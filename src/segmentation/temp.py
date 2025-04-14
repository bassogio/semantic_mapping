#!/usr/bin/env python3
import argparse
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import numpy as np
from PIL import Image as PILImage
import cv2
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerFeatureExtractor,
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
)

# Define segmentation labels with their IDs and corresponding colors.
labels = [
    {'name': 'road', 'id': 0, 'color': (128, 64, 128)},
    {'name': 'sidewalk', 'id': 1, 'color': (244, 35, 232)},
    {'name': 'building', 'id': 2, 'color': (70, 70, 70)},
    {'name': 'wall', 'id': 3, 'color': (102, 102, 156)},
    {'name': 'fence', 'id': 4, 'color': (190, 153, 153)},
    {'name': 'pole', 'id': 5, 'color': (153, 153, 153)},
    {'name': 'traffic light', 'id': 6, 'color': (250, 170, 30)},
    {'name': 'traffic sign', 'id': 7, 'color': (220, 220, 0)},
    {'name': 'vegetation', 'id': 8, 'color': (107, 142, 35)},
    {'name': 'terrain', 'id': 9, 'color': (152, 251, 152)},
    {'name': 'sky', 'id': 10, 'color': (70, 130, 180)},
    {'name': 'person', 'id': 11, 'color': (220, 20, 60)},
    {'name': 'rider', 'id': 12, 'color': (255, 0, 0)},
    {'name': 'car', 'id': 13, 'color': (0, 0, 142)},
    {'name': 'truck', 'id': 14, 'color': (0, 0, 70)},
    {'name': 'bus', 'id': 15, 'color': (0, 60, 100)},
    {'name': 'train', 'id': 16, 'color': (0, 80, 100)},
    {'name': 'motorcycle', 'id': 17, 'color': (0, 0, 230)},
    {'name': 'bicycle', 'id': 18, 'color': (119, 11, 32)},
    {'name': 'void', 'id': 29, 'color': (0, 0, 0)},
]

# Precompute a lookup for label id to color.
id_to_color = {label['id']: label['color'] for label in labels}

class DualSegmentationNode(Node):
    def __init__(self, model_type: str):
        super().__init__('dual_segmentation_node')
        self.model_type = model_type.lower()
        self.bridge = CvBridge()
        # Subscribe to camera topic.
        self.subscription = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.image_callback,
            10)
        # Publisher for the blended segmentation result.
        self.pub_segmented = self.create_publisher(Image, "segmentation/segmented_image", 10)

        # Choose device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        if self.model_type == "segformer":
            self.get_logger().info("Loading SegFormer model and feature extractor...")
            model_name = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.get_logger().info("SegFormer loaded successfully.")
        elif self.model_type == "clipseg":
            self.get_logger().info("Loading CLIPSeg model and processor...")
            model_name = "CIDAS/clipseg-rd64-refined"
            # Force the slow tokenizer to avoid tensor shape issues.
            self.processor = CLIPSegProcessor.from_pretrained(model_name, use_fast=False)
            self.model = CLIPSegForImageSegmentation.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self.get_logger().info("CLIPSeg loaded successfully.")
            # Prepare text prompts based on label names.
            self.prompts = [label['name'] for label in labels]
        else:
            self.get_logger().error(f"Unknown model type: {self.model_type}. Use 'segformer' or 'clipseg'.")
            rclpy.shutdown()

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV BGR image.
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        # Convert OpenCV image (BGR) to PIL RGB image.
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        # Resize image (for consistency) to 512x512.
        resized_image = pil_image.resize((512, 512))

        if self.model_type == "segformer":
            # Process with SegFormer.
            inputs = self.feature_extractor(images=pil_image, return_tensors="pt")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits  # shape: (batch_size, num_labels, H, W)
            segmentation = torch.argmax(logits.squeeze(), dim=0).cpu().numpy()
        else:  # clipseg branch
            # Process with CLIPSeg.
            inputs = self.processor(
                text=self.prompts,
                images=[pil_image] * len(self.prompts),
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits
            processed_preds = torch.sigmoid(logits)
            segmentation = processed_preds.squeeze(0).argmax(dim=0).cpu().numpy()

        # Create a colored segmentation mask using the id_to_color mapping.
        h, w = segmentation.shape
        segmentation_colored = np.zeros((h, w, 3), dtype=np.uint8)
        for label_id, color in id_to_color.items():
            segmentation_colored[segmentation == label_id] = color

        # Convert colored mask to PIL image.
        mask_image = PILImage.fromarray(segmentation_colored)
        # Ensure mask image size matches the resized original.
        if mask_image.size != resized_image.size:
            mask_image = mask_image.resize(resized_image.size)
        # Blend the original resized image with the colored mask (50% blend).
        blended = PILImage.blend(resized_image, mask_image.convert("RGB"), alpha=0.5)

        # Convert the blended image back to OpenCV BGR and then to a ROS Image message.
        blended_cv = cv2.cvtColor(np.array(blended), cv2.COLOR_RGB2BGR)
        try:
            seg_msg = self.bridge.cv2_to_imgmsg(blended_cv, encoding="bgr8")
            self.pub_segmented.publish(seg_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error during publishing: {e}')

def main(args=None):
    parser = argparse.ArgumentParser(description="ROS2 node for segmentation with SegFormer or CLIPSeg")
    parser.add_argument('--model', type=str, default="segformer",
                        help="Specify model type: 'segformer' or 'clipseg'")
    args, unknown = parser.parse_known_args()

    rclpy.init(args=args)
    node = DualSegmentationNode(model_type=args.model)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt, shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
