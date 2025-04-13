#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import numpy as np
from PIL import Image as PILImage
import cv2
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

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

class ClipSegNode(Node):
    def __init__(self):
        super().__init__('clipseg_node')
        self.bridge = CvBridge()

        # Create a subscription to the raw image topic.
        self.subscription = self.create_subscription(
            Image,
            "/davis/left/image_raw",
            self.image_callback,
            10)

        # Publisher for the segmentation mask output.
        self.publisher_ = self.create_publisher(Image, '/clipseg/segmented_image', 10)

        # Determine the device: use GPU if available.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # Load the pre-trained CLIPSeg processor and model.
        self.get_logger().info("Loading CLIPSeg model and processor...")
        # Forcing the use of the slow tokenizer by setting use_fast=False prevents tensor shape mismatches.
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=False)
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model.to(self.device)
        self.get_logger().info("CLIPSeg model loaded successfully.")

        # Prepare text prompts from the label names.
        self.prompts = [label['name'] for label in labels]

    def image_callback(self, msg):
        try:
            # Convert the ROS image message to an OpenCV image.
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        # Convert the OpenCV BGR image to a PIL RGB image.
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # Prepare inputs for the CLIPSeg model.
        # Here we activate padding and truncation for the text to ensure consistent tensor shapes.
        inputs = self.processor(
            text=self.prompts,
            images=[pil_image] * len(self.prompts),
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Move all tensors in inputs to the designated device.
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.device)

        # Run inference without gradient calculations.
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process the raw logits with a sigmoid activation.
        preds = outputs.logits
        processed_preds = torch.sigmoid(preds)

        # Determine the predicted label per pixel.
        combined_preds = processed_preds.squeeze(0).argmax(dim=0).cpu().numpy()

        # Create an image array to store the colored segmentation mask.
        colored_mask = np.zeros((combined_preds.shape[0], combined_preds.shape[1], 3), dtype=np.uint8)
        for label in labels:
            colored_mask[combined_preds == label['id']] = label['color']

        # Convert the segmentation mask into a ROS image message and publish.
        try:
            seg_msg = self.bridge.cv2_to_imgmsg(colored_mask, encoding="rgb8")
            self.publisher_.publish(seg_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error during publishing: {e}')
            return

def main(args=None):
    rclpy.init(args=args)
    node = ClipSegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down ClipSeg node due to KeyboardInterrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
