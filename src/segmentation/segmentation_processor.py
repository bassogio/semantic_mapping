#!/usr/bin/env python3
"""
Segmentation Node using CLIPSeg

This node:
  - Loads configuration parameters from a YAML file.
  - Checks if a GPU is available.
  - Subscribes to a raw color image topic.
  - Processes each incoming image using a CLIPSeg-based segmentation model.
  - Publishes the segmented image on a dedicated topic.
  - (Optionally) subscribes to camera info.
"""

import os
import yaml
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPSegProcessor as ClipProcessor, CLIPSegForImageSegmentation
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

def load_config():
    """
    Load configuration parameters.

    Returns:
        dict: A dictionary containing configuration data.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, '../../config/segmentation_config.yaml')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        return {}

class CLIPsegProcessorWrapper:
    """
    Wraps the CLIPSeg model for image segmentation.
    """
    def __init__(self, prompts, device):
        self.prompts = prompts
        self.device = device
        # Removed unused keyword arguments ('padding', 'truncation')
        self.processor = ClipProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=True)
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)
        self.model.eval()  # Ensure the model is in evaluation mode.
    
    def process_image(self, cv_image, label_colors):
        """
        Process the input image to generate a segmentation overlay.

        Args:
            cv_image (numpy.ndarray): Input color image in BGR format.
            label_colors (dict): Mapping of prompt indices to color tuples.

        Returns:
            numpy.ndarray: Combined segmented image.
        """
        original_height, original_width = cv_image.shape[:2]

        # Prepare inputs for all prompts without the removed kwargs.
        inputs = self.processor(
            text=self.prompts,
            images=[cv_image] * len(self.prompts),
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Optionally convert to half precision for faster inference.
        if self.device.type == 'cuda':
            self.model.half()
            inputs = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        preds = torch.sigmoid(outputs.logits)

        # Ensure the tensor has the shape (N, C, H, W).
        if preds.ndim == 3:  # Likely shape (N, H, W)
            preds = preds.unsqueeze(1)  # Now shape becomes (N, 1, H, W)

        # Use GPU-accelerated interpolation to resize predictions.
        preds = F.interpolate(preds, size=(original_height, original_width),
                              mode='bilinear', align_corners=False)

        # Remove the channel dimension if it's 1.
        if preds.shape[1] == 1:
            preds = preds.squeeze(1)

        preds = preds.cpu().numpy()  # Now shape: (N, original_height, original_width)

        combined_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)
        # Combine each prediction with its associated label color.
        for idx, confidence in enumerate(preds):
            color = np.array(label_colors.get(idx, (255, 255, 255)))
            colored_mask = (confidence[:, :, None] * color).astype(np.uint8)
            combined_image = cv.add(combined_image, colored_mask)

        return combined_image

class SegmentationNode(Node):
    def __init__(self, config):
        super().__init__('segmentation_node')
        # Check if GPU is available.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU for computation:", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            print("Using CPU for computation.")

        # Access configuration parameters.
        self.node_config = config['segmentation_processing']
        self.frame_id = self.node_config.get('frame_id', 'map')
        self.segmentation_topic = self.node_config['segmentation_topic']
        self.color_image_topic = self.node_config['color_image_topic']
        self.camera_parameters_topic = self.node_config['camera_parameters_topic']
        self.labels = self.node_config['labels']

        # Declare ROS2 parameters for potential runtime modifications.
        self.declare_parameter('frame_id', self.frame_id)
        self.declare_parameter('segmentation_topic', self.segmentation_topic)
        self.declare_parameter('color_image_topic', self.color_image_topic)
        self.declare_parameter('camera_parameters_topic', self.camera_parameters_topic)
        self.frame_id = self.get_parameter('frame_id').value
        self.segmentation_topic = self.get_parameter('segmentation_topic').value
        self.color_image_topic = self.get_parameter('color_image_topic').value
        self.camera_parameters_topic = self.get_parameter('camera_parameters_topic').value

        # Initialize CV bridge.
        self.bridge = CvBridge()
        self.prompts = [label['name'] for label in self.labels]
        self.label_colors = {idx: tuple(label['color']) for idx, label in enumerate(self.labels)}
        self.processor = CLIPsegProcessorWrapper(self.prompts, self.device)

        # Set up publisher and subscribers.
        self.segmentation_pub = self.create_publisher(Image, self.segmentation_topic, 10)
        self.image_sub = self.create_subscription(
            Image,
            self.color_image_topic,
            self.image_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_parameters_topic,
            self.camera_info_callback,
            10
        )
        self.get_logger().info(
            "SegmentationNode started on topics: color images from '{}' and publishing segmented images on '{}', using frame_id '{}'".format(
                self.color_image_topic, self.segmentation_topic, self.frame_id
            )
        )

    def image_callback(self, msg: Image):
        self.get_logger().info("Received color image")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            seg_image = self.processor.process_image(cv_image, self.label_colors)
            ros_seg_image = self.bridge.cv2_to_imgmsg(seg_image, "bgr8")
            ros_seg_image.header.stamp = self.get_clock().now().to_msg()
            ros_seg_image.header.frame_id = self.frame_id
            self.segmentation_pub.publish(ros_seg_image)
        except Exception as e:
            self.get_logger().error("Error processing image: {}".format(e))

    def camera_info_callback(self, msg: CameraInfo):
        self.get_logger().info("Received camera info")

def main(args=None):
    rclpy.init(args=args)
    config = load_config()
    node = SegmentationNode(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
