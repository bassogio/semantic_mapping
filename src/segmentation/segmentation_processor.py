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

Ensure your configuration file (../../config/segmentation_config.yaml)
contains the required keys. For example:

segmentation_processing:
  frame_id: "map"
  segmentation_topic: "/camera/segmentation"
  color_image_topic: "/camera/color/image_raw"
  camera_parameters_topic: "/camera/CameraInfo"
  labels:
    - name: "ball"
      id: 0
      color: [128, 64, 128]
    - name: "bottle"
      id: 8
      color: [107, 142, 35]
    - name: "person"
      id: 11
      color: [220, 20, 60]
    - name: "void"
      id: 29
      color: [255, 255, 255]
"""

import os
import yaml
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import torch
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
        self.processor = ClipProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=True)
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)

    def process_image(self, cv_image, label_colors):
        """
        Process the input image to generate a segmentation overlay.

        Args:
            cv_image (numpy.ndarray): Input color image in BGR format.
            label_colors (dict): Mapping of prompt indices to color tuples.

        Returns:
            numpy.ndarray: Combined segmented image.
        """
        
        inputs = self.processor(
            text=self.prompts,
            images=[cv_image] * len(self.prompts),
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        preds = torch.sigmoid(outputs.logits).cpu()
        original_height, original_width = cv_image.shape[:2]
        combined_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)
        for idx, confidence_map in enumerate(preds):
            confidence = confidence_map.numpy().squeeze()
            resized_confidence = cv.resize(confidence, (original_width, original_height), interpolation=cv.INTER_LINEAR)
            color = label_colors.get(idx, (255, 255, 255))
            for c in range(3):
                combined_image[:, :, c] += (resized_confidence * color[c]).astype(np.uint8)
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

        # Access the configuration section.
        self.node_config = config['segmentation_processing']
        self.frame_id = self.node_config.get('frame_id', 'map')
        self.segmentation_topic = self.node_config['segmentation_topic']
        self.color_image_topic = self.node_config['color_image_topic']
        self.camera_parameters_topic = self.node_config['camera_parameters_topic']
        # Load labels from configuration (do not declare them as a ROS parameter)
        self.labels = self.node_config['labels']

        # Declare ROS2 parameters for runtime modification of simple types.
        self.declare_parameter('frame_id', self.frame_id)
        self.declare_parameter('segmentation_topic', self.segmentation_topic)
        self.declare_parameter('color_image_topic', self.color_image_topic)
        self.declare_parameter('camera_parameters_topic', self.camera_parameters_topic)
        # Retrieve potential runtime overrides.
        self.frame_id = self.get_parameter('frame_id').value
        self.segmentation_topic = self.get_parameter('segmentation_topic').value
        self.color_image_topic = self.get_parameter('color_image_topic').value
        self.camera_parameters_topic = self.get_parameter('camera_parameters_topic').value

        # Initialize the bridge.
        self.bridge = CvBridge()
        # Build list of prompts and mapping of label colors.
        self.prompts = [label['name'] for label in self.labels]
        self.label_colors = { idx: tuple(label['color']) for idx, label in enumerate(self.labels) }
        # Initialize the segmentation processor.
        self.processor = CLIPsegProcessorWrapper(self.prompts, self.device)
        # Create a publisher for the segmented image.
        self.segmentation_pub = self.create_publisher(Image, self.segmentation_topic, 10)
        # Create subscribers.
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
