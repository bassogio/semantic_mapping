# -----------------------------------
# Import Statements
# -----------------------------------
import os
import yaml
import argparse
import cv2
from cv_bridge import CvBridge, CvBridgeError
import torch
import numpy as np
from PIL import Image as PILImage
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerFeatureExtractor,
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

# -----------------------------------
# Configuration Loader Function
# -----------------------------------
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
        raise FileNotFoundError(f"Config file not found at: {config_file}")

# -----------------------------------
# ROS2 Node Definition
# -----------------------------------
class SegmentationNode(Node):
    def __init__(self, config):
        # Initialize the node with a unique name.
        super().__init__('segmentation_node')

        # -------------------------------------------
        # Access the configuration section.
        # -------------------------------------------
        self.node_config = config['segmentation_processing']

        # -------------------------------------------
        # Load configuration parameters.
        # -------------------------------------------
        self.segmentated_rgb_topic    = self.node_config['segmentated_rgb_topic']
        self.segmentated_depth_topic  = self.node_config['segmentated_depth_topic']
        self.color_image_topic        = self.node_config['color_image_topic']
        self.depth_image_topic        = self.node_config['depth_image_topic']
        self.frame_id                 = self.node_config['frame_id']
        self.labels                   = self.node_config['labels']

        # -------------------------------------------
        # Declare ROS2 parameters for runtime modification.
        # -------------------------------------------
        self.declare_parameter('segmentated_rgb_topic',      self.segmentated_rgb_topic)
        self.declare_parameter('segmentated_depth_topic',    self.segmentated_depth_topic)
        self.declare_parameter('color_image_topic',          self.color_image_topic)
        self.declare_parameter('frame_id',                   self.frame_id)

        # -------------------------------------------
        # Retrieve final parameter values from the parameter server.
        # -------------------------------------------
        self.segmentated_rgb_topic       = self.get_parameter('segmentated_rgb_topic').value
        self.segmentated_depth_topic     = self.get_parameter('segmentated_depth_topic').value
        self.color_image_topic           = self.get_parameter('color_image_topic').value
        self.frame_id                    = self.get_parameter('frame_id').value

        # -------------------------------------------
        # Initialize additional attributes needed for processing.
        # -------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # Let user choose which segmentation model to use.
        self.model_input = ''
        while True:
            self.model_input = input("Please choose a model (clipseg/segformer): ").strip().lower()
            if self.model_input == 'clipseg':
                self.get_logger().info("You chose the CLIPSeg model.")
                self.clipseg_initialization()
                break
            elif self.model_input == 'segformer':
                self.get_logger().info("You chose the SegFormer model.")
                self.segformer_initialization()
                break
            else:
                self.get_logger().error("Invalid model choice. Please choose 'clipseg' or 'segformer'.")

        # Prepare text prompts from the label names.
        self.prompts = [label['name'] for label in self.labels]

        # Initialize CvBridge once for the node.
        self.bridge = CvBridge()

        # Initialize a variable to store the latest depth image.
        self.latest_depth_image = None

        # -------------------------------------------
        # Create Publishers.
        # -------------------------------------------
        self.segmentation_publisher = self.create_publisher(Image, self.segmentated_rgb_topic, 10)
        self.combined_depth_publisher = self.create_publisher(Image, self.segmentated_depth_topic, 10)

        # -------------------------------------------
        # Create Subscribers.
        # -------------------------------------------
        # Depth image subscriber with a dedicated callback.
        self.depth_subscription = self.create_subscription(
            Image,                      # Message type.
            self.depth_image_topic,     # Topic name.
            self.depth_callback,        # Callback function.
            10                          # Queue size.
        )

        # RGB image subscriber triggering segmentation.
        self.rgb_subscription = self.create_subscription(
            Image,                      # Message type.
            self.color_image_topic,     # Topic name.
            self.segmentation_callback, # Callback function.
            10                          # Queue size.
        )

        # -------------------------------------------
        # Initialize flags to track if each subscriber has received a message.
        # -------------------------------------------
        self.received_rgb_image = False
        self.received_depth_image = False

        # -------------------------------------------
        # Create a Timer to check if all subscribed topics have received at least one message.
        # -------------------------------------------
        self.subscription_check_timer = self.create_timer(2.0, self.check_initial_subscriptions)

    # -------------------------------------------
    # Function to initialize the ClipSeg model.
    # -------------------------------------------
    def clipseg_initialization(self):
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=False)
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model.to(self.device)
        self.get_logger().info("CLIPSeg model loaded successfully.")

    # -------------------------------------------
    # Function to initialize the SegFormer model.
    # -------------------------------------------
    def segformer_initialization(self):
        model_name = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.get_logger().info("SegFormer loaded successfully.")

    # -------------------------------------------
    # Timer Callback to Check if All Subscribed Topics Have Received at Least One Message.
    # -------------------------------------------
    def check_initial_subscriptions(self):
        waiting_topics = []
        if not self.received_rgb_image:
            waiting_topics.append(f"'{self.color_image_topic}'")
        if not self.received_depth_image:
            waiting_topics.append(f"'{self.depth_image_topic}'")
            
        if waiting_topics:
            self.get_logger().info(f"Waiting for messages on topics: {', '.join(waiting_topics)}")
        else:
            self.get_logger().info(
                "All subscribed topics have received at least one message.\n"
                f"Using '{self.model_input}' model for segmentation.\n"
                f"SegmentationNode started with publishers on '{self.segmentated_rgb_topic}' and '{self.segmentated_depth_topic}',\n"
                f"subscribers on '{self.color_image_topic}' and '{self.depth_image_topic}',\n"
                f"and frame_id '{self.frame_id}'."
            )
            self.subscription_check_timer.cancel()

    # -------------------------------------------
    # Depth Image Callback Function.
    # -------------------------------------------
    def depth_callback(self, msg):
        try:
            # Convert the ROS image message to a NumPy array.
            # Assuming the depth image encoding is "32FC1".
            cv_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
            self.latest_depth_image = cv_depth_image.astype(np.float32)
            if not self.received_depth_image:
                self.received_depth_image = True
            self.get_logger().debug("Received and stored a new depth image.")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error in depth callback: {e}')

    # -------------------------------------------
    # Segmentation Callback Function.
    # -------------------------------------------
    def segmentation_callback(self, msg):
        if not self.received_rgb_image:
            self.received_rgb_image = True

        try:
            # Convert the ROS image message to an OpenCV image.
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        # Convert the OpenCV BGR image to a PIL RGB image.
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        # Process the image based on the chosen model.
        if self.model_input == 'clipseg':
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
            preds = outputs.logits
            processed_preds = torch.sigmoid(preds)
            # For CLIPSeg we use argmax over the single batch dimension.
            combined_preds = processed_preds.squeeze(0).argmax(dim=0).cpu().numpy()
            
        elif self.model_input == 'segformer':
            inputs = self.feature_extractor(images=pil_image, return_tensors="pt")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            preds = outputs.logits  # Shape: [batch, num_labels, height, width]
            combined_preds = preds.argmax(dim=1)[0].cpu().numpy()
        else:
            self.get_logger().error("Invalid model selection.")
            return

        # Create a colored segmentation mask for visualization.
        colored_mask = np.zeros((combined_preds.shape[0], combined_preds.shape[1], 3), dtype=np.uint8)
        for label in self.labels:
            colored_mask[combined_preds == label['id']] = label['color']

        # Publish the colored segmentation mask.
        try:
            seg_msg = self.bridge.cv2_to_imgmsg(colored_mask, encoding="rgb8")
            self.segmentation_publisher.publish(seg_msg)
            self.get_logger().info("Published segmentation mask.")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error during segmentation publishing: {e}')
            return

        # ------------------------------
        # Combine Depth and Segmentation IDs
        # ------------------------------
        if self.latest_depth_image is None:
            self.get_logger().warn("No depth image available yet; skipping combined depth publication.")
            return

        # Check if the segmentation result and the depth image have the same spatial dimensions.
        seg_height, seg_width = combined_preds.shape
        depth_height, depth_width = self.latest_depth_image.shape[:2]
        if (seg_height, seg_width) != (depth_height, depth_width):
            self.get_logger().info("Resizing segmentation id image to match depth image resolution.")
            # Use nearest-neighbor to preserve label ids.
            combined_preds_resized = cv2.resize(combined_preds.astype(np.float32),
                                                (depth_width, depth_height),
                                                interpolation=cv2.INTER_NEAREST)
        else:
            combined_preds_resized = combined_preds.astype(np.float32)

        # Create a 2-channel image where:
        # Channel 0: Depth data.
        # Channel 1: Segmentation id data.
        combined_depth_image = np.dstack((self.latest_depth_image, combined_preds_resized))
        
        try:
            # Publish the combined depth image.
            # We use "32FC2" encoding to indicate a 2-channel float image.
            combined_depth_msg = self.bridge.cv2_to_imgmsg(combined_depth_image, encoding="32FC2")
            self.combined_depth_publisher.publish(combined_depth_msg)
            self.get_logger().info("Published combined depth + segmentation id image.")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error during combined depth publishing: {e}')
            return

# -----------------------------------
# Main Entry Point
# -----------------------------------
def main(args=None):
    """
    The main function initializes the ROS2 system, loads configuration parameters,
    creates an instance of the SegmentationNode, and spins to process messages until shutdown.
    """
    rclpy.init(args=args)
    config = load_config()
    node = SegmentationNode(config)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass  # Allow graceful shutdown on CTRL+C.
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
