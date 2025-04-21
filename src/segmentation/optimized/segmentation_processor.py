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
from contextlib import nullcontext  # Added to use as fallback context manager

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
    config_file = os.path.join(script_dir, '../../../config/segmentation_config.yaml')
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
        self.segmentation_topic = self.node_config['segmentation_topic']
        self.color_image_topic  = self.node_config['color_image_topic']
        self.frame_id           = self.node_config['frame_id']
        self.labels             = self.node_config['labels']
        self.model              = self.node_config['model']

        # -------------------------------------------
        # Declare ROS2 parameters for runtime modification.
        # -------------------------------------------
        self.declare_parameter('segmentation_topic', self.segmentation_topic)
        self.declare_parameter('color_image_topic',  self.color_image_topic)
        self.declare_parameter('frame_id',           self.frame_id)
        self.declare_parameter('model',              self.model)

        # -------------------------------------------
        # Retrieve final parameter values from the parameter server.
        # -------------------------------------------
        self.segmentation_topic = self.get_parameter('segmentation_topic').value
        self.color_image_topic  = self.get_parameter('color_image_topic').value
        self.frame_id           = self.get_parameter('frame_id').value
        self.model              = self.get_parameter('model').value
        
        # -------------------------------------------
        # Initialize additional attributes needed for processing.
        # -------------------------------------------
        # Determine the device: use GPU if available.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # Prepare text prompts from the label names.
        self.prompts = [label['name'] for label in self.labels]

        # Initialize CvBridge once for the node.
        self.bridge = CvBridge()

        # Let user choose which segmentation model to use.
        # self.model_input = ''
        self.model_input = self.model

        while True:
            # self.model_input = input("Please choose a model (clipseg/segformer): ").strip().lower()
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

        # -------------------------------------------
        # Create Publishers.
        # -------------------------------------------
        self.segmentation_publisher = self.create_publisher(Image, self.segmentation_topic, 10)

        # -------------------------------------------
        # Create Subscribers.
        # -------------------------------------------
        self.subscription = self.create_subscription(
            Image,                      # Message type.
            self.color_image_topic,     # Topic name.
            self.segmentation_callback, # Callback function.
            10                          # Queue size.
        )

        # -------------------------------------------
        # Initialize flags to track if a color image has been received.
        # -------------------------------------------
        self.received_color_image = False

        # Timer to check if the subscribed topic has received at least one message.
        self.subscription_check_timer = self.create_timer(2.0, self.check_initial_subscriptions)

        # Define the target resolution for processing (adjust as needed).
        self.target_width = 320
        self.target_height = 240

    # -------------------------------------------
    # Function to initialize the ClipSeg model.
    # -------------------------------------------
    def clipseg_initialization(self):
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=False)
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model.to(self.device)
        
        # Enable mixed precision if using GPU.
        if self.device.type == "cuda":
            self.model.half()

        # TorchScript conversion is commented out for CLIPSeg to avoid incompatibility issues.
        # self.model = torch.jit.script(self.model)

        self.get_logger().info("CLIPSeg model loaded successfully.")

        # Precompute text tokens from prompts and move them to the device.
        self.precomputed_text_inputs = self.processor(
            text=self.prompts,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        for key, value in self.precomputed_text_inputs.items():
            if isinstance(value, torch.Tensor):
                self.precomputed_text_inputs[key] = value.to(self.device)
        
    # -------------------------------------------
    # Function to initialize the SegFormer model.
    # -------------------------------------------
    def segformer_initialization(self):
        # model_name = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
        model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)  
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Enable mixed precision if using GPU.
        if self.device.type == "cuda":
            self.model.half()
        
        # TorchScript conversion is optional. Uncomment if supported.
        # self.model = torch.jit.script(self.model)

        self.get_logger().info("SegFormer model loaded successfully.")

    # -------------------------------------------
    # Timer Callback to Check Subscription Initialization
    # -------------------------------------------
    def check_initial_subscriptions(self):
        waiting_topics = []
        if not self.received_color_image:
            waiting_topics.append(f"'{self.color_image_topic}'")
        if waiting_topics:
            self.get_logger().info(f"Waiting for messages on topics: {', '.join(waiting_topics)}")
        else:
            self.get_logger().info(
                "All subscribed topics have received at least one message.\n"
                f"Using '{self.model_input}' model for segmentation.\n"
                f"SegmentationNode started with publishers on '{self.segmentation_topic}', "
                f"subscriber on '{self.color_image_topic}', and frame_id '{self.frame_id}'."
            )
            self.subscription_check_timer.cancel()

    # -------------------------------------------
    # Segmentation Callback Function 
    # -------------------------------------------
    def segmentation_callback(self, msg):
        if not self.received_color_image:
            self.received_color_image = True

        try:
            # Convert the ROS image message to an OpenCV image.
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        # Convert the OpenCV BGR image to a PIL RGB image.
        pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        # Resize image to lower resolution to speed up processing.
        pil_image = pil_image.resize((self.target_width, self.target_height))

        # Set up a proper autocast context using the recommended API.
        if self.device.type == "cuda":
            autocast_cm = torch.amp.autocast(device_type="cuda")
        else:
            autocast_cm = nullcontext()

        # Process the image based on the chosen model.
        if self.model_input == 'clipseg':
            # Process image portion and reuse precomputed text tokens.
            image_inputs = self.processor(
                images=[pil_image] * len(self.prompts),
                return_tensors="pt"
            )
            for key, value in image_inputs.items():
                if isinstance(value, torch.Tensor):
                    image_inputs[key] = value.to(self.device)

            # Merge precomputed text inputs into image_inputs.
            image_inputs.update(self.precomputed_text_inputs)

            # Use the updated autocast context for mixed precision inference.
            with torch.no_grad(), autocast_cm:
                outputs = self.model(**image_inputs)
            preds = outputs.logits

            # For CLIPSeg, apply sigmoid and then determine the class.
            processed_preds = torch.sigmoid(preds)
            combined_preds = processed_preds.squeeze(0).argmax(dim=0).cpu().numpy()

        elif self.model_input == 'segformer':
            inputs = self.feature_extractor(images=pil_image, return_tensors="pt")
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    inputs[key] = value.to(self.device)
            with torch.no_grad(), autocast_cm:
                outputs = self.model(**inputs)
            preds = outputs.logits  # Shape: [batch, num_labels, height, width]
            combined_preds = preds.argmax(dim=1)[0].cpu().numpy()
        else:
            self.get_logger().error("Invalid model selection.")
            return

        # Create a colored segmentation mask for visualization.
        colored_mask = np.zeros((combined_preds.shape[0], combined_preds.shape[1], 3), dtype=np.uint8)
        for label in self.labels:
            # Corrected dictionary access using square brackets
            colored_mask[combined_preds == label['id']] = label['color']

        # Convert the segmentation mask into a ROS image message and publish.
        try:
            seg_msg = self.bridge.cv2_to_imgmsg(colored_mask, encoding="rgb8")
            self.segmentation_publisher.publish(seg_msg)
            self.get_logger().info("Published segmentation mask.")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error during publishing: {e}')

# -----------------------------------
# Main Entry Point
# -----------------------------------
def main(args=None):
    """
    Main function to initialize the ROS2 system, load configuration parameters,
    create an instance of the SegmentationNode, and start processing.
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

# -----------------------------------
# Run the Node when Executed Directly
# -----------------------------------
if __name__ == '__main__':
    main()
