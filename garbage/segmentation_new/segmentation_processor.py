# segmentation_processor
import rclpy
from rclpy.node import Node  
from sensor_msgs.msg import Image
import numpy as np
from segmentation_publisher import SegmentationPublisher
from cv_bridge import CvBridge
import torch
class SegmentationProcessor(rclpy.node.Node):
    def __init__(self, config):
        super().__init__('segmentation_processor')
        
        # Access the config under 'point_cloud_processing'
        self.segmentation_processing = config['segmentation_processing']

        # Check if GPU is available and set the device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using GPU for computation:", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            print("Using CPU for computation.")

        # Load configuration parameters
        self.color_image_topic = self.segmentation_processing['color_image_topic']
        self.segmentation_topic = self.segmentation_processing['segmentation_topic']
        self.labels = self.segmentation_processing['labels']

        # Create dictionaries for easy lookup of label IDs to colors and names.
        self.label_colors = {label['id']: label['color'] for label in self.labels}
        self.label_names = {label['id']: label['name'] for label in self.labels}


        # Declare parameters for ROS 2
        self.declare_parameter('raw_image_topic', self.color_image_topic)
        self.declare_parameter('segmentation_topic', self.segmentation_topic)

        # Initialize the SegmentationPublisher
        self.publisher = SegmentationPublisher(self, self.segmentation_topic)

        # Initialize CvBridge
        self.bridge = CvBridge()

        self.create_subscription(
            Image,  
            self.raw_image_topic,
            self.segmentation_callback, 10)

    def segmentation_callback(self, msg):
        """Callback to process the raw image and generate a segmentated image."""
        try:


            # Publish the segmentation
            self.publisher.publish_segmentation()

            # Log debug information
            self.get_logger().debug(f"Published Segmentated Image: {self.segmentation_topic}")
        except Exception as e:
            self.get_logger().error(f"Error: {e}")
