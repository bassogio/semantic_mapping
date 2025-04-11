import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from clipseg_processor import CLIPsegProcessor
import cv2 as cv
import numpy as np

class CLIPsegPublisher(Node):
    def __init__(self, device):
        super().__init__("CLIPseg_segmentation_processor")
        self.device = device
        self.logger = self.get_logger()
        self.bridge = CvBridge()

        # Subscriber and Publisher
        self.color_sub = self.create_subscription(Image, '/camera/color/image_raw', self.color_image_callback, 10)
        self.segmented_image_pub = self.create_publisher(Image, '/camera/segmented_image', 10)

        # Labels and prompts
        self.labels = [
            {'name': 'floor', 'id': 0, 'color': (255, 0, 0)},
            {'name': 'person', 'id': 1, 'color': (0, 255, 0)},
            {'name': 'dog', 'id': 2, 'color': (0, 0, 255)},
            {'name': 'bottle', 'id': 3, 'color': (125, 125, 0)},
            {'name': 'ball', 'id': 4, 'color': (0, 125, 125)}]
        self.prompts = [label['name'] for label in self.labels]
        self.label_colors = {label['id']: label['color'] for label in self.labels}

        # Initialize CLIPsegProcessor with device
        self.processor = CLIPsegProcessor(self.prompts, self.device)
        self.logger.info("CLIPsegPublisher node initialized.")

    def color_image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            segmented_image = self.processor.process_image(cv_image, self.label_colors)

            ros_image = self.bridge.cv2_to_imgmsg(segmented_image, "bgr8")
            self.segmented_image_pub.publish(ros_image)
        except Exception as e:
            self.logger.error(f"Failed to process image: {e}")
