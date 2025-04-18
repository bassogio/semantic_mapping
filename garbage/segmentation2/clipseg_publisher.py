import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from clipseg_processor import CLIPsegProcessor
import cv2 as cv
import numpy as np
import torch

class CLIPsegPublisher(Node):
    def __init__(self):
        super().__init__("CLIPseg_segmentation_processor")

        self.logger = self.get_logger()
        self.bridge = CvBridge()

        # Subscriber and Publisher
        self.color_sub = self.create_subscription(Image, '/camera/color/image_raw', self.color_image_callback, 10)
        self.segmented_image_pub = self.create_publisher(Image, '/camera/segmented_image', 10)

        # Labels and prompts
        self.labels = [{'name': 'floor', 'id': 0, 'color': (255, 0, 0)},
                       {'name': 'person', 'id': 1, 'color': (0, 255, 0)}]
        self.prompts = [label['name'] for label in self.labels]
        self.label_colors = {label['id']: label['color'] for label in self.labels}

        # Initialize CLIPsegProcessor
        self.processor = CLIPsegProcessor(self.prompts)
        self.logger.info("CLIPsegPublisher node initialized.")

    def color_image_callback(self, msg):
        """Callback function for processing images."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            segmented_image = self.processor.process_image(cv_image, self.label_colors)

            # Display and publish segmented image
            cv.imshow("Combined Segmentation", segmented_image)
            cv.waitKey(1)
            ros_image = self.bridge.cv2_to_imgmsg(segmented_image, "bgr8")
            self.segmented_image_pub.publish(ros_image)
        except Exception as e:
            self.logger.error(f"Failed to process image: {e}")