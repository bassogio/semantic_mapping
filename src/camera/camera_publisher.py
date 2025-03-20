# camera_publisher.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image

class CameraPublisher(Node):
    def __init__(self, node, parameters_topic, timestamp_topic):
        super().__init__('camera_publisher')
        self.node = node

        # Initialize publishers for camera info, timestamp, and images
        self.parameters_pub = self.create_publisher(CameraInfo, parameters_topic, 10)
        self.timestamp_pub = self.create_publisher(String, timestamp_topic, 10)
        self.color_image_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)  # Color image topic
        self.depth_image_pub = self.create_publisher(Image, '/camera/depth/image_raw', 10)  # Depth image topic

    def publish_camera_info(self, camera_info):
        self.parameters_pub.publish(camera_info)

    def publish_timestamp(self, timestamp_str):
        msg = String()
        msg.data = timestamp_str
        self.timestamp_pub.publish(msg)

    def publish_image(self, topic, image_msg):
        if topic == '/camera/color/image_raw':
            self.color_image_pub.publish(image_msg)
        elif topic == '/camera/depth/image_raw':
            self.depth_image_pub.publish(image_msg)

