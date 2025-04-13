# segmentation_publisher.py
import rclpy
from rclpy.node import Node
import std_msgs.msg
from sensor_msgs.msg import Image

class SegmentationPublisher(Node):
    def __init__(self, node, segmentation_topic):
        super().__init__('segmentation_publisher')
        self.node = node
        self.segmentation_pub = self.create_publisher(Image, segmentation_topic, 10)

    def publish_segmentation(self, seg_img_msg, segmentation_processing):
        # Update header with current time and the frame_id from configuration
        header = seg_img_msg.header
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = segmentation_processing['frame_id']
        seg_img_msg.header = header

        # Publish the segmentation image message
        self.segmentation_pub.publish(seg_img_msg)
