# point_cloud_publisher.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
from std_msgs.msg import String
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from std_msgs.msg import Header


class PointCloudPublisher(Node):
    def __init__(self, node, point_cloud_topic):
        super().__init__('point_cloud_publisher')
        self.node = node

        # Initialize publisher for PointCloud2
        self.point_cloud_pub = self.create_publisher(PointCloud2, point_cloud_topic, 10)

    def publish_point_cloud(self, point_cloud):
        """Format the point cloud and publish."""
        # Create PointCloud2 message
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'  # Use the correct frame ID for your setup

        # Convert the point cloud to PointCloud2 format
        pc_data = pc2.create_cloud_xyz32(header, point_cloud)

        # Publish PointCloud
        self.point_cloud_pub.publish(pc_data)
