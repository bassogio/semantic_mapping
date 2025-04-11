# rotate_pose_publisher.py
import rclpy
from rclpy.node import Node
import std_msgs.msg
from std_msgs.msg import String
import numpy as np
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped

class RotatePosePublisher(Node):
    def __init__(self, node, rotate_pose_topic):
        super().__init__('rotate_pose_publisher')
        self.node = node

        # Initialize publisher for PoseStamped
        self.rotate_pose_pub = self.create_publisher(PoseStamped, rotate_pose_topic, 10)

    def publish_rotate_pose(self, rotated_pose, rotate_pose_processing):
        """Format and publish."""
        # Create PoseStamped message
        header = std_msgs.msg.Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = rotate_pose_processing['frame_id']

        # Publish 
        self.rotate_pose_pub.publish(rotated_pose)
