# rotate_pose_processor.py
import rclpy
from rclpy.node import Node  
from geometry_msgs.msg import PoseStamped
import numpy as np
from rotate_pose_publisher import RotatePosePublisher
from cv_bridge import CvBridge

class RotatePoseProcessor(Node):
    def __init__(self, config):
        super().__init__('rotate_pose_processor')

        # Access the configuration parameters under 'rotate_pose_processing'
        self.rotate_pose_processing = config['rotate_pose_processing']

        # Load topics and rotation matrix from the configuration
        self.pose_topic = self.rotate_pose_processing['pose_topic']
        self.rotate_pose_topic = self.rotate_pose_processing['rotate_pose_topic']
        rotation_matrix_config = self.rotate_pose_processing['rotation_matrix']
        self.rotation_matrix = np.array(rotation_matrix_config)

        # Declare parameters for ROS 2 if needed
        self.declare_parameter('pose_topic', self.pose_topic)
        self.declare_parameter('rotate_pose_topic', self.rotate_pose_topic)

        # Initialize the publisher with the current node context
        self.publisher = RotatePosePublisher(self, self.rotate_pose_topic)

        # Initialize CvBridge (if you plan to process images later)
        self.bridge = CvBridge()

        # Subscribe to the original pose topic
        self.create_subscription(
            PoseStamped, 
            self.pose_topic, 
            self.rotate_pose_callback, 
            10
        )

    def rotate_pose_callback(self, msg):
        """Callback that rotates the incoming pose and publishes it."""
        try:
            # Extract the original position
            pose_x = msg.pose.position.x
            pose_y = msg.pose.position.y
            pose_z = msg.pose.position.z

            # Create a numpy array for the point and apply the rotation
            input_pose = np.array([pose_x, pose_y, pose_z])
            rotated_position = input_pose @ self.rotation_matrix.T

            # Create a new PoseStamped message with the rotated position
            rotated_msg = PoseStamped()
            rotated_msg.header.stamp = self.get_clock().now().to_msg()
            rotated_msg.header.frame_id = self.rotate_pose_processing['frame_id']
            rotated_msg.pose.position.x = float(rotated_position[0])
            rotated_msg.pose.position.y = float(rotated_position[1])
            rotated_msg.pose.position.z = float(rotated_position[2])
            # Optionally, set the orientation (for now, leaving it default or identity)
            # rotated_msg.pose.orientation.w = 1.0
            # rotated_msg.pose.orientation.x = 0.0
            # rotated_msg.pose.orientation.y = 0.0
            # rotated_msg.pose.orientation.z = 0.0

            # Publish the rotated pose message
            self.publisher.publish_rotate_pose(rotated_msg, self.rotate_pose_processing)

            # Log debug information
            self.get_logger().debug("Published rotated pose.")
        except Exception as e:
            self.get_logger().error(f"Error in rotate_pose_callback: {e}")
