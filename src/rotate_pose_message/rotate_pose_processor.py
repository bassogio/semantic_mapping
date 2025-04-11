import rclpy
from rclpy.node import Node  
from geometry_msgs.msg import PoseStamped, Point
import numpy as np
from rotate_pose_publisher import RotatePosePublisher
from transforms3d.euler import euler2mat
from transforms3d.quaternions import quat2mat, mat2quat
from visualization_msgs.msg import Marker
from std_srvs.srv import Trigger

class RotatePoseProcessor(Node):
    def __init__(self, config):
        super().__init__('rotate_pose_processor')
        
         # Access the config file
        self.rotate_pose_processing = config['rotate_pose_processing']

        # Load configuration parameters
        self.pose_topic = self.rotate_pose_processing['pose_topic']
        self.roll = self.rotate_pose_processing['roll']
        self.pitch = self.rotate_pose_processing['pitch']
        self.yaw = self.rotate_pose_processing['yaw']   

        self.rotation_matrix = euler2mat(np.deg2rad(self.roll), 
                                         np.deg2rad(self.pitch), 
                                         np.deg2rad(self.yaw), 
                                         axes='sxyz')
        self.get_logger().info(
            f"Initial rotation matrix computed with roll={self.roll}°, pitch={self.pitch}°, yaw={self.yaw}°"
        )
    
        # Declare parameters for ROS 2
        self.declare_parameter('pose_topic', self.pose_topic)

        # Initialize the PointCloudPublisher
        self.publisher = RotatePosePublisher(self, self.pose_topic)

        self.create_subscription(
            PoseStamped, 
            self.pose_topic, 
            self.rotate_pose_callback, 10)

    def rotate_pose_callback(self, msg):
        try:
            # Extract original pose position
            self.x = msg.pose.position.x
            self.y = msg.pose.position.y
            self.z = msg.pose.position.z

            # Convert into a numpy vector and apply the rotation
            input_pose = np.array([self.x, self.y, self.z])
            rotated_position = self.rotation_matrix.dot(input_pose)

            # Extract the original quaternion
            self.qx = msg.pose.orientation.x
            self.qy = msg.pose.orientation.y
            self.qz = msg.pose.orientation.z
            self.qw = msg.pose.orientation.w
            
            # Convert into a numpy vector
            input_quat = np.array([self.qx ,  self.qy, self.qz, self.qw])
            
            # Convert the original quaternion to a 3x3 rotation matrix
            orig_orientation_matrix = quat2mat(input_quat)

            rotated_orientation_matrix = self.rotation_matrix.dot(orig_orientation_matrix)

            rotated_quat = mat2quat(rotated_orientation_matrix)

            # Publish the rotated point cloud
            self.publisher.publish_rotate_pose(rotated_position, rotated_quat, self.rotate_pose_processing)

            # Log debug information
            self.get_logger().debug(f"Published rotated pose.")
        except Exception as e:
            self.get_logger().error(f"Error: {e}")
