#point_cloud_processor
import rclpy
from rclpy.node import Node  
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
import numpy as np
from point_cloud_publisher import PointCloudPublisher
import std_msgs.msg
import cv2
from cv_bridge import CvBridge
import math
from config_loader import load_config

class PointCloudProcessor(rclpy.node.Node):
    def __init__(self, config):
        super().__init__('point_cloud_processor')

        # Load initial configuration
        self.config = config
        self.load_config_parameters()

        # Set up a timer to reload the config periodically (e.g., every 5 seconds)
        self.timer = self.create_timer(5.0, self.reload_config)  # Reload config every 5 seconds

        # Access the config under 'point_cloud_processing'
        point_cloud_processing = self.config['point_cloud_processing']

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Load configuration parameters
        self.depth_image_topic = point_cloud_processing['depth_image_topic']
        self.camera_parameters_topic = point_cloud_processing['camera_parameters_topic']
        self.point_cloud_topic = point_cloud_processing['point_cloud_topic']
        self.max_distance = point_cloud_processing['max_distance']
        self.depth_scale = point_cloud_processing['depth_scale']

        # Load the rotation matrix from the config
        rotation_matrix_config = point_cloud_processing['rotation_matrix']
        self.rotation_matrix = np.array(rotation_matrix_config)

        # Pose information
        self.current_position = np.zeros(3)  # Default to zeros matrix 
        self.current_orientation = np.eye(3)  # Default to identity matrix (no rotation)

        # Declare parameters for ROS 2
        self.declare_parameter('depth_image_topic', self.depth_image_topic)
        self.declare_parameter('camera_parameters_topic', self.camera_parameters_topic)
        self.declare_parameter('point_cloud_topic', self.point_cloud_topic)

        # Initialize the PointCloudPublisher
        self.publisher = PointCloudPublisher(self, self.point_cloud_topic)

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Create subscriptions
        self.create_subscription(
            CameraInfo, 
            self.camera_parameters_topic, 
            self.camera_info_callback, 10)
        
        self.create_subscription(
            Image,  
            self.depth_image_topic,
            self.point_cloud_callback, 10)

        self.create_subscription(
            PoseStamped, 
            '/davis/left/pose_fixed', 
            self.pose_callback, 10)

    def load_config_parameters(self):
        # Load the rotation matrix and other parameters from the current config
        point_cloud_processing = self.config['point_cloud_processing']
        self.rotation_matrix = np.array(point_cloud_processing['rotation_matrix'])
        self.depth_image_topic = point_cloud_processing['depth_image_topic']
        self.camera_parameters_topic = point_cloud_processing['camera_parameters_topic']
        self.point_cloud_topic = point_cloud_processing['point_cloud_topic']
        self.max_distance = point_cloud_processing['max_distance']
        self.depth_scale = point_cloud_processing['depth_scale']

    def reload_config(self):
        # Reload the config and update the parameters
        self.config = load_config()  # Reload the config file
        self.load_config_parameters()
        self.get_logger().info("Config reloaded successfully.")

    def camera_info_callback(self, msg):
        """Extract camera parameters."""
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def pose_callback(self, msg):
        """Callback to process the pose and extract the rotation matrix."""
        # Extract position and orientation quaternion from pose
        self.current_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        q = msg.pose.orientation

        # Convert quaternion to rotation matrix
        self.current_orientation = self.quaternion_to_rotation_matrix(q)

    def quaternion_to_rotation_matrix(self, q):
        """Convert a quaternion to a 3x3 rotation matrix."""
        # Extract quaternion components
        qw = q.w
        qx = q.x
        qy = q.y
        qz = q.z

        # Compute rotation matrix from quaternion
        R = np.array([
            [1 -  2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 -  2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 -  2 * (qx**2 + qy**2)]
        ])

        return R

    def point_cloud_callback(self, msg):
        """Callback to process the depth image and generate a point cloud."""
        # Ensure camera parameters are initialized
        if None in [self.fx, self.fy, self.cx, self.cy]:
            self.get_logger().warn("Camera parameters not initialized. Skipping point cloud processing.")
            return

        try:
            # Convert the depth image to an OpenCV format
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            # Validate depth image dimensions
            if len(depth_image.shape) != 2:
                self.get_logger().error(f"Invalid depth image shape: {depth_image.shape}. Expected a 2D image.")
                return

            # Apply depth scaling (convert raw depth to meters)
            depth = depth_image * self.depth_scale
            rows, cols = depth.shape

            # Create meshgrid for pixel indices
            u, v = np.meshgrid(np.arange(cols), np.arange(rows))
            u = u.astype(np.float32)
            v = v.astype(np.float32)

            # Compute 3D coordinates using the pinhole camera model
            z = depth
            x = z * (u - self.cx) / self.fx
            y = z * (v - self.cy) / self.fy

            # Filter out points beyond the maximum distance or invalid values
            valid = (z > 0) & (z < self.max_distance)
            x = x[valid]
            y = y[valid]
            z = z[valid]

            # Combine the valid 3D points into an (N, 3) numpy array
            points = np.stack((x, y, z), axis=-1)
            
            # Apply the pre-configured rotation matrix
            #points_rotated = points @ self.rotation_matrix.T

            # Apply rotation to the points using the current pose's rotation matrix
            points_rotated = points @ self.current_orientation.T

            

            # Apply translation to the rotated points
            points_transformed = points_rotated + self.current_position

            # Publish the transformed point cloud
            self.publisher.publish_point_cloud(points_rotated)

            # Log debug information
            self.get_logger().debug(f"Published rotated point cloud with {points_rotated.shape[0]} points.")
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")
