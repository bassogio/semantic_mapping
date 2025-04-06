# point_cloud_processor
import rclpy
from rclpy.node import Node  
from sensor_msgs.msg import CameraInfo, Image
import numpy as np
from point_cloud_publisher import PointCloudPublisher
import std_msgs.msg
import cv2
from cv_bridge import CvBridge

class PointCloudProcessor(rclpy.node.Node):
    def __init__(self, config):
        super().__init__('point_cloud_processor')

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Access the config under 'point_cloud_processing'
        self.point_cloud_processing = config['point_cloud_processing']

        # Load configuration parameters
        self.depth_image_topic = self.point_cloud_processing['depth_image_topic']
        self.camera_parameters_topic = self.point_cloud_processing['camera_parameters_topic']
        self.point_cloud_topic = self.point_cloud_processing['point_cloud_topic']
        self.max_distance = self.point_cloud_processing['max_distance']
        self.depth_scale = self.point_cloud_processing['depth_scale']

        # Load the rotation matrix from the config
        rotation_matrix_config = self.point_cloud_processing['rotation_matrix']
        self.rotation_matrix = np.array(rotation_matrix_config)

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

    def camera_info_callback(self, msg):
        # Extract camera parameters
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def point_cloud_callback(self, msg):
        """Callback to process the depth image and generate a point cloud."""
        # Ensure camera parameters are initialized
        if not self.fx or not self.fy or not self.cx or not self.cy:
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

            # Apply rotation to the points using the matrix from the config
            points_rotated = points @ self.rotation_matrix.T

            # Publish the rotated point cloud
            self.publisher.publish_point_cloud(points_rotated, self.point_cloud_processing)

            # Log debug information
            self.get_logger().debug(f"Published rotated point cloud with {points_rotated.shape[0]} points.")
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")
