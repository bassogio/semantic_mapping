# point_cloud_processor
import rclpy
from rclpy.node import Node  
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseStamped
import numpy as np
from point_cloud_publisher import PointCloudPublisher
import std_msgs.msg
import cv2
from cv_bridge import CvBridge
from transforms3d.quaternions import quat2mat

class PointCloudProcessor(rclpy.node.Node):
    def __init__(self, config):
        super().__init__('point_cloud_processor')

        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        
        self.Qx = 0.0
        self.Qy = 0.0
        self.Qz = 0.0
        self.Qw = 0.0

        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_z = 0.0

        # Access the config 
        self.point_cloud_processing = config['point_cloud_processing']

        # Load configuration parameters
        self.depth_image_topic = self.point_cloud_processing['depth_image_topic']
        self.camera_parameters_topic = self.point_cloud_processing['camera_parameters_topic']
        self.point_cloud_topic = self.point_cloud_processing['point_cloud_topic']
        self.max_distance = self.point_cloud_processing['max_distance']
        self.depth_scale = self.point_cloud_processing['depth_scale']
        self.pose_topic = self.point_cloud_processing['pose_topic']

        # Load the rotation matrix from the config
        # rotation_matrix_config = self.point_cloud_processing['rotation_matrix']
        # self.rotation_matrix = np.array(rotation_matrix_config)

        # Declare parameters for ROS 2
        self.declare_parameter('pose_topic', self.pose_topic)
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
            PoseStamped, 
            self.pose_topic, 
            self.pose_callback, 10)

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

    def pose_callback(self,msg):
        # Update quaternion components from the pose message
        self.Qx = msg.pose.orientation.x
        self.Qy = msg.pose.orientation.y
        self.Qz = msg.pose.orientation.z
        self.Qw = msg.pose.orientation.w
        # Update position (translation) components from the pose message
        self.pose_x = msg.pose.position.x
        self.pose_y = msg.pose.position.y
        self.pose_z = msg.pose.position.z


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
            
            # Convert the quaternion to a 3x3 rotation matrix.
            # transforms3d expects quaternion order as (w, x, y, z)
            quat = [self.Qw, self.Qx, self.Qy, self.Qz]
            rotation_matrix = quat2mat(quat)

            # Apply the rotation from the quaternion to the point cloud
            points_rotated = points @ rotation_matrix.T
            
            # Now apply the translation from the pose (after rotation)
            translation = np.array([self.pose_x, self.pose_y, self.pose_z])
            points_transformed = points_rotated + translation 

            # Publish the rotated point cloud
            self.publisher.publish_point_cloud(points_transformed, self.point_cloud_processing)

            # Log debug information
            self.get_logger().debug(f"Published rotated point cloud with {points_rotated.shape[0]} points.")
        except Exception as e:
            self.get_logger().error(f"Error: {e}")
