# segmentation_processor
import rclpy
from rclpy.node import Node  
from sensor_msgs.msg import Image
import numpy as np
from segmentation_publisher import SegmentationPublisher
from cv_bridge import CvBridge

class SegmentationProcessor(rclpy.node.Node):
    def __init__(self, config):
        super().__init__('segmentation_processor')

        # Access the config under 'point_cloud_processing'
        self.segmentation_processing = config['segmentation_processing']

        # Load configuration parameters
        self.raw_image_topic = self.segmentation_processing['raw_image_topic']
        self.segmentation_topic = self.segmentation_processing['segmentation_topic']

        # Declare parameters for ROS 2
        self.declare_parameter('raw_image_topic', self.raw_image_topic)
        self.declare_parameter('segmentation_topic', self.segmentation_topic)

        # Initialize the SegmentationPublisher
        self.publisher = SegmentationPublisher(self, self.segmentation_topic)

        # Initialize CvBridge
        self.bridge = CvBridge()

        self.create_subscription(
            Image,  
            self.raw_image_topic,
            self.segmentation_callback, 10)

    def point_cloud_callback(self, msg):
        """Callback to process the raw image and generate a segmentated image."""
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
