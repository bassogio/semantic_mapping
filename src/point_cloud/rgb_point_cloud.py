#!/usr/bin/env python3
# -----------------------------------
# Import Statements
# -----------------------------------
import os
import yaml
import numpy as np
import struct  # For packing RGB values into a single 32-bit field
import cv2   # For resizing images
from cv_bridge import CvBridge
from transforms3d.quaternions import quat2mat
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, CameraInfo, Image, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2

# -----------------------------------
# Configuration Loader Function
# -----------------------------------
def load_config():
    """
    Load configuration parameters from the configuration file.

    Returns:
        dict: A dictionary containing configuration data.
    """
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, '../../config/point_cloud_config.yaml')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        raise FileNotFoundError(f"Config file not found at: {config_file}")

# -----------------------------------
# ROS2 Node Definition
# -----------------------------------
class PointCloudNode(Node):
    def __init__(self, config):
        # Initialize the node with a unique name.
        super().__init__('point_cloud_node')

        # -------------------------------------------
        # Access the configuration section.
        # -------------------------------------------
        self.task_config = config['point_cloud_processing']

        # -------------------------------------------
        # Load configuration parameters.
        # -------------------------------------------
        self.point_cloud_topic       = self.task_config['point_cloud_topic']
        self.camera_parameters_topic = self.task_config['camera_parameters_topic']
        self.depth_image_topic       = self.task_config['depth_image_topic']
        self.semantic_image_topic    = self.task_config['semantic_image_topic']
        self.max_distance            = self.task_config['max_distance']
        self.depth_scale             = self.task_config['depth_scale']
        self.frame_id                = self.task_config['frame_id']

        # -------------------------------------------
        # Declare ROS2 parameters for runtime modification.
        # -------------------------------------------
        self.declare_parameter('point_cloud_topic',       self.point_cloud_topic)
        self.declare_parameter('camera_parameters_topic', self.camera_parameters_topic)
        self.declare_parameter('depth_image_topic',       self.depth_image_topic)
        self.declare_parameter('semantic_image_topic',    self.semantic_image_topic)
        self.declare_parameter('max_distance',            self.max_distance)
        self.declare_parameter('depth_scale',             self.depth_scale)
        self.declare_parameter('frame_id',                self.frame_id)

        # Retrieve the (possibly updated) parameter values.
        self.point_cloud_topic       = self.get_parameter('point_cloud_topic').value
        self.camera_parameters_topic = self.get_parameter('camera_parameters_topic').value
        self.depth_image_topic       = self.get_parameter('depth_image_topic').value
        self.semantic_image_topic    = self.get_parameter('semantic_image_topic').value
        self.max_distance            = self.get_parameter('max_distance').value
        self.depth_scale             = self.get_parameter('depth_scale').value
        self.frame_id                = self.get_parameter('frame_id').value

        # -------------------------------------------
        # Initialize additional attributes.
        # -------------------------------------------
        # Camera intrinsics will be set from the camera info callback.
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Placeholders for the depth and semantic images.
        self.depth_image = None
        self.semantic_image = None

        # Flags for tracking if at least one message has been received.
        self.received_camera   = False
        self.received_depth    = False
        self.received_semantic = False

        # -------------------------------------------
        # Initialize CvBridge once for the node.
        # -------------------------------------------
        self.bridge = CvBridge()

        # -------------------------------------------
        # Create a Publisher.
        # -------------------------------------------
        self.point_cloud_publisher = self.create_publisher(PointCloud2, self.point_cloud_topic, 10)

        # -------------------------------------------
        # Create Subscribers.
        # -------------------------------------------
        self.camera_subscription = self.create_subscription(
            CameraInfo,                      # Message type.
            self.camera_parameters_topic,    # Topic name.
            self.camera_callback,            # Callback function.
            10                               # Queue size.
        )

        self.depth_image_subscription = self.create_subscription(
            Image,                           # Message type.
            self.depth_image_topic,          # Topic name.
            self.depth_callback,             # Callback function.
            10                               # Queue size.
        )

        self.semantic_image_subscription = self.create_subscription(
            Image,                           # Message type.
            self.semantic_image_topic,       # Topic name.
            self.semantic_callback,          # Callback function.
            10                               # Queue size.
        )

        # -------------------------------------------
        # Create a Timer to check if subscribed topics have received at least one message.
        # -------------------------------------------
        self.subscription_check_timer = self.create_timer(2.0, self.check_initial_subscriptions)

    # -------------------------------------------
    # Timer Callback to Check Initial Subscriptions
    # -------------------------------------------
    def check_initial_subscriptions(self):
        not_received = []
        if not self.received_camera:
            not_received.append(f"'{self.camera_parameters_topic}'")
        if not self.received_depth:
            not_received.append(f"'{self.depth_image_topic}'")
        if not self.received_semantic:
            not_received.append(f"'{self.semantic_image_topic}'")
        if not_received:
            self.get_logger().info(f"Waiting for messages on topics: {', '.join(not_received)}")
        else:
            self.get_logger().info(
                f"All subscribed topics have received messages. "
                f"PointCloudNode started with publisher on '{self.point_cloud_topic}' and frame_id '{self.frame_id}'."
            )
            self.subscription_check_timer.cancel()

    # -------------------------------------------
    # Camera Callback Function
    # -------------------------------------------
    def camera_callback(self, msg):
        if not self.received_camera:
            self.received_camera = True
        # Extract camera intrinsics.
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    # -------------------------------------------
    # Depth Image Callback Function
    # -------------------------------------------
    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if not self.received_depth:
                self.received_depth = True
            self.create_pointcloud()  # Attempt to create the point cloud if semantic image is available.
        except Exception as e:
            self.get_logger().error(f"Error in depth_callback: {e}")

    # -------------------------------------------
    # Semantic Image Callback Function
    # -------------------------------------------
    def semantic_callback(self, msg):
        try:
            # Assumes semantic image is in 'rgb8' encoding.
            self.semantic_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if not self.received_semantic:
                self.received_semantic = True
            self.create_pointcloud()  # Attempt to create the point cloud if depth image is available.
        except Exception as e:
            self.get_logger().error(f"Error in semantic_callback: {e}")

    # -------------------------------------------
    # Function to Create and Publish the RGB Point Cloud
    # -------------------------------------------
    def create_pointcloud(self):
        # Ensure both depth and semantic images are available.
        if self.depth_image is None or self.semantic_image is None:
            return

        # Ensure camera parameters are initialized.
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            self.get_logger().warn("Camera parameters not initialized. Skipping point cloud creation.")
            return

        try:
            # Validate that the depth_image is 2D.
            if len(self.depth_image.shape) != 2:
                self.get_logger().error(f"Invalid depth image shape: {self.depth_image.shape}. Expected a 2D image.")
                return

            # Apply depth scaling to convert raw depth values to meters.
            depth = self.depth_image * self.depth_scale
            rows, cols = depth.shape

            # Resize the semantic image to match the depth image dimensions if necessary.
            if self.semantic_image.shape[0] != rows or self.semantic_image.shape[1] != cols:
                semantic_resized = cv2.resize(self.semantic_image, (cols, rows))
            else:
                semantic_resized = self.semantic_image

            # Generate pixel index meshgrid.
            u, v = np.meshgrid(np.arange(cols), np.arange(rows))
            u = u.astype(np.float32)
            v = v.astype(np.float32)

            # Compute 3D coordinates using the pinhole camera model.
            z = depth
            x = z * (u - self.cx) / self.fx
            y = z * (v - self.cy) / self.fy

            # Create a boolean mask to filter out points beyond the maximum distance.
            valid = (z > 0) & (z < self.max_distance)
            x = x[valid]
            y = y[valid]
            z = z[valid]

            # Stack the valid 3D points into an (N, 3) array.
            points = np.stack((x, y, z), axis=-1)

            # Get corresponding color information from the semantic image.
            r_channel = semantic_resized[..., 0]
            g_channel = semantic_resized[..., 1]
            b_channel = semantic_resized[..., 2]
            r = r_channel[valid]
            g = g_channel[valid]
            b = b_channel[valid]

            # Pack the RGB values into one float (using IEEE 754 conversion).
            rgb_packed = np.array([
                struct.unpack('f', struct.pack('I', (int(r_val) << 16 | int(g_val) << 8 | int(b_val))))[0]
                for r_val, g_val, b_val in zip(r, g, b)
            ])

            # Combine 3D points with the RGB data.
            points_with_color = np.column_stack((points, rgb_packed))

            # Build a PointCloud2 message with fields: x, y, z, and rgb.
            header = Header()
            header.stamp    = self.get_clock().now().to_msg()
            header.frame_id = self.frame_id

            # Define the structure of the point cloud with PointField.
            field_x = PointField()
            field_x.name = "x"
            field_x.offset = 0
            field_x.datatype = PointField.FLOAT32
            field_x.count = 1

            field_y = PointField()
            field_y.name = "y"
            field_y.offset = 4
            field_y.datatype = PointField.FLOAT32
            field_y.count = 1

            field_z = PointField()
            field_z.name = "z"
            field_z.offset = 8
            field_z.datatype = PointField.FLOAT32
            field_z.count = 1

            field_rgb = PointField()
            field_rgb.name = "rgb"
            field_rgb.offset = 12
            field_rgb.datatype = PointField.FLOAT32
            field_rgb.count = 1

            fields = [field_x, field_y, field_z, field_rgb]

            point_step = 16  # 4 fields * 4 bytes each.
            data = points_with_color.astype(np.float32).tobytes()

            cloud_msg = PointCloud2()
            cloud_msg.header = header
            cloud_msg.height = 1
            cloud_msg.width = points_with_color.shape[0]
            cloud_msg.fields = fields
            cloud_msg.is_bigendian = False
            cloud_msg.point_step = point_step
            cloud_msg.row_step = point_step * points_with_color.shape[0]
            cloud_msg.data = data
            cloud_msg.is_dense = True

            # Publish the RGB point cloud.
            self.point_cloud_publisher.publish(cloud_msg)
            self.get_logger().debug(f"Published RGB point cloud with {points_with_color.shape[0]} points.")
        except Exception as e:
            self.get_logger().error(f"Error creating point cloud: {e}")

# -----------------------------------
# Main Entry Point
# -----------------------------------
def main(args=None):
    """
    The main function initializes the ROS2 system, loads configuration parameters,
    creates an instance of the PointCloudNode, and spins to process messages until shutdown.
    """
    rclpy.init(args=args)
    config = load_config()
    node = PointCloudNode(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass  # Graceful shutdown on CTRL+C.
    finally:
        node.destroy_node()
        rclpy.shutdown()

# -----------------------------------
# Run the node when the script is executed directly.
# -----------------------------------
if __name__ == '__main__':
    main()
