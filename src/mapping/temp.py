#!/usr/bin/env python3
# -----------------------------------
# Import Statements
# -----------------------------------
import os
import yaml
import numpy as np
import struct  # For packing/unpacking RGB values as a single 32-bit field
import cv2   # For resizing images
from cv_bridge import CvBridge
from transforms3d.quaternions import quat2mat
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, CameraInfo, Image, PointField
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
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
# Helper function to unpack an RGB-packed float
# -----------------------------------
def unpack_rgb(rgb_float):
    i = struct.unpack('I', struct.pack('f', rgb_float))[0]
    r = ((i >> 16) & 0xFF) / 255.0
    g = ((i >> 8) & 0xFF) / 255.0
    b = (i & 0xFF) / 255.0
    return r, g, b

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
        self.pose_topic              = self.task_config['pose_topic']
        self.semantic_image_topic    = self.task_config['semantic_image_topic']
        self.max_distance            = self.task_config['max_distance']
        self.depth_scale             = self.task_config['depth_scale']
        self.frame_id                = self.task_config['frame_id']
        # Grid resolution for the occupancy grid markers (in meters).
        self.grid_resolution = self.task_config.get('grid_resolution', 0.1)

        # -------------------------------------------
        # Declare ROS2 parameters for runtime modification.
        # -------------------------------------------
        self.declare_parameter('point_cloud_topic',       self.point_cloud_topic)
        self.declare_parameter('camera_parameters_topic', self.camera_parameters_topic)
        self.declare_parameter('depth_image_topic',       self.depth_image_topic)
        self.declare_parameter('pose_topic',              self.pose_topic)
        self.declare_parameter('semantic_image_topic',    self.semantic_image_topic)
        self.declare_parameter('max_distance',            self.max_distance)
        self.declare_parameter('depth_scale',             self.depth_scale)
        self.declare_parameter('frame_id',                self.frame_id)
        self.declare_parameter('grid_resolution',         self.grid_resolution)

        # Retrieve the (possibly updated) parameter values.
        self.point_cloud_topic       = self.get_parameter('point_cloud_topic').value
        self.camera_parameters_topic = self.get_parameter('camera_parameters_topic').value
        self.depth_image_topic       = self.get_parameter('depth_image_topic').value
        self.pose_topic              = self.get_parameter('pose_topic').value
        self.semantic_image_topic    = self.get_parameter('semantic_image_topic').value
        self.max_distance            = self.get_parameter('max_distance').value
        self.depth_scale             = self.get_parameter('depth_scale').value
        self.frame_id                = self.get_parameter('frame_id').value
        self.grid_resolution         = self.get_parameter('grid_resolution').value

        # -------------------------------------------
        # Initialize additional attributes needed for processing.
        # (Pose-related parameters are retained in this code in case they
        # are used in the point cloud; if not needed, they can be removed.)
        # -------------------------------------------
        self.Qw     = 1.0  # Default orientation (identity quaternion)
        self.Qx     = 0.0
        self.Qy     = 0.0
        self.Qz     = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_z = 0.0

        # Placeholders for images.
        self.depth_image = None
        self.semantic_image = None

        # Flags for tracking if at least one message has been received.
        self.received_camera   = False
        self.received_pose     = False
        self.received_depth    = False
        self.received_semantic = False

        # -------------------------------------------
        # Initialize CvBridge.
        # -------------------------------------------
        self.bridge = CvBridge()

        # -------------------------------------------
        # Create Publishers.
        # -------------------------------------------
        # Publisher for the RGB point cloud.
        self.point_cloud_publisher = self.create_publisher(PointCloud2, self.point_cloud_topic, 10)
        # Publisher for the occupancy grid markers.
        self.marker_pub = self.create_publisher(Marker, "occupancy_grid_markers", 10)

        # -------------------------------------------
        # Create Subscribers.
        # -------------------------------------------
        self.camera_subscription = self.create_subscription(
            CameraInfo,
            self.camera_parameters_topic,
            self.camera_callback,
            10
        )

        self.pose_subscription = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            10
        )

        self.depth_image_subscription = self.create_subscription(
            Image,
            self.depth_image_topic,
            self.depth_callback,
            10
        )

        self.semantic_image_subscription = self.create_subscription(
            Image,
            self.semantic_image_topic,
            self.semantic_callback,
            10
        )

        # -------------------------------------------
        # Create a Timer to verify initial subscriptions.
        # -------------------------------------------
        self.subscription_check_timer = self.create_timer(2.0, self.check_initial_subscriptions)

    # -------------------------------------------
    # Timer Callback to Check Initial Subscriptions
    # -------------------------------------------
    def check_initial_subscriptions(self):
        not_received = []
        if not self.received_camera:
            not_received.append(f"'{self.camera_parameters_topic}'")
        if not self.received_pose:
            not_received.append(f"'{self.pose_topic}'")
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
    # Pose Callback Function
    # -------------------------------------------
    def pose_callback(self, msg):
        if not self.received_pose:
            self.received_pose = True
        self.Qx = msg.pose.orientation.x
        self.Qy = msg.pose.orientation.y
        self.Qz = msg.pose.orientation.z
        self.Qw = msg.pose.orientation.w
        self.pose_x = msg.pose.position.x
        self.pose_y = msg.pose.position.y
        self.pose_z = msg.pose.position.z

    # -------------------------------------------
    # Depth Image Callback Function
    # -------------------------------------------
    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if not self.received_depth:
                self.received_depth = True
            self.create_pointcloud()  # Proceed if semantic image is available.
        except Exception as e:
            self.get_logger().error(f"Error in depth_callback: {e}")

    # -------------------------------------------
    # Semantic Image Callback Function
    # -------------------------------------------
    def semantic_callback(self, msg):
        try:
            self.semantic_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if not self.received_semantic:
                self.received_semantic = True
            self.create_pointcloud()  # Proceed if depth image is available.
        except Exception as e:
            self.get_logger().error(f"Error in semantic_callback: {e}")

    # -------------------------------------------
    # Function to Create and Publish the RGB Point Cloud
    # Also creates occupancy grid markers from the point cloud.
    # -------------------------------------------
    def create_pointcloud(self):
        # Verify that both images are available.
        if self.depth_image is None or self.semantic_image is None:
            return

        # Verify camera parameters.
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            self.get_logger().warn("Camera parameters not initialized. Skipping point cloud creation.")
            return

        try:
            # Ensure the depth image is 2D.
            if len(self.depth_image.shape) != 2:
                self.get_logger().error(f"Invalid depth image shape: {self.depth_image.shape}. Expected a 2D image.")
                return

            # Scale depth values (to meters).
            depth = self.depth_image * self.depth_scale
            rows, cols = depth.shape

            # Resize the semantic image to match depth dimensions if needed.
            if self.semantic_image.shape[0] != rows or self.semantic_image.shape[1] != cols:
                semantic_resized = cv2.resize(self.semantic_image, (cols, rows))
            else:
                semantic_resized = self.semantic_image

            # Generate meshgrid of pixel coordinates.
            u, v = np.meshgrid(np.arange(cols), np.arange(rows))
            u = u.astype(np.float32)
            v = v.astype(np.float32)

            # Compute 3D coordinates using the pinhole camera model.
            z = depth
            x = z * (u - self.cx) / self.fx
            y = z * (v - self.cy) / self.fy

            # Create a valid mask based on depth.
            valid = (z > 0) & (z < self.max_distance)
            x = x[valid]
            y = y[valid]
            z = z[valid]

            # Stack the valid 3D points into an (N, 3) array.
            points = np.stack((x, y, z), axis=-1)

            # Obtain corresponding RGB information from the semantic image.
            r_channel = semantic_resized[..., 0]
            g_channel = semantic_resized[..., 1]
            b_channel = semantic_resized[..., 2]
            r = r_channel[valid]
            g = g_channel[valid]
            b = b_channel[valid]

            # Pack the RGB values into one float per point.
            rgb_packed = np.array([
                struct.unpack('f', struct.pack('I', (int(r_val) << 16 | int(g_val) << 8 | int(b_val))))[0]
                for r_val, g_val, b_val in zip(r, g, b)
            ])

            # Combine 3D points with their packed color.
            points_with_color = np.column_stack((points, rgb_packed))

            # (Optional) Apply pose transformation.
            # Convert quaternion to rotation matrix.
            quat = [self.Qw, self.Qx, self.Qy, self.Qz]
            rotation_matrix = quat2mat(quat)
            pts_xyz = points_with_color[:, :3]
            pts_rotated = pts_xyz @ rotation_matrix.T
            translation = np.array([self.pose_x, self.pose_y, self.pose_z])
            pts_transformed = pts_rotated + translation
            # Recombine transformed coordinates with RGB.
            points_final = np.column_stack((pts_transformed, points_with_color[:, 3]))

            # Publish the full RGB point cloud.
            header = Header()
            header.stamp    = self.get_clock().now().to_msg()
            header.frame_id = self.frame_id

            # Define point fields.
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
            point_step = 16  # 4 fields * 4 bytes
            data = points_final.astype(np.float32).tobytes()

            cloud_msg = PointCloud2()
            cloud_msg.header = header
            cloud_msg.height = 1
            cloud_msg.width = points_final.shape[0]
            cloud_msg.fields = fields
            cloud_msg.is_bigendian = False
            cloud_msg.point_step = point_step
            cloud_msg.row_step = point_step * points_final.shape[0]
            cloud_msg.data = data
            cloud_msg.is_dense = True

            self.point_cloud_publisher.publish(cloud_msg)
            self.get_logger().debug(f"Published RGB point cloud with {points_final.shape[0]} points.")

            # Create and publish occupancy grid markers.
            self.create_occupancy_grid_markers(points_final)
        except Exception as e:
            self.get_logger().error(f"Error creating point cloud: {e}")

    # -------------------------------------------
    # Function to Create and Publish Occupancy Grid Markers
    # -------------------------------------------
    def create_occupancy_grid_markers(self, points_final):
        # Group points into grid cells.
        grid = {}
        for pt in points_final:
            x, y, z, rgb = pt
            # Compute grid cell index.
            cell_x = int(np.floor(x / self.grid_resolution))
            cell_y = int(np.floor(y / self.grid_resolution))
            key = (cell_x, cell_y)
            if key not in grid:
                grid[key] = {'points': [], 'colors': []}
            grid[key]['points'].append([x, y, z])
            grid[key]['colors'].append(rgb)

        cell_points = []
        cell_colors = []
        for key, group in grid.items():
            pts = np.array(group['points'])
            avg_pt = pts.mean(axis=0)
            rgbs = [unpack_rgb(c) for c in group['colors']]
            avg_rgb = np.mean(rgbs, axis=0)
            cell_points.append(avg_pt)
            cell_colors.append(avg_rgb)

        # Create a Marker of type CUBE_LIST.
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.frame_id
        marker.ns = "occupancy_grid"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        # Set the scale (size) of each cube to be the grid resolution.
        marker.scale.x = self.grid_resolution
        marker.scale.y = self.grid_resolution
        marker.scale.z = self.grid_resolution
        marker.pose.orientation.w = 1.0

        # Add a cube for each grid cell.
        for pt, col in zip(cell_points, cell_colors):
            p = Point()
            p.x, p.y, p.z = pt
            marker.points.append(p)
            color = ColorRGBA()
            color.r, color.g, color.b = col
            color.a = 1.0
            marker.colors.append(color)

        self.marker_pub.publish(marker)
        self.get_logger().debug(f"Published occupancy grid marker with {len(cell_points)} cells.")

# -----------------------------------
# Main Entry Point
# -----------------------------------
def main(args=None):
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
