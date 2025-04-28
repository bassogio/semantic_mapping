# -----------------------------------
# Import Statements
# -----------------------------------
import os
import yaml
import numpy as np
import struct
import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, CameraInfo, Image, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2

# -----------------------------------
# Configuration Loader Function
# -----------------------------------
def load_config():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, '../../config/point_cloud_config.yaml')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found at: {config_file}")

# -----------------------------------
# ROS2 Node Definition
# -----------------------------------
class PointCloudNode(Node):
    def __init__(self, config):
        super().__init__('point_cloud_node')

        # Load configuration
        self.task_config            = config['point_cloud_processing']
        self.point_cloud_topic      = self.task_config['point_cloud_topic']
        self.camera_parameters_topic= self.task_config['camera_parameters_topic']
        self.depth_image_topic      = self.task_config['depth_image_topic']
        self.semantic_image_topic   = self.task_config['semantic_image_topic']
        self.max_distance           = self.task_config['max_distance']
        self.depth_scale            = self.task_config['depth_scale']
        self.frame_id               = self.task_config['frame_id']
        self.subsample_step         = self.task_config.get('subsample_step', 2)

        # Declare ROS2 parameters
        self.declare_parameter('point_cloud_topic',       self.point_cloud_topic)
        self.declare_parameter('camera_parameters_topic', self.camera_parameters_topic)
        self.declare_parameter('depth_image_topic',       self.depth_image_topic)
        self.declare_parameter('semantic_image_topic',    self.semantic_image_topic)
        self.declare_parameter('max_distance',            self.max_distance)
        self.declare_parameter('depth_scale',             self.depth_scale)
        self.declare_parameter('frame_id',                self.frame_id)
        self.declare_parameter('subsample_step',          self.subsample_step)

        # Retrieve (possibly updated) parameters
        self.point_cloud_topic       = self.get_parameter('point_cloud_topic').value
        self.camera_parameters_topic = self.get_parameter('camera_parameters_topic').value
        self.depth_image_topic       = self.get_parameter('depth_image_topic').value
        self.semantic_image_topic    = self.get_parameter('semantic_image_topic').value
        self.max_distance            = self.get_parameter('max_distance').value
        self.depth_scale             = self.get_parameter('depth_scale').value
        self.frame_id                = self.get_parameter('frame_id').value
        self.subsample_step          = self.get_parameter('subsample_step').value

        # Placeholders
        self.depth_image    = None
        self.semantic_image = None

        # Flags
        self.received_camera   = False
        self.received_depth    = False
        self.received_semantic = False

        # UV‐grid initialization flag
        self.uv_initialized = False

        # CvBridge
        self.bridge = CvBridge()

        # Publisher
        self.point_cloud_publisher = self.create_publisher(
            PointCloud2, self.point_cloud_topic, 10
        )

        # Subscribers
        self.camera_subscription = self.create_subscription(
            CameraInfo,
            self.camera_parameters_topic,
            self.camera_callback,
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

        # Check subscriptions timer
        self.subscription_check_timer = self.create_timer(
            2.0, self.check_initial_subscriptions
        )

    def check_initial_subscriptions(self):
        not_received = []
        if not self.received_camera:
            not_received.append(f"'{self.camera_parameters_topic}'")
        if not self.received_depth:
            not_received.append(f"'{self.depth_image_topic}'")
        if not self.received_semantic:
            not_received.append(f"'{self.semantic_image_topic}'")
        if not_received:
            self.get_logger().info(
                f"Waiting for messages on: {', '.join(not_received)}"
            )
        else:
            self.get_logger().info(
                f"PointCloudNode ready on '{self.point_cloud_topic}' "
                f"(frame_id '{self.frame_id}')."
            )
            self.subscription_check_timer.cancel()

    def camera_callback(self, msg):
        if not self.received_camera:
            self.received_camera = True
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough'
            )
            if not self.received_depth:
                self.received_depth = True
            self.create_pointcloud()
        except Exception as e:
            self.get_logger().error(f"Error in depth_callback: {e}")

    def semantic_callback(self, msg):
        try:
            self.semantic_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='rgb8'
            )
            if not self.received_semantic:
                self.received_semantic = True
            self.create_pointcloud()
        except Exception as e:
            self.get_logger().error(f"Error in semantic_callback: {e}")

    def create_pointcloud(self):
        # Require both inputs
        if self.depth_image is None or self.semantic_image is None:
            return
        # Require intrinsics
        if not hasattr(self, 'fx'):
            self.get_logger().warn(
                "Camera parameters not initialized. Skipping point cloud creation."
            )
            return

        try:
            # Depth → meters
            depth = self.depth_image * self.depth_scale
            rows, cols = depth.shape

            # Precompute UV grid once
            if not self.uv_initialized:
                ys = np.arange(rows)[::self.subsample_step]
                xs = np.arange(cols)[::self.subsample_step]
                u, v = np.meshgrid(xs, ys)
                self.u = u.astype(np.float32)
                self.v = v.astype(np.float32)
                self.uv_initialized = True

            # Software subsample depth
            if self.subsample_step > 1:
                depth = depth[::self.subsample_step, ::self.subsample_step]

            # Resize + subsample semantic image
            sem = self.semantic_image
            if sem.shape[:2] != (rows, cols):
                sem = cv2.resize(sem, (cols, rows))
            if self.subsample_step > 1:
                sem = sem[::self.subsample_step, ::self.subsample_step]

            # Back‐project
            z = depth
            x = z * (self.u - self.cx) / self.fx
            y = z * (self.v - self.cy) / self.fy

            # Filter valid
            valid = (z > 0) & (z < self.max_distance)
            x, y, z = x[valid], y[valid], z[valid]

            # Pack color
            r = sem[..., 0][valid].astype(np.uint8)
            g = sem[..., 1][valid].astype(np.uint8)
            b = sem[..., 2][valid].astype(np.uint8)
            rgb_uint32 = (
                (r.astype(np.uint32) << 16)
                | (g.astype(np.uint32) << 8)
                | b.astype(np.uint32)
            )
            rgb_packed = rgb_uint32.view(np.float32)

            # Combine XYZ + RGB
            points = np.stack((x, y, z), axis=-1)
            points_with_color = np.column_stack((points, rgb_packed))

            # Static rotation (apply only to XYZ)
            R = np.array([
                [1,  0,  0],
                [0,  0, -1],
                [0, -1,  0],
            ], dtype=np.float32)
            xyz_rot = points_with_color[:, :3] @ R.T
            pts_final = np.column_stack((xyz_rot, points_with_color[:, 3]))

            # Define PointField layout
            field_x   = PointField(name="x",   offset=0,  datatype=PointField.FLOAT32, count=1)
            field_y   = PointField(name="y",   offset=4,  datatype=PointField.FLOAT32, count=1)
            field_z   = PointField(name="z",   offset=8,  datatype=PointField.FLOAT32, count=1)
            field_rgb = PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1)

            # Build and publish PointCloud2
            header = Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id=self.frame_id
            )
            cloud_msg = PointCloud2(
                header=header,
                height=1,
                width=pts_final.shape[0],
                fields=[field_x, field_y, field_z, field_rgb],
                is_bigendian=False,
                point_step=16,
                row_step=16 * pts_final.shape[0],
                data=pts_final.astype(np.float32).tobytes(),
                is_dense=True
            )
            self.point_cloud_publisher.publish(cloud_msg)
            self.get_logger().debug(
                f"Published RGB point cloud with {pts_final.shape[0]} points."
            )

        except Exception as e:
            self.get_logger().error(f"Error creating point cloud: {e}")

# -----------------------------------
# Main Entry Point
# -----------------------------------
def main(args=None):
    rclpy.init(args=args)
    config = load_config()
    node   = PointCloudNode(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
