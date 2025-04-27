#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------
# Import Statements
# -----------------------------------
import os
import yaml
import numpy as np
import struct  # For packing RGB values into a single 32-bit field
import cv2     # For resizing images
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, CameraInfo, Image, PointField
from std_msgs.msg import Header

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
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found at: {config_file}")

# -----------------------------------
# ROS2 Node Definition
# -----------------------------------
class PointCloudNode(Node):
    def __init__(self, config):
        super().__init__('point_cloud_node')
        cfg = config['point_cloud_processing']

        # Topics & parameters
        self.point_cloud_topic       = cfg['point_cloud_topic']
        self.camera_parameters_topic = cfg['camera_parameters_topic']
        self.depth_image_topic       = cfg['depth_image_topic']
        self.semantic_image_topic    = cfg['semantic_image_topic']
        self.max_distance            = cfg['max_distance']
        self.depth_scale             = cfg['depth_scale']
        self.frame_id                = cfg['frame_id']

        # Declare & retrieve ROS2 parameters
        for name, default in [
            ('point_cloud_topic',       self.point_cloud_topic),
            ('camera_parameters_topic', self.camera_parameters_topic),
            ('depth_image_topic',       self.depth_image_topic),
            ('semantic_image_topic',    self.semantic_image_topic),
            ('max_distance',            self.max_distance),
            ('depth_scale',             self.depth_scale),
            ('frame_id',                self.frame_id),
        ]:
            self.declare_parameter(name, default)

        self.point_cloud_topic       = self.get_parameter('point_cloud_topic').value
        self.camera_parameters_topic = self.get_parameter('camera_parameters_topic').value
        self.depth_image_topic       = self.get_parameter('depth_image_topic').value
        self.semantic_image_topic    = self.get_parameter('semantic_image_topic').value
        self.max_distance            = self.get_parameter('max_distance').value
        self.depth_scale             = self.get_parameter('depth_scale').value
        self.frame_id                = self.get_parameter('frame_id').value

        # Placeholders & flags
        self.bridge            = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_image       = None
        self.semantic_image    = None
        self.received_camera   = False
        self.received_depth    = False
        self.received_semantic = False

        # Publisher
        self.point_cloud_publisher = \
            self.create_publisher(PointCloud2, self.point_cloud_topic, 10)

        # Subscribers
        self.create_subscription(
            CameraInfo, self.camera_parameters_topic,
            self.camera_callback, 10
        )
        self.create_subscription(
            Image, self.depth_image_topic,
            self.depth_callback, 10
        )
        self.create_subscription(
            Image, self.semantic_image_topic,
            self.semantic_callback, 10
        )

        # Timer to ensure we've seen at least one message on each
        self.subscription_check_timer = self.create_timer(2.0, self.check_initial_subscriptions)

    def check_initial_subscriptions(self):
        missing = []
        if not self.received_camera:
            missing.append(self.camera_parameters_topic)
        if not self.received_depth:
            missing.append(self.depth_image_topic)
        if not self.received_semantic:
            missing.append(self.semantic_image_topic)

        if missing:
            self.get_logger().info(f"Waiting for messages on: {missing}")
        else:
            self.get_logger().info("All topics ready — publishing point clouds now.")
            self.subscription_check_timer.cancel()

    def camera_callback(self, msg: CameraInfo):
        if not self.received_camera:
            self.received_camera = True
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def depth_callback(self, msg: Image):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if not self.received_depth:
                self.received_depth = True
            self.create_pointcloud()
        except Exception as e:
            self.get_logger().error(f"Depth callback error: {e}")

    def semantic_callback(self, msg: Image):
        try:
            self.semantic_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if not self.received_semantic:
                self.received_semantic = True
            self.create_pointcloud()
        except Exception as e:
            self.get_logger().error(f"Semantic callback error: {e}")

    def create_pointcloud(self):
        # Ensure data and intrinsics are ready
        if (self.depth_image is None or
            self.semantic_image is None or
            None in (self.fx, self.fy, self.cx, self.cy)):
            return

        try:
            # Convert depth to meters
            depth = self.depth_image * self.depth_scale
            rows, cols = depth.shape

            # Resize semantic image if needed
            sem = (cv2.resize(self.semantic_image, (cols, rows))
                   if self.semantic_image.shape[:2] != depth.shape
                   else self.semantic_image)

            # Pixel grid
            u, v = np.meshgrid(np.arange(cols), np.arange(rows))
            z = depth
            x = z * (u - self.cx) / self.fx
            y = z * (v - self.cy) / self.fy

            # Filter valid points
            valid = (z > 0) & (z < self.max_distance)
            x, y, z = x[valid], y[valid], z[valid]
            pts = np.stack((x, y, z), axis=-1)

            # ------ Apply your fixed rotation here ------
            R = np.array([[1,  0,  0],
                          [0,  0, -1],
                          [0,  -1,  0]], dtype=np.float32)
            pts = pts @ R.T

            # Extract colors
            r = sem[..., 0][valid].astype(np.uint8)
            g = sem[..., 1][valid].astype(np.uint8)
            b = sem[..., 2][valid].astype(np.uint8)

            # Pack into float
            rgb_packed = np.array([
                struct.unpack('f',
                    struct.pack('I',
                        (int(rv) << 16) |
                        (int(gv) << 8)  |
                        int(bv)
                    )
                )[0]
                for rv, gv, bv in zip(r, g, b)
            ])

            # Combine coords + color
            final_pts = np.column_stack((pts, rgb_packed))

            # Build PointCloud2
            header = Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id=self.frame_id
            )

            fields = []
            for idx, name in enumerate(['x','y','z','rgb']):
                f = PointField()
                f.name     = name
                f.offset   = idx * 4
                f.datatype = PointField.FLOAT32
                f.count    = 1
                fields.append(f)

            cloud = PointCloud2(
                header=header,
                height=1,
                width=final_pts.shape[0],
                fields=fields,
                is_bigendian=False,
                point_step=16,
                row_step=16 * final_pts.shape[0],
                data=final_pts.astype(np.float32).tobytes(),
                is_dense=True
            )

            self.point_cloud_publisher.publish(cloud)
            self.get_logger().debug(f"Published {final_pts.shape[0]} points.")
        except Exception as e:
            self.get_logger().error(f"Error creating point cloud: {e}")

def main(args=None):
    rclpy.init(args=args)
    cfg = load_config()
    node = PointCloudNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
