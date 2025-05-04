# rgb_point_cloud_processor.py

# -----------------------------------
# Import Statements
# -----------------------------------
import os
import yaml
import numpy as np
import struct  # left for compatibility, no longer used in packing
import cv2   # For resizing images
from cv_bridge import CvBridge
from transforms3d.quaternions import quat2mat
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, CameraInfo, Image, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
import sensor_msgs_py.point_cloud2 as pc2

# -----------------------------------
# Configuration Loader Function
# -----------------------------------
def load_config():
    """
    Load configuration parameters from the configuration file.
    """
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, '../../../config/point_cloud_config.yaml')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found at: {config_file}")

# -----------------------------------
# Main Node Class
# -----------------------------------
class PointCloudNode(Node):
    def __init__(self, config):
        super().__init__('point_cloud_node')
        self.task_config = config['point_cloud_processing']

        # Topics and parameters
        self.point_cloud_topic       = self.task_config['point_cloud_topic']
        self.camera_parameters_topic = self.task_config['camera_parameters_topic']
        self.depth_image_topic       = self.task_config['depth_image_topic']
        self.pose_topic              = self.task_config['pose_topic']
        self.semantic_image_topic    = self.task_config['semantic_image_topic']
        self.max_distance            = self.task_config['max_distance']
        self.depth_scale             = self.task_config['depth_scale']
        self.frame_id                = self.task_config['frame_id']

        # Declare and retrieve ROS2 parameters (allows runtime overrides)
        self.declare_parameter('point_cloud_topic',       self.point_cloud_topic)
        self.declare_parameter('camera_parameters_topic', self.camera_parameters_topic)
        self.declare_parameter('depth_image_topic',       self.depth_image_topic)
        self.declare_parameter('pose_topic',              self.pose_topic)
        self.declare_parameter('semantic_image_topic',    self.semantic_image_topic)
        self.declare_parameter('max_distance',            self.max_distance)
        self.declare_parameter('depth_scale',             self.depth_scale)
        self.declare_parameter('frame_id',                self.frame_id)

        self.point_cloud_topic       = self.get_parameter('point_cloud_topic').value
        self.camera_parameters_topic = self.get_parameter('camera_parameters_topic').value
        self.depth_image_topic       = self.get_parameter('depth_image_topic').value
        self.pose_topic              = self.get_parameter('pose_topic').value
        self.semantic_image_topic    = self.get_parameter('semantic_image_topic').value
        self.max_distance            = self.get_parameter('max_distance').value
        self.depth_scale             = self.get_parameter('depth_scale').value
        self.frame_id                = self.get_parameter('frame_id').value

        # -------------------------------------------
        # Initialize pose & quaternion
        # -------------------------------------------
        self.Qw     = 1.0  # identity quaternion
        self.Qx     = 0.0
        self.Qy     = 0.0
        self.Qz     = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_z = 0.0

        # Placeholders for incoming data
        self.depth_image    = None
        self.semantic_image = None
        self.received_camera   = False
        self.received_pose     = False
        self.received_depth    = False
        self.received_semantic = False

        # -------------------------------------------
        # Initialize CvBridge once for the node.
        # -------------------------------------------
        self.bridge = CvBridge()

        # ----------------------------------------------------------------
        # —– Begin optimization additions —–
        # ----------------------------------------------------------------
        # UV‐grid cache (to avoid regenerating meshgrid every frame)
        self.uv_initialized = False
        self.u = None
        self.v = None

        # Static PointField list (reuse instead of rebuilding per frame)
        self.fields = [
            PointField(name='x',   offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',   offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',   offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        # —– End optimization additions —–

        # -------------------------------------------
        # Publisher & Subscribers
        # -------------------------------------------
        self.point_cloud_publisher = self.create_publisher(PointCloud2, self.point_cloud_topic, 10)
        self.camera_subscription   = self.create_subscription(
            CameraInfo, self.camera_parameters_topic, self.camera_callback, 10
        )
        self.pose_subscription     = self.create_subscription(
            PoseStamped, self.pose_topic, self.pose_callback, 10
        )
        self.depth_subscription    = self.create_subscription(
            Image, self.depth_image_topic, self.depth_callback, 10
        )
        self.semantic_subscription = self.create_subscription(
            Image, self.semantic_image_topic, self.semantic_callback, 10
        )

    # -------------------------------------------
    # CameraInfo Callback Function
    # -------------------------------------------
    def camera_callback(self, msg: CameraInfo):
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]
        self.received_camera = True

    # -------------------------------------------
    # Pose Callback Function
    # -------------------------------------------
    def pose_callback(self, msg: PoseStamped):
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
    def depth_callback(self, msg: Image):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if not self.received_depth:
                self.received_depth = True
            self.create_pointcloud()
        except Exception as e:
            self.get_logger().error(f"Error in depth_callback: {e}")

    # -------------------------------------------
    # Semantic Image Callback Function
    # -------------------------------------------
    def semantic_callback(self, msg: Image):
        try:
            self.semantic_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            if not self.received_semantic:
                self.received_semantic = True
            self.create_pointcloud()
        except Exception as e:
            self.get_logger().error(f"Error in semantic_callback: {e}")

    # -------------------------------------------
    # Function to Create and Publish the RGB Point Cloud
    # -------------------------------------------
    def create_pointcloud(self):
        # Need depth, semantic & camera parameters
        if (self.depth_image is None or self.semantic_image is None or
            not getattr(self, 'fx', None) or not getattr(self, 'cx', None)):
            return

        try:
            # 1) Scale depth to meters
            depth = self.depth_image * self.depth_scale
            rows, cols = depth.shape

            # 2) Initialize & cache UV grid once
            if not self.uv_initialized:
                u_grid, v_grid = np.meshgrid(np.arange(cols), np.arange(rows))
                self.u = u_grid.astype(np.float32)
                self.v = v_grid.astype(np.float32)
                self.uv_initialized = True

            # 3) Resize semantic image if needed
            if (self.semantic_image.shape[0] != rows or
                self.semantic_image.shape[1] != cols):
                sem = cv2.resize(self.semantic_image, (cols, rows))
            else:
                sem = self.semantic_image

            # 4) Back-project to XYZ
            z = depth
            x = z * (self.u - self.cx) / self.fx
            y = z * (self.v - self.cy) / self.fy

            # 5) Mask valid points
            valid = (z > 0) & (z < self.max_distance)
            xyz = np.stack((x[valid], y[valid], z[valid]), axis=-1)

            # 6) Vectorized RGB packing
            r = sem[..., 0][valid].astype(np.uint32)
            g = sem[..., 1][valid].astype(np.uint32)
            b = sem[..., 2][valid].astype(np.uint32)
            rgb_uint32 = (r << 16) | (g << 8) | b
            rgb_packed = rgb_uint32.view(np.float32)

            # 7) Combine into N×4 array
            pts = np.column_stack((xyz, rgb_packed))

            # 8) Apply rotation & translation
            R = quat2mat([self.Qw, self.Qx, self.Qy, self.Qz])
            pts[:, :3] = pts[:, :3] @ R.T
            pts[:, :3] += np.array([self.pose_x, self.pose_y, self.pose_z],
                                   dtype=np.float32)

            # 9) Publish via create_cloud
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.frame_id
            cloud_msg = pc2.create_cloud(header, self.fields, pts)
            self.point_cloud_publisher.publish(cloud_msg)

        except Exception as e:
            self.get_logger().error(f"Error creating point cloud: {e}")

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