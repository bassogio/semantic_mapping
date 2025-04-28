# -----------------------------------
# Import Statements
# -----------------------------------
import os
import yaml
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2

# -----------------------------------
# Configuration Loader Function
# -----------------------------------
def load_config():
    """
    Load configuration parameters from the configuration file.
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

        # -------------------------------------------
        # Load configuration parameters
        # -------------------------------------------
        self.task_config = config['point_cloud_processing']
        self.point_cloud_topic       = self.task_config['point_cloud_topic']
        self.camera_parameters_topic = self.task_config['camera_parameters_topic']
        self.depth_image_topic       = self.task_config['depth_image_topic']
        self.max_distance            = self.task_config['max_distance']
        self.depth_scale             = self.task_config['depth_scale']
        self.frame_id                = self.task_config['frame_id']
        # New: subsampling step (1 = no subsampling)
        self.subsample_step          = self.task_config.get('subsample_step', 2)

        # -------------------------------------------
        # Declare ROS2 parameters
        # -------------------------------------------
        self.declare_parameter('subsample_step', self.subsample_step)
        self.subsample_step = self.get_parameter('subsample_step').value
        self.declare_parameter('point_cloud_topic',       self.point_cloud_topic)
        self.declare_parameter('camera_parameters_topic', self.camera_parameters_topic)
        self.declare_parameter('depth_image_topic',       self.depth_image_topic)
        self.declare_parameter('max_distance',            self.max_distance)
        self.declare_parameter('depth_scale',             self.depth_scale)
        self.declare_parameter('frame_id',                self.frame_id)

        # Retrieve possibly updated parameters
        self.point_cloud_topic       = self.get_parameter('point_cloud_topic').value
        self.camera_parameters_topic = self.get_parameter('camera_parameters_topic').value
        self.depth_image_topic       = self.get_parameter('depth_image_topic').value
        self.max_distance            = self.get_parameter('max_distance').value
        self.depth_scale             = self.get_parameter('depth_scale').value
        self.frame_id                = self.get_parameter('frame_id').value

        # -------------------------------------------
        # Initialize CvBridge, publisher, subscribers
        # -------------------------------------------
        self.bridge = CvBridge()
        self.point_cloud_publisher = self.create_publisher(PointCloud2, self.point_cloud_topic, 10)
        self.camera_subscription   = self.create_subscription(
            CameraInfo,
            self.camera_parameters_topic,
            self.camera_callback,
            10
        )
        self.depth_subscription    = self.create_subscription(
            Image,
            self.depth_image_topic,
            self.point_cloud_callback,
            10
        )

        # Flags
        self.received_camera = False
        self.received_depth  = False
        # UV grid initialization flag
        self.uv_initialized  = False

        # Timer to check initial subscriptions
        self.subscription_check_timer = self.create_timer(2.0, self.check_initial_subscriptions)

    def check_initial_subscriptions(self):
        not_received = []
        if not self.received_camera:
            not_received.append(f"'{self.camera_parameters_topic}'")
        if not self.received_depth:
            not_received.append(f"'{self.depth_image_topic}'")
        if not_received:
            self.get_logger().info(f"Waiting for messages on topics: {', '.join(not_received)}")
        else:
            self.get_logger().info(
                f"All subscribed topics received. Publisher on '{self.point_cloud_topic}' (frame_id '{self.frame_id}')."
            )
            self.subscription_check_timer.cancel()

    def camera_callback(self, msg):
        if not self.received_camera:
            self.received_camera = True
        self.fx, self.fy = msg.k[0], msg.k[4]
        self.cx, self.cy = msg.k[2], msg.k[5]

    def point_cloud_callback(self, msg):
        if not self.received_depth:
            self.received_depth = True
        # Ensure camera intrinsics are available
        if not hasattr(self, 'fx'):
            self.get_logger().warn("Camera parameters not initialized. Skipping point cloud.")
            return
        try:
            # Decode depth image
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            if depth_image.ndim != 2:
                self.get_logger().error(f"Invalid depth image shape: {depth_image.shape}")
                return

            depth = depth_image * self.depth_scale

            # Precompute UV meshgrid on first frame
            if not self.uv_initialized:
                rows, cols = depth.shape
                ys = np.arange(rows)[::self.subsample_step]
                xs = np.arange(cols)[::self.subsample_step]
                u, v = np.meshgrid(xs, ys)
                self.u = u.astype(np.float32)
                self.v = v.astype(np.float32)
                self.uv_initialized = True

            # Subsample depth
            if self.subsample_step > 1:
                depth = depth[::self.subsample_step, ::self.subsample_step]

            # Backproject to 3D
            z = depth
            x = z * (self.u - self.cx) / self.fx
            y = z * (self.v - self.cy) / self.fy

            # Filter invalid points
            valid = (z > 0) & (z < self.max_distance)
            xyz = np.stack((x[valid], y[valid], z[valid]), axis=-1)
            
            # ------ Apply your fixed rotation here ------
            R = np.array([[1,  0,  0],
                          [0,  0, -1],
                          [0,  -1,  0]], dtype=np.float32)
            xyz = xyz @ R.T

            # Build and publish PointCloud2
            header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame_id)
            cloud_msg = pc2.create_cloud_xyz32(header, xyz)
            self.point_cloud_publisher.publish(cloud_msg)

            self.get_logger().debug(f"Published point cloud with {xyz.shape[0]} points.")
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {e}")

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
