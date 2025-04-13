# -----------------------------------
# Import Statements
# -----------------------------------
import os       
import yaml     
import numpy as np
from cv_bridge import CvBridge
from transforms3d.quaternions import quat2mat
import rclpy   
from rclpy.node import Node  
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, '../../config/point_cloud_config.yaml')
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        return {}

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
        self.point_cloud_topic = self.task_config['point_cloud_topic']
        self.camera_parameters_topic = self.task_config['camera_parameters_topic']
        self.depth_image_topic = self.task_config['depth_image_topic']
        self.pose_topic = self.task_config['pose_topic']
        self.max_distance = self.task_config['max_distance']
        self.depth_scale = self.task_config['depth_scale']
        self.frame_id = self.task_config['frame_id']
        
        # -------------------------------------------
        # Declare ROS2 parameters for runtime modification.
        # This allows you to change the parameters without restarting the node.
        # -------------------------------------------
        self.declare_parameter('point_cloud_topic', self.point_cloud_topic)
        self.declare_parameter('camera_parameters_topic', self.camera_parameters_topic)
        self.declare_parameter('depth_image_topic', self.depth_image_topic)
        self.declare_parameter('pose_topic', self.pose_topic)
        self.declare_parameter('max_distance', self.max_distance)
        self.declare_parameter('depth_scale', self.depth_scale)
        self.declare_parameter('frame_id', self.frame_id)
        
        # Retrieve the (possibly updated) parameter values.
        self.point_cloud_topic = self.get_parameter('point_cloud_topic').value
        self.camera_parameters_topic = self.get_parameter('camera_parameters_topic').value
        self.depth_image_topic = self.get_parameter('depth_image_topic').value
        self.pose_topic = self.get_parameter('pose_topic').value
        self.max_distance = self.get_parameter('max_distance').value
        self.depth_scale = self.get_parameter('depth_scale').value
        self.frame_id = self.get_parameter('frame_id').value     

        # -------------------------------------------
        # Initialize additional attributes needed for processing.
        # These will be updated when pose messages are received.
        # -------------------------------------------
        self.Qw = 1.0  # Default orientation (identity quaternion)
        self.Qx = 0.0
        self.Qy = 0.0
        self.Qz = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_z = 0.0

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
            CameraInfo,                        # Message type.
            self.camera_parameters_topic,      # Topic name.
            self.camera_callback,              # Callback function.
            10                                 # Queue size.
        )

        self.pose_subscription = self.create_subscription(
            PoseStamped,                       # Message type.
            self.pose_topic,             # Topic name.
            self.pose_callback,                # Callback function.
            10                                 # Queue size.
        )

        self.depth_image_subscription = self.create_subscription(
            Image,                             # Message type.
            self.depth_image_topic,             # Topic name.
            self.point_cloud_callback,         # Callback function.
            10                                 # Queue size.
        )

        self.get_logger().info(
            f"PointCloudNode started with publisher on '{self.point_cloud_topic}'."
            f"frame_id '{self.frame_id}'."
        )
        
    # -------------------------------------------
    # Camera Callback Function
    # -------------------------------------------
    def camera_callback(self, msg):
        # Extract camera parameters
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    # -------------------------------------------
    # Pose Callback Function
    # -------------------------------------------
    def pose_callback(self, msg):
        # Update quaternion components from the pose message
        self.Qx = msg.pose.orientation.x
        self.Qy = msg.pose.orientation.y
        self.Qz = msg.pose.orientation.z
        self.Qw = msg.pose.orientation.w
        # Update position (translation) components from the pose message
        self.pose_x = msg.pose.position.x
        self.pose_y = msg.pose.position.y
        self.pose_z = msg.pose.position.z

    # -------------------------------------------
    # Point Cloud Callback Function
    # -------------------------------------------
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

            # Create PointCloud2 message
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.frame_id
            processed_msg = pc2.create_cloud_xyz32(header, points_transformed)

            self.point_cloud_publisher.publish(processed_msg)
            
            # self.get_logger().info(
            #     f"Published processed message with frame_id: '{processed_msg.header.frame_id}', "
            #     f"timestamp: {processed_msg.header.stamp}"
            # )

            # Log debug information
            self.get_logger().debug(f"Published rotated point cloud with {points_rotated.shape[0]} points.")
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

# -----------------------------------
# Main Entry Point
# -----------------------------------
def main(args=None):
    """
    The main function initializes the ROS2 system, loads configuration parameters,
    creates an instance of the GeneralTaskNode, and spins to process messages until shutdown.
    """
    rclpy.init(args=args)
    config = load_config()
    node = PointCloudNode(config)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass  # Allow graceful shutdown on CTRL+C.
    finally:
        node.destroy_node()
        rclpy.shutdown()

# -----------------------------------
# Run the node when the script is executed directly.
# -----------------------------------
if __name__ == '__main__':
    main()
