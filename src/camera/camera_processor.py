"""
Generic ROS2 Node Template for a Camera Node

This node loads configuration parameters, sets up a RealSense camera pipeline,
and publishes color and depth image messages (as well as camera info) continuously.
"""

# -----------------------------------
# Import Statements
# -----------------------------------
import os
import yaml
import pyrealsense2 as rs
import numpy as np
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

# -----------------------------------
# Configuration Loader Function
# -----------------------------------
def load_config():
    """
    Load configuration parameters.

    Returns:
        dict: A dictionary containing configuration data.
    """
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, '../../config/camera_config.yaml')
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        raise FileNotFoundError(f"Config file not found at: {config_file}")

# -----------------------------------
# ROS2 Node Definition
# -----------------------------------
class CameraNode(Node):
    def __init__(self, config):
        # Initialize the node with a unique name.
        super().__init__('camera_node')
        
        # -------------------------------------------
        # Access the configuration section.
        # -------------------------------------------
        self.node_config = config['camera_processing']
        
        # -------------------------------------------
        # Load configuration parameters.
        # -------------------------------------------
        self.color_image_topic = self.node_config['color_image_topic']
        self.depth_image_topic = self.node_config['depth_image_topic']
        self.camera_info_topic  = self.node_config['camera_info_topic']
        
        # -------------------------------------------
        # Declare ROS2 parameters for runtime modification.
        # -------------------------------------------
        self.declare_parameter('color_image_topic', self.color_image_topic)
        self.declare_parameter('depth_image_topic', self.depth_image_topic)
        self.declare_parameter('camera_info_topic',  self.camera_info_topic)
        
        # -------------------------------------------
        # Retrieve final parameter values from the parameter server.
        # -------------------------------------------
        self.color_image_topic = self.get_parameter('color_image_topic').value
        self.depth_image_topic = self.get_parameter('depth_image_topic').value
        self.camera_info_topic  = self.get_parameter('camera_info_topic').value
        
        # -------------------------------------------
        # Setup RealSense pipeline.
        # -------------------------------------------
        self.pipeline = rs.pipeline()
        self.config   = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.pipeline.start(self.config)
        
        # -------------------------------------------
        # Initialize CvBridge once for the node.
        # -------------------------------------------
        self.bridge = CvBridge()
        
        # -------------------------------------------
        # Create Publishers.
        # Use the topics specified in the configuration.
        # -------------------------------------------
        self.color_image_publisher = self.create_publisher(Image,       self.color_image_topic, 10)
        self.depth_image_publisher = self.create_publisher(Image,       self.depth_image_topic, 10)
        self.camera_info_publisher = self.create_publisher(CameraInfo,  self.camera_info_topic,  10)
        
        # -------------------------------------------
        # Create a Timer to process and publish camera data at a fixed interval (e.g., 10 Hz).
        # You can adjust the rate as needed.
        # -------------------------------------------
        self.create_timer(0.1, self.image_callback)
        
        self.get_logger().info(
            f"CameraNode started with publishers on '{self.color_image_topic}', "
            f"'{self.depth_image_topic}' and '{self.camera_info_topic}.'"
        )
    
    # -------------------------------------------
    # Image Processing Callback 
    # -------------------------------------------
    def image_callback(self):
        # Wait for frames from RealSense camera.
        frames = self.pipeline.wait_for_frames()
        
        # Get color and depth frames.
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            self.get_logger().warn("No frames received")
            return
        
        # Convert RealSense frames to OpenCV format.
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Convert to ROS Image messages.
        color_image_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        depth_image_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
        
        # Add header to images with timestamp.
        now = self.get_clock().now().to_msg()
        color_image_msg.header.stamp = now
        depth_image_msg.header.stamp = now
        
        # Publish the messages.
        self.color_image_publisher.publish(color_image_msg)
        self.depth_image_publisher.publish(depth_image_msg)
        
        # Process and publish camera info.
        self.camera_parameters_callback(color_frame)
    
    # -------------------------------------------
    # Camera Parameters Callback 
    # -------------------------------------------
    def camera_parameters_callback(self, color_frame):
        camera_info = CameraInfo()
        # Retrieve parameters from the RealSense stream profile.
        video_profile = color_frame.profile.as_video_stream_profile()
        intrinsics  = video_profile.intrinsics
        camera_info.width = intrinsics.width
        camera_info.height = intrinsics.height
        camera_info.k[0] = intrinsics.fx
        camera_info.k[4] = intrinsics.fy
        camera_info.k[2] = intrinsics.ppx
        camera_info.k[5] = intrinsics.ppy
        camera_info.header.stamp = self.get_clock().now().to_msg()
        
        self.camera_info_publisher.publish(camera_info)

# -----------------------------------
# Main Entry Point
# -----------------------------------
def main(args=None):
    rclpy.init(args=args)
    config = load_config()
    node   = CameraNode(config)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass  # Allow graceful shutdown on CTRL+C.
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
