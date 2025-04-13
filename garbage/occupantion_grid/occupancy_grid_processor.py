# coccupancy_grid_processor.py
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from camera_publisher import CameraPublisher
import pyrealsense2 as rs
import numpy as np
from cv_bridge import CvBridge
import datetime

class OccupancyGridProcessor(Node):
    def __init__(self, config):
        super().__init__('camera_processor')

        # Access the config under 'camera_processing'
        camera_processing = config['camera_processing']
        
        # Initialize topics
        self.color_image_topic = camera_processing['color_image_topic']
        self.depth_image_topic = camera_processing['depth_image_topic']
        self.parameters_topic = camera_processing['CameraInfo_topic']
        self.timestamp_topic = camera_processing['timestamp_topic']

        # Declare parameters for ROS 2
        self.declare_parameter('color_image_topic', self.color_image_topic)
        self.declare_parameter('depth_image_topic', self.depth_image_topic)
        self.declare_parameter('parameters_topic', self.parameters_topic)
        self.declare_parameter('timestamp_topic', self.timestamp_topic)

        # Initialize CameraPublisher
        self.publisher = CameraPublisher(self, self.parameters_topic, self.timestamp_topic)

        # Setup RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.pipeline.start(self.config)

        # Timer to call process_camera_data at a fixed interval (e.g., 10 Hz)
        self.create_timer(0.1, self.process_camera_data)  # 0.1 sec = 10 Hz

        # Create a CvBridge object for converting images
        self.bridge = CvBridge()

    def process_timestamp(self):
        # Get the current timestamp
        timestamp_str = self.get_timestamp(self.get_clock().now().to_msg())
        
        # Publish timestamp
        self.publisher.publish_timestamp(timestamp_str)

    def process_camera_data(self):
        # Wait for frames from RealSense camera
        frames = self.pipeline.wait_for_frames()

        # Get color and depth frames
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            self.get_logger().warn("No frames received")
            return

        # Convert RealSense frames to OpenCV format
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert to ROS Image message
        color_image_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
        depth_image_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")

        # Add header to images with timestamp
        color_image_msg.header.stamp = self.get_clock().now().to_msg()
        depth_image_msg.header.stamp = self.get_clock().now().to_msg()

        # Publish images
        self.publisher.publish_image('/camera/color/image_raw', color_image_msg)
        self.publisher.publish_image('/camera/depth/image_raw', depth_image_msg)

        # Publish the timestamp after processing frames
        self.process_timestamp()

        # Camera Info processing remains unchanged
        camera_info = CameraInfo()
        camera_info.width = color_frame.profile.as_video_stream_profile().width()
        camera_info.height = color_frame.profile.as_video_stream_profile().height()
        camera_info.k[0] = color_frame.profile.as_video_stream_profile().intrinsics.fx
        camera_info.k[4] = color_frame.profile.as_video_stream_profile().intrinsics.fy
        camera_info.k[2] = color_frame.profile.as_video_stream_profile().intrinsics.ppx
        camera_info.k[5] = color_frame.profile.as_video_stream_profile().intrinsics.ppy
        self.publisher.publish_camera_info(camera_info)

    def get_timestamp(self, header_stamp):
        """Helper function to extract and format timestamp."""
        timestamp_sec = header_stamp.sec
        timestamp_nsec = header_stamp.nanosec

        # Convert timestamp to human-readable format
        timestamp = datetime.datetime.fromtimestamp(timestamp_sec + timestamp_nsec / 1e9)
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')

        return timestamp_str

    def on_shutdown(self):
        self.pipeline.stop()
