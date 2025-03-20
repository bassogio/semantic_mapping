#camera_parameters_from_data_processor
import rclpy
from rclpy.node import Node  
from sensor_msgs.msg import CameraInfo
import datetime
from camera_publisher import CameraPublisher
from std_msgs.msg import Float64MultiArray  

class CameraProcessor(rclpy.node.Node):
    def __init__(self, config):
        super().__init__('camera_processor')
        
        # Access the config under 'camera_processing'
        camera_processing = config['camera_processing']

        # Load configuration parameters from the nested 'camera_processing'
        self.camera_info_topic = camera_processing['camera_info_topic']
        self.parameters_topic = camera_processing['parameters_topic']
        self.timestamp_topic = camera_processing['timestamp_topic']

        # Declare parameters for ROS 2
        self.declare_parameter('camera_info_topic', self.camera_info_topic)
        self.declare_parameter('parameters_topic', self.parameters_topic)
        self.declare_parameter('timestamp_topic', self.timestamp_topic)

        # Initialize CameraPublisher and pass the parameters
        self.publisher = CameraPublisher(
            self, 
            self.parameters_topic, 
            self.timestamp_topic)

        # Create a subscription to /davis/left/camera_info
        self.create_subscription(
            CameraInfo, 
            self.camera_info_topic, 
            self.camera_info_callback, 10)

        # Subscribe to the camera parameters 
        self.create_subscription(
            Float64MultiArray,  
            self.parameters_topic, 
            self.parameters_callback, 10)

    def camera_info_callback(self, msg):
        # Extract camera parameters
        fx = msg.k[0]
        fy = msg.k[4]
        cx = msg.k[2]
        cy = msg.k[5]

        # Extract timestamp from the header
        timestamp_sec = msg.header.stamp.sec
        timestamp_nsec = msg.header.stamp.nanosec

        # Convert timestamp to human-readable format
        timestamp = datetime.datetime.fromtimestamp(timestamp_sec + timestamp_nsec / 1e9)
        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')

        # Publish extracted parameters and timestamp
        self.publisher.publish_parameters(fx, fy, cx, cy)
        self.publisher.publish_timestamp(timestamp_str)

    def parameters_callback(self, msg):
        try:
            # Unpack the parameters from the received array
            fx, fy, cx, cy = msg.data
            #self.get_logger().info(f"Received camera parameters: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        except Exception as e:
            self.get_logger().error(f"Error unpacking camera parameters: {e}")
