# camera_parameters_from_data_publisher.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String

class CameraPublisher(Node):
    def __init__(self, node, parameters_topic, timestamp_topic):
        super().__init__('camera_publisher')
        self.node = node

        # Initialize publishers for parameters and timestamp
        self.parameters_pub = self.create_publisher(
            Float64MultiArray, 
            parameters_topic, 10)
        self.timestamp_pub = self.create_publisher(
            String, 
            timestamp_topic, 10)

    def publish_parameters(self, fx, fy, cx, cy):
        # Create a message for the parameters as an array
        msg = Float64MultiArray()
        
        # Fill the data field with an array of parameters
        msg.data = [float(fx), float(fy), float(cx), float(cy)]

        # Publish the message
        self.parameters_pub.publish(msg)

    def publish_timestamp(self, timestamp_str):
        # Publish the timestamp string
        msg = String()
        msg.data = timestamp_str
        self.timestamp_pub.publish(msg)
