#!/usr/bin/env python3
"""
Generic ROS2 Node Template with Multiple Publishers and Subscribers,
Optional Service, and Header Handling

This template demonstrates how to:
  - Load configuration parameters from a YAML file.
  - Declare and retrieve ROS2 parameters.
  - Create multiple publishers and subscribers.
  - Process incoming messages by updating header information (timestamp and frame_id)
    and publishing the resulting message on a second topic.
  - Optionally create a service server (e.g., using Trigger) that can respond to service calls.
  - Wait until messages are received from all subscribed topics before ceasing further checks.

Update your configuration file (config.yaml) to match these parameter names.
"""

# -----------------------------------
# Import Statements
# -----------------------------------
import os       # For file path operations.
import yaml     # For loading configuration files.
import rclpy    # ROS2 Python client library.
from rclpy.node import Node  # Base class for ROS2 nodes.
from geometry_msgs.msg import PoseStamped  # Message type including header and pose.
from std_srvs.srv import Trigger  # Standard service type for simple triggers.

# -----------------------------------
# Configuration Loader Function
# -----------------------------------
def load_config():
    """
    Load configuration parameters from 'config.yaml' in the same directory as this script.

    Expected YAML structure example:
      node_config:
        publisher_topic: "output_topic"
        publisher_topic2: "output_topic2"
        subscriber_topic: "input_topic"
        subscriber_topic2: "input_topic2"
        frame_id: "base_link"
        use_service: true
        service_name: "trigger_service"

    Returns:
        dict: A dictionary containing configuration data.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, 'config.yaml')
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        raise FileNotFoundError(f"Config file not found at: {config_file}")

# -----------------------------------
# ROS2 Node Definition
# -----------------------------------
class GeneralTaskNode(Node):
    """
    A generic ROS2 node that demonstrates header management, multiple publishers/subscribers,
    and optional service functionality.

    This node:
      - Loads configuration parameters from the configuration file section ("node_config").
      - Declares ROS2 parameters for runtime introspection.
      - Creates two publishers and two subscribers for PoseStamped messages.
      - Processes each incoming message by updating the header (with the current timestamp and
        configured frame_id) and publishing the processed message.
      - Optionally provides a service server (using std_srvs/Trigger) if enabled.
      - Waits until it receives a message from all subscribed topics before ceasing further checks.
    """
    def __init__(self, config):
        # Initialize the node with a unique name.
        super().__init__('general_task_node')
        
        # -------------------------------------------
        # Access the configuration section.
        #
        # The configuration file (config.yaml) should include a top-level key "node_config" with your
        # node's parameters (e.g., publisher_topic, subscriber_topic, etc.).
        # -------------------------------------------
        self.node_config = config['node_config']
        
        # -------------------------------------------
        # Load configuration parameters.
        # -------------------------------------------
        self.publisher_topic  = self.node_config['publisher_topic']
        self.publisher_topic2 = self.node_config['publisher_topic2']
        self.subscriber_topic  = self.node_config['subscriber_topic']
        self.subscriber_topic2 = self.node_config['subscriber_topic2']
        self.frame_id          = self.node_config['frame_id']
        self.use_service       = self.node_config.get('use_service', False)
        self.service_name      = self.node_config.get('service_name', 'trigger_service')
        
        # -------------------------------------------
        # Declare ROS2 parameters for runtime modification.
        # -------------------------------------------
        self.declare_parameter('publisher_topic', self.publisher_topic)
        self.declare_parameter('publisher_topic2', self.publisher_topic2)
        self.declare_parameter('subscriber_topic', self.subscriber_topic)
        self.declare_parameter('subscriber_topic2', self.subscriber_topic2)
        self.declare_parameter('frame_id', self.frame_id)
        
        # -------------------------------------------
        # Retrieve final parameter values from the parameter server.
        # This allows any runtime overrides (e.g., via launch files) to update these defaults.
        # -------------------------------------------
        self.publisher_topic  = self.get_parameter('publisher_topic').value
        self.publisher_topic2 = self.get_parameter('publisher_topic2').value
        self.subscriber_topic  = self.get_parameter('subscriber_topic').value
        self.subscriber_topic2 = self.get_parameter('subscriber_topic2').value
        self.frame_id          = self.get_parameter('frame_id').value
        
        # -------------------------------------------
        # Initialize additional attributes needed for processing.
        # These attributes hold pose and orientation data and must be initialized.
        # -------------------------------------------
        self.Qw = 1.0  # Default for an identity quaternion.
        self.Qx = 0.0
        self.Qy = 0.0
        self.Qz = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_z = 0.0

        # -------------------------------------------
        # Create Publishers.
        # -------------------------------------------
        # Publisher 1 sends PoseStamped messages on the first publisher topic.
        self.publisher_ = self.create_publisher(PoseStamped, self.publisher_topic, 10)
        # Publisher 2 sends PoseStamped messages on the second publisher topic.
        self.publisher2_ = self.create_publisher(PoseStamped, self.publisher_topic2, 10)
        
        # -------------------------------------------
        # Create Subscribers.
        # -------------------------------------------
        # Subscriber 1 listens for PoseStamped messages on the first subscriber topic.
        self.subscription = self.create_subscription(
            PoseStamped,                # Message type.
            self.subscriber_topic,      # Topic name.
            self.listener_callback,     # Callback function.
            10                          # Queue size.
        )
        # Subscriber 2 listens for PoseStamped messages on the second subscriber topic.
        self.subscription2 = self.create_subscription(
            PoseStamped,                # Message type.
            self.subscriber_topic2,     # Topic name.
            self.listener_callback2,    # Callback function.
            10                          # Queue size.
        )
        
        # -------------------------------------------
        # Initialize flags to track if each subscriber has received a message.
        # -------------------------------------------
        self.received_sub1 = False
        self.received_sub2 = False

        # -------------------------------------------
        # Create a Timer to check if all subscribed topics have received at least one message.
        # This timer will stop checking once messages from both topics have been received.
        # -------------------------------------------
        self.subscription_check_timer = self.create_timer(2.0, self.check_initial_subscriptions)
        
        # -------------------------------------------
        # OPTIONAL: Create a Service Server.
        # Enable this block if you want your node to provide service functionality.
        # To disable, either set 'use_service' to false in the configuration file or comment out this block.
        # -------------------------------------------
        if self.use_service:
            self.service_server = self.create_service(Trigger, self.service_name, self.service_callback)
            self.get_logger().info(f"Service server started on '{self.service_name}'")
    
    # -------------------------------------------
    # Timer Callback to Check if All Subscribed Topics Have Received at Least One Message
    # -------------------------------------------
    def check_initial_subscriptions(self):
        waiting_topics = []
        if not self.received_sub1:
            waiting_topics.append(f"'{self.subscriber_topic}'")
        if not self.received_sub2:
            waiting_topics.append(f"'{self.subscriber_topic2}'")
            
        if waiting_topics:
            self.get_logger().info(f"Waiting for messages on topics: {', '.join(waiting_topics)}")
        else:
            self.get_logger().info(
                "All subscribed topics have received at least one message."
                f"GeneralTaskNode started with publishers on '{self.publisher_topic}' and '{self.publisher_topic2}', "
                f"subscribers on '{self.subscriber_topic}' and '{self.subscriber_topic2}', "
                f"and frame_id '{self.frame_id}'."
            )
            self.subscription_check_timer.cancel()
    
    # -------------------------------------------
    # Subscriber Callback Function for Topic 1
    # -------------------------------------------
    def listener_callback(self, msg: PoseStamped):
        """
        Callback invoked when a PoseStamped message is received on the first subscriber topic.

        The function:
          - Logs the header information of the incoming message.
          - Creates a new message.
          - Sets the header's timestamp to the current time and frame_id to the configured value.
          - Copies (or processes) the pose data.
          - Publishes the processed message using Publisher 1.

        Parameters:
            msg (PoseStamped): The received message.
        """
        if not self.received_sub1:
            self.received_sub1 = True

        self.get_logger().info(
            f"(Subscriber 1) Received message with frame_id: '{msg.header.frame_id}', timestamp: {msg.header.stamp}"
        )
        
        processed_msg = PoseStamped()
        processed_msg.header.stamp = self.get_clock().now().to_msg()  # Update timestamp.
        processed_msg.header.frame_id = self.frame_id               # Update frame_id.
        processed_msg.pose = msg.pose                                # Copy or modify pose as needed.
        
        self.publisher_.publish(processed_msg)
        
        self.get_logger().info(
            f"(Publisher 1) Published processed message with frame_id: '{processed_msg.header.frame_id}', "
            f"timestamp: {processed_msg.header.stamp}"
        )
    
    # -------------------------------------------
    # Subscriber Callback Function for Topic 2
    # -------------------------------------------
    def listener_callback2(self, msg: PoseStamped):
        """
        Callback invoked when a PoseStamped message is received on the second subscriber topic.

        Similar to listener_callback, but publishes the processed message using Publisher 2.

        Parameters:
            msg (PoseStamped): The received message.
        """
        if not self.received_sub2:
            self.received_sub2 = True

        self.get_logger().info(
            f"(Subscriber 2) Received message with frame_id: '{msg.header.frame_id}', timestamp: {msg.header.stamp}"
        )
        
        processed_msg = PoseStamped()
        processed_msg.header.stamp = self.get_clock().now().to_msg()  # Update timestamp.
        processed_msg.header.frame_id = self.frame_id               # Update frame_id.
        processed_msg.pose = msg.pose                                # Copy or process pose as needed.
        
        self.publisher2_.publish(processed_msg)
        
        self.get_logger().info(
            f"(Publisher 2) Published processed message with frame_id: '{processed_msg.header.frame_id}', "
            f"timestamp: {processed_msg.header.stamp}"
        )
    
    # -------------------------------------------
    # OPTIONAL: Service Callback Function
    # -------------------------------------------
    def service_callback(self, request, response):
        """
        Callback function for the optional service server.

        Uses the Trigger service from std_srvs. When the service is called, logs the call,
        and returns a simple response.

        Parameters:
            request: The service request (unused for Trigger).
            response: The service response to be sent.

        Returns:
            The updated service response with success set to True.
        """
        self.get_logger().info("Service call received.")
        response.success = True
        response.message = "Service call processed successfully."
        return response

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
    node = GeneralTaskNode(config)
    
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
