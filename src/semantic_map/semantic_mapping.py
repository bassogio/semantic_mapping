# -----------------------------------
# Import Statements
# -----------------------------------
import os       
import yaml     
import rclpy    
from rclpy.node import Node  
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
import sensor_msgs_py.point_cloud2 as pc2

# -----------------------------------
# Configuration Loader Function
# -----------------------------------
def load_config():
    """
    Load configuration parameters.

    Returns:
        dict: A dictionary containing configuration data.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, '../../config/semantic_mapping_config.yaml')
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        raise FileNotFoundError(f"Config file not found at: {config_file}")

# -----------------------------------
# ROS2 Node Definition
# -----------------------------------
class SemanticMapNode(Node):

    def __init__(self, config):
        # Initialize the node with a unique name.
        super().__init__('semantic_map_node')
        
        # -------------------------------------------
        # Access the configuration section.
        # -------------------------------------------
        self.node_config = config['semantic_mapping']
        
        # -------------------------------------------
        # Load configuration parameters.
        # -------------------------------------------
        self.map_topic            = self.node_config['map_topic']   
        self.point_cloud_topic    = self.node_config['point_cloud_topic']
        self.frame_id             = self.node_config['frame_id']

        # -------------------------------------------
        # Declare ROS2 parameters for runtime modification.
        # -------------------------------------------
        self.declare_parameter('map_topic', self.map_topic)
        self.declare_parameter('point_cloud_topic', self.point_cloud_topic)
        self.declare_parameter('frame_id', self.frame_id)

        # -------------------------------------------
        # Retrieve final parameter values from the parameter server.
        # This allows any runtime overrides (e.g., via launch files) to update these defaults.
        # -------------------------------------------
        self.map_topic            = self.get_parameter('map_topic').value
        self.point_cloud_topic    = self.get_parameter('point_cloud_topic').value
        self.frame_id             = self.get_parameter('frame_id').value

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
    def listener_callback(self, msg):
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
      
# -----------------------------------
# Main Entry Point
# -----------------------------------
def main(args=None):
    rclpy.init(args=args)
    config = load_config()
    node = SemanticMapNode(config)
    
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
