# -----------------------------------
# Import Statements
# -----------------------------------
import os       
import yaml     
import numpy as np
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
    config_file = os.path.join(script_dir, '../../config/occupancy_grid_config.yaml')
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        raise FileNotFoundError(f"Config file not found at: {config_file}")

# -----------------------------------
# ROS2 Node Definition
# -----------------------------------
class OccupancyGridNode(Node):
    def __init__(self, config):
        # Initialize the node with a unique name.
        super().__init__('occupancy_grid_node')
        
        # -------------------------------------------
        # Access the configuration section.
        #
        # The configuration file (config.yaml) should include a top-level key "node_config" with your
        # node's parameters (e.g., publisher_topic, subscriber_topic, etc.).
        # -------------------------------------------
        self.node_config = config['occupancy_grid_processing']
        
        # -------------------------------------------
        # Load configuration parameters.
        # -------------------------------------------
        self.occupancy_grid_topic  = self.node_config['occupancy_grid_topic']
        self.point_cloud_topic     = self.node_config['point_cloud_topic']
        self.frame_id              = self.node_config['frame_id']
        self.grid_resolution       = self.node_config['grid_resolution']
        self.grid_size             = self.node_config['grid_size']
        self.grid_origin           = list(map(float, self.node_config['grid_origin']))

        # -------------------------------------------
        # Declare ROS2 parameters for runtime modification.
        # -------------------------------------------
        self.declare_parameter('occupancy_grid_topic', self.occupancy_grid_topic)
        self.declare_parameter('point_cloud_topic', self.point_cloud_topic)
        self.declare_parameter('frame_id', self.frame_id)
        self.declare_parameter('grid_resolution', self.grid_resolution)
        self.declare_parameter('grid_size', self.grid_size)
        self.declare_parameter('grid_origin', self.grid_origin)
        
        # -------------------------------------------
        # Retrieve final parameter values from the parameter server.
        # This allows any runtime overrides (e.g., via launch files) to update these defaults.
        # -------------------------------------------
        self.occupancy_grid_topic  = self.get_parameter('occupancy_grid_topic').value
        self.point_cloud_topic     = self.get_parameter('point_cloud_topic').value
        self.frame_id              = self.get_parameter('frame_id').value
        self.grid_resolution       = self.get_parameter('grid_resolution').value
        self.grid_size             = self.get_parameter('grid_size').value
        self.grid_origin           = self.get_parameter('grid_origin').value

        # -------------------------------------------
        # Create Publishers.
        # -------------------------------------------
        self.occupancy_grid_publisher = self.create_publisher(OccupancyGrid, self.occupancy_grid_topic, 10)

        # -------------------------------------------
        # Create Subscribers.
        # -------------------------------------------
        # Subscriber 1 listens for PoseStamped messages on the first subscriber topic.
        self.point_cloud_subscription = self.create_subscription(
            PointCloud2,                # Message type.
            self.point_cloud_topic,      # Topic name.
            self.point_cloud_callback,     # Callback function.
            10                          # Queue size.
        )
        
        # -------------------------------------------
        # Initialize flags to track if each subscriber has received a message.
        # -------------------------------------------
        self.received_point_cloud = False

        # -------------------------------------------
        # Create a Timer to check if all subscribed topics have received at least one message.
        # This timer will stop checking once messages from both topics have been received.
        # -------------------------------------------
        self.subscription_check_timer = self.create_timer(2.0, self.check_initial_subscriptions)

    # -------------------------------------------
    # Timer Callback to Check if All Subscribed Topics Have Received at Least One Message
    # -------------------------------------------
    def check_initial_subscriptions(self):
        waiting_topics = []
        if not self.received_point_cloud:
            waiting_topics.append(f"'{self.point_cloud_topic}'")

        if waiting_topics:
            self.get_logger().info(f"Waiting for messages on topics: {', '.join(waiting_topics)}")
        else:
            self.get_logger().info(
                "All subscribed topics have received at least one message."
                f"OccupancyGridNode started with publisher on '{self.occupancy_grid_topic}', "
                f"subscribers on '{self.point_cloud_topic}', "
                f"and frame_id '{self.frame_id}'."
            )
            self.subscription_check_timer.cancel()
    
    # -------------------------------------------
    # Point Cloud Callback Function
    # -------------------------------------------
    def point_cloud_callback(self, msg):

        if not self.received_point_cloud:
            self.received_point_cloud = True
        
        """Convert PointCloud2 data to an occupancy grid and publish it."""
        # Extract points
        points = np.array([(p[0], p[1]) for p in pc2.read_points(msg, 
                                                                 field_names=("x", "y", "z"), 
                                                                 skip_nans=True)])

        # Initialize occupancy grid (-1: unknown, 0: free, 100: occupied)
        grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)

        for x, y in points:
            # Convert world coordinates to grid coordinates
            grid_x = int((x - self.grid_origin[0]) / self.grid_resolution)
            grid_y = int((y - self.grid_origin[1]) / self.grid_resolution)

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                grid[grid_y, grid_x] = 100  # Mark as occupied

        # Create OccupancyGrid message
        occ_grid_msg = OccupancyGrid()
        occ_grid_msg.header.stamp = self.get_clock().now().to_msg()
        occ_grid_msg.header.frame_id = self.frame_id
        occ_grid_msg.info.resolution = self.grid_resolution
        occ_grid_msg.info.width = self.grid_size
        occ_grid_msg.info.height = self.grid_size
        occ_grid_msg.info.origin.position.x = self.grid_origin[0]
        occ_grid_msg.info.origin.position.y = self.grid_origin[1]
        occ_grid_msg.info.origin.position.z = 0.0

        # Flatten grid and assign to message
        occ_grid_msg.data = grid.flatten().tolist()

        # Publish the occupancy grid
        self.occupancy_grid_publisher.publish(occ_grid_msg)

# -----------------------------------
# Main Entry Point
# -----------------------------------
def main(args=None):
    """
    The main function initializes the ROS2 system, loads configuration parameters,
    creates an instance of the OccupancyGridNode, and spins to process messages until shutdown.
    """
    rclpy.init(args=args)
    config = load_config()
    node = OccupancyGridNode(config)
    
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
