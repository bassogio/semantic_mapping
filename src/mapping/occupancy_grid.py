# -----------------------------------
# Import Statements
# -----------------------------------
import os
import yaml
import numpy as np
import struct  
import cv2   
from cv_bridge import CvBridge
from transforms3d.quaternions import quat2mat
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid

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
class OccupancyGridNode(Node):

    def __init__(self, config):
        # Initialize the node with a unique name.
        super().__init__('occupancy_grid_node')

        # -------------------------------------------
        # Access the configuration section.
        # -------------------------------------------
        self.node_config = config['semantic_mapping']

        # -------------------------------------------
        # Load configuration parameters.
        # -------------------------------------------
        self.occupancy_grid_topic = self.node_config['occupancy_grid_topic']
        self.point_cloud_topic    = self.node_config['point_cloud_topic']
        self.pose_topic           = self.node_config['pose_topic']
        self.frame_id             = self.node_config['frame_id']
        self.prior_prob           = self.node_config['prior_prob']
        self.occupied_prob        = self.node_config['occupied_prob']
        self.free_prob            = self.node_config['free_prob']
        self.p_min                = self.node_config['p_min']
        self.p_max                = self.node_config['p_max']
        self.grid_resolution      = self.node_config['grid_resolution']
        self.grid_width           = self.node_config['grid_width']
        self.grid_height          = self.node_config['grid_height']
        self.grid_origin          = list(map(float, self.node_config['grid_origin']))

        # -------------------------------------------
        # Declare ROS2 parameters for runtime modification.
        # -------------------------------------------
        self.declare_parameter('occupancy_grid_topic', self.occupancy_grid_topic)
        self.declare_parameter('point_cloud_topic',    self.point_cloud_topic)
        self.declare_parameter('pose_topic',           self.pose_topic)
        self.declare_parameter('frame_id',             self.frame_id)
        self.declare_parameter('prior_prob',           self.prior_prob)
        self.declare_parameter('occupied_prob',        self.occupied_prob)
        self.declare_parameter('free_prob',            self.free_prob)
        self.declare_parameter('p_min',                self.p_min)
        self.declare_parameter('p_max',                self.p_max)
        self.declare_parameter('grid_resolution',      self.grid_resolution)
        self.declare_parameter('grid_width',           self.grid_width)
        self.declare_parameter('grid_height',          self.grid_height)
        self.declare_parameter('grid_origin',          self.grid_origin)

        # -------------------------------------------
        # Retrieve final parameter values from the parameter server.
        # This allows any runtime overrides (e.g., via launch files) to update these defaults.
        # -------------------------------------------
        self.occupancy_grid_topic = self.get_parameter('occupancy_grid_topic').value
        self.point_cloud_topic    = self.get_parameter('point_cloud_topic').value
        self.pose_topic           = self.get_parameter('pose_topic').value
        self.frame_id             = self.get_parameter('frame_id').value
        self.prior_prob           = self.get_parameter('prior_prob').value
        self.occupied_prob        = self.get_parameter('occupied_prob').value
        self.free_prob            = self.get_parameter('free_prob').value
        self.p_min                = self.get_parameter('p_min').value
        self.p_max                = self.get_parameter('p_max').value
        self.grid_resolution      = self.get_parameter('grid_resolution').value
        self.grid_width           = self.get_parameter('grid_width').value
        self.grid_height          = self.get_parameter('grid_height').value
        self.grid_origin          = self.get_parameter('grid_origin').value
        
        # -------------------------------------------
        # Initialize CvBridge.
        # -------------------------------------------
        self.bridge = CvBridge()

        # -------------------------------------------
        # Initialize occupancy grid.
        # -------------------------------------------
        l_0 = self.prob_to_log_odds(self.prior_prob)
        self.occupancy_map = np.full((self.grid_height, self.grid_width), l_0, dtype=np.float32)  # Initialize with log odds of prior probability

        # Precompute measurement updates
        self.l_occupied = self.prob_to_log_odds(self.occupied_prob) # Occupied probability
        self.l_free = self.prob_to_log_odds(self.free_prob) # Free probability
        
        # Define clamping limits:
        # Define clamping limits on the log-odds values to keep our occupancy probabilities
        # within a reasonable range and avoid runaway certainty:
        # - We never want any cell to report less than (p_min * 100)% chance of being occupied,
        #   nor more than (p_max * 100)% chance, regardless of how many observations we process.
        # - Converting these probabilities into log-odds gives us l_min and l_max
        self.l_min = self.prob_to_log_odds(self.p_min)
        self.l_max = self.prob_to_log_odds(self.p_max)

        # -------------------------------------------
        # Initialize pose variables.
        # -------------------------------------------
        self.Qw     = 1.0  # Default orientation (identity quaternion)
        self.Qx     = 0.0
        self.Qy     = 0.0
        self.Qz     = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_z = 0.0

        # -------------------------------------------
        # Create Publishers.
        # -------------------------------------------
        self.occupancy_grid_publisher = self.create_publisher(OccupancyGrid, self.occupancy_grid_topic, 10)

        # -------------------------------------------
        # Create Subscribers.
        # -------------------------------------------
        self.pose_subscription = self.create_subscription(
            PoseStamped,              # Message type.
            self.pose_topic,         # Topic name.
            self.pose_callback,      # Callback function.
            10                       # Queue size.
        )

        self.point_cloud_subscription = self.create_subscription(
            PointCloud2,               # Message type.
            self.point_cloud_topic,    # Topic name.
            self.point_cloud_callback, # Callback function.
            10                         # Queue size.
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
                f"OccupancyGridNode started with publishers on '{self.occupancy_grid_topic}', "
                f"subscribers on '{self.point_cloud_topic}', "
                f"and frame_id '{self.frame_id}'."
            )
            self.subscription_check_timer.cancel()
    
    # -------------------------------------------
    # Pose Callback Function
    # -------------------------------------------
    def pose_callback(self, msg):
        if not self.received_pose:
            self.received_pose = True
        # Update quaternion and translation from pose message.
        self.Qx = msg.pose.orientation.x
        self.Qy = msg.pose.orientation.y
        self.Qz = msg.pose.orientation.z
        self.Qw = msg.pose.orientation.w
        self.pose_x = msg.pose.position.x
        self.pose_y = msg.pose.position.y
        self.pose_z = msg.pose.position.z

    # TODO: I don't need to extract the rgb data from the point cloud
    # -------------------------------------------
    # Point cloud Callback Function
    # -------------------------------------------
    def point_cloud_callback(self, msg):
        if not self.received_point_cloud:
            self.received_point_cloud = True
        
        # Read all the points
        points = list(pc2.read_points(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True))

        #  Preallocate NumPy array
        self.all_points = np.zeros((len(points), 6), dtype=np.float32)  # x, y, z, r, g, b

        # Fill the array manually
        for i, (x, y, z, rgb_float) in enumerate(points):
            rgb_uint = struct.unpack('I', struct.pack('f', rgb_float))[0]
            r = (rgb_uint >> 16) & 0xFF
            g = (rgb_uint >> 8) & 0xFF
            b = rgb_uint & 0xFF
            self.all_points[i] = [x, y, z, r, g, b]

        self.publish_maps()

    # -------------------------------------------
    # Probability to Log Odds
    # -------------------------------------------
    def prob_to_log_odds(self, p):
        """
        Log odds ratio of p(x):

                      p(x)
        l(x) = log ----------
                    1 - p(x)

        """
        return np.log(p / (1 - p))
    
    # -------------------------------------------
    # Log Odds to Probability
    # -------------------------------------------
    def log_odds_to_prob(self, l):
        """
        Probability of p(x) from log odds ratio l(x):

                         1
        p(x) = 1 - ---------------
                    1 + e(l(x))

        """
        return 1 - (1 / (1 + np.exp(l)))

    # -------------------------------------------
    # Maps Publishing Function
    # -------------------------------------------
    def publish_maps(self):
        # Group points into 2D grid cells.
        grid = {}
        for pt in self.all_points:
            x, y, z, r, g, b = pt
            # world→grid with origin offset
            # Calculate grid cell index by quantizing x and y using resolution
            grid_x = int((x - self.grid_origin[0]) / self.grid_resolution)
            grid_y = int((y - self.grid_origin[1]) / self.grid_resolution)
            
            # Check if the grid cell is within bounds
            if grid_x < 0 or grid_x >= self.grid_width or grid_y < 0 or grid_y >= self.grid_height:
                continue

            key = (grid_x, grid_y) 

            # Initialize the cell entry if it doesn't exist yet
            if key not in grid:
                grid[key] = {'points': []}

            # Append the point corresponding cell
            grid[key]['points'].append([x, y, z])

        
        # Publish the occupancy grid.
        # self.occupancy_grid_publisher.publish()

# -----------------------------------
# Main Entry Point
# -----------------------------------
def main(args=None):
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