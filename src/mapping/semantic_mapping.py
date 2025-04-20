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
from visualization_msgs.msg import Marker
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
        self.semantic_map_topic   = self.node_config['semantic_map_topic']   
        self.occupancy_grid_topic = self.node_config['occupancy_grid_topic']
        self.point_cloud_topic    = self.node_config['point_cloud_topic']
        self.frame_id             = self.node_config['frame_id']
        self.prior_prob           = self.node_config['prior_prob']
        self.occupied_prob        = self.node_config['occupied_prob']
        self.free_prob            = self.node_config['free_prob']
        self.grid_resolution      = self.node_config['grid_resolution']
        self.grid_width           = self.node_config['grid_width']
        self.grid_height          = self.node_config['grid_height']
        self.grid_origin          = list(map(float, self.node_config['grid_origin']))

        # -------------------------------------------
        # Declare ROS2 parameters for runtime modification.
        # -------------------------------------------
        self.declare_parameter('semantic_map_topic',   self.semantic_map_topic)
        self.declare_parameter('occupancy_grid_topic', self.occupancy_grid_topic)
        self.declare_parameter('point_cloud_topic',    self.point_cloud_topic)
        self.declare_parameter('frame_id',             self.frame_id)
        self.declare_parameter('prior_prob',           self.prior_prob)
        self.declare_parameter('occupied_prob',        self.occupied_prob)
        self.declare_parameter('free_prob',            self.free_prob)
        self.declare_parameter('grid_resolution',      self.grid_resolution)
        self.declare_parameter('grid_width',           self.grid_width)
        self.declare_parameter('grid_height',          self.grid_height)
        self.declare_parameter('grid_origin',          self.grid_origin)

        # -------------------------------------------
        # Retrieve final parameter values from the parameter server.
        # This allows any runtime overrides (e.g., via launch files) to update these defaults.
        # -------------------------------------------
        self.semantic_map_topic   = self.get_parameter('semantic_map_topic').value
        self.occupancy_grid_topic = self.get_parameter('occupancy_grid_topic').value
        self.point_cloud_topic    = self.get_parameter('point_cloud_topic').value
        self.frame_id             = self.get_parameter('frame_id').value
        self.prior_prob           = self.get_parameter('prior_prob').value
        self.occupied_prob        = self.get_parameter('occupied_prob').value
        self.free_prob            = self.get_parameter('free_prob').value
        self.grid_resolution      = self.get_parameter('grid_resolution').value
        self.grid_width           = self.get_parameter('grid_width').value
        self.grid_height          = self.get_parameter('grid_height').value
        self.grid_origin          = self.get_parameter('grid_origin').value
        
        # -------------------------------------------
        # Initialize CvBridge.
        # -------------------------------------------
        self.bridge = CvBridge()

        # -------------------------------------------
        # Create Publishers.
        # -------------------------------------------
        self.occupancy_grid_publisher = self.create_publisher(OccupancyGrid, self.occupancy_grid_topic, 10)
        self.semantic_map_publisher   = self.create_publisher(Marker, self.semantic_map_topic, 10)

        # -------------------------------------------
        # Create Subscribers.
        # -------------------------------------------
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
                f"SemanticMapNode started with publishers on '{self.semantic_map_topic}' and '{self.occupancy_grid_topic}', "
                f"subscribers on '{self.point_cloud_topic}', "
                f"and frame_id '{self.frame_id}'."
            )
            self.subscription_check_timer.cancel()
    
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
    # Maps Publishing Function
    # -------------------------------------------
    def publish_maps(self):
        # Group points into 2D grid cells.
        grid = {}
        for pt in self.all_points:
            x, y, z, r, g, b = pt
            # Calculate grid cell index by quantizing x and y using resolution
            grid_x = int(np.floor(x / self.grid_resolution))
            grid_y = int(np.floor(y / self.grid_resolution))
            key = (grid_x, grid_y)

            # Initialize the cell entry if it doesn't exist yet
            if key not in grid:
                grid[key] = {'points': [], 'colors': []}

            # Append the point and its RGB color to the corresponding cell
            grid[key]['points'].append([x, y, z])
            grid[key]['colors'].append([r, g, b])
        
        #TODO: This average thingy will be removed later on and instead we will keep point with the highest probability.
        """we group all points that fall into the same grid cell. 
        But one cell may contain many points — maybe 5, 50, or even 500.
        So we average to get a single representative point and color for the entire cell."""
        cell_points = []  # Stores average positions of each grid cell
        cell_colors = []  # Stores average normalized RGB color per cell

        for key, group in grid.items():
            # Compute average 3D position from all points in the cell
            pts = np.array(group['points'])
            avg_pt = pts.mean(axis=0) 

            # Compute average RGB color and normalize to 0–1 for ROS
            rgbs = group['colors']
            avg_rgb = np.clip(np.mean(rgbs, axis=0) / 255.0, 0, 1)

            # Store the results
            cell_points.append(avg_pt)
            cell_colors.append(avg_rgb)

        # Create a Marker of type CUBE_LIST.
        marker = Marker()
        marker.header.stamp       = self.get_clock().now().to_msg()
        marker.header.frame_id    = self.frame_id
        marker.ns                 = "semantic_map" # Namespace for this marker
        marker.id                 = 0 # Marker ID
        marker.type               = Marker.CUBE_LIST # Use cube list to draw colored cells
        marker.action             = Marker.ADD # Action to add or modify marker
        marker.scale.x            = self.grid_resolution # Size of each cube along x
        marker.scale.y            = self.grid_resolution # Size of each cube along y
        marker.scale.z            = self.grid_resolution # Size of each cube along z
        marker.pose.orientation.w = 1.0 # Identity quaternion (no rotation)

        # Add a cube for each grid cell.
        for pt, col in zip(cell_points, cell_colors):
            """
            cell_points: list of average positions [x, y, z] per grid cell.
            cell_colors: list of average RGB values [r, g, b] (normalized to 0–1) per grid cell.
            zip(...) pairs each point with its matching color.
            """
            # Set cube position
            p = Point()
            p.x = float(pt[0]) # x coordinate
            p.y = float(pt[1]) # y coordinate
            p.z = 0.0          # z coordinate
            marker.points.append(p) # add the point to the marker

            # Set cube color
            color = ColorRGBA() 
            color.r = float(col[0]) # red channel
            color.g = float(col[1]) # green channel
            color.b = float(col[2]) # blue channel
            color.a = 1.0           # full opacity
            marker.colors.append(color) 

        # Publish the semantic map
        self.semantic_map_publisher.publish(marker)

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