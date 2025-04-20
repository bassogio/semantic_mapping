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
        self.pose_topic           = self.node_config['pose_topic'] # I don't think I need it because Point cloud is already subscribing to it
        self.frame_id             = self.node_config['frame_id']
        self.grid_resolution      = self.node_config['grid_resolution']
        self.grid_size            = self.node_config['grid_size']
        self.max_distance         = self.node_config['max_distance']
        self.grid_origin          = list(map(float, self.node_config['grid_origin']))

        # -------------------------------------------
        # Declare ROS2 parameters for runtime modification.
        # -------------------------------------------
        self.declare_parameter('semantic_map_topic',   self.semantic_map_topic)
        self.declare_parameter('occupancy_grid_topic', self.occupancy_grid_topic)
        self.declare_parameter('point_cloud_topic',    self.point_cloud_topic)
        self.declare_parameter('pose_topic',           self.pose_topic)
        self.declare_parameter('frame_id',             self.frame_id)
        self.declare_parameter('grid_resolution',      self.grid_resolution)
        self.declare_parameter('grid_size',            self.grid_size)
        self.declare_parameter('max_distance',         self.max_distance)
        self.declare_parameter('grid_origin',          self.grid_origin)

        # -------------------------------------------
        # Retrieve final parameter values from the parameter server.
        # This allows any runtime overrides (e.g., via launch files) to update these defaults.
        # -------------------------------------------
        self.semantic_map_topic   = self.get_parameter('semantic_map_topic').value
        self.occupancy_grid_topic = self.get_parameter('occupancy_grid_topic').value
        self.point_cloud_topic    = self.get_parameter('point_cloud_topic').value
        self.pose_topic           = self.get_parameter('pose_topic').value
        self.frame_id             = self.get_parameter('frame_id').value
        self.grid_resolution      = self.get_parameter('grid_resolution').value
        self.grid_size            = self.get_parameter('grid_size').value
        self.max_distance         = self.get_parameter('max_distance').value
        self.grid_origin          = self.get_parameter('grid_origin').value

        # -------------------------------------------
        # Initialize log-odds grid and known cell mask.
        # -------------------------------------------
        self.grid_log_odds = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.known = np.zeros((self.grid_size, self.grid_size), dtype=bool)

        # Set the update increment for an occupied cell.
        self.L_occ = 0.7  # You can adjust this value as needed.

        # -------------------------------------------
        # Create Publishers.
        # -------------------------------------------
        self.occupancy_grid_publisher = self.create_publisher(OccupancyGrid, self.occupancy_grid_topic, 10)
        self.semantic_map_publisher   = self.create_publisher(Marker, self.semantic_map_topic, 10)

        # -------------------------------------------
        # Create Subscribers.
        # -------------------------------------------
        self.pose_subscription = self.create_subscription(
            PoseStamped,         # Message type.
            self.pose_topic,     # Topic name.
            self.pose_callback,  # Callback function.
            10                   # Queue size.
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
        self.received_pose = False
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
        if not self.received_pose:
            waiting_topics.append(f"'{self.pose_topic}'")
        if not self.received_point_cloud:
            waiting_topics.append(f"'{self.point_cloud_topic}'")
            
        if waiting_topics:
            self.get_logger().info(f"Waiting for messages on topics: {', '.join(waiting_topics)}")
        else:
            self.get_logger().info(
                "All subscribed topics have received at least one message."
                f"SemanticMapNode started with publishers on '{self.semantic_map_topic}' and '{self.occupancy_grid_topic}', "
                f"subscribers on '{self.point_cloud_topic}' and '{self.pose_topic}', "
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
        # Group points into grid cells.
        grid = {}
        for pt in self.all_points:
            x, y, z, r, g, b = pt
            # Compute grid cell index.
            grid_x = int(np.floor(x / self.grid_resolution))
            grid_y = int(np.floor(y / self.grid_resolution))

            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                self.known[grid_y, grid_x] = True
                self.grid_log_odds[grid_y, grid_x] += self.L_occ

            key = (grid_x, grid_y)
            if key not in grid:
                grid[key] = {'points': [], 'colors': []}
            grid[key]['points'].append([x, y, z])
            grid[key]['colors'].append([r, g, b])

        occupancy = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        known_cells = self.known
        if np.any(known_cells):
            prob = 1.0 / (1.0 + np.exp(-self.grid_log_odds[known_cells]))
            occupancy[known_cells] = np.round(prob * 100).astype(np.int8)
        
        occ_grid_msg = OccupancyGrid()
        occ_grid_msg.header.stamp = self.get_clock().now().to_msg()
        occ_grid_msg.header.frame_id = self.frame_id
        occ_grid_msg.info.resolution = self.grid_resolution
        occ_grid_msg.info.width = self.grid_size
        occ_grid_msg.info.height = self.grid_size
        occ_grid_msg.info.origin.position.x = self.grid_origin[0]
        occ_grid_msg.info.origin.position.y = self.grid_origin[1]
        occ_grid_msg.info.origin.position.z = 0.0
        occ_grid_msg.data = occupancy.flatten().tolist()
        
        self.occupancy_grid_publisher.publish(occ_grid_msg)
        
        cell_points = []
        cell_colors = []
        for key, group in grid.items():
            pts = np.array(group['points'])
            avg_pt = pts.mean(axis=0) 
            rgbs = group['colors']
            avg_rgb = np.mean(rgbs, axis=0) / 255.0
            cell_points.append(avg_pt)
            cell_colors.append(avg_rgb)

        # Create a Marker of type CUBE_LIST.
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.frame_id
        marker.ns = "occupancy_grid"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD
        # Set the scale (size) of each cube to be the grid resolution.
        marker.scale.x = self.grid_resolution
        marker.scale.y = self.grid_resolution
        marker.scale.z = self.grid_resolution
        marker.pose.orientation.w = 1.0

        # Add a cube for each grid cell.
        for pt, col in zip(cell_points, cell_colors):
            p = Point()
            p.x, p.y, p.z = pt
            marker.points.append(p)
            color = ColorRGBA()
            color.r, color.g, color.b = col
            color.a = 1.0
            marker.colors.append(color)

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