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

def parse_optional_float(value):
    """
    Convert a value to a float if it is not None or the string "None" (case-insensitive).
    Otherwise, return None.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "none":
        return None
    return float(value)

class IncrementalOccupancyGridNode(Node):
    def __init__(self, config):
        super().__init__('incremental_occupancy_grid_node')
        
        # -------------------------------------------
        # Load configuration parameters.
        # -------------------------------------------
        self.node_config = config['occupancy_grid_processing']

        self.occupancy_grid_topic  = self.node_config['occupancy_grid_topic']
        self.point_cloud_topic     = self.node_config['point_cloud_topic']
        self.frame_id              = self.node_config['frame_id']
        self.grid_resolution       = self.node_config['grid_resolution']
        self.grid_size             = self.node_config['grid_size']
        self.grid_width            = self.node_config['grid_width']
        self.grid_height           = self.node_config['grid_height']
        self.grid_origin           = list(map(float, self.node_config['grid_origin']))
        
        # Declare parameters for runtime modification.
        self.declare_parameter('occupancy_grid_topic', self.occupancy_grid_topic)
        self.declare_parameter('point_cloud_topic', self.point_cloud_topic)
        self.declare_parameter('frame_id', self.frame_id)
        self.declare_parameter('grid_resolution', self.grid_resolution)
        self.declare_parameter('grid_size', self.grid_size)
        self.declare_parameter('grid_width', self.grid_width)
        self.declare_parameter('grid_height', self.grid_height)
        self.declare_parameter('grid_origin', self.grid_origin)
        
        # Retrieve final values from the parameter server.
        self.occupancy_grid_topic  = self.get_parameter('occupancy_grid_topic').value
        self.point_cloud_topic     = self.get_parameter('point_cloud_topic').value
        self.frame_id              = self.get_parameter('frame_id').value
        self.grid_resolution       = self.get_parameter('grid_resolution').value
        self.grid_size             = self.get_parameter('grid_size').value
        self.grid_width            = self.get_parameter('grid_width').value
        self.grid_height           = self.get_parameter('grid_height').value
        self.grid_origin           = self.get_parameter('grid_origin').value

        # -------------------------------------------
        # Persistent Grid State: Initialize log-odds grid and known cell mask.
        # -------------------------------------------
        self.grid_log_odds = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        self.known = np.zeros((self.grid_height, self.grid_width), dtype=bool)

        # Set the update increment for an occupied cell.
        self.L_occ = 0.7  # You can adjust this value as needed.
        
        # -------------------------------------------
        # Create Publishers and Subscribers.
        # -------------------------------------------
        self.occupancy_grid_publisher = self.create_publisher(OccupancyGrid, self.occupancy_grid_topic, 10)
        self.point_cloud_subscription = self.create_subscription(
            PointCloud2,                # Message type.
            self.point_cloud_topic,      # Topic name.
            self.point_cloud_callback,   # Callback function.
            10                          # Queue size.
        )
        
        self.received_point_cloud = False
        self.subscription_check_timer = self.create_timer(2.0, self.check_initial_subscriptions)
    
    def check_initial_subscriptions(self):
        waiting_topics = []
        if not self.received_point_cloud:
            waiting_topics.append(f"'{self.point_cloud_topic}'")
            
        if waiting_topics:
            self.get_logger().info(f"Waiting for messages on topics: {', '.join(waiting_topics)}")
        else:
            self.get_logger().info(
                f"All subscribed topics have received at least one message. "
                f"IncrementalOccupancyGridNode running with publisher on '{self.occupancy_grid_topic}', "
                f"subscriber on '{self.point_cloud_topic}', frame '{self.frame_id}'."
            )
            self.subscription_check_timer.cancel()
    
    def point_cloud_callback(self, msg):
        if not self.received_point_cloud:
            self.received_point_cloud = True

        # Read all x, y, z points from the PointCloud2 message.
        all_points = np.array([
            [p[0], p[1], p[2]] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        ])

        # Use only the x and y coordinates for occupancy mapping.
        points = all_points[:, :2]

        # Update the persistent occupancy grid for each point.
        for x, y in points:
            grid_x = int((x - self.grid_origin[0]) / self.grid_resolution)
            grid_y = int((y - self.grid_origin[1]) / self.grid_resolution)
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                self.known[grid_y, grid_x] = True
                self.grid_log_odds[grid_y, grid_x] += self.L_occ

        # Publish the updated occupancy grid.
        self.publish_occupancy_grid()

    def publish_occupancy_grid(self):
        """
        Convert the persistent log-odds grid to an occupancy grid message and publish it.
        Unknown cells remain marked as -1.
        """
        occupancy = np.full((self.grid_height, self.grid_width), -1, dtype=np.int8)
        known_cells = self.known
        if np.any(known_cells):
            prob = 1.0 / (1.0 + np.exp(-self.grid_log_odds[known_cells]))
            occupancy[known_cells] = np.round(prob * 100).astype(np.int8)
        
        occ_grid_msg = OccupancyGrid()
        occ_grid_msg.header.stamp = self.get_clock().now().to_msg()
        occ_grid_msg.header.frame_id = self.frame_id
        occ_grid_msg.info.resolution = self.grid_resolution
        occ_grid_msg.info.width = self.grid_width
        occ_grid_msg.info.height = self.grid_height
        occ_grid_msg.info.origin.position.x = self.grid_origin[0]
        occ_grid_msg.info.origin.position.y = self.grid_origin[1]
        occ_grid_msg.info.origin.position.z = 0.0
        occ_grid_msg.data = occupancy.flatten().tolist()
        
        self.occupancy_grid_publisher.publish(occ_grid_msg)
        self.get_logger().info("Published updated occupancy grid.")

def main(args=None):
    rclpy.init(args=args)
    config = load_config()
    node = IncrementalOccupancyGridNode(config)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass  # Allow graceful shutdown on CTRL+C.
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
