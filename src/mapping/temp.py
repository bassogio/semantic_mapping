# -----------------------------------
# Import Statements
# -----------------------------------
import os
import yaml
import numpy as np
import struct  
from cv_bridge import CvBridge
from transforms3d.quaternions import quat2mat
import rclpy
from rclpy.node import Node
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, Point, Pose, Quaternion
from nav_msgs.msg import OccupancyGrid, MapMetaData

# -----------------------------------
# Bresenham's line algorithm
# -----------------------------------
def bresenham(x0, y0, x1, y1):
    """
    Returns list of grid cells from (x0,y0) to (x1,y1).
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x1 > x0 else -1
    sy = 1 if y1 > y0 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    points.append((x1, y1))
    return points

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
            return yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found at: {config_file}")

# -----------------------------------
# ROS2 Node Definition
# -----------------------------------
class OccupancyGridNode(Node):

    def __init__(self, config):
        super().__init__('occupancy_grid_node')
        self.node_config         = config['semantic_mapping']
        # Load parameters
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
        
        # Declare & get ROS2 parameters (allow runtime overrides)
        for name in ['occupancy_grid_topic','point_cloud_topic','pose_topic','frame_id',
                     'prior_prob','occupied_prob','free_prob','p_min','p_max',
                     'grid_resolution','grid_width','grid_height','grid_origin']:
            self.declare_parameter(name, getattr(self, name))
            setattr(self, name, self.get_parameter(name).value)

        # Bridge & map initialization
        self.bridge = CvBridge()
        self.l0     = self.prob_to_log_odds(self.prior_prob)
        self.occupancy_map = np.full(
            (self.grid_height, self.grid_width),
            self.l0, dtype=np.float32
        )
        # precompute log-odds of measurements
        self.l_occ  = self.prob_to_log_odds(self.occupied_prob)
        self.l_free = self.prob_to_log_odds(self.free_prob)
        # clamp limits
        self.l_min  = self.prob_to_log_odds(self.p_min)
        self.l_max  = self.prob_to_log_odds(self.p_max)

        # pose
        self.received_pose = False
        self.pose_x = self.pose_y = self.pose_z = 0.0

        # Publishers & Subscribers
        self.occupancy_grid_pub = self.create_publisher(
            OccupancyGrid, self.occupancy_grid_topic, 10
        )
        self.create_subscription(
            PoseStamped, self.pose_topic, self.pose_callback, 10
        )
        self.create_subscription(
            PointCloud2, self.point_cloud_topic,
            self.point_cloud_callback, 10
        )

    def pose_callback(self, msg):
        self.received_pose = True
        self.pose_x = msg.pose.position.x
        self.pose_y = msg.pose.position.y
        self.pose_z = msg.pose.position.z

    def point_cloud_callback(self, msg):
        if not self.received_pose:
            self.get_logger().warn("Skipping update: no pose yet")
            return

        # read points
        points = pc2.read_points(
            msg, field_names=("x","y","z"), skip_nans=True
        )
        # sensor origin in grid coords
        sx = int((self.pose_x - self.grid_origin[0]) / self.grid_resolution)
        sy = int((self.pose_y - self.grid_origin[1]) / self.grid_resolution)

        # update each beam
        for x_w, y_w, _ in points:
            gx = int((x_w - self.grid_origin[0]) / self.grid_resolution)
            gy = int((y_w - self.grid_origin[1]) / self.grid_resolution)
            if not (0 <= gx < self.grid_width and 0 <= gy < self.grid_height):
                continue
            # cells along beam
            line = bresenham(sx, sy, gx, gy)
            # free cells (all except endpoint)
            for cx, cy in line[:-1]:
                self.occupancy_map[cy, cx] += (self.l_free - self.l0)
            # occupied cell at end
            ex, ey = line[-1]
            self.occupancy_map[ey, ex] += (self.l_occ - self.l0)

        # clamp
        np.clip(self.occupancy_map, self.l_min, self.l_max,
                out=self.occupancy_map)

        # publish
        self.publish_occupancy_grid()

    def publish_occupancy_grid(self):
        """Convert log-odds map → probabilities → [0..100], then publish."""
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = self.frame_id

        # metadata
        meta = MapMetaData()
        meta.resolution = self.grid_resolution
        meta.width      = self.grid_width
        meta.height     = self.grid_height
        origin = Pose()
        origin.position.x = self.grid_origin[0]
        origin.position.y = self.grid_origin[1]
        origin.orientation = Quaternion(
            x=0.0, y=0.0, z=0.0, w=1.0
        )
        meta.origin = origin
        grid_msg.info = meta

        # flatten map: log-odds → probability → occupancy [0..100]
        probs = 1 - 1/(1 + np.exp(self.occupancy_map))
        flat  = (probs * 100).astype(np.int8).flatten().tolist()
        grid_msg.data = flat

        self.occupancy_grid_pub.publish(grid_msg)

    def prob_to_log_odds(self, p):
        return np.log(p / (1 - p))

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
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
