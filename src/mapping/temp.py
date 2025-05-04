#!/usr/bin/env python3
import os
import yaml
import numpy as np
from cv_bridge import CvBridge
from transforms3d.quaternions import quat2mat
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
import math


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
    raise FileNotFoundError(f"Config file not found at: {config_file}")


class OccupancyGridNode(Node):
    def __init__(self, config):
        super().__init__('occupancy_grid_node')
        cfg = config['semantic_mapping']

        # Topics & frames
        self.occupancy_topic = cfg['occupancy_grid_topic']
        self.pc_topic        = cfg['point_cloud_topic']
        self.pose_topic      = cfg['pose_topic']
        self.frame_id        = cfg['frame_id']

        # Probabilities & resolution
        self.grid_resolution = float(cfg['grid_resolution'])
        self.grid_origin     = list(map(float, cfg['grid_origin']))
        self.prior_p         = float(cfg['prior_prob'])
        self.occ_p           = float(cfg['occupied_prob'])
        self.free_p          = float(cfg['free_prob'])
        self.p_min           = float(cfg['p_min'])
        self.p_max           = float(cfg['p_max'])
        self.publish_hz      = float(cfg.get('publish_rate', 1.0))  # Hz

        # Declare and override ROS2 parameters
        # Match declared parameter names to attribute names
        param_map = {
            'occupancy_grid_topic': 'occupancy_topic',
            'point_cloud_topic':    'pc_topic',
            'pose_topic':           'pose_topic',
            'frame_id':             'frame_id',
            'grid_resolution':      'grid_resolution',
            'grid_origin':          'grid_origin',
            'prior_prob':           'prior_p',
            'occupied_prob':        'occ_p',
            'free_prob':            'free_p',
            'p_min':                'p_min',
            'p_max':                'p_max',
            'publish_rate':         'publish_hz'
        }
        for param_name, attr_name in param_map.items():
            self.declare_parameter(param_name, getattr(self, attr_name))
            setattr(self, attr_name, self.get_parameter(param_name).value)

        # Log-odds conversion factors
        self.l0        = self.prob_to_log_odds(self.prior_p)
        self.l_occ     = self.prob_to_log_odds(self.occ_p)
        self.l_free    = self.prob_to_log_odds(self.free_p)
        self.l_min     = self.prob_to_log_odds(self.p_min)
        self.l_max     = self.prob_to_log_odds(self.p_max)

        # Sparse map & dynamic bounds
        self.occupancy_map = {}  # {(i,j): log_odds}
        self.min_x = math.inf; self.max_x = -math.inf
        self.min_y = math.inf; self.max_y = -math.inf

        # Robot pose
        self.Qw = 1.0; self.Qx = self.Qy = self.Qz = 0.0
        self.pose_x = self.pose_y = self.pose_z = 0.0

        # CvBridge (for future use)
        self.bridge = CvBridge()

        # Publishers & subscribers
        self.pub = self.create_publisher(OccupancyGrid, self.occupancy_topic, 10)
        self.create_subscription(PoseStamped, self.pose_topic, self.pose_cb, 10)
        self.create_subscription(PointCloud2, self.pc_topic, self.pc_cb, 10)

        # Timers
        self.publish_timer = self.create_timer(1.0/self.publish_hz, self.publish_map)

        # Flags
        self.received_pose = False
        self.received_pc   = False

    def pose_cb(self, msg):
        self.received_pose = True
        o = msg.pose.orientation
        t = msg.pose.position
        self.Qx, self.Qy, self.Qz, self.Qw = o.x, o.y, o.z, o.w
        self.pose_x, self.pose_y, self.pose_z = t.x, t.y, t.z

    def pc_cb(self, msg):
        if not self.received_pc:
            self.received_pc = True
        points = pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)
        pts = np.fromiter(points, dtype=np.float32).reshape(-1,3)
        self.update_map(pts)

    def prob_to_log_odds(self, p):
        return np.log(p / (1.0 - p))

    def bresenham(self, x0, y0, x1, y1):
        cells = []
        dx = abs(x1-x0); dy = abs(y1-y0)
        sx = 1 if x1>x0 else -1; sy = 1 if y1>y0 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            cells.append((x,y))
            if x==x1 and y==y1:
                break
            e2 = 2*err
            if e2 > -dy:
                err -= dy; x += sx
            if e2 < dx:
                err += dx; y += sy
        return cells

    def update_map(self, pts):
        if not self.received_pose:
            return
        rx = int((self.pose_x - self.grid_origin[0]) / self.grid_resolution)
        ry = int((self.pose_y - self.grid_origin[1]) / self.grid_resolution)
        R = quat2mat([self.Qw, self.Qx, self.Qy, self.Qz])
        for p in pts:
            w = R.dot(p) + np.array([self.pose_x, self.pose_y, self.pose_z])
            gx = int((w[0]-self.grid_origin[0]) / self.grid_resolution)
            gy = int((w[1]-self.grid_origin[1]) / self.grid_resolution)
            ray = self.bresenham(rx, ry, gx, gy)
            for cell in ray[:-1]:
                val = self.occupancy_map.get(cell, self.l0) + self.l_free
                self.occupancy_map[cell] = np.clip(val, self.l_min, self.l_max)
                self.min_x = min(self.min_x, cell[0]); self.max_x = max(self.max_x, cell[0])
                self.min_y = min(self.min_y, cell[1]); self.max_y = max(self.max_y, cell[1])
            end = ray[-1]
            val = self.occupancy_map.get(end, self.l0) + self.l_occ
            self.occupancy_map[end] = np.clip(val, self.l_min, self.l_max)
            self.min_x = min(self.min_x, end[0]); self.max_x = max(self.max_x, end[0])
            self.min_y = min(self.min_y, end[1]); self.max_y = max(self.max_y, end[1])

    def publish_map(self):
        if not self.occupancy_map:
            return
        min_x, max_x = int(self.min_x), int(self.max_x)
        min_y, max_y = int(self.min_y), int(self.max_y)
        width  = max_x - min_x + 1
        height = max_y - min_y + 1
        grid = np.full((height, width), self.l0, dtype=np.float32)
        for (i,j), l in self.occupancy_map.items():
            grid[j-min_y, i-min_x] = l
        prob = 1.0 - 1.0/(1.0+np.exp(grid))
        data = (prob*100).astype(np.int8).flatten().tolist()
        og = OccupancyGrid()
        og.header = Header()
        og.header.stamp = self.get_clock().now().to_msg()
        og.header.frame_id = self.frame_id
        og.info.resolution = self.grid_resolution
        og.info.width = width
        og.info.height = height
        og.info.origin.position.x = self.grid_origin[0] + min_x*self.grid_resolution
        og.info.origin.position.y = self.grid_origin[1] + min_y*self.grid_resolution
        og.info.origin.orientation.w = 1.0
        og.data = data
        self.pub.publish(og)


def main(args=None):
    rclpy.init(args=args)
    cfg = load_config()
    node = OccupancyGridNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()
