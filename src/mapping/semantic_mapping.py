# -----------------------------------
# Import Statements
# -----------------------------------
import os
import yaml
import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point

# -----------------------------------
# Configuration Loader Function
# -----------------------------------
def load_config():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, '../../config/semantic_mapping_config.yaml')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(f"Config file not found at: {config_file}")

# -----------------------------------
# ROS2 Node Definition
# -----------------------------------
class SemanticMapNode(Node):

    def __init__(self, config):
        super().__init__('semantic_map_node')
        nc = config['semantic_mapping']

        # Declare & read parameters
        self.semantic_map_topic = self.declare_parameter('semantic_map_topic', nc['semantic_map_topic']).value
        self.point_cloud_topic = self.declare_parameter('point_cloud_topic', nc['point_cloud_topic']).value
        self.frame_id          = self.declare_parameter('frame_id', nc['frame_id']).value
        self.resolution        = self.declare_parameter('grid_resolution', nc['grid_resolution']).value

        # Publisher & Subscriber
        self.map_pub = self.create_publisher(Marker, self.semantic_map_topic, 10)
        self.create_subscription(PointCloud2,
                                 self.point_cloud_topic,
                                 self.pc_callback,
                                 10)

    def pc_callback(self, msg: PointCloud2):
        # --- 1) Build a NumPy dtype matching the PointCloud2 fields ---
        dtype_map = {
            PointField.INT8:   np.int8,
            PointField.UINT8:  np.uint8,
            PointField.INT16:  np.int16,
            PointField.UINT16: np.uint16,
            PointField.INT32:  np.int32,
            PointField.UINT32: np.uint32,
            PointField.FLOAT32: np.float32,
            PointField.FLOAT64: np.float64,
        }
        names   = [f.name for f in msg.fields]
        formats = [dtype_map[f.datatype] for f in msg.fields]
        offsets = [f.offset for f in msg.fields]
        pc_dtype = np.dtype({
            'names':   names,
            'formats': formats,
            'offsets': offsets,
            'itemsize': msg.point_step
        })

        # --- 2) Interpret raw buffer as structured array ---
        pc_arr = np.frombuffer(msg.data, dtype=pc_dtype)
        if msg.is_bigendian:
            pc_arr = pc_arr.byteswap().newbyteorder()

        # --- 3) Mask-out invalid points ---
        # Assume fields 'x','y','z' exist
        valid = np.isfinite(pc_arr['x']) & np.isfinite(pc_arr['y']) & np.isfinite(pc_arr['z'])
        xyz = np.vstack((pc_arr['x'][valid],
                         pc_arr['y'][valid],
                         pc_arr['z'][valid])).T  # shape (N,3)

        # --- 4) Extract & unpack RGB ---
        if 'rgb' in pc_arr.dtype.names:
            raw_rgb = pc_arr['rgb'][valid]
        elif 'rgba' in pc_arr.dtype.names:
            raw_rgb = pc_arr['rgba'][valid]
        else:
            raise RuntimeError("PointCloud2 has no 'rgb' or 'rgba' field")
        # reinterpret bits if stored as float
        if raw_rgb.dtype == np.float32:
            rgb_uint = raw_rgb.view(np.uint32)
        else:
            rgb_uint = raw_rgb.astype(np.uint32)
        # split channels
        r = ((rgb_uint >> 16) & 0xFF).astype(np.float64)
        g = ((rgb_uint >>  8) & 0xFF).astype(np.float64)
        b = ( rgb_uint        & 0xFF).astype(np.float64)

        # --- 5) Compute grid indices i,j for each point ---
        coords = xyz[:, :2] / self.resolution
        ij = np.floor(coords).astype(np.int64)  # shape (N,2)

        # --- 6) Group by (i,j) via structured array + np.unique ---
        ij_dtype = np.dtype([('i', np.int64), ('j', np.int64)])
        ij_struct = np.empty(ij.shape[0], dtype=ij_dtype)
        ij_struct['i'], ij_struct['j'] = ij[:,0], ij[:,1]

        unique_cells, inverse = np.unique(ij_struct, return_inverse=True)
        counts = np.bincount(inverse)
        n_cells = unique_cells.shape[0]

        # --- 7) Sum positions & colors per cell ---
        sum_xyz = np.zeros((n_cells,3), dtype=np.float64)
        sum_r   = np.zeros(n_cells,    dtype=np.float64)
        sum_g   = np.zeros(n_cells,    dtype=np.float64)
        sum_b   = np.zeros(n_cells,    dtype=np.float64)

        np.add.at(sum_xyz, inverse, xyz)
        np.add.at(sum_r,   inverse, r)
        np.add.at(sum_g,   inverse, g)
        np.add.at(sum_b,   inverse, b)

        # --- 8) Compute averages ---
        avg_xyz = sum_xyz / counts[:,None]
        avg_rgb = np.vstack((sum_r, sum_g, sum_b)).T / counts[:,None] / 255.0  # shape (n_cells,3)

        # --- 9) Build & publish the Marker ---
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp    = self.get_clock().now().to_msg()
        marker.ns              = "semantic_map"
        marker.id              = 0
        marker.type            = Marker.CUBE_LIST
        marker.action          = Marker.ADD
        marker.scale.x         = self.resolution
        marker.scale.y         = self.resolution
        marker.scale.z         = self.resolution
        marker.pose.orientation.w = 1.0

        for idx in range(n_cells):
            x, y, _ = avg_xyz[idx]
            rr, gg, bb = avg_rgb[idx]

            pt = Point(x=float(x), y=float(y), z=0.0)
            col = ColorRGBA(r=float(rr), g=float(gg), b=float(bb), a=1.0)

            marker.points.append(pt)
            marker.colors.append(col)

        self.map_pub.publish(marker)


# -----------------------------------
# Main Entry Point
# -----------------------------------
def main(args=None):
    rclpy.init(args=args)
    cfg = load_config()
    node = SemanticMapNode(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
