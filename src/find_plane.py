import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class PlaneCheckNode(Node):
    def __init__(self):
        super().__init__('plane_check_node')
        self.sub = self.create_subscription(
            PointCloud2,
            '/velodyne_point_cloud',
            self.callback,
            10
        )

    def callback(self, msg):
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if not points:
            self.get_logger().info("No points in cloud.")
            return
        z_values = np.array([p[2] for p in points])
        if np.allclose(z_values, 0.0, atol=0.05):
            self.get_logger().info("Points lie mostly on the XY plane.")
        else:
            self.get_logger().info("Points are distributed in 3D space.")

def main():
    rclpy.init()
    node = PlaneCheckNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
