import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np

class OccupancyGridMapper(Node):
    def __init__(self):
        super().__init__('occupancy_grid_mapper')

        # Subscribe to the point cloud topic
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/pointcloud',  # Ensure this matches your publisher
            self.point_cloud_callback,
            10
        )

        # Occupancy grid publisher
        self.occupancy_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)

        # Grid parameters
        self.grid_resolution = 0.1  # Grid cell size in meters
        self.grid_size = 1000  # Number of cells in each dimension
        self.grid_origin = (-50.0, -50.0)  # Origin of the grid in meters

    def point_cloud_callback(self, msg):
        """Convert PointCloud2 data to an occupancy grid and publish it."""
        # Extract points and filter by Z > 0.5
        points = np.array([
            (p[0], p[1]) for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            if p[2] > 0.2  # Only include points where Z > 0.2
        ])

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
        occ_grid_msg.header.frame_id = "map"
        occ_grid_msg.info.resolution = self.grid_resolution
        occ_grid_msg.info.width = self.grid_size
        occ_grid_msg.info.height = self.grid_size
        occ_grid_msg.info.origin.position.x = self.grid_origin[0]
        occ_grid_msg.info.origin.position.y = self.grid_origin[1]
        occ_grid_msg.info.origin.position.z = 0.0

        # Flatten grid and assign to message
        occ_grid_msg.data = grid.flatten().tolist()

        # Publish the occupancy grid
        self.occupancy_pub.publish(occ_grid_msg)
        self.get_logger().info("Published Occupancy Grid with Z > 0.1")

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
