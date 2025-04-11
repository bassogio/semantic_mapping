#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class OccupancyGridMapper(Node):
    def __init__(self):
        super().__init__('occupancy_grid_mapper')
        # Subscription to the point cloud
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/pointcloud',  # Adjust if needed
            self.point_cloud_callback,
            10
        )

        # Publisher for the occupancy grid
        self.occupancy_pub = self.create_publisher(OccupancyGrid, '/occupancy_grid', 10)

        # Publisher for markers (object markers)
        self.marker_pub = self.create_publisher(Marker, '/object_markers', 10)

        # Grid parameters
        self.grid_resolution = 0.1  # Grid cell size in meters
        self.grid_size = 1000       # Number of cells in each dimension (grid will be grid_size x grid_size)
        self.grid_origin = (-50.0, -50.0)  # Origin of the grid in world coordinates

    def point_cloud_callback(self, msg):
        """
        Process the point cloud message to create an occupancy grid.
        Points above a certain z-threshold are considered to indicate an object.
        """
        # Extract points from the point cloud with z > 0.2 meters as an example threshold
        points = np.array([
            (p[0], p[1])
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            if p[2] > 0.2
        ])

        # Create an occupancy grid:
        # unknown: -1, free: 0, occupied: 100
        grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)

        # Iterate over the extracted points and mark the corresponding grid cell as occupied
        for x, y in points:
            grid_x = int((x - self.grid_origin[0]) / self.grid_resolution)
            grid_y = int((y - self.grid_origin[1]) / self.grid_resolution)
            if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                grid[grid_y, grid_x] = 100

        # Build the OccupancyGrid message
        occ_grid_msg = OccupancyGrid()
        occ_grid_msg.header.stamp = self.get_clock().now().to_msg()
        occ_grid_msg.header.frame_id = "map"
        occ_grid_msg.info.resolution = self.grid_resolution
        occ_grid_msg.info.width = self.grid_size
        occ_grid_msg.info.height = self.grid_size
        occ_grid_msg.info.origin.position.x = self.grid_origin[0]
        occ_grid_msg.info.origin.position.y = self.grid_origin[1]
        occ_grid_msg.info.origin.position.z = 0.0
        occ_grid_msg.info.origin.orientation.w = 1.0  # No rotation
        occ_grid_msg.data = grid.flatten().tolist()

        # Publish the occupancy grid
        self.occupancy_pub.publish(occ_grid_msg)

        # Prepare Marker message (using CUBE_LIST) for each occupied cell
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "map"
        marker.ns = "objects"
        marker.id = 0
        marker.type = Marker.CUBE_LIST  # Visualizes each point as a cube
        marker.action = Marker.ADD

        # Set the size for each cube in the marker list (each cube matches a grid cell)
        marker.scale.x = self.grid_resolution
        marker.scale.y = self.grid_resolution
        marker.scale.z = 0.1  # A fixed height for visualization

        # Set a color (red) and transparency for the markers
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8

        # Iterate through the grid and add a point to the marker for every occupied cell
        for grid_y in range(self.grid_size):
            for grid_x in range(self.grid_size):
                if grid[grid_y, grid_x] == 100:
                    # Compute the center of the cell in world coordinates
                    world_x = self.grid_origin[0] + grid_x * self.grid_resolution + self.grid_resolution / 2.0
                    world_y = self.grid_origin[1] + grid_y * self.grid_resolution + self.grid_resolution / 2.0
                    pt = Point()
                    pt.x = world_x
                    pt.y = world_y
                    pt.z = 0.0  # Adjust the z value if you need the marker offset vertically
                    marker.points.append(pt)

        # Publish the marker
        self.marker_pub.publish(marker)
        self.get_logger().info("Published occupancy grid and object markers.")


def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
