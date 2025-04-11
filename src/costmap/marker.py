#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
import random

class SemanticMapPublisher(Node):
    def __init__(self):
        super().__init__('semantic_map_publisher')
        # Publisher for the semantic map as a MarkerArray.
        self.publisher_ = self.create_publisher(MarkerArray, 'semantic_map', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        
        # Define map grid parameters.
        self.map_width = 10    # number of cells in x direction
        self.map_height = 10   # number of cells in y direction
        self.resolution = 1.0  # size (meters) of each cell

        # Define a dictionary of colors.
        # Colors are in normalized RGBA format (values between 0 and 1).
        self.colors = {
            'pink':   (1.0, 0.75, 0.8, 1.0),   # you can adjust these values for the desired pink
            'yellow': (1.0, 1.0, 0.0, 1.0),
            'green':  (0.0, 1.0, 0.0, 1.0),
            'red':    (1.0, 0.0, 0.0, 1.0)
        }
        # A list of color names for random selection.
        self.color_keys = list(self.colors.keys())

    def timer_callback(self):
        marker_array = MarkerArray()
        marker_id = 0  # Unique ID for each marker

        # Create a grid of cells.
        for i in range(self.map_height):
            for j in range(self.map_width):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "semantic_map"
                marker.id = marker_id
                marker_id += 1

                # Use a CUBE marker for each cell.
                marker.type = Marker.CUBE
                marker.action = Marker.ADD

                # Set the cell position. We position each cube at the center of its cell.
                marker.pose.position.x = j * self.resolution + self.resolution / 2.0
                marker.pose.position.y = i * self.resolution + self.resolution / 2.0
                marker.pose.position.z = 0.0  # lying flat on the ground
                marker.pose.orientation.x = 0.0
                marker.pose.orientation.y = 0.0
                marker.pose.orientation.z = 0.0
                marker.pose.orientation.w = 1.0

                # Set the size of the cube to match the cell dimensions.
                marker.scale.x = self.resolution
                marker.scale.y = self.resolution
                marker.scale.z = 0.1  # make it a thin tile

                # Choose a random color from the defined set.
                color_name = random.choice(self.color_keys)
                color = self.colors[color_name]
                marker.color.r = color[0]
                marker.color.g = color[1]
                marker.color.b = color[2]
                marker.color.a = color[3]

                # Set lifetime to 0 so markers persist until replaced.
                marker.lifetime.sec = 0

                marker_array.markers.append(marker)

        self.publisher_.publish(marker_array)
        self.get_logger().info("Published semantic map with colored cells.")

def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
