#!/usr/bin/env python3
import os
import yaml
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2

def load_config():
    """
    Loads the segmentation configuration file.
    Adjust the relative path as needed.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, '../../config/segmentation_config.yaml')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        raise FileNotFoundError(f"Config file not found at: {config_file}")

class OccupancyGridMapperNode(Node):
    def __init__(self):
        super().__init__('occupancy_grid_mapper')
        
        # Load configuration and extract segmentation labels.
        config = load_config()
        # Assuming the configuration file has a section called 'segmentation_processing'
        # that contains a list of label dictionaries.
        self.labels = config['segmentation_processing']['labels']

        # Subscriber to the combined depth + segmentation image.
        # The image is expected to be encoded as "32FC2" (2-channel float image).
        self.subscription = self.create_subscription(
            Image,
            '/camera/segmentation/depth',  # Change to the actual topic name.
            self.combined_depth_callback,
            10
        )
        
        # Publishers for the occupancy grid and markers.
        self.occ_grid_pub = self.create_publisher(OccupancyGrid, 'occupancy_grid', 10)
        self.marker_pub = self.create_publisher(Marker, 'segmentation_markers', 10)

        self.bridge = CvBridge()

        # Parameters (can also be declared as ROS2 parameters):
        self.declare_parameter('depth_threshold', 2.0)  # in meters
        self.declare_parameter('grid_resolution', 0.05)   # meters per grid cell (e.g., 5cm)
        self.declare_parameter('grid_frame', 'map')

        self.depth_threshold = self.get_parameter('depth_threshold').value
        self.grid_resolution = self.get_parameter('grid_resolution').value
        self.grid_frame = self.get_parameter('grid_frame').value

    def get_color_for_seg_id(self, seg_id: int) -> ColorRGBA:
        """
        Extract the color from the loaded configuration based on the segmentation id.
        The configuration color is assumed to be in a list of three integers [R,G,B] (0-255).
        """
        default_color = [127, 127, 127]
        selected_color = default_color
        for label in self.labels:
            if label['id'] == seg_id:
                selected_color = label.get('color', default_color)
                break
        
        # Convert 0-255 integers to floats between 0 and 1.
        color = ColorRGBA()
        color.r = selected_color[0] / 255.0
        color.g = selected_color[1] / 255.0
        color.b = selected_color[2] / 255.0
        color.a = 1.0  # Fully opaque.
        return color

    def combined_depth_callback(self, msg: Image):
        """
        Callback for the combined depth and segmentation image.
        The expected input is a 2-channel image:
           Channel 0: Depth (meters)
           Channel 1: Segmentation id (stored as float; converted to int)
        """
        try:
            # Convert ROS Image message to an OpenCV image (NumPy array)
            np_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC2")
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return

        # Check that we have a 2-channel image.
        if np_img.ndim != 3 or np_img.shape[2] != 2:
            self.get_logger().error(f'Expected a 2-channel image, got shape {np_img.shape}')
            return

        height, width, _ = np_img.shape
        depth_image = np_img[:, :, 0]  # Depth data (meters)
        seg_id_image = np_img[:, :, 1].astype(np.int32)  # Segmentation ids

        # Build the occupancy grid.
        occ_grid = OccupancyGrid()
        occ_grid.header.stamp = self.get_clock().now().to_msg()
        occ_grid.header.frame_id = self.grid_frame
        occ_grid.info.resolution = self.grid_resolution
        occ_grid.info.width = width
        occ_grid.info.height = height
        occ_grid.info.origin.position.x = 0.0
        occ_grid.info.origin.position.y = 0.0
        occ_grid.info.origin.position.z = 0.0
        occ_grid.info.origin.orientation.w = 1.0

        # Create occupancy grid data (occupied if depth is less than the threshold).
        occ_data = np.zeros((height, width), dtype=np.int8)
        occ_data[depth_image < self.depth_threshold] = 100  # Occupied
        
        occ_grid.data = occ_data.flatten().tolist()
        self.occ_grid_pub.publish(occ_grid)
        self.get_logger().info("Published occupancy grid.")

        # Create markers that show, for each occupied cell, a cube with the color from the segmentation.
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.grid_frame
        marker.ns = "segmentation"
        marker.id = 0
        marker.type = Marker.CUBE_LIST
        marker.action = Marker.ADD

        # Cube (marker) size equal to grid resolution (height is arbitrary for visualization).
        marker.scale.x = self.grid_resolution
        marker.scale.y = self.grid_resolution
        marker.scale.z = 0.05

        marker.lifetime.sec = 0  # persistent marker

        marker.points = []
        marker.colors = []

        # Iterate over every cell. For cells with depth below the threshold, add a marker cube.
        for y in range(height):
            for x in range(width):
                if depth_image[y, x] < self.depth_threshold:
                    pt = Point()
                    # Place marker at center of grid cell.
                    pt.x = (x + 0.5) * self.grid_resolution
                    pt.y = (y + 0.5) * self.grid_resolution
                    pt.z = 0.0  # Placed on z=0 plane.
                    marker.points.append(pt)
                    
                    seg_id = seg_id_image[y, x]
                    color = self.get_color_for_seg_id(seg_id)
                    marker.colors.append(color)

        self.marker_pub.publish(marker)
        self.get_logger().info("Published segmentation marker cubes.")

def main(args=None):
    rclpy.init(args=args)
    node = OccupancyGridMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
