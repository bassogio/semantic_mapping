#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
import numpy as np

class SegmentationBlockCostmap(Node):
    def __init__(self):
        super().__init__('segmentation_block_costmap_publisher')
        # Publisher for the costmap message
        self.publisher_ = self.create_publisher(OccupancyGrid, 'segmentation_costmap', 10)
        # Timer to publish at 2 Hz (every 0.5 seconds)
        self.timer = self.create_timer(0.5, self.timer_callback)
        
        # Define segmentation label IDs (the colors are defined below for reference)
        #   road          (id: 0)  -> [128, 64, 128]
        #   sidewalk      (id: 1)  -> [244, 35, 232]
        #   building      (id: 2)  -> [70, 70, 70]
        #   wall          (id: 3)  -> [102, 102, 156]
        #   fence         (id: 4)  -> [190, 153, 153]
        #   pole          (id: 5)  -> [153, 153, 153]
        #   traffic light (id: 6)  -> [250, 170, 30]
        #   traffic sign  (id: 7)  -> [220, 220, 0]
        #   vegetation    (id: 8)  -> [107, 142, 35]
        #   terrain       (id: 9)  -> [152, 251, 152]
        #   sky           (id: 10) -> [70, 130, 180]
        #   person        (id: 11) -> [220, 20, 60]
        #   rider         (id: 12) -> [255, 0, 0]
        #   car           (id: 13) -> [0, 0, 142]
        #   truck         (id: 14) -> [0, 0, 70]
        #   bus           (id: 15) -> [0, 60, 100]
        #   train         (id: 16) -> [0, 80, 100]
        #   motorcycle    (id: 17) -> [0, 0, 230]
        #   bicycle       (id: 18) -> [119, 11, 32]
        #   void          (id: 29) -> [0, 0, 0]
        #
        # We’ll use the sorted order of IDs.
        self.allowed_labels = sorted([0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                        10, 11, 12, 13, 14, 15, 16, 17, 18, 29])
        # Set up the block grid parameters: 5 blocks horizontally x 4 blocks vertically (20 blocks total)
        self.blocks_x = 5       # number of blocks along the x-axis
        self.blocks_y = 4       # number of blocks along the y-axis
        self.block_width = 20   # width (in cells) of each block
        self.block_height = 20  # height (in cells) of each block

        # Total occupancy grid size in cells
        self.width = self.blocks_x * self.block_width
        self.height = self.blocks_y * self.block_height

        # Resolution: size of each cell (meters per cell)
        self.resolution = 0.1  # adjust as needed

    def create_costmap_data(self):
        """
        Creates a 2D costmap (as a NumPy array) arranged in blocks.
        Each block is filled with one of the segmentation label IDs.
        The blocks are arranged in row-major order according to self.allowed_labels.
        If there are more blocks than labels, the labels will repeat.
        """
        # Initialize the costmap array with zeros (shape: height x width)
        costmap = np.zeros((self.height, self.width), dtype=np.int8)
        label_idx = 0  # counter for assigning labels
        
        # Iterate through blocks in row-major order
        for by in range(self.blocks_y):
            for bx in range(self.blocks_x):
                # Get the segmentation label for this block.
                seg_label = self.allowed_labels[label_idx % len(self.allowed_labels)]
                label_idx += 1
                # Compute pixel boundaries for the current block.
                y_start = by * self.block_height
                y_end = y_start + self.block_height
                x_start = bx * self.block_width
                x_end = x_start + self.block_width
                # Fill the block with the segmentation label.
                costmap[y_start:y_end, x_start:x_end] = seg_label
        return costmap

    def timer_callback(self):
        # Create the occupancy grid message.
        occ_grid = OccupancyGrid()
        occ_grid.header.stamp = self.get_clock().now().to_msg()
        occ_grid.header.frame_id = "map"
        
        # Set metadata for the map.
        occ_grid.info.resolution = self.resolution
        occ_grid.info.width = self.width
        occ_grid.info.height = self.height
        # Set the origin (position and orientation) of the map.
        occ_grid.info.origin.position.x = 0.0
        occ_grid.info.origin.position.y = 0.0
        occ_grid.info.origin.position.z = 0.0
        occ_grid.info.origin.orientation.w = 1.0  # no rotation
        
        # Generate the costmap data.
        costmap_array = self.create_costmap_data()
        # Flatten the 2D array into a 1D list in row-major order.
        occ_grid.data = costmap_array.flatten().tolist()

        # Publish the OccupancyGrid message.
        self.publisher_.publish(occ_grid)
        self.get_logger().info("Published costmap message with each segmentation color block.")

def main(args=None):
    rclpy.init(args=args)
    node = SegmentationBlockCostmap()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
