#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from message_filters import Subscriber, ApproximateTimeSynchronizer

class SyncNode(Node):
    def __init__(self):
        super().__init__('sync_node')
        # Create message_filters Subscribers for the specified topics.
        self.depth_sub = Subscriber(self, Image, '/davis/left/depth_image_raw')
        self.image_sub = Subscriber(self, Image, '/davis/left/image_raw')

        # Create an ApproximateTimeSynchronizer to synchronize the two topics.
        # Adjust `queue_size` and `slop` as needed. 'slop' defines the allowable time difference (in seconds).
        self.sync = ApproximateTimeSynchronizer(
            [self.depth_sub, self.image_sub],
            queue_size=50,
            slop=0.1
        )
        self.sync.registerCallback(self.callback)

    def callback(self, depth_msg, image_msg):
        # This callback is called when a pair of messages has been synchronized.
        self.get_logger().info("Synchronized messages received!")
        
        # Logging timestamps for illustration (timestamps in seconds.nanoseconds)
        depth_stamp = depth_msg.header.stamp
        image_stamp = image_msg.header.stamp
        self.get_logger().info(
            f"Depth Timestamp: {depth_stamp.sec}.{depth_stamp.nanosec}"
        )
        self.get_logger().info(
            f"Image Timestamp: {image_stamp.sec}.{image_stamp.nanosec}"
        )
        # Process the messages further as needed.

def main(args=None):
    rclpy.init(args=args)
    node = SyncNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt, shutting down node.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


