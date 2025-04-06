import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from message_filters import Subscriber, ApproximateTimeSynchronizer

class ImageDepthSyncNode(Node):
    def __init__(self):
        super().__init__('image_depth_sync_node')

        # Set up subscribers to the image and depth topics
        self.image_sub = Subscriber(self, Image, '/davis/left/image_raw')
        self.depth_image_sub = Subscriber(self, Image, '/davis/left/depth_image_raw')

        # Use ApproximateTimeSynchronizer to sync the two streams
        self.ts = ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_image_sub],
            queue_size=10,
            slop=0.1  # 50 ms tolerance
        )
        self.ts.registerCallback(self.synced_callback)

    def synced_callback(self, image_msg, depth_msg):
        self.get_logger().info(
            f"Received synced image + depth: "
            f"RGB stamp = {image_msg.header.stamp.sec}.{image_msg.header.stamp.nanosec}, "
            f"Depth stamp = {depth_msg.header.stamp.sec}.{depth_msg.header.stamp.nanosec}"
        )
        # Add your image + depth processing logic here

def main(args=None):
    rclpy.init(args=args)
    node = ImageDepthSyncNode()
    
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
