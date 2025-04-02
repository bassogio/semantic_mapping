import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

class TimestampComparator(Node):
    def __init__(self):
        super().__init__('timestamp_comparator')

        self.sub_rgb = self.create_subscription(Image, '/davis/left/image_raw', self.rgb_cb, 10)
        self.sub_depth = self.create_subscription(Image, '/davis/left/depth_image_raw', self.depth_cb, 10)

        self.last_rgb_time = None
        self.last_depth_time = None

        self.get_logger().info("⏳ Waiting for image topics...")

    def rgb_cb(self, msg):
        self.last_rgb_time = self._to_sec(msg.header.stamp)
        self.check_difference()

    def depth_cb(self, msg):
        self.last_depth_time = self._to_sec(msg.header.stamp)
        self.check_difference()

    def check_difference(self):
        if self.last_rgb_time is not None and self.last_depth_time is not None:
            diff = self.last_rgb_time - self.last_depth_time
            direction = ""

            if diff > 0:
                direction = "🕓 Depth is behind by {:.3f} seconds".format(abs(diff))
            elif diff < 0:
                direction = "🕓 RGB is behind by {:.3f} seconds".format(abs(diff))
            else:
                direction = "✅ Timestamps are exactly equal!"

            self.get_logger().info(
                f"RGB: {self.last_rgb_time:.6f}, Depth: {self.last_depth_time:.6f}, Δ: {abs(diff):.3f}s → {direction}"
            )

    def _to_sec(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9

def main():
    rclpy.init()
    node = TimestampComparator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
