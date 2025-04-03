import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker

class OdometryArrowPublisher(Node):
    def __init__(self):
        super().__init__('odometry_arrow_publisher')

        self.subscription = self.create_subscription(
            Odometry,
            '/davis/left/odometry',
            self.odometry_callback,
            10
        )
        self.publisher = self.create_publisher(
            Marker,
            '/odometry_arrow',
            10
        )
        self.get_logger().info('Publishing odometry arrow in "map" frame.')

    def odometry_callback(self, msg: Odometry):
        marker = Marker()
        marker.header.frame_id = 'map'  # <- Set fixed frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'odom_arrow'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Use odometry pose
        marker.pose.position = msg.pose.pose.position
        marker.pose.orientation = msg.pose.pose.orientation

        # Arrow size
        marker.scale.x = 0.5  # shaft length
        marker.scale.y = 0.1  # shaft diameter
        marker.scale.z = 0.1  # head diameter

        # Arrow color (red)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Arrow persists until updated
        marker.lifetime.sec = 0

        self.publisher.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = OdometryArrowPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
