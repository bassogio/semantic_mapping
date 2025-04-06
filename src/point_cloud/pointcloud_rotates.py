import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

class VehicleArrowNode(Node):
    def __init__(self):
        super().__init__('vehicle_arrow_node')
        
        # Subscriber to vehicle pose
        self.pose_subscriber = self.create_subscription(
            PoseStamped,
            '/davis/left/pose',  # Change this topic to the actual pose topic
            self.pose_callback,
            10
        )

        # Publisher for the arrow marker
        self.marker_publisher = self.create_publisher(Marker, '/vehicle/arrow', 10)
        
        # Initialize marker
        self.arrow_marker = Marker()
        self.arrow_marker.header.frame_id = 'map'  # Make sure this is the correct frame
        self.arrow_marker.type = Marker.ARROW
        self.arrow_marker.action = Marker.ADD
        self.arrow_marker.scale.x = 2.5  # Length of the arrow
        self.arrow_marker.scale.y = 0.2  # Width of the arrow
        self.arrow_marker.scale.z = 0.2  # Height of the arrow
        self.arrow_marker.color.r = 1.0  # Red
        self.arrow_marker.color.g = 0.0  # Green
        self.arrow_marker.color.b = 0.0  # Blue
        self.arrow_marker.color.a = 1.0  # Alpha (opacity)
    
    def pose_callback(self, msg: PoseStamped):
        # Update arrow position and orientation based on vehicle pose
        self.arrow_marker.header.stamp = self.get_clock().now().to_msg()
        self.arrow_marker.pose.position = msg.pose.position
        self.arrow_marker.pose.orientation = msg.pose.orientation
        
        # Publish the updated marker
        self.marker_publisher.publish(self.arrow_marker)


def main(args=None):
    rclpy.init(args=args)
    node = VehicleArrowNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
