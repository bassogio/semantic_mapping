#rotate
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class PoseFixer(Node):
    def __init__(self):
        super().__init__('pose_fixer')
        self.sub = self.create_subscription(PoseStamped, '/davis/left/pose', self.pose_callback, 10)
        self.pub = self.create_publisher(PoseStamped, '/davis/left/pose_fixed', 10)

    def pose_callback(self, msg):
        fixed_pose = PoseStamped()
        fixed_pose.header = msg.header
        fixed_pose.pose.position.x = msg.pose.position.x / 4
        fixed_pose.pose.position.y = msg.pose.position.y / 4
        fixed_pose.pose.position.z = 0.0
        fixed_pose.pose.orientation.x = msg.pose.orientation.x
        fixed_pose.pose.orientation.y = msg.pose.orientation.y  
        fixed_pose.pose.orientation.z = 0.0 
        self.pub.publish(fixed_pose)

rclpy.init()
node = PoseFixer()
rclpy.spin(node)

"""permutations = [
    #('x, y, z', (0, 1, 2)),
    #('x, z, y', (0, 2, 1)),
    #('y, x, z', (1, 0, 2)),
    #('y, z, x', (1, 2, 0)),
    ('z, x, y', (2, 0, 1)),
    ('z, y, x', (2, 1, 0))
]"""