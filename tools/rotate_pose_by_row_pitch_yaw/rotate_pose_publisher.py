# rotate_pose_publisher.py
from geometry_msgs.msg import PoseStamped

class RotatePosePublisher:
    def __init__(self, node, rotate_pose_topic):
        # Use the provided node to create a publisher for PoseStamped messages.
        self.node = node
        self.rotate_pose_pub = self.node.create_publisher(PoseStamped, rotate_pose_topic, 10)

    def publish_rotate_pose(self, rotated_msg):
        self.rotate_pose_pub.publish(rotated_msg)
        # self.node.get_logger().info(f"Published rotated pose: {rotated_msg}")