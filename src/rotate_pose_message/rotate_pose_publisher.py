# rotate_pose_publisher.py
import std_msgs.msg
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped

class RotatePosePublisher:
    def __init__(self, node, rotated_pose_topic):
        # Instead of creating a new node, simply save the provided node.
        self.node = node
        # Use the node's create_publisher to initialize the publisher.
        self.rotated_pose_pub = node.create_publisher(PoseStamped, rotated_pose_topic, 10)

    def publish_rotate_pose(self, rotated_position, rotated_quat, rotate_pose_processing):
        """Format the rotated pose and publish."""
        # Create header using the parent node's clock.
        header = std_msgs.msg.Header()
        header.stamp = self.node.get_clock().now().to_msg()
        # Use the 'frame_id' from the config, with a default fallback if needed.
        header.frame_id = rotate_pose_processing.get('frame_id', 'world')
        
        # Create the PoseStamped message.
        pose_stamped = PoseStamped()
        pose_stamped.header = header
        pose_stamped.pose.position.x = float(rotated_position[0])
        pose_stamped.pose.position.y = float(rotated_position[1])
        pose_stamped.pose.position.z = float(rotated_position[2])
        # Note: The transforms3d library returns quaternions as [w, x, y, z].
        # ROS expects quaternion fields in the order: x, y, z, w.
        pose_stamped.pose.orientation.x = rotated_quat[1]
        pose_stamped.pose.orientation.y = rotated_quat[2]
        pose_stamped.pose.orientation.z = rotated_quat[3]
        pose_stamped.pose.orientation.w = rotated_quat[0]

        # Publish the rotated pose.
        self.rotated_pose_pub.publish(pose_stamped)
