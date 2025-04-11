#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from transforms3d.euler import euler2mat
from transforms3d.quaternions import quat2mat, mat2quat

class PoseRotatorNode(Node):
    def __init__(self):
        super().__init__('pose_rotator_node')
        
        # Topic definitions
        self.input_topic = '/davis/left/pose'
        self.output_topic = '/davis/rotated_pose'
        self.marker_topic = 'visualization_marker'
        
        # Create a subscriber to receive PoseStamped messages
        self.subscription = self.create_subscription(
            PoseStamped,
            self.input_topic,
            self.pose_callback,
            10
        )
        # Create a publisher to publish rotated PoseStamped messages
        self.publisher = self.create_publisher(PoseStamped, self.output_topic, 10)
        # Create a publisher to publish the visualization marker for the path
        self.marker_pub = self.create_publisher(Marker, self.marker_topic, 10)
        
        # Store the rotated positions for the path marker
        self.path_points = []
        
        # Fixed rotation parameters (in degrees)
        self.roll = 277.5
        self.pitch = 7.0
        self.yaw = 0.0
        self.get_logger().info(
            f"Rotation angles set to roll={self.roll}°, pitch={self.pitch}°, yaw={self.yaw}°"
        )
        
        # Convert fixed rotation to a matrix using the 'sxyz' convention.
        roll_rad = np.deg2rad(self.roll)
        pitch_rad = np.deg2rad(self.pitch)
        yaw_rad = np.deg2rad(self.yaw)
        self.rotation_matrix = euler2mat(roll_rad, pitch_rad, yaw_rad, axes='sxyz')
        
    def pose_callback(self, msg: PoseStamped):
        try:
            # Extract original position and compute the rotated position.
            pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
            rotated_pos = self.rotation_matrix.dot(pos)
            
            # Append the rotated position to the path for the marker.
            new_point = Point()
            new_point.x = float(rotated_pos[0])
            new_point.y = float(rotated_pos[1])
            new_point.z = float(rotated_pos[2])
            self.path_points.append(new_point)
            
            # Process the orientation:
            # Convert the incoming ROS quaternion (x, y, z, w) to [w, x, y, z] for transforms3d.
            input_quat = np.array([
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z
            ])
            orig_orientation_matrix = quat2mat(input_quat)
            
            # Rotate the orientation using the fixed rotation matrix.
            rotated_orientation_matrix = self.rotation_matrix.dot(orig_orientation_matrix)
            rotated_quat = mat2quat(rotated_orientation_matrix)  # returns [w, x, y, z]
            
            # Prepare a new PoseStamped message with the rotated pose.
            rotated_msg = PoseStamped()
            rotated_msg.header = msg.header
            rotated_msg.pose.position.x = float(rotated_pos[0])
            rotated_msg.pose.position.y = float(rotated_pos[1])
            rotated_msg.pose.position.z = float(rotated_pos[2])
            # Convert rotated quaternion back to ROS order (x, y, z, w)
            rotated_msg.pose.orientation.x = rotated_quat[1]
            rotated_msg.pose.orientation.y = rotated_quat[2]
            rotated_msg.pose.orientation.z = rotated_quat[3]
            rotated_msg.pose.orientation.w = rotated_quat[0]
            
            # Publish the rotated pose.
            self.publisher.publish(rotated_msg)
            
            # Create and update a marker to visualize the path.
            marker = Marker()
            marker.header = msg.header
            marker.ns = "pose_path"
            marker.id = 0
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.05  # Line width.
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 0
            marker.points = self.path_points
            
            self.marker_pub.publish(marker)
            
            self.get_logger().info("Published rotated pose and updated path marker.")
        except Exception as e:
            self.get_logger().error(f"Error processing pose: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PoseRotatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt detected, shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
