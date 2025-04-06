import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from pynput import keyboard  # For keyboard input
import numpy as np

class PoseStampedKeyboardController(Node):
    def __init__(self):
        super().__init__('pose_stamped_keyboard_controller')

        # Initialize rotation angles (in radians)
        self.rotation_angles = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # Subscriber to the input PoseStamped topic
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/davis/left/pose',
            self.pose_callback,
            10
        )

        # Publisher for the rotated PoseStamped
        self.pose_pub = self.create_publisher(PoseStamped, '/rotated_pose', 10)

        # Start listening for keyboard inputs
        self.listener = keyboard.Listener(on_press=self.on_key_press)
        self.listener.start()

    def on_key_press(self, key):
        """Handle keyboard input to adjust rotation angles."""
        try:
            if key.char == 'w':  # Rotate up (increase x-axis rotation)
                self.rotation_angles['x'] += 0.1
            elif key.char == 's':  # Rotate down (decrease x-axis rotation)
                self.rotation_angles['x'] -= 0.1
            elif key.char == 'a':  # Rotate left (increase y-axis rotation)
                self.rotation_angles['y'] += 0.1
            elif key.char == 'd':  # Rotate right (decrease y-axis rotation)
                self.rotation_angles['y'] -= 0.1
            elif key.char == 'q':  # Rotate counterclockwise (increase z-axis rotation)
                self.rotation_angles['z'] += 0.1
            elif key.char == 'e':  # Rotate clockwise (decrease z-axis rotation)
                self.rotation_angles['z'] -= 0.1

            self.get_logger().info(f"Updated rotation angles: {self.rotation_angles}")
        except AttributeError:
            pass  # Ignore special keys

    def pose_callback(self, msg):
        """Callback to process and rotate the pose."""
        try:
            # Extract position and orientation (quaternion) from the PoseStamped message
            position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            orientation = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

            # Create rotation matrices for each axis
            rx = np.array([
                [1, 0, 0],
                [0, np.cos(self.rotation_angles['x']), -np.sin(self.rotation_angles['x'])],
                [0, np.sin(self.rotation_angles['x']), np.cos(self.rotation_angles['x'])]
            ])
            ry = np.array([
                [np.cos(self.rotation_angles['y']), 0, np.sin(self.rotation_angles['y'])],
                [0, 1, 0],
                [-np.sin(self.rotation_angles['y']), 0, np.cos(self.rotation_angles['y'])]
            ])
            rz = np.array([
                [np.cos(self.rotation_angles['z']), -np.sin(self.rotation_angles['z']), 0],
                [np.sin(self.rotation_angles['z']), np.cos(self.rotation_angles['z']), 0],
                [0, 0, 1]
            ])

            # Combine rotations
            rotation_matrix = rz @ ry @ rx

            # Apply the rotation to the position
            rotated_position = rotation_matrix @ position

            # For quaternion, use the same rotation matrix to rotate the orientation (you can use a library for more precise quaternion rotations)
            rotated_orientation = orientation  # Keep the original orientation for simplicity

            # Create a new PoseStamped message with the rotated position and orientation
            rotated_pose_stamped = PoseStamped()
            rotated_pose_stamped.header = msg.header  # Copy the header from the original message
            rotated_pose_stamped.pose.position.x = rotated_position[0]
            rotated_pose_stamped.pose.position.y = rotated_position[1]
            rotated_pose_stamped.pose.position.z = rotated_position[2]
            rotated_pose_stamped.pose.orientation.x = rotated_orientation[0]
            rotated_pose_stamped.pose.orientation.y = rotated_orientation[1]
            rotated_pose_stamped.pose.orientation.z = rotated_orientation[2]
            rotated_pose_stamped.pose.orientation.w = rotated_orientation[3]

            # Publish the rotated PoseStamped
            self.pose_pub.publish(rotated_pose_stamped)

        except Exception as e:
            self.get_logger().error(f"Error processing pose: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PoseStampedKeyboardController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
