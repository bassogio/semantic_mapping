import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from pynput import keyboard  # For keyboard input

class PointCloudKeyboardController(Node):
    def __init__(self):
        super().__init__('point_cloud_keyboard_controller')

        # Initialize rotation angles (in radians)
        self.rotation_angles = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        # Subscriber to the input point cloud topic
        self.point_cloud_sub = self.create_subscription(
            PointCloud2,
            '/camera/pointcloud',
            self.point_cloud_callback,
            10
        )

        # Publisher for the rotated point cloud
        self.point_cloud_pub = self.create_publisher(PointCloud2, '/rotated_pointcloud', 10)

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

    def point_cloud_callback(self, msg):
        """Callback to process and rotate the point cloud."""
        try:
            # Convert PointCloud2 message to numpy array
            points = np.array(list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)))

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

            # Apply the rotation to the point cloud
            rotated_points = points @ rotation_matrix.T

            # Create a new PointCloud2 message
            header = msg.header
            rotated_pc_msg = pc2.create_cloud_xyz32(header, rotated_points)

            # Publish the rotated point cloud
            self.point_cloud_pub.publish(rotated_pc_msg)
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudKeyboardController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
