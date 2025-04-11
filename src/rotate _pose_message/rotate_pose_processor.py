# rotate_pose_processor.py
import rclpy
from rclpy.node import Node  
from geometry_msgs.msg import PoseStamped, Point
import numpy as np
from rotate_pose_publisher import RotatePosePublisher
from transforms3d.euler import euler2mat
from transforms3d.quaternions import quat2mat, mat2quat
from rcl_interfaces.msg import SetParametersResult
from visualization_msgs.msg import Marker
from std_srvs.srv import Trigger

class RotatePoseProcessor(Node):
    def __init__(self, config):
        super().__init__('rotate_pose_processor')

        # Access configuration parameters
        self.rotate_pose_processing = config['rotate_pose_processing']
        self.pose_topic = self.rotate_pose_processing['pose_topic']
        self.rotate_pose_topic = self.rotate_pose_processing['rotate_pose_topic']
        self.frame_id = self.rotate_pose_processing['frame_id']

        # Declare and initialize parameters (angles in degrees using roll, pitch, yaw)
        initial_roll  = self.rotate_pose_processing.get('roll', 0.0)
        initial_pitch = self.rotate_pose_processing.get('pitch', 0.0)
        initial_yaw   = self.rotate_pose_processing.get('yaw', 0.0)

        self.declare_parameter('roll', initial_roll)
        self.declare_parameter('pitch', initial_pitch)
        self.declare_parameter('yaw', initial_yaw)

        self.roll  = self.get_parameter('roll').value
        self.pitch = self.get_parameter('pitch').value
        self.yaw   = self.get_parameter('yaw').value

        # Convert degrees to radians and compute the initial rotation matrix.
        # In our convention: roll (x), pitch (y), yaw (z)
        self.rotation_matrix = euler2mat(np.deg2rad(self.roll), 
                                         np.deg2rad(self.pitch), 
                                         np.deg2rad(self.yaw), 
                                         axes='sxyz')
        self.get_logger().info(
            f"Initial rotation matrix computed with roll={self.roll}°, pitch={self.pitch}°, yaw={self.yaw}°"
        )

        # Set up dynamic parameter update callback for angle changes
        self.add_on_set_parameters_callback(self.update_rotation_params)

        # Initialize the publisher for the rotated pose and publisher for markers
        self.publisher = RotatePosePublisher(self, self.rotate_pose_topic)
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)

        # Create a service to clear markers upon request
        self.clear_marker_srv = self.create_service(Trigger, 'clear_markers', self.handle_clear_markers)

        # Store the history of rotated poses as a list of geometry_msgs/Point
        self.path_points = []

        # Create subscription to the incoming pose topic
        self.create_subscription(PoseStamped, self.pose_topic, self.rotate_pose_callback, 10)

    def update_rotation_params(self, params):
        """Update roll, pitch, and yaw parameters (in degrees), recompute the rotation matrix, and reset the path history."""
        updated = False
        for param in params:
            if param.name == 'roll':
                self.roll = param.value
                updated = True
            elif param.name == 'pitch':
                self.pitch = param.value
                updated = True
            elif param.name == 'yaw':
                self.yaw = param.value
                updated = True
        if updated:
            self.rotation_matrix = euler2mat(np.deg2rad(self.roll), 
                                             np.deg2rad(self.pitch), 
                                             np.deg2rad(self.yaw), 
                                             axes='sxyz')
            self.get_logger().info(
                f"Updated rotation matrix with roll={self.roll}°, pitch={self.pitch}°, yaw={self.yaw}°"
            )
            # Clear the current path history when parameters change.
            self.path_points = []
            # Publish a DELETE marker to clear any existing marker in RViz.
            delete_marker = Marker()
            delete_marker.header.frame_id = self.frame_id
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.ns = "rotate_pose_path"
            delete_marker.id = 1
            delete_marker.action = Marker.DELETE
            self.marker_pub.publish(delete_marker)
            self.get_logger().info("Path history has been cleared due to parameter update.")
        return SetParametersResult(successful=True)

    def rotate_pose_callback(self, msg):
        """Apply the rotation to the pose (position and orientation),
        publish the rotated pose, and update the path marker."""
        try:
            # --- Process Position Rotation ---
            # Extract original pose position
            pose_x = msg.pose.position.x
            pose_y = msg.pose.position.y
            pose_z = msg.pose.position.z

            # Convert into a numpy vector and apply the rotation
            input_pose = np.array([pose_x, pose_y, pose_z])
            rotated_position = self.rotation_matrix.dot(input_pose)

            # Create a new PoseStamped message with the rotated coordinates
            rotated_msg = PoseStamped()
            rotated_msg.header.stamp = self.get_clock().now().to_msg()
            rotated_msg.header.frame_id = self.frame_id
            rotated_msg.pose.position.x = float(rotated_position[0])
            rotated_msg.pose.position.y = float(rotated_position[1])
            rotated_msg.pose.position.z = float(rotated_position[2])

            # --- Process Orientation Rotation ---
            # Extract the original quaternion from ROS (x, y, z, w)
            # and convert it into transforms3d order [w, x, y, z]
            orig_quat = [
                msg.pose.orientation.w,
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z
            ]

            # Convert the original quaternion to a 3x3 rotation matrix
            orig_orientation_matrix = quat2mat(orig_quat)

            # Apply the custom rotation on the original orientation.
            # This multiplies the rotation matrices so that the new orientation
            # is the result of applying self.rotation_matrix to the original orientation.
            rotated_orientation_matrix = self.rotation_matrix.dot(orig_orientation_matrix)

            # Convert the new rotation matrix back into a quaternion.
            # The returned quaternion is in the order [w, x, y, z].
            new_quat = mat2quat(rotated_orientation_matrix)

            # Assign the new quaternion to the rotated pose message.
            # Note: Convert back to ROS order (x, y, z, w)
            rotated_msg.pose.orientation.x = new_quat[1]
            rotated_msg.pose.orientation.y = new_quat[2]
            rotated_msg.pose.orientation.z = new_quat[3]
            rotated_msg.pose.orientation.w = new_quat[0]

            # --- Publish the Updated Pose and Visualize ---
            # Publish the rotated pose message with both rotated position and orientation.
            self.publisher.publish_rotate_pose(rotated_msg)

            # Append current rotated position to the path history for visualization.
            pt = Point()
            pt.x = rotated_msg.pose.position.x
            pt.y = rotated_msg.pose.position.y
            pt.z = rotated_msg.pose.position.z
            self.path_points.append(pt)

            # Create and publish a line strip marker to visualize the path history.
            path_marker = Marker()
            path_marker.header.frame_id = self.frame_id
            path_marker.header.stamp = self.get_clock().now().to_msg()
            path_marker.ns = "rotate_pose_path"
            path_marker.id = 1
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            path_marker.points = self.path_points
            path_marker.scale.x = 0.05  # line width
            path_marker.color.r = 0.0
            path_marker.color.g = 1.0
            path_marker.color.b = 0.0
            path_marker.color.a = 1.0  # fully opaque
            path_marker.lifetime.sec = 0  # persists indefinitely

            self.marker_pub.publish(path_marker)
            self.get_logger().debug("Published rotated pose (position and orientation) and path marker.")

        except Exception as e:
            self.get_logger().error(f"Error in rotate_pose_callback: {e}")

    def handle_clear_markers(self, request, response):
        """
        Clear markers from RViz and reset the path history.
        This will remove the currently published markers and clear the internal path list,
        so that the new path will start fresh.
        """
        self.get_logger().info("Clearing markers and resetting path history...")
        
        # Publish a DELETE marker to remove the current path marker.
        delete_marker = Marker()
        delete_marker.header.frame_id = self.frame_id
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.ns = "rotate_pose_path"
        delete_marker.id = 1
        delete_marker.action = Marker.DELETE
        self.marker_pub.publish(delete_marker)
        
        # Reset the internal path history.
        self.path_points = []

        response = Trigger.Response()
        response.success = True
        response.message = "Markers cleared and path history reset."
        return response