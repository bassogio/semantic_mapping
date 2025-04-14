# -----------------------------------
# Import Statements
# -----------------------------------
import os
import yaml
import numpy as np
from transforms3d.euler import euler2mat
from transforms3d.quaternions import quat2mat, mat2quat
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_srvs.srv import Trigger

# -----------------------------------
# Configuration Loader Function
# -----------------------------------
def load_config():
    """
    Load configuration parameters.

    Returns:
        dict: A dictionary containing configuration data.
    """
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, '../../config/rotate_pose_config.yaml')

    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        raise FileNotFoundError(f"Config file not found at: {config_file}")

# -----------------------------------
# ROS2 Node Definition
# -----------------------------------
class RotatedPoseNode(Node):
    def __init__(self, config):
        # Initialize the node with a unique name.
        super().__init__('rotated_pose_node')

        # -------------------------------------------
        # Access the configuration section.
        # -------------------------------------------
        self.node_config = config['rotate_pose_processing']

        # -------------------------------------------
        # Load configuration parameters.
        # -------------------------------------------
        self.rotated_pose_topic  = self.node_config['rotated_pose_topic']
        self.pose_topic          = self.node_config['pose_topic']
        self.marker_topic        = self.node_config['marker_topic']
        self.roll                = self.node_config['roll']
        self.pitch               = self.node_config['pitch']
        self.yaw                 = self.node_config['yaw']
        self.frame_id            = self.node_config['frame_id']
        self.use_service         = self.node_config['use_service']
        self.service_name        = self.node_config['service_name']

        # -------------------------------------------
        # Declare ROS2 parameters for runtime modification.
        # -------------------------------------------
        self.declare_parameter('rotated_pose_topic', self.rotated_pose_topic)
        self.declare_parameter('pose_topic',         self.pose_topic)
        self.declare_parameter('marker_topic',       self.marker_topic)
        self.declare_parameter('roll',               self.roll)
        self.declare_parameter('pitch',              self.pitch)
        self.declare_parameter('yaw',                self.yaw)
        self.declare_parameter('frame_id',           self.frame_id)
        self.declare_parameter('use_service',        self.use_service)
        self.declare_parameter('service_name',       self.service_name)

        # -------------------------------------------
        # Retrieve final parameter values from the parameter server.
        # This allows any runtime overrides (e.g., via launch files) to update these defaults.
        # -------------------------------------------
        self.rotated_pose_topic  = self.get_parameter('rotated_pose_topic').value
        self.pose_topic          = self.get_parameter('pose_topic').value
        self.marker_topic        = self.get_parameter('marker_topic').value
        self.roll                = self.get_parameter('roll').value
        self.pitch               = self.get_parameter('pitch').value
        self.yaw                 = self.get_parameter('yaw').value
        self.frame_id            = self.get_parameter('frame_id').value
        self.use_service         = self.get_parameter('use_service').value
        self.service_name        = self.get_parameter('service_name').value

        # -------------------------------------------
        # Initialize additional attributes needed for processing.
        # These attributes hold pose and orientation data and must be initialized.
        # -------------------------------------------
        # Store the rotated positions for the path marker
        self.path_points = []

        # Convert fixed rotation to a matrix using the 'sxyz' convention.
        roll_rad  = np.deg2rad(self.roll)
        pitch_rad = np.deg2rad(self.pitch)
        yaw_rad   = np.deg2rad(self.yaw)
        self.rotation_matrix = euler2mat(roll_rad, pitch_rad, yaw_rad, axes='sxyz')

        # -------------------------------------------
        # Create Publisher.
        # -------------------------------------------
        self.rotated_pose_publisher = self.create_publisher(PoseStamped, self.rotated_pose_topic, 10)
        self.marker_publisher       = self.create_publisher(Marker,      self.marker_topic,       10)

        # -------------------------------------------
        # Create Subscriber.
        # -------------------------------------------
        self.pose_subscription = self.create_subscription(
            PoseStamped,              # Message type.
            self.pose_topic,          # Topic name.
            self.rotate_pose_callback,# Callback function.
            10                        # Queue size.
        )

        # -------------------------------------------
        # Initialize flag to track if a message has been received.
        # -------------------------------------------
        self.received_pose = False

        # -------------------------------------------
        # Create a Timer to check if the subscribed topic has received at least one message.
        # This timer will stop checking once a message has been received.
        # -------------------------------------------
        self.subscription_check_timer = self.create_timer(2.0, self.check_initial_subscriptions)

        # -------------------------------------------
        # Create a Service Server.
        # -------------------------------------------
        if self.use_service:
            self.service_server = self.create_service(Trigger, self.service_name, self.service_callback)
            self.get_logger().info(f"Service server started on '{self.service_name}'")

    # -------------------------------------------
    # Timer Callback to Check if the Subscribed Topic Has Received at Least One Message
    # -------------------------------------------
    def check_initial_subscriptions(self):
        if not self.received_pose:
            self.get_logger().info(f"Waiting for messages on topic: '{self.pose_topic}'")
        else:
            self.get_logger().info(
                f"All subscribed topics have received at least one message. "
                f"RotatedPoseNode started with publishers on '{self.rotated_pose_topic}' and '{self.marker_topic}', "
                f"frame_id '{self.frame_id}',"
                f" rotation angles set to roll={self.roll}°, pitch={self.pitch}°, yaw={self.yaw}°"
            )
            self.subscription_check_timer.cancel()

    # -------------------------------------------
    # Pose Rotation Callback
    # -------------------------------------------
    def rotate_pose_callback(self, msg):
        try:
            # Set the flag on receiving the first message.
            if not self.received_pose:
                self.received_pose = True

            # Extract original position and compute the rotated position.
            pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
            rotated_pos = self.rotation_matrix.dot(pos)
            
            # Append the rotated position to the path for the marker.
            new_point    = Point()
            new_point.x  = float(rotated_pos[0])
            new_point.y  = float(rotated_pos[1])
            new_point.z  = float(rotated_pos[2])
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
            rotated_pose_msg                  = PoseStamped()
            rotated_pose_msg.header           = msg.header
            rotated_pose_msg.pose.position.x  = float(rotated_pos[0])
            rotated_pose_msg.pose.position.y  = float(rotated_pos[1])
            rotated_pose_msg.pose.position.z  = float(rotated_pos[2])
            # Convert rotated quaternion back to ROS order (x, y, z, w)
            rotated_pose_msg.pose.orientation.x = rotated_quat[1]
            rotated_pose_msg.pose.orientation.y = rotated_quat[2]
            rotated_pose_msg.pose.orientation.z = rotated_quat[3]
            rotated_pose_msg.pose.orientation.w = rotated_quat[0]

            # Publish the rotated pose.
            self.rotated_pose_publisher.publish(rotated_pose_msg)

            # Create and update a marker to visualize the path.
            marker                  = Marker()
            marker.header           = msg.header
            marker.ns               = "pose_path"
            marker.id               = 0
            marker.type             = Marker.LINE_STRIP
            marker.action           = Marker.ADD
            marker.scale.x          = 0.05  # Line width.
            marker.color.r          = 0.0
            marker.color.g          = 1.0
            marker.color.b          = 0.0
            marker.color.a          = 1.0
            marker.lifetime.sec     = 0
            marker.lifetime.nanosec = 0
            marker.points           = self.path_points

            self.marker_publisher.publish(marker)

            # self.get_logger().info("Published rotated pose and updated path marker.")
        except Exception as e:
            self.get_logger().error(f"Error processing pose: {e}")

    # -------------------------------------------
    # Service Callback Function
    # -------------------------------------------
    def service_callback(self, request, response):
        """
        Clear markers from RViz and reset the path history.
        This will remove the currently published markers and clear the internal path list,
        so that the new path will start fresh.
        """
        self.get_logger().info("Clearing markers and resetting path history...")

        # Publish a DELETE marker to remove the current path marker.
        delete_marker            = Marker()
        delete_marker.header.frame_id = self.frame_id
        delete_marker.header.stamp    = self.get_clock().now().to_msg()
        delete_marker.ns              = "rotate_pose_path"
        delete_marker.id              = 1
        delete_marker.action          = Marker.DELETE
        self.marker_publisher.publish(delete_marker)

        # Reset the internal path history.
        self.path_points = []

        response = Trigger.Response()
        response.success = True
        response.message = "Markers cleared and path history reset."
        return response

# -----------------------------------
# Main Entry Point
# -----------------------------------
def main(args=None):
    rclpy.init(args=args)
    config = load_config()
    node   = RotatedPoseNode(config)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass  # Allow graceful shutdown on CTRL+C.
    finally:
        node.destroy_node()
        rclpy.shutdown()

# -----------------------------------
# Run the node when the script is executed directly.
# -----------------------------------
if __name__ == '__main__':
    main()