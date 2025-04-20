# -----------------------------------
# Import Statements
# -----------------------------------
import os       # For file path operations.
import yaml     # For loading configuration files.
import rclpy    # ROS2 Python client library.
from rclpy.node import Node  # Base class for ROS2 nodes.
from geometry_msgs.msg import PoseStamped  # Message type including header and pose.

# -----------------------------------
# Configuration Loader Function
# -----------------------------------
def load_config():
    """
    Load configuration parameters

    Returns:
        dict: A dictionary containing configuration data.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, '../config/semantic_mapping_config.yaml')

    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        raise FileNotFoundError(f"Config file not found at: {config_file}")

# -----------------------------------
# ROS2 Node Definition
# -----------------------------------
class PoseDataNode(Node):
    def __init__(self, config):
        super().__init__('pose_data_node')

        # -------------------------------------------
        # Access the configuration section.
        # -------------------------------------------
        self.node_config = config['semantic_mapping']
    
        # -------------------------------------------
        # Load configuration parameters.
        # -------------------------------------------
        self.publisher_topic   = "/pose_bounds"
        self.subscriber_topic   = self.node_config['pose_topic']
        self.grid_resolution   = self.node_config['grid_resolution']

        # -------------------------------------------
        # Declare ROS2 parameters for runtime modification.
        # -------------------------------------------
        self.declare_parameter('publisher_topic', self.publisher_topic)
        self.declare_parameter('subscriber_topic', self.subscriber_topic)
        self.declare_parameter('grid_resolution', self.grid_resolution)
        
        # -------------------------------------------
        # Retrieve final parameter values from the parameter server.
        # This allows any runtime overrides (e.g., via launch files) to update these defaults.
        # -------------------------------------------
        self.publisher_topic = self.get_parameter('publisher_topic').value
        self.subscriber_topic = self.get_parameter('subscriber_topic').value
        self.grid_resolution = self.get_parameter('grid_resolution').value
        
        # Initialize min/max bounds
        self.min_x = float('inf')
        self.max_x = float('-inf')
        self.min_y = float('inf')
        self.max_y = float('-inf')

        self.received_data = False

        # Create subscriber
        self.subscription = self.create_subscription(
            PoseStamped,
            self.subscriber_topic,
            self.pose_callback,
            10
        )

        # Track the time of the last received message
        self.last_msg_time = self.get_clock().now()

        # Timer to detect inactivity
        self.inactivity_timer = self.create_timer(5.0, self.check_inactivity)

    def pose_callback(self, msg: PoseStamped):
        x = msg.pose.position.x
        y = msg.pose.position.y

        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)

        self.last_msg_time = self.get_clock().now()
        self.received_data = True

    def check_inactivity(self):
        now = self.get_clock().now()
        duration = now - self.last_msg_time

        if self.received_data and duration.nanoseconds > 5e9:
            self.get_logger().info("No new pose messages received for 5 seconds. Assuming bag finished.")
            self.report_bounds()
            self.inactivity_timer.cancel()

    def report_bounds(self):
        center_x = (self.min_x + self.max_x) / 2
        center_y = (self.min_y + self.max_y) / 2
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        
        self.grid_resolution = 0.1  # same as your config
        padding = 0.0

        # --- Ask user if they want to add padding ---
        try:
            user_input = input("Do you want to add padding? (yes/no): ").strip().lower()
            if user_input in ['yes', 'y']:
                padding_input = input("Enter padding amount in meters: ").strip()
                padding = float(padding_input)
                self.get_logger().info(f"Using padding: {padding} meters")
            elif user_input in ['no', 'n']:
                padding = 0.0
                self.get_logger().info("No padding will be applied.")
            else:
                self.get_logger().warn("Unrecognized input. Proceeding without padding.")
        except Exception as e:
            self.get_logger().warn(f"Invalid input. Proceeding without padding. Error: {e}")

        # --- Apply padding to min/max bounds ---
        padded_min_x = self.min_x - padding
        padded_max_x = self.max_x + padding
        padded_min_y = self.min_y - padding
        padded_max_y = self.max_y + padding

        padded_width = padded_max_x - padded_min_x
        padded_height = padded_max_y - padded_min_y

        # --- Compute grid size and origin centered around the middle ---
        total_width = padded_width
        total_height = padded_height
        grid_width = int(total_width / self.grid_resolution)
        grid_height = int(total_height / self.grid_resolution)

        grid_origin_x = center_x - (total_width / 2)
        grid_origin_y = center_y - (total_height / 2)
        grid_origin = [round(grid_origin_x, 2), round(grid_origin_y, 2)]

        # --- Print final result ---
        self.get_logger().info(
            f"\nPose bounds:\n"
            f"  x: [{self.min_x:.2f}, {self.max_x:.2f}]\n"
            f"  y: [{self.min_y:.2f}, {self.max_y:.2f}]\n"
            f"  center: ({center_x:.2f}, {center_y:.2f})\n"
            f"  size: width = {width:.2f}, height = {height:.2f}\n"
            f"  padding: {padding} m\n\n"
            f"Recommended YAML config snippet:\n"
            f"  grid_resolution: {self.grid_resolution}\n"
            f"  grid_width: {grid_width}  # in cells\n"
            f"  grid_height: {grid_height}  # in cells\n"
            f"  grid_origin: {grid_origin}  # in meters"
        )

# -----------------------------------
# Main Entry Point
# -----------------------------------
def main(args=None):
    rclpy.init(args=args)
    config = load_config()
    node = PoseDataNode(config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# -----------------------------------
# Run the node
# -----------------------------------
if __name__ == '__main__':
    main()
