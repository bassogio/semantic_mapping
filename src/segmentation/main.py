import rclpy
from clipseg_publisher import CLIPsegPublisher
import torch

def main(args=None):
    """Main function to initialize and spin the node."""
    rclpy.init(args=args)
    
    # Check if GPU is available and set the device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for computation:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("Using CPU for computation.")
    
    # Pass the device information to the node if necessary
    node = CLIPsegPublisher(device=device)  # Assuming `CLIPsegPublisher` accepts a `device` parameter

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node interrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
