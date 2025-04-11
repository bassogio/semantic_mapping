#rotatte_pose_main

# Stop creating __pycache__
import sys
sys.dont_write_bytecode = True  

import rclpy
from rotate_pose_processor import RotatePoseProcessor
from config_loader import load_config

def main():
    rclpy.init()

    # Load config
    config = load_config()

    # Initialize RotatePoseProcessor with config data
    node = RotatePoseProcessor(config=config)  # Pass the config as an argument

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
