#camera_main

# Stop creating __pycache__
import sys
sys.dont_write_bytecode = True  

import rclpy
from camera_processor import CameraProcessor
from config_loader import load_config

def main():
    rclpy.init()

    # Load config
    config = load_config()

    # Initialize CameraProcessor with config data
    node = CameraProcessor(config=config)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
