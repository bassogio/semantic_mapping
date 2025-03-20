#camera_main
import rclpy
from camera_processor import CameraProcessor
from config_loader import load_config
import sys

sys.dont_write_bytecode = True # Stop creating __pycache__

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
