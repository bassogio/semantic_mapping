#point_cloud_main
import rclpy
from point_cloud_processor import PointCloudProcessor
from config_loader import load_config

def main():
    rclpy.init()

    # Load config
    config = load_config()

    # Initialize PointCloudProcessor with config data
    node = PointCloudProcessor(config=config)  # Pass the config as an argument

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
