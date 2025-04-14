# -----------------------------------
# Import Statements
# -----------------------------------
import os      
import yaml    
import rclpy    
from rclpy.node import Node  

# -----------------------------------
# Configuration Loader Function
# -----------------------------------
def load_config():
    """
    Load configuration parameters.

    Returns:
        dict: A dictionary containing configuration data.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, '../../config/segmentation_config.yaml')
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        raise FileNotFoundError(f"Config file not found at: {config_file}")

# -----------------------------------
# ROS2 Node Definition
# -----------------------------------
class SegmentationNode(Node):
    def __init__(self, config):
        # Initialize the node with a unique name.
        super().__init__('segmentation_node')
        
        # -------------------------------------------
        # Access the configuration section.
        # -------------------------------------------
        self.node_config = config['segmentation_processing']
        
        # -------------------------------------------
        # Load configuration parameters.
        # -------------------------------------------
        self.segmentation_topic  = self.node_config['segmentation_topic']
        
        # -------------------------------------------
        # Declare ROS2 parameters for runtime modification.
        # -------------------------------------------
        self.declare_parameter('segmentation_topic', self.segmentation_topic)

        
        # -------------------------------------------
        # Retrieve final parameter values from the parameter server.
        # -------------------------------------------
        self.segmentation_topic  = self.get_parameter('segmentation_topic').value

        # -------------------------------------------
        # Initialize additional attributes needed for processing.
        # -------------------------------------------
        self.model_input = ''

        while True:
            self.model_input = input("Please choose a model (clipseg/segformer): ").strip().lower()

            if self.model_input == 'clipseg':
                self.get_logger().info("You chose the CLIPSeg model.")
                break
            elif self.model_input == 'segformer':
                self.get_logger().info("You chose the SegFormer model.")
                break
            else:
                self.get_logger().error("Invalid model choice. Please choose 'clipseg' or 'segformer'.")


        self.get_logger().info(
            "SegmentationNode started."
            f"Using '{self.model_input}' model for segmentation."
        )
              
# -----------------------------------
# Main Entry Point
# -----------------------------------
def main(args=None):
    """
    The main function initializes the ROS2 system, loads configuration parameters,
    creates an instance of the GeneralTaskNode, and spins to process messages until shutdown.
    """
    rclpy.init(args=args)
    config = load_config()
    node = SegmentationNode(config)

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