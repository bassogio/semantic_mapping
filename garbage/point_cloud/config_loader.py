# camera_config_loader
import os
import yaml

def load_config():
    # Get the absolute path to the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the config file path
    config_path = os.path.join(script_dir, '../../config/point_cloud_config.yaml')

    # Open and read the config YAML file
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
