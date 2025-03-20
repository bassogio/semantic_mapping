#camera_config_loader
import os
import yaml

def is_docker():
    """Check if the code is running inside a Docker container."""
    try:
        with open('/proc/1/cgroup', 'r') as f:
            return 'docker' in f.read()
    except FileNotFoundError:
        return False

def load_config():
    # Get the absolute path to the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check if running inside a Docker container
    if is_docker():
        # If inside the container, use the local config path
        config_path = os.path.join(script_dir, '../../config/camera_config.yaml')
    else:
        # If running on the host, use the parent directory path
        config_path = os.path.join(script_dir, '../../config/camera_config.yaml')

    # Open and read the config YAML file
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
