#!/bin/bash
# entrypoint.sh

# Source ROS 2 environment setup
source /opt/ros/galactic/setup.bash

# Change to the point_cloud directory
cd /workspace/point_cloud

# Run the Python script
python3 main.py

# Or if you prefer to use the relative path:
# python3 point_cloud/main.py

# Keep the container running (optional, if you need to keep the container alive after the script finishes)
# tail -f /dev/null
