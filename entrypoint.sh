#!/bin/bash

# Source ROS 2 environment setup
source /opt/ros/humble/setup.bash

# Change to the  directory
cd /workspace/data

# Run both ros2 bag play commands in the background
ros2 bag play ros2_bag_data_dir --loop & ros2 bag play ros2_bag_gt_dir --loop &

cd ..

# Start both Python scripts in the background
#python3 src/camera/main.py & 
python3 src/point_cloud/main.py &

# Wait for all background processes to finish
wait
