#!/bin/bash

# Source ROS 2 environment setup
source /opt/ros/humble/setup.bash

# Change to the data directory
cd /workspace/data

# Play the RGB bag immediately
ros2 bag play ros2_bag_data_dir --loop --rate 0.5 &

# Sleep for ~10.2 seconds to match the 5.1s timestamp offset 
sleep 10.2

# Then play the Depth bag
ros2 bag play ros2_bag_gt_dir --loop --rate 0.5 &

cd ..

# Start your Python node(s) in the background
# python3 src/camera/main.py & 
python3 src/point_cloud/main.py &

# Wait for all background processes to finish
wait
