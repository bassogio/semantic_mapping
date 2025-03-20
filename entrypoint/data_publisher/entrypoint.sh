#Data_publisher_entrypoint
#!/bin/bash
# Source ROS 2 environment setup
source /opt/ros/galactic/setup.bash

ros2 daemon stop
ros2 daemon start

# Change to the  directory
cd /workspace/data

# Run both ros2 bag play commands in the background
ros2 bag play ros2_bag_data_dir --loop & ros2 bag play ros2_bag_gt_dir --loop 