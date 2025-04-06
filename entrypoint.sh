#!/bin/bash

# Stop all background processes on exit (including Ctrl+C)
cleanup() {
  echo "Cleaning up..."
  kill ${PIDS[@]} 2>/dev/null
  wait
}
trap cleanup

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# Go to data directory
cd /workspace/data

# Start RGB bag playback
ros2 bag play ros2_bag_data_dir --loop --clock &
PIDS+=($!)

# Wait to align playback
# sleep 10.2
# sleep 5.1


# Start Depth bag playback
ros2 bag play ros2_bag_gt_dir --loop --clock &
PIDS+=($!)

# Go back to project root
cd ..

# Start Python nodes
# python3 src/camera/main.py &
# PIDS+=($!)
# python3 src/point_cloud/main.py &
# PIDS+=($!)

# Wait for all background jobs
wait
