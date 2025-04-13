#!/bin/bash

# Array to hold process IDs
PIDS=()

# Cleanup function to kill all background processes
cleanup() {
  echo "Cleaning up..."
  for pid in "${PIDS[@]}"; do
    echo "Killing process with PID: $pid"
    kill "$pid" 2>/dev/null
  done
  exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM to trigger cleanup
trap cleanup SIGINT SIGTERM

# Source the ROS 2 environment
source /opt/ros/humble/setup.bash

# Start additional Python nodes (they continue running)
# python3 src/point_cloud/point_cloud_processor.py &
# PIDS+=($!)

# python3 src/segmentation/segmentation_processor.py &
# PIDS+=($!)

# --- Wait for the segmentation node to be ready ---
# Instead of waiting for the process to exit (which never happens normally),
# we poll for its readiness by checking if the segmentation topic is available.
# echo "Waiting for the segmentation node to publish /camera/segmentation..."
# timeout=60  # seconds to wait before timing out
# while ! ros2 topic list | grep -q "/camera/segmentation"; do
#   sleep 1
#   ((timeout--))
#   if [ $timeout -le 0 ]; then
#     echo "Timeout waiting for segmentation topic /camera/segmentation. Exiting."
#     cleanup
#   fi
# done
# echo "Segmentation node is ready. Proceeding with bag playback."

# Change to the data directory
cd /workspace/data

# Start RGB bag playback in the background
ros2 bag play ros2_bag_data_dir --clock &
PIDS+=($!)

# Wait a bit to align playback timing
sleep 5.1

# Start Depth bag playback in the background
ros2 bag play ros2_bag_gt_dir --clock &
PIDS+=($!)

# Optionally, return to the project root if necessary
cd ..

# Wait for all background processes to finish
wait
