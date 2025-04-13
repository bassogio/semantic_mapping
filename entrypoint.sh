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
  # Optionally, if child processes spawn subprocesses, you can kill the entire process group:
  # kill -- -$$
  exit 0
}

# Trap SIGINT (Ctrl+C) and SIGTERM to trigger cleanup
trap cleanup SIGINT SIGTERM

# Source the ROS 2 environment
source /opt/ros/humble/setup.bash

# Change to the data directory
cd /workspace/data

# Start RGB bag playback in the background
ros2 bag play ros2_bag_data_dir --clock &
# Record the process ID
PIDS+=($!)

# Wait a bit to align playback timing
sleep 5.1

# Start Depth bag playback in the background
ros2 bag play ros2_bag_gt_dir --clock &
# Record the process ID
PIDS+=($!)

# Optionally, return to the project root if necessary
cd ..

# Start any additional Python nodes if needed (currently commented out)
# python3 src/camera/main.py &
# PIDS+=($!)
# python3 src/point_cloud/main.py &
# PIDS+=($!)

# Wait for all background processes to finish
wait
