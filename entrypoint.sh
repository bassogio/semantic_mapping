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

# # Start additional Python nodes (they continue running)
python3 src/point_cloud/point_cloud_processor.py &
PIDS+=($!)
python3 src/rotated_pose_message/rotated_pose_processor.py &
PIDS+=($!)
# python3 src/occupancy_grid/occupancy_grid_processor.py &
# PIDS+=($!)
# python3 src/segmentation/segmentation_processor.py &
# PIDS+=($!)

# Prompt the user to press Enter or Space to continue
echo "Press 'Enter' or 'Space' to continue..."
while true; do
  # -n1: read one character
  # -rs: silent mode (won't echo the character)
  read -n1 -rs key
  # Check if the pressed key is Space (" ") or Enter (newline, interpreted as an empty string)
  if [ -z "$key" ] || [ "$key" = " " ]; then
    break
  fi
done

# Continue with your commands:
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
