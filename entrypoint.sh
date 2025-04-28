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
  rm -f /tmp/segmentation_log
  exit 0
}

trap cleanup SIGINT SIGTERM

# Source ROS 2 environment
source /opt/ros/humble/setup.bash

# # TEMP LOG FILE TO MONITOR SEGMENTATION STDOUT
# SEG_LOG=/tmp/segmentation_log

# # Start segmentation node, log to file and stdout
# echo "Starting segmentation_processor.py..."
# python3 src/segmentation/optimized/segmentation_processor.py 2>&1 | tee "$SEG_LOG" &
# SEG_PID=$!
# PIDS+=($SEG_PID)

# # Wait for the log line
# echo "Waiting for segmentation node to be ready..."
# while ! grep -q "Waiting for messages on topics: '/davis/left/image_raw'" "$SEG_LOG"; do
#   sleep 0.5
# done

# echo "Segmentation is ready. Starting the other ROS2 nodes..."

# # # Start other nodes now that segmentation is waiting for input
# python3 src/point_cloud/point_cloud_processor.py &
# PIDS+=($!)
# # python3 src/point_cloud/pose_following_point_cloud.py &
# # PIDS+=($!)

# python3 src/rotated_pose_message/rotated_pose_processor.py &
# PIDS+=($!)

# python3 src/mapping/semantic_mapping.py &
# PIDS+=($!)

# python3 src/combined/combined.py &

# Continue with bag playback
cd /workspace/bags

ros2 bag play data_1 --clock &
PIDS+=($!)

sleep 5.1

ros2 bag play gt_1 --clock &
PIDS+=($!)

cd ..

# Wait for all background processes
wait
