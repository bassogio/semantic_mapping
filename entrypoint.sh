#!/bin/bash

PIDS=()
cleanup() {
  echo "Cleaning up..."
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null; done
  rm -f /tmp/segmentation.log
  exit 0
}
trap cleanup SIGINT SIGTERM

source /opt/ros/humble/setup.bash

# Launch node and tee its logs
LOG=/tmp/segmentation.log
python3 src/segmentation/segmentation_processor.py 2>&1 | tee "$LOG" &
PIDS+=($!)

# Block until we see "Waiting for messages" in the log
echo "Waiting for segmentation node to report it's waiting for messages..."
until grep -q "Waiting for messages" "$LOG"; do
  sleep 0.1
done

# Launch node and tee its logs
LOG=/tmp/rotation.log
python3 src/rotated_pose_message/rotated_pose_processor.py 2>&1 | tee "$LOG" &
PIDS+=($!)

# Block until we see "Waiting for messages" in the log
echo "Waiting for pose rotation node to report it's waiting for messages..."
until grep -q "Waiting for messages" "$LOG"; do
  sleep 0.1
done

# Launch node and tee its logs
LOG=/tmp/pointcloud.log
python3 src/point_cloud/point_cloud_processor.py 2>&1 | tee "$LOG" &
PIDS+=($!)

# Block until we see "Waiting for messages" in the log
echo "Waiting for pointcloud node to report it's waiting for messages..."
until grep -q "Waiting for messages" "$LOG"; do
  sleep 0.1
done

# Continue with bag playback
cd /workspace/bags

# Play data_1 but jump 5.105 seconds in
ros2 bag play data_1 --start-offset 5.105 &
PIDS+=($!)

# Play gt_1 from its very start
ros2 bag play gt_1 &
PIDS+=($!)

# # Play data_2 but jump 43.474 seconds in
# ros2 bag play data_2 --start-offset 43.474 &
# PIDS+=($!)

# # Play gt_2 from its very start
# ros2 bag play gt_2 &
# PIDS+=($!)

cd ..

# Wait for all background processes
wait





# # TEMP LOG FILE TO MONITOR SEGMENTATION STDOUT
# SEG_LOG=/tmp/segmentation_log

# # Start segmentation node, log to file and stdout
# echo "Starting  segmentation_processor.py..."
# python3 src/segmentation/optimized/segmentation_processor.py 2>&1 | tee "$SEG_LOG" &
# SEG_PID=$!
# PIDS+=($SEG_PID)

# # Wait for the log line
# echo "Waiting for segmentation node to be ready..."
# while grep -q "Waiting for messages on topics: '/davis/left/image_raw'" "$SEG_LOG"; do
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