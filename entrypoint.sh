#!/bin/bash

# Start both Python scripts in the background
python3 src/camera/main.py & 
python3 src/point_cloud/main.py &

# Wait for all background processes to finish
wait
