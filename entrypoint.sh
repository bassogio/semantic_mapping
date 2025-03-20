#!/bin/bash

# Start both Python scripts in the background
python3 camera/main.py & 
python3 point_cloud/main.py &

# Wait for all background processes to finish
wait
