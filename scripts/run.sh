#!/bin/bash

# Set container and image names
CONTAINER_NAME="semantic_mapping_container"
IMAGE_NAME="semantic_mapping"

echo "Starting container: $CONTAINER_NAME"

# Run the container with necessary options
docker run -it --rm \
    --name "$CONTAINER_NAME" \
    --runtime nvidia \
    --network host \
    --privileged \
    -v /dev:/dev \
    -v /dev/bus/usb:/dev/bus/usb \
    "$IMAGE_NAME" /workspace/entrypoint.sh 

echo "Container $CONTAINER_NAME is running."
