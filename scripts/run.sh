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
    --gpus all \
    -v /dev:/dev \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /home/orin_nano1/SemanticMapping-GiordanoBasso/bags:/workspace/bags \
    -v /home/orin_nano1/SemanticMapping-GiordanoBasso/src:/workspace/src \
    -v /home/orin_nano1/SemanticMapping-GiordanoBasso/config:/workspace/config \
    -v /home/orin_nano1/SemanticMapping-GiordanoBasso/entrypoint.sh:/workspace/entrypoint.sh \
    "$IMAGE_NAME"
    # "$IMAGE_NAME" /workspace/entrypoint.sh

 && echo "Container $CONTAINER_NAME is running."
