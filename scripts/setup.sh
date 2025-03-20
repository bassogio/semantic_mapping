#!/bin/bash

echo "Setting up..."

# Set the image name
IMAGE_NAME="semantic_mapping"

# Set the path to the Dockerfile
DOCKERFILE_PATH="../semantic_mapping.Dockerfile"

echo "Updating system packages..."
sudo apt-get update

echo "Building the Docker image: $IMAGE_NAME using $DOCKERFILE_PATH"
docker build -t $IMAGE_NAME -f "$DOCKERFILE_PATH" "$(dirname "$DOCKERFILE_PATH")"

echo "Environment setup completed."
