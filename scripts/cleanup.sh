#!/bin/bash

CONTAINER_NAME="semantic_mapping_container"
IMAGE_NAME="semantic_mapping"

echo "Removing container (if exists): $CONTAINER_NAME"
docker rm -f $CONTAINER_NAME

echo "Removing image (if exists): $IMAGE_NAME"
docker rmi -f $IMAGE_NAME

echo "Cleanup completed!"
