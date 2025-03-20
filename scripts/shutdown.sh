#!/bin/bash

CONTAINER_NAME="semantic_mapping_container"

echo "Stopping container: $CONTAINER_NAME"

docker stop $CONTAINER_NAME

echo "Container: $CONTAINER_NAME has been stopped."
