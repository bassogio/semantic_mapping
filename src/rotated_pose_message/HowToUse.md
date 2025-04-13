
  

# How to Use: Rotate Pose Processor

  

This document explains how adjust rotation parameters at runtime, clear the path manually and visualize the results in RViz.

 ## Initialization
 First, in the rotate_pose_config.yaml, set:

    roll: 0.0  # rotation around x-axis in degrees
    pitch: 0.0  # rotation around y-axis in degrees
    yaw: 0.0  # rotation around z-axis in degrees

## Launch the Node

Run the node with the following command:

  
  

python3 rotated_pose_processor.py

  

This starts the node, which subscribes to an incoming pose topic, applies the rotation defined by roll, pitch, and yaw (in degrees), and publishes the rotated pose along with its trajectory path.

## Setting Rotation Parameters at Runtime

The node uses ROS 2 parameters for the rotation values. These parameters are interpreted as:

  

-  **Roll**: Rotation about the X-axis.

-  **Pitch**: Rotation about the Y-axis.

-  **Yaw**: Rotation about the Z-axis.

  

To update these values while the node is running, open a new terminal and use the following commands:

  

    ros2 param set /rotated_pose_node roll 10.0
    
    ros2 param set /rotated_pose_node pitch 20.0
    
    ros2 param set /rotated_pose_node yaw 30.0

  

**Note**: Every time a parameter is updated, the node logs the new raw values, recomputes the rotation matrix, and automatically clears the current path history.

  

## Clearing the Path Manually

If you wish to manually clear the markers and reset the internal path history, you can use the provided ROS 2 service:

  

ros2 service call /clear_markers std_srvs/srv/Trigger "{}"

  

This call removes the current visual markers from RViz and resets the stored path so that the new path starts fresh.

  

## Visualizing in RViz

  

1.  **Start RViz2**

Launch RViz2 with:

    rviz2

3.  **Configure RViz Displays**

-  **Marker Display:**

Add a Marker display and set its topic to visualization_marker to see the trajectory rendered as a line strip.

-  **Pose Display (Optional):**

Optionally, add a Pose display to monitor the /davis/rotated_pose topic, which shows the current rotated pose.
