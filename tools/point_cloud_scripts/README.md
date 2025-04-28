# Point Cloud Scripts

## point_cloud_processor.py
- Publish a point cloud message. Rotating it is possible by adjusting the Rotation matrix R.

## rgb_point_cloud.py
- Publish a RGB point cloud message. RGB is taken from the semantic image but can also be replaced with the original color image. Rotating it is possible by adjusting the Rotation matrix R.

## pose_following_point_cloud.py
- Publish a point cloud message that follows a pose location and oriantation. Rotating it is possible by adjusting the Raw, Yaw and pitch in the config file.

## rgb_point_cloud_processor.py
- Publish a RGB point cloud message that follows a pose location and oriantation. Rotating it is possible by adjusting the Raw, Yaw and pitch in the config file.

## faster_pc.py
- Publish a point cloud message faster.