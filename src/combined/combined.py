#!/usr/bin/env python3
# combined_processor.py

# -----------------------------------
# Import Statements
# -----------------------------------
import os
import yaml
import numpy as np
import struct      # For packing RGB values into a single 32-bit field
import cv2         # For image resizing
import torch
from PIL import Image as PILImage
from cv_bridge import CvBridge, CvBridgeError
from transforms3d.euler import euler2mat
from transforms3d.quaternions import quat2mat, mat2quat

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
from std_srvs.srv import Trigger
import sensor_msgs_py.point_cloud2 as pc2

from transformers import (
    SegformerForSemanticSegmentation,
    SegformerFeatureExtractor,
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
)

# -----------------------------------
# Config Loaders
# -----------------------------------
def load_rotate_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(script_dir, '../../config/rotate_pose_config.yaml')
    with open(cfg_file, 'r') as f:
        return yaml.safe_load(f)

def load_segmentation_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(script_dir, '../../config/segmentation_config.yaml')
    with open(cfg_file, 'r') as f:
        return yaml.safe_load(f)

def load_point_cloud_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_file = os.path.join(script_dir, '../../config/point_cloud_config.yaml')
    with open(cfg_file, 'r') as f:
        return yaml.safe_load(f)

# -----------------------------------
# RotatedPoseNode
# -----------------------------------
class RotatedPoseNode(Node):
    def __init__(self, config):
        super().__init__('rotated_pose_node')
        cfg = config['rotate_pose_processing']
        self.pose_topic = cfg['pose_topic']
        self.rotated_pose_topic = cfg['rotated_pose_topic']
        self.marker_topic = cfg['marker_topic']
        self.roll, self.pitch, self.yaw = cfg['roll'], cfg['pitch'], cfg['yaw']
        self.frame_id = cfg['frame_id']
        self.use_service = cfg['use_service']
        self.service_name = cfg['service_name']

        # Declare & get parameters
        for name, val in [
            ('pose_topic', self.pose_topic),
            ('rotated_pose_topic', self.rotated_pose_topic),
            ('marker_topic', self.marker_topic),
            ('roll', self.roll), ('pitch', self.pitch), ('yaw', self.yaw),
            ('frame_id', self.frame_id),
            ('use_service', self.use_service),
            ('service_name', self.service_name),
        ]:
            self.declare_parameter(name, val)
            setattr(self, name, self.get_parameter(name).value)

        # Fixed rotation matrix
        self.rotation_matrix = euler2mat(
            np.deg2rad(self.roll),
            np.deg2rad(self.pitch),
            np.deg2rad(self.yaw),
            axes='sxyz'
        )
        self.path_points = []

        # Publishers & Subscriber
        self.rotated_pose_publisher = self.create_publisher(PoseStamped, self.rotated_pose_topic, 10)
        self.marker_publisher = self.create_publisher(Marker, self.marker_topic, 10)
        self.create_subscription(PoseStamped, self.pose_topic, self.rotate_pose_callback, 10)

        # Optional clear service
        if self.use_service:
            self.create_service(Trigger, self.service_name, self.service_callback)

        self.received_pose = False
        # Timer handle to cancel later
        self.subscription_check_timer = self.create_timer(2.0, self.check_initial_subscriptions)

    def check_initial_subscriptions(self):
        if not self.received_pose:
            self.get_logger().info(f"Waiting for '{self.pose_topic}' messages...")
        else:
            self.get_logger().info("RotatedPoseNode ready")
            self.subscription_check_timer.cancel()

    def rotate_pose_callback(self, msg):
        try:
            if not self.received_pose:
                self.received_pose = True

            # Rotate position
            p_arr = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            rp = self.rotation_matrix.dot(p_arr)
            pt = Point(x=float(rp[0]), y=float(rp[1]), z=float(rp[2]))
            self.path_points.append(pt)

            # Rotate orientation
            iq = np.array([msg.pose.orientation.w,
                           msg.pose.orientation.x,
                           msg.pose.orientation.y,
                           msg.pose.orientation.z])
            orig_m = quat2mat(iq)
            rot_m = self.rotation_matrix.dot(orig_m)
            rq = mat2quat(rot_m)  # [w,x,y,z]

            # Publish rotated pose
            out = PoseStamped()
            out.header = msg.header
            out.pose.position = pt
            out.pose.orientation.x, out.pose.orientation.y, out.pose.orientation.z, out.pose.orientation.w = (
                rq[1], rq[2], rq[3], rq[0]
            )
            self.rotated_pose_publisher.publish(out)

            # Publish path marker
            m = Marker()
            m.header = msg.header
            m.ns = "pose_path"
            m.id = 0
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.05
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 1.0
            m.points = self.path_points
            self.marker_publisher.publish(m)

        except Exception as e:
            self.get_logger().error(f"Error in rotate_pose: {e}")

    def service_callback(self, request, response):
        # Clear markers
        dm = Marker()
        dm.header.frame_id = self.frame_id
        dm.header.stamp = self.get_clock().now().to_msg()
        dm.ns = "rotate_pose_path"
        dm.id = 1
        dm.action = Marker.DELETE
        self.marker_publisher.publish(dm)
        self.path_points = []
        response.success = True
        response.message = "Markers cleared"
        return response

# -----------------------------------
# SegmentationNode
# -----------------------------------
class SegmentationNode(Node):
    def __init__(self, config):
        super().__init__('segmentation_node')
        cfg = config['segmentation_processing']
        self.segmentation_topic = cfg['segmentation_topic']
        self.color_image_topic = cfg['color_image_topic']
        self.frame_id = cfg['frame_id']
        self.labels = cfg['labels']
        self.model_input = cfg['model']

        for name, val in [
            ('segmentation_topic', self.segmentation_topic),
            ('color_image_topic', self.color_image_topic),
            ('frame_id', self.frame_id),
            ('model', self.model_input),
        ]:
            self.declare_parameter(name, val)
            setattr(self, name, self.get_parameter(name).value)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Segmentation on {self.device}")
        self.prompts = [lbl['name'] for lbl in self.labels]
        self.bridge = CvBridge()
        self.target_width, self.target_height = 320, 240

        # Initialize chosen model
        if self.model_input == 'clipseg':
            self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=False)
            self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)
            if self.device.type == "cuda": self.model.half()
            self.precomputed_text_inputs = self.processor(
                text=self.prompts,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            for k, v in self.precomputed_text_inputs.items():
                if isinstance(v, torch.Tensor):
                    self.precomputed_text_inputs[k] = v.to(self.device)
        else:
            model_name = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
            if self.device.type == "cuda": self.model.half()

        self.segmentation_publisher = self.create_publisher(Image, self.segmentation_topic, 10)
        self.create_subscription(Image, self.color_image_topic, self.segmentation_callback, 10)

        self.received_color_image = False
        self.subscription_check_timer = self.create_timer(2.0, self.check_initial_subscriptions)

    def check_initial_subscriptions(self):
        if not self.received_color_image:
            self.get_logger().info(f"Waiting for '{self.color_image_topic}'...")
        else:
            self.get_logger().info("SegmentationNode ready")
            self.subscription_check_timer.cancel()

    def segmentation_callback(self, msg):
        if not self.received_color_image:
            self.received_color_image = True

        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return

        pil_img = PILImage.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
        pil_img = pil_img.resize((self.target_width, self.target_height))

        autocast_cm = torch.amp.autocast(device_type="cuda") if self.device.type == "cuda" else nullcontext()
        with torch.no_grad(), autocast_cm:
            if self.model_input == 'clipseg':
                img_inputs = self.processor(images=[pil_img]*len(self.prompts), return_tensors="pt")
                for k, v in img_inputs.items():
                    if isinstance(v, torch.Tensor): img_inputs[k]=v.to(self.device)
                img_inputs.update(self.precomputed_text_inputs)
                outputs = self.model(**img_inputs).logits
                combined = torch.sigmoid(outputs).squeeze(0).argmax(dim=0).cpu().numpy()
            else:
                inputs = self.feature_extractor(images=pil_img, return_tensors="pt")
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor): inputs[k]=v.to(self.device)
                preds = self.model(**inputs).logits
                combined = preds.argmax(dim=1)[0].cpu().numpy()

        mask = np.zeros((*combined.shape, 3), dtype=np.uint8)
        for lbl in self.labels:
            mask[combined == lbl['id']] = lbl['color']

        try:
            out_msg = self.bridge.cv2_to_imgmsg(mask, encoding="rgb8")
            self.segmentation_publisher.publish(out_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Publish error: {e}")

# -----------------------------------
# PointCloudNode
# -----------------------------------
class PointCloudNode(Node):
    def __init__(self, config):
        super().__init__('point_cloud_node')
        cfg = config['point_cloud_processing']
        self.point_cloud_topic       = cfg['point_cloud_topic']
        self.camera_parameters_topic = cfg['camera_parameters_topic']
        self.depth_image_topic       = cfg['depth_image_topic']
        self.pose_topic              = cfg['pose_topic']
        self.semantic_image_topic    = cfg['semantic_image_topic']
        self.max_distance            = cfg['max_distance']
        self.depth_scale             = cfg['depth_scale']
        self.frame_id                = cfg['frame_id']

        for name, val in [
            ('point_cloud_topic', self.point_cloud_topic),
            ('camera_parameters_topic', self.camera_parameters_topic),
            ('depth_image_topic', self.depth_image_topic),
            ('pose_topic', self.pose_topic),
            ('semantic_image_topic', self.semantic_image_topic),
            ('max_distance', self.max_distance),
            ('depth_scale', self.depth_scale),
            ('frame_id', self.frame_id),
        ]:
            self.declare_parameter(name, val)
            setattr(self, name, self.get_parameter(name).value)

        # pose & image placeholders
        self.Qw, self.Qx, self.Qy, self.Qz = 1.0, 0.0, 0.0, 0.0
        self.pose_x, self.pose_y, self.pose_z = 0.0, 0.0, 0.0
        self.depth_image, self.semantic_image = None, None

        self.received_camera, self.received_pose, self.received_depth, self.received_sem = False, False, False, False
        self.bridge = CvBridge()
        self.point_cloud_publisher = self.create_publisher(PointCloud2, self.point_cloud_topic, 10)
        self.create_subscription(CameraInfo, self.camera_parameters_topic, self.camera_callback, 10)
        self.create_subscription(PoseStamped, self.pose_topic, self.pose_callback, 10)
        self.create_subscription(Image, self.depth_image_topic, self.depth_callback, 10)
        self.create_subscription(Image, self.semantic_image_topic, self.semantic_callback, 10)

        self.subscription_check_timer = self.create_timer(2.0, self.check_initial_subscriptions)

    def check_initial_subscriptions(self):
        missing = [
            t for flag, t in [ (self.received_camera, self.camera_parameters_topic),
                                (self.received_pose,   self.pose_topic           ) ]
            if not flag
        ]
        if missing:
            self.get_logger().info("Waiting for: " + ", ".join(missing))
        else:
            self.get_logger().info("PointCloudNode ready")
            self.subscription_check_timer.cancel()

    def camera_callback(self, msg):
        self.received_camera = True
        self.fx, self.fy, self.cx, self.cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]

    def pose_callback(self, msg):
        self.received_pose = True
        o, p = msg.pose.orientation, msg.pose.position
        self.Qx, self.Qy, self.Qz, self.Qw = o.x, o.y, o.z, o.w
        self.pose_x, self.pose_y, self.pose_z = p.x, p.y, p.z

    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.received_depth = True
            self.create_pointcloud()
        except Exception as e:
            self.get_logger().error(f"Depth cb error: {e}")

    def semantic_callback(self, msg):
        try:
            self.semantic_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.received_sem = True
            self.create_pointcloud()
        except Exception as e:
            self.get_logger().error(f"Semantic cb error: {e}")

    def create_pointcloud(self):
        if self.depth_image is None or self.semantic_image is None:
            return
        # Validate camera intrinsics
        if not hasattr(self, 'fx'):
            self.get_logger().warn("Camera parameters not set; skipping point cloud.")
            return
        try:
            depth = self.depth_image * self.depth_scale
            rows, cols = depth.shape
            if self.semantic_image.shape[:2] != (rows, cols):
                sem = cv2.resize(self.semantic_image, (cols, rows))
            else:
                sem = self.semantic_image
            u, v = np.meshgrid(np.arange(cols), np.arange(rows))
            u, v = u.astype(np.float32), v.astype(np.float32)
            z = depth
            x = z * (u - self.cx) / self.fx
            y = z * (v - self.cy) / self.fy
            valid = (z > 0) & (z < self.max_distance)
            x, y, z = x[valid], y[valid], z[valid]
            points = np.stack((x, y, z), axis=-1)
            r, g, b = sem[...,0][valid], sem[...,1][valid], sem[...,2][valid]
            rgb_packed = np.array([
                struct.unpack('f', struct.pack('I', (int(rv)<<16 | int(gv)<<8 | int(bv))))[0]
                for rv,gv,bv in zip(r,g,b)
            ])
            pts_color = np.column_stack((points, rgb_packed))
            quat = [self.Qw, self.Qx, self.Qy, self.Qz]
            R = quat2mat(quat)
            pts_xyz = pts_color[:,:3]
            pts_rot = pts_xyz @ R.T
            trans = np.array([self.pose_x, self.pose_y, self.pose_z])
            pts_tf = pts_rot + trans
            final = np.column_stack((pts_tf, pts_color[:,3]))

            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.frame_id
            fields = []
            for i,name in enumerate(['x','y','z','rgb']):
                pf = PointField()
                pf.name = name
                pf.offset = 4*i
                pf.datatype = PointField.FLOAT32
                pf.count = 1
                fields.append(pf)
            cloud = PointCloud2(
                header=header,
                height=1,
                width=final.shape[0],
                fields=fields,
                is_bigendian=False,
                point_step=16,
                row_step=16*final.shape[0],
                data=final.astype(np.float32).tobytes(),
                is_dense=True
            )
            self.point_cloud_publisher.publish(cloud)
        except Exception as e:
            self.get_logger().error(f"Error creating point cloud: {e}")

# -----------------------------------
# Main
# -----------------------------------
def main():
    rclpy.init()
    rot_cfg = load_rotate_config()
    seg_cfg = load_segmentation_config()
    pc_cfg  = load_point_cloud_config()

    rot_node = RotatedPoseNode(rot_cfg)
    seg_node = SegmentationNode(seg_cfg)
    pc_node  = PointCloudNode(pc_cfg)

    executor = MultiThreadedExecutor()
    for nd in (rot_node, seg_node, pc_node):
        executor.add_node(nd)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for nd in (rot_node, seg_node, pc_node):
            nd.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
