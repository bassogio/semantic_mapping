#!/usr/bin/env python3
# segmentation_processor.py

import os
import yaml
import cv2
from contextlib import nullcontext
from cv_bridge import CvBridge, CvBridgeError
import torch
import torch.cuda.amp as amp
import numpy as np
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerFeatureExtractor,
    CLIPSegForImageSegmentation,
    CLIPSegProcessor,
)
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image

def load_config():
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    cfg_file    = os.path.join(script_dir, '../../../config/segmentation_config.yaml')
    if not os.path.exists(cfg_file):
        raise FileNotFoundError(f"Config file not found at: {cfg_file}")
    with open(cfg_file, 'r') as f:
        return yaml.safe_load(f)

class SegmentationNode(Node):
    def __init__(self, config):
        super().__init__('segmentation_node')
        cfg = config['segmentation_processing']

        # Topics & frame
        self.segmentation_topic = cfg['segmentation_topic']
        self.color_image_topic  = cfg['color_image_topic']
        self.frame_id           = cfg['frame_id']

        # Model choice & labels
        self.labels      = cfg['labels']
        self.model_input = cfg['model']           # "segformer" or "clipseg"

        # Speed tweaks
        self.target_width  = cfg.get('target_width', 320)
        self.target_height = cfg.get('target_height', 240)
        self.use_fp16      = cfg.get('use_fp16', True)
        self.frame_skip    = cfg.get('frame_skip', 1)

        # (Optional) expose as ROS params
        for p in ['segmentation_topic','color_image_topic','frame_id','model_input']:
            self.declare_parameter(p, getattr(self, p))
            setattr(self, p, self.get_parameter(p).value)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # Build prompts list
        self.prompts = [lbl['name'] for lbl in self.labels]

        # Load & prepare model
        if self.model_input == 'clipseg':
            self.get_logger().info("Initializing CLIPSeg...")
            self.processor = CLIPSegProcessor.from_pretrained(
                "CIDAS/clipseg-rd64-refined", use_fast=False
            )
            self.model = CLIPSegForImageSegmentation.from_pretrained(
                "CIDAS/clipseg-rd64-refined"
            )
            self.model.to(self.device).eval()

            # Precompute text‐only inputs
            self.precomputed_text_inputs = self.processor(
                text=self.prompts,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            for k,v in self.precomputed_text_inputs.items():
                if isinstance(v, torch.Tensor):
                    self.precomputed_text_inputs[k] = v.to(self.device)

            # CLIP mean/std
            self._mean = torch.tensor(
                [0.48145466, 0.4578275, 0.40821073],
                device=self.device
            ).view(1,3,1,1)
            self._std  = torch.tensor(
                [0.26862954, 0.26130258, 0.27577711],
                device=self.device
            ).view(1,3,1,1)

        else:
            self.get_logger().info("Initializing SegFormer...")
            model_name = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            self.model.to(self.device).eval()

            # ImageNet mean/std for SegFormer
            self._mean = torch.tensor(
                [0.485, 0.456, 0.406],
                device=self.device
            ).view(1,3,1,1)
            self._std  = torch.tensor(
                [0.229, 0.224, 0.225],
                device=self.device
            ).view(1,3,1,1)

        # Mixed precision
        if self.device.type=="cuda" and self.use_fp16:
            self.model.half()

        # TorchScript for extra speed
        try:
            self.model = torch.jit.script(self.model)
            self.get_logger().info("Model scripted to TorchScript.")
        except Exception as e:
            self.get_logger().warn(f"TorchScript failed: {e}")

        # Build color LUT
        max_id = max(lbl['id'] for lbl in self.labels)
        self.color_lut = np.zeros((max_id+1,3), dtype=np.uint8)
        for lbl in self.labels:
            self.color_lut[lbl['id']] = lbl['color']

        # CvBridge / pubs & subs
        self.bridge = CvBridge()
        self.segmentation_publisher = self.create_publisher(
            Image, self.segmentation_topic, 10
        )
        self.subscription = self.create_subscription(
            Image, self.color_image_topic, self.segmentation_callback, 10
        )

        self.received_color_image = False
        self._frame_cnt = 0
        self.create_timer(2.0, self.check_initial_subscriptions)

    def check_initial_subscriptions(self):
        if not self.received_color_image:
            self.get_logger().info(f"Waiting for '{self.color_image_topic}'...")
        else:
            self.get_logger().info(
                f"SegmentationNode ready → publishing on '{self.segmentation_topic}'"
            )

    def segmentation_callback(self, msg):
        # Frame skipping
        self._frame_cnt += 1
        if (self._frame_cnt % self.frame_skip) != 0:
            return

        if not self.received_color_image:
            self.received_color_image = True

        # 1) ROS→OpenCV
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return
        h0, w0 = cv_image.shape[:2]

        # 2) Pre‐resize: 
        #   - SegFormer: any WxH
        #   - CLIPSeg: square (min dimension), divisible by patch size (32)
        if self.model_input == 'clipseg':
            size = min(self.target_width, self.target_height)
            # round size to nearest multiple of 32
            size = (size // 32) * 32
            cv_proc = cv2.resize(cv_image, (size, size), interpolation=cv2.INTER_LINEAR)
        else:
            cv_proc = cv2.resize(
                cv_image,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_LINEAR
            )

        # 3) BGR→RGB & to tensor & normalize
        img_rgb = cv2.cvtColor(cv_proc, cv2.COLOR_BGR2RGB)
        tensor  = torch.from_numpy(img_rgb).permute(2,0,1).float().div(255.0)
        tensor  = (tensor.unsqueeze(0).to(self.device) - self._mean) / self._std
        if self.device.type=="cuda" and self.use_fp16:
            tensor = tensor.half()

        # 4) Build inputs
        if self.model_input == 'clipseg':
            n = len(self.prompts)
            pixel_values = tensor.repeat(n, 1, 1, 1)
            inputs = dict(self.precomputed_text_inputs)
            inputs['pixel_values'] = pixel_values
        else:
            inputs = {'pixel_values': tensor}

        # 5) Inference under autocast
        cm = amp.autocast if self.device.type=="cuda" else nullcontext
        with torch.no_grad(), cm():
            outputs = self.model(**inputs)

        # 6) Extract combined map
        if self.model_input == 'clipseg':
            preds = outputs.logits                  # (n,1,H',W')
            probs = torch.sigmoid(preds).squeeze(1) # (n,H',W')
            combined = probs.argmax(dim=0).cpu().numpy()
        else:
            combined = outputs.logits.argmax(dim=1)[0].cpu().numpy()

        # 7) Colorize + upsample to original
        mask_colored = self.color_lut[combined]
        mask_up      = cv2.resize(
            mask_colored, (w0, h0), interpolation=cv2.INTER_NEAREST
        )

        # 8) Publish
        try:
            out_msg = self.bridge.cv2_to_imgmsg(mask_up, encoding="rgb8")
            out_msg.header.frame_id = self.frame_id
            self.segmentation_publisher.publish(out_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge publish error: {e}")

def main(args=None):
    rclpy.init(args=args)
    cfg  = load_config()
    node = SegmentationNode(cfg)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
