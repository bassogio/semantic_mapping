import cv2 as cv
import numpy as np
from transformers import CLIPSegProcessor as ClipProcessor, CLIPSegForImageSegmentation
import torch

class CLIPsegProcessor:
    def __init__(self, prompts, device):
        self.prompts = prompts
        self.device = device
        self.processor = ClipProcessor.from_pretrained("CIDAS/clipseg-rd64-refined", use_fast=True)
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(self.device)

    def process_image(self, cv_image, label_colors):
        inputs = self.processor(text=self.prompts, images=[cv_image] * len(self.prompts), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        preds = torch.sigmoid(outputs.logits).cpu()  # move back to CPU for numpy conversion
        original_height, original_width = cv_image.shape[:2]
        combined_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)

        for idx, confidence_map in enumerate(preds):
            confidence = confidence_map.numpy().squeeze()
            resized_confidence = cv.resize(confidence, (original_width, original_height), interpolation=cv.INTER_LINEAR)
            color = label_colors.get(idx, (255, 255, 255))
            for c in range(3):
                combined_image[:, :, c] += (resized_confidence * color[c]).astype(np.uint8)

        return combined_image
