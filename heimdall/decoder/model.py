"""
Stage 3 — Training Wrapper Model

Encapsulates the frozen Depth Anything V2 backbone and the trainable Domain Adaptation Head.
Handles the forward pass and loss computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from PIL import Image
import numpy as np

class DomainAdaptationWrapper(nn.Module):
    """
    Wraps the Hugging Face Depth Anything V2 model (frozen) and our custom
    DomainAdaptationHead (trainable).
    """
    def __init__(self, model_key: str = "vit-b", device_str: str = "cpu"):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        from heimdall.depth.depth_anything import MODEL_REGISTRY
        from heimdall.decoder.head import DomainAdaptationHead

        model_id = MODEL_REGISTRY[model_key]
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        
        # Load backbone and freeze it entirely
        self.backbone = AutoModelForDepthEstimation.from_pretrained(model_id)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # Trainable head
        self.head = DomainAdaptationHead(in_channels=4, hidden_dims=[64, 32, 16])
        
        # Loss function (L1 loss is a good baseline for metric depth)
        self.loss_fn = nn.L1Loss()

    def train(self, mode: bool = True):
        """Override train to ensure backbone stays in eval mode (no dropout/batchnorm updates)."""
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, rgb_tensor: torch.Tensor, target_height: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        rgb_tensor: (B, 3, H, W) float tensor [0, 1]
        target_height: (B, 1, H, W) float tensor
        """
        # Convert the raw tensor to the format expected by the HF processor.
        # The processor expects PIL images or numpy arrays typically, 
        # but since we are in a training loop and need gradients for the head 
        # (though not backbone), we can pass tensors if we normalize manually.
        # Actually, AutoImageProcessor is hard to use purely in-graph with tensors.
        # Let's normalize the tensor manually matching ImageNet stats which DepthAnything uses.
        
        # DepthAnything V2 uses standard ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=rgb_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=rgb_tensor.device).view(1, 3, 1, 1)
        
        pixel_values = (rgb_tensor - mean) / std

        # Forward pass through frozen backbone
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
            relative_depth = outputs.predicted_depth.unsqueeze(1) # (B, 1, H', W')
            
        # The head takes the original RGB [0,1] and the relative depth
        pred_height = self.head(rgb_tensor, relative_depth)
        
        # Interpolate pred to target size if needed
        if target_height is not None and pred_height.shape[2:] != target_height.shape[2:]:
            pred_height = torch.nn.functional.interpolate(
                pred_height, size=target_height.shape[2:], mode="bilinear", align_corners=False
            )
            
        loss = None
        if target_height is not None:
            # Mask out invalid pixels (e.g. nodata values like <= -9999 or 0 if 0 is invalid)
            valid_mask = target_height > -1000
            if valid_mask.sum() > 0:
                loss = self.loss_fn(pred_height[valid_mask], target_height[valid_mask])
            else:
                loss = torch.tensor(0.0, device=rgb_tensor.device, requires_grad=True)
                
        return {"pred_height": pred_height, "loss": loss}
