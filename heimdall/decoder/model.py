"""
Stage 3 — Training Wrapper Model

Encapsulates the frozen Depth Anything V2/V3 backbone and the trainable Domain Adaptation Head.
Handles the forward pass and loss computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class DomainAdaptationWrapper(nn.Module):
    """
    Wraps the Depth Anything backbone (frozen) and our custom
    DomainAdaptationHead (trainable).
    
    Supports both DA2 (via HF transformers) and DA3 (via native API).
    """
    def __init__(self, model_key: str = "vit-b", device_str: str = "cpu"):
        super().__init__()
        from heimdall.depth.depth_anything import MODEL_REGISTRY, _load_model
        from heimdall.decoder.head import DomainAdaptationHead

        self.model_key = model_key
        self.is_da3 = model_key.startswith("da3")

        # Load backbone via our unified loader (returns processor, model)
        self.processor, self.backbone = _load_model(model_key, device_str)

        # Freeze backbone entirely
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        
        # Trainable ASPP head
        self.head = DomainAdaptationHead(in_channels=4, hidden_dim=128)
        
        # SOTA Combined Loss Function
        from heimdall.decoder.loss import MetricDepthLoss
        self.loss_fn = MetricDepthLoss(alpha=1.0, beta=0.5, gamma=0.1)

    def train(self, mode: bool = True):
        """Override train to ensure backbone stays in eval mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def _extract_relative_depth_da2(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Extract relative depth from DA2 backbone using ImageNet normalization."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=rgb_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=rgb_tensor.device).view(1, 3, 1, 1)
        pixel_values = (rgb_tensor - mean) / std

        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
            relative_depth = outputs.predicted_depth.unsqueeze(1)  # (B, 1, H', W')
        return relative_depth

    def _extract_relative_depth_da3(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Extract depth from DA3 backbone using its native inference API.
        
        DA3's inference() expects a list of numpy images (H, W, 3) uint8 or file paths.
        We convert the batch tensor back to numpy, run inference, and return as tensor.
        """
        B = rgb_tensor.shape[0]
        device = rgb_tensor.device
        
        # Convert (B, 3, H, W) float [0,1] tensor -> list of (H, W, 3) uint8 numpy
        images_np = []
        for i in range(B):
            img = rgb_tensor[i].detach().cpu().permute(1, 2, 0).numpy()  # (H, W, 3) float [0,1]
            img = (img * 255).clip(0, 255).astype(np.uint8)
            images_np.append(img)
        
        with torch.no_grad():
            prediction = self.backbone.inference(images_np)
            # prediction.depth is [N, H, W] float32 numpy
            depth_np = prediction.depth  # (B, H, W)
        
        depth_tensor = torch.from_numpy(depth_np).to(device).unsqueeze(1)  # (B, 1, H, W)
        return depth_tensor

    def forward(self, rgb_tensor: torch.Tensor, target_height: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """
        rgb_tensor: (B, 3, H, W) float tensor [0, 1]
        target_height: (B, 1, H, W) float tensor
        """
        # Extract relative depth from the frozen backbone
        if self.is_da3:
            relative_depth = self._extract_relative_depth_da3(rgb_tensor)
        else:
            relative_depth = self._extract_relative_depth_da2(rgb_tensor)
        
        # The head takes the original RGB [0,1] and the relative depth
        pred_height = self.head(rgb_tensor, relative_depth)
        
        # Interpolate pred to target size if needed
        if target_height is not None and pred_height.shape[2:] != target_height.shape[2:]:
            pred_height = torch.nn.functional.interpolate(
                pred_height, size=target_height.shape[2:], mode="bilinear", align_corners=False
            )
            
        loss_dict = {"loss": None, "silog": None, "grad": None, "l1": None}
        if target_height is not None:
            loss_dict = self.loss_fn(pred_height, target_height)
                
        return {"pred_height": pred_height, **loss_dict}
