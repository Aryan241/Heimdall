"""
Stage 3 — Domain Adaptation Head

A lightweight regression head that takes the frozen Depth Anything V2 features
(specifically, the relative depth map + RGB image) and predicts the adapted height.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class DomainAdaptationHead(nn.Module):
    """Lightweight CNN to adapt relative depth + RGB to target height.
    
    Instead of building a massive DPT decoder from scratch, we leverage the
    already-excellent relative depth map from the frozen Depth Anything V2,
    concatenate it with the original RGB image (for texture/boundary guidance),
    and pass it through a few convolutional blocks.
    """
    def __init__(self, in_channels: int = 4, hidden_dims: list[int] = [32, 32, 16]):
        super().__init__()
        
        layers = []
        current_in = in_channels
        
        # Build lightweight conv blocks
        for h_dim in hidden_dims:
            layers.extend([
                nn.Conv2d(current_in, h_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.ReLU(inplace=True),
            ])
            current_in = h_dim
            
        # Final projection to 1-channel depth
        layers.append(nn.Conv2d(current_in, 1, kernel_size=3, padding=1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, rgb: torch.Tensor, relative_depth: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) normalized image tensors.
            relative_depth: (B, 1, H, W) relative depth map from frozen backbone.
            
        Returns:
            adapted_height: (B, 1, H, W) absolute/adapted height map.
        """
        # Ensure spatial dimensions match
        if relative_depth.shape[2:] != rgb.shape[2:]:
            relative_depth = F.interpolate(
                relative_depth, size=rgb.shape[2:], mode="bilinear", align_corners=False
            )
            
        x = torch.cat([rgb, relative_depth], dim=1) # (B, 4, H, W)
        out = self.net(x)                           # (B, 1, H, W)
        
        # We add the relative depth back as a skip connection (scaled) to ease learning,
        # so the network only has to learn the residual/scale transform.
        return out + relative_depth
