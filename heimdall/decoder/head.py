"""
Stage 3 — Domain Adaptation Head

A robust regression head that takes the frozen Depth Anything V2 features
(relative depth map + RGB image) and predicts the adapted metric height.
Now upgraded with ASPP (Atrous Spatial Pyramid Pooling) for multi-scale context.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module for multi-scale context aggregation."""
    def __init__(self, in_channels: int, out_channels: int, dilations: list[int] = [1, 6, 12, 18]):
        super().__init__()
        
        self.branches = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 1 if dilation == 1 else 3
            padding = 0 if dilation == 1 else dilation
            
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
            
        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Bottleneck to fuse all branches
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        branch_outs = [branch(x) for branch in self.branches]
        
        pool_out = self.global_pool(x)
        pool_out = F.interpolate(pool_out, size=size, mode='bilinear', align_corners=False)
        branch_outs.append(pool_out)
        
        out = torch.cat(branch_outs, dim=1)
        return self.bottleneck(out)


class DomainAdaptationHead(nn.Module):
    """Advanced CNN with ASPP to adapt relative depth + RGB to target height.
    
    Concatenates RGB (3ch) + relative depth (1ch), extracts multi-scale features
    using ASPP, and regresses absolute metric height.
    """
    def __init__(self, in_channels: int = 4, hidden_dim: int = 128):
        super().__init__()
        
        # Initial stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Multi-scale context
        self.aspp = ASPP(hidden_dim, hidden_dim)
        
        # Decoder/Projection
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1)
        )
        
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
        
        feat = self.stem(x)
        feat = self.aspp(feat)
        out = self.decoder(feat)
        
        # We add the relative depth back as a skip connection (scaled) to ease learning,
        # so the network only has to learn the residual/scale transform.
        return out + relative_depth
