"""
Multi-scale height decoder.

The original head (``heimdall/decoder/head.py``) sees only the backbone's final *scalar*
depth map — one number per pixel — and rescales it. Everything the transformer knows about
roof planes, façades, shadows and object extent has already been collapsed into that number.

Depth Anything's DPT neck exposes the pyramid *before* that collapse: four 128-channel
feature maps at strides ~1/14, 1/7, 1/3.5 and 1/1.75 of the input. This decoder consumes
that pyramid, fuses it coarse-to-fine, and predicts the same affine field as before:

    nDSM(x, y) = ReLU( S(x, y) · depth(x, y) + T(x, y) )

Keeping the affine parameterisation preserves the inductive bias that the backbone's
relative depth already carries the scene's structure; the pyramid only has to supply the
spatially varying scale and offset — which is what the ground truth actually disagrees with.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, max(1, c // 8)), num_channels=c)


class _FusionBlock(nn.Module):
    """Upsample the coarser stream, add the skip connection, refine (DPT-style residual)."""

    def __init__(self, channels: int):
        super().__init__()
        self.skip = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), _gn(channels))
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), _gn(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False), _gn(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, coarse: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(coarse, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = up + self.skip(skip)
        return self.act(x + self.refine(x))


class MultiScaleHeightHead(nn.Module):
    """DPT-pyramid → per-pixel scale and shift → metric height above ground.

    Args:
        feature_channels: channels of each neck stage (128 for Depth Anything V2/V3 DPT necks).
        hidden_dim: working width of the decoder.
        context_dilations: dilated convolutions giving the decoder a wide receptive field,
            so it can tell a tall building from a bright roof on the ground.
    """

    def __init__(self, feature_channels: int = 128, hidden_dim: int = 128,
                 context_dilations: tuple[int, ...] = (1, 6, 12, 18)):
        super().__init__()
        self.project = nn.ModuleList(
            nn.Sequential(nn.Conv2d(feature_channels, hidden_dim, 1, bias=False), _gn(hidden_dim),
                          nn.ReLU(inplace=True))
            for _ in range(4)
        )
        self.fuse = nn.ModuleList(_FusionBlock(hidden_dim) for _ in range(3))

        self.context = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim // 2, 1 if d == 1 else 3,
                          padding=0 if d == 1 else d, dilation=d, bias=False),
                _gn(hidden_dim // 2), nn.ReLU(inplace=True))
            for d in context_dilations
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_dim, hidden_dim // 2, 1, bias=False),
            _gn(hidden_dim // 2), nn.ReLU(inplace=True))

        fused = (hidden_dim // 2) * (len(context_dilations) + 1)
        self.merge = nn.Sequential(
            nn.Conv2d(fused, hidden_dim, 3, padding=1, bias=False), _gn(hidden_dim), nn.ReLU(inplace=True))

        # Final refinement at full resolution, conditioned on the image and the backbone depth.
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim + 4, hidden_dim // 2, 3, padding=1, bias=False),
            _gn(hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, 3, padding=1, bias=False),
            _gn(hidden_dim // 2), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 2, 1),
        )
        with torch.no_grad():  # start as the identity: height = depth
            self.refine[-1].bias.data[0] = 1.0
            self.refine[-1].bias.data[1] = 0.0

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor, features: list[torch.Tensor]) -> torch.Tensor:
        """rgb (B,3,H,W) in [0,1]; depth (B,1,H,W) raw backbone output; features: 4 neck stages."""
        assert len(features) == 4, f"expected 4 neck stages, got {len(features)}"
        p = [proj(f) for proj, f in zip(self.project, features)]
        x = p[0]                                  # coarsest
        for block, skip in zip(self.fuse, p[1:]):
            x = block(x, skip)                    # → finest stage (~H/2)

        ctx = [c(x) for c in self.context]
        pooled = self.global_pool(x)
        ctx.append(F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False))
        x = self.merge(torch.cat(ctx, dim=1))

        x = F.interpolate(x, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        if depth.shape[-2:] != rgb.shape[-2:]:
            depth = F.interpolate(depth, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
        out = self.refine(torch.cat([x, rgb, depth], dim=1))
        scale, shift = out[:, 0:1], out[:, 1:2]
        return F.relu(scale * depth + shift)
