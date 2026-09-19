"""
Stage 3 — Training / inference wrapper.

Encapsulates the frozen Depth Anything V2/V3 backbone and the trainable Domain
Adaptation Head, which regresses height above ground (nDSM, metres) from
RGB + the backbone's raw depth output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DomainAdaptationWrapper(nn.Module):
    """Frozen Depth Anything backbone + trainable ASPP DomainAdaptationHead."""

    def __init__(self, model_key: str = "da3-metric-l", device_str: str = "cpu"):
        super().__init__()
        from heimdall.depth.depth_anything import _load_model
        from heimdall.decoder.head import DomainAdaptationHead
        from heimdall.decoder.loss import MetricDepthLoss

        self.model_key = model_key
        self.is_da3 = model_key.startswith("da3")
        self.processor, self.backbone = _load_model(model_key, device_str)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self.head = DomainAdaptationHead(in_channels=4, hidden_dim=128)
        self.loss_fn = MetricDepthLoss(alpha=1.0, beta=0.5, gamma=0.1)

    def train(self, mode: bool = True):
        """Keep the backbone in eval mode whatever the wrapper's mode."""
        super().train(mode)
        self.backbone.eval()
        return self

    def _extract_relative_depth_da2(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=rgb_tensor.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=rgb_tensor.device).view(1, 3, 1, 1)
        with torch.no_grad():
            outputs = self.backbone(pixel_values=(rgb_tensor - mean) / std)
        return outputs.predicted_depth.unsqueeze(1)

    def _extract_relative_depth_da3(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        """Raw DA3 depth for each image in the batch.

        DA3's inference() treats a list of images as multiple *views of one scene*
        (cross-view attention), so unrelated tiles must be run one at a time —
        otherwise training sees different backbone outputs than single-image inference.
        """
        maps = []
        with torch.no_grad():
            for i in range(rgb_tensor.shape[0]):
                img = rgb_tensor[i].detach().float().cpu().permute(1, 2, 0).numpy()
                img = (img * 255).clip(0, 255).astype(np.uint8)
                maps.append(np.asarray(self.backbone.inference([img]).depth[0], dtype=np.float32))
        return torch.from_numpy(np.stack(maps)).to(rgb_tensor.device).unsqueeze(1)

    def forward(self, rgb_tensor: torch.Tensor, target_height: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """rgb_tensor: (B,3,H,W) in [0,1]; target_height: (B,1,H,W) metres (optional)."""
        if self.is_da3:
            relative_depth = self._extract_relative_depth_da3(rgb_tensor)
        else:
            relative_depth = self._extract_relative_depth_da2(rgb_tensor)

        pred_height = self.head(rgb_tensor, relative_depth)
        if target_height is not None and pred_height.shape[2:] != target_height.shape[2:]:
            pred_height = torch.nn.functional.interpolate(
                pred_height, size=target_height.shape[2:], mode="bilinear", align_corners=False
            )

        loss_dict = {"loss": None, "silog": None, "grad": None, "l1": None}
        if target_height is not None:
            loss_dict = self.loss_fn(pred_height, target_height)
        return {"pred_height": pred_height, **loss_dict}


def load_wrapper(weights: str | Path, device: torch.device, model_key: str | None = None) -> DomainAdaptationWrapper:
    """Build a wrapper and load a head checkpoint written by scripts/train_decoder.py."""
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "head_state_dict" in ckpt:
        state = ckpt["head_state_dict"]
        model_key = model_key or ckpt.get("model_key")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    model_key = model_key or "da3-metric-l"
    state = {k.replace("module.", "").removeprefix("head."): v for k, v in state.items()}

    wrapper = DomainAdaptationWrapper(model_key=model_key, device_str=str(device))
    wrapper.head.load_state_dict(state, strict=True)
    wrapper = wrapper.to(device).eval()
    logger.info("Loaded decoder head %s (backbone %s, epoch %s)", Path(weights).name, model_key,
                ckpt.get("epoch") if isinstance(ckpt, dict) else "?")
    return wrapper


@torch.no_grad()
def predict_ndsm(
    wrapper: DomainAdaptationWrapper,
    image: np.ndarray,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 128,
    tta: bool = False,
    progress=None,
) -> np.ndarray:
    """Tiled nDSM prediction (metres above ground) for an H×W×3 uint8 image.

    Tiles match the training crop size (512²) so the head sees the same context it
    was trained on; overlaps are feather-blended. ``tta`` averages the horizontal
    flip (heights are flip-invariant), costing 2× runtime.
    """
    from heimdall.ingestion.tiling import stitch_tiles, tile_image

    tiles = tile_image(image, tile_size=tile_size, overlap=overlap)
    outs = []
    for i, (tile, meta) in enumerate(tiles):
        x = torch.from_numpy(np.ascontiguousarray(tile)).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        pred = wrapper(x)["pred_height"]
        if tta:
            pred_f = wrapper(torch.flip(x, dims=[3]))["pred_height"]
            pred = 0.5 * (pred + torch.flip(pred_f, dims=[3]))
        if pred.shape[2:] != x.shape[2:]:
            pred = torch.nn.functional.interpolate(pred, size=x.shape[2:], mode="bilinear", align_corners=False)
        outs.append((pred[0, 0].float().cpu().numpy(), meta))
        if progress:
            progress(i + 1, len(tiles))
    return stitch_tiles(outs, image.shape[:2], tile_size=tile_size, overlap=overlap)
