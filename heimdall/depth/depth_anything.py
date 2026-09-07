"""
Stage 2 — Relative depth extraction using Depth Anything V2 (frozen backbone).

Loads via Hugging Face transformers (AutoModelForDepthEstimation) which provides
ViT-B/ViT-L checkpoints. The backbone is kept FROZEN — we only run inference here.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from heimdall.device import get_device

logger = logging.getLogger(__name__)

# HF model identifiers for Depth Anything V2
MODEL_REGISTRY: dict[str, str] = {
    "vit-s": "depth-anything/Depth-Anything-V2-Small-hf",
    "vit-b": "depth-anything/Depth-Anything-V2-Base-hf",
    "vit-l": "depth-anything/Depth-Anything-V2-Large-hf",
}


@lru_cache(maxsize=1)
def _load_model(model_key: str, device_str: str):
    """Load and cache the Depth Anything V2 model + processor.

    Uses lru_cache so repeated calls with the same key don't re-download.
    """
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    model_id = MODEL_REGISTRY[model_key]
    logger.info("Loading Depth Anything V2 model: %s → %s", model_key, model_id)

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    device = torch.device(device_str)
    model = model.to(device).eval()

    # Freeze everything — we never update backbone weights in this project.
    for p in model.parameters():
        p.requires_grad_(False)

    logger.info("Model loaded on %s  (%.1f M params, all frozen)",
                device, sum(p.numel() for p in model.parameters()) / 1e6)
    return processor, model


def predict_depth(
    image: np.ndarray,
    model_key: str = "vit-b",
    device: torch.device | None = None,
) -> np.ndarray:
    """Run Depth Anything V2 on a single RGB image tile.

    Args:
        image: H×W×3 uint8 numpy array (RGB).
        model_key: One of 'vit-s', 'vit-b', 'vit-l'.
        device: Override device. Auto-detected if None.

    Returns:
        depth_map: H×W float32 numpy array of relative (inverse) depth values.
                   Higher = closer to camera. NOT metric — needs calibration (Stage 5).
    """
    if device is None:
        device = get_device()

    processor, model = _load_model(model_key, str(device))

    # Convert numpy → PIL for the HF processor
    pil_img = Image.fromarray(image)
    h, w = image.shape[:2]

    # Prepare inputs
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference — no grad, no AMP on MPS (unsupported)
    with torch.no_grad():
        outputs = model(**inputs)

    # The model returns predicted_depth of shape (1, H', W')
    # where H', W' are the model's internal resolution.
    depth = outputs.predicted_depth  # (1, H', W')

    # Interpolate back to original input resolution
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),          # (1, 1, H', W')
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).squeeze()                      # (H, W)

    depth_np = depth.cpu().numpy().astype(np.float32)

    logger.debug("Depth map stats: min=%.3f  max=%.3f  mean=%.3f",
                 depth_np.min(), depth_np.max(), depth_np.mean())
    return depth_np


def predict_depth_tiled(
    image: np.ndarray,
    model_key: str = "vit-b",
    tile_size: int = 512,
    overlap: int = 64,
    device: torch.device | None = None,
    weights_path: str | None = None,
) -> np.ndarray:
    """Run depth estimation on a large image by tiling.

    Tiles the image, runs per-tile inference, and stitches results.
    This is the main entry point for production use on large inputs.
    """
    from heimdall.ingestion.tiling import tile_image, stitch_tiles

    h, w = image.shape[:2]

    # Small images: skip tiling overhead
    if h <= tile_size and w <= tile_size:
        logger.info("Image fits in a single tile — running direct inference.")
        return predict_depth(image, model_key=model_key, device=device)

    logger.info("Image %dx%d exceeds tile size %d — tiling.", h, w, tile_size)
    tiles = tile_image(image, tile_size=tile_size, overlap=overlap)

    depth_tiles = []
    for i, (tile_arr, meta) in enumerate(tiles):
        logger.info("Processing tile %d/%d", i + 1, len(tiles))
        d = predict_depth(tile_arr, model_key=model_key, device=device)
        depth_tiles.append((d, meta))

    return stitch_tiles(depth_tiles, original_shape=(h, w),
                        tile_size=tile_size, overlap=overlap)
