"""
Stage 2 — Relative depth extraction using Depth Anything V2 (frozen backbone).

Loads via Hugging Face transformers (AutoModelForDepthEstimation) which provides
ViT-B/ViT-L checkpoints. The backbone is kept FROZEN — we only run inference here.
"""

from __future__ import annotations

import sys
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from heimdall.device import get_device

logger = logging.getLogger(__name__)

# HF model identifiers for Depth Anything V2 & V3
MODEL_REGISTRY: dict[str, str] = {
    "vit-s": "depth-anything/Depth-Anything-V2-Small-hf",
    "vit-b": "depth-anything/Depth-Anything-V2-Base-hf",
    "vit-l": "depth-anything/Depth-Anything-V2-Large-hf",
    "da3-metric-l": "depth-anything/DA3METRIC-LARGE",
    "da3-mono-l": "depth-anything/DA3MONO-LARGE",
}


@lru_cache(maxsize=1)
def _load_model(model_key: str, device_str: str, use_fp16: bool = False):
    """Load and cache the Depth Anything V2 or V3 model + processor."""
    import sys
    from pathlib import Path
    
    # Ensure local heimdall path is in sys.path for DA3 absolute imports
    repo_root = Path(__file__).parent.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    model_id = MODEL_REGISTRY[model_key]
    device = torch.device(device_str)

    if model_key.startswith("da3"):
        logger.info("Loading Depth Anything V3 model natively: %s → %s", model_key, model_id)
        # Import the local package
        from depth_anything_3.api import DepthAnything3
        
        # DA3 doesn't use AutoImageProcessor
        model = DepthAnything3.from_pretrained(model_id)
        processor = None
    else:
        logger.info("Loading Depth Anything V2 model: %s → %s", model_key, model_id)
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)

    model = model.to(device).eval()

    # Apply FP16 for CUDA devices to halve VRAM and double throughput
    if use_fp16 and device.type == "cuda":
        model = model.half()
        logger.info("FP16 half-precision enabled for CUDA inference.")

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

    # Convert numpy → PIL for the HF processor (if DA2)
    h, w = image.shape[:2]

    if processor is None:
        # Depth Anything V3 native inference
        import cv2
        # DA3 inference takes BGR image path or RGB numpy array, but let's pass a list of numpy images
        # The inference function accepts a list of inputs. Let's pass a list with one element.
        # Wait, the example says: prediction = model.inference(images)
        with torch.no_grad():
            prediction = model.inference([image])
        
        # prediction.depth is [N, H, W]
        depth = torch.from_numpy(prediction.depth[0]).to(device) # (H, W)
        
        # It's already original resolution, but we can ensure dimensions match
        if depth.shape != (h, w):
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=(h, w),
                mode="bilinear",
                align_corners=False
            ).squeeze()
            
        return depth.cpu().numpy()

    # Depth Anything V2 HF processor
    pil_img = Image.fromarray(image)
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
