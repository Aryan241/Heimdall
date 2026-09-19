"""
Stage 2 — Monocular depth extraction with a frozen Depth Anything backbone.

Output polarity differs between model families and matters downstream:

* Depth Anything V2 (HF transformers) returns *relative disparity*: higher = closer.
* Depth Anything V3 (native API) returns *depth*: higher = farther from the camera.

For a near-nadir satellite view "closer to the sensor" means "taller", so
``to_closeness`` flips DA3 output before it is used as a relative height map.
The trained decoder head consumes the *raw* backbone output (that is what it was
trained on), so ``predict_depth`` keeps the native polarity.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from heimdall.device import get_device

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, str] = {
    "vit-s": "depth-anything/Depth-Anything-V2-Small-hf",
    "vit-b": "depth-anything/Depth-Anything-V2-Base-hf",
    "vit-l": "depth-anything/Depth-Anything-V2-Large-hf",
    "da3-metric-l": "depth-anything/DA3METRIC-LARGE",
    "da3-mono-l": "depth-anything/DA3MONO-LARGE",
}


def is_depth_polarity(model_key: str) -> bool:
    """True if the model outputs depth (higher = farther) rather than disparity."""
    return model_key.startswith("da3")


@lru_cache(maxsize=2)
def _load_model(model_key: str, device_str: str, use_fp16: bool = False):
    """Load and cache the Depth Anything V2 or V3 model (+ processor for V2)."""
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    model_id = MODEL_REGISTRY[model_key]
    device = torch.device(device_str)

    if model_key.startswith("da3"):
        logger.info("Loading Depth Anything V3: %s → %s", model_key, model_id)
        from depth_anything_3.api import DepthAnything3
        model = DepthAnything3.from_pretrained(model_id)
        processor = None
    else:
        logger.info("Loading Depth Anything V2: %s → %s", model_key, model_id)
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)

    model = model.to(device).eval()
    if use_fp16 and device.type == "cuda":
        model = model.half()
        logger.info("FP16 half-precision enabled for CUDA inference.")

    for p in model.parameters():
        p.requires_grad_(False)

    logger.info("Model loaded on %s (%.1f M params, frozen)",
                device, sum(p.numel() for p in model.parameters()) / 1e6)
    return processor, model


def predict_depth(
    image: np.ndarray,
    model_key: str = "da3-metric-l",
    device: torch.device | None = None,
) -> np.ndarray:
    """Run the backbone on one RGB image (H×W×3 uint8) and return its raw output at H×W."""
    if device is None:
        device = get_device()
    processor, model = _load_model(model_key, str(device))
    h, w = image.shape[:2]

    if processor is None:
        # DA3: one image per call — batching several images makes DA3 treat them as
        # views of the same scene (cross-view attention), which changes the output.
        with torch.no_grad():
            prediction = model.inference([image])
        depth = torch.from_numpy(np.asarray(prediction.depth[0], dtype=np.float32))
    else:
        inputs = processor(images=Image.fromarray(image), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            depth = model(**inputs).predicted_depth[0].float().cpu()

    if tuple(depth.shape) != (h, w):
        depth = F.interpolate(depth[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    return depth.numpy().astype(np.float32)


def to_closeness(raw: np.ndarray, model_key: str) -> np.ndarray:
    """Convert raw backbone output to a relative height proxy (higher = taller).

    For a distant, near-nadir sensor, height ≈ H_sensor − depth, so negating depth is
    the affine-correct transform (1/depth would add a spurious non-linearity).
    """
    return (-raw if is_depth_polarity(model_key) else raw).astype(np.float32)


def predict_relative_height(
    image: np.ndarray,
    model_key: str = "da3-metric-l",
    device: torch.device | None = None,
    max_side: int = 1536,
) -> np.ndarray:
    """Globally consistent relative height map (higher = taller) at the image resolution.

    Relative depth is only defined up to an affine transform *per inference call*, so
    tiling would give every tile its own scale/shift. We therefore run a single global
    pass (on a ≤ max_side view) and upsample.
    """
    h, w = image.shape[:2]
    view = image
    if max(h, w) > max_side:
        s = max_side / max(h, w)
        view = np.asarray(Image.fromarray(image).resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS))
    raw = predict_depth(view, model_key=model_key, device=device)
    rel = to_closeness(raw, model_key)
    if rel.shape != (h, w):
        rel = np.asarray(Image.fromarray(rel).resize((w, h), Image.Resampling.BILINEAR))
    return rel.astype(np.float32)
