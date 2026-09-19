"""
Stage 6 — Scale heuristics for imagery *without* georeferencing.

Estimates ground sample distance (GSD) from the apparent size of vehicles. This is
a weak prior (COCO detectors are trained on oblique views) and is only used when
neither a GeoTIFF transform nor a user-supplied ``--gsd`` is available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def estimate_gsd(image: np.ndarray, assumed_car_length_m: float = 4.6,
                 min_detections: int = 5) -> tuple[float | None, dict]:
    """Estimate GSD (m/px) of *image* (H×W×3 uint8) from detected vehicles.

    Returns (gsd or None, info). None means "not enough evidence" — callers should
    fall back to an explicit default rather than trusting a single detection.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return None, {"method": "vehicle-size", "reason": "ultralytics not installed"}

    weights = _REPO_ROOT / "yolov8n.pt"
    model = YOLO(str(weights) if weights.exists() else "yolov8n.pt")

    # Detect on a ≤1536 px view to keep it fast, then rescale lengths.
    h, w = image.shape[:2]
    scale = min(1.0, 1536 / max(h, w))
    view = image
    if scale < 1.0:
        from PIL import Image
        view = np.asarray(Image.fromarray(image).resize((int(w * scale), int(h * scale))))

    results = model(view, classes=[2, 7], conf=0.25, verbose=False)
    boxes = results[0].boxes if results else None
    n = 0 if boxes is None else len(boxes)
    if n < min_detections:
        logger.info("Vehicle-based GSD: only %d detections — not trusted.", n)
        return None, {"method": "vehicle-size", "detections": n}

    lengths = np.array([max(b[2].item(), b[3].item()) for b in boxes.xywh]) / scale
    gsd = float(assumed_car_length_m / np.median(lengths))
    if not (0.03 <= gsd <= 3.0):
        logger.info("Vehicle-based GSD %.3f m/px outside plausible range — discarded.", gsd)
        return None, {"method": "vehicle-size", "detections": n, "rejected_gsd": gsd}
    logger.info("Vehicle-based GSD: %.3f m/px from %d detections.", gsd, n)
    return gsd, {"method": "vehicle-size", "detections": n}
