"""
Vehicle detection as a *ground prior* for height regularisation.

Monocular height models confuse large, featureless paved areas (parking lots, plazas)
with flat concrete roofs. Parked vehicles disambiguate this: the surface immediately
around a car is ground. We detect vehicles with a DOTA-trained oriented-box YOLO model
(aerial imagery, classes "small vehicle" / "large vehicle") and rasterise them onto the
refinement grid.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
DETECT_GSD_M = 0.15            # DOTA imagery is ~0.1–0.5 m; small vehicles detect best near 0.15 m
VEHICLE_CLASSES = {"small vehicle", "large vehicle"}


def detect_vehicles(rgb: np.ndarray, gsd: float, out_shape: tuple[int, int], conf: float = 0.25,
                    tile: int = 1024) -> tuple[np.ndarray, int]:
    """Return (label image on *out_shape* with one id per vehicle, count). 0 = no vehicle.

    *rgb* is the native image with ground sample distance *gsd* (m/px).
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return np.zeros(out_shape, np.int32), 0
    weights = _REPO / "yolov8n-obb.pt"
    try:
        model = YOLO(str(weights) if weights.exists() else "yolov8n-obb.pt")
    except Exception as exc:  # offline without cached weights
        logger.warning("Vehicle detector unavailable: %s", exc)
        return np.zeros(out_shape, np.int32), 0

    s = min(1.0, gsd / DETECT_GSD_M) if gsd else 1.0
    view = rgb if s >= 0.999 else np.asarray(
        Image.fromarray(rgb).resize((max(1, int(rgb.shape[1] * s)), max(1, int(rgb.shape[0] * s))), Image.Resampling.LANCZOS))
    vh, vw = view.shape[:2]
    polys: list[np.ndarray] = []
    step = tile - 128
    for y0 in range(0, max(1, vh - 128), step):
        for x0 in range(0, max(1, vw - 128), step):
            crop = view[y0:y0 + tile, x0:x0 + tile]
            res = model(crop, conf=conf, verbose=False)[0]
            if res.obb is None:
                continue
            for cls, pts in zip(res.obb.cls.tolist(), res.obb.xyxyxyxy.cpu().numpy()):
                if model.names[int(cls)] in VEHICLE_CLASSES:
                    polys.append(pts + np.array([x0, y0]))

    # Rasterise (dedupe overlapping detections from tile overlaps by drawing into one canvas).
    oh, ow = out_shape
    sx, sy = ow / vw, oh / vh
    canvas = Image.new("I", (ow, oh), 0)
    draw = ImageDraw.Draw(canvas)
    for i, p in enumerate(polys, start=1):
        draw.polygon([(float(x * sx), float(y * sy)) for x, y in p], fill=i)
    labels = np.asarray(canvas, dtype=np.int32)
    n = len(np.unique(labels)) - 1
    logger.info("Vehicle prior: %d vehicles detected", n)
    return labels, n
