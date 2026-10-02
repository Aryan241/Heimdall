"""
Stage 6 — Scale heuristics for imagery *without* georeferencing.

Estimates ground sample distance (GSD) from the apparent size of vehicles: a car is about
4.6 m long, so its length in pixels implies the image scale. Used only when neither a GeoTIFF
transform nor a user-supplied ``--gsd`` is available, and it returns ``None`` rather than a
guess whenever the evidence is thin.

Two things make or break this in practice, both established by measurement:

* **The detector must be trained on aerial imagery.** A COCO detector sees oblique, eye-level
  photographs; on nadir views it finds nothing at all — zero detections on 0.25 m aerial
  imagery at every scale tried, which made this estimator silently useless on exactly the
  imagery it exists for. The DOTA-trained oriented-box model is used instead.
* **The image must be tiled, not shrunk.** YOLO resizes its input to 640 px, so a car in a
  2400 px scene becomes about five pixels and disappears. ``detect_vehicles`` already tiles at
  full resolution, so this reuses it rather than repeating the mistake.

Even so it frequently finds nothing, because many aerial scenes have no clearly visible cars.
That is expected: the caller then falls back to an explicit default and warns.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The detector is happiest near this resolution. We do not know the true GSD yet — that is the
# point — so ask for no rescaling and let the tiling carry the small objects.
_ASSUMED_DETECT_GSD_M = 0.15


def estimate_gsd(image: np.ndarray, assumed_car_length_m: float = 4.6,
                 min_detections: int = 5) -> tuple[float | None, dict]:
    """Estimate GSD (m/px) of *image* (H×W×3 uint8) from detected vehicles.

    Returns (gsd or None, info). None means "not enough evidence" — callers should fall back to
    an explicit default rather than trusting a couple of detections.
    """
    try:
        from skimage.measure import regionprops

        from heimdall.refine.vehicles import detect_vehicles
    except ImportError as exc:
        return None, {"method": "vehicle-size", "reason": f"{type(exc).__name__}: {exc}"}

    try:
        labels, n = detect_vehicles(image, gsd=_ASSUMED_DETECT_GSD_M, out_shape=image.shape[:2],
                                    classes={"small vehicle"})
    except Exception as exc:
        logger.warning("Vehicle detection failed during GSD estimation: %s", exc)
        return None, {"method": "vehicle-size", "reason": str(exc)}

    if n < min_detections:
        logger.info("Vehicle-based GSD: only %d aerial vehicle detections — not trusted.", n)
        return None, {"method": "vehicle-size", "detections": int(n)}

    # Oriented boxes are rasterised into the label image, so each blob's major axis is the
    # vehicle's length whichever direction it is parked in.
    lengths = np.array([max(r.axis_major_length, r.axis_minor_length) for r in regionprops(labels)])
    lengths = lengths[lengths > 1.0]
    if lengths.size < min_detections:
        return None, {"method": "vehicle-size", "detections": int(lengths.size)}

    median_px = float(np.median(lengths))
    gsd = float(assumed_car_length_m / median_px)
    if not (0.03 <= gsd <= 3.0):
        logger.info("Vehicle-based GSD %.3f m/px outside the plausible range — discarded.", gsd)
        return None, {"method": "vehicle-size", "detections": int(n), "rejected_gsd": gsd}

    logger.info("Vehicle-based GSD: %.3f m/px from %d detections (median car %.1f px).",
                gsd, n, median_px)
    return gsd, {"method": "vehicle-size", "detections": int(n), "median_vehicle_px": median_px}
