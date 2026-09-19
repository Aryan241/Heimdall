"""
Stage 5b — Scale calibration of *relative* depth via RANSAC.

Used only when no trained decoder head is available: the backbone's scale-agnostic
"closeness" map (higher = nearer the sensor = taller) is mapped to metres with a
robust affine fit against a reference DEM that has already been reprojected onto
the image grid (see ``heimdall.calibration.dem``).
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import RANSACRegressor

logger = logging.getLogger(__name__)


def fit_affine_transform(
    relative: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray | None = None,
    num_samples: int = 20000,
    residual_threshold: float | None = None,
    seed: int = 42,
) -> dict:
    """Fit ``reference ≈ scale * relative + shift`` robustly.

    Args:
        relative: H×W relative closeness (higher = taller).
        reference: H×W reference elevation in metres (NaN = no data), same grid.
        mask: optional H×W bool of pixels to sample (e.g. predicted ground).
        residual_threshold: inlier tolerance in metres; defaults to 1.5 × MAD of the reference.

    Returns dict(scale, shift, inlier_fraction, n_samples).
    """
    assert relative.shape == reference.shape, "reproject the DEM onto the image grid first"
    valid = np.isfinite(reference) & np.isfinite(relative)
    if mask is not None and (valid & mask).sum() > 500:
        valid &= mask
    idx = np.flatnonzero(valid)
    if idx.size < 100:
        logger.warning("Too few valid pixels (%d) for RANSAC — returning identity.", idx.size)
        return {"scale": 1.0, "shift": 0.0, "inlier_fraction": 0.0, "n_samples": int(idx.size)}

    rng = np.random.default_rng(seed)
    if idx.size > num_samples:
        idx = rng.choice(idx, num_samples, replace=False)
    X = relative.ravel()[idx].reshape(-1, 1).astype(np.float64)
    y = reference.ravel()[idx].astype(np.float64)

    if residual_threshold is None:
        mad = np.median(np.abs(y - np.median(y)))
        residual_threshold = float(max(1.0, 1.5 * 1.4826 * mad))

    ransac = RANSACRegressor(residual_threshold=residual_threshold, random_state=seed, max_trials=500)
    ransac.fit(X, y)
    scale = float(ransac.estimator_.coef_[0])
    shift = float(ransac.estimator_.intercept_)
    inl = float(ransac.inlier_mask_.mean())
    if scale < 0:
        logger.warning("RANSAC returned a negative scale (%.4f) — the relative map is anti-correlated "
                       "with the reference; check depth polarity.", scale)
    logger.info("RANSAC fit: scale=%.4f shift=%.2f inliers=%.1f%% (thr=%.2fm, n=%d)",
                scale, shift, inl * 100, residual_threshold, len(y))
    return {"scale": scale, "shift": shift, "inlier_fraction": inl, "n_samples": int(len(y)),
            "residual_threshold_m": residual_threshold}


def apply_transform(relative: np.ndarray, scale: float, shift: float) -> np.ndarray:
    return (relative * scale + shift).astype(np.float32)
