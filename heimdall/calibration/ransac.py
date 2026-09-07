"""
Stage 5 — Scale Calibration via RANSAC

Anchors unscaled relative depth maps to a reference metric DEM (e.g. SRTM)
by robustly fitting a linear affine transform (scale and shift).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import RANSACRegressor

logger = logging.getLogger(__name__)

def fit_affine_transform(
    relative_depth: torch.Tensor,
    reference_dem: torch.Tensor,
    mask: torch.Tensor | None = None,
    num_samples: int = 10000,
    residual_threshold: float = 2.0
) -> tuple[float, float]:
    """
    Fits: metric_height = (scale * relative_depth) + shift
    
    Args:
        relative_depth: (H, W) or (1, 1, H, W) tensor of raw relative depth.
        reference_dem: (H, W) or (1, 1, H, W) tensor of reference heights (e.g. SRTM).
        mask: Optional (H, W) boolean mask of valid pixels to sample (e.g., ground only).
        num_samples: Max number of random pixels to sample for RANSAC.
        residual_threshold: Max error (in meters) for an inlier.
        
    Returns:
        scale (float), shift (float)
    """
    # Normalize shapes
    if relative_depth.dim() == 4:
        rel_d = relative_depth.squeeze()
    else:
        rel_d = relative_depth
        
    if reference_dem.dim() == 4:
        ref_d = reference_dem.squeeze()
    else:
        ref_d = reference_dem
        
    # Resize DEM to match relative depth if they differ
    if rel_d.shape != ref_d.shape:
        logger.info("Resizing reference DEM from %s to match relative depth %s", ref_d.shape, rel_d.shape)
        # Add batch/channel dims for interpolate
        ref_d = ref_d.unsqueeze(0).unsqueeze(0)
        ref_d = F.interpolate(ref_d, size=rel_d.shape, mode="bilinear", align_corners=False)
        ref_d = ref_d.squeeze()

    rel_np = rel_d.detach().cpu().numpy()
    ref_np = ref_d.detach().cpu().numpy()
    
    # Identify valid pixels
    valid = np.ones_like(rel_np, dtype=bool)
    if mask is not None:
        valid = valid & mask.squeeze().detach().cpu().numpy()
    
    # Filter out DEM nodata values (e.g., heavily negative or exactly 0 depending on format)
    valid = valid & (ref_np > -100)
    
    valid_indices = np.where(valid)
    num_valid = valid_indices[0].size
    
    if num_valid < 100:
        logger.warning("Too few valid pixels (%d) for RANSAC. Returning dummy scale=1, shift=0", num_valid)
        return 1.0, 0.0

    # Sample points
    if num_valid > num_samples:
        choices = np.random.choice(num_valid, num_samples, replace=False)
        y_idx = valid_indices[0][choices]
        x_idx = valid_indices[1][choices]
    else:
        y_idx = valid_indices[0]
        x_idx = valid_indices[1]
        
    X = rel_np[y_idx, x_idx].reshape(-1, 1) # Relative depth (features)
    y = ref_np[y_idx, x_idx]                # Reference metric height (target)

    # RANSAC
    logger.info("Running RANSAC on %d samples with residual_threshold=%.1fm", X.shape[0], residual_threshold)
    ransac = RANSACRegressor(residual_threshold=residual_threshold, random_state=42)
    ransac.fit(X, y)
    
    scale = float(ransac.estimator_.coef_[0])
    shift = float(ransac.estimator_.intercept_)
    
    inlier_mask = ransac.inlier_mask_
    inlier_pct = inlier_mask.sum() / len(inlier_mask) * 100
    
    logger.info("RANSAC Fit Complete. Scale: %.4f, Shift: %.2f. Inliers: %.1f%%", scale, shift, inlier_pct)
    
    return scale, shift

def apply_transform(
    relative_depth: torch.Tensor,
    scale: float,
    shift: float
) -> torch.Tensor:
    """
    Applies the scale and shift to generate a metric DSM.
    """
    return (relative_depth * scale) + shift
