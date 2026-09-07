"""
Stage 7 (partial) — Output writing for depth maps.

Supports:
- 16-bit PNG heightmap (non-georeferenced path)
- GeoTIFF DSM (georeferenced path, preserving CRS)
- Matplotlib colorized visualization
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def save_depth_png16(depth: np.ndarray, path: str | Path) -> Path:
    """Save depth map as a 16-bit grayscale PNG.

    Normalizes to [0, 65535] range. Suitable for non-georeferenced outputs
    where we only have relative depth.
    """
    path = Path(path)
    mn, mx = depth.min(), depth.max()
    if mx - mn < 1e-8:
        scaled = np.zeros_like(depth, dtype=np.uint16)
    else:
        scaled = ((depth - mn) / (mx - mn) * 65535).astype(np.uint16)

    img = Image.fromarray(scaled, mode="I;16")
    img.save(path)
    logger.info("Saved 16-bit depth PNG: %s  (range: %.2f–%.2f)", path, mn, mx)
    return path


def save_depth_colorized(depth: np.ndarray, path: str | Path) -> Path:
    """Save a colorized (viridis) visualization of the depth map."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.imshow(depth, cmap="inferno")
    ax.set_title("Relative Depth (Depth Anything V2)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved colorized depth visualization: %s", path)
    return path


def save_depth_geotiff(
    depth: np.ndarray,
    path: str | Path,
    crs_wkt: str,
    transform_tuple: tuple,
) -> Path:
    """Save depth map as a single-band GeoTIFF, preserving the source CRS and transform."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    path = Path(path)
    h, w = depth.shape
    transform = Affine(*transform_tuple[:6])

    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=h, width=w,
        count=1,
        dtype="float32",
        crs=CRS.from_wkt(crs_wkt),
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(depth.astype(np.float32), 1)
        dst.update_tags(
            software="Heimdall/DepthWizard",
            description="Relative depth map from Depth Anything V2 (uncalibrated)",
        )

    logger.info("Saved GeoTIFF depth: %s", path)
    return path
