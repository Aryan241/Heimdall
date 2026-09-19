"""
Geospatial helpers shared by the pipeline: ground sample distance (GSD) in metres,
resampling to a working resolution while keeping the geotransform consistent, and
coordinate conversions for metadata consumed by the viewer.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from PIL import Image

from heimdall.ingestion.loader import GeoInfo

logger = logging.getLogger(__name__)

# The Heimdall decoder head was trained on GAMUS (0.33 m GSD, 1024² tiles cropped to 512²).
TRAINING_GSD_M = 0.33


def gsd_from_geo(geo: GeoInfo, shape: tuple[int, int]) -> float:
    """Return mean ground sample distance in metres per pixel for a georeferenced raster."""
    res_x, res_y = abs(geo.resolution[0]), abs(geo.resolution[1])
    if not geo.is_geographic:
        # Projected CRS. Assume metre units (true for UTM and virtually all EO products);
        # rasterio exposes linear units but many files omit them.
        return float((res_x + res_y) / 2)
    lat = center_latlon(geo, shape)[0]
    mx = res_x * 111_320.0 * math.cos(math.radians(lat))
    my = res_y * 110_540.0
    return float((mx + my) / 2)


def pixel_to_crs(transform: tuple, row: float, col: float) -> tuple[float, float]:
    a, b, c, d, e, f = transform[:6]
    return a * col + b * row + c, d * col + e * row + f


def center_latlon(geo: GeoInfo, shape: tuple[int, int]) -> tuple[float, float]:
    """Latitude/longitude (EPSG:4326) of the raster centre."""
    h, w = shape
    x, y = pixel_to_crs(geo.transform, h / 2, w / 2)
    if geo.is_geographic:
        return float(y), float(x)
    try:
        from rasterio.crs import CRS
        from rasterio.warp import transform as warp_transform
        xs, ys = warp_transform(CRS.from_wkt(geo.crs_wkt), CRS.from_epsg(4326), [x], [y])
        return float(ys[0]), float(xs[0])
    except Exception:
        return float("nan"), float("nan")


def scaled_transform(transform: tuple, src_shape: tuple[int, int], dst_shape: tuple[int, int]) -> tuple:
    """Geotransform for the same footprint sampled on a grid of *dst_shape*."""
    a, b, c, d, e, f = transform[:6]
    sy = src_shape[0] / dst_shape[0]
    sx = src_shape[1] / dst_shape[1]
    return (a * sx, b * sy, c, d * sx, e * sy, f)


def resize_image(img: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if img.shape[:2] == (h, w):
        return img
    method = Image.Resampling.LANCZOS if (h * w) < img.shape[0] * img.shape[1] else Image.Resampling.BICUBIC
    return np.asarray(Image.fromarray(img).resize((w, h), method))


def resize_float(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if arr.shape[:2] == (h, w):
        return arr.astype(np.float32)
    return np.asarray(Image.fromarray(arr.astype(np.float32)).resize((w, h), Image.Resampling.BILINEAR))


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if mask.shape[:2] == (h, w):
        return mask
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 127


def working_shape(
    shape: tuple[int, int],
    native_gsd: float | None,
    target_gsd: float | None,
    max_upsample: float = 3.0,
    max_side: int = 8192,
) -> tuple[tuple[int, int], float | None]:
    """Choose the processing grid.

    The decoder head learned metric heights at ~0.33 m/px, so imagery is resampled
    towards that GSD. Up-sampling is capped (coarse 5 m imagery can't be turned into
    0.33 m detail) and so is the longest side (memory / runtime).

    Returns (new_shape, working_gsd).
    """
    h, w = shape
    scale = 1.0
    if native_gsd and target_gsd:
        scale = native_gsd / target_gsd            # >1 means up-sample
        scale = min(scale, max_upsample)
    if max(h, w) * scale > max_side:
        scale = max_side / max(h, w)
    new_shape = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
    work_gsd = native_gsd / scale if native_gsd else None
    return new_shape, work_gsd
