"""
Stage 1 — Ingestion: detect input type, extract metadata, route to pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes for pipeline routing
# ---------------------------------------------------------------------------

@dataclass
class GeoInfo:
    """Geospatial metadata extracted from a GeoTIFF."""
    crs_wkt: str
    crs_epsg: int | None
    transform: tuple  # rasterio Affine as tuple
    bounds: tuple      # (left, bottom, right, top) in CRS units
    resolution: tuple  # (res_x, res_y) in CRS units


@dataclass
class ImagePayload:
    """Unified container for an ingested image, carrying everything downstream
    stages need to decide what to do."""
    path: Path
    image: np.ndarray                        # H×W×C, uint8, RGB
    kind: Literal["georeferenced", "plain"]
    geo: GeoInfo | None = None
    original_shape: tuple = ()               # (H, W) before any tiling
    metadata: dict = field(default_factory=dict)


def _try_load_geotiff(path: Path) -> ImagePayload | None:
    """Attempt to open *path* as a GeoTIFF with valid CRS + transform.

    Returns an ImagePayload with kind='georeferenced' on success,
    or None if the file isn't a GeoTIFF / lacks georeferencing.
    """
    try:
        import rasterio
    except ImportError:
        logger.warning("rasterio not installed — skipping GeoTIFF detection")
        return None

    try:
        with rasterio.open(path) as src:
            # A valid georeferenced raster must have a CRS and a non-identity transform.
            if src.crs is None or src.transform.is_identity:
                logger.info("File opened by rasterio but lacks CRS/transform — treating as plain image.")
                return None

            # Read as RGB (bands 1-3). Handle single-band gracefully.
            if src.count >= 3:
                rgb = np.stack([src.read(i) for i in (1, 2, 3)], axis=-1)  # H×W×3
            elif src.count == 1:
                band = src.read(1)
                rgb = np.stack([band, band, band], axis=-1)
            else:
                band1 = src.read(1)
                rgb = np.stack([band1, band1, band1], axis=-1)

            # Ensure uint8 — some GeoTIFFs are 16-bit or float.
            if rgb.dtype != np.uint8:
                if np.issubdtype(rgb.dtype, np.floating):
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                else:
                    # Integer but not uint8 — scale from actual range.
                    mn, mx = rgb.min(), rgb.max()
                    if mx > mn:
                        rgb = ((rgb.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
                    else:
                        rgb = np.zeros_like(rgb, dtype=np.uint8)

            epsg = src.crs.to_epsg() if src.crs else None
            geo = GeoInfo(
                crs_wkt=src.crs.to_wkt(),
                crs_epsg=epsg,
                transform=tuple(src.transform),
                bounds=tuple(src.bounds),
                resolution=(src.res[0], src.res[1]),
            )
            logger.info("Loaded georeferenced image: %s  CRS=%s  shape=%s", path.name, src.crs, rgb.shape[:2])
            return ImagePayload(
                path=path,
                image=rgb,
                kind="georeferenced",
                geo=geo,
                original_shape=rgb.shape[:2],
            )
    except rasterio.errors.RasterioIOError:
        return None


def _load_plain_image(path: Path) -> ImagePayload:
    """Load a plain PNG/JPG/BMP image."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img)
    logger.info("Loaded plain image: %s  shape=%s", path.name, arr.shape[:2])
    return ImagePayload(
        path=path,
        image=arr,
        kind="plain",
        original_shape=arr.shape[:2],
    )


def ingest(path: str | Path) -> ImagePayload:
    """Detect image type and load into an ImagePayload.

    Tries GeoTIFF first (via rasterio); falls back to PIL for plain images.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    # Try GeoTIFF first
    payload = _try_load_geotiff(path)
    if payload is not None:
        return payload

    # Fallback: plain image
    return _load_plain_image(path)
