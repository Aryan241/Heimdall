"""
Stage 1 — Ingestion: detect input type, extract metadata, route to pipeline.

Handles:
- Plain PNG/JPG/BMP/TIFF (no georeferencing)  -> kind="plain"
- GeoTIFF with CRS + geotransform               -> kind="georeferenced"
- 8/16-bit and float rasters, 1/3/4+ bands, configurable band order
  (ISRO multispectral products are often ordered B, G, R, NIR).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Large scenes are legitimate input — don't let PIL treat them as decompression bombs.
Image.MAX_IMAGE_PIXELS = None


@dataclass
class GeoInfo:
    """Geospatial metadata extracted from a GeoTIFF."""
    crs_wkt: str
    crs_epsg: int | None
    transform: tuple  # rasterio Affine as 6/9-tuple (a, b, c, d, e, f, ...)
    bounds: tuple      # (left, bottom, right, top) in CRS units
    resolution: tuple  # (res_x, res_y) in CRS units
    is_geographic: bool = False


@dataclass
class ImagePayload:
    """Unified container for an ingested image, carrying everything downstream
    stages need to decide what to do."""
    path: Path
    image: np.ndarray                        # H×W×3, uint8, RGB
    kind: Literal["georeferenced", "plain"]
    geo: GeoInfo | None = None
    original_shape: tuple = ()               # (H, W) as read from disk
    valid_mask: np.ndarray | None = None     # H×W bool, False where source is nodata
    metadata: dict = field(default_factory=dict)


def to_uint8(bands: np.ndarray, valid: np.ndarray | None = None,
             low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    """Convert an H×W×C raster of any dtype to uint8 with a per-band percentile stretch.

    A min/max stretch lets a handful of saturated or dark outlier pixels crush the
    contrast of a 16-bit scene; a 1–99 % stretch is the standard remote-sensing default.
    """
    if bands.dtype == np.uint8:
        return bands
    out = np.zeros(bands.shape, dtype=np.uint8)
    for c in range(bands.shape[-1]):
        band = bands[..., c].astype(np.float32)
        sample = band[valid] if valid is not None and valid.any() else band.ravel()
        sample = sample[np.isfinite(sample)]
        if sample.size == 0:
            continue
        lo, hi = np.percentile(sample, [low_pct, high_pct])
        if hi <= lo:
            hi = lo + 1.0
        out[..., c] = (np.clip((band - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return out


def _try_load_geotiff(path: Path, band_order: Sequence[int] | None) -> ImagePayload | None:
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
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            src = rasterio.open(path)
    except Exception:
        return None

    with src:
        if src.crs is None or src.transform.is_identity:
            logger.info("File opened by rasterio but lacks CRS/transform — treating as plain image.")
            return None

        if band_order:
            idx = list(band_order)
        elif src.count >= 3:
            idx = _guess_rgb_bands(src)
        else:
            idx = [1, 1, 1]
        idx = [min(max(i, 1), src.count) for i in idx]
        bands = np.stack([src.read(i) for i in idx], axis=-1)

        valid = None
        masks = src.read_masks(idx[0])
        if (masks == 0).any():
            valid = masks > 0
        if src.nodata is not None:
            nd = np.all(bands == src.nodata, axis=-1)
            valid = ~nd if valid is None else (valid & ~nd)

        rgb = to_uint8(bands, valid)
        epsg = src.crs.to_epsg() if src.crs else None
        geo = GeoInfo(
            crs_wkt=src.crs.to_wkt(),
            crs_epsg=epsg,
            transform=tuple(src.transform),
            bounds=tuple(src.bounds),
            resolution=(src.res[0], src.res[1]),
            is_geographic=bool(src.crs.is_geographic),
        )
        logger.info("Loaded georeferenced image: %s  CRS=%s  shape=%s  bands=%s  dtype=%s",
                    path.name, src.crs, rgb.shape[:2], idx, src.dtypes[0])
        return ImagePayload(
            path=path, image=rgb, kind="georeferenced", geo=geo,
            original_shape=rgb.shape[:2], valid_mask=valid,
            metadata={"bands_used": idx, "source_dtype": src.dtypes[0], "band_count": src.count},
        )


def _guess_rgb_bands(src) -> list[int]:
    """Pick R, G, B band indices (1-based) from band descriptions / colour interpretation."""
    from rasterio.enums import ColorInterp
    ci = list(src.colorinterp)
    lookup = {ColorInterp.red: None, ColorInterp.green: None, ColorInterp.blue: None}
    for i, c in enumerate(ci, start=1):
        if c in lookup and lookup[c] is None:
            lookup[c] = i
    if all(v is not None for v in lookup.values()):
        return [lookup[ColorInterp.red], lookup[ColorInterp.green], lookup[ColorInterp.blue]]

    descs = [(d or "").lower() for d in src.descriptions]
    by_name = {}
    for i, d in enumerate(descs, start=1):
        for key in ("red", "green", "blue"):
            if key in d and key not in by_name:
                by_name[key] = i
    if len(by_name) == 3:
        return [by_name["red"], by_name["green"], by_name["blue"]]
    return [1, 2, 3]


def _load_plain_image(path: Path, band_order: Sequence[int] | None) -> ImagePayload:
    """Load a plain PNG/JPG/BMP/TIFF image."""
    img = Image.open(path)
    if img.mode in ("I;16", "I;16B", "I", "F"):
        arr = np.asarray(img)
        arr = to_uint8(np.stack([arr] * 3, axis=-1))
    else:
        arr = np.asarray(img.convert("RGB"))
    if band_order and arr.ndim == 3:
        arr = arr[..., [min(max(i, 1), 3) - 1 for i in band_order]]
    logger.info("Loaded plain image: %s  shape=%s", path.name, arr.shape[:2])
    return ImagePayload(path=path, image=np.ascontiguousarray(arr), kind="plain",
                        original_shape=arr.shape[:2])


def ingest(path: str | Path, band_order: Sequence[int] | None = None) -> ImagePayload:
    """Detect image type and load into an ImagePayload.

    Tries GeoTIFF first (via rasterio); falls back to PIL for plain images.
    ``band_order`` is a 1-based list such as ``[3, 2, 1]`` for B,G,R-ordered products.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    payload = _try_load_geotiff(path, band_order)
    if payload is not None:
        return payload
    return _load_plain_image(path, band_order)
