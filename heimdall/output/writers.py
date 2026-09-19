"""
Stage 7 — Output writers.

- Float32 GeoTIFF (georeferenced DSM / nDSM, or pixel-grid GeoTIFF for plain images)
- 16-bit PNG heightmap with the metre scale recorded in PNG text chunks
- Colourised + hill-shaded preview PNG
- Texture JPEG for the 3D mesh
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

logger = logging.getLogger(__name__)

NODATA = -9999.0


def save_geotiff(
    arr: np.ndarray,
    path: str | Path,
    transform_tuple: tuple | None,
    crs_wkt: str | None,
    description: str,
    units: str = "metre",
    valid_mask: np.ndarray | None = None,
    tags: dict | None = None,
) -> Path:
    """Write a single-band float32 GeoTIFF. Without CRS/transform the raster is written on
    the pixel grid (still a valid GeoTIFF that GIS tools and the validator can read)."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine

    path = Path(path)
    h, w = arr.shape
    data = arr.astype(np.float32).copy()
    if valid_mask is not None:
        data[~valid_mask] = NODATA
    data[~np.isfinite(data)] = NODATA

    profile = dict(driver="GTiff", height=h, width=w, count=1, dtype="float32",
                   nodata=NODATA, compress="deflate", predictor=3, tiled=True,
                   blockxsize=256, blockysize=256)
    if crs_wkt and transform_tuple:
        profile.update(crs=CRS.from_wkt(crs_wkt), transform=Affine(*transform_tuple[:6]))
    if h < 256 or w < 256:
        profile.update(tiled=False)
        profile.pop("blockxsize"); profile.pop("blockysize")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # NotGeoreferencedWarning for plain inputs
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data, 1)
            dst.set_band_description(1, description)
            dst.update_tags(software="Heimdall/DepthWizard", description=description, units=units, **(tags or {}))
            dst.update_tags(1, units=units)
    logger.info("Saved GeoTIFF: %s", path)
    return path


def save_depth_png16(arr: np.ndarray, path: str | Path) -> tuple[Path, dict]:
    """16-bit PNG with a linear mapping stored in text chunks:
    value_m = offset_m + pixel * scale_m."""
    path = Path(path)
    finite = arr[np.isfinite(arr)]
    mn = float(finite.min()) if finite.size else 0.0
    mx = float(finite.max()) if finite.size else 1.0
    scale = (mx - mn) / 65535.0 if mx > mn else 1.0
    q = np.clip(np.nan_to_num((arr - mn) / scale, nan=0.0), 0, 65535).astype(np.uint16)
    info = PngImagePlugin.PngInfo()
    info.add_text("heimdall:offset_m", f"{mn:.6f}")
    info.add_text("heimdall:scale_m", f"{scale:.9f}")
    Image.fromarray(q).save(path, pnginfo=info)
    logger.info("Saved 16-bit PNG: %s (%.2f – %.2f m)", path, mn, mx)
    return path, {"offset_m": mn, "scale_m": scale}


def hillshade(arr: np.ndarray, gsd: float = 1.0, azimuth: float = 315, altitude: float = 45) -> np.ndarray:
    gy, gx = np.gradient(arr.astype(np.float64), gsd)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(azimuth), np.radians(altitude)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip(shade, 0, 1)


def save_colorized(arr: np.ndarray, path: str | Path, title: str, gsd: float = 1.0,
                   units: str = "m", cmap: str = "turbo") -> Path:
    """Colour-mapped, hill-shaded preview with a colourbar."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    path = Path(path)
    finite = arr[np.isfinite(arr)]
    lo, hi = (np.percentile(finite, [1, 99]) if finite.size else (0, 1))
    if hi <= lo:
        hi = lo + 1
    norm = colors.Normalize(lo, hi)
    rgb = plt.get_cmap(cmap)(norm(np.nan_to_num(arr, nan=lo)))[..., :3]
    shade = hillshade(np.nan_to_num(arr, nan=lo), gsd)
    rgb = rgb * (0.55 + 0.45 * shade[..., None])
    rgb[~np.isfinite(arr)] = 0.12  # nodata → neutral dark grey

    aspect = arr.shape[1] / max(arr.shape[0], 1)
    fig, ax = plt.subplots(figsize=(8 * min(aspect, 1.6) + 1.2, 8))
    ax.imshow(rgb)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02, label=units)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved preview: %s", path)
    return path


def save_texture(image: np.ndarray, path: str | Path, max_side: int = 4096, quality: int = 90) -> Path:
    path = Path(path)
    img = Image.fromarray(image)
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)), Image.Resampling.LANCZOS)
    img.save(path, quality=quality)
    return path


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return path
