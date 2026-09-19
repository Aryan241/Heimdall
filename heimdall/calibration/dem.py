"""
Stage 5 — Absolute elevation reference.

Provides the low-resolution terrain datum that turns the decoder's height-above-ground
(nDSM) into an absolute DSM:

    DSM(x, y) = DTM(x, y) + nDSM(x, y)

Sources, in order of preference:
  1. A user-supplied DEM (SRTM / Copernicus / CartoDEM / any GeoTIFF), read as float
     and reprojected onto the image grid (CRS, footprint and resolution are honoured).
  2. Copernicus GLO-30 (30 m) fetched on demand from the public AWS Open Data bucket
     (no API key; only the blocks covering the scene are read, results are cached).
  3. Ground Control Points (CSV) — used to remove residual vertical bias / tilt, or on
     their own to define a planar datum when no DEM is available.

The 30 m products are surface models at radar wavelengths, so buildings and canopy leak
into them. A grey-scale morphological opening at the DEM's native resolution suppresses
those positive bumps before up-sampling, giving a smoother bare-earth estimate.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

COPERNICUS_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
)
CACHE_DIR = Path(os.environ.get("HEIMDALL_CACHE", Path.home() / ".cache" / "heimdall")) / "dem"


@dataclass
class TerrainReference:
    dtm: np.ndarray                 # float32 H×W, metres (orthometric, EGM2008 for Copernicus)
    source: str                     # human-readable description
    native_res_m: float = 30.0
    details: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def _latlon_bounds(transform: tuple, crs_wkt: str, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    from rasterio.crs import CRS
    from rasterio.transform import Affine, array_bounds
    from rasterio.warp import transform_bounds
    h, w = shape
    left, bottom, right, top = array_bounds(h, w, Affine(*transform[:6]))
    return transform_bounds(CRS.from_wkt(crs_wkt), CRS.from_epsg(4326), left, bottom, right, top, densify_pts=21)


def _reproject_to_grid(src_path: str, transform: tuple, crs_wkt: str, shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly reproject band 1 of *src_path* onto the destination grid. NaN = no data."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject

    dst = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(src_path) as src:
        src_nodata = src.nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_nodata=src_nodata,
            dst_transform=Affine(*transform[:6]),
            dst_crs=CRS.from_wkt(crs_wkt),
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    # Common sentinel values that some DEMs store without declaring nodata.
    dst[(dst < -1000) | (dst > 9000)] = np.nan
    return dst


def _fill_nan(arr: np.ndarray) -> np.ndarray:
    mask = ~np.isfinite(arr)
    if not mask.any():
        return arr
    if mask.all():
        return np.zeros_like(arr)
    from scipy import ndimage
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return arr[tuple(idx)]


def _coarse_grid(transform: tuple, shape: tuple[int, int], gsd_m: float, target_res_m: float):
    """Grid over the same footprint at ~target_res_m (with a one-cell margin)."""
    from heimdall.geo import scaled_transform
    h, w = shape
    factor = max(1.0, target_res_m / max(gsd_m, 1e-6))
    ch, cw = max(3, int(math.ceil(h / factor))), max(3, int(math.ceil(w / factor)))
    ct = scaled_transform(transform, (h, w), (ch, cw))
    # Add a one-cell margin so the opening has context at the borders.
    a, b, c, d, e, f = ct
    ct = (a, b, c - a - b, d, e, f - d - e)
    return ct, (ch + 2, cw + 2)


def _terrain_from_source(src: str, transform, crs_wkt, shape, gsd_m, native_res_m, opening_cells):
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, reproject
    from scipy import ndimage

    ct, cshape = _coarse_grid(transform, shape, gsd_m, native_res_m)
    coarse = _reproject_to_grid(src, ct, crs_wkt, cshape)
    valid_frac = float(np.isfinite(coarse).mean())
    if valid_frac < 0.05:
        raise ValueError(f"DEM {src} does not cover the image footprint (valid fraction {valid_frac:.1%}).")
    coarse = _fill_nan(coarse)
    raw_coarse = coarse.copy()
    if opening_cells and opening_cells > 1 and min(cshape) >= opening_cells:
        coarse = ndimage.grey_opening(coarse, size=(opening_cells, opening_cells))
    coarse = ndimage.gaussian_filter(coarse, sigma=0.75)

    fine = np.empty(shape, dtype=np.float32)
    reproject(
        source=coarse.astype(np.float32), destination=fine,
        src_transform=Affine(*ct[:6]), src_crs=CRS.from_wkt(crs_wkt),
        dst_transform=Affine(*transform[:6]), dst_crs=CRS.from_wkt(crs_wkt),
        resampling=Resampling.cubic_spline,
    )
    return fine, {
        "coverage": valid_frac,
        "coarse_shape": list(cshape),
        "coarse_min": float(np.nanmin(raw_coarse)),
        "coarse_max": float(np.nanmax(raw_coarse)),
        "opening_cells": opening_cells,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def copernicus_urls(bounds_ll: tuple[float, float, float, float]) -> list[str]:
    west, south, east, north = bounds_ll
    urls = []
    for lat in range(math.floor(south), math.floor(north) + 1):
        for lon in range(math.floor(west), math.floor(east) + 1):
            urls.append(COPERNICUS_URL.format(
                ns="N" if lat >= 0 else "S", lat=abs(lat),
                ew="E" if lon >= 0 else "W", lon=abs(lon),
            ))
    return urls


def fetch_copernicus(transform: tuple, crs_wkt: str, shape: tuple[int, int]) -> str:
    """Build (and cache) a GeoTIFF of Copernicus GLO-30 covering the image footprint.

    Returns a local path. Only the needed blocks are streamed from the COGs.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject

    west, south, east, north = _latlon_bounds(transform, crs_wkt, shape)
    pad = 0.003  # ~300 m margin
    west, south, east, north = west - pad, south - pad, east + pad, north + pad
    key = hashlib.sha1(f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}".encode()).hexdigest()[:16]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / f"glo30_{key}.tif"
    if out.exists():
        logger.info("Using cached Copernicus GLO-30 extract: %s", out)
        return str(out)

    res = 1.0 / 3600.0  # 1 arc-second
    w = max(2, int(math.ceil((east - west) / res)))
    h = max(2, int(math.ceil((north - south) / res)))
    dst_t = from_bounds(west, south, east, north, w, h)
    mosaic = np.full((h, w), np.nan, dtype=np.float32)

    env = rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                       GDAL_HTTP_TIMEOUT="30", GDAL_HTTP_MAX_RETRY="3")
    with env:
        for url in copernicus_urls((west, south, east, north)):
            logger.info("Streaming Copernicus GLO-30 tile: %s", url.rsplit("/", 1)[-1])
            part = np.full((h, w), np.nan, dtype=np.float32)
            try:
                with rasterio.open("/vsicurl/" + url) as src:
                    reproject(source=rasterio.band(src, 1), destination=part,
                              dst_transform=dst_t, dst_crs=CRS.from_epsg(4326),
                              dst_nodata=np.nan, resampling=Resampling.bilinear)
            except Exception as exc:  # ocean tiles don't exist in GLO-30
                logger.warning("Tile unavailable (%s): %s", url.rsplit("/", 1)[-1], exc)
                continue
            fill = np.isnan(mosaic) & np.isfinite(part)
            mosaic[fill] = part[fill]

    if not np.isfinite(mosaic).any():
        raise RuntimeError("Copernicus GLO-30 returned no data for this footprint (offline, or over ocean).")

    with rasterio.open(out, "w", driver="GTiff", height=h, width=w, count=1, dtype="float32",
                       crs=CRS.from_epsg(4326), transform=dst_t, nodata=np.nan, compress="deflate") as dst:
        dst.write(mosaic, 1)
        dst.update_tags(source="Copernicus DEM GLO-30 (ESA/Airbus), AWS Open Data", vertical_datum="EGM2008")
    return str(out)


def terrain_reference(
    transform: tuple,
    crs_wkt: str,
    shape: tuple[int, int],
    gsd_m: float,
    dem_path: str | None = None,
    auto_fetch: bool = True,
    opening_cells: int = 3,
) -> TerrainReference | None:
    """Return a bare-earth DTM on the working grid, or None if no source is available."""
    if dem_path:
        src, desc, native = str(dem_path), f"user DEM ({Path(dem_path).name})", None
        try:
            import rasterio
            with rasterio.open(src) as ds:
                if ds.crs is None or ds.transform.is_identity:
                    return _terrain_from_unreferenced(src, shape, desc)
                native = _native_res_m(ds)
        except Exception as exc:
            raise ValueError(f"Cannot read reference DEM {dem_path}: {exc}") from exc
    elif auto_fetch:
        try:
            src = fetch_copernicus(transform, crs_wkt, shape)
        except Exception as exc:
            logger.warning("Automatic Copernicus GLO-30 download failed: %s", exc)
            return None
        desc, native = "Copernicus GLO-30 (auto)", 30.0
    else:
        return None

    dtm, details = _terrain_from_source(src, transform, crs_wkt, shape, gsd_m, native or 30.0, opening_cells)
    logger.info("Terrain reference from %s: %.1f – %.1f m", desc, float(dtm.min()), float(dtm.max()))
    return TerrainReference(dtm=dtm, source=desc, native_res_m=native or 30.0, details=details)


def _native_res_m(ds) -> float:
    rx, ry = abs(ds.res[0]), abs(ds.res[1])
    if ds.crs.is_geographic:
        lat = (ds.bounds.top + ds.bounds.bottom) / 2
        return float((rx * 111_320 * math.cos(math.radians(lat)) + ry * 110_540) / 2)
    return float((rx + ry) / 2)


def _terrain_from_unreferenced(path: str, shape: tuple[int, int], desc: str) -> TerrainReference:
    """DEM without georeferencing: assume it covers exactly the image footprint."""
    import rasterio
    from heimdall.geo import resize_float
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype(np.float32)
            if ds.nodata is not None:
                arr[arr == ds.nodata] = np.nan
    arr[(arr < -1000) | (arr > 9000)] = np.nan
    arr = _fill_nan(arr)
    logger.warning("Reference DEM has no CRS — assuming it spans exactly the image footprint.")
    return TerrainReference(dtm=resize_float(arr, shape), source=desc + " [unreferenced, footprint assumed]")


def load_reference_dem_raw(path: str, shape: tuple[int, int], transform=None, crs_wkt=None) -> np.ndarray:
    """Reference DEM as float on the working grid, *without* bare-earth filtering (for RANSAC)."""
    import rasterio
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path) as ds:
            referenced = ds.crs is not None and not ds.transform.is_identity
    if referenced and transform is not None and crs_wkt is not None:
        return _reproject_to_grid(path, transform, crs_wkt, shape)
    return _terrain_from_unreferenced(path, shape, "").dtm


# ─────────────────────────────────────────────────────────────────────────────
# Ground Control Points
# ─────────────────────────────────────────────────────────────────────────────

def load_gcps(csv_path: str, transform: tuple | None, crs_wkt: str | None) -> list[tuple[float, float, float]]:
    """Read GCPs as (row, col, z) on the working grid.

    Accepted headers (case-insensitive):
      lon,lat,z   — WGS84 degrees
      x,y,z       — coordinates in the image CRS
      row,col,z   — pixel coordinates on the *working* grid
    'z' may also be called elevation / height / h / alt.
    """
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return []
    keys = {k.lower().strip(): k for k in rows[0].keys()}

    def col(*names):
        for n in names:
            if n in keys:
                return keys[n]
        return None

    zk = col("z", "elevation", "height", "h", "alt", "altitude", "elev")
    if zk is None:
        raise ValueError("GCP CSV needs a z/elevation/height column.")

    out: list[tuple[float, float, float]] = []
    if col("row") and col("col"):
        for r in rows:
            out.append((float(r[col("row")]), float(r[col("col")]), float(r[zk])))
        return out

    if transform is None:
        raise ValueError("Geographic GCPs require a georeferenced image; use row,col,z for plain images.")
    from rasterio.transform import Affine
    inv = ~Affine(*transform[:6])

    if col("lon", "longitude") and col("lat", "latitude"):
        from rasterio.crs import CRS
        from rasterio.warp import transform as warp_transform
        lons = [float(r[col("lon", "longitude")]) for r in rows]
        lats = [float(r[col("lat", "latitude")]) for r in rows]
        xs, ys = warp_transform(CRS.from_epsg(4326), CRS.from_wkt(crs_wkt), lons, lats)
    elif col("x", "easting") and col("y", "northing"):
        xs = [float(r[col("x", "easting")]) for r in rows]
        ys = [float(r[col("y", "northing")]) for r in rows]
    else:
        raise ValueError("GCP CSV needs lon/lat, x/y or row/col columns.")

    for x, y, r in zip(xs, ys, rows):
        c_, r_ = inv * (x, y)
        out.append((float(r_), float(c_), float(r[zk])))
    return out


def fit_gcp_correction(surface: np.ndarray, gcps: list[tuple[float, float, float]]):
    """Fit an offset (1–2 GCPs) or a robust plane (≥3) to GCP residuals.

    Returns (correction_array, report_dict). Points outside the grid are ignored.
    """
    h, w = surface.shape
    pts = [(r, c, z) for r, c, z in gcps if 0 <= r < h and 0 <= c < w]
    if not pts:
        return np.zeros_like(surface), {"used": 0, "note": "no GCPs inside the image"}
    rr = np.array([p[0] for p in pts])
    cc = np.array([p[1] for p in pts])
    zz = np.array([p[2] for p in pts])
    pred = surface[np.clip(rr.astype(int), 0, h - 1), np.clip(cc.astype(int), 0, w - 1)]
    resid = zz - pred

    if len(pts) >= 3 and np.linalg.matrix_rank(np.column_stack([rr, cc, np.ones_like(rr)])) == 3:
        A = np.column_stack([rr / h, cc / w, np.ones_like(rr)])
        keep = np.ones(len(pts), bool)
        for _ in range(3):  # iteratively reweighted: drop >2.5 σ outliers
            coef, *_ = np.linalg.lstsq(A[keep], resid[keep], rcond=None)
            fit = A @ coef
            err = resid - fit
            s = 1.4826 * np.median(np.abs(err[keep] - np.median(err[keep]))) + 1e-6
            new_keep = np.abs(err) <= max(2.5 * s, 0.5)
            if new_keep.sum() < 3 or (new_keep == keep).all():
                break
            keep = new_keep
        yy, xx = np.mgrid[0:h, 0:w]
        corr = (coef[0] * yy / h + coef[1] * xx / w + coef[2]).astype(np.float32)
        model = "plane"
    else:
        corr = np.full_like(surface, float(np.median(resid)), dtype=np.float32)
        model = "offset"
        keep = np.ones(len(pts), bool)

    after = zz - (pred + corr[np.clip(rr.astype(int), 0, h - 1), np.clip(cc.astype(int), 0, w - 1)])
    return corr, {
        "used": int(keep.sum()),
        "total": len(pts),
        "model": model,
        "rmse_before_m": float(np.sqrt(np.mean(resid ** 2))),
        "rmse_after_m": float(np.sqrt(np.mean(after[keep] ** 2))),
    }
