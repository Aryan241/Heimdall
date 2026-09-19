"""
Validation of a predicted DSM against a reference DSM / LiDAR raster.

* Georeferenced rasters: the reference is reprojected onto the prediction grid
  (average resampling when the reference is finer, bilinear otherwise).
* Non-georeferenced: the reference is assumed to cover the same footprint and is resized.

Reports metrics on the raw surfaces *and* after removing the median vertical offset
(reference DSMs are often on a different vertical datum — ellipsoid vs geoid — which is
a constant shift unrelated to the height model). Also compares above-ground heights,
using a morphological ground estimate of the reference (white top-hat), which is the
fair comparison for relative (rDSM / nDSM) products.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np

from heimdall.eval.metrics import height_metrics, pixel_strata

logger = logging.getLogger(__name__)


def _read(path: str | Path):
    import rasterio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with rasterio.open(path) as ds:
            arr = ds.read(1).astype(np.float32)
            if ds.nodata is not None:
                arr[arr == ds.nodata] = np.nan
            georef = ds.crs is not None and not ds.transform.is_identity
            res = abs(ds.res[0])
            return arr, (ds.transform if georef else None), (ds.crs if georef else None), res


def reference_on_grid(ref_path, shape, transform, crs, pred_res) -> np.ndarray:
    ref, rt, rcrs, rres = _read(ref_path)
    ref[(ref < -1000) | (ref > 9000)] = np.nan
    if transform is not None and rt is not None:
        from rasterio.warp import Resampling, reproject
        dst = np.full(shape, np.nan, dtype=np.float32)
        finer = rres < pred_res * 0.75 if (not rcrs.is_geographic and not crs.is_geographic) else False
        reproject(source=ref, destination=dst, src_transform=rt, src_crs=rcrs, src_nodata=np.nan,
                  dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                  resampling=Resampling.average if finer else Resampling.bilinear)
        return dst
    from heimdall.geo import resize_float
    if transform is not None or rt is not None:
        logger.warning("Only one raster is georeferenced — assuming identical footprints.")
    nan = ~np.isfinite(ref)
    out = resize_float(np.nan_to_num(ref, nan=float(np.nanmedian(ref))), shape)
    if nan.any():
        from heimdall.geo import resize_mask
        out[resize_mask(nan, shape)] = np.nan
    return out


def ground_estimate(surface: np.ndarray, gsd: float, window_m: float = 40.0) -> np.ndarray:
    """Approximate bare earth of a surface model by a grey opening (removes objects < window)."""
    from scipy import ndimage
    fill = float(np.nanmedian(surface))
    s = np.nan_to_num(surface, nan=fill)
    k = max(3, int(round(window_m / max(gsd, 1e-3))) | 1)
    # Opening on a downsampled copy for speed on large rasters.
    f = max(1, k // 15)
    small = s[::f, ::f]
    ks = max(3, (k // f) | 1)
    opened = ndimage.grey_opening(small, size=(ks, ks))
    opened = ndimage.uniform_filter(opened, size=max(3, ks // 3))
    from heimdall.geo import resize_float
    return resize_float(opened, s.shape)


def compare_dsm(pred_path, ref_path, out_dir, prefix="validation", rgb: np.ndarray | None = None,
                pred_is_relative: bool = False, gsd: float | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred, pt, pcrs, pres = _read(pred_path)
    gsd = gsd or (pres if pt is not None and not pcrs.is_geographic else 1.0)
    ref = reference_on_grid(ref_path, pred.shape, pt, pcrs, pres)
    overlap = np.isfinite(pred) & np.isfinite(ref)
    if overlap.mean() < 0.01:
        raise ValueError("Prediction and reference do not overlap (check CRS / footprint).")

    ref_ground = ground_estimate(ref, gsd)
    ref_agl = np.clip(ref - ref_ground, 0, None)
    pred_ground = ground_estimate(pred, gsd)
    pred_agl = np.clip(pred - pred_ground, 0, None)

    offset = float(np.nanmedian((pred - ref)[overlap]))
    report: dict = {
        "reference": Path(ref_path).name,
        "prediction": Path(pred_path).name,
        "overlap_fraction": float(overlap.mean()),
        "gsd_m": gsd,
        "surface_raw": height_metrics(pred, ref),
        "surface_offset_removed": height_metrics(pred - offset, ref),
        "vertical_offset_m": offset,
        "above_ground": height_metrics(pred_agl, ref_agl),
    }
    if pred_is_relative:
        report["note"] = "Relative product: compare 'above_ground' (and correlation) — absolute offsets are undefined."

    if rgb is not None:
        from heimdall.geo import resize_image
        rgb_g = resize_image(rgb, pred.shape)
        strata = pixel_strata(rgb_g, np.nan_to_num(ref_agl), ref, gsd)
        report["by_class"] = {k: height_metrics(pred - offset, ref, m) for k, m in strata.items() if m.sum() > 50}
        report["by_class_above_ground"] = {k: height_metrics(pred_agl, ref_agl, m) for k, m in strata.items()
                                           if m.sum() > 50}

    # Error map (offset-removed) + scatter/histogram data for the UI.
    err = pred - offset - ref
    _save_error_png(err, out_dir / f"{prefix}_error.png")
    idx = np.flatnonzero(overlap)
    rng = np.random.default_rng(0)
    pick = rng.choice(idx, min(3000, idx.size), replace=False)
    report["scatter"] = {"ref": ref.ravel()[pick].round(3).tolist(), "pred": (pred.ravel()[pick] - offset).round(3).tolist(),
                         "ref_agl": ref_agl.ravel()[pick].round(3).tolist(), "pred_agl": pred_agl.ravel()[pick].round(3).tolist()}
    e = err[overlap]
    lim = float(np.percentile(np.abs(e), 99)) or 1.0
    hist, edges = np.histogram(np.clip(e, -lim, lim), bins=41, range=(-lim, lim))
    report["histogram"] = {"counts": hist.tolist(), "edges": edges.round(3).tolist()}
    report["files"] = {"error_png": f"{prefix}_error.png", "json": f"{prefix}.json"}
    (out_dir / f"{prefix}.json").write_text(json.dumps(report, indent=2))
    return report


def _save_error_png(err: np.ndarray, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = err[np.isfinite(err)]
    lim = float(np.percentile(np.abs(f), 98)) if f.size else 1.0
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(err, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title("Prediction − reference (median offset removed)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="m")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
