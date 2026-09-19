"""
Height-accuracy metrics and stratification used by eval.py (GAMUS benchmark) and
validate.py / the web validation panel (predicted DSM vs a reference DSM / LiDAR).

All metrics are computed over *every* valid pixel — ground included. Excluding
zero-height ground (as depth-estimation code often does) hides most of the error
budget of a height model and is not what DSM validation against LiDAR measures.
"""

from __future__ import annotations

import numpy as np

HEIGHT_OBJECT_M = 2.5      # nDSM threshold separating objects (buildings/trees) from ground


def height_metrics(pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """RMSE, MAE, bias, NMAD, LE90, Pearson r, R² (and δ-accuracies on object pixels)."""
    valid = np.isfinite(pred) & np.isfinite(ref)
    if mask is not None:
        valid &= mask
    p = pred[valid].astype(np.float64)
    t = ref[valid].astype(np.float64)
    n = int(p.size)
    if n < 10:
        return {"n": n}
    err = p - t
    abs_err = np.abs(err)
    out = {
        "n": n,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(abs_err.mean()),
        "bias": float(err.mean()),
        "median_abs": float(np.median(abs_err)),
        "nmad": float(1.4826 * np.median(np.abs(err - np.median(err)))),
        "le90": float(np.percentile(abs_err, 90)),
        "rmse_debiased": float(np.sqrt(np.mean((err - err.mean()) ** 2))),
    }
    if p.std() > 1e-9 and t.std() > 1e-9:
        r = float(np.corrcoef(p, t)[0, 1])
        out["pearson_r"] = r
        out["r2"] = float(1 - np.sum(err ** 2) / np.sum((t - t.mean()) ** 2))
    obj = t >= 1.0
    if obj.sum() >= 10 and (p[obj] > 0).any():
        ratio = np.maximum(p[obj] / t[obj], t[obj] / np.maximum(p[obj], 1e-3))
        out["delta1"] = float((ratio < 1.25).mean())
        out["delta2"] = float((ratio < 1.25 ** 2).mean())
        out["delta3"] = float((ratio < 1.25 ** 3).mean())
    return out


def vegetation_mask(rgb: np.ndarray) -> np.ndarray:
    """Excess-green vegetation index on chromatic coordinates (RGB-only imagery)."""
    x = rgb.astype(np.float32) + 1e-3
    s = x.sum(-1, keepdims=True)
    r, g, b = np.moveaxis(x / s, -1, 0)
    return (2 * g - r - b) > 0.06


def pixel_strata(rgb: np.ndarray, ref_ndsm: np.ndarray, ref_surface: np.ndarray | None = None,
                 gsd: float = 1.0) -> dict[str, np.ndarray]:
    """Per-pixel classes: building, tree, ground (+ steep terrain when an absolute surface is given)."""
    veg = vegetation_mask(rgb)
    tall = ref_ndsm >= HEIGHT_OBJECT_M
    strata = {
        "building": tall & ~veg,
        "tree": tall & veg,
        "ground": ~tall,
        "low_vegetation": ~tall & veg,
    }
    if ref_surface is not None:
        from scipy import ndimage
        smooth = ndimage.gaussian_filter(np.nan_to_num(ref_surface, nan=np.nanmedian(ref_surface)),
                                         sigma=max(1.0, 15.0 / gsd))
        gy, gx = np.gradient(smooth, gsd)
        strata["steep_terrain"] = np.degrees(np.arctan(np.hypot(gx, gy))) > 8.0
    return strata


def scene_stratum(rgb: np.ndarray, ref_ndsm: np.ndarray, terrain_relief_m: float | None = None) -> str:
    """Coarse landscape label for a whole tile: urban / forested / sparse / hilly / mixed."""
    if terrain_relief_m is not None and terrain_relief_m > 30:
        return "hilly"
    veg = vegetation_mask(rgb)
    tall = ref_ndsm >= HEIGHT_OBJECT_M
    building_frac = float((tall & ~veg).mean())
    tree_frac = float((tall & veg).mean())
    if building_frac >= 0.12:
        return "urban"
    if tree_frac >= 0.20:
        return "forested"
    if (building_frac + tree_frac) < 0.05:
        return "sparse"
    return "mixed"


class RunningMetrics:
    """Accumulates predictions per group (subsampled) and reports metrics per group."""

    def __init__(self, max_pixels_per_sample: int = 200_000, seed: int = 0):
        self.groups: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        self.max_px = max_pixels_per_sample
        self.rng = np.random.default_rng(seed)

    def add(self, group: str, pred: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None):
        valid = np.isfinite(pred) & np.isfinite(ref)
        if mask is not None:
            valid &= mask
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            return
        if idx.size > self.max_px:
            idx = self.rng.choice(idx, self.max_px, replace=False)
        self.groups.setdefault(group, []).append((pred.ravel()[idx].astype(np.float32), ref.ravel()[idx].astype(np.float32)))

    def report(self) -> dict[str, dict]:
        out = {}
        for g, items in sorted(self.groups.items()):
            p = np.concatenate([a for a, _ in items])
            t = np.concatenate([b for _, b in items])
            out[g] = {**height_metrics(p, t), "samples": len(items)}
        return out
