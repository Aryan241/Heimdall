"""
Object-based regularisation of a predicted nDSM (height above ground).

A monocular model predicts every pixel independently, so man-made surfaces come out
bumpy (texture such as solar-panel grids leaks into height) and walls become ramps.
We segment the optical image into colour-homogeneous regions (so a roof including its
blurred edge is one region), split regions whose predicted heights are clearly bimodal,
then:

* built regions (raised, not vegetation) → one robust plane per region
  (flat or evenly pitched roofs; walls snap to the region boundary),
* vegetation → kept (canopies are not planar), lightly smoothed,
* ground → one level per region (terrain relief lives in the DTM, not the nDSM).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _vegetation(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32) + 1e-3
    s = x.sum(-1, keepdims=True)
    r, g, b = np.moveaxis(x / s, -1, 0)
    return (2 * g - r - b) > 0.06


def _robust_plane(yy: np.ndarray, xx: np.ndarray, zz: np.ndarray, iters: int = 3):
    """Least-squares plane z = a·y + b·x + c with iterative trimming of outliers."""
    A = np.column_stack([yy, xx, np.ones_like(yy)])
    keep = np.ones(zz.size, bool)
    coef = np.array([0.0, 0.0, float(np.median(zz))])
    for _ in range(iters):
        if keep.sum() < 6:
            break
        coef, *_ = np.linalg.lstsq(A[keep], zz[keep], rcond=None)
        res = zz - A @ coef
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-3
        keep = np.abs(res) < 2.5 * mad
    res = zz - A @ coef
    nmad = 1.4826 * np.median(np.abs(res - np.median(res)))
    return coef, float(nmad)


def segment(rgb: np.ndarray, gsd: float, min_area_m2: float = 6.0, scale: float = 150.0) -> np.ndarray:
    """Graph-based (Felzenszwalb) segmentation on Lab colour."""
    from skimage.color import rgb2lab
    from skimage.segmentation import felzenszwalb

    lab = rgb2lab(rgb)
    feat = np.dstack([lab[..., 0] / 100.0, lab[..., 1] / 60.0, lab[..., 2] / 60.0])
    min_size = max(8, int(min_area_m2 / (gsd * gsd)))
    return felzenszwalb(feat, scale=scale, sigma=0.6, min_size=min_size, channel_axis=-1)


def split_bimodal(labels: np.ndarray, h: np.ndarray, rgb: np.ndarray, gsd: float, min_gap_m: float = 2.0,
                  min_px: int = 20) -> tuple[np.ndarray, int]:
    """Split colour segments whose predicted heights are clearly bimodal (e.g. a grey roof next to a
    grey road). The cut follows *colour* sub-segments (a finer segmentation), never a contour of the
    blurry predicted height — otherwise the boundary wanders through the middle of a roof."""
    from scipy import ndimage
    from skimage.color import rgb2lab
    from skimage.filters import threshold_otsu
    from skimage.segmentation import felzenszwalb

    out = labels.copy()
    nxt = int(labels.max()) + 1
    splits = 0
    lab = None
    for k, sl in enumerate(ndimage.find_objects(labels + 1)):
        if sl is None:
            continue
        m = labels[sl] == k
        z = h[sl][m]
        if z.size < 2 * min_px or float(np.ptp(z)) < min_gap_m:
            continue
        t = threshold_otsu(z)
        hi = z > t
        if hi.sum() < min_px or (~hi).sum() < min_px or z[hi].mean() - z[~hi].mean() < min_gap_m:
            continue
        if lab is None:
            lab = rgb2lab(rgb)
        feat = np.dstack([lab[sl][..., 0] / 100.0, lab[sl][..., 1] / 60.0, lab[sl][..., 2] / 60.0])
        sub = felzenszwalb(feat, scale=40, sigma=0.5, min_size=max(4, int(1.5 / (gsd * gsd))), channel_axis=-1)
        sub = np.where(m, sub, -1)
        high = np.zeros_like(m)
        for s_id in np.unique(sub[m]):
            sm = sub == s_id
            if np.median(h[sl][sm]) > t:
                high |= sm
        if high.sum() < min_px or (m & ~high).sum() < min_px:
            continue
        comp, n = ndimage.label(high)
        view = out[sl]
        for c in range(1, n + 1):
            view[comp == c] = nxt
            nxt += 1
        splits += 1
    return out, splits


def refine_ndsm(ndsm: np.ndarray, rgb: np.ndarray, gsd: float, object_min_m: float = 1.5,
                max_plane_nmad_m: float = 1.0, max_roof_slope_deg: float = 35.0,
                pitch_gain_m: float = 0.4, merge_tol_m: float = 0.5,
                vehicles: np.ndarray | None = None, vehicle_height_m: float = 1.4) -> tuple[np.ndarray, dict]:
    """Return (refined nDSM, report). *rgb* must be on the same grid as *ndsm*.

    *vehicles* (optional) is a label image of detected vehicles on the same grid. Segments that
    vehicles are parked on are treated as ground, and vehicles become low boxes on that ground.
    """
    from scipy import ndimage

    valid = np.isfinite(ndsm)
    h = np.where(valid, ndsm, 0.0).astype(np.float32)
    labels = _relabel(_absorb_thin(segment(rgb, gsd)))
    labels, n_split = split_bimodal(labels, h, rgb, gsd)
    labels = _relabel(_absorb_thin(_relabel(labels)))  # slivers created by the split become fins otherwise
    veg = _vegetation(rgb)
    smooth = ndimage.gaussian_filter(h, sigma=max(0.5, 0.5 / gsd))
    out = smooth.copy()

    n_seg = int(labels.max()) + 1
    idx = np.arange(n_seg)
    med = ndimage.median(h, labels, idx)
    veg_frac = ndimage.mean(veg.astype(np.float32), labels, idx)
    slices = ndimage.find_objects(labels + 1)  # find_objects numbers labels from 1
    max_grad = np.tan(np.radians(max_roof_slope_deg)) * gsd  # height change per pixel

    planar = kept_veg = n_pitched = 0
    levelled = np.zeros(n_seg, bool)  # built segments that received a single flat level
    ground_votes = _vehicle_ground_votes(labels, vehicles, gsd, n_seg) if vehicles is not None else np.zeros(n_seg, int)
    for k, sl in enumerate(slices):
        if sl is None:
            continue
        if veg_frac[k] > 0.5:
            if med[k] >= object_min_m:
                kept_veg += 1          # canopy: keep the (smoothed) prediction
            continue
        if ground_votes[k] > 0 and med[k] >= object_min_m:
            # Vehicles are parked on it → it is ground, whatever the model says (parking lots and
            # plazas are easily mistaken for flat concrete roofs). Level it with the ground around it.
            out[labels == k] = _surrounding_ground_level(labels == k, h, object_min_m)
            continue
        if med[k] < object_min_m:
            # Ground patch: terrain relief lives in the DTM, so above-ground height is ~constant.
            m = labels[sl] == k
            out[sl][m] = float(med[k])  # not merged: chaining small ground steps would bias the datum
            continue
        m = labels[sl] == k
        yy, xx = np.nonzero(m)
        zz = h[sl][m]
        coef, nmad_plane = _robust_plane(yy.astype(np.float64), xx.astype(np.float64), zz.astype(np.float64))
        level = float(np.median(zz))
        nmad_flat = float(1.4826 * np.median(np.abs(zz - level)))
        a, b, c = coef
        g = np.hypot(a, b)
        # Monocular noise easily fits a spurious tilt, so a roof is flat unless a pitched plane
        # explains it much better (and the pitch is plausible).
        pitched = (g <= max_grad and nmad_plane < 0.5 * nmad_flat and nmad_flat - nmad_plane > pitch_gain_m
                   and nmad_plane <= max_plane_nmad_m)
        if pitched:
            out[sl][m] = a * yy + b * xx + c
            n_pitched += 1
        else:
            out[sl][m] = level
            levelled[k] = True
        planar += 1

    merged = _merge_levels(out, h, labels, levelled, merge_tol_m)

    n_veh = 0
    if vehicles is not None and vehicles.any():
        # Vehicles: low boxes on the surrounding ground level.
        ring = ndimage.binary_dilation(vehicles > 0, iterations=max(1, int(round(1.0 / gsd)))) & (vehicles == 0)
        for v, sl in enumerate(ndimage.find_objects(vehicles), start=1):
            if sl is None:
                continue
            pad = tuple(slice(max(0, x.start - 8), x.stop + 8) for x in sl)
            vm = vehicles[pad] == v
            rm = ring[pad] & ndimage.binary_dilation(vm, iterations=max(1, int(round(1.0 / gsd))))
            if rm.any():
                out[pad][vm] = float(np.median(out[pad][rm])) + vehicle_height_m
                n_veh += 1

    out = np.clip(out, 0, None)
    out[~valid] = np.nan
    rep = {"segments": n_seg, "built_surfaces": planar, "pitched": n_pitched, "vegetation_objects": kept_veg,
           "bimodal_splits": n_split, "merged_pairs": merged, "vehicles": n_veh,
           "segments_grounded_by_vehicles": int((ground_votes > 0).sum())}
    logger.info("Planar refinement: %d segments, %d built surfaces regularised, %d vegetation kept",
                n_seg, planar, kept_veg)
    return out.astype(np.float32), rep


def _merge_levels(out: np.ndarray, h: np.ndarray, labels: np.ndarray, levelled: np.ndarray, tol: float) -> int:
    """Merge adjacent flat segments whose levels differ by < tol (union-find), re-levelling each group.

    Over-segmentation (e.g. a solar array split in two) otherwise leaves small steps inside one roof.
    """
    from scipy import ndimage

    n = labels.max() + 1
    level = np.full(n, np.nan)
    lv = ndimage.median(out, labels, np.arange(n))
    level[levelled] = np.asarray(lv)[levelled]

    pairs = set()
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        d = a != b
        for i, j in zip(a[d].ravel(), b[d].ravel()):
            pairs.add((min(i, j), max(i, j)))

    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merged = 0
    for i, j in sorted(pairs, key=lambda p: abs(level[p[0]] - level[p[1]]) if levelled[p[0]] and levelled[p[1]] else 1e9):
        if not (levelled[i] and levelled[j]) or abs(level[i] - level[j]) >= tol:
            continue
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
            merged += 1
    if merged:
        roots = np.array([find(k) for k in range(n)])
        group = roots[labels]
        mask = levelled[labels]
        gl = ndimage.median(h, np.where(mask, group, -1), np.unique(group[mask]))
        lut = dict(zip(np.unique(group[mask]).tolist(), np.atleast_1d(gl).tolist()))
        vals = np.vectorize(lambda g: lut.get(g, np.nan))(group[mask])
        out[mask] = vals
    return merged


def _relabel(labels: np.ndarray) -> np.ndarray:
    _, inv = np.unique(labels, return_inverse=True)
    return inv.reshape(labels.shape)


def _absorb_thin(labels: np.ndarray, min_core_frac: float = 0.35) -> np.ndarray:
    """Reassign pixels of thin segments (mixed-colour strips along edges) to the nearest thick segment.

    Such strips sit on the blurred wall ramp; left alone they become spurious ledges.
    """
    from scipy import ndimage

    thin = np.zeros(labels.shape, bool)
    for k, sl in enumerate(ndimage.find_objects(labels + 1)):
        if sl is None:
            continue
        m = labels[sl] == k
        core = ndimage.binary_erosion(m, iterations=2)
        if core.sum() < min_core_frac * m.sum():
            thin[sl] |= m
    if not thin.any() or thin.all():
        return labels
    idx = ndimage.distance_transform_edt(thin, return_distances=False, return_indices=True)
    return labels[tuple(idx)]


def _vehicle_ground_votes(labels: np.ndarray, vehicles: np.ndarray, gsd: float, n_seg: int) -> np.ndarray:
    """For each segment, the number of vehicles whose immediate surroundings (a ~1 m ring) mostly
    fall inside it — i.e. vehicles parked on that surface."""
    from scipy import ndimage

    votes = np.zeros(n_seg, int)
    it = max(1, int(round(1.0 / gsd)))
    for v, sl in enumerate(ndimage.find_objects(vehicles), start=1):
        if sl is None:
            continue
        pad = tuple(slice(max(0, x.start - it - 1), x.stop + it + 1) for x in sl)
        vm = vehicles[pad] == v
        ring = ndimage.binary_dilation(vm, iterations=it) & (vehicles[pad] == 0)
        if not ring.any():
            continue
        segs, counts = np.unique(labels[pad][ring], return_counts=True)
        best = counts.argmax()
        if counts[best] >= 0.4 * ring.sum():
            votes[segs[best]] += 1
    return votes


def _surrounding_ground_level(mask: np.ndarray, h: np.ndarray, ground_max_m: float, ring_px: int = 6) -> float:
    """Median predicted height of ground-like pixels (< ground_max_m) around *mask*; 0 if none."""
    from scipy import ndimage
    ring = ndimage.binary_dilation(mask, iterations=ring_px) & ~mask
    z = h[ring]
    z = z[z < ground_max_m]
    return float(np.median(z)) if z.size >= 10 else 0.0
