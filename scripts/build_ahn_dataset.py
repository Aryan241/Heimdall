#!/usr/bin/env python3
"""
Build a GAMUS-style training set (RGB + nDSM at 0.33 m, 512² tiles) from open Dutch data:
PDOK aerial orthophotos + AHN LiDAR (nDSM = DSM − gap-filled DTM).

Tiles are sampled at random across the Netherlands' land area, stratified over a coarse
grid, and never within `--exclude-km` of a benchmark site (scripts/benchmark_ahn.py), so
the benchmark stays an honest held-out test.

    python scripts/build_ahn_dataset.py --out data/AHN --train 1500 --val 150
    python scripts/train_decoder.py --dataset-dir data/AHN --resume checkpoints/decoder/decoder_best_all.pth ...

Layout: <out>/images/{train,val}/<id>_RGB.png  and  <out>/heights/{train,val}/<id>_AGL.tif
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from scripts.benchmark_ahn import SITES, WCS, WMS  # noqa: E402

GSD = 0.33
PX = 512
SIZE_M = PX * GSD  # 169 m

# Rough land extent of the Netherlands in RD New (x_min, y_min, x_max, y_max)
EXTENT = (13000, 306000, 278000, 612000)


def _fetch(url: str, retries: int = 3) -> bytes:
    for a in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(2 + 2 * a)
    return b""


def make_tile(x0: float, y0: float, layer: str):
    import rasterio
    from PIL import Image
    from rasterio.io import MemoryFile
    from scipy import ndimage

    x1, y1 = x0 + SIZE_M, y0 + SIZE_M
    rgb = np.asarray(Image.open(io.BytesIO(_fetch(WMS.format(layer=layer, x0=x0, y0=y0, x1=x1, y1=y1, w=PX, h=PX))))
                     .convert("RGB"))
    arrs = []
    for cov in ("dsm_05m", "dtm_05m"):
        with MemoryFile(_fetch(WCS.format(cov=cov, x0=x0, x1=x1, y0=y0, y1=y1))) as mf, mf.open() as s:
            a = s.read(1).astype(np.float32)
            a[(a == s.nodata) | (np.abs(a) > 1e30)] = np.nan
            arrs.append(a)
    dsm, dtm = arrs
    if np.isnan(dsm).mean() > 0.05 or np.isfinite(dtm).mean() < 0.2:
        return None  # sea / no LiDAR coverage
    if (rgb.mean(-1) < 5).mean() > 0.05 or (rgb.mean(-1) > 250).mean() > 0.2:
        return None  # outside orthophoto coverage
    idx = ndimage.distance_transform_edt(np.isnan(dtm), return_distances=False, return_indices=True)
    dtm = ndimage.uniform_filter(dtm[tuple(idx)], size=5)
    ndsm = np.clip(np.nan_to_num(dsm - dtm, nan=0.0), 0, 150)
    ndsm = np.asarray(Image.fromarray(ndsm).resize((PX, PX), Image.Resampling.BILINEAR))
    if np.isnan(ndsm).any():
        return None
    return rgb, ndsm.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "AHN")
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--layer", default="2023_ortho25")
    ap.add_argument("--exclude-km", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import rasterio
    from PIL import Image

    rng = np.random.default_rng(args.seed)
    excl = [(x + 200, y + 200) for x, y, _ in SITES.values()]

    def candidates():
        while True:
            x = rng.uniform(EXTENT[0], EXTENT[2] - SIZE_M)
            y = rng.uniform(EXTENT[1], EXTENT[3] - SIZE_M)
            if all(np.hypot(x - ex, y - ey) > args.exclude_km * 1000 for ex, ey in excl):
                yield round(x), round(y)

    for split, n in (("train", args.train), ("val", args.val)):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "heights" / split).mkdir(parents=True, exist_ok=True)
        have = len(list((args.out / "images" / split).glob("*_RGB.png")))
        gen = candidates()
        with ThreadPoolExecutor(args.workers) as ex:
            while have < n:
                batch = [next(gen) for _ in range(args.workers * 2)]
                for (x, y), res in zip(batch, ex.map(lambda p: _safe(make_tile, p[0], p[1], args.layer), batch)):
                    if res is None or have >= n:
                        continue
                    rgb, ndsm = res
                    tid = f"NL_{x}_{y}"
                    Image.fromarray(rgb).save(args.out / "images" / split / f"{tid}_RGB.png")
                    with rasterio.open(args.out / "heights" / split / f"{tid}_AGL.tif", "w", driver="GTiff",
                                       height=PX, width=PX, count=1, dtype="float32", compress="deflate") as o:
                        o.write(ndsm, 1)
                    have += 1
                print(f"{split}: {have}/{n}", flush=True)
    return 0


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as exc:
        print("skip:", exc, flush=True)
        return None


if __name__ == "__main__":
    sys.exit(main())
