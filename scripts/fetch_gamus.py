#!/usr/bin/env python3
"""
Fetch the GAMUS benchmark (the dataset recommended by the problem statement) from Hugging
Face and store it compactly.

GAMUS ships one HDF5 file per tile per modality: 5 004 train / 859 val / 2 861 test tiles,
~7 MB per tile for image+height — about 37 GB for the train split alone, which does not fit
in a Kaggle working directory. This script downloads tile by tile, converts to
JPEG + float32 GeoTIFF (+ PNG class labels), deletes the HDF5 immediately, and can resume.
That brings a tile down to roughly 1 MB, so the whole dataset fits in ~9 GB.

    python scripts/fetch_gamus.py --out data/GAMUS                       # everything
    HF_TOKEN=hf_...  python scripts/fetch_gamus.py --out data/GAMUS      # higher rate limits
    python scripts/fetch_gamus.py --out data/GAMUS --tiles 2000 --splits train val
    python scripts/fetch_gamus.py --out data/GAMUS --splits test --tiles 400   # for eval.py

Layout (what heimdall's dataset loader and eval.py expect):
    <out>/images/<split>/<tile>_RGB.jpg
    <out>/heights/<split>/<tile>_AGL.tif      metres above ground
    <out>/classes/<split>/<tile>_CLS.png      0 other, 1 ground, 2 low veg, 3 building,
                                              4 water, 5 road, 6 tree
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

DATASET = "earthflow/GAMUS"
MODALITIES = {"images": "RGB", "heights": "AGL", "classes": "CLS"}
# GAMUS is not consistent: DC and PHL name their imagery <tile>_RGB.h5, NYC uses <tile>_IMG.h5
# (2167 of 8724 tiles). Heights and labels are always _AGL / _CLS.
IMAGE_SUFFIXES = ("RGB", "IMG")


def _download(kind: str, split: str, tile: str, suffix: str, tmp: Path, tries: int = 6) -> Path:
    """hf_hub_download with backoff. Hugging Face rate-limits anonymous clients (HTTP 429);
    without this, a parallel run silently loses a large fraction of the dataset."""
    from huggingface_hub import hf_hub_download
    try:  # the error class moved between huggingface_hub versions
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:  # older releases
        from huggingface_hub.utils import HfHubHTTPError

    last: Exception | None = None
    for attempt in range(tries):
        try:
            return Path(hf_hub_download(DATASET, f"{kind}/{split}/{tile}_{suffix}.h5",
                                        repo_type="dataset", local_dir=str(tmp),
                                        token=os.environ.get("HF_TOKEN") or None))
        except HfHubHTTPError as exc:
            last = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status not in (429, 500, 502, 503, 504):
                raise
            wait = float(getattr(exc, "response", None).headers.get("Retry-After", 0) or 0) \
                if getattr(exc, "response", None) is not None else 0
            time.sleep(max(wait, min(60, 4 * 2 ** attempt)) + random.uniform(0, 2))
    raise last if last else RuntimeError("download failed")


def _read_h5(path: Path) -> np.ndarray:
    import h5py
    with h5py.File(path, "r") as f:
        return np.array(f[list(f.keys())[0]])


def convert_tile(tile: str, split: str, out: Path, want_classes: bool, quality: int,
                 image_suffix: str = "RGB") -> str | None:
    """Download one tile's HDF5 files, write compact outputs, delete the originals."""
    import rasterio
    from huggingface_hub import hf_hub_download
    from PIL import Image

    img_path = out / "images" / split / f"{tile}_RGB.jpg"
    hgt_path = out / "heights" / split / f"{tile}_AGL.tif"
    if img_path.exists() and hgt_path.exists() and hgt_path.stat().st_size > 0:
        return None  # already converted

    tmp = out / ".hf"
    downloaded: list[Path] = []
    written: list[Path] = []
    try:
        # Image and height are required; class labels are optional (a few tiles lack them).
        for kind in ("images", "heights"):
            suffix = image_suffix if kind == "images" else MODALITIES[kind]
            local = _download(kind, split, tile, suffix, tmp)
            downloaded.append(local)
            arr = _read_h5(local)
            if kind == "images":
                Image.fromarray(arr.astype(np.uint8)).save(img_path, quality=quality)
                written.append(img_path)
            else:
                a = np.nan_to_num(arr.astype(np.float32), nan=0.0)
                with rasterio.open(hgt_path, "w", driver="GTiff", height=a.shape[0], width=a.shape[1],
                                   count=1, dtype="float32", compress="deflate", predictor=3) as d:
                    d.write(a, 1)
                written.append(hgt_path)
        if want_classes:
            try:
                local = _download("classes", split, tile, MODALITIES["classes"], tmp)
                downloaded.append(local)
                Image.fromarray(np.clip(_read_h5(local), 0, 255).astype(np.uint8)).save(
                    out / "classes" / split / f"{tile}_CLS.png")
            except Exception as exc:
                print(f"note {tile}: no class labels ({type(exc).__name__})", flush=True)
        return tile
    except Exception as exc:
        # Never leave a half-written tile behind: the pair would be unusable and the resume
        # check would consider it done.
        for w in written:
            w.unlink(missing_ok=True)
        print(f"skip {tile}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        return None
    finally:
        for p in downloaded:
            p.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "data" / "GAMUS")
    ap.add_argument("--splits", nargs="*", default=["train", "val"], choices=["train", "val", "test"])
    ap.add_argument("--tiles", type=int, default=0, help="Max tiles per split (0 = all).")
    ap.add_argument("--no-classes", action="store_true", help="Skip semantic labels (saves ~20%%).")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel downloads. Hugging Face rate-limits anonymous clients; >6 causes HTTP 429.")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality for the imagery.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(DATASET, repo_type="dataset")
    rng = np.random.default_rng(args.seed)
    total = 0
    for split in args.splits:
        tiles = []
        for f in files:
            if not f.startswith(f"images/{split}/"):
                continue
            name = f.split("/")[-1].removesuffix(".h5")
            for suf in IMAGE_SUFFIXES:
                if name.endswith("_" + suf):
                    tiles.append((name.removesuffix("_" + suf), suf))
                    break
        tiles.sort()
        if args.tiles and len(tiles) > args.tiles:
            # Random (but reproducible) subset, so a partial download still covers every city.
            tiles = [tiles[i] for i in sorted(rng.choice(len(tiles), args.tiles, replace=False))]
        for kind in ("images", "heights") + (() if args.no_classes else ("classes",)):
            (args.out / kind / split).mkdir(parents=True, exist_ok=True)

        done = 0
        with ThreadPoolExecutor(args.workers) as ex:
            futures = [ex.submit(convert_tile, t, split, args.out, not args.no_classes, args.quality, suf)
                       for t, suf in tiles]
            for i, fut in enumerate(futures, 1):
                fut.result()
                if i % 50 == 0 or i == len(futures):
                    print(f"{split}: {i}/{len(tiles)}", flush=True)
            done = len(tiles)
        total += done
        shutil.rmtree(args.out / ".hf", ignore_errors=True)

    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file()) / 1e9
    print(f"\nGAMUS ready: {total} tiles across {args.splits} → {args.out}  ({size:.1f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
