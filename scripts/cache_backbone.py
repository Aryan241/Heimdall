#!/usr/bin/env python3
"""
Pre-compute frozen-backbone outputs for a training set, so head training does not re-run
the backbone every epoch (it is frozen, so its output never changes).

For each tile two variants are cached:
  * "sharp"     — the tile as-is,
  * "degraded"  — down-sampled to a random satellite-like GSD and back, so the head also
                  learns the blur/detail level of coarser sensors.

    python scripts/cache_backbone.py --dataset-dir data/AHN --model vit-b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def degrade(rgb: np.ndarray, factor: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    small = Image.fromarray(rgb).resize((max(8, int(w / factor)), max(8, int(h / factor))), Image.Resampling.BOX)
    return np.asarray(small.resize((w, h), Image.Resampling.BILINEAR))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, nargs="+", required=True)
    ap.add_argument("--cache-root", type=Path, default=None,
                    help="Write caches under this folder instead of beside the dataset (read-only inputs).")
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--model", default="vit-b")
    ap.add_argument("--device", default=None)
    ap.add_argument("--degrade-range", nargs=2, type=float, default=[1.5, 3.0],
                    help="Random down-sampling factors for the degraded variant (0.33 m → 0.5–1.0 m).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from heimdall.decoder.dataset import RemoteSensingHeightDataset
    from heimdall.depth.depth_anything import predict_depth
    from heimdall.device import get_device

    device = get_device(args.device)
    rng = np.random.default_rng(args.seed)
    for dataset_dir, split in [(d, s) for d in args.dataset_dir for s in args.splits]:
        ds = RemoteSensingHeightDataset(dataset_dir, split=split, patch_size=512, augment=False)
        root = (args.cache_root / dataset_dir.name) if args.cache_root else dataset_dir
        out_dir = root / "backbone" / args.model / split
        out_dir.mkdir(parents=True, exist_ok=True)
        todo = [(p, out_dir / f"{p.stem}.npz") for p, _ in ds.samples]
        todo = [(p, o) for p, o in todo if not o.exists()]
        print(f"{dataset_dir.name}/{split}: {len(todo)} tiles to cache ({len(ds.samples)} total)", flush=True)
        for img_path, out in tqdm(todo, file=sys.stdout):
            rgb = ds._load_image(img_path)
            with torch.no_grad():
                sharp = predict_depth(rgb, model_key=args.model, device=device)
                f = float(rng.uniform(*args.degrade_range))
                deg = predict_depth(degrade(rgb, f), model_key=args.model, device=device)
            np.savez_compressed(out, sharp=sharp.astype(np.float16), degraded=deg.astype(np.float16), factor=f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
