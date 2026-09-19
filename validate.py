#!/usr/bin/env python3
"""
Heimdall — validate.py
======================
Validate a Heimdall DSM against a reference DSM / LiDAR raster (GeoTIFF).

    # Using the metadata written by infer.py (finds the DSM + texture automatically)
    python validate.py --meta outputs/scene_meta.json --reference lidar_dsm.tif

    # Or any two rasters
    python validate.py --pred outputs/scene_dsm.tif --reference lidar_dsm.tif --out-dir outputs/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate a DSM against a reference raster")
    ap.add_argument("--meta", type=Path, help="*_meta.json written by infer.py")
    ap.add_argument("--pred", type=Path, help="Predicted DSM GeoTIFF (if --meta not given)")
    ap.add_argument("--reference", "--ref", dest="reference", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--json-only", action="store_true", help="Print only the JSON report path event.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")

    from heimdall.eval.compare import compare_dsm
    from heimdall.pipeline import emit

    rgb = None
    relative = False
    gsd = None
    if args.meta:
        meta = json.loads(args.meta.read_text())
        base = args.meta.parent
        pred = base / meta["files"]["dsm"]
        relative = meta.get("product") == "relative_dsm"
        gsd = (meta.get("gsd") or {}).get("working_m")
        tex = meta["files"].get("texture")
        if tex and (base / tex).exists():
            from PIL import Image
            rgb = np.asarray(Image.open(base / tex).convert("RGB"))
        prefix = args.prefix or Path(meta["files"]["dsm"]).stem.rsplit("_", 1)[0] + "_validation"
    elif args.pred:
        pred = args.pred
        prefix = args.prefix or args.pred.stem + "_validation"
    else:
        ap.error("--meta or --pred is required")

    out_dir = args.out_dir or Path(pred).parent
    try:
        rep = compare_dsm(pred, args.reference, out_dir, prefix=prefix, rgb=rgb, pred_is_relative=relative, gsd=gsd)
    except Exception as exc:
        logging.exception("Validation failed")
        emit("error", message=f"{type(exc).__name__}: {exc}")
        return 1

    emit("validation", json=rep["files"]["json"], error_png=rep["files"]["error_png"])
    if not args.json_only:
        for key in ("surface_raw", "surface_offset_removed", "above_ground"):
            m = rep[key]
            print(f"{key:<24} RMSE {m.get('rmse', float('nan')):7.3f} m   MAE {m.get('mae', float('nan')):7.3f} m   "
                  f"bias {m.get('bias', float('nan')):+7.3f} m   r {m.get('pearson_r', float('nan')):.3f}")
        print(f"vertical offset (pred − ref, median): {rep['vertical_offset_m']:+.2f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
