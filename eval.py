#!/usr/bin/env python3
"""
Heimdall — eval.py
==================
Benchmark the decoder head on a held-out split of GAMUS (or any images/heights dataset)
at the dataset's *native* resolution, using exactly the tiled inference used in
production (512² tiles, feather-blended).

Reports RMSE, MAE, bias, NMAD, LE90, Pearson r, R² and δ-accuracies:
  * overall
  * per landscape type (urban / forested / sparse / mixed — inferred per tile)
  * per pixel class (building / tree / ground / low vegetation)
  * per city (GAMUS tile-id prefix)
  * against trivial baselines (predict 0 m; predict the training-set mean), so the
    numbers can be read in context.

Usage
-----
    python eval.py --weights checkpoints/decoder/decoder_best_all.pth \
                   --data-dir /kaggle/input/gamus --split test --output results/gamus_test.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Heimdall GAMUS benchmark")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--val-data", "--data-dir", "--dataset-dir", "--val_data", "--data_dir", "--dataset_dir",
                    dest="data_dir", type=Path, required=True)
    ap.add_argument("--split", default="test", help="Split folder name (test / val / train).")
    ap.add_argument("--output", "--output-dir", dest="output", type=Path, default=Path("eval_results.json"))
    ap.add_argument("--model", default=None, help="Backbone key (default: stored in checkpoint).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N tiles.")
    ap.add_argument("--every", type=int, default=1, help="Evaluate every k-th tile (quick runs).")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--refine", action="store_true", help="Apply the planar roof/wall regularisation (as infer.py does).")
    ap.add_argument("--overlap", type=int, default=128)
    ap.add_argument("--save-examples", type=int, default=6, help="Write N qualitative example PNGs.")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("heimdall.eval")

    from heimdall.decoder.dataset import RemoteSensingHeightDataset
    from heimdall.decoder.model import load_wrapper, predict_ndsm
    from heimdall.device import get_device
    from heimdall.eval.metrics import RunningMetrics, height_metrics, pixel_strata, scene_stratum

    device = get_device(args.device)
    ds = RemoteSensingHeightDataset(args.data_dir, split=args.split, patch_size=512)
    if len(ds) == 0:
        log.error("No image/height pairs found for split '%s' in %s", args.split, args.data_dir)
        return 1
    samples = ds.samples[:: max(1, args.every)]
    if args.limit:
        samples = samples[: args.limit]
    log.info("Evaluating %d tiles from %s/%s on %s", len(samples), args.data_dir, args.split, device)

    wrapper = load_wrapper(args.weights, device, model_key=args.model)

    overall = RunningMetrics()
    by_scene = RunningMetrics()
    by_class = RunningMetrics()
    by_city = RunningMetrics()
    baseline_zero = RunningMetrics()
    per_tile = []
    ex_dir = args.output.with_suffix("") if args.output.suffix else args.output
    ex_dir = Path(str(ex_dir) + "_examples")
    t0 = time.time()

    for i, (img_path, hgt_path) in enumerate(samples):
        rgb = ds._load_image(img_path)
        ref = ds._load_height(hgt_path).astype(np.float32)
        if ref.shape != rgb.shape[:2]:
            from heimdall.geo import resize_float
            ref = resize_float(ref, rgb.shape[:2])
        ref[(ref < -100) | (ref > 1000)] = np.nan

        pred = np.clip(predict_ndsm(wrapper, rgb, device, overlap=args.overlap, tta=args.tta), 0, None)
        if args.refine:
            from heimdall.refine import refine_ndsm
            pred, _ = refine_ndsm(pred, rgb, 0.33)

        scene = scene_stratum(rgb, np.nan_to_num(ref))
        city = ds._extract_tile_id(img_path.stem).split("_")[0]
        overall.add("all", pred, ref)
        by_scene.add(scene, pred, ref)
        by_city.add(city, pred, ref)
        baseline_zero.add("zero", np.zeros_like(ref), ref)
        for name, m in pixel_strata(rgb, np.nan_to_num(ref)).items():
            by_class.add(name, pred, ref, m)

        tm = height_metrics(pred, ref)
        per_tile.append({"tile": img_path.name, "scene": scene, "city": city,
                         **{k: tm.get(k) for k in ("rmse", "mae", "bias", "pearson_r")}})

        if i < args.save_examples:
            _save_example(ex_dir, img_path.stem, rgb, ref, pred, tm)
        if (i + 1) % 25 == 0 or i + 1 == len(samples):
            rate = (i + 1) / (time.time() - t0)
            cur = overall.report()["all"]
            log.info("%d/%d tiles (%.2f tiles/s) — running RMSE %.3f m, MAE %.3f m, r %.3f",
                     i + 1, len(samples), rate, cur["rmse"], cur["mae"], cur.get("pearson_r", float("nan")))

    results = {
        "dataset": str(args.data_dir), "split": args.split, "tiles": len(samples),
        "weights": str(args.weights), "backbone": wrapper.model_key, "tta": args.tta, "refine": args.refine,
        "overall": overall.report()["all"],
        "by_landscape": by_scene.report(),
        "by_pixel_class": by_class.report(),
        "by_city": by_city.report(),
        "baselines": {"predict_zero": baseline_zero.report()["zero"]},
        "per_tile": per_tile,
        "seconds": round(time.time() - t0, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.is_dir():
        args.output = args.output / "eval_results.json"
    args.output.write_text(json.dumps(results, indent=2))
    _print_table(results)
    log.info("Saved %s", args.output)
    return 0


def _print_table(res: dict) -> None:
    cols = ("n", "rmse", "mae", "bias", "nmad", "pearson_r", "delta1")
    hdr = f"{'group':<22}" + "".join(f"{c:>11}" for c in cols)
    print("\n" + "=" * len(hdr) + "\n" + hdr + "\n" + "-" * len(hdr))

    def row(name, m):
        cells = []
        for c in cols:
            v = m.get(c)
            cells.append(f"{v:>11,d}" if c == "n" and v is not None else (f"{v:>11.3f}" if v is not None else f"{'—':>11}"))
        print(f"{name:<22}" + "".join(cells))

    row("OVERALL", res["overall"])
    row("baseline: predict 0", res["baselines"]["predict_zero"])
    for section in ("by_landscape", "by_pixel_class", "by_city"):
        print("-" * len(hdr))
        for k, m in res[section].items():
            row(f"{section[3:]}:{k}", m)
    print("=" * len(hdr) + "\n")


def _save_example(out_dir: Path, name: str, rgb, ref, pred, m) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    vmax = float(np.nanpercentile(ref, 99)) or 1.0
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4))
    ax[0].imshow(rgb); ax[0].set_title("RGB")
    ax[1].imshow(ref, cmap="turbo", vmin=0, vmax=vmax); ax[1].set_title("Reference nDSM")
    im = ax[2].imshow(pred, cmap="turbo", vmin=0, vmax=vmax); ax[2].set_title("Heimdall nDSM")
    e = ax[3].imshow(pred - ref, cmap="RdBu_r", vmin=-vmax / 2, vmax=vmax / 2)
    ax[3].set_title(f"Error  RMSE {m.get('rmse', 0):.2f} m  MAE {m.get('mae', 0):.2f} m")
    for a in ax:
        a.axis("off")
    fig.colorbar(im, ax=ax[1:3], fraction=0.02, label="m")
    fig.colorbar(e, ax=ax[3], fraction=0.046, label="m")
    fig.savefig(out_dir / f"{name}.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
