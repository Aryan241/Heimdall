#!/usr/bin/env python3
"""Score one or more checkpoints on the same GAMUS tiles, so the numbers are comparable."""
import sys, json, argparse
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parent
sys.path.insert(0, "/Users/aryanachary/.gemini/antigravity-ide/scratch/Heimdall")
from torch.utils.data import DataLoader
from heimdall.decoder.dataset import RemoteSensingHeightDataset
from heimdall.decoder.model import load_wrapper
from heimdall.eval.metrics import RunningMetrics

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--weights", nargs="+", required=True)
a = ap.parse_args()

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ds = RemoteSensingHeightDataset(a.data, split=a.split, patch_size=512, augment=False)
print(f"{len(ds)} tiles from {a.data}/{a.split}  ·  device {dev}", flush=True)
out = {}
for w in a.weights:
    model = load_wrapper(w, dev)
    model.eval()
    acc = RunningMetrics(max_pixels_per_sample=50_000)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
    with torch.no_grad():
        for i, b in enumerate(dl):
            pred = model(b["image"].to(dev))["pred_height"].float().cpu().numpy()
            for p, r in zip(pred, b["height"].numpy()):
                r = r[0].copy(); r[(r < -100) | (r > 1000)] = np.nan
                acc.add("m", np.clip(p[0], 0, None), r)
            if i % 20 == 0: print(f"  {Path(w).name}: {i*2}/{len(ds)}", flush=True)
    m = acc.report().get("m", {})
    out[Path(w).name] = m
    print(f"\n{Path(w).name}:  RMSE {m['rmse']:.3f}  MAE {m['mae']:.3f}  "
          f"r {m.get('pearson_r', float('nan')):.3f}  median {m.get('median_abs', float('nan')):.3f}\n", flush=True)
# predict-zero on the identical tiles
acc = RunningMetrics(max_pixels_per_sample=50_000)
for b in DataLoader(ds, batch_size=2, shuffle=False, num_workers=0):
    for r in b["height"].numpy():
        r = r[0].copy(); r[(r < -100) | (r > 1000)] = np.nan
        acc.add("z", np.zeros_like(r), r)
out["predict_zero"] = acc.report().get("z", {})
print(f"predict-zero:  RMSE {out['predict_zero']['rmse']:.3f}")
Path(a.data + "_compare.json").write_text(json.dumps(out, indent=1))
