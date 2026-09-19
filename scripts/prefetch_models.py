#!/usr/bin/env python3
"""Download every model the pipeline needs so it can run fully offline afterwards.

    python scripts/prefetch_models.py            # DA3-Metric-Large + YOLOv8n
    python scripts/prefetch_models.py --all      # also Depth Anything V2 and SegFormer
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Also fetch optional models (DA2, SegFormer).")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download

    repos = ["depth-anything/DA3METRIC-LARGE"]
    if args.all:
        repos += ["depth-anything/Depth-Anything-V2-Base-hf", "nvidia/segformer-b0-finetuned-ade-512-512"]
    for r in repos:
        print(f"→ {r}")
        snapshot_download(r)

    import urllib.request
    for name, url in [("yolov8n.pt", "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"),
                      ("yolov8n-obb.pt", "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-obb.pt")]:
        if not (REPO / name).exists():
            print(f"→ {name}")
            urllib.request.urlretrieve(url, REPO / name)

    head = REPO / "checkpoints" / "decoder" / "decoder_best_all.pth"
    print(f"Decoder head: {'OK' if head.exists() else 'MISSING — relative mode only'} ({head})")
    print("Done. Set HF_HUB_OFFLINE=1 to run without network access.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
