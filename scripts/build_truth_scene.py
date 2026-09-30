#!/usr/bin/env python3
"""Build a flythrough scene from a reference height raster, matching a finished job's grid.

    python scripts/build_truth_scene.py --meta outputs/jobs/<id>/result_meta.json \
        --reference truth_heights.tif --out-dir outputs/jobs/<id> --stem truth_abc123

The result is a self-contained scene the web viewer loads exactly like the model's own: same
grid, same texture, same vertical origin, so the two can be flown through and compared directly.
Only the heights differ. Costs about 0.6 s on top of a job — no model inference is involved.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", type=Path, required=True, help="The finished job's *_meta.json.")
    ap.add_argument("--reference", type=Path, required=True, help="Reference height raster (above ground).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stem", required=True, help="Filename stem for the scene's files.")
    a = ap.parse_args()

    from scripts.build_demo_site import build_truth_scene
    info = build_truth_scene(a.meta.parent, a.out_dir, a.reference, a.stem, meta_path=a.meta)
    files = info.get("files", info) if isinstance(info, dict) else {}
    print("@@HEIMDALL " + json.dumps({"event": "truth_scene", "meta": f"{a.stem}_meta.json"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
