#!/usr/bin/env bash
# Install a freshly trained head and produce every number and artefact that depends on it.
#
#   ./scripts/finalise_model.sh ~/Downloads/decoder_best_kaggle.pth
#
# Steps (each is skippable with the flags below):
#   1. install   copy the checkpoint to checkpoints/decoder/decoder_best.pth (picked up automatically)
#   2. tests     unit tests, so a broken checkpoint can't silently ship
#   3. bench     LiDAR benchmark over 8 held-out landscapes -> outputs/benchmark_ahn/results.md
#   4. demo      regenerate the bundled web demo scene with the new model
#   5. report    print the table to paste into PROJECT_REPORT.md section 4.4
set -uo pipefail
cd "$(dirname "$0")/.."

CKPT=${1:-}
PY=${PY:-./venv/bin/python}
SKIP=${SKIP:-}
[ -z "$CKPT" ] && { echo "usage: $0 <trained-checkpoint.pth> [SKIP=bench,demo]"; exit 1; }
[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT"; exit 1; }
has() { [[ ",$SKIP," != *",$1,"* ]]; }
say() { printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; }

say "1/5 Installing $CKPT"
mkdir -p checkpoints/decoder
cp "$CKPT" checkpoints/decoder/decoder_best.pth
$PY - <<'EOF'
import torch
c = torch.load("checkpoints/decoder/decoder_best.pth", map_location="cpu", weights_only=False)
val = c.get("val") or {}
print(f"   backbone : {c.get('model_key')}")
print(f"   head     : {c.get('head', 'affine (legacy)')}")
print(f"   epoch    : {c.get('epoch')}")
print(f"   val RMSE : {val.get('rmse', float('nan')):.3f} m   MAE {val.get('mae', float('nan')):.3f} m"
      f"   r {val.get('pearson_r', float('nan')):.3f}")
EOF

if has tests; then
  say "2/5 Unit tests"
  $PY -m pytest -q tests -m "not slow" || { echo "tests failed — stopping"; exit 1; }
fi

if has bench; then
  say "3/5 LiDAR benchmark (8 held-out landscapes, ~25 min)"
  $PY scripts/benchmark_ahn.py --out outputs/benchmark_ahn 2>&1 | tail -20
fi

if has demo; then
  say "4/5 Regenerating the web demo scene"
  $PY infer.py -i data/uploaded_image_1789657102996.tif -o heimdall-web/public/demo \
      --name demo --tta --quiet-events 2>&1 | tail -4
fi

say "5/5 Numbers for PROJECT_REPORT.md section 4.4"
[ -f outputs/benchmark_ahn/results.md ] && cat outputs/benchmark_ahn/results.md
[ -f outputs/gamus_test.json ] && $PY - <<'EOF'
import json
from pathlib import Path
r = json.loads(Path("outputs/gamus_test.json").read_text())
o, z = r["overall"], r["baselines"]["predict_zero"]
print(f"\nGAMUS test ({r['tiles']} tiles): RMSE {o['rmse']:.2f} m  MAE {o['mae']:.2f} m  "
      f"bias {o['bias']:+.2f} m  r {o.get('pearson_r', float('nan')):.3f}   "
      f"[predict-zero baseline RMSE {z['rmse']:.2f} m]")
EOF

say "Done. Commit with:"
echo "  git add -f checkpoints/decoder/decoder_best.pth heimdall-web/public/demo outputs/benchmark_ahn/results.md"
echo "  git commit -m 'Trained head: <numbers>' && git push"
