#!/usr/bin/env bash
# End-to-end retraining on a multi-GPU Linux workstation.
#
#   tmux new -s heimdall            # so it survives the SSH session dropping
#   ./scripts/train_workstation.sh  2>&1 | tee outputs/train_run.log
#
# Stages (each is skippable and resumable — re-running only does what is missing):
#   1. data    GAMUS (recommended by the problem statement) + AHN LiDAR supplement
#   2. cache   frozen-backbone features, computed in parallel across the spare GPUs
#   3. train   the decoder head on the fastest GPU
#   4. bench   accuracy vs GAMUS test and vs airborne LiDAR
#
# Defaults assume: cuda:0 GTX 1080, cuda:1 RTX 3070 (Ampere → mixed precision), cuda:2 GTX 1080.
set -euo pipefail
cd "$(dirname "$0")/.."

# ── Notifications ────────────────────────────────────────────────────────────
# Set HEIMDALL_NOTIFY_URL to get a push when the run finishes or dies:
#   ntfy (no signup):  export HEIMDALL_NOTIFY_URL=https://ntfy.sh/<your-secret-topic>
#   Slack / Discord :  export HEIMDALL_NOTIFY_URL=<incoming-webhook-url>
NOTIFY_URL=${HEIMDALL_NOTIFY_URL:-}
STAGE="starting"
LOG_TAIL=${LOG_TAIL:-outputs/train_run_tail.log}
mkdir -p "$(dirname "$LOG_TAIL")"

notify() {                       # notify "<title>" "<body>"
  [[ -z "$NOTIFY_URL" ]] || {
    local title="$1" body="$2" payload
    case "$NOTIFY_URL" in
      *hooks.slack.com*)  payload=$(printf '{"text":"*%s*\n%s"}' "$title" "$(echo "$body" | sed 's/"/\\"/g; s/$/\\n/' | tr -d '\n')")
                          curl -fsS -m 20 -H 'Content-Type: application/json' -d "$payload" "$NOTIFY_URL" >/dev/null || true ;;
      *discord.com*)      payload=$(printf '{"content":"**%s**\n%s"}' "$title" "$(echo "$body" | sed 's/"/\\"/g; s/$/\\n/' | tr -d '\n')")
                          curl -fsS -m 20 -H 'Content-Type: application/json' -d "$payload" "$NOTIFY_URL" >/dev/null || true ;;
      *)                  curl -fsS -m 20 -H "Title: $title" -d "$body" "$NOTIFY_URL" >/dev/null || true ;;
    esac
  }
}

on_exit() {
  local code=$?
  if (( code == 0 )); then
    notify "Heimdall training finished" "$(tail -n 25 "$LOG_TAIL" 2>/dev/null || echo 'done')"
  else
    notify "Heimdall training FAILED (stage: $STAGE, exit $code)" \
           "$(tail -n 25 "$LOG_TAIL" 2>/dev/null || echo 'no log captured')"
  fi
}
trap on_exit EXIT

PY=${PY:-python}
MODEL=${MODEL:-vit-b}              # vit-b (default) | vit-l | da3-mono-l
TRAIN_GPU=${TRAIN_GPU:-1}          # RTX 3070: Ampere, so AMP actually helps
CACHE_GPU_A=${CACHE_GPU_A:-0}
CACHE_GPU_B=${CACHE_GPU_B:-2}
GAMUS_TILES=${GAMUS_TILES:-0}      # 0 = all 5004 train + 859 val
AHN_TILES=${AHN_TILES:-3000}
EPOCHS=${EPOCHS:-40}
BATCH=${BATCH:-4}                  # 8 GB cards at 512²; raise if memory allows
LR=${LR:-3e-4}
TAG=${TAG:-v2}
STAGES=${STAGES:-data,cache,train,bench}

has() { [[ ",$STAGES," == *",$1,"* ]]; }
say() { STAGE="$*"; printf '\n\033[1;36m▶ %s\033[0m\n' "$*"; echo "▶ $*" >> "$LOG_TAIL"; }
# Keep a short tail of progress for the failure notification.
keep() { tee -a "$LOG_TAIL" | tail -n 2000 > /dev/null; }

say "GPUs"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

if has data; then
  say "1/4 Datasets"
  # GAMUS: the dataset the problem statement recommends. Streamed tile by tile and converted
  # to JPEG + GeoTIFF (~1 MB/tile instead of ~7 MB), so the full set is ~9 GB.
  $PY scripts/fetch_gamus.py --out data/GAMUS --splits train val test --tiles "$GAMUS_TILES" --workers 12
  # AHN: open Dutch aerial + LiDAR, a second geography and landscape mix. Excludes the
  # benchmark sites, so scripts/benchmark_ahn.py stays an honest held-out test.
  $PY scripts/build_ahn_dataset.py --out data/AHN --train "$AHN_TILES" --val 300 --workers 16
fi

if has cache; then
  say "2/4 Caching frozen-backbone features on cuda:$CACHE_GPU_A and cuda:$CACHE_GPU_B"
  # The backbone is frozen, so its output is computed once per tile instead of every epoch.
  # Two datasets, two spare GPUs, in parallel.
  $PY scripts/cache_backbone.py --dataset-dir data/GAMUS --model "$MODEL" --device "cuda:$CACHE_GPU_A" \
      --splits train val &
  pid_a=$!
  $PY scripts/cache_backbone.py --dataset-dir data/AHN --model "$MODEL" --device "cuda:$CACHE_GPU_B" \
      --splits train val &
  pid_b=$!
  wait $pid_a $pid_b
fi

if has train; then
  say "3/4 Training the head on cuda:$TRAIN_GPU ($MODEL, $EPOCHS epochs)"
  $PY scripts/train_decoder.py \
      --dataset-dir data/GAMUS data/AHN \
      --model "$MODEL" --backbone-cache --gpu "$TRAIN_GPU" \
      --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" --num-workers 6 \
      --output-dir checkpoints/decoder --tag "$TAG" 2>&1 | tee -a "$LOG_TAIL"
fi

if has bench; then
  CKPT="checkpoints/decoder/decoder_best_${TAG}.pth"
  say "4/4 Accuracy — GAMUS test split"
  $PY eval.py --weights "$CKPT" --data-dir data/GAMUS --split test \
      --device "cuda:$TRAIN_GPU" --output outputs/gamus_test_${TAG}.json
  say "4/4 Accuracy — airborne LiDAR (held-out Dutch sites)"
  $PY scripts/benchmark_ahn.py --out outputs/benchmark_${TAG}
  cat outputs/benchmark_${TAG}/results.md | tee -a "$LOG_TAIL"
fi

say "Done. Install the new head with:"
echo "  cp checkpoints/decoder/decoder_best_${TAG}.pth checkpoints/decoder/decoder_best.pth"
echo "  git add -f checkpoints/decoder/decoder_best.pth && git commit -m 'Retrained head' && git push"
