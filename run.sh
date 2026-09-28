#!/usr/bin/env bash
# One-command launcher: sets up the Python env + web app (first run) and serves the app.
#   ./run.sh            → http://localhost:3000
#   PORT=8080 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-3000}"

step() { printf '\n\033[1;36m▶ %s\033[0m  (%s)\n' "$1" "$(date +%H:%M:%S)"; }

# PyTorch from PyPI defaults to the CUDA build on Linux: the wheel plus its nvidia-* runtime
# packages come to roughly 3.5 GB, a 50-minute first run on a normal connection — and most of
# it is wasted on a machine with no NVIDIA GPU. So pick the build that matches the hardware:
# CUDA where an NVIDIA GPU is actually present, CPU otherwise. Override with TORCH_INDEX.
if [ -z "${TORCH_INDEX:-}" ]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    TORCH_KIND="CUDA (NVIDIA GPU detected) — about 3.5 GB"
  else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_KIND="CPU (no NVIDIA GPU found) — about 200 MB"
  fi
else
  TORCH_KIND="from ${TORCH_INDEX##*/} (TORCH_INDEX override)"
fi

if [ ! -x venv/bin/python ]; then
  echo "First run: this sets everything up and takes roughly 5-10 minutes."
  step "Creating the Python virtualenv"
  python3 -m venv venv
  venv/bin/pip install -q --upgrade pip
  step "Installing PyTorch: $TORCH_KIND"
  venv/bin/pip install torch torchvision --index-url "$TORCH_INDEX"
  step "Installing the remaining Python dependencies (~10 packages)"
  venv/bin/pip install -r requirements.txt
  step "Downloading model weights (~450 MB, cached afterwards)"
  venv/bin/python scripts/prefetch_models.py
  step "Python environment ready"
fi

cd heimdall-web
if [ ! -d node_modules ]; then
  step "Installing web dependencies (~1 minute)"
  npm ci
fi
if [ ! -f .next/BUILD_ID ] || [ -n "${REBUILD:-}" ]; then
  step "Building the web app (~1 minute)"
  npm run build
fi

export HEIMDALL_ROOT="$(cd .. && pwd)"
export HEIMDALL_PYTHON="$HEIMDALL_ROOT/venv/bin/python"

# `output: standalone` produces a self-contained server bundle, but Next deliberately leaves
# the static assets out of it — they must be staged alongside, or every script, stylesheet and
# demo scene 404s and the 3D viewer renders nothing. `next start` does not work with this
# output mode at all (it warns and misbehaves), so run the bundle directly.
if [ -f .next/standalone/server.js ]; then
  mkdir -p .next/standalone/.next
  rm -rf .next/standalone/.next/static .next/standalone/public
  cp -R .next/static .next/standalone/.next/static
  cp -R public .next/standalone/public
  echo "▶ Heimdall running at http://localhost:$PORT"
  ( sleep 3; command -v open >/dev/null && open "http://localhost:$PORT" || true ) &
  exec env PORT="$PORT" HOSTNAME=0.0.0.0 node .next/standalone/server.js
fi

echo "▶ Heimdall running at http://localhost:$PORT"
( sleep 3; command -v open >/dev/null && open "http://localhost:$PORT" || true ) &
exec npx next start -p "$PORT"
