#!/usr/bin/env bash
# One-command launcher: sets up the Python env + web app (first run) and serves the app.
#   ./run.sh            → http://localhost:3000
#   PORT=8080 ./run.sh
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-3000}"

if [ ! -x venv/bin/python ]; then
  echo "▶ Creating Python virtualenv"
  python3 -m venv venv
  venv/bin/pip install --upgrade pip
  venv/bin/pip install -r requirements.txt
  venv/bin/python scripts/prefetch_models.py
fi

cd heimdall-web
if [ ! -d node_modules ]; then
  echo "▶ Installing web dependencies"
  npm ci
fi
if [ ! -f .next/BUILD_ID ] || [ -n "${REBUILD:-}" ]; then
  echo "▶ Building web app"
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
