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
echo "▶ Heimdall running at http://localhost:$PORT"
( sleep 3; command -v open >/dev/null && open "http://localhost:$PORT" || true ) &
exec npx next start -p "$PORT"
