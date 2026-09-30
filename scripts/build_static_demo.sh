#!/usr/bin/env bash
# Build the read-only public demo: the viewer, the bundled LiDAR-comparison scenes and the
# Accuracy tab, exported as plain files that any static host will serve.
#
#   ./scripts/build_static_demo.sh                       # build only  -> outputs/static_demo
#   HF_TOKEN=hf_... ./scripts/build_static_demo.sh yann-n/heimdall    # build and deploy
#
# Processing an upload runs the PyTorch pipeline server-side, which a static host cannot do,
# so the API routes are dropped and the upload panel is replaced with a note (NEXT_PUBLIC_
# STATIC_DEMO). Everything that runs in the browser — the 3D flythrough, layers, measurement,
# the ground-truth comparison and the measured results — works exactly as it does locally.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

SPACE="${1:-}"
STAGE="${STAGE:-$(mktemp -d)/static}"
OUT="$REPO/outputs/static_demo"

echo "▶ Staging the web app in $STAGE"
mkdir -p "$STAGE"
cp -R heimdall-web/. "$STAGE/"
rm -rf "$STAGE/.next" "$STAGE/node_modules" "$STAGE/out"

# The public demo is bandwidth-bound: each scene pair is ~25 MB of mesh and texture, so it
# ships a subset while a local install keeps them all. Override with STATIC_SCENES.
STATIC_SCENES="${STATIC_SCENES:-utrecht_suburb veluwe_forest gamus_philadelphia}"
python3 "$REPO/scripts/_trim_static_scenes.py" "$STAGE" $STATIC_SCENES
# `output: export` cannot build dynamic route handlers, and they are useless without a backend.
rm -rf "$STAGE/src/app/api"

cat > "$STAGE/next.config.ts" <<'CFG'
import type { NextConfig } from "next";
import path from "path";

// Static export for the public demo: no server, no API routes, plain files.
const nextConfig: NextConfig = {
  turbopack: { root: path.join(__dirname) },
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;
CFG

cd "$STAGE"
echo "▶ Installing dependencies"
npm ci --silent
echo "▶ Building"
NEXT_PUBLIC_STATIC_DEMO=1 npm run build

rm -rf "$OUT"
mkdir -p "$(dirname "$OUT")"
cp -R "$STAGE/out" "$OUT"
echo "▶ Static demo built: $OUT  ($(du -sh "$OUT" | cut -f1))"

[ -z "$SPACE" ] && { echo "   Preview it with:  npx serve $OUT"; exit 0; }
[ -z "${HF_TOKEN:-}" ] && { echo "HF_TOKEN is not set — needed to deploy"; exit 1; }

echo "▶ Deploying to Hugging Face Space $SPACE (static SDK)"
python - "$SPACE" <<'PYEOF'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(sys.argv[1], repo_type="space", space_sdk="static", exist_ok=True)
print("   space ready")
PYEOF

DEPLOY="$(mktemp -d)/space"
mkdir -p "$DEPLOY"
cp -R "$OUT/." "$DEPLOY/"
cat > "$DEPLOY/README.md" <<'MD'
---
title: Heimdall — DepthWizard
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: mit
---

# Heimdall — Single-View Height Estimation and 3D Flythrough

SIH 2026, Problem Statement 26175 (ISRO / SAC).

Each scene ships with its airborne-LiDAR twin, so the reconstruction can be compared
against measured truth in the viewer. The **Accuracy** tab carries every measured number:
5.32 m mean RMSE and r = 0.67 against LiDAR across eight held-out Dutch landscapes, versus
8.37 m for predicting zero height everywhere, and stability from 0.25 m to 1 m imagery.

This deployment is read-only — processing an upload runs a PyTorch pipeline server-side.
Run it locally for the full application: https://github.com/Aryan241/Heimdall
MD

cd "$DEPLOY"
git init -q
git lfs install --local >/dev/null 2>&1 || true
# Hugging Face rejects a push that carries sizeable binaries outside LFS.
cat > .gitattributes <<'ATTR'
*.glb filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.jpg filter=lfs diff=lfs merge=lfs -text
*.jpeg filter=lfs diff=lfs merge=lfs -text
*.tif filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
ATTR
git add -A
git -c user.email=deploy@heimdall -c user.name=Heimdall commit -qm "Deploy Heimdall static demo"
git branch -M main
git remote add space "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE}"
git push -q --force space main
echo "▶ Live at https://huggingface.co/spaces/${SPACE}"
