#!/usr/bin/env bash
# Deploy Heimdall to a Hugging Face Space (Docker runtime).
#
#   HF_TOKEN=hf_...  ./scripts/deploy_hf_space.sh <username>/<space-name>
#
# Hugging Face Spaces is used rather than Vercel or Netlify because this app is not a static
# site: every job runs the PyTorch pipeline as a subprocess for 10-60 s and needs ~3 GB of
# RAM. Serverless Node hosts cap function bundles at a few hundred megabytes and time out long
# before a job finishes. Spaces runs an arbitrary Dockerfile with 16 GB of RAM on the free CPU
# tier, which fits.
#
# The Space gets a clean checkout: source, the trained head and the bundled scenes, but not
# venv/, data/, outputs/ or the git history, so the push stays small.
set -euo pipefail
cd "$(dirname "$0")/.."

SPACE="${1:-}"
[ -z "$SPACE" ] && { echo "usage: HF_TOKEN=hf_... $0 <username>/<space-name>"; exit 1; }
[ -z "${HF_TOKEN:-}" ] && { echo "HF_TOKEN is not set — create one at https://huggingface.co/settings/tokens (write scope)"; exit 1; }

STAGE="${STAGE:-$(mktemp -d)/space}"
echo "▶ Staging deployment in $STAGE"
mkdir -p "$STAGE"

# Only what the running app needs.
for p in heimdall depth_anything_3 scripts heimdall-web/src heimdall-web/public; do
  mkdir -p "$STAGE/$(dirname "$p")"
  cp -R "$p" "$STAGE/$p"
done
mkdir -p "$STAGE/checkpoints/decoder"
cp checkpoints/decoder/decoder_best.pth "$STAGE/checkpoints/decoder/"
cp requirements.txt infer.py validate.py eval.py Dockerfile "$STAGE/"
cp heimdall-web/package.json heimdall-web/package-lock.json heimdall-web/next.config.ts \
   heimdall-web/tsconfig.json "$STAGE/heimdall-web/"
[ -f heimdall-web/eslint.config.mjs ] && cp heimdall-web/eslint.config.mjs "$STAGE/heimdall-web/"
[ -f heimdall-web/next-env.d.ts ] && cp heimdall-web/next-env.d.ts "$STAGE/heimdall-web/"
find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

# Spaces routes traffic to one port and reads its configuration from the README front matter.
sed -i.bak 's/^ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=\/app\/.hf \\/ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=\/app\/.hf \\/' "$STAGE/Dockerfile"
sed -i.bak 's/PORT=3000/PORT=7860/; s/EXPOSE 3000/EXPOSE 7860/' "$STAGE/Dockerfile"
rm -f "$STAGE/Dockerfile.bak"

cat > "$STAGE/README.md" <<'MD'
---
title: Heimdall — DepthWizard
emoji: 🛰️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Heimdall — Single-View Height Estimation and 3D Flythrough

SIH 2026, Problem Statement 26175 (ISRO / SAC). Upload a single optical remote-sensing image
and get a Digital Surface Model in metres plus a navigable 3D scene.

**Measured accuracy** (held-out, against airborne LiDAR at eight Dutch sites): 5.68 m mean
RMSE, r = 0.66, versus 8.37 m for predicting zero height everywhere. Accuracy holds from
0.25 m to 1 m imagery. Full breakdown in the app's **Accuracy** tab.

Three bundled scenes each ship with their airborne-LiDAR twin, so the reconstruction can be
compared against measured truth in the viewer.

Note: the free CPU tier has no GPU, so a job takes a few minutes rather than seconds.
MD

cd "$STAGE"
git init -q
git lfs install --local 2>/dev/null || true
cat > .gitattributes <<'ATTR'
*.glb filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.tif filter=lfs diff=lfs merge=lfs -text
ATTR
git add -A
git -c user.email=deploy@heimdall -c user.name=Heimdall commit -qm "Deploy Heimdall"
git branch -M main
git remote add space "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE}"
echo "▶ Pushing to https://huggingface.co/spaces/${SPACE}"
git push -q --force space main
echo "▶ Done — the Space will build (10-20 min the first time)."
echo "   Watch it at: https://huggingface.co/spaces/${SPACE}"
