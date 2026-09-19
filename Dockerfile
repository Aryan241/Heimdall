# Heimdall / DepthWizard — self-contained image (pipeline + web app).
#   docker build -t heimdall .                       # CPU
#   docker build -t heimdall --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 .   # CUDA
#   docker run --rm -p 3000:3000 [--gpus all] -v $PWD/outputs:/app/outputs heimdall

# ── Web build ────────────────────────────────────────────────────────────────
FROM node:20-bookworm-slim AS web
WORKDIR /app/heimdall-web
COPY heimdall-web/package.json heimdall-web/package-lock.json ./
RUN npm ci
COPY heimdall-web/ ./
RUN npm run build

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf \
    HEIMDALL_ROOT=/app HEIMDALL_PYTHON=/usr/local/bin/python HEIMDALL_CACHE=/app/.cache \
    NODE_ENV=production PORT=3000 HOSTNAME=0.0.0.0
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl ca-certificates \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install torch torchvision --index-url ${TORCH_INDEX} && pip install -r requirements.txt
COPY heimdall/ heimdall/
COPY depth_anything_3/ depth_anything_3/
COPY scripts/ scripts/
COPY infer.py validate.py eval.py ./
COPY checkpoints/decoder/decoder_best_all.pth checkpoints/decoder/
# Bake model weights into the image so it runs offline.
RUN python scripts/prefetch_models.py
COPY --from=web /app/heimdall-web/.next/standalone ./heimdall-web/
COPY --from=web /app/heimdall-web/.next/static ./heimdall-web/.next/static
COPY --from=web /app/heimdall-web/public ./heimdall-web/public
EXPOSE 3000
WORKDIR /app/heimdall-web
CMD ["node", "server.js"]
