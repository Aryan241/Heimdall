#!/usr/bin/env python3
"""
Heimdall — infer.py
===================
End-to-end CLI: ingest image → run Depth Anything V2 → save depth outputs.

Implements Stage 1 (ingestion/routing) + Stage 2 (relative depth extraction)
as a standalone runnable pipeline.

Usage
-----
    # Single plain image (Mac MPS or CPU):
    python infer.py --input sample.jpg --output-dir outputs/

    # GeoTIFF input:
    python infer.py --input scene.tif --output-dir outputs/

    # Use ViT-Large backbone:
    python infer.py --input sample.jpg --output-dir outputs/ --model vit-l

    # Force CPU:
    python infer.py --input sample.jpg --output-dir outputs/ --device cpu
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Heimdall: Single-view depth estimation (Stage 1+2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Path to input image (PNG/JPG/BMP or GeoTIFF).",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=Path("outputs"),
        help="Directory to write outputs. Created if it doesn't exist.",
    )
    parser.add_argument(
        "--model", "-m", choices=["vit-s", "vit-b", "vit-l"], default="vit-b",
        help="Depth Anything V2 backbone size (default: vit-b).",
    )
    parser.add_argument(
        "--device", "-d", type=str, default=None,
        help="Force device (cuda/mps/cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--tile-size", type=int, default=512,
        help="Tile size for large-image tiling (default: 512).",
    )
    parser.add_argument(
        "--overlap", type=int, default=64,
        help="Overlap between tiles in pixels (default: 64).",
    )
    parser.add_argument(
        "--no-colorize", action="store_true",
        help="Skip saving colorized depth visualization.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    # ── Logging ──────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("heimdall.infer")

    # ── Imports (deferred so --help is instant) ──────────────────────────
    from heimdall.device import get_device
    from heimdall.ingestion.loader import ingest
    from heimdall.depth.depth_anything import predict_depth_tiled
    from heimdall.output.writers import (
        save_depth_png16,
        save_depth_colorized,
        save_depth_geotiff,
    )

    # ── Stage 1: Ingest ──────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("HEIMDALL — Single-View Depth Estimation")
    log.info("═" * 60)

    device = get_device(args.device)
    log.info("Device: %s", device)

    payload = ingest(args.input)
    log.info("Input type: %s  |  Shape: %s", payload.kind, payload.image.shape[:2])

    if payload.kind == "georeferenced":
        log.info("CRS: %s  |  Bounds: %s", payload.geo.crs_epsg, payload.geo.bounds)

    # ── Stage 2: Depth inference ─────────────────────────────────────────
    log.info("Running Depth Anything V2 (%s) …", args.model)
    t0 = time.perf_counter()

    depth = predict_depth_tiled(
        payload.image,
        model_key=args.model,
        tile_size=args.tile_size,
        overlap=args.overlap,
        device=device,
    )

    elapsed = time.perf_counter() - t0
    log.info("Depth inference complete in %.1fs  |  Output shape: %s", elapsed, depth.shape)
    log.info("Depth stats — min: %.3f  max: %.3f  mean: %.3f  std: %.3f",
             depth.min(), depth.max(), depth.mean(), depth.std())

    # ── Stage 7 (partial): Save outputs ──────────────────────────────────
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = args.input.stem

    # Always save 16-bit heightmap
    png16_path = save_depth_png16(depth, out_dir / f"{stem}_depth16.png")
    log.info("✓ 16-bit depth PNG: %s", png16_path)

    # Colorized visualization
    if not args.no_colorize:
        vis_path = save_depth_colorized(depth, out_dir / f"{stem}_depth_vis.png")
        log.info("✓ Colorized depth viz: %s", vis_path)

    # Save raw float32 numpy for downstream stages
    npy_path = out_dir / f"{stem}_depth.npy"
    np.save(npy_path, depth)
    log.info("✓ Raw depth array: %s", npy_path)

    # If georeferenced, also save as GeoTIFF
    if payload.kind == "georeferenced":
        geo_path = save_depth_geotiff(
            depth,
            out_dir / f"{stem}_depth.tif",
            crs_wkt=payload.geo.crs_wkt,
            transform_tuple=payload.geo.transform,
        )
        log.info("✓ GeoTIFF depth: %s", geo_path)

    # ── Summary ──────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("Pipeline complete. Outputs in: %s", out_dir.resolve())
    log.info("═" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
