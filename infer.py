#!/usr/bin/env python3
"""
Heimdall — infer.py
===================
Single-view optical image → DSM (GeoTIFF) + textured 3D mesh (GLB).

Examples
--------
    # GeoTIFF → absolute DSM (terrain from Copernicus GLO-30 is fetched automatically)
    python infer.py --input scene.tif --output-dir outputs/

    # GeoTIFF with your own SRTM / CartoDEM tile and a few GCPs
    python infer.py --input scene.tif --reference-dem srtm.tif --gcps gcps.csv

    # Plain JPG/PNG → relative DSM (metres above ground if the GSD is known)
    python infer.py --input image.jpg --gsd 0.5

    # Backbone only (no trained head): relative height, RANSAC-calibrated if a DEM is given
    python infer.py --input scene.tif --weights none
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    from heimdall.pipeline import DEFAULT_WEIGHTS
    p = argparse.ArgumentParser(description="Heimdall: single-view height estimation",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--input", "-i", required=True, type=Path, help="PNG/JPG/TIFF or GeoTIFF image.")
    p.add_argument("--output-dir", "-o", type=Path, default=Path("outputs"))
    p.add_argument("--name", default=None, help="Output file prefix (default: input file stem).")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                   help="Decoder head checkpoint (.pth), or 'none' for backbone-only relative mode.")
    p.add_argument("--model", "-m", default=None,
                   choices=["vit-s", "vit-b", "vit-l", "da3-metric-l", "da3-mono-l"],
                   help="Backbone (default: the one stored in the checkpoint, else vit-b).")
    p.add_argument("--device", "-d", default=None, help="cuda / mps / cpu (auto if omitted).")
    p.add_argument("--reference-dem", type=Path, default=None, help="Low-res DEM GeoTIFF (SRTM, Copernicus, CartoDEM…).")
    p.add_argument("--no-auto-dem", action="store_true", help="Don't download Copernicus GLO-30 automatically.")
    p.add_argument("--gcps", type=Path, default=None, help="CSV of ground control points (lon,lat,z | x,y,z | row,col,z).")
    p.add_argument("--gsd", type=float, default=None, help="Ground sample distance override (m/px).")
    p.add_argument("--target-gsd", type=float, default=None, help="Working GSD (default 0.33 m = training GSD).")
    p.add_argument("--max-side", type=int, default=8192, help="Cap on the working grid's longest side.")
    p.add_argument("--bands", default=None, help="1-based band order for R,G,B, e.g. '3,2,1' for BGR products.")
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--tta", action="store_true", help="Flip test-time augmentation (≈2× slower, slightly better).")
    p.add_argument("--no-refine", action="store_true",
                   help="Skip planar roof/wall regularisation (keep the raw per-pixel prediction).")
    p.add_argument("--no-mesh", action="store_true", help="Skip GLB mesh export.")
    p.add_argument("--export-mesh", action="store_true", help=argparse.SUPPRESS)  # legacy flag (mesh is default)
    p.add_argument("--mesh-grid", type=int, default=640, help="Max mesh vertices per side.")
    p.add_argument("--max-height", type=float, default=None, help="Relative mode: scale 0–1 output to metres.")
    p.add_argument("--use-segmentation", action="store_true", help="Relative mode: ground mask for RANSAC.")
    p.add_argument("--quiet-events", action="store_true", help="Don't print @@HEIMDALL progress events.")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-26s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    from heimdall.pipeline import PipelineConfig, emit, run

    weights = None if str(args.weights).lower() in ("none", "", "off") else Path(args.weights)
    cfg = PipelineConfig(
        input=args.input,
        output_dir=args.output_dir,
        name=args.name,
        weights=weights,
        model_key=args.model,
        device=args.device,
        reference_dem=args.reference_dem,
        auto_dem=not args.no_auto_dem,
        gcps=args.gcps,
        gsd=args.gsd,
        target_gsd=args.target_gsd,
        max_side=args.max_side,
        tile_size=args.tile_size,
        overlap=args.overlap,
        tta=args.tta,
        refine=not args.no_refine,
        band_order=[int(b) for b in args.bands.split(",")] if args.bands else None,
        export_mesh=not args.no_mesh,
        mesh_grid=args.mesh_grid,
        max_height=args.max_height,
        use_segmentation=args.use_segmentation,
        events=not args.quiet_events,
    )
    try:
        result = run(cfg)
    except Exception as exc:
        from heimdall.ingestion.loader import UnreadableImageError
        logging.getLogger("heimdall").exception("Pipeline failed")
        if isinstance(exc, (UnreadableImageError, FileNotFoundError, MemoryError)):
            message = str(exc) if not isinstance(exc, MemoryError) else (
                "Ran out of memory. Try a smaller --max-side (e.g. 4096) or a machine with more RAM.")
        else:
            # Don't leak server paths to API clients; the full traceback is in the log.
            message = f"{type(exc).__name__}: {exc}"
            message = re.sub(r"(/[\w.\-]+)+/([\w.\-]+)", r"\2", message)
        if cfg.events:
            emit("error", message=message)
        return 1

    s = result.meta["stats"]["surface"]
    print(f"\n{result.product}: min {s['min']:.2f}  max {s['max']:.2f}  mean {s['mean']:.2f} "
          f"{result.meta['units']}  →  {Path(cfg.output_dir).resolve()}")
    for k, v in result.files.items():
        print(f"  {k:<11} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
