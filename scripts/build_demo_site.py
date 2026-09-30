#!/usr/bin/env python3
"""
Build a side-by-side demo scene: Heimdall's reconstruction and the airborne-LiDAR truth for
the same place, on the same grid, so the two can be flown through and compared directly.

    python scripts/build_demo_site.py --site rotterdam_centre --size 600

For the chosen benchmark site (scripts/benchmark_ahn.py, none of which were trained on) this

  1. downloads the open Dutch orthophoto and the AHN LiDAR DSM/DTM,
  2. runs the production pipeline on the orthophoto, exactly as a user's upload would be,
  3. builds a second scene whose heights come from the LiDAR instead of the model, with the
     same texture, grid, mesh resolution and vertical origin,
  4. scores one against the other and writes the metrics into both scenes' metadata.

The truth scene reuses the *pipeline's own* terrain datum and only replaces the above-ground
layer, so the comparison isolates what the model predicts (height above ground) instead of
mixing in the constant offset between the coarse DEM's geoid and the LiDAR's NAP datum.

Output: <out>/<name>/ and <out>/<name>-truth/, each a self-contained scene the web viewer
loads with no special handling.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

from scripts.benchmark_ahn import SITES, fetch_site  # noqa: E402

log = logging.getLogger("demo-site")


def _resample_to_grid(src_path: Path, transform, crs_wkt: str, shape: tuple[int, int]) -> np.ndarray:
    """Bring a raster onto the pipeline's working grid (same CRS here, so this is a resample)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    if transform is None or crs_wkt is None:
        # A plain (ungeoreferenced) input — GAMUS ships with no CRS — so the pipeline works on
        # the image's own pixel grid and the truth raster is already aligned to it. Resize only
        # if the pipeline downscaled, and do it with area averaging to match how it resamples.
        from PIL import Image
        with rasterio.open(src_path) as s:
            a = s.read(1).astype(np.float32)
            if s.nodata is not None:
                a[a == s.nodata] = np.nan
        a[np.abs(a) > 1e30] = np.nan
        if a.shape != tuple(shape):
            a = np.asarray(Image.fromarray(a).resize((shape[1], shape[0]), Image.Resampling.BOX))
        return a

    dst = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(src_path) as s:
        a = s.read(1).astype(np.float32)
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        a[np.abs(a) > 1e30] = np.nan
        reproject(a, dst, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=transform, dst_crs=crs_wkt,
                  src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.average)
    return dst


def build_truth_scene(model_dir: Path, out_dir: Path, ahn_ndsm: Path, stem: str,
                      meta_path: Path | None = None) -> dict:
    """Clone the model scene with LiDAR heights in place of predicted ones.

    *meta_path* lets a caller point at the metadata explicitly; a bundled demo scene names it
    after its own directory, but a web job writes <JOB_NAME>_meta.json instead.
    """
    import rasterio
    from heimdall.mesh.mesher import build_textured_mesh, export_glb
    from heimdall.output import writers
    from heimdall.pipeline import _grid_layer

    meta = json.loads((meta_path or model_dir / f"{model_dir.name}_meta.json").read_text())
    wt = meta.get("working_transform")
    transform = rasterio.transform.Affine(*wt) if wt else None
    shape = tuple(meta["working_shape"])
    crs_wkt = meta["crs_wkt"]
    gsd = meta["gsd"]["working_m"]

    out_dir.mkdir(parents=True, exist_ok=True)
    # An ungeoreferenced input (GAMUS) yields a relative DSM with no terrain datum, so there is
    # no DTM to add: the reference *is* the height above ground and the two scenes share a zero
    # ground plane. A georeferenced run adds the model's own DTM so both scenes sit on the same
    # terrain and only the above-ground layer differs.
    absolute = "dtm" in meta["files"]
    if absolute:
        with rasterio.open(model_dir / meta["files"]["dtm"]) as s:
            dtm = s.read(1).astype(np.float32)
            dtm[s.read_masks(1) == 0] = np.nan
    else:
        dtm = np.zeros(shape, dtype=np.float32)

    ndsm = np.clip(np.nan_to_num(_resample_to_grid(ahn_ndsm, transform, crs_wkt, shape), nan=0.0), 0, None)
    surface = dtm + ndsm
    valid = np.isfinite(surface)

    src_name = "AHN airborne LiDAR (reference)" if absolute else "GAMUS LiDAR nDSM (reference)"
    tags = {"product": "absolute_dsm" if absolute else "relative_dsm", "source": src_name}
    files = {
        "dsm": writers.save_geotiff(surface, out_dir / f"{stem}_dsm.tif", transform, crs_wkt,
                                    "Absolute DSM from AHN LiDAR", valid_mask=valid, tags=tags).name,
        "ndsm": writers.save_geotiff(ndsm, out_dir / f"{stem}_ndsm.tif", transform, crs_wkt,
                                     "Height above ground from AHN LiDAR", valid_mask=valid, tags=tags).name,
    }
    if absolute:
        files["dtm"] = writers.save_geotiff(dtm, out_dir / f"{stem}_dtm.tif", transform, crs_wkt,
                                            "Terrain datum (shared with the model scene)",
                                            valid_mask=valid, tags=tags).name
    png_path, png_scale = writers.save_depth_png16(surface, out_dir / f"{stem}_height16.png")
    files["height_png"] = png_path.name
    files["preview"] = writers.save_colorized(surface, out_dir / f"{stem}_preview.png",
                                              f"{'Absolute DSM' if absolute else 'Height above ground'} — LiDAR reference (m)",
                                              gsd=gsd, units="m").name
    # Same texture as the model scene: the imagery is identical, only the heights differ.
    shutil.copy(model_dir / meta["files"]["texture"], out_dir / f"{stem}_texture.jpg")
    files["texture"] = f"{stem}_texture.jpg"

    # Identical vertical origin and grid resolution, so the two scenes are directly comparable.
    offset = meta["mesh"]["height_offset"]
    grid_side = max(meta["mesh"]["grid_shape"])
    mesh, minfo = build_textured_mesh(surface, out_dir / files["texture"], gsd,
                                      max_grid_side=grid_side, height_offset=offset)
    files["mesh"] = export_glb(mesh, out_dir / f"{stem}_mesh.glb").name

    gr, gc = minfo["grid_shape"]
    layers = {"surface": _grid_layer(surface, (gr, gc)), "ndsm": _grid_layer(ndsm, (gr, gc)),
              "dtm": _grid_layer(dtm, (gr, gc))}
    dy, dx = minfo["depth_m"] / max(gr - 1, 1), minfo["width_m"] / max(gc - 1, 1)
    s_ = np.nan_to_num(layers["surface"], nan=float(np.nanmin(layers["surface"])))
    gy, gx = np.gradient(s_, dy, dx)
    layers["slope_deg"] = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    order = list(layers)
    (out_dir / f"{stem}_grid.bin").write_bytes(
        np.stack([np.nan_to_num(layers[k], nan=-9999.0).astype("<f4") for k in order]).tobytes())
    files["grid"] = f"{stem}_grid.bin"

    truth_meta = dict(meta)
    truth_meta.update({
        "input": "AHN airborne LiDAR",
        "product_description": "Reference surface measured by airborne LiDAR (AHN, 0.5 m), "
                               "resampled to the model's working grid.",
        "mode": "reference",
        "model": None,
        "weights": None,
        "is_reference": True,
        "mesh": {**minfo, "grid_layers": order, "z_scale": 1.0,
                 "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))},
        "stats": {"surface": _stats(surface), "ndsm": _stats(ndsm), "dtm": _stats(dtm)},
        "height_png_scale": png_scale,
        "files": files,
        "warnings": [],
    })
    truth_meta["files"]["meta"] = f"{stem}_meta.json"
    (out_dir / f"{stem}_meta.json").write_text(json.dumps(truth_meta, indent=2))
    return {"meta": truth_meta, "ndsm": ndsm, "surface": surface}


def _stats(a: np.ndarray) -> dict:
    f = a[np.isfinite(a)]
    if not f.size:
        return {}
    return {"min": float(f.min()), "max": float(f.max()), "mean": float(f.mean()),
            "p2": float(np.percentile(f, 2)), "p98": float(np.percentile(f, 98))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default="rotterdam_centre", choices=sorted(SITES))
    ap.add_argument("--gamus-rgb", type=Path, default=None,
                    help="Build from a GAMUS tile instead of an AHN site: the tile's RGB image. "
                         "GAMUS ships ungeoreferenced, so --gamus-gsd sets the scale.")
    ap.add_argument("--gamus-truth", type=Path, default=None,
                    help="The tile's LiDAR-derived _AGL height raster (metres above ground).")
    ap.add_argument("--gamus-gsd", type=float, default=0.33)
    ap.add_argument("--size", type=int, default=600, help="Site extent in metres.")
    ap.add_argument("--name", default=None, help="Scene folder name (default: the site name).")
    ap.add_argument("--out", type=Path, default=REPO / "heimdall-web" / "public")
    ap.add_argument("--cache", type=Path, default=REPO / "outputs" / "demo_sites")
    ap.add_argument("--layer", default="2023_ortho25")
    ap.add_argument("--tta", action="store_true",
                    help="Test-time augmentation. Measured to change accuracy by <0.02 m for 2.3x the\n"
                         "runtime, so it is off by default and the scenes match how the benchmark scores them.")
    ap.add_argument("--mesh-grid", type=int, default=512,
                    help="Mesh vertices per side. 640 is the pipeline default; 512 keeps the bundled "
                         "scenes small enough to ship without a visible loss of detail.")
    ap.add_argument("--keep-rasters", action="store_true",
                    help="Keep the GeoTIFF/16-bit PNG products in the scene folder. They are the real "
                         "deliverable for a user's own upload, but ~16 MB per scene of dead weight for a "
                         "bundled demo, so they are removed by default.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    from heimdall.eval.metrics import height_metrics
    from heimdall.pipeline import PipelineConfig, run

    if args.gamus_rgb:
        if not args.gamus_truth:
            raise SystemExit("--gamus-rgb needs --gamus-truth (the tile's _AGL raster)")
        name = args.name or args.gamus_rgb.stem.replace("_RGB", "")
        landscape = "US urban (GAMUS)"
        site = {"rgb": args.gamus_rgb, "ndsm": args.gamus_truth}
        extent = None
        log.info("GAMUS tile %s at %.2f m GSD", name, args.gamus_gsd)
    else:
        name = args.name or args.site
        x0, y0, landscape = SITES[args.site]
        log.info("Site %s (%s), %d m", args.site, landscape, args.size)
        site = fetch_site(args.site, x0, y0, args.size, args.cache, args.layer)
        extent = args.size

    model_dir = args.out / name
    log.info("Running the production pipeline on the imagery → %s", model_dir)
    result = run(PipelineConfig(input=str(site["rgb"]), output_dir=str(model_dir), name=name,
                                tta=args.tta, mesh_grid=args.mesh_grid,
                                **({"gsd": args.gamus_gsd} if args.gamus_rgb else {})))

    truth_dir = args.out / f"{name}-truth"
    log.info("Building the LiDAR reference scene → %s", truth_dir)
    truth = build_truth_scene(model_dir, truth_dir, site["ndsm"], f"{name}-truth")

    # Score the model against the LiDAR on the shared grid.
    import rasterio
    with rasterio.open(model_dir / result.files["ndsm"]) as s:
        pred = s.read(1).astype(np.float32)
    ref = truth["ndsm"]
    m = height_metrics(pred, ref)
    zero = height_metrics(np.zeros_like(pred), ref)
    comparison = {
        "site": name, "landscape": landscape,
        "extent_m": extent if extent is not None else round(int(np.shape(ref)[1]) * args.gamus_gsd),
        "reference": ("GAMUS LiDAR-derived nDSM, height above ground" if args.gamus_rgb
                      else "AHN airborne LiDAR (0.5 m), height above ground"),
        "held_out": True,
        "metrics": {k: m.get(k) for k in ("rmse", "mae", "bias", "nmad", "le90", "pearson_r", "r2")},
        "predict_zero_rmse": zero.get("rmse"),
        "truth_scene": f"{name}-truth",
        "model_scene": name,
    }
    for d, stem in ((model_dir, name), (truth_dir, f"{name}-truth")):
        p = d / f"{stem}_meta.json"
        meta = json.loads(p.read_text())
        meta["comparison"] = comparison
        p.write_text(json.dumps(meta, indent=2))

    if not args.keep_rasters:
        for d, stem in ((model_dir, name), (truth_dir, f"{name}-truth")):
            p = d / f"{stem}_meta.json"
            meta = json.loads(p.read_text())
            for key in ("dsm", "ndsm", "dtm", "height_png"):
                f = meta["files"].pop(key, None)
                if f:
                    (d / f).unlink(missing_ok=True)
            p.write_text(json.dumps(meta, indent=2))

    print(f"\n{args.site} ({landscape}), {args.size} m — Heimdall vs AHN LiDAR")
    print(f"  RMSE {m['rmse']:.2f} m   MAE {m['mae']:.2f} m   bias {m['bias']:+.2f} m   "
          f"r {m['pearson_r']:.3f}   [predict zero: {zero['rmse']:.2f} m]")
    print(f"  scenes: {model_dir}  and  {truth_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
