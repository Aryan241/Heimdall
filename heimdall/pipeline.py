"""
Heimdall end-to-end pipeline (shared by the CLI, the web app and the evaluators).

    image ─► ingest ─► GSD ─► resample to working GSD ─► height estimation ─► calibration ─► outputs ─► mesh

Height estimation
  * "head"     — frozen Depth Anything V3 + trained ASPP head → nDSM (metres above ground).
                 Imagery is resampled to the training GSD (0.33 m) and tiled at 512² exactly
                 as during training. The nDSM is then regularised object-by-object (flat /
                 pitched roofs, vertical walls; see heimdall.refine).
  * "relative" — backbone only → relative height (scale-free), used when no checkpoint exists.

Calibration
  * Georeferenced + head:   DSM = DTM(user DEM | Copernicus GLO-30) + nDSM   [+ GCP plane]
  * Georeferenced + relative: RANSAC affine fit of relative height to the DEM  [+ GCP plane]
  * Plain image:            rDSM = nDSM (head) or normalised relative height   [+ row/col GCPs]

Progress is reported as JSON lines on stdout prefixed with ``@@HEIMDALL`` so that the
web server doesn't have to scrape log text.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("heimdall.pipeline")

REPO_ROOT = Path(__file__).resolve().parents[1]
_CKPT_DIR = REPO_ROOT / "checkpoints" / "decoder"
# Preference order: a head trained on the current (Depth Anything V2) backbone, then the
# legacy GAMUS/DA3 head, which the AHN LiDAR benchmark showed has almost no skill.
DEFAULT_WEIGHTS = next((p for p in (_CKPT_DIR / "decoder_best.pth", _CKPT_DIR / "decoder_best_ahn.pth",
                                    _CKPT_DIR / "decoder_best_all.pth") if p.exists()),
                       _CKPT_DIR / "decoder_best_all.pth")
EVENT_PREFIX = "@@HEIMDALL "


def emit(event: str, **data) -> None:
    """Machine-readable progress event on stdout (one line, flushed)."""
    sys.stdout.write(EVENT_PREFIX + json.dumps({"event": event, **data}, default=float) + "\n")
    sys.stdout.flush()


@dataclass
class PipelineConfig:
    input: Path
    output_dir: Path = Path("outputs")
    name: str | None = None                  # output file prefix (default: input stem)
    weights: Path | None = DEFAULT_WEIGHTS   # None → relative mode
    model_key: str | None = None             # default: from checkpoint, else da3-metric-l
    device: str | None = None
    reference_dem: Path | None = None
    auto_dem: bool = True
    gcps: Path | None = None
    gsd: float | None = None                 # user override, metres/pixel
    target_gsd: float | None = None          # default: training GSD in head mode
    max_side: int = 8192
    tile_size: int = 512
    overlap: int = 128
    tta: bool = False
    refine: bool = True                      # object-based planar regularisation of the nDSM
    band_order: list[int] | None = None
    export_mesh: bool = True
    mesh_grid: int = 640
    texture_side: int = 4096
    max_height: float | None = None          # relative mode, plain image: scale 0..1 to metres
    use_segmentation: bool = False           # relative mode: SegFormer ground mask for RANSAC
    dtm_opening_cells: int = 3
    events: bool = True


@dataclass
class PipelineResult:
    product: str
    files: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


def _stats(a: np.ndarray) -> dict:
    f = a[np.isfinite(a)]
    if f.size == 0:
        return {}
    return {
        "min": float(f.min()), "max": float(f.max()), "mean": float(f.mean()), "std": float(f.std()),
        "p02": float(np.percentile(f, 2)), "p50": float(np.percentile(f, 50)), "p98": float(np.percentile(f, 98)),
    }


def run(cfg: PipelineConfig) -> PipelineResult:
    import torch  # noqa: F401  (deferred heavy import)
    from heimdall.device import get_device
    from heimdall.geo import (TRAINING_GSD_M, center_latlon, gsd_from_geo, is_rotated, resize_image,
                              resize_mask, scaled_transform, working_shape)
    from heimdall.ingestion.loader import ingest
    from heimdall.output import writers

    t_start = time.perf_counter()
    timings: dict[str, float] = {}
    warnings_: list[str] = []
    say = emit if cfg.events else (lambda *a, **k: None)

    def warn(msg: str):
        logger.warning(msg)
        warnings_.append(msg)
        say("warning", message=msg)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = cfg.name or Path(cfg.input).stem
    device = get_device(cfg.device)

    # ── Stage 1: ingest ─────────────────────────────────────────────────────
    say("stage", stage="ingest", message="Reading image")
    t0 = time.perf_counter()
    payload = ingest(cfg.input, band_order=cfg.band_order)
    native_shape = payload.image.shape[:2]
    geo = payload.geo
    georef = payload.kind == "georeferenced"
    timings["ingest"] = time.perf_counter() - t0
    say("info", kind=payload.kind, width=native_shape[1], height=native_shape[0],
        crs=(geo.crs_epsg if geo else None))

    head_mode = cfg.weights is not None and Path(cfg.weights).exists()
    if cfg.weights is not None and not head_mode:
        warn(f"Decoder weights not found at {cfg.weights} — falling back to relative mode.")

    # ── Stage 2: GSD ────────────────────────────────────────────────────────
    say("stage", stage="gsd", message="Determining ground sample distance")
    gsd_info: dict = {}
    if cfg.gsd:
        native_gsd, gsd_info = float(cfg.gsd), {"source": "user"}
    elif georef:
        native_gsd, gsd_info = gsd_from_geo(geo, native_shape), {"source": "geotransform"}
        if is_rotated(geo.transform):
            warn("The raster has a rotated geotransform (not north-up). Elevation products stay "
                 "correctly georeferenced, but mesh axes and the viewer's lat/lon readout assume "
                 "north-up and are approximate.")
    else:
        from heimdall.calibration.heuristic import estimate_gsd
        native_gsd, gsd_info = estimate_gsd(payload.image)
        gsd_info["source"] = "vehicle-size" if native_gsd else "assumed"
        if native_gsd is None:
            native_gsd = TRAINING_GSD_M
            warn(f"No georeferencing and GSD could not be estimated — assuming {TRAINING_GSD_M} m/px "
                 "(pass --gsd for correct metric scale).")
    gsd_info["native_m"] = native_gsd

    target = cfg.target_gsd or (TRAINING_GSD_M if head_mode else None)
    work_shape, work_gsd = working_shape(native_shape, native_gsd, target, max_side=cfg.max_side)
    image = resize_image(payload.image, work_shape)
    valid = resize_mask(payload.valid_mask, work_shape) if payload.valid_mask is not None else None
    work_transform = scaled_transform(geo.transform, native_shape, work_shape) if georef else None
    gsd_info["working_m"] = work_gsd
    if work_gsd and target and abs(work_gsd - target) / target > 0.25:
        warn(f"Working GSD {work_gsd:.2f} m differs from the {target:.2f} m training GSD "
             "(up-sampling is capped at 3×); metric heights will be less reliable.")
    logger.info("GSD native %.3f m → working %.3f m; grid %s → %s", native_gsd, work_gsd, native_shape, work_shape)
    say("info", gsd_native=native_gsd, gsd_working=work_gsd, work_width=work_shape[1], work_height=work_shape[0])

    # ── Stage 3: height estimation ─────────────────────────────────────────
    t0 = time.perf_counter()
    ndsm = rel = None
    model_key = cfg.model_key
    if head_mode:
        from heimdall.decoder.model import load_wrapper, predict_ndsm
        say("stage", stage="depth", message="Loading Depth Anything V3 + Heimdall head")
        wrapper = load_wrapper(cfg.weights, device, model_key=model_key)
        model_key = wrapper.model_key
        if model_key.startswith("da3"):
            warn("This decoder head was trained on a Depth Anything V3 metric backbone, which is nearly "
                 "blind to height in nadir imagery (LiDAR benchmark: r≈0.06, RMSE 7.4 m vs 8.4 m for "
                 "predicting zero). Heights here are unreliable — retrain on a V2 backbone "
                 "(docs/KAGGLE_TRAINING.md).")
        say("stage", stage="depth", message="Estimating height above ground")
        ndsm = predict_ndsm(wrapper, image, device, tile_size=cfg.tile_size, overlap=cfg.overlap, tta=cfg.tta,
                            progress=lambda i, n: say("progress", stage="depth", current=i, total=n))
        ndsm = np.clip(ndsm, 0.0, None)
        if cfg.refine:
            from heimdall.geo import resize_float
            from heimdall.refine import refine_ndsm
            say("stage", stage="depth", message="Regularising roofs and walls")
            # If the source is sharper than the working grid, regularise on a 2× finer grid so walls
            # follow image edges with half-size stair-steps; the products are written on that grid.
            up = min(2.0, work_gsd / native_gsd) if native_gsd else 1.0
            if up >= 1.5:
                fine = (int(round(work_shape[0] * up)), int(round(work_shape[1] * up)))
                if work_transform is not None:
                    work_transform = scaled_transform(work_transform, work_shape, fine)
                ndsm = resize_float(ndsm, fine)
                image = resize_image(payload.image, fine)
                valid = resize_mask(payload.valid_mask, fine) if payload.valid_mask is not None else None
                work_gsd = work_gsd * work_shape[0] / fine[0]
                work_shape = fine
                gsd_info["output_m"] = work_gsd
            from heimdall.refine.vehicles import detect_vehicles
            vehicles, _ = detect_vehicles(payload.image, native_gsd, work_shape)
            ndsm, refine_report = refine_ndsm(np.where(valid, ndsm, np.nan) if valid is not None else ndsm,
                                              image, work_gsd, vehicles=vehicles)
            ndsm = np.nan_to_num(ndsm, nan=0.0)
    else:
        from heimdall.depth.depth_anything import predict_relative_height
        model_key = model_key or "vit-b"
        say("stage", stage="depth", message=f"Estimating relative height ({model_key})")
        rel = predict_relative_height(image, model_key=model_key, device=device)
    timings["depth"] = time.perf_counter() - t0

    # ── Stage 4: calibration ───────────────────────────────────────────────
    say("stage", stage="calibration", message="Calibrating to absolute elevation")
    t0 = time.perf_counter()
    from heimdall.calibration.dem import fit_gcp_correction, load_gcps, load_reference_dem_raw, terrain_reference
    calib: dict = {}
    if head_mode and cfg.refine:
        calib["refinement"] = {"method": "segment-wise planar regularisation", **refine_report}
    dtm = None
    crs_wkt = geo.crs_wkt if georef else None

    if head_mode:
        surface = ndsm.copy()
        product = "ndsm" if georef else "relative_dsm"
        if georef:
            try:
                terr = terrain_reference(work_transform, crs_wkt, work_shape, work_gsd,
                                         dem_path=str(cfg.reference_dem) if cfg.reference_dem else None,
                                         auto_fetch=cfg.auto_dem, opening_cells=cfg.dtm_opening_cells)
            except Exception as exc:
                terr = None
                warn(f"Reference DEM unusable: {exc}")
            if terr is not None:
                dtm = terr.dtm
                surface = dtm + ndsm
                product = "absolute_dsm"
                calib.update({"method": "DTM + nDSM", "terrain_source": terr.source, **terr.details})
            elif not cfg.gcps:
                warn("No terrain reference available (offline and no DEM supplied) — output is height above "
                     "ground (nDSM), not absolute elevation.")
        elif cfg.reference_dem:
            warn("A reference DEM was supplied for a non-georeferenced image; it cannot be aligned and was ignored.")
    else:
        product = "relative_dsm"
        surface = rel
        ref = None
        if cfg.reference_dem:
            ref = load_reference_dem_raw(str(cfg.reference_dem), work_shape, work_transform, crs_wkt)
            src_desc = f"user DEM ({Path(cfg.reference_dem).name})"
        elif georef and cfg.auto_dem:
            try:
                from heimdall.calibration.dem import fetch_copernicus
                ref = load_reference_dem_raw(fetch_copernicus(work_transform, crs_wkt, work_shape),
                                             work_shape, work_transform, crs_wkt)
                src_desc = "Copernicus GLO-30 (auto)"
            except Exception as exc:
                warn(f"Automatic Copernicus GLO-30 download failed: {exc}")
        fit = None
        if ref is not None:
            # A coarse DEM can only constrain the relative map's scale if the scene spans many DEM
            # cells *and* the DEM itself has relief; otherwise the fit is ill-posed.
            finite = ref[np.isfinite(ref)]
            relief = float(np.percentile(finite, 98) - np.percentile(finite, 2)) if finite.size else 0.0
            cells = max(work_shape) * (work_gsd or 1.0) / 30.0
            if relief < 5.0 or cells < 8:
                warn(f"Reference DEM cannot calibrate relative heights here (relief {relief:.1f} m over "
                     f"~{cells:.0f} DEM cells) — use the trained head for metric heights.")
            else:
                from heimdall.calibration.ransac import apply_transform, fit_affine_transform
                mask = None
                if cfg.use_segmentation:
                    from heimdall.segmentation.segformer import GroundSegmenter
                    mask = GroundSegmenter(device=device).get_ground_mask(image).cpu().numpy()
                fit = fit_affine_transform(rel, ref, mask=mask)
                if fit["scale"] <= 0:
                    warn("RANSAC scale was non-positive (relative map anti-correlated with the DEM) — fit rejected.")
                    fit = None
        if fit is not None:
            surface = apply_transform(rel, fit["scale"], fit["shift"])
            product = "absolute_dsm" if georef else "relative_dsm"
            calib.update({"method": "RANSAC affine (relative → DEM)", "terrain_source": src_desc, **fit})
        else:
            lo, hi = np.percentile(rel[valid] if valid is not None else rel, [1, 99.5])
            surface = np.clip((rel - lo) / max(hi - lo, 1e-6), 0, None)
            if cfg.max_height:
                surface = surface * cfg.max_height
                calib.update({"method": "percentile normalisation × user max height", "max_height": cfg.max_height})
            else:
                calib.update({"method": "percentile normalisation (unitless 0–1)"})

    if cfg.gcps:
        try:
            gcps = load_gcps(str(cfg.gcps), work_transform, crs_wkt)
            corr, rep = fit_gcp_correction(surface, gcps)
            surface = surface + corr
            calib["gcp"] = rep
            if georef and product == "ndsm" and rep.get("used"):
                product = "absolute_dsm"
                calib.setdefault("method", "nDSM + GCP datum")
            logger.info("GCP correction: %s", rep)
        except Exception as exc:
            warn(f"GCPs could not be applied: {exc}")
    timings["calibration"] = time.perf_counter() - t0

    units = "relative" if (not head_mode and "scale" not in calib and not cfg.max_height) else "m"
    if valid is not None:
        surface = np.where(valid, surface, np.nan)

    # ── Stage 5: outputs ───────────────────────────────────────────────────
    say("stage", stage="outputs", message="Writing GeoTIFF / PNG outputs")
    t0 = time.perf_counter()
    files: dict[str, str] = {}
    descriptions = {
        "absolute_dsm": "Absolute DSM (metres, orthometric; terrain datum + predicted above-ground height)",
        "ndsm": "Normalised DSM: predicted height above ground (metres)",
        "relative_dsm": "Relative DSM (rDSM): height above local ground" + (" (metres, GSD-assumed scale)"
                                                                           if units == "m" else " (unitless)"),
    }
    main_name = {"absolute_dsm": "dsm", "ndsm": "ndsm", "relative_dsm": "rdsm"}[product]
    tags = {"product": product, "gsd_m": f"{work_gsd:.4f}", "model": model_key or "",
            "calibration": calib.get("method", "")}
    files["dsm"] = writers.save_geotiff(surface, out_dir / f"{stem}_{main_name}.tif", work_transform, crs_wkt,
                                        descriptions[product], units="metre" if units == "m" else "relative",
                                        valid_mask=valid, tags=tags).name
    if ndsm is not None and product != "ndsm":
        files["ndsm"] = writers.save_geotiff(ndsm, out_dir / f"{stem}_ndsm.tif", work_transform, crs_wkt,
                                             descriptions["ndsm"], valid_mask=valid, tags=tags).name
    if dtm is not None:
        files["dtm"] = writers.save_geotiff(dtm, out_dir / f"{stem}_dtm.tif", work_transform, crs_wkt,
                                            "Terrain datum (bare-earth estimate from reference DEM)",
                                            valid_mask=valid, tags=tags).name
    png_path, png_scale = writers.save_depth_png16(surface, out_dir / f"{stem}_height16.png")
    files["height_png"] = png_path.name
    title = {"absolute_dsm": "Absolute DSM (m)", "ndsm": "Height above ground (m)",
             "relative_dsm": "Relative DSM" + (" (m)" if units == "m" else "")}[product]
    files["preview"] = writers.save_colorized(surface, out_dir / f"{stem}_preview.png", title, gsd=work_gsd,
                                              units=units).name
    # Texture comes from the native image (usually finer than the working grid), capped in size.
    files["texture"] = writers.save_texture(payload.image, out_dir / f"{stem}_texture.jpg",
                                            max_side=cfg.texture_side).name
    timings["outputs"] = time.perf_counter() - t0

    # ── Stage 6: mesh ──────────────────────────────────────────────────────
    mesh_meta = None
    if cfg.export_mesh:
        say("stage", stage="mesh", message="Building textured 3D mesh")
        t0 = time.perf_counter()
        from heimdall.mesh.mesher import build_textured_mesh, export_glb
        mesh_surface = surface.copy()  # NaN (nodata) cells are cut out of the mesh
        # Relative (unitless) surfaces get a nominal vertical scale so the terrain is visible.
        mesh_z_scale = 1.0
        if units == "relative":
            mesh_z_scale = 0.08 * max(work_shape) * work_gsd
            mesh_surface = mesh_surface * mesh_z_scale
        mesh, minfo = build_textured_mesh(mesh_surface, out_dir / files["texture"], work_gsd,
                                          max_grid_side=cfg.mesh_grid)
        glb = export_glb(mesh, out_dir / f"{stem}_mesh.glb")
        files["mesh"] = glb.name

        # Height layers sampled on the mesh grid, in vertex order, for exact readouts in the viewer.
        gr, gc = minfo["grid_shape"]
        layers = {"surface": _grid_layer(surface, (gr, gc))}
        if ndsm is not None:
            layers["ndsm"] = _grid_layer(ndsm, (gr, gc))
        if dtm is not None:
            layers["dtm"] = _grid_layer(dtm, (gr, gc))
        dy, dx = minfo["depth_m"] / max(gr - 1, 1), minfo["width_m"] / max(gc - 1, 1)
        s = np.nan_to_num(layers["surface"], nan=float(np.nanmin(layers["surface"])))
        if units == "relative":
            s = s * mesh_z_scale
        gy, gx = np.gradient(s, dy, dx)
        layers["slope_deg"] = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
        order = list(layers.keys())
        blob = np.stack([np.nan_to_num(layers[k], nan=-9999.0).astype("<f4") for k in order])
        (out_dir / f"{stem}_grid.bin").write_bytes(blob.tobytes())
        files["grid"] = f"{stem}_grid.bin"
        mesh_meta = {**minfo, "grid_layers": order, "z_scale": mesh_z_scale,
                     "vertices": int(len(mesh.vertices)), "faces": int(len(mesh.faces))}
        timings["mesh"] = time.perf_counter() - t0

    # ── Metadata ───────────────────────────────────────────────────────────
    meta = {
        "version": 2,
        "input": Path(cfg.input).name,
        "kind": payload.kind,
        "product": product,
        "product_description": descriptions[product],
        "units": units,
        "mode": "head" if head_mode else "relative",
        "model": model_key,
        "weights": Path(cfg.weights).name if head_mode else None,
        "native_shape": list(native_shape),
        "working_shape": list(work_shape),
        "gsd": gsd_info,
        "crs_epsg": geo.crs_epsg if georef else None,
        "crs_wkt": crs_wkt,
        "working_transform": list(work_transform) if work_transform else None,
        "center_latlon": list(center_latlon(geo, native_shape)) if georef else None,
        "bounds": list(geo.bounds) if georef else None,
        "calibration": calib,
        "stats": {"surface": _stats(surface), **({"ndsm": _stats(ndsm)} if ndsm is not None else {}),
                  **({"dtm": _stats(dtm)} if dtm is not None else {})},
        "height_png_scale": png_scale,
        "mesh": mesh_meta,
        "files": files,
        "warnings": warnings_,
        "timings_s": {k: round(v, 2) for k, v in timings.items()} | {"total": round(time.perf_counter() - t_start, 2)},
        "config": {k: (Path(v).name if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},  # no local paths
    }
    if ndsm is not None:
        meta["stats"]["built_fraction"] = float((ndsm > 2.5).mean())
    meta_path = writers.save_json(meta, out_dir / f"{stem}_meta.json")
    files["meta"] = meta_path.name
    meta["files"] = files
    writers.save_json(meta, meta_path)
    say("done", meta=meta_path.name, product=product, stats=meta["stats"]["surface"], files=files)
    logger.info("Pipeline complete in %.1fs → %s", meta["timings_s"]["total"], out_dir.resolve())
    return PipelineResult(product=product, files=files, meta=meta)


def _grid_layer(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image
    gr, gc = shape
    a = arr.astype(np.float32)
    if a.shape == (gr, gc):
        return a
    fill = float(np.nanmin(a)) if np.isfinite(a).any() else 0.0
    nan_mask = ~np.isfinite(a)
    img = Image.fromarray(np.nan_to_num(a, nan=fill))
    out = np.asarray(img.resize((gc, gr), Image.Resampling.BOX if a.shape[0] > gr else Image.Resampling.BILINEAR))
    if nan_mask.any():
        m = np.asarray(Image.fromarray(nan_mask.astype(np.uint8) * 255).resize((gc, gr))) > 127
        out = np.where(m, np.nan, out)
    return out.astype(np.float32)
