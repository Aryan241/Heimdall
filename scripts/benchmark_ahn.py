#!/usr/bin/env python3
"""
Independent accuracy benchmark against airborne LiDAR (Netherlands, fully open data).

For each site the script downloads
  * the national aerial orthophoto (PDOK Luchtfoto RGB, 25 cm; EPSG:28992),
  * the AHN LiDAR DSM and DTM (0.5 m; heights in NAP),
runs the *production* pipeline on the orthophoto (optionally degraded to satellite-like
GSDs), and scores

  * height above ground  — Heimdall nDSM  vs  AHN DSM − AHN DTM   (model skill)
  * absolute elevation   — Heimdall DSM   vs  AHN DSM             (model + terrain datum)

overall, per pixel class (building / tree / ground / low vegetation) and against a
predict-zero baseline. None of these sites were used for training.

    python scripts/benchmark_ahn.py                       # all sites, native 0.25 m imagery
    python scripts/benchmark_ahn.py --gsd 0.5 1.0         # also simulate 0.5 m / 1 m sensors
    python scripts/benchmark_ahn.py --sites amsterdam_centre veluwe_forest --size 300
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np

# name: (x_min, y_min) in EPSG:28992 (RD New), landscape label (for reporting only)
SITES: dict[str, tuple[int, int, str]] = {
    "amsterdam_centre": (121000, 487000, "dense historic urban"),
    "rotterdam_centre": (92400, 437000, "high-rise urban"),
    "utrecht_suburb": (137500, 452500, "suburban residential"),
    "rotterdam_port": (75500, 434500, "industrial"),
    "westland_greenhouses": (76500, 445500, "greenhouses"),
    "veluwe_forest": (187500, 452000, "forest"),
    "flevoland_farmland": (165000, 505000, "sparse rural"),
    "valkenburg_hills": (186500, 318500, "hilly town"),
}

WCS = ("https://service.pdok.nl/rws/ahn/wcs/v1_0?service=WCS&request=GetCoverage&version=2.0.1"
       "&coverageId={cov}&subset=x({x0},{x1})&subset=y({y0},{y1})&format=image/tiff")
WMS = ("https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0?service=WMS&version=1.3.0&request=GetMap"
       "&layers={layer}&styles=&crs=EPSG:28992&bbox={x0},{y0},{x1},{y1}&width={w}&height={h}&format=image/jpeg")

log = logging.getLogger("benchmark")


def _get(url: str, dst: Path) -> Path:
    if not dst.exists():
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(url, dst)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    return dst


def fetch_site(name: str, x0: int, y0: int, size: int, root: Path, layer: str) -> dict[str, Path]:
    """Download orthophoto + AHN DSM/DTM and derive the reference nDSM (cached)."""
    import rasterio
    from rasterio.transform import from_origin
    from PIL import Image

    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    x1, y1 = x0 + size, y0 + size
    dsm = _get(WCS.format(cov="dsm_05m", x0=x0, x1=x1, y0=y0, y1=y1), d / "ahn_dsm.tif")
    dtm = _get(WCS.format(cov="dtm_05m", x0=x0, x1=x1, y0=y0, y1=y1), d / "ahn_dtm.tif")
    px = int(size / 0.25)
    jpg = _get(WMS.format(layer=layer, x0=x0, y0=y0, x1=x1, y1=y1, w=px, h=px), d / "ortho.jpg")

    rgb_tif = d / "ortho_0.25m.tif"
    if not rgb_tif.exists():
        rgb = np.asarray(Image.open(jpg).convert("RGB"))
        with rasterio.open(rgb_tif, "w", driver="GTiff", height=px, width=px, count=3, dtype="uint8",
                           crs="EPSG:28992", transform=from_origin(x0, y1, 0.25, 0.25), compress="deflate") as o:
            o.write(rgb.transpose(2, 0, 1))

    ndsm_path = d / "ahn_ndsm.tif"
    if not ndsm_path.exists():
        from scipy import ndimage
        with rasterio.open(dsm) as s:
            a = s.read(1).astype(np.float32); prof = s.profile; nd = s.nodata
        with rasterio.open(dtm) as s:
            t = s.read(1).astype(np.float32); ndt = s.nodata
        a[(a == nd) | (a > 1e30)] = np.nan
        t[(t == ndt) | (t > 1e30)] = np.nan
        # AHN DTM has holes under buildings/water: fill by nearest valid terrain, then smooth.
        if np.isnan(t).any() and np.isfinite(t).any():
            idx = ndimage.distance_transform_edt(np.isnan(t), return_distances=False, return_indices=True)
            t = ndimage.uniform_filter(t[tuple(idx)], size=5)
        ndsm = np.clip(a - t, 0, None)
        prof.update(dtype="float32", nodata=np.nan, compress="deflate")
        with rasterio.open(ndsm_path, "w", **prof) as o:
            o.write(ndsm.astype(np.float32), 1)
    return {"dir": d, "rgb": rgb_tif, "dsm": dsm, "ndsm": ndsm_path}


def degrade(rgb_tif: Path, gsd: float) -> Path:
    """Resample the orthophoto to a coarser GSD (area averaging) to mimic a satellite sensor."""
    import rasterio
    from rasterio.enums import Resampling
    out = rgb_tif.with_name(f"ortho_{gsd:g}m.tif")
    if out.exists():
        return out
    with rasterio.open(rgb_tif) as s:
        f = s.res[0] / gsd
        h, w = int(s.height * f), int(s.width * f)
        data = s.read(out_shape=(3, h, w), resampling=Resampling.average)
        prof = s.profile
        prof.update(height=h, width=w, transform=s.transform * s.transform.scale(s.width / w, s.height / h))
    with rasterio.open(out, "w", **prof) as o:
        o.write(data)
    return out


def score(pred_path: Path, ref_path: Path, rgb_path: Path) -> dict:
    import rasterio
    from PIL import Image
    from heimdall.eval.compare import reference_on_grid
    from heimdall.eval.metrics import height_metrics, pixel_strata

    with rasterio.open(pred_path) as s:
        pred = s.read(1).astype(np.float32)
        pred[pred == s.nodata] = np.nan
        tr, crs, res, shape = s.transform, s.crs, s.res[0], (s.height, s.width)
    ref = reference_on_grid(ref_path, shape, tr, crs, res)
    rgb = np.asarray(Image.open(rgb_path).convert("RGB").resize(shape[::-1])) if rgb_path.suffix == ".jpg" else None
    if rgb is None:
        with rasterio.open(rgb_path) as s:
            rgb = np.asarray(Image.fromarray(s.read().transpose(1, 2, 0)).resize(shape[::-1]))
    out = {"overall": height_metrics(pred, ref), "zero_baseline": height_metrics(np.zeros_like(ref), ref)}
    out["by_class"] = {k: height_metrics(pred, ref, m) for k, m in pixel_strata(rgb, np.nan_to_num(ref)).items()
                       if m.sum() > 100}
    return out, pred, ref


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="*", default=list(SITES))
    ap.add_argument("--size", type=int, default=400, help="Site size in metres.")
    ap.add_argument("--gsd", nargs="*", type=float, default=[], help="Extra simulated sensor GSDs (m).")
    ap.add_argument("--layer", default="2023_ortho25", help="PDOK orthophoto layer (close in time to AHN).")
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "benchmark_ahn")
    ap.add_argument("--no-refine-compare", action="store_true", help="Only run the default (refined) pipeline.")
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)

    from heimdall.pipeline import PipelineConfig, run

    args.out.mkdir(parents=True, exist_ok=True)
    variants = [("refined", True)] + ([] if args.no_refine_compare else [("raw", False)])
    results = []
    for name in args.sites:
        x0, y0, kind = SITES[name]
        log.info("── %s (%s)", name, kind)
        site = fetch_site(name, x0, y0, args.size, args.out / "data", args.layer)
        for gsd in [0.25] + list(args.gsd):
            img = site["rgb"] if gsd == 0.25 else degrade(site["rgb"], gsd)
            for vname, refine in variants:
                tag = f"{name}_{gsd:g}m_{vname}"
                run_dir = args.out / "runs" / tag
                t0 = time.time()
                res = run(PipelineConfig(input=img, output_dir=run_dir, name="pred", refine=refine,
                                         export_mesh=False, events=False, tta=args.tta))
                files = res.files
                ndsm_file = run_dir / files.get("ndsm", files["dsm"])
                s_ndsm, pred, ref = score(ndsm_file, site["ndsm"], img)
                entry = {"site": name, "kind": kind, "gsd": gsd, "variant": vname,
                         "product": res.product, "seconds": round(time.time() - t0, 1),
                         "ndsm": s_ndsm}
                if res.product == "absolute_dsm":
                    s_dsm, *_ = score(run_dir / files["dsm"], site["dsm"], img)
                    entry["dsm"] = s_dsm
                results.append(entry)
                o = s_ndsm["overall"]
                log.info("  %-6s %-8s nDSM RMSE %5.2f  MAE %5.2f  bias %+5.2f  r %.3f   (predict-0 RMSE %5.2f)",
                         f"{gsd:g}m", vname, o["rmse"], o["mae"], o["bias"], o.get("pearson_r", np.nan),
                         s_ndsm["zero_baseline"]["rmse"])
                if vname == "refined" and gsd == 0.25:
                    _figure(args.out / f"{name}.png", img, ref, pred, o)

    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    md = _markdown(results)
    (args.out / "results.md").write_text(md)
    print("\n" + md)
    return 0


def _markdown(results: list[dict]) -> str:
    lines = ["| Site | Type | GSD | Variant | nDSM RMSE | nDSM MAE | Bias | r | Building MAE | Tree MAE | Ground MAE | Predict-0 RMSE | DSM RMSE (offset removed) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "—" if v is None or not np.isfinite(v) else f"{v:.2f}"
    for r in results:
        o, c = r["ndsm"]["overall"], r["ndsm"].get("by_class", {})
        dsm_rmse = None
        if "dsm" in r:
            dsm_rmse = r["dsm"]["overall"].get("rmse_debiased")
        lines.append(f"| {r['site']} | {r['kind']} | {r['gsd']:g} m | {r['variant']} | {f(o.get('rmse'))} | {f(o.get('mae'))} | "
                     f"{f(o.get('bias'))} | {f(o.get('pearson_r'))} | {f(c.get('building', {}).get('mae'))} | "
                     f"{f(c.get('tree', {}).get('mae'))} | {f(c.get('ground', {}).get('mae'))} | "
                     f"{f(r['ndsm']['zero_baseline'].get('rmse'))} | {f(dsm_rmse)} |")
    # Aggregate per variant/GSD (pixel-weighted means are approximated by site means here).
    lines.append("")
    lines.append("| GSD | Variant | mean nDSM RMSE | mean nDSM MAE | mean r | mean predict-0 RMSE |")
    lines.append("|---|---|---|---|---|---|")
    keys = sorted({(r["gsd"], r["variant"]) for r in results})
    for g, v in keys:
        rs = [r for r in results if r["gsd"] == g and r["variant"] == v]
        m = lambda k: np.nanmean([r["ndsm"]["overall"].get(k, np.nan) for r in rs])
        z = np.nanmean([r["ndsm"]["zero_baseline"]["rmse"] for r in rs])
        lines.append(f"| {g:g} m | {v} | {m('rmse'):.2f} | {m('mae'):.2f} | {m('pearson_r'):.3f} | {z:.2f} |")
    return "\n".join(lines) + "\n"


def _figure(path: Path, img: Path, ref: np.ndarray, pred: np.ndarray, m: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    with rasterio.open(img) as s:
        rgb = s.read().transpose(1, 2, 0)
    vmax = float(np.nanpercentile(ref, 99)) or 1
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
    ax[0].imshow(rgb); ax[0].set_title("Orthophoto")
    ax[1].imshow(ref, cmap="turbo", vmin=0, vmax=vmax); ax[1].set_title("AHN LiDAR nDSM")
    im = ax[2].imshow(pred, cmap="turbo", vmin=0, vmax=vmax); ax[2].set_title("Heimdall nDSM")
    e = ax[3].imshow(pred - ref, cmap="RdBu_r", vmin=-vmax / 2, vmax=vmax / 2)
    ax[3].set_title(f"Error: RMSE {m['rmse']:.2f} m, MAE {m['mae']:.2f} m, r {m.get('pearson_r', 0):.2f}")
    for a in ax:
        a.axis("off")
    fig.colorbar(im, ax=ax[1:3], fraction=0.02, label="m")
    fig.colorbar(e, ax=ax[3], fraction=0.046, label="m")
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
