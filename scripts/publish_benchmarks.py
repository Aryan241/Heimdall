#!/usr/bin/env python3
"""
Collect every measured accuracy number into one file the web app can display, so the
results are visible in the product itself rather than only in a report.

    python scripts/publish_benchmarks.py

Reads whatever exists:
    outputs/benchmark_ahn/results.json   LiDAR benchmark at native 0.25 m
    outputs/benchmark_gsd/results.json   the same sites at simulated satellite resolutions
    outputs/gamus_test.json              held-out GAMUS test split
and writes heimdall-web/public/benchmarks.json plus the per-site error maps.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WEB = REPO / "heimdall-web" / "public"


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _sites(rows: list, gsd: float = 0.25, variant: str = "raw") -> list[dict]:
    out = []
    for r in rows:
        if r.get("gsd") != gsd or r.get("variant") != variant:
            continue
        o = (r.get("ndsm") or {}).get("overall") or {}
        nd = r.get("ndsm") or {}
        base = nd.get("zero_baseline") or nd.get("baselines", {}).get("predict_zero", {})
        out.append({
            "site": r["site"], "landscape": r.get("kind"),
            "rmse": o.get("rmse"), "mae": o.get("mae"), "bias": o.get("bias"),
            "pearson_r": o.get("pearson_r"), "predict_zero_rmse": base.get("rmse"),
        })
    return sorted(out, key=lambda x: (x["rmse"] is None, x["rmse"]))


def _by_gsd(rows: list, variant: str = "raw") -> list[dict]:
    from statistics import mean
    out = []
    for g in sorted({r.get("gsd") for r in rows if r.get("gsd")}):
        sel = [r for r in rows if r.get("gsd") == g and r.get("variant") == variant]
        vals = [((r.get("ndsm") or {}).get("overall") or {}) for r in sel]
        vals = [v for v in vals if v.get("rmse") is not None]
        if not vals:
            continue
        zero = [((r.get("ndsm") or {}).get("zero_baseline")
                 or (r.get("ndsm") or {}).get("baselines", {}).get("predict_zero", {})).get("rmse")
                for r in sel]
        zero = [z for z in zero if z is not None]
        out.append({"gsd": g, "sites": len(vals),
                    "rmse": mean(v["rmse"] for v in vals),
                    "mae": mean(v["mae"] for v in vals),
                    "pearson_r": mean(v["pearson_r"] for v in vals if v.get("pearson_r") is not None),
                    "predict_zero_rmse": mean(zero) if zero else None})
    return out


def main() -> int:
    ahn = _load(REPO / "outputs" / "benchmark_ahn" / "results.json") or []
    gsd = _load(REPO / "outputs" / "benchmark_gsd" / "results.json") or []
    gamus = _load(REPO / "outputs" / "gamus_test.json")

    rows = gsd or ahn  # the GSD run repeats the native-resolution pass, so prefer it
    payload = {
        "generated": __import__("datetime").date.today().isoformat(),
        "model": {"backbone": "Depth Anything V2 Base (frozen)", "head": "multi-scale DPT-pyramid decoder",
                  "trained_on": "9 004 tiles — GAMUS + Dutch AHN LiDAR at 0.33 m and 1 m",
                  "val_rmse": (gamus or {}).get("training", {}).get("val_rmse")},
        "lidar_sites": _sites(rows),
        "by_gsd": _by_gsd(rows),
        "gamus_test": None,
    }
    if gamus:
        payload["gamus_test"] = {
            "tiles": gamus.get("tiles"),
            "overall": gamus["overall"],
            "predict_zero": gamus["baselines"]["predict_zero"],
            "by_landscape": {k: {"rmse": v["rmse"], "mae": v["mae"], "pearson_r": v.get("pearson_r")}
                             for k, v in gamus.get("by_landscape", {}).items()},
            "by_pixel_class": {k: {"rmse": v["rmse"], "mae": v["mae"], "bias": v["bias"]}
                               for k, v in gamus.get("by_pixel_class", {}).items()},
        }

    # Per-site error maps, so the numbers come with a picture.
    maps = {}
    dst = WEB / "benchmark"
    dst.mkdir(parents=True, exist_ok=True)
    for png in sorted((REPO / "outputs" / "benchmark_ahn").glob("*.png")):
        shutil.copy(png, dst / png.name)
        maps[png.stem] = f"/benchmark/{png.name}"
    payload["site_maps"] = maps

    out = WEB / "benchmarks.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}  ({len(payload['lidar_sites'])} sites, {len(payload['by_gsd'])} resolutions, "
          f"{len(maps)} error maps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
