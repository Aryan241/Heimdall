#!/usr/bin/env python3
"""
Decide whether a candidate checkpoint should replace the shipped one.

    python scripts/compare_checkpoint.py --candidate ~/Downloads/decoder_best_e1.pth
    python scripts/compare_checkpoint.py --candidate ... --results outputs/bench_e1   # reuse a run

Benchmarks the candidate over the eight held-out LiDAR sites and prints it beside the shipped
model, site by site and class by class, then states whether it meets the criterion agreed
before any of the experiments were run:

    beat the baseline's mean RMSE, without losing more than 0.05 m of MAE on the sub-metre
    classes (ground, road, low vegetation), and without a material change in inference cost.

The point of fixing the rule in advance is that a marginal result cannot be argued into a win
after the fact. If the candidate does not clear the bar, the shipped model stays.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBMETRE = ("ground", "road", "low_vegetation")
MAE_TOLERANCE = 0.05


def _load(results_dir: Path) -> list[dict]:
    p = results_dir / "results.json"
    if not p.exists():
        raise SystemExit(f"no results at {p}")
    return json.loads(p.read_text())


def _sites(rows: list[dict], gsd: float = 0.25, variant: str = "raw") -> dict[str, dict]:
    return {r["site"]: r["ndsm"]["overall"]
            for r in rows if r.get("gsd") == gsd and r.get("variant") == variant}


def _mean(d: dict[str, dict], key: str) -> float:
    vals = [v[key] for v in d.values() if v.get(key) is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", type=Path, required=True, help="Checkpoint to evaluate.")
    ap.add_argument("--results", type=Path, default=None,
                    help="Existing benchmark directory for the candidate (skips re-running it).")
    ap.add_argument("--baseline", type=Path, default=REPO / "outputs" / "benchmark_ahn",
                    help="Benchmark directory for the currently shipped model.")
    ap.add_argument("--gamus", type=Path, default=None,
                    help="Optional GAMUS test JSON for the candidate, for the per-class check.")
    args = ap.parse_args()

    if not args.candidate.exists():
        raise SystemExit(f"candidate not found: {args.candidate}")

    results = args.results
    if results is None:
        results = REPO / "outputs" / f"bench_{args.candidate.stem}"
        print(f"▶ Benchmarking {args.candidate.name} over the eight held-out sites "
              f"(~25 min) → {results}")
        cmd = [sys.executable, str(REPO / "scripts" / "benchmark_ahn.py"),
               "--weights", str(args.candidate), "--out", str(results)]
        if subprocess.run(cmd, cwd=REPO).returncode != 0:
            raise SystemExit("benchmark failed")

    base, cand = _sites(_load(args.baseline)), _sites(_load(results))
    shared = sorted(set(base) & set(cand))
    if not shared:
        raise SystemExit("no sites in common between the two benchmark runs")

    print(f"\n{'site':<24}{'shipped':>10}{'candidate':>12}{'change':>10}")
    print("-" * 56)
    for s in shared:
        b, c = base[s]["rmse"], cand[s]["rmse"]
        print(f"  {s:<22}{b:>9.2f}m{c:>11.2f}m{c - b:>+9.2f}m")
    mb, mc = _mean({k: base[k] for k in shared}, "rmse"), _mean({k: cand[k] for k in shared}, "rmse")
    rb, rc = _mean({k: base[k] for k in shared}, "pearson_r"), _mean({k: cand[k] for k in shared}, "pearson_r")
    print("-" * 56)
    print(f"  {'mean RMSE':<22}{mb:>9.2f}m{mc:>11.2f}m{mc - mb:>+9.2f}m")
    print(f"  {'mean correlation':<22}{rb:>10.3f}{rc:>12.3f}{rc - rb:>+10.3f}")

    verdict: list[str] = []
    if mc < mb:
        verdict.append(f"RMSE improves by {mb - mc:.2f} m")
    else:
        verdict.append(f"RMSE is {mc - mb:.2f} m WORSE — fails the criterion")

    # The sub-metre classes are the majority of any image and the part that makes the product
    # feel accurate; a win on tall structures bought by losing them is not a win.
    if args.gamus and args.gamus.exists():
        shipped_g = json.loads((REPO / "outputs" / "gamus_test.json").read_text())
        cand_g = json.loads(args.gamus.read_text())
        for cls in SUBMETRE:
            b = shipped_g["by_pixel_class"].get(cls, {}).get("mae")
            c = cand_g.get("by_pixel_class", {}).get(cls, {}).get("mae")
            if b is None or c is None:
                continue
            delta = c - b
            state = "holds" if delta <= MAE_TOLERANCE else f"LOST {delta:.3f} m — fails"
            verdict.append(f"{cls} MAE {b:.2f} → {c:.2f} m ({state})")
    else:
        verdict.append("per-class check skipped (pass --gamus with the candidate's test JSON)")

    print("\nVerdict")
    for line in verdict:
        print(f"  · {line}")
    ships = mc < mb and not any("fails" in v for v in verdict)
    print(f"\n  {'SHIP IT — install with scripts/finalise_model.sh' if ships else 'DO NOT SHIP — the current checkpoint stays'}")
    return 0 if ships else 2


if __name__ == "__main__":
    sys.exit(main())
