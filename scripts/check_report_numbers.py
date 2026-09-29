#!/usr/bin/env python3
"""
Check that the figures quoted in the documentation still match the measured results.

    python scripts/check_report_numbers.py          # exits non-zero if anything has drifted

Every accuracy claim in PROJECT_REPORT.md and README.md is supposed to come from a file in
outputs/. Documentation drifts silently — a model is retrained, a benchmark is re-run, and a
table keeps the old number — and the failure is invisible right up until a reviewer checks one.
This reads the tables back out of the Markdown and compares them against the JSON they came
from, so the drift is caught here instead of there.

It checks what can be tied to a source unambiguously: the per-site LiDAR table, the per-surface
GAMUS table, and the headline figures. Prose is left alone.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOL = 0.006  # figures are quoted to two decimals


def _num(cell: str) -> float | None:
    """Pull a number out of a table cell, tolerating units and Unicode minus signs."""
    cell = cell.replace("−", "-").replace("m", "").strip()
    m = re.match(r"^\*{0,2}(-?\d+(?:\.\d+)?)\*{0,2}$", cell)
    return float(m.group(1)) if m else None


# The same site appears in several tables measured over different extents — the eight-site
# benchmark uses 400 m sites, the bundled demo scenes 600 m — so matching on the site name
# alone would flag correct figures as drift. Only the canonical tables are checked, identified
# by their header row.
SITE_HEADER = "predict-zero"
CLASS_HEADER = "surface"


def _tables(text: str) -> list[tuple[list[str], list[list[str]]]]:
    """Return (header, rows) for every Markdown table in *text*."""
    tables, header, rows = [], None, []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue                      # separator row
            if header is None:
                header = cells
            else:
                rows.append(cells)
        else:
            if header is not None:
                tables.append((header, rows))
            header, rows = None, []
    if header is not None:
        tables.append((header, rows))
    return tables


def main() -> int:
    # benchmark_ahn is what scripts/finalise_model.sh rewrites when a model is installed, so
    # it is canonical. Preferring benchmark_gsd here meant that after installing a new head
    # this script happily validated the documentation against the PREVIOUS model's results
    # and reported no drift — the precise failure it exists to catch.
    bench_path = REPO / "outputs" / "benchmark_ahn" / "results.json"
    if not bench_path.exists():
        bench_path = REPO / "outputs" / "benchmark_gsd" / "results.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else []
    ckpt = REPO / "checkpoints" / "decoder" / "decoder_best.pth"
    if bench_path.exists() and ckpt.exists() and bench_path.stat().st_mtime < ckpt.stat().st_mtime:
        print(f"  STALE  {bench_path.relative_to(REPO)} predates the installed checkpoint — "
              f"re-run scripts/benchmark_ahn.py before trusting this")
        return 1
    other = REPO / "outputs" / "benchmark_gsd" / "results.json"
    if other.exists() and ckpt.exists() and other.stat().st_mtime < ckpt.stat().st_mtime:
        print(f"  note: {other.relative_to(REPO)} is older than the checkpoint; any figure "
              f"quoted from it is stale")
    gamus_path = REPO / "outputs" / "gamus_test.json"
    gamus = json.loads(gamus_path.read_text()) if gamus_path.exists() else {}

    sites = {}
    for r in bench:
        if r.get("gsd") == 0.25 and r.get("variant") == "raw":
            o = r["ndsm"]["overall"]
            z = (r["ndsm"].get("zero_baseline") or {}).get("rmse")
            sites[r["site"].replace("_", " ")] = (o["rmse"], o["mae"], o["bias"],
                                                 o.get("pearson_r"), z)
    classes = {k.replace("_", " "): (v["rmse"], v["mae"], v["bias"])
               for k, v in (gamus.get("by_pixel_class") or {}).items()}

    checked = failed = 0
    for doc in ("PROJECT_REPORT.md", "README.md"):
        path = REPO / doc
        if not path.exists():
            continue
        for header, rows in _tables(path.read_text()):
            head = " ".join(header).lower()
            first = header[0].strip().lower()
            # "Site" is the eight-site benchmark at 400 m; "Scene" is the bundled demo pairs at
            # 600 m, which carry the same column names but different measurements.
            if SITE_HEADER in head and first == "site":
                lookup = sites
            elif head.startswith(CLASS_HEADER):
                lookup = classes
            else:
                continue                      # not a table tied to a measured source
            for cells in rows:
                key = cells[0].strip().strip("`*").lower().replace("_", " ")
                key = re.sub(r"\s*\(incl\. road\)", "", key)
                source = lookup.get(key)
                if source is None:
                    continue
                quoted = [q for q in (_num(c) for c in cells[1:]) if q is not None]
                remaining = [v for v in source if v is not None]
                for q in quoted:
                    if not remaining:
                        break
                    if any(abs(q - v) < TOL for v in remaining):
                        remaining = [v for v in remaining if abs(q - v) >= TOL]
                        checked += 1
                    else:
                        print(f"  DRIFT  {doc}: '{cells[0]}' quotes {q}, not among "
                              f"{[round(v, 3) for v in source if v is not None]}")
                        failed += 1
                        break

    print(f"\n  {checked} figures verified against {bench_path.name} and "
          f"{gamus_path.name}, {failed} drifted")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
