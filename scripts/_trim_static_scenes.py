#!/usr/bin/env python3
"""Drop every bundled scene except the ones named, from both the payload and the picker.

The public demo is bandwidth-bound: each scene pair is roughly 25 MB of mesh and texture,
so it ships a subset while a local install keeps them all.
"""
import re
import shutil
import sys
from pathlib import Path

stage, keep = Path(sys.argv[1]), set(sys.argv[2:])
for d in sorted((stage / "public").iterdir()):
    if not d.is_dir() or d.name == "benchmark":
        continue
    base = d.name[:-6] if d.name.endswith("-truth") else d.name
    if base not in keep:
        shutil.rmtree(d)

page = stage / "src" / "app" / "page.tsx"
t = page.read_text()
i = t.index("const BUNDLED_SCENES")
j = t.index("];", i) + 2
lines = []
for line in t[i:j].splitlines():
    m = re.search(r"id: '([^']+)'", line)
    if m and m.group(1) not in keep:
        continue
    lines.append(line)
page.write_text(t[:i] + "\n".join(lines) + t[j:])
print("   static demo keeps: " + ", ".join(sorted(keep)))
