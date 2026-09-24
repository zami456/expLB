#!/usr/bin/env python
"""M1 claim checks F-42 / F-43 -- the NDIF attention-memory rule, read from the raw logs.

Counts OOM and permanent-failure lines in the two windows F-41->F-42 and F-42->F-43 names,
finds the capacity line the per-process-cap diagnosis rests on, tracks the size of the
attempted allocations, and checks the post-1024 window for OOMs.

READ-ONLY.  Run on the lab PC.
"""
import re
import sys
from collections import Counter
from pathlib import Path

R = Path("C:/Users/user4/expLB")
LOG = R / "logs" / "scale_coords_8b.log"

raw = LOG.read_bytes()
enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
text = raw.decode(enc, "replace")
lines = text.splitlines()
print(f"{LOG}  {len(raw):,} B  encoding={enc}  lines={len(lines)}")

TS = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})")
stamped = []
for i, l in enumerate(lines):
    m = TS.match(l)
    if m:
        stamped.append((i, m.group(1), m.group(2), l))
print(f"timestamped lines: {len(stamped)}")
print(f"dates present    : {sorted(Counter(d for _, d, _, _ in stamped).items())}")
print(f"first: {stamped[0][3][:130]}")
print(f"last : {stamped[-1][3][:130]}")
print()

# ---- markers -------------------------------------------------------------------
pat_oom = re.compile(r"out of memory|OutOfMemory|CUDA out of memory", re.I)
pat_perm = re.compile(r"attempt 4/4|permanently|giving up|FAILED after", re.I)
pat_alloc = re.compile(r"Tried to allocate ([\d.]+) ([KMG]iB)")
pat_cap = re.compile(r"total capacity of ([\d.]+) GiB of which ([\d.]+) GiB is free")
pat_chunk = re.compile(r"chunk_(\d+)\.pt|wrote .*chunk")
pat_start = re.compile(r"START|attn_budget|max_length|session_batches|session_tokens", re.I)

print("== lines matching START / attn_budget / max_length / session_* ==")
for i, d, t, l in stamped:
    if pat_start.search(l) and ("layers:" in l or "budget" in l.lower() or "START" in l
                                or "max_length" in l):
        print(f"  [{i}] {l[:180]}")
print()


def window(d0, t0, t1):
    return [(i, d, t, l) for i, d, t, l in stamped if d == d0 and t0 <= t <= t1]


def report(name, w):
    ooms = [x for x in w if pat_oom.search(x[3])]
    perms = [x for x in w if pat_perm.search(x[3])]
    allocs = [pat_alloc.search(x[3]) for x in w]
    allocs = [a for a in allocs if a]
    mib = []
    for a in allocs:
        v = float(a.group(1))
        mib.append(v * 1024 if a.group(2) == "GiB" else v / 1024 if a.group(2) == "KiB" else v)
    print(f"-- {name}: {len(w)} timestamped lines")
    print(f"     OOM lines              : {len(ooms)}")
    print(f"     permanent-failure lines: {len(perms)}")
    if mib:
        print(f"     'Tried to allocate' n={len(mib)}  first {mib[0]:.2f} MiB  "
              f"last {mib[-1]:.2f} MiB  min {min(mib):.2f}  max {max(mib):.2f}")
    caps = [pat_cap.search(x[3]) for x in w]
    caps = [c for c in caps if c]
    if caps:
        s = {(c.group(1), c.group(2)) for c in caps}
        print(f"     capacity/free pairs    : {sorted(s)}")
    return ooms, perms


print("== F-41 window (restart at attn 0.75; entry: 57 OOMs, 14 permanent in 14 min) ==")
report("2026-09-20 17:32:00-17:48:37", window("2026-09-20", "17:32:00", "17:48:37"))
print()
print("== F-42 window (attn 0.55; entry: 50 OOMs, 12 permanent, 902 -> 438 MiB) ==")
report("2026-09-20 17:48:34-18:02:06", window("2026-09-20", "17:48:34", "18:02:06"))
print()
print("== F-43 window (session_batches 1 then max_length 1024; entry: 0 OOMs after 18:04) ==")
report("2026-09-20 18:02:00-18:04:20", window("2026-09-20", "18:02:00", "18:04:20"))
report("2026-09-20 18:04:20-19:30:00", window("2026-09-20", "18:04:20", "19:30:00"))
print()

print("== every 'total capacity' line in the file ==")
for i, d, t, l in stamped:
    if pat_cap.search(l):
        print(f"  [{i}] {d} {t}  {l[:240]}")
print()

print("== every line mentioning 'padded tokens' with '1 batches' ==")
for i, d, t, l in stamped:
    if "padded tokens" in l and re.search(r"\b1 batches\b", l):
        print(f"  [{i}] {d} {t}  {l[:200]}")
print()

print("== harm-pressure-knowledge-report chunk writes on 2026-09-20 17:30-17:50 ==")
for i, d, t, l in stamped:
    if d == "2026-09-20" and "17:3" in t and ("chunk" in l.lower()):
        print(f"  [{i}] {t}  {l[:200]}")
print()

print("== run_coords_chain.ps1 passB invocation ==")
p = R / "src" / "assistant_axis_ndif" / "run_coords_chain.ps1"
if p.exists():
    src = p.read_text(encoding="utf-8", errors="replace").splitlines()
    for n, l in enumerate(src, 1):
        if 80 <= n <= 105:
            print(f"  {n:4} {l.rstrip()[:160]}")
    print("  --- occurrences of attn_budget_gib in the file:",
          sum("attn_budget_gib" in l for l in src))
else:
    print("  NOT FOUND", p)
print()

print("== scale/scripts/run_scale_coords.ps1 max_length / attn settings ==")
for cand in (R / "scale" / "scripts" / "run_scale_coords.ps1",):
    if cand.exists():
        for n, l in enumerate(cand.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if re.search(r"max_length|attn_budget|session_batches|session_tokens|n_heads|8_coords",
                         l):
                print(f"  {n:4} {l.rstrip()[:180]}")
    else:
        print("  NOT FOUND", cand)
