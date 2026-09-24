#!/usr/bin/env python
"""M1 F-42/F-43 pass 3 -- exact windows, allocation sequences, chunk writes, config lines."""
import re
from pathlib import Path

R = Path("C:/Users/user4/expLB")
LOG = R / "logs" / "scale_coords_8b.log"
lines = LOG.read_text(encoding="utf-8", errors="replace").splitlines()
TS = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})")
st = [(i, m.group(1), m.group(2), l) for i, l in enumerate(lines) if (m := TS.match(l))]

ALLOC = re.compile(r"Tried to allocate ([\d.]+) ([KMG]i?B)")
PERM = "failed permanently"


def mib(m):
    v = float(m.group(1))
    return v * 1024 if m.group(2).startswith("G") else v / 1024 if m.group(2).startswith("K") else v


print("== find the window that gives exactly 57 OOM / 14 permanent (F-42's claim) ==")
perm_t = [t for _, d, t, l in st if d == "2026-09-20" and PERM in l]
for start in ("17:32:00", "17:32:29", "17:32:30"):
    for end in perm_t:
        if end < start:
            continue
        w = [x for x in st if x[1] == "2026-09-20" and start <= x[2] <= end]
        o = sum(1 for x in w if "OutOfMemoryError" in x[3])
        p = sum(1 for x in w if PERM in x[3])
        if p in (13, 14, 15) and 54 <= o <= 60:
            print(f"   start {start} end {end}: OOM {o}  perm {p}   span "
                  f"{int(end[:2])*60+int(end[3:5]) - (int(start[:2])*60+int(start[3:5]))} min")
print()

print("== allocation sequence, 17:32:29 -> 18:04:20 (MiB, in time order) ==")
seq = []
for i, d, t, l in st:
    if d != "2026-09-20" or not ("17:32:29" <= t <= "18:04:20"):
        continue
    m = ALLOC.search(l)
    if m:
        seq.append((t, round(mib(m), 1)))
for k in range(0, len(seq), 8):
    print("   " + "  ".join(f"{t} {v:7.1f}" for t, v in seq[k:k + 8]))
print()
print("== distinct allocation sizes per phase ==")
for name, t0, t1 in [("<=17:32:28 (pre-F41-restart)", "00:00:00", "17:32:28"),
                     ("F41 17:32:29-17:48:37 (attn .75)", "17:32:29", "17:48:37"),
                     ("F42 17:48:38-18:02:05 (attn .55)", "17:48:38", "18:02:05"),
                     ("F43a 18:02:06-18:04:19 (sess 1)", "18:02:06", "18:04:19")]:
    vals = [round(mib(m), 1) for i, d, t, l in st
            if d == "2026-09-20" and t0 <= t <= t1 and (m := ALLOC.search(l))]
    if vals:
        print(f"   {name:36} n={len(vals):3} max {max(vals):8.1f}  min {min(vals):8.1f}  "
              f"first {vals[0]:8.1f}  last {vals[-1]:8.1f}")
        print(f"      distinct: {sorted(set(vals), reverse=True)[:14]}")
print()

print("== '5250 padded tokens' / '1 batches' lines ==")
for i, d, t, l in st:
    if "5250" in l or re.search(r"\(1 batches, \d+ padded tokens\)", l):
        print(f"   [{i}] {d} {t} {l[:210]}")
print()

print("== the 18:02:46 capacity line, full text ==")
for i, d, t, l in st:
    if d == "2026-09-20" and "18:02:4" <= t <= "18:02:50":
        print(f"   [{i}] {t} {l[:400]}")
print()

print("== lines 17:33:00-17:35:00 (chunk writes F-42 claims) ==")
for i, d, t, l in st:
    if d == "2026-09-20" and "17:33:00" <= t <= "17:35:00" and "HTTP Request" not in l:
        print(f"   [{i}] {t} {l[:190]}")
print()

print("== chunk_*.pt mtimes in run-scale/8b/coords/harm-pressure-knowledge-report ==")
import datetime
for base in (R / "run-scale" / "8b" / "coords",):
    if not base.exists():
        print("   MISSING", base); break
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        ch = sorted(d.glob("chunk_*.pt"))
        rows = d / "rows.jsonl"
        nrow = sum(1 for _ in rows.open(encoding="utf-8")) if rows.exists() else None
        it = d / "items.pt"
        print(f"   {d.name:34} chunks {len(ch):4}  rows.jsonl {nrow}  "
              f"items.pt {it.stat().st_size if it.exists() else '-'} "
              f"({datetime.datetime.fromtimestamp(it.stat().st_mtime) if it.exists() else '-'})")
print()

print("== '# batches, widest attention' + 'sessions, widest' config lines, 17:00-18:10 ==")
for i, d, t, l in st:
    if d == "2026-09-20" and "17:00" <= t <= "18:10" and (
            "widest" in l or "cached spans" in l or "to do" in l or "layers:" in l):
        print(f"   [{i}] {t} {l[:190]}")
