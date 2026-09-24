#!/usr/bin/env python
"""M1 F-42/F-43 pass 2 -- resolve what the log actually counts."""
import re
from collections import Counter
from pathlib import Path

R = Path("C:/Users/user4/expLB")
LOG = R / "logs" / "scale_coords_8b.log"
text = LOG.read_text(encoding="utf-8", errors="replace")
lines = text.splitlines()
TS = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})")
stamped = [(i, m.group(1), m.group(2), l) for i, l in enumerate(lines) if (m := TS.match(l))]


def norm(l):
    s = re.sub(r"^\S+ \S+ - \S+ - ", "", l).strip()
    s = re.sub(r"\d+", "#", s)
    return s[:110]


def win(t0, t1, d="2026-09-20"):
    return [x for x in stamped if x[1] == d and t0 <= x[2] <= t1]


for name, t0, t1 in [("F41 17:32:29-17:48:37", "17:32:29", "17:48:37"),
                     ("F42 17:48:38-18:02:06", "17:48:38", "18:02:06"),
                     ("F43 18:02:06-18:04:20", "18:02:06", "18:04:20"),
                     ("post-1024 18:04:20-20:00", "18:04:20", "20:00:00")]:
    w = win(t0, t1)
    print("=" * 78)
    print(f"{name}   lines={len(w)}   first idx {w[0][0] if w else '-'}  last {w[-1][0] if w else '-'}")
    c = Counter(norm(l) for _, _, _, l in w)
    for k, v in c.most_common(18):
        print(f"   {v:5}  {k}")
    att = Counter()
    for _, _, _, l in w:
        m = re.search(r"session attempt (\d)/(\d)", l)
        if m:
            att[m.group(0)] += 1
    print("   session-attempt counts:", dict(sorted(att.items())))
    oom = sum(1 for _, _, _, l in w if "OutOfMemoryError" in l)
    print("   lines containing OutOfMemoryError:", oom)
    perm = [l for _, _, _, l in w if re.search(r"permanent|gave up|giving up|unrecoverable|dropped",
                                               l, re.I)]
    print("   'permanent'-ish lines:", len(perm))
    for l in perm[:6]:
        print("      ", l[:170])
    print()

print("=" * 78)
print("ALL lines anywhere in the log mentioning 'permanent'")
for i, d, t, l in stamped:
    if "permanent" in l.lower():
        print(f"  [{i}] {d} {t} {l[:200]}")
print()
print("=" * 78)
print("Lines with 'skipped' / 'failed' summary per dataset, 17:30-18:10")
for i, d, t, l in stamped:
    if d == "2026-09-20" and "17:30" <= t <= "18:10" and re.search(
            r"wrote|chunk|summary|skipp|failed .*rows|rows failed|Done", l, re.I):
        print(f"  [{i}] {t} {l[:190]}")
