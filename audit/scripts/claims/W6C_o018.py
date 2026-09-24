"""W6C-briefing / O-018: was the withdrawn "0.811 on the same six" the knowledge-denial bank?

The current briefing says:
  "We previously reported the +0.023 as the headline and said our own contrast 'loses by
   0.082' on the same six. Both statements mixed two dataset sets and two banks -- the 0.811
   we quoted is our *knowledge-denial* bank, not a register axis."

The withdrawn version (commit aea9f53, 2026-09-21 21:25) stated 0.811 on the six and 0.733
on all eleven, zero-shot contrast, L40. This script reads the zero-shot block of every
locally available L40 matrix and finds which bank produces 0.811 / 0.733.

Local git working-tree copies are used (the lab PC is unreachable this session); the md5 of
each file is printed so the copy can be identified, and the kd11 root file is compared with
the audit's cached lab copy.
"""
import hashlib
import io
import json
import os
import sys

p = sys.stdout.write

SIX = ["instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
       "convincing-game", "insider-trading", "ai_liar"]

STORES = [
    ("kd11 root (paper16 bank)", "run-crossgen/kd11/matrices"),
    ("kd11/banks/kd", "run-crossgen/kd11/banks/kd/matrices"),
    ("kd11/banks/kd_s23", "run-crossgen/kd11/banks/kd_s23/matrices"),
    ("run-crossgen root", "run-crossgen/matrices"),
    ("banks/paper16_shared", "run-crossgen/banks/paper16_shared/matrices"),
    ("banks/paper16_famcentred", "run-crossgen/banks/paper16_famcentred/matrices"),
    ("banks/own100", "run-crossgen/banks/own100/matrices"),
    ("banks/hf100", "run-crossgen/banks/hf100/matrices"),
]


def load(path):
    with io.open(path, "rb") as f:
        b = f.read()
    return json.loads(b.decode("utf-8")), hashlib.md5(b).hexdigest(), len(b)


def subset_mean(names, aur, keep):
    v = [a for n, a in zip(names, aur) if n in keep and a is not None]
    return (sum(v) / len(v), len(v)) if v else (None, 0)


for label, d in STORES:
    f = os.path.join(d, "mean_L40_s42.json")
    if not os.path.exists(f):
        p("  %-28s  (no mean_L40_s42.json)\n" % label)
        continue
    m, md5, n = load(f)
    ds = m["datasets"]
    p("\n== %-26s %s\n   md5 %s  %d B  bank=%s bank_name=%s k=%s  datasets=%d\n"
      % (label, f, md5, n, m.get("bank"), m.get("bank_name"), m.get("k"), len(ds)))
    have6 = [x for x in SIX if x in ds]
    p("   of the six overlapping datasets, this store has %d: %s\n"
      % (len(have6), ", ".join(have6)))
    for z in m.get("zeroshot", []):
        allm, na = subset_mean(ds, z["auroc"], set(ds))
        sixm, ns = subset_mean(ds, z["auroc"], set(SIX))
        p("   %-34s all(%2d) %.4f   six(%d) %s\n"
          % (z["direction"], na, allm, ns,
             ("%.4f" % sixm) if sixm is not None else "-"))

p("\n--- cross-check: local kd11 root matrix vs the audit's cached LAB copy ---\n")
a = load("run-crossgen/kd11/matrices/mean_L40_s42.json")
b = load("audit/cache/labpc/run-crossgen/kd11/matrices/mean_L40_s42.json")
p("  local md5 %s (%d B)\n  lab   md5 %s (%d B)   %s\n"
  % (a[1], a[2], b[1], b[2], "IDENTICAL" if a[1] == b[1] else "DIFFERENT"))
if a[1] != b[1]:
    za = {z["direction"]: z["auroc"] for z in a[0]["zeroshot"]}
    zb = {z["direction"]: z["auroc"] for z in b[0]["zeroshot"]}
    for k in sorted(set(za) | set(zb)):
        if k in za and k in zb:
            d = max(abs(x - y) for x, y in zip(za[k], zb[k]))
            p("    zeroshot %-20s max|diff| %.6f\n" % (k, d))
        else:
            p("    zeroshot %-20s present only in %s\n" % (k, "local" if k in za else "lab"))
