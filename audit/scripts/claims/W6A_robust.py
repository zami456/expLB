"""W6A-thesis1 -- robustness sweeps over audit/phase2/layer_sensitivity.csv.

Used for O-070, O-071, O-072, O-073 (and as context for O-059): does the
claim survive other layers, poolings, seeds, probes and stores?

layer_sensitivity.csv holds off-diagonal / diagonal means per
(store, bank, k, pooling, layer, seed, dataset_set, feature, probe).
NOTE the known traps: seeds 0-4 carry only a reduced feature/probe list in
passA / passC / <root> (P3-D), and dataset_set is 'all' (10 or 11 datasets)
or 'paper5' (5) -- never the 9-dataset set that run-coords uses.

Usage: python audit/scripts/claims/W6A_robust.py
"""
from __future__ import annotations

import collections
import csv
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parents[3]
CSVP = ROOT / "audit/phase2/layer_sensitivity.csv"

rows = []
with open(CSVP, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        try:
            r["layer"] = int(r["layer"])
            r["offdiag_mean"] = float(r["offdiag_mean"])
            r["diag_mean"] = float(r["diag_mean"]) if r["diag_mean"] else None
        except (ValueError, TypeError):
            continue
        rows.append(r)

print("rows loaded:", len(rows), "from", CSVP)

# Trap check: `raw` and a bank store's own `datapca` must never be joined. Verify
# that no bank store even carries a `raw` row, so every join below is within a
# store that owns its raw baseline.
_st = collections.defaultdict(set)
for r in rows:
    _st[r["store"]].add(r["feature"])
print("\nTRAP CHECK -- which stores carry a `raw` baseline:")
for s in sorted(_st):
    print(f"   {s:<26} raw={'raw' in _st[s]:<5} axes={'axes' in _st[s]:<5} "
          f"pc:deception={'pc:deception' in _st[s]:<5} datapca={'datapca' in _st[s]}")
print("   -> no banks/* store has `raw`, so no axes-vs-raw join below can pick up a "
       "bank store's misleading column.")
print("TRAP CHECK -- P3-D: no multi-seed mean is computed anywhere in this script; "
      "every seed is reported as its own configuration row.")

KEY = ("store", "bank_name", "k", "pooling", "layer", "seed", "dataset_set",
       "n_datasets", "probe")
cells = collections.defaultdict(dict)
for r in rows:
    cells[tuple(str(r[k]) for k in KEY)][r["feature"]] = r["offdiag_mean"]


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ------------------------------------------------------------------- O-073
hdr("O-073  does pc:deception EVER beat raw at layers >= 20?")
tot = pos = 0
wins = []
for k, fv in cells.items():
    if "pc:deception" in fv and "raw" in fv and int(k[4]) >= 20:
        tot += 1
        d = fv["pc:deception"] - fv["raw"]
        if d > 0:
            pos += 1
            wins.append((d, k))
print(f"cells with both pc:deception and raw at layer >= 20: {tot}")
print(f"cells where pc:deception > raw: {pos}  ({pos/tot*100:.1f}%)")
wins.sort(reverse=True)
print("largest 12 pc-over-raw gains (gain, store, bank, k, pooling, layer, seed, "
      "dataset_set, n_datasets, probe):")
for d, k in wins[:12]:
    print(f"   {d:+.4f}  {k}")
diffs = [fv["pc:deception"] - fv["raw"] for k, fv in cells.items()
         if "pc:deception" in fv and "raw" in fv and int(k[4]) >= 20]
print(f"\nsigned pc-minus-raw over all {len(diffs)} cells: median {statistics.median(diffs):+.4f}, "
      f"mean {statistics.fmean(diffs):+.4f}, max {max(diffs):+.4f}, min {min(diffs):+.4f}")
# restricted to the paper-like setting: main bank family, 10/11 datasets
sub = [(k, fv["pc:deception"] - fv["raw"]) for k, fv in cells.items()
       if "pc:deception" in fv and "raw" in fv and int(k[4]) >= 20
       and k[6] == "all"]
print(f"restricted to dataset_set='all': n={len(sub)}, positive "
      f"{sum(1 for _, d in sub if d > 0)}, median {statistics.median([d for _, d in sub]):+.4f}")
print("k values present for pc:deception:",
      collections.Counter(r["k"] for r in rows if r["feature"] == "pc:deception"))

# ------------------------------------------------------------------- O-070
hdr("O-070  axes minus raw, best-vs-best, across stores (dataset_set='all')")
byset = collections.defaultdict(dict)   # (store,bank,pool,seed,probe,nds) -> layer -> feats
for k, fv in cells.items():
    if k[6] != "all":
        continue
    kk = (k[0], k[1], k[3], k[5], k[8], k[7])
    byset[kk][int(k[4])] = fv
print(f"{'store':<26} {'bank':<20} {'pool':<5} {'seed':<4} {'probe':<9} {'nds':<4} "
      f"{'raw best':<16} {'axes best':<16} {'axes-raw':>9}")
out = []
for kk, lay in sorted(byset.items()):
    ok = {L: f for L, f in lay.items() if "raw" in f and "axes" in f and L >= 20}
    if len(ok) < 3:
        continue
    braw = max(ok, key=lambda L: ok[L]["raw"])
    bax = max(ok, key=lambda L: ok[L]["axes"])
    d = ok[bax]["axes"] - ok[braw]["raw"]
    out.append(d)
    print(f"{kk[0]:<26} {kk[1]:<20} {kk[2]:<5} {kk[3]:<4} {kk[4]:<9} {kk[5]:<4} "
          f"L{braw:<3}{ok[braw]['raw']:.4f}      L{bax:<3}{ok[bax]['axes']:.4f}      "
          f"{d:>+9.4f}")
if out:
    print(f"\nbest-vs-best axes-raw over {len(out)} configurations: "
          f"median {statistics.median(out):+.4f}, positive "
          f"{sum(1 for d in out if d > 0)}/{len(out)}")

# ------------------------------------------------------------- O-071 / O-072
hdr("O-071 / O-072  axes vs raw at the same layer, layers >= 20")
same = [(k, fv["axes"] - fv["raw"]) for k, fv in cells.items()
        if "axes" in fv and "raw" in fv and int(k[4]) >= 20]
print(f"n cells = {len(same)}; axes > raw in {sum(1 for _, d in same if d > 0)} "
      f"({sum(1 for _, d in same if d>0)/len(same)*100:.1f}%); "
      f"median {statistics.median([d for _, d in same]):+.4f}")
for L in (20, 28, 32, 36, 40, 44, 48, 60):
    s = [d for k, d in same if int(k[4]) == L]
    if s:
        print(f"   layer {L:<3} n={len(s):<4} median {statistics.median(s):+.4f} "
              f"positive {sum(1 for d in s if d>0)}/{len(s)}")

print("\nDONE")
