"""Re-derive the load-bearing ledger corrections for F-123 (6D-4 / Phase 6).

DF-71  F-61's '+0.006' and 'twelvefold'  -> +0.0093 and 8.1x
DF-42  F-26/F-33/F-73's '0.553 vs 0.665' -> 0.5555 / 0.6623  (11 ds) and 0.5494 / 0.6652 (10 ds)
DF-04  F-73's 'confirmed to three decimals'

Source: run-crossgen/kd11/matrices/mean_L40_s42.json (bank paper16, probe 'paper', k=5, seed 42).
Also demonstrates the DF-71 bug live: d['rows'] is a 22-char caption, not a label list.
"""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
M = ROOT / "run-crossgen/kd11/matrices/mean_L40_s42.json"
d = json.loads(M.read_text())
ds, block9 = d["datasets"], d["block9"]

def offdiag(feature, probe="paper", labels=None):
    labels = labels or ds
    keep = [i for i, name in enumerate(ds) if name in labels]
    for r in d["results"]:
        if r["feature"] == feature and r["probe"] == probe:
            a = r["auroc"]
            cells = [a[i][j] for i in keep for j in keep if i != j]
            return sum(cells) / len(cells), len(cells)
    raise KeyError(feature)

print(f"file      : {M.relative_to(ROOT)}")
print(f"bank      : {d['bank_name']} ({d['bank']}), layer {d['layer']}, "
      f"{d['pooling']} pooling, seed {d['seed']}, k={d['k']}")
print(f"DF-71 bug : d['rows'] = {d['rows']!r} -> {len(d['rows'])} chars, "
      f"NOT a label list. zip() over this iterates CHARACTERS.\n")

feats = sorted({r["feature"] for r in d["results"] if r["probe"] == "paper"})
print("features with probe='paper':", feats, "\n")

print("--- DF-71 / DF-42 : 11 datasets (all) ---")
for f in ("raw", "datapca", "length", "pc:unified"):
    if f in feats:
        m, n = offdiag(f)
        print(f"  {f:12s} off-diagonal mean = {m:.4f}   ({n} pairs)")
print("\n--- 10 datasets (block9 = all but sycophancy_mt) ---")
for f in ("raw", "datapca", "length", "pc:unified"):
    if f in feats:
        m, n = offdiag(f, labels=block9)
        print(f"  {f:12s} off-diagonal mean = {m:.4f}   ({n} pairs)")

if "pc:unified" in feats and "raw" in feats:
    u11, _ = offdiag("pc:unified"); r11, _ = offdiag("raw"); p11, _ = offdiag("datapca")
    print(f"\nDF-71: pc:unified - raw     = {u11 - r11:+.4f}   (F-61 printed +0.006)")
    print(f"       pc:unified - datapca = {u11 - p11:+.4f}")
    print(f"       0.0757 / {u11-r11:.4f} = {0.0757/(u11-r11):.1f}x   <- like-for-like vs raw")
    print(f"       0.0757 / {u11-p11:.4f} = {0.0757/(u11-p11):.1f}x   <- the mixed ratio F-61 printed")

# DF-71's buggy path, reproduced: zip over the caption string
buggy_labels = list(d["rows"])
print(f"\nDF-71 buggy path: zip(datasets, list(d['rows'])) pairs "
      f"{len(ds)} datasets against {len(buggy_labels)} CHARACTERS "
      f"-> diagonal test tr==te compares letters.")
