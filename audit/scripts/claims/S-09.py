"""S-09: the first Q2 signal (3 datasets, the 'axes residualised' arm).
Reads run-coords/smoke.json, which the session's table was printed from, and
recomputes the off-diagonal means from the per-cell matrices it stores."""
import json, statistics
d = json.load(open("run-coords/smoke.json"))
print("layer", d["layer"], "pooling", d["pooling"], "k", d["k"], "bank", d["bank"])
print("datasets:", d["datasets"])
print("label-shuffle null:", d["null"])
print()
print(f"{'features':16s}{'probe':10s}{'diag':>8}{'offdiag(file)':>15}{'offdiag(recomputed)':>21}{'n':>4}")
for r in d["p1"]:
    m = r["matrix"]
    off = [v for k, v in m.items() if k.split("->")[0] != k.split("->")[1]]
    print(f"{r['features']:16s}{r['probe']:10s}{r['diagonal']:8.4f}{r['off_diagonal']:15.4f}"
          f"{statistics.fmean(off):21.4f}{len(off):4d}")
print()
print("per-cell matrix, axes_resid / paper:")
for r in d["p1"]:
    if r["features"].startswith("axes_resid") and r["probe"] == "paper":
        for k, v in r["matrix"].items():
            print(f"   {k:70s} {v:.4f}")
