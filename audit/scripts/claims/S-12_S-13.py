"""S-12 / S-13: composition-dependence of the unified-axis margin, and the block
structure / inverted-cell count of the raw baseline, from the crossgen L40 matrix.
Read-only. Usage: python S-12_S-13.py [matrix.json]"""
import json, sys, itertools
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "run-crossgen/matrices/mean_L40_s42.json"
d = json.load(open(path))
DS = d["datasets"]; n = len(DS)
print("file:", path)
print("pooling", d["pooling"], "layer", d["layer"], "seed", d["seed"], "bank", d["bank"], "k", d["k"], "frac_train", d["frac_train"])
print("datasets (%d):" % n, DS)

M = {}
for r in d["results"]:
    M[(r["feature"], r["probe"])] = np.array(r["auroc"], dtype=float)

INSTRUCTED = ["instructed-deception", "harm-pressure-knowledge-report", "harm-pressure-choice",
              "insider-trading", "convincing-game", "ai_liar", "sandbagging_v2"]  # 18_q5_semantics.py SEMANTICS
idx = {ds: i for i, ds in enumerate(DS)}

def offdiag(mat, keep):
    ks = [idx[x] for x in keep]
    vals = [mat[i, j] for i in ks for j in ks if i != j]
    return float(np.mean(vals)), len(vals)

comps = {
    "all10": DS,
    "no_sycophancy_mt": [x for x in DS if x != "sycophancy_mt"],
    "instructed7": INSTRUCTED,
}
print("\n=== S-12: off-diagonal means by composition ===")
for probe in ["paper", "lr_reg10", "mms"]:
    print(f"\n-- probe = {probe}")
    for cname, keep in comps.items():
        raw, nc = offdiag(M[("raw", probe)], keep)
        uni, _ = offdiag(M[("pc:unified@2", probe)], keep)
        dpc, _ = offdiag(M[("datapca@2", probe)], keep)
        rnd, _ = offdiag(M[("random@2", probe)], keep)
        print(f"   {cname:18s} cells={nc:3d}  raw {raw:.4f}  pc:unified@2 {uni:.4f}  "
              f"(uni-raw {uni-raw:+.4f})  datapca@2 {dpc:.4f} (uni-datapca {uni-dpc:+.4f})  "
              f"random@2 {rnd:.4f} (uni-random {uni-rnd:+.4f})")

print("\n=== S-13: sycophancy row / column deltas (pc:unified@2 - raw), probe=paper ===")
P = "paper"
U, R = M[("pc:unified@2", P)], M[("raw", P)]
s = idx["sycophancy_mt"]
row = float(np.mean([U[s, j] - R[s, j] for j in range(n) if j != s]))
col = float(np.mean([U[i, s] - R[i, s] for i in range(n) if i != s]))
print(f"   sycophancy_mt ROW (train syc -> others, 9 cells): {row:+.4f}")
print(f"   sycophancy_mt COL (others -> syc, 9 cells):       {col:+.4f}")
print(f"   row incl. diagonal: {float(np.mean(U[s]-R[s])):+.4f}   col incl. diagonal: {float(np.mean(U[:,s]-R[:,s])):+.4f}")

print("\n=== S-13: below-chance cells in the RAW matrix, probe=paper ===")
for label, mask in [("all 100 cells", None), ("off-diagonal 90", "off")]:
    cells = [(i, j) for i in range(n) for j in range(n)
             if (mask is None or i != j)]
    below = [(i, j, R[i, j]) for i, j in cells if R[i, j] < 0.5]
    print(f"   {label}: {len(below)} cells with AUROC < 0.5")
below = sorted([(R[i, j], DS[i], DS[j]) for i in range(n) for j in range(n) if R[i, j] < 0.5])
print("   worst 6:")
for v, a, b in below[:6]:
    print(f"      {a} -> {b}: {v:.4f}")

# "significantly inverted": bootstrap over the column dataset's rows is not available in
# this file, so use the binomial test implied by the column dataset's n (unknown here) ->
# instead report the count under several fixed thresholds, and the Mann-Whitney normal
# approximation is impossible without row scores. Report thresholds.
for t in [0.45, 0.40, 0.35, 0.30]:
    print(f"   cells with AUROC < {t}: {sum(1 for v,_,_ in below if v < t)}")

print("\n=== other probes: below-chance count in raw ===")
for probe in ["lr_reg10", "mms"]:
    Rp = M[("raw", probe)]
    c_all = int((Rp < 0.5).sum()); c_off = int(sum(1 for i in range(n) for j in range(n) if i != j and Rp[i, j] < 0.5))
    mn = np.unravel_index(np.argmin(Rp), Rp.shape)
    print(f"   {probe}: {c_all} of 100 ({c_off} off-diagonal); worst {DS[mn[0]]} -> {DS[mn[1]]} {Rp[mn]:.4f}")
