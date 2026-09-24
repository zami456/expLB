"""F-04 - 'Almost nothing clears a dataset-level bootstrap'.

Claim (kd/docs/FINDINGS.md:77-85):
  Across 384 significance tests in the 11-dataset paper16 run, 1 is significant at dataset
  level after BH correction.  At L40 / mean / paper probe: axes-raw = +0.015, dataset 95% CI
  [-0.170,+0.187], q = 0.931; axes-datapca = +0.026, q = 0.855.
  source: run-crossgen/kd11/significance.json (10,000 dataset bootstraps)

What this script does, from the raw test list only:
  1. reports the shape of the family actually stored (tests, blocks, layers, poolings, probes,
     comparisons) and what 'block' means (pairs -> number of datasets);
  2. reads the two quoted tests and their stored q;
  3. RE-DERIVES the BH q-values over candidate sub-families, to find which family reproduces
     q = 0.931 / 0.855 and '1 significant' -- i.e. reconstructs the 384-test run-1 family
     that the log describes (layers 40+44 x mean+last x 3 probes x 16 comparisons x 2 blocks).

Run on the git/laptop copy and, unchanged, on the lab copy.
"""
import json, os, sys
from collections import Counter

PATH = sys.argv[1] if len(sys.argv) > 1 else "run-crossgen/kd11/significance.json"
d = json.load(open(PATH, encoding="utf-8"))
T = d["tests"]
print("file :", os.path.abspath(PATH))
print("size :", os.path.getsize(PATH), "B")
print("meta : boot_datasets =", d.get("boot_datasets"), "| note:", d.get("note"))
print("n tests:", len(T))

blocks = sorted({t["block"] for t in T})
layers = sorted({t["layer"] for t in T})
pool = sorted({t["pooling"] for t in T})
probes = sorted({t["probe"] for t in T})
comps = sorted({t["comparison"] for t in T})
print("blocks  :", blocks)
print("layers  :", layers)
print("poolings:", pool)
print("probes  :", probes)
print("comparisons (%d):" % len(comps), comps)
for b in blocks:
    pr = sorted({t["pairs"] for t in T if t["block"] == b})
    print(f"  block {b!r}: n={sum(1 for t in T if t['block']==b)}  pairs values={pr}"
          f"   -> n_datasets = {[int(round((1+(1+4*p)**0.5)/2)) for p in pr]} (pairs = n*(n-1))")

print()
print("=== the two quoted tests (block 9, L40, mean, paper) ===")
for comp in ("axes - raw", "axes - datapca"):
    for t in T:
        if (t["block"] == "9" and t["layer"] == 40 and t["pooling"] == "mean"
                and t["probe"] == "paper" and t["comparison"] == comp):
            print(f"  {comp:16} diff={t['diff']:+.4f}  dataset_ci95="
                  f"[{t['dataset_ci95'][0]:+.4f},{t['dataset_ci95'][1]:+.4f}]  "
                  f"dataset_p={t['dataset_p']:.4f}  q_bh(stored)={t['dataset_q_bh']:.4f}  "
                  f"pairs={t['pairs']}")

print()
nsig = sum(1 for t in T if t.get("significant_datasets"))
print(f"significant_datasets == True : {nsig} of {len(T)}")
print("per block:", {b: sum(1 for t in T if t["block"] == b and t.get("significant_datasets")) for b in blocks})


def bh(ps):
    """Benjamini-Hochberg q-values, same convention as statsmodels fdr_bh."""
    n = len(ps)
    order = sorted(range(n), key=lambda i: ps[i])
    q = [0.0] * n
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        v = min(prev, ps[i] * n / rank)
        q[i] = v
        prev = v
    return q


def famq(sub, label):
    """BH over a sub-family; print the q of the two quoted tests and the count significant."""
    ps = [t["dataset_p"] for t in sub]
    qs = bh(ps)
    got = {}
    for t, q in zip(sub, qs):
        if (t["block"] == "9" and t["layer"] == 40 and t["pooling"] == "mean"
                and t["probe"] == "paper" and t["comparison"] in ("axes - raw", "axes - datapca")):
            got[t["comparison"]] = q
    nsig = sum(1 for q in qs if q < 0.05)
    print(f"  {label:58} n={len(sub):5}  q(axes-raw)={got.get('axes - raw', float('nan')):.4f}"
          f"  q(axes-datapca)={got.get('axes - datapca', float('nan')):.4f}  n_sig(q<.05)={nsig}")
    return got, nsig


print()
print("=== BH q re-derived over candidate families (stated: 0.931 / 0.855, 1 significant) ===")
famq([t for t in T if t["block"] == "9"], "block 9 only (as stored: BH per block)")
famq(T, "all tests in the file")
for lay in ([40], [40, 44], [40, 44, 20]):
    sub = [t for t in T if t["layer"] in lay]
    famq(sub, f"layers {lay} - both blocks pooled")
    famq([t for t in sub if t["block"] == "9"], f"layers {lay} - block 9 only")

print()
print("=== the run-1 family the log describes: 16 comparisons x 3 probes x 2 poolings x 2 layers x 2 blocks ===")
sub = [t for t in T if t["layer"] in (40, 44)]
print("  tests present at layers 40+44:", len(sub),
      "| comparisons", len({t['comparison'] for t in sub}),
      "| probes", len({t['probe'] for t in sub}),
      "| poolings", len({t['pooling'] for t in sub}),
      "| blocks", len({t['block'] for t in sub}))
print("  16 x 3 x 2 x 2 x 2 =", 16 * 3 * 2 * 2 * 2)
# BH computed per block over the 384/2 = 192-test half-family
for b in blocks:
    famq([t for t in sub if t["block"] == b], f"run-1 reconstruction: block {b}, layers 40+44")
# and over all 384
famq(sub, "run-1 reconstruction: all 384 (BH over both blocks together)")

print()
print("=== how many tests would clear q<0.05 in each reconstruction ===")
for b in blocks:
    s = [t for t in sub if t["block"] == b]
    qs = bh([t["dataset_p"] for t in s])
    win = [(q, t["pooling"], t["layer"], t["probe"], t["comparison"]) for q, t in zip(qs, s) if q < 0.05]
    print(f"  block {b}: {len(win)} significant ->", win[:5])
qs = bh([t["dataset_p"] for t in sub])
win = [(round(q, 4), t["block"], t["pooling"], t["layer"], t["probe"], t["comparison"])
       for q, t in zip(qs, sub) if q < 0.05]
print(f"  all 384 pooled: {len(win)} significant ->", win[:5])

print()
print("=== smallest dataset_p in the 384 reconstruction (what 'the 1 significant' would be) ===")
for t in sorted(sub, key=lambda x: x["dataset_p"])[:5]:
    print(f"  p={t['dataset_p']:.4f} block={t['block']} {t['pooling']} L{t['layer']} "
          f"{t['probe']:9} {t['comparison']:22} diff={t['diff']:+.4f}")
