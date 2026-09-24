"""F-34 - the kd_s23 gain is depth-dependent, not an L40 artifact.
Checks: the 21-layer profile of `pc:unified - raw` (paper probe, mean pooling, block 9 = 90 pairs),
the same against datapca, how many layers are positive, the peak layer, the best q, the number of
tests in the BH family, and which tests clear the dataset bootstrap.

Usage: python F-34.py [<banks/kd_s23/significance.json>] [<bank matrices dir>] [<baseline matrices dir>]
"""
import glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sig = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "..", "..", "..", "run-crossgen", "kd11", "banks", "kd_s23", "significance.json")
d = json.load(open(sig, encoding="utf-8"))
t = d["tests"]
print("file    :", os.path.abspath(sig))
print("n tests :", len(t), "| boot_datasets:", d.get("boot_datasets"))

def sel(comp, probe="paper", pooling="mean", block="9"):
    return {x["layer"]: x for x in t
            if x["comparison"] == comp and x["probe"] == probe
            and x["pooling"] == pooling and x["block"] == block}

print("\n=== profile: pc:unified - raw, paper, mean, block 9 (90 pairs) ===")
a = sel("pc:unified - raw")
for L in sorted(a):
    x = a[L]
    ci = x["dataset_ci95"]
    print(f"  L{L:<3} diff={x['diff']:+.4f} pairs={x['pairs_won']}/{x['pairs']}"
          f" ci95=[{ci[0]:+.3f}, {ci[1]:+.3f}] p={x['dataset_p']:.3f} q={x['dataset_q_bh']:.3f}"
          f" {'CI>0' if ci[0] > 0 else ''}")
pos = sum(1 for L in a if a[L]["diff"] > 0)
print(f"  positive at {pos} of {len(a)} layers; range {min(a[L]['diff'] for L in a):+.4f} .. "
      f"{max(a[L]['diff'] for L in a):+.4f}; peak at L{max(a, key=lambda L: a[L]['diff'])}")

print("\n=== profile: pc:unified - datapca, paper, mean, block 9 ===")
b = sel("pc:unified - datapca")
for L in sorted(b):
    x = b[L]; ci = x["dataset_ci95"]
    print(f"  L{L:<3} diff={x['diff']:+.4f} ci95=[{ci[0]:+.3f}, {ci[1]:+.3f}] q={x['dataset_q_bh']:.3f}"
          f" {'CI>0' if ci[0] > 0 else ''}")
posb = sum(1 for L in b if b[L]["diff"] > 0)
print(f"  positive at {posb} of {len(b)} layers; peak at L{max(b, key=lambda L: b[L]['diff'])}")
print(f"  best (lowest) q among pc:unified - datapca, this slice: {min(b[L]['dataset_q_bh'] for L in b):.3f}"
      f" at L{min(b, key=lambda L: b[L]['dataset_q_bh'])}")

print("\n=== tests that clear the dataset bootstrap in this file ===")
cl = [x for x in t if x.get("significant_datasets")]
print("  count:", len(cl), "| negative:", sum(1 for x in cl if x["diff"] < 0))
for x in sorted(cl, key=lambda y: (y["comparison"], y["layer"])):
    print(f"  L{x['layer']:<3} {x['pooling']:<5} blk{x['block']} {x['probe']:<9} {x['comparison']:<28}"
          f" diff={x['diff']:+.4f} q={x['dataset_q_bh']:.3f}")

print("\n=== best q over the whole file for any POSITIVE pc:unified comparison vs raw/datapca ===")
cand = [x for x in t if x["comparison"] in ("pc:unified - raw", "pc:unified - datapca") and x["diff"] > 0]
if cand:
    m = min(cand, key=lambda x: x["dataset_q_bh"])
    print(f"  q={m['dataset_q_bh']:.3f}  {m['comparison']}  L{m['layer']} {m['pooling']} blk{m['block']} {m['probe']} diff={m['diff']:+.4f}")

# ---- Level 2: off-diagonal means straight from the matrices ----
sys.path.insert(0, HERE)
import _dsboot as DB
mdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(sig)), "matrices")
bdir = sys.argv[3] if len(sys.argv) > 3 else os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(sig)), "..", "..", "matrices"))
print("\n=== LEVEL 2: recomputed off-diagonal means (block 9), bank =", mdir, "===")
for fp in sorted(glob.glob(os.path.join(mdir, "mean_L*_s42.json")),
                 key=lambda s: int(re.search(r"_L(\d+)_", s).group(1))):
    L = int(re.search(r"_L(\d+)_", fp).group(1))
    bp = os.path.join(bdir, f"mean_L{L}_s42.json")
    if not os.path.exists(bp):
        print(f"  L{L}: baseline missing"); continue
    m = json.load(open(fp, encoding="utf-8")); bb = json.load(open(bp, encoding="utf-8"))
    ds = m["datasets"]
    by = {(r["feature"], r["probe"]): DB.arr(r["auroc"]) for r in m["results"]}
    for r in bb["results"]:
        by.setdefault((r["feature"], r["probe"]), DB.arr(r["auroc"]))
    ids9 = [ds.index(x) for x in m.get("block9", ds[:-1])]
    u = DB.offdiag_mean(by[("pc:unified", "paper")], ids9)
    rr = DB.offdiag_mean(by[("raw", "paper")], ids9)
    dp = DB.offdiag_mean(by[("datapca", "paper")], ids9)
    print(f"  L{L:<3} bank={m.get('bank')} name={m.get('bank_name')} | pc:unified={u:.4f} raw={rr:.4f}"
          f" datapca={dp:.4f} | -raw={u-rr:+.4f} -datapca={u-dp:+.4f}")
