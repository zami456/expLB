"""F-66 - the cap result clears the DATASET-LEVEL bootstrap on the `sycophancy_mt` column.

Independent reimplementation of cap/scripts/cap_significance.py: paired i.i.d. bootstrap over the
10 SOURCE datasets that each contribute one trained transfer AUROC to the sycophancy_mt column,
B = 10,000, numpy default_rng(42), one shared resampling for every pair (so the difference carries
no between-sample variance). Also prints the per-source table and the win counts.

Usage: python F-66.py [<kd11 root>]
"""
import json, os, sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/user4/expLB/run-crossgen/kd11"
DS = "sycophancy_mt"
B = 10000
PAIRS = [("pc:capitulation@2", "raw"), ("pc:capitulation@2", "datapca@2"),
         ("pc:capitulation@2", "random@2"), ("pc:capitulation@2", "length"),
         ("pc:capitulation@2", "pc:cap_only@2"), ("pc:cap_only@2", "raw"),
         ("pc:capitulation", "raw")]


def column(d, feat, probe="paper"):
    ds = d["datasets"]; j = ds.index(DS)
    for r in d["results"]:
        if r["feature"] == feat and r["probe"] == probe:
            return (np.array([r["auroc"][i][j] for i in range(len(ds)) if i != j], dtype=float),
                    [ds[i] for i in range(len(ds)) if i != j])
    return None, None


for bank, L in (("banks/cap_s23", 20), ("banks/cap_s23", 40), ("banks/cap_s23", 44),
                ("banks/cap_all", 40), ("banks/cap_all", 20), ("banks/cap_all", 44)):
    p = os.path.join(root, bank, "matrices", f"mean_L{L}_s42.json")
    if not os.path.exists(p):
        print("MISSING", p); continue
    d = json.load(open(p, encoding="utf-8"))
    print("=" * 78)
    print(f"{bank}  L{L}  bank={d.get('bank')} name={d.get('bank_name')} pooling={d['pooling']} seed={d['seed']}")
    cols, names = {}, None
    for a, b in PAIRS:
        for f in (a, b):
            if f not in cols:
                cols[f], nm = column(d, f)
                if nm:
                    names = nm
    n = len(names)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(B, n))        # one shared resampling -> paired
    print(f"  resampling unit: the {n} SOURCE datasets in the '{DS}' column, B={B}")
    for a, b in PAIRS:
        ca, cb = cols.get(a), cols.get(b)
        if ca is None or cb is None:
            print(f"  {a+' - '+b:<40}  MISSING"); continue
        dif = ca - cb
        bs = dif[idx].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ple = float((bs <= 0).mean())
        mark = "CLEARS" if lo > 0 else ("sig. negative" if hi < 0 else "does NOT clear")
        print(f"  {a+' - '+b:<40}{dif.mean():>+9.4f}  [{lo:+.4f}, {hi:+.4f}]  P(<=0)={ple:.4f}  {mark}")
    ca, cb = cols["pc:capitulation@2"], cols["raw"]
    print(f"  cap@2 beats raw on {int((ca > cb).sum())} of {n} source datasets")
    if L == 40 and bank == "banks/cap_s23":
        print("  per-source, L40, s23:")
        for i in np.argsort(-(ca - cb)):
            print(f"    {names[i]:<34}{ca[i]:>8.3f}{cb[i]:>8.3f}{ca[i]-cb[i]:>+9.3f}")

# ---- robustness the entry did not run: the same paired dataset bootstrap at EVERY stored layer ----
import glob, re
print("\n" + "=" * 78)
print("ROBUSTNESS: pc:capitulation@2 vs raw / datapca@2 / pc:cap_only@2, every stored layer, s23")
for pool in ("mean", "last"):
    print(f"-- pooling={pool}")
    for fp in sorted(glob.glob(os.path.join(root, "banks/cap_s23", "matrices", f"{pool}_L*_s42.json")),
                     key=lambda s: int(re.search(r"_L(\d+)_", s).group(1))):
        L = int(re.search(r"_L(\d+)_", fp).group(1))
        d = json.load(open(fp, encoding="utf-8"))
        cols = {}
        for f in ("pc:capitulation@2", "raw", "datapca@2", "pc:cap_only@2", "random@2"):
            cols[f], names = column(d, f)
        n = len(names)
        rng = np.random.default_rng(42)
        idx = rng.integers(0, n, size=(B, n))
        out = []
        for a, b in (("pc:capitulation@2", "raw"), ("pc:capitulation@2", "datapca@2"),
                     ("pc:capitulation@2", "pc:cap_only@2")):
            dif = cols[a] - cols[b]
            bs = dif[idx].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out.append(f"{b}: {dif.mean():+.4f} [{lo:+.4f},{hi:+.4f}] {'CLEARS' if lo > 0 else '-     '}")
        print(f"  L{L:<3} " + " | ".join(out))
