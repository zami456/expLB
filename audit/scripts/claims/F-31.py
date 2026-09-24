"""F-31 - the kd bank's +0.076 (pc:unified - raw, paper probe, L40, mean pooling) does NOT clear
the dataset-level bootstrap; and '6 of 4,032 tests clear', four of them negative.

Usage: python F-31.py [<path to banks/kd/significance.json>] [<matrices dir for level-2 recompute>]
"""
import json, os, statistics, sys

sig = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "run-crossgen", "kd11", "banks", "kd", "significance.json")
d = json.load(open(sig, encoding="utf-8"))
t = d["tests"]
print("file        :", os.path.abspath(sig))
print("n tests     :", len(t), "| boot_datasets:", d.get("boot_datasets"))
print("BH note     :", d.get("note"))

HEAD = [("pc:unified - raw", "paper"), ("pc:unified - datapca", "paper"),
        ("pc:unified - raw", "lr_reg10"), ("axes - raw", "paper")]
print("\n=== F-31 headline table: block 10, mean pooling, L40 ===")
for comp, probe in HEAD:
    for x in t:
        if (x["comparison"] == comp and x["probe"] == probe and x["layer"] == 40
                and x["pooling"] == "mean" and x["block"] == "10"):
            print(f"  {comp:<26} {probe:<9} diff={x['diff']:+.4f}  pairs={x['pairs_won']}/{x['pairs']}"
                  f"  ci95=[{x['dataset_ci95'][0]:+.3f}, {x['dataset_ci95'][1]:+.3f}]"
                  f"  p={x['dataset_p']}  q_BH={x['dataset_q_bh']:.3f}  clears={x['significant_datasets']}")

print("\n=== every test that clears the dataset bootstrap (significant_datasets) ===")
cl = [x for x in t if x.get("significant_datasets")]
print("  count:", len(cl))
for x in sorted(cl, key=lambda y: (y["comparison"], y["layer"])):
    print(f"  L{x['layer']:<3} {x['pooling']:<5} blk{x['block']} {x['probe']:<9} {x['comparison']:<26}"
          f" diff={x['diff']:+.4f} q={x['dataset_q_bh']:.3f}")
print("  negative among them:", sum(1 for x in cl if x["diff"] < 0))

print("\n=== how many clear by the RAW (uncorrected) CI, ignoring BH ===")
raw_cl = [x for x in t if x["dataset_ci95"] and (x["dataset_ci95"][0] > 0 or x["dataset_ci95"][1] < 0)]
print("  count:", len(raw_cl), "of", len(t))

# ---- Level 2: recompute the headline diff AND its bootstrap CI from the AUROC matrices ----
# The bank matrices hold only persona features; raw/datapca/random come from the baseline
# (kd11/matrices), which is bank-independent. Both are merged here, as 27_crossgen.py does.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import _dsboot as DB

mdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(sig)), "matrices")
bdir = sys.argv[3] if len(sys.argv) > 3 else os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(sig)), "..", "..", "matrices"))
mp, bp = os.path.join(mdir, "mean_L40_s42.json"), os.path.join(bdir, "mean_L40_s42.json")
if os.path.exists(mp) and os.path.exists(bp):
    m = json.load(open(mp, encoding="utf-8")); b = json.load(open(bp, encoding="utf-8"))
    assert m["datasets"] == b["datasets"], "dataset order differs between bank and baseline"
    ds = m["datasets"]
    by = {(r["feature"], r["probe"]): DB.arr(r["auroc"]) for r in m["results"]}
    for r in b["results"]:
        by.setdefault((r["feature"], r["probe"]), DB.arr(r["auroc"]))
    print("\n=== LEVEL 2 ===")
    print("  bank matrix  :", os.path.abspath(mp), "| bank:", m.get("bank"), "| name:", m.get("bank_name"), "| k:", m.get("k"))
    print("  baseline     :", os.path.abspath(bp), "| bank:", b.get("bank"), "| name:", b.get("bank_name"))
    ids10 = list(range(len(ds))); ids9 = [ds.index(x) for x in b.get("block9", ds[:-1])]
    for name, ids in (("block10 (11 ds)", ids10), ("block9 (10 ds)", ids9)):
        vals = {f: DB.offdiag_mean(by[(f, "paper")], ids) for f in ("pc:unified", "raw", "datapca", "axes")}
        print(f"  {name:<18} " + "  ".join(f"{k}={v:.4f}" for k, v in vals.items())
              + f" | pc:unified-raw={vals['pc:unified']-vals['raw']:+.4f}"
                f" pc:unified-datapca={vals['pc:unified']-vals['datapca']:+.4f}"
                f" axes-raw={vals['axes']-vals['raw']:+.4f}")
    # exact re-run of the bootstrap: one rng seeded with main_seed, block 10 drawn first
    rng = np.random.default_rng(42)
    W10 = DB.make_weights(len(ids10), d.get("boot_datasets", 10000), rng)
    print("\n  re-run of the paired dataset bootstrap (B=%d, rng=default_rng(42), block 10 first):"
          % d.get("boot_datasets", 10000))
    for fa, fb, pr in (("pc:unified", "raw", "paper"), ("pc:unified", "datapca", "paper"),
                       ("pc:unified", "raw", "lr_reg10"), ("axes", "raw", "paper")):
        dd = DB.dboot(by[(fa, pr)], ids10, W10, len(ids10)) - DB.dboot(by[(fb, pr)], ids10, W10, len(ids10))
        dd = dd[~np.isnan(dd)]
        print(f"    {fa} - {fb:<9} {pr:<9} diff={DB.offdiag_mean(by[(fa,pr)],ids10)-DB.offdiag_mean(by[(fb,pr)],ids10):+.4f}"
              f" ci95=[{np.percentile(dd,2.5):+.3f}, {np.percentile(dd,97.5):+.3f}] p={DB.two_sided_p(dd):.4f}")
else:
    print("\n(matrices not found at", mp, "/", bp, "- level 2 skipped)")
