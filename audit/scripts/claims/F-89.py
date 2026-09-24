"""F-89 - Tier 0.4: the format-matched control. Swapping the six holding personas for six
topically unrelated ones costs only ~+0.03; removing them entirely costs two to four times that.

The four conditions differ only in which six personas join the six capitulation roles + anchor
in the PCA basis:
  paraphrased holding = pc:cap_vs_para@2     (bank cap_ctrl)
  original  holding   = pc:capitulation@2    (bank cap_s23)
  neutral             = pc:cap_vs_neutral@2  (bank cap_ctrl)
  none (anchor only)  = pc:cap_only@2        (bank cap_s23)  == pc:cap_only_ctrl@2 (cap_ctrl)

All are read as the off-diagonal mean of the `sycophancy_mt` column (10 source datasets),
paper probe, mean pooling, seed 42. The three contrasts are re-bootstrapped over the same
10 sources, paired (one shared resampling), B = 10,000, default_rng(42).

Usage: python F-89.py [<kd11 root>]
"""
import json, os, sys
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/user4/expLB/run-crossgen/kd11"
DS, B = "sycophancy_mt", 10000
COND = [("paraphrased holding (13)", "banks/cap_ctrl", "pc:cap_vs_para@2"),
        ("original holding (13)", "banks/cap_s23", "pc:capitulation@2"),
        ("neutral / format-matched (13)", "banks/cap_ctrl", "pc:cap_vs_neutral@2"),
        ("none - anchor only (7)", "banks/cap_s23", "pc:cap_only@2"),
        ("none - ctrl copy (7)", "banks/cap_ctrl", "pc:cap_only_ctrl@2"),
        ("raw 8192-d", "banks/cap_s23", "raw")]


def column(bank, L, feat, pool="mean", probe="paper"):
    p = os.path.join(root, bank, "matrices", f"{pool}_L{L}_s42.json")
    if not os.path.exists(p):
        return None, None, None
    d = json.load(open(p, encoding="utf-8"))
    ds = d["datasets"]; j = ds.index(DS)
    for r in d["results"]:
        if r["feature"] == feat and r["probe"] == probe:
            return (np.array([r["auroc"][i][j] for i in range(len(ds)) if i != j], dtype=float),
                    [ds[i] for i in range(len(ds)) if i != j], d.get("bank"))
    return None, None, d.get("bank")


def boot(dif, idx):
    bs = dif[idx].mean(axis=1)
    return dif.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), float((bs <= 0).mean())


for L in (20, 40, 44):
    print("=" * 78)
    print(f"LAYER {L}, mean pooling, paper probe, target column = {DS}")
    cols, order, names = {}, [], None
    for label, bank, feat in COND:
        c, nm, bankfile = column(bank, L, feat)
        if c is None:
            print(f"  {label:<32} MISSING ({bank} {feat})"); continue
        cols[label] = c
        names = names or nm
        assert nm == names, "source dataset order differs between banks"
        order.append(label)
    n = len(names)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(B, n))
    raw = cols["raw 8192-d"]
    for label in order:
        c = cols[label]
        if label == "raw 8192-d":
            print(f"  {label:<32} AUROC={c.mean():.4f}")
            continue
        m, lo, hi, p = boot(c - raw, idx)
        print(f"  {label:<32} AUROC={c.mean():.4f}  vs raw={m:+.4f}  [{lo:+.4f}, {hi:+.4f}]"
              f"  P(<=0)={p:.3f}  {'clears' if lo > 0 else 'null'}")
    print("  --- the three contrasts ---")
    for lab, a, b in (("holding - neutral", "original holding (13)", "neutral / format-matched (13)"),
                      ("paraphrase - neutral", "paraphrased holding (13)", "neutral / format-matched (13)"),
                      ("any six - none", "original holding (13)", "none - anchor only (7)")):
        if a in cols and b in cols:
            m, lo, hi, p = boot(cols[a] - cols[b], idx)
            print(f"  {lab:<24} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]  P(<=0)={p:.3f}"
                  f"  {'clears' if lo > 0 else 'does NOT clear'}")
    if "original holding (13)" in cols and "neutral / format-matched (13)" in cols:
        h = cols["original holding (13)"].mean() - raw.mean()
        nn = cols["neutral / format-matched (13)"].mean() - raw.mean()
        print(f"  retention: neutral keeps {100*nn/h:.0f}% of holding's margin over raw"
              f" ({nn:+.4f} of {h:+.4f})")
    # structural identities the entry rests on
    for a, b, bank_a, bank_b in (("pc:capitulation@2", "pc:cap_vs_default@2", "banks/cap_s23", "banks/cap_s23"),
                                 ("pc:cap_only@2", "pc:cap_only_ctrl@2", "banks/cap_s23", "banks/cap_ctrl")):
        pa = os.path.join(root, bank_a, "matrices", f"mean_L{L}_s42.json")
        pb = os.path.join(root, bank_b, "matrices", f"mean_L{L}_s42.json")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        da, db = json.load(open(pa, encoding="utf-8")), json.load(open(pb, encoding="utf-8"))
        ra = next((r for r in da["results"] if r["feature"] == a and r["probe"] == "paper"), None)
        rb = next((r for r in db["results"] if r["feature"] == b and r["probe"] == "paper"), None)
        if ra and rb:
            A = np.array([[np.nan if v is None else v for v in row] for row in ra["auroc"]])
            Bm = np.array([[np.nan if v is None else v for v in row] for row in rb["auroc"]])
            off = ~np.eye(A.shape[0], dtype=bool)
            print(f"  identity {a} vs {b}: max|diff| over the {int(off.sum())} off-diagonal cells"
                  f" = {np.nanmax(np.abs(A - Bm)[off]):.6g}")
