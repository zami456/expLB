"""RF-16: recompute the three-rung scale comparison (F-63, F-96, F-100..F-102) with the bank made explicit.

Runs ON the lab PC, read-only. For every rung and every layer it computes, from the stored matrices, the
off-diagonal mean AUROC of each feature over the paper's five datasets, exactly as kd/scripts/paper_comparison.py
does (train != test pairs, probe=paper), and then:

    best_persona = max(axes, pc:deception)      # F-100's "best", a per-layer max over two features
    delta_vs_raw, delta_vs_datapca

Rungs compared:
    3b, 8b              run-crossgen/scale_{3b,8b}          paper16 bank of that model
    70b_kd              run-crossgen/scale_70b              the 36-persona kd bank (what F-100 actually swept)
    70b_paper16fp32     run-crossgen/scale_70b_paper16fp32  like-for-like paper16 bank (the audit's RF-16 run)

Usage: python fix_rf16_recompute.py [pooling]      (default: mean)
"""
import json, os, statistics, sys

REPO = "C:/Users/user4/expLB"
DS5 = ["instructed-deception", "mask", "convincing-game", "ai_liar", "roleplaying"]
POOL = sys.argv[1] if len(sys.argv) > 1 else "mean"
EXCLUDE = {int(x) for x in sys.argv[2].split(",")} if len(sys.argv) > 2 else set()
RUNGS = [("3b", "scale_3b"), ("8b", "scale_8b"), ("70b_kd", "scale_70b"), ("70b_paper16fp32", "scale_70b_paper16fp32")]


def offdiag(matrix, feature, probe="paper"):
    """Mean AUROC over train != test pairs among DS5 — exactly kd/scripts/paper_comparison.py's recipe:
    results is a list of {feature, probe, auroc}, where auroc is a train x test matrix indexed by datasets."""
    ds = matrix["datasets"]
    block = {(r["feature"], r["probe"]): r["auroc"] for r in matrix["results"]}.get((feature, probe))
    if block is None:
        return None
    vals = [v for tr, row in zip(ds, block) for te, v in zip(ds, row)
            if tr != te and v is not None and tr in DS5 and te in DS5]
    return statistics.fmean(vals) if vals else None


def per_layer(store):
    d = os.path.join(REPO, "run-crossgen", store, "matrices")
    if not os.path.isdir(d):
        return {}
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.startswith(POOL + "_L") or not fn.endswith("_s42.json"):
            continue
        L = int(fn[len(POOL) + 2:].split("_")[0])
        m = json.load(open(os.path.join(d, fn), encoding="utf-8"))
        if not set(DS5) <= set(m["datasets"]) or L in EXCLUDE:
            continue
        raw = offdiag(m, "raw"); dpca = offdiag(m, "datapca")
        cands = [f for f in ("axes", "pc:deception") if offdiag(m, f) is not None]
        if raw is None or not cands:
            continue
        best_f = max(cands, key=lambda f: offdiag(m, f))
        best = offdiag(m, best_f)
        out[L] = dict(raw=raw, datapca=dpca, best=best, best_feature=best_f, bank=m.get("bank"), bank_name=m.get("bank_name"), k=m.get("k"))
    return out


print(f"pooling={POOL}  datasets={DS5}  probe=paper\n")
tables = {}
for label, store in RUNGS:
    t = per_layer(store)
    tables[label] = t
    if not t:
        print(f"{label:18s} no matrices at {store}")
        continue
    banks = {(v["bank"], v["bank_name"], v["k"]) for v in t.values()}
    print(f"{label:18s} layers={sorted(t)}  bank(s)={banks}")
print()
hdr = f"{'rung':18s} {'n':>3s} {'raw':>8s} {'best-raw':>9s} {'best-dPCA':>10s}   per-layer best-raw"
print(hdr)
for label, _ in RUNGS:
    t = tables.get(label) or {}
    if not t:
        continue
    ls = sorted(t)
    raw = statistics.fmean(t[L]["raw"] for L in ls)
    dr = statistics.fmean(t[L]["best"] - t[L]["raw"] for L in ls)
    dd = statistics.fmean(t[L]["best"] - t[L]["datapca"] for L in ls if t[L]["datapca"] is not None)
    detail = " ".join(f"L{L}:{t[L]['best'] - t[L]['raw']:+.3f}" for L in ls)
    print(f"{label:18s} {len(ls):3d} {raw:8.4f} {dr:+9.4f} {dd:+10.4f}   {detail}")
print("\nF-100 reported (kd-bank sweep, 19 layers): best-datapca +0.120 / +0.103 / +0.075 for 3B / 8B / 70B,")
print("and 'best - raw' +0.0131 (3B) / +0.0079 (70B). Compare the rows above, matching layer sets.")
