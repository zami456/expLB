"""Verify F-62: kd (s3) vs kd_s23 bank deltas vs raw, off-diagonal, L40 mean paper probe.
Usage: p2v_F62.py <baseline_matrix> <bank_matrix_kd> <bank_matrix_kd_s23>"""
import json, statistics, sys

def offdiag(index, feature, probe, datasets):
    blk = index.get((feature, probe))
    if blk is None: return None
    vals = []
    for tr, row in zip(datasets, blk):
        for te, v in zip(datasets, row):
            if tr == te or v is None: continue
            vals.append(v)
    return vals

base_p, kd_p, s23_p = sys.argv[1], sys.argv[2], sys.argv[3]
bd = json.load(open(base_p, encoding="utf-8"))
bidx = {(r["feature"], r["probe"]): r["auroc"] for r in bd["results"]}
raw = offdiag(bidx, "raw", "paper", bd["datasets"])
base = statistics.fmean(raw)
print(f"baseline {base_p}: layer={bd['layer']} pooling={bd['pooling']} k={bd.get('k')} bank={bd.get('bank')} name={bd.get('bank_name')}")
print(f"raw off-diagonal mean = {base:.4f} over {len(raw)} pairs")
res = {}
for label, p in (("kd(s3)", kd_p), ("kd_s23", s23_p)):
    d = json.load(open(p, encoding="utf-8"))
    idx = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}
    print(f"\n{label} {p}: layer={d['layer']} pooling={d['pooling']} k={d.get('k')} bank={d.get('bank')} name={d.get('bank_name')} ndatasets={len(d['datasets'])}")
    feats = sorted({f for f, pr in idx if pr == "paper"})
    res[label] = {}
    for f in feats:
        v = offdiag(idx, f, "paper", d["datasets"])
        if not v: continue
        m = statistics.fmean(v)
        res[label][f] = m - base
        print(f"   {f:<24} mean={m:.4f}  d_vs_raw={m-base:+.4f}  pairs={len(v)}")
print("\n--- F-62 table ---")
print(f"{'feature':<24}{'kd(s3)':>10}{'kd_s23':>10}{'|diff|':>10}")
for f in ("pc:kd_assertion", "pc:knowledge_denial", "pc:unified", "axes"):
    a = res["kd(s3)"].get(f); b = res["kd_s23"].get(f)
    if a is None or b is None:
        print(f"{f:<24}{'MISSING':>10}{'MISSING':>10}")
    else:
        print(f"{f:<24}{a:>+10.4f}{b:>+10.4f}{abs(a-b):>10.4f}")
