"""F-61 - locate the '+0.006' reading for the paper16 bank and compare it like-for-like
with the kd bank's '+0.076'.

Runs on the LOCAL git copy of run-crossgen/kd11 (fmap: lab = git for these matrices).
Mirrors kd/scripts/paper_comparison.py's off-diagonal definition (skip diagonal and None).
"""
import json, os, statistics, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
KD11 = os.path.join(ROOT, "run-crossgen", "kd11")


def offdiag(doc, feature, probe="paper", keep=None):
    ds = doc["datasets"]
    idx = {(r["feature"], r["probe"]): r["auroc"] for r in doc["results"]}
    blk = idx.get((feature, probe))
    if blk is None:
        return None, 0
    vals = []
    for tr, row in zip(ds, blk):
        for te, v in zip(ds, row):
            if tr == te or v is None:
                continue
            if keep and (tr not in keep or te not in keep):
                continue
            vals.append(v)
    return (statistics.fmean(vals) if vals else None), len(vals)


def load(rel):
    p = os.path.join(KD11, rel)
    return (json.load(open(p, encoding="utf-8")), p) if os.path.exists(p) else (None, p)


print("### the root (paper16) matrices, both poolings, probe=paper")
for pool in ("mean", "last"):
    doc, p = load(f"matrices/{pool}_L40_s42.json")
    if doc is None:
        print("  MISSING", p); continue
    raw, nr = offdiag(doc, "raw")
    uni, nu = offdiag(doc, "pc:unified")
    uni2, nu2 = offdiag(doc, "pc:unified@2")
    dpc, nd = offdiag(doc, "datapca")
    dpc2, nd2 = offdiag(doc, "datapca@2")
    print(f"  {pool}_L40_s42.json  bank={doc.get('bank')} name={doc.get('bank_name')} "
          f"k={doc.get('k')} datasets={len(doc['datasets'])}")
    print(f"     raw        = {raw:.4f}  over {nr} pairs")
    if uni is not None:
        print(f"     pc:unified = {uni:.4f} ({nu})   pc:unified - raw = {uni-raw:+.4f}")
    if uni2 is not None:
        print(f"     pc:unified@2={uni2:.4f} ({nu2})  pc:unified@2 - raw = {uni2-raw:+.4f}")
    if dpc is not None:
        print(f"     datapca    = {dpc:.4f} ({nd})   pc:unified - datapca = {uni-dpc:+.4f}")
    if dpc2 is not None:
        print(f"     datapca@2  = {dpc2:.4f} ({nd2})  pc:unified@2 - datapca@2 = {uni2-dpc2:+.4f}")
    feats = sorted({r["feature"] for r in doc["results"]})
    print(f"     features: {feats}")

print("\n### the kd bank matrix (its controls must come from the ROOT store)")
for pool in ("mean", "last"):
    bank, pb = load(f"banks/kd/matrices/{pool}_L40_s42.json")
    root, pr = load(f"matrices/{pool}_L40_s42.json")
    if bank is None or root is None:
        print("  MISSING", pb, pr); continue
    raw, nr = offdiag(root, "raw")
    uni, nu = offdiag(bank, "pc:unified")
    dpc, nd = offdiag(root, "datapca")
    print(f"  banks/kd {pool}_L40  bank={bank.get('bank')} name={bank.get('bank_name')} "
          f"k={bank.get('k')}")
    print(f"     kd pc:unified = {uni:.4f} ({nu})   root raw = {raw:.4f} ({nr})   "
          f"kd pc:unified - raw = {uni-raw:+.4f}")
    print(f"     root datapca  = {dpc:.4f}          kd pc:unified - datapca = {uni-dpc:+.4f}")
    for f in ("pc:kd_assertion", "pc:knowledge_denial", "axes"):
        v, n = offdiag(bank, f)
        if v is not None:
            print(f"     kd {f:<22} = {v:.4f} ({n})   - raw = {v-raw:+.4f}")

print("\n### 6_axes.py: what goes into `unified`")
ax = os.path.join(ROOT, "src", "assistant_axis_ndif", "6_axes.py")
if os.path.exists(ax):
    lines = open(ax, encoding="utf-8").read().splitlines()
    for i, l in enumerate(lines, 1):
        if 165 <= i <= 200:
            print(f"   {i}: {l}")
else:
    print("   6_axes.py NOT FOUND at", ax)
