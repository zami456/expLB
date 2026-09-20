"""The cap result across all three layers, both filters, plus the zero-shot contrast rows.

The trained `pc:<name>` features project onto the named PCA BASIS and never touch the contrast,
so variants A and B (identical role sets) are identical by construction. The ZERO-SHOT rows are
the only place the contrast direction itself is scored.
"""
import json, statistics, sys, glob, os

DS = "sycophancy_mt"
KEEP = ("raw", "random", "random@2", "datapca", "datapca@2", "length",
        "pc:capitulation", "pc:capitulation@2", "pc:cap_vs_default", "pc:cap_vs_default@2",
        "pc:cap_only", "pc:cap_only@2")

def col(A, ds, j):
    t = [A[i][j] for i in range(len(ds)) if i != j and A[i][j] is not None]
    return statistics.fmean(t) if t else float("nan")

for bank in ("cap_s23", "cap_all"):
    for lay in (20, 40, 44):
        p = f"run-crossgen/kd11/banks/{bank}/matrices/mean_L{lay}_s42.json"
        if not os.path.exists(p):
            print(f"{p} MISSING"); continue
        d = json.load(open(p, encoding="utf-8"))
        ds = d["datasets"]; j = ds.index(DS)
        print(f"\n=== {bank}  L{lay}  (as TARGET on {DS}, off-diagonal mean) ===")
        vals = {}
        for r in d["results"]:
            if r["probe"] != "paper" or r["feature"] not in KEEP:
                continue
            vals[r["feature"]] = col(r["auroc"], ds, j)
        for f in KEEP:
            if f in vals:
                print(f"    {f:<22} {vals[f]:.4f}")
        zs = d.get("zeroshot") or []
        if zs:
            print(f"  -- zero-shot (the CONTRAST direction, 100% of rows) --")
            for z in zs:
                v = z.get("auroc") or z.get("aurocs")
                if isinstance(v, dict):
                    print(f"    {z['direction']:<22} {DS}: {v.get(DS)}")
                elif isinstance(v, list):
                    print(f"    {z['direction']:<22} {DS}: {v[j]}")
                else:
                    print(f"    {z['direction']:<22} {json.dumps(z)[:200]}")
