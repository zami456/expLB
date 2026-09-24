"""READ-ONLY (lab PC): identify which persona bank produced a matrix by comparing its zero-shot direction AUROCs
(per dataset) with matrices whose bank is known. Same bank + same activations => identical zero-shot AUROCs."""
import json, glob, os
B = "C:/Users/user4/expLB/run-crossgen/"
import sys
L = sys.argv[1] if len(sys.argv) > 1 else "40"
target = B + f"scale_70b/matrices/mean_L{L}_s42.json"
def zs(path):
    d = json.load(open(path, encoding="utf-8"))
    out = {}
    for z in d.get("zeroshot", []):
        for ds, a in zip(d["datasets"], z["auroc"]):
            out[(z["direction"], ds)] = round(a, 6)
    return d.get("bank"), d.get("bank_name"), out
tb, tn, T = zs(target)
print("TARGET", target, tb, tn)
for k, v in sorted(T.items()): print("   ", k, v)
cands = [p for p in glob.glob(B + f"**/matrices/mean_L{L}_s42.json", recursive=True) if "scale_70b" not in p]
print("\ncandidates:", len(cands))
for p in sorted(cands):
    try: b, n, Z = zs(p)
    except Exception as e: print("ERR", p, e); continue
    common = [k for k in T if k in Z]
    if not common: continue
    same = sum(abs(T[k] - Z[k]) < 1e-9 for k in common)
    maxd = max(abs(T[k] - Z[k]) for k in common)
    print(f"{p[len(B):]:60s} bank={b} name={n} common={len(common)} identical={same} max|d|={maxd:.4f}")
