"""Structure probe for a crossgen matrix file (read-only, lab PC)."""
import json
import os

p = r"C:\Users\user4\expLB\run-crossgen\kd11\banks\cap_s23\matrices\mean_L40_s42.json"
j = json.load(open(p, encoding="utf-8"))
res = j["results"]
print("type(results)", type(res).__name__)
if isinstance(res, dict):
    print("results keys:", sorted(res)[:30])
    k0 = sorted(res)[0]
    v = res[k0]
    print("  first key", k0, "->", type(v).__name__,
          sorted(v)[:20] if isinstance(v, dict) else f"len {len(v)}")
    if isinstance(v, dict):
        k1 = sorted(v)[0]
        w = v[k1]
        print("    ", k1, "->", type(w).__name__,
              sorted(w)[:20] if isinstance(w, dict) else str(w)[:200])
        if isinstance(w, dict):
            k2 = sorted(w)[0]
            print("      ", k2, "->", str(w[k2])[:300])
elif isinstance(res, list):
    print("len", len(res), "first:", json.dumps(res[0])[:400])
print()
print("zeroshot:", json.dumps(j["zeroshot"])[:400])
print()
print("block9:", json.dumps(j["block9"])[:300])
print()
print("row_bootstrap_offdiag_diff:", json.dumps(j["row_bootstrap_offdiag_diff"])[:400])
