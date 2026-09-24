"""Phase-1 MAPPING helper (preliminary values only, not verification).
Prints, for a crossgen matrix JSON, the off-diagonal row mean ("as source") and column mean
("as target") per feature/probe for the datasets named on the command line, plus zero-shot AUROC.
Read-only. Usage:
  python audit/scripts/p1_fmap_F001-F027_kd11cells.py <matrix.json> [probe=paper] [dataset ...]
"""
import json, statistics, sys

p = sys.argv[1]
probe = sys.argv[2] if len(sys.argv) > 2 else "paper"
want = sys.argv[3:]
d = json.load(open(p, encoding="utf-8"))
ds = d["datasets"]
print("file", p)
print("bank", d.get("bank"), d.get("bank_name"), "layer", d.get("layer"), "pooling", d.get("pooling"), "k", d.get("k"))
print("datasets:", ds)
print("block9:", d.get("block9"))
for r in d["results"]:
    if r["probe"] != probe:
        continue
    A = r["auroc"]
    for name in want:
        if name not in ds:
            continue
        i = ds.index(name)
        row = [A[i][j] for j in range(len(ds)) if j != i and A[i][j] is not None]
        col = [A[j][i] for j in range(len(ds)) if j != i and A[j][i] is not None]
        print(f"{r['feature']:22} {name:34} as_source(row mean)={statistics.mean(row):.3f}  as_target(col mean)={statistics.mean(col):.3f}  diag={A[i][i]}")
for z in d.get("zeroshot", []):
    print("zeroshot", z["direction"], {n: z["auroc"][ds.index(n)] for n in want if n in ds})
