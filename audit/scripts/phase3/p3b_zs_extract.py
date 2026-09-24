"""P3-B: extract per-dataset zero-shot AUROC from the owner's crossgen matrix JSONs.

Usage: python p3b_zs_extract.py <repo_root> [glob ...]
Prints: store,pooling,layer,seed,bank,bank_name,k,direction,dataset,auroc
"""
import json, sys, glob, os, csv

root = sys.argv[1]
pats = sys.argv[2:] or ["run-crossgen/**/matrices/*_s42.json"]
w = csv.writer(sys.stdout, lineterminator="\n")
w.writerow("store pooling layer seed bank bank_name k direction dataset auroc".split())
for pat in pats:
    for p in sorted(glob.glob(os.path.join(root, pat), recursive=True)):
        try:
            d = json.load(open(p))
        except Exception as e:
            print("ERR", p, e, file=sys.stderr); continue
        if "zeroshot" not in d:
            continue
        rel = os.path.relpath(p, root).replace("\\", "/")
        store = rel.split("/matrices/")[0]
        ds = d["datasets"]
        for r in d["zeroshot"]:
            for i, name in enumerate(ds):
                w.writerow([store, d.get("pooling"), d.get("layer"), d.get("seed"),
                            d.get("bank"), d.get("bank_name"), d.get("k"),
                            r.get("direction"), name, r["auroc"][i]])
