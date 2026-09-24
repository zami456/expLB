#!/usr/bin/env python
"""W4 step 1 (LOCAL, read-only): harvest every stored ZERO-SHOT axis AUROC from the
run-crossgen matrix JSONs, plus the stored random-direction null from
run-coords/zeroshot_nulls.json.

A crossgen matrix file's `zeroshot` block holds, for each fixed direction (contrast / PC1 of
each behaviour axis in the bank), one AUROC per dataset -- no training, whole dataset scored.
That is the axis side of the control table.  The null side is computed separately
(w4_null_lab.py), because it needs the activations.

Output: audit/phase3/out/w4_zeroshot_axes.csv
"""
import csv, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "audit" / "phase3" / "out" / "w4_zeroshot_axes.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

rows = []
for p in sorted((ROOT / "run-crossgen").rglob("matrices/*.json")):
    d = json.load(open(p, encoding="utf-8"))
    if "zeroshot" not in d:
        continue
    store = str(p.parent.parent.relative_to(ROOT / "run-crossgen")).replace("\\", "/")
    if store == ".":
        store = "<root>"
    ds = d["datasets"]
    for z in d["zeroshot"]:
        for i, name in enumerate(ds):
            rows.append(dict(store=store, bank=d.get("bank", ""), bank_name="", k=d.get("k", ""),
                             pooling=d["pooling"], layer=d["layer"], seed=d["seed"],
                             direction=z["direction"], dataset=name,
                             auroc=z["auroc"][i], src=str(p.relative_to(ROOT)).replace("\\", "/")))

with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} zero-shot axis rows -> {OUT}")
import collections
print("stores:", dict(collections.Counter(r["store"] for r in rows)))
print("directions:", sorted({r["direction"] for r in rows}))
print("layers:", sorted({r["layer"] for r in rows}))
print("datasets:", sorted({r["dataset"] for r in rows}))
