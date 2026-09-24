#!/usr/bin/env python
"""Read-only structural peek for M1: sarcasm chunk store + the two kd banks + span stats.

Run on the lab PC through labpc_run.py + p1_labpc_runpy.ps1.  Reads nothing else.
"""
import json
from pathlib import Path

import torch

R = Path("C:/Users/user4/expLB")
S = R / "run-coords" / "sarcasm" / "sarcasm_natural"

print("== sarcasm store ==")
for p in sorted(S.iterdir()):
    print(f"  {p.name:20} {p.stat().st_size:>12,}  {p.stat().st_mtime}")

b = torch.load(S / "chunk_0000.pt", weights_only=False, map_location="cpu")
print("  chunk_0000 keys:", sorted(b))
print("  layers:", list(b["layers"]))
print("  pooling:", list(b.get("pooling", ["mean", "last"])))
print("  pooled shape/dtype:", tuple(b["pooled"].shape), b["pooled"].dtype)
print("  n row_ids:", len(b["row_ids"]), "first:", b["row_ids"][0])
del b

n = 0
labs = {}
for line in (S / "rows.jsonl").open(encoding="utf-8"):
    r = json.loads(line)
    n += 1
    labs[r["label"]] = labs.get(r["label"], 0) + 1
print("  rows.jsonl:", n, "labels", labs)

for name in ("axes_vectors_s23.pt", "axes_collab_s23.pt"):
    p = R / "run-kd" / name
    print(f"\n== {name}  {p.stat().st_size:,} B  mtime {p.stat().st_mtime} ==")
    bk = torch.load(p, map_location="cpu", weights_only=False)
    print("  top keys:", sorted(bk))
    if "axes" in bk:
        for k, v in bk["axes"].items():
            try:
                print(f"    axes[{k}] shape {tuple(v.shape)} dtype {v.dtype}")
            except AttributeError:
                print(f"    axes[{k}] {type(v)}")
    if "pca" in bk:
        print("  pca keys:", sorted(bk["pca"]))
    for k in ("meta", "config", "roles", "personas"):
        if k in bk:
            print(f"  {k}:", str(bk[k])[:500])
    del bk

print("\n== HP-KR / kr-matched span stats (F-43/F-44) ==")
import numpy as np
for ds in ("harm-pressure-knowledge-report", "harm-pressure-choice", "harm-pressure-kr-matched"):
    p = R / "run-coords" / "passB" / ds / "rows.jsonl"
    if not p.exists():
        print(f"  {ds}: rows.jsonl MISSING")
        continue
    sl, ids = [], []
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        ids.append(r["row_id"])
        if r.get("span_len") is not None:
            sl.append(r["span_len"])
    a = np.array(sl)
    print(f"  {ds}: rows={len(ids)} span_len n={len(a)} "
          f"p50={np.percentile(a,50):.1f} p90={np.percentile(a,90):.1f} "
          f"p99={np.percentile(a,99):.1f} max={a.max()} >1024:{int((a>1024).sum())}")
