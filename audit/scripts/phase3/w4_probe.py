"""W4 probe (READ-ONLY, lab PC): chunk structure and label composition of run-coords/passB."""
import json, os
import torch
R = "C:/Users/user4/expLB/run-coords/passB"
for ds in sorted(os.listdir(R)):
    d = os.path.join(R, ds)
    if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "rows.jsonl")):
        continue
    lab = {}
    for line in open(os.path.join(d, "rows.jsonl"), encoding="utf-8"):
        r = json.loads(line); lab[r.get("label")] = lab.get(r.get("label"), 0) + 1
    cs = sorted(f for f in os.listdir(d) if f.startswith("chunk_"))
    b = torch.load(os.path.join(d, cs[0]), weights_only=False, map_location="cpu")
    print(f"{ds:34s} chunks={len(cs):4d} labels={lab} keys={list(b.keys())}")
    print(f"{'':34s} pooled{tuple(b['pooled'].shape)} {b['pooled'].dtype} layers={list(b['layers'])} pooling={list(b.get('pooling',['mean','last']))}")
    del b
