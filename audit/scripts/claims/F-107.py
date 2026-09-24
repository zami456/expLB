#!/usr/bin/env python
"""M1 claim check F-107 -- Gate 3 on natural sarcasm, recomputed from the stored activations.

F-107's numbers were never persisted (fmap: producing script NOT-FOUND, inline).  This is a
Level-2-only recompute: load run-coords/sarcasm/sarcasm_natural/chunk_*.pt, project onto the
axes of run-kd/axes_vectors_s23.pt (+ collab_extended from run-kd/axes_collab_s23.pt) exactly as
9_persona_probes.zeroshot() does -- roc_auc_score(y, X @ unit(direction)) -- and rebuild the
random-direction null as the entry describes it (200 unit directions per cell, scored unsigned
as max(a, 1-a)).

READ-ONLY.  Run on the lab PC via labpc_run.py + p1_labpc_runpy.ps1.
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

R = Path("C:/Users/user4/expLB")
S = R / "run-coords" / "sarcasm" / "sarcasm_natural"
LAYERS = [20, 40]
POOLS = ["mean", "last"]
NDRAW = 200
SEEDS = [0, 1, 42]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


meta = {}
for line in (S / "rows.jsonl").open(encoding="utf-8"):
    r = json.loads(line)
    meta[r["row_id"]] = r

store = {}          # (pool, layer) -> (X, y)
for pool in POOLS:
    for layer in LAYERS:
        store[(pool, layer)] = ([], [])

for c in sorted(S.glob("chunk_*.pt")):
    blob = torch.load(c, weights_only=False, map_location="cpu")
    layers = list(blob["layers"])
    pools = list(blob.get("pooling", ["mean", "last"]))
    for pool in POOLS:
        pi = pools.index(pool)
        for layer in LAYERS:
            li = layers.index(layer)
            pooled = blob["pooled"][:, li, :, pi].to(torch.float32).numpy()
            xs, ys = store[(pool, layer)]
            for i, rid in enumerate(blob["row_ids"]):
                m = meta.get(rid)
                if m is None or m.get("label") is None:
                    continue
                xs.append(pooled[i]); ys.append(m["label"])
    del blob

bank = torch.load(R / "run-kd" / "axes_vectors_s23.pt", map_location="cpu", weights_only=False)
collab = torch.load(R / "run-kd" / "axes_collab_s23.pt", map_location="cpu", weights_only=False)

print("bank axes_vectors_s23.pt roles:", len(bank["roles"]))
print("bank axes_collab_s23.pt roles:", len(collab["roles"]))
print()

rows = []
for pool in POOLS:
    for layer in LAYERS:
        xs, ys = store[(pool, layer)]
        X = np.stack(xs); y = np.array(ys, dtype=int)
        print("=" * 76)
        print(f"pooling={pool}  layer={layer}   X{X.shape}  labels 1={int(y.sum())} "
              f"0={int((1 - y).sum())}")
        print("=" * 76)
        dirs = {k: unit(bank["axes"][k][layer].numpy()) for k in bank["axes"]}
        dirs["collab_extended"] = unit(collab["axes"]["collab_extended"][layer].numpy())
        comps = bank["pca"]["deception"]["components"][layer].numpy()
        for j in range(min(3, comps.shape[0])):
            dirs[f"pc_deception_PC{j + 1}"] = unit(comps[j])
        for name, d in sorted(dirs.items()):
            a = float(roc_auc_score(y, X @ d))
            print(f"   {name:24} AUROC {a:.4f}   unsigned {max(a, 1 - a):.4f}")
            rows.append({"pooling": pool, "layer": layer, "direction": name, "auroc": a})

        # random-direction null, unsigned as the entry specifies
        H = X.shape[1]
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            V = rng.standard_normal((NDRAW, H)).astype(np.float32)
            V /= np.linalg.norm(V, axis=1, keepdims=True)
            aa = np.array([roc_auc_score(y, X @ v) for v in V])
            un = np.maximum(aa, 1 - aa)
            print(f"   RANDOM n={NDRAW} seed={seed}: mean {un.mean():.4f}  "
                  f"p95 {np.percentile(un, 95):.4f}  max {un.max():.4f}")
        rng = np.random.default_rng(12345)
        V = rng.standard_normal((2000, H)).astype(np.float32)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        aa = np.array([roc_auc_score(y, X @ v) for v in V])
        un = np.maximum(aa, 1 - aa)
        print(f"   RANDOM n=2000 seed=12345: mean {un.mean():.4f}  "
              f"p95 {np.percentile(un, 95):.4f}  p99 {np.percentile(un, 99):.4f}  "
              f"max {un.max():.4f}")
        print()

print("JSON " + json.dumps(rows))
