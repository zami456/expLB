"""Lead QA of F-107 (batch M1's top nomination) — independent re-derivation.

F-107's numbers exist in NO stored artifact, and it is the project's sharpest limitation claim:
"the deception axis does not fire on natural sarcasm; the axis that fires is sycophancy". This
rebuilds it with its own loader, its own AUROC and its own null draws. Read-only.
"""
import glob, json, os

import numpy as np
import torch

ROOT = "C:/Users/user4/expLB"
STORE = f"{ROOT}/run-coords/sarcasm/sarcasm_natural"
ROWS = f"{ROOT}/cap/run-sarcasm-natural/rows.jsonl"


def auroc(y, s):
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=np.float64)
    order = np.argsort(s, kind="mergesort"); sv = s[order]
    rank = np.empty(len(s), float); i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        rank[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(s), float); r[order] = rank
    npos = int(y.sum()); nneg = len(y) - npos
    return (r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


chunks = sorted(glob.glob(f"{STORE}/chunk_*.pt"))
pooled, rids, layers, pooling = [], [], None, None
for c in chunks:
    d = torch.load(c, map_location="cpu", weights_only=False)
    p = d["pooled"]
    pooled.append(p.float().numpy() if hasattr(p, "float") else np.asarray(p))
    rids += list(d["row_ids"])
    layers = list(d["layers"]) if layers is None else layers
    pooling = d.get("pooling") if pooling is None else pooling
X = np.concatenate(pooled, 0)
print(f"chunks {len(chunks)}  X {X.shape}  rows {len(rids)}  layers {layers}  pooling {pooling}")

rows = {json.loads(l)["row_id"]: json.loads(l) for l in open(ROWS, encoding="utf-8") if l.strip()}
y = np.array([int(rows[r]["label"]) for r in rids])
print(f"labels aligned by row_id: {y.sum()} sarcastic / {len(y)-y.sum()} literal")

# X is (n, n_layers, H) or (n, n_layers, n_pool, H)
def slab(L, pool="mean"):
    """X is (n, n_layers, H, n_pool) for this store — the LAST axis is the pooling."""
    j = layers.index(L)
    a = X[:, j]                                   # (n, H, n_pool)
    if a.ndim == 3:
        k = list(pooling).index(pool) if isinstance(pooling, (list, tuple)) and pool in pooling else 0
        a = a[:, :, k]                            # (n, H)
    return a.astype(np.float64)


def unit(v):
    v = np.asarray(v, dtype=np.float64); n = np.linalg.norm(v)
    return v / n if n else v


BANKS = [f"{ROOT}/run-paper16/axes_vectors_s3_fp32.pt", f"{ROOT}/run-kd/axes_vectors_s23.pt"]
rng = np.random.default_rng(42)
for bp in BANKS:
    if not os.path.exists(bp):
        continue
    bank = torch.load(bp, map_location="cpu", weights_only=False)
    if "axes" not in bank:
        continue
    print(f"\n=== {os.path.basename(bp)}  axes: {sorted(bank['axes'])}")
    for L in [l for l in (40, 44) if l in layers]:
      for pool in ("mean", "last"):
        A = slab(L, pool); Ac = A - A.mean(0, keepdims=True)
        out = []
        for name in sorted(bank["axes"]):
            try:
                d = unit(bank["axes"][name][L].numpy())
            except Exception:
                continue
            if d.shape[0] != A.shape[1]:
                continue
            a = auroc(y, Ac @ d)
            out.append((name, max(a, 1 - a)))
        R = rng.standard_normal((2000, A.shape[1])); R /= np.linalg.norm(R, axis=1, keepdims=True)
        S = Ac @ R.T
        nulls = np.array([auroc(y, S[:, i]) for i in range(S.shape[1])])
        nulls = np.maximum(nulls, 1 - nulls)
        p95, mx = np.percentile(nulls, 95), nulls.max()
        print(f"  L{L} {pool:4s}  random null (n=2000): p95 {p95:.4f}  max {mx:.4f}  median {np.median(nulls):.4f}")
        for name, a in out:
            flag = "ABOVE p95" if a > p95 else "below p95"
            beat = "beats null max" if a > mx else ""
            print(f"       {name:12s} {a:.4f}   {flag} {beat}")
