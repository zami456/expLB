"""S-14 level 2 (READ-ONLY, lab PC): recompute the per-dataset zero-shot random-direction null
for the paper16 deception axis at L40, mean pooling, on run-coords/passB -- the 9 datasets of
`corpus_L40_mean` in run-coords/zeroshot_nulls.json.

Method mirrors src/assistant_axis_ndif/25_zeroshot_nulls.py: a random direction's sign is
arbitrary, so the comparison is on |AUROC - 0.5|; p = (1 + #{dev_rand >= dev_axis}) / (n+1);
`random_p95_auroc_dev` = 0.5 + the 95th percentile of the random deviations.
"""
import json, os, sys, numpy as np, torch
from scipy.stats import rankdata

R = "C:/Users/user4/expLB/run-coords/passB"
BANK = "C:/Users/user4/expLB/run-paper16/axes_vectors_s3_fp32.pt"
L, NR, SEED = 40, 1000, 42
DS = ["ai_liar", "convincing-game", "harm-pressure-choice", "harm-pressure-knowledge-report",
      "insider-trading", "instructed-deception", "mask", "roleplaying", "sandbagging_v2"]

def aurocs_columns(S, y):
    r = rankdata(S, axis=0); n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2) / (n1 * n0)

def load(ds):
    d = os.path.join(R, ds)
    meta = {}
    with open(os.path.join(d, "rows.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line); meta[r["row_id"]] = r
    X, y, li, pi = [], [], None, None
    for c in sorted(f for f in os.listdir(d) if f.startswith("chunk_")):
        b = torch.load(os.path.join(d, c), weights_only=False)
        if li is None:
            li = list(b["layers"]).index(L)
            pi = list(b.get("pooling", ["mean", "last"])).index("mean")
        for i, rid in enumerate(b["row_ids"]):
            m = meta.get(rid)
            if m is None or m.get("label") is None: continue
            X.append(b["pooled"][i, li, :, pi].float().numpy()); y.append(int(m["label"]))
        del b
    return np.array(X, dtype=np.float32), np.array(y)

bank = torch.load(BANK, map_location="cpu", weights_only=False)
v = bank["axes"]["deception"][L].float().numpy()
v = v / np.linalg.norm(v)
a = bank["axes"]["assistant"][L].float().numpy(); a = a / np.linalg.norm(a)
rng = np.random.default_rng(SEED)
D = rng.standard_normal((len(v), NR)).astype(np.float32)
D /= np.linalg.norm(D, axis=0, keepdims=True)

print(f"bank {BANK}  layer {L}  pooling mean  n_random {NR}  seed {SEED}")
print(f"\n{'dataset':34s}{'n':>6}{'dec AUROC':>11}{'rand p95':>10}{'p':>8}{'beats':>7}{'asst AUROC':>12}")
allS, allSa, allR, ally = [], [], [], []
for ds in DS:
    X, y = load(ds)
    sd = X @ v; sa = X @ a; SR = X @ D
    allS.append(sd); allSa.append(sa); allR.append(SR); ally.append(y)
    ad = aurocs_columns(sd[:, None], y)[0]; aa = aurocs_columns(sa[:, None], y)[0]
    ar = aurocs_columns(SR, y)
    dev = np.abs(ar - 0.5); dax = abs(ad - 0.5)
    p95 = float(np.quantile(dev, 0.95)) + 0.5
    p = (1 + int((dev >= dax).sum())) / (NR + 1)
    print(f"{ds:34s}{len(y):6d}{ad:11.4f}{p95:10.4f}{p:8.4f}{str(dax > (p95-0.5)):>7}{aa:12.4f}")
y = np.concatenate(ally); sd = np.concatenate(allS); sa = np.concatenate(allSa); SR = np.vstack(allR)
ad = aurocs_columns(sd[:, None], y)[0]; aa = aurocs_columns(sa[:, None], y)[0]
ar = aurocs_columns(SR, y); dev = np.abs(ar - 0.5)
print(f"\nPOOLED  n={len(y)}  deception {ad:.4f}  assistant {aa:.4f}")
print(f"  random p95 {0.5+float(np.quantile(dev,0.95)):.4f}  random max {0.5+float(dev.max()):.4f}")
print(f"  p(deception) {(1+int((dev>=abs(ad-0.5)).sum()))/(NR+1):.6f}   p(assistant) {(1+int((dev>=abs(aa-0.5)).sum()))/(NR+1):.6f}")
