"""S-04 substitution test (READ-ONLY, lab PC).
Does swapping run-kd/axes_vectors_s3.pt (score-3-judged rollouts) for
run-kd/axes_vectors_all.pt (ALL rollouts, no judge filter) change the kd bank's
zero-shot readings?  Same rows, same pooling, same layer -- only the bank changes.

Streams run-coords/passA chunk by chunk and keeps only the layer-L mean-pooled slice.
"""
import json, os, sys, numpy as np, torch
from sklearn.metrics import roc_auc_score

R = "C:/Users/user4/expLB/run-coords/passA"
BANKS = {"s3": "C:/Users/user4/expLB/run-kd/axes_vectors_s3.pt",
         "all": "C:/Users/user4/expLB/run-kd/axes_vectors_all.pt"}
LAYERS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["20", "40", "44"])]
DSETS = sys.argv[2].split(",") if len(sys.argv) > 2 else None

banks = {t: torch.load(p, map_location="cpu", weights_only=False) for t, p in BANKS.items()}
AXES = ["deception", "unified", "knowledge_denial", "kd_assertion", "assistant", "sycophancy"]

def load_layer(ds, layers):
    d = os.path.join(R, ds)
    meta = {}
    with open(os.path.join(d, "rows.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line); meta[r["row_id"]] = r
    X, y, lay_idx = {L: [] for L in layers}, [], None
    for c in sorted(f for f in os.listdir(d) if f.startswith("chunk_")):
        blob = torch.load(os.path.join(d, c), weights_only=False)
        if lay_idx is None:
            ls = list(blob["layers"]); pools = list(blob.get("pooling", ["mean", "last"]))
            pi = pools.index("mean"); lay_idx = {L: ls.index(L) for L in layers}
        for i, rid in enumerate(blob["row_ids"]):
            m = meta.get(rid)
            if m is None or m.get("label") is None: continue
            for L in layers:
                X[L].append(blob["pooled"][i, lay_idx[L], :, pi].float().numpy())
            y.append(int(m["label"]))
        del blob
    return {L: np.array(X[L]) for L in layers}, np.array(y)

names = DSETS or [d for d in sorted(os.listdir(R))
                  if os.path.isdir(os.path.join(R, d)) and d != "alpaca"]
print("datasets:", names, "layers:", LAYERS)
print(f"\n{'dataset':32s}{'L':>4}{'axis':20s}{'AUROC s3':>10}{'AUROC all':>11}{'diff':>9}{'n':>6}")
agg = {}
for ds in names:
    try:
        X, y = load_layer(ds, LAYERS)
    except Exception as e:
        print(f"{ds:32s}  LOAD FAILED: {e}"); continue
    if len(set(y.tolist())) < 2:
        print(f"{ds:32s}  one class only, skipped"); continue
    for L in LAYERS:
        for ax in AXES:
            s3 = banks["s3"]["axes"][ax][L].float().numpy()
            al = banks["all"]["axes"][ax][L].float().numpy()
            a1 = roc_auc_score(y, X[L] @ s3); a2 = roc_auc_score(y, X[L] @ al)
            agg.setdefault((L, ax), []).append((a1, a2))
            print(f"{ds:32s}{L:4d}{ax:20s}{a1:10.4f}{a2:11.4f}{a2-a1:+9.4f}{len(y):6d}")
print("\nmean over datasets:")
for (L, ax), v in sorted(agg.items()):
    a = np.array(v)
    print(f"  L{L:<4d}{ax:20s} s3 {a[:,0].mean():.4f}  all {a[:,1].mean():.4f}  diff {a[:,1].mean()-a[:,0].mean():+.4f}"
          f"  max|diff| {np.abs(a[:,1]-a[:,0]).max():.4f}  n_ds {len(v)}")
