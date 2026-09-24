import sys, torch, numpy as np
p = sys.argv[1] if len(sys.argv)>1 else "run-paper16/axes_vectors_s3.pt"
d = torch.load(p, map_location="cpu", weights_only=False)
ax = d["axes"]
for k,v in ax.items():
    print("axis", k, "shape", tuple(v.shape), v.dtype)
print("counts", {k:(v if not torch.is_tensor(v) else v.tolist()) for k,v in d["counts"].items()})
A = {k: v.float() for k,v in ax.items()}
names = list(A)
L = A[names[0]].shape[0]
print("layers", L)
# per-layer norms
import numpy as np
N = {k: A[k].norm(dim=-1).numpy() for k in names}
print("\nlayer  " + "  ".join(f"{k:>12}" for k in names))
for l in range(L):
    print(f"{l:5d}  " + "  ".join(f"{N[k][l]:12.4f}" for k in names))
print("\nsummary per axis: min / median / max / mean over 80 layers")
for k in names:
    print(f"  {k:12s} min {N[k].min():.4f} med {np.median(N[k]):.4f} max {N[k].max():.4f}")
# which layer best matches the transcript quartet
targ = {"assistant":1.012, "deception":1.0037, "sycophancy":1.535, "unified":0.9722}
best=None
for l in range(L):
    err = sum(abs(N[k][l]-targ[k]) for k in targ)
    if best is None or err < best[1]: best=(l,err)
print("\nbest-matching layer for transcript quartet:", best[0], "sum|diff| =", f"{best[1]:.6f}")
l=best[0]
print("   at that layer:", {k: round(float(N[k][l]),4) for k in targ})
# PCA component norms at that layer
pc = d["pca_components"].float()
print("\npca_components norms at layer", l, [round(float(x),4) for x in pc[l].norm(dim=-1)])
print("pca_components norms: global min %.6f max %.6f" % (float(pc.norm(dim=-1).min()), float(pc.norm(dim=-1).max())))
print("pca_variance_ratio at layer", l, [round(float(x),4) for x in d["pca_variance_ratio"][l]])
