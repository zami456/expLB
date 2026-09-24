"""M2: (a) verify 6_axes.py:19-21's cancellation argument at EVERY layer and for every
contrast; (b) measure what including the default anchor in the PCA basis costs. Read-only."""
import torch, json, numpy as np
import torch.nn.functional as F
from pathlib import Path
R = Path(__file__).resolve().parents[2]
vecs = {}
for f in sorted((R/"run-paper16"/"vectors_s3").glob("*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    vecs[d["role"]] = d["vector"].double()
g = json.loads((R/"src/assistant_axis_ndif/pc_role_groups.json").read_text())["groups"]
DEC,HON = g["deception"]["roles"], g["honesty"]["roles"]
SYC,ANT = g["sycophancy"]["roles"], g["anti_sycophancy"]["roles"]
CASES = {"deception": (DEC, HON, DEC+HON+["default"]),
         "sycophancy": (SYC, ANT, SYC+ANT+["default"]),
         "unified":   (DEC+SYC, HON+ANT, DEC+HON+SYC+ANT+["default"])}
print("(a) Eq(2) centering cancels in Eq(4) -- max |d_uncentered - d_centered| over ALL 80 layers")
for name,(P,N,ALL) in CASES.items():
    S = torch.stack([vecs[r] for r in ALL])           # (n,80,H)
    mu = S.mean(0)                                    # (80,H)  Eq (2) mean over the persona set
    iP=[ALL.index(r) for r in P]; iN=[ALL.index(r) for r in N]
    d_raw = S[iP].mean(0) - S[iN].mean(0)
    d_ctr = (S[iP]-mu).mean(0) - (S[iN]-mu).mean(0)
    scale = d_raw.abs().max().item()
    print(f"  {name:11} max abs diff = {(d_raw-d_ctr).abs().max().item():.3e}"
          f"   (largest |d| coord = {scale:.3f} -> relative {(d_raw-d_ctr).abs().max().item()/scale:.1e})")

print("\n(b) anchor in the PCA basis: PC1 with vs without `default`")
def pc(names, k=3):
    Y = torch.stack([vecs[r] for r in names])                 # (n,80,H)
    Y = Y - Y.mean(0, keepdim=True)
    out=[]
    for L in range(Y.shape[1]):
        _,S,Vh = torch.linalg.svd(Y[:,L,:], full_matrices=False)
        vr=(S**2/(S**2).sum())
        out.append((Vh[:k], vr[:k]))
    return out
for name,(P,N,ALL) in CASES.items():
    w  = pc(ALL); wo = pc([r for r in ALL if r!="default"])
    cs=[abs(F.cosine_similarity(w[L][0][0], wo[L][0][0], dim=0).item()) for L in range(80)]
    vr_w=[w[L][1][0].item() for L in range(80)]; vr_o=[wo[L][1][0].item() for L in range(80)]
    print(f"  {name:11} n={len(ALL)} vs {len(ALL)-1} | |cos(PC1)| min {min(cs):.4f} med {np.median(cs):.4f}"
          f" | PC1 var L40 {vr_w[40]:.3f} -> {vr_o[40]:.3f}")
