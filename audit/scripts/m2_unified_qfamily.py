"""M2: does the paper16 unified PCA's PC1 track the QUESTION SET rather than harmfulness?
Read-only. Reproduces the memory claim from raw vectors."""
import torch, json, numpy as np
from pathlib import Path
R = Path(__file__).resolve().parents[2]
V = R / "run-paper16" / "vectors_s3"
vecs = {}
for f in sorted(V.glob("*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    vecs[d["role"]] = d["vector"].float()
g = json.loads((R/"src/assistant_axis_ndif/pc_role_groups.json").read_text())["groups"]
DEC, HON = g["deception"]["roles"], g["honesty"]["roles"]
SYC, ANT = g["sycophancy"]["roles"], g["anti_sycophancy"]["roles"]
UNI = DEC + HON + SYC + ANT + ["default"]
# which question bank each persona was rolled out on (verified from responses/*.jsonl)
qfam = np.array([1.0 if r in DEC + HON + ["default"] else 0.0 for r in UNI])
harm = np.array([1.0 if r in DEC + SYC else (0.0 if r in HON + ANT else 0.5) for r in UNI])
for L in (20, 40, 60):
    X = torch.stack([vecs[r][L] for r in UNI]).double()
    Xc = X - X.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    vr = (S**2 / (S**2).sum()).numpy()
    load = (Xc @ Vh[0]).numpy()          # PC1 loading per persona
    load2 = (Xc @ Vh[1]).numpy()
    d_uni = (torch.stack([vecs[r][L] for r in DEC+SYC]).double().mean(0)
             - torch.stack([vecs[r][L] for r in HON+ANT]).double().mean(0))
    cq = np.corrcoef(load, qfam)[0, 1]
    ch = np.corrcoef(load, harm)[0, 1]
    cq2 = np.corrcoef(load2, qfam)[0, 1]
    ch2 = np.corrcoef(load2, harm)[0, 1]
    print(f"L{L}: PC1 var={vr[0]:.3f} PC2 var={vr[1]:.3f} | "
          f"corr(PC1,question-set)={cq:+.3f} corr(PC1,harmful)={ch:+.3f} | "
          f"corr(PC2,question-set)={cq2:+.3f} corr(PC2,harmful)={ch2:+.3f} | "
          f"cos(PC1,unified contrast)={float(torch.nn.functional.cosine_similarity(Vh[0], d_uni, dim=0)):+.3f}")
