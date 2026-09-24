"""M2: verify the extraction-vs-paper claims against the real artifacts. Read-only."""
import torch, json
from pathlib import Path
R = Path(__file__).resolve().parents[2]
V = R / "run-paper16" / "vectors_s3"
A = R / "run-paper16" / "axes_vectors_s3.pt"

vecs, meta = {}, {}
for f in sorted(V.glob("*.pt")):
    d = torch.load(f, map_location="cpu", weights_only=False)
    vecs[d["role"]] = d["vector"].float()
    meta[d["role"]] = {"type": d.get("type"), "n": d.get("n"), "layers": len(d.get("layers", []))}
print("roles:", len(vecs), "| shape:", tuple(next(iter(vecs.values())).shape))
print("types:", sorted({m['type'] for m in meta.values()}))
print("n per role (min/max/median):",
      min(m['n'] for m in meta.values()), max(m['n'] for m in meta.values()))
for r in ("default", "pc_handler", "pc_yes_man"):
    print("  ", r, meta[r])

g = json.loads((R / "src/assistant_axis_ndif/pc_role_groups.json").read_text())["groups"]
D = [f"{r}" for r in g["deception"]["roles"]]
H = [f"{r}" for r in g["honesty"]["roles"]]
P16 = D + H + ["default"]     # the paper's 16-persona deception set

L = 40
# --- Eq(2)+Eq(4) centering-cancellation, on the real layer-40 vectors ---
X   = torch.stack([vecs[r][L] for r in P16]).double()
mu  = X.mean(0)
Xc  = X - mu
d_raw = torch.stack([vecs[r][L] for r in D]).double().mean(0) - \
        torch.stack([vecs[r][L] for r in H]).double().mean(0)
iD = [P16.index(r) for r in D]; iH = [P16.index(r) for r in H]
d_ctr = Xc[iD].mean(0) - Xc[iH].mean(0)
print("\nEq(2)->Eq(4) cancellation at L%d over P=%d personas:" % (L, len(P16)))
print("  max |d_raw - d_centered| =", float((d_raw - d_ctr).abs().max()))
print("  ||d_raw|| =", float(d_raw.norm()), " cos =",
      float(torch.nn.functional.cosine_similarity(d_raw, d_ctr, dim=0)))
# and what if the anchor were dropped from the centering mean?
X15 = torch.stack([vecs[r][L] for r in D + H]).double()
mu15 = X15.mean(0)
d_ctr15 = (X15[:len(D)] - mu15).mean(0) - (X15[len(D):] - mu15).mean(0)
print("  centering WITHOUT the anchor: max diff =", float((d_raw - d_ctr15).abs().max()))

# --- does the anchor's presence change PCA / PC1? ---
def pc1(names):
    Y = torch.stack([vecs[r][L] for r in names]).double()
    Y = Y - Y.mean(0, keepdim=True)
    _, S, Vh = torch.linalg.svd(Y, full_matrices=False)
    return Vh[0], (S**2 / (S**2).sum())
u_with, vr_with = pc1(P16)
u_wo,  vr_wo  = pc1(D + H)
c = float(torch.nn.functional.cosine_similarity(u_with, u_wo, dim=0))
print("\nPC1 with anchor (n=16) vs without (n=15): cos = %+.4f" % c)
print("  PC1 var ratio  with=%.4f  without=%.4f" % (vr_with[0], vr_wo[0]))
print("  cos(PC1_with, contrast) = %+.4f   cos(PC1_without, contrast) = %+.4f" % (
      torch.nn.functional.cosine_similarity(u_with, d_raw, dim=0),
      torch.nn.functional.cosine_similarity(u_wo,  d_raw, dim=0)))

# --- what the saved axes file actually holds ---
ax = torch.load(A, map_location="cpu", weights_only=False)
print("\naxes file:", A.name)
print("  axes:", list(ax["axes"].keys()))
print("  counts:", ax["counts"])
print("  pca bases:", {k: len(v["roles"]) for k, v in ax["pca"].items()})
print("  pcs kept:", tuple(ax["pca"]["deception"]["components"].shape))
print("  anchor in deception basis:", "default" in ax["pca"]["deception"]["roles"])
print("  anchor in unified basis:", "default" in ax["pca"]["unified"]["roles"])
sd = ax["axes"]["deception"][L].double()
print("  saved deception axis == uncentered class-mean diff: max|Δ| =",
      float((sd - d_raw).abs().max()))
