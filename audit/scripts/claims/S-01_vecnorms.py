"""S-01 level 2: persona vector norms at L20 and re-derivation of the four axes
as unnormalised difference-of-means from run-paper16/vectors_s3/*.pt."""
import torch, numpy as np, glob, os
AX = torch.load("run-paper16/axes_vectors_s3.pt", map_location="cpu", weights_only=False)
roles = AX["roles"]; groups = AX["role_groups"]; L = 20
V = {}
for r in roles:
    f = f"run-paper16/vectors_s3/{r}.pt"
    d = torch.load(f, map_location="cpu", weights_only=False)
    if isinstance(d, dict):
        # find the (layers, hidden) tensor
        t = None
        for k, v in d.items():
            if torch.is_tensor(v) and v.ndim == 2 and v.shape[0] >= 80:
                t = v; break
        if t is None:
            print("STRUCT", r, {k: (tuple(v.shape) if torch.is_tensor(v) else type(v).__name__) for k, v in d.items()}); raise SystemExit
        V[r] = t.float()
    else:
        V[r] = d.float()
print("n roles", len(V), "vec shape", tuple(next(iter(V.values())).shape))
n = np.array([float(V[r][L].norm()) for r in roles])
print(f"persona vector norms at L{L}: min {n.min():.4f}  median {np.median(n):.4f}  max {n.max():.4f}  mean {n.mean():.4f}")
print("role_groups:", {k: (v if isinstance(v, list) else v) for k, v in groups.items()})
print("counts:", AX["counts"])
# re-derive axes
def mean(rs): return torch.stack([V[r][L] for r in rs]).mean(0)
G = groups
cand = {
 "assistant": (G.get("assistant_anchor"), [r for r in roles if r not in (G.get("assistant_anchor") or [])]),
 "deception": (G.get("deception"), G.get("honesty")),
 "sycophancy": (G.get("sycophancy"), G.get("anti_sycophancy")),
 "unified": ((G.get("deception") or []) + (G.get("sycophancy") or []), (G.get("honesty") or []) + (G.get("anti_sycophancy") or [])),
}
for name, (pos, neg) in cand.items():
    if not pos or not neg: print(name, "SKIP", pos, neg); continue
    v = mean(pos) - mean(neg)
    s = AX["axes"][name][L].float()
    cos = float(torch.dot(v, s) / (v.norm() * s.norm()))
    print(f"{name:11s} n_pos={len(pos):2d} n_neg={len(neg):2d}  recomputed norm {float(v.norm()):.4f}  stored norm {float(s.norm()):.4f}  cos {cos:.6f}  max|diff| {float((v-s).abs().max()):.3e}")
