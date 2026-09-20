"""Is pc:capitulation == pc:cap_vs_default a RESULT or a tautology?

`make_features` for `pc:<name>` reads bank["pca"][name]["components"] -- the top-k PCs of the
named BASIS. It never touches the contrast direction. So if A and B are built over the same
role set, they are identical by construction and the crossgen agreement says nothing about
whether the holding personas matter to the contrast.
"""
import torch, torch.nn.functional as F

b = torch.load("run-cap/axes_s23.pt", map_location="cpu", weights_only=False)
p, ax = b["pca"], b["axes"]

print("PCA bases:")
for k in p:
    print(f"  {k:16} n={len(p[k]['roles']):>3}  {sorted(p[k]['roles'])}")

A, B, C = (p[n]["components"] for n in ("capitulation", "cap_vs_default", "cap_only"))
print()
print("A roles == B roles :", sorted(p["capitulation"]["roles"]) == sorted(p["cap_vs_default"]["roles"]))
print("max|A-B| over all layers/PCs :", (A - B).abs().max().item())
print("max|A-C| over all layers/PCs :", (A - C).abs().max().item())

print("\ncontrast directions at L40 (these are what A vs B was meant to isolate):")
for n in ("capitulation", "cap_vs_default", "cap_only"):
    c = F.cosine_similarity(ax["capitulation"][40], ax[n][40], dim=0).item()
    print(f"  cos(capitulation, {n:16}) = {c:+.4f}   norm={ax[n][40].norm():.3f}")

print("\ncos(PC1, own contrast) at L40  -- the 2e gate, never printed by the report:")
for n in ("capitulation", "cap_vs_default", "cap_only"):
    pc1 = p[n]["components"][40, 0]
    print(f"  {n:16} {F.cosine_similarity(pc1, ax[n][40], dim=0).item():+.4f}"
          f"   cos(PC1,assistant)={F.cosine_similarity(pc1, ax['assistant'][40], dim=0).item():+.4f}"
          f"   PC1var={p[n]['variance_ratio'][40,0]:.1%}")
