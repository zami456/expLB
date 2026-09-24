"""F-65 (b)/(a) geometry from run-cap/axes_s23.pt at L40."""
import torch
o = torch.load("C:/Users/user4/expLB/run-cap/axes_s23.pt", map_location="cpu", weights_only=True)
L = 40
def cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))
ax = o["axes"]
pc = o["pca"]
for name in ("capitulation", "cap_vs_default", "cap_only", "all"):
    comp = pc[name]["components"][L]          # (5, 8192)
    vr = pc[name]["variance_ratio"][L]
    sign = pc[name]["sign"][L]
    pc1 = comp[0] * float(sign[0])
    print(f"{name:<16} PC1 var={float(vr[0]):.4f} | cos(PC1, assistant)={cos(pc1, ax['assistant'][L]):+.4f} "
          f"| cos(PC1, own contrast)={cos(pc1, ax[name][L]) if name in ax else float('nan'):+.4f} "
          f"| n_roles={len(pc[name]['roles'])}")
print(f"cos(contrast capitulation, contrast cap_vs_default) = {cos(ax['capitulation'][L], ax['cap_vs_default'][L]):+.4f}")
print(f"cos(contrast capitulation, contrast cap_only)       = {cos(ax['capitulation'][L], ax['cap_only'][L]):+.4f}")
print("roles capitulation:", pc["capitulation"]["roles"])
print("roles cap_vs_default:", pc["cap_vs_default"]["roles"])
print("roles all:", pc["all"]["roles"])
