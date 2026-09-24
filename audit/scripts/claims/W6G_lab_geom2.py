"""W6G Level-2 part 2 (lab, read-only):

S-10  own100 bank `run/axes.pt`: cos(dec,asst), cos(syc,asst), cos(dec,syc), cos(PC1,asst)
      by layer; plus the group-mean projection table at L40 and the Spearman rho between the
      deception-axis and assistant-axis orderings of the six role groups.
S-24  cap bank: are cap_vs_default and cap_only literally the same contrast vector?
"""
import statistics
import torch
import torch.nn.functional as F

print("==================== S-10  run/axes.pt (own100)")
b = torch.load(r"C:\Users\user4\expLB\run\axes.pt", map_location="cpu", weights_only=False)
print("top keys:", [k for k in b])
ax = b["axes"]
print("axes:", sorted(ax))
pca = b.get("pca")
if pca is not None:
    print("pca bases:", sorted(pca))


def cos(a, c, L):
    return F.cosine_similarity(ax[a][L].float(), ax[c][L].float(), dim=0).item()


print("\n  layer  cos(dec,asst)  cos(syc,asst)  cos(dec,syc)")
for L in (20, 22, 40, 60, 79):
    print("  %5d      %+.4f        %+.4f       %+.4f" % (L, cos("deception", "assistant", L),
                                                         cos("sycophancy", "assistant", L),
                                                         cos("deception", "sycophancy", L)))
nL = ax["assistant"].shape[0]
sc = [cos("sycophancy", "assistant", L) for L in range(nL)]
print("  cos(syc,asst) over all %d layers: min %+.4f  max %+.4f" % (nL, min(sc), max(sc)))
c40 = cos("deception", "assistant", 40)
print("  cos(dec,asst) L40 = %+.4f  -> cos^2 = %.4f  -> orthogonal share = %.1f%%"
      % (c40, c40 ** 2, 100 * (1 - c40 ** 2)))

if pca is not None and "all" in pca:
    for L in (20, 22, 40, 60):
        pc1 = pca["all"]["components"][L, 0].float()
        print("  L%-3d cos(PC1,assistant) = %+.4f  PC1 var %.1f%%"
              % (L, F.cosine_similarity(pc1, ax["assistant"][L].float(), dim=0).item(),
                 100 * float(pca["all"]["variance_ratio"][L, 0])))

# group-mean projections at L40 and the Spearman between the two orderings
groups = b.get("role_groups")
roles = b.get("roles")
vec = b.get("role_vectors") or b.get("vectors")
print("\n  role_groups present:", bool(groups), " roles:", len(roles) if roles else None,
      " vectors key present:", vec is not None)
if groups:
    print("  groups:", {g: len(v["roles"]) if isinstance(v, dict) else len(v) for g, v in groups.items()})

print("\n  Spearman rho, from the six group means the external review quoted")
dec = {"deception": -0.78, "breadth_control": -1.25, "anti_sycophancy": -1.86,
       "sycophancy": -1.95, "honesty": -2.50, "assistant_anchor": -2.96}
asst = {"assistant_anchor": 1.86, "honesty": 1.13, "sycophancy": 0.77,
        "anti_sycophancy": 0.54, "deception": -0.00, "breadth_control": -0.66}
ks = sorted(dec)
rd = {k: i for i, k in enumerate(sorted(ks, key=lambda k: dec[k]), 1)}
ra = {k: i for i, k in enumerate(sorted(ks, key=lambda k: asst[k]), 1)}
n = len(ks)
d2 = sum((rd[k] - ra[k]) ** 2 for k in ks)
print("  n=%d  sum d^2=%d  rho = 1 - 6*%d/(%d*(%d^2-1)) = %.6f" % (n, d2, d2, n, n, 1 - 6 * d2 / (n * (n * n - 1))))

print("\n==================== S-24  cap contrasts: identical or merely collinear?")
c = torch.load(r"C:\Users\user4\expLB\run-cap\axes_s23.pt", map_location="cpu", weights_only=False)
cax = c["axes"]
a, bb, cc = cax["capitulation"][40].float(), cax["cap_vs_default"][40].float(), cax["cap_only"][40].float()
print("  cos(cap_vs_default, cap_only) at L40 = %+.6f" % F.cosine_similarity(bb, cc, dim=0).item())
print("  max |cap_vs_default - cap_only| over ALL layers = %r"
      % (cax["cap_vs_default"].float() - cax["cap_only"].float()).abs().max().item())
print("  norms at L40: capitulation %.4f  cap_vs_default %.4f  cap_only %.4f"
      % (a.norm().item(), bb.norm().item(), cc.norm().item()))
print("  counts:", c.get("counts"))
