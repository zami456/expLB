"""W6G Level-2 recomputation of persona-bank geometry, run READ-ONLY on the lab PC.

Covers:
  S-24  cap bank: cos(capitulation, {capitulation, cap_vs_default, cap_only}) and norms at L40
  S-25  cap bank: role sets of the three PCA bases; max|A-B| and max|A-C| over all layers/PCs
  S-26  kd s3 bank: all-roles PCA variance share (PC1, top-5) and cos(PC1, assistant) by layer
  S-31  8B s23 bank: same geometry block, plus per-behaviour PCA

Loads only the axes bundles (.pt) - no activations, no model, nothing written.
"""
import torch
import torch.nn.functional as F


def hdr(t):
    print("\n==================== " + t)


def load(p):
    return torch.load(p, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------- S-24 / S-25
hdr("S-24 / S-25  run-cap/axes_s23.pt")
b = load(r"C:\Users\user4\expLB\run-cap\axes_s23.pt")
p, ax = b["pca"], b["axes"]
print("axes keys :", sorted(ax))
print("pca  keys :", sorted(p))
for k in sorted(p):
    print("  basis %-16s n=%3d  %s" % (k, len(p[k]["roles"]), sorted(p[k]["roles"])))

A, B, C = (p[n]["components"] for n in ("capitulation", "cap_vs_default", "cap_only"))
print("\ncomponents shape:", tuple(A.shape), A.dtype)
print("A roles == B roles :", sorted(p["capitulation"]["roles"]) == sorted(p["cap_vs_default"]["roles"]))
print("A roles == C roles :", sorted(p["capitulation"]["roles"]) == sorted(p["cap_only"]["roles"]))
print("max|A-B| over all layers/PCs : %r" % (A - B).abs().max().item())
print("max|A-C| over all layers/PCs : %r" % (A - C).abs().max().item())
# sign-invariant version: is the SUBSPACE the same?
print("max|A-B| after per-(layer,PC) sign alignment : %r"
      % (A - B * torch.sign((A * B).sum(-1, keepdim=True))).abs().max().item())
print("max|A-C| after per-(layer,PC) sign alignment : %r"
      % (A - C * torch.sign((A * C).sum(-1, keepdim=True))).abs().max().item())

print("\ncontrast directions at L40:")
for n in ("capitulation", "cap_vs_default", "cap_only"):
    c = F.cosine_similarity(ax["capitulation"][40].float(), ax[n][40].float(), dim=0).item()
    print("  cos(capitulation, %-16s) = %+.4f   norm=%.3f" % (n, c, ax[n][40].float().norm().item()))

print("\ncos(PC1, own contrast) at L40:")
for n in ("capitulation", "cap_vs_default", "cap_only"):
    pc1 = p[n]["components"][40, 0].float()
    print("  %-16s %+.4f  cos(PC1,assistant)=%+.4f  PC1var=%.1f%%"
          % (n, F.cosine_similarity(pc1, ax[n][40].float(), dim=0).item(),
             F.cosine_similarity(pc1, ax["assistant"][40].float(), dim=0).item(),
             100 * float(p[n]["variance_ratio"][40, 0])))

# ---------------------------------------------------------------- S-26
hdr("S-26  run-kd/axes_vectors_s3.pt  (all-roles PCA)")
k = load(r"C:\Users\user4\expLB\run-kd\axes_vectors_s3.pt")
kp, kax = k["pca"], k["axes"]
print("axes keys :", sorted(kax))
print("pca  keys :", sorted(kp))
allb = kp["all"]
print("all-basis n roles :", len(allb["roles"]))
vr = allb["variance_ratio"]
print("variance_ratio shape:", tuple(vr.shape))
print("L40  PC1 = %.4f%%   top-5 = %.4f%%" % (100 * float(vr[40, 0]), 100 * float(vr[40, :5].sum())))
cos = []
nL = allb["components"].shape[0]
for L in range(nL):
    pc1 = allb["components"][L, 0].float()
    cos.append(F.cosine_similarity(pc1, kax["assistant"][L].float(), dim=0).item())
import statistics
print("cos(PC1,assistant) at L40 = %+.4f" % cos[40])
print("across %d layers: min %+.4f  median %+.4f  max %+.4f" % (nL, min(cos), statistics.median(cos), max(cos)))
print("per-behaviour PCA at L40:")
for n in sorted(kp):
    if n == "all":
        continue
    pc1 = kp[n]["components"][40, 0].float()
    cc = F.cosine_similarity(pc1, kax[n][40].float(), dim=0).item() if n in kax else float("nan")
    print("  %-18s n=%3d cos(PC1,contrast)=%+.3f cos(PC1,assistant)=%+.3f PC1var=%.1f%%"
          % (n, len(kp[n]["roles"]), cc,
             F.cosine_similarity(pc1, kax["assistant"][40].float(), dim=0).item(),
             100 * float(kp[n]["variance_ratio"][40, 0])))

# ---------------------------------------------------------------- S-31
hdr("S-31  run-scale/8b/axes_8b.pt  (which build survives?)")
s = load(r"C:\Users\user4\expLB\run-scale\8b\axes_8b.pt")
sp, sax = s["pca"], s["axes"]
print("axes keys :", sorted(sax))
print("meta keys :", [kk for kk in s if kk not in ("axes", "pca")])
for kk in s:
    if kk not in ("axes", "pca"):
        print("   ", kk, "=", repr(s[kk])[:300])
ab = sp["all"]
print("all-basis n roles :", len(ab["roles"]))
print("L16  PC1 = %.4f%%   top-5 = %.4f%%" % (100 * float(ab["variance_ratio"][16, 0]),
                                              100 * float(ab["variance_ratio"][16, :5].sum())))
cos = []
for L in range(ab["components"].shape[0]):
    cos.append(F.cosine_similarity(ab["components"][L, 0].float(), sax["assistant"][L].float(), dim=0).item())
print("cos(PC1,assistant) L16 = %+.4f ; across %d layers min %+.4f median %+.4f"
      % (cos[16], len(cos), min(cos), statistics.median(cos)))
print("per-behaviour PCA at L16:")
for n in sorted(sp):
    if n == "all":
        continue
    pc1 = sp[n]["components"][16, 0].float()
    cc = F.cosine_similarity(pc1, sax[n][16].float(), dim=0).item() if n in sax else float("nan")
    print("  %-14s n=%3d cos(PC1,contrast)=%+.3f cos(PC1,assistant)=%+.3f PC1var=%.1f%%"
          % (n, len(sp[n]["roles"]), cc,
             F.cosine_similarity(pc1, sax["assistant"][16].float(), dim=0).item(),
             100 * float(sp[n]["variance_ratio"][16, 0])))
print("assistant norm at L16 = %.3f  deception %.3f  sycophancy %.3f  unified %.3f"
      % tuple(sax[n][16].float().norm().item() for n in ("assistant", "deception", "sycophancy", "unified")))
