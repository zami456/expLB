"""S-03 / S-04 (READ-ONLY, lab PC): the kd bank's own geometry.
S-03: the six axis norms at L40, the PC1 variance shares of the three contrast bases, and the
      'default' anchor's relative position on PC1.
S-04: does run-kd/axes_vectors_all.pt exist, and how does it differ from axes_vectors_s3.pt?
Both .pt files were rewritten on 2026-09-22 15:41 (RF-13), so the per-role vectors (09-20, untouched)
are used to re-derive the axes independently."""
import json, os, torch, statistics

R = "C:/Users/user4/expLB/run-kd"
L = 40

def describe(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    print("=" * 72); print(path, os.path.getsize(path), "bytes, mtime",
          __import__("datetime").datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M"))
    print("top keys:", list(d.keys()))
    print("n roles:", len(d.get("roles", [])), "| vectors_dir:", d.get("vectors_dir"))
    print("counts:", d.get("counts"))
    ax = d["axes"]
    print(f"axis norms at L{L}:")
    for k in ax: print(f"   {k:20s} {float(ax[k][L].float().norm()):.4f}")
    if "pca" in d:
        for name, sub in d["pca"].items():
            if isinstance(sub, dict):
                vr = sub.get("variance_ratio")
                if vr is not None:
                    v = vr[L] if hasattr(vr, "__getitem__") else None
                    print(f"   pca[{name}] variance_ratio at L{L}: {[round(float(x),4) for x in v][:5]}")
                for kk in sub:
                    if kk not in ("variance_ratio",):
                        val = sub[kk]
                        print(f"      pca[{name}].{kk}: {tuple(val.shape) if torch.is_tensor(val) else str(val)[:120]}")
    if "pca_variance_ratio" in d:
        print(f"   pca_variance_ratio (all-roles basis) at L{L}: {[round(float(x),4) for x in d['pca_variance_ratio'][L]][:5]}")
    return d

s3 = describe(f"{R}/axes_vectors_s3.pt")
al = describe(f"{R}/axes_vectors_all.pt")

print("\n" + "=" * 72)
print("S-04: s3 vs all --- difference")
print("roles identical:", s3["roles"] == al["roles"])
for k in s3["axes"]:
    a, b = s3["axes"][k][L].float(), al["axes"][k][L].float()
    cos = float(torch.dot(a, b) / (a.norm() * b.norm()))
    print(f"   {k:20s} norm s3 {float(a.norm()):.4f}  all {float(b.norm()):.4f}   cos {cos:.6f}")
if "pca_components" in s3 and "pca_components" in al:
    A, B = s3["pca_components"][L].float(), al["pca_components"][L].float()
    print("   pca_components max|cos| per PC:",
          [round(float(abs(torch.dot(A[i], B[i]) / (A[i].norm() * B[i].norm()))), 4) for i in range(A.shape[0])])

print("\n" + "=" * 72)
print("S-03: default anchor's relative position on PC1 (over the role vectors of each bank)")
for tag, d, vdir in (("s3", s3, f"{R}/vectors_s3"), ("all", al, f"{R}/vectors_all")):
    roles = d["roles"]
    V = []
    for r in roles:
        t = torch.load(f"{vdir}/{r}.pt", map_location="cpu", weights_only=False)
        if isinstance(t, dict):
            t = [v for v in t.values() if torch.is_tensor(v) and v.ndim == 2][0]
        V.append(t[L].float())
    V = torch.stack(V)
    pc1 = d["pca_components"][L][0].float()
    proj = V @ pc1
    i = roles.index("default")
    lo, hi = float(proj.min()), float(proj.max())
    print(f"   [{tag}] n roles {len(roles)}  default proj {float(proj[i]):.4f}  min {lo:.4f} max {hi:.4f}"
          f"  -> relative position {(float(proj[i]) - lo) / (hi - lo):.4f}")
    # also with the mean-centred convention
    Vc = V - V.mean(0, keepdim=True)
    projc = Vc @ pc1
    lo2, hi2 = float(projc.min()), float(projc.max())
    print(f"        mean-centred: default {float(projc[i]):.4f} -> relative position "
          f"{(float(projc[i]) - lo2) / (hi2 - lo2):.4f}")
    # cos(PC1, assistant axis) across layers
    cs = []
    for l in range(d["pca_components"].shape[0]):
        p = d["pca_components"][l][0].float(); a = d["axes"]["assistant"][l].float()
        cs.append(abs(float(torch.dot(p, a) / (p.norm() * a.norm()))))
    print(f"        |cos(PC1, assistant)| over {len(cs)} layers: min {min(cs):.4f} median {statistics.median(cs):.4f} max {max(cs):.4f}")
    vr = d["pca_variance_ratio"]
    print(f"        all-roles PCA: PC1 {float(vr[L][0]):.4f}  top5 {float(vr[L][:5].sum()):.4f}")

print("\n" + "=" * 72)
print("S-03 level 2: re-derive the six axes from the per-role vectors (09-20, untouched by the 09-22 rewrite)")
g = s3["role_groups"]
roles = s3["roles"]
V = {}
for r in roles:
    t = torch.load(f"{R}/vectors_s3/{r}.pt", map_location="cpu", weights_only=False)
    if isinstance(t, dict):
        t = [v for v in t.values() if torch.is_tensor(v) and v.ndim == 2][0]
    V[r] = t[L].float()
print("role_groups:", {k: (v.get("roles") if isinstance(v, dict) else v) for k, v in g.items()})
