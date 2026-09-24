"""F-65 geometry sub-claims, read from the stored axes files (read-only torch.load).

Checks: cos(PC1, contrast) for every basis (the re-run gate 0.987/0.949/0.936), PC1 variance share,
cos(PC1, assistant) = -0.838, cos(contrast_A, contrast_B) = 0.931, and which roles each basis spans.
Also fingerprints cap_s23 vs cap_ctrl, whose bank files share the basename `axes_s23.pt`.

Usage: python F-65b.py <axes file> [<axes file> ...]   (e.g. run-cap/axes_s23.pt run-cap-ctrl/axes_s23.pt)
"""
import hashlib, os, sys
import numpy as np
import torch


def cos(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


for path in sys.argv[1:]:
    print("=" * 78)
    if not os.path.exists(path):
        print("MISSING", path); continue
    h = hashlib.md5(open(path, "rb").read()).hexdigest()
    d = torch.load(path, map_location="cpu", weights_only=False)
    print(path, "| size", os.path.getsize(path), "| md5", h)
    print("  top-level keys:", sorted(d.keys()) if isinstance(d, dict) else type(d))
    axes = d.get("axes", {})
    pca = d.get("pca", {})
    print("  axes    :", sorted(axes))
    print("  pca     :", sorted(pca))
    for k in ("roles", "role_groups", "layers", "bank_name", "source", "report"):
        if k in d:
            v = d[k]
            print(f"  {k:<11}:", (sorted(v) if isinstance(v, dict) else v) if len(str(v)) < 400 else str(v)[:400] + " ...")
    L = 40
    for name in sorted(pca):
        p = pca[name]
        comps = p["components"] if isinstance(p, dict) else None
        if comps is None:
            continue
        c0 = np.asarray(comps[L][0], dtype=np.float64)
        sign = 1.0
        if isinstance(p, dict) and "sign" in p and p["sign"] is not None:
            try:
                sign = float(p["sign"][L][0])
            except Exception:
                sign = 1.0
        pc1 = c0 * sign
        line = f"  L{L} basis {name:<18}"
        if name in axes:
            line += f" cos(PC1,contrast)={cos(pc1, np.asarray(axes[name][L], dtype=np.float64)):+.4f}"
        if isinstance(p, dict) and "explained_variance_ratio" in p and p["explained_variance_ratio"] is not None:
            try:
                line += f"  PC1 var={float(p['explained_variance_ratio'][L][0]):.4f}"
            except Exception:
                pass
        if "assistant" in axes:
            line += f"  cos(PC1,assistant)={cos(pc1, np.asarray(axes['assistant'][L], dtype=np.float64)):+.4f}"
        if isinstance(p, dict) and "roles" in p:
            line += f"  n_roles={len(p['roles'])}"
        print(line)
    print(f"  L{L} contrast-vs-contrast cosines:")
    names = sorted(axes)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            print(f"    cos({a}, {b}) = {cos(np.asarray(axes[a][L], dtype=np.float64), np.asarray(axes[b][L], dtype=np.float64)):+.4f}")
    for name in sorted(pca):
        p = pca[name]
        if isinstance(p, dict) and "roles" in p:
            print(f"  roles[{name}] ({len(p['roles'])}):", sorted(p["roles"]))
