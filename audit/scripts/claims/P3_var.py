"""F-65's 'PC1 var 56.8%': read pca_variance_ratio / pca[*] out of the cap bank properly."""
import numpy as np, torch
d = torch.load("C:/Users/user4/expLB/run-cap/axes_s23.pt", map_location="cpu", weights_only=False)
vr = d["pca_variance_ratio"]
print("pca_variance_ratio:", type(vr), getattr(vr, "shape", None))
a = np.asarray(vr)
print("as array shape:", a.shape)
if a.ndim >= 2:
    print("  row 40 (layer 40):", np.round(a[40][:5], 4))
print("\npca sub-dicts:")
for name, p in sorted(d["pca"].items()):
    print(" ", name, type(p), sorted(p.keys()) if isinstance(p, dict) else "")
    if isinstance(p, dict):
        for k in p:
            v = p[k]
            try:
                arr = np.asarray(v)
                extra = ""
                if arr.ndim >= 2 and arr.shape[0] > 40:
                    extra = f"  [40][:3]={np.round(np.asarray(arr[40]).ravel()[:3], 4)}"
                print(f"    {k}: shape={arr.shape}{extra}")
            except Exception as e:
                print(f"    {k}: {type(v)} ({e})")
