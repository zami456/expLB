"""P3 batch leftovers: file mtimes (RF-13 regeneration window), the cap bank's PC1 variance share,
the collab bank's 12-v-13 role groups, and the run-cap-ctrl rollout count. Read-only."""
import datetime, glob, hashlib, json, os
import numpy as np
import torch

LAB = "C:/Users/user4/expLB"
print("=== mtimes of every file this batch read (lab local time) ===")
files = [
    f"{LAB}/run-crossgen/kd11/matrices/mean_L40_s42.json",
    f"{LAB}/run-crossgen/kd11/banks/kd/significance.json",
    f"{LAB}/run-crossgen/kd11/banks/kd_s23/significance.json",
    f"{LAB}/run-crossgen/kd11/banks/cap_s23/matrices/mean_L40_s42.json",
    f"{LAB}/run-crossgen/kd11/banks/cap_all/matrices/mean_L40_s42.json",
    f"{LAB}/run-crossgen/kd11/banks/cap_ctrl/matrices/mean_L40_s42.json",
    f"{LAB}/run-crossgen/kd11/banks/cap_s23/significance.json",
    f"{LAB}/run-crossgen/kd11/banks/cap_all/significance.json",
    f"{LAB}/run-crossgen/kd11/banks/collab/matrices/mean_L20_s42.json",
    f"{LAB}/run-crossgen/kd11/banks/collab/matrices/mean_L40_s42.json",
    f"{LAB}/run-premise/premise_L20_mean_s42.json",
    f"{LAB}/run-premise/premise_L40_mean_s42.json",
    f"{LAB}/run-cap/axes_s23.pt",
    f"{LAB}/run-cap-ctrl/axes_s23.pt",
    f"{LAB}/run-kd/axes_collab_s23.pt",
]
for f in files:
    if os.path.exists(f):
        st = os.stat(f)
        print(f"  {datetime.datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M:%S}  {st.st_size:>10}  {f}")
    else:
        print(f"  MISSING                        {f}")

print("\n=== cap bank PC1 variance share at L40 (F-65's '56.8%') ===")
d = torch.load(f"{LAB}/run-cap/axes_s23.pt", map_location="cpu", weights_only=False)
vr = d.get("pca_variance_ratio")
if isinstance(vr, dict):
    for name in sorted(vr):
        try:
            v = np.asarray(vr[name][40], dtype=float)
            print(f"  {name:<18} PC1={v[0]:.4f}  PC2={v[1]:.4f}  (sum top2={v[:2].sum():.4f})")
        except Exception as e:
            print(f"  {name}: {e}")
else:
    print("  pca_variance_ratio type:", type(vr))

print("\n=== collab bank role groups (F-98's '12 v 13') ===")
p = f"{LAB}/run-kd/axes_collab_s23.pt"
if os.path.exists(p):
    c = torch.load(p, map_location="cpu", weights_only=False)
    print("  md5:", hashlib.md5(open(p, "rb").read()).hexdigest())
    print("  axes:", sorted(c.get("axes", {})))
    rg = c.get("role_groups", {})
    for k, v in (rg.items() if isinstance(rg, dict) else []):
        roles = v.get("roles") if isinstance(v, dict) else v
        print(f"  group {k:<22} n={len(roles)}  {sorted(roles)}")
    print("  vectors_dir:", c.get("vectors_dir"))
else:
    print("  MISSING", p)

print("\n=== run-cap-ctrl rollout count (F-89's '1,200 rollouts, 12 personas x 5 variants x 20 questions') ===")
for sub in ("responses", "scores", "vectors_s23", "vectors_all"):
    d2 = f"{LAB}/run-cap-ctrl/{sub}"
    if os.path.isdir(d2):
        fs = glob.glob(os.path.join(d2, "*"))
        n = 0
        for f in fs:
            if f.endswith(".jsonl"):
                n += sum(1 for _ in open(f, encoding="utf-8"))
        print(f"  {sub}: {len(fs)} files, {n} jsonl lines")
    else:
        print(f"  {sub}: not a directory")
print("  run-cap-ctrl top level:", sorted(os.path.basename(x) for x in glob.glob(f"{LAB}/run-cap-ctrl/*")))
