"""F-65 layer profile: trained target-column AUROC for cap features at L20/40/44 (s23), plus axes_s23.pt geometry."""
import json, os, statistics
root = "C:/Users/user4/expLB/run-crossgen/kd11"
TARGET = "sycophancy_mt"
for L in (20, 40, 44):
    p = os.path.join(root, "banks/cap_s23/matrices", f"mean_L{L}_s42.json")
    b = os.path.join(root, "matrices", f"mean_L{L}_s42.json")
    d = json.load(open(p, encoding="utf-8")); ds = d["datasets"]; i = ds.index(TARGET)
    idx = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}
    bd = json.load(open(b, encoding="utf-8")); bidx = {(r["feature"], r["probe"]): r["auroc"] for r in bd["results"]}
    def col(ix, feat):
        a = ix.get((feat, "paper"))
        if a is None: return None
        v = [a[j][i] for j in range(len(ds)) if j != i and a[j][i] is not None]
        return statistics.fmean(v)
    raw = col(bidx, "raw")
    out = {f: col(idx, f) for f in ("pc:capitulation@2", "pc:cap_only@2", "pc:capitulation", "pc:cap_only")}
    print(f"L{L}: raw={raw:.4f} " + " ".join(f"{k}={v:.4f}" for k, v in out.items() if v is not None)
          + f" | cap@2-cap_only@2={out['pc:capitulation@2']-out['pc:cap_only@2']:+.4f}")
print("\n=== run-cap/axes_s23.pt ===")
import torch
o = torch.load("C:/Users/user4/expLB/run-cap/axes_s23.pt", map_location="cpu", weights_only=True)
def walk(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict): walk(v, pre + k + ".")
        elif hasattr(v, "shape"): print(f"  {pre}{k}: tensor{tuple(v.shape)} {v.dtype}")
        else: print(f"  {pre}{k}: {v!r}"[:200])
walk(o if isinstance(o, dict) else {"root": o})
