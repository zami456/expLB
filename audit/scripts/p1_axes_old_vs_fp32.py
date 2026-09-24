"""READ-ONLY (lab PC): how different are the OLD paper16 axes (run-paper16/axes_vectors_s3.pt, 09-15) and the
fp32 re-extraction (axes_vectors_s3_fp32.pt, 09-16)? Prints the file structure, then per axis the cosine between
old and new at a set of layers, and the relative norm difference."""
import torch
B = "C:/Users/user4/expLB/run-paper16/"
old = torch.load(B + "axes_vectors_s3.pt", map_location="cpu", weights_only=False)
new = torch.load(B + "axes_vectors_s3_fp32.pt", map_location="cpu", weights_only=False)
def desc(o, pre=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if hasattr(v, "shape"): print(f"  {pre}{k}: {tuple(v.shape)} {v.dtype}")
            elif isinstance(v, dict): desc(v, pre + k + ".")
            else: print(f"  {pre}{k}: {type(v).__name__} {str(v)[:80]}")
print("OLD keys:"); desc(old)
print("NEW keys:"); desc(new)
def tensors(o, pre=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            if hasattr(v, "shape"): out[pre + k] = v
            elif isinstance(v, dict): out.update(tensors(v, pre + k + "."))
    return out
T0, T1 = tensors(old), tensors(new)
for k in sorted(set(T0) & set(T1)):
    a, b = T0[k].float(), T1[k].float()
    if a.shape != b.shape: print(k, "shape differs", a.shape, b.shape); continue
    if a.dim() == 2 and a.shape[0] == 80:  # [layer, d]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
        rel = (a - b).norm(dim=1) / a.norm(dim=1).clamp_min(1e-12)
        ls = [0, 10, 20, 22, 30, 40, 44, 60, 79]
        print(f"{k}: cos@L{ls} = {[round(float(cos[l]), 4) for l in ls]} | min cos {float(cos.min()):.4f} (L{int(cos.argmin())}) | max rel.diff {float(rel.max()):.4f}")
    elif a.dim() == 3 and a.shape[1] == 80:  # e.g. [k, layer, d] PCs
        cos = torch.nn.functional.cosine_similarity(a, b, dim=2).abs()
        print(f"{k}: |cos| per component at L20/L40 = {[round(float(x), 4) for x in cos[:, 20]]} / {[round(float(x), 4) for x in cos[:, 40]]}")
    else:
        print(f"{k}: shape {tuple(a.shape)} max|diff| {float((a-b).abs().max()):.4g}")
