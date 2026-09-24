"""S-01/S-02: paper16 axis norms. Read-only.
Loads run-paper16/axes_vectors_s3.pt (laptop copy = the OLD bank, RF-15) and prints,
per layer, the L2 norm of each named axis and of the PCA components."""
import sys, torch, numpy as np, json

p = sys.argv[1] if len(sys.argv) > 1 else "run-paper16/axes_vectors_s3.pt"
d = torch.load(p, map_location="cpu", weights_only=False)
print("PATH", p)
print("TOP KEYS", list(d.keys()))
for k, v in d.items():
    if isinstance(v, dict):
        print(" ", k, "-> dict keys", list(v.keys())[:12], "..." if len(v) > 12 else "")
    elif torch.is_tensor(v):
        print(" ", k, "-> tensor", tuple(v.shape), v.dtype)
    else:
        print(" ", k, "->", type(v).__name__, str(v)[:200])
