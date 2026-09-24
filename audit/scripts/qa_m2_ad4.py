"""Which judge filter is baked into each scale rung's axes bank, and into the 70B bank it is
compared against? Read-only. Settles AD-4 and the run_scale_chain.ps1 overwrite."""
import glob, os, torch

def describe(p):
    if not os.path.exists(p):
        return f"{p}\n    ABSENT"
    b = torch.load(p, map_location="cpu", weights_only=False)
    keys = sorted(b.keys()) if isinstance(b, dict) else type(b).__name__
    out = [f"{p}  ({os.path.getsize(p)/1e6:.1f} MB)", f"    keys: {keys}"]
    for k in ("meta", "counts", "n_pos", "n_neg", "keep_scores", "source", "vectors_dir", "roles_file"):
        if isinstance(b, dict) and k in b:
            out.append(f"    {k}: {str(b[k])[:220]}")
    return "\n".join(out)

print("=== the scale rungs' axes banks ===")
for p in sorted(glob.glob("C:/Users/user4/expLB/run-scale/*/axes_*.pt")):
    print(describe(p)); print()
print("=== the 70B banks they are compared against ===")
for p in ("C:/Users/user4/expLB/run-paper16/axes_vectors_s3_fp32.pt",
          "C:/Users/user4/expLB/run-paper16/axes_vectors_all_fp32.pt"):
    print(describe(p)); print()
print("=== do both vector dirs exist per rung? (the overwrite leaves both dirs but one bank) ===")
for d in sorted(glob.glob("C:/Users/user4/expLB/run-scale/*/vectors_*")):
    n = len(glob.glob(d + "/*.pt"))
    print(f"    {d}  {n} vector files")
