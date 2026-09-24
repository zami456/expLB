"""READ-ONLY on the lab PC. Level-2 recompute of F-37's geometry numbers straight from the
axes .pt files, plus the F-27 3B generation-timing evidence. Writes nothing."""
import glob, json, os, re, datetime

R = r"C:\Users\user4\expLB"
import torch

TARGETS = [
    ("3B", os.path.join(R, "run-scale", "3b", "axes_3b.pt"), 14),
    ("8B", os.path.join(R, "run-scale", "8b", "axes_8b.pt"), 16),
    ("70B all (shared)", os.path.join(R, "run-paper16-shared", "axes_vectors_all.pt"), 40),
    ("70B s3 (shared)", os.path.join(R, "run-paper16-shared", "axes_vectors_s3.pt"), 40),
    ("70B s3 (plain)", os.path.join(R, "run-paper16", "axes_vectors_s3.pt"), 40),
]

print("= F-37 Level-2: cos(PC1_all, assistant axis) recomputed from the .pt files =")
for lab, p, mid in TARGETS:
    if not os.path.exists(p):
        print(f"  {lab:20s} MISSING {p}")
        continue
    d = torch.load(p, map_location="cpu", weights_only=False)
    a = d["axes"]["assistant"].double()          # (L, H)
    pc = d["pca"]["all"]["components"].double()  # (L, k, H)
    vr = d["pca"]["all"]["variance_ratio"].double()
    sign = d["pca"]["all"].get("sign")
    L = a.shape[0]

    def sgn(s, l, j=0):
        if s is None:
            return 1.0
        t = s
        try:
            t = t[l]
        except Exception:
            return float(t)
        try:
            t = t[j]
        except Exception:
            pass
        return float(t)

    def cos(l):
        u = pc[l, 0] * sgn(sign, l)
        v = a[l]
        return float((u @ v) / (u.norm() * v.norm()))

    vals = [cos(l) for l in range(L)]
    import statistics
    print(f"  {lab:20s} {os.path.basename(p)}  L={L}  mid={mid}  sign_type="
          f"{type(sign).__name__}{getattr(sign,'shape','')}")
    print(f"      cos(PC1,assistant)@mid = {vals[mid]:+.4f}   "
          f"min={min(vals):+.4f}  median={statistics.median(vals):+.4f}")
    print(f"      PC1 var@mid = {100*float(vr[mid][0]):.1f}%   top5 = {100*float(vr[mid][:5].sum()):.1f}%")
    for basis in ("deception", "sycophancy", "unified"):
        if basis in d["pca"]:
            b = d["pca"][basis]
            u = b["components"][mid, 0].double() * sgn(b.get("sign"), mid)
            c = d["axes"][basis].double()[mid] if basis in d["axes"] else None
            cc = float((u @ c) / (u.norm() * c.norm())) if c is not None else float("nan")
            print(f"      basis {basis:11s} cos(PC1,contrast)={cc:+.4f}  "
                  f"PC1var={100*float(b['variance_ratio'][mid][0]):.1f}%")

print()
print("= F-27: 3B generation timing evidence =")
for rung in ("3b", "8b"):
    d = os.path.join(R, "run-scale", rung, "responses")
    fs = sorted(glob.glob(os.path.join(d, "*.jsonl")), key=os.path.getmtime)
    if not fs:
        print(f"  {rung}: none")
        continue
    t0 = datetime.datetime.fromtimestamp(os.path.getmtime(fs[0]))
    t1 = datetime.datetime.fromtimestamp(os.path.getmtime(fs[-1]))
    print(f"  {rung}: {len(fs)} files  first mtime {t0:%Y-%m-%d %H:%M:%S}  last {t1:%Y-%m-%d %H:%M:%S}  "
          f"span {(t1-t0).total_seconds()/60:.0f} min")
print("  logs matching scale_3b / scale_8b:")
for p in sorted(glob.glob(os.path.join(R, "logs", "*scale*3b*")) +
                glob.glob(os.path.join(R, "logs", "*scale*8b*"))):
    print(f"    {os.path.getsize(p):>10} {datetime.datetime.fromtimestamp(os.path.getmtime(p)):%Y-%m-%d %H:%M} "
          f"{os.path.basename(p)}")
