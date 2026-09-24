"""READ-ONLY on the lab PC.
(a) F-37: cos(PC1, assistant axis), PC1 variance share, contrast alignment, anchor position —
    first from the axes logs, then recomputed from the axes_*.pt files themselves.
(b) F-27: can the 21-role subset numbers be reconstructed from the full 25-role score files?
Writes nothing.
"""
import collections, glob, itertools, json, os, re, sys

R = r"C:\Users\user4\expLB"

print("=" * 78)
print("(a) F-37 source logs")
for name in ("scale_3b_axes.log", "scale_8b_axes.log", "paper16_shared_axes.log"):
    p = os.path.join(R, "logs", name)
    if not os.path.exists(p):
        print(f"  MISSING {p}")
        continue
    raw = open(p, "rb").read()
    txt = raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8", "replace")
    print(f"  --- {name} ({len(txt.splitlines())} lines, {len(raw)} B)")
    for l in txt.splitlines():
        if l.strip():
            print(f"      {l.strip()[:190]}")

print()
print("=" * 78)
print("(a2) F-37 recomputed from the axes .pt files")
try:
    import torch
except Exception as e:
    print("  torch unavailable:", e)
    torch = None

CAND = {
    "3B": [os.path.join(R, "run-scale", "3b", "axes_vectors_s23.pt"),
           os.path.join(R, "run-scale", "3b", "axes_vectors_s3.pt"),
           os.path.join(R, "run-scale", "3b", "axes_vectors_all.pt")],
    "8B": [os.path.join(R, "run-scale", "8b", "axes_vectors_s23.pt"),
           os.path.join(R, "run-scale", "8b", "axes_vectors_s3.pt"),
           os.path.join(R, "run-scale", "8b", "axes_vectors_all.pt")],
    "70B": [os.path.join(R, "run-paper16", "axes_vectors_s3_shared.pt"),
            os.path.join(R, "run-paper16", "axes_vectors_shared.pt"),
            os.path.join(R, "run-paper16", "axes_vectors_s3_fp32.pt"),
            os.path.join(R, "run-paper16", "axes_vectors_s3.pt")],
}
MID = {"3B": 14, "8B": 16, "70B": 40}

print("  available axes files:")
for rung in ("3B", "8B", "70B"):
    d = os.path.dirname(CAND[rung][0])
    for f in sorted(glob.glob(os.path.join(d, "axes_*.pt"))):
        print(f"    {rung:4s} {os.path.getsize(f):>10} {os.path.basename(f)}")


def cosdeg(a, b):
    import torch
    a = a.double().flatten(); b = b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm()))


if torch is not None:
    for rung in ("3B", "8B", "70B"):
        path = next((p for p in CAND[rung] if os.path.exists(p)), None)
        if path is None:
            print(f"  {rung}: no axes file found")
            continue
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  {rung}: load failed {e}")
            continue
        print(f"\n  --- {rung}  {path}")
        print(f"      top keys: {list(d)[:12]}")
        # structure discovery, then the numbers if we recognise it
        ax = d.get("axes", d)
        if isinstance(ax, dict):
            print(f"      axes keys: {list(ax)[:12]}")
            for k in list(ax)[:1]:
                v = ax[k]
                print(f"      axes[{k!r}] type={type(v).__name__} "
                      f"{'keys=' + str(list(v)[:8]) if isinstance(v, dict) else getattr(v, 'shape', '')}")
        pca = d.get("pca")
        if isinstance(pca, dict):
            print(f"      pca keys: {list(pca)[:12]}")
            for k in list(pca)[:1]:
                v = pca[k]
                print(f"      pca[{k!r}] type={type(v).__name__} "
                      f"{'keys=' + str(list(v)[:10]) if isinstance(v, dict) else getattr(v, 'shape', '')}")
        for k in list(d)[:12]:
            v = d[k]
            if not isinstance(v, (dict, list)):
                print(f"      {k} = {v!r}"[:180])

print()
print("=" * 78)
print("(b) F-27: 21-of-26-role subset reconstruction (8B)")
sc = os.path.join(R, "run-scale", "8b", "scores")
per_role = {}
for f in sorted(glob.glob(os.path.join(sc, "*.json"))):
    d = json.load(open(f, encoding="utf-8"))
    per_role[os.path.basename(f)[:-5]] = collections.Counter(d.values())
groups = json.load(open(os.path.join(R, "src", "assistant_axis_ndif", "pc_role_groups.json"),
                        encoding="utf-8"))["groups"]
TARGET = {"deception": (8, 765, 800), "honesty": (5, 485, 500),
          "sycophancy": (4, 397, 400), "anti_sycophancy": (4, 393, 400)}
for g, (nr, kept_t, n_t) in TARGET.items():
    roles = [r for r in groups[g].get("roles", []) if r in per_role]
    sols = []
    for comb in itertools.combinations(roles, nr):
        n = sum(sum(per_role[r].values()) for r in comb)
        kept = sum(per_role[r][2] + per_role[r][3] for r in comb)
        if n == n_t and kept == kept_t:
            sols.append(comb)
    print(f"  {g:18s} want {nr} of {len(roles)} roles -> kept {kept_t}/{n_t}: "
          f"{len(sols)} subset(s) match" + (f"  e.g. {sols[0]}" if sols else "  NONE"))
print("  (70B reference quoted by F-27 and by scale/scripts/score_summary.py: 16/1500 = 1.07%)")
p16 = os.path.join(R, "run-paper16", "scores")
n16 = sum(len(json.load(open(f, encoding='utf-8'))) for f in glob.glob(os.path.join(p16, "*.json")))
d16 = sum(sum(1 for v in json.load(open(f, encoding='utf-8')).values() if v in (0, 1))
          for f in glob.glob(os.path.join(p16, "*.json")))
print(f"  measured now: {d16}/{n16} = {100*d16/n16:.2f}%  ({len(glob.glob(os.path.join(p16,'*.json')))} files)")
