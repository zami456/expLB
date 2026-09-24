"""P6 batch, lab-side part B: F-11, F-19 (assistant-axis drift), F-12 (permutation test),
F-18 (paper16 vs paper16-shared).  READ-ONLY: loads .pt banks and vector dirs, writes nothing.

Every number is recomputed from the per-role vectors, not read from a log.  The recomputation
follows src/assistant_axis_ndif/6_axes.py (group centroid . unit(axis)), perm.py and
cmp_banks.py, which are quoted in the F-entries as the producers.
"""
import glob, itertools, json, os, re, sys
import numpy as np
import torch

R = "C:/Users/user4/expLB"
L = 40


def head(t):
    print("\n" + "=" * 78 + "\n== " + t + "\n" + "=" * 78)


def st(p):
    import datetime
    return f"{p}  {os.path.getsize(p)} B  mtime={datetime.datetime.fromtimestamp(os.path.getmtime(p)):%Y-%m-%d %H:%M:%S}"


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def load_vecs(vdir, layer=L):
    """role -> vector at `layer` (float32 numpy), exactly as 6_axes.py loads them."""
    out, layers = {}, None
    for f in sorted(glob.glob(os.path.join(vdir, "*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        layers = list(d.get("layers", [])) or layers
        out[d.get("role", os.path.splitext(os.path.basename(f))[0])] = \
            d["vector"][layers.index(layer)].float().numpy()
    return out, layers


def grep(path, pat, limit=60):
    raw = open(path, "rb").read()
    txt = raw.decode("utf-16", "replace") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")
    hits = [(i, l.rstrip()) for i, l in enumerate(txt.splitlines(), 1) if re.search(pat, l)]
    for i, l in hits[:limit]:
        print(f"    {os.path.basename(path)}:{i} | {l}")
    return hits


# =================================================================== F-11 / F-19
def drift(bankpath, vdir, label):
    print("\n--- " + label + " ---")
    print("  " + st(bankpath))
    print("  " + st(vdir))
    b = torch.load(bankpath, map_location="cpu", weights_only=False)
    layers = list(b["layers"]); li = layers.index(L)
    vecs, _ = load_vecs(vdir)
    groups = b["role_groups"]
    print(f"  roles in bank: {len(b['roles'])}   vector files: {len(vecs)}   "
          f"vectors_dir recorded in bank: {b.get('vectors_dir')}")
    asst = unit(b["axes"]["assistant"][li].numpy())
    print(f"  {'group':20} {'n':>3} {'proj on assistant axis':>24}")
    proj = {}
    for g, spec in groups.items():
        rs = [r for r in spec["roles"] if r in vecs]
        if not rs:
            continue
        centroid = np.stack([vecs[r] for r in rs]).mean(0)
        proj[g] = float(centroid @ asst)
        print(f"  {g:20} {len(rs):>3} {proj[g]:>24.2f}")
    if "knowledge_denial" in proj and "sincere_ignorance" in proj:
        print(f"  gap knowledge_denial vs sincere_ignorance = "
              f"{proj['sincere_ignorance'] - proj['knowledge_denial']:.2f}")
    if "deception" in proj and "honesty" in proj:
        print(f"  gap honesty vs deception (paper16 groups inside this bank) = "
              f"{proj['honesty'] - proj['deception']:.2f}")
    # cos(PC1_all, assistant) and the default anchor's relative position
    comps = b["pca"]["all"]["components"] if "all" in b.get("pca", {}) else None
    if comps is not None:
        pc1 = unit(comps[li][0].numpy())
        print(f"  cos(PC1[all-roles], assistant) = {float(pc1 @ asst):+.3f}")
        names = sorted(vecs)
        pr = {r: float(vecs[r] @ pc1) for r in names}
        dl = [r for r in groups["assistant_anchor"]["roles"] if "default" in r and r in vecs]
        dv = sum(pr[r] for r in dl) / len(dl)
        lo, hi = min(pr.values()), max(pr.values())
        print(f"  default anchor ({dl}) relative position on PC1 = {(dv-lo)/(hi-lo):.3f}")
    print(f"  {'basis':18} {'cos(PC1, own contrast)':>24} {'PC1 var':>9}")
    for tag in ("deception", "sycophancy", "unified", "knowledge_denial", "all"):
        if tag in b.get("pca", {}) and tag in b["axes"]:
            p1 = unit(b["pca"][tag]["components"][li][0].numpy())
            con = unit(b["axes"][tag][li].numpy())
            var = float(b["pca"][tag]["variance_ratio"][li][0])
            print(f"  {tag:18} {float(p1@con):>+24.3f} {var:>9.1%}")
    return proj


head("F-11 / F-19  assistant-axis drift of the kd contrast, s3 vs s23")
p3 = drift(f"{R}/run-kd/axes_vectors_s3.pt", f"{R}/run-kd/vectors_s3", "s3 bank")
p23 = drift(f"{R}/run-kd/axes_vectors_s23.pt", f"{R}/run-kd/vectors_s23", "s23 bank")
print("\n  stated F-11 (s3): kd -0.22, sincere +1.60, plain +0.90, deception 1.51, honesty 2.20,")
print("                    gap 1.82 vs 0.69; cos(PC1,assistant)=0.583; default anchor 0.882")
print("  stated F-19 (s23): kd -0.01, sincere +1.66, plain +1.37; gap 1.67 (-8%);")
print("                    cos(PC1,contrast) dec 0.961->0.965, syc 0.993->0.993, unified 0.887->0.909;")
print("                    cos(PC1,assistant) 0.583->0.620")
if "knowledge_denial" in p3 and "knowledge_denial" in p23:
    g3 = p3["sincere_ignorance"] - p3["knowledge_denial"]
    g23 = p23["sincere_ignorance"] - p23["knowledge_denial"]
    print(f"  recomputed gap s3={g3:.4f}  s23={g23:.4f}  fall={(g3-g23)/g3:.1%}  (stated 'about 8%')")

print("\n  --- what the logs print (Level 1) ---")
grep(f"{R}/logs/kd_axes.log", r"knowledge_denial|sincere_ignorance|plain_assertion|cos\(PC1|relative position|deception |honesty ")
print()
grep(f"{R}/logs/kd_axes_s23.log", r"knowledge_denial|sincere_ignorance|plain_assertion|cos\(PC1|relative position|deception |honesty ")

# =================================================================== F-12
head("F-12  permutation test on PC1 alignment (reimplementation of perm.py)")
bank = torch.load(f"{R}/run-kd/axes_vectors_s3.pt", map_location="cpu", weights_only=False)
groups = bank["role_groups"]
vecdir = str(bank["vectors_dir"])
print("  bank:", st(f"{R}/run-kd/axes_vectors_s3.pt"))
print("  vectors_dir recorded in the bank:", vecdir)
vecs, _ = load_vecs(f"{R}/run-kd/vectors_s3")
for tag, pos_g, neg_g in (("sycophancy", "sycophancy", "anti_sycophancy"),
                          ("deception", "deception", "honesty"),
                          ("knowledge_denial", "knowledge_denial", "sincere_ignorance")):
    pos = [r for r in groups[pos_g]["roles"] if r in vecs]
    neg = [r for r in groups[neg_g]["roles"] if r in vecs]
    members = pos + neg + [r for r in groups["assistant_anchor"]["roles"] if r in vecs]
    X = np.stack([vecs[r] for r in members]); Xc = X - X.mean(0)
    _, _, Vh = np.linalg.svd(Xc, full_matrices=False)
    pc1 = unit(Vh[0])
    real = unit(np.stack([vecs[r] for r in pos]).mean(0) - np.stack([vecs[r] for r in neg]).mean(0))
    true_cos = abs(float(pc1 @ real))
    pool = pos + neg
    null = []
    for combo in itertools.combinations(range(len(pool)), len(pos)):
        a = [pool[i] for i in combo]; b = [pool[i] for i in range(len(pool)) if i not in combo]
        d = unit(np.stack([vecs[r] for r in a]).mean(0) - np.stack([vecs[r] for r in b]).mean(0))
        null.append(abs(float(pc1 @ d)))
    null = np.array(null)
    print(f"  {tag:18} |pos|={len(pos)} |neg|={len(neg)} n_members={len(members):>2}  "
          f"cos(PC1,real)={true_cos:.3f}  splits={len(null):>5}  median={np.median(null):.3f}  "
          f"max={null.max():.3f}  p={(null >= true_cos).mean():.4f}")
print("  stated: sycophancy 5v5 / 252 / 0.993 / 0.534 / p=0.0079;")
print("          deception 8v7 / 6,435 / 0.961 / 0.528 / p=0.0092;")
print("          knowledge_denial 4v3 / 35 / 0.894 / 0.397 / p=0.143")

# =================================================================== F-18
head("F-18  paper16 vs paper16-shared (reimplementation of cmp_banks.py)")
banks = {"paper16 (own question sets), OLD axes": f"{R}/run-paper16/axes_vectors_s3.pt",
         "paper16 (own question sets), fp32 axes": f"{R}/run-paper16/axes_vectors_s3_fp32.pt",
         "paper16-shared (one 20-q set)": f"{R}/run-paper16-shared/axes_vectors_s3.pt"}
loaded = {}
for name, p in banks.items():
    if not os.path.exists(p):
        print(f"MISSING {p}"); continue
    b = torch.load(p, map_location="cpu", weights_only=False)
    layers = list(b["layers"]); li = layers.index(L)
    loaded[name] = (b, li)
    vd = b["vectors_dir"]
    resp = os.path.join(os.path.dirname(str(vd)), "responses")
    nroll = 0
    nfiles = 0
    for f in sorted(glob.glob(os.path.join(resp, "*.jsonl"))):
        nfiles += 1
        nroll += sum(1 for l in open(f, encoding="utf-8") if l.strip())
    print(f"\n=== {name} ===")
    print("  " + st(p))
    print(f"  roles {len(b['roles'])}   rollout files {nfiles}   rollouts {nroll}   "
          f"vectors_dir {vd}   responses {resp}")
    for tag in ("deception", "sycophancy", "unified"):
        if tag not in b.get("pca", {}):
            continue
        pc1 = unit(b["pca"][tag]["components"][li][0].numpy())
        con = unit(b["axes"][tag][li].numpy())
        asst = unit(b["axes"]["assistant"][li].numpy())
        var = float(b["pca"][tag]["variance_ratio"][li][0])
        sign = b["pca"][tag].get("sign")
        print(f"  {tag:11} cos(PC1,own contrast) {float(pc1@con):+.3f}   "
              f"cos(PC1,assistant) {float(pc1@asst):+.3f}   PC1 var {var:.1%}   "
              f"pinned sign at L{L} = {float(sign[li][0]) if sign is not None else 'n/a'}")

print("\n=== cross-bank cosines of the AXES at L40 ===")
for a_name, c_name in (("paper16 (own question sets), OLD axes", "paper16-shared (one 20-q set)"),
                       ("paper16 (own question sets), fp32 axes", "paper16-shared (one 20-q set)"),
                       ("paper16 (own question sets), OLD axes", "paper16 (own question sets), fp32 axes")):
    if a_name not in loaded or c_name not in loaded:
        continue
    (a, ia), (c, ic) = loaded[a_name], loaded[c_name]
    print(f"  {a_name}  vs  {c_name}")
    for tag in ("deception", "sycophancy", "unified", "assistant"):
        if tag in a["axes"] and tag in c["axes"]:
            x = unit(a["axes"][tag][ia].numpy()); y = unit(c["axes"][tag][ic].numpy())
            print(f"    cos({tag:11}) = {float(x@y):+.3f}")

print("\n=== sign-sensitivity check: cos(PC1, own contrast) at EVERY stored layer, both banks ===")
for tag in ("deception", "sycophancy", "unified"):
    print(f"  {tag}:")
    for name in banks:
        if name not in loaded:
            continue
        b, _ = loaded[name]
        if tag not in b.get("pca", {}):
            continue
        ls = list(b["layers"])
        vals = []
        for j, lay in enumerate(ls):
            p1 = unit(b["pca"][tag]["components"][j][0].numpy())
            con = unit(b["axes"][tag][j].numpy())
            vals.append((lay, float(p1 @ con)))
        print(f"    {name[:38]:40} " + " ".join(f"L{l}:{v:+.2f}" for l, v in vals))
print("\n  stated: deception +0.966/+0.952, sycophancy +0.979/+0.993, unified +0.389/+0.896;")
print("          PC1 var 46.7%/36.8%; rollouts 1,400/2,600;")
print("          cross-bank cos deception +0.949, sycophancy +0.966, unified +0.984, assistant +0.631")
