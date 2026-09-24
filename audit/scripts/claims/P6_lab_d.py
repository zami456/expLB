"""P6 batch, lab-side part D: layer robustness for F-11, F-19 (drift gap) and F-12 (permutation
test), none of which the ledger sweeps.  READ-ONLY."""
import glob, itertools, os
import numpy as np
import torch

R = "C:/Users/user4/expLB"
LAYERS = [0, 4, 8, 12, 16, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def load_all(vdir):
    out, layers = {}, None
    for f in sorted(glob.glob(os.path.join(vdir, "*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        layers = list(d.get("layers", [])) or layers
        out[d.get("role", os.path.splitext(os.path.basename(f))[0])] = d["vector"].float().numpy()
    return out, layers


print("=== F-11 / F-19: assistant-axis projection by group, EVERY stored layer ===")
res = {}
for arm, bankp, vdir in (("s3", f"{R}/run-kd/axes_vectors_s3.pt", f"{R}/run-kd/vectors_s3"),
                         ("s23", f"{R}/run-kd/axes_vectors_s23.pt", f"{R}/run-kd/vectors_s23")):
    b = torch.load(bankp, map_location="cpu", weights_only=False)
    layers = list(b["layers"])
    vecs, _ = load_all(vdir)
    groups = b["role_groups"]
    print(f"\n  arm={arm}  (bank {os.path.basename(bankp)})")
    print(f"  {'layer':>5} {'kd':>7} {'sincere':>8} {'plain':>7} {'gap(kd-sinc)':>13} "
          f"{'dec':>7} {'hon':>7} {'gap(hon-dec)':>13} {'cos(PC1,asst)':>14}")
    res[arm] = {}
    for lay in LAYERS:
        li = layers.index(lay)
        asst = unit(b["axes"]["assistant"][li].numpy())
        pr = {}
        for g in ("knowledge_denial", "sincere_ignorance", "plain_assertion", "deception", "honesty"):
            rs = [r for r in groups[g]["roles"] if r in vecs]
            pr[g] = float(np.stack([vecs[r][li] for r in rs]).mean(0) @ asst)
        pc1 = unit(b["pca"]["all"]["components"][li][0].numpy())
        gap = pr["sincere_ignorance"] - pr["knowledge_denial"]
        gap2 = pr["honesty"] - pr["deception"]
        res[arm][lay] = gap
        print(f"  {lay:>5} {pr['knowledge_denial']:>7.2f} {pr['sincere_ignorance']:>8.2f} "
              f"{pr['plain_assertion']:>7.2f} {gap:>13.2f} {pr['deception']:>7.2f} "
              f"{pr['honesty']:>7.2f} {gap2:>13.2f} {float(pc1@asst):>+14.3f}")
print("\n  gap s3 vs s23 at every layer (F-19 measures only L40: 1.82 -> 1.67, -8%):")
print(f"  {'layer':>5} {'s3':>7} {'s23':>7} {'fall':>8}")
for lay in LAYERS:
    g3, g23 = res["s3"][lay], res["s23"][lay]
    print(f"  {lay:>5} {g3:>7.2f} {g23:>7.2f} {((g3-g23)/g3 if g3 else float('nan')):>8.1%}")

print("\n=== F-12: permutation test at EVERY stored layer ===")
b = torch.load(f"{R}/run-kd/axes_vectors_s3.pt", map_location="cpu", weights_only=False)
groups = b["role_groups"]
layers = list(b["layers"])
vecs, _ = load_all(f"{R}/run-kd/vectors_s3")
for tag, pg, ng in (("sycophancy", "sycophancy", "anti_sycophancy"),
                    ("deception", "deception", "honesty"),
                    ("knowledge_denial", "knowledge_denial", "sincere_ignorance")):
    pos = [r for r in groups[pg]["roles"] if r in vecs]
    neg = [r for r in groups[ng]["roles"] if r in vecs]
    members = pos + neg + [r for r in groups["assistant_anchor"]["roles"] if r in vecs]
    print(f"  {tag} ({len(pos)}v{len(neg)}):")
    line_c, line_p = [], []
    for lay in LAYERS:
        li = layers.index(lay)
        X = np.stack([vecs[r][li] for r in members]); Xc = X - X.mean(0)
        _, _, Vh = np.linalg.svd(Xc, full_matrices=False)
        pc1 = unit(Vh[0])
        real = unit(np.stack([vecs[r][li] for r in pos]).mean(0)
                    - np.stack([vecs[r][li] for r in neg]).mean(0))
        tc = abs(float(pc1 @ real))
        pool = pos + neg
        null = []
        for combo in itertools.combinations(range(len(pool)), len(pos)):
            a = [pool[i] for i in combo]; bb = [pool[i] for i in range(len(pool)) if i not in combo]
            d = unit(np.stack([vecs[r][li] for r in a]).mean(0)
                     - np.stack([vecs[r][li] for r in bb]).mean(0))
            null.append(abs(float(pc1 @ d)))
        null = np.array(null)
        line_c.append(f"L{lay}:{tc:.2f}")
        line_p.append(f"L{lay}:{(null >= tc).mean():.3f}")
    print("    cos: " + " ".join(line_c))
    print("    p  : " + " ".join(line_p))
