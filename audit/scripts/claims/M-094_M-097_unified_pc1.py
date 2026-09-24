"""W6F2-memory: claims M-094 (unified PC1 projections by persona family)
and M-096 / M-097 (ceiling/regression effect of 2-d projections).

M-094 (memory/unified-pc1-is-question-family.md:20): at layer 40 on
  run-paper16/axes_vectors_s3.pt + vectors_s3/, the unified PC1 projections are
  deception -2.75 / honesty -3.60 vs sycophancy +1.49 / anti-syc +0.07 (a question-family split).
M-096 (line 29): a random 2-d projection gains +0.056 where raw off-diagonal AUROC < 0.6
  and loses -0.44 where raw >= 0.85; corr(raw, delta) = -0.76.
M-097 (line 30): unified@2 gains +0.18 in the same low-raw bin, dataset-PCA@2 +0.12.

Raw files (git working tree):
  run-paper16/axes_vectors_s3.pt, run-paper16/vectors_s3/*.pt   (OLD paper16 bank, RF-15)
  run-crossgen/kd11/matrices/mean_L40_s42.json                  (11x11, L40, mean, seed 42)
Read-only.
"""
import torch, json, os, glob, math, hashlib

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")

print("=" * 78)
print("M-094 -- unified PC1 projections by persona family, layer 40")
print("=" * 78)
BP = os.path.join(ROOT, "run-paper16", "axes_vectors_s3.pt")
b = torch.load(BP, map_location="cpu", weights_only=False)
print("bank: run-paper16/axes_vectors_s3.pt md5=%s" % hashlib.md5(open(BP, 'rb').read()).hexdigest())
V = {os.path.basename(f)[:-3]: torch.load(f, map_location='cpu', weights_only=False)['vector'].float()
     for f in glob.glob(os.path.join(ROOT, "run-paper16", "vectors_s3", "*.pt"))}
print("role vectors: %d from run-paper16/vectors_s3" % len(V))
u = b['pca']['unified']
i40 = b['layers'].index(40)
roles = u['roles']
print("unified PCA roles (n=%d): %s" % (len(roles), roles))
print("variance_ratio L40 (PC1..PC3): %s"
      % [round(float(x), 4) for x in u['variance_ratio'][i40][:3]])
sign = u['sign']
print("stored sign object: %r" % (sign if not hasattr(sign, 'shape') else (tuple(sign.shape), sign[i40][:3].tolist()),))

X = torch.stack([V[r][i40] for r in roles])
groups = {g: b['role_groups'][g]['roles'] for g in b['role_groups']}
NOTE = {'deception': -2.75, 'honesty': -3.60, 'sycophancy': +1.49, 'anti_sycophancy': +0.07}
# CONVENTION: the note's numbers reproduce under the UNCENTERED dot product with the
# sign-pinned PC. Re-centering on the 26-role mean shifts ALL FOUR group means by the same
# +1.65, so only the SPLIT between families is convention-free. Both are printed.
for centering in ("uncentered (reproduces the note)", "centered on the 26-role mean"):
  Xu = X if centering.startswith("uncentered") else X - X.mean(0, keepdim=True)
  print("\n### convention: %s" % centering)
  for pcidx in (0, 1, 2):
    pc = u['components'][i40][pcidx]
    s = float(sign[i40][pcidx]) if hasattr(sign, 'shape') else 1.0
    proj = (Xu @ pc) * s
    m = {r: float(proj[i]) for i, r in enumerate(roles)}
    print("\n  PC%d (var %.3f, stored sign %+g):" % (pcidx + 1, float(u['variance_ratio'][i40][pcidx]), s))
    for g, rs in groups.items():
        vals = [m[r] for r in rs if r in m]
        note = "   note %+0.2f" % NOTE[g] if (pcidx == 0 and g in NOTE) else ""
        print("     %-16s n=%d mean=%+.4f%s" % (g, len(vals), sum(vals) / len(vals), note))
    print("     %-16s     default=%+.4f" % ("", m['default']))
    dec_hon = [m[r] for g in ('deception', 'honesty') for r in groups[g]]
    syc = [m[r] for g in ('sycophancy', 'anti_sycophancy') for r in groups[g]]
    print("     deception-QUESTION family mean=%+.4f (n=%d) vs sycophancy-QUESTION family mean=%+.4f (n=%d)"
          % (sum(dec_hon) / len(dec_hon), len(dec_hon), sum(syc) / len(syc), len(syc)))

# ------------------------------------------------------------- M-096 / M-097
print()
print("=" * 78)
print("M-096 / M-097 -- ceiling/regression effect of 2-d projections")
print("=" * 78)
MP = os.path.join(ROOT, "run-crossgen", "kd11", "matrices", "mean_L40_s42.json")
d = json.load(open(MP))
DS = d['datasets']
res = {(r['feature'], r['probe']): r for r in d['results']}
PROBE = 'paper'
raw = res[('raw', PROBE)]['auroc']
pairs = [(i, j) for i in range(len(DS)) for j in range(len(DS)) if i != j]
print("matrix: run-crossgen/kd11/matrices/mean_L40_s42.json probe=%s, %d off-diagonal cells"
      % (PROBE, len(pairs)))


def corr(a, c):
    n = len(a); ma = sum(a) / n; mc = sum(c) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a)); sc = math.sqrt(sum((x - mc) ** 2 for x in c))
    return sum((x - ma) * (y - mc) for x, y in zip(a, c)) / (sa * sc)


print("\n%-14s %8s %8s %8s %8s %8s" % ("feature", "n<0.6", "d<0.6", "n>=0.85", "d>=0.85", "corr"))
for feat in ('random@2', 'pc:unified@2', 'datapca@2', 'random', 'pc:unified', 'datapca',
             'axes', 'pc:deception'):
    if (feat, PROBE) not in res:
        continue
    m = res[(feat, PROBE)]['auroc']
    rv = [raw[i][j] for i, j in pairs]
    dv = [m[i][j] - raw[i][j] for i, j in pairs]
    lo = [d_ for r, d_ in zip(rv, dv) if r < 0.60]
    hi = [d_ for r, d_ in zip(rv, dv) if r >= 0.85]
    print("%-14s %8d %+8.4f %8d %+8.4f %+8.4f"
          % (feat, len(lo), sum(lo) / len(lo), len(hi), sum(hi) / len(hi), corr(rv, dv)))
print("\nnote: random@2 +0.056 / -0.44 / corr -0.76 ; unified@2 +0.18 ; datapca@2 +0.12")

# the same bins restricted to block9 (= the 10 datasets WITHOUT sycophancy_mt; in the kd11 era
# the label "block9" means 10 datasets -- see the audit trap list)
keep = {i for i, x in enumerate(DS) if x in d['block9']}
pairs9 = [(i, j) for i in keep for j in keep if i != j]
print("\nblock9 restriction (%d datasets, %d off-diagonal cells), probe=%s:" % (len(keep), len(pairs9), PROBE))
for feat in ('random@2', 'pc:unified@2', 'datapca@2'):
    m = res[(feat, PROBE)]['auroc']
    rv = [raw[i][j] for i, j in pairs9]
    dv = [m[i][j] - raw[i][j] for i, j in pairs9]
    lo = [x for r, x in zip(rv, dv) if r < 0.60]
    hi = [x for r, x in zip(rv, dv) if r >= 0.85]
    print("%-14s %8d %+8.4f %8d %+8.4f %+8.4f"
          % (feat, len(lo), sum(lo) / len(lo), len(hi), sum(hi) / len(hi), corr(rv, dv)))
