"""Phase 2 batch P7 — F-73 (length AUROC row) and F-74 (ceiling-null / coupling) recomputed
from the raw matrix file.  Read-only.

Run:  PYTHONIOENCODING=utf-8 python audit/scripts/claims/F-73_F-74.py [matrix.json]
Default matrix: run-crossgen/kd11/matrices/mean_L40_s42.json  (local git copy; fmap says LAB=GIT)
"""
import json, sys, math, random

MAT = sys.argv[1] if len(sys.argv) > 1 else "run-crossgen/kd11/matrices/mean_L40_s42.json"
d = json.load(open(MAT, encoding="utf-8"))
res = {(r["feature"], r["probe"]): r for r in d["results"]}
ds = d["datasets"]
n = len(ds)
print(f"matrix: {MAT}")
print(f"  pooling={d['pooling']} layer={d['layer']} seed={d['seed']} bank={d['bank']} "
      f"bank_name={d['bank_name']} k={d['k']} n_datasets={n}")


def offdiag(feature, probe="paper"):
    a = res[(feature, probe)]["auroc"]
    return [a[i][j] for i in range(n) for j in range(n) if i != j]


def mean(x):
    return sum(x) / len(x)


def sd(x, ddof=1):
    m = mean(x)
    return math.sqrt(sum((v - m) ** 2 for v in x) / (len(x) - ddof))


def corr(x, y):
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den


def ols(x, y):
    """slope of y on x"""
    mx, my = mean(x), mean(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x)


print("\n=== F-73 : off-diagonal AUROC, paper probe, 110 cells ===")
print(f"{'feature':<14}{'offdiag mean':>14}{'n':>6}   F-73 states")
stated = {"raw": 0.6623, "datapca": 0.6651, "random": 0.5849, "length": 0.5555}
f73 = {}
for feat in ["raw", "datapca", "random", "length"]:
    v = offdiag(feat)
    f73[feat] = mean(v)
    print(f"{feat:<14}{mean(v):>14.4f}{len(v):>6}   {stated[feat]}")

print("\n  extra (robustness, not quoted by F-73):")
for feat in ["raw", "datapca", "random", "length"]:
    for probe in ["lr_reg10", "mms"]:
        print(f"    {feat:<12} {probe:<9} {mean(offdiag(feat, probe)):.4f}")
print("    random@2   paper    %.4f" % mean(offdiag("random@2")))
print("    datapca@2  paper    %.4f" % mean(offdiag("datapca@2")))

print("\n=== F-74 : coupling of corr(raw, X - raw) ===")
raw = offdiag("raw")
rnd2 = offdiag("random@2")
print(f"  sd(raw)        = {sd(raw):.4f}   (F-74 states 0.2128)")
print(f"  sd(random@2)   = {sd(rnd2):.4f}   (F-74 states 0.1898)")
meas = corr(raw, [b - a for a, b in zip(raw, rnd2)])
print(f"  measured corr(raw, random@2 - raw) = {meas:.4f}   (F-74 states -0.7684)")

s_raw, s_x = sd(raw), sd(rnd2)
analytic = -s_raw / math.sqrt(s_raw ** 2 + s_x ** 2)
print(f"  analytic null  -s_raw/sqrt(s_raw^2+s_X^2) = {analytic:.4f}   (F-74 states -0.7462)")

# permuted pairing null: shuffle X against raw, marginals preserved
rng = random.Random(42)
B = 20000
perm = []
for _ in range(B):
    xs = rnd2[:]
    rng.shuffle(xs)
    perm.append(corr(raw, [b - a for a, b in zip(raw, xs)]))
perm.sort()
print(f"  permuted null  mean={mean(perm):.4f}  95% [{perm[int(.025*B)]:.4f}, {perm[int(.975*B)]:.4f}]"
      f"   (F-74: -0.7456 [-0.7986, -0.6892])")
p_perm = sum(1 for v in perm if v <= meas) / B
print(f"  P(permuted null <= measured) = {p_perm:.4f}   (F-74 states P=0.2135)")

# simulated-noise null: X ~ independent normal at the observed spread
rng2 = random.Random(42)
Bs = 2000
sim = []
mx = mean(rnd2)
for _ in range(Bs):
    xs = [rng2.gauss(mx, s_x) for _ in range(len(raw))]
    sim.append(corr(raw, [b - a for a, b in zip(raw, xs)]))
sim.sort()
print(f"  simulated null mean={mean(sim):.4f}  95% [{sim[int(.025*Bs)]:.4f}, {sim[int(.975*Bs)]:.4f}]"
      f"   (F-74: -0.7462 [-0.8106, -0.6683])")
p_sim = sum(1 for v in sim if v <= meas) / Bs
print(f"  P(simulated null <= measured) = {p_sim:.4f}")

print("\n  --- the clean (uncoupled) test ---")
pu2 = offdiag("pc:unified@2")
dp2 = offdiag("datapca@2")
dp1 = offdiag("datapca")
m1 = [a - b for a, b in zip(pu2, dp2)]
print(f"  pc:unified@2 - datapca@2 : slope vs raw = {ols(raw, m1):+.4f}  r = {corr(raw, m1):+.4f}  "
      f"mean margin = {mean(m1):+.4f}   (F-74: -0.100 / -0.067 / +0.0063)")
m2 = [a - b for a, b in zip(raw, dp2)]
print(f"  raw - datapca@2          : slope vs raw = {ols(raw, m2):+.4f}  r = {corr(raw, m2):+.4f}  "
      f"mean margin = {mean(m2):+.4f}   (F-74: +0.630 / +0.571 / +0.0290)")
m3 = [a - b for a, b in zip(raw, dp1)]
print(f"  [alt] raw - datapca(k=5) : slope vs raw = {ols(raw, m3):+.4f}  r = {corr(raw, m3):+.4f}  "
      f"mean margin = {mean(m3):+.4f}")
