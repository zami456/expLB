"""F-17 - 'Matched controls erase the persona-coordinate advantage'.

Claim (kd/docs/FINDINGS.md:272-279):
  Against dataset PCA at the same k, the unified persona axis reads -0.003.  A random 2-d
  projection gains +0.056 where raw < 0.6 and loses 0.44 where raw >= 0.85
  (corr(raw, delta) = -0.76), reproducing the paper's qualitative claim with meaningless
  directions.
  source: run-crossgen/significance.json; memory 'matched-control-kills-the-gain'

Four numbers.  This script locates each one and says WHICH file holds it.
  (a) -0.003   : significance.json, comparison 'pc:unified@2 - datapca@2'
  (b) +0.056   : per-cell matrix, mean(random@2 - raw) over cells with raw < 0.6
  (c) -0.44    : same, over cells with raw >= 0.85
  (d) -0.76    : corr(raw, random@2 - raw) over the cells
It sweeps probe x cell-set (all / off-diagonal) so the reading is not assumed, and it repeats
(b)-(d) at every stored layer to show how layer-specific the numbers are.
"""
import glob, json, os, sys, math

ROOT = sys.argv[1] if len(sys.argv) > 1 else "run-crossgen"

# ---------------------------------------------------------------- (a) significance.json
sig = os.path.join(ROOT, "significance.json")
d = json.load(open(sig, encoding="utf-8"))
T = d["tests"]
print("=== (a) significance.json ===")
print("file :", os.path.abspath(sig), "|", os.path.getsize(sig), "B")
print("n tests:", len(T), "| blocks", sorted({t['block'] for t in T}),
      "| layers", sorted({t['layer'] for t in T}))
for comp in ("pc:unified@2 - datapca@2", "pc:unified - datapca"):
    for t in T:
        if (t["comparison"] == comp and t["layer"] == 40 and t["pooling"] == "mean"
                and t["probe"] == "paper"):
            print(f"  block {t['block']:>2} {comp:26} diff={t['diff']:+.4f} "
                  f"dataset_p={t['dataset_p']:.4f} q_bh={t['dataset_q_bh']:.4f} pairs={t['pairs']}")

# ---------------------------------------------------------------- (b)-(d) the matrices
def load(path):
    m = json.load(open(path, encoding="utf-8"))
    by = {(r["feature"], r["probe"]): r["auroc"] for r in m["results"]}
    return m, by


def gains(raw, ran, offdiag_only):
    n = len(raw)
    xs, ys = [], []
    for i in range(n):
        for j in range(n):
            if offdiag_only and i == j:
                continue
            xs.append(raw[i][j]); ys.append(ran[i][j] - raw[i][j])
    lo = [y for x, y in zip(xs, ys) if x < 0.6]
    hi = [y for x, y in zip(xs, ys) if x >= 0.85]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs)); sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    r = cov / (sx * sy) if sx and sy else float("nan")
    return (sum(lo) / len(lo) if lo else float("nan"), len(lo),
            sum(hi) / len(hi) if hi else float("nan"), len(hi), r, len(xs))


main = os.path.join(ROOT, "matrices", "mean_L40_s42.json")
m, by = load(main)
print()
print("=== (b)-(d) per-cell matrix ===")
print("file :", os.path.abspath(main), "|", os.path.getsize(main), "B")
print("bank :", m.get("bank"), "| k", m.get("k"), "| pooling", m.get("pooling"),
      "| layer", m.get("layer"), "| seed", m.get("seed"), "| datasets", len(m["datasets"]))
print(f"{'probe':10} {'cells':>12} {'mean(d) raw<0.6':>16} {'n':>4} {'mean(d) raw>=0.85':>18} {'n':>4} {'corr(raw,d)':>12}")
for probe in ("paper", "lr_reg10", "mms"):
    for off in (False, True):
        g = gains(by[("raw", probe)], by[("random@2", probe)], off)
        print(f"{probe:10} {'off-diag' if off else 'all':>12} {g[0]:>+16.4f} {g[1]:>4} "
              f"{g[2]:>+18.4f} {g[3]:>4} {g[4]:>12.4f}")

print()
print("  same, with 'random' (k=5) instead of random@2, probe=paper:")
for off in (False, True):
    g = gains(by[("raw", "paper")], by[("random", "paper")], off)
    print(f"    {'off-diag' if off else 'all':>12} lo={g[0]:+.4f} (n={g[1]}) hi={g[2]:+.4f} (n={g[3]}) corr={g[4]:.4f}")

print()
print("=== layer sweep of (b)-(d), probe=paper, all cells, mean pooling ===")
print(f"{'layer':>6} {'raw<0.6':>10} {'n':>4} {'raw>=0.85':>11} {'n':>4} {'corr':>8}")
rows = []
for p in sorted(glob.glob(os.path.join(ROOT, "matrices", "mean_L*_s42.json")),
                key=lambda p: int(os.path.basename(p).split("_L")[1].split("_")[0])):
    mm, bb = load(p)
    if ("random@2", "paper") not in bb:
        continue
    g = gains(bb[("raw", "paper")], bb[("random@2", "paper")], False)
    rows.append((mm["layer"], g))
    print(f"{mm['layer']:>6} {g[0]:>+10.4f} {g[1]:>4} {g[2]:>+11.4f} {g[3]:>4} {g[4]:>8.4f}")
if rows:
    print(f"  layers covered: {[r[0] for r in rows]}")
