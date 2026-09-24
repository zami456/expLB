"""M-018 (matched-control-kills-the-gain.md:19) and M-095 (unified-pc1-is-question-family.md:28)

Both notes state the same regression-to-chance measurement:
  "a RANDOM 2-d projection gains +0.056 where raw < 0.6 and loses 0.44 where raw >= 0.85
   (corr(raw, delta) = -0.76, 23 of 90 raw cells are below 0.5)"
M-095 adds: "unified@2 gains +0.18 there, dataset-PCA@2 +0.12" (i.e. in the raw<0.6 bucket).

Context given by the notes: layer 40, mean pooling, paper probe (C=1), 10 datasets / 90
off-diagonal pairs, run-crossgen (80/20 retrain store).

Recomputes per-cell deltas over the 90 off-diagonal cells of the 10x10 matrix.
Read-only.
"""
import json, sys, os, math

path = sys.argv[1] if len(sys.argv) > 1 else 'run-crossgen/matrices/mean_L40_s42.json'
probes = sys.argv[2].split(',') if len(sys.argv) > 2 else ['paper', 'mms', 'lr_reg10']

d = json.load(open(path))
ds = d['datasets']
n = len(ds)
print(f'### {path}')
print(f'    pooling={d["pooling"]} layer={d["layer"]} seed={d["seed"]} bank={d["bank"]} '
      f'k={d["k"]} n_datasets={n} design={d["design"][:60]}...')


def get(feat, probe):
    for r in d['results']:
        if r['feature'] == feat and r['probe'] == probe:
            return r['auroc']
    return None


def corr(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sx * sy)


for probe in probes:
    raw = get('raw', probe)
    if raw is None:
        continue
    off = [(i, j) for i in range(n) for j in range(n) if i != j]
    rawv = [raw[i][j] for i, j in off]
    print(f'\n--- probe={probe} : {len(off)} off-diagonal cells ---')
    print(f'    raw off-diag mean = {sum(rawv)/len(rawv):.4f}; '
          f'cells with raw<0.5: {sum(1 for v in rawv if v < 0.5)}; '
          f'raw<0.6: {sum(1 for v in rawv if v < 0.6)}; '
          f'raw>=0.85: {sum(1 for v in rawv if v >= 0.85)}')
    for feat in ['random@2', 'pc:unified@2', 'datapca@2', 'random', 'pc:unified', 'datapca']:
        m = get(feat, probe)
        if m is None:
            continue
        dv = [m[i][j] - raw[i][j] for i, j in off]
        lo = [x for x, r in zip(dv, rawv) if r < 0.6]
        hi = [x for x, r in zip(dv, rawv) if r >= 0.85]
        won = sum(1 for x in dv if x > 0)
        print(f'    {feat:14s} mean delta vs raw = {sum(dv)/len(dv):+.4f}  pairs won {won}/{len(dv)}'
              f'   | raw<0.6 (n={len(lo)}): {sum(lo)/len(lo):+.4f}'
              f'   | raw>=0.85 (n={len(hi)}): {sum(hi)/len(hi):+.4f}'
              f'   | corr(raw,delta) = {corr(rawv, dv):+.3f}')
