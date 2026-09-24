# -*- coding: utf-8 -*-
"""W5D-memory batch, robustness half. READ-ONLY over C:/Users/user4/expLB.

  M-004  : the instructed-deception ("ID") row mean over the 9 other datasets, every stored
           layer (mean pooling, seed 42) and every stored seed at L40.
  M-018 / M-095 : the random@2 / unified@2 / datapca@2 bucket deltas vs raw at every layer.
  M-032  : sycophancy_mt as target (column) and source (row) for `axes`, every kd11 layer.
"""
import glob
import io
import json
import math
import os
import re

R = r'C:\Users\user4\expLB'


def jload(p):
    with io.open(p, encoding='utf-8', errors='replace') as f:
        return json.load(f)


def get(d, feat, probe):
    for r in d['results']:
        if r['feature'] == feat and r['probe'] == probe:
            return r['auroc']
    return None


def corr(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float('nan')


def lay(p):
    m = re.search(r'_L(\d+)_s(\d+)\.json$', p)
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


print('=' * 78)
print('M-004 / M-018 / M-095 robustness : run-crossgen/matrices, mean pooling')
print('=' * 78)
allf = sorted(glob.glob(os.path.join(R, 'run-crossgen', 'matrices', 'mean_L*_s*.json')),
              key=lambda p: lay(p))
files = [p for p in allf if lay(p)[1] == 42] + [p for p in allf if lay(p)[0] == 40 and lay(p)[1] != 42]
print(f'files found: {len(allf)}; shown (seed 42 sweep + all seeds at L40): {len(files)}')
print(f'{"layer":>5} {"seed":>4} | {"ID raw":>8} {"ID pcdec":>8} {"ID axes":>8} | '
      f'{"raw offdg":>9} {"r@2<0.6":>8} {"r@2>=.85":>9} {"corr":>6} {"u@2<0.6":>8} {"dp@2<0.6":>8}')
for p in files:
    L, s = lay(p)
    d = jload(p)
    ds = d['datasets']
    n = len(ds)
    i = ds.index('instructed-deception')
    out = []
    for feat in ['raw', 'pc:deception', 'axes']:
        a = get(d, feat, 'mms')
        out.append(sum(a[i][j] for j in range(n) if j != i) / (n - 1))
    raw = get(d, 'raw', 'paper')
    if raw is None:
        print(f'{L:>5} {s:>4} | {out[0]:>8.4f} {out[1]:>8.4f} {out[2]:>8.4f} | '
              f'(no paper probe: features={sorted({(r["feature"], r["probe"]) for r in d["results"]})})')
        continue
    off = [(x, y) for x in range(n) for y in range(n) if x != y]
    rawv = [raw[x][y] for x, y in off]
    buck = {}
    for feat in ['random@2', 'pc:unified@2', 'datapca@2']:
        a = get(d, feat, 'paper')
        if a is None:
            buck[feat] = (float('nan'),) * 3
            continue
        dv = [a[x][y] - raw[x][y] for x, y in off]
        lo = [u for u, v in zip(dv, rawv) if v < 0.6]
        hi = [u for u, v in zip(dv, rawv) if v >= 0.85]
        buck[feat] = (sum(lo) / len(lo) if lo else float('nan'),
                      sum(hi) / len(hi) if hi else float('nan'),
                      corr(rawv, dv))
    print(f'{L:>5} {s:>4} | {out[0]:>8.4f} {out[1]:>8.4f} {out[2]:>8.4f} | '
          f'{sum(rawv)/len(rawv):>9.4f} {buck["random@2"][0]:>8.4f} {buck["random@2"][1]:>9.4f} '
          f'{buck["random@2"][2]:>6.2f} {buck["pc:unified@2"][0]:>8.4f} {buck["datapca@2"][0]:>8.4f}')

print()
print('=' * 78)
print('M-032 robustness : run-crossgen/kd11/matrices, axes vs random, sycophancy_mt')
print('=' * 78)
files = sorted(glob.glob(os.path.join(R, 'run-crossgen', 'kd11', 'matrices', '*_L*_s42.json')),
               key=lambda p: (os.path.basename(p).split('_')[0], lay(p)))
print(f'files found: {len(files)}')
print(f'{"pool":>5} {"layer":>5} | {"axes tgt":>8} {"axes src":>8} | {"rand tgt":>8} {"rand src":>8} '
      f'| {"raw tgt":>8} {"raw src":>8}')
for p in files:
    L, s = lay(p)
    d = jload(p)
    ds = d['datasets']
    n = len(ds)
    k = ds.index('sycophancy_mt')
    vals = []
    for feat in ['axes', 'random', 'raw']:
        a = get(d, feat, 'paper')
        if a is None:
            vals.append((float('nan'), float('nan')))
            continue
        vals.append((sum(a[i][k] for i in range(n) if i != k) / (n - 1),
                     sum(a[k][j] for j in range(n) if j != k) / (n - 1)))
    print(f'{d["pooling"]:>5} {L:>5} | {vals[0][0]:>8.4f} {vals[0][1]:>8.4f} | '
          f'{vals[1][0]:>8.4f} {vals[1][1]:>8.4f} | {vals[2][0]:>8.4f} {vals[2][1]:>8.4f}')
print('\nDONE')
