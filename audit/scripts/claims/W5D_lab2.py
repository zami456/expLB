# -*- coding: utf-8 -*-
"""W5D-memory batch, lab-PC half. READ-ONLY over C:/Users/user4/expLB.

Covers:
  A. version check (size / mtime / md5) of every file the batch's local recomputes used
  B. M-004 / M-018 / M-095 recomputed on the LAB copies of run-crossgen/matrices
  C. M-032 + M-036 (0.374 / 0.409) on the LAB copy of run-crossgen/kd11/matrices/mean_L40_s42.json
  D. M-014 / M-015 judge agreement on the LAB copies of run-paper16/scores{,_405b}
  E. M-070 Pass C LODO layer table on the LAB copy of run-coords/q8_trained_pertoken.json
  F. M-087 significance comparisons on the LAB copy of run-coords/significance.json
  G. M-082 per-dataset length-only AUROC / median / mean from run-coords/passB/*/rows.jsonl
     and from run-coords/confounds_L44_last.json
  H. M-012 axis norms from run-paper16/axes_vectors_s3.pt (+ the fp32 rebuild), if torch loads
"""
import glob
import hashlib
import io
import json
import math
import os
import datetime

R = r'C:\Users\user4\expLB'


def P(*a):
    return os.path.join(R, *a)


def stat(p):
    try:
        st = os.stat(p)
        h = hashlib.md5()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 20), b''):
                h.update(b)
        return f'{st.st_size:>12} B  {datetime.datetime.fromtimestamp(st.st_mtime):%Y-%m-%d %H:%M}  {h.hexdigest()}'
    except Exception as e:
        return f'ERR {e}'


def jload(p):
    with io.open(p, encoding='utf-8', errors='replace') as f:
        return json.load(f)


def auroc(y, x):
    """Mann-Whitney U with midranks == sklearn.roc_auc_score."""
    pairs = sorted(zip(x, y))
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    npos = sum(1 for _, yy in pairs if yy == 1)
    nneg = len(pairs) - npos
    if npos == 0 or nneg == 0:
        return float('nan')
    s = sum(rk for rk, (_, yy) in zip(ranks, pairs) if yy == 1)
    return (s - npos * (npos + 1) / 2.0) / (npos * nneg)


def median(v):
    v = sorted(v)
    n = len(v)
    return float('nan') if not n else (v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0)



print('=' * 78)
print('D2. does run-paper16/scores_405b exist anywhere on the lab?')
print('=' * 78)
hits = []
for base in [R, r'C:\Users\user4']:
    for root, dirs, files in os.walk(base):
        if root.count(os.sep) - base.count(os.sep) > 3:
            dirs[:] = []
            continue
        for d in list(dirs):
            if '405' in d:
                hits.append(os.path.join(root, d))
print('   dirs with 405 in the name:', hits if hits else 'NONE')
print()
print()
print('=' * 78)
print('E. M-070  Pass C LODO on the LAB run-coords/q8_trained_pertoken.json')
print('=' * 78)
q = jload(P('run-coords', 'q8_trained_pertoken.json'))
print(f'   protocol={q["protocol"]} layers={q["layers"]} folds={len(q["held_out"])}')
tab = {}
for r in q['results']:
    if isinstance(r, dict) and 'auroc_mean' in r:
        tab.setdefault(f'{r.get("probe")}/{r.get("aggregation")}', {})[r['layer']] = r['auroc_mean']
for name in ['pool_mean_mms/pooled', 'pool_last_mms/pooled', 'pool_mean_lr/pooled']:
    if name in tab:
        t = tab[name]
        best = max(t, key=lambda L: t[L])
        print(f'   {name:22s} ' + '  '.join(f'L{L}={t[L]:.4f}' for L in sorted(t)) +
              f'   best=L{best} {t[best]:.4f}  L40-L28={t[40]-t[28]:+.4f}')
q6 = jload(P('run-coords', 'q6_layers.json'))
for r in q6['rows']:
    if 'Pass C' in r['source'] and 'pool_mean_mms' in r['method']:
        print(f'   q6_layers row: {r["source"]} | {r["method"]} best_layer={r["best_layer"]} best={r["best"]:.4f}')

print()
print('=' * 78)
print('F. M-087  on the LAB run-coords/significance.json')
print('=' * 78)
s = jload(P('run-coords', 'significance.json'))
print(f'   boot_rows={s["boot_rows"]} boot_datasets={s["boot_datasets"]} datasets={len(s["datasets"])}')
for c in s['comparisons']:
    nm = c.get('name', '')
    if nm.startswith('axes') and ('raw' in nm):
        print(f'   {nm:46s} diff={c["diff"]:+.4f} row_ci95=[{c["row_ci95"][0]:+.4f},{c["row_ci95"][1]:+.4f}] '
              f'row_p={c.get("row_p")} dataset_q_bh={c.get("dataset_q_bh")}')

print()
print('=' * 78)
print('G. M-082  per-dataset length-only AUROC (passB rows.jsonl) + confounds_L44_last.json')
print('=' * 78)
cf = jload(P('run-coords', 'confounds_L44_last.json'))
print(f'   confounds_L44_last.json: layer={cf["layer"]} pooling={cf["pooling"]} bank={cf["bank"]} '
      f'datasets={len(cf["datasets"])}')
lo = cf['confounds']['length_only_auroc']
above = 0
for r in lo:
    above += r['auroc_from_length'] > 0.5
    print(f'     {r["dataset"]:32s} auroc={r["auroc_from_length"]:.4f}  '
          f'median {r["median_pos"]:.0f} vs {r["median_neg"]:.0f}')
print(f'   rows in that table: {len(lo)}; AUROC>0.5 in {above}; '
      f'median_pos>median_neg in {sum(1 for r in lo if r["median_pos"] > r["median_neg"])}')

print('\n   --- recomputed from run-coords/passB/<dataset>/rows.jsonl ---')
tot = {}
for dd in sorted(glob.glob(P('run-coords', 'passB', '*'))):
    rp = os.path.join(dd, 'rows.jsonl')
    if not os.path.isfile(rp):
        continue
    ys, xs = [], []
    with io.open(rp, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            lab = o.get('label')
            if lab is None:
                continue
            try:
                y = int(lab)
            except Exception:
                continue
            if y not in (0, 1):
                continue
            ys.append(y)
            xs.append(float(o.get('span_len', 0)))
    if len(set(ys)) < 2:
        print(f'     {os.path.basename(dd):32s} n={len(ys)} single-class, skipped')
        continue
    pos = [x for x, y in zip(xs, ys) if y == 1]
    neg = [x for x, y in zip(xs, ys) if y == 0]
    a = auroc(ys, xs)
    tot[os.path.basename(dd)] = a
    print(f'     {os.path.basename(dd):32s} n={len(ys):5d} pos={len(pos):5d} neg={len(neg):5d} '
          f'auroc={a:.4f}  median {median(pos):.0f} vs {median(neg):.0f}  '
          f'mean {sum(pos)/len(pos):.2f} vs {sum(neg)/len(neg):.2f} '
          f'(gap {sum(pos)/len(pos)-sum(neg)/len(neg):+.2f})')
print(f'   passB datasets with AUROC>0.5: {sum(1 for a in tot.values() if a > 0.5)} of {len(tot)}')
print(f'   passB datasets with AUROC>0.5, excluding sycophancy_mt: '
      f'{sum(1 for k, a in tot.items() if a > 0.5 and k != "sycophancy_mt")} of '
      f'{sum(1 for k in tot if k != "sycophancy_mt")}')

print()
print('=' * 78)
print('H. M-012  |deception axis| at layer 0 and 76, old bank vs fp32 rebuild')
print('=' * 78)
try:
    import torch
    for f in ['axes_vectors_s3.pt', 'axes_vectors_s3_fp32.pt']:
        p = P('run-paper16', f)
        if not os.path.exists(p):
            print(f'   {f}: MISSING')
            continue
        b = torch.load(p, map_location='cpu', weights_only=False)
        L = b['layers']
        ax = b['axes']['deception']
        g = (lambda i: torch.as_tensor(ax[i]).float().norm()) if isinstance(ax, dict) else \
            (lambda i: torch.as_tensor(ax).float()[L.index(i)].norm())
        vals = {i: float(g(i)) for i in [0, 20, 40, 76, 79]}
        print(f'   {f}: layers={len(L)} ' + '  '.join(f'L{k}={v:.4f}' for k, v in vals.items()) +
              f'  vectors_dir={b.get("vectors_dir")}')
except Exception as e:
    print('   torch unavailable or load failed:', repr(e))
print('\nDONE')
