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
print('A. VERSION CHECK  (size / mtime / md5 on the lab PC)')
print('=' * 78)
FILES = ['run-crossgen/matrices/mean_L40_s42.json',
         'run-crossgen/matrices/last_L40_s42.json',
         'run-crossgen/kd11/matrices/mean_L40_s42.json',
         'run-coords/q6_layers.json',
         'run-coords/q8_trained_pertoken.json',
         'run-coords/significance.json',
         'run-coords/q10_sycophancy_axis.json',
         'run-coords/confounds_L44_last.json',
         'run-paper16/axes_vectors_s3.pt',
         'run-paper16/axes_vectors_s3_fp32.pt',
         'run-paper16/scores/pc_archivist.json',
         'run-paper16/scores_405b/pc_archivist.json']
for f in FILES:
    print(f'  {f:52s} {stat(P(*f.split("/")))}')

print()
print('=' * 78)
print('B. M-004 / M-018 / M-095  on the LAB run-crossgen/matrices')
print('=' * 78)


def offrow(d, feat, probe, row_ds, exclude=()):
    ds = d['datasets']
    i = ds.index(row_ds)
    for r in d['results']:
        if r['feature'] == feat and r['probe'] == probe:
            v = [r['auroc'][i][j] for j in range(len(ds)) if j != i and ds[j] not in exclude]
            return sum(v) / len(v), len(v)
    return None, 0


def corr(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


for pool in ['mean', 'last']:
    p = P('run-crossgen', 'matrices', f'{pool}_L40_s42.json')
    d = jload(p)
    print(f'\n-- {pool}_L40_s42.json  bank={d["bank"]} k={d["k"]} datasets={len(d["datasets"])}')
    for probe in ['mms', 'lr_reg10']:
        for feat in ['raw', 'pc:deception', 'axes']:
            m, n = offrow(d, feat, probe, 'instructed-deception')
            m2, n2 = offrow(d, feat, probe, 'instructed-deception', exclude=('sycophancy_mt',))
            print(f'   ID-row {probe:8s} {feat:13s} mean/{n} = {m:.4f}   w/o syc mean/{n2} = {m2:.4f}')
    ds = d['datasets']
    k = ds.index('harm-pressure-knowledge-report')
    for feat in ['random', 'length']:
        for r in d['results']:
            if r['feature'] == feat and r['probe'] == 'mms':
                print(f'   ID->HP-KR  {feat:8s} (mms) = {r["auroc"][0][k]:.4f}')
    if pool == 'mean':
        raw = None
        for r in d['results']:
            if r['feature'] == 'raw' and r['probe'] == 'paper':
                raw = r['auroc']
        off = [(i, j) for i in range(len(ds)) for j in range(len(ds)) if i != j]
        rawv = [raw[i][j] for i, j in off]
        print(f'   raw(paper) off-diag mean={sum(rawv)/len(rawv):.4f}  cells<0.5: '
              f'{sum(1 for v in rawv if v < 0.5)}  <0.6: {sum(1 for v in rawv if v < 0.6)}  '
              f'>=0.85: {sum(1 for v in rawv if v >= 0.85)}')
        for feat in ['random@2', 'pc:unified@2', 'datapca@2']:
            for r in d['results']:
                if r['feature'] == feat and r['probe'] == 'paper':
                    dv = [r['auroc'][i][j] - raw[i][j] for i, j in off]
                    lo = [a for a, b in zip(dv, rawv) if b < 0.6]
                    hi = [a for a, b in zip(dv, rawv) if b >= 0.85]
                    print(f'   {feat:13s} mean delta={sum(dv)/len(dv):+.4f} won={sum(1 for a in dv if a>0)}/90'
                          f'  raw<0.6: {sum(lo)/len(lo):+.4f}  raw>=0.85: {sum(hi)/len(hi):+.4f}'
                          f'  corr={corr(rawv, dv):+.3f}')

print()
print('=' * 78)
print('C. M-032 / M-036  on the LAB run-crossgen/kd11/matrices/mean_L40_s42.json')
print('=' * 78)
d = jload(P('run-crossgen', 'kd11', 'matrices', 'mean_L40_s42.json'))
ds = d['datasets']
n = len(ds)
print(f'   bank={d["bank"]} bank_name={d.get("bank_name")} k={d["k"]} datasets={n}')
for feat in ['axes', 'random']:
    for r in d['results']:
        if r['feature'] == feat and r['probe'] == 'paper':
            for t in ['harm-pressure-knowledge-report', 'sycophancy_mt']:
                k = ds.index(t)
                col = sum(r['auroc'][i][k] for i in range(n) if i != k) / (n - 1)
                row = sum(r['auroc'][k][j] for j in range(n) if j != k) / (n - 1)
                print(f'   {feat:7s} {t:32s} target(col)={col:.4f} source(row)={row:.4f}')
ksyc = ds.index('sycophancy_mt')
for z in d['zeroshot']:
    print(f'   zeroshot {z["direction"]:10s} sycophancy_mt = {z["auroc"][ksyc]:.4f}')

print()
print('=' * 78)
print('D. M-014 / M-015  judge agreement on the LAB run-paper16 scores')
print('=' * 78)
A, B = {}, {}
for sub, D in (('scores', A), ('scores_405b', B)):
    for p in sorted(glob.glob(P('run-paper16', sub, '*.json'))):
        role = os.path.basename(p)[:-5]
        for k, v in jload(p).items():
            D[(role, k)] = v
common = sorted(set(A) & set(B))
ag = sum(1 for k in common if A[k] == B[k])


def dist(D, keys):
    o = {}
    for k in keys:
        o[D[k]] = o.get(D[k], 0) + 1
    return dict(sorted(o.items()))


print(f'   70B items {len(A)}  405B items {len(B)}  comparable {len(common)}')
print(f'   agreement {ag}/{len(common)} = {100*ag/len(common):.4f}%')
print(f'   70B  dist {dist(A, common)}')
print(f'   405B dist {dist(B, common)}')
print(f'   score-3 rate 70B  {sum(1 for k in common if A[k]==3)}/{len(common)} = '
      f'{100*sum(1 for k in common if A[k]==3)/len(common):.2f}%')
print(f'   score-3 rate 405B {sum(1 for k in common if B[k]==3)}/{len(common)} = '
      f'{100*sum(1 for k in common if B[k]==3)/len(common):.2f}%')
roles = {}
for k in common:
    roles.setdefault(k[0], [0, 0])
    roles[k[0]][1] += 1
    roles[k[0]][0] += (A[k] == B[k])
for r, (ok, nn) in sorted(roles.items(), key=lambda kv: kv[1][0] / kv[1][1])[:4]:
    print(f'     {r:26s} {ok}/{nn} = {100*ok/nn:.1f}%')
nresp = 0
for p in sorted(glob.glob(P('run-paper16', 'responses', '*'))):
    try:
        o = jload(p)
        nresp += len(o)
    except Exception:
        pass
print(f'   responses/ records: {nresp}')

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
