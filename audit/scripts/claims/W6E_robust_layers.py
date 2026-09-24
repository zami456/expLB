# W6E: independent robustness look at the layer profile, from audit/phase2/layer_sensitivity.csv
# NOTE: this is the run-crossgen 80/20 protocol, NOT the run-coords pairwise-transfer protocol
# behind the Q-series numbers. Used only as a direction-of-effect cross-check.
import csv, collections, statistics, os
ROOT = r'C:/Users/SURAIYA/Documents/STUDY/thesis/expLB'
agg = collections.defaultdict(list)
with open(os.path.join(ROOT, 'audit/phase2/layer_sensitivity.csv'), encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r['store'] != 'passA': continue
        if r['pooling'] != 'mean' or r['probe'] != 'mms': continue
        agg[(r['feature'], int(r['layer']), r['seed'], r['n_datasets'])].append(float(r['offdiag_mean']))
byfeat = collections.defaultdict(dict)
for (feat, L, seed, nds), v in agg.items():
    byfeat[(feat, seed, nds)][L] = statistics.mean(v)
print('store=passA, pooling=mean, probe=mms  (run-crossgen 80/20 protocol)')
for key in sorted(byfeat, key=lambda k: (k[0], k[1])):
    d = byfeat[key]
    peak = max(d, key=d.get)
    print(f'feature={key[0]:13s} seed={key[1]:3s} n_datasets={key[2]:>3s}  n_layers={len(d):2d}  '
          f'PEAK L{peak} = {d[peak]:.4f}   L20={d.get(20, float("nan")):.4f} L32={d.get(32, float("nan")):.4f} '
          f'L40={d.get(40, float("nan")):.4f} L44={d.get(44, float("nan")):.4f}')

print()
print('=== axes vs raw across all 80 layers, store=passA, seed=42, mean/mms ===')
import csv as _csv
cur = {}
with open(os.path.join(ROOT, 'audit/phase2/layer_sensitivity.csv'), encoding='utf-8') as f:
    for r in _csv.DictReader(f):
        if r['store'] != 'passA' or r['pooling'] != 'mean' or r['probe'] != 'mms': continue
        if r['seed'] != '42' or r['n_datasets'] != '10': continue
        cur[(r['feature'], int(r['layer']))] = float(r['offdiag_mean'])
wins = [L for L in range(80) if ('axes', L) in cur and ('raw', L) in cur and cur[('axes', L)] > cur[('raw', L)]]
print('layers where axes > raw :', wins)
for L in (20, 22, 32, 36, 40, 44, 48, 76, 79):
    if ('raw', L) in cur:
        print(f'  L{L:<3} raw={cur[("raw",L)]:.4f} axes={cur[("axes",L)]:.4f} pc:deception={cur.get(("pc:deception",L),float("nan")):.4f}  '
              f'axes-raw={cur[("axes",L)]-cur[("raw",L)]:+.4f}')
