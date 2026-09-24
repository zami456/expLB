"""P9 / F-29: recompute the sycophancy_mt inversion table from the cell-level matrix.
Source: run-crossgen/kd11/matrices/mean_L40_s42.json  (kd11 store, paper16 fp32 bank, k=5).
Read-only."""
import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
P = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'run-crossgen', 'kd11', 'matrices', 'mean_L40_s42.json')
d = json.load(open(P, encoding='utf-8'))
ds = d['datasets']
j = ds.index('sycophancy_mt')
print('file   :', P)
print('bank   :', d['bank'], '| bank_name:', d['bank_name'], '| k:', d['k'], '| layer:', d['layer'], '| seed:', d['seed'], '| pooling:', d['pooling'])
print('datasets (%d):' % len(ds), ds)
print()

# zero-shot directions
print('--- zero-shot AUROC on sycophancy_mt ---')
for z in d['zeroshot']:
    print('  %-12s %.4f   (inverted %.4f)' % (z['direction'], z['auroc'][j], 1 - z['auroc'][j]))
print()

# trained probes: target column mean (exclude diagonal) and source row mean (exclude diagonal)
print('--- trained probes, probe=paper: sycophancy_mt as TARGET (column) and SOURCE (row) ---')
print('%-16s %8s %8s | %8s %8s' % ('feature', 'target', 'inv', 'source', 'inv'))
for r in d['results']:
    if r['probe'] != 'paper':
        continue
    A = r['auroc']
    col = [A[i][j] for i in range(len(ds)) if i != j]
    row = [A[j][i] for i in range(len(ds)) if i != j]
    mt, ms = sum(col) / len(col), sum(row) / len(row)
    print('%-16s %8.4f %8.4f | %8.4f %8.4f  (n=%d each)' % (r['feature'], mt, 1 - mt, ms, 1 - ms, len(col)))
print()

def get(feat, probe='paper'):
    for r in d['results']:
        if r['feature'] == feat and r['probe'] == probe:
            A = r['auroc']
            col = [A[i][j] for i in range(len(ds)) if i != j]
            row = [A[j][i] for i in range(len(ds)) if i != j]
            return sum(col)/len(col), sum(row)/len(row)
    return None, None

ax_t, ax_s = get('axes')
rd_t, rd_s = get('random')
print('--- F-29 ledger table, recomputed ---')
print('  zero-shot PC1      raw %.4f   inverted %.4f' % (
    [z for z in d['zeroshot'] if z['direction'] != 'contrast'][0]['auroc'][j],
    1 - [z for z in d['zeroshot'] if z['direction'] != 'contrast'][0]['auroc'][j]))
print('  trained as TARGET  raw %.4f   inverted %.4f   random %.4f   inv-random = %+.4f' % (ax_t, 1-ax_t, rd_t, (1-ax_t)-rd_t))
print('  trained as SOURCE  raw %.4f   inverted %.4f   random %.4f   inv-random = %+.4f' % (ax_s, 1-ax_s, rd_s, (1-ax_s)-rd_s))
