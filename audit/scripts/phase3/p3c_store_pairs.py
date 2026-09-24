"""P3-C: pairwise store comparison on persona features, matched on
(pooling, layer, seed, dataset_set, probe, feature, k). Two stores that are byte-identical
on persona features used the SAME bank -- a label collision. Read-only."""
import csv, itertools
ROWS = list(csv.DictReader(open('audit/phase2/layer_sensitivity.csv', encoding='utf-8-sig')))
KEY = ('k','pooling','layer','seed','dataset_set','probe','feature')
idx = {}
for r in ROWS:
    if r['feature'] == 'axes' or r['feature'].startswith('pc:'):
        idx.setdefault(r['store'], {})[tuple(r[k] for k in KEY)] = float(r['offdiag_mean'])
stores = sorted(idx)
for a, b in itertools.combinations(stores, 2):
    common = set(idx[a]) & set(idx[b])
    if len(common) < 50: continue
    diffs = [abs(idx[a][k]-idx[b][k]) for k in common]
    ident = sum(1 for d in diffs if d == 0.0)
    if ident / len(common) > 0.30:
        print(f"{a:26s} {b:26s} n={len(common):5d} identical={ident:5d} "
              f"({100*ident/len(common):.0f}%) max|d|={max(diffs):.6f}")
