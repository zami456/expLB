"""M-070: memory note q6-layer-choice-depends-on-method.md:19
"... Pass C leave-one-dataset-out peaks at L40 (pooled-mean mms 0.883) with L28 within 0.007.
 Pass C has no 32/36 stored."
Context (line 11): compiled by 22_q6_layers.py into run-coords/q6_layers.json.

Prints every row of q6_layers.json, marks the Pass C / LODO rows, and lists the layers each
row actually stores. Read-only.
"""
import json, sys

p = sys.argv[1] if len(sys.argv) > 1 else 'run-coords/q6_layers.json'
d = json.load(open(p))
print(f'### {p}  targets={d["targets"]} tol={d["tol"]} rows={len(d["rows"])}')
for r in d['rows']:
    at = {int(k): v for k, v in r['at'].items()}
    print(f'\n  source = {r["source"]}')
    print(f'  method = {r["method"]}   file = {r["file"]}')
    print(f'    at targets: ' + '  '.join(f'L{k}={v:.4f}' for k, v in sorted(at.items())))
    print(f'    best_layer = {r["best_layer"]}  best = {r["best"]:.4f}  '
          f'gap40-bestof(20,22) = {r["gap_40_minus_best_of_20_22"]:+.4f}  verdict={r["verdict"]}')
    for k in r:
        if k not in ('source', 'method', 'file', 'at', 'best_layer', 'best', 'gap_20_22',
                     'gap_40_minus_best_of_20_22', 'spread_20_22_40', 'verdict'):
            print(f'    extra {k} = {r[k]}')
