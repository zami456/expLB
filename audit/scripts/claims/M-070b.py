"""M-070 (part 2): the L28 value and the stored-layer set for Pass C LODO pooled-mean mms.
Claim: "Pass C leave-one-dataset-out peaks at L40 (pooled-mean mms 0.883) with L28 within 0.007.
Pass C has no 32/36 stored."
Source file named by q6_layers.json: run-coords/q8_trained_pertoken.json
Read-only.
"""
import json, sys

p = sys.argv[1] if len(sys.argv) > 1 else 'run-coords/q8_trained_pertoken.json'
d = json.load(open(p))
print(f'### {p}')
print(f'    protocol={d["protocol"]} control={d["control"]} layers={d["layers"]} '
      f'n_rows={d["n_rows"]} tokens_per_row={d["tokens_per_row"]}')
print(f'    held_out folds ({len(d["held_out"])}): {d["held_out"]}')

res = d['results']
print(f'    results type={type(res).__name__} len={len(res)}')
rows = res if isinstance(res, list) else [res]
keys = set()
for r in rows:
    if isinstance(r, dict):
        keys |= set(r.keys())
print(f'    result keys: {sorted(keys)}')

# collect (layer, probe/aggregation) -> auroc_mean for the pooled entries
tab = {}
for r in rows:
    if not isinstance(r, dict):
        continue
    L = r.get('layer')
    name = f'{r.get("probe")}/{r.get("aggregation")}'
    if 'auroc_mean' in r:
        tab.setdefault(name, {})[L] = r['auroc_mean']
for name in sorted(tab):
    d2 = tab[name]
    best = max(d2, key=lambda L: d2[L])
    line = '  '.join(f'L{L}={d2[L]:.4f}' for L in sorted(d2))
    print(f'  {name:28s} {line}   best=L{best} {d2[best]:.4f}')
    if 28 in d2 and 40 in d2:
        print(f'{"":30s} L40 - L28 = {d2[40]-d2[28]:+.4f}')

# layer sweep files in run-coords: which layers does each store?
import glob, os
print('\n--- layers stored by other run-coords sweep files ---')
for f in sorted(glob.glob(os.path.join(os.path.dirname(p), 'layer_sweep_*.json')) +
                glob.glob(os.path.join(os.path.dirname(p), 'pertoken_q8.json'))):
    try:
        o = json.load(open(f))
    except Exception as e:
        print(' ', f, 'ERR', e); continue
    ls = o.get('layers')
    print(f'  {os.path.basename(f):32s} layers={ls}')
