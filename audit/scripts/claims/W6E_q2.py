# W6E-memory-q : Q2 claims M-054..M-057  (memory/q2-answer-is-layer-dependent.md)
import json, os
ROOT = r'C:/Users/SURAIYA/Documents/STUDY/thesis/expLB'
def J(p): return json.load(open(os.path.join(ROOT, p)))

print('=== A. probes_L20.json (5-dataset subset, L20) vs FULL_L20.json (full corpus, L20) ===')
sub = J('run-coords/probes_L20.json'); full = J('run-coords/FULL_L20.json')
print('subset: layer', sub['layer'], 'pooling', sub['pooling'], 'k', sub['k'], 'bank', sub['bank'], 'datasets', list(sub['datasets']))
print('full  : layer', full['layer'], 'pooling', full['pooling'], 'k', full['k'], 'bank', full['bank'], 'n_datasets', len(full['datasets']))
for tag, d in (('SUBSET', sub), ('FULL', full)):
    for e in d['p1']:
        if e['features'] in ('raw', 'pc:deception', 'axes'):
            print(f'  {tag:6s} {e["features"]:13s} {e["probe"]:9s} off_diag={e["off_diagonal"]:.4f} '
                  f'within={e["within_semantics"]:.4f} across={e["across_semantics"]:.4f} diag={e["diagonal"]:.4f} n_pairs={e["n_pairs"]}')

print()
print('=== B. Q4-at-L20/L22/L40 erasure costs from the FULL_L* files (the figure the Q2 note quotes) ===')
for f in ('run-coords/FULL_L20.json', 'run-coords/FULL_L22.json', 'run-coords/FULL_L40.json'):
    d = J(f)
    by = {(e['features'], e['probe']): e['off_diagonal'] for e in d['p1']}
    print(f'-- {f}  layer={d["layer"]} pooling={d["pooling"]} bank={d["bank"]}')
    for probe in sorted({p for (ft, p) in by}):
        if ('raw', probe) in by and ('erase:assistant', probe) in by:
            cA = by[('raw', probe)] - by[('erase:assistant', probe)]
            cR = by[('raw', probe)] - by.get(('erase:random', probe), float('nan'))
            print(f'   probe={probe:9s} raw={by[("raw",probe)]:.4f} erase:assistant={by[("erase:assistant",probe)]:.4f} '
                  f'cost_assistant={cA:+.4f}   erase:random={by.get(("erase:random",probe),float("nan")):.4f} cost_random={cR:+.4f}')

print()
print('=== C. layer_sweep_passB_mms.json (mean pooling, mms, fp32 bank): the RESOLVED table ===')
sw = J('run-coords/layer_sweep_passB_mms.json')
print('pooling', sw['pooling'], 'probe', sw['probe'], 'k', sw['k'], 'bank', sw['bank'])
best = {}
for r in sw['sweep']:
    print(f'  L{r["layer"]:>2}  raw={r["raw"]:.4f}  pc:deception={r["pc:deception"]:.4f}  axes={r["axes"]:.4f}   '
          f'{"axes>raw" if r["axes"]>r["raw"] else "":8s} {"pc>raw" if r["pc:deception"]>r["raw"] else ""}')
for k in ('raw', 'pc:deception', 'axes'):
    m = max(sw['sweep'], key=lambda r: r[k])
    print(f'  PEAK {k:13s} = {m[k]:.4f} at layer {m["layer"]}')
n_pc_beats = sum(1 for r in sw['sweep'] if r['pc:deception'] > r['raw'])
print('  layers where pc:deception > raw :', n_pc_beats, '/', len(sw['sweep']))
print('  layers where axes > raw         :', [r['layer'] for r in sw['sweep'] if r['axes'] > r['raw']])
print()
print('  label-shuffle null in FULL_L20.json:', json.dumps(full['null']))
