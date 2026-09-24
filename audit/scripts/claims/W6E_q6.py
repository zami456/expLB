# W6E-memory-q : Q6 claims M-067..M-071  (memory/q6-layer-choice-depends-on-method.md)
import json, os, statistics
ROOT = r'C:/Users/SURAIYA/Documents/STUDY/thesis/expLB'
d = json.load(open(os.path.join(ROOT, 'run-coords/q6_layers.json')))
rows = d['rows']
print('targets', d['targets'], 'tol', d['tol'], 'n_rows(methods)', len(rows))
print()
print(f'{"source":34s} {"method":34s} {"file":28s} {"L20":>8s} {"L22":>8s} {"L40":>8s} {"gap2022":>8s} {"40-best":>8s} {"bestL":>5s} {"best":>7s} verdict')
for r in rows:
    a = r['at']
    f = lambda x: ('%8.4f' % x) if x is not None else '    None'
    print(f'{r["source"][:34]:34s} {r["method"][:34]:34s} {r["file"][:28]:28s} {f(a.get("20"))} {f(a.get("22"))} {f(a.get("40"))} '
          f'{f(r["gap_20_22"])} {f(r["gap_40_minus_best_of_20_22"])} {str(r["best_layer"]):>5s} {r["best"]:7.4f} {r["verdict"]}')

gaps = [abs(r['gap_20_22']) for r in rows if r['gap_20_22'] is not None]
print()
print('n rows with both 20 and 22 :', len(gaps))
print('median |gap_20_22| =', round(statistics.median(gaps), 5), ' max =', round(max(gaps), 5), ' min =', round(min(gaps), 5))
print('all |gaps| sorted  :', [round(g, 4) for g in sorted(gaps)])

def cls(m):
    ml = m.lower()
    if ml.startswith('raw') or 'length' in ml: return 'raw/other'
    return 'persona'
groups = {}
for r in rows:
    groups.setdefault(cls(r['method']), []).append(r)
print()
for g, rs in groups.items():
    v = [r['gap_40_minus_best_of_20_22'] for r in rs if r['gap_40_minus_best_of_20_22'] is not None]
    print(f'{g:10s} n={len(v):2d} median(40 - best_of_20_22) = {statistics.median(v):+.4f}  '
          f'range [{min(v):+.4f}, {max(v):+.4f}]')
    for r in rs:
        print(f'    {r["method"]:38s} {r["gap_40_minus_best_of_20_22"]:+.4f}   ({r["source"]})')
