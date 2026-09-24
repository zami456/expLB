"""W6B-thesis2 — claims O-076..O-096 of docs/thesis/persona-coordinates-70b-results.md
sections 7.4 (Q4), 7.5 (Q5), 7.6 (Q6), 7.7 (Q7).

Read-only. Run from the repo root:
    python audit/scripts/claims/W6B_local.py

Every number printed is read (or recomputed) from a file under run-coords/ or run-paper16/.
No model calls. No writes outside stdout.
"""
import glob
import hashlib
import json
import os
import statistics
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
os.chdir(ROOT)


def finfo(p):
    st = os.stat(p)
    h = hashlib.md5(open(p, 'rb').read()).hexdigest()
    return f'{p}  size={st.st_size}  mtime={time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))}  md5={h}'


def hdr(t):
    print('\n' + '=' * 78)
    print(t)
    print('=' * 78)


# ---------------------------------------------------------------- Q4 (O-076..O-079)
hdr('Q4 / section 7.4 -- O-076, O-077, O-078, O-079')
P = 'run-coords/q3_q4_bootstrap.json'
print(finfo(P))
o = json.load(open(P))
print('top-level keys:', list(o))
print('boot_rows=%s boot_datasets=%s' % (o.get('boot_rows'), o.get('boot_datasets')))
print('\n-- full q4_table --')
for r in o['q4_table']:
    print(json.dumps(r))

comps = o.get('comparisons', [])
print('\nBH family size in this file = %d' % len(comps))
from collections import Counter
print('by question:', dict(Counter(c.get('question') for c in comps)))
print('\n-- every Q4 comparison --')
for c in comps:
    if c.get('question') == 'Q4':
        print(json.dumps(c))

print('\n-- O-079: cost as a FRACTION of the raw transfer AUROC --')
for r in o['q4_table']:
    lab = 'L%s %s, %s' % (r['layer'], r['pooling'], r['probe'])
    frac = r['cost_assistant'] / r['raw']
    print('  %-22s raw=%.4f erased=%.4f cost=%.4f  cost/raw=%.4f = %.2f%%'
          % (lab, r['raw'], r['erase_assistant'], r['cost_assistant'], frac, 100 * frac))
costs = [r['cost_assistant'] for r in o['q4_table']]
fr = [100 * r['cost_assistant'] / r['raw'] for r in o['q4_table']]
print('  absolute cost range  : %.4f .. %.4f (AUROC points)' % (min(costs), max(costs)))
print('  relative cost range  : %.2f%% .. %.2f%%' % (min(fr), max(fr)))
allrand = [c for r in o['q4_table'] for c in r['cost_random']]
print('  random-erasure costs : n=%d  mean=%+.6f  max|.|=%.6f'
      % (len(allrand), sum(allrand) / len(allrand), max(abs(x) for x in allrand)))
sig = [(r['layer'], r['pooling'], r['probe']) for r in o['q4_table']]
print('\n-- significance of the cost, per row (from comparisons) --')
for c in comps:
    if c.get('question') == 'Q4' and 'eras' in c.get('name', '').lower():
        print('  %-52s diff=%.4f q_bh=%.4f significant=%s dataset_ci=%s'
              % (c['name'], c['diff'], c['dataset_q_bh'], c['significant'],
                 [round(x, 4) for x in c['dataset_ci95']]))

# pooled zero-shot assistant-direction reading quoted just below the table
for name in ('zeroshot_paper16.json', 'zeroshot_nulls.json'):
    p = 'run-coords/' + name
    if os.path.exists(p):
        z = json.load(open(p))
        txt = json.dumps(z)
        if 'assistant' in txt:
            print('\n-- %s : entries mentioning assistant --' % name)
            def walk(x, path=''):
                if isinstance(x, dict):
                    for k, v in x.items():
                        walk(v, path + '/' + str(k))
                elif isinstance(x, list):
                    for i, v in enumerate(x):
                        walk(v, path + '[%d]' % i)
                else:
                    if 'assistant' in path.lower():
                        print('   ', path, '=', x)
            walk(z)

# ---------------------------------------------------------------- Q5 (O-081)
hdr('Q5 / section 7.5 -- O-081')
P5 = 'run-coords/q5_semantics_L44_last.json'
print(finfo(P5))
q5 = json.load(open(P5))
print('source matrix:', q5['source'])
print('datasets in the semantics map: %d -> %s' % (len(q5['semantics']), q5['semantics']))
print('\n-- results rows --')
for r in q5['results']:
    print('  features=%-14s probe=%-8s within=%.4f (n=%d)  across=%.4f (n=%d)  diff=%.4f  perm_p=%.4f'
          % (r['features'], r['probe'], r['within_semantics'], r['n_within'],
             r['across_semantics'], r['n_across'], r['difference'], r['permutation_p']))
print('\n-- blocks for features=raw probe=mms (the O-081 row) --')
row = [r for r in q5['results'] if r['features'] == 'raw' and r['probe'] == 'mms'][0]
for k, v in row['blocks'].items():
    print('   %-34s %s' % (k, v))
print('\n-- cross-provenance split for the same row --')
print('   same-construct  n=%d mean=%.4f' % (len(row['cross_provenance_same_construct']),
                                             row['cross_provenance_same_mean']))
print('   diff-construct  n=%d mean=%.4f' % (len(row['cross_provenance_diff_construct']),
                                             row['cross_provenance_diff_mean']))

# ---------------------------------------------------------------- Q6 (O-083..O-094)
hdr('Q6 / section 7.6 -- O-083..O-094')
P6 = 'run-coords/q6_layers.json'
print(finfo(P6))
q6 = json.load(open(P6))
print('targets=%s tol=%s  n_rows=%d' % (q6['targets'], q6['tol'], len(q6['rows'])))
print('\n%-34s %-30s %8s %8s %8s   %-10s' % ('method', 'source', 'L20', 'L22', 'L40', 'own best'))
for r in q6['rows']:
    a = r['at']
    f = lambda v: ('%8.4f' % v) if v is not None else '    None'
    print('%-34s %-30s %s %s %s   L%-3s %.4f'
          % (r['method'], r['source'], f(a['20']), f(a['22']), f(a['40']),
             r['best_layer'], r['best']))

g2022 = [r for r in q6['rows'] if r['gap_20_22'] is not None]
gaps = [r['gap_20_22'] for r in g2022]
print('\n-- O-091: 20 vs 22 --')
print('   rows with both layers: %d' % len(g2022))
print('   median |gap| = %.6f   max = %.6f  (max row: %s)'
      % (statistics.median(gaps), max(gaps),
         max(g2022, key=lambda r: r['gap_20_22'])['method']))
print('   min = %.6f' % min(gaps))

RAWKEY = ('raw /', 'tok_lr', 'tok_mms', 'pool_mean', 'pool_last')
raw_rows, pers_rows = [], []
for r in q6['rows']:
    m = r['method']
    if any(m.startswith(k) for k in RAWKEY):
        raw_rows.append(r)
    else:
        pers_rows.append(r)
print('\n-- O-092 / O-093: layer-40 gain over the better of 20/22 --')
print('   RAW-feature methods (%d):' % len(raw_rows))
for r in raw_rows:
    print('      %-34s %+.4f' % (r['method'], r['gap_40_minus_best_of_20_22']))
rv = [r['gap_40_minus_best_of_20_22'] for r in raw_rows]
print('      median %+.4f   min %+.4f   max %+.4f' % (statistics.median(rv), min(rv), max(rv)))
print('   PERSONA-feature methods (%d):' % len(pers_rows))
for r in pers_rows:
    print('      %-34s %+.4f' % (r['method'], r['gap_40_minus_best_of_20_22']))
pv = [r['gap_40_minus_best_of_20_22'] for r in pers_rows]
print('      median %+.4f   min %+.4f   max %+.4f' % (statistics.median(pv), min(pv), max(pv)))
print('   ratio of medians (persona / raw) = %.2fx' % (statistics.median(pv) / statistics.median(rv)))

# ---------------------------------------------------------------- O-094 late layers
hdr('O-094 -- zero-shot deception axis in late layers (dotproducts_mean.json)')
PD = 'run-coords/dotproducts_mean.json'
print(finfo(PD))
d = json.load(open(PD))
print('top-level keys:', list(d) if isinstance(d, dict) else type(d))


def collect(x, path=''):
    out = []
    if isinstance(x, dict):
        for k, v in x.items():
            out += collect(v, path + '/' + str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out += collect(v, path + '[%d]' % i)
    else:
        out.append((path, x))
    return out


flat = collect(d)
cand = [(p, v) for p, v in flat if isinstance(v, float) and ('60' in p or '68' in p)]
print('-- entries whose path mentions layer 60 or 68 --')
for p, v in cand[:80]:
    print('   ', p, '=', v)

# ---------------------------------------------------------------- Q7 (O-095, O-096)
hdr('Q7 / section 7.7 -- O-095, O-096 (judge agreement)')
a_dir, b_dir = 'run-paper16/scores', 'run-paper16/scores_405b'
A, B = {}, {}
for dd, D in ((a_dir, A), (b_dir, B)):
    for p in sorted(glob.glob(os.path.join(dd, '*.json'))):
        role = os.path.basename(p)[:-5]
        for k, v in json.load(open(p)).items():
            D[(role, k)] = v
print('70B  store %s : %d files, %d scored items' % (a_dir, len(glob.glob(a_dir + '/*.json')), len(A)))
print('405B store %s : %d files, %d scored items' % (b_dir, len(glob.glob(b_dir + '/*.json')), len(B)))
common = sorted(set(A) & set(B))
print('comparable: %d   only-70B: %d   only-405B: %d' % (len(common), len(set(A) - set(B)), len(set(B) - set(A))))
agree = sum(1 for k in common if A[k] == B[k])
print('exact agreement: %d/%d = %.4f%%' % (agree, len(common), 100 * agree / len(common)))
print('70B  dist over all   :', dict(sorted(Counter(A.values()).items())))
print('405B dist over all   :', dict(sorted(Counter(B.values()).items())))
print('70B  dist over common:', dict(sorted(Counter(A[k] for k in common).items())))
print('405B dist over common:', dict(sorted(Counter(B[k] for k in common).items())))
for nm, D, ks in (('70B all', A, list(A)), ('405B all', B, list(B)),
                  ('70B common', A, common), ('405B common', B, common)):
    n3 = sum(1 for k in ks if D[k] == 3)
    print('score-3 rate %-12s: %d/%d = %.2f%%' % (nm, n3, len(ks), 100 * n3 / len(ks)))
n3a = sum(1 for k in common if A[k] == 3) / len(common)
n3b = sum(1 for k in common if B[k] == 3) / len(common)
print('score-3 gap on common items = %.2f pp' % (100 * (n3a - n3b)))
roles = {}
for k in common:
    roles.setdefault(k[0], [0, 0])
    roles[k[0]][1] += 1
    if A[k] == B[k]:
        roles[k[0]][0] += 1
print('\nper-role agreement, lowest 6:')
for r, (ok, n) in sorted(roles.items(), key=lambda kv: kv[1][0] / kv[1][1])[:6]:
    print('   %-28s %d/%d = %.3f' % (r, ok, n, ok / n))
tot = 0
for p in sorted(glob.glob('run-paper16/responses/*')):
    try:
        if p.endswith('.jsonl'):
            tot += sum(1 for _ in open(p, encoding='utf-8'))
        else:
            tot += len(json.load(open(p, encoding='utf-8')))
    except Exception as e:
        print('  skip', p, e)
print('\nresponses/: %d files, %d rollout records' % (len(glob.glob('run-paper16/responses/*')), tot))
