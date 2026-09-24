# W6E-memory-q : Q5 claims M-064..M-066  (memory/q5-semantics-does-not-predict-transfer.md)
import json, os, glob
ROOT = r'C:/Users/SURAIYA/Documents/STUDY/thesis/expLB'
def J(p): return json.load(open(os.path.join(ROOT, p)))

print('=== A. q5_semantics_L44_last.json (source:', J('run-coords/q5_semantics_L44_last.json')['source'], ') ===')
d = J('run-coords/q5_semantics_L44_last.json')
for r in d['results']:
    print(f'{r["features"]:14s} {r["probe"]:9s} within={r["within_semantics"]:.4f} (n={r["n_within"]}) '
          f'across={r["across_semantics"]:.4f} (n={r["n_across"]}) diff={r["difference"]:+.4f} perm_p={r["permutation_p"]:.4f}')
    for b, v in r['blocks'].items():
        print(f'      {b:55s} mean={v["mean"]:.4f} n={v["n"]}')

print()
print('=== B. q5_layers/p1_L*.json : intent->intent and belief->belief blocks across 8 layers ===')
import re
rows = {}
for f in sorted(glob.glob(os.path.join(ROOT, 'run-coords/q5_layers/p1_L*.json')),
                key=lambda p: int(re.search(r'L(\d+)', p).group(1))):
    L = int(re.search(r'L(\d+)', f).group(1))
    dd = json.load(open(f))
    print(f'-- L{L}  keys={list(dd.keys())}')
    res = dd.get('results', dd.get('p1'))
    for r in res:
        key = (r['features'], r['probe'])
        blocks = r.get('blocks', {})
        ii = blocks.get('instructed_intent->instructed_intent', {}).get('mean')
        bb = blocks.get('belief_inconsistency->belief_inconsistency', {}).get('mean')
        rows.setdefault(key, []).append((L, ii, bb, r.get('within_semantics'), r.get('across_semantics')))
        print(f'   {r["features"]:14s} {r["probe"]:9s} intent->intent={ii} belief->belief={bb} '
              f'within={r.get("within_semantics")} across={r.get("across_semantics")}')
print()
print('=== C. mean over the 8 layers ===')
for key, v in rows.items():
    ii = [x[1] for x in v if x[1] is not None]
    bb = [x[2] for x in v if x[2] is not None]
    if ii:
        print(f'{key[0]:14s} {key[1]:9s} n_layers={len(v)} mean intent->intent={sum(ii)/len(ii):.4f} '
              f'mean belief->belief={sum(bb)/len(bb):.4f}  (layers {[x[0] for x in v]})')

print()
print('=== D. diagonals for mask and sycophancy_mt (p1_L44_last_withsyc.json) across layers ===')
w = J('run-coords/p1_L44_last_withsyc.json')
print('   keys:', list(w.keys()))

print()
print('=== E. blocks recomputed from the stored matrices, q5_layers/p1_L*.json (8 layers, last pooling, mms) ===')
SEM = J('run-coords/q5_semantics_L44_last.json')['semantics']
import re, glob
files = sorted(glob.glob(os.path.join(ROOT, 'run-coords/q5_layers/p1_L*.json')),
               key=lambda p: int(re.search(r'L(\d+)', p).group(1)))
agg = {}
for f in files:
    L = int(re.search(r'L(\d+)', f).group(1))
    dd = json.load(open(f))
    for e in dd['p1']:
        M = e['matrix']
        ii, bb, diag = [], [], {}
        for k, v in M.items():
            s, t = k.split('->')
            if s == t:
                diag[s] = v; continue
            if SEM.get(s) == 'instructed_intent' and SEM.get(t) == 'instructed_intent': ii.append(v)
            if SEM.get(s) == 'belief_inconsistency' and SEM.get(t) == 'belief_inconsistency': bb.append(v)
        rec = (L, sum(ii)/len(ii), len(ii), sum(bb)/len(bb), len(bb),
               diag.get('mask'), diag.get('sycophancy_mt'),
               M.get('mask->convincing-game'), M.get('sycophancy_mt->sandbagging_v2'),
               M.get('mask->sycophancy_mt'), M.get('sycophancy_mt->mask'))
        agg.setdefault((e['features'], e['probe']), []).append(rec)
        print(f'L{L:<3} {e["features"]:6s} {e["probe"]:5s} intent->intent={rec[1]:.4f}(n={rec[2]}) belief->belief={rec[3]:.4f}(n={rec[4]}) '
              f'diag_mask={rec[5]:.4f} diag_syc={rec[6]:.4f} mask->cg={rec[7]:.4f} syc->sand={rec[8]:.4f} '
              f'mask->syc={rec[9]:.4f} syc->mask={rec[10]:.4f}')
print()
for key, v in agg.items():
    n = len(v)
    print(f'MEAN over {n} layers {key[0]:6s} {key[1]:5s}: intent->intent={sum(x[1] for x in v)/n:.4f} '
          f'belief->belief={sum(x[3] for x in v)/n:.4f}  diag_mask range=[{min(x[5] for x in v):.4f},{max(x[5] for x in v):.4f}] '
          f'diag_syc range=[{min(x[6] for x in v):.4f},{max(x[6] for x in v):.4f}]')
    print(f'    belief->belief per layer: {[round(x[3],4) for x in v]}')
