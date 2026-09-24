# W6E-memory-q : Q8 claims M-072..M-078  (memory/q8-per-token-beats-pooling.md)
import json, os, statistics
ROOT = r'C:/Users/SURAIYA/Documents/STUDY/thesis/expLB'
def J(p): return json.load(open(os.path.join(ROOT, p)))

print('=== HALF 1: pertoken_q8.json (zero-shot projection on the deception axis, Pass C) ===')
d1 = J('run-coords/pertoken_q8.json')
print('axis =', d1['axis'], ' bank =', d1['bank'])
print(f'{"layer":>5} {"mean":>8} {"last":>8} {"last8":>8} {"max":>8} {"top10pct":>9} {"first":>8} {"top10-mean":>11}')
for r in d1['aggregations']:
    print(f'{r["layer"]:>5} {r["mean"]:8.4f} {r["last"]:8.4f} {r["last8"]:8.4f} {r["max"]:8.4f} {r["top10pct"]:9.4f} {r["first"]:8.4f} {r["top10pct"]-r["mean"]:+11.4f}')
gaps = [(r['layer'], r['top10pct'] - r['mean']) for r in d1['aggregations']]
print('top10pct > mean at every layer :', all(g > 0 for _, g in gaps))
print('MAX gain over layers           :', f'{max(gaps, key=lambda t: t[1])[1]:+.4f} at layer {max(gaps, key=lambda t: t[1])[0]}')
print('gain at the top10pct-peak layer:', f'{[g for L,g in gaps if L==max(d1["aggregations"], key=lambda r: r["top10pct"])["layer"]][0]:+.4f} '
      f'at layer {max(d1["aggregations"], key=lambda r: r["top10pct"])["layer"]}')

print()
print('=== HALF 2: q8_trained_pertoken.json (LODO, Pass C) ===')
d2 = J('run-coords/q8_trained_pertoken.json')
print('protocol', d2['protocol'], '| control', d2['control'], '| layers', d2['layers'],
      '| n_rows', d2['n_rows'], '| tokens_per_row', d2['tokens_per_row'], '| held_out', d2['held_out'])
print('self_checks:', json.dumps(d2['self_checks'])[:300])
rows = d2['results']
keys = sorted({(r['probe'], r['aggregation']) for r in rows})
print()
print(f'{"probe/agg":32s} ' + ' '.join(f'{("L%d" % L):>16}' for L in d2['layers']))
tbl = {}
for k in keys:
    line = f'{k[0] + "/" + k[1]:32s} '
    for L in d2['layers']:
        m = [r for r in rows if r['probe'] == k[0] and r['aggregation'] == k[1] and r['layer'] == L]
        if m:
            tbl[(k, L)] = (m[0]['auroc_mean'], m[0]['tpr_mean'])
            line += f'{m[0]["auroc_mean"]:.3f}/{m[0]["tpr_mean"]:.3f}'.rjust(17)
        else:
            line += ' ' * 17
    print(line)

print()
print('=== pooled vs per-token, per layer (best pooled minus best token-trained) ===')
POOLED = [k for k in keys if k[0].startswith('pool')]
TOKEN = [k for k in keys if k[0].startswith('tok')]
for L in d2['layers']:
    bp = max((tbl[(k, L)][0] for k in POOLED if (k, L) in tbl), default=None)
    bt = max((tbl[(k, L)][0] for k in TOKEN if (k, L) in tbl), default=None)
    pmm = tbl.get((('pool_mean_mms', 'pooled'), L))
    print(f'L{L:<3} best pooled={bp:.4f}  best token-trained={bt:.4f}  diff={bp-bt:+.4f}   '
          f'pool_mean_mms={pmm[0]:.4f}/{pmm[1]:.3f}')

print()
print('=== per-dataset: token LR/mean minus pooled mean mms, at each layer ===')
for L in d2['layers']:
    a = [r for r in rows if r['probe'] == 'tok_lr' and r['aggregation'] == 'mean' and r['layer'] == L]
    b = [r for r in rows if r['probe'] == 'pool_mean_mms' and r['layer'] == L]
    if not a or not b: continue
    fa = {f['held_out']: f['auroc'] for f in a[0]['per_fold']}
    fb = {f['held_out']: f['auroc'] for f in b[0]['per_fold']}
    print(f'-- L{L}: ' + '  '.join(f'{k}={fa[k]-fb[k]:+.3f}' for k in sorted(fa)))

print()
print('=== zero-shot axis top10pct TPR@1%FPR vs pooled mean mms ===')
for L in d2['layers']:
    z = [r for r in rows if r['probe'] == 'zeroshot_axis' and r['aggregation'] == 'top10pct' and r['layer'] == L]
    p = [r for r in rows if r['probe'] == 'pool_mean_mms' and r['layer'] == L]
    if z and p:
        print(f'L{L:<3} zeroshot_axis/top10pct auroc={z[0]["auroc_mean"]:.4f} tpr={z[0]["tpr_mean"]:.4f} | '
              f'pool_mean_mms auroc={p[0]["auroc_mean"]:.4f} tpr={p[0]["tpr_mean"]:.4f}')
print()
print('=== length_only rows ===')
for r in rows:
    if r['probe'] == 'length_only':
        print(f'L{r["layer"]:<3} auroc={r["auroc_mean"]:.4f} tpr={r["tpr_mean"]:.4f}')

print()
print('=== M-077 detail: every token-vs-pooled pairing, per held-out dataset, at L40 and L44 ===')
d2 = J('run-coords/q8_trained_pertoken.json'); rows = d2['results']
def fold(probe, agg, L):
    m = [r for r in rows if r['probe'] == probe and r['aggregation'] == agg and r['layer'] == L]
    return {f['held_out']: f['auroc'] for f in m[0]['per_fold']} if m else None
TOK = [('tok_lr', 'mean'), ('tok_lr', 'top10pct'), ('tok_mms', 'mean'), ('tok_mms', 'top10pct'), ('tok_mms', 'max'), ('tok_lr', 'max')]
POO = [('pool_mean_mms', 'pooled'), ('pool_last_mms', 'pooled'), ('pool_mean_lr', 'pooled')]
for L in (40, 44):
    print(f'--- layer {L} ---')
    for tp in TOK:
        ft = fold(*tp, L)
        for pp in POO:
            fp = fold(*pp, L)
            if not ft or not fp: continue
            diffs = {k: ft[k] - fp[k] for k in ft}
            print(f'  {tp[0]}/{tp[1]:9s} - {pp[0]:14s}: ' + ' '.join(f'{k[:18]}={v:+.3f}' for k, v in sorted(diffs.items())))
    print(f'  BEST-token-per-dataset minus BEST-pooled-per-dataset, layer {L}:')
    bt = {}; bp = {}
    for k in ft:
        bt[k] = max(fold(*tp, L)[k] for tp in TOK if fold(*tp, L))
        bp[k] = max(fold(*pp, L)[k] for pp in POO if fold(*pp, L))
    print('   ' + ' '.join(f'{k[:18]}={bt[k]-bp[k]:+.3f}' for k in sorted(bt)))

print()
print('=== fold-mean consistency check (M-074, M-075): stored auroc_mean vs mean of per_fold ===')
import statistics as _st
worst = 0.0
for r in rows:
    if not r.get('per_fold'): continue
    m = _st.mean(f['auroc'] for f in r['per_fold'])
    t = _st.mean(f['tpr_at_1pct_fpr'] for f in r['per_fold'])
    da, dt = abs(m - r['auroc_mean']), abs(t - r['tpr_mean'])
    worst = max(worst, da, dt)
    if r['probe'] in ('pool_mean_mms', 'tok_lr') and r['aggregation'] in ('pooled', 'mean'):
        print(f'  {r["probe"]:14s}/{r["aggregation"]:7s} L{r["layer"]:<3} stored={r["auroc_mean"]:.6f} recomputed={m:.6f} '
              f'(n_folds={len(r["per_fold"])})  tpr stored={r["tpr_mean"]:.6f} recomputed={t:.6f}')
print(f'  worst |stored - recomputed| over ALL {sum(1 for r in rows if r.get("per_fold"))} records = {worst:.2e}')
