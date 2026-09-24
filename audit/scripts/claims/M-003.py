# M-003: smoke L44 last, 9-block 100%-target means vs earlier sweep
import json, sys
sys.path.insert(0, 'audit/scripts/claims')
from _cg import mat, offdiag_mean
P='run-crossgen/smoke/last_L44_s42.json'
d=json.load(open(P,encoding='utf-8'))
print('file:',P,'pooling',d['pooling'],'layer',d['layer'],'seed',d['seed'],'bank',d['bank'],'k',d['k'])
print('seconds recorded:', d['seconds'])
sc=d['smoke_checks']['consistency_with_earlier_sweep']
print('\nLEVEL 1 (stored smoke_checks):')
for f,v in sc.items():
    print(f'  {f:14s} new_9block={v["new_9block"]:.6f} earlier_sweep={v["earlier_sweep"]:.6f} absdiff={v["abs_diff"]:.6f}')
print('\nLEVEL 2 (recomputed from the stored 10x10 matrices, paper probe, 9-block off-diagonal):')
b9=d['block9']; ds=d['datasets']
idx=[ds.index(x) for x in b9]
for f in ['raw','pc:deception','axes']:
    M=mat(d,f,'paper')
    vals=[M[i][j] for i in idx for j in idx if i!=j]
    print(f'  {f:14s} 9-block offdiag mean = {sum(vals)/len(vals):.6f}  (n={len(vals)})')
    M10=mat(d,f,'paper'); m10,n10=offdiag_mean(M10)
    print(f'  {f:14s} 10-dataset offdiag mean = {m10:.6f} (n={n10})')
