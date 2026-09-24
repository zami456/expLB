# M-006 & M-007: Figure-6-style panel, ROOT store, L40 mean, paper probe, 10 datasets
import json, sys
sys.path.insert(0,'audit/scripts/claims')
from _cg import mat, offdiag_mean
P='run-crossgen/matrices/mean_L40_s42.json'
d=json.load(open(P,encoding='utf-8'))
ds=d['datasets']; n=len(ds)
print('file',P,'bank',d['bank'],'k',d['k'],'pooling',d['pooling'],'layer',d['layer'],'seed',d['seed'])
R=mat(d,'raw','paper')
rm,npairs=offdiag_mean(R)
print(f'raw off-diag mean (paper, mean pool, 10 ds) = {rm:.4f}  n_pairs={npairs}')
for f in ['random@2','datapca@2','pc:unified@2','random','datapca','pc:unified','axes','pc:deception']:
    try: M=mat(d,f,'paper')
    except KeyError: print(f,'MISSING'); continue
    m,_=offdiag_mean(M)
    won=sum(1 for i in range(n) for j in range(n) if i!=j and M[i][j]>R[i][j])
    print(f'  {f:14s} mean={m:.4f}  delta_vs_raw={m-rm:+.4f}  pairs_won={won}/90')
print()
print('M-007: named cells, pc:unified@2 minus raw (paper, mean pool)')
U=mat(d,'pc:unified@2','paper')
short={'instructed-deception':'ID','harm-pressure-choice':'HP-C','harm-pressure-knowledge-report':'HP-KR',
 'convincing-game':'CG','insider-trading':'IT','ai_liar':'AIL','sandbagging_v2':'SB','roleplaying':'RP',
 'mask':'MASK','sycophancy_mt':'SYC'}
pairs=[('instructed-deception','insider-trading'),('sandbagging_v2','convincing-game'),
       ('sandbagging_v2','insider-trading'),('harm-pressure-choice','ai_liar')]
for a,b in pairs:
    i,j=ds.index(a),ds.index(b)
    print(f'  {short[a]}->{short[b]}: raw={R[i][j]:.4f} unified@2={U[i][j]:.4f} delta={U[i][j]-R[i][j]:+.4f}')
print()
print('HP-KR row (as source) deltas unified@2 - raw:')
i=ds.index('harm-pressure-knowledge-report')
print('  ', [f'{short[ds[j]]}:{U[i][j]-R[i][j]:+.2f}' for j in range(n) if j!=i])
print('SYC row (as source):')
i=ds.index('sycophancy_mt')
print('  ', [f'{short[ds[j]]}:{U[i][j]-R[i][j]:+.2f}' for j in range(n) if j!=i])
print('SYC column (as target):')
j=ds.index('sycophancy_mt')
print('  ', [f'{short[ds[i]]}:{U[i][j]-R[i][j]:+.2f}' for i in range(n) if i!=j])
