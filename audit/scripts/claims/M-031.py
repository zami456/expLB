# M-031 / M-033: persona-axes-encode-register table, kd11 grid L40 mean paper probe (11 datasets)
import json, sys
sys.path.insert(0,'audit/scripts/claims')
from _cg import mat, offdiag_mean
P='run-crossgen/kd11/matrices/mean_L40_s42.json'
d=json.load(open(P,encoding='utf-8'))
ds=d['datasets']; n=len(ds)
print(f'{P}: bank={d["bank"]} bank_name={d["bank_name"]} k={d["k"]} pooling={d["pooling"]} layer={d["layer"]} seed={d["seed"]} n_ds={n}')
def tgt(M,j):  # dataset j as TARGET: column mean over other sources
    return sum(M[i][j] for i in range(n) if i!=j)/(n-1)
def src(M,i):  # dataset i as SOURCE: row mean over other targets
    return sum(M[i][j] for j in range(n) if j!=i)/(n-1)
for probe in ['paper','mms','lr_reg10']:
    print(f'\n--- probe = {probe}')
    for name in ['harm-pressure-knowledge-report','sycophancy_mt']:
        i=ds.index(name)
        out=[]
        for f in ['axes','random','raw','pc:deception','pc:unified','datapca','length']:
            try: M=mat(d,f,probe)
            except KeyError: continue
            out.append(f'{f}: target {tgt(M,i):.3f} / source {src(M,i):.3f}')
        print(f'  {name}:')
        for o in out: print('     ',o)
# 11x11 off-diagonal means, and the drop-sycophancy comparison
print('\n--- off-diagonal means over ALL 11 datasets (paper probe), and without sycophancy_mt')
SY=ds.index('sycophancy_mt')
R=mat(d,'raw','paper')
for f in ['raw','axes','pc:deception','pc:unified','random','datapca']:
    M=mat(d,f,'paper')
    m11,_=offdiag_mean(M)
    keep=[k for k in range(n) if k!=SY]
    v10=[M[i][j] for i in keep for j in keep if i!=j]
    print(f'  {f:14s} 11ds={m11:.4f}  10ds(no syc)={sum(v10)/len(v10):.4f}  lift_from_dropping_syc={sum(v10)/len(v10)-m11:+.4f}')
