# M-008: paper16_shared bank, L40, paper probe, delta vs raw (10 datasets, 90 pairs)
# Bank stores hold PERSONA features only; raw baseline is merged from the ROOT store
# (run-crossgen/matrices), per the memory note itself.
import json, sys, glob
sys.path.insert(0,'audit/scripts/claims')
from _cg import mat, offdiag_mean
for pool in ['mean','last']:
    Pb=f'run-crossgen/banks/paper16_shared/matrices/{pool}_L40_s42.json'
    Pr=f'run-crossgen/matrices/{pool}_L40_s42.json'
    b=json.load(open(Pb,encoding='utf-8')); r=json.load(open(Pr,encoding='utf-8'))
    assert b['datasets']==r['datasets'], 'dataset order differs'
    n=len(b['datasets'])
    print(f'== bank {Pb}')
    print(f'   bank={b.get("bank")} bank_name={b.get("bank_name")} k={b["k"]} pooling={b["pooling"]} seed={b["seed"]}')
    print(f'   raw from {Pr} (bank field {r.get("bank")})')
    R=mat(r,'raw','paper'); rm,np_=offdiag_mean(R)
    feats=sorted({x['feature'] for x in b['results']})
    print(f'   raw offdiag={rm:.4f} n_pairs={np_}  bank features={feats}')
    for f in ['pc:unified','pc:unified@2','pc:deception','pc:deception@2','axes','axes_resid']:
        if f not in feats: continue
        M=mat(b,f,'paper'); m,_=offdiag_mean(M)
        won=sum(1 for i in range(n) for j in range(n) if i!=j and M[i][j]>R[i][j])
        print(f'   {f:15s} mean={m:.4f} delta={m-rm:+.4f} won={won}/{np_}')
