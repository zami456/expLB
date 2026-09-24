# M-005 / M-006 / M-007: crossgen ROOT store L40 seed42 (10 datasets)
import json, sys
sys.path.insert(0,'audit/scripts/claims')
from _cg import mat, offdiag_mean
for pool in ['mean','last']:
    P=f'run-crossgen/matrices/{pool}_L40_s42.json'
    d=json.load(open(P,encoding='utf-8'))
    ds=d['datasets']; ID=ds.index('instructed-deception'); KR=ds.index('harm-pressure-knowledge-report')
    SY=ds.index('sycophancy_mt')
    print(f'== {P}  bank={d["bank"]} k={d["k"]} seed={d["seed"]} design-ok')
    print('   datasets:', ds)
    for probe in ['mms','paper','lr_reg10']:
        line=[]
        for f in ['raw','pc:deception','axes','random','length','datapca']:
            try: M=mat(d,f,probe)
            except KeyError: continue
            row=[M[ID][j] for j in range(len(ds)) if j!=ID]
            row9=[M[ID][j] for j in range(len(ds)) if j not in (ID,SY)]
            line.append(f'{f}={sum(row)/len(row):.4f}(9ds w/o syc {sum(row9)/len(row9):.4f})')
        print(f'   ID-row offdiag [{probe}]: ' + '  '.join(line))
    # ID -> HP-KR single cells
    print('   ID->HP-KR single cells:')
    for probe in ['mms','paper','lr_reg10']:
        s=[]
        for f in ['raw','pc:deception','axes','random','length','datapca']:
            try: M=mat(d,f,probe)
            except KeyError: continue
            s.append(f'{f}={M[ID][KR]:.4f}')
        print(f'      [{probe}] ' + '  '.join(s))
    # HP-KR column mean (as target) for random/length
    print('   HP-KR as TARGET (column mean over other sources):')
    for probe in ['mms','paper','lr_reg10']:
        s=[]
        for f in ['raw','random','length','axes','datapca']:
            try: M=mat(d,f,probe)
            except KeyError: continue
            col=[M[i][KR] for i in range(len(ds)) if i!=KR]
            s.append(f'{f}={sum(col)/len(col):.4f}')
        print(f'      [{probe}] ' + '  '.join(s))
    print()
