import json,sys
path=sys.argv[1]; subs=sys.argv[2].split(',') if len(sys.argv)>2 else None
d=json.load(open(path))
ds=d['datasets']
print(path); print(' meta: pooling=%s layer=%s seed=%s k=%s bank=%s bank_name=%s ndatasets=%d'%(d['pooling'],d['layer'],d['seed'],d.get('k'),d.get('bank'),d.get('bank_name'),len(ds)))
print(' datasets:',ds)
if subs:
    idx=[ds.index(s) for s in subs]
else:
    idx=list(range(len(ds)))
print(' subset idx',idx,[ds[i] for i in idx])
res={}
for r in d['results']:
    m=r['auroc']; s=0.0;c=0
    for i in idx:
        for j in idx:
            if i!=j: s+=m[i][j]; c+=1
    res[(r['feature'],r['probe'])]=s/c
    print('  %-22s %-9s offdiag=%.4f (n=%d)'%(r['feature'],r['probe'],s/c,c))
