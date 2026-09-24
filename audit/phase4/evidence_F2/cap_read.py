import json,sys,glob,os
TARGET='sycophancy_mt'
def colmean(m, ds, target):
    j=ds.index(target); n=len(ds)
    v=[m[i][j] for i in range(n) if i!=j]
    return sum(v)/len(v), len(v)
for p in sorted(glob.glob('audit/phase4/_f2work/cap_*_mean_L*_s42.json')):
    d=json.load(open(p)); ds=d['datasets']
    print('='*80); print(os.path.basename(p),'| bank=%s k=%s layer=%s pooling=%s seed=%s ndat=%d'%(d.get('bank_name'),d.get('k'),d['layer'],d['pooling'],d['seed'],len(ds)))
    rows=[]
    for r in d['results']:
        if r['probe']!='paper': continue
        cm,n=colmean(r['auroc'],ds,TARGET)
        rows.append((cm,r['feature'],n))
    for cm,f,n in sorted(rows,reverse=True):
        print('   %-24s colmean(%s as TARGET, paper probe)=%.4f  (n=%d)'%(f,TARGET,cm,n))
    z=d.get('zeroshot')
    if z:
        for e in z:
            a=e.get('auroc')
            if a: print('   ZEROSHOT %-10s sycophancy_mt=%.4f  | all: %s'%(e.get('direction'), a[ds.index(TARGET)], [round(x,3) for x in a]))
