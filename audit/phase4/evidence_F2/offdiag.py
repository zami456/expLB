import json,sys,os
def offdiag(m):
    n=len(m); s=0.0;c=0
    for i in range(n):
        for j in range(n):
            if i!=j: s+=m[i][j]; c+=1
    return s/c, c
def diag(m):
    n=len(m); return sum(m[i][i] for i in range(n))/n
for path in sys.argv[1:]:
    d=json.load(open(path))
    print('='*90)
    print(path)
    print('  pooling=%s layer=%s seed=%s k=%s bank=%s bank_name=%s control=%s frac_train=%s'%(d.get('pooling'),d.get('layer'),d.get('seed'),d.get('k'),d.get('bank'),d.get('bank_name'),d.get('control'),d.get('frac_train')))
    print('  datasets=',d.get('datasets'))
    raws={}
    for r in d['results']:
        od,c=offdiag(r['auroc'])
        raws[(r['feature'],r['probe'])]=od
        print('  TRAINED feat=%-22s probe=%-8s offdiag=%.4f (n=%d) diag=%.4f'%(r['feature'],r['probe'],od,c,diag(r['auroc'])))
    z=d.get('zeroshot')
    if z:
        print('  --- zeroshot ---')
        for e in z:
            if isinstance(e,dict):
                print('   ', {k:(round(v,4) if isinstance(v,float) else v) for k,v in e.items() if not isinstance(v,(list,dict))})
                for k,v in e.items():
                    if isinstance(v,(list,dict)): print('      ',k,'=',v)
