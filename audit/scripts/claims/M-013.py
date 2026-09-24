# M-013 / M-028: the HP-KR "lexical null" AUROC. Independent recomputation of the
# denial-regex AUROC against the deceptive label, on all rows and on the Llama-3.3-70B rows.
import csv, re, ast, sys, json
csv.field_size_limit(10**9)
P='results/subsets/harm-pressure-knowledge-report.csv'
DEN=re.compile(r"\b(do not|don't|doesn't|not)\b[^.]{0,20}\bknow\b", re.I)
ASS=re.compile(r"^\s*(yes\b|i know\b)", re.I)
def denies(t):
    return bool(DEN.search(t or '')) and not ASS.match(t or '')
def auroc(y, s):
    # rank AUROC with ties averaged, written independently of sklearn
    pairs=sorted(zip(s,y))
    ranks={}; i=0; rk=[0]*len(pairs)
    while i<len(pairs):
        j=i
        while j+1<len(pairs) and pairs[j+1][0]==pairs[i][0]: j+=1
        r=(i+j)/2.0+1
        for t in range(i,j+1): rk[t]=r
        i=j+1
    sp=sum(rk[t] for t in range(len(pairs)) if pairs[t][1]==1)
    n1=sum(y); n0=len(y)-n1
    return (sp - n1*(n1+1)/2)/(n1*n0)
rows=list(csv.DictReader(open(P,encoding='utf-8')))
print('file',P,'rows',len(rows))
print('columns',list(rows[0].keys()))
from collections import Counter
print('models:',Counter(r.get('model','?') for r in rows))
def resp(r):
    m=r.get('messages') or ''
    try:
        msgs=ast.literal_eval(m)
        for x in reversed(msgs):
            if x.get('role')=='assistant': return x.get('content','')
    except Exception:
        pass
    return ''
def lab(r):
    v=str(r.get('deceptive','')).strip().lower()
    return 1 if v in ('true','1','yes') else 0
for name,sel in [('ALL rows', lambda r: True),
                 ('Llama-3.3-70B rows', lambda r: '70b' in str(r.get('model','')).lower() and 'llama' in str(r.get('model','')).lower())]:
    sub=[r for r in rows if sel(r)]
    y=[lab(r) for r in sub]; s=[1 if denies(resp(r)) else 0 for r in sub]
    a=auroc(y,s)
    tp=sum(1 for yy,ss in zip(y,s) if yy==1 and ss==1); fn=sum(1 for yy,ss in zip(y,s) if yy==1 and ss==0)
    fp=sum(1 for yy,ss in zip(y,s) if yy==0 and ss==1); tn=sum(1 for yy,ss in zip(y,s) if yy==0 and ss==0)
    print(f'{name}: n={len(sub)}  pos={sum(y)}  AUROC={a:.4f}  TPR={tp/(tp+fn):.4f} FPR={fp/(fp+tn):.4f}  missed_deceptive={fn}')
