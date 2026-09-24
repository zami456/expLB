"""S-14b: which rows go into the POOLED corpus_L40_mean number?
25_zeroshot_nulls.py pools EVERY passB dataset with a rows.jsonl except sycophancy_mt,
including the alpaca control and harm-pressure-kr-matched. Test that reading."""
import json, os, numpy as np, torch
from scipy.stats import rankdata
R="C:/Users/user4/expLB/run-coords/passB"; BANK="C:/Users/user4/expLB/run-paper16/axes_vectors_s3_fp32.pt"
L=40
def auc(s,y):
    r=rankdata(s); n1=int(y.sum()); n0=len(y)-n1
    return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
def load(ds):
    d=os.path.join(R,ds); meta={}
    with open(os.path.join(d,"rows.jsonl"),encoding="utf-8") as fh:
        for line in fh:
            r=json.loads(line); meta[r["row_id"]]=r
    X,y,li,pi=[],[],None,None
    for c in sorted(f for f in os.listdir(d) if f.startswith("chunk_")):
        b=torch.load(os.path.join(d,c),weights_only=False)
        if li is None:
            li=list(b["layers"]).index(L); pi=list(b.get("pooling",["mean","last"])).index("mean")
        for i,rid in enumerate(b["row_ids"]):
            m=meta.get(rid)
            if m is None or m.get("label") is None: continue
            X.append(b["pooled"][i,li,:,pi].float().numpy()); y.append(int(m["label"]))
        del b
    return np.array(X,dtype=np.float32), np.array(y)
bank=torch.load(BANK,map_location="cpu",weights_only=False)
v=bank["axes"]["deception"][L].float().numpy(); v/=np.linalg.norm(v)
a=bank["axes"]["assistant"][L].float().numpy(); a/=np.linalg.norm(a)
keys=sorted(d for d in os.listdir(R) if os.path.isdir(os.path.join(R,d))
            and os.path.exists(os.path.join(R,d,"rows.jsonl")) and d!="sycophancy_mt")
print("keys pooled by the script's rule:",keys)
S,A,Y=[],[],[]
for k in keys:
    X,y=load(k); S.append(X@v); A.append(X@a); Y.append(y)
    print(f"  {k:34s} n={len(y):6d} pos={int(y.sum()):6d} neg={int((1-y).sum()):6d}")
    del X
s=np.concatenate(S); aa=np.concatenate(A); y=np.concatenate(Y)
print(f"\nPOOLED all-of-them n={len(y)} pos={int(y.sum())}  deception {auc(s,y):.4f}  assistant {auc(aa,y):.4f}")
# variants
def sub(drop):
    idx=[i for i,k in enumerate(keys) if k not in drop]
    ss=np.concatenate([S[i] for i in idx]); yy=np.concatenate([Y[i] for i in idx]); ab=np.concatenate([A[i] for i in idx])
    print(f"  without {sorted(drop)}: n={len(yy)}  deception {auc(ss,yy):.4f}  assistant {auc(ab,yy):.4f}")
sub({"alpaca"}); sub({"harm-pressure-kr-matched"}); sub({"alpaca","harm-pressure-kr-matched"})
