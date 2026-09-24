import os
R="C:/Users/user4/expLB/run-coords/passA"
tot=0
for d in sorted(os.listdir(R)):
    p=os.path.join(R,d)
    if not os.path.isdir(p): continue
    s=sum(os.path.getsize(os.path.join(p,f)) for f in os.listdir(p))
    n=len([f for f in os.listdir(p) if f.startswith("chunk_")])
    rows=os.path.join(p,"rows.jsonl")
    nr=sum(1 for _ in open(rows,encoding="utf-8")) if os.path.exists(rows) else -1
    tot+=s
    print(f"{d:36s} {s/1e6:9.1f} MB  chunks {n:3d}  rows {nr}")
print("TOTAL %.1f GB" % (tot/1e9))
