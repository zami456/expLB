import os
R="C:/Users/user4/expLB/run-coords/passB"
tot=0
for d in sorted(os.listdir(R)):
    p=os.path.join(R,d)
    if not os.path.isdir(p): continue
    fs=os.listdir(p); s=sum(os.path.getsize(os.path.join(p,f)) for f in fs)
    rows=os.path.join(p,"rows.jsonl")
    nr=sum(1 for _ in open(rows,encoding="utf-8")) if os.path.exists(rows) else -1
    tot+=s; print(f"{d:34s} {s/1e6:9.1f} MB rows {nr} files {len(fs)}")
print("TOTAL %.2f GB"%(tot/1e9))
