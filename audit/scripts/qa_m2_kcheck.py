import json, glob, os, collections
for store in ("scale_3b","scale_8b","scale_70b","scale_70b_paper16fp32"):
    d=f"C:/Users/user4/expLB/run-crossgen/{store}/matrices"
    if not os.path.isdir(d): print(f"{store:26s} absent"); continue
    ks=collections.Counter(); banks=collections.Counter()
    for f in glob.glob(d+"/*.json"):
        try:
            m=json.load(open(f,encoding="utf-8")); ks[m.get("k")]+=1; banks[str(m.get("bank_name"))]+=1
        except Exception: pass
    print(f"{store:26s} k={dict(ks)}  bank_name={dict(banks)}")
