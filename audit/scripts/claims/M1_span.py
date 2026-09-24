import json, numpy as np
from pathlib import Path
p = Path("C:/Users/user4/expLB/run-coords/passB/harm-pressure-knowledge-report/rows.jsonl")
s = np.array([json.loads(l)["span_len"] for l in p.open(encoding="utf-8")])
print("n", len(s), "min", s.min(), "max", s.max(), "mean %.2f" % s.mean())
for q in (50, 90, 95, 99):
    print(f"  p{q}: linear {np.percentile(s,q):.2f}  nearest {np.percentile(s,q,method='nearest')}"
          f"  higher {np.percentile(s,q,method='higher')}  rank {np.sort(s)[int(np.ceil(q/100*len(s)))-1]}")
print("  rows > 1024:", int((s > 1024).sum()), " > 70:", int((s > 70).sum()))
print("  frac >= 13:", float((s >= 13).mean()), " frac >= 12:", float((s >= 12).mean()))
