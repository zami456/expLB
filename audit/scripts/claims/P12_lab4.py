"""P12 O-001 attribution completion: every stored zero-shot direction of the paper16 bank on the
two datasets the briefing's Claim 1 names, wherever such a matrix exists. Read-only."""
import json, os
FILES = [
 ("passA11 (paper16 bank, 200-row subsamples)", "C:/Users/user4/expLB/run-crossgen/kd11/passA11/matrices/mean_L40_s42.json"),
 ("passC11 (paper16 bank)",                     "C:/Users/user4/expLB/run-crossgen/kd11/passC11/matrices/mean_L40_s42.json"),
 ("passB kd11 (paper16 bank, the F-13 file)",   "C:/Users/user4/expLB/run-crossgen/kd11/matrices/mean_L40_s42.json"),
 ("passB kd11 banks/collab (OUR kd s23 bank)",  "C:/Users/user4/expLB/run-crossgen/kd11/banks/collab/matrices/mean_L40_s42.json"),
]
TARGETS = ["sycophancy_mt", "harm-pressure-knowledge-report"]
for tag, p in FILES:
    if not os.path.exists(p):
        print("MISSING", p); continue
    d = json.load(open(p, encoding="utf-8"))
    print("=" * 78)
    print(f"{tag}\n  {p}\n  bank={d.get('bank')} bank_name={d.get('bank_name')} k={d.get('k')}")
    idx = {t: d["datasets"].index(t) for t in TARGETS if t in d["datasets"]}
    for z in d["zeroshot"]:
        vals = "  ".join(f"{t}={z['auroc'][i]:.4f}" for t, i in idx.items())
        print(f"    {z['direction']:<28} {vals}")
