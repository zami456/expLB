import json, os
p = "C:/Users/user4/expLB/run-crossgen/scale_70b/matrices/mean_L40_s42.json"
d = json.load(open(p, encoding="utf-8"))
print("L40 file:", p)
print("  bank", d.get("bank"), "| bank_name", d.get("bank_name"), "| k", d.get("k"))
print("  zeroshot dirs:", [z["direction"] for z in d.get("zeroshot", [])])
print("  features:", sorted({r['feature'] for r in d['results']}))
q = "C:/Users/user4/expLB/run-crossgen/scale_70b/matrices/mean_L44_s42.json"
e = json.load(open(q, encoding="utf-8"))
print("L44 file: bank", e.get("bank"), "| name", e.get("bank_name"), "| zs", [z["direction"] for z in e.get("zeroshot", [])])
