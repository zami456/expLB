"""W5E-ownerdoc: local (laptop/git) checks for O-075, O-082, O-086, O-088, O-040, O-065.

All inputs are the working-tree copies under run-coords/ , kd/ and run-crossgen/ .
Read-only. Prints every number to the precision of the file.
"""
import json, os, statistics, sys, glob, csv, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RC = os.path.join(ROOT, "run-coords")


def st(p):
    s = os.stat(p)
    return f"{p}  size={s.st_size}  mtime={__import__('datetime').datetime.fromtimestamp(s.st_mtime)}"


print("#" * 78)
print("O-086 / O-088 -- thesis 7.6 Q6 table rows (docs/thesis/persona-coordinates-70b-results.md:725,727)")
p = os.path.join(RC, "q6_layers.json")
print(st(p))
d = json.load(open(p, encoding="utf-8"))
print(f"  targets={d['targets']}  tol={d['tol']}  n_rows={len(d['rows'])}")
for r in d["rows"]:
    if r["method"] in ("pc:deception / mms / mean", "PC1 / mean", "deception contrast axis / mean"):
        print(f"  [{r['source']}] {r['method']}  file={r['file']}")
        print(f"     L20={r['at']['20']!r}")
        print(f"     L22={r['at']['22']!r}")
        print(f"     L40={r['at']['40']!r}")
        print(f"     best_layer=L{r['best_layer']}  best={r['best']!r}")

print()
print("-- LEVEL 2: recompute the same rows from the files q6_layers.json says it read --")
# pc:deception / mms / mean  <- layer_sweep_passB_mms.json
p2 = os.path.join(RC, "layer_sweep_passB_mms.json")
print(st(p2))
sw = json.load(open(p2, encoding="utf-8"))
print("  top-level keys:", list(sw.keys()) if isinstance(sw, dict) else type(sw))
print("  " + json.dumps(sw)[:1200])

print()
p3 = os.path.join(RC, "dotproducts_mean.json")
print(st(p3))
dp = json.load(open(p3, encoding="utf-8"))
print("  layers:", dp["layers"])
print("  matched (per-layer pooled-corpus zero-shot AUROC):")
for m in dp["matched"]:
    print(f"    L{m['layer']:<3} contrast={m['contrast']!r}  pc1={m['pc1']!r}")
best_c = max(dp["matched"], key=lambda m: m["contrast"])
best_p = max(dp["matched"], key=lambda m: m["pc1"])
print(f"  argmax contrast = L{best_c['layer']} {best_c['contrast']:.4f}")
print(f"  argmax pc1      = L{best_p['layer']} {best_p['pc1']:.4f}")
for k in dp:
    if k not in ("layers", "matched", "axis_cos", "cross_auroc"):
        print(f"  other key {k}: {json.dumps(dp[k])[:400]}")

print()
print("#" * 78)
print("O-075 -- thesis 7.4 Q4 table row 'L20 mean, mms' (line 656)")
p4 = os.path.join(RC, "q3_q4_bootstrap.json")
print(st(p4))
q = json.load(open(p4, encoding="utf-8"))
print(f"  boot_rows={q['boot_rows']} boot_datasets={q['boot_datasets']}")
print("  q4_table:")
for row in q["q4_table"]:
    print("    " + json.dumps(row))
print("  comparisons whose name mentions eras/assistant:")
for c in q["comparisons"]:
    if "eras" in c["name"].lower() or "assistant" in c["name"].lower() or c.get("question") == "Q4":
        print("    " + json.dumps(c))
print(f"  total comparisons in this file (BH family size) = {len(q['comparisons'])}")
fam = {}
for c in q["comparisons"]:
    fam[c.get("question")] = fam.get(c.get("question"), 0) + 1
print("  comparisons by question:", fam)

p5 = os.path.join(RC, "erase_test.json")
if os.path.exists(p5):
    print(" ", st(p5))
    e = json.load(open(p5, encoding="utf-8"))
    print(f"    layer={e['layer']} pooling={e['pooling']} k={e['k']} bank={e['bank']} datasets={list(e['datasets'])}")

print()
print("#" * 78)
print("O-082 -- thesis 7.5: 'dataset identity 99.3% decodable, label lift +0.032' (line 702-703)")
p6 = os.path.join(RC, "validate_L40.json")
print(st(p6))
v = json.load(open(p6, encoding="utf-8"))
print("  " + json.dumps(v, indent=2)[:2500])

print()
print("#" * 78)
print("O-065 -- thesis 4.1 own100 bank row (line 377)")
for cand in ["run-hf275", "run", "run-paper16"]:
    pth = os.path.join(ROOT, cand)
    if os.path.isdir(pth):
        print(f"  {pth}: {sorted(os.listdir(pth))[:30]}")
