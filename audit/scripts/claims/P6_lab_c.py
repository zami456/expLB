"""P6 batch, lab-side part C.  READ-ONLY.
 (1) F-20: try to RECONSTRUCT the vanished 98-row / 89:9 passA/passC matched slice from the
     stored splits (which may still carry the pre-top-up 200 HP-KR row_ids), and otherwise
     bound it with the hypergeometric expectation.
 (2) F-20: does any passA/passC crossgen matrix contain harm-pressure-kr-matched?
 (3) F-17 / F-04 on the LAB copies of the two significance files + the 21-layer root matrices.
"""
import glob, json, os, re, sys, math
from collections import Counter

R = "C:/Users/user4/expLB"


def head(t):
    print("\n" + "=" * 78 + "\n== " + t + "\n" + "=" * 78)


def st(p):
    import datetime
    return f"{p}  {os.path.getsize(p)} B  mtime={datetime.datetime.fromtimestamp(os.path.getmtime(p)):%Y-%m-%d %H:%M:%S}"


head("F-20 (1)  can the 98-row / 89:9 passA & passC state be reconstructed?")
matched = {}
p = f"{R}/run-coords/passB/harm-pressure-kr-matched/rows.jsonl"
for l in open(p, encoding="utf-8"):
    if l.strip():
        d = json.loads(l)
        matched[d["row_id"]] = d.get("label")
print("  matched slice (passB):", st(p), "->", len(matched), "row_ids",
      dict(Counter(matched.values())))

for store in ("passA", "passB", "passC"):
    for cand in (f"{R}/run-crossgen/{store}/splits.json",
                 f"{R}/run-crossgen/kd11/{store}/splits.json"):
        if not os.path.exists(cand):
            print(f"  no splits at {cand}")
            continue
        print("  " + st(cand))
        s = json.load(open(cand, encoding="utf-8"))
        ks = list(s)[:6]
        print(f"    top-level keys ({len(s)}): {ks}")
        # find HP-KR entries and count how many of their row ids are in the matched slice
        def walk(o, path=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    yield from walk(v, path + "/" + str(k))
            elif isinstance(o, list):
                yield path, o
        for path, lst in walk(s):
            if "knowledge-report" in path or "kr-matched" in path:
                ids = [x for x in lst if isinstance(x, str)]
                if not ids:
                    print(f"    {path}: {len(lst)} entries (not strings)"); continue
                inter = [i for i in ids if i in matched]
                print(f"    {path}: {len(ids)} row_ids; in matched slice: {len(inter)}; "
                      f"labels {dict(Counter(matched[i] for i in inter))}")

print("\n  --- hypergeometric expectation for a 200-row (100D/100H) HP-KR subsample ---")
D_tot, H_tot = 139, 2000
D_m, H_m = 124, 134
print(f"    corpus: {D_tot} deceptive / {H_tot} honest; matched slice keeps {D_m} / {H_m}")
print(f"    100 deceptive drawn from {D_tot}: expected in slice = 100*{D_m}/{D_tot} = {100*D_m/D_tot:.1f}"
      f"  (sd ~ {math.sqrt(100*(D_m/D_tot)*(1-D_m/D_tot)*(D_tot-100)/(D_tot-1)):.1f})")
print(f"    100 honest    drawn from {H_tot}: expected in slice = 100*{H_m}/{H_tot} = {100*H_m/H_tot:.1f}"
      f"  (sd ~ {math.sqrt(100*(H_m/H_tot)*(1-H_m/H_tot)*(H_tot-100)/(H_tot-1)):.1f})")
print(f"    -> F-20's 89 / 9 (total 98) sits at those expectations.")

head("F-20 (2)  do the passA/passC crossgen matrices include harm-pressure-kr-matched?")
for pat in (f"{R}/run-crossgen/kd11/passA/matrices/*.json",
            f"{R}/run-crossgen/kd11/passC/matrices/*.json",
            f"{R}/run-crossgen/passA/matrices/*.json",
            f"{R}/run-crossgen/passC/matrices/*.json",
            f"{R}/run-crossgen/kd11/matrices/*.json"):
    fs = sorted(glob.glob(pat))
    if not fs:
        print(f"  {pat}: none")
        continue
    m = json.load(open(fs[0], encoding="utf-8"))
    print(f"  {pat}: {len(fs)} files; first = {os.path.basename(fs[0])}")
    print(f"    datasets ({len(m['datasets'])}): {m['datasets']}")
    print(f"    'harm-pressure-kr-matched' present: {'harm-pressure-kr-matched' in m['datasets']}")

head("F-17 / F-04 on the LAB copies")
sys.path.insert(0, "C:/Users/user4/audit_scratch/scripts")
os.chdir(R)
for mod, arg in (("F-04", f"{R}/run-crossgen/kd11/significance.json"), ("F-17", f"{R}/run-crossgen")):
    print("\n" + "#" * 70 + f"\n# {mod} on the lab copy\n" + "#" * 70)
    sys.argv = [mod, arg]
    exec(open(f"C:/Users/user4/audit_scratch/scripts/{mod}.py", encoding="utf-8").read(),
         {"__name__": "__main__", "__file__": f"{mod}.py"})
