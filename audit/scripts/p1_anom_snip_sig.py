# READ-ONLY snippet for the lab PC (Task B, anomalies). Run with:
#   python audit/scripts/labpc_run.py audit/scripts/p1_fmap_F001-F027_pyc.ps1 "CODE=$(cat audit/scripts/p1_anom_snip_sig.py)"
# (cwd on the lab = C:/Users/user4/expLB; opens files for reading only)
import json, collections
for p in ["run-crossgen/passA/significance.json","run-crossgen/kd11/passA/significance.json","run-crossgen/passC/significance.json","run-crossgen/kd11/passC/significance.json","run-crossgen/kd11/passC/significance.json.stale-576","run-crossgen/kd11/passA/summary.json","run-crossgen/passA/summary.json"]:
    d = json.load(open(p, encoding="utf-8"))
    print("==", p, type(d).__name__, list(d.keys())[:12] if isinstance(d, dict) else len(d))
    if isinstance(d, dict):
        for k, v in d.items():
            if k in ("tests","records","results"):
                print("   ", k, len(v))
                t = v[0] if v else {}
                print("    keys", list(t.keys())[:14])
                for f in ("block","layer","pooling","bank","probe","seed"):
                    if f in t:
                        c = collections.Counter(str(x.get(f)) for x in v)
                        print("    ", f, len(c), sorted(c)[:12])
            elif not isinstance(v, (list, dict)) or (isinstance(v, list) and len(v) < 20):
                print("   ", k, str(v)[:200])
            elif isinstance(v, dict):
                print("   ", k, "dict", list(v.keys())[:8])
