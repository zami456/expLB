# READ-ONLY snippet for the lab PC (Task B): value distribution of own100 score files that are byte-identical.
# Run: python audit/scripts/labpc_run.py audit/scripts/p1_fmap_F001-F027_pyc.ps1 "CODE=$(cat audit/scripts/p1_anom_snip_scores.py)"
import json, collections
for r in ["bard", "void", "whale", "default"]:
    try:
        d = json.load(open("run/scores/%s.json" % r, encoding="utf-8"))
    except Exception as e:
        print(r, "ERR", e); continue
    print(r, len(d), dict(collections.Counter(d.values())), list(d)[:2], list(d)[-2:])
