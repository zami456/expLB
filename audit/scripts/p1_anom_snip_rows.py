# READ-ONLY snippet for the lab PC (Task B, anomalies). Run with:
#   python audit/scripts/labpc_run.py audit/scripts/p1_fmap_F001-F027_pyc.ps1 "CODE=$(cat audit/scripts/p1_anom_snip_rows.py)"
# (cwd on the lab = C:/Users/user4/expLB; opens files for reading only)
import json, collections
for p in ["run-coords/passB/harm-pressure-knowledge-report/rows.jsonl","run-coords/passA/harm-pressure-knowledge-report/rows.jsonl","run-coords/passB_fine/harm-pressure-knowledge-report/rows.jsonl","run-coords/passP/harm-pressure-knowledge-report/rows.jsonl","run-coords/passB/harm-pressure-kr-matched/rows.jsonl","run-coords/passA/harm-pressure-kr-matched/rows.jsonl"]:
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    keys = sorted(set().union(*[r.keys() for r in rows]))
    lab = collections.Counter(str(r.get("label", r.get("deceptive"))) for r in rows)
    ids = [str(r.get("id", r.get("row_id", r.get("idx")))) for r in rows]
    print(p, "n=", len(rows), "keys=", keys[:12], "labels=", dict(lab), "first_ids=", ids[:3])
