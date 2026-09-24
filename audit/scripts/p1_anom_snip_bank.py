# READ-ONLY snippet for the lab PC (Task B, anomalies). Run with:
#   python audit/scripts/labpc_run.py audit/scripts/p1_fmap_F001-F027_pyc.ps1 "CODE=$(cat audit/scripts/p1_anom_snip_bank.py)"
# (cwd on the lab = C:/Users/user4/expLB; opens files for reading only)
import json
for p in ["run-crossgen/scale_70b/matrices/mean_L40_s42.json","run-crossgen/scale_70b/matrices/mean_L20_s42.json","run-crossgen/scale_3b/matrices/mean_L16_s42.json","run-crossgen/kd11/matrices/mean_L40_s42.json","run-crossgen/matrices/mean_L40_s42.json","run-crossgen/kd11/banks/kd/matrices/mean_L40_s42.json"]:
    d = json.load(open(p, encoding="utf-8"))
    print(p, {k: (v if not isinstance(v, (dict, list)) else (type(v).__name__ + str(len(v)))) for k, v in d.items() if k not in ("results",)})
