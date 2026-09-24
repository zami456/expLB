# READ-ONLY snippet for the lab PC (Task B, anomalies). Run with:
#   python audit/scripts/labpc_run.py audit/scripts/p1_fmap_F001-F027_pyc.ps1 "CODE=$(cat audit/scripts/p1_anom_snip_feat.py)"
# (cwd on the lab = C:/Users/user4/expLB; opens files for reading only)
import json
for p in ["run-crossgen/scale_70b/matrices/mean_L40_s42.json","run-crossgen/scale_70b/matrices/mean_L20_s42.json","run-crossgen/scale_3b/matrices/mean_L16_s42.json","run-crossgen/kd11/banks/kd/matrices/mean_L40_s42.json","run-crossgen/kd11/matrices/mean_L40_s42.json"]:
    d = json.load(open(p, encoding="utf-8"))
    r = d.get("results")
    feats = sorted(r.keys()) if isinstance(r, dict) else (sorted({x.get("feature") for x in r}) if isinstance(r, list) else None)
    print(p)
    print("   zeroshot:", [ (z.get("axis") or z.get("feature") or str(z)[:60]) if isinstance(z, dict) else str(z)[:60] for z in d.get("zeroshot", [])][:10])
    print("   features:", feats[:40] if feats else type(r).__name__)
