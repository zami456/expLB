"""P3-B: the supervised single-source baseline, restricted to the five shared datasets.

His  = persona_ood*/matrix_baseline.csv  (8192-d pooled acts, L2 logistic reg_coeff 10, no intercept)
Hers = run-crossgen/kd11/matrices/mean_L*_s42.json, feature 'raw', each probe recipe.
Off-diagonal mean over the 5 datasets both evaluate, same rows and labels.
"""
import csv, json, sys, statistics as st

sp, repo = sys.argv[1], sys.argv[2]
HIS5 = ["convince_me__plain", "harm_refusal_choice__plain",
        "harm_refusal_knowledge_report__plain", "insider_trading_cadenza__plain",
        "instruct_dishonesty__plain"]
HER5 = ["convincing-game", "harm-pressure-choice", "harm-pressure-knowledge-report",
        "insider-trading", "instructed-deception"]

def his_off(path, keep):
    rows = list(csv.reader(open(path)))
    cols = rows[0][1:]
    m = {r[0]: {c: float(v) for c, v in zip(cols, r[1:])} for r in rows[1:]}
    return st.mean(m[r][c] for r in keep for c in keep if r != c)

def her_off(path, feature, probe, keep):
    d = json.load(open(path))
    ds = d["datasets"]
    idx = [ds.index(x) for x in keep]
    for r in d["results"]:
        if r["feature"] == feature and r["probe"] == probe:
            a = r["auroc"]
            return st.mean(a[i][j] for i in idx for j in idx if i != j)
    return None

print("layer,his_store,his_baseline_5ds,her_probe,her_raw_5ds,gap")
for L, store in ((20, "persona_ood"), (20, "persona_ood_ext_L20"),
                 (40, "persona_ood_L40"), (40, "persona_ood_ext_L40"),
                 (44, "persona_ood_ext_L44")):
    tag = {"persona_ood": "pood_L20", "persona_ood_L40": "pood_L40",
           "persona_ood_ext_L20": "ext_L20", "persona_ood_ext_L40": "ext_L40",
           "persona_ood_ext_L44": "ext_L44"}[store]
    try:
        h = his_off(f"{sp}/his_{tag}_matrix_baseline.csv", HIS5)
    except FileNotFoundError:
        continue
    for probe in ("paper", "lr_reg10", "mms"):
        v = her_off(f"{repo}/run-crossgen/kd11/matrices/mean_L{L}_s42.json", "raw", probe, HER5)
        print(f"{L},{store},{h:.4f},{probe},{v:.4f},{h - v:+.4f}")
