"""P3-B: match the collaborator's stores to the owner's stores BY BANK, per dataset.

his store                     her store                          bank (both)
persona_ood / persona_ood_L40 run-crossgen/banks/paper16_shared  26-persona shared-question bank
persona_ood_ext_L*            run-crossgen/kd11/banks/kd         36-persona kd bank, score-3 filter
persona_ood_mod_L*            run-crossgen/kd11/banks/kd_s23     36-persona kd bank, score-2+3 filter
persona_ood_ext_L* (ext arm)  run-crossgen/kd11/banks/collab     <-- MISMATCH: collab is score-2+3
"""
import csv, sys, statistics as st

sp = sys.argv[1]
MAP = {"convince_me__plain": "convincing-game",
       "harm_refusal_choice__plain": "harm-pressure-choice",
       "harm_refusal_knowledge_report__plain": "harm-pressure-knowledge-report",
       "insider_trading_cadenza__plain": "insider-trading",
       "instruct_dishonesty__plain": "instructed-deception"}

def hers(path, store, pooling="mean"):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["pooling"] == pooling and r["store"] == store:
            out[(int(r["layer"]), r["dataset"], r["direction"])] = float(r["auroc"])
    return out

def his(path, col, layer):
    out = {}
    for r in csv.DictReader(open(path)):
        if r.get(col):
            out[(layer, r["dataset"])] = float(r[col])
    return out

zsm, zsc = f"{sp}/zs_mean.csv", f"{sp}/zs_collab.csv"
zss = f"{sp}/zs_shared.csv"   # lab PC: run-crossgen/banks/paper16_shared, all layers
PAIRS = [
    # label, his rows, her store+direction, layers
    ("26-persona shared bank | 8v7 deception axis",
     {20: (f"{sp}/his_pood_L20_zero_shot.csv", "auroc_axis"),
      40: (f"{sp}/his_pood_L40_zero_shot.csv", "auroc_axis")},
     hers(zss, "banks/paper16_shared"), "contrast"),
    ("36-persona kd bank s3 | 8v7 deception axis",
     {L: (f"{sp}/his_ext_L{L}_zero_shot.csv", "auroc_axis") for L in (20, 40, 44)},
     hers(zsm, "run-crossgen/kd11/banks/kd"), "contrast"),
    ("36-persona kd bank s23 | 8v7 deception axis",
     {L: (f"{sp}/his_mod_L{L}_zero_shot.csv", "auroc_axis") for L in (20, 40, 44)},
     hers(zsm, "run-crossgen/kd11/banks/kd_s23"), "contrast"),
    ("36-persona kd bank s23 | 12v13 extended axis",
     {L: (f"{sp}/his_mod_L{L}_zero_shot.csv", "auroc_ext_axis") for L in (20, 40, 44)},
     hers(zsc, "kd11/banks/collab"), "contrast:collab_extended"),
    ("CROSS-BANK | his s3 12v13 vs her s23 12v13",
     {L: (f"{sp}/his_ext_L{L}_zero_shot.csv", "auroc_ext_axis") for L in (20, 40, 44)},
     hers(zsc, "kd11/banks/collab"), "contrast:collab_extended"),
    ("26-persona ORIGINAL paper16 bank (her headline) vs his shared-bank axis",
     {20: (f"{sp}/his_pood_L20_zero_shot.csv", "auroc_axis"),
      40: (f"{sp}/his_pood_L40_zero_shot.csv", "auroc_axis")},
     hers(zsm, "run-crossgen/kd11"), "contrast"),
]
w = csv.writer(sys.stdout, lineterminator="\n")
w.writerow(["pairing", "layer", "dataset", "his", "hers", "gap"])
for label, hism, hersm, direc in PAIRS:
    gaps = []
    for L, (path, col) in sorted(hism.items()):
        h = his(path, col, L)
        for dh, dn in MAP.items():
            a, b = h.get((L, dh)), hersm.get((L, dn, direc))
            if a is None or b is None:
                continue
            w.writerow([label, L, dn, round(a, 4), round(b, 4), round(a - b, 4)])
            gaps.append(abs(a - b))
    if gaps:
        w.writerow([label, "ALL", f"n={len(gaps)}", "", "",
                    f"mean|gap|={st.mean(gaps):.4f} max={max(gaps):.4f}"])
