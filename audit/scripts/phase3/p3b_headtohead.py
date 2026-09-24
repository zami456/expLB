"""P3-B: owner vs collaborator, per-dataset zero-shot AUROC on the same arms.

Inputs (all produced during this audit):
  <sp>/his_ext_L{20,40,44}_zero_shot.csv   git show origin/feat/persona-pc-ood:.../persona_ood_ext_L*/zero_shot.csv
  <sp>/his_mod_L{20,40,44}_zero_shot.csv   ditto for persona_ood_mod_L*
  <sp>/zs_collab.csv                       her run-crossgen/kd11/banks/collab (lab PC, read-only)
  <sp>/zs_mean.csv                         her local run-crossgen stores
"""
import csv, sys, collections

sp = sys.argv[1]

HIS2HERS = {
    "convince_me__plain": "convincing-game",
    "harm_refusal_choice__plain": "harm-pressure-choice",
    "harm_refusal_knowledge_report__plain": "harm-pressure-knowledge-report",
    "insider_trading_cadenza__plain": "insider-trading",
    "instruct_dishonesty__plain": "instructed-deception",
    "repe_honesty__plain": None,           # not in her eleven
}
# his column  ->  her zero-shot `direction` in the collab bank
ARMS = {
    "auroc_axis": "contrast",                       # 8v7 deception
    "auroc_ext_axis": "contrast:collab_extended",   # 12v13
    "auroc_denial_axis": "contrast:knowledge_denial",  # 4v3 (hers) / 4v6 (his)
    "auroc_pc1": "PC1",
    "auroc_syc_axis": "contrast:sycophancy",
}

def load_his(tag):
    out = {}
    for L in (20, 40, 44):
        try:
            rows = list(csv.DictReader(open(f"{sp}/his_{tag}_L{L}_zero_shot.csv")))
        except FileNotFoundError:
            continue
        for r in rows:
            for col in ARMS:
                if r.get(col):
                    out[(L, r["dataset"], col)] = float(r[col])
    return out

def load_hers(path, store=None, pooling="mean"):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["pooling"] != pooling:
            continue
        if store and r["store"] != store:
            continue
        out[(int(r["layer"]), r["dataset"], r["direction"])] = float(r["auroc"])
    return out

his = load_his("ext")
his_mod = load_his("mod")
hers_collab = load_hers(f"{sp}/zs_collab.csv")
hers_root = load_hers(f"{sp}/zs_mean.csv", store="run-crossgen/kd11")

w = csv.writer(sys.stdout, lineterminator="\n")
w.writerow(["layer", "dataset_his", "dataset_hers", "arm_his", "arm_hers",
            "his_ext", "his_mod", "hers_collab_s23", "gap_ext_vs_hers",
            "gap_mod_vs_hers", "gap_ext_vs_mod"])
for L in (20, 40, 44):
    for dh, dn in HIS2HERS.items():
        if dn is None:
            continue
        for ac, an in ARMS.items():
            a = his.get((L, dh, ac))
            b = his_mod.get((L, dh, ac))
            c = hers_collab.get((L, dn, an))
            if a is None and c is None:
                continue
            f = lambda x: "" if x is None else round(x, 4)
            g = lambda x, y: "" if (x is None or y is None) else round(x - y, 4)
            w.writerow([L, dh, dn, ac, an, f(a), f(b), f(c),
                        g(a, c), g(b, c), g(a, b)])
