"""
Phase 2, batch P8-draft-rest.
Level-1 checks of the collaborator's draft (sections 5, 6 and the appendices)
against the result CSVs HE committed on origin/feat/persona-pc-ood.

His activations are gone (owner's decision), so nothing here is a Level-2
recompute of his pipeline.  What IS recomputed: aggregates that his own
cell-level CSVs allow (row/column means of the transfer matrices, floor
counts, dataset means over the five Liars' Bench sets).

Read-only.  Reads a `git archive` export of his branch made into the session
scratchpad; the same bytes are obtainable with
    git show origin/feat/persona-pc-ood:<path>

Usage:  python P8_draft_rest.py <root-of-exported-results-dir>
"""
import csv
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."

LB5 = ["convince_me__plain", "harm_refusal_choice__plain",
       "harm_refusal_knowledge_report__plain", "insider_trading_cadenza__plain",
       "instruct_dishonesty__plain"]
ALL6 = LB5 + ["repe_honesty__plain"]
SHORT = {"convince_me__plain": "CG", "harm_refusal_choice__plain": "HP-C",
         "harm_refusal_knowledge_report__plain": "HP-KR",
         "insider_trading_cadenza__plain": "IT",
         "instruct_dishonesty__plain": "ID", "repe_honesty__plain": "RepE"}

def rows(path):
    with open(os.path.join(ROOT, path), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def matrix(path):
    """matrix_*.csv : index = TRAINING set, columns = TEST set."""
    with open(os.path.join(ROOT, path), newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        hdr = next(r)[1:]
        m = {}
        for line in r:
            m[line[0]] = {h: float(v) if v != "" else None for h, v in zip(hdr, line[1:])}
        return m

def zs(store):
    return {r["dataset"]: r for r in rows(f"{store}/zero_shot.csv")}

def op(store):
    """zero_shot_operating.csv keyed (arm, threshold_source, dataset)."""
    return {(r["arm"], r["threshold_source"], r["dataset"]): r
            for r in rows(f"{store}/zero_shot_operating.csv")}

def summ(store):
    return {r["features"]: r for r in rows(f"{store}/summary.csv")}

def summ_ba(store):
    return {r["features"]: r for r in rows(f"{store}/summary_balacc.csv")}

def hdr(cid, text):
    print("\n" + "=" * 78)
    print(f"{cid}  {text}")
    print("=" * 78)

def cmp(label, stated, found, tol=5e-4):
    ok = "MATCH" if (found is not None and abs(found - stated) <= tol) else "DIFFERS"
    print(f"  {label:<58} draft={stated}  file={found}  -> {ok}")
    return ok

EXT = {20: "persona_ood_ext_L20", 40: "persona_ood_ext_L40", 44: "persona_ood_ext_L44"}
FUP = "persona_ood_followup_L20"
POS = {20: "persona_positions_L20", 40: "persona_positions_L40", 44: "persona_positions_L44"}

# ---------------------------------------------------------------- section 5.1
hdr("D-081", "base probe inverts at L20 (0.109); LB Mean Probe 0.12")
o20 = op(EXT[20])
cmp("base probe (RepE) AUROC on HP-KR, L20",
    0.109, float(o20[("base probe (RepE-trained)", "control",
                      "harm_refusal_knowledge_report__plain")]["auroc"]))
print("  ('Mean Probe 0.12' is a Liars' Bench published number -> checked against the PDF, not here)")

hdr("D-084", "extended contrast (12v13) on HP-KR: 0.882 / 0.912 / 0.894")
for L, want in ((20, 0.882), (40, 0.912), (44, 0.894)):
    cmp(f"zero_shot.csv auroc_ext_axis HP-KR L{L}", want,
        float(zs(EXT[L])["harm_refusal_knowledge_report__plain"]["auroc_ext_axis"]))

hdr("D-085", "published grouping inverts on ORIGINAL bank (0.103) and on re-extracted (0.148)")
cmp("persona_ood/zero_shot.csv auroc_axis HP-KR (original bank, L20)", 0.103,
    float(zs("persona_ood")["harm_refusal_knowledge_report__plain"]["auroc_axis"]))
cmp("persona_ood_ext_L20/zero_shot.csv auroc_axis HP-KR (re-extracted)", 0.148,
    float(zs(EXT[20])["harm_refusal_knowledge_report__plain"]["auroc_axis"]))

hdr("D-094", "base probe L40 HP-KR AUROC 0.936 but balanced accuracy 0.500; so do 8v7 and 12v13")
o40 = op(EXT[40])
cmp("base probe AUROC HP-KR L40", 0.936,
    float(o40[("base probe (RepE-trained)", "control", "harm_refusal_knowledge_report__plain")]["auroc"]))
for arm in ["base probe (RepE-trained)", "deception axis (8v7)", "extended axis (12v13)"]:
    r = o40[(arm, "control", "harm_refusal_knowledge_report__plain")]
    cmp(f"balacc HP-KR L40, {arm}", 0.500, float(r["balacc"]))
    print(f"      recall={r['recall']}  specificity={r['specificity']}")

hdr("D-095", "L20 knowledge-denial threshold flags 88% of honest HP-KR answers; balacc 0.533")
r = o20[("knowledge-denial axis (4v6)", "control", "harm_refusal_knowledge_report__plain")]
spec = float(r["specificity"])
cmp("fraction of honest HP-KR answers flagged (1 - specificity)", 0.88, 1 - spec, tol=0.005)
cmp("balanced accuracy", 0.533, float(r["balacc"]))
print(f"      recall={r['recall']}  specificity={spec}")

# ---------------------------------------------------------------- section 5.2
hdr("D-096", "raw probes: in-distribution 0.983 at L40, off-diagonal 0.732, loss 0.251")
s40 = summ(EXT[40])
diag = float(s40["baseline"]["diagonal_mean"]); off = float(s40["baseline"]["off_diagonal_mean"])
cmp("summary.csv baseline diagonal_mean L40", 0.983, diag)
cmp("summary.csv baseline off_diagonal_mean L40", 0.732, off)
cmp("stated drop 0.251 = diagonal - off-diagonal", 0.251, diag - off, tol=0.0015)
m = matrix(f"{EXT[40]}/matrix_baseline.csv")
rec_diag = sum(m[d][d] for d in ALL6) / 6
rec_off = sum(m[tr][te] for tr in ALL6 for te in ALL6 if tr != te) / 30
print(f"  RECOMPUTED from matrix_baseline.csv: diagonal={rec_diag:.4f}  off-diagonal={rec_off:.4f}")

hdr("D-097", "worst cell: probe trained on CG, tested on HP-KR, scores 0.06 (L40)")
cmp("matrix_baseline L40 [train=CG][test=HP-KR]", 0.06,
    m["convince_me__plain"]["harm_refusal_knowledge_report__plain"], tol=0.005)
cells = sorted(((m[tr][te], SHORT[tr], SHORT[te]) for tr in ALL6 for te in ALL6 if tr != te))
print(f"  lowest five off-diagonal cells: {[(round(v,4), a, b) for v, a, b in cells[:5]]}")

hdr("D-101", "L20 excluding HP-KR: published contrast 0.919 vs extended 0.909")
s20 = summ(EXT[20])
cmp("summary.csv axis1 off_diagonal_excl_hpkr L20 (published 8v7)", 0.919,
    float(s20["axis1"]["off_diagonal_excl_hpkr"]))
cmp("summary.csv ext_axis1 off_diagonal_excl_hpkr L20 (extended 12v13)", 0.909,
    float(s20["ext_axis1"]["off_diagonal_excl_hpkr"]))

hdr("D-102", "L40 and L44 the extended contrast leads with or without HP-KR (0.917 vs 0.880 at L40)")
cmp("ext_axis1 off_diagonal_excl_hpkr L40", 0.917, float(s40["ext_axis1"]["off_diagonal_excl_hpkr"]))
cmp("axis1   off_diagonal_excl_hpkr L40", 0.880, float(s40["axis1"]["off_diagonal_excl_hpkr"]))
s44 = summ(EXT[44])
for lab, s in (("L44", s44),):
    print(f"  {lab}: ext_axis1 excl-HPKR={float(s['ext_axis1']['off_diagonal_excl_hpkr']):.4f} "
          f"vs axis1={float(s['axis1']['off_diagonal_excl_hpkr']):.4f}; "
          f"all-six ext={float(s['ext_axis1']['off_diagonal_mean']):.4f} vs "
          f"axis1={float(s['axis1']['off_diagonal_mean']):.4f}")
print(f"  L40 all six: ext_axis1={float(s40['ext_axis1']['off_diagonal_mean']):.4f} "
      f"vs axis1={float(s40['axis1']['off_diagonal_mean']):.4f}")

hdr("D-103", "knowledge-denial contrast alone: 0.912 off-diagonal at L40")
cmp("summary.csv denial_axis1 off_diagonal_mean L40", 0.912,
    float(s40["denial_axis1"]["off_diagonal_mean"]))

# ---------------------------------------------------------------- section 5.3
hdr("D-105", "extended PCs(k=3) 0.684/0.866/0.880 vs contrast 0.904/0.916/0.903; gap 0.220/0.050/0.023")
gaps = {}
for L, pc, ax in ((20, 0.684, 0.904), (40, 0.866, 0.916), (44, 0.880, 0.903)):
    s = summ(EXT[L])
    p = float(s["ext_pc3"]["off_diagonal_mean"]); a = float(s["ext_axis1"]["off_diagonal_mean"])
    cmp(f"ext_pc3  off_diagonal_mean L{L}", pc, p)
    cmp(f"ext_axis1 off_diagonal_mean L{L}", ax, a)
    gaps[L] = a - p
for L, want in ((20, 0.220), (40, 0.050), (44, 0.023)):
    cmp(f"gap (axis1 - pc3) L{L}", want, gaps[L], tol=0.0015)

hdr("D-106", "extended PC1 0.768 at L40; published PCs(k=3) 0.679 / 0.639 / 0.617")
cmp("ext_pc1 off_diagonal_mean L40", 0.768, float(s40["ext_pc1"]["off_diagonal_mean"]))
for L, want in ((20, 0.679), (40, 0.639), (44, 0.617)):
    cmp(f"pc3 off_diagonal_mean L{L} (published bank)", want,
        float(summ(EXT[L])["pc3"]["off_diagonal_mean"]))

hdr("D-107", "three random directions score 0.527 / 0.504 / 0.493")
for L, want in ((20, 0.527), (40, 0.504), (44, 0.493)):
    cmp(f"random3 off_diagonal_mean L{L}", want,
        float(summ(EXT[L])["random3"]["off_diagonal_mean"]))
print("  NOTE: the store carries ONE random-direction draw; no seed column and no spread is recorded.")

# ---------------------------------------------------------------- section 5.4
hdr("D-118", "knowledge-denial is the best detector at the threshold: 0.725 balacc, 0.487 recall (L40)")
r = o40[("knowledge-denial axis (4v6)", "control", "Average (Liars' Bench)")]
cmp("Average (Liars' Bench) balacc, 4v6, L40", 0.725, float(r["balacc"]))
cmp("Average (Liars' Bench) recall, 4v6, L40", 0.487, float(r["recall"]))
for arm in ["base probe (RepE-trained)", "deception axis (8v7)",
            "extended axis (12v13)", "knowledge-denial axis (4v6)"]:
    a = o40[(arm, "control", "Average (Liars' Bench)")]
    print(f"      {arm:<32} balacc={a['balacc']} recall={a['recall']} auroc={a['auroc']}")

hdr("D-120", "L20: base probe leads on balacc (0.721 vs 0.591) and recall (0.489 vs 0.187)")
b = o20[("base probe (RepE-trained)", "control", "Average (Liars' Bench)")]
e = o20[("extended axis (12v13)", "control", "Average (Liars' Bench)")]
cmp("base probe balacc L20", 0.721, float(b["balacc"]))
cmp("extended  balacc L20", 0.591, float(e["balacc"]))
cmp("base probe recall L20", 0.489, float(b["recall"]))
cmp("extended  recall L20", 0.187, float(e["recall"]))

hdr("D-121", "CG at L20: base probe recall 0.677 vs extended 0.174; neither flags an honest response")
bc = o20[("base probe (RepE-trained)", "control", "convince_me__plain")]
ec = o20[("extended axis (12v13)", "control", "convince_me__plain")]
cmp("base probe recall on CG, L20", 0.677, float(bc["recall"]))
cmp("extended  recall on CG, L20", 0.174, float(ec["recall"]))
print(f"      specificity: base={bc['specificity']}  extended={ec['specificity']}  (1.0 = flags no honest)")

hdr("D-122", "IT/ID at L20: base probe flags 13% and 11% of honest vs 0% and 2%; AUROC 0.990/0.973 IT, 0.954/0.950 ID")
for ds, want_b, want_e, want_ae, want_ab in (
        ("insider_trading_cadenza__plain", 0.13, 0.00, 0.990, 0.973),
        ("instruct_dishonesty__plain", 0.11, 0.02, 0.954, 0.950)):
    bb = o20[("base probe (RepE-trained)", "control", ds)]
    ee = o20[("extended axis (12v13)", "control", ds)]
    cmp(f"{SHORT[ds]}: base-probe false-positive rate (1-specificity)", want_b,
        1 - float(bb["specificity"]), tol=0.005)
    cmp(f"{SHORT[ds]}: extended false-positive rate (1-specificity)", want_e,
        1 - float(ee["specificity"]), tol=0.005)
    cmp(f"{SHORT[ds]}: extended AUROC", want_ae, float(ee["auroc"]))
    cmp(f"{SHORT[ds]}: base-probe AUROC", want_ab, float(bb["auroc"]))

hdr("D-124", "base probe inverts on one dataset per depth: HP-KR L20 0.109, IT L40 0.338, L44 0.289")
cmp("base probe AUROC HP-KR L20", 0.109,
    float(o20[("base probe (RepE-trained)", "control", "harm_refusal_knowledge_report__plain")]["auroc"]))
for L, want in ((40, 0.338), (44, 0.289)):
    o = op(EXT[L])
    cmp(f"base probe AUROC IT L{L}", want,
        float(o[("base probe (RepE-trained)", "control", "insider_trading_cadenza__plain")]["auroc"]))
    below = [(SHORT[d], float(o[("base probe (RepE-trained)", "control", d)]["auroc"]))
             for d in ALL6 if float(o[("base probe (RepE-trained)", "control", d)]["auroc"]) < 0.5]
    print(f"      L{L}: base-probe datasets below 0.5 -> {below}")
below20 = [(SHORT[d], float(o20[("base probe (RepE-trained)", "control", d)]["auroc"]))
           for d in ALL6 if float(o20[("base probe (RepE-trained)", "control", d)]["auroc"]) < 0.5]
print(f"      L20: base-probe datasets below 0.5 -> {below20}")

hdr("D-125", "HP-C: extended contrast AUROC 0.712 at L40; only 4v6 flags any HP-C lie, recall 0.015")
cmp("extended AUROC on HP-C, L40", 0.712,
    float(o40[("extended axis (12v13)", "control", "harm_refusal_choice__plain")]["auroc"]))
for arm in ["base probe (RepE-trained)", "deception axis (8v7)",
            "extended axis (12v13)", "knowledge-denial axis (4v6)", "sycophancy axis (5v5)"]:
    a = o40[(arm, "control", "harm_refusal_choice__plain")]
    print(f"      {arm:<32} recall={a['recall']} balacc={a['balacc']} auroc={a['auroc']}")

hdr("D-127", "base probe balanced accuracy peaks at L20: 0.721 / 0.626 / 0.626")
for L, want in ((20, 0.721), (40, 0.626), (44, 0.626)):
    cmp(f"base probe Average (Liars' Bench) balacc L{L}", want,
        float(op(EXT[L])[("base probe (RepE-trained)", "control", "Average (Liars' Bench)")]["balacc"]))

# ---------------------------------------------------------------- section 6
hdr("D-157", "balanced accuracy sits at its 0.5 floor in 18 of the 30 off-diagonal cells, raw probes, L40")
sb40 = summ_ba(EXT[40])
print(f"  summary_balacc.csv baseline: cells_at_floor={sb40['baseline']['cells_at_floor']} "
      f"of off_diagonal_cells={sb40['baseline']['off_diagonal_cells']}")
mb = matrix(f"{EXT[40]}/matrix_balacc_baseline.csv")
floor = [(SHORT[tr], SHORT[te]) for tr in ALL6 for te in ALL6
         if tr != te and abs(mb[tr][te] - 0.5) < 1e-9]
print(f"  RECOMPUTED from matrix_balacc_baseline.csv: {len(floor)} of 30 cells exactly 0.5")
print(f"  cells: {floor}")

# ---------------------------------------------------------------- appendix C
hdr("D-197", "at the 'No' token IT falls 0.990 -> 0.173 and HP-KR 0.882 -> 0.190 (extended contrast, L20)")
zf = zs(FUP); z20 = zs(EXT[20])
for ds, resp, tok in (("insider_trading_cadenza__plain", 0.990, 0.173),
                      ("harm_refusal_knowledge_report__plain", 0.882, 0.190)):
    cmp(f"{SHORT[ds]} response-token auroc_ext_axis", resp, float(z20[ds]["auroc_ext_axis"]))
    cmp(f"{SHORT[ds]} follow-up-'No' auroc_ext_axis", tok, float(zf[ds]["auroc_ext_axis"]))

hdr("D-198", "Follow-Up Probe scores 0.088 on HP-KR and flags 88% of its honest answers")
of = op(FUP)
r = of[("base probe (RepE-trained)", "control", "harm_refusal_knowledge_report__plain")]
cmp("follow-up store, base probe AUROC on HP-KR", 0.088, float(r["auroc"]))
cmp("fraction of honest HP-KR answers flagged (1 - specificity)", 0.88,
    1 - float(r["specificity"]), tol=0.005)
print(f"      balacc={r['balacc']} recall={r['recall']} specificity={r['specificity']}")

# ---------------------------------------------------------------- appendix D
hdr("D-206", "CG read at its maximum: 0.993 vs 0.912 mean (L40); 0.991 vs 0.859 (L44)")
for L, mx, mn in ((40, 0.993, 0.912), (44, 0.991, 0.859)):
    rr = {(x["axis"], x["read"], x["dataset"]): x for x in rows(f"{POS[L]}/reads.csv")}
    cmp(f"L{L} extended/max/CG auroc", mx,
        float(rr[("extended", "max", "convince_me__plain")]["auroc"]))
    cmp(f"L{L} extended/mean/CG auroc", mn,
        float(rr[("extended", "mean", "convince_me__plain")]["auroc"]))

hdr("D-207", "top-10% read at L40: 0.737 balacc and 0.531 recall vs 0.694 and 0.413 for the mean")
rs40 = {(x["axis"], x["read"]): x for x in rows(f"{POS[40]}/reads_summary.csv")}
cmp("extended/top10 balacc L40", 0.737, float(rs40[("extended", "top10")]["balacc"]))
cmp("extended/top10 recall L40", 0.531, float(rs40[("extended", "top10")]["recall"]))
cmp("extended/mean  balacc L40", 0.694, float(rs40[("extended", "mean")]["balacc"]))
cmp("extended/mean  recall L40", 0.413, float(rs40[("extended", "mean")]["recall"]))
print("  All six reads at L40 (extended):")
for rd in ["mean", "top10", "max", "first", "last", "pre"]:
    x = rs40[("extended", rd)]
    print(f"      {rd:<6} auroc={x['auroc']} balacc={x['balacc']} recall={x['recall']}")

hdr("D-208", "the single maximum lowers RepE's balanced accuracy from 0.998 to 0.638")
found = {}
for L in (20, 40, 44):
    rr = {(x["axis"], x["read"], x["dataset"]): x for x in rows(f"{POS[L]}/reads.csv")}
    k_mean = ("extended", "mean", "repe_honesty__plain")
    k_max = ("extended", "max", "repe_honesty__plain")
    if k_mean in rr:
        found[L] = (float(rr[k_mean]["balacc"]), float(rr[k_max]["balacc"]))
        print(f"  L{L}: extended RepE balacc  mean={found[L][0]}  max={found[L][1]}")
print("  draft says 0.998 -> 0.638; the layer is not named in the text.")

# ---------------------------------------------------------------- appendix E
hdr("D-216", "Table 10 (trained on RepE, L20): response 0.887 / 0.736 / 0.734; follow-up 0.699 / 0.636 / 0.552")
def repe_row_mean(store, feat):
    m = matrix(f"{store}/matrix_{feat}.csv")
    return sum(m["repe_honesty__plain"][d] for d in LB5) / 5
for feat, want, lab in (("ext_axis1", 0.887, "extended contrast (1-d)"),
                        ("ext_pc3", 0.736, "extended PCs (k=3)"),
                        ("baseline", 0.734, "raw activations (8192-d)"),
                        ("denial_pc3", 0.810, "knowledge-denial PCs (k=3)"),
                        ("pc3", 0.733, "published PCs (k=3)")):
    cmp(f"response tokens, {lab}", want, repe_row_mean(EXT[20], feat), tol=0.0006)
for feat, want, lab in (("denial_pc3", 0.699, "knowledge-denial PCs (k=3)"),
                        ("baseline", 0.636, "raw activations (8192-d)"),
                        ("ext_axis1", 0.552, "extended contrast (1-d)"),
                        ("ext_pc3", 0.579, "extended PCs (k=3)"),
                        ("pc3", 0.574, "published PCs (k=3)")):
    cmp(f"follow-up 'No', {lab}", want, repe_row_mean(FUP, feat), tol=0.0006)

# ---------------------------------------------------------------- appendix F
hdr("D-225", "deception contrast off-diagonal 0.832 at L40 vs 0.663 for the sycophancy contrast; leads at every layer")
for L in (20, 40, 44):
    s = summ(EXT[L])
    print(f"  L{L}: axis1(8v7)={float(s['axis1']['off_diagonal_mean']):.4f}  "
          f"syc_axis1(5v5)={float(s['syc_axis1']['off_diagonal_mean']):.4f}")
cmp("axis1 off_diagonal_mean L40", 0.832, float(s40["axis1"]["off_diagonal_mean"]))
cmp("syc_axis1 off_diagonal_mean L40", 0.663, float(s40["syc_axis1"]["off_diagonal_mean"]))

# ---------------------------------------------------------------- appendix G
hdr("D-233", "'AUROC, balanced accuracy and recall matched the pipeline's own results table to four decimal places'")
base = os.path.join(ROOT, "..", "..", "..", "..")  # unused; listing is done below
for d in sorted(os.listdir(ROOT)):
    if d.startswith("XXXX-9"):
        files = []
        for dp, _, fn in os.walk(os.path.join(ROOT, d)):
            for x in fn:
                files.append(os.path.relpath(os.path.join(dp, x), ROOT).replace("\\", "/"))
        print(f"  pipeline run dir {d}: {sorted(files)}")

hdr("D-234", "our Follow-Up Probe measurement: 0.636 AUROC and 0.622 balanced accuracy")
r = of[("base probe (RepE-trained)", "control", "Average (Liars' Bench)")]
cmp("follow-up store, base probe Average (Liars' Bench) auroc", 0.636, float(r["auroc"]))
cmp("follow-up store, base probe Average (Liars' Bench) balacc", 0.622, float(r["balacc"]))
lb = {x["method"]: x for x in rows(f"{FUP}/liars_bench_comparison.csv")}
print("  his recorded published values (liars_bench_comparison.csv):")
for k, v in lb.items():
    print(f"      {k:<34} source={v['source']:<16} balacc={v['balacc_1pct_fpr']} auroc={v['auroc']} n={v['n_datasets']}")

hdr("D-235", "our base probe (Mean Probe recipe, L20): 0.697 AUROC and 0.721 balanced accuracy")
cmp("ext_L20 store, base probe Average (Liars' Bench) auroc", 0.697, float(b["auroc"]))
cmp("ext_L20 store, base probe Average (Liars' Bench) balacc", 0.721, float(b["balacc"]))

hdr("D-236", "on insider trading our base probe reaches 0.909 balanced accuracy")
cmp("base probe balacc on IT, L20", 0.909,
    float(o20[("base probe (RepE-trained)", "control", "insider_trading_cadenza__plain")]["balacc"]))
print("  per-dataset base-probe balanced accuracy at L20:")
for d in LB5:
    x = o20[("base probe (RepE-trained)", "control", d)]
    print(f"      {SHORT[d]:<6} balacc={x['balacc']} auroc={x['auroc']} recall={x['recall']}")

print("\nDONE")
