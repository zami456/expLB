"""
Phase 2, batch P8-draft-rest: writes audit/phase2/claims_batch_P8-draft-rest.csv
and audit/claims/<id>.md from the two check scripts' output.

The verdicts and the prose are the auditor's; the numbers come from
  audit/phase2/out/P8_level1.txt      (P8_draft_rest.py)
  audit/phase2/out/P8_appG_floor.txt  (P8_appG_and_floor.py)
and, for the claims about other papers, from the PDFs in papers/ read during
this audit (page numbers given per row).

Usage: PYTHONIOENCODING=utf-8 python audit/scripts/claims/P8_write_rows.py
"""
import csv
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT_CSV = os.path.join(REPO, "audit", "phase2", "claims_batch_P8-draft-rest.csv")
NOTES = os.path.join(REPO, "audit", "claims")
L1 = os.path.join(REPO, "audit", "phase2", "out", "P8_level1.txt")
L2 = os.path.join(REPO, "audit", "phase2", "out", "P8_appG_floor.txt")

COLS = ["id", "claim_plain", "stated_numbers", "source_ref", "evidence_file", "evidence_key",
        "level1_value", "level1_match", "level2_value", "level2_method", "version_used",
        "robustness", "verdict", "confidence", "importance", "script", "notes"]

# --- slice the level-1 output into per-claim blocks -------------------------
blocks = {}
cur = None
for line in open(L1, encoding="utf-8").read().splitlines():
    m = re.match(r"^(D-\d{3})\s+(.*)$", line)
    if m:
        cur = m.group(1)
        blocks[cur] = []
    elif cur and not line.startswith("====="):
        blocks[cur].append(line)
for k in blocks:
    blocks[k] = "\n".join(blocks[k]).strip("\n")
APPG = open(L2, encoding="utf-8").read()

PCOOD = "origin/feat/persona-pc-ood (read with `git show`; never checked out)"
VER = ("his branch @ tip; his own committed CSVs; NOT touched by the 2026-09-22 re-run wave "
       "(RF-13 hit the OWNER's lab stores). Persona bank = the owner's kd bank, byte-identical "
       "to lab run-kd/vectors_s3 (RF-1 RESOLVED). Activations gone -> Level 2 on his pipeline impossible.")
SCRIPT1 = "audit/scripts/claims/P8_draft_rest.py"
SCRIPT2 = "audit/scripts/claims/P8_appG_and_floor.py"
L1ONLY = "MATCHES HIS DATA (Level 1 only)"
NOLVL2 = ("Level 2 on his pipeline NOT POSSIBLE (his activations are gone; owner's rule). "
          "Recomputed where his own cell-level CSVs allow it.")

R = []  # rows


def row(**kw):
    d = {c: "" for c in COLS}
    d.update(kw)
    R.append(d)


P = "draft main (1).pdf"

# ---------------------------------------------------------------- section 5.1
row(id="D-081",
    claim_plain="On HP-KR at layer 20 the RepE-trained base probe also inverts, AUROC 0.109; so does Liars' Bench's own Mean Probe at 0.12.",
    stated_numbers="0.109; 0.12", source_ref=f"{P} p.7 sec 5.1 + Table 2",
    evidence_file="persona_ood_ext_L20/zero_shot_operating.csv; papers/2511.16035v2 (1).pdf Table 6 p.54",
    evidence_key="arm='base probe (RepE-trained)', threshold_source='control', dataset=harm_refusal_knowledge_report__plain, col auroc | LB Table 6 AUROC row, Llama 3.3 70B, HP-KR",
    level1_value="0.109 (his file); 0.12 (LB PDF)", level1_match="YES both (tol 5e-4)",
    level2_value=NOLVL2, level2_method="n/a",
    version_used=VER,
    robustness="The same base probe at L40 reads 0.9358 and at L44 0.8281 on HP-KR: the inversion is layer-20-only (see D-124).",
    verdict=L1ONLY + "; the Liars' Bench 0.12 VERIFIED in the PDF",
    confidence="HIGH - exact match in his file and in the published table",
    importance="HIGH - the 'it is not only the persona contrast' half of the HP-KR story",
    script=SCRIPT1,
    notes="LB Table 6 (Mean Probe), Llama-3.3-70B row, HP-KR AUROC = 0.12. Draft reproduces it exactly.")

row(id="D-084",
    claim_plain="The extended (12v13) contrast keeps nearly all of the HP-KR repair: zero-shot AUROC 0.882 / 0.912 / 0.894 at layers 20 / 40 / 44, response-mean pooling, no seed.",
    stated_numbers="0.882; 0.912; 0.894", source_ref=f"{P} p.7 sec 5.1 + Table 2 + Table 6",
    evidence_file="persona_ood_ext_L{20,40,44}/zero_shot.csv",
    evidence_key="row harm_refusal_knowledge_report__plain, col auroc_ext_axis",
    level1_value="0.881568 / 0.912392 / 0.893637", level1_match="YES (tol 5e-4)",
    level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="The 4v6 knowledge-denial contrast is slightly higher at all three layers (0.890/0.919/0.908, D-083, P5). The 'mod' bank (more rollouts, persona_ood_mod_L40) gives 0.9066 not 0.9158 off-diagonal - the draft reports the ext bank only.",
    verdict=L1ONLY, confidence="HIGH - three exact matches", importance="HIGH - Table 2, Figure 3, the repair claim",
    script=SCRIPT1, notes="")

row(id="D-085",
    claim_plain="The HP-KR repair comes from the new personas, not from re-extracting the old ones: the published 8v7 grouping inverts on the original bank (0.103 at L20) and still inverts on the re-extracted bank (0.148).",
    stated_numbers="0.103; 0.148", source_ref=f"{P} p.7 sec 5.1",
    evidence_file="persona_ood/zero_shot.csv; persona_ood_ext_L20/zero_shot.csv",
    evidence_key="row harm_refusal_knowledge_report__plain, col auroc_axis",
    level1_value="0.102856 (original bank); 0.148277 (re-extracted)", level1_match="YES (tol 5e-4)",
    level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="This is a two-point comparison at one layer. The stronger version of the same argument (0.103 vs 0.148 at L20 only) is not repeated at L40/L44, where the published contrast is near chance rather than inverted, so the 'still inverts' wording is layer-20-specific.",
    verdict=L1ONLY, confidence="HIGH - both numbers exact",
    importance="HIGH - it is the causal attribution behind contribution 2",
    script=SCRIPT1,
    notes="The causal reading ('comes from the new personas') is an inference from two banks, not an experiment that varies one factor; the two banks differ in BOTH rollout count and persona membership. See also persona_ood_mod_* which varies rollout count alone.")

row(id="D-094",
    claim_plain="At layer 40 the base probe ranks HP-KR well (AUROC 0.936) yet flags nothing at the Alpaca threshold (balanced accuracy 0.500); the published and extended contrasts also flag nothing.",
    stated_numbers="0.936; 0.500", source_ref=f"{P} p.8 sec 5.1 + Table 2",
    evidence_file="persona_ood_ext_L40/zero_shot_operating.csv",
    evidence_key="threshold_source='control', dataset=harm_refusal_knowledge_report__plain; arms 'base probe (RepE-trained)', 'deception axis (8v7)', 'extended axis (12v13)'",
    level1_value="auroc 0.9358; balacc 0.5 / 0.5 / 0.5 with recall 0.0 and specificity 1.0 in all three",
    level1_match="YES (0.9358 rounds to 0.936; balacc exact)",
    level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="recall=0 and specificity=1 in all three arms, so the 0.500 is the metric's floor, not a measurement.",
    verdict=L1ONLY, confidence="HIGH", importance="HIGH - the AUROC-vs-threshold gap is a main message",
    script=SCRIPT1, notes="")

row(id="D-095",
    claim_plain="The benchmark threshold is unreliable even for the knowledge-denial contrast: at layer 20 on HP-KR it flags 88% of honest answers and balanced accuracy falls to 0.533.",
    stated_numbers="88%; 0.533", source_ref=f"{P} p.8 sec 5.1",
    evidence_file="persona_ood_ext_L20/zero_shot_operating.csv",
    evidence_key="arm='knowledge-denial axis (4v6)', control, HP-KR: specificity, balacc",
    level1_value="specificity 0.1165 -> 88.35% honest flagged; balacc 0.5331; recall 0.9496",
    level1_match="YES (tol 0.005 on the percentage; balacc exact)",
    level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="At L40 the same arm has specificity 0.94 and balacc 0.910 (D-093, P5): the failure is layer-20-specific.",
    verdict=L1ONLY, confidence="HIGH", importance="MED - a limitation the draft states against itself",
    script=SCRIPT1, notes="")

# ---------------------------------------------------------------- section 5.2
row(id="D-096",
    claim_plain="Probes on raw 8192-d activations reach mean in-distribution AUROC 0.983 at layer 40 but lose 0.251 off the diagonal, falling to 0.732.",
    stated_numbers="0.983; 0.251; 0.732", source_ref=f"{P} p.8 sec 5.2 + Table 3",
    evidence_file="persona_ood_ext_L40/summary.csv; matrix_baseline.csv",
    evidence_key="features=baseline: diagonal_mean, off_diagonal_mean",
    level1_value="diagonal 0.983273, off-diagonal 0.732033, difference 0.251240", level1_match="YES (tol 5e-4)",
    level2_value="RECOMPUTED from his matrix_baseline.csv (rounded cells): diagonal 0.9833, off-diagonal 0.7320 - agrees",
    level2_method="means over the 6 diagonal and 30 off-diagonal cells of his own matrix CSV",
    version_used=VER,
    robustness="Off-diagonal baseline is 0.648 at L20 and 0.716 at L44 (D-098, P5), so 0.732 is the best of the three layers for the raw probe.",
    verdict=L1ONLY + " (aggregates re-derived from his cell-level CSV)",
    confidence="HIGH - three exact matches plus an independent re-aggregation",
    importance="HIGH - the baseline the headline transfer gain is measured against",
    script=SCRIPT1, notes="")

row(id="D-097",
    claim_plain="The worst cell of the layer-40 raw-probe transfer matrix is a probe trained on Convincing-Game and tested on HP-KR, AUROC 0.06.",
    stated_numbers="0.06", source_ref=f"{P} p.8 sec 5.2 + Fig 4 left",
    evidence_file="persona_ood_ext_L40/matrix_baseline.csv",
    evidence_key="row (training set) convince_me__plain, column (test set) harm_refusal_knowledge_report__plain",
    level1_value="0.0598", level1_match="YES (rounds to 0.06)",
    level2_value="RECOMPUTED: it is indeed the minimum of the 30 off-diagonal cells; next lowest are RepE->IT 0.2735 and CG->HP-C 0.4282",
    level2_method="sort of all 30 off-diagonal cells of his matrix", version_used=VER,
    robustness="Matrix orientation checked: rows are the TRAINING set, columns the TEST set (the 1-d contrast matrices are constant down a column). Fig 4's axes are transposed relative to the CSV; the quoted cell is the same one either way.",
    verdict=L1ONLY, confidence="HIGH", importance="MED - a vivid illustration, not a headline",
    script=SCRIPT1, notes="")

row(id="D-101",
    claim_plain="At layer 20 with HP-KR removed, the published 8v7 contrast is marginally ahead of the extended 12v13 one (0.919 vs 0.909 mean off-diagonal AUROC), so at that layer the whole gain is the HP-KR repair.",
    stated_numbers="0.919; 0.909", source_ref=f"{P} p.8 sec 5.2 + p.12 sec 6 + Table 3",
    evidence_file="persona_ood_ext_L20/summary.csv",
    evidence_key="off_diagonal_excl_hpkr for features axis1 and ext_axis1",
    level1_value="axis1 0.919401; ext_axis1 0.908627", level1_match="YES (tol 5e-4)",
    level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="At L40 and L44 the ordering reverses (D-102). The gap at L20 is 0.0108 with no interval reported in the draft; the store has no bootstrap for the excl-HP-KR aggregate.",
    verdict=L1ONLY, confidence="HIGH - exact match",
    importance="HIGH - this is the draft's own strongest caveat, restated in section 6 (D-156)",
    script=SCRIPT1,
    notes="D-156 (section 6 Limitations) restates the same fact and is covered by this row.")

row(id="D-102",
    claim_plain="At layers 40 and 44 the extended contrast leads with or without HP-KR; at layer 40 excluding HP-KR it is 0.917 against the published contrast's 0.880.",
    stated_numbers="0.917; 0.880", source_ref=f"{P} p.8 sec 5.2 + Table 3",
    evidence_file="persona_ood_ext_L40/summary.csv; persona_ood_ext_L44/summary.csv",
    evidence_key="off_diagonal_excl_hpkr, features ext_axis1 and axis1",
    level1_value="L40 ext 0.916524 vs axis1 0.879757; L44 ext 0.9053 vs axis1 0.8734; all-six L40 0.9158 vs 0.8319, L44 0.9034 vs 0.8260",
    level1_match="YES (tol 5e-4)", level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="Checked at both layers and in both dataset sets: the 'leads with or without HP-KR' part is true at L40 and L44 in all four aggregates.",
    verdict=L1ONLY, confidence="HIGH", importance="HIGH - the answer to the D-101 caveat",
    script=SCRIPT1, notes="")

row(id="D-103",
    claim_plain="The knowledge-denial (4v6) contrast alone transfers nearly as well as the extended one: 0.912 mean off-diagonal AUROC at layer 40 over all six test sets.",
    stated_numbers="0.912", source_ref=f"{P} p.8 sec 5.2 + Table 3",
    evidence_file="persona_ood_ext_L40/summary.csv", evidence_key="features=denial_axis1, off_diagonal_mean",
    level1_value="0.911535", level1_match="YES (tol 5e-4)", level2_value=NOLVL2, level2_method="n/a",
    version_used=VER,
    robustness="Extended is 0.9158 at the same layer, so 'nearly as well' means a 0.004 gap. Excluding HP-KR the denial contrast is 0.9101 vs extended 0.9165.",
    verdict=L1ONLY, confidence="HIGH", importance="MED", script=SCRIPT1, notes="")

# ---------------------------------------------------------------- section 5.3
row(id="D-105",
    claim_plain="A probe on the top three components of the extended bank reaches 0.684 / 0.866 / 0.880 mean off-diagonal AUROC against 0.904 / 0.916 / 0.903 for the 1-d contrast; the gap narrows from 0.220 to 0.050 to 0.023 and never reverses.",
    stated_numbers="0.684/0.866/0.880; 0.904/0.916/0.903; 0.220/0.050/0.023",
    source_ref=f"{P} p.9 sec 5.3 + Table 3",
    evidence_file="persona_ood_ext_L{20,40,44}/summary.csv",
    evidence_key="off_diagonal_mean for features ext_pc3 and ext_axis1",
    level1_value="pc3 0.683697 / 0.865534 / 0.879936; axis1 0.904117 / 0.915835 / 0.903388; gaps 0.220420 / 0.050301 / 0.023453",
    level1_match="YES (tol 5e-4; gaps tol 1.5e-3)",
    level2_value="RECOMPUTED: the three gaps are exactly axis1 - pc3 from his own summary rows",
    level2_method="arithmetic on his summary.csv", version_used=VER,
    robustness="Only three layers exist, so 'narrows with depth' rests on three points. No seed or spread is stored for the PCA features.",
    verdict=L1ONLY, confidence="HIGH - nine exact matches",
    importance="HIGH - contribution 4 / Table 3 (P5 checked the 0.220 and 0.023 endpoints as D-028)",
    script=SCRIPT1, notes="")

row(id="D-106",
    claim_plain="The extended bank's first component alone is far behind (0.768 off-diagonal at layer 40), and the published bank's three components fall with depth (0.679, 0.639, 0.617).",
    stated_numbers="0.768; 0.679/0.639/0.617", source_ref=f"{P} p.9 sec 5.3 + Table 3",
    evidence_file="persona_ood_ext_L{20,40,44}/summary.csv",
    evidence_key="off_diagonal_mean, features ext_pc1 (L40) and pc3",
    level1_value="ext_pc1 L40 0.767950; pc3 0.679191 / 0.639362 / 0.616589", level1_match="YES (tol 5e-4)",
    level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="ext_pc1 at L20/L44 is 0.661/0.780 (Table 3), so 'far behind' holds at every layer.",
    verdict=L1ONLY, confidence="HIGH", importance="MED", script=SCRIPT1, notes="")

row(id="D-107",
    claim_plain="Three random directions used as probe features score 0.527, 0.504 and 0.493 mean off-diagonal AUROC at layers 20, 40 and 44, so the persona subspace is not empty of signal.",
    stated_numbers="0.527; 0.504; 0.493", source_ref=f"{P} p.9 sec 5.3 + Table 3",
    evidence_file="persona_ood_ext_L{20,40,44}/summary.csv", evidence_key="features=random3, off_diagonal_mean",
    level1_value="0.526938 / 0.503798 / 0.492709", level1_match="YES (tol 5e-4)",
    level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="WEAK AS A NULL: his store keeps a single random draw (persona_ood.py builds it from one QR of a standard-normal matrix). No seed column, no repeats, no spread - so this cannot support a statement about what a random direction typically does. The owner's own work has a many-draw random null (memory note 'significance and random nulls'); the draft does not use it.",
    verdict=L1ONLY + "; the control itself is UNDERPOWERED (n=1 draw)",
    confidence="HIGH for the number, LOW for it being a null",
    importance="MED - it is the only control in Table 3", script=SCRIPT1,
    notes="Flag for the lead: a one-draw random baseline is quoted in Table 3 as if it were a null distribution.")

# ---------------------------------------------------------------- section 5.4
row(id="D-118",
    claim_plain="Averaged over the five Liars' Bench datasets at layer 40, the knowledge-denial (4v6) contrast is the best detector at the 1%-FPR threshold: 0.725 balanced accuracy and 0.487 recall.",
    stated_numbers="0.725; 0.487", source_ref=f"{P} p.9 sec 5.4 + Table 4",
    evidence_file="persona_ood_ext_L40/zero_shot_operating.csv",
    evidence_key="arm='knowledge-denial axis (4v6)', control, dataset='Average (Liars' Bench)'",
    level1_value="balacc 0.7251, recall 0.4868", level1_match="YES (tol 5e-4)",
    level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="All four arms at L40: base probe 0.6255/0.2814, 8v7 0.675/0.3819, 12v13 0.6939/0.4129, 4v6 0.7251/0.4868. So 'best' holds on both metrics at L40. At L20 the base probe is ahead (D-120).",
    verdict=L1ONLY, confidence="HIGH", importance="HIGH - Table 4, the headline detection claim",
    script=SCRIPT1, notes="")

row(id="D-120",
    claim_plain="Layer 20 is the exception: there the base probe leads the extended contrast on balanced accuracy (0.721 vs 0.591) and recall (0.489 vs 0.187).",
    stated_numbers="0.721 vs 0.591; 0.489 vs 0.187", source_ref=f"{P} p.9 sec 5.4 + Table 4",
    evidence_file="persona_ood_ext_L20/zero_shot_operating.csv",
    evidence_key="control, 'Average (Liars' Bench)', arms base probe and extended axis",
    level1_value="base 0.7209 / 0.4893; extended 0.5914 / 0.1872", level1_match="YES (tol 5e-4)",
    level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="On AUROC at the same layer the ordering is the other way (base 0.6972, extended 0.8868), which the draft says next.",
    verdict=L1ONLY, confidence="HIGH", importance="MED - an honest negative the draft reports",
    script=SCRIPT1, notes="")

row(id="D-121",
    claim_plain="On Convincing-Game at layer 20 the base probe's recall is 0.677 against 0.174 for the extended contrast, and neither flags an honest response.",
    stated_numbers="0.677; 0.174", source_ref=f"{P} p.10 sec 5.4",
    evidence_file="persona_ood_ext_L20/zero_shot_operating.csv",
    evidence_key="control, convince_me__plain, arms base probe and extended axis: recall, specificity",
    level1_value="recall 0.6766 and 0.1737; specificity 1.0 for both", level1_match="YES (tol 5e-4)",
    level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="'neither flags an honest response' is exact: specificity = 1.0 for both arms.",
    verdict=L1ONLY, confidence="HIGH", importance="MED", script=SCRIPT1, notes="")

row(id="D-122",
    claim_plain="On Insider-Trading and Instructed-Deception at layer 20 the Alpaca threshold lets the base probe flag 13% and 11% of honest responses against 0% and 2% for the extended contrast, which ranks both at least as well (AUROC 0.990 vs 0.973 on IT, 0.954 vs 0.950 on ID).",
    stated_numbers="13%; 11%; 0%; 2%; 0.990/0.973; 0.954/0.950", source_ref=f"{P} p.10 sec 5.4",
    evidence_file="persona_ood_ext_L20/zero_shot_operating.csv",
    evidence_key="control, IT and ID rows, arms base probe and extended axis: specificity, auroc",
    level1_value="IT: 1-spec 0.1309 (base) vs 0.0 (ext); AUROC 0.9726 vs 0.9899. ID: 1-spec 0.1064 vs 0.0222; AUROC 0.9498 vs 0.9539",
    level1_match="YES (percentages tol 0.005; AUROCs tol 5e-4)",
    level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="All eight quoted quantities check out. 11% is 10.64% rounded up; 2% is 2.22% rounded down.",
    verdict=L1ONLY, confidence="HIGH", importance="MED", script=SCRIPT1, notes="")

row(id="D-124",
    claim_plain="The base probe inverts on one dataset at each depth: HP-KR at layer 20 (AUROC 0.109), and Insider-Trading at layers 40 and 44 (0.338 and 0.289).",
    stated_numbers="0.109; 0.338; 0.289", source_ref=f"{P} p.10 sec 5.4",
    evidence_file="persona_ood_ext_L{20,40,44}/zero_shot_operating.csv",
    evidence_key="arm='base probe (RepE-trained)', control, per-dataset auroc",
    level1_value="0.109 / 0.3378 / 0.2888", level1_match="YES (tol 5e-4) for the three quoted numbers",
    level2_value="RECOMPUTED sweep of all six datasets per layer: at L20 the base probe is below 0.5 on TWO datasets, HP-KR 0.109 AND HP-C 0.4708; at L40 and L44 only IT is below 0.5.",
    level2_method="scan of his zero_shot_operating.csv for auroc < 0.5", version_used=VER,
    robustness="The under-statement is small (HP-C 0.4708 is barely inverted) but the sentence 'one dataset at each depth' is not literally true at layer 20.",
    verdict="PARTLY - the three numbers match his data; the count 'one dataset' does not hold at layer 20 (two datasets sit below 0.5)",
    confidence="HIGH - the full per-layer scan is in the script output",
    importance="MED - a wording fix, not a result change", script=SCRIPT1,
    notes="Proposed correction for DOC_FIXES: 'at layer 20 HP-KR (0.109), and marginally HP-C (0.471)'.")

row(id="D-125",
    claim_plain="On Harm-Pressure Choice the persona contrasts still rank lies above honest answers (extended contrast AUROC 0.712 at layer 40) but almost nothing crosses the Alpaca threshold: only the knowledge-denial contrast flags any HP-C lie, with recall 0.015.",
    stated_numbers="0.712; 0.015", source_ref=f"{P} p.10 sec 5.4",
    evidence_file="persona_ood_ext_L40/zero_shot_operating.csv",
    evidence_key="control, harm_refusal_choice__plain, all five arms",
    level1_value="extended auroc 0.7116; knowledge-denial recall 0.0149 (balacc 0.5075); every other arm recall 0.0",
    level1_match="YES (tol 5e-4)", level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="Verified across all five arms at L40, including the sycophancy axis (recall 0.0): 'only the knowledge-denial contrast flags any HP-C lie' is exact.",
    verdict=L1ONLY, confidence="HIGH", importance="MED - the calibration limitation",
    script=SCRIPT1, notes="")

row(id="D-127",
    claim_plain="The base probe's balanced accuracy peaks at layer 20: 0.721, 0.626, 0.626 at layers 20, 40, 44 (five-dataset mean at the 1%-FPR threshold).",
    stated_numbers="0.721/0.626/0.626", source_ref=f"{P} p.10 sec 5.5 + Table 4",
    evidence_file="persona_ood_ext_L{20,40,44}/zero_shot_operating.csv",
    evidence_key="arm='base probe (RepE-trained)', control, 'Average (Liars' Bench)', col balacc",
    level1_value="0.7209 / 0.6255 / 0.6256", level1_match="YES under round-half-up (0.6255 -> 0.626); NOTE 0.6255 rounds to 0.625 under banker's rounding, so the L40 digit is on the boundary",
    level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="The L40 and L44 values differ in the 4th decimal (0.6255 vs 0.6256) and the draft prints both as 0.626; that is defensible but hides the ordering.",
    verdict=L1ONLY, confidence="HIGH", importance="LOW", script=SCRIPT1, notes="")

# ---------------------------------------------------------------- section 6
row(id="D-157",
    claim_plain="In the layer-40 transfer matrix, balanced accuracy at the Alpaca threshold sits at its 0.5 floor in 18 of the 30 off-diagonal cells for the raw probes.",
    stated_numbers="18/30", source_ref=f"{P} p.12 sec 6 Calibration",
    evidence_file="persona_ood_ext_L40/summary_balacc.csv; matrix_balacc_baseline.csv; his persona_ood.py:481",
    evidence_key="features=baseline: cells_at_floor / off_diagonal_cells",
    level1_value="18 of 30", level1_match="YES - his file says 18",
    level2_value="RECOMPUTED: only 14 of the 30 cells are EXACTLY 0.5. His counter is cells_at_floor(matrix, floor=0.5, tol=0.005), and the +-0.005 band adds exactly four cells: HP-C->HP-KR 0.4992, HP-KR->ID 0.5011, HP-C->CG 0.503, IT->ID 0.5048.",
    level2_method="tolerance sweep over the 30 off-diagonal cells of his own matrix CSV; definition read from his script",
    version_used=VER,
    robustness="Sweep: tol 0 -> 14, 0.001 -> 15, 0.002 -> 16, 0.005 -> 18, 0.01 -> 18.",
    verdict=L1ONLY + "; but 'sits AT its 0.5 floor' overstates by 4 cells (14 exactly, 18 within +-0.005)",
    confidence="HIGH - the tolerance is hard-coded in his own script and I reproduced the count",
    importance="MED - it is a limitation paragraph, and the honest version is barely weaker",
    script=SCRIPT2,
    notes="Proposed wording for DOC_FIXES: 'within 0.005 of its 0.5 floor in 18 of the 30 cells (exactly 0.5 in 14)'.")

# ---------------------------------------------------------------- appendix C
row(id="D-197",
    claim_plain="Read at the prefilled 'No' of the follow-up protocol (layer 20), the extended contrast falls from 0.990 to 0.173 AUROC on Insider-Trading and from 0.882 to 0.190 on HP-KR.",
    stated_numbers="0.990->0.173; 0.882->0.190", source_ref=f"{P} p.17 App C + Table 8",
    evidence_file="persona_ood_ext_L20/zero_shot.csv; persona_ood_followup_L20/zero_shot.csv",
    evidence_key="col auroc_ext_axis, rows insider_trading_cadenza__plain and harm_refusal_knowledge_report__plain",
    level1_value="IT 0.989925 -> 0.172555; HP-KR 0.881568 -> 0.189906", level1_match="YES (tol 5e-4)",
    level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="Both stores are the same layer (20) and the same datasets; only the read position differs, so the comparison is like for like.",
    verdict=L1ONLY, confidence="HIGH", importance="MED - the appendix that limits the method's scope",
    script=SCRIPT1, notes="")

row(id="D-198",
    claim_plain="The Liars' Bench Follow-Up Probe, re-measured here, scores 0.088 AUROC on HP-KR and flags 88% of HP-KR's honest answers at the Alpaca threshold.",
    stated_numbers="0.088; 88%", source_ref=f"{P} p.17 App C",
    evidence_file="persona_ood_followup_L20/zero_shot_operating.csv",
    evidence_key="arm='base probe (RepE-trained)', control, HP-KR: auroc, specificity",
    level1_value="auroc 0.0876; specificity 0.1195 -> 88.05% flagged; balacc 0.2144; recall 0.3094",
    level1_match="YES (tol 5e-4 / 0.005)",
    level2_value="Cross-checked against the pipeline's own results table for the same run (XXXX-9-llama-all-followup/results_table.csv): auroc 0.08755, balacc 0.21443, recall 0.30935, fpr 0.88050 - identical to 4 d.p.",
    level2_method="cell-by-cell comparison, see D-233", version_used=VER,
    robustness="This is a third probe family inverting on HP-KR (0.088), below the Mean Probe (0.109) and the published contrast (0.148).",
    verdict=L1ONLY + " and cross-checked against his pipeline table",
    confidence="HIGH", importance="MED", script=SCRIPT2, notes="")

# ---------------------------------------------------------------- appendix D
row(id="D-206",
    claim_plain="Reading the extended contrast at its single maximum token beats the response mean on Convincing-Game: 0.993 vs 0.912 AUROC at layer 40, and 0.991 vs 0.859 at layer 44.",
    stated_numbers="0.993/0.912; 0.991/0.859", source_ref=f"{P} p.18 App D + Table 9",
    evidence_file="persona_positions_L40/reads.csv; persona_positions_L44/reads.csv",
    evidence_key="axis=extended, read in {max, mean}, dataset=convince_me__plain, col auroc",
    level1_value="L40 0.9927 / 0.9123; L44 0.9909 / 0.8586", level1_match="YES (tol 5e-4)",
    level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="His read_vs_mean.csv gives the bootstrap delta for the same comparison: CG max-vs-mean +0.0804 [0.0521, 0.1084] at L40, which excludes zero.",
    verdict=L1ONLY, confidence="HIGH", importance="LOW-MED - appendix, but it is the basis of a future-work recommendation",
    script=SCRIPT1, notes="")

row(id="D-207",
    claim_plain="For detection at the 1%-FPR threshold the top-10% token read is best: at layer 40 it reaches 0.737 balanced accuracy and 0.531 recall against 0.694 and 0.413 for the mean (five-dataset average, extended contrast).",
    stated_numbers="0.737; 0.531; 0.694; 0.413", source_ref=f"{P} p.18 App D + Table 9",
    evidence_file="persona_positions_L40/reads_summary.csv",
    evidence_key="axis=extended, read in {top10, mean}: balacc, recall",
    level1_value="top10 0.7372 / 0.5306; mean 0.6939 / 0.4129", level1_match="YES (tol 5e-4)",
    level2_value="RECOMPUTED ordering: at L40 the single MAXIMUM is actually higher than top-10% on both metrics (0.7497 / 0.548), so 'the top-10% read is best' is not true at layer 40 on these two metrics.",
    level2_method="full six-read table printed from his reads_summary.csv", version_used=VER,
    robustness="All six reads at L40 (extended): mean 0.899/0.6939/0.4129, top10 0.9075/0.7372/0.5306, max 0.9095/0.7497/0.548, first 0.8631/0.6059/0.227, last 0.8797/0.6556/0.4221, pre 0.8254/0.6515/0.3829.",
    verdict="PARTLY - all four quoted numbers match his data, but 'the top-10% read is best' is contradicted by the maximum read in the same table at layer 40 (Table 9 shows this too)",
    confidence="HIGH - his own Table 9 carries the contradicting row",
    importance="MED - it feeds the future-work recommendation 'the top tenth of response tokens is a better read than the mean'",
    script=SCRIPT1,
    notes="The draft's own argument for preferring top-10% over max is the RepE instability in D-208, which is a different dataset; on the five-dataset mean at L40 max wins.")

row(id="D-208",
    claim_plain="The single-maximum read is noisier: it lowers RepE's balanced accuracy from 0.998 to 0.638.",
    stated_numbers="0.998 -> 0.638", source_ref=f"{P} p.18 App D",
    evidence_file="persona_positions_L40/reads.csv",
    evidence_key="axis=extended, dataset=repe_honesty__plain, read in {mean, max}, col balacc",
    level1_value="L40: mean 0.9982, max 0.6382", level1_match="YES (tol 5e-4) - and the layer, which the text does not name, is 40",
    level2_value="RECOMPUTED at all three layers: L20 0.9473 -> 0.8509, L40 0.9982 -> 0.6382, L44 0.9982 -> 0.5727",
    level2_method="scan of his reads.csv at the three layers", version_used=VER,
    robustness="The effect is in the same direction at all three layers; the quoted pair is the L40 one. At L44 the drop is larger still (0.998 -> 0.573).",
    verdict=L1ONLY + " (UNDERSPECIFIED: the layer is not stated in the text; it is 40)",
    confidence="HIGH", importance="LOW", script=SCRIPT1, notes="")

# ---------------------------------------------------------------- appendix E
row(id="D-216",
    claim_plain="Trained on RepE at layer 20 and averaged over the five Liars' Bench datasets: on response tokens the 1-d extended contrast (0.887) beats the three extended components (0.736) and the raw 8192-d probe (0.734); at the follow-up 'No' the knowledge-denial components are best (0.699), above the raw probe (0.636) and the contrast (0.552).",
    stated_numbers="0.887; 0.736; 0.734; 0.699; 0.636; 0.552", source_ref=f"{P} p.18 App E + Table 10",
    evidence_file="persona_ood_ext_L20/matrix_*.csv; persona_ood_followup_L20/matrix_*.csv",
    evidence_key="row (training set) repe_honesty__plain, averaged over the five Liars' Bench columns",
    level1_value="response: ext_axis1 0.88678, ext_pc3 0.73642, baseline 0.73442, denial_pc3 0.80980, pc3 0.73266. follow-up: denial_pc3 0.69894, baseline 0.63640, ext_axis1 0.55178, ext_pc3 0.57910, pc3 0.57374",
    level1_match="YES - all ten cells of Table 10 (tol 6e-4)",
    level2_value="RECOMPUTED: every Table 10 cell is the mean of the RepE training row of his own transfer matrix over the five Liars' Bench test columns. I re-derived all ten from the matrix CSVs, not from a summary.",
    level2_method="row-mean over five columns of his matrix_<feature>.csv", version_used=VER,
    robustness="The orientation assumption (rows = training set) is confirmed by these ten independent reproductions.",
    verdict=L1ONLY + " (all ten Table 10 cells re-derived from his cell-level matrices)",
    confidence="HIGH", importance="MED - Appendix E is the reply to Mahadik & Skapars' third method",
    script=SCRIPT1, notes="")

# ---------------------------------------------------------------- appendix F
row(id="D-225",
    claim_plain="The contrast built from deception personas transfers further than the sycophancy (5v5) contrast: mean off-diagonal AUROC 0.832 vs 0.663 at layer 40, and it leads at every layer.",
    stated_numbers="0.832; 0.663", source_ref=f"{P} p.19 App F + Table 11",
    evidence_file="persona_ood_ext_L{20,40,44}/summary.csv",
    evidence_key="off_diagonal_mean, features axis1 and syc_axis1",
    level1_value="L20 0.6511 vs 0.6153; L40 0.8319 vs 0.6626; L44 0.8260 vs 0.7078", level1_match="YES (tol 5e-4)",
    level2_value=NOLVL2, level2_method="n/a", version_used=VER,
    robustness="'leads at every layer' checks out, but the L20 margin is 0.036, an order of magnitude smaller than the L40 margin of 0.169.",
    verdict=L1ONLY, confidence="HIGH",
    importance="MED - it is the draft's own register caveat, and it lines up with the owner's F-13 / cap-bank result",
    script=SCRIPT1,
    notes="The owner's independent finding (memory note 'persona axes encode register'; cap bank 0.374 -> 0.745) is the constructive version of the same observation. Worth connecting in Phase 3.")

# ---------------------------------------------------------------- appendix G
row(id="D-233",
    claim_plain="Appendix G: rescoring the pooled activations with the pipeline's stored RepE probe matched the pipeline's own results table to four decimal places on AUROC, balanced accuracy and recall on all five Liars' Bench datasets, at each layer, and likewise at the follow-up token.",
    stated_numbers="4 decimal places, 5 datasets, 3 layers + follow-up", source_ref=f"{P} p.19 App G",
    evidence_file="XXXX-9-llama-all-followup/results_table.csv vs persona_ood_followup_L20/zero_shot_operating.csv",
    evidence_key="20 cells: 5 datasets x {auroc, balanced_accuracy_at_fpr_1%, recall 1.0%, fpr 1.0%}",
    level1_value="FOLLOW-UP RUN: 20/20 cells identical to 4 d.p. (max |difference| = 0.0).",
    level1_match="YES for the follow-up run; NOT CHECKABLE for the three response-token runs",
    level2_value="The pipeline results tables for the response-token runs are NOT on his branch: XXXX-9-llama-all, XXXX-9-llama-all-L40, XXXX-9-llama-all-L44 and XXXX-9_lr__repe_honesty__... each contain only plots/*.png. Only XXXX-9-llama-all-followup has cfg.yaml + results_table.csv + summary.md.",
    level2_method="directory listing of every XXXX-9* run on his branch + cell-by-cell comparison of the one table that exists",
    version_used=VER,
    robustness="The follow-up half of the claim is fully supported. The layer-by-layer half rests on files that were not committed, so it cannot be confirmed or refuted from his branch.",
    verdict="PARTLY - the follow-up half MATCHES HIS DATA exactly (20/20 cells); the 'for each layer' half is NOT VERIFIABLE (no results table committed for the L20/L40/L44 runs)",
    confidence="HIGH for what is checkable; the rest is a missing-file gap, not a disagreement",
    importance="HIGH - it is the draft's own reproducibility guarantee, and P5 flagged it unchecked",
    script=SCRIPT2,
    notes="Ask the collaborator for the three missing results_table.csv files (or the runs' output dirs). Related: D-237's paper/make_numbers.py does not exist either.")

row(id="D-234",
    claim_plain="Our measurement of the Liars' Bench Follow-Up Probe averages 0.636 AUROC and 0.622 balanced accuracy over the five datasets, against published values of 0.634 and 0.578 (Kretschmar et al. 2025, Table 7).",
    stated_numbers="0.636; 0.622; 0.634; 0.578", source_ref=f"{P} p.19 App G",
    evidence_file="persona_ood_followup_L20/zero_shot_operating.csv; papers/2511.16035v2 (1).pdf Table 7 p.55",
    evidence_key="arm='base probe (RepE-trained)', control, 'Average (Liars' Bench)'; LB Table 7, Llama 3.3 70B row",
    level1_value="his file: auroc 0.636, balacc 0.6223", level1_match="YES (tol 5e-4)",
    level2_value="PUBLISHED VALUES VERIFIED IN THE PDF: Table 7, Llama-3.3-70B, AUROC HP-C 0.50 / HP-KR 0.14 / CG 0.75 / ID 0.93 / IT 0.85 -> five-dataset mean 0.634; balanced accuracy at 1% FPR 0.50 / 0.26 / 0.60 / 0.87 / 0.66 -> mean 0.578. Both reproduce exactly.",
    level2_method="re-averaged the five shared datasets from the published table (the paper's own AVERAGE column is over seven datasets and reads 0.57 / 0.54)",
    version_used=VER + " Published side: papers/2511.16035v2 (1).pdf, printed page 55.",
    robustness="The comparison is not like for like on Insider-Trading (his scores exclude the reasoning scratchpad; see D-064, D-160), which the draft states.",
    verdict="HOLDS - his measured side matches his data and the published side is correct in the PDF",
    confidence="HIGH - both sides checked", importance="HIGH - it is the reproduction argument",
    script=SCRIPT1 + " + PDF read", notes="")

row(id="D-235",
    claim_plain="Our base probe, the Mean Probe recipe at layer 20, averages 0.697 AUROC and 0.721 balanced accuracy over the five datasets, against the published 0.682 and 0.636 (Kretschmar et al. 2025, Table 6).",
    stated_numbers="0.697; 0.721; 0.682; 0.636", source_ref=f"{P} p.19 App G",
    evidence_file="persona_ood_ext_L20/zero_shot_operating.csv; papers/2511.16035v2 (1).pdf Table 6 p.54",
    evidence_key="arm='base probe (RepE-trained)', control, 'Average (Liars' Bench)'; LB Table 6, Llama 3.3 70B row",
    level1_value="his file: auroc 0.6972, balacc 0.7209", level1_match="YES (tol 5e-4)",
    level2_value="PUBLISHED VALUES VERIFIED IN THE PDF: Table 6, Llama-3.3-70B, AUROC 0.41 / 0.12 / 1.00 / 0.95 / 0.93 -> mean 0.682; balanced accuracy at 1% FPR 0.50 / 0.50 / 0.83 / 0.85 / 0.50 -> mean 0.636. Both reproduce exactly.",
    level2_method="re-averaged the five shared datasets from the published table",
    version_used=VER + " Published side: papers/2511.16035v2 (1).pdf, printed page 54.",
    robustness="Insider-Trading is not like for like (D-064). Without IT his probe's balacc mean would fall.",
    verdict="HOLDS - measured side matches his data, published side correct in the PDF",
    confidence="HIGH", importance="HIGH", script=SCRIPT1 + " + PDF read", notes="")

row(id="D-236",
    claim_plain="Nearly all of the balanced-accuracy gap to Liars' Bench is Insider-Trading, where our scores exclude the reasoning scratchpad: our base probe reaches 0.909 there against the published 0.50.",
    stated_numbers="0.909; 0.50", source_ref=f"{P} p.19 App G",
    evidence_file="persona_ood_ext_L20/zero_shot_operating.csv; papers/2511.16035v2 (1).pdf Table 6 p.54",
    evidence_key="base probe, control, insider_trading_cadenza__plain, col balacc; LB Table 6 balanced accuracy at 1% FPR, Llama row, IT",
    level1_value="his file 0.9085 (rounds to 0.909 half-up, 0.908 banker's); published 0.50 VERIFIED",
    level1_match="YES within rounding",
    level2_value="RECOMPUTED the gap: his per-dataset balanced accuracies at L20 are CG 0.8383, HP-C 0.5, HP-KR 0.5, IT 0.9085, ID 0.8578 (mean 0.7209) against published 0.83 / 0.50 / 0.50 / 0.50 / 0.85 (mean 0.636). Per-dataset differences: CG +0.008, HP-C 0.000, HP-KR 0.000, IT +0.409, ID +0.008. So IT is 0.409 of the 0.085 x 5 = 0.425 total, i.e. 96% of the gap - 'nearly all' is exact.",
    level2_method="per-dataset difference decomposition from his file and the published table",
    version_used=VER + " Published side: PDF printed page 54.",
    robustness="LB's own ablation (Table 17, p.70) confirms the mechanism: at layer 20, removing reasoning activations moves the Mean Probe from balacc 0.50 to 0.66 and AUROC 0.93 to 0.97 (see D-064).",
    verdict="HOLDS - both sides checked, and the 'nearly all' decomposition reproduces (96%)",
    confidence="HIGH", importance="HIGH - it is the honesty caveat on the headline reproduction",
    script=SCRIPT1 + " + PDF read", notes="")

row(id="D-237",
    claim_plain="Compute/provenance: persona rollouts, judging and all activation extraction ran on NDIF; every analysis runs on CPU from pooled activations and persona vectors; and paper/make_numbers.py regenerates every number from the released result files.",
    stated_numbers="-", source_ref=f"{P} p.19 App G Compute",
    evidence_file="git ls-tree -r origin/feat/persona-pc-ood; git ls-tree -r HEAD",
    evidence_key="search for 'make_numbers' and for any path starting 'paper/'",
    level1_value="NO SUCH FILE. `paper/make_numbers.py` does not exist on his branch, and there is no `paper/` directory on his branch; it does not exist on asix HEAD either.",
    level1_match="NO",
    level2_value="Related gap (D-233): the pipeline results tables the reproduction claim compares against are missing for three of the four runs.",
    level2_method="git ls-tree on both branches", version_used=PCOOD + " and asix HEAD",
    robustness="Independently re-checked here; P5 reported the same for the abstract/contribution rows.",
    verdict="DOES NOT MATCH HIS DATA - the named script is not in the repository",
    confidence="HIGH - a file either exists in the tree or it does not",
    importance="HIGH - a reproducibility claim in the paper that the repository does not support",
    script="git ls-tree (inline)",
    notes="The NDIF half of the sentence is consistent with the owner's pipeline and was not separately audited here.")

# ---------------------------------------------------------------- other papers
row(id="D-013",
    claim_plain="Every method Liars' Bench evaluates has recall near zero on HP-KR at a 1% false-positive rate, and its probe's AUROC on Llama-3.3-70B is 0.12.",
    stated_numbers="0.12; 'recall near zero'", source_ref=f"{P} p.1 sec 1 Introduction",
    evidence_file="papers/2511.16035v2 (1).pdf, Tables 6-11, printed pages 54-59",
    evidence_key="HP-KR column, Llama 3.3 70B row, RECALL block of each table",
    level1_value="Mean Probe AUROC on HP-KR = 0.12 (Table 6): CORRECT.",
    level1_match="YES for the 0.12",
    level2_value="HP-KR recall at 1% FPR, Llama-3.3-70B, per method: Mean Probe 0.00, Follow-Up Probe 0.37, Self-Evaluation 0.05, LLM-as-a-Judge 0.07, Unrelated-Question Classifier 0.12, Self-Eval (deception prompt) 0.01, Upper-Bound Probe (privileged) 0.97. Across-model averages: 0.00, 0.12, 0.02, 0.08, 0.04, 0.01.",
    level2_method="read every per-method table of the published PDF during this audit",
    version_used="papers/2511.16035v2 (1).pdf (v2), pages 54-59",
    robustness="Whether 'near zero' covers 0.37 is a judgement call; the Follow-Up Probe buys that recall with an 84% false-positive rate on HP-KR, so its balanced accuracy is 0.26.",
    verdict="PARTLY HOLDS - the 0.12 AUROC is exact, but 'every method ... recall near zero' is false for the Follow-Up Probe on Llama-3.3-70B (0.37); it is true of the cross-model averages",
    confidence="HIGH - read in the published tables",
    importance="MED - it is a motivating sentence in the introduction, repeated as D-037 in section 2",
    script="PDF read (pypdf, audit/tools/pylib)",
    notes="Same claim as D-037 (section 2 Related work), which this row also covers. Suggested fix: 'recall at or below 0.12 for every unprivileged method except the Follow-Up Probe, whose 0.37 comes with an 84% false-positive rate'.")

row(id="D-023",
    claim_plain="For Llama-3.3-70B on HP-KR, Liars' Bench reports 0.50 balanced accuracy for its Mean Probe, 0.26 for its Follow-Up Probe, and 0.95 for a probe trained on privileged in-distribution labels.",
    stated_numbers="0.50; 0.26; 0.95", source_ref=f"{P} p.1 Abstract (D-007) and p.2 sec 1 Contributions (D-023) + Table 2",
    evidence_file="papers/2511.16035v2 (1).pdf, Tables 6, 7 and 8 (printed pages 54, 55, 56)",
    evidence_key="BALANCED ACCURACY AT 1% FPR block, Llama 3.3 70B row, HP-KR column (Table 8's block is labelled BALANCED ACCURACY)",
    level1_value="Table 6 -> 0.50; Table 7 -> 0.26; Table 8 (Upper Bound Probe) -> 0.95. All three exact.",
    level1_match="YES", level2_value="Also checked the AUROC side quoted in his Table 2: Mean Probe 0.12, Follow-Up 0.14, Upper-Bound 0.97 - all three exact in the same tables.",
    level2_method="direct read of the published tables",
    version_used="papers/2511.16035v2 (1).pdf (v2)",
    robustness="The Upper-Bound Probe is privileged (trained on Liars' Bench labels including HP-KR), which the draft states.",
    verdict="HOLDS - all six published values quoted from Liars' Bench for HP-KR are correct",
    confidence="HIGH", importance="HIGH - the comparison the abstract and contribution 2 rest on",
    script="PDF read (pypdf, audit/tools/pylib)",
    notes="Covers D-007 (abstract) and D-023 (contributions), which state the same three numbers.")

row(id="D-035",
    claim_plain="Liars' Bench collects 72,863 on-policy lies and honest responses across seven datasets and four models.",
    stated_numbers="72,863; 7; 4", source_ref=f"{P} p.3 sec 2 Related work",
    evidence_file="papers/2511.16035v2 (1).pdf, abstract (page 1) and page 2",
    evidence_key="abstract: 'a testbed consisting of 72,863 examples of lies and honest responses generated by four open-weight models across seven datasets'",
    level1_value="72,863 / four models / seven datasets - exact", level1_match="YES",
    level2_value="n/a (a count stated in the source paper)", level2_method="PDF text search for '72,863'",
    version_used="papers/2511.16035v2 (1).pdf (v2)",
    robustness="Page 2 repeats it as 'a total of 72,863 labeled on-policy' responses, which also supports the draft's 'on-policy' wording.",
    verdict="HOLDS", confidence="HIGH", importance="LOW - background",
    script="PDF read (pypdf, audit/tools/pylib)", notes="")

row(id="D-042",
    claim_plain="Kumar (2026) finds the top principal component alone captures only 0.61-0.80 AUROC, which he takes as a rejection of the single-direction hypothesis.",
    stated_numbers="0.61-0.80", source_ref=f"{P} p.3 sec 2 Related work",
    evidence_file="papers/Pressure-Testing Deception Probes in LLMs Scaling, Robustness, and the.pdf",
    evidence_key="page 1 abstract; page 5 results table (k=1 row: 0.609 0.661 0.804 0.672, min 0.609 max 0.804); page 8 'REJECTED 4/4 models k=1: 0.61-0.80'; page 15",
    level1_value="The paper states '0.61-0.80 AUROC' in four places and its k=1 row runs 0.609-0.804.",
    level1_match="YES",
    level2_value="n/a (a claim about another paper's reported values)",
    level2_method="PDF text search for '0.61', '0.80', 'single-direction'",
    version_used="papers/Pressure-Testing Deception Probes in LLMs ... .pdf (18 pages), arXiv 2605.27958 per audit/papers.csv",
    robustness="The draft's next sentence (D-043) correctly distinguishes Kumar's PC1-of-activations from its own persona-group contrast, so the citation is not being misused.",
    verdict="HOLDS - the range and the 'rejected' framing are the paper's own words",
    confidence="HIGH", importance="MED - it is the citation that could have contradicted contribution 4",
    script="PDF read (pypdf, audit/tools/pylib)",
    notes="This PDF has no notes file in docs/papers/ (audit/papers.csv: 'PDF, NO NOTES').")

row(id="D-064",
    claim_plain="Liars' Bench's own ablation shows that excluding the reasoning scratchpad raises its Mean Probe's insider-trading AUROC from 0.93 to 0.97.",
    stated_numbers="0.93 -> 0.97", source_ref=f"{P} p.6 sec 3.5 Activations",
    evidence_file="papers/2511.16035v2 (1).pdf, Appendix P.1, Table 17, printed page 70",
    evidence_key="Table 17 'Effect of layer selection and reasoning transcript access on mean probe performance', Layer 20 row: AUROC w/ Reas. 0.93, w/o Reas. 0.97",
    level1_value="0.93 -> 0.97 at layer 20 - exact", level1_match="YES",
    level2_value="At layer 22 the same table reads 0.87 -> 0.97. The draft quotes the layer-20 row, which is the layer its own base probe and Liars' Bench's reported Mean Probe use, so the quotation is the right one.",
    level2_method="read Table 17 and the surrounding Appendix P.1 text",
    version_used="papers/2511.16035v2 (1).pdf (v2), printed page 70",
    robustness="The same table shows balanced accuracy moving 0.50 -> 0.66 and recall 0.00 -> 0.89 at layer 20, i.e. the effect on the threshold metrics is much larger than on AUROC - which is exactly the direction of the draft's D-236 gap.",
    verdict="HOLDS", confidence="HIGH",
    importance="HIGH - it is the justification for the draft's non-like-for-like insider-trading caveat",
    script="PDF read (pypdf, audit/tools/pylib)", notes="")

row(id="D-082",
    claim_plain="The published deceptive personas all lie by omission - they 'control timing, emphasis, and selective disclosure', 'bury or exclude material facts' and give 'defensible half-disclosures' (Mahadik & Skapars 2026, Table 2) - and none of them lies by claiming not to know.",
    stated_numbers="3 verbatim quotes", source_ref=f"{P} p.7 sec 5.1",
    evidence_file="papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf, Appendix A 'Deception Axis Provenance', Table 2, PDF page 9 (repeated p.11)",
    evidence_key="Table 2 rows: handler / coverup operator / plausible denier",
    level1_value="Verbatim in the PDF: handler 'controls timing, emphasis, and selective disclosure'; coverup operator 'buries or excludes material facts'; plausible denier 'gives defensible half-disclosures'. All three quotes are exact (the draft changes the verb to the infinitive, 'bury or exclude').",
    level1_match="YES",
    level2_value="I read the WHOLE of Table 2 (8 deceptive personas: handler, broker, impression manager, coverup operator, gatekeeper, spin doctor, plausible denier, narrative manager) and Table 3 (7 honest personas: ombudsman, transparency officer, disclosure counsel, whistleblower, documentarian, fact checker, archivist). Every one of the eight deceptive descriptions is about withholding, sequencing or framing information; none describes claiming not to know. So the draft's 'none of them lies by claiming not to know' is supported by the full list, not only by the three quoted rows. The 8-and-7 counts also confirm the draft's '8 v 7' label.",
    level2_method="full read of Tables 2 and 3 of the published PDF during this audit",
    version_used="papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf (14 pages), arXiv 2605.09391 per audit/papers.csv",
    robustness="This is the mechanism the whole paper rests on: the reason the published contrast has nothing to detect on HP-KR. It is a reading of persona descriptions, not a measurement, but the reading is unambiguous.",
    verdict="HOLDS - the quotes are verbatim and the generalisation is supported by all eight rows of the source table",
    confidence="HIGH - read in the source PDF",
    importance="HIGH - it is the explanation behind contributions 1 and 2",
    script="PDF read (pypdf, audit/tools/pylib)",
    notes="Also confirms the '8 v 7' persona counts used throughout the draft: Table 2 has 8 deceptive personas, Table 3 has 7 honest ones.")

row(id="D-046",
    claim_plain="Mahadik & Skapars publish the roles' names and one-line descriptions but not their prompts, so the collaborator wrote five prompt variants per persona in the style of the Assistant Axis.",
    stated_numbers="5 variants", source_ref=f"{P} p.4 sec 3.2 Persona bank",
    evidence_file="papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf, Tables 2 and 3 (PDF page 9)",
    evidence_key="the persona tables and their surrounding appendix text",
    level1_value="Tables 2 and 3 give only a persona name and a one-line description each (e.g. 'gatekeeper | decides what information passes through'). No system prompts appear there.",
    level1_match="YES for the 'names and one-line descriptions, not prompts' half",
    level2_value="The '5 prompt variants' half is about the OWNER's bank, not this paper: the prompts live in the owner's run-kd / paper16 persona files. RF-1 (Phase 1) established that his vectors are byte-identical to the owner's lab run-kd/vectors_s3 and that the stored system prompts match the instruction text committed at 3dfdb67. The variant COUNT per persona was not re-counted in this batch.",
    level2_method="PDF read; provenance carried over from RF-1 (audit/PROGRESS.md), not re-derived here",
    version_used="papers/... Persona Coordinates.pdf; provenance from RF-1",
    robustness="Deferred: counting the five variants per persona in the owner's persona files, and checking they are the ones the rollouts used.",
    verdict="PARTLY - the statement about the published paper HOLDS; the '5 variants' half is NOT TESTED in this batch",
    confidence="HIGH for the paper half, NOT TESTED for the bank half",
    importance="MED - it is the scope caveat repeated in section 6 (D-159)",
    script="PDF read (pypdf, audit/tools/pylib)",
    notes="For a later batch: count the prompt variants per persona in the owner's kd/paper16 persona files and check the rollout files used them.")

# ---------------------------------------------------------------- write
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
os.makedirs(NOTES, exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    for d in R:
        w.writerow(d)
print("wrote", OUT_CSV, len(R), "rows")

for d in R:
    cid = d["id"]
    out = [f"# {cid} — {d['claim_plain']}", ""]
    out += [f"**Batch:** P8-draft-rest (the collaborator's draft, sections 5, 6 and the appendices, "
            f"plus his claims about other papers).",
            f"**Where stated:** {d['source_ref']}",
            f"**Numbers as stated:** {d['stated_numbers']}", ""]
    out += ["## Checking rule for this claim", ""]
    if cid in ("D-013", "D-023", "D-035", "D-042", "D-064", "D-082", "D-046"):
        out += ["This is a claim about a *published paper*, so it is checked properly: against the "
                "PDF in `papers/`, opened during this audit.", ""]
    else:
        out += ["This number is computed from the collaborator's activations, which are gone "
                "(owner's decision). So only **Level 1** is possible: does the draft's number equal "
                "the number in the data he committed on `origin/feat/persona-pc-ood`? The verdict is "
                "recorded as *matches his data* / *does not match his data*, never as *holds*. "
                "Where his own cell-level CSVs allow an aggregate to be re-derived, I did that too "
                "and say so.", ""]
    out += ["## Evidence", "",
            f"- **File(s):** `{d['evidence_file']}`",
            f"- **Key/row:** {d['evidence_key']}",
            f"- **Version used:** {d['version_used']}", ""]
    out += ["## Level 1 — document vs file", "",
            f"- file value: {d['level1_value']}",
            f"- match: {d['level1_match']}", ""]
    if d["level2_value"]:
        out += ["## Level 2 / re-derivation", "", d["level2_value"], "",
                f"*method:* {d['level2_method']}", ""]
    if d["robustness"]:
        out += ["## Robustness", "", d["robustness"], ""]
    if blocks.get(cid):
        out += ["## Script output (verbatim)", "",
                f"`{SCRIPT1}` — full log at `audit/phase2/out/P8_level1.txt`", "",
                "```", blocks[cid], "```", ""]
    if cid in ("D-233", "D-198", "D-157"):
        out += ["## Script output (verbatim), second script", "",
                f"`{SCRIPT2}` — full log at `audit/phase2/out/P8_appG_floor.txt`", "",
                "```", APPG.strip(), "```", ""]
    out += ["## Verdict", "", f"**{d['verdict']}**", "",
            f"- confidence: {d['confidence']}",
            f"- importance for the paper: {d['importance']}", ""]
    if d["notes"]:
        out += ["## Notes", "", d["notes"], ""]
    with open(os.path.join(NOTES, f"{cid}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
print("wrote", len(R), "notes into", NOTES)
