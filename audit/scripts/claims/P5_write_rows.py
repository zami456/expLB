"""Writes audit/phase2/claims_batch_P5-draft-headlines.csv from the verdicts of
audit/scripts/claims/P5_draft_headlines.py (whose full output is saved at
audit/claims/P5-draft-headlines_script_output.txt).

Re-runnable; overwrites the batch CSV only.
"""
import csv
import os

OUT = r"C:/Users/SURAIYA/Documents/STUDY/thesis/expLB/audit/phase2/claims_batch_P5-draft-headlines.csv"
COLS = ["id", "claim_plain", "stated_numbers", "source_ref", "evidence_file",
        "evidence_key", "level1_value", "level1_match", "level2_value",
        "level2_method", "version_used", "robustness", "verdict", "confidence",
        "importance", "script", "notes"]

SCRIPT = "audit/scripts/claims/P5_draft_headlines.py"
GITRES = "git show origin/feat/persona-pc-ood:src/probes/deception-detection/results"
VER = ("origin/feat/persona-pc-ood @ tip, read with `git show` (never checked out); "
       "his activations are gone -> Level 2 on his pipeline impossible (owner's rule). "
       "Files are his own committed CSVs; not touched by the 09-22 re-run wave (RF-13), "
       "which hit the OWNER's lab stores only.")
L2 = ("Level 2 on his pipeline: NOT POSSIBLE (activations gone). What IS recomputed: his "
      "aggregates re-derived from his own committed cell-level CSVs.")

R = []


def add(**kw):
    row = {c: "" for c in COLS}
    row.update(kw)
    row["script"] = row["script"] or SCRIPT
    row["version_used"] = row["version_used"] or VER
    R.append(row)


add(id="D-003",
    claim_plain="On HP-KR the published 8-vs-7 persona contrast inverts: zero-shot AUROC 0.148 at layer 20 on Llama-3.3-70B, re-extracted persona bank, response-mean pooling, no seed (deterministic direction).",
    stated_numbers="0.148",
    source_ref="draft main (1).pdf p.1 Abstract + p.7 sec 5.1 + Table 2; draft_claims_raw.csv D-003",
    evidence_file=f"{GITRES}/persona_ood_ext_L20/zero_shot.csv",
    evidence_key="row harm_refusal_knowledge_report__plain, col auroc_axis",
    level1_value="0.1482769784172662",
    level1_match="YES (draft 0.148, tol 5e-4)",
    level2_value=L2,
    level2_method="Cross-checked against his zero_shot_operating.csv (same layer/arm/dataset): 0.1483, identical. n=2139, n_deceptive=139.",
    robustness="Same direction at L40 0.5926 and L44 0.5895 (draft quotes 0.593/0.589; both match) -> the INVERSION is layer-20-only; at L40/L44 it is near chance, not inverted. On the ORIGINAL (pre-re-extraction) bank the same grouping reads 0.1029 (persona_ood/zero_shot.csv).",
    verdict="MATCHES HIS DATA (Level 1 only)",
    confidence="HIGH - exact match, cross-checked in two of his files",
    importance="HIGH - abstract, contribution 3, Fig 3, Table 2",
    notes="Single run, single bank, one pooling. The word 'inverts' is exact only at L20.")

add(id="D-004",
    claim_plain="Every HP-KR lie is a denial of knowledge, and none of Mahadik & Skapars' 8 published deceptive personas lies by denying knowledge.",
    stated_numbers="(none - qualitative)",
    source_ref="draft p.1 Abstract + p.7 sec 5.1; draft_claims_raw.csv D-004",
    evidence_file="papers/2511.16035v2 (1).pdf p.5; papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf p.9 Table 2",
    evidence_key="LB p.5 HP-KR definition; M&S Table 2 (8 deceptive personas) and Table 3 (7 honest)",
    level1_value="LB p.5: HP-KR is 'labeled deceptive if the model denies knowing the answer despite that established belief, and honest otherwise'. M&S Table 2: handler 'controls timing, emphasis, and selective disclosure'; coverup operator 'buries or excludes material facts'; plausible denier 'gives defensible half-disclosures'; plus broker, impression manager, gatekeeper, spin doctor, narrative manager. None mentions claiming not to know.",
    level1_match="YES - all three quotations in the draft are verbatim/near-verbatim from M&S Table 2, and the HP-KR label rule is as the benchmark states it",
    level2_value="NOT TESTED - the causal half ('that is why the contrast has nothing to go on') is argued from persona descriptions, not measured.",
    level2_method="Read both source PDFs during this audit (pypdf text extraction).",
    robustness="The owner's own work bears on this: HP-KR's honest class pools affirmations with knowledge-free denials (LB p.5), so a denial-wording detector scores well without detecting deception - the draft concedes this in sec 6. The inversion is L20-only, which the persona-coverage story does not by itself predict.",
    verdict="PARTLY HOLDS (quotations and the benchmark's label rule verified; the causal claim is untested)",
    confidence="MED - the descriptive half is verified from primary sources; the explanatory half has no test in the draft or on his branch",
    importance="HIGH - it is the paper's causal story",
    notes="Also checked the bank arithmetic: M&S Tables 2-5 give 8+7+5+6=26 personas, matching the draft's '26 personas of Mahadik and Skapars'; 8+4 v 7+3+3 = 12 v 13 as the draft says.")

add(id="D-006",
    claim_plain="At layer 40 on HP-KR, one zero-shot contrast direction (the 4-vs-6 knowledge-denial contrast) reaches AUROC 0.919 and balanced accuracy 0.910 at a threshold flagging <=1% of the Alpaca control.",
    stated_numbers="0.919; 0.910; 1% FPR",
    source_ref="draft p.1 Abstract; draft_claims_raw.csv D-006",
    evidence_file=f"{GITRES}/persona_ood_ext_L40/zero_shot.csv and .../zero_shot_operating.csv",
    evidence_key="auroc_denial_axis @ harm_refusal_knowledge_report__plain; operating row (arm='knowledge-denial axis (4v6)', dataset=HP-KR, threshold_source='control')",
    level1_value="AUROC 0.918536 (operating file 0.9185); balacc 0.9102; recall 0.8849; specificity 0.9355",
    level1_match="YES (0.9185 -> 0.919; 0.9102 -> 0.910)",
    level2_value=L2,
    level2_method="Cross-file check: all 54 zero_shot.csv AUROCs equal the zero_shot_operating.csv AUROCs (0 mismatches).",
    robustness="Bootstrap 95% CI (his, 1000 resamples over examples): AUROC [0.8878, 0.9450], balacc [0.8835, 0.9369]. Same axis at L20 balacc 0.5331 and L44 0.7571 -> the 0.910 is a LAYER-40-ONLY operating point. n_deceptive=139 of n=2139.",
    verdict="MATCHES HIS DATA (Level 1 only)",
    confidence="HIGH for the arithmetic; MED for the claim as read, see notes",
    importance="HIGH - abstract headline",
    notes="AMBIGUITY: the abstract's 'a single contrast direction' here is the 4v6 knowledge-denial axis, while the abstract's transfer headline (D-008) uses the 12v13 EXTENDED axis. Two different directions carry the two halves of one abstract; only Table 2 disambiguates. The extended axis's balanced accuracy on HP-KR is 0.500 at all three layers.")

add(id="D-008",
    claim_plain="Over the 6x6 train-test matrix at layer 40, the extended 12-vs-13 contrast reaches 0.916 mean off-diagonal AUROC against 0.732 for 8192-d logistic probes each trained on one dataset.",
    stated_numbers="0.916; 0.732",
    source_ref="draft p.1 Abstract + p.8 sec 5.2 + Table 3; draft_claims_raw.csv D-008",
    evidence_file=f"{GITRES}/persona_ood_ext_L40/summary.csv",
    evidence_key="rows features=ext_axis1 and features=baseline, col off_diagonal_mean",
    level1_value="ext_axis1 0.915835; baseline 0.732033",
    level1_match="YES (both, tol 5e-4)",
    level2_value="Re-derived from his matrix_ext_axis1.csv / matrix_baseline.csv: 0.9158 and 0.7320 (30 off-diagonal cells). All 24 (layer x feature) summary values reproduce from the matrices.",
    level2_method="Mean of the 30 off-diagonal cells of each matrix CSV; audit/scripts/claims/P5_draft_headlines.py section 'Off-diagonal mean recomputed from the matrices'.",
    robustness="Excluding HP-KR: 0.9165 vs 0.7581 - the gap survives (+0.158). Random 3-d directions give 0.504 at L40, so the comparison is not a ceiling artifact of 1-d features alone. BUT see notes.",
    verdict="MATCHES HIS DATA (Level 1; aggregate also re-derived from his cells)",
    confidence="HIGH for the arithmetic",
    importance="HIGH - abstract + contribution 4 + Table 3 + Fig 4",
    notes=("STRUCTURAL CAVEAT (audit's, not his): for a 1-d feature the trained probe learns only a sign, so his "
           "ext_axis1 matrix rows are constant and its 'mean off-diagonal AUROC' equals its mean ZERO-SHOT AUROC over "
           "the 6 datasets (0.9158 both ways, verified). The headline therefore compares a fixed direction against "
           "probes that fit each training set - which the draft does say in the Fig 4 caption, but the number is not a "
           "transfer measurement in the same sense. DECOMPOSITION: at L40 raw 0.732 -> published 8v7 0.832 (+0.100) -> "
           "extended 0.916 (+0.084); i.e. 46% of the margin over raw comes from the 10 added knowledge-denial roles "
           "(which are byte-identical to the OWNER's run-kd/vectors_s3). At L20: raw 0.648 -> 8v7 0.651 (+0.003) -> "
           "ext 0.904, i.e. ~99% of the margin. No matched low-dimensional control (e.g. dataset PCA) is reported; the "
           "owner's own work found that control removes the advantage in her setting."))

add(id="D-009",
    claim_plain="Averaged over training sets, the extended contrast beats the raw 8192-d probes on every held-out dataset (layer 40, six test sets).",
    stated_numbers="(none in the abstract; Table 7 carries the numbers)",
    source_ref="draft p.1 Abstract; Table 7 p.17; draft_claims_raw.csv D-009",
    evidence_file=f"{GITRES}/persona_ood_ext_L40/matrix_ext_axis1.csv, matrix_baseline.csv",
    evidence_key="per test-set column, mean over the 5 rows that are not that dataset",
    level1_value="L40 per test set (raw | 12v13): CG 0.7875|0.9123, HP-C 0.5709|0.7116, HP-KR 0.5836|0.9124, IT 0.7999|0.9803, ID 0.8267|0.9784, RepE 0.8236|1.0000",
    level1_match="YES - all 12 numbers equal Table 7's layer-40 columns to 3 dp",
    level2_value="Recomputed the whole of Table 7 (36 cells, 3 layers) from his matrices: 0 mismatches.",
    level2_method="Column mean over the 5 non-self training rows; P5_draft_headlines.py section C.",
    robustness="Holds at L20 and L44 as well (18/18 cells). At L40 the smallest margin is CG (+0.1248) and the largest HP-KR (+0.3288).",
    verdict="MATCHES HIS DATA (Level 1; Table 7 fully re-derived from his cells)",
    confidence="HIGH",
    importance="HIGH - abstract",
    notes="Single cells still favour a raw probe (the draft says so): at L40 the ID-trained probe scores 0.9831 on CG vs 0.9123 for the contrast (matrix_baseline L40, row ID, col CG) - verified.")

add(id="D-010",
    claim_plain="The persona principal components add nothing beyond the 1-d contrast (mean off-diagonal AUROC, all three layers).",
    stated_numbers="(Table 3: ext_pc3 0.684/0.866/0.880, ext_pc1 0.661/0.768/0.780 vs ext_axis1 0.904/0.916/0.903)",
    source_ref="draft p.1 Abstract + Table 3; draft_claims_raw.csv D-010",
    evidence_file=f"{GITRES}/persona_ood_ext_L{{20,40,44}}/summary.csv",
    evidence_key="rows ext_axis1, ext_pc3, ext_pc1, col off_diagonal_mean",
    level1_value="L20 0.9041 / 0.6837 / 0.6613; L40 0.9158 / 0.8655 / 0.7680; L44 0.9034 / 0.8799 / 0.7801",
    level1_match="YES - all 48 Table 3 cells (8 feature sets x 3 layers x 2 dataset sets) match, 0 mismatches",
    level2_value="All 24 off-diagonal means re-derived from his matrix CSVs; max |diff| 0.0001.",
    level2_method="P5_draft_headlines.py sections B and 'Off-diagonal mean recomputed'.",
    robustness="No reversal anywhere, but the margin is HP-KR-driven at L20: excluding HP-KR, ext_axis1 0.9086 vs ext_pc3 0.9000 (+0.0086, against +0.2204 with HP-KR in). At L40/L44 the excl-HP-KR margins are +0.050 and +0.035. Random 3-d control: 0.527/0.504/0.493.",
    verdict="MATCHES HIS DATA (Level 1; aggregates re-derived)",
    confidence="HIGH for the numbers; MED for the wording 'add nothing' (it is 'score lower', not an orthogonality test)",
    importance="HIGH - abstract + contribution 5; it is the claim that contradicts the method being replicated",
    notes="'Add nothing beyond it' is not tested as an incremental-value claim (no model with contrast + PCs together). What is shown is that a probe on 3 PCs transfers worse than a probe on the 1 contrast.")

add(id="D-027",
    claim_plain="The extended contrast beats the raw probes' average over training sets on every held-out dataset at every layer tested (L20, L40, L44).",
    stated_numbers="(Table 7, 36 cells)",
    source_ref="draft p.2 Contributions; Table 7; draft_claims_raw.csv D-027",
    evidence_file=f"{GITRES}/persona_ood_ext_L{{20,40,44}}/matrix_ext_axis1.csv, matrix_baseline.csv",
    evidence_key="18 (test set x layer) column means over the 5 other training rows",
    level1_value="12v13 > raw in 18 of 18 cells. Smallest margins: L44 CG +0.1015, L20 HP-C +0.1044, L40 CG +0.1248. Largest: L20 HP-KR +0.4880.",
    level1_match="YES",
    level2_value="Table 7 fully recomputed from his matrices: 36/36 cells match the printed table.",
    level2_method="P5_draft_headlines.py section C + 'D-009/D-027/D-195 verdict inputs'.",
    robustness="'Every layer we tested' = 3 layers only (20, 40, 44); the draft states this limit in sec 5.5. No seeds (the direction is deterministic); no CIs on these per-test-set means.",
    verdict="MATCHES HIS DATA (Level 1; Table 7 re-derived)",
    confidence="HIGH",
    importance="HIGH - contribution 4",
    notes="Three layers is not a layer sweep. The owner's repeated experience (memory: 'No margin from one layer') is that margins from few layers do not survive a full sweep; his activations are gone, so a sweep is impossible on his side.")

add(id="D-028",
    claim_plain="The 1-d extended contrast outscores a probe on the 3 leading extended-persona PCs at every depth, by 0.220 mean off-diagonal AUROC at layer 20 and 0.023 at layer 44.",
    stated_numbers="0.220; 0.023",
    source_ref="draft p.2 Contributions + p.8 sec 5.3; draft_claims_raw.csv D-028",
    evidence_file=f"{GITRES}/persona_ood_ext_L{{20,40,44}}/summary.csv",
    evidence_key="off_diagonal_mean(ext_axis1) - off_diagonal_mean(ext_pc3)",
    level1_value="L20 0.904117-0.683697 = 0.220420; L40 0.915835-0.865534 = 0.050301; L44 0.903388-0.879936 = 0.023453",
    level1_match="YES (0.220, 0.050 and 0.023 as printed)",
    level2_value="Both operands re-derived from his matrices (0 mismatches).",
    level2_method="P5_draft_headlines.py section 'D-028 / D-104 / D-010'.",
    robustness="Positive at all three depths but shrinking monotonically (0.220 -> 0.050 -> 0.023). Excluding HP-KR the L20 gap collapses to 0.0086 (0.9086 vs 0.9000), so the 0.220 is almost entirely the HP-KR column. No CI on the difference.",
    verdict="MATCHES HIS DATA (Level 1; aggregates re-derived)",
    confidence="HIGH for the arithmetic; the excl-HP-KR collapse is the caveat",
    importance="HIGH - contribution 5",
    notes="AUDIT FLAG: the headline 0.220 is HP-KR-driven. His own Table 3 excl-HP-KR column shows it (0.909 vs 0.900) but the contributions bullet does not.")

add(id="D-083",
    claim_plain="With no benchmark labels the knowledge-denial 4-vs-6 contrast reaches AUROC 0.890, 0.919 and 0.908 on HP-KR at layers 20, 40 and 44.",
    stated_numbers="0.890; 0.919; 0.908",
    source_ref="draft p.7 sec 5.1 + Table 2; draft_claims_raw.csv D-083",
    evidence_file=f"{GITRES}/persona_ood_ext_L{{20,40,44}}/zero_shot.csv",
    evidence_key="row harm_refusal_knowledge_report__plain, col auroc_denial_axis",
    level1_value="0.889932 / 0.918536 / 0.907683",
    level1_match="YES (all three)",
    level2_value=L2,
    level2_method="All 54 Table 6 cells (6 datasets x 3 contrasts x 3 layers) checked against zero_shot.csv: 0 mismatches.",
    robustness="Extended 12v13 on the same rows: 0.882/0.912/0.894 (also matches Table 2). Original-bank published grouping: 0.1029. Sycophancy axis L20: 0.0429.",
    verdict="MATCHES HIS DATA (Level 1 only)",
    confidence="HIGH",
    importance="HIGH - sec 5.1 headline, Table 2, Fig 3",
    notes=("The 4-vs-6 split is by ROLE NAME only - his own summary.md (persona_ood_ext_L20, 'Caveats') says 'the files "
           "carry no group metadata, so the 4/6 split is a reading of the role names. A different split would give a "
           "different axis.' Also: the personas are the OWNER's (his 36-vector bank is byte-identical to run-kd/vectors_s3)."))

add(id="D-093",
    claim_plain="At layer 40 on HP-KR the knowledge-denial contrast reaches 0.910 balanced accuracy at the 1%-Alpaca-FPR threshold, recall 0.885, and flags 6% of the honest answers.",
    stated_numbers="0.910; 0.885; 6%",
    source_ref="draft p.8 sec 5.1; draft_claims_raw.csv D-093",
    evidence_file=f"{GITRES}/persona_ood_ext_L40/zero_shot_operating.csv",
    evidence_key="arm='knowledge-denial axis (4v6)', dataset=harm_refusal_knowledge_report__plain, threshold_source='control'",
    level1_value="balacc 0.9102; recall 0.8849; specificity 0.9355 -> 6.45% of honest answers flagged",
    level1_match="PARTLY - 0.910 and 0.885 match exactly; the file's honest-flag rate is 6.45%, printed as '6%'",
    level2_value=L2,
    level2_method="Read his operating CSV; specificity -> 1-specificity.",
    robustness="95% CI: balacc [0.8835,0.9369], recall [0.8345,0.9353]. At L20 the same axis flags 88.35% of honest answers (balacc 0.5331) and at L44 4.7% (balacc 0.7571) - the operating point is highly layer-dependent, which the draft states.",
    verdict="MATCHES HIS DATA, one rounding exception (6.45% written as 6%)",
    confidence="HIGH",
    importance="HIGH - sec 5.1, Fig 3, Fig 5",
    notes="6.45% rounds to 6.5%, not 6%. Cosmetic, but it is the number that makes the detector look clean at the benchmark threshold.")

add(id="D-098",
    claim_plain="Mean off-diagonal AUROC of the extended contrast is 0.904, 0.916, 0.903 at layers 20/40/44 against 0.648, 0.732, 0.716 for the raw 8192-d probes.",
    stated_numbers="0.904/0.916/0.903 vs 0.648/0.732/0.716",
    source_ref="draft p.8 sec 5.2 + Table 3; draft_claims_raw.csv D-098",
    evidence_file=f"{GITRES}/persona_ood_ext_L{{20,40,44}}/summary.csv",
    evidence_key="rows ext_axis1 / baseline, col off_diagonal_mean",
    level1_value="0.904117 / 0.915835 / 0.903388 and 0.647770 / 0.732033 / 0.715649",
    level1_match="YES - all six",
    level2_value="All six re-derived from his matrix CSVs (30 off-diagonal cells each): identical to 4 dp.",
    level2_method="P5_draft_headlines.py sections B and 'Off-diagonal mean recomputed'.",
    robustness="Also checked the two sentences around it: baseline in-distribution 0.983 at L40 and the 0.251 drop off the diagonal both match (0.983273, 0.251240); the worst raw cell (CG-trained on HP-KR) is 0.0598, printed as 0.06.",
    verdict="MATCHES HIS DATA (Level 1; aggregates re-derived)",
    confidence="HIGH",
    importance="HIGH - sec 5.2 headline, same numbers as the abstract",
    notes="Same structural caveat as D-008: for a 1-d feature the off-diagonal mean equals the zero-shot mean over the 6 datasets.")

add(id="D-099",
    claim_plain="Averaged over training sets the extended contrast beats the raw probes on every held-out dataset at every layer, and matches or beats the published 8v7 contrast.",
    stated_numbers="(Table 7; Table 6 for the published contrast)",
    source_ref="draft p.8 sec 5.2; draft_claims_raw.csv D-099",
    evidence_file=f"{GITRES}/persona_ood_ext_L{{20,40,44}}/matrix_ext_axis1.csv, matrix_baseline.csv, matrix_axis1.csv, zero_shot.csv",
    evidence_key="per-test-set means over the 5 other training rows; and Table 6 zero-shot cells",
    level1_value="vs raw: 18/18 cells won. vs published 8v7 ON THE SAME MEAN-OVER-TRAINING-SETS BASIS: 18/18 won (min margin +0.0000 at L40/L44 RepE, both 1.0000). vs published 8v7 ON THE ZERO-SHOT BASIS (Table 6 as printed): 12v13 is BELOW 8v7 in 3 of 18 cells, all at L20 - HP-C 0.639 vs 0.717, ID 0.954 vs 0.959, RepE 0.991 vs 0.994.",
    level1_match="PARTLY - true under the Table 7 reading the sentence cites, false under the Table 6 reading a reader would use",
    level2_value="Table 7 recomputed in full (36/36 cells match); Table 6 recomputed in full (54/54 cells match).",
    level2_method="P5_draft_headlines.py sections C and 'D-099 second half'.",
    robustness="The two readings differ because the trained 8v7 probe can flip sign: at L20 the HP-KR-trained row of matrix_axis1 is inverted (0.0601, 0.2831, 0.0121, 0.0413, 0.0065), which drags the published contrast's Table-7 means down.",
    verdict="PARTLY MATCHES HIS DATA",
    confidence="HIGH - both readings computed from his files",
    importance="HIGH",
    notes=("Phase 0's flag confirmed: Table 7 as printed has NO published-contrast column, so the sentence's pointer "
           "'(Table 7, Appendix B)' cannot be checked where it points. Recommend the draft either add the column or "
           "say 'on the mean-over-training-sets basis'."))

add(id="D-104",
    claim_plain="At 70B the contrast between persona group means beats probes on the persona principal components at every depth.",
    stated_numbers="(Table 3)",
    source_ref="draft p.8 sec 5.3; draft_claims_raw.csv D-104",
    evidence_file=f"{GITRES}/persona_ood_ext_L{{20,40,44}}/summary.csv",
    evidence_key="ext_axis1 vs ext_pc3 and ext_pc1, col off_diagonal_mean",
    level1_value="L20 0.9041 > 0.6837 and > 0.6613; L40 0.9158 > 0.8655 and > 0.7680; L44 0.9034 > 0.8799 and > 0.7801",
    level1_match="YES - 6/6 comparisons",
    level2_value="Operands re-derived from his matrices.",
    level2_method="P5_draft_headlines.py section 'D-028 / D-104 / D-010'.",
    robustness="Excluding HP-KR at L20 the margin over ext_pc3 shrinks from 0.220 to 0.009. On the PUBLISHED bank the same statement is stronger (axis1 beats pc3 at L40/L44 but LOSES at L20: 0.651 vs 0.679) - so 'the contrast beats the PCs at every depth' is true for the EXTENDED bank only.",
    verdict="MATCHES HIS DATA (Level 1)",
    confidence="HIGH",
    importance="HIGH - sec 5.3, contradicts the replicated method's central operation",
    notes="AUDIT FLAG: on the published bank, at L20, pc3 (0.679) beats axis1 (0.651). The draft's Table 3 shows this, but sec 5.3's sentence does not qualify which bank.")

add(id="D-117",
    claim_plain="At layer 40, averaged over the five Liars' Bench datasets, the extended contrast reaches 0.694 balanced accuracy, 0.413 recall and 0.899 AUROC; the RepE-trained base probe reaches 0.626, 0.281 and 0.778.",
    stated_numbers="0.694/0.413/0.899 vs 0.626/0.281/0.778",
    source_ref="draft p.9 sec 5.4 + Table 4; draft_claims_raw.csv D-117",
    evidence_file=f"{GITRES}/persona_ood_ext_L40/zero_shot_operating.csv",
    evidence_key="dataset=\"Average (Liars' Bench)\", threshold_source='control', arms 'extended axis (12v13)' and 'base probe (RepE-trained)'",
    level1_value="extended 0.6939 / 0.4129 / 0.8990; base probe 0.6255 / 0.2814 / 0.7784",
    level1_match="YES (0.6255 -> 0.626 and 0.6535 -> 0.654 are the only two of 36 Table-4 cells that need round-half-up)",
    level2_value=L2,
    level2_method="All 36 Table-4 cells checked (3 layers x 4 detectors x 3 metrics).",
    robustness="At this threshold the knowledge-denial 4v6 axis beats the extended one on BA and recall (0.725 / 0.487), as the draft says. At L20 the base probe leads on BA (0.721 vs 0.591) and recall (0.489 vs 0.187).",
    verdict="MATCHES HIS DATA (Level 1 only)",
    confidence="HIGH",
    importance="HIGH - sec 5.4, Table 4, Fig 5",
    notes=("RESOLVES Phase-0 red flag RF-3(c): the base probe's identical 0.626/0.281 at L40 and L44 is NOT a copied "
           "file. Per dataset the two layers differ (CG AUROC 0.9723 vs 0.9395; HP-C 0.675 vs 0.7577; HP-KR 0.9358 vs "
           "0.8283; IT 0.3378 vs 0.2888; thresholds -2.4762 vs -2.9885), and the averages are 0.6255/0.2814 vs "
           "0.6256/0.2812 - they only LOOK identical at 3 dp because 4 of 6 datasets flag nothing (BA 0.5, recall 0.0) "
           "at both layers."))

add(id="D-166",
    claim_plain="Persona coordinates scale to a 70B model, but only as far as the persona bank covers the kind of lie being told.",
    stated_numbers="(none)",
    source_ref="draft p.13 sec 8 Conclusion; draft_claims_raw.csv D-166",
    evidence_file=f"{GITRES}/persona_ood_ext_L{{20,40,44}}/zero_shot.csv, summary.csv",
    evidence_key="auroc_axis vs auroc_denial_axis on HP-KR; summary.csv off_diagonal_mean",
    level1_value="Support in his files: the published bank reads 0.148/0.593/0.589 on HP-KR and 0.651/0.832/0.826 off-diagonal; adding 10 knowledge-denial roles gives 0.890/0.919/0.908 and 0.904/0.916/0.903.",
    level1_match="YES as a summary of Tables 2 and 3",
    level2_value="NOT TESTED as a general claim - one added lie type, one model, three layers, one benchmark.",
    level2_method="Read his summary.csv / zero_shot.csv (above).",
    robustness="The generalisation 'only as far as the bank covers the lie type' is supported by exactly one instance (HP-KR). The draft's own sec 6 concedes the personas were written after seeing HP-KR fail, so this is a designed repair, not a blind test.",
    verdict="MATCHES HIS DATA as a restatement; NOT TESTED as a general claim",
    confidence="MED - the numbers behind it are verified; the generalisation is n=1",
    importance="HIGH - it is the paper's conclusion sentence",
    notes="The owner's independent capitulation experiment (cap/, F-65/F-66: 0.374 -> 0.745 zero-shot) is a second instance of the same pattern and is NOT cited by the draft.")

add(id="D-167",
    claim_plain="With no benchmark labels the extended bank's contrast detects HP-KR lies at the benchmark's own threshold nearly as well as a probe trained on the benchmark itself (0.910 vs 0.95 balanced accuracy).",
    stated_numbers="0.910 vs 0.95",
    source_ref="draft p.13 sec 8 Conclusion + Table 2; draft_claims_raw.csv D-167",
    evidence_file=f"{GITRES}/persona_ood_ext_L40/zero_shot_operating.csv ; papers/2511.16035v2 (1).pdf p.56 Table 8",
    evidence_key="his: knowledge-denial axis, HP-KR, control threshold, balacc. LB: Table 8 'BALANCED ACCURACY', row Llama 3.3 70B, column HP-KR",
    level1_value="his 0.9102; Liars' Bench Upper Bound Probe = 0.95 (and AUROC 0.97). Also verified: LB Mean Probe HP-KR Llama-3.3-70B AUROC 0.12 / BA 0.50 (Table 6, p.54) and Follow-Up Probe 0.14 / 0.26 (Table 7, p.55) - exactly the numbers the draft's Table 2 and abstract quote.",
    level1_match="YES - both sides, including all six Liars'-Bench-reported values in the draft's Table 2",
    level2_value=L2,
    level2_method="pypdf text extraction of the Liars' Bench PDF, pages 54-56, read during this audit.",
    robustness="The 0.910 is the 4v6 axis at L40 only; the EXTENDED (12v13) axis, which the conclusion sentence's subject actually names, scores 0.500 balanced accuracy on HP-KR at all three layers.",
    verdict="PARTLY MATCHES HIS DATA - both numbers are right, but the sentence attributes 0.910 to the wrong contrast",
    confidence="HIGH - his file and the LB paper both checked directly",
    importance="HIGH - conclusion",
    notes=("Wording problem: 'Adding roles that deny knowledge turns it into one that ... detects those lies at the "
           "benchmark's own threshold' reads as the extended bank's direction. In his own Table 2 the extended (12v13) "
           "contrast has balanced accuracy 0.500 on HP-KR at every layer; only the 4v6 knowledge-denial contrast "
           "reaches 0.910. Same ambiguity as D-006."))

add(id="D-168",
    claim_plain="Across datasets, one zero-shot contrast over the extended bank transfers better on average than probes trained on single datasets, and better than the persona subspace.",
    stated_numbers="(Table 3)",
    source_ref="draft p.13 sec 8 Conclusion; draft_claims_raw.csv D-168",
    evidence_file=f"{GITRES}/persona_ood_ext_L{{20,40,44}}/summary.csv",
    evidence_key="off_diagonal_mean rows ext_axis1, baseline, ext_pc3",
    level1_value="ext_axis1 0.904/0.916/0.903 vs baseline 0.648/0.732/0.716 vs ext_pc3 0.684/0.866/0.880",
    level1_match="YES - ext_axis1 is highest at all three layers on both comparisons",
    level2_value="All operands re-derived from his matrices.",
    level2_method="P5_draft_headlines.py sections B and 'Off-diagonal mean recomputed'.",
    robustness="Survives removing HP-KR (0.909/0.917/0.905 vs raw 0.728/0.758/0.730 and vs ext_pc3 0.900/0.867/0.870), though at L20 the margin over ext_pc3 falls to 0.009.",
    verdict="MATCHES HIS DATA (Level 1; aggregates re-derived)",
    confidence="HIGH",
    importance="HIGH - conclusion",
    notes="Same structural caveat as D-008 (a 1-d zero-shot direction has no training set, so 'transfers better' is a comparison of kinds, and no matched low-dimensional control is reported).")

add(id="D-195",
    claim_plain="Table 7 caption: the extended contrast is higher than the raw probe on every test set at every layer.",
    stated_numbers="(36 cells of Table 7)",
    source_ref="draft p.17 Table 7 caption; draft_claims_raw.csv D-195",
    evidence_file=f"{GITRES}/persona_ood_ext_L{{20,40,44}}/matrix_ext_axis1.csv, matrix_baseline.csv",
    evidence_key="18 column means (6 test sets x 3 layers), each over the 5 non-self training rows",
    level1_value="18/18 cells higher. All 36 printed Table-7 values reproduce to 3 dp.",
    level1_match="YES",
    level2_value="Table 7 recomputed in full from his matrices; 0 mismatches.",
    level2_method="P5_draft_headlines.py section C.",
    robustness="Identical evidence to D-027/D-009; only three layers exist.",
    verdict="MATCHES HIS DATA (Level 1; Table 7 re-derived)",
    confidence="HIGH",
    importance="MED - it is a caption restating D-027",
    notes="")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    for row in R:
        w.writerow(row)
print(f"wrote {len(R)} rows to {OUT}")
