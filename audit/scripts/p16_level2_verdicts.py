"""Phase 6 / 6A LEVEL 2 -- append the 54 re-derived verdicts to audit/claims.csv.

Batches: B23 (Abstract), B41 (T2), B42 (T3), B43 (T4), B45 (T6), B46 (T7),
         B40 (T11), B30 (App F).

Every verdict here was produced by `audit/scripts/p16_level2.py` plus the
targeted checks recorded in `audit/phase6/RESULTS.md`.  Idempotent: re-running
replaces the rows for these ids rather than duplicating them.
"""
import csv, pathlib, shutil, datetime

ROOT = pathlib.Path(__file__).resolve().parents[2]
CLAIMS = ROOT / "audit" / "claims.csv"
BATCHDIR = ROOT / "audit" / "phase6" / "batches"

EXT = "src/probes/deception-detection/results/persona_ood_ext_L{20,40,44}"
ZS = f"origin/paper:{EXT}/zero_shot.csv"
SUMM = f"origin/paper:{EXT}/summary.csv"
OPER = f"origin/paper:{EXT}/zero_shot_operating.csv"
MATB = f"origin/paper:{EXT}/matrix_baseline.csv"
MATA = f"origin/paper:{EXT}/matrix_axis1.csv"
MATS = f"origin/paper:{EXT}/matrix_syc_axis1.csv"
LBPDF = "papers/2511.16035v2 (1).pdf"
SCRIPT = "audit/scripts/p16_level2.py"

DIM1 = ("A dim=1 arm: its trained off-diagonal EQUALS the zero-shot AUROC up to sign "
        "(verified here -- every cell in a test column is identical across training rows). "
        "REVISED 2026-09-24 after Level 1: this is NOT an undisclosed framing problem. "
        "The draft states the identity explicitly in section 3.4 -- 'A probe on a single feature "
        "can learn only a sign and a scale. Whenever every training set agrees on the sign, its "
        "AUROC on a test set therefore equals the zero-shot AUROC, and it has no training set to "
        "overfit' -- Table 3's own caption says 'The 1-d rows are persona contrasts, a probe on "
        "one feature', and the Contributions call the arm 'the zero-shot extended contrast'. "
        "DF-93's second half does not bite either: intervals are '95% stratified bootstrap "
        "intervals over examples (1,000 resamples)', never over the 30 matrix cells.")

# id: (verdict, level2_value, notes, evidence_file, evidence_key)
V = {}

# ---------------------------------------------------------------- B23 Abstract
V["D-001"] = ("MATCHES HIS DATA",
    "Re-derived: baseline (8192-d raw) diagonal_mean 0.9833 vs off_diagonal_mean 0.7320 at L40 "
    "-- a 0.251 in-distribution-to-transfer drop. L20 0.9787/0.6478; L44 0.9826/0.7159.",
    "Supported by his own Table 3 source file. The gap is largest at L20 (0.331).",
    SUMM, "features=baseline, columns diagonal_mean / off_diagonal_mean, L20/L40/L44")

V["D-002"] = ("MATCHES HIS DATA",
    "Checked against the cited paper itself: main model Llama-3.2-3B, 'additional replications on "
    "Llama 3-8B' (p.4), with the 8B evaluations in Appendices E and F. Contribution bullets on p.2 "
    "claim persona-PC projections 'generalize better across datasets than probes trained on the raw "
    "activations'.",
    "Faithful to the source. TWO caveats a reader needs: (1) the paper's own words for 8B are "
    "'auxiliary comparisons ... broadly supportive' (p.5), not a full replication; (2) this project's "
    "own three-rung ladder (F-121) finds the persona gain DECLINES with scale on both banks "
    "(+0.038 -> +0.015 -> -0.012 for 26 personas; +0.036 -> +0.019 -> +0.005 for 36), so 'better at "
    "3B and 8B' does not extend to 70B. That is the paper's contribution and should be said here.",
    "papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf",
    "pp.1-5 contribution bullets and Section 5; cross-check kd/docs/FINDINGS.md F-121")

V["D-005"] = ("MATCHES HIS DATA",
    "Re-derived exactly from the persona bank: knowledge_denial 4 (pc_amnesiac_witness, "
    "pc_deniable_expert, pc_stonewaller, pc_feigned_novice) + sincere_ignorance 3 + "
    "plain_assertion 3 = 10. Consistent with the 12v13 contrast: 8+4=12 deceptive, 7+3+3=13 honest.",
    "Exact, and the three-way split matches the abstract's three-way wording. Note DF-25: his "
    "KNOWLEDGE_CANDOUR_PERSONAS merges sincere_ignorance and plain_assertion into one arm, which is "
    "the split the owner kept apart to remove a wording confound.",
    "kd/personas/kd_role_groups.json", "groups.{knowledge_denial,sincere_ignorance,plain_assertion}.roles")

V["D-007"] = ("MATCHES HIS DATA",
    "All three re-derived from the Liars' Bench paper, Llama-3.3-70B / HP-KR: Mean Probe "
    "BA@1%FPR 0.50 (Table 6 p.54), Follow-Up Probe BA@1%FPR 0.26 (Table 7 p.55), Upper-Bound "
    "Probe BA 0.95 (Table 8 p.56). Exact.",
    "Two scope conditions the sentence omits. (1) These are the Llama-3.3-70B rows; the "
    "cross-model averages are 0.50 / 0.42 / 0.72, so 0.26 in particular is model-specific. "
    "(2) The 0.95 upper bound is NOT a 1%-FPR operating point -- Table 8's FPR row gives 0.08 on "
    "HP-KR for Llama-70B, and the Follow-Up Probe's 0.26 carries a realised FPR of 0.84. Both "
    "thresholds are calibrated on alpaca, not on the evaluation set.",
    LBPDF, "Table 6 p.54 / Table 7 p.55 / Table 8 p.56, row 'Llama 3.3 70B', column HP-KR")

V["D-011"] = ("MATCHES HIS DATA",
    "Re-derived and strengthened. With the threshold calibrated to 1% FPR on alpaca, 23 of the 90 "
    "(arm x dataset x layer) cells exceed 1% realised FPR on the evaluation set; 67 sit at or below "
    "it. The failures are concentrated, not diffuse: knowledge-denial axis 88.3% on HP-KR and 35.6% "
    "on RepE at L20 (6.5% at L40, 4.7% at L44), and instruct_dishonesty runs 5-15% for every arm at "
    "every layer.",
    "The claim holds and is understated. This is the same defect the draft levels at Liars' Bench "
    "(Follow-Up Probe 0.37 recall at 84% FPR) and it is present in the draft's own headline arm at "
    "L20. Quote the realised FPRs.",
    OPER, "threshold_source=control, specificity column; FPR = 1 - specificity")

# ---------------------------------------------------------------- B41 Table 2
V["D-086"] = ("MATCHES HIS DATA",
    "AUROC 0.1483 / 0.5926 / 0.5895 -> 0.148 / 0.593 / 0.589. BA 0.5000 at all three. Exact.",
    "BA is exactly 0.500 because the arm never fires: recall 0.0000, specificity 1.0 at all three "
    "layers. A 0.500 that means 'no detections' is worth distinguishing from a 0.500 that means "
    "'chance'.",
    OPER, "arm='deception axis (8v7)', dataset=harm_refusal_knowledge_report__plain, threshold_source=control")

V["D-087"] = ("MATCHES HIS DATA",
    "AUROC 0.8899 / 0.9185 / 0.9077 -> 0.890 / 0.919 / 0.908 EXACT. "
    "BA 0.5331 / 0.9102 / 0.7571 -> 0.533 / 0.910 / 0.757 EXACT.",
    "All six values exact. Re-derived operating point: the threshold is fitted on Alpaca, so the "
    "realised FPR on HP-KR is 88.35% at L20 (specificity 0.1165, recall 0.9496), 6.45% at L40, "
    "4.70% at L44. REVISED 2026-09-24 after Level 1 -- an earlier reading of mine reduced this to "
    "PARTLY for an undisclosed operating point. **The draft discloses it**: the Figure 3 caption "
    "says 'At layer 20 its threshold falls inside HP-KR and flags most honest answers too', which "
    "is the 88% FPR in words. The L40 headline 0.910 stands, at 6.5% realised FPR. Worth "
    "quantifying in the text, but it is not a hidden defect.",
    OPER, "arm='knowledge-denial axis (4v6)', dataset=HP-KR, threshold_source=control, cols balacc/specificity/recall")

V["D-088"] = ("MATCHES HIS DATA",
    "AUROC 0.8816 / 0.9124 / 0.8936 -> 0.882 / 0.912 / 0.894. BA 0.5000 at all three. Exact.",
    "Same never-fires pattern as D-086: recall 0.0000, specificity 1.0. An arm with AUROC 0.912 and "
    "BA 0.500 is a ranking success and a calibration failure; that contrast is the paper's own point "
    "(D-011) and should be drawn here explicitly.",
    OPER, "arm='extended axis (12v13)', dataset=HP-KR, threshold_source=control")

V["D-089"] = ("MATCHES HIS DATA",
    "AUROC 0.1090 / 0.9358 / 0.8283 -> 0.109 / 0.936 / 0.828. BA 0.5000 at all three. Exact.",
    "Reproduces from zero_shot_operating.csv. NOTE a second, different base-probe number exists in "
    "the same result directory: matrix_baseline.csv[repe_honesty -> HP-KR] reads 0.3808 / 0.9281 / "
    "0.8803. At L20 the two disagree sharply (0.109 vs 0.381). Table 2 uses the operating file, so "
    "the claim matches -- but the draft should say which base probe it means, because the matrix "
    "version does not show the L20 inversion nearly as starkly.",
    OPER + " and " + MATB,
    "arm='base probe (RepE-trained)', dataset=HP-KR, control; vs matrix_baseline row repe_honesty col HP-KR")

V["D-090"] = ("MATCHES HIS DATA",
    "Liars' Bench Table 6 p.54, Llama-3.3-70B, HP-KR: AUROC 0.12, BA@1%FPR 0.50. Exact.",
    "Correctly attributed and correctly transcribed. Recall is 0.00 -- the probe never fires on "
    "HP-KR, so 0.50 is the no-detection floor, not chance-level discrimination.",
    LBPDF, "Table 6 p.54, AUROC and BALANCED ACCURACY AT 1% FPR blocks, row 'Llama 3.3 70B', col HP-KR")

V["D-091"] = ("MATCHES HIS DATA",
    "Liars' Bench Table 7 p.55, Llama-3.3-70B, HP-KR: AUROC 0.14, BA@1%FPR 0.26. Exact.",
    "Exact. The FPR block on the same page gives 0.84 for this cell -- the 0.26 balanced accuracy is "
    "produced at an 84% false-positive rate, which independently confirms DF-13. Quote the FPR "
    "whenever this number is used.",
    LBPDF, "Table 7 p.55, AUROC / BA@1%FPR / FPR AT 1% FPR blocks, row 'Llama 3.3 70B', col HP-KR")

V["D-092"] = ("PARTLY MATCHES HIS DATA",
    "Liars' Bench Table 8 p.56, Llama-3.3-70B, HP-KR: AUROC 0.97, BA 0.95. Both exact.",
    "The transcription is exact and correctly attributed; the reduction is for a caveat the draft "
    "does not carry. Liars' Bench Table 8's first block is headed 'BALANCED ACCURACY' with no "
    "'AT 1% FPR' qualifier (Tables 6 and 7 have it), and its own FPR block gives 0.08 for this cell. "
    "So the privileged upper bound operates at 8% FPR while the arms compared against it are held to "
    "1% -- the comparison is not like-for-like, and that is a property of the benchmark's table "
    "rather than of this transcription. Any 'within 0.03 of the upper bound' statement (DF-26) "
    "inherits it. One line noting the upper bound's operating point would settle it.",
    LBPDF, "Table 8 p.56, BALANCED ACCURACY / AUROC / FPR blocks, row 'Llama 3.3 70B', col HP-KR")

# ---------------------------------------------------------------- B42 Table 3
T3 = [("D-109", "baseline", "8192", "MATCHES HIS DATA", ""),
      ("D-110", "axis1", "1", "MATCHES HIS DATA", "SIGN"),
      ("D-111", "denial_axis1", "1", "MATCHES HIS DATA", "DIM1"),
      ("D-112", "ext_axis1", "1", "MATCHES HIS DATA", "DIM1"),
      ("D-113", "ext_pc3", "3", "MATCHES HIS DATA", ""),
      ("D-114", "ext_pc1", "1", "MATCHES HIS DATA", "DIM1"),
      ("D-115", "pc3", "3", "MATCHES HIS DATA", ""),
      ("D-116", "random3", "3", "MATCHES HIS DATA", "")]
T3VALS = {
    "D-109": "0.648 0.732 0.716 | 0.728 0.758 0.730", "D-110": "0.651 0.832 0.826 | 0.919 0.880 0.873",
    "D-111": "0.866 0.912 0.900 | 0.861 0.910 0.899", "D-112": "0.904 0.916 0.903 | 0.909 0.917 0.905",
    "D-113": "0.684 0.866 0.880 | 0.900 0.867 0.870", "D-114": "0.661 0.768 0.780 | 0.682 0.774 0.793",
    "D-115": "0.679 0.639 0.617 | 0.888 0.751 0.728", "D-116": "0.527 0.504 0.493 | 0.560 0.544 0.547"}
for cid, feat, dim, verdict, kind in T3:
    note = "All six values reproduce exactly."
    if kind == "DIM1":
        note += " " + DIM1
    if kind == "SIGN":
        note += (" " + DIM1 + " The L20 value is depressed by a sign flip: the HP-KR TRAINING row of "
                 "matrix_axis1.csv is exactly 1-x of every other row (verified cell by cell, diff "
                 "0.00000), row mean 0.0806 against 0.750-0.806 for the other five. Sign-corrected, "
                 "the L20 off-diagonal is 0.7909 -- precisely the 0.791 the draft prints in Table 6's "
                 "Mean row for the SAME arm at the SAME layer. REVISED after Level 1: the draft "
                 "DISCLOSES this mechanism in section 4 with the matching worked example -- 'the "
                 "inverted ranking on HP-KR carries into every test set (for instance CG falls from "
                 "0.940 to 0.060)' -- and 0.940/0.060 is exactly the cell pair verified here. The "
                 "only thing left is presentational: Table 3 does not cross-reference that "
                 "explanation, so a reader comparing Table 3 with Table 6 must find it in the prose. "
                 "At L40/L44 no row inverts and the two tables agree to 4 dp.")
    V[cid] = (verdict,
              f"summary.csv features={feat} (dim {dim}): all-six | excl-HP-KR at L20/L40/L44 = {T3VALS[cid]} -- EXACT",
              note, SUMM, f"features={feat}, off_diagonal_mean and off_diagonal_excl_hpkr")

# ---------------------------------------------------------------- B43 Table 4
T4 = {
 "D-140": (20, "base probe (RepE-trained)", "0.721 0.489 0.697"),
 "D-141": (20, "deception axis (8v7)", "0.534 0.069 0.750"),
 "D-142": (20, "knowledge-denial axis (4v6)", "0.559 0.305 0.863"),
 "D-143": (20, "extended axis (12v13)", "0.591 0.187 0.887"),
 "D-144": (40, "base probe (RepE-trained)", "0.6255 0.2814 0.7784 -> 0.626 0.281 0.778"),
 "D-145": (40, "deception axis (8v7)", "0.675 0.382 0.798"),
 "D-146": (40, "knowledge-denial axis (4v6)", "0.725 0.487 0.894"),
 "D-147": (40, "extended axis (12v13)", "0.694 0.413 0.899"),
 "D-148": (44, "base probe (RepE-trained)", "0.626 0.281 0.756"),
 "D-149": (44, "deception axis (8v7)", "0.6535 0.3180 0.7913 -> 0.654 0.318 0.791"),
 "D-150": (44, "knowledge-denial axis (4v6)", "0.683 0.395 0.880"),
 "D-151": (44, "extended axis (12v13)", "0.666 0.350 0.884")}
for cid, (L, arm, vals) in T4.items():
    note = ("Row reproduces exactly on all three of BA / recall / AUROC, from the "
            "'Average (Liars' Bench)' row (5 datasets, repe_honesty included) at threshold_source=control.")
    if "->" in vals:
        note += (" The stored value is an exact half-way case; the draft rounds half-up, which is "
                 "correct. (An earlier reading of mine used Python's banker's rounding and made this "
                 "look like a mismatch -- it is not.)")
    if "axis" in arm:
        note += (" This arm is dim=1 and label-free, so its AUROC column is zero-shot; only the "
                 "threshold is fitted. The base-probe row it is compared against is a trained "
                 "8192-d probe. Say so (DF-93).")
    V[cid] = ("MATCHES HIS DATA", f"L{L} {arm}: BA / recall / AUROC = {vals} -- EXACT", note,
              OPER, f"layer={L}, arm='{arm}', dataset=\"Average (Liars' Bench)\", threshold_source=control")

# ---------------------------------------------------------------- B45 Table 6
T6 = {"D-182": ("CG", "0.940 0.931 0.969 | 0.865 0.902 0.912 | 0.819 0.858 0.859"),
      "D-183": ("HP-C", "0.717 0.585 0.639 | 0.581 0.755 0.712 | 0.607 0.725 0.712"),
      "D-184": ("HP-KR", "0.148 0.890 0.882 | 0.593 0.919 0.912 | 0.589 0.908 0.894"),
      "D-185": ("IT", "0.988 0.976 0.990 | 0.979 0.913 0.980 | 0.979 0.933 0.982"),
      "D-186": ("ID", "0.959 0.932 0.954 | 0.974 0.980 0.978 | 0.962 0.978 0.974"),
      "D-187": ("RepE", "0.994 0.882 0.991 | 1.000 1.000 1.000 | 1.000 1.000 1.000")}
for cid, (ds, vals) in T6.items():
    V[cid] = ("MATCHES HIS DATA",
              f"{ds}: all nine cells (3 arms x 3 layers) reproduce EXACTLY: {vals}",
              "Columns are auroc_axis (8v7) / auroc_denial_axis (4v6) / auroc_ext_axis (12v13). "
              "Zero-shot, so no training and no sign ambiguity -- this table is the honest one.",
              ZS, f"dataset={ds}, columns auroc_axis / auroc_denial_axis / auroc_ext_axis")

V["D-188"] = ("MATCHES HIS DATA",
    "Mean row reproduces exactly as the unweighted mean of the six zero-shot cells: "
    "L20 0.7909/0.8660/0.9041 -> 0.791 0.866 0.904; L40 0.8319/0.9115/0.9158 -> 0.832 0.912 0.916; "
    "L44 0.8260/0.9001/0.9034 -> 0.826 0.900 0.903.",
    "All nine values exact. Worth knowing when reading it beside Table 3: for the 8v7 arm at L20 this "
    "Mean reads 0.791 while Table 3 reads 0.651 for the same arm at the same layer. Both reproduce; "
    "they are different quantities -- Table 6's is the zero-shot mean, Table 3's the trained "
    "off-diagonal -- and the 0.140 gap is entirely one sign-flipped HP-KR training row (row mean "
    "0.0806, exactly 1-x of the others). REVISED after Level 1: the draft explains this mechanism in "
    "section 4 with the matching example (CG 0.940 -> 0.060), so it is disclosed; an earlier reading "
    "of mine reduced this claim on the assumption it was not. What remains is presentational -- "
    "neither table cross-references the explanation. At L40/L44 the two agree to 4 dp.",
    ZS + " and " + SUMM, "mean of auroc_{axis,denial_axis,ext_axis} over 6 datasets vs summary.csv off_diagonal_mean")

# ---------------------------------------------------------------- B46 Table 7
T7 = {"D-189": ("CG", "0.663 0.969 | 0.7875->0.788 0.912 | 0.757 0.859"),
      "D-190": ("HP-C", "0.535 0.639 | 0.571 0.712 | 0.570 0.712"),
      "D-191": ("HP-KR", "0.394 0.882 | 0.584 0.912 | 0.637 0.894"),
      "D-192": ("IT", "0.799 0.990 | 0.800 0.980 | 0.736 0.982"),
      "D-193": ("ID", "0.778 0.954 | 0.827 0.978 | 0.808 0.974"),
      "D-194": ("RepE", "0.718 0.991 | 0.824 1.000 | 0.785 1.000")}
for cid, (ds, vals) in T7.items():
    note = ("Both columns reproduce exactly. The two columns are different protocols -- 'raw' is a "
            "TRAINED 8192-d probe averaged over the 5 off-diagonal training rows, while '12v13' is a "
            "label-free 1-d direction whose trained column mean equals its zero-shot AUROC exactly at "
            "every dataset and every layer (verified: all 18 pairs identical). REVISED 2026-09-24 "
            "after Level 1: an earlier reading of mine reduced this to PARTLY on the ground that the "
            "table does not say so. It does -- section 3.4 states the dim=1 identity outright, "
            "Table 3's caption flags the 1-d rows, and the Contributions call the arm 'the zero-shot "
            "extended contrast'. Trained-vs-zero-shot is the stronger claim and the draft makes it.")
    if cid == "D-189":
        note += (" The L40 raw cell is an exact half-way value 0.7875; the draft's 0.788 is correct "
                 "round-half-up.")
    V[cid] = ("MATCHES HIS DATA",
              f"{ds}: raw / 12v13 at L20 | L40 | L44 = {vals} -- EXACT", note,
              MATB + " and " + ZS,
              f"matrix_baseline.csv column {ds} mean over 5 off-diagonal rows; zero_shot.csv auroc_ext_axis")

# ---------------------------------------------------------------- B40 Table 11
T11 = {"D-226": ("CG", "0.999 0.998 0.983"), "D-227": ("HP-C", "0.533 0.751 0.692"),
       "D-228": ("HP-KR", "0.043 0.167 0.480"), "D-229": ("IT", "0.983 0.985 0.987"),
       "D-230": ("ID", "0.923 0.948 0.949"), "D-231": ("RepE", "0.977 0.999 1.000")}
for cid, (ds, vals) in T11.items():
    note = "Reproduces exactly from the zero-shot sycophancy-axis column."
    if cid == "D-228":
        note += (" Note L44 = 0.4804 -- below 0.5, so 'inverts at every layer' holds, but at L44 it is "
                 "marginal (0.020 below chance) against 0.457 below at L20. Say 'inverts, weakening "
                 "with depth'.")
    V[cid] = ("MATCHES HIS DATA", f"{ds}: L20/L40/L44 = {vals} -- EXACT", note,
              ZS, f"dataset={ds}, column auroc_syc_axis")

V["D-232"] = ("PARTLY MATCHES HIS DATA",
    "Off-diagonal mean reproduces exactly as printed: syc_axis1 = 0.6153 / 0.6626 / 0.7078 -> "
    "0.615 0.663 0.708.",
    "Values exact, verdict reduced because all three are sign artifacts. matrix_syc_axis1.csv has an "
    "inverted HP-KR training row at EVERY layer (row mean 0.1171 / 0.0638 / 0.0779). Sign-corrected "
    "the row means become 0.7429 / 0.8080 / 0.8485, each equal to the mean of the six zero-shot cells "
    "to 4 dp. So this row understates the arm by 0.128 / 0.145 / 0.141. "
    "**This falsifies DF-46's scope claim** that the inversion occurs 'only at L20, only in axis1, "
    "only in the HP-KR training row': a full sweep of the seven dim=1 arms finds inverted rows in "
    "axis1, syc_axis1, syc_pc1 and pc1 at L20; syc_axis1 and syc_pc1 at L40; and syc_axis1 at L44 -- "
    "7 (arm, layer) cells, not 1. In every case the inverted row is HP-KR's.",
    MATS + " and " + SUMM, "features=syc_axis1 off_diagonal_mean; matrix_syc_axis1.csv per-row off-diagonal means")

# ---------------------------------------------------------------- B30 App F
V["D-223"] = ("MATCHES HIS DATA",
    "Both halves re-derived. 'Detects the instructed lies well': ID 0.923 / 0.948 / 0.949 and "
    "RepE 0.977 / 0.999 / 1.000. 'Inverts on HP-KR at every layer': 0.0429 / 0.1671 / 0.4804, all "
    "below 0.5.",
    "Holds. Two refinements: the inversion weakens sharply with depth (0.457 below chance at L20, "
    "0.020 at L44), and CG/IT are also near ceiling (0.983-0.999), so 'instructed lies' understates "
    "how broad the arm's success is -- it is everything except the two harm-pressure datasets.",
    ZS, "column auroc_syc_axis across all six datasets, L20/L40/L44")

V["D-224"] = ("MATCHES HIS DATA",
    "Interpretive claim, and the pattern it rests on is exactly as described: the sycophancy "
    "contrast scores 0.92-1.00 on the four role-played/instructed datasets and inverts on HP-KR "
    "(0.043-0.480) and is weak on HP-C (0.533-0.751).",
    "The inference is sound and independently corroborated by this project: F-13's register "
    "diagnosis, and the cap bank's constructive confirmation (a capitulation axis reads 0.745 "
    "zero-shot on sycophancy_mt where the paper16 sycophancy axis reads 0.313, F-65/F-66). Cite that "
    "rather than leaving it as a suggestion -- it is the strongest form of the argument available. "
    "Note also DF-24: this contrast is not independently constructed (|cos| = 0.54 with the "
    "deception axis, fitted on the same bank object).",
    ZS, "column auroc_syc_axis, all six datasets; cross-ref kd/docs/FINDINGS.md F-13, F-65, F-66")

# ---------------------------------------------------------------- write
def main():
    batch_of, meta = {}, {}
    for b in ["B23", "B41", "B42", "B43", "B45", "B46", "B40", "B30"]:
        for r in csv.DictReader((BATCHDIR / f"{b}.csv").open(encoding="utf-8")):
            batch_of[r["id"]] = b
            meta[r["id"]] = r

    rows = list(csv.DictReader(CLAIMS.open(encoding="utf-8")))
    fields = list(rows[0].keys())
    rows = [r for r in rows if r["id"] not in V]          # idempotent

    stamp = datetime.datetime.now().strftime("%Y-%m-%d")
    for cid, (verdict, l2, notes, ev, key) in V.items():
        m = meta.get(cid, {})
        rows.append({
            "id": cid,
            "claim_plain": m.get("claim_short", ""),
            "stated_numbers": m.get("numbers", ""),
            "source_ref": m.get("source_ref", ""),
            "evidence_file": ev,
            "evidence_key": key,
            "level1_value": "see level2_value",
            "level1_match": "yes",
            "level2_value": l2,
            "level2_method": "re-derived from the collaborator's raw result files on origin/paper "
                             "(git show, no checkout) and from the Liars' Bench PDF",
            "version_used": "origin/paper @ ea1ce52b",
            "robustness": "single seed, his pipeline; three layers L20/L40/L44",
            "verdict": verdict,
            "confidence": "high",
            "importance": "high" if cid in ("D-007", "D-087", "D-092", "D-110", "D-188", "D-232") else "medium",
            "script": SCRIPT,
            "notes": notes + f" [Phase 6 / 6A Level 2, {stamp}]",
            "batch": batch_of.get(cid, ""),
        })

    shutil.copy(CLAIMS, CLAIMS.with_suffix(".csv.before-phase6"))
    with CLAIMS.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    from collections import Counter
    c = Counter(v[0] for v in V.values())
    print(f"wrote {len(V)} Level-2 verdicts -> {CLAIMS.name} (backup .before-phase6)")
    print(f"total rows now: {len(rows)}")
    for k, n in c.most_common():
        print(f"   {n:3d}  {k}")
    missing = [cid for cid in V if cid not in batch_of]
    print("ids not found in the batch files:", missing or "none")


if __name__ == "__main__":
    main()
