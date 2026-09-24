"""Write audit/phase2/claims_batch_W5E-ownerdoc.csv with the columns of CLAIM_INSTRUCTIONS.md."""
import csv, os

COLS = ["id", "claim_plain", "stated_numbers", "source_ref", "evidence_file", "evidence_key",
        "level1_value", "level1_match", "level2_value", "level2_method", "version_used",
        "robustness", "verdict", "confidence", "importance", "script", "notes"]

R = []

R.append(dict(
    id="O-014",
    claim_plain=("What the knowledge-denial persona bank is worth (ext_axis1 minus axis1, zero-shot, "
                 "meaned over the owner's eleven datasets) is +0.055 mean-pooled and +0.059 last-token "
                 "as a layer-mean over all 21 stored layers, and +0.081 at L20 mean-pooled; so the raw "
                 "difference, not the share, is the number to quote."),
    stated_numbers="+0.055; 0.059; +0.081 (also 21/21 and 20/21 layers; share median 63%/50%; range 2-114% and -156% to +145%)",
    source_ref="docs/COLLABORATOR_BRIEFING.md:304 (working tree; :277 in the committed version) - ledger F-99, F-103",
    evidence_file=("C:/Users/user4/expLB/run-crossgen/kd11/banks/collab/matrices/{mean,last}_L*_s42.json "
                   "(21+21 files, mtimes 2026-09-22 00:42 and 02:07); baseline "
                   "C:/Users/user4/expLB/run-crossgen/kd11/matrices/{mean,last}_L*_s42.json"),
    evidence_key=("zeroshot[direction='contrast'] and zeroshot[direction='contrast:collab_extended'], "
                  "meaned over the 11 datasets, per layer; raw = results[feature='raw',probe='paper'] "
                  "110 off-diagonal cells"),
    level1_value="ledger F-99/F-103 state +0.0551 (mean) / +0.0590 (last), range [+0.0019,+0.0814], 21/21 and 20/21",
    level1_match="YES - the briefing rounds the ledger exactly",
    level2_value=("mean pooling: layer-mean +0.0551, min +0.0019 (L0), max +0.0814 (L20), positive 21/21, "
                  "share median 62.9% range 2.2-113.9%, axis1<raw at L60/L64/L68. "
                  "last pooling: layer-mean +0.0590, min -0.0327 (L0), max +0.0848 (L36), positive 20/21, "
                  "share median 50.0% range -155.9% to +144.9%. L20 mean-pooled = +0.08137, L20 last = +0.07501."),
    level2_method=("own reader over all 42 matrix files + 42 baseline files; off-diagonal raw recomputed "
                   "from the 110 cells; no owner analysis code imported"),
    version_used=("lab only (RF-10). collab matrices 2026-09-22 00:42/02:07, BEFORE the 15:40-16:26 wave "
                  "(RF-13). Bank taken from fields inside the matrix: bank=axes_collab_s23.pt, "
                  "bank_name=collab, k=5, datasets=11, seed=42 (RF-16 rule). Baseline kd11 matrices WERE "
                  "in the rewrite set, but the laptop/git copies give the identical off-diagonal raw at "
                  "L20 (0.6289) and L40 (0.6623). No BH q quoted."),
    robustness=("the claim IS the robustness sweep (21 layers x 2 poolings). One seed (42). The adjacent "
                "sentence's HP-KR ranges (axis1 0.087-0.86, ext_axis1 0.57-0.91) are mean-pooled only; "
                "last-token ext_axis1 on HP-KR falls to 0.263 (L32)."),
    verdict="HOLDS",
    confidence="HIGH - all 42 layer cells recomputed from the stored matrices",
    importance="HIGH - this is the number the briefing instructs the collaborator to put in the paper",
    script="audit/scripts/claims/W5E_lab.py (output audit/phase2/out/W5E/W5E_lab.txt); audit/scripts/claims/W5E_local2.py for the baseline cross-check",
    notes=("Exact on every quantity in the paragraph. The sentence is sound: the raw difference moves by "
           "0.004 between poolings while the share moves by 13 points and spans a factor of 50."),
))

R.append(dict(
    id="O-015",
    claim_plain=("On the collaborator's own six datasets at L40, his ext_axis1 zero-shot contrast (0.916) "
                 "beats his own single-source trained baseline (0.732) by +0.184, and both numbers come "
                 "from his files on the same rows, so the comparison is like-for-like."),
    stated_numbers="0.916; 0.732; +0.184",
    source_ref="docs/COLLABORATOR_BRIEFING.md:332 (working tree; :305 in the committed version)",
    evidence_file=("origin/feat/persona-pc-ood:src/probes/deception-detection/results/persona_ood_ext_L40/"
                   "{summary.csv,matrix_baseline.csv,matrix_ext_axis1.csv,zero_shot.csv} (read with git show)"),
    evidence_key="summary.csv rows features='ext_axis1' and 'baseline', column off_diagonal_mean",
    level1_value="ext_axis1 0.9158352734669751; baseline 0.7320325942227636; difference +0.183803",
    level1_match="YES (0.916 / 0.732 / +0.184 to 3 dp)",
    level2_value=("matrix_baseline.csv 30 off-diagonal cells -> 0.732027 (matrix rounded to 4 dp); "
                  "matrix_ext_axis1.csv 30 cells -> 0.915833; zero_shot.csv auroc_ext_axis over the 6 "
                  "datasets -> 0.9158352734669751, i.e. identical to summary.csv. L20 equivalents: "
                  "ext 0.9041173922099752, baseline 0.6477701196654847, margin +0.256347."),
    level2_method="own reader over his committed 6x6 matrices and per-dataset zero-shot column",
    version_used=("collaborator's branch as committed (git show only, no checkout). By the owner's Phase-2 "
                  "decision his pipeline is Level 1 by definition - his activations are gone. No bank or "
                  "BH question arises."),
    robustness=("L20 checked too and consistent. Note for the same paragraph: axis1-baseline is +0.0999 at "
                "L40 (matching the briefing's '+0.100') but only +0.0033 at L20 in THESE files - the "
                "briefing's '+0.143 (L20)' comes from persona_ood/persona_ood_L20, a different run, and is "
                "a separate claim row."),
    verdict="HOLDS",
    confidence="HIGH - both sides recomputed from the underlying matrices",
    importance="HIGH - the replication's headline margin as the briefing tells him to state it",
    script="audit/scripts/claims/W5E_local2.py (output audit/phase2/out/W5E/W5E_local2.txt)",
    notes="'like-for-like' verified: same file, same six datasets, same layer, same protocol.",
))

R.append(dict(
    id="O-022",
    claim_plain=("Aggregating the 110 single-source transfer pairs as a sign-oriented mean, max(a,1-a), "
                 "gives 0.735 - against 0.676 for the plain mean the abstract is compared with and 0.722 "
                 "for the median (L40, mean pooling, paper probe, seed 42, eleven datasets)."),
    stated_numbers="0.735 (row of a table that also gives 0.676, 0.722, 0.897)",
    source_ref="docs/COLLABORATOR_BRIEFING.md:354 (working tree; :327 in the committed version) - ledger F-90",
    evidence_file="C:/Users/user4/expLB/run-premise/audit/audit_L40_s42.json (md5 712bca920f38b6c321fd0eeeed50bc4b, mtime 2026-09-21 20:59:35)",
    evidence_key="baselines.oriented_mean (and .mean_over_sources, .median_over_sources, .best_source_oracle); pairs[110]",
    level1_value="baselines.oriented_mean = 0.735361430160015 (mean 0.6763163062289969, median 0.7219698908893769, oracle 0.8966842158241568)",
    level1_match="YES, all four to 3 dp",
    level2_value=("recomputed from the 110 stored pairs: oriented 0.735361430160015, mean 0.6763163062289967, "
                  "median 0.7219698908893769, oracle 0.8966842158241566; pairs below 0.5 = 21, below 0.4 = 14, "
                  "min 0.05455 (convincing-game -> harm-pressure-knowledge-report)"),
    level2_method="own per-target aggregation of the stored pair AUROCs (mean / median / mean of max(a,1-a) / max), then mean over the 11 targets",
    version_used="lab only (RF-10: run-premise is not on Drive and not in git). mtime 09-21, before the 09-22 wave. No bank. No BH q.",
    robustness=("ONE layer, ONE seed, ONE pooling - there is no audit_L20_s42.json, so none of the four "
                "baselines is depth-checked. The aggregation choice moves the headline margin from +0.178 "
                "(vs mean) to +0.119 (vs oriented) to -0.042 (vs oracle)."),
    verdict="HOLDS",
    confidence="HIGH - recomputed from the 110 stored pairs with an independent max(a,1-a)",
    importance="MEDIUM-HIGH - the pre-emption the briefing and the roadmap both rely on",
    script="audit/scripts/claims/W5E_lab.py + W5E_lab2.py (outputs audit/phase2/out/W5E/W5E_lab.txt, W5E_lab2.txt)",
    notes="Per-target and flat per-pair oriented means coincide because every target has exactly 10 sources.",
))

R.append(dict(
    id="O-038",
    claim_plain=("The roadmap records that a reviewer will ask why the single-source baseline is the mean "
                 "and not the median (0.722) or the sign-oriented mean (0.735), and says to pre-empt it."),
    stated_numbers="0.735 (and 0.722 in the same sentence)",
    source_ref="docs/ROADMAP_TO_DEFENCE.md:28 (unmodified in the working tree) - ledger F-90",
    evidence_file="C:/Users/user4/expLB/run-premise/audit/audit_L40_s42.json",
    evidence_key="baselines.median_over_sources, baselines.oriented_mean",
    level1_value="0.7219698908893769 and 0.735361430160015",
    level1_match="YES",
    level2_value="recomputed from the 110 stored pairs: 0.7219698908893769 and 0.735361430160015 (identical)",
    level2_method="same script as O-022",
    version_used="lab only, mtime 2026-09-21 20:59:35, md5 712bca920f38b6c321fd0eeeed50bc4b. No BH q, no bank.",
    robustness="one layer / one seed / one pooling, as O-022.",
    verdict="HOLDS",
    confidence="HIGH",
    importance="MEDIUM - defence preparation, but it feeds the briefing table the collaborator receives",
    script="audit/scripts/claims/W5E_lab2.py",
    notes=("Completeness gap rather than an error: the line names only the median and the oriented mean. "
           "The fourth baseline in the same file - best source per target, 0.897 - is the one where pooled "
           "training is SIGNIFICANTLY worse (-0.042, CI [-0.073,-0.011]). A reviewer following this line "
           "pre-empts the easy half of the question."),
))

R.append(dict(
    id="O-040",
    claim_plain=("Of claim 9 (the structurally empty 2x2 cell in knowledge-report deception data), only the "
                 "quantification is the project's own: a denial regex reaches 0.91 AUROC on a dataset where "
                 "probes report ~0.87."),
    stated_numbers="0.91; 0.87",
    source_ref="docs/NOVELTY_ASSESSMENT.md:30 (defined at docs/NOVELTY_CHECK_PROMPT.md:103-108; repeated at docs/LAYER_PREREGISTRATION.md:93) - ledger F-70:2489",
    evidence_file=("results/subsets/harm-pressure-knowledge-report.csv (git, mtime 2026-09-09) for the regex; "
                   "run-crossgen/kd11/matrices/*_L*_s42.json (42 files) and docs/papers/liars-bench-notes.md:127-131 "
                   "for the probe search"),
    evidence_key="denial regex vs the deceptive column, Llama-3.3-70B rows; HP-KR diagonal / column mean / row mean for every feature x probe",
    level1_value="regex 0.9125 (Llama rows, n=2,139) and 0.8699 (all 7,068 rows) - F-01's own re-derivation in this audit",
    level1_match="0.91 YES; 0.87 NO SOURCE FOUND",
    level2_value=("no probe reads ~0.87 on HP-KR: raw/paper diagonal 0.9741 (L40 mean), 0.9608 (L20 mean), "
                  "0.9629 (L20 last), 0.9731 (L44 last); as transfer TARGET 0.5478-0.7278; as SOURCE "
                  "0.4665-0.702. A sweep of all 42 kd11 matrices for any HP-KR cell in [0.86,0.88] returns "
                  "only incidental values (pc:unified@2 diagonal 0.873, random@2 diagonal 0.870). The "
                  "benchmark's own mean probe: avg AUROC 0.60, recall 0.00 on HP-KR; upper-bound probe AUROC "
                  "0.91 averaged over all datasets."),
    level2_method="own regex+AUROC over the released CSV (via F-01) and an exhaustive scan of the stored per-cell matrices",
    version_used=("CSV git-tracked, untouched by the 09-22 wave (no results/ CSV was rewritten). kd11 matrices: "
                  "laptop copies, cross-checked equal to the lab at L20/L40 raw off-diagonal. Bank recorded "
                  "inside as axes_vectors_s3_fp32.pt."),
    robustness="the 0.91 is layer-free and bank-free (a text feature vs a label); the negative half was checked at every stored layer and both poolings.",
    verdict="PARTLY HOLDS",
    confidence="HIGH for the regex half; MEDIUM for the negative half (exhaustive over stored matrices and paper notes, but an unindexed log could hold a number I did not find)",
    importance="HIGH",
    script="audit/scripts/claims/W5E_local2.py (output audit/phase2/out/W5E/W5E_local2.txt); regex half from audit/scripts/claims/F-01_F-02_F-16.py",
    notes=("REACHES THE PAPER. 0.91 is verified (0.9125 on the Llama-3.3-70B rows). 'probes report ~0.87' has "
           "no source: the only 0.87 attached to HP-KR anywhere in this project is the project's OWN "
           "superseded regex AUROC over all 7,068 rows (0.8699), which F-01 replaced with 0.9125 and which "
           "kd/docs/DESIGN.md:71, kd/docs/REPORT.md:18 and the 2026-09-18 session transcript all call 'HP-KR "
           "is 87% a lexical task'. If that is the origin, claim 9 compares the regex against ITSELF on two "
           "row sets while calling one of them 'probes'. This sentence is the only part of claim 9 the "
           "novelty review left as the project's own contribution, and it is repeated verbatim in a "
           "PRE-REGISTRATION document (docs/LAYER_PREREGISTRATION.md:93). Needs the owner's call."),
))

R.append(dict(
    id="O-065",
    claim_plain=("The own100 persona bank is 100 AA roles x AA's 240 shared questions x 5 prompt variants, "
                 "about 1,200 rollouts per role of which a median 1,163 are kept, judged by Llama-3.1-70B."),
    stated_numbers="100 roles; 240 questions; 5 prompts; ~1,200 per role; kept median 1,163; judge Llama-3.1-70B",
    source_ref="docs/thesis/persona-coordinates-70b-results.md:377 (unchanged by the audit's edits to this file)",
    evidence_file=("C:/Users/user4/expLB/run/responses (100 .jsonl), C:/Users/user4/expLB/run/scores (99 .json), "
                   "C:/Users/user4/expLB/logs/vectors.log, logs/shard{0,1,2}.log; "
                   "assistant_axis/data/extraction_questions.jsonl (local, 240 lines)"),
    evidence_key="rollout line counts; distinct question_index / prompt_index; count of judge score == 3 per role; 'Role X: N kept' log lines; 'judge=' log field",
    level1_value="no intermediate result file - the table is stated against the raw stores",
    level1_match="n/a",
    level2_value=("100 response files (one is `default`) x exactly 1,200 rows = 120,000; question_index 0-239 "
                  "(240 distinct), prompt_index 0-4; score-3 per role min 548, MEDIAN 1163, max 1200, mean "
                  "1119; logs/vectors.log independently prints 'kept min=548 median=1163 max=1200'; "
                  "logs/shard0-2 record model=meta-llama/Llama-3.3-70B-Instruct judge=meta-llama/"
                  "Llama-3.1-70B-Instruct; assistant_axis/data/extraction_questions.jsonl = 240 lines and is "
                  "1_generate_ndif.py's default --questions_file with --question_count 240."),
    level2_method="own counters over the rollout and score stores + the extraction log's own kept lines; 'kept' definition read from 2_activations_ndif.py:449,459,469 (judge score in --keep_scores, default '3')",
    version_used="lab only. None of these paths is in RF-13's rewrite list. No bank ambiguity (this IS the bank's raw material). No BH q.",
    robustness=("kept counts are uneven - 548 (proofreader) to 1,200 - a 2.2x spread that the median hides; "
                "score distribution over the 99 judged roles is {3: 110,780, 2: 4,154, 1: 3,271, 0: 595}."),
    verdict="PARTLY HOLDS",
    confidence="HIGH - rollouts, scores and the extraction log agree exactly",
    importance="MEDIUM - a methods data table; no result moves",
    script="audit/scripts/claims/W5E_lab3.py (output audit/phase2/out/W5E/W5E_lab3.txt); W5E_lab.py O-065 block",
    notes=("Every number is exact, including the median 1,163 which reproduces twice independently. ONE "
           "error: '100 AA roles' counts `default`, the no-persona anchor, as a role - there are 99 AA roles "
           "+ default. The same table's paper16 row writes '25 paper personas + default', so the "
           "inconsistency is inside one table. A reader also meets a third count, the 45 labelled roles in "
           "own100_role_groups_strict.json, with no note reconciling them. Also confirms the thesis's "
           "'AA's 240 shared questions' really is the Assistant-Axis repo's extraction_questions.jsonl."),
))

R.append(dict(
    id="O-075",
    claim_plain=("At L20 with mean pooling and the mms probe, erasing the paper16 assistant direction costs "
                 "0.006 transfer AUROC (0.652 -> 0.646), random erasures cost ~0, the erased probe's "
                 "dataset-level 95% CI is [0.513, 0.796], and the cost is not significant across datasets "
                 "(BH q 0.52)."),
    stated_numbers="0.652; 0.646; 0.006; ~0.000; [0.513, 0.796]; q 0.52",
    source_ref="docs/thesis/persona-coordinates-70b-results.md:656 (inventory :655)",
    evidence_file="run-coords/q3_q4_bootstrap.json (git working tree; lab copy md5-identical 2f616b3ed18c3d74863635bb0ba4053e, mtime 2026-09-17)",
    evidence_key="q4_table[layer=20,pooling='mean',probe='mms']; comparisons[name='cost of erasing Assistant Axis, L20 mean mms']",
    level1_value=("raw 0.6523819821789706; erase_assistant 0.646230375193732; cost_assistant "
                  "0.0061516069852385336; cost_random 5 values mean +5.23e-05; erased_dataset_ci95 "
                  "[0.5127429605373054, 0.7956263084897787]; dataset_q_bh 0.5151058823529412 "
                  "(dataset_p 0.3284, row_p 0.001)"),
    level1_match="YES - all six cells, and all five sibling rows of the same table (q 0.54, 0.003, 0.16, 0.003, 0.04)",
    level2_value="not recomputed from activations (needs the run-coords passB/passC tensor store); internal consistency checked instead",
    level2_method="own reader over the stored bootstrap file; BH family size counted (26 tests: Q3 14 + Q4 12)",
    version_used=("lab and laptop copies md5-identical (2f616b3e...), mtime 2026-09-17, BEFORE the 09-22 "
                  "15:40-16:26 wave and not in its rewrite set. THE DOC DOES NOT SAY WHICH BH FAMILY: it is "
                  "this file's own 26 comparisons, not the 4,788-test crossgen family RF-13 shifted. "
                  "erase_test.json (the 4-dataset pilot, not this row's source) records bank "
                  "axes_vectors_s3.pt = the OLD paper16 axes (RF-15); RF-15's measurement bounds any "
                  "old-vs-fp32 effect at ~1% of norm, well under the 0.006 cost."),
    robustness=("the table is itself the sweep: cost 0.005-0.018 across 3 layer/pooling settings x 2 probes, "
                "always positive, always small against a 0.28-wide dataset CI. Only FIVE random erasures "
                "back the '~0.000'."),
    verdict="HOLDS",
    confidence="HIGH for Level 1 (6/6 cells plus 5 sibling rows exact); MEDIUM overall (no recompute from activations)",
    importance="MEDIUM - the non-significant row of Q4",
    script="audit/scripts/claims/W5E_local.py (output audit/phase2/out/W5E/W5E_local.txt)",
    notes="The one Level-2 gap in this batch: re-fitting the erased probes needs the ~30 GB passB/passC activation store (CPU-feasible, no model calls).",
))

R.append(dict(
    id="O-082",
    claim_plain=("Dataset identity is 99.3% decodable from the L40 mean-pooled eval activations while the "
                 "deception label is decodable at only +0.032 over its majority baseline, so a high "
                 "within-construct transfer mean is expected whether transfer clusters by construct or by "
                 "provenance."),
    stated_numbers="99.3%; +0.032 (and 7.12 as the cross-reference)",
    source_ref="docs/thesis/persona-coordinates-70b-results.md:702-703, cross-referencing 7.12 (:879-891)",
    evidence_file="run-coords/validate_L40.json (lab md5 d17705873619231f6be457c450f58269 = laptop, mtime 2026-09-16 18:45:12); C:/Users/user4/expLB/logs/validate.log (UTF-16)",
    evidence_key="dataset_acc; label_acc; the log's 'label : 0.646 (majority baseline 0.614) lift +0.032' and 'dataset : 0.993 (majority baseline 0.110) lift +0.883 over 10 datasets'",
    level1_value="dataset_acc = 0.9933774834437085; label_acc = 0.6462472406181016; lift +0.032 is LOG-ONLY (17_validate.py:126-131 never writes it)",
    level1_match="YES - 99.3% in the JSON, +0.032 in the run's own log, mutually consistent (0.6462 - 0.614)",
    level2_value="not recomputed (needs the passB activation store); 7.12's PC label AUROCs 0.704/0.503/0.507 confirmed to be max(a,1-a) of the stored 0.2959/0.4973/0.4927",
    level2_method="own reader over the result JSON + decoded run log; 'lift' definition read from 17_validate.py:126-131",
    version_used="lab and laptop copies md5-identical, mtime 2026-09-16, not in the 09-22 rewrite set. No bank ambiguity (bank axes_vectors_s3_fp32.pt is the script default, used only for the axis-vs-PC cosines). No BH q.",
    robustness="one layer (L40), one pooling, one subsample seed (default_rng(0)), 3-fold CV; no validate_L20/L44 exists, so the decodability gap is not depth-checked.",
    verdict="HOLDS",
    confidence="HIGH for both numbers (result file + producing run's log); MEDIUM that the probe fit itself is right",
    importance="MEDIUM-HIGH - the load-bearing step of the Q5 verdict, reused at line 1101",
    script="audit/scripts/claims/W5E_local.py and W5E_lab2.py (outputs W5E_local.txt, W5E_lab2.txt)",
    notes=("Two precision points for DOC_FIXES. (1) 7.12 says '400 rows per dataset'; 400 is the --cap, and "
           "the realised run was 3,624 rows over 10 datasets (avg 362), with the dataset accuracy computed "
           "over the 10 sets with >=100 rows. (2) 7.5 borrows a 10-dataset pooled measurement into a "
           "9/11-dataset pair-matrix argument without saying so; the inference survives, the row sets differ. "
           "Bonus cross-check from the same log: cos(deception, assistant) = -0.371, matching the -0.37 "
           "contamination figure quoted elsewhere."),
))

R.append(dict(
    id="O-086",
    claim_plain=("Trained cross-dataset transfer with the pc:deception feature, mms probe, mean pooling "
                 "reads 0.524 at L20, 0.517 at L22 and 0.661 at L40, and its own best layer is L40 at 0.661."),
    stated_numbers="0.524; 0.517; 0.661; L40 0.661",
    source_ref="docs/thesis/persona-coordinates-70b-results.md:725 (inventory :724)",
    evidence_file="run-coords/q6_layers.json (md5 01e35cf0b472cb13e05ec871d7ab27c6, lab = laptop) and run-coords/layer_sweep_passB_mms.json (md5 3506b84d16e7b686c094de79c468f708, lab = laptop)",
    evidence_key="rows[source='trained transfer, full corpus', method='pc:deception / mms / mean'].at{20,22,40}, .best_layer, .best; sweep[layer].['pc:deception']",
    level1_value="0.5237417568658953 / 0.5174643904798112 / 0.6607664342269597; best_layer 40, best 0.6607664342269597",
    level1_match="YES, all four to 3 dp",
    level2_value="layer_sweep_passB_mms.json gives the identical three values; argmax over its full 21-layer sweep = L40, 0.6607664342269597",
    level2_method="own reader; argmax recomputed over all 21 stored layers rather than taken from the compiled row",
    version_used="lab and laptop md5-identical for both files; mtimes 2026-09-16/17, not in the 09-22 rewrite set. Bank read from INSIDE the sweep file: axes_vectors_s3_fp32.pt, k=5 (the re-extracted fp32 axes, RF-15's good side). No BH q.",
    robustness="one seed, one probe, one pooling per row - the table's purpose is the within-row layer comparison, which is what was checked. pc:deception loses to raw (0.652/0.646/0.669) at all three layers, as the verdict text says.",
    verdict="HOLDS",
    confidence="HIGH - Level 1 and Level 2 exact, argmax independently recomputed",
    importance="MEDIUM - one row of the layer-choice table",
    script="audit/scripts/claims/W5E_local.py (output audit/phase2/out/W5E/W5E_local.txt)",
    notes=("The compilation step is faithful. Caveat for the table as a whole, not this row: rows come from "
           "different protocols and corpora (trained off-diagonal transfer here, pooled-corpus zero-shot two "
           "rows down, LODO Pass C at the bottom) and the thesis names the provenance only in aggregate "
           "('compiled from every result file'); per-row provenance is in q6_layers.json's source/file fields. "
           "Cross-row comparisons are not like-for-like."),
))

R.append(dict(
    id="O-088",
    claim_plain=("The zero-shot projection onto PC1 of the paper16 DECEPTION axis, mean pooling, reads 0.673 "
                 "at L20, 0.699 at L22 and 0.877 at L40, and its own best layer is L40 at 0.877."),
    stated_numbers="0.673; 0.699; 0.877; L40 0.877",
    source_ref="docs/thesis/persona-coordinates-70b-results.md:727 (inventory :726)",
    evidence_file="run-coords/q6_layers.json and run-coords/dotproducts_mean.json (md5 a12563f5c24afc0a6b001ed26236c6d9, lab = laptop, mtime 2026-09-16 11:09)",
    evidence_key="rows[source='zero-shot, full corpus', method='PC1 / mean']; dotproducts_mean.json matched[layer].pc1",
    level1_value="0.672741905827186 / 0.6985357918446898 / 0.8768949185173067; best_layer 40",
    level1_match="YES, all four to 3 dp",
    level2_value=("dotproducts_mean.json gives the identical values; argmax pc1 over all 21 layers = L40 "
                  "0.8768949185173067 (argmax contrast = L40 0.8630105619589489, matching the row above at "
                  "0.608/0.630/0.863)"),
    level2_method="own reader; axis identity settled from the producing code, 12_dotproduct_layers.py:79 takes bank['pca']['deception']['components'][L,0]",
    version_used="lab and laptop md5-identical; mtime 2026-09-16, not in the 09-22 rewrite set. Bank = run-paper16/axes_vectors_s3_fp32.pt (script default, fp32 re-extraction). RF-15: old-vs-fp32 PCA sign pinning differs at a few layers but NOT at L20/L40/L44. No BH q.",
    robustness=("mean pooling only, one seed. Reconciles with 8.2's 0.881 for the 'pooled corpus L40 mean "
                "(incl. alpaca)' - this file EXCLUDES single-class sets, hence 0.877 vs 0.881 on different "
                "row sets, which neither section says. 8.2 also records that zero-shot deception-axis "
                "readings sit inside the random-direction null in several tests."),
    verdict="HOLDS",
    confidence="HIGH - Level 1 and Level 2 exact; the axis label verified against the producing code, not the column name",
    importance="MEDIUM - a layer-choice row, but the one behind 'moving to L40 is worth ~+0.113 for persona methods'",
    script="audit/scripts/claims/W5E_local.py (output audit/phase2/out/W5E/W5E_local.txt)",
    notes=("Checked specifically against DF-31: here the 'deception PC1' label is CORRECT - the script names "
           "the deception axis explicitly, so this is not another bare-PC1 mis-attribution. Caveat: this row "
           "is a POOLED-CORPUS zero-shot AUROC while the trained rows above it are means over 90-110 "
           "off-diagonal transfer cells; within-row layer comparison is fine, across-row is not."),
))

out = os.path.join(os.path.dirname(__file__), "..", "..", "phase2", "claims_batch_W5E-ownerdoc.csv")
out = os.path.abspath(out)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    for r in R:
        w.writerow({c: r.get(c, "") for c in COLS})
print("wrote", out, len(R), "rows")
