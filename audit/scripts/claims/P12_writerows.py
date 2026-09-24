"""Write audit/phase2/claims_batch_P12-briefing.csv (batch P12: docs/COLLABORATOR_BRIEFING.md)."""
import csv
import os

H = ["id", "claim_plain", "stated_numbers", "source_ref", "evidence_file", "evidence_key",
     "level1_value", "level1_match", "level2_value", "level2_method", "version_used",
     "robustness", "verdict", "confidence", "importance", "script", "notes"]
R = []

LAB = "C:/Users/user4/expLB"
S_LAB = "audit/scripts/claims/P12_lab.py (output audit/phase2/out/P12/P12_lab.txt)"
S_LAB2 = "audit/scripts/claims/P12_lab2.py (output audit/phase2/out/P12/P12_lab2.txt)"
S_LAB4 = "audit/scripts/claims/P12_lab4.py (output audit/phase2/out/P12/P12_lab4.txt)"
S_LOC = "audit/scripts/claims/P12_local.py (output audit/phase2/out/P12/P12_local.txt)"

R.append(dict(
    id="O-001",
    claim_plain="The paper's sycophancy persona axis scores 0.374 on our capitulation dataset (sycophancy_mt) - below chance and worse than a random direction (0.541) - and the same axis family scores 0.300 on the knowledge-report dataset (HP-KR).",
    stated_numbers="0.374; 0.541; 0.300",
    source_ref="docs/COLLABORATOR_BRIEFING.md:78-79 (ledger F-65 kd/docs/FINDINGS.md:2097-2099 and F-13 :44-48,223-224)",
    evidence_file=f"{LAB}/run-crossgen/kd11/matrices/mean_L40_s42.json (md5 d7616aa50814267e38bd38931e25a6a0, 2026-09-19 20:12; byte-size-identical git working-tree copy used for the same reads); {LAB}/run-crossgen/kd11/passA11/matrices/mean_L40_s42.json; .../passC11/matrices/mean_L40_s42.json; {LAB}/run-paper16/axes_vectors_s3_fp32.pt (md5 76fee3969b41dcabb5a667b00107fd25)",
    evidence_key="zeroshot[direction='PC1'|'contrast'|'PC1:sycophancy'|'contrast:sycophancy'].auroc[sycophancy_mt|harm-pressure-knowledge-report]; results[feature,probe='paper'].auroc column mean (as target) and row mean (as source)",
    level1_value="ledger F-65:2099 '0.374 ... worse than a random direction (0.541)'; F-13:220,223 '0.300' and 'SYC 0.541/0.633' - the briefing restates the ledger exactly",
    level1_match="YES - every number and the (mis)attribution are copied from F-65/F-13",
    level2_value="zero-shot PC1 on sycophancy_mt = 0.3740 and contrast = 0.4089, but BOTH are the DECEPTION axis (27_crossgen.py:411 picks _zs_axis='deception'); the paper16 SYCOPHANCY axis reads 0.3061 (contrast) / 0.2996 (PC1) on sycophancy_mt and 0.1625 / 0.2768 on HP-KR. random = 0.5413 is the TRAINED random feature, sycophancy_mt as target. axes (paper16 stack, k=5, trained) = 0.3004 as source on full HP-KR, 0.3551 as target on sycophancy_mt.",
    level2_method="own reader over the stored per-cell AUROC matrices; axis identity resolved from the bank's own keys (axes=['assistant','deception','sycophancy','unified']) plus the _zs_axis selection in src/assistant_axis_ndif/27_crossgen.py:411-428",
    version_used="lab originals, mtimes 2026-09-19 20:12 (passB kd11) and 2026-09-21 12:20/12:24 (passA11/passC11) - all before the 09-22 15:40-16:26 re-run wave (RF-13). No BH q quoted. Bank resolved by content (axes_vectors_s3_fp32.pt), not by name.",
    robustness="the three numbers come from three DIFFERENT protocols: 0.374 zero-shot, 0.541 trained-as-target, 0.300 trained-as-source. The like-for-like trained pair is axes 0.3551 vs random 0.5413 (target) and 0.3787 vs 0.6327 (source), so 'worse than a random direction' survives a matched comparison. No zero-shot random direction exists in these stores.",
    verdict="PARTLY HOLDS",
    confidence="HIGH - every value re-derived from the stored per-cell matrices; the axis identity is fixed by the producing script and by the bank's own key list",
    importance="HIGH - it is the briefing's Claim 1 and the sentence the collaborator is asked to put in the paper",
    script=f"{S_LOC}; {S_LAB}; {S_LAB4}",
    notes="ATTRIBUTION ERROR, inherited from F-65. 0.374 is the paper16 DECEPTION axis's zero-shot PC1, not the sycophancy axis: in these files the bare 'contrast'/'PC1' rows are the bank's primary behaviour contrast, which 27_crossgen.py hardcodes to 'deception'. The paper16 sycophancy axis reads 0.3061/0.2996 on sycophancy_mt - i.e. WORSE, so the argument is unharmed and the number is wrong. Second issue: 0.374 (zero-shot) is compared against 0.541 (a trained probe on a random direction); the matched trained comparison is 0.3551 vs 0.5413. Third: 0.300 is the trained paper16 'axes' stack as SOURCE on full HP-KR, not a sycophancy-axis or zero-shot number (the sycophancy axis on HP-KR reads 0.1625/0.2768)."))

R.append(dict(
    id="O-002",
    claim_plain="For the capitulation bank the filtered (judge score-3 only) and unfiltered banks give results that differ by 0.0004, so the judge filter does not drive the cap result.",
    stated_numbers="0.0004 (plus the same table's +0.112, 95% CI [+0.050, +0.185], and 'positive at 21 of 21 layers')",
    source_ref="docs/COLLABORATOR_BRIEFING.md:104-110 (ledger F-82 kd/docs/FINDINGS.md:3257-3271; F-66 :2221; F-77 :2884)",
    evidence_file=f"{LAB}/run-crossgen/kd11/banks/cap_s23/matrices/mean_L*_s42.json (21 files, mtimes 2026-09-21 03:48-13:41) and .../banks/cap_all/matrices/mean_L*_s42.json (21 files, 2026-09-21)",
    evidence_key="results[feature,probe='paper'].auroc, the 10 off-diagonal cells of the sycophancy_mt COLUMN; margin = pc:capitulation@2 minus datapca@2 (and minus raw)",
    level1_value="F-82's table: margin at L40 +0.1542 (s23) vs +0.1538 (all), 21/21 both, CI [+0.0913,+0.2113] vs [+0.0907,+0.2118]",
    level1_match="YES - the briefing's 0.0004 is F-82's difference, and the +0.112 / [+0.050,+0.185] / 21-of-21 rows are F-66 and F-77",
    level2_value="L40: pc:capitulation@2 s23 0.7136 vs all 0.7132 (|diff| 0.0004); margin vs datapca@2 +0.1542 vs +0.1538 (|diff| 0.0004); margin vs raw +0.1123 vs +0.1118 (|diff| 0.0004). raw/datapca@2/random@2/length identical to 4 dp between banks. Own bootstrap (B=10,000, seed 42, resampling the 10 source datasets): -raw +0.1123 CI [+0.0502,+0.1847] P(<=0) 0.0000; -datapca@2 +0.1542 CI [+0.0913,+0.2113] P(<=0) 0.0000. Positive at 21/21 layers against BOTH raw and datapca@2, for both banks.",
    level2_method="own loader + own paired dataset bootstrap (numpy default_rng(42), index resampling of the 10 sources), not importing cap/scripts/cap_layer_profile.py",
    version_used="lab only (run-cap / cap banks are in no branch; RF-10). Matrices dated 2026-09-21 03:48-13:41, before the 09-22 wave. Bank files resolved by path; no BH q quoted.",
    robustness="argmax median L36 (s23) / L32 (all), 95% [L28,L60] for both; L40 is the argmax in only 3.9% / 3.1% of resamples, so the band should be quoted rather than L40",
    verdict="HOLDS",
    confidence="HIGH - both banks recomputed cell by cell and the bootstrap reimplemented independently",
    importance="MEDIUM - it is a robustness row of Claim 2, not the claim itself",
    script=S_LAB,
    notes="The 0.0004 is exact, and the rest of the Claim-2 table reproduces to 4 dp. One caveat the briefing does not carry: the batch P3 check of F-65 found that pc:cap_only@2 - the feature the 'basis needs the holding personas' argument rests on - differs between filters by 0.0142 at L40, not 0.0004. 'The judge filter does not drive it' is established for the headline feature; it is not true of every feature in the bank."))

R.append(dict(
    id="O-003",
    claim_plain="Under leave-one-CLUSTER-out (every dataset sharing the target's provenance removed from training) pooled training still beats single-source transfer by +0.129 (L20), +0.131 (L32), +0.125 (L40) over all eleven targets and +0.157 / +0.117 / +0.150 over the nine uncontaminated targets, positive on 8-11 of 11 at every layer.",
    stated_numbers="+0.129; +0.131; +0.125; +0.157; +0.117; +0.150; 8-11 of 11",
    source_ref="docs/COLLABORATOR_BRIEFING.md:135-143 (ledger F-108 kd/docs/FINDINGS.md:5140-5186)",
    evidence_file=f"{LAB}/run-premise/clustered/clustered_L{{20,32,40}}_mean_s42.json (md5 90eda6baec4ab35539fa9f0301baa068 / 73686f8ab57716d48a9ac003664f8af2 / 524c1340b2f7bd557cdce51aa84cabf7, mtimes 2026-09-22 16:58:55 / 16:59:17 / 16:59:39)",
    evidence_key="summary.all.gain_loco, summary.clean.gain_loco, summary.*.loco_positive; recomputed from per_dataset[t].pooled_loco - per_dataset[t].single_offdiag",
    level1_value="stored summary: L20 all +0.12894 clean +0.15710 (loco_positive 10 / 9); L32 all +0.13087 clean +0.11681 (11 / 9); L40 all +0.12544 clean +0.15039 (10 / 8)",
    level1_match="YES for the six gains (briefing rounds to 3 dp); NO for the count phrase - see notes",
    level2_value="recomputed from the 11 per-target cells: L20 +0.1289 (all, 10/11) / +0.1571 (clean 9, 9/9); L32 +0.1309 (11/11) / +0.1168 (9/9); L40 +0.1254 (10/11) / +0.1504 (8/9). Clusters = lb_harm{HP-C,HP-KR,HP-KR-matched}, lb_scenario{ID,CG,IT}, apollo{ai_liar,roleplaying,sandbagging_v2}, mask, ours{sycophancy_mt}; leaked = HP-KR, HP-KR-matched.",
    level2_method="own script: mean over the per-target pooled_loco and single_offdiag entries, and an independent count of positive targets; the file's own summary block is printed alongside and agrees",
    version_used="lab only (run-premise is on no branch; RF-10). The clustered files were written 2026-09-22 16:58-16:59, i.e. AFTER the 15:40-16:26 boot-trigger wave (RF-13) - they are the F-108 run itself, not a re-run of an older result. No BH q quoted.",
    robustness="only three layers, one seed, one probe (paper), mean pooling - as F-108's own caveats say. The counter-intuitive clean>all ordering reproduces (L20 +0.157 vs +0.129; L40 +0.150 vs +0.125). HP-KR collapses 0.9674 (LODO) -> 0.6492 (LOCO) at L40 and 0.7090 -> 0.4311 at L20.",
    verdict="HOLDS (with one count phrase to fix)",
    confidence="HIGH - every gain recomputed from the per-target cells, and the file's stored summary agrees",
    importance="HIGH - this is the corrected form of the briefing's strongest claim (Claim 3)",
    script=S_LAB2,
    notes="All six gains are exact. The phrase 'positive on 8-11 of 11 at every layer' mixes two denominators: over all eleven targets it is 10/11, 11/11, 10/11; over the nine clean targets it is 9/9, 9/9, 8/9. '8 of 11' is not measured anywhere. F-108's own table states both correctly; the briefing's one-line compression is what loses it."))

R.append(dict(
    id="O-004",
    claim_plain="The knowledge-report matched-slice persona margin at the pre-specified layer does not clear a dataset-level bootstrap: 95% CI [-0.069, +0.189].",
    stated_numbers="[-0.069, +0.189]",
    source_ref="docs/COLLABORATOR_BRIEFING.md:171 (ledger F-76 kd/docs/FINDINGS.md:2846)",
    evidence_file=f"{LAB}/run-crossgen/kd11/passA11/matrices/mean_L*_s42.json and .../passA11/banks/kd_s23/matrices/mean_L*_s42.json (160 files, mtimes 2026-09-21 12:20-13:28, originals)",
    evidence_key="80-layer profile of the matched-slice column; margin = axes minus datapca at L40; paired dataset bootstrap over the 10 transfer datasets, B=10,000, seed 42",
    level1_value="F-76 states +0.0655, CI [-0.0693, +0.1887], P(<=0) 0.161 - the briefing rounds to [-0.069, +0.189]",
    level1_match="YES",
    level2_value="already re-derived TWICE in this audit: batch P7 measured +0.0655, CI [-0.0693,+0.1887], P 0.1611; the lead's independent QA bootstrap (audit/scripts/qa_p7.py) measured +0.0656, CI [-0.0738,+0.1904], P 0.1646. Both agree the interval contains zero.",
    level2_method="not recomputed a third time in this batch, by the lead's instruction; the two existing independent re-derivations used different loaders and different implementations of the same design",
    version_used="lab originals (passA11, 2026-09-21 12:20-13:28), untouched by the 09-22 wave; no BH q quoted",
    robustness="the verdict is robust to the control (datapca and raw both fail) and to the implementation (two bootstraps, different code); the argmax interval [L5,L79] and the label-permutation p=0.38 (O-005) point the same way",
    verdict="HOLDS",
    confidence="HIGH - settled by two independent re-derivations recorded in audit/claims/F-76.md and audit/PROGRESS.md",
    importance="HIGH - it is the project's own strongest self-correction and the briefing reports it correctly",
    script="audit/scripts/claims/P7_lab.py f7576; audit/scripts/qa_p7.py (both pre-existing in this audit)",
    notes="No re-derivation owed. The adjacent row of the same table, '95% interval [L5, L79] - 93% of the network', is also right: F-76 measures 74 of 80 layers = 92.5%."))

R.append(dict(
    id="O-005",
    claim_plain="The matched-slice persona band does not survive a cluster-based label permutation test: p = 0.38.",
    stated_numbers="0.38",
    source_ref="docs/COLLABORATOR_BRIEFING.md:173 (ledger F-84 kd/docs/FINDINGS.md:3357-3394)",
    evidence_file=f"{LAB}/run-premise/cluster_perm_kd_matched.json (md5 e43dba906883c2876d6decc40fe508ad, 5,168 B, mtime 2026-09-22 16:07:02)",
    evidence_key="keys p, B, observed_max_mass, observed_clusters, observed_margins, null_max_mass",
    level1_value="file: p = 0.38, B = 200, observed_max_mass = 0.91552, observed_clusters = [0.91552, 0.05881], 20 layers",
    level1_match="YES - exact",
    level2_value="p = 0.38 read verbatim from the stored artifact; the null distribution (200 draws of null_max_mass) spans 0.0000 to 3.9097, so the observed 0.9155 sits well inside it. Not re-simulated (B=200 permutations with probe refits is a re-run, not a recomputation).",
    level2_method="Level 1 only for the p itself: the artifact is the output of src/assistant_axis_ndif/30c_cluster_perm.py and the permutation loop was not re-executed. The stored null and observed statistics were read and checked for internal consistency (p = fraction of null_max_mass >= 0.9155).",
    version_used="lab only. WARNING: the file's mtime 2026-09-22 16:07:02 falls INSIDE the RF-13 boot-trigger re-run window (15:40-16:26), while F-84 is dated 2026-09-21 - so the artifact read here is a regenerated copy. Its p matches F-84's stated 0.38, and the run is seeded, so no discrepancy is visible; there is no pre-wave copy to compare against.",
    robustness="B = 200 is modest, as F-84's own caveat says. The briefing uses 'p = 0.38' TWICE, at :146 and :173 - they are two different tests (see notes), not a copy error.",
    verdict="HOLDS",
    confidence="MEDIUM-HIGH - the number is exact in its artifact, but the artifact was rewritten inside the 09-22 re-run window and the permutation loop was not re-executed",
    importance="MEDIUM - a supporting row of the 'what did not survive' table",
    script=S_LAB,
    notes="Verified. Two things found on the way. (1) The briefing's OTHER p = 0.38, at :146 ('pooled ~6,700 rows and ~666 rows are statistically indistinguishable'), is a different test and its stored p is ONE-SIDED: 30b_premise_boot.py:62 computes p = (bs <= 0).mean(). My own two-sided bootstrap of pooled_lodo - pooled_lodo_match gives +0.0045, 95% CI [-0.0235, +0.0331], two-sided p 0.744 (one-sided 0.372) - same conclusion, but the quoted 0.38 is a one-sided tail and the CI is the statistic that supports 'indistinguishable'. (2) The same sentence's '~666 rows': the realised row-budget-matched LODO training pool is 412-500 rows (mean 449) per target; 666 is n_train.single_mean, the budget the control is CAPPED at (30_premise.py:167 per_o = budget // 10, then min(per_o, len(tr))). The control is therefore stricter than advertised."))

R.append(dict(
    id="O-006",
    claim_plain="The abstract's 0.904 / 0.916 are the ext_axis1 off_diagonal_mean values 0.9041173922 and 0.9158352735 in the collaborator's persona_ood_ext_L20/L40 summary.csv, identical to the mean of auroc_ext_axis in the matching zero_shot.csv to 1e-16; ext_axis1 is the 12-vs-13 extended contrast whose ten extra roles are the owner's knowledge-denial bank, matching kd/personas/kd_role_groups.json name for name, over the base bank paper16_L20_sharedquestions/paper16_L20.pt.",
    stated_numbers="0.9041173922; 0.9158352735; 1e-16; 8+4 vs 7+6",
    source_ref="docs/COLLABORATOR_BRIEFING.md:199-216 (ledger F-97 kd/docs/FINDINGS.md:4384-4466)",
    evidence_file="origin/feat/persona-pc-ood (and f386a72): src/probes/deception-detection/results/persona_ood_ext_L{20,40}/{summary.csv,zero_shot.csv}; .../deception_detection/persona_pcs.py; .../deception_detection/scripts/persona_ood.py - all read with `git show`, no checkout. Plus kd/personas/kd_role_groups.json (asix working tree).",
    evidence_key="summary.csv row features='ext_axis1' column off_diagonal_mean; mean of zero_shot.csv column auroc_ext_axis over the 6 scored datasets (the Alpaca control row XXXX-*__plain has empty AUROCs and is dropped); persona_pcs.py:31 BANK_PATH; persona_pcs.py axes['extended']",
    level1_value="summary.csv verbatim: L20 0.9041173922099752, L40 0.9158352734669751 - identical at f386a72 and at the branch head 45b16d9",
    level1_match="YES - the briefing truncates both to 10 decimal places and loses nothing",
    level2_value="mean(auroc_ext_axis) = 0.904117392209975 (L20) and 0.9158352734669751 (L40); |difference| from off_diagonal_mean = 2.220e-16 (L20) and exactly 0.000 (L40). Arms: persona_pcs.py axes['extended'] = mean_of(DECEPTION_PERSONAS + KNOWLEDGE_DENIAL_PERSONAS) - mean_of(HONESTY_PERSONAS + KNOWLEDGE_CANDOUR_PERSONAS), i.e. 8+4 = 12 vs 7+6 = 13. His 10 extra role names are SET-IDENTICAL to the owner's knowledge_denial (4) + sincere_ignorance (3) + plain_assertion (3); symmetric difference empty. persona_pcs.py line 31 is exactly BANK_PATH = REPO_ROOT / 'paper16_L20_sharedquestions' / 'paper16_L20.pt'.",
    level2_method="own script: every mean recomputed from the per-dataset rows of zero_shot.csv rather than read off summary.csv; role names parsed out of his source and compared as sets against kd_role_groups.json",
    version_used="collaborator's branch, read out of git at both f386a72 (the commit the ledger cites) and the current head 45b16d9 - the persona_ood_ext results are identical at both. His activations are gone, so the AUROCs themselves are Level 1 by the owner's decision (RF-2).",
    robustness="the same identity holds for denial_axis1 (|diff| 1.11e-16 at L20, 0.0 at L40) but NOT for axis1 at L20 (|diff| 1.398e-01), because one training set disagrees on the sign - so 'the trained off-diagonal equals the zero-shot mean' is a property of sign-consistent 1-d features, not a general one",
    verdict="HOLDS",
    confidence="HIGH - digits read verbatim from his committed CSV and every mean recomputed from the per-dataset rows",
    importance="HIGH - it is the correction the briefing exists to deliver",
    script=S_LOC,
    notes="Digits and attribution both check out. Three wording fixes for the sentence around them. (1) The briefing cites `scripts/persona_ood.py::extended_arms()`; the file is at deception_detection/scripts/persona_ood.py, and extended_arms() re-FITS the geometry - the axis itself is defined in deception_detection/persona_pcs.py (axes['extended']). (2) 'they match kd/personas/kd_role_groups.json name for name' is true of the NAMES but not of the GROUPING: his KNOWLEDGE_CANDOUR_PERSONAS merges the owner's two distinct honest groups, sincere_ignorance (3) and plain_assertion (3), into one six-role arm. (3) 'identical ... to 1e-16' is exact at L40 (0.0) and 2.2e-16 at L20, i.e. 1 ulp above the stated bound - harmless, but 'to within 1e-15' would be safer."))

R.append(dict(
    id="O-007",
    claim_plain="On the collaborator's six datasets, zero-shot, the paper's 8-vs-7 contrast beats his trained single-source baseline by +0.143 at layer 20 and +0.100 at layer 40.",
    stated_numbers="+0.143; +0.100",
    source_ref="docs/COLLABORATOR_BRIEFING.md:220-228 (ledger F-97 kd/docs/FINDINGS.md:4384-4466)",
    evidence_file="origin/feat/persona-pc-ood (and f386a72): src/probes/deception-detection/results/persona_ood_ext_L{20,40}/{zero_shot.csv,summary.csv,matrix_baseline.csv}, read with `git show`",
    evidence_key="mean of zero_shot.csv auroc_axis over the 6 scored datasets, minus summary.csv row features='baseline' column off_diagonal_mean",
    level1_value="F-97's table: baseline 0.6478 / 0.7320, axis1 0.7909 / 0.8319, margins +0.143 / +0.100",
    level1_match="YES",
    level2_value="L20 axis1 zero-shot mean 0.7908803603 - baseline 0.6477701197 = +0.1431; L40 0.8318937061 - 0.7320 = +0.0999. Extended-contrast margins +0.2563 / +0.1838; kd personas worth +0.1132 / +0.0839 = 44.2% / 45.7% of the headline margin. Dropping repe_honesty against the 5-dataset baseline (0.6085 / 0.7064): axis1 margin +0.1418 / +0.0919, kd worth +0.1364 / +0.1007 - the briefing's '+0.142 / +0.092' and '+0.136 / +0.101'.",
    level2_method="own script: the baseline recomputed independently as the off-diagonal mean of matrix_baseline.csv (n=30 over six datasets, n=20 over five) and cross-checked against summary.csv's own value; the zero-shot means recomputed from the per-dataset rows",
    version_used="collaborator's branch at f386a72 and at head 45b16d9 - identical. Level 1 by the owner's decision on his activations (RF-2).",
    robustness="the whole §5(a) table re-derives: 0.6478/0.7320 baseline, 0.7909/0.8319 axis1, 0.9041/0.9158 ext_axis1, +0.256/+0.184, 44%/46% share. The drop-repe_honesty variant needs the 5-dataset baseline (0.6085/0.7064); against the 6-dataset baseline the margins would be +0.1026/+0.0662, so the briefing's reading is the correct one.",
    verdict="HOLDS",
    confidence="HIGH - recomputed from the per-dataset rows and from the raw transfer matrix, at two commits",
    importance="HIGH - it is the number that separates the paper's own contrast from the abstract's headline",
    script=S_LOC,
    notes="Exact. The one thing worth adding to the briefing is which baseline the drop-repe_honesty row uses: +0.142 / +0.092 are against a baseline recomputed over the five remaining datasets (0.6085 / 0.7064), not against the six-dataset baseline printed one row above."))

R.append(dict(
    id="O-009",
    claim_plain="Rebuilt on our eleven datasets, both of the collaborator's zero-shot arms lose to a pooled leave-one-dataset-out probe - by 0.064 at layer 20 and 0.055 at layer 40.",
    stated_numbers="0.064; 0.055",
    source_ref="docs/COLLABORATOR_BRIEFING.md:243-252 (ledger F-98 kd/docs/FINDINGS.md:4469-4542)",
    evidence_file=f"{LAB}/run-crossgen/kd11/banks/collab/matrices/mean_L{{20,40}}_s42.json (md5 .../3f9fe86b652dda3aa89b18d9466371c5 at L40, mtimes 2026-09-22 00:33); {LAB}/run-premise/premise_L{{20,40}}_mean_s42.json (md5 d975eb33f26f328bede48aa7f9b6e03a / e5ddaaa13ea6aa5a4b97bbafa38592ba, 2026-09-21 14:35 / 14:26); baseline {LAB}/run-crossgen/kd11/matrices/mean_L{{20,40}}_s42.json",
    evidence_key="zeroshot['contrast'] and zeroshot['contrast:collab_extended'] means over the 11 datasets; summary.pooled_lodo from the premise files",
    level1_value="F-98's table: L20 pooled 0.8027 vs ext 0.7386; L40 0.8545 vs 0.7996 - i.e. -0.064 / -0.055",
    level1_match="YES",
    level2_value="L20: pooled LODO 0.8027; axis1 0.6572 (gap +0.1455); ext_axis1 0.7386 (gap +0.0641). L40: pooled LODO 0.8545; axis1 0.7325 (gap +0.1219); ext_axis1 0.7996 (gap +0.0548).",
    level2_method="own reader: zero-shot means recomputed from the 11 per-dataset entries, pooled figure read from the premise summary key, raw baseline recomputed as the mean of the 110 off-diagonal cells",
    version_used="lab only. collab matrices 2026-09-22 00:33 (the 00:52-01:33 sweep window, before the 15:40 wave); premise files 2026-09-21 14:26/14:35, untouched by the wave. No BH q quoted. Bank composition was verified inside axes_collab_s23.pt by batch P3 (12 v 13, vectors_dir run-kd/vectors_s23).",
    robustness="pooled LODO beats ext_axis1 at all 21 stored layers under mean pooling, by +0.0396 (L0) to +0.1202 (L60); the gap is smallest at L40 (+0.0548) and L22 (+0.0482), so quoting L20/L40 is if anything conservative",
    verdict="HOLDS",
    confidence="HIGH - both sides recomputed from the per-dataset raw values on the lab copies",
    importance="HIGH - it is the briefing's central argument against the abstract's framing",
    script=S_LAB,
    notes="Both numbers exact. Wording only: 0.064 and 0.055 are the gaps for the EXTENDED arm alone; the paper's 8v7 arm loses by 0.146 (L20) and 0.122 (L40). 'Both of your arms lose ... by 0.064 and 0.055' reads as if the pair covers both arms; it is the smaller of the two gaps in each case."))

R.append(dict(
    id="O-011",
    claim_plain="The knowledge-denial bank is worth +0.055 mean-pooled and +0.059 last-token (ext_axis1 minus axis1, zero-shot, our eleven datasets), positive at 21/21 and 20/21 layers respectively, the single exception being L0, the embedding layer.",
    stated_numbers="+0.055; +0.059; 21/21; 20/21; L0",
    source_ref="docs/COLLABORATOR_BRIEFING.md:263-270 (ledger F-99 kd/docs/FINDINGS.md covers the mean-pooled half only)",
    evidence_file=f"{LAB}/run-crossgen/kd11/banks/collab/matrices/{{mean,last}}_L*_s42.json (42 files, mtimes 2026-09-22 00:33-00:49 and 02:07-02:11); baseline {LAB}/run-crossgen/kd11/matrices/{{mean,last}}_L*_s42.json (2026-09-19 20:12-20:58)",
    evidence_key="zeroshot['contrast:collab_extended'].auroc mean minus zeroshot['contrast'].auroc mean, over the 11 datasets, at each of the 21 stored layers, under each pooling",
    level1_value="F-99 states the mean-pooled half (+0.0551, 21 of 21, range [+0.0019 L0, +0.0814 L20]); the LAST-TOKEN half is stated only in the briefing - audit/claims/F-99.md records 'Last pooling was not swept'",
    level1_match="PARTIAL - the mean-pooled half matches F-99 exactly; the last-token half has no ledger entry behind it and is checked here for the first time",
    level2_value="mean pooling: ext - axis1 positive at 21/21 layers, mean +0.0551, range [+0.0019 (L0), +0.0814 (L20)]. last-token: positive at 20/21, mean +0.0590, range [-0.0327 (L0), +0.0848 (L36)]; the only non-positive layer is L0.",
    level2_method="own reader over all 42 collab matrices, recomputing each zero-shot mean from the 11 per-dataset entries; no summary line read",
    version_used="lab only; the last_L* collab matrices are dated 2026-09-22 02:07-02:11 and the mean_L* 00:33-00:49, all before the 15:40-16:26 wave (RF-13). No BH q quoted.",
    robustness="the claim's own robustness check: under last pooling the peak moves from L20 (mean) to L36, and the range widens; but the size of the effect is stable (+0.0551 vs +0.0590) and the sign holds everywhere except the embedding layer",
    verdict="HOLDS",
    confidence="HIGH - the mean-pooled half reproduces F-99 exactly and the last-token half is re-derived here from the raw matrices",
    importance="HIGH - it is the layer/pooling robustness that the briefing offers in place of a two-layer quote",
    script=S_LAB,
    notes="Exact, including 'the single exception is L0'. NOTE FOR THE LEDGER, not the briefing: the last-token sweep has no F-entry - F-99 is explicitly a mean-pooling claim (audit/claims/F-99.md) - so this briefing line is currently the only place the 20/21 result is written down."))

R.append(dict(
    id="O-012",
    claim_plain="The SHARE of the margin over raw attributable to the knowledge-denial bank is unstable: it reads 2%-114% across depth with a median of 63% mean-pooled and 50% last-token, while the underlying difference barely moves (+0.055 vs +0.059).",
    stated_numbers="2%; 114%; 63%; 50%; +0.055; +0.059",
    source_ref="docs/COLLABORATOR_BRIEFING.md:272-278 (ledger F-99, mean-pooled half only)",
    evidence_file=f"{LAB}/run-crossgen/kd11/banks/collab/matrices/{{mean,last}}_L*_s42.json; raw baseline {LAB}/run-crossgen/kd11/matrices/{{mean,last}}_L*_s42.json",
    evidence_key="share = (ext_axis1 - axis1) / (ext_axis1 - raw), where raw is the off-diagonal mean of results[feature='raw',probe='paper'] over the 110 pairs, at each of the 21 layers",
    level1_value="F-99 states median 62.9%, mean 63.8%, range [2.2%, 113.9%] for mean pooling; it states nothing for last-token",
    level1_match="PARTIAL - the mean-pooled figures match F-99 exactly; the 50% last-token median has no ledger entry",
    level2_value="mean pooling: median 62.9%, mean 63.8%, range [2.2% (L0), 113.9% (L68)]. last-token: median 50.0%, mean 48.0%, range [-155.9% (L0), +144.9% (L8)]. Underlying difference +0.0551 (mean) vs +0.0590 (last).",
    level2_method="own reader: every share recomputed from the three recomputed quantities at each layer; the raw baseline recomputed from the 110 off-diagonal cells of the matching baseline matrix under the SAME pooling",
    version_used="lab only, same files and mtimes as O-011, all pre-wave",
    robustness="the instability the claim asserts is real and is larger than stated once pooling is varied: under last-token pooling the share goes NEGATIVE at L0 (-155.9%) and exceeds 144% at L8, because the denominator (ext - raw) shrinks where the arms sit near raw",
    verdict="HOLDS",
    confidence="HIGH - both poolings recomputed from the raw matrices",
    importance="MEDIUM-HIGH - it is the caveat that tells the collaborator not to quote a share",
    script=S_LAB,
    notes="63% and 50% are exact (62.9% and 50.0%). One scoping point: the range '2%-114%' is the MEAN-POOLED range; the sentence puts it before 'mean-pooled', so a reader can take it as covering both poolings. Under last-token the range is -156% to +145%. Adding 'mean-pooled' before the range, or giving both ranges, would make the sentence airtight - and the wider last-token range strengthens the point the sentence is making."))

out = os.path.join(os.path.dirname(__file__), "..", "..", "phase2",
                   "claims_batch_P12-briefing.csv")
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=H)
    w.writeheader()
    for r in R:
        w.writerow(r)
print("wrote", os.path.abspath(out), len(R), "rows")
