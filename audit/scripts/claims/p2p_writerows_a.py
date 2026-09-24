"""Append the first four P2-pooled rows (F-79, F-80, F-81, F-83) to the batch CSV."""
import csv
import os

P = "audit/phase2/claims_batch_P2-pooled.csv"
COLS = ["id", "claim_plain", "stated_numbers", "source_ref", "evidence_file", "evidence_key",
        "level1_value", "level1_match", "level2_value", "level2_method", "version_used",
        "robustness", "verdict", "confidence", "importance", "script", "notes"]

ROWS = [
    dict(
        id="F-79",
        claim_plain="At L40/mean/paper/seed42 over 11 datasets, a probe trained on the pooled rows of ten datasets and tested on a held-out eleventh scores 0.8545 vs single-source transfer 0.6763 (+0.178); row-budget-matched it still scores 0.8500 (+0.174), so the gain is diversity not data volume.",
        stated_numbers="0.9506; 0.6763; 0.9447; 0.8924; 0.8545; 0.8500; +0.178; +0.174; cos R2 0.244; mahal R2 0.017; n=7336/414/666",
        source_ref="kd/docs/FINDINGS.md:3045",
        evidence_file="audit/cache/labpc/run-premise/premise_L40_mean_s42.json (lab C:/Users/user4/expLB/run-premise/, md5-verified)",
        evidence_key="summary.{single_own_mean,single,pooled_full,pooled_match,pooled_lodo,pooled_lodo_match}; n_train; predictors",
        level1_value="0.950642 / 0.676316 / 0.944721 / 0.892400 / 0.854475 / 0.850015; margins +0.178159 and +0.173698; cos R2 0.2440, mahal R2 0.0165; n_train 666/7336/414",
        level1_match="YES (all |d|<=5e-5)",
        level2_value="not recomputed from activations",
        level2_method="Level 1 only: re-fitting needs run-coords/passB (lab, ~10GB) and would re-run the experiment. Caveat (4)'s class balances VERIFIED from logs/falsify.log (134/2000, 139/2000).",
        version_used="lab copy, mtime 2026-09-21 14:26:55, md5 == 20:56 census; NOT in the 09-22 RF-13 wave; covariance field null (pre-switch); no bank/persona involved",
        robustness="positive at 21/21 layers (F-81); 5/5 seeds +0.173..+0.198 (F-83); bootstrap clears on all 4 comparisons (F-80)",
        verdict="HOLDS",
        confidence="HIGH - every quoted number reproduces from an md5-verified pre-regeneration file; producing script read line by line",
        importance="HIGH",
        script="audit/scripts/claims/p2p_premise_core.py",
        notes="The comparison table's persona numbers (F-03 0.752, F-65 0.714, paper16 pc:unified ~0.672) belong to other claims and were not re-derived."),
    dict(
        id="F-80",
        claim_plain="A paired dataset-level bootstrap (unit=target dataset, n=11, B=10000, seed 42) on the L40 premise run gives pooled-LODO minus single-source +0.1782 [+0.1430,+0.2129] with P(<=0)=0; the volume control (LODO minus row-matched LODO) is +0.0045 p=0.378; pooling wins on 11/11 targets.",
        stated_numbers="+0.1782 [+0.1430,+0.2129]; +0.1737 [+0.1256,+0.2239]; +0.0045 [-0.0234,+0.0325] p=0.378; +0.0962 [+0.0553,+0.1360]; 11/11; mask +0.072; HP-KR 0.972 (+0.279); sycophancy_mt 0.802",
        source_ref="kd/docs/FINDINGS.md:3121",
        evidence_file="audit/cache/labpc/run-premise/premise_L40_mean_s42.json (per_dataset block)",
        evidence_key="per_dataset.*.{single_offdiag,single_own,pooled_lodo,pooled_lodo_match}",
        level1_value="all 11 per-target rows match to 3 dp; no CI exists in any premise JSON",
        level1_match="YES for point estimates; CIs are console-only (in no file)",
        level2_value="+0.1782 [+0.1430,+0.2129] P=0.0000; +0.1737 [+0.1256,+0.2239] P=0.0000; +0.0045 [-0.0234,+0.0325] P=0.3779; +0.0962 [+0.0553,+0.1360] P=0.0000; wins 11/11",
        level2_method="re-implemented 30b_premise_boot.py's bootstrap (sorted names, default_rng(42), integers(0,n,(B,n)), paired diff, 2.5/97.5 pct) over the md5-verified per_dataset values",
        version_used="lab copy 2026-09-21 14:26:55, md5 == census; no BH q-value so RF-13's test-family issue does not apply",
        robustness="per-layer version clears at 21/21 (F-81); on the six-dataset subset the same bootstrap does NOT clear at L20 (+0.0926 [-0.0450,+0.2102] P=0.089) - n=11 is what makes this one tight",
        verdict="HOLDS",
        confidence="HIGH - Level 2: all four intervals reproduced digit-for-digit by an independent implementation",
        importance="HIGH",
        script="audit/scripts/claims/p2p_premise_core.py",
        notes="This recomputation also LOCATES F-83's '+0.143 to +0.213' CI that an earlier agent could not find: it exists in no file at all, only in console output, and is reproduced here exactly."),
    dict(
        id="F-81",
        claim_plain="The pooled-training margin is positive at all 21 stored layers and all 21 clear the dataset-level bootstrap; range +0.128 (L4) to +0.184 (L52); 11/11 targets won at 13 layers, the rest 9-10/11 and all early or very late.",
        stated_numbers="21/21 positive; 21/21 clear; +0.128..+0.184; 13 layers at 11/11; per-layer table L0/8/16/20/28/36/40/52/68/76",
        source_ref="kd/docs/FINDINGS.md:3193",
        evidence_file="audit/cache/labpc/run-premise/premise_L{0,4,...,76}_mean_s42.json (21 files)",
        evidence_key="summary.*; per_dataset.*",
        level1_value="all 10 tabulated layers match; two rounding slips: L40 pooled_lodo is 0.8545 (printed 0.855), L20 margin is +0.1705 (printed +0.171)",
        level1_match="YES except the two rounding slips and the 'targets won' count",
        level2_value="21/21 positive; 21/21 CIs exclude zero; range +0.1279 (L4) to +0.1837 (L52); 11/11 targets won at 12 layers (20,22,24,28,36,40,44,48,52,56,68,72), NOT 13; the nine exceptions include L32 (10/11), which is mid-network",
        level2_method="recomputed the per-layer paired dataset bootstrap (B=10000, seed 42) for all 21 layers; no CI is stored in any premise JSON",
        version_used="21 lab files, mtimes 2026-09-21 14:33-14:42, all md5 == 20:56 census; not in the RF-13 wave",
        robustness="F-83's 5 seeds at L40; the absolute LODO level does move with depth (0.659 at L0 to 0.860 at L36)",
        verdict="PARTLY HOLDS",
        confidence="HIGH - both headline counts confirmed by my own recomputation of all 21 intervals; the failing sub-claim is a direct count",
        importance="HIGH",
        script="audit/scripts/claims/p2p_premise_core.py",
        notes="Corrections owed: 'targets won at 13 of 21 layers' -> 12; 'the rest all early or very late' -> L32, the peak layer, is one of them; L40 lodo prints as 0.855 but is 0.8545."),
    dict(
        id="F-83",
        claim_plain="The pooled-LODO margin at L40 is stable across five split seeds (+0.173 to +0.198, spread 0.025, inside the bootstrap CI width 0.070), and Ying et al.'s Mahalanobis predictor does not reproduce with either covariance (within-class R2 0.023, total 0.017, vs plain cosine 0.244).",
        stated_numbers="+0.1728/+0.1979/+0.1845/+0.1782/+0.1777; spread 0.025; CI +0.143..+0.213; R2 0.244 / 0.023 / 0.017",
        source_ref="kd/docs/FINDINGS.md:3290",
        evidence_file="audit/cache/labpc/run-premise/premise_L40_mean_s{1,7,13,42,99}.json; run-premise/mahal/premise_L40_mean_s42{,_total}.json",
        evidence_key="summary.{single,pooled_lodo}; predictors.{cos,mahal_cos}.r2; covariance",
        level1_value="margins +0.1728/+0.1979/+0.1845/+0.1782/+0.1777 (|d|=0.0000 on all five); cos R2 0.2440; mahal within 0.0230; mahal total 0.0165",
        level1_match="YES",
        level2_value="seed spread 0.0250; recomputed bootstrap CI [+0.14295,+0.21291], width 0.0699",
        level2_method="the CI F-83 quotes exists in no file; recomputed from premise_L40_mean_s42.json per_dataset with the 30b_premise_boot.py procedure",
        version_used="the four extra-seed files and both mahal/ files were written 2026-09-22 15:55-15:57, INSIDE the RF-13 reboot wave and AFTER F-83 was written - yet they reproduce F-83's numbers to 4dp, so that regeneration was value-faithful. All md5 == 20:56 census. premise_L40_mean_s42.json has covariance=null (pre-switch), which is why its mahal R2 equals the _total variant's.",
        robustness="covariance choice cannot affect the seed margins: in 30_premise.py the covariance block runs after all probe fits/AUROCs and consumes no RNG. The Mahalanobis comparison is one layer only (pairwise cosines stored only at L40).",
        verdict="HOLDS",
        confidence="HIGH - all eight quoted numbers reproduce; the one number with no file was independently recomputed and matches",
        importance="HIGH",
        script="audit/scripts/claims/p2p_premise_core.py",
        notes="Answers the briefing's open question: the '+0.143 to +0.213' CI is F-80's console-only interval, not a file value; now verified."),
]

new = not os.path.exists(P)
with open(P, "a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    if new:
        w.writeheader()
    for r in ROWS:
        w.writerow(r)
print("wrote", P, "rows", len(ROWS))
