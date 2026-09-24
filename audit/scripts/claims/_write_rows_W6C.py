"""Write audit/phase2/claims_batch_W6C-briefing.csv from the per-claim dicts below.

Run after the four W6C scripts have produced their outputs under audit/phase2/out/W6C/.
"""
import csv
import io
import os

COLS = ["id", "claim_plain", "stated_numbers", "source_ref", "evidence_file", "evidence_key",
        "level1_value", "level1_match", "level2_value", "level2_method", "version_used",
        "robustness", "verdict", "confidence", "importance", "script", "notes"]

COLLAB = ("origin/feat/persona-pc-ood (head 45b16d9) and f386a72, read with `git show`: "
          "src/probes/deception-detection/results/persona_ood_ext_L{20,40,44}/"
          "{summary.csv,zero_shot.csv,matrix_baseline.csv}")
SC1 = "audit/scripts/claims/W6C_collab.py (out audit/phase2/out/W6C/W6C_collab.txt)"
SC2 = "audit/scripts/claims/W6C_premise.py (out audit/phase2/out/W6C/W6C_premise.txt)"
SC3 = "audit/scripts/claims/W6C_collab2.py (out audit/phase2/out/W6C/W6C_collab2.txt)"
SC4 = "audit/scripts/claims/W6C_o018.py (out audit/phase2/out/W6C/W6C_o018.txt)"
SC5 = "audit/scripts/claims/W6C_eleven.py (out audit/phase2/out/W6C/W6C_eleven.txt)"
SC6 = "audit/scripts/claims/W6C_misc.py (out audit/phase2/out/W6C/W6C_misc.txt)"

LABVER = ("audit's cached lab copies under audit/cache/labpc/ (lab PC UNREACHABLE this "
          "session: Tailscale down). No BH q-value quoted by any claim in this batch.")

ROWS = [
 dict(id="O-008",
   claim_plain="On the collaborator's six datasets the extended 12v13 contrast beats his trained "
     "single-source baseline by +0.256 at layer 20 and +0.184 at layer 40 (zero-shot AUROC).",
   stated_numbers="+0.256; +0.184",
   source_ref="docs/COLLABORATOR_BRIEFING.md:252 (table in 5a)",
   evidence_file=COLLAB,
   evidence_key="summary.csv row features='ext_axis1' col off_diagonal_mean, minus row "
     "features='baseline' col off_diagonal_mean; cross-checked against mean(zero_shot.csv "
     "auroc_ext_axis) over the 6 scored datasets",
   level1_value="L20 0.9041173922 - 0.6477701197 = +0.2563 ; L40 0.9158352735 - 0.7320325942 "
     "= +0.1838",
   level1_match="YES (briefing rounds to 3 dp)",
   level2_value="+0.25635 (L20) and +0.18380 (L40); identical whether ext_axis1 is taken from "
     "summary.csv or recomputed as the mean of the per-dataset auroc_ext_axis rows "
     "(|diff| 1.1e-16 / 0.0). Baseline independently recomputed as the off-diagonal mean of "
     "matrix_baseline.csv: 0.647773 (L20), 0.732027 (L40), n=30 pairs.",
   level2_method="own reader over his committed CSVs; both terms recomputed from the "
     "per-dataset rows and from the raw 6x6 transfer matrix",
   version_used="his branch, read out of git at head 45b16d9 AND at f386a72 - byte-identical "
     "values at both. His activations are gone (RF-2), so the AUROCs themselves are Level 1.",
   robustness="L44 gives +0.1877, so the two quoted layers are not a lucky pair. Against the "
     "five-dataset baseline (repe_honesty dropped, 0.6085/0.7064) the extended margins are "
     "+0.2782 / +0.1926.",
   verdict="HOLDS", confidence="HIGH - digits read verbatim from his committed CSV and both "
     "terms recomputed two ways", importance="HIGH - it is the size of the abstract's headline "
     "claim, and the number the kd-bank share is a fraction of",
   script=SC1,
   notes="Exact. Same table as P12's O-007 (the 8v7 row); this is its extended twin."),

 dict(id="O-010",
   claim_plain="Rebuilt on our eleven datasets, the extended contrast ext_axis1 beats the paper's "
     "axis1 at all 21 stored layers, by a mean of +0.055, range [+0.002, +0.081], peaking at L20 "
     "(zero-shot, mean pooling).",
   stated_numbers="21 of 21; +0.055; +0.002; +0.081; peak L20",
   source_ref="docs/COLLABORATOR_BRIEFING.md:286-287",
   evidence_file="C:/Users/user4/expLB/run-crossgen/kd11/banks/collab/matrices/mean_L*_s42.json "
     "(21 files, 2026-09-22 00:33-00:49) and the paper16 baseline "
     "run-crossgen/kd11/matrices/mean_L*_s42.json - LAB ONLY, not in git, not in audit/cache/",
   evidence_key="zeroshot['contrast:collab_extended'].auroc mean over the 11 datasets minus "
     "zeroshot['contrast'].auroc mean, per layer",
   level1_value="F-99 states +0.0551, 21 of 21, range [+0.0019 (L0), +0.0814 (L20)]",
   level1_match="YES",
   level2_value="21/21 positive; mean +0.0551; range [+0.0019 at L0, +0.0814 at L20]; the "
     "maximum is at L20, so 'peaking at L20' is exact",
   level2_method="DEFERRED (lab unreachable). Reused the audit's own re-derivation from those "
     "raw matrices: batch P12, audit/phase2/out/W6C/../P12/P12_lab.txt lines 218-242, script "
     "audit/scripts/claims/P12_lab.py. Not re-run this session.",
   version_used=LABVER + " Would need run-crossgen/kd11/banks/collab/matrices/mean_L*_s42.json.",
   robustness="Under LAST-token pooling the same difference is +0.0590, positive at 20/21, and "
     "the peak MOVES from L20 to L36 (+0.0848). So 'peaking at L20' is a mean-pooling fact; the "
     "briefing states it inside the mean-pooled paragraph, which is correct but not signposted.",
   verdict="HOLDS", confidence="MEDIUM - every figure matches, but the level-2 re-derivation is "
     "batch P12's, not re-run this session because the lab PC is unreachable",
   importance="HIGH - it is the layer robustness offered in place of a two-layer quote",
   script=SC1 + " [context only]; numbers from audit/phase2/out/P12/P12_lab.txt",
   notes="LEVEL-2 DEFERRED (lab unreachable). 'Peaking at L20' holds under mean pooling only; "
     "under last-token the peak is L36."),

 dict(id="O-013",
   claim_plain="The kd bank's contribution barely moves with pooling (+0.055 mean-pooled vs "
     "+0.059 last-token) and shares above 100% are real: at L60-L68 axis1 falls below raw "
     "activations, so the kd bank supplies more than the whole margin there.",
   stated_numbers="+0.055; +0.059; >100% at L60-L68",
   source_ref="docs/COLLABORATOR_BRIEFING.md:301-305",
   evidence_file="C:/Users/user4/expLB/run-crossgen/kd11/banks/collab/matrices/"
     "{mean,last}_L*_s42.json plus the raw baseline run-crossgen/kd11/matrices/"
     "{mean,last}_L*_s42.json - LAB ONLY for the collab half",
   evidence_key="mean over the 11 datasets of zeroshot['contrast:collab_extended'] and "
     "zeroshot['contrast']; raw = off-diagonal mean of results[feature='raw',probe='paper'] "
     "under the same pooling; share = (ext-axis1)/(ext-raw)",
   level1_value="F-99 (mean half) +0.0551; the last-token half is stated only in the briefing",
   level1_match="YES for the mean half; the last-token half has no ledger entry",
   level2_value="mean +0.0551, last +0.0590. At L60 axis1 0.6379 vs raw 0.6386; L64 0.6337 vs "
     "0.6370; L68 0.6260 vs 0.6339 - axis1 below raw at all three, shares 101.4%, 106.3%, "
     "113.9%. At L72 axis1 (0.6406) is back above raw (0.6343) and the share falls to 87.4%, so "
     "'L60-L68' names exactly the right window.",
   level2_method="DEFERRED (lab unreachable). Numbers from batch P12's re-derivation of the raw "
     "collab matrices (audit/phase2/out/P12/P12_lab.txt lines 218-269). The RAW baseline half of "
     "the arithmetic WAS re-derived this session from the local kd11 matrices (0.6289 at L20, "
     "0.6623 at L40, 110 pairs) - see W6C_eleven.txt.",
   version_used=LABVER + " Local run-crossgen/kd11/matrices/mean_L40_s42.json is byte-identical "
     "to the cached lab copy (md5 d7616aa50814267e38bd38931e25a6a0), so the baseline half is "
     "version-safe.",
   robustness="Under last-token pooling the share range widens to [-155.9%, +144.9%] and goes "
     "negative at L0 - which is the 09-23 audit correction now in the text, and it is right.",
   verdict="HOLDS", confidence="MEDIUM - exact on every figure, but the collab half is P12's "
     "re-derivation, not re-run this session",
   importance="MEDIUM-HIGH - it is the caveat that stops the collaborator quoting a share",
   script=SC5 + "; collab half from audit/phase2/out/P12/P12_lab.txt",
   notes="LEVEL-2 DEFERRED for the collab half. The 09-23 audit correction that scoped '2%-114%' "
     "to mean pooling and added the last-token range '-156% to +145%' is itself CORRECT "
     "(measured -155.9% to +144.9%, median 50.0%)."),

 dict(id="O-016",
   claim_plain="Against the collaborator's 0.916, our pooled leave-one-dataset-out probe on our "
     "six overlapping datasets reads 0.868 at L40, a margin of +0.048 - explicitly flagged as a "
     "mixed-corpora comparison.",
   stated_numbers="0.916; 0.868; +0.048",
   source_ref="docs/COLLABORATOR_BRIEFING.md:333",
   evidence_file="audit/cache/labpc/run-premise/six/premise_L40_mean_s42.json "
     "(md5 02a7643673e6629249bd78f29635a3cb, 8,262 B); 0.916 from his summary.csv (see O-008)",
   evidence_key="summary.pooled_lodo; his summary.csv row ext_axis1 off_diagonal_mean",
   level1_value="file summary.pooled_lodo = 0.8683400560 ; 0.9158352735 - 0.8683400560 = +0.0475",
   level1_match="YES (briefing rounds 0.8683 to 0.868 and the margin to +0.048)",
   level2_value="+0.04750. The six datasets in the file are instructed-deception, "
     "harm-pressure-choice, harm-pressure-knowledge-report, convincing-game, insider-trading, "
     "ai_liar - i.e. FIVE of his six plus ai_liar substituting repe_honesty, exactly as the "
     "briefing says two paragraphs above.",
   level2_method="own reader over the cached premise-six file; margin computed from his "
     "committed CSV value rather than from the briefing's rounded 0.916",
   version_used=LABVER + " run-premise is on no branch and has no Drive copy (RF-10); the cached "
     "copy is the audit's own tar-verified pull.",
   robustness="At L32 the same comparison is 0.916 - 0.8965 = +0.0193, i.e. the margin nearly "
     "halves one layer band earlier - which is 5e's point and is consistent with it. At L20 "
     "pooled reads only 0.7365.",
   verdict="HOLDS", confidence="HIGH - both terms read from primary files and the arithmetic "
     "redone", importance="HIGH - it is the comparison the abstract's margin is judged against",
   script=SC2,
   notes="Exact, and the briefing's own 'mixed corpora' flag is accurate: his six and our six "
     "share only five datasets."),

 dict(id="O-017",
   claim_plain="Against the collaborator's 0.916, our row-budget-matched pooled LODO probe on "
     "our six reads 0.893 at L40, a margin of +0.023 - again mixed corpora.",
   stated_numbers="0.916; 0.893; +0.023",
   source_ref="docs/COLLABORATOR_BRIEFING.md:334",
   evidence_file="audit/cache/labpc/run-premise/six/premise_L40_mean_s42.json "
     "(md5 02a7643673e6629249bd78f29635a3cb)",
   evidence_key="summary.pooled_lodo_match",
   level1_value="file summary.pooled_lodo_match = 0.8929720785 ; 0.9158352735 - 0.8929720785 "
     "= +0.0229",
   level1_match="YES",
   level2_value="+0.02286. n_train.pooled_match = 544 for the six-dataset store; the per-target "
     "realised matched pools are n_lodo_match 452-622.",
   level2_method="own reader over the cached premise-six file",
   version_used=LABVER,
   robustness="At L32 the same row reads 0.9010 and the margin falls to +0.0148, which is the "
     "'+0.015' 5e quotes - so 5e's follow-up is consistent with this row. Note the row-budget "
     "matched probe BEATS the full pooled probe here (0.893 > 0.868), i.e. more rows do not help.",
   verdict="HOLDS", confidence="HIGH", importance="HIGH - the smallest margin in the briefing "
     "and the one a reviewer will seize on",
   script=SC2,
   notes="Exact. Same caveat as O-016: five of six datasets overlap."),

 dict(id="O-018",
   claim_plain="An earlier version of this briefing reported the +0.023 as the headline and said "
     "our own contrast 'loses by 0.082' on the same six; both statements mixed two dataset sets "
     "and two banks, because the 0.811 quoted there is our knowledge-denial bank, not a register "
     "axis.",
   stated_numbers="+0.023; 0.082; (0.811)",
   source_ref="docs/COLLABORATOR_BRIEFING.md:309-311",
   evidence_file="git: docs/COLLABORATOR_BRIEFING.md at aea9f53 (2026-09-21 21:25) lines "
     "198-205; and the L40 matrices run-crossgen/{kd11/matrices, kd11/banks/kd, "
     "kd11/banks/kd_s23, matrices, banks/*}/mean_L40_s42.json (local working tree)",
   evidence_key="zeroshot['contrast'].auroc, mean over the six overlapping datasets and over all "
     "eleven, per store",
   level1_value="aea9f53 says verbatim: 'the margin goes from +0.184 to +0.023' and 'like-for-"
     "like on our data the persona contrast loses by 0.082' (0.811 against 0.893)",
   level1_match="YES - the self-quotation is exact, including the 0.811 and 0.893",
   level2_value="The withdrawn table gave 0.811 on the six and 0.733 on all eleven. Measured: "
     "kd11/banks/kd_s23 (the knowledge-denial bank, s23 vectors) gives 0.8110 / 0.7325 - both to "
     "4 dp. kd11 root (paper16 bank, axes_vectors_s3_fp32.pt) gives 0.8106 / 0.7287; "
     "kd11/banks/kd (s3) gives 0.8205 / 0.7392; own100 0.8052 / 0.7299; hf100 0.7958 / 0.7221. "
     "Only kd_s23 reproduces BOTH quoted numbers.",
   level2_method="own reader over every locally available L40 matrix; the bank is identified by "
     "its zero-shot fingerprint on two subsets at once, not by file name (RF-16)",
   version_used="local git working-tree matrices; the kd11 root copy is byte-identical to the "
     "audit's cached lab copy (md5 d7616aa50814267e38bd38931e25a6a0), so RF-11 does not bite here.",
   robustness="0.811 alone does NOT identify the bank - the paper16 root reads 0.8106 and also "
     "rounds to 0.811. It is the companion 0.733 that pins it to kd_s23.",
   verdict="HOLDS", confidence="HIGH - the self-quotation is verbatim in git and the bank is "
     "fixed by a two-number fingerprint",
   importance="MEDIUM - it is a withdrawal, but a wrong withdrawal would leave the collaborator "
     "with a number he cannot place",
   script=SC4,
   notes="One precision worth adding to the text: the bank is the s23 variant (kd_s23), not the "
     "s3 kd bank, which reads 0.8205 / 0.7392. Per the standing trap, s3 and s23 are not "
     "interchangeable."),

 dict(id="O-019",
   claim_plain="The paper's own 8-vs-7 contrast beats the collaborator's own single-source "
     "trained baseline by +0.143 at L20 and +0.100 at L40, entirely inside his files.",
   stated_numbers="+0.143; +0.100",
   source_ref="docs/COLLABORATOR_BRIEFING.md:342-343 (and the 5a table, line 251)",
   evidence_file=COLLAB,
   evidence_key="mean(zero_shot.csv auroc_axis over the 6 scored datasets) minus summary.csv row "
     "baseline off_diagonal_mean",
   level1_value="0.7908803603 - 0.6477701197 = +0.1431 (L20); 0.8318937061 - 0.7320325942 = "
     "+0.0999 (L40)",
   level1_match="YES",
   level2_value="+0.14311 and +0.09986. The baseline was independently recomputed as the "
     "off-diagonal mean of matrix_baseline.csv (0.647773 / 0.732027, n=30).",
   level2_method="own reader; zero-shot means recomputed from the per-dataset rows",
   version_used="his branch at head 45b16d9 and at f386a72 - identical",
   robustness="Care needed: at L20 summary.csv's TRAINED axis1 row is 0.6511, not 0.7909, so "
     "this margin exists only for the ZERO-SHOT arm. The briefing labels the row "
     "'(zero_shot.csv, auroc_axis)' and is therefore right. At L44 the margin is +0.1104.",
   verdict="HOLDS", confidence="HIGH", importance="HIGH - the briefing calls it 'the number the "
     "replication is entitled to claim'",
   script=SC1,
   notes="Exact. Duplicate of P12's O-007, re-derived independently here and agreeing."),

 dict(id="O-020",
   claim_plain="Across our 110 source-to-target pairs, 21 fall below 0.5 and 14 below 0.4, with "
     "a minimum of 0.055 - which is what drags the single-source baseline down.",
   stated_numbers="110; 21; 14; 0.055",
   source_ref="docs/COLLABORATOR_BRIEFING.md:346-348",
   evidence_file="audit/cache/labpc/run-premise/audit/audit_L40_s42.json "
     "(md5 712bca920f38b6c321fd0eeeed50bc4b, 17,175 B)",
   evidence_key="pairs[] (110 entries, each source/target/auroc); pairs_below_half; n_pairs",
   level1_value="file: pairs_below_half = 21, n_pairs = 110",
   level1_match="YES",
   level2_value="recomputed from the 110 stored pairs: n=110, below 0.5 = 21, below 0.4 = 14, "
     "below 0.3 = 6, minimum 0.0545535714 (convincing-game -> "
     "harm-pressure-knowledge-report). Mean 0.6763163062.",
   level2_method="own reader; every count and the minimum recomputed from the per-pair list "
     "rather than read from the summary block",
   version_used=LABVER + " Layer 40, mean pooling, paper probe, seed 42 - the briefing does not "
     "name the layer; L40 is the only layer for which this audit file exists.",
   robustness="'14 below 0.4' is not stored anywhere in the file - it is recomputed here and "
     "matches. The eleven-dataset set is the full kd11 set (110 = 11 x 10), not block9.",
   verdict="HOLDS", confidence="HIGH - counts and minimum recomputed from the raw pair list",
   importance="MEDIUM - supporting evidence for the 'weak baseline' argument",
   script=SC2,
   notes="Exact, including the minimum to 3 dp (0.0546 -> 0.055)."),

 dict(id="O-021",
   claim_plain="Of four single-source baselines computed on the same 110 pairs, the mean over "
     "sources - the estimator the abstract compares against - is 0.676 (median 0.722, "
     "sign-oriented mean 0.735, best-source oracle 0.897).",
   stated_numbers="0.676 (and 0.722, 0.735, 0.897)",
   source_ref="docs/COLLABORATOR_BRIEFING.md:352 (table 350-355)",
   evidence_file="audit/cache/labpc/run-premise/audit/audit_L40_s42.json",
   evidence_key="baselines.{mean_over_sources, median_over_sources, oriented_mean, "
     "best_source_oracle}; per_target.* for the recomputation",
   level1_value="0.6763163062 / 0.7219698909 / 0.7353614302 / 0.8966842158",
   level1_match="YES - all four round to the briefing's values",
   level2_value="mean over sources recomputed from the 110 pairs = 0.6763163062 (exact). The "
     "other three are per-TARGET statistics averaged over the 11 targets, not statistics of the "
     "flat 110: mean of per-target medians 0.7219698909, of per-target oriented means "
     "0.7353614302, of per-target maxima 0.8966842158 - all reproduce exactly. The plain median "
     "of the 110 pairs is 0.6844, which is NOT the quoted 0.722.",
   level2_method="own reader; each estimator rebuilt from the per-pair list and compared with "
     "the stored per_target arrays",
   version_used=LABVER + " L40, mean pooling, paper probe, seed 42.",
   robustness="The estimator definition matters: 'median over sources' means per-target median "
     "then average. Anyone recomputing it as the median of all 110 pairs gets 0.684 and will "
     "think the table is wrong.",
   verdict="HOLDS", confidence="HIGH - all four values reproduced from the per-pair list",
   importance="MEDIUM-HIGH - these four numbers are what section 7 then mis-attaches to HIS "
     "baseline (see O-036)",
   script=SC2,
   notes="Values exact. The row label '(what the abstract compares against)' is true of the "
     "ESTIMATOR but not of the VALUE - the abstract compares against his 0.732 on his six, not "
     "our 0.676 on our eleven. Harmless here; it becomes an error in section 7 (O-036)."),

 dict(id="O-023",
   claim_plain="The principal components are not doing the work: at L20 ext_pc1 trails ext_axis1 "
     "by 0.243, and the 1-d contrast beats the 3-d subspace at every depth the collaborator ran.",
   stated_numbers="0.243; 'every depth'",
   source_ref="docs/COLLABORATOR_BRIEFING.md:373-374",
   evidence_file=COLLAB + " (L20, L40, L44 summary.csv)",
   evidence_key="summary.csv off_diagonal_mean for features ext_pc1, ext_axis1, pc1, axis1, "
     "pc3, ext_pc3",
   level1_value="L20 ext_pc1 0.6612748487 vs ext_axis1 0.9041173922 -> -0.2428",
   level1_match="YES for the 0.243",
   level2_value="0.24284 at L20 (also -0.1479 at L40 and -0.1232 at L44). 1-d vs 3-d, EXTENDED "
     "bank: ext_axis1 - ext_pc3 = +0.2204 (L20), +0.0503 (L40), +0.0235 (L44) - positive at all "
     "three. PAPER bank: axis1 - pc3 = -0.0281 (L20), +0.1925 (L40), +0.2095 (L44) - NEGATIVE at "
     "L20, i.e. the 3-d subspace (0.6792) beats the 1-d contrast (0.6511) there.",
   level2_method="own reader over his summary.csv at all three layers",
   version_used="his branch at head 45b16d9 and f386a72 - identical",
   robustness="On the off_diagonal_excl_hpkr column the paper bank's L20 case flips to positive "
     "(+0.0315), so the exception is driven by harm-pressure-knowledge-report - the same dataset "
     "that drives everything else in this file.",
   verdict="PARTLY HOLDS",
   confidence="HIGH - every value read verbatim from his committed CSV",
   importance="MEDIUM - it is the resolution of the withdrawn 'PC1 contradiction'",
   script=SC1,
   notes="The 0.243 is exact. 'The 1-d contrast beats the 3-d subspace at every depth you ran' "
     "is true of the EXTENDED bank (all three depths) but FALSE of the paper bank at L20, where "
     "axis1 0.6511 < pc3 0.6792. Suggested fix: say 'the extended 1-d contrast beats the "
     "extended 3-d subspace at all three depths', or add 'except axis1 vs pc3 at L20'."),

 dict(id="O-024",
   claim_plain="An earlier version of this briefing asserted that the collaborator scored per "
     "token while our pipeline mean-pools, and concluded that 'token aggregation alone is worth "
     "up to +0.103'; that premise is withdrawn because his pipeline mean-pools too.",
   stated_numbers="+0.103",
   source_ref="docs/COLLABORATOR_BRIEFING.md:355-356",
   evidence_file="git: docs/COLLABORATOR_BRIEFING.md at 1a482b7 (line 227) and aea9f53 (line "
     "247); origin/feat/persona-pc-ood at f386a72: "
     "src/probes/deception-detection/deception_detection/scripts/pool_activations.py; "
     "docs/thesis/persona-coordinates-70b-results.md:770,1048",
   evidence_key="the quoted sentence; pool_activations.py line 84; the Q8 pooling table",
   level1_value="aea9f53:247 reads verbatim '**Token aggregation alone is worth up to +0.103.** "
     "Add the six-dataset subset effect (+0.079, section 5a) and 0.811 -> 0.916 is fully "
     "accounted for without anyone being wrong.'",
   level1_match="YES - the self-quotation is verbatim, including the second sentence",
   level2_value="The +0.103 traces to the project's own Q8 measurement (top-decile vs mean "
     "pooling, zero-shot): docs/thesis/persona-coordinates-70b-results.md:770 gives +0.103 at "
     "L20, and :1048 records '+0.103 (L20), +0.064 (L22), +0.053 (L28), <= +0.014 at 40-48'. "
     "The retraction's own evidence also checks out: at f386a72, pool_activations.py LINE 84 is "
     "exactly `pooled.append(acts[row_mask].float().mean(0).squeeze(0).numpy())`, and the module "
     "docstring says 'the mean over the assistant response tokens'.",
   level2_method="git history of the briefing; line-exact read of his script at the commit the "
     "briefing cites; grep of the project's own Q8 tables",
   version_used="git only (both branches); no lab file needed",
   robustness="The line number is version-specific: at branch head 45b16d9 the same statement "
     "has moved to line 123 and reads `example.acts[example.mask]`. The briefing cites head "
     "f386a72 in section 5, so ':84' is right for the commit it names.",
   verdict="HOLDS", confidence="HIGH - quotation, code line and source of +0.103 all verified",
   importance="MEDIUM - a self-correction; getting it wrong would re-open a settled dispute",
   script="git show (see evidence_file); " + SC1 + " for the branch reads",
   notes="Path nit, same family as the one the 09-23 audit already fixed for persona_ood.py: the "
     "file is `deception_detection/scripts/pool_activations.py`, not `scripts/"
     "pool_activations.py`. The docstring is paraphrased as 'the model's response tokens'; it "
     "actually says 'the assistant response tokens'."),

 dict(id="O-025",
   claim_plain="Kumar's abstract says the single-direction hypothesis is decisively rejected, "
     "k=1 capturing only 0.61-0.80 AUROC of the signal, with cross-domain transfer failure "
     "confirmed as geometric rather than layer-mismatch-driven.",
   stated_numbers="0.61; 0.80",
   source_ref="docs/COLLABORATOR_BRIEFING.md:418-420 (block quote)",
   evidence_file="papers/Pressure-Testing Deception Probes in LLMs Scaling, Robustness, and "
     "the.pdf (18 pages, Sachin Kumar, LexisNexis)",
   evidence_key="page 1 abstract, clause (b); Table 6 on page 5; Figure 2 caption on page 15",
   level1_value="abstract, verbatim: '(b) the single-direction hypothesis is decisively rejected "
     "(k=1 captures only 0.61-0.80 AUROC of the signal, with cross-domain transfer failure "
     "confirmed as geometric rather than layer-mismatch-driven;'",
   level1_match="YES - word for word",
   level2_value="Table 6 (page 5) gives the k=1 row 0.609 / 0.661 / 0.804 / 0.672 for 1B / 4B / "
     "12B / 27B, min 0.609 max 0.804 - so 0.61-0.80 is the rounded min-max of the k=1 row. Body "
     "text: 'all four models capture only 0.609-0.804 of the deception signal'.",
   level2_method="pypdf text extraction of the PDF present in the repo, searched for the "
     "quoted strings and for the underlying table",
   version_used="the repo's tracked PDF; read during this audit",
   robustness="Also checked the two other Kumar figures the same section quotes: style-augmented "
     "0.979-0.983 (abstract) and 'CIs ... spanning nearly the full network, indicating weak "
     "localization' (page 5) - both present verbatim.",
   verdict="HOLDS", confidence="HIGH - quotation checked against the PDF opened in this audit",
   importance="MEDIUM - it is the concurrent-paper threat the briefing tells the collaborator to "
     "answer",
   script=SC6,
   notes="The briefing silently repairs an unbalanced parenthesis in the original ('of the "
     "signal), with' vs the paper's 'of the signal, with'). Substantively identical; if it is "
     "reproduced as a block quote in the paper, use [sic] or re-punctuate openly."),

 dict(id="O-026",
   claim_plain="Run like-for-like under leave-one-dataset-out at L40, all three persona feature "
     "sets lose to raw 8192-d logistic regression: axes 0.679->0.745 (-0.109, CI [-0.187,"
     "-0.039]), pc:deception 0.621->0.731 (-0.124, [-0.203,-0.055]), pc:unified@2 0.574->0.692 "
     "(-0.162, [-0.288,-0.059]), against raw 0.676->0.854.",
   stated_numbers="0.676; 0.854; 0.679; 0.745; -0.109; 0.621; 0.731; -0.124; 0.574; 0.692; "
     "-0.162; the three CIs",
   source_ref="docs/COLLABORATOR_BRIEFING.md:475-480 (5d table)",
   evidence_file="audit/cache/labpc/run-premise/audit/audit_L40_s42.json "
     "(md5 712bca920f38b6c321fd0eeeed50bc4b); cross-check "
     "audit/cache/labpc/run-premise/premise_L40_mean_s42.json (md5 e5ddaaa13ea6aa5a4b97bbafa38592ba)",
   evidence_key="persona_lodo[feature].{single,lodo,lodo_per_target}; pooled_lodo_full; "
     "baselines.mean_over_sources; per_target.pooled_lodo_full",
   level1_value="axes 0.6794089978 / 0.7453755531; pc:deception 0.6213570249 / 0.7307243420; "
     "pc:unified@2 0.5739826680 / 0.6920640950; raw 0.6763163062 / 0.8544753650",
   level1_match="YES - all eight values round to the table",
   level2_value="vs raw pooled: axes -0.1091, pc:deception -0.1238, pc:unified@2 -0.1624. My own "
     "paired dataset-level bootstrap over the 11 targets (B=10,000, default_rng(42)) gives CI95 "
     "[-0.187,-0.039] P(>=0) 0.0006 ; [-0.203,-0.055] P 0.0001 ; [-0.288,-0.059] P 0.0002 - the "
     "three stated intervals reproduce to 3 dp, and none of them is in the file, so this is a "
     "genuine re-derivation rather than a read-back.",
   level2_method="own loader + own paired dataset bootstrap over the 11 per-target values "
     "(the CIs are not stored in the artifact)",
   version_used=LABVER + " The audit file was written by the 5d review run; run-premise has no "
     "Drive copy (RF-10). No BH q involved.",
   robustness="One layer (40), one pooling (mean), one seed, one probe ('paper'). The raw "
     "pooled figure 0.8545 also appears independently in premise_L40_mean_s42.json's summary, "
     "and the two agree exactly.",
   verdict="HOLDS", confidence="HIGH - the three confidence intervals were independently "
     "re-derived, not copied",
   importance="HIGH - it is the briefing's single strongest claim against the method",
   script=SC2,
   notes="All eleven quantities exact. Worth telling the collaborator that this is a "
     "single-layer, single-seed result, which the briefing does not say."),

 dict(id="O-027",
   claim_plain="Pooling helps the persona features too - pc:unified@2 gains +0.118 from "
     "leave-one-dataset-out pooling - so diversity is not something personas provide.",
   stated_numbers="+0.118",
   source_ref="docs/COLLABORATOR_BRIEFING.md:484",
   evidence_file="audit/cache/labpc/run-premise/audit/audit_L40_s42.json",
   evidence_key="persona_lodo['pc:unified@2'].lodo - .single",
   level1_value="0.6920640950 - 0.5739826680 = +0.1181",
   level1_match="YES",
   level2_value="+0.11808. The other two features gain +0.0660 (axes) and +0.1094 "
     "(pc:deception); raw gains +0.1782. So pc:unified@2 is the LARGEST persona gain and the "
     "briefing quotes the most favourable of the three.",
   level2_method="own reader; all four gains computed so the quoted one can be placed",
   version_used=LABVER,
   robustness="The argument ('diversity is not something personas provide') holds for all three "
     "features, so picking the largest does not change the conclusion - but raw still gains more "
     "(+0.178) than any persona feature.",
   verdict="HOLDS", confidence="HIGH", importance="MEDIUM",
   script=SC2,
   notes="Exact. The sentence would be stronger with 'between +0.066 and +0.118' than with the "
     "single best figure."),

 dict(id="O-028",
   claim_plain="The secondary mean-of-directions result is +0.028 over pooled leave-one-dataset-"
     "out, 95% CI [+0.005, +0.052].",
   stated_numbers="+0.028; +0.005; +0.052",
   source_ref="docs/COLLABORATOR_BRIEFING.md:496",
   evidence_file="audit/cache/labpc/run-premise/audit/audit_L40_s42.json; also "
     "audit/cache/labpc/run-premise/why/why_L40_s42.json (md5 db563a7ccb9ceb98cd1dc2ab799a69da)",
   evidence_key="bootstrap['mean_of_directions-pooled_lodo_full'].{diff,lo,hi}; "
     "per_target.mean_of_directions vs per_target.pooled_lodo_full",
   level1_value="file: diff 0.0283541533, lo 0.0045082540, hi 0.0523446105",
   level1_match="YES",
   level2_value="recomputed from the 11 per-target values: +0.028354, own CI95 [+0.004610, "
     "+0.052037], P(<=0) 0.0088 (B=10,000, default_rng(42)). Scalars: mean_of_directions "
     "0.8828295183 (the same value appears in why_L40_s42.json), pooled_lodo_full 0.8544753650.",
   level2_method="own paired dataset bootstrap over the 11 targets, independent of the stored "
     "interval",
   version_used=LABVER + " L40, mean pooling, seed 42.",
   robustness="The interval is narrow and its lower bound sits at +0.0045; a different bootstrap "
     "implementation (mine) moves it to +0.0046, i.e. the conclusion 'it clears, barely' is "
     "implementation-stable.",
   verdict="HOLDS", confidence="HIGH - stored interval and an independent re-derivation agree",
   importance="MEDIUM - the briefing deliberately demotes it",
   script=SC2,
   notes="Exact. This is the +0.028 that F-86's 0.883 'mean of ten single-source directions' "
     "sits behind, and the briefing's demotion of it is warranted."),

 dict(id="O-029",
   claim_plain="The mean-of-directions result clears, but its lower bound is +0.005, so it is "
     "not a headline.",
   stated_numbers="+0.005",
   source_ref="docs/COLLABORATOR_BRIEFING.md:497",
   evidence_file="audit/cache/labpc/run-premise/audit/audit_L40_s42.json",
   evidence_key="bootstrap['mean_of_directions-pooled_lodo_full'].lo",
   level1_value="lo = 0.0045082540",
   level1_match="YES to the stated precision (0.0045 -> +0.005)",
   level2_value="my own bootstrap gives +0.004610. Both are above zero, so 'it clears' is "
     "correct; the exact bound is +0.0045, which the briefing rounds UP to +0.005.",
   level2_method="own paired dataset bootstrap over the 11 targets",
   version_used=LABVER,
   robustness="P(<=0) = 0.0088 in my bootstrap, i.e. it clears a one-sided 1% threshold as well. "
     "The claim is a self-deflation, so rounding the bound up is the conservative direction only "
     "if you are trying to sound confident - here it makes the result look very slightly better "
     "than it is.",
   verdict="HOLDS", confidence="HIGH", importance="LOW-MEDIUM",
   script=SC2,
   notes="Exact to the quoted precision. If a reviewer recomputes, they get +0.0045; writing "
     "'+0.0045' would be unimpeachable."),

 dict(id="O-030",
   claim_plain="In the sarcasm pilot, deadpan is producible: the marker rate falls to 0.00 at "
     "variants v4 and v5.",
   stated_numbers="0.00 (v4, v5)",
   source_ref="docs/COLLABORATOR_BRIEFING.md:575 (and the table label at :573)",
   evidence_file="audit/cache/labpc/cap/run-sarc-pilot/marker_report.json "
     "(md5 bcc633166d53e73a565d76318e977c75, 4,938 B, n=300)",
   evidence_key="markers[arm].per_variant (five entries, v1..v5) for the three arms",
   level1_value="pc_ironic_critic [0.85, 0.15, 0.10, 0.00, 0.00]; pc_sycophant [0.85, 1.00, "
     "0.50, 0.00, 0.05]; pc_sincere_critic [0.60, 0.15, 0.15, 0.25, 0.05]",
   level1_match="PARTIAL - exactly 0.00/0.00 for the ironic arm only",
   level2_value="ironic v4 = 0.00, v5 = 0.00 (both exact). Sycophant v4 = 0.00 but v5 = 0.05. "
     "Sincere-critic v4 = 0.25, v5 = 0.05. The lexical table the same section prints does "
     "reproduce exactly: ironic vs sycophant, all v1-v5 raw 0.9525 / stripped 0.9472; v4+v5 raw "
     "0.9194 / stripped 0.9213.",
   level2_method="own reader over the cached marker report; all three arms printed, not just "
     "the one the sentence is about",
   version_used=LABVER + " cap/run-sarc-pilot is lab-only with no Drive copy (RF-10). A "
     "marker_report_partial.json (n=100, ironic arm only) carries the same per-variant vector.",
   robustness="PROGRESS.md records that the sarcasm-pilot status file logged a stripped AUROC of "
     "0.947 at 02:29 and 0.921 at 15:55 for different subsets - the 0.921 here is the v4+v5 "
     "subset, so the briefing's table uses the later, narrower one and labels it correctly.",
   verdict="PARTLY HOLDS",
   confidence="HIGH - read directly from the artifact",
   importance="MEDIUM - it is the 'encouraging half' of a negative result",
   script=SC6,
   notes="True of the IRONIC arm, which is the arm 'deadpan' refers to. The table header "
     "'v4+v5 (marker rate 0.00)' labels the ironic-vs-sycophant comparison, and the sycophant "
     "arm reads 0.05 at v5. Safe wording: 'the ironic arm's marker rate falls to 0.00 at v4 and "
     "v5 (the sycophant arm to 0.00 and 0.05)'."),

 dict(id="O-031",
   claim_plain="Running both of the collaborator's arms through our crossgen on our eleven "
     "datasets is done: the kd bank supplies 49-74% of the margin, and both arms lose to a "
     "pooled probe.",
   stated_numbers="49%; 74%",
   source_ref="docs/COLLABORATOR_BRIEFING.md:614 (section 7 table), backed by the 5a table at "
     ":269-275",
   evidence_file="raw baseline recomputed from run-crossgen/kd11/matrices/mean_L{20,40}_s42.json "
     "(local, md5 414a8f1adc1bd55534e9cc2145397249 and d7616aa50814267e38bd38931e25a6a0 - the "
     "latter byte-identical to the cached lab copy); pooled from "
     "audit/cache/labpc/run-premise/premise_L{20,40}_mean_s42.json; the two arms from "
     "run-crossgen/kd11/banks/collab/matrices/mean_L{20,40}_s42.json (LAB ONLY)",
   evidence_key="share = (ext_axis1 - axis1) / (ext_axis1 - raw); raw = off-diagonal mean of "
     "results[feature='raw',probe='paper'] over the 110 pairs; pooled = summary.pooled_lodo",
   level1_value="briefing 5a table: 74% (L20) and 49% (L40)",
   level1_match="YES",
   level2_value="raw recomputed here: 0.6289 (L20) and 0.6623 (L40), 110 pairs - matching the "
     "5a table exactly. pooled recomputed here: 0.8027 and 0.8545 - matching exactly. With "
     "P12's lab-side arms (axis1 0.6572/0.7325, ext_axis1 0.7386/0.7996) the share is 74.2% and "
     "48.9%, and both arms lose to pooled: axis1 by +0.1455 / +0.1220 and ext_axis1 by +0.0641 / "
     "+0.0549.",
   level2_method="three of the five rows of the 5a table re-derived this session from local and "
     "cached files; the two collab arms DEFERRED (lab unreachable) and taken from batch P12",
   version_used=LABVER + " Would need run-crossgen/kd11/banks/collab/matrices/mean_L*_s42.json "
     "for a full re-derivation.",
   robustness="'49-74%' is the range over TWO layers. Over all 21 layers, mean-pooled, the same "
     "share runs 2%-114% (median 63%) and under last-token -156% to +145% - which the same "
     "document says two sections earlier. The section-7 one-liner does not carry that caveat.",
   verdict="HOLDS", confidence="MEDIUM-HIGH - the arithmetic and three of five inputs re-derived "
     "here; the two arms rest on P12's re-derivation",
   importance="HIGH - it is the headline of the whole revision",
   script=SC5,
   notes="LEVEL-2 PARTLY DEFERRED. The raw and pooled rows of the 5a table reproduce exactly. "
     "The 09-23 audit addition 'and axis1 by 0.146 and 0.122' is CORRECT (+0.1455, +0.1220). "
     "Suggest 'roughly half at L40, three quarters at L20 - but see the 2%-114% range across "
     "depth' so the section-7 line does not travel alone."),

 dict(id="O-032",
   claim_plain="Run through the same leave-one-dataset-out harness, the persona method loses to "
     "raw pooled logistic regression by 0.11-0.16 AUROC.",
   stated_numbers="0.11; 0.16",
   source_ref="docs/COLLABORATOR_BRIEFING.md:616 (section 7 table)",
   evidence_file="audit/cache/labpc/run-premise/audit/audit_L40_s42.json",
   evidence_key="persona_lodo[feature].lodo minus pooled_lodo_full",
   level1_value="the 5d table it points at: -0.109, -0.124, -0.162",
   level1_match="YES",
   level2_value="-0.1091 (axes), -0.1238 (pc:deception), -0.1624 (pc:unified@2). Rounded, the "
     "range is 0.11 to 0.16.",
   level2_method="own reader; all three differences recomputed from the stored scalars and "
     "cross-checked against the per-target arrays",
   version_used=LABVER + " Single layer (40), single seed, mean pooling.",
   robustness="All three losses clear a paired dataset-level bootstrap over the 11 targets "
     "(P(>=0) 0.0006 / 0.0001 / 0.0002), so the range is not an artefact of picking the ends.",
   verdict="HOLDS", confidence="HIGH", importance="HIGH - the briefing proposes this as the "
     "paper's revised contribution",
   script=SC2,
   notes="Exact. Same measurement as O-034, which states it in prose."),

 dict(id="O-033",
   claim_plain="In the format-matched control, swapping what the six negative-side personas say "
     "costs only about +0.03, clears no significance test at any layer, and removing the six "
     "entirely costs two to four times as much.",
   stated_numbers="~+0.03; 'no significance test at any layer'; 'two to four times'",
   source_ref="docs/COLLABORATOR_BRIEFING.md:630-631",
   evidence_file="C:/Users/user4/expLB/run-crossgen/kd11/banks/{cap_s23,cap_ctrl}/matrices/"
     "{mean,last}_L*_s42.json (90 files) - LAB ONLY, in no branch and not in audit/cache/",
   evidence_key="sycophancy_mt COLUMN of each feature's 11x11 matrix, mean over the 10 sources; "
     "content = pc:capitulation@2 - pc:cap_vs_neutral@2, presence = pc:capitulation@2 - "
     "pc:cap_only_ctrl@2; per-layer paired dataset bootstrap over the 10 sources",
   level1_value="F-94 (kd/docs/FINDINGS.md:4107-4180): content mean +0.0356, clears at 3 of 21 "
     "layers (L0, L12, L36); presence mean +0.0730, clears at 15 of 21; ratio about 2:1 "
     "mean-pooled and about 1:1 last-token; 'No ratio should be quoted at all'",
   level1_match="NO for two of the three parts - see notes",
   level2_value="DEFERRED. Using the audit's own earlier re-derivation of those raw matrices "
     "(audit/claims/F-94.md, batch P2-pooled, script audit/scripts/claims/p2p_F94_lab.py): "
     "content mean +0.0356 and it CLEARS at L0 (+0.0524), L12 (+0.0171) and L36 (+0.0290); "
     "presence mean +0.0730, clearing at 15 of 21; presence/content = 2.05x on the means, with "
     "per-layer ratios spanning 0.42x (L8) to 4.86x (L40).",
   level2_method="DEFERRED (lab unreachable). Ratios computed here from the F-94.md table; the "
     "underlying matrices were read read-only on the lab PC by batch P2-pooled.",
   version_used=LABVER + " Would need run-crossgen/kd11/banks/{cap_s23,cap_ctrl}/matrices/. "
     "The cap_ctrl completeness gate is separately broken by RF-6, but all 42 matrices exist.",
   robustness="Under LAST-token pooling F-94 records content +0.0346 and presence +0.0373, i.e. "
     "about 1:1, and content CLEARS at L28, L32 and L36 there too. So the 'clears nothing' claim "
     "fails under both poolings.",
   verdict="PARTLY HOLDS",
   confidence="MEDIUM-HIGH - the magnitude and the counter-examples come from an audit "
     "re-derivation of the raw matrices, but not one re-run this session",
   importance="MEDIUM-HIGH - it is the qualification the briefing attaches to Claim 2",
   script=SC6,
   notes="The '~+0.03' is right (+0.0356). The other two clauses are not. (1) 'clears no "
     "significance test at any layer' is FALSE: the content effect clears at L0, L12 and L36 "
     "mean-pooled, and at L28/L32/L36 last-token. (2) 'two to four times that' is a ratio F-94 "
     "explicitly says should not be quoted at all: it is 2.05x on the 21-layer means, ~1:1 under "
     "last-token pooling, and ranges 0.42x-4.86x by layer. Safe wording, from F-94's own bottom "
     "line: 'swapping what the negative-side personas say costs ~+0.035 and rarely clears; "
     "removing them costs ~+0.073 and usually does.' THIS ONE WOULD TRAVEL INTO THE PAPER."),

 dict(id="O-034",
   claim_plain="In a like-for-like leave-one-dataset-out comparison the trained persona-"
     "coordinates method loses to ordinary logistic regression on raw activations by 0.11-0.16 "
     "AUROC.",
   stated_numbers="0.11; 0.16",
   source_ref="docs/COLLABORATOR_BRIEFING.md:644-645",
   evidence_file="audit/cache/labpc/run-premise/audit/audit_L40_s42.json",
   evidence_key="persona_lodo[feature].lodo minus pooled_lodo_full, all three features",
   level1_value="5d table: -0.109 / -0.124 / -0.162",
   level1_match="YES",
   level2_value="-0.1091, -0.1238, -0.1624 - range 0.109 to 0.162, i.e. 0.11-0.16 rounded. All "
     "three clear a paired dataset bootstrap over 11 targets.",
   level2_method="own reader plus own bootstrap (see O-026)",
   version_used=LABVER + " One layer, one seed, one pooling, one probe.",
   robustness="The claim is stated in the paragraph that proposes it as the paper's contribution, "
     "so its single-layer basis matters: no layer sweep of the persona LODO harness exists in "
     "run-premise (only L20/L32/L40 for the clustered variant, and those do not carry persona "
     "features).",
   verdict="HOLDS", confidence="HIGH for the number; the SCOPE is single-layer and the sentence "
     "does not say so", importance="HIGH - proposed as the paper's revised headline",
   script=SC2,
   notes="Exact. Recommend adding 'at layer 40, mean-pooled' - four of this project's headline "
     "margins have already failed a layer sweep (memory note 'No margin from one layer')."),

 dict(id="O-035",
   claim_plain="The 0.732 the abstract compares against is a single-source trained baseline - "
     "one dataset trained on, another tested on - averaged over the collaborator's six datasets "
     "at layer 40.",
   stated_numbers="0.732",
   source_ref="docs/COLLABORATOR_BRIEFING.md:653-654 (also :248, :317)",
   evidence_file=COLLAB,
   evidence_key="summary.csv row features='baseline' col off_diagonal_mean; matrix_baseline.csv",
   level1_value="summary.csv L40 baseline off_diagonal_mean = 0.7320325942",
   level1_match="YES",
   level2_value="recomputed as the off-diagonal mean of his 6x6 matrix_baseline.csv = 0.732027 "
     "over n=30 pairs (the 6e-6 gap is his own rounding in the CSV). It is a per-target mean over "
     "the five other sources, averaged over the six targets - i.e. a single-source mean, exactly "
     "as described. L20 reads 0.6478 and L44 0.7156.",
   level2_method="own reader; the summary value recomputed from the raw transfer matrix",
   version_used="his branch at head 45b16d9 and f386a72 - identical",
   robustness="The per-target spread is wide: 0.571 (harm_refusal_choice) to 0.827 "
     "(instruct_dishonesty) at L40, and the worst single pair is 0.060.",
   verdict="HOLDS", confidence="HIGH", importance="MEDIUM-HIGH - it is the baseline the whole "
     "abstract rests on",
   script=SC1 + "; " + SC3,
   notes="Exact, and the characterisation 'single-source mean' is right."),

 dict(id="O-036",
   claim_plain="A reviewer may not accept 0.732 as 'the trained probe' when the median over the "
     "same pairs is 0.722 and the sign-oriented mean is 0.735.",
   stated_numbers="0.722; 0.735",
   source_ref="docs/COLLABORATOR_BRIEFING.md:653-655",
   evidence_file=COLLAB + " (matrix_baseline.csv at L20/L40/L44) vs "
     "audit/cache/labpc/run-premise/audit/audit_L40_s42.json",
   evidence_key="per-target median over sources and per-target sign-oriented mean, averaged over "
     "targets - the same two estimators as our audit file's baselines block",
   level1_value="the briefing's 0.722 and 0.735 are, verbatim, OUR values in the 5b table "
     "(audit_L40_s42.json baselines.median_over_sources 0.7219698909 and oriented_mean "
     "0.7353614302), computed over our 110 pairs on our eleven datasets",
   level1_match="NO for the stated scope: 'the same pairs' as 0.732 means HIS 30 pairs",
   level2_value="Applying the identical estimators to HIS 6x6 matrix_baseline.csv: L40 median "
     "over sources 0.7608, sign-oriented mean 0.7850 (mean over sources 0.7320, oracle 0.9186). "
     "L20: 0.6581 / 0.7448 against a mean of 0.6478. L44: 0.7435 / 0.7659 against 0.7157. So on "
     "HIS pairs the two alternatives are 0.761 and 0.785, not 0.722 and 0.735.",
   level2_method="own script applying the audit file's own estimator definitions (per-target "
     "statistic, then averaged over targets) to his committed transfer matrix",
   version_used="his branch read out of git; our side from the cached lab audit file. Both "
     "verified during this session.",
   robustness="The direction of the argument is unaffected and in fact strengthened: on his own "
     "pairs the median (0.761) and the sign-oriented mean (0.785) both exceed 0.732 by MORE than "
     "our figures do, so the abstract's margin shrinks further under either alternative.",
   verdict="DOES NOT HOLD",
   confidence="HIGH - both sides measured this session with the same estimator definitions",
   importance="HIGH - it is the closing sentence of the document and the one the collaborator is "
     "most likely to paste into a rebuttal",
   script=SC3 + "; " + SC2,
   notes="ERROR THAT WOULD REACH THE COLLABORATOR. 0.722 and 0.735 are our eleven-dataset "
     "numbers; 'the same pairs' as 0.732 are his six-dataset ones, where the same estimators "
     "read 0.7608 and 0.7850. Fix: either quote his numbers ('the median over the same pairs is "
     "0.761 and the sign-oriented mean 0.785') or keep ours and say so ('on our own eleven "
     "datasets the same three estimators read 0.676 / 0.722 / 0.735'). The section-5b table's "
     "row label '(what the abstract compares against)' is what seeds the confusion."),
]

path = "audit/phase2/claims_batch_W6C-briefing.csv"
os.makedirs(os.path.dirname(path), exist_ok=True)
with io.open(path, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    for r in ROWS:
        w.writerow({c: r.get(c, "") for c in COLS})
print("wrote %s with %d rows" % (path, len(ROWS)))
