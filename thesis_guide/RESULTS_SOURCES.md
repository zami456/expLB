# RESULTS_SOURCES — where every Results-chapter number comes from

_Written 2026-09-24 for future sessions drafting the thesis Results chapter (LaTeX, `thesis/`).
Read-only survey; every statement cites the file it was read from (`path:line`). "UNKNOWN" = not
established by any file opened for this guide. This file is a MAP, not evidence: per the project's
evidence rule, re-open the raw source before writing any number (CLAUDE.local.md "Evidence rules")._

**Pins (use these, never a moving ref).** `origin/paper` = `16def2deec34aeee532511df116adb7de5522ca7`
(collaborator zami456); `origin/feat/persona-pc-ood` = `45b16d90…` (ancestor, nothing newer); owner =
`asix` HEAD `e917ecd3` + working tree (`audit/BRANCH_PINS.md:12-18`). Helpers `show()/tab()` read a
file at a pinned SHA (`audit/scripts/p17_common.py:8-33`). Ledger `kd/docs/FINDINGS.md` = F-01…F-128,
md5 `f81856ef` after the F-128 append (`audit/BRANCH_PINS.md:70`).

---

## (a) Where every kind of thing lives

| kind | location | notes |
|---|---|---|
| Findings ledger (owner) | `kd/docs/FINDINGS.md` (6,771 lines, F-01…F-128) | append-only; corrections are NEW entries (`FINDINGS.md:1-6`). F-128 (`:6719-6772`) partially corrects F-111/112/117/118/120/127. |
| Claim verdicts | `audit/claims.csv` (643 rows; ids D-/D2-/F-/L2-/M-/O-/S-/H-/A-) | F-01…F-110 have `F-` rows; F-111…F-127 are `L2-001…L2-017` (source_ref = ledger line). F-128 has no row. Verdict counts `audit/p17/P17_FACTS.md:77`. |
| F-entry → primary file | `audit/phase1/fmap_all.csv` (F-01…F-127) | columns `resolved_primary`, `producing_script`, `superseded_by`, `mapping_status`. |
| Experiments | `audit/experiments.csv` (119 rows) | O## = owner, C## = collaborator, AN# = print-only analyses, BG# = upstream. `results_location` + `f_ids`. |
| Cross-branch comparison | `audit/phase3/DISAGREEMENTS.md` | the only like-for-like numeric comparison of the two pipelines (§0 `:24-57`). |
| Result drafts (audit) | `audit/RESULTS_CHAPTER.md` (collaborator draft tables re-derived), `audit/RESULTS_BY_AXIS.md` (per-axis, per-scale), `audit/METHODS_CHAPTER.md`, `audit/LIMITATIONS_AND_FUTURE_WORK.md`, `audit/HPKR_WORDING_CONTROLLED.md` | drafts carry dated CORRECTION/UPDATE blocks; read the block, not the text under it. |
| Collaborator record | `audit/COLLABORATOR_FINDINGS.md`, `audit/COLLAB_PAPER_BRANCH.md` | his numbers are "attested, not reproduced": no activations committed (`COLLABORATOR_FINDINGS.md:19-23`). |
| P17 re-pin facts | `audit/p17/P17_FACTS.md`, `A_REPORT.md`, `C_REPORT.md`, `paper_new_claims.csv`, `csv_drift.csv` | P17_FACTS may not be cited alone; cite its evidence file (`P17_FACTS.md:3-5`). |
| Collaborator paper | `origin/paper@16def2de:src/probes/deception-detection/paper/` — `sections/*.tex`, `generated/numbers.tex` (1,176 macros @16def2de, F-128 `FINDINGS.md:6753`), 11 `generated/tab_*.tex`, 5 `generated/fig_*.pdf`, `make_numbers.py`, `make_figures.py` | `git ls-tree` @16def2de. |
| Collaborator results | `origin/paper@16def2de:src/probes/deception-detection/results/` — `persona_ood_ext_L{20..76 step 4,79}/` (summary.csv, zero_shot.csv, zero_shot_operating.csv, matrix_<arm>.csv), `persona_depth/depth.csv`, `loo_L{20,40}/`, `persona_positions_L{20,40,44}/`, `persona_ood_followup_L20/`, `persona_ood_mod_L*/`, `gender_secret/` | `XXXX-9-llama-all*/` hold only plots (`RESULTS_CHAPTER.md:519-525`). |
| Draft PDF | `full paper draft for this experiments/main (1).pdf` (collaborator's, `audit/PROJECT_MAP.md:37`) | superseded by `paper/main.pdf` @16def2de for current wording. |
| Owner results, laptop | `run-crossgen/` (kd11, banks/kd, banks/kd_s23, root matrices), `run-coords/*.json` (Q1–Q11 outputs), `run-paper16/`, `kd/results/fig_data.json`, `cap/`, `scale/`, `results/` (Liars' Bench), `docs/thesis/*.md` | laptop holds partial copies only (`PROJECT_MAP.md:19`). |
| LAB-ONLY stores | lab PC `C:\Users\user4\expLB\` (run-coords 77.6 GB, run-crossgen, run-premise, run-cap, run-kd, run-scale) and `C:\Users\user4\expLB-resplit\` (audit reruns) | `PROJECT_MAP.md:21`; read-only, `ssh -o BatchMode=yes` (CLAUDE.local.md). |
| Local caches of lab files | `audit/cache/labpc/` (expLB-resplit mean_L{20,40,44} + km/, run-premise/ incl. clustered/, six/, why/, falsify/; run-crossgen/kd11/banks/{collab,kd,kd_s23}; scale_3b/8b; run-kd responses/scores) and `audit/p17/labcache/` (kdscale_*, scale_70b_paper16fp32, kd11/passA11/passC11, banks/cap_s23, passP, passB_fine, phase6/) | directory listings above; `.meta.json` sidecars in p17/labcache. |
| Drive | `tamara:expLB/` (copy of lab results; `share/`, `thinking/plan.md`, `thesis-docs/`) (`PROJECT_MAP.md:23`); `tamara:expLB/_labpc_only_backup_2026-09-24/run-scale-kd` (`FINDINGS.md:6057`) | rclone copy-only rules in CLAUDE.local.md. |
| Plan defining Q1–Q11 | `~/.claude/plans/harmonic-cuddling-kay.md` (`docs/ONBOARDING.md:31`); one-line answers `docs/ONBOARDING.md:57-81` | dated 09-17; stale per `P17_FACTS.md:93`. |
| Thesis LaTeX | `thesis/main.tex` inputs `chapters/chapter_{1,2,3,5,6,9}.tex` (`thesis/main.tex:125-145`) | no results chapter file identified. |

---

## (b) How to verify a number before writing it

1. **Raw data is the only source of truth**; prose (ledger, reports, memory, drafts, this file) is a
   map (CLAUDE.local.md "Evidence rules" 1-3). Open the matrix/CSV/JSON and cite path + key/row.
2. **Other branches only at a pinned SHA**, via `git show <SHA>:<path>` or `p17_common.show/tab`
   (`audit/scripts/p17_common.py:8-26`). `origin/paper` moved ea1ce52b→2ad758fb→d1b88154→16def2de during
   Phase 6 (`BRANCH_PINS.md:44-51`); 80 baseline cells drifted (max |Δ| 0.0054, raw-probe only)
   (`P17_FACTS.md:21`), e.g. CG raw L40 0.7875→0.7873 (`FINDINGS.md:6733`).
3. **Cite the path and key**, as `REPORT_TECHNICAL.md:195-206` does (`premise_L40_mean_s42.json`
   `summary.single`, `clustered_L40_mean_s42.json` `summary.all.gain_loco`). For LAB-ONLY numbers give
   the lab path AND the local cache path.
4. **Check the correction chain first**: `fmap_all.csv` `superseded_by`, the F-entry's
   `supersedes:` line, and F-111/F-114/F-123/F-128 correction lists.
5. **Name the setting**: bank (s3 vs s23; paper16 vs paper16-shared), arm (zero-shot 1-d vs trained),
   dataset set (his 6 incl. `repe_honesty` vs our 11/10/5), layer, pooling, probe, k. Cross-branch
   numbers are on different scales unless DISAGREEMENTS.md says otherwise (§4 below).
6. **Controls travel with every transfer number**: raw, `datapca` (matched control, from the ROOT
   store — bank stores carry their own `datapca`, `FINDINGS.md:5462-5466, 5508-5511`), `random`,
   `length` (0.5555 off-diagonal, `docs/ONBOARDING.md:84-88`); zero-shot AUROCs vs random directions,
   not 0.5 (`ONBOARDING.md:94-96`). Significance bar = dataset-level bootstrap (`FINDINGS.md:15-17`).
7. **1-d arms are zero-shot**: trained off-diagonal of a dim=1 arm equals its zero-shot AUROC; 30 cells
   hold 6 distinct numbers (`RESULTS_CHAPTER.md:139-162`); put intervals over test datasets.

---

## (c) Results by theme

### T1 — Replication of persona coordinates at 70B: persona axes vs raw and the matched control (Q2, Q6)

- **Verified claims.** Against raw the persona gain is ~0 or negative at 70B; against the matched
  dataset-PCA control it is not significant. F-17 (HOLDS; `FINDINGS.md:272-279`, "inherited").
  Q2: `pc:deception` never beats raw; persona axes peak L44 0.731, raw L32 0.717
  (`ONBOARDING.md:59-60`; M-057 HOLDS). Q2/Q6 gaps significant over rows, not over datasets (M-085,
  M-088 HOLDS; `ONBOARDING.md:90-97`). F-04: 1 of 384 tests clears (HOLDS "on the reconstructed family";
  F-111 item 3-4 `FINDINGS.md:5332-5344`: the "11-dataset" figures are block "9" = all but
  `sycophancy_mt`; the one significant test is `random@2 − raw` losing). F-112 (L2-002 PARTLY):
  `pc:kd_assertion − raw` +0.10…+0.15 from L32, 1 BH-clearing test (L68), 0 vs `datapca`
  (`FINDINGS.md:5439-5473`; F-128 scopes "8 clear BH" → 34 family-wide, `:6749`). passB_fine: `axes − raw`
  negative at 36/40 cells; `hf275_strict` `pc:deception` +0.039 over raw at all 10 layers, never
  bootstrapped (F-126, L2-016 HOLDS, `FINDINGS.md:6609-6617`).
- **Do not quote:** F-31 "nothing beats raw" (superseded by F-112, `:5505`); "block 9 = 9 datasets"
  (`:5336-5338`); the 70B "essentially nil vs raw" (F-100 → F-110).
- **Raw source.** `run-crossgen/kd11/matrices/mean_L{20,40,44}_s42.json` (laptop + lab), significance
  `run-crossgen/kd11/significance.json` and lab-canonical `run-crossgen/kd11/banks/kd/significance.json`
  (6,300 tests, `FINDINGS.md:5423-5437`); Q2/Q6 `run-coords/{layer_sweep_*,q6_layers,significance}.json`;
  passB_fine LAB-ONLY `run-crossgen/passB_fine/` (cache `audit/p17/labcache/expLB/run-crossgen/passB_fine`).
  Scripts `src/assistant_axis_ndif/27_crossgen.py`, `22_q6_layers.py`, `23_significance.py`;
  `audit/scripts/p13_all_results_by_axis.py`.
- **Both branches?** Yes, different estimators. His raw baseline trains on 100% unbalanced source,
  ours on a balanced 80% split → his raw is higher by +0.056/+0.065 at L40/L44 on the 5 shared
  datasets; never quote the two margins in one sentence (DISAGREEMENTS.md:201-237, D-04).
- **Figures.** `figures/persona-coords/{q2_layer_sweep,q6_layers,significance_forest,transfer_L44_last}.pdf`;
  `figures/crossgen/figure6_style/` (unified panel confounded by question family — M-094 HOLDS).

### T2 — Scale ladder 3B / 8B / 70B

- **Verified claim (use this).** F-121 (L2-011 HOLDS): best persona feature minus raw, layer-averaged,
  5 datasets, k=3, mean: paper16 +0.0382 / +0.0148 / **−0.0116**; kd all +0.0317 / +0.0169 / +0.0032;
  kd s23 +0.0360 / +0.0194 / +0.0054 — both banks decline (`FINDINGS.md:6026-6043`). Scope: 5 datasets
  contain no harm-pressure data (`:6045-6048`). Per-feature table `RESULTS_BY_AXIS.md:61-87`
  (`pc:deception` 3B +0.0324 → 70B −0.0321). Versus matched control: +0.120 / +0.103 / +0.054 like-for-like
  (F-110 HOLDS, `FINDINGS.md:5259-5271`); mean-pooling result only (F-102, F-110 `:5271`).
- **Chain / do not quote:** F-58 "replicates at 3B/8B, dies at 70B" (REVISED → F-63; `:2034`); F-63's
  3B +0.202 is max of 7 layers (F-96, M-027); F-100's 70B +0.075 and "+0.0079 vs raw" ran the kd bank
  under a paper16 label (F-110 `:5242-5252, 5283`); CLAUDE.local "+0.036 → +0.019 → +0.005" for the
  36-persona bank matches neither F-121 row exactly (kd all ends +0.0032, s23 +0.0054) — re-read before use.
- **Raw source.** LAB-ONLY `C:\Users\user4\expLB-resplit\run-crossgen\kdscale_{3b,8b,70b}_{all,s23}\matrices\`
  and `expLB\run-crossgen\scale_{3b,8b,70b_paper16fp32}\` (`FINDINGS.md:6054-6055`); caches
  `audit/p17/labcache/expLB-resplit/run-crossgen/kdscale_*`, `audit/p17/labcache/expLB/run-crossgen/scale_*`,
  `audit/cache/labpc/run-crossgen/scale_{3b,8b}`. Scripts `audit/scripts/p12_kd_scale_ladder.py` (runs on
  lab), `audit/scripts/fix_rf16_recompute.py`, `kd/scripts/paper_comparison.py`.
- **Both branches?** No — owner only (experiments O72-O77, O88). His paper cites Mahadik & Skapars'
  3B/8B only (`RESULTS_CHAPTER.md:348-355`).
- **Figures.** `figures/thesis/fig4_geometry_by_scale.pdf` (F-37 geometry, not the AUROC ladder).

### T3 — HP-KR: the inversion and the register diagnosis

- **Verified claims.** HP-KR is largely lexical: denial regex 0.913 on Llama rows (F-01 PARTLY: AUROC
  print-only, `fmap_all.csv` F-01); label = knows AND denies, fourth cell empty (F-02 PARTLY; cells sum
  6,943 not 7,068 per fmap notes). F-13 (HOLDS, status provisional): trained paper16 axes 0.427/0.300 on
  HP-KR, 0.355/0.379 on `sycophancy_mt`, random beats both (`FINDINGS.md:217-237`). His published 8v7
  contrast zero-shot HP-KR 0.148 (L20) / 0.593 (L40); base probe 0.109 L20 (`RESULTS_CHAPTER.md:61-83`).
  Length alone 0.776 (M-080 HOLDS).
- **Do not quote:** 0.870 lexical AUROC (all 7,068 rows; superseded, `FINDINGS.md:29`); "0.374 is the
  sycophancy axis" — it is the paper16 **deception** axis PC1; sycophancy axis = 0.313 (F-111 item 8,
  `:5370-5384`); "0.553 vs 0.665" length pair (matches no matrix, `:5357-5368`).
- **Raw source.** `results/subsets/harm-pressure-knowledge-report.csv` (git = lab); feature code
  `src/assistant_axis_ndif/eval_datasets.py:363-389` (fmap F-01); `run-crossgen/kd11/matrices/mean_L40_s42.json`;
  his `persona_ood_ext_L{20,40,44}/zero_shot.csv` @16def2de.
- **Both branches?** Yes, and **"the paper's 8v7 axis" is a different bank on each side**: his =
  our repaired `paper16_shared` (match 0.0015 over 10 cells); our kd11/root stores use the
  question-confounded bank, up to 0.148 apart on HP-KR L20 (0.1029 vs 0.2504) (DISAGREEMENTS.md:110-150, D-02).
  Five HP-KR-L20 values circulate (D-05, `:239`).
- **Figures.** `figures/thesis/fig1_register.pdf` (F-57 PARTLY); his `generated/fig_hpkr.pdf`
  (figure↔table mapping UNVERIFIED).

### T4 — The knowledge-denial extension (the collaborator's headline)

- **Verified claims.** Zero-shot, his 6 datasets: `ext_axis1` (12v13) 0.904/0.916/0.903 L20/40/44 mean;
  `denial_axis1` (4v6) HP-KR 0.890/0.919/0.908; published 8v7 mean 0.791/0.832/0.826 (Table 6,
  `RESULTS_CHAPTER.md:218-226`; 165/9/2 draft verdicts after P17, `FINDINGS.md:6733-6736`). F-97 (HOLDS):
  ~44-46% of the headline margin is the kd bank (`FINDINGS.md:4384-4420`); F-98/F-99 (HOLDS) replicate on
  our 11 datasets (21/21 layers); F-103 pooling-robust. Our `pc:kd_assertion` (4 feigned vs 3 plain
  assertion) is the best-transferring persona feature: +0.150 over raw at L40, 11 datasets, k=5
  (`RESULTS_BY_AXIS.md:93-105`), but does not clear `datapca` (F-112).
- **Key qualification (D-03, VERIFIED).** Same s23 bank: his 4v6 reads HP-KR 0.87-0.88; our 4-vs-3
  sincere-ignorance split reads 0.59-0.67; the plain-assertion pole 0.91-0.93 — the merged axis is carried
  by plain assertion (DISAGREEMENTS.md:156-199; `HPKR_WORDING_CONTROLLED.md:115-141`).
- **Do not quote:** F-85's "+0.184 → +0.023" (superseded by F-97, `FINDINGS.md:3405-3409`); F-98's
  "reproduced exactly" without saying it rebuilt from s23 while his headline uses s3 (D-01,
  DISAGREEMENTS.md:59-108); "his s3→s23 swap costs nothing" (F-62) — his control shows a loss
  (`COLLABORATOR_FINDINGS.md:227-235`).
- **Raw source.** His `persona_ood_ext_L{20,40,44}/{zero_shot,summary,matrix_*}.csv` and
  `paper/generated/numbers.tex` @16def2de; ours LAB-ONLY `run-crossgen/kd11/banks/{kd,collab}/matrices/`
  (cache `audit/cache/labpc/run-crossgen/kd11/banks/`). Scripts `audit/scripts/p16_level2.py`,
  `p16_level2_verdicts.py`, `p16_level1*.py`, `p17_level{1,2}_pinned.py`; `audit/scripts/phase3/p3b_bankmatch.py`.
- **Both branches?** Yes. Pipelines agree on 55 bank-matched cells, mean |gap| 0.0015
  (DISAGREEMENTS.md:24-57). Different dataset sets explain 0.9158 (his 6) vs 0.7996 (our 11); on the
  5 shared datasets 0.8990 vs 0.8882 (s3 vs s23) (DISAGREEMENTS.md:417-429).
- **Figures/tables.** His `generated/fig_personas.pdf` (Figure 2 geometry; 0.47 cosine etc. rebuilt
  exactly, F-126 `FINDINGS.md:6621-6642`), `fig_transfer.pdf`, `tab_hpkr.tex`, `tab_zeroshot.tex`,
  `tab_transfer.tex`, `tab_pertest.tex`, `tab_roster.tex`.

### T5 — The wording-matched convergence (report as ONE finding)

- **Verified claims.** Three tests where both classes deny: (1) his 12v13 axis on our `kr-matched`
  slice, zero-shot 0.4630/0.6203/0.5383 (`HPKR_WORDING_CONTROLLED.md:121`, reproduces from the collab-bank
  kd11 matrices — F-128 correction to F-118's pointer, `FINDINGS.md:6751`); (2) F-116 (L2-006 HOLDS)
  `knowledge_matched` 2,001 rows: raw 0.999 in-distribution, persona `axes` transfer 0.3850/0.3335/0.5267,
  `axes − raw` −0.19 to −0.38 (`FINDINGS.md:5849-5867`); (3) F-118 (L2-008 PARTLY, pointer only)
  Gender-Secret, every direction 0.436-0.623 (`FINDINGS.md:5926-5942`). F-115 (L2-005 HOLDS): leak
  guard leaves F-03's margin (−0.2545/+0.0309/+0.0050) but `KR→matched` at L40 falls to 0.4262
  (`FINDINGS.md:5809-5826`). F-03/F-72/F-76: L40 matched-slice margin not significant, inverts at L20
  (F-76 HOLDS, `FINDINGS.md:2846`).
- **Do not quote:** F-75's "strengthens" verdict (withdrawn by F-76, `:2904`); "F-115 §5" as a source;
  F-116's zero-shot 0.2405/0.6466/0.6085 as evidence (93.5% overlap by construction, `:5872-5875`).
- **Raw source.** Cached: `audit/cache/labpc/expLB-resplit/mean_L{20,40,44}_s42.json` (F-115) and
  `.../km/mean_L{20,40,44}_s42.json` (F-116); LAB-ONLY originals in `C:\Users\user4\expLB-resplit\`.
  Collab-bank matrices cache `audit/cache/labpc/run-crossgen/kd11/banks/collab/`. Gender-Secret:
  `origin/paper@16def2de:.../results/gender_secret/{summary.md,zero_shot.csv}` (not re-derived; no
  activations, `FINDINGS.md:5949`). Scripts `audit/scripts/p6_patch_leakguard.py`,
  `p6_resplit_compare.py`, `p6_build_km_view.py`, `p5_hpkr_matched.py`.
- **Both branches?** Yes, complementary, not the same test: (1)+(2) ours, (3) his; direction agrees.
  Our Q11 gender-secret (base-model probe on passQ11) is also at chance (`ONBOARDING.md:78-82`) —
  different detector; like-for-like UNKNOWN.
- **Figures.** None identified. UNKNOWN.

### T6 — Sycophancy and the capitulation bank (Q10, cap/)

- **Verified claims.** Q10: paper16 sycophancy axis inverted on `sycophancy_mt` (0.28-0.31), only the
  Assistant Axis detects it (0.71) (`ONBOARDING.md:75-77`; M-046/M-047/M-048 HOLDS). F-65 (PARTLY — 0.374
  attribution): cap contrast zero-shot 0.759/0.745/0.728 L20/40/44 vs 0.374 (deception PC1) / 0.313
  (sycophancy axis) (`FINDINGS.md:2095-2120`; F-111 `:5370-5384`). F-66 (HOLDS): `pc:capitulation@2`
  beats raw +0.112 [+0.050,+0.185] and both matched controls at L40/L44 on the `sycophancy_mt` column
  (`:2206-2235`). F-67 (PARTLY): local — over 11×11 +0.098 vs raw, CI spans 0 (`:2285-2310`). F-77
  21/21 layers; F-87 seed-stable; F-89→F-94 presence beats content ~2:1 (F-94 PARTLY, supersedes F-89's
  "78% / three-to-one"). His side: `syc_pc1` tops his excl-HP-KR table L40-56, margin is mostly
  `convince_me`, zero-shot (F-122, L2-012 HOLDS; `RESULTS_BY_AXIS.md:134-171`). Ours: `pc:sycophancy`
  trained loses to raw at all 6 (layer,pooling) cells on 11 datasets; 0/264 bootstrap comparisons clear
  (F-126 `FINDINGS.md:6551-6577`).
- **Do not quote:** "pc:sycophancy never computed anywhere" (`RESULTS_BY_AXIS.md:109-114`); F-32's
  bank-v2 acceptance counts (F-111 item 2, `:5315-5328`); "worse than random (0.541)" as a matched pair
  (zero-shot vs trained; matched pair 0.3551 vs 0.5413, `:5379-5381`).
- **Raw source.** LAB-ONLY `run-crossgen/kd11/banks/{cap_s23,cap_all,cap_ctrl}/` (cache
  `audit/p17/labcache/expLB/run-crossgen/kd11/banks/cap_s23`); `run-cap/responses` (cache
  `audit/p17/labcache/expLB/run-cap/responses`); Q10 `run-coords/q10_sycophancy_axis.json`; his
  `persona_ood_ext_L*/matrix_syc_*.csv`, `zero_shot.csv` col `auroc_syc_axis`. F-126 LAB-ONLY
  `expLB-resplit\phase6\run-crossgen\` (cache `audit/p17/labcache/expLB-resplit/phase6/run-crossgen/`).
  Scripts `cap/scripts/{cap_significance,cap_layer_profile,cap_zeroshot}.py`,
  `src/assistant_axis_ndif/24_q10_sycophancy_axis.py`.
- **Both branches?** Sycophancy axis yes (his 6 datasets zero-shot vs our 11 trained — not like-for-like;
  both say "best away from sycophancy", F-126 `:6572-6577`). Cap bank: owner only. His Table 11 HP-KR
  inversion 0.043/0.167/0.480 (`RESULTS_CHAPTER.md:276-295`).
- **Figures.** `figures/thesis/fig5_capitulation.pdf`, `figures/persona-coords/q10_sycophancy_axis.pdf`,
  his `generated/tab_syc.tex`.

### T7 — The positive control: pooled training without personas

- **Verified claims.** F-79/F-80 (HOLDS): pooled LODO 0.8545 vs single-source 0.6763, +0.178, clears
  dataset bootstrap (`FINDINGS.md:3045-3070`). F-108 (HOLDS): `kr-matched` is 100% HP-KR rows;
  leave-one-CLUSTER-out gain +0.1254 at L40 (10/11 positive), +0.1289 L20, +0.1309 L32; clean-9
  +0.150-0.157 (`:5164-5189`). **Publish +0.125** (`REPORT_TECHNICAL.md:195-206`). F-86/F-95 why-pooling
  (mean-of-directions 0.8828, `:3761-3790` F-90); F-90 Result 2 +0.028 clears "only just".
- **Caution.** The "five seeds, ±0.035" travel with the LODO gain (seeds +0.173…+0.198, F-83), not with
  the LOCO +0.125, which is one seed and has no dataset bootstrap (F-108 caveats `:5198-5202`;
  `REPORT_TECHNICAL.md:205-206`). `gain_lodo` is 0.1782 in `premise_L40` but 0.1814 in `clustered_L40`
  (`REPORT_TECHNICAL.md:198`, `REPORT_PLAIN.md:112`) — name the file.
- **Do not quote:** +0.178 as the headline; F-79's HP-KR 0.972 ("substantially the leak", `:5174-5189`);
  "0.850 or 0.806" single draws (F-90 → 0.8344 ± 0.0200).
- **Raw source.** LAB-ONLY `run-premise/{premise_L*_mean_s*.json,clustered/,six/,why/,falsify/,audit/}`,
  cached in `audit/cache/labpc/run-premise/`. Scripts `src/assistant_axis_ndif/30_premise.py`, `30b`…`30g`.
- **Both branches?** Yes: his `loo_L{20,40}/` (C15/C16). Axis minus his pooled baseline, excl. HP-KR:
  +0.1408/−0.0197 (L20 naive/balanced), +0.0016/+0.0640 (L40) (`REPORT_TECHNICAL.md:285-296`,
  `audit/scripts/p4_loo_baseline.py`). Our six ≠ his six (`ai_liar` for `repe_honesty`, F-109 `:5228`).
  Like-for-like pooled-vs-pooled comparison: UNKNOWN (none found).
- **Figures.** His `loo_L40/loo.png`; ours none identified.

### T8 — Layer / depth

- **Verified claims.** Q6: L20 vs L22 immaterial (median 0.012); L40 worth ~+0.025 raw, ~+0.113 persona
  (M-067/M-068/M-071 HOLDS). F-91 (PARTLY): L32 peak ~0.02 above L40. F-75/F-76: plateau not
  significant. His 16-layer sweep: `ext_axis1` peaks L36 0.9275 vs L40 0.9158; vs his baseline positive
  16/16 (+0.115…+0.273) (F-117 L2-007 PARTLY, `FINDINGS.md:5889-5909`); excl-HP-KR `ext_axis1 − ext_pc3`
  11/16 positive, −0.033…+0.063, 6 sign changes (F-128 `:6750`). Late-layer rise = raw degrading +
  anisotropy; random gains +0.127 (F-120 L2-010 PARTLY: 5-dataset reading; all-10 raw L40 = 0.6760,
  `:6752`). D2-005 and D2-019 DOES NOT HOLD (`P17_FACTS.md:27-37`).
- **Do not quote:** "within ±0.06 at every layer", "crosses zero five times" (F-117); "outscores at
  every depth"; raw 0.8195 as an all-dataset figure; the paper sentences at `introduction.tex:68-70`
  and `limitations.tex:17-21`.
- **Raw source.** His `persona_depth/depth.csv`, `persona_ood_ext_L*/summary.csv` @16def2de
  (`audit/scripts/p8_collab_layer_sweep.py`); ours `audit/phase2/layer_sensitivity.csv`
  (`audit/scripts/p2_w1_layer_sensitivity.py`, `p9_late_layer.py`), LAB-ONLY `run-crossgen/kd11/passA11/`.
- **Both branches?** Yes; different setups (his 6 datasets zero-shot; ours 11/10 trained). Peak L36 vs
  L32 is not a like-for-like disagreement (D-11 predates his sweep, `DISAGREEMENTS.md:427`). UNKNOWN.
- **Figures.** His `generated/fig_depth.pdf`, `persona_depth/depth.png`; ours
  `figures/thesis/fig2_layer_profile.pdf`, `figures/persona-coords/q6_layers.pdf`.

### T9 — Token position and pooling (Q8)

- **Verified claims.** Q8: zero-shot top-decile beats mean (+0.053); trained pooled-mean beats
  token-trained at every layer (−0.03…−0.06) (M-072/M-073 PARTLY, M-076 HOLDS). His Tables 8-10:
  follow-up-token sign not universal; single max beats top-10% at L40 (0.750/0.548)
  (`RESULTS_CHAPTER.md:388-406`). Pooling: F-101/F-102 (PARTLY) 70B gain is pooling-dependent;
  F-103 kd share pooling-robust. passP is a prompt-span store: CG and sandbagging 1.000 from the prompt
  alone (F-126 `FINDINGS.md:6583-6607`).
- **Raw source.** `run-coords/{pertoken_q8,q8_trained_pertoken}.json`; his
  `persona_positions_L{20,40,44}/reads_summary.csv`, `persona_ood_followup_L20/`; passP cache
  `audit/p17/labcache/expLB/run-coords/passP`.
- **Both branches?** Yes: same direction, magnitudes ~4× apart because ours pools all datasets into
  one AUROC and his averages per-dataset AUROCs (D-10, `DISAGREEMENTS.md:390-415`; our side DOC-ONLY there).
- **Figures.** `figures/persona-coords/q8_pertoken.pdf`, `q8_trained_pertoken.pdf`; his `tab_reads.tex`,
  `tab_followup.tex`.

### T10 — Operating point and calibration

- **Verified claims.** Alpaca-calibrated 1%-FPR thresholds do not hold: 23/90 cells exceed 1%; L20
  4v6 HP-KR BA 0.533 at 88.3% FPR; L40 0.910 at 6.5% (`RESULTS_CHAPTER.md:85-99`). LB upper bound 0.95
  runs at 8% FPR (`:107-111`). Table 4: at L20 the base probe wins BA/recall, persona arms only AUROC
  (`:311-333`). Our M-078 (PARTLY) zero-shot axis best at 1% FPR at L40-44.
- **Raw source.** His `persona_ood_ext_L{20,40,44}/zero_shot_operating.csv` @16def2de; Liars' Bench
  Tables 6-8 in `papers/2511.16035v2 (1).pdf` p.54-56.
- **Both branches?** His measured; ours only M-078. Comparison UNKNOWN.
- **Figures.** His `generated/fig_operating.pdf`, `tab_operating.tex`.

### T11 — Remaining plan questions and the sarcasm gates

- **Q1** paper16 halves Assistant-Axis contamination −0.66 → −0.37 (M-043 HOLDS, M-044 PARTLY);
  vs Lu's published axis −0.14…−0.23 (M-030 PARTLY). **Q3** probe-dependent (M-058 HOLDS). **Q4** erasure
  costs 0.005-0.018 (M-056 PARTLY, M-059…M-063 HOLDS). **Q5** semantics does not predict transfer
  (M-055, M-064, M-065 HOLDS). **Q7** judges 91.2% agree (M-014/M-015 HOLDS). **Q9** peak at n=4: no
  M- row found — UNKNOWN verdict. **Q11** sleeper 0.96 = falsity detection (M-050…M-053 HOLDS)
  (`ONBOARDING.md:57-82`). Sources `run-coords/{bank_run*,q3_q4_bootstrap,q5_*,q11_sleeper_*,persona_count_L40,zeroshot_nulls}.json`;
  scripts `src/assistant_axis_ndif/2{0..6}_*.py`. Owner only. Figures `figures/persona-coords/q{1,3,5,9,11}_*.pdf`.
- **Sarcasm:** G2 fails — ironic vs sycophantic 0.92 separable by bag-of-words (F-93 HOLDS, re-verified
  F-105); Gate 3: deception axis at random on natural sarcasm, sycophancy fires 0.71 (F-107 HOLDS; result
  numbers not persisted per fmap F-107). `docs/thesis/sarcasm-pilot-negative-result.md`; cache
  `audit/cache/labpc/cap/run-sarc-pilot/`. Owner only.

### T12 — Setup facts the chapter needs (corpus, bank)

- Table 1 counts (CG 267/167 … Alpaca 2,000/0) and Table 5 (3,749 kept of 5,400) exact
  (`RESULTS_CHAPTER.md:375-386`); his follow-up run uses RepE 612/306 not 550/275. `repe_honesty` is not a
  Liars' Bench dataset — "six LB datasets" is five (`RESULTS_CHAPTER.md:591-592`). Bank definitions:
  `kd/personas/kd_role_groups.json`, `src/assistant_axis_ndif/pc_role_groups.json` (`RESULTS_BY_AXIS.md:12-51`).
  Row identity across branches proven for 5 shared datasets (`DISAGREEMENTS.md:40-44`).

---

## (d) Known-open / unverified items

1. **202 inventory claims have no verdict**: 110 ledger (6C, TODO) + 92 collaborator notes (6B,
   CANDIDATE); coverage 533/735 = 72.5% (`FINDINGS.md:6762-6763`; `P17_FACTS.md:82-84`). The "45%" and
   "378 unchecked" in CLAUDE.local.md / `audit/COVERAGE_GAP.md` predate P17 (`P17_FACTS.md:84`).
2. **D2-024, D2-025 UNRESOLVED** — pooling constants hard-coded; `results_table.csv` not committed
   (`FINDINGS.md:6768`). **D-030, D-063 UNRESOLVED** — Appendix-G four-decimal reproduction; artifact
   exists nowhere outside the collaborator's machine (`FINDINGS.md:6653-6657`).
3. **0.9597 regex** not recorded; a guess gives 0.9413 (`FINDINGS.md:6769`;
   `COLLABORATOR_FINDINGS.md:176-182`). Also unverified: "median gap 0.012 over 17 methods" (`:6760`).
4. **LAB-ONLY evidence** (no git copy): F-110, F-113 (part), F-114, F-115, F-116, F-120, F-121, F-126
   (`FINDINGS.md:6766`). F-121's 290-file Drive backup not checked (`:6767`).
5. **Print-only numbers** with no persisted output: F-01 AUROC, F-02 cross-tab, F-107 results, AN1-AN5
   analyses incl. F-96/F-98…F-103 (`audit/experiments.csv` AN1-AN5; `fmap_all.csv`).
6. **F-20's 98-row / 89:9** unrecoverable (`FINDINGS.md:5484-5503`). Figure 2 not regenerable from the
   repo (36 vectors uncommitted, `RESULTS_CHAPTER.md:515-517`); `persona_depth/summary.md` cites two
   folders not on the branch (`FINDINGS.md:5961-5962`). F-57's fig3 matches no committed script
   (`FINDINGS.md:5759`). C4: third bank (`persona knowledge denial modified`) unreported
   (`COLLABORATOR_FINDINGS.md:188-190`).
7. **Single seed / no significance** on most tables (his three-layer tables, F-121, F-117) —
   `RESULTS_CHAPTER.md:584-590`, `RESULTS_BY_AXIS.md:193-202`.
8. **95 DOC_FIXES proposals**: correction blocks were applied to 10 docs but the header still says none
   were (`P17_FACTS.md:68-72, 90`) — read the blocks, not the old text.
