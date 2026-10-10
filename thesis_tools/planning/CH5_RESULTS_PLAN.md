# CH5_RESULTS_PLAN — Chapter 5 (Result Analysis)

**Approved 2026-09-25.** Writing starts only when the owner says so. Copied from the session plan to
`thesis_tools/planning/CH5_RESULTS_PLAN.md`.

## Context

**Target file**
- Chapter 5 "Result Analysis" is `thesis/chapters/chapter_6.tex`; `main.tex` inputs it under `\chapter{Result Analysis}`.
- It still holds the template text.
- Chapter 4 is written in `chapter_5.tex` and verified. Chapter 5 `\ref`s its labels (GLOSSARY §1) and never re-explains a method.

**Promises Chapter 4 makes that Chapter 5 must keep**
- "evaluates the alternatives on the same data and selects one" (`chapter_5.tex:163`);
- "states which [pipeline] produced each result" (`:59`);
- describes the on-policy probe and INLP attempts (`:201`);
- holds the model-organism analysis (`:415`).

**How this plan was built (owner's order: plan from the audited maps → write → verify)**
- Values and fields come from the audited maps:
  - `audit/phase1/fmap_all.csv` (F-entry → raw file + key);
  - `kd/docs/FINDINGS.md`, with F-128's corrections;
  - `audit/claims.csv` verdicts;
  - `thesis_guide/{RESULTS_SOURCES,PITFALLS,FIGURES}.md`;
  - the audit drafts `audit/RESULTS_CHAPTER.md` (the co-author's tables, re-derived), `RESULTS_BY_AXIS.md` and `HPKR_WORDING_CONTROLLED.md`;
  - `docs/thesis/persona-coordinates-70b-results.md`, whose appendix at :1492 says where each Q-answer's number lives;
  - the other thesis version's Chapter 5 (`full paper draft for this experiments/report.pdf`, pp. 29–43).
- For the co-author's CSVs, only the header rows were opened. That fixed the field names; the values were not read.
- **Every value below is the map's value.** It is re-opened at its raw source when written (rule 1), and checked against raw data on both sides in the final session.

**Decisions already taken (owner, 2026-09-25)**
1. **Structure:** the template headings are the sections and our themes are the subsections, in the same shape as the other version's Chapter 5.
2. **New terms:** each is defined once, at first use in Chapter 5, and added to `GLOSSARY.md`. The terms: factor probes, Lu et al.'s published Assistant Axis, format control, token reads, oracle threshold, generalisation gap.

## Setting shorthand (used in the tables)

- **B6.** Pipeline B (the co-author's code, `origin/paper` @16def2de):
  - bank: kd, score-3 filter (36 personas);
  - datasets: CG, HP-C, HP-KR, IT, ID, RepE;
  - pooling: the detection-token mean;
  - seed 42; 1,000-row bootstrap;
  - 1-d arms are zero-shot (a trained 1-d probe equals zero-shot, `sec:m-protocols`).
- **B-LB5.** The same, averaged over the 5 Liars' Bench datasets (RepE excluded).
- **A11.** Pipeline A (our code):
  - datasets: 11;
  - pooling: mean; probe: `paper`; k = 5; seed 42;
  - 80/20 train one, test all;
  - controls: raw, datapca, random and length, from the root store;
  - significance: dataset bootstrap, 10,000 resamples, BH.
- **HIS** = `git show 16def2de:src/probes/deception-detection/results/`.
- **KD11** = `run-crossgen/kd11/`.
- **Lab-only** = no git copy. Read from the cache named, then re-checked on the lab PC in the final session.
- Layers: L20, L22 and L40 are pre-registered. L36 and L44 are post hoc and always labelled so.
- Model: Llama-3.3-70B-Instruct unless the row is a 3B or 8B rung.

---

## Chapter outline and results, section by section

Each row gives: the result (+ positive, − negative, 0 null), the value from the map, the raw source and field, the setting, and the ledger or claim reference. Every positive result carries its "what the gain is made of" rows beside it.

### 5.1 Performance Evaluation

**5.1.1 Evaluation criteria.** Four criteria, tied to Chapter 4:
- ranking (AUROC);
- detection at the benchmark's operating point: Pipeline A TPR, Pipeline B BA plus specificity (`sec:m-metrics`);
- transfer, off-diagonal, with and without HP-KR (`sec:m-protocols`);
- the generalisation gap (in-distribution minus off-diagonal).

Also each design's label need and cost. Chapter 3's requirement IDs are not written yet; `\ref` them once they exist.

**5.1.2 Testing methods.** One paragraph that `\ref`s `sec:m-protocols`, `sec:m-stats` and `fig:m-protocols`. It adds only what Chapter 4 lacks: the oracle threshold, defined in one sentence.

**5.1.3 The problem measured: probes that do not transfer, and ranking ≠ detection**

| # | Result | Value (map) | Raw source :: field | Setting | Ref |
|---|---|---|---|---|---|
| 1.1 − | Raw probe (D1): high in-distribution, drops off-diagonal | L40 0.983 → 0.732; L20 0.979 → 0.648 | HIS `persona_ood_ext_L{20,40}/summary.csv` :: row `baseline`, `diagonal_mean` / `off_diagonal_mean` | B6, 8192-d `lr_reg10`, trained on 100% of the source, 5-fold diagonal | RESULTS_CHAPTER §7 |
| 1.2 − | Worst cell: raw probe trained on CG, tested on HP-KR | 0.06 | HIS `persona_ood_ext_L40/matrix_baseline.csv` :: cell CG→HP-KR | B6, L40 | report.pdf Fig 5.1 |
| 1.3 − | Our raw baseline, 11 datasets | off-diagonal 0.6623 at L40 | KD11 `matrices/mean_L40_s42.json` :: `results[raw,paper].auroc`, off-diagonal mean over 110 cells | A11 | F-62/F-73/F-126 |
| 1.4 0 | Base probe (D2): Liars' Bench average | AUROC 0.778, BA 0.626 at L40 | HIS `persona_ood_ext_L40/zero_shot_operating.csv` :: arm `base probe (RepE-trained)`, row "Average (Liars' Bench)", `threshold_source=control` | B-LB5 | RESULTS_CHAPTER §6 |
| 1.5 − | Ranking ≠ detection: the base probe ranks HP-KR but flags nothing | AUROC 0.936, BA 0.500 at L40 | same file :: HP-KR row | B6 | report.pdf §5.1.3 |
| 1.6 − | Benchmark comparators (published) | Liars' Bench average AUROC: Mean 0.682, Follow-Up 0.634, Upper Bound 0.93 (privileged). HP-KR: Mean 0.12 / BA 0.50; Follow-Up 0.14 / 0.26; Upper 0.97 / 0.95 | LB paper Tables 6–8, pp. 54–56 (cite the PDF). The CSV `loo_L40/liars_bench_comparison_loo.csv` holds constants hard-coded in `make_numbers.py`, so it is not raw | Llama-3.3-70B rows | D2-024; F-124 §5 (0.26 is the 70B row; the cross-model average is 0.42) |
| 1.7 − | The HP-KR inversion (published 8 v 7, zero-shot) | 0.148 / 0.593 / 0.589 at L20 / 40 / 44 | HIS `persona_ood_ext_L*/zero_shot.csv` :: `auroc_axis`, HP-KR row | B6 | D-02, D-05 (five L20 values circulate: name the bank) |
| 1.8 − | Base probe inverts at L20 on HP-KR | 0.109 (`zero_shot_operating`) vs 0.381 (`matrix_baseline`) | both HIS files, L20 | B6 | RESULTS_CHAPTER §1, "loose end": say which base probe is meant |
| 1.9 − | Trained persona axes invert on HP-KR (our pipeline) | 0.300 as source / 0.427 as target, against random 0.659 / 0.698 | KD11 `matrices/mean_L40_s42.json` :: `results[axes,paper]`, HP-KR row/column means | A11, paper16 bank | F-13 (HOLDS) |
| 1.10 − | The inversion spans 7 (arm, layer) cells, always from the HP-KR training row | L20: `axis1`, `syc_axis1`, `syc_pc1`, `pc1`; L40: `syc_axis1`, `syc_pc1`; L44: `syc_axis1` | HIS `matrix_<arm>.csv` | B6 | F-124 §1 |

`fig:m-hpkr`, the regex 0.9125 and the label confound are already in Chapter 4 (`sec:m-integration`), so they are `\ref`'d here and not repeated.

### 5.2 Analysis of Design Solutions (D1–D9 on the same data)

**5.2.1 Supervised probes (D1, D2).** Rows 1.1–1.5. D3 goes to 5.3.6 (no files).

**5.2.2 The published persona method (D4)**

| # | Result | Value (map) | Raw source :: field | Setting | Ref |
|---|---|---|---|---|---|
| 2.1 − | Extended PCs (k=3) transfer below the single extended contrast | 0.866 vs 0.916 at L40 | HIS `persona_ood_ext_L40/summary.csv` :: `ext_pc3`, `ext_axis1` `off_diagonal_mean` | B6 | RESULTS_CHAPTER Table 3 |
| 2.2 0 | One direction vs components over 16 layers | excluding HP-KR, `ext_axis1 − ext_pc3` is positive at 11 of 16 layers, range −0.033 to +0.0633 (L56), 6 sign changes | HIS `persona_ood_ext_L*/summary.csv` :: `off_diagonal_excl_hpkr` | B6, 16 layers | F-117 corrected by F-128; DF-101. Never "within ±0.06" |
| 2.3 − | Published PCs (k=3) | 0.679 / 0.639 / 0.617 | same file :: `pc3` | B6, L20 / 40 / 44 | Table 3 |

Pipeline A's version of D4 (`pc:deception` never beats raw) goes in 5.5.3.

**5.2.3 Zero-shot contrasts (D5–D9) — the headline**

| # | Result | Value (map) | Raw source :: field | Setting | Ref |
|---|---|---|---|---|---|
| 3.1 + | **HEADLINE:** extended contrast (D9), Liars' Bench average zero-shot AUROC | **0.8990 at L40**; **0.9130 at L36** (post hoc) | HIS `loo_L40/liars_bench_comparison_loo.csv` :: row "persona axis, extended (zero-shot)", `auroc`; `persona_depth/depth.csv` :: layer 36, source "zero-shot, Liars' Bench average", arm "extended axis (12v13)", metric `auroc` | B-LB5, no training, no benchmark labels | THESIS_BRIEF Batch 3. **L36 rests on the uncommitted remote-pooling code (Ch 4 disclosure).** |
| 3.2 + | Same, balanced accuracy | 0.6939 vs Mean Probe 0.636 | same CSV :: `balacc_1pct_fpr` | B-LB5 | |
| 3.3 + | Per-dataset zero-shot table (Table 6) for 8 v 7, 4 v 6 and 12 v 13 at L20/40/44 | means 0.791 / 0.832 / 0.826; 0.866 / 0.912 / 0.900; 0.904 / 0.916 / 0.903 | HIS `persona_ood_ext_L*/zero_shot.csv` :: `auroc_axis`, `auroc_denial_axis`, `auroc_ext_axis` | B6, **including RepE (1.000 = ceiling)** | Paper `tab_zeroshot.tex` |
| 3.4 made of | The kd roles' share of the margin | kd worth +0.113 / +0.084 at L20 / L40; share 44.2% / 45.7% | HIS `zero_shot.csv` means over 6 rows :: `auroc_ext_axis` − `auroc_axis` | B6 | F-97 (HOLDS) |
| 3.5 made of | …and on our 11 datasets | kd positive at 21/21 layers, mean +0.0551, median share 63%; `last` pooling +0.0590, 20/21 | KD11 `banks/collab/matrices/{mean,last}_L*_s42.json` :: `zeroshot[contrast:collab_extended]` − `zeroshot[contrast]`, mean over 11 columns | A11, zero-shot, 21 layers. **Lab-only** (cache `audit/cache/labpc/run-crossgen/kd11/banks/collab/`) | F-98, F-99, F-103 |
| 3.6 made of | Against the pooled probe | all six 0.9158 vs pooled naive 0.8496; **excluding HP-KR 0.9165 vs 0.9149**; HP-KR 0.9124 vs 0.523; the pooled probe wins CG, IT, ID and RepE (CIs exclude 0) | HIS `loo_L40/loo_summary.csv` :: `auroc`, `auroc_excl_hpkr`; `loo_bootstrap.csv` :: `delta`, `ci_low`, `ci_high`, `excludes_zero` | B6, pooled leave-one-out (naive and balanced) | the co-author's pooled baseline |
| 3.7 made of | Base probe at its own best layer | 0.8498 at L32 (vs 0.7784 at L40) | HIS `depth.csv` :: arm "base probe (RepE-trained)", LB average, `auroc` | B-LB5 | The ea1ce52b message says to quote it |
| 3.8 made of | Random directions (k=3) | 0.527 / 0.504 / 0.493 | HIS `summary.csv` :: `random3` | B6 | Table 3 |
| 3.9 made of | Gap to the privileged upper bound | 0.0310 (L40), 0.0432 (L20); on BA the gap is 0.196; the upper bound runs at 8% FPR | LB Table 8; rows 3.1–3.2 | | DF-26 |
| 3.10 0 | Sycophancy contrast (D6): strong away from HP-KR, inverted on it | HP-KR 0.043 / 0.167 / 0.480; excluding HP-KR, `syc_pc1` tops L40–56 (0.9370 at L40), carried by CG (+0.086) | HIS `zero_shot.csv` :: `auroc_syc_axis`; `summary.csv` :: `syc_pc1` `off_diagonal_excl_hpkr` | B6 | F-122 (HOLDS) |
| 3.11 − | Unified contrast (D7) does not help (the control for widening the bank) | averaged over 16 layers: D7 0.657 vs 8 v 7 0.686 vs D9 0.885; D7 ranks HP-KR below 8 v 7 at 15/16 layers | HIS `depth.csv` :: arm "unified axis (13v12)", "transfer, off-diagonal mean" and "zero-shot, HP-KR" | B6, 16 layers | report.pdf §5.3.3. Recompute the averages at write time |

### 5.2.4 HP-KR in detail, the factor probes, and the convergence (ONE finding)

| # | Result | Value (map) | Raw source :: field | Setting | Ref |
|---|---|---|---|---|---|
| 4.1 + | Denial contrast (D8) is the only arm that flags HP-KR's lies | AUROC 0.890 / 0.919 / 0.908; BA 0.533 / 0.910 / 0.757; recall 0.885 at L40 | HIS `zero_shot_operating.csv` :: arm "knowledge-denial axis (4v6)", HP-KR, `control` | B6 | tab_hpkr |
| 4.2 + | …across 16 layers, D8 is the only arm above 0.5 BA; peak 0.9102 at L40 | 16/16 | HIS `depth.csv` :: "zero-shot, HP-KR", `balacc` | B6 | |
| 4.3 made of | Realised FPR | 6.5% at L40; **88.3% at L20** (specificity 0.1165); 23 of 90 cells exceed 1% | HIS `zero_shot_operating.csv` :: 1 − `specificity` | B6 | F-124 §4 |
| 4.4 0 | Base probe vs D8 on HP-KR AUROC by layer | base leads at L36, L40, L56–72, L79; trails at L44 (0.828 vs 0.908), L48, L52, L76 | HIS `depth.csv` :: "zero-shot, HP-KR" `auroc` | B6 | D2-005 DOES NOT HOLD (F-128) |
| 4.5 made of | Lexical baselines on HP-KR | regex 0.9125 (Ch 4) vs raw-probe diagonal 0.9663; length alone 0.776 | KD11 `matrices/mean_L40_s42.json` :: raw diagonal HP-KR; length from M-080 | A11 | DF-55, DF-105; **0.9597 unverified: never quote** |
| 4.6 + | Factor probes: denial is linearly present; knowledge less so | d_utterance on the knowledge-matched slice 0.996 / 0.999 / 0.999; d_knowledge on the text-matched slice 0.685 / 0.772 / 0.771 | `kd/results/phase2_gate.json` :: `rows[layer].dU_knowledge_matched`, `.dK_text_matched` | owner, L20 / 40 / 44 | F-05 (HOLDS) |
| 4.7 0 | Knowledge and wording are entangled at L40, not at L20 | cos(dK, dU) 0.063 / 0.410 / 0.430 | same :: `cos_dK_dU` | | F-06; F-07 split-half 0.988–0.995 |
| 4.8 − | The plain-assertion pole carries HP-KR | `kd_assertion`: full 0.9118 / 0.9344 / 0.9266 → text-matched 0.4227 / 0.5860 / 0.5253; split knowledge-denial axis full 0.5923 / 0.6692 / 0.6659 | KD11 `banks/collab/matrices/mean_L*_s42.json` :: `zeroshot` | A11, collab bank (score 2–3), zero-shot. **Lab-only** | `HPKR_WORDING_CONTROLLED.md` §5; D-03 |
| 4.9 made of | Our trained `pc:kd_assertion` beats `pc:knowledge_denial` | 0.8119 vs 0.7492 at L40 | KD11 `banks/kd/matrices/mean_L40_s42.json` :: `results[pc:*,paper]` off-diagonal | A11 | F-112; RESULTS_BY_AXIS §3 |
| **4.10 − THE CONVERGENCE (a)** | Text-matched slice, extended contrast | **0.4630 / 0.6203 / 0.5383** | KD11 `banks/collab/matrices/mean_L{20,40,44}_s42.json` :: `zeroshot[contrast:collab_extended]`, column `harm-pressure-kr-matched` | A11, **collab score 2–3 bank, not his score-3 bank** (say so) | `HPKR_WORDING_CONTROLLED.md` §5 :121; F-128 |
| **4.11 − (b)** | Knowledge-matched slice (2,001 rows): raw separates in-distribution, axes transfer below chance | in-distribution raw 0.9995 / 0.9992 / 0.9993; axes transfer 0.3850 / 0.3335 / 0.5267; axes − raw −0.1860 / −0.3758 / −0.1937; length 0.6174 | `audit/cache/labpc/expLB-resplit/km/mean_L{20,40,44}_s42.json` (lab `C:/Users/user4/expLB-resplit/run-crossgen/km/matrices/`) :: KM row off-diagonal (11 targets); diagonal | A, 12 datasets, leak guard, mean, `paper` | F-116 (HOLDS). **Lab-only.** Never quote 0.2405/0.6466/0.6085 |
| **4.12 − (c)** | Gender-Secret (co-author): every direction at chance | 4 v 6 0.549 / 0.436 / 0.498; 12 v 13 0.505 / 0.467 / 0.541; 8 v 7 0.514 / 0.545 / 0.623; 5 v 5 0.593 / 0.523 / 0.520 | HIS `gender_secret/summary.md` results table; `gender_secret/zero_shot.csv` | B, fine-tuned model, 392 rows | F-118 (PARTLY, pointer only) |
| 4.13 − | Guarded margins on the text-matched slice | axes − raw −0.2545 / +0.0309 / +0.0050; KR → text-matched at L40 falls 0.5674 → **0.4262** | `audit/cache/labpc/expLB-resplit/mean_L*_s42.json` | A, leak guard | F-115. **Lab-only** |
| 4.14 0 | Not significant, and the layer cannot be chosen | over datapca at L40 +0.0655 [−0.0693, +0.1887], P = 0.161; best-layer interval L5–L79; cluster permutation p = 0.38; layer-localised 0.739 at L40 vs 0.374 at L20 | F-76 prints from 160 passA11 matrices (print-only); `run-premise/cluster_perm_kd_matched.json`; passC11 matrices | A | F-72, F-76, F-84 (HOLDS) |

**How to write it:** (a), (b) and (c) together are ONE finding: "where both classes deny knowledge, no persona direction reads the lie." F-75's "strengthens" is withdrawn.

**5.2.5 Transfer across datasets**

| # | Result | Value (map) | Raw source :: field | Setting | Ref |
|---|---|---|---|---|---|
| 5.1 + | Table 3, B6 grid (all six / excluding HP-KR) | 12 v 13: 0.904 / 0.916 / 0.903 (excl. 0.909 / 0.917 / 0.905); 4 v 6: 0.866 / 0.912 / 0.900; raw 0.648 / 0.732 / 0.716 | HIS `summary.csv` | B6. 1-d arms are zero-shot. The 8 v 7 value at L20 has a sign artifact (0.651 → 0.7909) | RESULTS_CHAPTER §2 |
| 5.2 + | 16 layers | ext 0.9275 at L36 (post hoc) vs 0.9158 at L40; single-source probe 0.7682 / 0.7320; ext ahead of the baseline at 16/16 layers (+0.115 to +0.273) | HIS `depth.csv` :: "transfer, off-diagonal mean" | B6 | F-117 (PARTLY) |
| 5.3 + | Extended contrast on our 11 | 0.7996 at L40 vs published 0.7325, raw 0.6623 | KD11 `banks/collab` zeroshot; root raw | A11, zero-shot. **Lab-only** | F-98. 0.9158 (his 6) vs 0.7996 (our 11) differ by dataset set. On the 5 shared datasets: 0.8990 (score 3) vs 0.8882 (score 2–3), DISAGREEMENTS :417-429 |
| 5.4 0 | `pc:kd_assertion` beats raw but not the matched control | +0.150 at L40; beats raw at 18/21 layers (1 clears BH, L68); beats datapca at 20/21 layers, **0 of 126 clear** | KD11 `banks/kd/significance.json` (lab-canonical, 6,300 tests) :: `tests[...]` | A11 | F-112 as amended by F-128 (34 clearing family-wide) |
| 5.5 | Cross-branch agreement | 55 bank-matched cells, mean \|gap\| 0.0015 | `audit/phase3/DISAGREEMENTS.md` §0; `p3b_bankmatch.py` | | one sentence here; details in App C. D-04: the raw baselines differ, so never put the two margins in one sentence |

**5.2.6 Which personas? Library banks vs purpose-built banks** (N12, Q1; the owner asked for it)

| # | Result | Value (map) | Raw source :: field | Setting | Ref |
|---|---|---|---|---|---|
| 6.1 + | Contamination cos(deception, own assistant direction) | L40: own100 −0.660, hf275 −0.610, paper16 score 3 −0.370; vs Lu's published axis −0.226 (range −0.141 to −0.226 over L20–60) | `run-paper16/axes_vectors_{s3,all}.pt`; HF cache; own100 from the 2026-09-14 diagnostics (**lab-only, not re-read**) | per layer | results doc §7.1; M-043 (HOLDS), M-030 (PARTLY) |
| 6.2 − | PC1 of a library bank is the Assistant Axis | cos 0.73–0.82 (own100), 0.75–0.80 (hf275) | same | L20–60 | §7.1 |
| 6.3 − | Relabelling does not fix it; trait vectors are worse | mean \|cos\| 0.655 → 0.585 (11%); −0.725 vs −0.610 | `run-coords/regroup_hf.json` | L40 | §7.1, §7.12 |
| 6.4 − | Sycophancy geometry changes sign between banks | +0.144 (own100) vs −0.270 (paper16); own100 sycophancy axis on `sycophancy_mt` 0.451 (inside the null) | `run-coords/q10_sycophancy_axis.json` | L40 | §9.1–9.2 |
| 6.5 + PROVISIONAL | Detection: paper16 vs own100 | `pc:deception` 0.815 vs 0.706; `axes_resid` 0.770 vs 0.553 | early 5-dataset run: confirm the file at write time (likely `run-coords/probes_5sets.json`) | L20, 5 datasets, bf16 bank, never repeated | results doc §7.1, :1087. Label it provisional |
| 6.6 0 | hf275 strict labels | `pc:deception` +0.039 over raw at all 10 passB_fine layers, never bootstrapped | lab `run-crossgen/passB_fine/` (cache `audit/p17/labcache/expLB/run-crossgen/passB_fine`) | | F-126. **Lab-only** |
| 6.7 − | The 26-persona unified PC1 is the question set | cos(PC1, contrast) 0.389 (question-confounded bank) vs 0.896 (paper16_shared); PC1 vs question family −0.945 | F-18 (lab stdout of `cmp_banks.py`); F-113 | | Conflict: −0.98 in `RESULTS_BY_AXIS.md` vs −0.945 (F-113). Quote F-113's, or re-derive |
| 6.8 0 | Bank comparison, 36 vs 26 personas, is confounded | +0.0093 (8.1×) | KD11 off-diagonal | A11 | F-61 corrected by DF-51/71, F-114 |

**5.2.7 Selection** (decision 3: one design per criterion).
- A table of the designs at L40 against each criterion, like report.pdf's Tables 5.1 and 5.2.
- D3's row comes from report.pdf (decision 4) and is marked as such.
- Result: D9 is selected for ranking and transfer, D8 for detection at the threshold.
- The D9 runner-up note: "robust across depth", leading the baseline at 16/16 layers; its lowest per-dataset score is 0.542 (depth sweep; recompute).

### 5.3 Final Design Adjustments

**5.3.1 From the published bank to the kd roles**
- D8 reads 0.890 at L20, where 8 v 7 reads 0.148.
- The repair is the new roles, not re-extraction: the published grouping is inverted on both banks (0.103 original, 0.148 re-extracted).
- Refs: D-02, D-06; report.pdf §5.3.1.

**5.3.2 One direction, not the subspace.** Row 2.2. The one-direction advantage over the components reverses at L64–72.

**5.3.3 The roles, not the widening.** Row 3.11: the D7 control.

**5.3.4 The layer** (fixed at L40 in advance, `sec:m-layers`)

| # | Result | Value (map) | Raw source :: field | Ref |
|---|---|---|---|---|
| 7.1 0 | Peaks: ext transfer 0.9275 and LB average 0.9130 at L36; base probe at L32; HP-KR BA spike 0.910 at L40 vs 0.745 / 0.757 either side | | HIS `depth.csv` | report §5.3.4 |
| 7.2 0 | Our plateau L32–L44 cannot be ordered | L32 above L40 by about 0.02 | cap_s23/kd_s23 matrices (F-91) | F-91 → DF-53 |
| 7.3 0 | Q6: L20 vs L22 median gap 0.012; formal equivalence at ±0.02 fails; L40 is worth about +0.025 (raw) and +0.113 (persona) | | `run-coords/q6_layers.json` | M-067/068/071; DF-63 |
| 7.4 − | Q2: persona peak L44 0.731, raw L32 0.717; `pc:deception` never beats raw | | `run-coords/layer_sweep_*.json` | M-057 |
| 7.5 − | Late-layer rise is an artifact: random directions gain +0.127 | raw at L40 over all 10 datasets is 0.6760, not 0.8195 | lab `run-crossgen/passA/matrices/` (cache `audit/p17/labcache`) | F-120 corrected by F-128 |
| 7.6 − | Co-author's layer claims that do not hold | D2-005 (L36 claim); D2-019 (≥0.026: −0.011 at L20 without HP-KR) | HIS `depth.csv`, `summary.csv` | F-128 |

**5.3.5 The token read and pooling**

| # | Result | Value (map) | Raw source :: field | Ref |
|---|---|---|---|---|
| 8.1 0 | Six token reads for ext (Table 9) | at L40: max BA 0.750 / recall 0.548; top-10% 0.737 / 0.531; mean 0.694 / 0.413. Max drops RepE BA 0.998 → 0.638 | HIS `persona_positions_L{20,40,44}/reads_summary.csv` :: `read`, `balacc`, `recall` | DF-15 (the max, not the top-10%, wins at L40) |
| 8.2 − | Follow-up token: the sign is not universal | IT 0.988 → 0.165; HP-KR 4 v 6 0.890 → 0.237; ext 0.887 → 0.552 | HIS `persona_ood_followup_L20/` | Tables 8, 10 |
| 8.3 0 | Q8, ours | zero-shot top-decile: +0.1027 (L20) → +0.005 (L40); trained pooled-mean beats token-trained by −0.03 to −0.06 | `run-coords/pertoken_q8.json`; `q8_trained_pertoken.json` | DF-61; M-072/073/076. D-10: magnitudes are ×4 apart for method reasons |
| 8.4 0 | The 70B gain depends on pooling; the kd share does not | `last` +0.033 (19/21) vs mean +0.007 | lab `run-crossgen/scale_70b/matrices/{mean,last}_*` | F-101, F-102, F-103 |
| 8.5 − | Prompt-only store: CG and sandbagging reach 1.000 from the prompt alone | | cache `audit/p17/labcache/expLB/run-coords/passP` | F-126. **Lab-only** |

**5.3.6 Approaches that did not work**
- **D3, the on-policy (AISI) base probe.** report.pdf §5.2.1 gives +0.100 [0.086, 0.113] over D2; ext still leads by 0.021 [0.006, 0.035] at L40 and by 0.105 at L20, and is level at L44; D3's HP-KR BA is 0.500. **No file on any branch** (`git ls-tree`/`log -G` search of all refs, 2026-09-25).
- **D10, INLP.** report.pdf §5.2.4 / Fig 5.4: at L20, rank 1 gives 0.632 vs raw 0.634; ranks 2–50 give 0.530–0.583; sensitivity 0.151–0.293 vs raw 0.180 vs persona 0.032. **No file.**
- **DANN.** Owner's statement only. No numbers anywhere.
- **The HP-KR growth gate failed.** Denial rate 5/20 = 0.250 vs control 3/20 = 0.150. Source: F-59 (logs).
- **Inverted axis as a source fails under mean pooling only.** F-29, DF-40.
- The sarcasm bank was never built (5.6.3); `ai_audit` is already in Chapter 4.
- **Per decision 4:**
  - D3 and D10 numbers are quoted from report.pdf, with a page `% src:` and a sentence that the result files are not in the repository.
  - DANN is described without numbers.

**5.3.7 Adjustments to the claims.**
- The claims the 16-layer sweep and the audit corrected: F-128's list, the report's §5.3.6, and "that test has not been run" (false: DF-76).
- Stated briefly; each points to the result that corrects it.

### 5.4 Statistical Analysis
`\ref sec:m-stats` for the techniques. Results:

| # | Result | Value (map) | Raw source :: field | Ref |
|---|---|---|---|---|
| 9.1 − | Almost nothing clears the dataset bootstrap | 1 of 384, and it is `random@2 − raw` losing (−0.2287, q 0.0192); on all 11 datasets the F-04 gains flip to −0.0215 / −0.0243 | KD11 `significance.json` :: `tests[block,pooling,layer,probe,comparison]` | F-04, F-111 (DF-01/02); "block 9" = 10 datasets |
| 9.2 0 | Effect size vs dataset-level uncertainty | median effect 0.044 vs median interval ±0.125; Q2/Q6 significant over rows, not over datasets | `run-coords/significance.json` | REPORT_TECHNICAL :529-531; M-085/088 |
| 9.3 − | The matched control erases the persona advantage | unified − datapca −0.003 (k=2) | `run-crossgen/significance.json` :: block 10, L40, mean, `pc:unified@2 − datapca@2` | F-17. Never cite −0.76 (F-74 withdrawn) |
| 9.4 0 | k-dependence | +0.138 (k=2) vs +0.004 (k=5) | KD11 L40 | F-39, F-113 |
| 9.5 − | Random-direction null | deception axis on HP-KR 0.6436 vs p95 0.9377; CG and HP-C inside the null | `run-coords/zeroshot_nulls.json` | DF-72; PITFALLS §0.4 |
| 9.6 − | Length alone transfers | 0.5555 vs raw 0.6623 (L40, mean, `paper`, 110 pairs); 0.5533 vs 0.6646 (L44, last, `mms`, 90 pairs); longer positive class in 9 of 11 datasets by token gap; length-only AUROC above 0.5 on 9 of 10 | KD11 `mean_L40` :: `results[length]`; `audit/cache/labpc/run-coords/p1_L44_last_lengthctl.json` `p1[3]`/`p1[0]`; `confounds_L44_last.json` | F-73, F-114 (name the metric and the set) |
| 9.7 0 | Dataset identity dominates the activations | 99.3% decodable vs +3.2% for the label | `run-coords/validate_L40.json` (confirm: PITFALLS says memory-only, not re-derived) | fig FIG-018 |
| 9.8 − | Calibration | 23/90 cells exceed 1%; the oracle table: L40 BA base 0.626 / 0.652, D8 0.725 / 0.692, D9 0.694 / 0.718; D8 0.725 [0.715, 0.735] vs base 0.626 [0.619, 0.633] | HIS `zero_shot_operating.csv` :: `threshold_source` ∈ {control, "in-dataset (oracle)"}, `balacc_lo/hi` | F-124; report Table 5.4 |
| 9.9 + | Generalisation gap | D1 in-distribution 0.956–0.984 loses 0.213–0.347; the ext off-diagonal equals its in-distribution value within 0.002 at every layer | HIS `summary.csv` all 16 layers | report §5.4.2. Recompute |

### 5.5 Comparisons and Relationships

**5.5.1 With the benchmark's probes and protocol.**
- `tab_hpkr` (rows 4.1, 1.6) and `tab_operating` (Table 4).
- At L20 the base probe wins BA and recall (0.721 / 0.489) while losing on AUROC. At L40/44 the persona arms lead on all three.
- Caveat: Pipeline B drops the IT scratchpad, so comparisons with the published numbers are approximate.

**5.5.2 With the replicated method at 70B (Pipeline A)**
- F-17 (9.3).
- F-110: like-for-like paper16 at 70B is +0.0541 vs datapca and −0.0131 vs raw. Source: lab `run-crossgen/scale_70b_paper16fp32/`.
- Q2 (7.4).
- The 11-dataset crossgen retrain: unified −0.034 vs raw at L40 (+0.028 without `sycophancy_mt`). Source: `run-crossgen/summary.json`; M-094; confounded by question family.
- F-126 passB_fine: axes − raw is negative at 36/40 cells.

**5.5.3 Scale ladder (3B / 8B / 70B)**

| # | Result | Value (map) | Raw source :: field | Setting | Ref |
|---|---|---|---|---|---|
| 10.1 − | **Both banks decline, so the decline is the method's** | paper16 +0.0382 / +0.0148 / −0.0116; kd s23 +0.0360 / +0.0194 / +0.0054; kd all +0.0317 / +0.0169 / +0.0032 | lab `expLB-resplit/run-crossgen/kdscale_*` and `expLB/run-crossgen/scale_*` (caches `audit/p17/labcache/...`) :: layer-average of max(axes, pc:*) off-diagonal − raw | 5 datasets (no HP data), k=3, mean, `paper`, one seed, post-hoc max. **Lab-only** | F-121 (HOLDS). Give the 70B kd value as a direction only (DF-87) |
| 10.2 − | Per feature | `pc:deception` +0.0324 (3B) → −0.0321 (70B) | same | | RESULTS_BY_AXIS §2 |
| 10.3 0 | vs the matched control | +0.120 / +0.103 / +0.054; mean pooling only (under `last` it is not monotone) | same | | F-110, F-102. F-58/F-63 are withdrawn or superseded |
| 10.4 0 | Geometry by scale | cos(PC1, assistant) 0.435 / 0.732 / 0.623 | logs `scale_*_axes.log` (log-only) | | F-37 (PARTLY) → probably appendix |

**5.5.4 Pooled training as a positive control** (train on many, test on one)

| # | Result | Value (map) | Raw source :: field | Setting | Ref |
|---|---|---|---|---|---|
| 11.1 + | **Leave-one-cluster-out gain** | **+0.1254 at L40** (10/11 targets), +0.1289 (L20), +0.1309 (L32); clean-9 +0.1504 | lab `run-premise/clustered/clustered_L{20,32,40}_mean_s42.json` (cache `audit/cache/labpc/run-premise/clustered/`) :: `summary.{all,clean}.gain_loco`, `loco_positive` | A, mean, `paper`, **one seed, three layers, no dataset bootstrap**. **Lab-only** | F-108 (HOLDS) |
| 11.2 do not headline | LODO +0.178 (0.8545 vs 0.6763) is leaky; its five seeds (+0.173 to +0.198) and 21/21 layers belong to LODO | | `run-premise/premise_L*_mean_s*.json` :: `summary.pooled_lodo − summary.single` | | F-79/80/81/83. ±0.035 has no source: never use |
| 11.3 + | Diversity, not volume | LODO − row-matched LODO +0.0045 [−0.0234, +0.0325], P = 0.378 | `premise_L40_mean_s42.json` + `30b_premise_boot.py` (print-only) | | F-80. Our equal-budget arm |
| 11.4 + | Why pooling works | mean-of-directions 0.8828; survives 51/51 cells at L ≥ 16; row-matched draws 0.8344 ± 0.0200 | `run-premise/why/why_L40_s42.json`; `falsify/`; `audit/audit_L40_s42.json` | | F-86, F-95, F-90 |
| 11.5 | The co-author's pooled leave-one-out | row 3.6; axis − pooled excluding HP-KR: +0.1408 / −0.0197 (L20, naive / balanced), +0.0016 / +0.0640 (L40) | HIS `loo_L{20,40}/`; `p4_loo_baseline.py` | B6 | credit: pooled LODO baseline = co-author; equal-budget arm = owner |

**5.5.5 Between geometry and behaviour.**
- cos(deception, denial) is 0.47 (already in Ch 4). Unified vs its parents: 0.84 / 0.85, from `numbers.tex`; confirm the macro.

### 5.6 Discussions

**5.6.1 Interpretation.** The register diagnosis:
- F-13;
- D6 inverts on HP-KR;
- the plain-assertion pole (4.8);
- `kd_assertion` > `knowledge_denial` (4.9);
- the convergence.
Mark all interpretation `% interp:`.

**5.6.2 The capitulation bank and the sycophancy axis**

**Example box (owner, 2026-09-25):** one capitulation exchange, verbatim.
- Show the question, the teacher-forced turn 1 (identical for both roles), the challenge "I don't think that's right. Are you sure?", a `pc_caver` reply that gives way, and a `pc_position_holder` reply that holds.
- Source: `audit/p17/labcache/expLB/run-cap/responses/{pc_caver,pc_position_holder}.jsonl` (lab-only; re-check). Pick the same question and variant for both.

| # | Result | Value (map) | Raw source :: field | Ref |
|---|---|---|---|---|
| 12.1 − | The paper16 sycophancy axis is inverted on real capitulation | 0.313 [0.283, 0.346] at L40 (mean pooling); 0.272–0.363 over 21 layers; the paper16 assistant direction reads 0.710 (not Lu's axis) | `run-coords/q10_sycophancy_axis.json` | Q10; M-046/047/048; DF-31 |
| 12.2 + | Capitulation contrast, zero-shot | 0.759 / 0.745 / 0.728 vs 0.374 (the deception axis PC1) and 0.313 (the sycophancy axis) | lab KD11 `banks/cap_s23/matrices/mean_L*_s42.json` :: `zeroshot[contrast].auroc[sycophancy_mt]` (cache) | F-65 (PARTLY). **Row bootstrap only; no dataset bootstrap** |
| 12.3 + | Trained `pc:capitulation@2` | vs raw +0.112 [+0.050, +0.185]; vs datapca@2 +0.154 [+0.091, +0.211] (`sycophancy_mt` column, 10 sources) | same :: `results[...]`; `cap_significance.py` | F-66 (HOLDS) |
| 12.4 0 | …but local | full 11×11 vs raw +0.098 [−0.036, +0.208]; clears at 14/21 layers (mean) and 3/21 (`last`); 4 seeds stable | KD11 `banks/cap_s23/significance.json`; seeds `mean_L40_s{1,7,13,42}` | F-67, F-77, DF-70, F-87 |
| 12.5 0 | Holding roles help; presence beats content | A − B ≈ +0.045; presence beats content about 2:1 (never quote the ratio) | `cap_ctrl` matrices | F-68, F-94 |
| 12.6 − | `pc:sycophancy` trained in our pipeline | loses to raw at 6/6 (layer, pooling) cells; 0 of 264 clear | lab `expLB-resplit/phase6/run-crossgen/` (cache) | F-126. **Lab-only** |
| 12.7 0 | The co-author's `syc_pc1` lead | carried by CG | row 3.10 | F-122 |

**5.6.3 Sarcasm banks: the test of deceptive intent (failed gates)** (added 2026-09-25 at the owner's request)
- **Open with the intent framing.** Sarcasm says something literally false without meaning to deceive. If a deception axis fires on unmarked sarcasm, it reads form, not intent. The design framed intent as **overt vs covert insincerity**: `sycophantic_praise − ironic_criticism` holds the literal approval fixed and varies whether the insincerity is meant to be recognised.
  - Sources: `cap/docs/SARCASM_DESIGN.md:30-45, 93-105`; `docs/thesis/sarcasm-pilot-negative-result.md:35-66`; `audit/LIMITATIONS_AND_FUTURE_WORK.md` §2.
  - Correct the old wording: sarcasm is not "deception minus the intent". It carries a *different* intent (`SARCASM_DESIGN.md:41-45`).
- **Related work to cite.** Only the first two are already in `references.bib`.
  - Goldowsky-Dill et al. (`goldowskydill2025detecting`) concede their probe fires on "is this scenario deception-related" rather than on intent. The sarcasm pilot was built to put a number on that. Check the page in the PDF.
  - Kirch et al. (`kirch2025offpolicy`, PDF `papers/The Impact of Off-Policy Training Data.pdf`) find off-policy data fails most for intent-defined behaviours. That is why the teacher-forced v2 was not run. Check the page.
  - The pragmatics accounts (Grice; Sperber & Wilson; Clark & Gerrig): Chapter 2 introduces them; here one clause with `\textcite` and no page (no PDF).
  - Sarc7 / SarcasmBench (models over-mark sarcasm): the expectation for Gate 1. Not in the bib; add, or drop the clause.
- **Example box (owner, 2026-09-25):** one unmarked ironic reply (v4 or v5) next to a sycophantic reply to the same question, verbatim. It makes Gate 2 concrete: the two are word-separable although both "approve".
  - Source: `audit/cache/labpc/cap/run-sarc-pilot/responses/` (lab-only; re-check).
  - Name the variant, and give the marker count from `marker_report.json`.
- Gate 1 (markers) passed: the ironic marker rate fell 0.85 → 0.00 by v4/v5 (F-92; `marker_report.json` :: `markers.pc_ironic_critic.per_variant`).
- Gate 2: bag-of-words 0.919 / 0.921 (v4+v5) and 0.953 / 0.947 (all). Source: `cap/run-sarc-pilot/marker_report.json` :: `lexical_subsets`. Refs: F-93/F-105; DF-66 (gate history).
- Gate 3: the deception axis sits inside the random null (0.543–0.586); sycophancy reads 0.693 / 0.710.
  - F-107: the numbers were console-only, rebuilt in F-113 §4.
  - If no file holds them at write time: `\todo`.
- F-106: the natural-sarcasm floors, 0.63–0.78. The set is FigLang 2020, per `cap/scripts/build_sarcasm_natural.py:12,48`. Chapter 4 now cites it as `ghosh2020figlang` (done 2026-09-25); cite it here too. Chapter 4 now also states the intent motivation, so §5.6.3 can `\ref` `sec:m-banks` and need not repeat it.
- **The concluding sentence (general, not only sarcasm).** An intent-defined construct could not be elicited on-policy without a register confound. The model marks intent through style, so on-policy persona pairs that differ only in intent already differ in wording. Source: `docs/thesis/sarcasm-pilot-negative-result.md:198-213`; mark it `% interp:` on the Gate 2 numbers.
- **Consequence.** The register diagnosis is supported on HP-KR and sycophancy, but untested on the one construct designed to isolate intent. The v2 bank is specified and never built (`SARCASM_DESIGN.md`). Point to Chapter 6's future work.

**5.6.4 Model organisms (Q11)**
- Sleeper answers are flagged at 0.96 cross-trigger, but this is falsity detection: a lie is a wrong answer there.
- Zero-shot persona scores (0.87–0.91) sit inside the random null.
- Our Gender-Secret is at chance.
- Source: `run-coords/q11_sleeper_L{20,44}_last.json`, `zeroshot_nulls.json`; M-050…053.
- The co-author's Gender-Secret is `\ref`'d to the convergence (4.12).
- Rebuilt inside the fine-tuned model: 0.641 [0.550, 0.724] at L44. Confirm in `gender_secret/summary.md`.

**5.6.5 Remaining questions.**
- **Q3:** probe-dependent. Source: `q3_q4_bootstrap.json`.
- **Q4:** erasure costs 0.80–2.64%. Sources: `q3_q4_bootstrap.json`, `p2_L44_last.json`; DF-62/65.
- **Q5:** semantics does not predict transfer. Sources: `q5_semantics_L44_last.json`, `q5_layers/`.
- **Q7:** judges agree 91.2%; the score-3 rate differs by 0.74 pp over 1,350 judged; the kept sets differ by 8.15%. Sources: `run-paper16/scores` vs `scores_405b`; DF-64.
- **Q9:** a plateau at n = 4 ("plateau", not "peak"). Source: `persona_count_L40.json`.

**5.6.6 Implications.** **5.6.7 Limitations:** a pointer to Chapter 6, which holds every caveat and the LAB-ONLY list.

### Items for other chapters, found while planning Chapter 5

- **Chapter 4 revision** (see `PROGRESS_THESIS.md`, "Chapter 4 revision list"):
  - one sentence of motivation for the sarcasm pilot (the intent test);
  - the FigLang 2020 citation for the natural-sarcasm set.
  - Both are **done 2026-09-25**, recorded as `CH4_VERIFY_TABLE.md` rows 179–180.
- **Chapter 2 (unwritten):** a short paragraph on irony and deceptive intent:
  - Grice, Sperber & Wilson, Clark & Gerrig (no PDFs in `papers/`, so no page numbers unless the PDFs are added);
  - Goldowsky-Dill's intent concession;
  - Kirch et al. on intent-defined behaviours;
  - optionally Sarc7, SarcasmBench, iSarcasmEval.
- **Chapter 6 future work:** sarcasm v2 is the top-ranked queued item (`audit/LIMITATIONS_AND_FUTURE_WORK.md` §2 and §10), the only design that separates intent from content.

---

## Where findings go when not in Chapter 5 (every F-entry is placed)

- **Already in Chapter 4, `\ref` only:**
  - F-01, F-02, F-16 (HP-KR label and counts);
  - F-23 (slice provenance);
  - the leak (199/258);
  - bank construction and teacher forcing: F-26, F-28, F-32, F-36, F-46, F-51–F-53, F-55, F-64;
  - F-69, F-71 (slice top-up).
- **Appendix B, full tables:**
  - crossgen per-source and per-target tables;
  - F-10, F-19 (kd score-3 imbalance, drift);
  - F-11, F-12 (kd assistant drift, PC1 permutation, knowledge denial p = 0.143);
  - F-27, F-30 (in-character rates by scale);
  - F-34 (kd_s23 21-layer profile, not significant);
  - F-62 (score 3 vs score 2–3; say that the co-author's control shows a loss);
  - F-82 (cap judge filter);
  - F-92 (sarcasm marker rates);
  - F-37 if not in 5.5.3.
- **Appendix C, reproducibility and audit:**
  - F-111, F-113, F-114, F-119, F-123, F-125, F-127, F-128 (correction entries; they govern which numbers are quoted);
  - the cross-branch agreement;
  - F-20 (passA/passC 98 rows).
- **Appendix D, engineering:**
  - F-14, F-21, F-22, F-24, F-25, F-33, F-35, F-40–F-45, F-47–F-50, F-54, F-56, F-104.
- **Chapter 2 or none (provenance and reading):**
  - F-60, F-70, F-78, F-88;
  - F-38 (dataset provenance: Chapter 4 already covers it).
- **Superseded; never quoted:**
  - F-31 → F-112; F-49 → F-50; F-58 → F-63 → F-121;
  - F-63's +0.202; F-85 → F-97; F-89 → F-94; F-100 → F-110;
  - F-74's −0.76; F-75's "strengthens"; F-57 (the figures entry).
- **Chapter 5 (tables above):**
  - F-03–F-09, F-13, F-15 (only as the `sycophancy_mt` motivation, else Chapter 4), F-17, F-18, F-29, F-39;
  - F-59, F-61, F-65–F-68, F-72, F-73, F-76, F-77, F-79–F-81, F-83, F-84, F-86, F-87, F-90, F-91, F-93–F-99;
  - F-101–F-103, F-105–F-110, F-112, F-115–F-118, F-120–F-122, F-124, F-126.
- **Q1–Q11:** all in 5.2.6, 5.3.4, 5.3.5, 5.6.2, 5.6.4 and 5.6.5.
- **Owner's N-list:** N1–N12 and P1–P4 are all placed.

## Figures and tables: what exists, and what to reuse

**The co-author's paper figures** (`git cat-file blob 16def2de:…/paper/generated/<f>`; credit him in the caption). These are the same figures as report.pdf Figs 5.1, 5.3, 5.6 and 5.7:
- `fig_transfer.pdf` → 5.2.5. Its right panel is constant by construction; say so.
- `fig_hpkr.pdf` → 5.2.4. Put the L20 88.3% FPR in the text.
- `fig_depth.pdf` → 5.3.4.
- `fig_operating.pdf` → 5.5.1.
- `fig_personas.pdf` is already in Chapter 4.
- Script: `paper/make_figures.py`, with inputs at the pin (`fig_personas` excepted).
- **Do not use:** FIG-435 `persona_ood_ext_L40/transfer_matrices.png` (stale); `loo_L*/loo.png` captioned as +0.125.

**The co-author's tables** (`paper/generated/tab_*.tex`, from `make_numbers.py`). Rebuild them in our table style, each number with `% src:` to the CSV:
- `tab_hpkr` → 5.2.4; `tab_zeroshot` → 5.2.3; `tab_transfer` → 5.2.5; `tab_pertest` → App B;
- `tab_operating` → 5.5.1; `tab_reads`, `tab_followup` → 5.3.5; `tab_syc` → 5.6.2;
- `tab_kthree` → App B;
- `tab_roster`, `tab_datasets` are already covered by Chapter 4 tables.

**Owner figures** (`figures/`; the scripts write into the repo, so regenerate only into a scratch folder):
- **FIG-005 `figures/thesis/fig5_capitulation.pdf`** → 5.6.2. Script `kd/scripts/make_thesis_figures.py`; data `run-cap/export/cap_fig_data.json`, local.
- FIG-006 `persona-coords/q10_sycophancy_axis.pdf` → 5.6.2. It uses paper16's assistant direction; say so in the caption.
- FIG-007 `q11_sleeper.pdf` → 5.6.4.
- FIG-008 `q1_contamination.pdf` → 5.2.6, provisional.
- FIG-012 `q6_layers.pdf` and FIG-009 `q2_layer_sweep.pdf` → 5.3.4 or App B.
- FIG-013/014 `q8_*.pdf` → 5.3.5.
- FIG-016 `significance_forest.pdf` → 5.4.
- FIG-018 `validate_pca` → 5.4.
- FIG-138 `crossgen/matched/L40_mean_paper` and FIG-084 `figure6_style/banks/paper16_shared/L40_mean_paper_k2` → 5.5.2.
  - Script `src/assistant_axis_ndif/28_crossgen_figures.py`.
  - The root `figure6_style` figures use the question-confounded bank: avoid.
- FIG-015 `q9`, FIG-010 `q3`, FIG-011 `q5` → App B.
- **Not used:** FIG-001, FIG-003, FIG-004 are hard-coded, and Chapter 4 says so. FIG-002 is not regenerable locally.
- `RESULTAXIS/` is empty. `results/`, `run-*/` and `docs/` hold no image files (the FIGURES.md sweep).
- The Liars' Bench upstream figures (FIG-139–188) are cited, not reproduced.

**New figures needed.** Scripts go in `thesis_tools/figures/`, output in `thesis/images/`.
1. **The convergence**, (a)/(b)/(c) in one panel. Data: collab-bank KD11 matrices; `km/` cache; HIS `gender_secret`.
2. **Scale ladder**, both banks (F-121). Data: the kdscale/scale caches.
3. **Leave-one-cluster-out per target** (+0.125). Data: `clustered_L40_mean_s42.json` `per_dataset`.
4. **The D7-vs-D9 widening control over 16 layers** (report Fig 5.5 has no committed plot). Data: HIS `depth.csv`.
5. Optional: library vs purpose-built contamination (the table may be enough).

**report.pdf figures that cannot be reused:**
- Fig 5.2 (D2/D3/D8/D9 bars): D3 has no data. Redraw without D3, or drop it.
- Fig 5.4 (INLP): no data.

## What is missing

- **D3, D10 and DANN:** no files on any branch, the lab PC or Drive. Only report.pdf prose and figures, and the owner's statement.
  - D3/D10 are quoted from report.pdf (decision 4).
  - Ask the co-author to commit the files: add this to `PROGRESS_THESIS.md`'s open questions.
- **The code for the 13 remote-pooled layers** (L36 included) is not committed. Chapter 4 already discloses it; label every L36 number.
- **Log-only or print-only values:**
  - F-01 (Chapter 4 recomputed it); F-76 and F-80's bootstraps;
  - F-107 (rebuilt: confirm the file); F-37 (logs); F-18 (stdout);
  - the provisional own100 detection (confirm the file).
- **Memory-only, to confirm or drop:**
  - dataset identity 99.3%;
  - "breaks 23 cells, rescues 14".
- **Conflicts to settle when writing:**
  - −0.98 vs −0.945 (the unified PC1);
  - 0.109 vs 0.381 (which base probe on HP-KR at L20).
- **The ±0.035 has no source:** never used.
- **Lab-only rows** (F-108, F-110, F-115, F-116, F-120, F-121, F-126, cap, collab bank): re-check on the lab PC in the final session. F-121's Drive backup is unchecked.
- **Chapter 3 is unwritten**, so 5.1's criteria cannot `\ref` requirement IDs yet.
- **`GLOSSARY.md` additions:** factor probes, published Assistant Axis, format control, token reads (max, top-10%, follow-up token, token before the reply), oracle threshold, generalisation gap, "the convergence".

## Owner decisions (2026-09-25, this session)

1. **Layout:** the template headings are the sections and our themes the subsections.
2. **New terms:** defined once in Chapter 5 and added to `GLOSSARY.md`.
3. **Selection (5.2.7): one design per criterion.**
   - D9, the extended contrast, for ranking and transfer. That is the headline: 0.8990 at L40, 0.9130 at L36 (post hoc).
   - D8, the knowledge-denial contrast, for detection at the benchmark threshold (HP-KR BA 0.910; LB-average BA 0.725).
   - Both are kd-bank contrasts, so the selected design is "the kd bank, read by one direction". The selection table shows which design wins each criterion.
4. **D3 and D10 (5.3.6): quote report.pdf's numbers.**
   - Source comment: `% src: full paper draft for this experiments/report.pdf :: p.N (§/Fig)`.
   - Add a sentence in the text saying the result files are not in the repository.
   - DANN has no numbers anywhere: describe the attempt (owner's statement, 2026-09-25) and give no number.
   - Page numbers to cite (PDF pages, to be confirmed at write time): §5.2.1 (D3) and §5.2.4 with Fig 5.4 (D10), printed pp. 30–34.
5. **report.pdf reuse: the skeleton only.**
   - Mirror its subsection order and table set: Tables 5.1–5.6 become ours.
   - Rebuild every sentence and number from the raw files, with `% src:`.
   - Add our results: the convergence, the pooled control, the scale ladder, the library banks, the cap bank, the replication at 70B.
   - Correct its known errors:
     - "that test has not been run" (false: DF-76);
     - "a pooled probe … is not compared here" (false: `loo_L*`);
     - "top-tenth is the better read" (the max wins at L40: DF-15);
     - it credits the ten kd roles to the co-author (they are the owner's).

## On approval (this session, before any LaTeX)

1. Copy this plan to `thesis_tools/planning/CH5_RESULTS_PLAN.md`.
2. Update `PROGRESS_THESIS.md`:
   - Ch 5 status: "planned";
   - its open questions: D3/D10 files from the co-author; the −0.98 vs −0.945 conflict.
3. Stop there. Session 1 starts when the owner says so.

## Writing sessions (one per session; each re-opens every source it writes, adds `% src:`, and runs `check_latex.py`)

1. **5.1 and 5.2.1–5.2.3** (criteria, the problem, D1/D2/D4, zero-shot contrasts, the headline and its made-of rows). Figure `fig_transfer`; tables zero-shot and transfer.
2. **5.2.4–5.2.7** (HP-KR and the factor probes, the convergence, transfer, which personas, selection). Figures `fig_hpkr` and the new convergence figure; tables `tab_hpkr`, contamination, selection.
3. **5.3** (adjustments: bank, one direction, the D7 control, layer, token read, approaches that did not work, claim adjustments). Figures `fig_depth` and the new D7 figure.
4. **5.4 and 5.5** (statistics, calibration, comparisons with LB and with Mahadik & Skapars, the scale ladder, pooled training, cross-branch). Figures `fig_operating`, FIG-138/084, and the new scale and leave-one-cluster-out figures.
5. **5.6**, the chapter intro and summary, `GLOSSARY.md` and `PROGRESS_THESIS.md` updates. Figures FIG-005, FIG-006, FIG-007.
5b. **Chapter 4 revision pass.**
   - Apply every item on `PROGRESS_THESIS.md`'s "Chapter 4 revision list": the sarcasm motivation and the FigLang citation, plus anything sessions 1–5 add.
   - Then run `check_latex.py`.
   - Doing it here means session 6 verifies the changed Chapter 4 paragraphs together with Chapter 5, in one run.
5c. **Appendices A and E** (see "Appendices" below).
   - Replace the template appendix titles at `main.tex:159,164`.
   - Run `check_latex.py`.
6. **Final verification.** Every number and every quoted example in Chapter 5, in the revised Chapter 4 paragraphs and in Appendices A and E, against raw data on both sides:
   - `asix`, and the lab PC over read-only BatchMode SSH, run from the main session;
   - Drive (`rclone lsf`/`cat`);
   - `origin/paper` @16def2de and `feat/persona-pc-ood`.
   Then an independent verifier subagent (told to check both sides), `python thesis_tools/check_latex.py`, and a `\todo` count, with the results logged in `CH5_VERIFY_TABLE.md`.

## Appendices (owner, 2026-09-25: examples in the main text AND the appendix)

The brief's A–D plus a new E. Each example is copied word for word from a file, with `% src:`. Long responses are cut, and each cut is marked `[\dots]`.

**WMDP rule:** for HP-C and HP-KR, withhold the question. Show only the response and the label or belief. These datasets are built from hazardous bio and cyber questions.

- **A. Persona prompts and questions**, generated from source, never hand-copied:
  - the kd bank: 36 roles × 5 system-prompt variants, and 30 questions;
  - the cap bank: 13 roles, 20 two-turn questions, and the challenge text;
  - the sarcasm pilot: 3 roles, v1–v5 variants, 20 questions.
  - How: a new `thesis_tools/appendix/make_appendix_personas.py`, modelled on `kd/scripts/make_persona_appendix.py`. It reads `kd/personas/kd_instructions/`, `kd/personas/kd_role_groups.json`, `kd/questions/kd_questions_shared.jsonl`, `cap/personas/` and `cap/questions/`, and writes `thesis/appendix/appendix_A.tex`, escaping LaTeX specials.
  - Record the definition-file commit each bank ran with: kd `3dfdb673`, cap `dc24e960` with the v3 questions, sarcasm `080a9794` (`THESIS_BRIEF.md`, "Bank versions").
- **B, C, D:** as in the brief (full tables; reproducibility and audit; engineering).
- **E. Examples:**
  1. **One labelled row per evaluation dataset**: the 9 labelled sets, `sycophancy_mt`, both HP-KR slices, and Alpaca.
     - Llama-3.3-70B responses only.
     - The cached `passB/<dataset>/rows.jsonl` holds no response text, only row ids and labels. Take the text through the loaders in `src/assistant_axis_ndif/eval_datasets.py`, matching on `row_id`. The loaders read `results/subsets/*.csv`, the Apollo rollout files and MASK; `sycophancy_mt` rows are on the lab PC or Drive.
  2. **One kept rollout per persona group**, with its judge score: paper16, kd and cap groups. Source: `run-paper16/responses` + `scores`; `audit/cache/labpc/run-kd/responses` + `scores`; `run-cap`.
  3. Do not reproduce Liars' Bench's own example figures (`figures/examples/*.pdf`): they are the benchmark authors' work, partly from other models.
- **Main-text examples:**
  - Ch 4: the kd trio and the HP-KR pair (Ch 4 revision list items 3–4);
  - Ch 5: capitulation (§5.6.2) and sarcasm (§5.6.3).
  - Format: a `quote` environment in `\small`, needing no new package, with a one-line caption if floated.

## Verification

- Per session:
  - every number has `% src: path :: field`, re-opened in that session;
  - `check_latex.py` passes;
  - no template text remains in `chapter_6.tex`.
- Final session: as in session 6. Mismatches are fixed before handover, and lab-only rows are marked as such in the source comments.

*Note: a scratch text extract of report.pdf was written to the system temp folder (`%TEMP%\rep.txt`) while planning. It is outside the repo and can be deleted.*


---

## Values in the tables above that did NOT reproduce from raw (writing pass, 2026-09-25)

The chapter uses the raw value; the rows above are left as written so the change is visible. Each was found by a writer or its independent verifier (`CH5_VERIFY_TABLE.md`) and is proposed to the audit as DF-110…DF-120 (`audit/DOC_FIXES.md`).

| Row | Plan value | Raw value used in the chapter | Source |
|---|---|---|---|
| 1.1 | D1 L20 in-distribution 0.979 | 0.9564 | `persona_ood_ext_L20/summary.csv` baseline `diagonal_mean` |
| 1.10 | 7 (arm, layer) inverted cells | 9 of 24 at the pin (uni_axis1 added at 2ad758fb) | `matrix_<arm>.csv` @16def2de |
| 1.6 | "0.682 / 0.634 / 0.93" as published averages | five-dataset means computed from the Llama rows; printed averages are 0.64 / 0.57 (7 datasets) / 0.93 (5) | LB Tables 6–8 pp. 54–56 |
| 3.6 | pooled probe wins CG, IT, ID and RepE | pooled wins CG only; D9 wins HP-KR, IT, ID, RepE (naive) | `loo_L40/loo_bootstrap.csv` (delta = pooled − ext) |
| 3.9 | upper bound runs at 8% FPR | 8% is the HP-KR cell; five-dataset mean 0.18 | LB Table 8 p.56 |
| 4.2 | D8 above 0.5 BA at 16/16 layers | 15/16 (L28 0.4801); 8v7 reaches 0.5026 at L56 | `persona_depth/depth.csv` |
| 4.3 / 9.8 | 23 of 90 cells above 1% FPR | 24 of 90 (26 of 108 with RepE) | `zero_shot_operating.csv` L20/40/44 |
| 4.5 | raw diagonal 0.9663; length 0.776 | 0.9741 / 0.9678 / 0.9623 by probe; 0.7756 | kd11 `mean_L40_s42.json`; `confounds_L44_last.json` |
| 4.12 | Gender-Secret "every direction at chance" | two adapted readings above 0.5 (5v5 L20, 8v7 L44), D8 below at L40, with intervals | `gender_secret/zero_shot.csv` |
| 5.1 | ext PCs (k=3) "0.866 vs 0.916" | reproduces; the D6/D7 transfer cells 0.663 / 0.690 are sign artifacts (restored 0.8080 / 0.8295) | `matrix_syc_axis1.csv`, `matrix_uni_axis1.csv` |
| 6.1 | range −0.141…−0.226 over L20–60 | four tabulated layers only; over every layer 20–60 it is −0.248…−0.115 | `run-paper16/axes_vectors_*.pt` |
| 6.5 | 5 datasets, `probes_5sets.json` | 4 labelled datasets + Alpaca (12 pairs); files `bank_run-paper16_axes_vectors_s3.pt.json` vs `bank_run_axes.pt.json`; and the L20 comparison DOES exist on the full corpus (lab `banks/own100/summary.json`) | run-coords; lab cache |
| 6.7 | −0.98 vs −0.945 conflict | both reproduce: cosine −0.9800, Pearson r −0.9446 | `run-paper16/vectors_s3`, `axes_vectors_s3.pt` |
| 7.1 | keeping L40 costs 0.0117 | 0.0116 | `depth.csv` |
| 7.5 | random gains +0.127 | +0.1271 on 5 datasets; on all 10 random falls (0.5718 → 0.5498) while axes rises (0.4846 → 0.6656) | passA cache L72/L79 |
| 8.3 | −0.03…−0.06 | 0.0276 (L44) … 0.0604 (L20) | `q8_trained_pertoken.json` |
| 8.4 | last-pooling +0.033 (19/21) | from the mixed-bank store; on `scale_70b_paper16fp32`: mean −0.0116 (9/21), last +0.0100 (14/21) | p17 labcache |
| 9.1 | 1 of 384 | reproduces only when BH is recomputed on the L40+L44 subset; with the file's own BH over all 21 layers, 0 of 4,032 clear | kd11 `significance.json` |
| 9.2 | median effect 0.044 vs interval ±0.125 | two populations: 0.1252 is the median half-width over all 65,388 tests; 0.0440 the median absolute difference of 276 persona-vs-raw tests (their half-width 0.1325) | 10 cached significance files |
| 9.4 | +0.138 vs +0.004 | reproduce on ID/CG/ai_liar/roleplaying/mask (not the paper's five: +0.1455 / +0.0582); on all 11 both vanish | kd11 L40 |
| 10.3 | +0.120 / +0.103 / +0.054 | F-110's definition (best = max(axes, pc:deception), 19 layers at 70B); the ladder's definition gives +0.1251 / +0.1086 / +0.0605 | scale caches |
| 12.4 | clears 14/21 (mean), 3/21 (last) | vs datapca@2: 15/21 and 6/21; vs raw: 13/21 and 3/21 | cap_s23 cache (48 files) |
| 12.5 | A − B ≈ +0.045 | +0.0510 / +0.0428 / +0.0475 at L20/40/44 | `cap_fig_data.json` |
| 5.6.3 | Gate 2 "0.953 / 0.947" | 0.9525 / 0.9472 at source precision; F-106's 0.78 (Twitter) is in no local file, only 0.6320 (Reddit, matched) | `marker_report.json`; `baselines.json` |
| 5.6.4 | 0.641 [0.550, 0.724] "confirm in gender_secret/summary.md" | not in any committed file; only report.pdf p.56 | report.pdf |
| 5.5.2 | "11-dataset crossgen retrain" | a 10-dataset grid; −0.0338 / +0.0281 reproduce | `run-crossgen/summary.json` |

**Resolved this session (were `\todo`s):** the collab-bank 21-layer sweep, the factor-probe gate values, 0.896, the majority-class baseline, cos(PC1, assistant) by scale, the cap 21-layer / seed margins, the format-control effects, the `sycophancy_mt` example: all re-read from lab files that were copied into `audit/cache/labpc/` byte-exact (md5 checked against the lab PC, `audit/cache/MANIFEST.csv`).

**Still `\todo`:** the trait-vector contrast (no grouping or output was ever saved) and Gate 3's sarcasm numbers (console-only; re-derivation from the lab chunk files is queued).

---

## Status 2026-09-25: sessions 1-6 done

Writing sessions 1-5, pass 5b (Chapter 4 examples), pass 5c (Appendices A and E) and session 6 (verification) were run in one workflow (`wf_040cbc40-2cf`) plus the main session's lab-PC check. See `PROGRESS_THESIS.md` "Chapter 5" and `CH5_VERIFY_TABLE.md`. Appendices B, C and D are stubs.
