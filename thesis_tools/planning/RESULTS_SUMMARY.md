# RESULTS_SUMMARY: every result in Chapter 5, and every limitation and open item

**Written 2026-09-25, end of the Chapter 5 session.** Read it with `PROGRESS_THESIS.md`, `THESIS_BRIEF.md` and `GLOSSARY.md`.

**How it was built.** From Chapter 5 (`thesis/chapters/chapter_6.tex`) and the thesis guide (`thesis_guide/`) only; no raw file was opened for it.
- The values are the ones the chapter prints. They come from the approved plan rows (`CH5_RESULTS_PLAN.md`), as corrected by the writing pass: the writers' "did not reproduce" notes, the verifiers' fixes, and the `\todo` values filled in the main session.
- It was compiled at the end of the session, under a usage limit, without re-reading `chapter_6.tex` line by line. **Before quoting a line elsewhere, confirm it against the chapter at its `\label`** (open item U-1). The chapter's `% src:` comment is the source; this file is a map.

**Columns.**
- **Scope:** model (Llama-3.3-70B-Instruct unless stated); pipeline (A = ours, B = the co-author's @16def2de); bank and judge filter (score 3, or scores 2–3); datasets; layer (20, 22 and 40 fixed in advance; every other layer post hoc); pooling; seeds; control compared against.
- **Polarity:** positive / negative / null / inconclusive, for the design or claim the row tests.
- **Guide ID:** the `thesis_guide/RESULTS_SOURCES.md` theme (T1–T12), then ledger F-numbers (`kd/docs/FINDINGS.md`), then audit claim IDs (M-, D-, DF-) or a PITFALLS pointer.
- **\label:** where the result sits in Chapter 5 (`GLOSSARY.md` §8.1).

**Default scope shorthand.**
- **B6:** Pipeline B, kd bank at score 3, six datasets (CG, HP-C, HP-KR, IT, ID, RepE), mean over detection tokens, 1,000-row bootstrap.
- **B-LB5:** the same, averaged over the five Liars' Bench datasets (RepE excluded).
- **A11:** Pipeline A, 11 datasets, mean pooling, `paper` probe, k = 5, seed 42, controls from the root store.

---

## 1. Results

### 5.1 Performance Evaluation: the problem measured (`sec:r-problem`)

| # | Result | Scope | Polarity | Guide ID | \label |
|---|---|---|---|---|---|
| 1 | The published deception contrast (D5, 8 v 7) inverts on HP-KR zero-shot: 0.148 / 0.593 / 0.589 AUROC at layers 20 / 40 / 44. | B6; L44 post hoc | negative | T3; F-13; D-02, D-05 | sec:r-problem |
| 2 | Pipeline A's zero-shot deception contrast shows the same pattern on HP-KR: 0.2504 / 0.2029 / 0.6436 / 0.6032 at L20 / L22 / L40 / L44. With the kd bank's vectors instead it reads 0.1482 / 0.5936 / 0.5913, within 0.002 of Pipeline B, so the gap between the pipelines comes from the bank. | A, paper16 at score 3 (and the kd bank), zero-shot; L44 post hoc | negative | T3; D-02 | sec:r-problem |
| 3 | Trained persona axes invert on HP-KR: 0.3004 as source and 0.4273 as target, against random 0.6590 / 0.6984. On `sycophancy_mt`: 0.3787 / 0.3551 against random 0.6327 / 0.5413. | A11, paper16, `axes`, L40; the grid has no leak guard (one "other" dataset is the text-matched slice) | negative | T3; F-13 | sec:r-problem |
| 4 | Training on HP-KR reverses the sign of a one-dimensional probe: its row inverts in 9 of 24 (arm, layer) cells at the pin. F-124 counted 7 of 21 at ea1ce52b; `uni_axis1` adds the other two. | B6, eight 1-d arms, L20/40/44 | negative | T3, T4; F-124 §1; DF-111 | sec:r-problem |
| 5 | Ranking is not detection: the base probe (D2) ranks HP-KR at 0.9358 AUROC yet flags none of its lies (BA 0.5, recall 0). Pipeline A's `paper` probe trained on MASK ranks HP-KR at 0.9085 with TPR 0. | B6 at L40; A11 at L40 | negative | T10 | sec:r-problem |
| 6 | At layer 20 the per-token base probe reads 0.109 on HP-KR; a D1 probe trained on pooled RepE activations reads 0.3808. | B, L20 | negative | T3; RESULTS_CHAPTER §1 | sec:r-problem |
| 7 | The benchmark's own probes on HP-KR (Llama-3.3-70B, layer 20): Mean Probe AUROC 0.12, BA 0.50; Follow-Up Probe 0.14 / 0.26, with recall 0.37 at FPR 0.84; Upper-Bound Probe 0.97 / 0.95 (privileged). The BA of 0.26 is the Llama row; the average over the three models scored on HP-KR is 0.42. | Liars' Bench Tables 6–8, pp. 54–56 | negative | T10; F-124 §5; DF-113 | sec:r-problem, tab:r-hpkr |

### 5.2.1–5.2.2 Supervised probes and the published persona method (`sec:r-supervised`, `sec:r-d4`)

| # | Result | Scope | Polarity | Guide ID | \label |
|---|---|---|---|---|---|
| 8 | The raw single-source probe (D1) fits its own dataset but transfers poorly: 0.9832 in distribution against 0.7320 off-diagonal at L40, and 0.9564 against 0.6478 at L20. | B6, 8192-d `lr_reg10`, trained on 100% of the source | negative | T1, T12; RESULTS_CHAPTER §7 | sec:r-supervised |
| 9 | The worst transfer cell, trained on CG and tested on HP-KR, is 0.0598. | B6, L40 | negative | T1 | sec:r-supervised |
| 10 | Pipeline A's raw probe is 0.6623 off-diagonal, level with the 5-dimensional `datapca` control's 0.6651 at L40. At L20 raw is ahead, 0.6289 against 0.6102. | A11, L20/L40 | null | T1; F-62, F-73 | sec:r-supervised |
| 11 | The base probe (D2) averages 0.7784 AUROC and 0.6255 BA on the Liars' Bench datasets. | B-LB5, L40 | negative | T10 | sec:r-supervised |
| 12 | A probe on the extended bank's first three components (D4) transfers at 0.866, below the single extended contrast's 0.916. The published components (k = 3) read 0.679 / 0.639 / 0.617. | B6, L40 (and L20/40/44) | negative | T1, T4; RESULTS_CHAPTER Table 3 | sec:r-d4 |
| 13 | One direction against the components over 16 layers: the contrast leads at 13 of 16 on all six datasets; the components lead at L64/68/72 by 0.0010 / 0.0205 / 0.0146. Excluding HP-KR it leads at 11 of 16, ranging from −0.0325 (L68) to +0.0633 (L56), with 6 sign changes. | B6, 16 layers (13 of them post hoc) | inconclusive | T8; F-117, F-128; DF-101 | sec:r-d4, sec:r-adj-direction |

### 5.2.3 Zero-shot persona contrasts and the headline (`sec:r-zeroshot`, `tab:r-zeroshot`)

| # | Result | Scope | Polarity | Guide ID | \label |
|---|---|---|---|---|---|
| 14 | **Headline.** The extended contrast (D9) reaches a Liars' Bench-average zero-shot AUROC of **0.8990 at L40** and **0.9130 at L36** (post hoc). The benchmark's Mean Probe reads 0.682, its Follow-Up Probe 0.634, and its privileged Upper Bound 0.93. | B-LB5, no training, no benchmark labels; L36 rests on remote-pooling code not in the repository | positive | T4; THESIS_BRIEF headline | sec:r-zeroshot |
| 15 | On balanced accuracy D9 reaches 0.6939, against 0.636 (Mean), 0.578 (Follow-Up) and 0.89 (Upper Bound). | B-LB5, L40, Alpaca threshold | positive | T4, T10 | sec:r-zeroshot |
| 16 | Zero-shot per dataset, the six-dataset means are: 8 v 7, 0.791 / 0.832 / 0.826; 4 v 6, 0.866 / 0.912 / 0.900; 12 v 13, 0.904 / 0.916 / 0.903. RepE (the base probe's training set) saturates at 1.000. | B6, L20/40/44 | positive | T4 | tab:r-zeroshot |
| 17 | What the gain is made of, part 1: the kd roles add +0.113 / +0.084 over the published contrast (L20 / L40), a share of 44.2% / 45.7% / 41.2% of the margin (L20 / 40 / 44). | B6 | positive | T4; F-97 | sec:r-zeroshot |
| 18 | Part 1, on our datasets: the kd roles add 0.0814 / 0.0671 / 0.0610 (shares 74.1% / 48.8% / 51.7%). Over 21 layers the difference is positive at 21/21, with a mean of 0.0551 and a median share of 62.9%. Under last-token pooling it is positive at 20/21, with a mean of 0.0590. On HP-KR at L20 it runs 0.1652 → 0.8057. | A, kd bank at scores 2–3, 11 datasets, zero-shot, mean (and last) pooling | positive | T4; F-98, F-99, F-103 | sec:r-zeroshot |
| 19 | Part 2, against the pooled probe: D9 scores 0.9158 against the naive pooled probe's 0.8496. **Without HP-KR the two are level, 0.9165 against 0.9149.** On HP-KR it is 0.9124 against 0.523. The pooled probe beats D9 on CG only. | B6, L40, pooled leave-one-out (naive and balanced) | null | T7; DF-112 | sec:r-zeroshot |
| 20 | Part 3, the base probe at its own best layer: 0.8498 at L32 (post hoc), against 0.7784 at L40. This narrows the headline margin. | B-LB5 | negative | T8; the ea1ce52b commit message | sec:r-zeroshot |
| 21 | Part 4, random directions (k = 3, averaged over five draws) transfer at 0.5269 / 0.5038 / 0.4927, so the persona arms are not a low-dimensional artifact. | B6, L20/40/44 | positive | T1 | sec:r-zeroshot |
| 22 | Part 5, the gap to the privileged upper bound is 0.0310 AUROC at L40 and 0.0432 at L20, and 0.196 on BA. The upper bound runs at 8% FPR on HP-KR, 18% on the five-dataset mean. | B-LB5 | inconclusive | T10; DF-26, DF-113 | sec:r-zeroshot |
| 23 | The sycophancy contrast (D6) inverts on HP-KR: 0.043 and 0.167 at L20 and L40, and 0.480 at L44, whose interval (0.4293–0.5309) includes 0.5. Without HP-KR, `syc_pc1` tops Pipeline B at L40–56 (0.9370 at L40), carried mostly by CG (+0.0862). Pipeline A agrees (0.0448 / 0.1647 / 0.4587). | B6; A kd bank at scores 2–3 | negative | T6; F-122 | sec:r-zeroshot |

### 5.2.4 HP-KR, the factor probes and the convergence (`sec:r-hpkr`)

| # | Result | Scope | Polarity | Guide ID | \label |
|---|---|---|---|---|---|
| 24 | The knowledge-denial contrast (D8) is the only detector of Pipeline B's six whose HP-KR BA leaves 0.5: AUROC 0.890 / 0.919 / 0.908, BA 0.533 / 0.910 / 0.757, recall 0.885 at L40. | B6, Alpaca threshold | positive | T4, T10 | tab:r-hpkr |
| 25 | The realised false-positive rate on HP-KR behind D8's BA is 88.35% at L20, 6.45% at L40 and 4.70% at L44. Over 16 layers it is 83–97.5% at L20–32 and 43.1% at L36, and below 10% at 10 of 16 layers. | B6 | inconclusive | T10; F-124 §4 | sec:r-hpkr |
| 26 | D8's HP-KR BA is above 0.5 at 15 of 16 layers (L28 reads 0.4801) and peaks at 0.9102 at L40. The other arms range from 0.4861 to 0.5026. | B6, 16 layers | positive | T8; DF-115 | sec:r-hpkr, fig:r-hpkr |
| 27 | Ranking HP-KR by layer: the base probe leads at L36, L40, L56–72 and L79, and trails D8 at L44 (0.828 vs 0.908), L48, L52 and L76. | B6, 16 layers | inconclusive | T8; F-128 (D2-005) | sec:r-hpkr, fig:r-hpkr |
| 28 | A label-free denial regex scores 0.9125 on HP-KR's 2,139 rows. The raw probe's in-distribution score is 0.9741 / 0.9678 / 0.9623 (`paper` / `lr_reg10` / `mms`), and response length alone scores 0.7756. | HP-KR CSV; A, L40 | negative | T3; F-01; DF-55, DF-105 | sec:r-hpkr |
| 29 | The factor probes: d_utterance scores 0.9960 / 0.9994 / 0.9995 on the knowledge-matched slice, but 0.3512 / 0.7842 / 0.7850 on the text-matched slice. **The pre-committed gate passes at L20 and fails at L40/L44.** d_knowledge scores 0.6847 / 0.7717 / 0.7706 (text-matched). cos(dK, dU) = 0.0630 / 0.4102 / 0.4304, so knowledge and wording are entangled deeper in the network. Split-half cosines are 0.9876–0.9955 against a floor of about 0.008. | A (owner), teacher-forced corpora, mean pooling, L20/40/44 | negative | T3; F-05, F-06, F-07 | sec:r-hpkr |
| 30 | The plain-assertion pole carries HP-KR: `kd_assertion` reads 0.9118 / 0.9344 / 0.9266 on full HP-KR but 0.4227 / 0.5860 / 0.5253 on the text-matched slice. The split knowledge-denial axis reads 0.5923 / 0.6692 / 0.6659. At L40 the full-HP-KR readings sit inside the random-direction null. | A, collab bank at scores 2–3, zero-shot | negative | T4, T5; D-03 | sec:r-hpkr |
| 31 | Pipeline B's merged 4 v 6 on the same scores-2–3 bank reads 0.8734 / 0.8826 / 0.8734, between our two 4 v 3 splits. Its re-extracted extended contrast (0.8053 / 0.8762 / 0.8486) agrees with Pipeline A's (0.8057 / 0.8765 / 0.8491) to within 0.0005. | B `persona_ood_mod`; A collab bank | positive | T4; D-03 | sec:r-hpkr |
| 32 | **The convergence (a).** On the text-matched slice the extended contrast reads 0.4630 / 0.6203 / 0.5383. At L40, 8 of 12 directions have row intervals above 0.5, 5 of 12 at L44 and none at L20. There is no random-direction null and no dataset bootstrap. | A, collab bank at scores 2–3 (not B's score-3 bank), zero-shot | negative | T5; F-118, F-128 | sec:r-hpkr, fig:r-convergence |
| 33 | **The convergence (b).** On the knowledge-matched slice (2,001 rows) raw separates in distribution (0.9995 / 0.9992 / 0.9993), but the axes transfer below chance: 0.3850 / 0.3335 / 0.5267, i.e. axes − raw = −0.1860 / −0.3758 / −0.1937. Length alone reads 0.6174. | A, paper16, 12 datasets, leak guard, mean, `paper`; lab-only store | negative | T5; F-116 | sec:r-hpkr, fig:r-convergence |
| 34 | **The convergence (c).** On Gender-Secret (392 rows, adapted read), 4 v 6 reads 0.549 / 0.436 / 0.498, 12 v 13 0.505 / 0.467 / 0.541, 8 v 7 0.514 / 0.545 / 0.623 and 5 v 5 0.593 / 0.523 / 0.520. Two readings exclude 0.5 upward (5 v 5 at L20, 8 v 7 at L44) and D8 excludes it downward at L40. In the base read D8 is inverted (0.386). | B, fine-tuned model, L20/40/44 | negative | T5; F-118 | sec:r-hpkr, fig:r-convergence |
| 35 | With the leak guard, the text-matched margins (axes − raw) are −0.2545 / +0.0309 / +0.0050. HP-KR → text-matched at L40 falls from 0.5674 to 0.4262 on 59 guarded rows (at L20 it rises, 0.6004 → 0.7190). | A, paper16, leak guard | negative | T5; F-115 | sec:r-hpkr |
| 36 | The layer-40 margin over `datapca` is +0.0655 [−0.0693, +0.1887], P = 0.1611. The best-layer interval is L5–L79 and a cluster permutation gives p = 0.38. The unguarded passA11 store gives 0.3743 at L20 and 0.7393 at L40. | A, 80-layer store, dataset bootstrap B = 10,000 | null | T5, T8; F-72, F-76, F-84 | sec:r-hpkr |

### 5.2.5 Transfer across datasets (`sec:r-transfer`)

| # | Result | Scope | Polarity | Guide ID | \label |
|---|---|---|---|---|---|
| 37 | Mean off-diagonal AUROC, all six (excluding HP-KR): 12 v 13 0.904 / 0.916 / 0.903 (0.909 / 0.917 / 0.905); 4 v 6 0.866 / 0.912 / 0.900; raw 0.648 / 0.732 / 0.716. | B6, L20/40/44; 1-d arms equal zero-shot | positive | T4; RESULTS_CHAPTER §2 | tab:r-transfer, fig:r-transfer |
| 38 | Sign artifacts: the 8 v 7 cell at L20 prints 0.651 where the zero-shot value is 0.7909. The D6 and D7 transfer cells at L40 (0.663, 0.690) are artifacts too (restored: 0.8080, 0.8295). | B6 | inconclusive | T4; F-124; DF-111 | sec:r-transfer, tab:r-designs |
| 39 | Over 16 layers D9 transfers at 0.9275 at L36 (post hoc) and 0.9158 at L40, against the single-source probe's 0.7682 / 0.7320. It is ahead of the baseline at all 16 layers (+0.115 to +0.273). | B6 | positive | T8; F-117 | sec:r-transfer |
| 40 | On our 11 datasets D9 reads 0.7996 at L40, against 0.7325 for the published contrast and 0.6623 for raw. On the five shared datasets it reads 0.8990 (score 3) against 0.8882 (scores 2–3). | A, collab bank, zero-shot; lab cache | positive | T4; F-98; DISAGREEMENTS D-11 | sec:r-transfer |
| 41 | `pc:kd_assertion` beats raw by +0.150 [+0.070, +0.237] at L40, and at 18 of 21 layers (1 clears BH, at L68). It beats `datapca` at 20 of 21 layers, but **0 of 126 tests clear**. | A11, k = 5, dataset bootstrap and BH | null | T1, T4; F-112, F-128 | sec:r-transfer |
| 42 | The two pipelines agree on bank-matched cells to a mean \|gap\| of 0.0015. | A vs B, bank-matched | positive | T4; DISAGREEMENTS §0 | sec:r-transfer (Appendix C) |

### 5.2.6 Which personas: library and purpose-built banks (`sec:r-personas`)

| # | Result | Scope | Polarity | Guide ID | \label |
|---|---|---|---|---|---|
| 43 | The deception contrast's cosine with an assistant direction at L40: own100 −0.660 (provisional), hf275 −0.610, paper16 at score 3 −0.370. Against the published Assistant Axis, paper16 reads −0.226 (−0.248 to −0.115 over L20–60). Purpose-built roles roughly halve the contamination. | A; own100 from the 2026-09-14 diagnostics (lab, not re-read) | positive | T11; M-043, M-030 | tab:r-personas |
| 44 | PC1 of a library bank is the Assistant Axis: cos 0.73–0.82 (own100) and 0.75–0.80 (hf275). | A, L20–60 | negative | T11 | sec:r-personas |
| 45 | Relabelling the library by disclosure stance moves mean \|cos\| only from 0.655 to 0.585 (about 11%). | A, hf275 | negative | T11 | sec:r-personas |
| 46 | Trait-vector contrast: not quoted (`\todo`; nothing was saved). | — | inconclusive | T11 | sec:r-personas |
| 47 | The sycophancy geometry changes sign between banks: +0.144 (own100) against −0.270 (paper16). own100's sycophancy axis reads 0.451 on `sycophancy_mt`, inside the null. | A, L40 | negative | T6, T11 | sec:r-personas |
| 48 | Detection, paper16 against own100: the provisional early run reads 0.815 vs 0.706 (`axes_resid` 0.770 vs 0.553). On the full corpus at L20 the gap shrinks to about 0.01, and both fall below raw. | A; early run = 4 labelled datasets + Alpaca, first bf16 extraction; full corpus = lab own100 store | inconclusive | T11; results doc §7.1 | sec:r-personas |
| 49 | hf275 with strict labels: `pc:deception` beats raw by +0.0387 at all 10 fine-sweep layers (L37–39, L41–47, post hoc), never bootstrapped. own100 reads −0.0401 and paper16_shared −0.0702. | A, passB_fine, 10 datasets; lab-only | inconclusive | T1, T11; F-126 | sec:r-personas |
| 50 | The unified PC1 of paper16 tracks the question set: cos −0.9800, r −0.9446, 46.7% of variance. Its cosine with the unified contrast is 0.389; on the shared-question rebuild it is 0.896 (36.8%). | A, L40 | negative | T1; F-18, F-113; DF-120 | sec:r-personas |
| 51 | The 36- vs 26-persona bank comparison is confounded (+0.0093, 8.1×). | A11 | inconclusive | F-61; DF-51, DF-71 | sec:r-personas |

### 5.2.7 Selection (`sec:r-selection`, `tab:r-designs`)

| # | Result | Scope | Polarity | Guide ID | \label |
|---|---|---|---|---|---|
| 52 | Selection, one per criterion (owner decision): **D9** for ranking and transfer (the headline), **D8** for detection at the threshold (the only design that flags HP-KR's lies, BA 0.910 at L40). Both come from the kd bank. | B6, L40 | positive | T4; THESIS_BRIEF "Chapter 5 decisions" | sec:r-selection, tab:r-designs |
| 53 | The D3 row of the designs table comes from report.pdf and is marked with a dagger. | report.pdf | inconclusive | — | tab:r-designs |

### 5.3 Final Design Adjustments (`sec:r-adjust`)

| # | Result | Scope | Polarity | Guide ID | \label |
|---|---|---|---|---|---|
| 54 | The repair is the roles: D8 reads 0.890 at L20, where the published grouping inverts on every build scored there (0.103 shared-question bank, 0.148, 0.2504, 0.1652). | A and B, L20 | positive | T3, T4; D-02, D-06 | sec:r-adj-bank |
| 55 | The widening control: the unified contrast (D7), which widens the bank with sycophantic roles, transfers no better than the published contrast, and ranks HP-KR below it at most layers. The extended contrast (D9), widened with the kd roles, stays high. Values: `sec:r-adj-roles` and `fig:r-widening`. | B6, 16 layers; A collab bank (unified 0.0600 / 0.3358 / 0.5220 on HP-KR) | negative | T4 | sec:r-adj-roles, fig:r-widening |
| 56 | Layer peaks: transfer 0.9275 and LB average 0.9130 at L36; the base probe peaks at L32 (0.8498); keeping L40 costs 0.0116 transfer AUROC. HP-KR BA spikes at 0.910 at L40, against 0.745 / 0.757 either side. | B, 16 layers (post hoc except 20/40) | inconclusive | T8 | sec:r-layer, fig:r-depth |
| 57 | L20 against L22: median gap 0.0115 over 17 methods, and formal equivalence at ±0.02 fails in all four tests. Moving to L40 gains +0.0251 for raw (10 methods) and +0.1135 for persona methods (13). | A, `run-coords` | inconclusive | T8; M-067, M-068, M-071; DF-63 | sec:r-layer |
| 58 | The persona axes peak at L44 (0.731, post hoc) and raw at L32 (0.717). Best against best, axes lead by +0.0142 [−0.1123, +0.1613], p 0.8612. `pc:deception` is below raw at every layer from 20. | A, paper16, four sweep runs | null | T1, T8; M-057 | sec:r-layer |
| 59 | Our L32–L44 plateau cannot be ordered: raw at L32 is 0.6818 against 0.6623 at L40. | A11 | null | T8; F-91; DF-53 | sec:r-layer |
| 60 | The late-layer rise: on five datasets random directions gain +0.1271 (L72 → L79). On all ten, random directions fall (0.5718 → 0.5498) while `axes` rises (0.4846 → 0.6656). Raw at L40 is 0.8195 on five datasets and 0.6760 on ten. | A, passA store (200 rows per dataset), lab-only | inconclusive | T8; F-120, F-128; DF-114 | sec:r-layer |
| 61 | Six token reads at L40: the single maximum reaches BA 0.750 and recall 0.548, the top-10% mean 0.737 / 0.531, the mean 0.694 / 0.413. The maximum drops RepE's BA from 0.998 to 0.638. | B-LB5, extended contrast | inconclusive | T9; DF-15 | sec:r-token, tab:r-reads |
| 62 | At the follow-up token the contrast's sign is not universal: IT 0.988 → 0.165, HP-KR (4 v 6) 0.890 → 0.237. The extended contrast reads 0.887 on the response and 0.552 at that token. | B, L20 | negative | T9 | sec:r-token |
| 63 | Our Q8 answer: zero-shot top-decile beats the mean by +0.1027 at L20 and +0.005 at L40. Trained, the pooled-mean probe beats the token-trained one by 0.0276 (L44) to 0.0604 (L20). | A, `run-coords` | inconclusive | T9; M-072, M-073, M-076; DF-61 | sec:r-token |
| 64 | 70B pooling (best persona feature − raw): mean pooling −0.0116 (ahead at 9 of 21 layers), last-token +0.0100 (14 of 21); against `datapca@3`, +0.0605 and +0.0712. | A, paper16 fp32, 5 datasets, k = 3 | inconclusive | T2, T9; F-101, F-102, F-110; DF-118 | sec:r-token |
| 65 | A prompt-only store detects CG (diagonal 1.0) and sandbagging (0.999–1.0) from the prompt alone; its off-diagonal is 0.5584 / 0.5687 / 0.5697. | A, passP, 10 datasets, L20/40/44; lab-only | negative | T9; F-126 | sec:r-token |
| 66 | **Did not work, D3** (on-policy AISI base probe): +0.100 [0.086, 0.113] over D2, but the extended contrast still leads by 0.021 [0.006, 0.035] at L40 and by 0.105 at L20, and they are level at L44. D3's HP-KR BA is 0.500. | report.pdf (no result files) | negative | — | sec:r-failed |
| 67 | **Did not work, D10** (INLP, L20): rank 1 reads 0.632 against raw 0.634; ranks 2–50 read 0.530–0.583. Its spread across training sets is 0.151–0.293, against 0.180 for raw and 0.032 for persona components. PCA beats INLP at ranks 2–50, not at rank 1. | report.pdf (no result files) | negative | — | sec:r-failed |
| 68 | **Did not work, DANN:** tried and failed; no numbers exist. | owner's statement 2026-09-25 | negative | — | sec:r-failed |
| 69 | **Did not work, the HP-KR growth gate:** denial rates were 5/20 = 0.250 (smoke) and 3/20 = 0.150 (control), below the 0.40 floor; the margin route reached 5/12 = 0.417 (marginal). The slice stayed at 258 rows. | A, logs (now cached, md5-checked) | negative | F-59 | sec:r-failed |
| 70 | The inverted axis used as a source fails under mean pooling only; under last-token pooling it beats random at 18 of 21 layers. | A | inconclusive | F-29; DF-40 | sec:r-failed |

### 5.4 Statistical Analysis (`sec:r-stats`)

| # | Result | Scope | Polarity | Guide ID | \label |
|---|---|---|---|---|---|
| 71 | Dataset-bootstrap census: 65,388 tests in 10 files, median half-width 0.1252, and 166 (0.25%) clear BH. Of 276 persona-vs-raw tests at L20/40/44 (mean, `paper`), the median \|diff\| is 0.0440, the median signed difference −0.0263, 95 are positive and 0 clear. | A, 9–11 datasets per block | null | T1; PITFALLS §0.5 | sec:r-stats |
| 72 | The F-04 family: 1 of 384 tests clears only when BH is recomputed on the L40+L44 subset, and that test is `random@2 − raw` **losing** (−0.2287, q 0.0192). With the file's own BH, 0 of 4,032 clear. On all 11 datasets the gains flip to −0.0215 / −0.0243. | A, kd11 significance file | null | T1; F-04, F-111; DF-01, DF-02 | sec:r-stats |
| 73 | The matched control erases the persona advantage: `pc:unified@2 − datapca@2` = −0.0028, P 0.9632. | A, L40, mean, `paper` | null | T1; F-17 | sec:r-stats |
| 74 | The k-dependence: +0.1382 at k = 2 against +0.0041 at k = 5 on ID, CG, ai_liar, roleplaying and mask (on the paper's own five, +0.1455 / +0.0582). On all 11 datasets both vanish (about +0.006, with wide intervals). | A, L40 | null | F-39, F-113; DF-117 | sec:r-stats |
| 75 | The random-direction null: the deception axis on HP-KR, 0.6436, lies inside the null (p95 0.9377), as do CG and HP-C. Pooled over the nine labelled sets plus Alpaca it clears (0.8810 vs 0.7636). No null was stored at L20. | A, L40 mean, 1,000 directions | inconclusive | PITFALLS §0.4; DF-72 | sec:r-stats |
| 76 | Response length alone transfers at 0.5555 against raw 0.6623 (L40, mean, `paper`, 110 pairs), and 0.5533 against 0.6646 (L44, last, `mms`, 90 pairs). The deceptive class is longer in 9 of 11 datasets by token gap; length-only AUROC is above 0.5 on 9 of 10 (roleplaying 0.4415). | A | negative | F-73, F-114 | sec:r-stats |
| 77 | Dataset identity dominates the activations: the dataset is decoded at 0.9934 accuracy and the label at 0.6462. The majority baselines are 0.614 (label) and 0.110 (dataset), so the lifts are +0.032 and +0.883. PCs 2–3 separate HP-C and IT perfectly, and the deception contrast has 16.58% of its norm in the top 10 PCs. | A, L40 mean, ≤400 rows per dataset, 3-fold CV | negative | PITFALLS §0.7 | sec:r-stats |
| 78 | Calibration: 24 of 90 cells exceed 1% realised FPR (26 of 108 with RepE). At L40, BA under the Alpaca vs the oracle threshold: base 0.6255 / 0.6519, D8 0.7251 / 0.6921, D9 0.6939 / 0.7183. D8's 0.725 [0.715, 0.735] is clearly above the base probe's 0.626 [0.619, 0.633]. | B, L20/40/44 (L36 in the oracle table) | negative | T10; F-124; DF-110 | sec:r-stats, tab:r-oracle |
| 79 | The generalisation gap: D1 scores 0.956–0.984 in distribution and loses 0.213–0.347 off-diagonal over 16 layers. The extended contrast's off-diagonal equals its in-distribution value within 0.002. | B6, 16 layers | positive | T1 | sec:r-stats |
| 80 | Q2/Q6 differences are significant over rows but not over datasets; the largest Q6 difference (axes, L40 vs L22) has q 0.0973. | A, `run-coords` significance file | null | M-085, M-088 | sec:r-stats, fig:r-significance |

### 5.5 Comparisons and Relationships (`sec:r-compare`)

| # | Result | Scope | Polarity | Guide ID | \label |
|---|---|---|---|---|---|
| 81 | Under the benchmark's protocol at L20, the base probe wins BA and recall (0.7209 / 0.4893) while the persona arms win AUROC (D9 0.887). At L40 and L44 the persona arms lead on all three (L40: D8 BA 0.725 and recall 0.487; D9 AUROC 0.899). | B-LB5, Alpaca threshold, mean pooling | inconclusive | T10 | sec:r-cmp-lb, tab:r-protocol, fig:r-operating |
| 82 | The replication at 70B, like for like: paper16 is −0.0131 against raw and +0.0541 against `datapca` (F-110's definition, 19 layers; −0.0171 / +0.0550 over 21). | A, paper16 fp32, 5 datasets, k = 3 | null | T1, T2; F-110 | sec:r-cmp-replication |
| 83 | The 10-dataset crossgen retrain: `pc:unified@2` is −0.0338 against raw at L40 (+0.0281 without `sycophancy_mt`); none of the differences clears (P 0.665–0.7938). | A, 10 datasets | null | T1; M-094 | sec:r-cmp-replication |
| 84 | Fine sweep: `axes − raw` is negative at 36 of 40 cells (mean, `paper`, 10 datasets), 11 of 40 without `sycophancy_mt`, and 21 of 40 under last-token pooling. | A, passB_fine; lab-only | negative | T1; F-126 | sec:r-cmp-replication |
| 85 | The scale ladder, layer-averaged (best persona feature − raw): paper16 +0.0382 / +0.0148 / −0.0116; kd scores 2–3 +0.0360 / +0.0194 / 70B a whisker above zero; kd all +0.0317 / +0.0169 / +0.0032. **Both banks decline, so the decline is the method's.** Against `datapca`: +0.1251 / +0.1086 / +0.0605. | A, 3B/8B/70B, 5 datasets without HP data, k = 3, one seed, post-hoc best feature, mean pooling; 800-row cap and 1,024 tokens at the rungs; lab-only | negative | T2; F-121, F-110, F-102; DF-87 | sec:r-scale, fig:r-scale |
| 86 | Per feature, `pc:deception` falls from +0.0324 at 3B to −0.0321 at 70B. | A, same scope | negative | T2; RESULTS_BY_AXIS §2 | sec:r-scale |
| 87 | Geometry by scale is not monotone: cos(PC1, assistant) is 0.435 (3B, L14), 0.732 (8B, L16) and 0.623 (70B shared-question rebuild, L40); PC1 explains 33.4% / 39.6% / 37.2%. | A, paper16 banks, logs (cached, md5-checked) | inconclusive | F-37 | sec:r-scale |
| 88 | **Pooled training without personas, leave-one-cluster-out:** +0.1254 at L40 (10 of 11 targets), +0.1289 at L20, +0.1309 at L32 (11 of 11); clean-9 +0.1504. **One seed, three layers, no dataset bootstrap.** | A, mean, `paper`; lab-only (md5-checked) | positive | T7; F-108 | sec:r-pooled, fig:r-loco |
| 89 | Leave-one-dataset-out gives +0.1782 (0.8545 vs 0.6763; five seeds +0.1728 to +0.1979; 21/21 layers), but it is leaky through the text-matched slice. It is never the headline. | A | inconclusive | T7; F-79, F-80, F-81, F-83 | sec:r-pooled |
| 90 | Diversity, not volume: leave-one-dataset-out minus the row-matched arm is +0.0045 [−0.0234, +0.0325], one-sided P = 0.3779. | A, L40 | null | T7; F-80 | sec:r-pooled |
| 91 | Why pooling works: a mean of single-source directions scores 0.8828, above all 20 row-matched draws (mean 0.8344, SD 0.0200, max 0.8736), and survives 51 of 51 cells at L ≥ 16. | A | positive | T7; F-86, F-95, F-90 | sec:r-pooled |
| 92 | Pipeline B's pooled leave-one-out against the axis, excluding HP-KR: +0.1408 / −0.0197 (L20, naive / balanced), +0.0016 / +0.0640 (L40). | B, `loo_L20`, `loo_L40` | inconclusive | T7 | sec:r-pooled |
| 93 | The knowledge-denial direction is only partly aligned with the published one (cos 0.47, Ch 4). The unified contrast sits close to both its parents (0.84 / 0.85) and shares the published contrast's failure. | kd vectors, L40 | negative | T4 | sec:r-geometry |

### 5.6 Discussions (`sec:r-discuss`)

| # | Result | Scope | Polarity | Guide ID | \label |
|---|---|---|---|---|---|
| 94 | Q10: paper16's sycophancy axis inverts on real capitulation, 0.313 [0.283, 0.346] (0.272–0.363 over 21 layers). paper16's own assistant direction (not Lu's axis) reads 0.710. | A, `sycophancy_mt`, L40 mean, 2,000-resample row bootstrap | negative | T6; M-046, M-047, M-048 | sec:r-cap |
| 95 | The capitulation contrast reads 0.759 / 0.745 / 0.728 zero-shot, against 0.374 (deception PC1) and 0.313 (sycophancy axis). The 0.745 [0.717, 0.772] interval is a row bootstrap only. | A, cap bank at scores 2–3 | positive | T6; F-65 | sec:r-cap, fig:r-cap |
| 96 | Trained `pc:capitulation@2` beats raw by +0.112 [+0.050, +0.185] and `datapca@2` by +0.154 [+0.091, +0.211] (beating raw from 9 of 10 sources). At L20 it clears raw only (+0.084); at L44 it clears both. | A, `sycophancy_mt` column, 10 sources, dataset bootstrap | positive | T6; F-66 | sec:r-cap |
| 97 | Over 21 layers (mean pooling) it beats `datapca@2` at 21/21 and clears at 15/21 (13/21 against raw). Under last-token pooling it is positive at 16/21 and clears at 6/21 (3 against raw). At L40 it is seed-stable over seeds 1/7/13/42 (+0.1110 to +0.1198 over raw). | A; cap matrices (lab, md5-checked) | positive | T6; F-77, F-87; DF-116 | sec:r-cap |
| 98 | The gain is local: over the full 11 × 11 grid it is +0.098 [−0.036, +0.208]; without `sycophancy_mt`, +0.073 [−0.077, +0.204]. | A, L40 | null | T6; F-67 | sec:r-cap |
| 99 | The holding roles matter: zero-shot A − B is +0.0510 / +0.0428 / +0.0475. Removing them costs +0.120 [+0.034, +0.213], and the reduced basis ties raw (−0.008). | A | positive | T6; F-68 | sec:r-cap |
| 100 | The format control, under mean pooling: replacing the holding roles costs +0.0356 (clears 3/21); removing them costs +0.0730 (15/21), so presence matters more than content. Under last-token pooling the two are level (+0.0346 / +0.0373; 11 and 10 of 21). | A, `cap_ctrl` + `cap_s23` (lab, md5-checked) | negative | T6; F-89, F-94 | sec:r-cap |
| 101 | `pc:sycophancy` trained in our pipeline loses to raw at 6 of 6 (layer, pooling) cells (−0.0075 to −0.1182), and 0 of 264 tests clear. | A, 11 datasets; lab-only | negative | T6; F-126 | sec:r-cap |
| 102 | Sarcasm Gate 1 passed: the ironic marker rate fell from 0.85 to 0.00 by v4/v5. | sarcasm pilot, 100 rollouts | positive | T11; F-92 | sec:r-sarcasm |
| 103 | Sarcasm Gate 2 failed: bag-of-words separates unmarked ironic from sycophantic replies at 0.921 (v4+v5; 0.919 with markers kept), and at 0.9472 over all variants. | sarcasm pilot | negative | T11; F-93, F-105; DF-66 | sec:r-sarcasm |
| 104 | Sarcasm Gate 3: the natural-sarcasm floors are bag-of-words 0.6320 and length 0.5000. The gate's own numbers are not quoted (console-only; `\todo`). | FigLang 2020 Reddit, 600 rows | inconclusive | T11; F-106, F-107 | sec:r-sarcasm |
| 105 | Q11 sleepers: base-model probes flag LoRA sleeper answers across triggers (raw, cross-trigger 0.7679 at L20 and 0.9639 at L44), but a lie is a wrong answer there. Gender-Secret is at chance (0.5413 / 0.4720). Zero-shot persona scores (0.87–0.91) sit inside the random null. | A, off-policy organisms, L20/L44 (post hoc) | inconclusive | T11; M-050…M-053 | sec:r-organisms, fig:r-sleeper |
| 106 | Q3: sample efficiency depends on the probe (with `lr_reg10` raw degrades while axes improve; with `mms` raw improves too) and is not significant. | A | null | T11; M-058 | sec:r-remaining |
| 107 | Q4: erasing the assistant direction costs 0.80–2.64% (3 of 6 settings below 1%), and the signal survives. | A, paper16 direction | positive | T11; M-056…M-063; DF-62, DF-65 | sec:r-remaining |
| 108 | Q5: label semantics does not predict transfer (mask ↔ `sycophancy_mt` at chance for raw, below it for axes). | A | null | T11; M-055, M-064, M-065 | sec:r-remaining |
| 109 | Q7: the 70B and 405B judges agree on 91.2% (1,231 / 1,350); the score-3 rate differs by 0.7407 pp and the kept sets by 8.15%. | A, paper16 rollouts | null | T11; M-014, M-015; DF-64 | sec:r-remaining |
| 110 | Q9: more personas buy reliability, not accuracy (a plateau at n = 4). | A, L40 | inconclusive | T11 (no M- row) | sec:r-remaining |

---

## 2. Limitations, and unresolved or unfinished items

### 2a. Limitations stated in Chapters 4–5 or in the guide

| # | Limitation | Where | Affects |
|---|---|---|---|
| L-1 | One model (Llama-3.3-70B-Instruct), English only; the 3B/8B rungs cover only 5 datasets without harm-pressure data. | Ch 5 `sec:r-scale`; Ch 4 `sec:m-model` | every result; the scale ladder |
| L-2 | **About eleven datasets:** a dataset-level bootstrap can rarely clear (median half-width 0.1252 against a median effect of 0.044). | Ch 5 `sec:r-stats`; PITFALLS §0.5 | every persona-vs-control margin |
| L-3 | Pipeline B has no dataset bootstrap and no multiple-testing correction; its intervals resample rows only. | Ch 5 `sec:r-stats`; `CH4_VERIFY_TABLE` writer items | every Pipeline B interval |
| L-4 | 1-d arms are zero-shot even when "trained": their 30 off-diagonal cells hold six distinct numbers, so intervals must be over test datasets. | Ch 4 `sec:m-protocols`; RESULTS_SOURCES (b)7 | `tab:r-transfer`, `tab:r-designs` |
| L-5 | **Post-hoc layers:** L36 (the headline's best layer), L44, L32, L48–56 and every sweep layer except 20/22/40 were chosen after results. | Ch 4 `sec:m-layers`; Ch 5 opening | headline 0.9130; `sec:r-layer` |
| L-6 | L36 and 12 other Pipeline B layers rest on remote-pooling code (`trace_layer_outputs(span_mask=…)`) that is not in the repository. | Ch 4 `sec:m-transform`; Ch 5 opening | 0.9130, 0.9275, `fig:r-depth`, `fig:r-hpkr` |
| L-7 | **HP-KR is where the kd repair was designed:** the roles were written after HP-KR was seen to fail, so HP-KR is not a blind test. | Ch 5 `sec:r-adj-bank` | headline, D8 detection |
| L-8 | **HP-KR's label is `knows AND denies`,** with one cell empty. A denial regex reaches 0.9125 and length 0.7756, and the full-HP-KR readings sit inside the random null. | Ch 4 `sec:m-integration`, `fig:m-hpkr`; Ch 5 `sec:r-hpkr` | every HP-KR number |
| L-9 | **The convergence:** where both classes deny, no persona direction reads the lie well. The kd contrasts partly read wording (the plain-assertion pole). | Ch 5 `sec:r-hpkr` | D8, D9 interpretation |
| L-10 | The 4 v 3 sincerity contrast isolates sincerity only on the four "does not know" questions; on the six "knows" questions the sincere-ignorance roles admit. | Ch 4 `sec:m-banks` (revised); `kd/docs/DESIGN.md:126-131` | `pc:knowledge_denial`, the convergence (a) |
| L-11 | **Calibration:** an Alpaca threshold does not transfer (24 of 90 cells above 1% FPR; D8 at 88.35% FPR at L20). The Upper Bound runs at 8% FPR on HP-KR. | Ch 5 `sec:r-stats`, `sec:r-hpkr`; PITFALLS addendum | every BA@1% number |
| L-12 | **What the gain is made of:** without HP-KR, D9 ties a pooled supervised probe (0.9165 vs 0.9149), and the base probe at its best layer narrows the margin (0.8498). | Ch 5 `sec:r-zeroshot` | the headline |
| L-13 | **The matched control erases the replicated method's gain at 70B** (−0.0028; 0 of 126 kd tests clear against `datapca`). | Ch 5 `sec:r-stats`, `sec:r-transfer` | the persona-coordinate claims |
| L-14 | The scale ladder uses a post-hoc best feature, one seed and no test, and it is a mean-pooling result only. The rungs differ in judge filter, 800-row cap and 1,024-token cut. | Ch 5 `sec:r-scale`; PITFALLS §A | `fig:r-scale` |
| L-15 | The pooled leave-one-cluster-out +0.1254 rests on one seed and three layers, with no dataset bootstrap. The LODO +0.178 is leaky. The "±0.035" has no source. | Ch 5 `sec:r-pooled`; PITFALLS §A | `fig:r-loco` |
| L-16 | Dataset identity dominates the activations, and response length transfers on its own. | Ch 5 `sec:r-stats` | every raw-probe transfer number |
| L-17 | The cap gain is local to `sycophancy_mt`, and weaker under last-token pooling. Its zero-shot 0.745 has a row bootstrap only. | Ch 5 `sec:r-cap` | cap results |
| L-18 | Library-bank results rest on an early 5-dataset run for own100 (first bf16 extraction), and own100's geometry was not re-read (lab only). hf275's strict labels were drafted by an AI assistant, and their review was not confirmed. | Ch 5 `sec:r-personas` | `tab:r-personas`, rows 43–49 |
| L-19 | Pipeline B excludes the insider-trading scratchpad, so its comparison with the published Mean Probe is approximate. | Ch 4 `sec:m-data`; Ch 5 `sec:r-cmp-lb` | LB comparisons |
| L-20 | The model organisms are off-policy: Pipeline A reads the fine-tuned text through the base model, and Pipeline B applies the adapter. In the sleepers a lie is a wrong answer. | Ch 4 `sec:m-cleaning`; Ch 5 `sec:r-organisms` | Q11, Gender-Secret |
| L-21 | An intent-defined construct (sarcasm) could not be elicited on-policy without a register confound, so the register diagnosis is untested on the one construct built to isolate intent. | Ch 5 `sec:r-sarcasm` | the register diagnosis |
| L-22 | **Lab-only evidence** (no git copy; read from md5-checked caches): F-110, F-113 (part), F-114, F-115, F-116, F-120, F-121, F-126, the passA/B/C/P stores, the cap, collab and scale matrices, and `run-sycophancy-mt`. | PITFALLS §E; RESULTS_SOURCES (d)4 | every lab-only row |
| L-23 | The co-author's numbers are "attested, not reproduced": no activations are committed on his side. | RESULTS_SOURCES (a); PITFALLS §E | every Pipeline B number |
| L-24 | Single seed, and no significance on most tables (the three-layer tables, F-121, F-117). | RESULTS_SOURCES (d)7 | most tables |

### 2b. Unresolved or unfinished items

| # | Item | Where | Status | What would resolve it |
|---|---|---|---|---|
| U-1 | This summary was not checked line by line against `chapter_6.tex` (usage limit). | this file | unverified | Read each `\label` section and confirm the numbers; fix this file, not the chapter |
| U-2 | The trait-vector contrast's cosine at L40. | Ch 5 `sec:r-personas` | `\todo` in text | Re-run a trait-vector contrast with a saved grouping, or drop the sentence |
| U-3 | Gate 3 (natural sarcasm) AUROCs. | Ch 5 `sec:r-sarcasm` | `\todo` in text; console-only | Re-derive from lab PC `run-coords/sarcasm/sarcasm_natural/chunk_*.pt` and save the result (a new computation, so it needs the owner's approval) |
| U-4 | Appendices B (full tables), C (reproducibility, pins, audit) and D (engineering notes). | `appendix_B/C/D.tex` | stub | Write them; planned content is listed in each stub and in `CH5_RESULTS_PLAN.md` "Appendices" |
| U-5 | D3 and D10 numbers come only from report.pdf; DANN has no numbers. | Ch 5 `sec:r-failed`, `tab:r-designs` | no result files | The co-author commits the D3/D10 result files |
| U-6 | Remote-pooling code for 13 Pipeline B layers (L36 included). | Ch 4 `sec:m-transform` | not in the repository | The co-author commits `trace_layer_outputs(span_mask=…)`; then delete the disclosure |
| U-7 | Pipeline B's four-decimal reproduction of the benchmark and its L40 pooling check (0.99989). | Ch 4 `sec:m-verification`, `sec:m-verify` | not reproducible (D2-024, D2-025, D-063) | The co-author commits `results_table.csv` and the check40 vectors |
| U-8 | Figure 2 (`fig:m-personas`) cannot be regenerated: the 36 vectors behind it are committed only as `.pt` files, and the plotting inputs are missing. | Ch 4; FIGURES §3; F-127 | not regenerable | Commit the plotting inputs, or keep the numbers-reproduce note |
| U-9 | The `aisi2026liedetection` bib entry takes its year from report.pdf. | `references.bib` | unverified | Open the Hugging Face page and check the year |
| U-10 | Six classic references (BH, Maris, Kriegeskorte, Ravfogel, Ganin, Sharma) have volume and pages from memory. | `references.bib` | unverified | A DOI check |
| U-11 | 7 template bib entries and template images `fig-5/17/18.jpg`. | `references.bib`, `thesis/images/` | leftover | Delete them once no chapter cites them |
| U-12 | DF-110 to DF-120 (corrections found while writing Ch 5) are proposals. | `audit/DOC_FIXES.md` | proposed | The owner decides; append ledger entries if wanted (the ledger is append-only) |
| U-13 | Chapter 4 items the writers found: RepE "left out of its averages" (LB average only); Pipeline B's two RepE probes; unnamed Pipeline B arms; no dataset bootstrap in Pipeline B; "any layer other than 20/22/40 is post hoc"; the paper16 anchor's questions; the cap `{model_name}` literal (20 rollouts); role descriptions verbatim for 22 of 25 (pp. 9–10); the cap judge filter (scores 2–3); the format-control banks and factor-probe corpora not listed; "assistant direction" undefined; two paper16 extractions. | `CH4_VERIFY_TABLE.md` (last paragraph) | pending | One Chapter 4 revision pass |
| U-14 | The guide's own open list: 202 inventory claims without a verdict; the 0.9597 regex unrecorded; print-only numbers (F-01, F-02, AN1–AN5); F-20's 98-row set unrecoverable; F-121's Drive backup unchecked. | RESULTS_SOURCES (d) | unverified in guide | Audit phases 6B/6C, or leave as disclosed |
| U-15 | Five paper16_shared L20 bank-matched cells are lab-only (the audit counts 55; the text's count comes from the cached cells). | Ch 5 `sec:r-transfer` | lab-only | Read them on the lab PC |
| U-16 | hf275_strict at the pre-registered layers (the lab store `banks/hf275_strict`, 21 layers × 6 seeds) was not read; the text reports only the post-hoc fine sweep. | Ch 5 `sec:r-personas` | lab-only | Read it on the lab PC before claiming anything beyond the fine sweep |
| U-17 | Stale figures not to use: FIG-435 `transfer_matrices.png`; FIG-001, FIG-003 and FIG-004 (hard-coded); FIG-002 (not regenerable). | FIGURES §5 | stale in guide | — (not used in Ch 5) |
| U-18 | Chapter 2 needs a paragraph on irony and deceptive intent; there are no PDFs for Grice, Sperber & Wilson or Clark & Gerrig. | `CH5_RESULTS_PLAN.md` "Items for other chapters" | not written | Add the PDFs or cite without pages |
| U-19 | Chapter 6 must collect every caveat above and the lab-only list, and put sarcasm v2 first in future work. | `sec:r-limits` points there | not written | Write Chapter 6 |
| U-20 | Nothing has been compiled with TeX yet; only the static checker ran. | whole thesis | unverified | Compile in Overleaf |
