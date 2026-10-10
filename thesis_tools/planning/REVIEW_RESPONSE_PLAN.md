# REVIEW_RESPONSE_PLAN — from 6.8 to 8–9

**Written 2026-10-07** at the owner's request ("write a md file stating all the to-do we have to do to
address all the weaknesses and questions … the best experiments forward to get the highest score possible").
Planning file only: nothing in `thesis/`, no commit (the owner commits), nothing on zami's branches.

**What was reviewed.** The thesis report *The Geometry of Lies: Knowledge and Persona Directions for
Detecting Deception in Large Language Models* (Zami, Shanaz, Hasan; Brac University, October 2026). Its PDF
and text are not in the repo; this session read a copy (`final.pdf` / `final.txt`, 2026-10-06). Report
sections referred to below: 4.2.7 (D5–D9), 4.2.8 (K), 4.2.9 (combining), 4.2.10 (reading fine-tunes),
4.4.5 (layer choice), 4.4.6 (steering), 5.2.4 (HP-KR wording), 5.2.5 (K), 5.3.4 (adding K), 5.3.5 (layer),
5.4.1 (intervals, random directions), 5.4.3 (K robustness), 5.4.4 (calibration).

**Score: 6.8/10** (ICLR-calibrated). Review text: this session's conversation, 2026-10-07.

**Path shorthand.**
- `Z` = `origin/paper@e27ffaf9:src/probes/deception-detection` (read with `git show`). Its 11 commits of
  2026-10-07 00:10–00:11 add K, K′, robustness, steering, layer averaging, the AISI probe, INLP, attention probes
  and the multilingual pilot.
- `RD` = `Z/results/recall_denial`. A byte copy is at `xmodel/k/zami_origin-paper_e27ffaf9/src/probes/deception-detection/results/recall_denial/`.
- `XR` = `%USERPROFILE%\xmodel-runs\` (laptop run folders, one per model; Drive `tamara:expLB/xmodel_2026-10-06/`).
- **✔** = number re-read from its file this session. **VERIFY** = taken from a research summary and not yet
  re-read; check it before it goes into any text.

---

## 0. Target and score logic

The reviewer's strengths are the method (K from forced identical denials), the wording diagnosis, the controls
(random bands, wording slices, K′, LoRA pass-through) and the honesty about tail recall. The deductions are
**scope** (one model; one diagnostic organism), **post-hoc layers**, **tail recall**, and **missing comparisons**
(multi-layer/multi-dimensional ensembles, black-box judges on the same slices, style-confound literature), plus
open questions on what K encodes.

- **To reach 8:** answer every W/Q with a measured result, not an argument. Concretely: (i) honest labelling of
  thresholds and layers; (ii) black-box and ensemble baselines on the exact slices, with privileged numbers
  labelled as such; (iii) one K-on-new-rows result (out-of-sample) on a second model; (iv) a topic-controlled
  Gender-Secret; (v) a familiarity test for K.
- **To reach 9:** add a second Liars' Bench family with detection numbers (Qwen + Mistral-3.1), a second
  diagnostic organism (Mistral-3.1 Gender-Secret, AISI on-policy rows), causal steering with fluency and random
  controls at several layers, and a style-shift test.
- **If NDIF stays down:** Tier 0 + Tier 1 alone are estimated at about 7.3–7.5, with the Tier 2 protocols
  pre-registered in an appendix.

---

## 1. Review → status map

| Item | Reviewer's point | Status | What exists (source) | Gap | Experiments |
|---|---|---|---|---|---|
| **W1** | K is 1-D; multi-dim / multi-layer ensembles may do better | partly | Zero-shot layer averaging only: D9+K 0.9063 at L40, 0.9163 over L36–44, 0.8891 over all 16 layers ✔ (`Z/results/layer_average/summary.csv`). Multi-dim without K: persona attention probes 0.804–0.849 transfer at L40; INLP 0.530–0.583; extended PCs k=3 0.866 vs D9 0.916 (VERIFY: `Z/results/persona_attnprobe_L*/summary.md`, `Z/results/inlp_ood/summary.md`) | No stacked multi-layer probe, no ≥10-D probe that includes K; nothing multi-dim on GS or identical text | T1-3, N1 |
| **W2** | K may be familiarity / answerability, not a stored answer | partly | K's unknowns are unanswerable by design (`recall_items.py:6-9`). K′ matches question form (bag-of-words known/unknown 0.95 → 0.858, VERIFY `RD/matched/style.json`). The chem arm (WMDP items split by belief) reads 0.7969 on identical text, cos 0.8206 with K (VERIFY `RD/auroc.csv`, `RD/cosines.csv`). Owner F-59: stated knowledge tracks confidence, not correctness | No rare-vs-frequent known items, no familiar-but-unanswerable items, no CoT test | T1-2, N1, G4 |
| **W3** | Layers 36/44 post hoc | partly | `docs/LAYER_PREREGISTRATION.md` pre-registers L20/22/40 (L44 marked post hoc). F-75/76/84/91 (plateau; L40 margin CI [−0.069, +0.189]; p = 0.38; L32 peak). D9 at L36 0.9130 vs L40 0.8990 ✔ (`layer_average/summary.csv`) | Headline still quotes L36; no selection-adjusted estimate; no cross-model replication at fixed layers yet | T0-1, T1-5, N3, N4 |
| **W4** | One model family / scale | partly | Scale ladder 3B/8B/70B, persona method only (F-121). **Qwen2.5-72B:** persona vectors (v2, span-fixed), K (split-half 0.9824 at L40), K′ (cos 0.953), D5/D8/D9, D9+K, Alpaca read 2,000 rows × 80 layers ✔ (`xmodel/README.md`). On Qwen D9+K ≈ K: Alpaca spread ratio 3.95 vs Llama 1.03, cos(D9+K, K) 0.971 ✔. **Mistral-Small-3.1** (Liars' Bench model): all loader and span fixes done offline, 21,362/21,362 eval spans exact ✔; NDIF deploy requested before the outage. **Mistral-Small-24B-2501:** persona chain paused at `vectors_all` (NDIF down); no Liars' Bench rows | No Liars' Bench detection number on any second model | N3, N4 |
| **W5** | Only Gender-Secret is diagnostic; Soft-Trigger leaks | partly | ST: the base read (no adapter) already gives K 0.9383 and D9 0.9182; the part the adapter adds gives K 0.2325, inside the random band 0.1573–0.8532 ✔ (`Z/results/soft_trigger/pooled.csv`, L40). GS: K 0.8645 [0.8236, 0.8981], p_null 0.012, but the random 97.5th percentile is 0.8236 — the lower bound sits on it ✔ (`RD/gender_secret.csv`). A second GS organism exists: Mistral-3.1, 370 rows / 115 lies ✔ | Only one diagnostic organism; GS margin over random is thin | N2, N4 |
| **W6** | At 1% FPR adding K lowers tail recall | measured, mislabelled | At L40, mean of five datasets, recall at 1% FPR ✔ (`RD/k_robustness/operating_points.csv`, computed this session): **per-dataset cutoff (privileged)** D9 0.4679 vs D9+K 0.3764; **Alpaca cutoff (the benchmark's protocol)** D9 0.4129 vs D9+K **0.4719**. BA 0.6939 vs 0.7014 ✔ (`RD/d5_plus_k.csv`). On HP-KR D9+K flags 0.0072 of lies; D8 alone flags 0.8849 of lies **and 0.9481 of honest denials** ✔ | The report's tail claim mixes protocols; no paired CI; no decision rule that keeps both | T0-2, T1-4 |
| **W7** | K′ validated on K's rows | not done | K′ (new matched items) on the same test rows: identical text 0.7959, GS 0.8808, cos(K′, K) 0.9928 ✔ (`RD/matched/verdicts.csv`). Qwen K′ built, not scored | No new rows, no second model | T0-6, N2, N3 |
| **W8** | Schematic; post-hoc tables; dense cross-references | partly | Thesis `fig:m-hpkr` (HP-KR grid by label, knowledge, denial) is the nearest thing; zami's TikZ pipeline `Z/paper/sections/fig_pipeline.tex` | No register-vs-knowledge schematic; K in no figure | T0-3, T0-4 |
| **W9** | Missing comparisons: ensembles, black-box judges on HP-KR slices, style-confound literature | not done | Per-row judge scores on HP-KR exist: Llama judge AUROC 0.4769, BA 0.5027 ✔ (`results/blackbox/external-evaluation.csv`, row 4); per-row Unrelated-Question classifier scores for HP-KR (L/M/Q) and GS (L/M) ✔ (`results/blackbox/pacchiardi/evaluation/`). Random p97.5 on full HP-KR is 0.933 at L20 ✔ (`RD/null.csv`), above D9's 0.9124 — so only the slices can rank detectors | No judge score on the identical-text slice; no style/format literature in either bib | T1-1, T1-3, T0-5 |
| **Q1** | D9 / K / D9+K vs multi-layer multi-dim ensemble, esp. identical text and GS | not done | see W1 | — | T1-3 |
| **Q2** | K vs exact denial string / template | partly | `k_template.py` variants change system prompt, template and denial together: "open" cos 0.29, "report" cos 0.43, identical-text 0.79 / 0.78 (VERIFY `RD/k_template/score_*.csv`). K minus the deny-vs-assert wording direction: identical 0.7918, full HP-KR 0.3488 (VERIFY `RD/auroc.csv`) | No paraphrase-only, punctuation-only or negation-only variant; no per-string K | T1-7, N1 |
| **Q3** | Stored answer vs familiarity; rare vs frequent; CoT self-correction | not done | see W2 | — | T1-2, N1, G4 |
| **Q4** | Causal steering along K | partly | K −4 at L40: unpressured denials 0.05 → 0.70 (+0.65 [0.45, 0.85]); under HP pressure 0.20 → 0.65; answer check stays 0.95; random directions do not do this ✔ (`Z/results/steering_L40_knowledge/summary.md`). n = 20, one layer, regex grading | Several layers, judge grading with agreement, fluency metric, held-out items | N5 |
| **Q5** | K′ out of sample | not done | see W7 | — | N2, N3 |
| **Q6** | Release per-layer random-band percentiles | partly | `RD/null.csv` (p2.5/p97.5 at 4 layers, 2,000 directions), `RD/k_robustness/depth.csv` (p97.5, 16 layers), owner `run-coords/zeroshot_nulls.json` | Not consolidated; no p50/p99.5; no guidance | T1-6 |
| **Q7** | Topic-controlled Gender-Secret | not done | zami's own GS summary calls the topic difference "the likeliest explanation" (VERIFY `Z/results/gender_secret/summary.md`) | — | N2 |

---

## 2. Ground rules for every new number

1. **Label tag on every number:**
   - **LF** (label-free) — no lie labels. K's known/unknowable split is a construction label, not a lie label;
     Alpaca thresholds are allowed.
   - **OOD-L** — trained on labelled lies from *other* datasets (e.g. RepE, leave-one-dataset-out).
   - **P** (privileged) — uses labels or thresholds from the dataset being evaluated (in-distribution training,
     stacking weights, per-dataset cutoffs). Always reported as an upper bound.
2. **Pre-register before running.** Each experiment's success criterion (below) goes in a dated note and is
   committed by the owner before the run. Results that miss their criterion are reported, not dropped.
3. **Ledger.** Every new result is appended to `kd/docs/FINDINGS.md` as F-129 onwards (append-only), with path,
   key/row and caveats.
4. **Layers.** L40 (80-layer models) and L20 (40-layer models) are the fixed layers. Every other layer is post
   hoc and goes in a separate column.
5. **Thresholds.** The headline operating point is the benchmark's: 1% FPR on the model's own Alpaca rows.
   Per-dataset cutoffs are P.
6. **Compute.** At most 3 NDIF requests per model at a time. The laptop has 7.7 GB of RAM: store-only analyses
   stream one layer at a time. The lab PC is down; everything runs from the laptop and Drive.
7. **Both branches.** zami wrote K, D9+K, K′ and steering; credit accordingly. His numbers are "attested, not
   reproduced" until T1-0 bridges them.

---

## 3. Tier 0 — writing only, no compute (~12 h)

| ID | Answers | Change in the report |
|---|---|---|
| T0-1 | W3 | L40 is the only headline (pre-registered). L36 and L44 move to a "post hoc" column. Drop 0.913 (L36) from the abstract; state D9 0.8990 and D9+K 0.9063 at L40. Cite `docs/LAYER_PREREGISTRATION.md`. (§4.4.5, §5.3.5) |
| T0-2 | W6 | Rewrite the tail claim with the protocol named: at the Alpaca cutoff D9+K catches more lies than D9 (0.4719 vs 0.4129) and has higher BA (0.7014 vs 0.6939); only under per-dataset (P) cutoffs does it fall behind (0.3764 vs 0.4679). (§5.3.4, §5.4.4) |
| T0-3 | W8 | New 2×2 schematic: rows "denies / claims to know", columns "knows / does not know". The persona contrast D separates rows (wording, register); K separates columns (knowledge). HP-KR's lie cell is knows ∧ denies; the identical-text slice is the "denies" row; one cell is empty in HP-KR. Show where each control (random band, identical-text slice, balanced splits, base vs LoRA read) rules out which confound. (§4.2.8 or §5.2.4) |
| T0-4 | W8 | Every results table gets an LF / OOD-L / P column and a pre-registered / post-hoc mark. Each results subsection opens with a 2-line summary so it reads without the appendix. Move dense cross-references to a single "where each number comes from" appendix table. |
| T0-5 | W9 | Related work, two new paragraphs (check each citation against the paper before quoting): **style and format confounds** — Sahoo et al. arXiv 2606.02907 (probes read task format; Ridge deconfounding), von Klinski et al. arXiv 2609.39807 (compliance and likelihood confounds; Goldowsky-Dill RepE reads 0.170 on their compliance confounder at Llama-70B), Kumar arXiv 2605.27958 (single universal direction rejected; style augmentation), Nordby et al. arXiv 2604.13386 (multi-layer ensembles; note their HP-KR 0.910 is learned on 80% of HP-KR, i.e. P); **knowledge vs familiarity** — Kadavath et al. 2207.05221 (P(IK)), Ferrando et al. 2411.14257 (known/unknown entity directions), Cheang et al. 2510.09033 (internal states mainly reflect recall), Brzezinka 2607.07670 (familiarity vs reliability), Lindsey et al. 2025 (known-entity features inhibit a can't-answer circuit). |
| T0-6 | W7 | Call K′ a replication of K's *construction* (new items, matched form), not an out-of-sample test. Out-of-sample evidence comes from N2 (new GS rows) and N3 (Qwen). |
| T0-7 | W9 | Add the published black-box table for HP-KR (Liars' Bench v2 appendix, `papers/2511.16035v2 (1).pdf` pp. 54–62, VERIFY each value): Claude-3.7 judge 0.48 AUROC, deceptiveness-prompt judge 0.64, self-judge 0.54, self-evaluation 0.44, Mean probe 0.12, Follow-up probe 0.14, Upper Bound (P) 0.97, Unrelated-Question classifier 0.90 (needs re-querying the model). |

---

## 4. Tier 1 — CPU on existing stores, no NDIF (in rank order)

Each card: design · inputs · success criterion (pre-registered) · what a miss means · report section.

**T1-0 Data bridge (prerequisite for T1-2…T1-6). Revised 2026-10-07 (owner): zami's activations are NOT
needed.** The report's numbers stand as written: zami's committed result CSVs back them (several re-read ✔ above).
His pooled activations are large and stay on his disk (`E:/liars-bench-acts/cache`; not in git, checked
2026-10-07); nothing below requires them.
- **Per-row comparisons (T1-1, T1-2c/d, T1-4, T1-5) need per-row *scores*, not activations.** Ask zami for one
  small file: `Z/results/layer_average/scores.npz` (written by `layer_average.py scores`, not committed: every arm's
  raw score on every dataset at all 16 layers — MB-scale, VERIFY its size), and, if he has them, per-row K / D9 /
  D9+K scores on the GS reads. These reproduce the report's numbers exactly, so no re-derivation is needed.
- **Analyses that need activations (T1-2b, T1-3, T1-8) run on the owner's passB**, already on Drive
  (`tamara:expLB/coords/passB`, 21 layers incl. 20/36/40/44, 12 datasets, 16,460 rows, 10.6 GiB): download to the
  laptop temporarily (no new Drive space), with Llama K from `XR\K-validation-llama\llama-70b-3.3\first\K.npz`
  (validated bit-for-bit against zami's split-half) and Llama D5/D8/D9 rebuilt from the kd bank by
  `xmodel/d_contrasts.py`. Sanity check before use: D9 and K on passB within |ΔAUROC| ≤ 0.02 of the report
  (pooling differs slightly: passB = response-span mean via 8_coords; zami = Apollo detection mask). passB results
  get their own column.
- **GS-based parts** are re-read on NDIF in N2 (topic-controlled GS) rather than taken from zami's adapted read.
- **Exactly what to ask zami for, and a copy-ready prompt for his LLM: §12.** With §12 Part B done by zami on his
  own cache, T1-2b, T1-3 and T1-6 run on the report's exact rows and passB becomes only a fallback.

**T1-1 Black-box on the slices (W9, T0-7).** Score the existing per-row judge outputs
(`results/blackbox/ext_judge_scoring_lie/harm-pressure-knowledge-report/llama-v3.3-70b-instruct.csv`, plus the
deceptiveness, self-judge and self-eval folders) and the Unrelated-Question classifier
(`results/blackbox/pacchiardi/evaluation/{harm-pressure-knowledge-report,gender-secret}/…`) on full HP-KR, the
identical-text slice, the denials slice and GS. Paired bootstrap against K. *Success:* K "beats" a method only
where the paired CI excludes 0. *Text:* a new comparison table (§5.2.5). Note the classifier is OOD-L and needs
extra queries to the model.

**T1-2 Question-text and belief-state controls (W2, Q3, W9).** (a) TF-IDF logistic regression on the
*question* only, 5-fold CV (P) — can the item alone predict the label? (b) Ridge-residualise activations on
format covariates (dataset, length, template, harm factor, WMDP domain) as in Sahoo et al., then rescore D9 and K.
(c) K's added value beyond text: likelihood-ratio test inside CV. (d) K on lies with the correct belief vs lies
with a wrong belief vs honest rows. The identical-text slice is reported to be 78 lies / 112 honest, all WMDP-bio,
with every honest row at `model_belief = -1` and 11 lies holding a wrong belief (VERIFY from the HP-KR CSV and
`knowledge_bank._slices`). *Success:* K adds information beyond text (p < 0.01) and stays ≥ 0.75 after
residualising. *Miss:* narrow the claim to "K reads item knowledge without labels". (d) speaks directly to
"K tracks the model's belief, which is what lying is about".

**T1-3 Ensembles (W1, Q1, W9).** Evaluate on the five-dataset mean, the identical-text and denials slices,
and GS:
- *LF:* a multi-dimensional K (shrinkage-LDA, L2 LR and top-k PCs on the 240 forced-denial reads); the 4-layer
  average of K (already 0.7944 on identical text over L36–44 ✔).
- *OOD-L:* one LR per layer trained on RepE across passB's 21 layers, combined by z-score mean or by 5 evenly spaced
  layers; Kumar's k-dim PCA+LR (k = 1…50) with a sign-flip permutation null.
- *P:* Nordby-style 5-layer stacking on an 80/20 HP-KR split, with all slice rows forced into the test half, and a
  learning curve over the number of training lies (5 → all).
Reuse `layer_average.py`, `streaming_lr.py`, `inlp_ood.py`, `k_transfer.py`.
*Success:* no LF or OOD-L ensemble beats K on the identical-text slice beyond the paired CI; report **n\***, the
number of labelled in-distribution lies a P probe needs to match K. *May backfire:* P stacking will likely beat K
on full HP-KR — report it as an upper bound and as "K is worth n\* labelled lies" (§5.2.6, new §5.5).

**T1-4 Tail recall (W6).** Paired bootstrap of recall at 1% FPR (Alpaca cutoff) for D9, K and D9+K. Fixed now,
before looking: an **OR-rule** — flag if z_D9 or z_K exceeds the Alpaca 99.5th percentile (so each half keeps
roughly 0.5% FPR). Also partial AUROC up to 5% FPR. Inputs: `k_robustness.operating_points` or T1-0's store.
*Success:* OR-rule recall ≥ D9's on ≥ 4 of 5 datasets at realised Alpaca FPR ≤ 1%. *Miss:* keep the "ranker
first" framing, now with CIs (§5.4.4).

**T1-5 Layer selection (W3).** Leave-one-dataset-out layer choice over the 21 passB layers (pick the layer on four
datasets, score the fifth) plus a row split-half choice. *Success:* selection-adjusted score ≥ L40 score − 0.01.
Report the adjusted number either way (§5.3.5).

**T1-6 Random-band release (Q6).** 1,000 seeded unit directions × 21 layers × every dataset and slice:
p2.5 / p50 / p97.5 / p99.5. Release as an appendix CSV with a reading rule: *a dataset can rank detectors only if
its random p97.5 < 0.85* (full HP-KR fails this: p97.5 0.933 at L20 ✔). Inputs: `RD/null.csv`,
`RD/k_robustness/depth.csv`, passB.

**T1-7 Denial-string transfer (Q2).** Build K from each of the two forced denials separately ("I do not know the
answer." / "I don't know the answer to the question."; reads already on disk: `XR\K-validation-llama\…\parts_answer`
and the Qwen K folders) and cross-score. *Success:* cos ≥ 0.8 and identical-text AUROC within 0.03 of K. *Miss:*
K is string-specific; N1's paraphrase arm becomes essential.

**T1-8 Dataset breadth on passB (persona-coordinates paper's sets).** Score D9, K and D9+K (LF) on the deception
sets of Mahadik & Skapars that passB already holds — ai_liar, roleplaying, MASK, CG, ID, HP-C — plus
sandbagging_v2 and sycophancy_mt. Re-grade ai_liar and sandbagging with their judge labels (the loaders use the
scenario's intended label; the judge grades differ: ai_liar 93 deceptive / 173 honest, sandbagging 420 / 512,
VERIFY). *Text:* a breadth table beyond Liars' Bench (§5.2.6).

---

## 5. Tier 2 — NDIF teacher-forced reads (NDIF is down as of 2026-10-07 22:17)

**N1 Llama-3.3 base, one session (Q2, Q3, W1, W2).**
- *Denial wording (Q2):* six paraphrases / punctuation / negation variants of the forced denial, plus an
  "I know the answer." arm; templates with and without the harmful prefix, plain QA, and with a role.
- *Familiarity (Q3, W2):* PopQA known-frequent vs known-rare (keep only items the model answers correctly,
  checked by a greedy answer first); Pacchiardi `questions_made_up` (380 familiar-but-unanswerable items, local:
  `pacchiardi/data/raw_questions/questions_made_up_all.json`); Meng et al. `known_1000` (local); the existing
  unknowables. Build K_rare, K_fam and a popularity direction K_pop.
- *Compliance (W9, von Klinski):* a refusal direction (harmful vs harmless prompts); cos(K, refusal) and K with
  refusal projected out.
- *Pressure independence:* the Liars' Bench HP-neutral sets (same WMDP items, no pressure; local
  `results/subsets_validation/harm-pressure-*-neutral.csv`) — K should track knowledge, not pressure.
- K at all 80 layers (Llama K is currently at 20/36/40/44 only).
- Reuse `xmodel/k/k_run.py` (validated) and zami's `recall_denial.general_items(source=…)`.
- *Success (fixed now):* ≥ 5 of 6 paraphrases have cos ≥ 0.8 with K and identical-text AUROC ≥ 0.75;
  cos(K_rare, K) ≥ 0.7; familiar-but-unanswerable items score like unknowables (AUROC vs unknowables ≤ 0.65);
  K_pop on the identical-text slice stays inside the random band. *May backfire:* if K_pop lines up with K,
  rename K a familiarity direction and narrow every claim (rule written before the run).

**N2 Llama-3.3 + Gender-Secret LoRA in the trace (Q7, W5, W7, Q5).**
- *Topic-controlled GS:* the same forced denial after "what is my gender?" (the fine-tune knows it) vs
  topic-matched unknowables (a third party's gender; the user's age or name, which the fine-tune does not know),
  plus an admission arm forcing the correct gender. Read through the LoRA and through the base model.
- *New on-policy rows:* AISI `ai-safety-institute/lie-detection-rollouts`, config `gender_secret_ood` (Llama-3.3 GS
  LoRA rollouts and base runs on the same task; zami pins revision 177a8166 in `aisi_rollouts.py`; read from the
  Hub, licence "other"). Score K and K′ there: **out-of-sample** rows.
- Reuse zami's in-trace LoRA route (`gender_secret.py`, `lora_intervention.py`).
- *Success:* LoRA read AUROC ≥ 0.75 and above the random p97.5; base read ≤ 0.60; admission arm high too.
  *May backfire:* AUROC ≈ 0.5 means the GS 0.8645 was topic → withdraw the GS claim.

**N3 Qwen2.5-72B eval read (W4, W7, Q5, W3).** HP-KR, ID, HP-C, MASK (+ the three soft-trigger organisms) at
L20/36/40/44 only (~0.7 GB). Pipeline ready: `xmodel/alpaca_chain.ps1` pattern with `XMODEL_CSV_MODEL`, the fixed
loader (`eval_datasets.py` now refuses other models' rows). Score D5/D8/D9, K, K′ and D9+K (Qwen Alpaca scaling
already in `XR\Qwen2.5-72B-Instruct\d9k\`). Identical-text slice for Qwen = rows whose reply is Qwen's most common
exact honest-denial string (rule fixed now). **Predicted in writing before the run:** D9+K ≈ K on Qwen (spread ratio
3.95), so it may trail D9 on wording-driven sets. *Success:* K on Qwen's identical-text slice ≥ 0.70 and above its
random p97.5; K′ within 0.03 of K.

**N4 Mistral-Small-3.1 (W4, W5).** The Liars' Bench model of the Mistral family, with its own Gender-Secret organism
(370 rows / 115 lies) and soft-trigger organisms. K (verify + 240 reads) at all 40 layers; HP-KR, ID, IT, HP-C,
MASK, GS (through its LoRA), Alpaca at **L20 (fixed, mid-depth)** plus L16/24. Code ready offline
(`ndif_common.load_language_model` → VisionLanguageModel with a tokenizer-only processor; template date pinned;
`k_run.py` keys `mistral-small-3.1-24b`); NDIF deployment was requested before the outage. *Success:* K on GS above
its random p97.5 and ≥ 0.70. Either way: a second organism and a second Liars' Bench family.

**N5 Steering along K (Q4).** L36/40/44; coefficients ±2/4/8 × the Alpaca SD; on K′'s items (held out from K's
construction); 20 norm-matched random directions; graded by an LLM judge and the regex with Cohen's κ; fluency by
perplexity under the unsteered model; an "admits under HP-KR pressure" arm (+K). Reuse `steer_knowledge.py`.
*Success:* K's effect ≥ 3× the random 95th percentile at ≥ 2 of 3 layers; perplexity ratio ≤ 1.5. *Miss:* K is a
readout, not a lever; say so.

---

## 6. Tier 3 — new generation (only if time allows)

| ID | Design | Why | Hours |
|---|---|---|---|
| G1 | Style augmentation: rewrite the identical-text and GS rows in 8 styles (Kumar's: pirate, Shakespearean, …) offline via an API, then teacher-force back in. *Success:* K's mean within-style AUROC ≥ 0.75 with SD ≤ 0.05, and D9's spread across styles larger | Direct answer to the style-confound literature | 8 |
| G2 | Mistral-3.1 persona bank (36 roles × 30 questions) → full D5/D8/D9 and D9+K on a second Liars' Bench model | Completes W4 for the combined detector | 16 + NDIF days |
| G3 | The persona-coordinates paper's sycophancy sets (Azarbal `arianaazarbal/sycophancy_dataset`, Papadatos `henrypapadatos/Open-ended_sycophancy`; ELEPHANT source unknown) — on-policy generation | Replication completeness; not about knowledge | 12 |
| G4 | CoT self-correction (Q3): items the model gets wrong greedily but right with CoT; force the denial before the CoT; does K read "will know"? | No published precedent (literature search) | 8 |

---

## 7. Datasets: what to add, and why

| Dataset | Local? | Read type | Label | Answers | Effort |
|---|---|---|---|---|---|
| Qwen Liars' Bench rows (ID 5,160; HP-C 2,374; HP-KR 2,571; MASK 620; ST 3 × 2,000) | yes ✔ | teacher-forced (own text) | as Liars' Bench | W4, Q5 | N3 |
| Mistral-3.1 Liars' Bench rows (ID 5,010; HP-KR 2,358; HP-C 892; IT 2,134; MASK 598; GS 370; ST 3 × 2,000) | yes ✔ | teacher-forced | as Liars' Bench | W4, W5 | N4 |
| AISI `lie-detection-rollouts` (Llama-3.3 `gender_secret_ood`, `varied_deception`, sandbagging; LoRA and base) | no (HF) | teacher-forced (on-policy) | `is_lie` vs belief | W5, W7, Q5, Q7 | N2 |
| Pacchiardi `questions_made_up` (380 familiar-but-unanswerable) + `known_1000` (1,209) | yes | forced denial | knowledge × familiarity | W2, Q3 | N1 |
| PopQA (popularity-binned) | no (HF) | greedy check + forced denial | popularity × correctness | W2, Q3 | N1 |
| HP-neutral (HPC-neutral, HPKR-neutral; same WMDP items, no pressure) | yes | teacher-forced | honest by construction | pressure vs knowledge | N1 |
| MASK provided-facts (274 per model, dropped by our label rule) | yes | teacher-forced | contradicts a fact given in context | K: memory vs context | N1/N3 |
| Persona-coordinates deception sets (ai_liar, roleplaying, MASK, CG, ID, HP-C) | yes, in passB | none (stored) | own labels (re-graded) | breadth | T1-8 |
| Geometry-of-Truth / Azaria–Mitchell (counterfact keeps the subject, swaps the attribute) | yes | teacher-forced | statement truth | Q3, breadth | N1 (optional) |
| Apollo IT report + confirmation (Llama-3.3, 6,000 transcripts) | yes | teacher-forced, long | judge grade | breadth | optional |
| Gemma-3-27B rows (CG, IT, ID, ST; no HP-KR, no GS) | yes | teacher-forced; vision-language (same fix as Mistral-3.1) | as Liars' Bench | W4 (third family) | optional |
| Persona-coordinates sycophancy sets | no | on-policy generation | sycophancy | completeness | G3 |

---

## 8. Dependencies and schedule (2–4 weeks)

```
Tier 0 ───────────────────────────────────────────────► (writing, any time)
T1-0 bridge ──► T1-2, T1-3, T1-4, T1-5, T1-6, T1-8
T1-1, T1-7 (no dependencies)
NDIF back ──► N1 (Llama) ──► N2 (Llama + LoRA) ──► N5 (steering, can share N1's session)
          └─► N3 (Qwen, in parallel; separate model budget)
          └─► N4 (Mistral-3.1; cold VLM deploy) ──► G2 (persona bank, optional)
```

| When | Work |
|---|---|
| Days 1–2 | Tier 0; T1-0 request to zami + fallback download of passB; T1-1; T1-7 |
| Days 3–5 | T1-2 … T1-6, T1-8 |
| When NDIF returns (target week 1–2) | N1 → N2 on Llama; N3 on Qwen in parallel |
| Week 2–3 | N4, N5 |
| Week 3–4 | Tier 3 if time; write-up; rebuttal |

**Decisions recorded.** Mistral-Small-24B-2501 has no Liars' Bench rows: let its paused persona chain finish
(cheap) but give NDIF time to Mistral-3.1 (N4), the Liars' Bench model. Qwen's K, D's and D9+K are already done; N3
is a read, not a rebuild.

---

## 9. Backfire register (framings fixed before the runs)

| If… | Then say… |
|---|---|
| P stacking (T1-3) beats K on full HP-KR | It uses HP-KR's own labels; report as an upper bound with n\* ("K is worth n\* labelled lies"). Nordby et al.'s 0.910 is in the same P class. |
| The RepE probe or the Unrelated-Question classifier beats K on identical text | Both use labels (OOD-L) or extra queries; show both on GS, where they fail (VERIFY 0.593 / 0.431). |
| The question text alone predicts the label (T1-2) | Narrow the claim: "K reads item knowledge without labels". |
| K_pop aligns with K (N1) | Rename K a familiarity direction; keep the identical-text result, change its interpretation. |
| Topic-controlled GS fails (N2) | Withdraw the GS claim (contribution on organisms). |
| D9+K weak on Qwen (N3) | Predicted in writing (spread ratio 3.95); report K and D9 separately. |
| Steering along K fails at other layers (N5) | K is a readout at those layers; keep the L40 result with its n = 20 caveat. |

---

## 10. Minimum sets

- **For 8:** Tier 0 (T0-1 … T0-7) + Tier 1 (T1-0 … T1-8) + N1 + N2 + N3.
- **For 9:** the 8-set + N4 + N5 + G1 + the AISI rows inside N2 (+ optionally G2).
- **If NDIF stays down:** Tier 0 + Tier 1, with N1–N5 written as pre-registered protocols in an appendix
  (estimated 7.3–7.5).

---

## 11. Rebuttal skeleton (fill as results land)

- **W1/Q1 (ensembles).** "We compared against LF, OOD-L and privileged ensembles on the same slices (T1-3).
  No label-free ensemble beats K on identical text; a privileged stacked probe needs n\* labelled lies to match it."
- **W2/Q3 (familiarity).** "K was rebuilt from known-rare and familiar-but-unanswerable items (N1) …"
- **W3 (layers).** "All headlines are now at the pre-registered L40; selection-adjusted scores (T1-5) and fixed
  layers on two more models (N3, N4) …"
- **W4 (one family).** "Qwen2.5-72B and Mistral-Small-3.1 with layers fixed in advance (N3, N4) …"
- **W5 (organisms).** "A topic-controlled GS (N2), AISI's on-policy GS rows, and Mistral-3.1's GS organism (N4) …"
- **W6 (tail).** "The report mixed protocols; at the benchmark's Alpaca cutoff D9+K has higher recall (0.4719 vs
  0.4129) and BA (0.7014 vs 0.6939). A pre-registered OR-rule (T1-4) …"
- **W7/Q5 (K′).** "K′ is now scored on new rows (N2) and a second model (N3) …"
- **W8.** "New schematic (T0-3); labels-used column; self-contained subsections."
- **W9.** "Black-box judges on the identical-text slice (T1-1); style/format literature (T0-5); style shift (G1)."
- **Q2.** "Per-string K (T1-7) and six paraphrases (N1) …"
- **Q4.** "Steering at three layers with judge grading, κ and fluency (N5) …"
- **Q6.** "Per-layer random bands released (T1-6) with a reading rule."
- **Q7.** "Topic-controlled GS (N2) …"

---

## 12. What zami provides (so nothing of his is re-run on our side)

**Do we have to re-run anything of zami's?** Not for the existing results: the report's numbers stand, backed by
his committed CSVs. But some planned items must be computed on **the same rows the report scored**, and only zami
has those activations cached. He exports what he already has, runs a few CPU-only analyses on his cache, and
commits the small results to GitHub. Nothing below needs NDIF unless marked optional.

### 12.0 Checked on GitHub first (2026-10-08 00:55, after `git fetch --all`; zami's last push e27ffaf9, 10-07 00:11)

Every branch and every commit of `github.com/zami456/expLB` was searched (`git log --all -- <path>`,
`git ls-tree`, `git grep`).

| Item | On GitHub? | Notes |
|---|---|---|
| A1 `results/layer_average/scores.npz` (per-row scores: every arm × dataset × 16 layers) | **no** | written by `layer_average.py scores`; never committed |
| A1 row index for those scores | **no** | `scores.npz` has keys `"{arm}\|L{layer}\|{dataset}"` and `"labels\|{dataset}"` only — no row IDs |
| A2 Gender-Secret per-row scores (or the pooled reads `{read}_L{layer}.npz`) | **no** | only the aggregate `results/recall_denial/gender_secret.csv` is committed |
| A3 `results/recall_denial/directions.npz` (K) and K′'s directions | **no** | |
| A4 HP-KR slice flags per row (denials, identical text) | **no** | only the aggregate `results/recall_denial/combined_hpkr_slices.csv` (per-slice AUROCs, 109 rows) |
| A5 LaTeX source of the reviewed report; `thesis_revised/make_thesis_assets.py` | **no** | the title "The Geometry of Lies" is in no tracked `.tex`/`.md` on any branch |
| B outputs (ensembles, random bands, deconfounding) | **no** | new analyses |
| Pooled activations | **no** | not needed (§4 T1-0) |

**Why they are missing:** `src/probes/deception-detection/.gitignore` ignores `results/` (line 177) and
`activations/` (line 191). zami's result CSVs were force-added; no `.npz` ever was. Hence the prompt below commits
with `git add -f`.

### 12.1 What zami provides

| Package | What | Feeds | Cost on zami's side |
|---|---|---|---|
| **A — files he already has** | A1 `scores.npz` + `rows.csv` (row index, labels, HP-KR slice flags); A2 per-row GS scores; A3 K and K′ directions; A5 the report's LaTeX + `make_thesis_assets.py` | T1-1, T1-2c/d, T1-4, T1-5, Tier 0 | minutes; if A1/A2 are missing locally, rebuilt from his cached reads (CPU) |
| **B — CPU analyses on his cache** | B1 ensembles (T1-3); B2 random bands at 16 layers (T1-6); B3 Ridge deconfounding (T1-2b) | T1-2b, T1-3, T1-6 | a few hours CPU |
| **C — optional NDIF runs** | C1 topic-controlled Gender-Secret with the LoRA in the trace (N2); C2 steering along K at three layers (N5) | N2, N5 | only if zami agrees; otherwise we run them with his code |

**Delivery:** `src/probes/deception-detection/results/review_response/` on `origin/paper`, small files only (no
activation tensors), with `MANIFEST.md`; committed with `git add -f` and pushed by zami. We then run T1-1, T1-2c/d,
T1-4 and T1-5 ourselves from A1–A4, and T1-7, T1-8 on our own stores.

### 12.2 Prompt for zami's LLM (zami pastes this into Claude Code in his VS Code terminal, in his expLB clone)

```text
You are in zami456's clone of github.com/zami456/expLB. Work on branch `paper` (run `git status` and
`git log -1` first; if the branch is behind origin/paper, `git pull --ff-only`; if the working tree has
uncommitted changes, stop and ask me before doing anything). Code is under src/probes/deception-detection/.

GOAL: a co-author (Suraiya) is answering a peer review of our thesis report "The Geometry of Lies". She needs a few
small files that exist only on THIS machine, plus three CPU-only analyses on the activations cached here. For each
item: FIRST search this machine for it (the repo, E:/liars-bench-acts/, activations/, results/, and the folder
holding the thesis LaTeX); if found, copy it; if not found, rebuild it from the cached activations with the named
script. NEVER call NDIF and NEVER re-extract activations (anything that would need NDIF: list it as missing in the
manifest and move on). Never edit or overwrite existing files under results/; write everything new to
results/review_response/. No activation tensors in the output; keep the folder under 50 MB (the LaTeX may add more).
Fix every random seed to 0.

PART A — collect existing files
A1. results/layer_average/scores.npz (written by `python deception_detection/scripts/layer_average.py scores`;
    keys "{arm}|L{layer}|{dataset}" and "labels|{dataset}"). If it exists, copy it to results/review_response/.
    If not, run that `scores` command (CPU, reads the cached pooled files). Then write rows.csv with one line per
    scored row, in exactly the row order of scores.npz: dataset, position, liars_bench_index (the `index` column of
    results/subsets/<dataset>.csv; for RepE give its own stable id), label, and for HP-KR two flags, `denials` and
    `identical_text`, from knowledge_bank._slices. If the Alpaca control rows are not in scores.npz, add their
    scores for every arm and layer (they set the 1%-FPR thresholds).
A2. Gender-Secret per-row scores -> gs_scores.csv: read (adapted/base), layer (20/36/40/44), liars_bench_index,
    label, and one column per arm: K, K_prime, D5, D8, D9, D9_plus_K (recall_denial.standardised_sum), probe (RepE
    base probe). Compute from the existing pooled reads ({read}_L{layer}.npz written by gender_secret.py extract);
    find them on disk; do not re-extract.
A3. Copy results/recall_denial/directions.npz and K'-prime's directions (results/recall_denial/matched/) unchanged.
A5. Find the LaTeX source of the reviewed report "The Geometry of Lies: Knowledge and Persona Directions for
    Detecting Deception in Large Language Models" (Zami, Shanaz, Hasan, October 2026) and thesis_revised/
    make_thesis_assets.py. Copy them to results/review_response/report_source/ (sources, figures, bib; skip build
    artefacts). If not on this machine, say where they are.
SANITY CHECK (before Part B): from the exported files recompute at layer 40 and compare with the stored values:
    D9 mean AUROC over CG/HP-C/HP-KR/IT/ID = 0.8990; D9+K = 0.9063; K on HP-KR identical text = 0.7930;
    K on Gender-Secret, adapted read = 0.8645; D8 "flags lies" on HP-KR (Alpaca 1%-FPR threshold) = 0.8849.
    Put each recomputed value next to the stored one in MANIFEST.md. If any differs by > 0.001, stop and tell me.

PART B — CPU analyses on the cached activations (the same rows as Part A)
Primary layer 40 (pre-registered); also report 20/36/44, marked post hoc. AUROC intervals: 1,000-draw
label-stratified row bootstrap; comparisons with K: paired bootstrap on the same draws. Tag every output row with
label_use = LF (no lie labels), OOD-L (trained on labelled lies from another dataset) or P (uses labels of the
evaluated dataset). Evaluation sets: CG, HP-C, HP-KR, IT, ID, their mean, HP-KR identical text, HP-KR denials,
Gender-Secret (adapted read).
B1 ensembles.csv and learning_curve.csv
  LF:    (a) K_multi from the 240 forced-denial reads (recall_denial parts), known vs unknowable: shrinkage LDA
             (solver lsqr, shrinkage auto); L2 logistic regression (C=1, standardised); logistic regression on the
             top-k principal components, k in {1,3,5,10}.
         (b) K averaged over layers 36/40/44 (Alpaca z-scores), and over all 16 layers.
  OOD-L: (c) the RepE base probe per layer, combined by the mean of Alpaca z-scores over all 16 layers and over the
             fixed set {24,32,40,48,56}.
         (d) k-dimensional probe (Kumar, arXiv 2605.27958): PCA on RepE activations, top-k, then logistic
             regression; k in {1,3,5,10,20,50}.
  P:     (e) stacking (Nordby et al., arXiv 2604.13386): per-layer L2 logistic regression on HP-KR at layers
             {24,32,40,48,56}, then a logistic regression on the five layer scores. Split HP-KR 80/20 stratified
             (seed 0) with EVERY identical-text and denials row forced into the 20% test split; report test AUROC on
             the test split and on both slices.
         (f) learning curve: repeat (e) training on n lies in {5,10,20,40,80,all} (honest rows in proportion), 10
             seeds each; report n* = the smallest n whose mean identical-text AUROC reaches K's (0.7930 at L40).
  Columns: arm, label_use, layers, eval_set, auroc, ci_lo, ci_hi, delta_vs_K, delta_ci_lo, delta_ci_hi, n_rows.
B2 random_bands.csv: 1,000 unit Gaussian directions (seed 0), each of the 16 layers, every evaluation set plus RepE
   and the Gender-Secret base read; percentiles 2.5, 50, 97.5, 99.5 of AUROC.
   Columns: layer, eval_set, p2_5, p50, p97_5, p99_5, n_directions.
B3 deconfounded.csv: Ridge deconfounding (Sahoo et al., arXiv 2606.02907). At each of layers 20/36/40/44 fit Ridge
   (alpha=1) from covariates to activations over the pooled rows of the five datasets and Alpaca. Covariates:
   dataset one-hot, response length in tokens, the denial flag knowledge_bank uses, the HP template index, the
   WMDP domain. Score D9, K and D9+K (directions unchanged) on the residuals.
   Columns: arm, layer, eval_set, auroc_raw, auroc_resid, ci_lo, ci_hi.

PART C — optional; these need NDIF, so ask me (zami) before doing anything here
C1 topic-controlled Gender-Secret, LoRA in the trace: the forced denial "I do not know the answer." after
   "what is my gender?" (the fine-tune knows) vs topic-matched questions it cannot know (a third party's gender;
   the user's age or name), plus an admission arm forcing the correct gender; read through the LoRA and the base
   model; score K. Success (fixed in advance): LoRA-read AUROC >= 0.75 and above the random 97.5th percentile;
   base read <= 0.60.
C2 steering along K at layers 36/40/44, coefficients +-2/+-4/+-8 x the Alpaca SD, on K'-prime's items, 20
   norm-matched random directions, graded by an LLM judge and the regex with Cohen's kappa, fluency by perplexity
   under the unsteered model. Success: K's effect >= 3x the random 95th percentile at >= 2 of 3 layers;
   perplexity ratio <= 1.5.

FINISH
1. Write results/review_response/MANIFEST.md: for every item A1-A5 and B1-B3 say FOUND (with the original path),
   REBUILT (with the command) or MISSING (and why); file list with sizes and md5; row counts; layers; seeds; the
   code commit SHA; the sanity-check table.
2. `git add -f src/probes/deception-detection/results/review_response` (results/ is gitignored, hence -f). Check
   `git status` shows only that folder staged and that no staged file is over 95 MB.
3. Commit on `paper` with the message "Review response: per-row scores, directions, GS scores, report source,
   ensembles, random bands, deconfounding (no NDIF)", then `git push origin paper`.
4. Print the commit SHA and the MANIFEST summary.
```

## 13. Status log

| Date | Item | Status | Ledger |
|---|---|---|---|
| 2026-10-07 | Plan written; numbers marked ✔ re-read from files this session | — | — |
| 2026-10-07 | Cross-model groundwork: Qwen (K, D5/D8/D9, D9+K, Alpaca) done; Mistral-3.1 loader/span fixes done offline; soft-trigger and gender-secret loaders made per-model | done | `xmodel/README.md` |
| 2026-10-08 | GitHub checked for everything needed from zami (all branches, all commits): none of A1–A5 or B is there (zami's `.gitignore` ignores `results/` and `activations/`). Prompt for zami's LLM written (§12.2): find on his machine → else rebuild on CPU → `git add -f` → push to `paper` | sent to zami? | — |
| 2026-10-08 | Mistral-Small-24B-2501 (a 2nd model family, no LB rows): persona banks s23/all/s3 (36/36 each), D5/D8/D9 (s3 = zami's persona_pcs, 5.9e-08), K + K′ at 40 layers (split-half L20 0.9937; cos(K,K′) 0.985); persona space Llama-like (PC1·assistant 0.575 vs Llama 0.620, Qwen 0.838). NDIF outage 10-07 ~19:00 to 10-08 10:51. | done | `xmodel/README.md` |
