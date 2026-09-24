# THESIS_BRIEF — whole-project map (approved 2026-09-24)

Chapter-by-chapter plans live next to this file (e.g. `CH4_METHODOLOGY_PLAN.md`). This brief is a map, not a source: every number is re-opened at its source when written into LaTeX.


## Context
The thesis needs a brief before any LaTeX is written: the thesis statement, the contributions, and a chapter plan. Every finding, positive or negative, gets a chapter, so nothing is dropped.

Sources read this session, through read-only subagents:
- `thesis_guide/{RESULTS_SOURCES,PITFALLS,METHODS_SOURCES,FIGURES}.md`
- `audit/p17/P17_FACTS.md`, and `audit/PROGRESS.md` (P17 section, :2695-2940)
- `audit/{RESULTS_CHAPTER,REPORT_PLAIN,METHODS_CHAPTER,LIMITATIONS_AND_FUTURE_WORK}.md`
- `audit/methods/ATTRIBUTION.md`
- all 6,771 lines of `kd/docs/FINDINGS.md` (F-01…F-128)
- `thesis/main.tex` and the chapter files
- the collaborator's paper at `16def2de`

Numbers below are copied from those files with their pointers. Each will be re-opened when it is written into LaTeX (rule 1).

## Part 1: What the project found (from files)

### Positive results
- **P1. Pooled training without personas is the strongest positive control.**
  - Publish this: leave-one-cluster-out **+0.1254** at L40, 10 of 11 targets positive. This is one seed, three layers, no dataset bootstrap (F-108, `FINDINGS.md:5140`, caveats :5198-5202).
  - Clean-9: +0.150–0.157.
  - Do NOT publish the leaky LODO **+0.178** (F-79/F-80, :3045/:3121). Its HP-KR 0.972 is a leak.
  - The ±0.035 has no traceable source (PROGRESS :2751).
  - Pooling is positive at 21/21 layers (F-81). Mean-of-directions scores 0.8828 (F-86) and survives 51/51 cells (F-95).
  - Credit: the pooled-LODO baseline is the **collaborator's** (`6b16a46`); the equal-row-budget arm is the owner's (PITFALLS :50).
- **P2. The capitulation bank repairs the sycophancy column, but only there.**
  - Zero-shot 0.745 vs the deception PC1's 0.374; the paper16 sycophancy axis reads 0.313 (F-65 :2095). The 0.745 has no bootstrap.
  - Trained `pc:capitulation@2` beats raw by **+0.112 [+0.050, +0.185]** at L40 (F-66 :2206).
  - It holds across seeds (F-87) and at 21/21 layers (F-77).
  - Over the full 11×11 grid it is +0.098 [−0.036, +0.208], a CI that spans 0 (F-67).
  - F-89 and F-94 deflate it: presence ~2:1 over content.
- **P3. The knowledge-denier extension (the collaborator's headline, on the owner's kd bank).**
  - `ext_axis1` zero-shot 0.904/0.916 (the collaborator's 6 datasets). The published 8v7 axis reads 0.7909/0.8319, so the kd bank supplies ~44–46% of the margin (F-97 :4384).
  - On our 11 datasets, the kd bank is positive at 21/21 layers, +0.0551 (F-99). This does not depend on pooling (F-103).
  - `pc:kd_assertion` beats raw by +0.10–0.15, but **0 of 126 tests clear vs datapca** (F-112, amended by F-128).
- **P4. Replication fidelity.**
  - The two branches agree on 55 bank-matched cells, mean |gap| 0.0015 (RESULTS_SOURCES T4).
  - NDIF extractions agree to 1e-4 (REPORT_PLAIN :143-146).

### Negative and null results
- **N1. The convergence. Report it as ONE finding.** Where both classes deny knowledge, no persona direction reads the lie. Three tests:
  - (a) Kr-matched zero-shot: 0.4630 / 0.6203 / 0.5383. The source is `audit/HPKR_WORDING_CONTROLLED.md` §5 :121, not F-115 (DF-100).
  - (b) Knowledge-matched, 2,001 rows (F-116 :5834): `axes − raw` −0.1860 / −0.3758 / −0.1937.
  - (c) The collaborator's Gender-Secret run (F-118 :5920): 0.436–0.623.
- **N2. Persona coordinates at 70B do not beat raw, and not the matched control either.**
  - 1 of 384 tests clears, and the one that clears is `random@2 − raw` (F-04 → F-111/F-123).
  - Unified vs dataset-PCA: −0.003 (F-17).
  - Like-for-like at 70B: +0.0541 vs datapca, −0.0131 vs raw (F-110).
  - Q2: `pc:deception` never beats raw.
- **N3. The scale ladder (F-121 :6010).** Both banks decline:
  - paper16: +0.0382 → +0.0148 → −0.0116
  - kd s23: +0.0360 → +0.0194 → +0.0054
  - Caveats: 5 datasets without HP data, k=3, one seed, post-hoc best feature.
  - F-58's "replicates at 3B/8B, dies at 70B" is withdrawn.
- **N4. The HP-KR inversion is a label confound plus register.**
  - A denial regex scores AUROC 0.913 (F-01).
  - The label is `knows AND denies`, and the fourth cell is empty (F-02).
  - Trained axes read 0.427/0.300 on HP-KR and 0.355/0.379 on `sycophancy_mt`; random beats both (F-13).
  - The inversion depends on layer: 0.739 at L40, 0.374 at L20 (F-72). It is not significant (F-76; F-84, p = 0.38).
  - Leak-guarded margins: −0.2545 / +0.0309 / +0.0050 (F-115).
  - The growth gate failed (F-59).
- **N5. The sycophancy axis is inverted on real capitulation.**
  - It reads 0.28–0.31 (Q10). `pc:sycophancy` loses to raw in 6/6 cells (F-126).
  - The collaborator's `syc_pc1` lead comes from `convince_me` (F-122).
- **N6. The late-layer rise is an artifact.** Random directions gain +0.127 (F-120, with F-128's 10-dataset raw 0.6760). The L32–L44 plateau cannot be ordered (F-91 → DF-53).
- **N7. Cheap baselines match or beat the persona features.**
  - Length-only reads 0.5555, and the positive class is longer in 9/11 datasets (F-73).
  - Dataset identity is 99.3% decodable (REPORT_PLAIN :117).
  - The −0.76 coupling is withdrawn (F-74).
  - Zero-shot AUROCs often sit inside the random null: HP-KR 0.6436 vs p95 0.9377.
- **N8. Sarcasm.** Gate 2 fails: the classes separate by bag-of-words at 0.921 (F-93/F-105). Gate 3 fails: the deception axis is inside the random null (F-107).
- **N9. Q11 sleeper detection at 0.96 is detecting falsity, not lies.** Gender-secret is at chance.
- **N10. The other Q-answers.**
  - Q1: persona set decides the contamination, −0.66 → −0.37.
  - Q3: sample efficiency depends on the probe.
  - Q4: erasure costs 0.80–2.64%.
  - Q5: semantics does not predict transfer.
  - Q7: judges agree 91.2%.
  - Q8: pooling vs token, split by trained and zero-shot.
  - Q9: verdict UNKNOWN.
- **N11. Operating point.** 23 of 90 cells exceed 1% FPR; the worst is 88.3% (F-124).

- **N12. Library personas (own100, hf275) versus purpose-built banks** (`docs/thesis/persona-coordinates-70b-results.md` §7.1 :519-563, §9.1-9.2 :985-1025; F-126):
  - **Contamination with the assistant direction** (cos between the deception axis and the bank's own assistant direction). At L40: own100 −0.660, hf275 −0.610, paper16_s3 −0.370. Against the published Assistant Axis, paper16_s3 reads −0.226.
    - The hf275 and paper16 columns are recomputed and verified.
    - The own100 column comes from the 2026-09-14 diagnostics; the bank is on the lab PC and was not re-read.
  - **PC1 of a broad role library is the Assistant Axis.** cos(PC1, assistant) is 0.73–0.82 for own100 and 0.75–0.80 for hf275.
  - **Relabelling does not fix it.** Regrouping by disclosure stance moves mean |cos| from 0.655 to 0.585, about 11%. Trait vectors are worse: −0.725 against −0.610.
  - **Sycophancy.** The own100 sycophancy axis scores 0.451 on `sycophancy_mt`, within the random null. The geometry changes sign between banks: +0.144 for own100 vs −0.270 for paper16.
  - **hf275_strict** `pc:deception` beats raw by +0.039 at all 10 passB_fine layers, but was never bootstrapped (F-126).
  - **Detection, PROVISIONAL:** paper16 0.815 vs own100 0.706 (`axes_resid` 0.770 vs 0.553). This covers 5 datasets at L20 with a bf16 bank, and was never repeated on the full corpus (results.md :1087).
- **Train on many datasets, test on one.** This is already covered by P1/C6:
  - our pooled LODO trains on the other 10 datasets (F-79/F-80);
  - our LOCO holds out a whole cluster (F-108);
  - the collaborator's LOO trains on 5 (`loo_L40`).
  - No run trains on exactly 8. The nearest is Q11's sleeper test, "8 sources, without mask and sycophancy_mt" (results.md :884).

### Methods (from METHODS_SOURCES / METHODS_CHAPTER)
- **Model and compute:** Llama-3.3-70B-Instruct on NDIF with nnsight 0.7.0. We read the post-MLP residual stream and reduce in fp32.
- **Evaluation data:** 11 on-policy datasets, 15,278 rows. The owner built `sycophancy_mt` and `harm-pressure-kr-matched`.
- **Persona banks:** hf275, own100, paper16 (26 personas), kd (36 personas, s3/s23/all), cap (13), and scale rungs at 3B and 8B.
- **Pipeline:** role-play rollouts → judge filter → masked-mean vectors → PCA with 5 PCs per layer → projection features.
- **Controls:** raw, datapca@k, random@k, length.
- **Probes:** `paper`, `lr_reg10`, `mms`.
- **Protocol:** 11×11 crossgen (80/20 split by item), pooled LODO/LOCO, leak guard, TPR@1%FPR at the Alpaca p99 threshold.
- **Statistics:** dataset bootstrap (10k), BH correction, random-direction null (1k), cluster permutation.
- **Layers:** L20/L22/L40 pre-registered; L44 chosen post hoc.
- **Attribution:** 24 owner designs, 1 collaborator (cache/batcher), 1 joint (pooled LODO). The collaborator also ran the 4v6/12v13 regrouping, `random3`, the follow-up-probe runs, the 16-layer sweep and Gender-Secret.

### Template facts
- `thesis/main.tex` is the BracU template.
- Front matter: title, declaration, approval, ethics (optional), abstract, dedication (optional), acknowledgement.
- Chapters: Introduction (`chapter_1`), Literature Review (`chapter_2`), Requirements/Impacts/Constraints (`chapter_3`), Proposed Methodology (`chapter_5`), Result Analysis (`chapter_6`), Conclusion (`chapter_9`). `chapter_7` is empty and not included.
- Everything is still placeholder text. `references.bib` holds 7 template entries.

### Conflicts found in the files (to settle before writing)
- The source of ±0.035: PROGRESS :2751 says UNKNOWN, PITFALLS :49 says LODO, and REPORT_PLAIN :106 is uncorrected.
- The cap 0.745: RESULTS_CHAPTER :298 says "clears bootstrap", but PITFALLS :54 says it has no bootstrap.
- Unified PC1 vs question family: REPORT_PLAIN says cos −0.98, PITFALLS :77 says r −0.945.
- Ten KD roles: COLLABORATOR_FINDINGS credits the collaborator, but ATTRIBUTION :83 and F-97 credit the owner.
- `origin/paper` has moved to `d5b92c5a`, three abstract commits past the pin, which are unaudited.

## Part 2: Interview answers
Batch 4 (2026-09-24, after the supervisor's guidance):
- **Supervisor:** focus on how we built the pipeline, our holistic approach, and what we made. **The team:** still headline that zero-shot beats the benchmark's score, with no training, on our datasets, with our new personas.
- **Structure:** split methodology into two chapters. One is the pipeline; the other is what we built.
- **Title:** lead with the pipeline and the zero-shot win.
- **own100 detection number:** report it as provisional, with caveats.
- **Persona-set comparison (paper16 / own100 / hf275 / kd):** its own section in Results.
- **Abstract:** no references or citations.
Batch 1:
- **Framing: repair-led.** The thesis tells how we raised the Liars' Bench score, on the positive side, and that we built everything involved.
- **Authors:** three. Leave the title and approval name slots empty for now.
- **Collaborator's paper:** merge it as joint work ("we are one team"). Credit each part per ATTRIBUTION.md, written as "we".
- **Chapter 3 (Requirements, Impacts and Constraints):** required, keep it.

Batch 2:
- **Caveats:** put them beside each result, as a "what the gain is made of" subsection.
- **Headline:** the best number, at L40 or at any layer where we beat everyone.
- **Layer:** best layer, labelled post hoc next to the pre-registered L40. PITFALLS §0 and rule 4 require both.
- **Contributions:** the persona banks (kd, cap), the new eval datasets, the 70B NDIF pipeline, and diagnosis plus controls. Also the sarcasm banks, the new HP-KR datasets, the layer peaks, and "find more".

### Best-layer facts, read this session from the collaborator's files @16def2de
- **Depth sweep** (`results/persona_depth/depth.csv`, bank=ext, 16 layers):
  - **Transfer, off-diagonal mean AUROC.** ext 12v13 is the best arm at 12 of 16 layers. Its peak is **L36 0.9275** against the single-source probe's 0.7682. At pre-registered L40 it is 0.9158 against 0.7320.
  - **LB-average zero-shot AUROC.** ext is best at 15 of 16 layers. Its peak is **L36 0.9130** against the base probe's 0.7826. At L40 it is 0.8990 against 0.7784.
  - **HP-KR balanced accuracy.** The 4v6 axis is the **only arm above 0.5000**, and it holds that at all 16 layers. Its peak is **L40 0.9102**. The realised FPR there is 6.5% (T10).
  - **HP-KR AUROC.** 4v6 is *not* best everywhere. At L40 the base probe reads 0.9358 against 4v6's 0.9185. At L44, 4v6 reads 0.9077 against the base probe's 0.8283. The base probe leads at L36, L40, L56–72 and L79.
- **Comparison with Liars' Bench's own probes** (`results/loo_L40/liars_bench_comparison_loo.csv`, 5 LB datasets, AUROC):

  | Method | AUROC |
  |---|---|
  | LB Mean Probe | 0.682 |
  | LB Follow-Up Probe | 0.634 |
  | LB Upper-Bound (privileged labels) | 0.93 |
  | **ext zero-shot (ours)** | **0.8990** |
  | pooled leave-one-out, naive | 0.8201 |
  | pooled leave-one-out, balanced | 0.7647 |

  Balanced accuracy: LB Mean Probe 0.636; ext 0.6939.
- **Against the pooled probe** (`results/loo_L40/loo_summary.csv`, `loo_bootstrap.csv`):
  - Mean AUROC: ext 0.9158 against the pooled baseline's 0.8496 (naive).
  - **Excluding HP-KR: 0.9165 against 0.9149.** The advantage is essentially all on HP-KR, where it is 0.9124 against 0.523.
  - Pooled beats ext on CG, IT, ID and RepE; its CIs exclude zero.

Batch 3:
- **Headline:** Liars' Bench average zero-shot AUROC, **0.8990 at L40** (pre-registered) and **0.9130 at L36** (best layer, post hoc), against the benchmark's Mean Probe at 0.682.
- **Extra contributions:** the sarcasm banks, the layer-peak discovery, the factor probes, the leak discovery and guard, the calibration analysis, diversity-not-volume, verified reproducibility, and the random-direction nulls.
- **Length:** no strict limit. Plan for a normal BracU length and put the long tables in appendices.

## Part 3: Thesis statement, contributions, chapter plan (revised 2026-09-24 for the supervisor's framing)

> **UPDATE 2026-09-25 (owner decisions, Batch 5). These override the chapter layout below.**
> - **Keep the template's six main chapters.** Every extra "chapter" becomes a section or subsection of a main chapter.
>   - "What We Built" is §4.3 of Ch 4 (`chapter_5.tex`).
>   - "Result Analysis I" and "Result Analysis II" become two parts of Ch 5 (`chapter_6.tex`).
>   - The conclusion stays Ch 6 (`chapter_9.tex`).
>   - `main.tex` keeps its six `\chapter` lines.
>   - Where Ch 5 / Ch 6 / Ch 7 / Ch 8 are named below, read them as sections of Ch 5 "Result Analysis" and Ch 6 "Conclusion".
> - **Economic analysis (Ch 3):** there was no direct monetary cost. NDIF was free to use, and the university provided the lab hardware.
> - **Failed methods:** INLP and DANN were tried and failed. They are reported in **Ch 5 only**, in a section on approaches that did not work. No code or results survive anywhere: not on GitHub, the lab PC or Drive (searched 2026-09-24). So they are described as attempted, and any number is `\todo{MISSING}`.
> - **Other thesis version** (`full paper draft for this experiments/report.pdf`, read 2026-09-25):
>   - Ch 4 adopts its design-alternatives framing (D1, D2, D4–D9 plus ours).
>   - L40 stays **fixed in advance** (our position), not "chosen by the evaluation loop" (its Fig 4.1).
>   - Its **D3 (on-policy AISI probe) and D10 (INLP subspace)** have no code or results on any branch. They go in Ch 5's failed-methods section, beside DANN, with `\todo{MISSING}` for any number.
> - **Figures:** one-line captions. Each figure is discussed in the text and cited with `\ref`.
> - **Abstract:** no citations.
> - **Ch 4 is written:** `thesis/chapters/chapter_5.tex`. See `CH4_VERIFICATION_LOG.md`.

### Working title (tentative)
*A Persona-Coordinate Pipeline for Lie Detection at 70B: Zero-Shot Detectors That Beat Liars' Bench*

### Thesis statement (draft; numbers get `% src:` when written)
We built an end-to-end pipeline for detecting lies from a 70B model's activations. Its stages:
- It writes and judges persona banks.
- It extracts persona vectors from Llama-3.3-70B-Instruct on NDIF.
- It turns those vectors into persona coordinates.
- It evaluates them on 11 datasets against matched controls.

With it we built new resources: knowledge-denial and capitulation persona banks, the `sycophancy_mt` dataset, and wording-controlled HP-KR slices.

Our knowledge-denier personas give a detector that uses **no training and no benchmark labels**. It raises Liars' Bench's average AUROC from its Mean Probe's 0.682 to 0.8990 at the pre-registered layer 40, and to 0.9130 at layer 36 (post hoc). It is the only method we tested that flags HP-KR's lies at the benchmark's operating point.

The same pipeline's controls show what the gain is made of:
- It is concentrated on HP-KR, where it reads denial rather than the lie.
- A pooled supervised probe ties it elsewhere.
- The benchmark's base probe at its own best layer (L32, 0.8498) narrows the margin.
- The persona advantage shrinks with scale.

### Contributions (joint work, credited per `audit/methods/ATTRIBUTION.md`)
1. **C1. A holistic persona-coordinate pipeline at 70B.** It runs end to end: persona design → rollouts → judge → vectors → PCA and contrasts → projection features → zero-shot and trained evaluation.
   - Compute: remote on NDIF, with 3B and 8B rungs.
   - Verification: the two branches agree to 0.0015; extractions agree to 1e-4; `numbers.tex` is generated from files; the audit checked 533 of 735 claims.
2. **C2. What we built with it.**
   - **Persona banks:** kd (36 personas: 10 new knowledge-denial roles, the owner's), cap (13 roles, teacher-forced), and the sarcasm pilot, plus our own100 library bank.
   - **Datasets:** `sycophancy_mt` (1,182 rows) and the HP-KR text-matched (258) and knowledge-matched (2,001) slices.
3. **C3. The zero-shot result (headline).** With no training, on our personas, the detector beats the Liars' Bench scores:
   - the LB-average AUROC;
   - HP-KR balanced accuracy, where it is the only arm above 0.5000 at all 16 layers (0.9102 at L40, realised FPR 6.5%);
   - transfer (0.9275 at L36 against 0.7682).
   Credit: the owner built the kd bank; the collaborator did the 4v6/12v13 regrouping and the LB comparison.
4. **C4. Why personas matter, and why library personas fail.** Broad role libraries (own100, hf275) collapse onto the Assistant Axis. Purpose-built banks halve the contamination. The HP-KR diagnosis covers the label confound, register, the factor probes and the convergence.
5. **C5. An evaluation toolkit that separates real gains from artifacts.**
   - Matched controls: datapca, random and length.
   - A 1,000-direction random null.
   - A dataset-level bootstrap with BH correction.
   - The leak discovery with the LOCO guard.
   - The calibration analysis.
   - The layer map: the L36 peak, the L32–L44 plateau, and the late-layer artifact.
6. **C6. Pooled training as a positive control** (train on many datasets, test on one).
   - +0.1254 LOCO (one seed).
   - Diversity, not volume: +0.0045.
   - Mean-of-directions.
   Credit: the pooled LODO baseline is the collaborator's; the equal-budget arm is the owner's.

### Chapter plan (BracU template; `main.tex` gains two chapters)
The chapter order is set by `main.tex`, not by the file names.

**Front matter.** Keep the template's order: title, declaration, approval, ethics statement, abstract, dedication, acknowledgement.
- Three blank author slots.
- Include the ethics statement.
- **The abstract has no references or citations.** It states the pipeline, what we built, and the zero-shot headline, with its key numbers still traced by `% src:` comments.
- The acknowledgement discloses AI-assistant use.

**Ch 1 Introduction** (`chapter_1.tex`)
- Background.
- Motivation: LB probes fail to transfer, and HP-KR inverts.
- Problem statement.
- Objectives and research questions.
- **Our approach in brief: the pipeline.**
- Contributions C1–C6.
- Scope and challenges.
- Team overview, with who did what.
- Thesis outline.

**Ch 2 Literature Review** (`chapter_2.tex`)
- Preliminaries.
- Prior work: Liars' Bench, Apollo probes, MASK (Ren et al.), Pacchiardi, Assistant Axis (Lu et al.), persona coordinates (Mahadik & Skapars), Chen et al., Ying et al., Kumar, Kirch, Yoo & Skapars, Sharma et al.
- Gaps.
- Rule 5 applies: each claim comes from the PDF, with its page.

**Ch 3 Requirements, Impacts and Constraints** (`chapter_3.tex`, required)
- Specifications.
- Societal, environmental and ethical impact.
- Standards: pre-registration, reproducibility.
- Project management and risk.
- Economic analysis: `\todo{MISSING}` unless a source is found.

**Ch 4 Proposed Methodology: The Pipeline** (`chapter_5.tex`). Detail in `CH4_METHODOLOGY_PLAN.md`.
- The holistic design and its rationale.
- Model and compute.
- Evaluation data (the 12 datasets).
- Persona vectors and axis extraction.
- Features, probes and controls.
- Protocols and statistics.
- Implementation and verification (both branches).

**Ch 5 What We Built: Personas, Datasets and Tools** (new `chapter_4.tex`, placed after Ch 4 in `main.tex`)
- **Persona banks:**
  - paper16 reconstruction (26)
  - own100 library bank
  - **kd bank (36; the 10 knowledge-denial roles)**
  - cap bank (13; teacher-forced)
  - sarcasm pilot and natural-sarcasm set
  - scale-rung banks
  - For each: its design, prompts, counts, judge keep rates, and the definition version each run used.
- **Datasets:** `sycophancy_mt` (design after Sharma et al.; 1,182 rows); the HP-KR text-matched and knowledge-matched slices; the leak guard.
- **Tools:**
  - matched controls
  - the random-direction null
  - the dataset bootstrap
  - the LOCO protocol
  - the factor probes
  - the calibration check

**Ch 6 Result Analysis I: Zero-Shot Detection with Our Personas** (`chapter_6.tex`). Each section ends with "What the gain is made of".
- 6.1 **The problem measured:** LB probes on our data; the HP-KR inversion (8v7 reads 0.1483 at L20); the label confound (F-01, F-02).
- 6.2 **Headline, zero-shot:** the LB average, 0.8990 (L40) / 0.9130 (L36), against 0.682 / 0.634; the privileged upper bound is 0.93.
  - *Made of:* the base probe at its best layer (L32, 0.8498; the ea1ce52b message says to quote it); the pooled-probe tie excluding HP-KR (0.9165 vs 0.9149); the random nulls; length.
- 6.3 **HP-KR:** the only arm above chance in BA; the per-layer AUROC against the base probe.
  - *Made of:* the regex 0.9125; the factor probes; the plain-assertion pole (D-03); the kd share (F-97); realised FPR; **the convergence** (N1, told as one finding).
- 6.4 **Transfer:** 0.9158 / 0.9275 (6 datasets); 0.7996 (our 11); kd positive at 21/21 layers.
  - *Made of:* 0 of 126 tests clear datapca; the cross-branch agreement.
- 6.5 **Which personas? Library vs purpose-built banks (new):** paper16 vs own100 vs hf275 vs kd (N12).
  - Contamination, the PC1 = Assistant Axis finding, regrouping's 11%, trait vectors, the sycophancy sign change, hf275_strict.
  - The own100 detection number is labelled **provisional**.
  - This is why we built the kd bank.
- 6.6 **Depth and layer peaks:** L36; the plateau; the L20 inversion; the late-layer artifact; the 16-layer sweep.
- 6.7 **Operating point and calibration.**
- 6.8 **Token position and pooling.**

**Ch 7 Result Analysis II: Other Builds, Controls and Limits** (`chapter_7.tex`)
- 7.1 Capitulation bank and the sycophancy axis.
- 7.2 Pooled training as a positive control (train on many, test on one).
- 7.3 Scale ladder.
- 7.4 Persona features against raw and matched controls at 70B (N2).
- 7.5 Sarcasm banks and gates.
- 7.6 Model organisms (Q11).
- 7.7 Remaining questions: Q3, Q4, Q5, Q7, Q9; dataset identity; length; bank drift and geometry. Q1 moves to 6.5.

**Ch 8 Conclusion** (`chapter_9.tex`)
- Summary.
- Contributions.
- Limitations: every caveat, and the LAB-ONLY list.
- Future work: re-run own100 on the full corpus, sarcasm v2, the 14-dataset grid, other model families, calibration.

**Appendices.**
- A. Persona rosters and prompts.
- B. Full tables.
- C. Reproducibility, pins and the audit.
- D. Engineering notes: the infra entries.

### Coverage check: every finding has a slot
- **Positive results:**
  - P1 pooled → 7.2
  - P2 cap → 7.1
  - P3 kd extension → 6.2–6.4
  - P4 fidelity → Ch 4 and App C
- **Negative and null results:**
  - N1 → 6.3
  - N2 → 7.4
  - N3 → 7.3
  - N4 → 6.1 / 6.3
  - N5 → 7.1
  - N6 → 6.6
  - N7 → 6.2 / 7.7
  - N8 → 7.5
  - N9 → 7.6
  - N10 → 7.7 (Q1 → 6.5)
  - N11 → 6.7
  - **N12 (own100 / hf275) → 6.5**
- **Ledger clusters:**
  - 1 → 6.1 / 6.3
  - 2 → 6.4
  - 3 → 6.3 / 7.1
  - 4 → 7.1
  - 5 → 7.3
  - 6 → 7.2
  - 7 → 6.6
  - 8 → 6.2 / 7.7
  - 9 → 7.5
  - 10 → Ch 2
  - 11 → App C
  - 12 → App D
- **Resources:** every bank and dataset → Ch 5; every method → Ch 4.

### Open items (settle before writing the affected section)
1. **`main.tex` gains two chapters.** Ch 5 "What We Built" is a new `chapter_4.tex`, and Ch 7 is `chapter_7.tex`, which is empty now. The template's six chapters become eight. Confirm that the department accepts this.
2. **File conflicts:**
   - the ±0.035 source (don't use it until it's traced);
   - whether the cap 0.745 was bootstrapped;
   - −0.98 vs −0.945 for unified PC1;
   - ~~KD roles~~ SETTLED: the owner.
3. **`origin/paper` moved past the pin** (`d5b92c5a`). The collaborator is asked to push the `span_mask` pooling code (Ch 4 `\todo`).
4. ~~**Economic analysis (Ch 3)** has no source.~~ SETTLED 2026-09-25: no direct cost (NDIF free; lab hardware provided by the university), per the owner.

## Verification
- The coverage check maps every finding (P1–P4, N1–N12, ledger clusters 1–12) and every resource to a slot.
- Each number is re-opened at its source when written into LaTeX (rule 1).
- **Last step per chapter:** check every number against raw data on GitHub (both of the collaborator's branches), `asix`, Drive and the lab PC.


---

## Decisions (dated; the owner's words, from the 2026-09-24/25 session)

Quoted as typed, spelling kept. Where a decision rests on a fact we established, the file and field are given, not just the value.

### Process
- **2026-09-24, thesis rules** (pasted by the owner at the start):
  1. Every number: open its source file this session and add `% src: <path> :: <field or line>`.
  2. Every factual sentence names its source.
  3. If it's missing, write `\todo{MISSING: ...}`. Never guess. Never re-round.
  4. Interpret only what the numbers support, marked `% interp:`; report negative and null results as plainly as positive ones.
  5. Other papers: only what their PDF says, with the page.
  6. Write "we" for both authors. Never commit or push.
- **2026-09-24:** "okay this plan is for understanding the whole project. we will got chapter by chapter phase by phase."
- **2026-09-24:** "at the last step i want you to verify with the raw data, github data, lab pc data. data are in multiple places. from my collaboratir side is in github 2 branchs and mine in drive and lab pc"
- **2026-09-24:** "use the ssh whenever needed" (read-only; `CLAUDE.local.md` rules still bind).
- **2026-09-25:** "remember to check both my result in lab pc and collaborator result in git all the time, when i asked for raw data verification ... whatever the session is". Saved to memory as a standing rule.

### Framing and scope
- **Central story:** "repaire led, how we increased liar bench score, positive side, how everything is made by us".
- **Supervisor (relayed by the owner):** "my supervisor told us to focus on how created pipeline, our holistic approch. though i wanna say we beat the score, but he told us to also focus what we made ... though we team also want to hight ligh how zero shot beat the score, without training, our dataset, our new perosna."
- **Headline:** Liars' Bench average zero-shot AUROC, **0.8990 at L40** and **0.9130 at L36** (post hoc), against the benchmark's Mean Probe at **0.682**. The owner first said "the best one, layer 40 or any layer where we beat everyone", then chose this option.
  - 0.8990: `thesis_tools/collab_evidence/16def2de/results/loo_L40/liars_bench_comparison_loo.csv`, row "persona axis, extended (zero-shot)", col auroc.
  - 0.9130: `results/persona_depth/depth.csv`, layer 36, source "zero-shot, Liars' Bench average", arm "extended axis (12v13)", metric auroc.
  - 0.682: the same comparison file, row "Mean Probe" (LB Table 6, p. 54).
- **Caveats:** placed beside each result ("What the gain is made of").
- **Collaborator's work:** "merge as joint work, we are one team".
- **Authors:** "3 authors, but keeps those empty for now, we will fill them out later".
- **Chapter 3:** required, keep it.
- **Length:** "No strict limit".
- **Title:** lead with the pipeline and the zero-shot win (working title in Part 3).
- **Contributions:** the owner selected the persona banks (kd, cap), the new eval datasets, the 70B NDIF pipeline, and diagnosis plus controls, and added "also the sarcasm banks, new hpkr dataset, how we found new layers that is peak, find more". From the offered list, they chose: calibration analysis, diversity-not-volume, verified reproducibility, random-direction nulls, sarcasm banks, layer-peak discovery, factor probes, leak discovery plus guard.
- **own100 / hf275:** "will you also mention the resut for our 100 and 275 persona dataset result". Report as **provisional, with caveats**, in its own Results section. The detection number 0.706 comes from `docs/thesis/persona-coordinates-70b-results.md` section 7.1 (5 datasets, L20, never repeated).
- **Chapters:** "go ahead with multiple chapters. but try to make them subchapter of main chapters." So the 6 template chapters are kept, and everything else becomes a section.
- **Economic analysis:** "ndif was free, lab hardware provided by the uni. so yeah, no direct cost."
- **Abstract:** "also no refernce in abstract".
- **Figures:** "Image, figure, caption in one line. But u can discuss in visuals later in the paragraph while citing them." Ch 4 figures chosen: evaluation protocols, 36-persona geometry, kd bank construction, HP-KR 2x2.
- **Failed methods:** "we will add both postiive negative failed method in the thesis report. we have tried inlp, dann but failed. iclude those please in writing somewhere".
  - "we have tried but files are messy. nit anywhere near repo or lab pc". Placement: **Ch 5 only**.
  - Later: "it has no source inside this repo. i just added it bcx i tested it files is elsewere." The Ch 4 sentence is sourced to that statement.
  - The other version's D3 (on-policy AISI probe) and D10 (INLP) are also reported in Ch 5 as failed methods.

### Method decisions (Chapter 4)
- **ai_audit:** "two few items and memory issues, dont mention ndif here".
  - `src/assistant_axis_ndif/eval_datasets.py:286`: "8 deceptive / 8 honest items -- too few to train on".
  - Memory: 4.23-13.57 GiB per row, computed from `run-coords/ai_audit/items.pt` token lengths (min 4,211, max 7,543) x 64 heads x T^2 x 4 bytes.
- **Pre-registration:** disclose the amendment. Layer 40 is **fixed in advance (our position)**, not "chosen by the evaluation loop" as the other version's Fig 4.1 says. Established from git:
  - `5d7b8560` (2026-09-10): `--layers target` = the middle layer, axis[40];
  - `50709657` (2026-09-15): `--layer 40` default;
  - Q6 sweep `9630f141` (2026-09-16);
  - pre-registration `f5d6904` (09-21 13:00), amended at `63aabb4b` (09-21 13:19);
  - Pipeline B moved to L40 at `06da2de2` (2026-09-20), before the kd personas (`97740753`).
- **Metrics:** AUROC is primary. At the operating point, label each pipeline's metric: Pipeline A TPR@1%FPR (`27_crossgen.py:275-279`); Pipeline B BA@1%FPR (`persona_ood.py:203-217`) plus specificity (`zero_shot_operating.csv`, column `specificity`).
- **The collaborator's missing pooling code:** the owner asked "can we proceede without my collaborator codes?". We proceed, with a disclosure.
  - `layer_sweep.py:155/198` calls `trace_layer_outputs(span_mask=...)`.
  - No committed `activations.py` accepts it: `git log --all -S span_mask` finds only `ea1ce52b`.
  - It is not on the lab PC (`C:\Users\user4`) or on Drive (`tamara:`).
- **Collaborator evidence:** "Thesis evidence only". Copied to `thesis_tools/collab_evidence/16def2de/` with `MANIFEST.md` md5s.
- **Other thesis version** (`full paper draft for this experiments/report.pdf`): adopt its design-alternatives framing.

### Facts established from files (use these; don't re-derive from prose)
- **Bank versions the runs used:**
  - paper16 = `9511ba8d` (all 26 cached response sets match);
  - kd = `3dfdb673` (v2; cached rollouts match v2 not v1 `5b43a7af`: 30/30 questions vs 25/30);
  - cap = `dc24e960` with `cap/questions/cap_questions_v3.jsonl`;
  - sarcasm pilot = `080a9794`.
  - Source: rollout-vs-definition comparison in `CH4_METHODOLOGY_PLAN.md` "Bank versions".
- **kd persona vectors are 80 layers:** `audit/p17/labcache/expLB/run-kd/vectors_s3/*.pt`, field `vector`, shape (80, 8192). 36/36 are md5-identical to the collaborator's `persona including knowlede denial/*.pt` (`148ddeb9`).
- **The ten KD roles are the owner's:** every commit to `kd/personas`, `kd/questions` and `kd/scripts/kd_personas.py` is by Suraiya Shanaz.
- **Leak:** 199/258 (77.13%) of kr-matched rows are in HP-KR's train split; 39/52 test rows; 59 survive the guard.
  - Source: `expLB-resplit/run-crossgen/splits.json` -> `splits["42"][...]["test_row_ids"]`, and the lab PC's `expLB/run-crossgen/kd11/splits.json` (same values).
  - F-113 section 5's "~81% (42/52)" does not reproduce.
- **Question provenance:**
  - kd first 20 = `src/assistant_axis_ndif/pc_questions_shared.jsonl` (20/20 identical);
  - sycophancy questions are verbatim from the paper PDF, pp. 9-10 (12/12);
  - the deception questions are our renderings of p. 9's topic bullets.
- **Persona geometry at L40:** recomputed from the kd `vectors_s3`, centred on 36. The means reproduce `numbers.tex:1166-1176`. Sincere ignorance sits level with the published deceptive personas on the denial axis; its nearest 2-D group is the default assistant (0.602).
- **Pooled size:** 13,602 rows (`loo_L40/loo_scores.npz` keys plus Alpaca 2,000) x 8192 x 4 B = **445.7 MB per layer** (float32 per `layer_sweep.py:124`).
- **Pipeline B constants that are not read from result files:** `paper/make_numbers.py:239` (`PAPER_PROBES`, transcribed from LB Tables 6-8) and `:495-505` (the check40 values). These are audit rows D2-024, D-063 and D2-025, all UNRESOLVED.
- **Follow-Up Probe:** 0.6360 against the published 0.634, from `results/persona_ood_followup_L20/liars_bench_comparison.csv` rows 3 and 9 (audit D-031).

## Writing conventions (used in Chapter 4; keep them)

**Voice**
- "We" for all three authors, in the active voice.
- Credit is given through the design, not the person ("Pipeline A", "Pipeline B", "the kd bank"). Authorship goes in Ch 1's Team Overview.

**Tense**
- The present tense for what the method does ("The judge filter keeps answers scored 3").
- The past tense for what was run or decided ("We amended it 19 minutes later").
- The present tense for what a figure or table shows.

**Spelling:** British: behaviour, centring, standardised, normalised, labelled, relabelling, modelling.

**Numbers**
- Copy the source's precision exactly; never re-round (0.9125, 77.13\%, 0.4630).
- Thousands separator `1{,}000` in LaTeX.
- Minus `$-0.08$`, ranges with `--` (en dash), percent `\%`.
- "layer 40" in prose; "L40" only in comments and tables.
- Contrasts written "4 v 6", "12 v 13".

**Names**
- Code identifiers in `\texttt{}`: `sycophancy\_mt`, `raw`, `pc:$b$@$k$`, `lr\_reg10`.
- Pipeline names `Pipeline~A` and `Pipeline~B`.
- Dataset abbreviations as in `GLOSSARY.md` section 6.

**Source comments** (never printed)
- `% src: <path> :: <line or field>` after every number or factual sentence, on the same line or directly after a float.
- Collaborator files: `collaborator <path> @16def2de`.
- Lab files: `lab PC C:/Users/user4/...`.
- The owner's word: `owner's statement, <date>`.
- `% interp: <which numbers>` for an interpretation; `% verified:` for a check; `% note:` for bookkeeping.
- **Never put a comment in the middle of a line**: everything after `%` is dropped.

**Missing values:** `\todo[inline]{MISSING: ...}`, always inline, since margin notes break inside tables. `todonotes` is loaded in `main.tex`.

**Citations**
- `\parencite[p.~N]{key}` or `\textcite[p.~N]{key}`, with the page checked in the PDF.
- Papers with no PDF are cited without a page.
- No citations in the abstract.
- MASK is Ren et al., not Cadenza Labs.

**Figures**
- One-line caption; discussed and `\ref`-cited in the text.
- Files go in `thesis/images/`. Generating scripts go in `thesis_tools/figures/`, previews in `thesis_tools/figures/preview/`, never in `thesis/`.

**Checks before handing over a chapter**
1. An independent verifier subagent, checking both sides.
2. `python thesis_tools/check_latex.py`.
3. A `\todo` count.

**Editing tip:** in Git Bash, never use `sed` replacements containing backslashes on `.tex` files. `\t`, `\b` and `\u` become control characters or case changes. Edit through a Python script file or the Edit tool.
