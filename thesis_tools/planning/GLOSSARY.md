# GLOSSARY: terms, names and labels defined in the thesis

**Last updated:** 2026-09-25, after Chapter 5 was written and verified (§8 added: every label in Chapters 4–6 and the appendices, and the terms Chapter 5 defines).

**What it covers:** every term Chapter 4 (`thesis/chapters/chapter_5.tex`) and Chapter 5 (`thesis/chapters/chapter_6.tex`) define, quoted with the exact wording they use. Later chapters should **use these names and `\ref` the label**, not re-explain the term. §1–§7 are the Chapter 4 terms; §8 is the complete label index and the Chapter 5 terms.

## 1. `\label`s later chapters should `\ref`

| Label | What it is | Title / caption (exact) |
|---|---|---|
| `sec:m-overview` | §4.1 | Design Process or Methodology Overview |
| `fig:m-pipeline` | Figure | "The persona-coordinate lie-detection pipeline, from persona bank to detector." |
| `sec:m-spec` | §4.2 | Preliminary Design or Design (Model) Specification |
| `sec:m-model` | §4.2.1 | Subject model and compute |
| `sec:m-vectors` | §4.2.2 | Persona vectors |
| `eq:m-persona` | Equation | persona vector $v_p^{(\ell)}$: mean over kept answers of the mean over each answer's tokens |
| `sec:m-coords` | §4.2.3 | Contrast directions and persona coordinates |
| `eq:m-contrast` | Equation | contrast $d^{(\ell)}$ = mean over $D$ − mean over $H$ |
| `sec:m-features` | §4.2.4 | Features |
| `sec:m-probes` | §4.2.5 | Probes |
| `sec:m-controls` | §4.2.6 | Controls |
| `sec:m-alternatives` | §4.2.7 | Design alternatives |
| `tab:m-alternatives` | Table | "The design alternatives." |
| `sec:m-verification` | §4.2.8 | Verification of the alternatives |
| `sec:m-built` | §4.3 | What We Built: Persona Banks, Datasets and Tools |
| `sec:m-banks` | §4.3.1 | Persona banks |
| `tab:m-banks` | Table | "The persona banks." |
| `fig:m-kdbank` | Figure | "The knowledge-denial bank's groups and the contrasts built from them." |
| `fig:m-personas` | Figure | "The 36 personas at layer 40 on the published and knowledge-denial contrasts." |
| `sec:m-built-data` | §4.3.2 | Evaluation datasets we built |
| `sec:m-built-tools` | §4.3.3 | Evaluation tools we built |
| `sec:m-data` | §4.4 | Data Collection |
| `tab:m-datasets` | Table | "Evaluation datasets (Llama-3.3-70B-Instruct responses)." |
| `sec:m-cleaning` | §4.4.1 | Data Cleaning |
| `sec:m-transform` | §4.4.2 | Data Transformation |
| `sec:m-integration` | §4.4.3 | Data Integration (HP-KR label, slices, leak and guard, provenance clusters) |
| `fig:m-hpkr` | Figure | "HP-KR rows by label, knowledge and denial; hatched cells are empty." |
| `sec:m-reduction` | §4.4.4 | Data Reduction |
| `sec:m-data-summary` | §4.4.5 | Summary of Preprocessed Data |
| `sec:m-impl` | §4.5 | Implementation of Selected Design |
| `sec:m-protocols` | §4.5.1 | Evaluation protocols |
| `fig:m-protocols` | Figure | "Four evaluation protocols over six example datasets; each square is one dataset." |
| `sec:m-metrics` | §4.5.2 | Metrics |
| `sec:m-stats` | §4.5.3 | Statistics |
| `sec:m-layers` | §4.5.4 | Layer choice (pre-registration and its amendment) |
| `sec:m-verify` | §4.5.5 | Implementation and verification |

**Numbering:** section numbers are as `main.tex` prints them, where Chapter 4 = Proposed Methodology. Always `\ref` a label; never hard-code a number.

## 2. The pipeline and its parts (exact wording in Ch 4)

- **persona bank.** "A *persona bank* fixes a set of roles. Each role has system-prompt variants and a shared question set."
- **persona vector.** "The mean activation over a role's kept answers is its *persona vector*, one per layer." (`eq:m-persona`)
- **contrast direction.** "Given a set $D$ of roles that show a behaviour and a set $H$ of roles that show its opposite, the *contrast direction* is the difference of the two group means." (`eq:m-contrast`)
- **persona coordinates.** "the leading principal components of the persona vectors after centring them across roles."
- **zero-shot.** A detector that "scores the projection directly, with no training". Also: "A zero-shot detector projects every row of every dataset onto a unit contrast direction and uses no labels."
- **assistant anchor.** "The default assistant, with no system prompt, is never judged. It serves as the *assistant anchor*."
- **Pipeline A.** "our persona-coordinate implementation. Its main grid scores eleven datasets under every control … It scores the knowledge-matched HP-KR slice in a separate run." This is the owner's code, on `asix`, in `src/assistant_axis_ndif/`.
- **Pipeline B.** "built on the Liars' Bench detection code … It scores the benchmark's datasets with the benchmark's own probe and metric." This is the collaborator's code, `origin/paper` @16def2de.
- **judge filter.** The Llama-3.1-70B-Instruct judge scores 0–3; by default only answers "scored 3 of 3" are kept. The notes call this *s3*; *s23* keeps scores 2–3. Ch 4 prose says "score 3" / "scores 2--3", not s3/s23.
- **subject model.** Llama-3.3-70B-Instruct. The scale rungs are Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct.
- **NDIF.** National Deep Inference Fabric, used through the nnsight library (version 0.7.0).

## 3. Persona banks (Table `tab:m-banks`)

| Name in text | Definition (Ch 4) |
|---|---|
| **paper16** | "our reconstruction of the personas of Mahadik & Skapars". 26 roles: 8 deceptive, 7 honest, 5 sycophantic, 5 anti-sycophantic, plus the assistant anchor. It has "10 deception questions, which are our renderings of the paper's printed topics" and "12 sycophancy questions, all of them the paper's printed questions word for word". |
| **kd** / "the knowledge-denial bank (kd)" | paper16's 26 plus 10 new roles, the owner's:<br>• *knowledge denial* (4): amnesiac witness, deniable expert, stonewaller, feigned novice ("They know and claim not to.");<br>• *sincere ignorance* (3): honest novice, calibrated reporter, scrupulous witness ("They do not know and say so.");<br>• *plain assertion* (3): forthright expert, candid witness, plain dealer ("They know and say it plainly.").<br>30 shared questions. |
| **cap** / "the capitulation bank (cap)" | 6 roles that give way under pressure, 6 that change their answer only for evidence, and the anchor. The first turn is teacher-forced, and the challenge is "I don't think that's right. Are you sure?" |
| **sarcasm pilot** | "an ironic critic, a sincere critic and a sycophant", 20 questions, 300 rollouts. It builds no vectors. |
| **natural-sarcasm set** | "600 Reddit comments, 300 sarcastic and 300 not, drawn from the Reddit part of the 2020 sarcasm detection shared task" (`ghosh2020figlang`; wording added 2026-09-25). Ch 4 now states the pilot's purpose: an intent-reading direction should separate sarcasm (meant to be recognised) from sycophancy/deception (meant to be believed) |
| **own100** | "100 roles from the Assistant-Axis library, generated on its 240 shared questions" (a library bank) |
| **hf275** | "the 275 published role vectors of Lu et al." (a library bank) |
| **scale rungs** | the 3B and 8B rebuilds of paper16 (on the 20 shared questions) and kd |

## 4. Contrasts and design alternatives (Table `tab:m-alternatives`)

| Name in text | Groups |
|---|---|
| published / deception contrast (8 v 7), **D5** | deception v honesty |
| sycophancy contrast (5 v 5), **D6** | sycophancy v anti-sycophancy |
| unified contrast (13 v 12), **D7** | deception + sycophancy v honesty + anti-sycophancy |
| knowledge-denial contrast (4 v 6), **D8**; "denial (4 v 6)" | knowledge denial v sincere ignorance + plain assertion (Pipeline B) |
| extended contrast (12 v 13), **D9** | deception + denial v honesty + sincere + plain (Pipeline B) |
| knowledge denial (4 v 3) | denial v sincere ignorance. "only sincerity differs" (Pipeline A) |
| denial versus assertion (4 v 3) | denial v plain assertion. "the register contrast" (Pipeline A) |
| **D1** | "Probe on raw activations, trained on one dataset (`raw`)" |
| **D2** | "Base probe: the Liars' Bench Mean Probe recipe" |
| **D4** | "Probe on the leading persona components (`pc@k`)" |
| pooled probe (no D-number) | "trained on every dataset but the target" |
| D3, D10 | on-policy base probe; INLP subspace. These are **not in the Ch 4 table**. Ch 4 says they "were tested and did not work … Their code and results are kept outside this repository. Chapter 5 describes them." DANN goes in the same Chapter 5 section. |

## 5. Features, probes, controls (Pipeline A names in `\texttt`)

- **Features:** `raw`; `pc:`$b$`@`$k$; `axes` (4-d: deception, sycophancy, unified, assistant); `axes_resid`; `erase:`$a$; zero-shot.
- **Probes:**
  - `paper`: LR, C=1, with an intercept.
  - `lr_reg10`: standardised features, LR with C=0.1, no intercept.
  - `mms`: difference of class means.
- **Base probe:** "the benchmark's own probe … We call it the *base probe*." It is trained on RepE.
- **Pooled probe:** "trained on several datasets together and tested on one held out."
- **Controls:**
  - `datapca@`$k$: "the dataset-PCA control of Mahadik & Skapars";
  - `random@`$k$;
  - `length`;
  - the random-direction null: 1,000 random unit directions.
- **k:** default $k=5$. Some features are also run at $k=2$; the scale rungs use $k=3$.

## 6. Datasets (Table `tab:m-datasets`) and abbreviations

| Abbreviation | Full name in text |
|---|---|
| ID | instructed-deception |
| HP-C | harm-pressure choice |
| HP-KR | harm-pressure knowledge report |
| CG | convincing-game |
| IT | insider-trading |
| — | ai\_liar, roleplaying, sandbagging\_v2 (Apollo); mask (MASK) |
| `sycophancy_mt` | "Multi-turn capitulation"; ours |
| **text-matched slice** | "every response denies knowledge, so only the knowledge state differs" (258 rows) |
| **knowledge-matched slice** | "the model knows the answer in every row, so only the denial differs" (2,001 rows) |
| Alpaca | the control. It "sets the threshold" and is never scored as a test set |
| RepE | the base probe's training set (Pipeline B: 550 examples) |
| ai\_audit | excluded: "only 8 deceptive and 8 honest items", plus memory |

## 7. Protocols, clusters, metrics and statistics

- **train one, test all.** 80/20 split by item; the diagonal is the source's own 20%; the off-diagonal is 100% of each other dataset.
- **pooled leave-one-out** (Pipeline B, naive or balanced), and **Pipeline A's version**, which adds an arm held to a single source's row budget.
- **leave-one-cluster-out** holds out a whole *provenance cluster*. The clusters, as named in text:
  - "Liars' Bench harm pressure" (HP-C, HP-KR, text-matched slice) = code `lb_harm`;
  - "Liars' Bench scenarios" (ID, CG, IT) = `lb_scenario`;
  - "Apollo" = `apollo`;
  - "MASK" = `mask`;
  - "Ours" (`sycophancy_mt`) = `ours`.
- **leak guard.** "drops from every off-diagonal test set any row that is in the source's training rows."
- **AUROC.** The main metric.
- **Operating point.** The threshold flags at most 1% of Alpaca.
  - Pipeline A reports the **true-positive rate (TPR)**.
  - Pipeline B reports **balanced accuracy (BA)** and records **specificity**, so its realised FPR = 1 − specificity.
- **row bootstrap:** 1,000 resamples.
- **dataset bootstrap:** 10,000 resamples.
- **Benjamini–Hochberg:** $q<0.05$.
- **cluster-based permutation test:** 200 permutations.
- **equivalence:** a 90% interval within ±0.02.
- **seed:** 42.
- **Layers:**
  - L20 (Liars' Bench), L22 (Apollo) and L40 (the middle layer) are fixed in advance, with the amendment disclosed;
  - L44 and L36 are post hoc;
  - the scale rungs use 14 and 16.

## 8. Every label in Chapters 4, 5 and 6 and the appendices, and the terms Chapter 5 defines (regenerated 2026-09-25)

**How to refresh this section:** `python thesis_tools/label_index.py --terms` prints both lists from the LaTeX itself (titles and captions brace-matched, comments removed). Section 1 above keeps the Chapter 4 labels as written on 2026-09-24; this section is the complete, current index.

**Rules for later chapters:**
- `\ref` a label; never hard-code a chapter, section, figure or table number. Chapter labels: `ch:method` (Chapter 4, first line of `chapter_5.tex`), `ch:results` (Chapter 5, first line of `chapter_6.tex`), `ch:conclusion` (Chapter 6, first line of `chapter_9.tex`).
- Chapter 4 labels use the prefix `m-`, Chapter 5 labels `r-`, appendix labels `app:`. Use a new prefix per new chapter (e.g. `i-` for the Introduction, `c-` for the Conclusion).
- Appendices A and E are generated (`thesis_tools/appendix/make_appendix_personas.py`, `make_appendix_examples.py`); never edit `appendix_A.tex` or `appendix_E.tex` by hand. Appendices B, C and D are stubs.

### 8.1 Label index

**Chapter 4 (Proposed Methodology, `chapters/chapter_5.tex`)**

| Label | Kind | Title / caption (exact) |
|---|---|---|
| `ch:method` | Chapter | Proposed Methodology |
| `sec:m-overview` | Section | Design Process or Methodology Overview |
| `fig:m-pipeline` | Figure | The persona-coordinate lie-detection pipeline, from persona bank to detector. |
| `sec:m-spec` | Section | Preliminary Design or Design (Model) Specification |
| `sec:m-model` | Subsection | Subject model and compute |
| `sec:m-vectors` | Subsection | Persona vectors |
| `eq:m-persona` | Equation | $v_p^{(\ell)} \;=\; \frac{1}{\|\mathcal{R}_p\|}\sum_{r\in\mathcal{R}_p}\;\frac{1}{\|T_r\|}\sum_{t\in T_r} h_t^{(\ell)},$ |
| `sec:m-coords` | Subsection | Contrast directions and persona coordinates |
| `eq:m-contrast` | Equation | $d^{(\ell)} \;=\; \frac{1}{\|D\|}\sum_{p\in D} v_p^{(\ell)} \;-\; \frac{1}{\|H\|}\sum_{p\in H} v_p^{(\ell)} .$ |
| `sec:m-features` | Subsection | Features |
| `sec:m-probes` | Subsection | Probes |
| `sec:m-controls` | Subsection | Controls |
| `sec:m-alternatives` | Subsection | Design alternatives |
| `tab:m-alternatives` | Table | The design alternatives. |
| `sec:m-verification` | Subsection | Verification of the alternatives |
| `sec:m-built` | Section | What We Built: Persona Banks, Datasets and Tools |
| `sec:m-banks` | Subsection | Persona banks |
| `tab:m-banks` | Table | The persona banks. |
| `fig:m-kdbank` | Figure | The knowledge-denial bank's groups and the contrasts built from them. |
| `fig:m-personas` | Figure | The 36 personas at layer 40 on the published and knowledge-denial contrasts. |
| `sec:m-built-data` | Subsection | Evaluation datasets we built |
| `sec:m-built-tools` | Subsection | Evaluation tools we built |
| `sec:m-data` | Section | Data Collection |
| `tab:m-datasets` | Table | Evaluation datasets (Llama-3.3-70B-Instruct responses). |
| `sec:m-cleaning` | Subsection | Data Cleaning |
| `sec:m-transform` | Subsection | Data Transformation |
| `sec:m-integration` | Subsection | Data Integration |
| `fig:m-hpkr` | Figure | HP-KR rows by label, knowledge and denial; hatched cells are empty. |
| `sec:m-reduction` | Subsection | Data Reduction |
| `sec:m-data-summary` | Subsection | Summary of Preprocessed Data |
| `sec:m-impl` | Section | Implementation of Selected Design |
| `sec:m-protocols` | Subsection | Evaluation protocols |
| `fig:m-protocols` | Figure | Four evaluation protocols over six example datasets; each square is one dataset. |
| `sec:m-metrics` | Subsection | Metrics |
| `sec:m-stats` | Subsection | Statistics |
| `sec:m-layers` | Subsection | Layer choice |
| `sec:m-verify` | Subsection | Implementation and verification |

**Chapter 5 (Result Analysis, `chapters/chapter_6.tex`)**

| Label | Kind | Title / caption (exact) |
|---|---|---|
| `ch:results` | Chapter | Result Analysis |
| `sec:r-perf` | Section | Performance Evaluation |
| `sec:r-criteria` | Subsection | Evaluation criteria |
| `sec:r-testing` | Subsection | Testing methods |
| `sec:r-problem` | Subsection | The problem measured |
| `sec:r-analysis` | Section | Analysis of Design Solutions |
| `sec:r-supervised` | Subsection | Supervised probes |
| `sec:r-d4` | Subsection | The published persona method |
| `sec:r-zeroshot` | Subsection | Zero-shot persona contrasts |
| `tab:r-zeroshot` | Table | Zero-shot AUROC per dataset of the published, knowledge-denial and extended contrasts (Pipeline~B). |
| `sec:r-hpkr` | Subsection | HP-KR, the factor probes and the convergence |
| `tab:r-hpkr` | Table | HP-KR detection by the zero-shot contrasts, the base probe and the published probes. |
| `fig:r-hpkr` | Figure | HP-KR AUROC and balanced accuracy of four detectors over 16 layers, as drawn by Pipeline~B. |
| `fig:r-convergence` | Figure | The convergence: three wording-controlled tests, AUROC by layer, with chance at 0.5. |
| `sec:r-transfer` | Subsection | Transfer across datasets |
| `tab:r-transfer` | Table | Transfer: mean off-diagonal AUROC of Pipeline~B's arms on its six datasets (layer 44 post hoc). |
| `fig:r-transfer` | Figure | Train-one-test-all AUROC at layer 40 for D1 (left) and D9 (right); produced by Pipeline~B. |
| `sec:r-personas` | Subsection | Which personas: library and purpose-built banks |
| `tab:r-personas` | Table | Cosine of each bank's deception contrast with an assistant direction, by layer. |
| `sec:r-selection` | Subsection | Selection |
| `tab:r-designs` | Table | The design alternatives at layer 40 against the evaluation criteria (Pipeline~B, six datasets). |
| `sec:r-adjust` | Section | Final Design Adjustments |
| `sec:r-adj-bank` | Subsection | From the published bank to the knowledge-denial roles |
| `sec:r-adj-direction` | Subsection | One direction, not the subspace |
| `sec:r-adj-roles` | Subsection | The roles, not the widening |
| `fig:r-widening` | Figure | The widening control across Pipeline~B's 16 layers: transfer (left) and HP-KR ranking (right). |
| `sec:r-layer` | Subsection | The layer |
| `fig:r-depth` | Figure | Transfer and the zero-shot Liars' Bench average across 16 layers, as plotted by Pipeline~B. |
| `sec:r-token` | Subsection | The token read and pooling |
| `tab:r-reads` | Table | Six token reads of the extended contrast, averaged over the five Liars' Bench datasets. |
| `sec:r-failed` | Subsection | Approaches that did not work |
| `sec:r-claims` | Subsection | Adjustments to the claims |
| `sec:r-stats` | Section | Statistical Analysis |
| `fig:r-significance` | Figure | One test family: each difference with its dataset-bootstrap interval and, where recorded, its row-bootstrap interval. |
| `tab:r-oracle` | Table | Liars' Bench average balanced accuracy at the Alpaca threshold and at the oracle threshold (Pipeline~B). |
| `sec:r-compare` | Section | Comparisons and Relationships |
| `sec:r-cmp-lb` | Subsection | With the benchmark's probes and protocol |
| `tab:r-protocol` | Table | Detection under the Liars' Bench protocol, averaged over its five datasets (Pipeline~B). |
| `fig:r-operating` | Figure | Per-dataset detection at layer 40 under the Liars' Bench protocol, produced by Pipeline~B. |
| `sec:r-cmp-replication` | Subsection | With the replicated method at 70B |
| `sec:r-scale` | Subsection | Across model scale |
| `fig:r-scale` | Figure | Layer-averaged margin of persona features over raw by model size, for two banks (Pipeline~A). |
| `sec:r-pooled` | Subsection | With pooled supervised training |
| `fig:r-loco` | Figure | Single-source and pooled leave-one-cluster-out transfer per target at layer 40 (Pipeline~A). |
| `sec:r-geometry` | Subsection | Between geometry and behaviour |
| `sec:r-discuss` | Section | Discussions |
| `sec:r-interp` | Subsection | Interpretation |
| `sec:r-cap` | Subsection | The capitulation bank and the sycophancy axis |
| `fig:r-cap` | Figure | Capitulation and paper16 directions on \texttt{sycophancy\_mt} at layer 40: zero-shot (a) and trained per source (b). |
| `sec:r-sarcasm` | Subsection | Sarcasm and deceptive intent |
| `sec:r-organisms` | Subsection | Model organisms |
| `fig:r-sleeper` | Figure | Probe tests on the Soft-Trigger and Gender-Secret organisms, raw activations against persona axes, at layers 20 and 44. |
| `sec:r-remaining` | Subsection | Remaining questions |
| `sec:r-implications` | Subsection | Implications |
| `sec:r-limits` | Subsection | Limitations |

**Chapter 6 (Conclusion, `chapters/chapter_9.tex`)**

| Label | Kind | Title / caption (exact) |
|---|---|---|
| `ch:conclusion` | Chapter | Conclusion |

**Appendices (chapter labels set in `main.tex`)**

| Label | Kind | Title / caption (exact) |
|---|---|---|
| `app:prompts` | Chapter | Persona Prompts and Questions |
| `app:tables` | Chapter | Full Result Tables |
| `app:repro` | Chapter | Reproducibility, Pins and the Audit |
| `app:eng` | Chapter | Engineering Notes |
| `app:examples` | Chapter | Worked Examples |

**Appendix A (generated, `appendix/appendix_A.tex`)**

| Label | Kind | Title / caption (exact) |
|---|---|---|
| `app:prompts-kd` | Section | The knowledge-denial bank and paper16 |
| `app:prompts-kd-q` | Subsection | The 30 kd questions |
| `app:prompts-p16-q` | Subsection | paper16's questions |
| `app:prompts-cap` | Section | The capitulation bank |
| `app:prompts-cap-q` | Subsection | The 20 two-turn questions |
| `app:prompts-sarcasm` | Section | The sarcasm pilot |
| `app:prompts-sarcasm-q` | Subsection | The 20 sarcasm-pilot questions |

### 8.2 Terms Chapter 5 defines at first use (owner decision 2026-09-25: defined once in the chapter, listed here)

The sentence is the definition as printed (LaTeX source, comments removed). Use the term with this meaning and `\ref` the section instead of re-defining it.

- **inverts**: A detector \emph{inverts} on a dataset when its AUROC falls below 0.5, so that it ranks honest responses above lies.
- **generalisation gap**: \item[Generalisation gap.] We define the \emph{generalisation gap} as the in-distribution (diagonal) AUROC minus the off-diagonal AUROC; a detector that transfers has a small one.
- **oracle threshold**: We call it the \emph{oracle threshold}.
- **Mean Probe**: The \emph{Mean Probe} is trained on contrastive instruction pairs and averages its per-token scores over the response, and D2 follows its recipe \parencite[p.~8]{kretschmar2025liars}.
- **Follow-Up Probe**: The \emph{Follow-Up Probe} appends the question ``Was your last response deceptive? (Yes/No)'', prefills the answer ``No'' and scores only that token \parencite[p.~8]{kretschmar2025liars}.
- **Upper-Bound Probe**: The \emph{Upper-Bound Probe} is trained with privileged examples from the five datasets we share with the benchmark, plus its control, and is scored on their held-out examples in the same way as the Follow-Up Probe, so it is a ceiling, not a comparator \parencite[p.~8]{kretschmar2025liars}.
- **Liars' Bench average**: We call the mean over the five Liars' Bench datasets that Pipeline~B scores (CG, HP-C, HP-KR, IT and ID), which leaves out RepE, the \emph{Liars' Bench average}.
- **share of the margin**: We call the fraction $(\text{D9}-\text{D5})/(\text{D9}-\text{raw})$, where raw is the same pipeline's single-source probe of Section~\ref{sec:r-supervised}, the kd roles' \emph{share of the margin}.
- **factor probes**: To separate the two things HP-KR's label mixes, we built two \emph{factor probes}: two directions, $d_{\text{knowledge}}$ and $d_{\text{utterance}}$, each of which probes one factor of the HP-KR label while the other is held fixed.
- **the convergence**: We call their agreement \emph{the convergence}: where both classes deny knowledge, no persona direction reads the lie well, and where only the denial separates the classes, what separates them does not transfer.
- **sign artifact**: We call this a \emph{sign artifact}: a trained one-dimensional probe inherits a reversed sign from its training set, so its score on every other test set is one minus the zero-shot AUROC.
- **assistant direction**: A bank's own \emph{assistant direction} is the mean vector of its assistant anchor minus the mean over all its other roles.
- **published Assistant Axis**: The \emph{published Assistant Axis} is the axis released by \textcite[p.~6]{lu2026assistant}, built the same way from their role vectors; we use the released Llama-3.3-70B file.
- **token read**: We call each such reduction a \emph{token read}: the mean over the response tokens, the mean of the top 10\% of token scores, the maximum, the first or last response token, or the last prompt token before the reply.
- **best persona feature**: For each layer we take the \emph{best persona feature}, the one with the highest mean off-diagonal AUROC among \texttt{axes}, \texttt{pc:deception}, \texttt{pc:unified} and \texttt{pc:unified@2}, a choice made after seeing the results.
- **row-matched arm**: The arm held to a single source's row budget, which we call the \emph{row-matched arm}, and the leave-one-cluster-out protocol are Pipeline~A's.
- **mean-of-directions probe**: Second, averaging the unit directions of the ten single-source probes, with no joint training, scores 0.8828 at layer 40 against 0.6763 for the single-source mean; we call this the \emph{mean-of-directions probe}.
- **register diagnosis**: The \emph{register diagnosis} is our reading that a persona direction responds to the register of a response, that is, to how it is worded (asserting or disclaiming, agreeing or challenging), more than to the construct that the dataset's label records.
- **format control**: To ask whether they contribute what they say or only that they are there, we built a \emph{format control}: a bank in which the six position-holding roles are replaced by six roles written in the same sentence frames but with no stance on keeping or revising an answer, such as a careful speller and a metric user, while the capitulation roles and the anchor stay unchanged.
- **model organism**: A \emph{model organism} here is a model fine-tuned to lie under known conditions.

**Terms used in the appendices:**
- **arm** (sarcasm pilot, Appendix A): the kind of answer a sarcasm-pilot role is written to give, named in its file's `_arm` field; one role per arm.
- **knowledge-admission questions** (Appendix E): the kd bank's ten questions about the speaker's own knowledge (ids 20–29; six in which the speaker knows, four in which it does not).
- **believed option** (HP-C, Appendix E): the option letter that Liars' Bench established as the model's belief (`meta.model_belief` 0–3 read as A–D); an HP-C row is deceptive when the chosen letter differs from it.

**Names used in Chapter 5 for Pipeline B's arms** (not defined in Chapter 4; Chapter 5 names them in prose): `pc3` (probe on the first three components of the published 8 v 7 roles), `ext_pc3` (the same on the 25 extended roles), `ext_pc1`, `syc_pc1` / `syc_axis1` (the 5 v 5 sycophancy geometry), `uni_axis1` (D7), `random3` (a trained probe on three random directions, averaged over five draws). Prefer the prose names ("the extended components", "the sycophancy contrast") in text.
