# GLOSSARY: terms, names and labels defined in the thesis

**Last updated:** 2026-09-25, after Chapter 4 was written and verified.

**What it covers:** every term Chapter 4 (`thesis/chapters/chapter_5.tex`) defines, quoted with the exact wording it uses. Later chapters should **use these names and `\ref` the label**, not re-explain the term.

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
| **natural-sarcasm set** | "600 Reddit comments, 300 sarcastic and 300 not" |
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
