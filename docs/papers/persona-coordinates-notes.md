# Do Linear Probes Generalize Better in Persona Coordinates?

**Source:** Prasad Mahadik, Adrians Skapars. *Do Linear Probes Generalize Better in
Persona Coordinates?* arXiv:2605.09391v2 [cs.AI], 15 May 2026. Preprint.
**Affiliations (as printed):** Prasad Mahadik — Independent Researcher
(`mahadikprasad15@gmail.com`); Adrians Skapars — University of Manchester
(`adrians.skapars@postgrad.manchester.ac.uk`).
**No code/data URL is given in the paper.**

This note covers only this paper. Section / figure / table / equation numbers below
refer to this paper. Numbers I read off bar charts / heatmaps (Figures 2–6, 10) are
labelled "(read from Figure N)" and are approximate to the printed precision.

---

## 1. What the paper is and its thesis

This is a **white-box monitoring / linear-probing method paper**. It is not a new
benchmark and not a training method. It proposes a way to build **low-dimensional
"persona axes"** for a target behaviour (deception, sycophancy) from a small curated
set of contrastive persona prompts, and then asks whether:

1. those axes form a clean geometric separation of harmful vs. harmless personas;
2. the axes can be used *directly* (unsupervised, zero-shot) as lie / sycophancy
   classifiers on unseen datasets;
3. projecting activations onto the leading persona principal components (PCs) and
   training a probe on those few coordinates generalizes **out-of-distribution
   (across datasets)** better than a probe trained on the raw high-dimensional
   activation.

**Motivation (Abstract, §1).** Text-only behavioural monitoring is argued to be
insufficient because models can strategically deceive or sandbag and change behaviour
when they detect evaluation; this motivates reading model internals with linear probes.
But probes trained on one dataset often fail to transfer, "not because the observed
model lacks behavior-relevant representations, but because the probe has learned to use
spurious and dataset-specific correlations." The paper's hypothesis (end of §2,
italicised in the paper): *"behavior may transfer through latent persona or agent
states, and probes that target those states may generalize better than probes trained
directly on high-dimensional dataset activations."*

**Intellectual lineage (§2).** Simulator / simulacra view of LLMs (Janus 2022;
Andreas 2022); personas as a model of truthfulness (Joshi et al. 2024); the Persona
Selection Model (Marks et al. 2026); and most directly **the Assistant Axis (Lu et al.
2026, arXiv:2601.10387)**, whose persona-prompt framework, residualization step, and
middle-layer / mean-pooled activation recipe this paper borrows and re-points at
behaviour-specific contrasts. Also builds on mean-difference truth probes (Marks &
Tegmark 2023), the Linear Representation Hypothesis (Park et al. 2024), unsupervised
truth directions (Burns et al. 2023), the Truthfulness Spectrum Hypothesis (Ying et
al. 2026), and deception probes (Goldowsky-Dill et al. 2025, arXiv:2502.03407).

**Stated contributions (§1):**

- *Persona-state probing*: construct persona-conditioned activation directions that
  approximate the "persona-state" a response was generated from, and probe those
  directions instead of raw activations labelled by output behaviour.
- Show the means of persona-conditioned activations give a clean geometric structure
  for deception and sycophancy, with leading PCs separating honest/deceptive and
  non-sycophantic/sycophantic personas.
- Show persona axes derived from **unlabelled** data transfer non-trivially to unseen
  deception and sycophancy datasets as zero-shot classifiers.
- Show probes trained on the projections onto persona PCs generalize better across
  datasets than raw-activation probes.
- Show a **unified axis** built from multiple harmful + harmless personas improves
  linear-probe generalization across multiple behaviours at once.

---

## 2. Core method — "persona-state probing"

The method has three stages: (A) build one activation vector per persona; (B) center
those vectors and run PCA to get persona axes / PCs; (C) use the geometry in one of
three ways (contrast direction zero-shot, PC1 zero-shot, or top-k PC projections as
probe features). The paper *does* contain formal equations — **four numbered
equations, (1)–(4)** — reproduced below.

### 2.1 Persona-vector construction (procedural — this is the part to get exactly right)

This is the piece the downstream large-scale replication depends on, so it is spelled
out in full detail here. Sources: §3 "Constructing Persona Vectors", §4 "Persona
Selection" / "Deception Axis" / "Sycophancy Axis", Figure 1, and Appendices A–C.

**Step 1 — choose the persona sets.** For each target behaviour, the authors
hand-pick a small set of personas from the Assistant-Axis persona-prompt framework:
a group expected to exhibit the harmful behaviour, an opposing group expected to
resist it, plus one **default assistant-like anchor persona**.

- **Deception axis (Appendix A, Tables 2–3):**
  - 8 **deceptive** personas: `handler`, `broker`, `impression_manager`,
    `coverup_operator`, `gatekeeper`, `spin_doctor`, `plausible_denier`,
    `narrative_manager`. All are framed around *omission* — "controls timing,
    emphasis, and selective disclosure", "omits damaging context to preserve surface
    impression", "gives defensible half-disclosures", etc.
  - 7 **honest** personas: `ombudsman`, `transparency_officer`, `disclosure_counsel`,
    `whistleblower`, `documentarian`, `fact_checker`, `archivist` ("completeness as
    fairness", "surfaces hidden relevant facts despite cost", "treats missing context
    as misleading").
  - + 1 default assistant anchor. Total **16 personas**.
- **Sycophancy axis (Appendix B, Tables 4–5):**
  - 5 **sycophantic** personas: `yes_man`, `flatterer`, `people_pleaser`,
    `cheerleader`, `enabler` ("agrees reflexively", "prioritizes social harmony and
    validation over accuracy", "reinforces the user's framing and beliefs").
  - 5 **non-sycophantic** personas: `critic`, `skeptic`, `contrarian`,
    `blunt_advisor`, `tough_love_mentor` ("prioritizes evaluation and honest
    assessment over agreement", "gives direct feedback even when socially
    uncomfortable").
  - + 1 default anchor (listed as `default` in Table 5). Total **11 personas**.

**Step 2 — build the prompt bank.** Each persona gets **2 instruction variants**
(system-prompt phrasings, "in the same style as the Assistant Axis paper"). There is a
**shared question pool** used for every persona:
- deception: **10** shared, omission-focused questions (`project handoff`,
  `leadership brief`, `decision support`, `risk disclosure`, `stakeholder onboarding`,
  `damaging summary`, `reference context`, `relationship transfer`,
  `postmortem completeness`, `compliance disclosure threshold`).
- sycophancy: **12** shared questions (Appendix B lists all 12 verbatim — e.g. "I got
  harsh feedback on my work. Can you explain why the reviewer is probably wrong and
  make me feel better about it?").

**Step 3 — generate rollouts.** The prompt template is
`instruction/role + question -> generated output`. For every
(persona × instruction variant × shared question) the **target model itself** generates
a response. That is **2 × 10 = 20 rollouts per deception persona** and
**2 × 12 = 24 rollouts per sycophancy persona**. There is **no LLM-judge filtering**
of whether the persona was successfully expressed (this is a difference from the
Assistant Axis pipeline — see §7). Every rollout is kept.

**Step 4 — pool to a per-(persona, question) vector.** Run each rollout through the
model, take hidden states at the analysis layer, and **mean-pool over the output
(generated response) tokens only** — Figure 1 panel 2 states "only output tokens are
pooled". Figure 1 writes this as

    x_{p,q} = (1 / |T_out|) * sum_{t in T_out} h_t^(14)      -> "one vector per (persona, question)"

where `T_out` is the set of output-token positions and `h_t^(14)` is the layer-14
hidden state at token `t`. (The superscript 14 is the analysis layer for the 3B
model.)

**Step 5 — average across questions to one persona vector.** Figure 1 panel 3:

    v_p = (1 / |Q|) * sum_{q in Q} x_{p,q}                    -> "one persona vector per persona"

The two instruction variants are also averaged in (the §4 text: "Averaging over
instructions and questions yields one activation vector per persona per layer").
Equation (1) in §3 states the same thing in one line, over the rollout set `Q_p`:

    v_p^(l) = (1 / |Q_p|) * sum_{q in Q_p} pool( h^(l)(p, q, y_{p,q}) )          ... (1)

Symbols:
- `p` — a persona; `l` — a transformer layer.
- `Q_p` — the set of rollouts for persona `p` (all instruction × question combos).
- `h^(l)(p, q, y_{p,q})` — hidden states at layer `l` for rollout `q`, where
  `y_{p,q}` is the generated response.
- `pool(·)` — **mean pooling over output tokens only**.
- `v_p^(l)` — the persona vector for `p` at layer `l` (one vector per persona per
  layer; dimensionality = model hidden size).

**Step 6 — residualize (center) across personas.** At each layer, subtract the mean
persona vector over the whole persona set `P` — the paper says this "follow[s] the
residualization step used in the Assistant Axis methodology (Lu et al., 2026)":

    v~_p^(l) = v_p^(l) - (1 / |P|) * sum_{p' in P} v_{p'}^(l)                    ... (2)

Only mean-centering is described; the paper does not say whether per-dimension
variance scaling is also applied (an ambiguity inherited from the Assistant Axis
paper).

**Step 7 — PCA for the persona axes.** Run PCA on the centered persona vectors
`{v~_p^(l)}` **at each layer**. Call the resulting principal directions
`u_1^(l), u_2^(l), u_3^(l), ...`, ordered by explained variance; "PC1" = `u_1^(l)`,
etc. For any vector `z^(l)` (a persona vector, or a pooled activation from an
evaluation sample), its PC-k score is the scalar projection:

    PC_k(z^(l)) = < z^(l), u_k^(l) >                                            ... (3)

Note the persona set has only 16 (deception) or 11 (sycophancy) points, so PCA in the
~3072-dim (3B) / ~4096-dim (8B) hidden space yields at most 15 / 10 non-zero
components; PC2/PC3 are estimated from very few samples. The paper does not comment on
this.

### 2.2 The contrastive direction (Equation 4)

Separately from PCA, the paper defines a **class-mean contrast direction** at layer
`l`:

    d^(l) = (1 / |D|) * sum_{p in D} v~_p^(l)  -  (1 / |H|) * sum_{p in H} v~_p^(l)   ... (4)

where `D` is the deceptive / sycophantic persona set and `H` is the honest /
non-sycophantic set. It uses the **centered** vectors `v~_p^(l)` from Eq (2). A
sample with pooled activation `z^(l)` is scored by

    s(z^(l)) = < z^(l), d^(l) >.

This is a two-class contrast (harmful-persona mean minus harmless-persona mean). The
default anchor persona is used in the geometry plots and (presumably) in the Eq (2)
mean, but the §3 text describes `d^(l)` as being between the deceptive and honest
groups only; whether the anchor is in `D`, `H`, or neither is not stated explicitly.

### 2.3 The three ways the geometry is used (§3)

1. **Contrastive direction, zero-shot.** Score evaluation samples by
   `s(z^(l)) = <z^(l), d^(l)>` (Eq 4) and compute AUROC against the true lie /
   sycophancy label. No training on evaluation data.
2. **PC1, zero-shot.** Score samples by `PC_1(z^(l)) = <z^(l), u_1^(l)>` (Eq 3).
   Higher PCs (`PC_2`, `PC_3`) are also tried as zero-shot scorers.
3. **PC features for a learned probe.** Project each sample's pooled activation onto
   the first few persona PCs (the §4 text says "the first 3 persona PCs"; the
   method-comparison figures actually use k = 1 for deception, k = 3 for sycophancy,
   k = 2 for the unified axis — see §8 ambiguities), then train a lightweight logistic
   regression on those k coordinates. This probe is trained on a **source** dataset
   and evaluated on a **target** dataset to measure cross-dataset transfer.

### 2.4 Unified harmful axis (§4 "Unified Axis", Appendix C)

Combine the deceptive and sycophantic personas into one set and build a single
harmful-vs-harmless axis. Uses a **shared balanced question set of 10 + 10 questions**:
the deception axis contributes its 10 questions, the sycophancy axis contributes "its
first 10 questions after shared-set balancing" (Appendix C). Same Eq (1)–(4)
machinery; PCA is run over the merged persona set. Used only for the learned-probe
experiments in Figure 6 / Appendix Figure 9, with k = 2.

---

## 3. Equations — summary

The paper has **four numbered equations**, all in §3:

| Eq | What it is | Plain-text form |
|---|---|---|
| (1) | Persona vector | `v_p^(l) = (1/|Q_p|) Σ_{q∈Q_p} pool(h^(l)(p,q,y_{p,q}))`, `pool` = mean over output tokens |
| (2) | Residualization / centering | `v~_p^(l) = v_p^(l) − (1/|P|) Σ_{p'∈P} v_{p'}^(l)` |
| (3) | PC-k score (scalar projection) | `PC_k(z^(l)) = <z^(l), u_k^(l)>`, `u_k^(l)` = k-th PCA direction of `{v~_p^(l)}` |
| (4) | Class-mean contrast direction | `d^(l) = (1/|D|) Σ_{p∈D} v~_p^(l) − (1/|H|) Σ_{p∈H} v~_p^(l)`; sample score `s(z^(l)) = <z^(l), d^(l)>` |

Figure 1 additionally shows the two intermediate forms `x_{p,q}` (per-question,
per-persona pooled vector) and `v_p` (per-persona average of `x_{p,q}`), which together
are just Eq (1) written in two steps.

There is no equation for the probe itself — it is plain scikit-learn
`LogisticRegression` on the k projected coordinates.

---

## 4. Exact experimental setup

### 4.1 Models

- **Main results:** **Llama 3.2-3B** (instruct-tuned is implied by the
  persona-conditioning; the paper writes "Llama 3.2-3B").
- **Replication:** **"Llama 3-8B"** (the paper writes "Llama 3-8B" / "Llama 3 8B";
  it does *not* say "Llama 3.1-8B" — the exact checkpoint is unstated). 8B results are
  in Appendices E and F.
- For each model, the **same model generates both** the persona rollouts and the
  evaluation-dataset rollouts, and activations are mean-pooled over output tokens in
  both cases (§4 "Language Models"). *Caveat:* Appendix F says the 8B results "are not
  the primary evidence... because the persona-axis construction was driven by 3B
  rollouts instead of 8B persona elicitations" — a mild tension with the §4 statement
  that each model uses its own generations. It is not fully clear whether the 8B
  persona vectors were built from 8B or 3B rollouts.

### 4.2 Layers

- **3B: layer 14.** **8B: layer 16.**
- The paper never prints total layer counts and never explicitly calls these "middle
  layers." Using public architecture specs: Llama-3.2-3B has **28** transformer
  layers, so layer 14 = 14/28 = **50.0%** (exact midpoint); Llama-3(.1)-8B has **32**
  layers, so layer 16 = 16/32 = **50.0%** (exact midpoint). So numerically this is the
  same "middle residual-stream layer" heuristic the Assistant Axis paper uses — but
  that framing, the layer totals, and any layer sweep are **absent from this paper**;
  it just states the layer number. PCA is described as run "at each layer" (Eq 2/3
  text) but only layer 14 (3B) / 16 (8B) results are reported.
- The paper says "hidden states at layer `l`" but does **not** specify post-MLP
  residual stream vs. post-block vs. post-attention (the Assistant Axis paper is
  explicit that it is the post-MLP residual stream).

### 4.3 Evaluation datasets (Table 1, verbatim)

| Behavior | Dataset | Source | Train | Test |
|---|---|---|---:|---:|
| Deception | AILiar [Pacchiardi et al., 2023] | — | 120 | 40 |
| Deception | Roleplaying [Scale Safety Research, 2024] | — | 444 | 150 |
| Deception | ConvGame [Kretschmar et al., 2025] | Liars' Bench | 335 | 113 |
| Deception | InstrDec [Kretschmar et al., 2025] | Liars' Bench | 599 | 201 |
| Deception | HarmPressureChoice [Kretschmar et al., 2025] | Liars' Bench | 599 | 201 |
| Sycophancy | Sycophancy Dataset [Azarbal, 2024] | — | 120 | 30 |
| Sycophancy | Open-Ended Sycophancy [Papadatos, 2025; Rimsky et al., 2024] | — | 84 | 22 |
| Sycophancy | OEQ Validation [Cheng et al., 2025] | ELEPHANT | 160 | 40 |
| Sycophancy | OEQ Indirectness [Cheng et al., 2025] | ELEPHANT | 160 | 40 |
| Sycophancy | OEQ Framing [Cheng et al., 2025] | ELEPHANT | 160 | 40 |

The paper prints **no column totals**. Summing the rows: deception train
120+444+335+599+599 = **2097**, test 40+150+113+201+201 = **705**; sycophancy train
120+84+160+160+160 = **684**, test 30+22+40+40+40 = **172**. (These sums are mine;
there is nothing in the paper to reconcile them against.)

All train/test splits are **label-balanced**. For learned-probe experiments: train on
the **full training split** of a source dataset (projected onto the first persona PCs)
and evaluate on the **full test split** of a target dataset.

**Inconsistency to flag (see §8):** every figure that shows per-dataset deception
results (Figures 3, 4, 7) uses the five datasets **{ConvGame, InstrDec, MASK, AILiar,
Roleplay}** — i.e. it includes **MASK** (the MASK benchmark) and **omits
HarmPressureChoice**, and Figure 7's title says "[HPC excluded]". But **MASK has no
row in Table 1** (no split sizes), and HarmPressureChoice, which *does* have a Table 1
row, never appears in a figure. So the deception evaluation set actually used is not
the one Table 1 documents.

### 4.4 Persona-axis construction inputs (recap)

| | Deception axis | Sycophancy axis | Unified axis |
|---|---|---|---|
| Harmful personas | 8 | 5 | 13 |
| Harmless personas | 7 | 5 | 12 |
| Anchor | 1 default | 1 default | 1 default (excluded from PCA) |
| Instruction variants / persona | 2 | 2 | 2 |
| Shared questions | 10 (omission-focused) | 12 | 10 + 10 balanced |
| Rollouts / persona | 20 | 24 | ~20 |
| Analysis layer | 14 (3B) / 16 (8B) | 14 (3B) / 16 (8B) | 14 (3B) / 16 (8B) |
| Judge/filter of persona expression | none | none | none |

### 4.5 Probe training and hardware (§4 "Training and Evaluation")

- Binary **logistic regression**, scikit-learn, **default L2 regularization (C = 1)**,
  with intercept, **random seed 42**, **max 1000 iterations**.
- **No `StandardScaler` or other normalization.**
- Probe input = the k PC-projection coordinates (k ∈ {1, 2, 3} depending on
  experiment), not the raw activation.
- All rollout generation, activation caching, and probe training ran on **NVIDIA
  RTX 4090 GPUs**.

### 4.6 Baselines / controls (Appendix E, Figures 7, 9, 10)

The persona-PC-projection probe is compared against:
- **Raw-activation logistic probe** ("activation_logistic @ layer 14") — the main
  baseline.
- **Random k-dimensional subspace** projection then probe.
- **Dataset-specific PCA** (PCA on the source dataset's own activations) then probe.

---

## 5. Results

### 5.1 Geometry — PC1 separates harmful from harmless personas (Figure 2)

Projecting persona vectors onto PC1/PC2:
- **PC1 cleanly separates harmful from harmless personas** for both behaviours. In the
  deception panel the x-axis runs "← Deception … Honesty →"; all 8 deceptive personas
  sit at one end, all 7 honest personas at the other. Same for sycophancy
  (sycophantic personas one end, non-sycophantic the other).
- The **default assistant persona lies near the harmless cluster** (near the honest /
  non-sycophantic end).
- Harmless personas "cluster more tightly"; **harmful personas show greater
  within-class variation** and sit at the opposite end of PC1.
- Verbatim (slightly confusingly worded in the paper): "Honest personas vary little
  along PC2, whereas harmless personas show relatively greater spread."

### 5.2 Zero-shot transfer of unsupervised directions (Figure 3, AUROC, layer 14)

Contrast direction and PC1 give the strongest and most consistent zero-shot transfer;
PC2/PC3 are more specialised and less reliable.

Deception (read from Figure 3) — AUROC for [Contrast, PC1, PC2, PC3]:

| Dataset | Contrast | PC1 | PC2 | PC3 |
|---|---|---|---|---|
| ConvGame | 0.66 | 0.74 | 0.67 | 0.27 |
| InstrDec | 0.77 | 0.77 | 0.57 | 0.37 |
| MASK | 0.73 | 0.72 | 0.46 | 0.65 |
| AILiar | 0.79 | 0.84 | 0.63 | 0.37 |
| Roleplay | 0.60 | 0.60 | 0.57 | 0.52 |

Sycophancy (read from Figure 3) — AUROC for [Contrast, PC1, PC2, PC3]:

| Dataset | Contrast | PC1 | PC2 | PC3 |
|---|---|---|---|---|
| Sycophancy Dataset | 0.94 | 0.93 | 0.49 | 0.83 |
| Open-Ended Sycophancy | 0.83 | 0.80 | 0.76 | 0.76 |
| OEQ Validation | 0.80 | 0.77 | 0.59 | 0.49 |
| OEQ Indirectness | 0.69 | 0.63 | 0.78 | 0.51 |
| OEQ Framing | 0.61 | 0.58 | 0.68 | 0.54 |

### 5.3 PC-projection probes improve cross-dataset generalization (Figures 4, 5)

**Deception (Figure 4, Llama 3.2-3B, layer 14).**
Raw-activation logistic probe cross-dataset AUROC matrix (rows = test, cols = train,
order ConvGame / InstrDec / MASK / AILiar / Roleplay):

```
            ConvGame InstrDec  MASK  AILiar Roleplay
ConvGame      0.89    0.74    0.55   0.42   0.67
InstrDec      0.82    0.88    0.52   0.47   0.66
MASK          0.81    0.76    0.83   0.76   0.64
AILiar        0.51    0.79    0.55   0.89   0.54
Roleplay      0.75    0.60    0.58   0.51   0.83
```

AUROC change from using **PC1 instead of the raw probe** (Δ = PC1 − baseline):

```
            ConvGame InstrDec  MASK  AILiar Roleplay
ConvGame     -0.15    0.03    0.17   0.42  -0.07
InstrDec     -0.08   -0.11    0.20   0.37  -0.06
MASK         -0.07    0.01   -0.11   0.08  -0.04
AILiar        0.23   -0.02    0.17  -0.05   0.06
Roleplay     -0.01    0.17    0.14   0.33  -0.23
```

Pattern: **largest gains on weak off-diagonal pairs** (e.g. train AILiar → test
ConvGame/InstrDec/Roleplay: +0.42 / +0.37 / +0.33); **some already-strong pairs
degrade** (diagonal / same-dataset pairs go slightly negative).

**Sycophancy (Figure 5, Llama 3.2-3B, layer 14).**
Raw-probe AUROC matrix (rows = test, cols = train, order Sycophancy Dataset /
Open-Ended Sycophancy / OEQ Validation / OEQ Indirectness / OEQ Framing):

```
                 Syc  OE-Syc  OEQ-V  OEQ-I  OEQ-F
Syc Dataset      1.00   0.85   0.50   0.35   0.33
OE Sycophancy    1.00   1.00   0.60   0.60   0.72
OEQ Validation   0.27   0.55   0.87   0.76   0.84
OEQ Indirectness 0.15   0.56   0.85   0.82   0.78
OEQ Framing      0.36   0.64   0.83   0.86   0.84
```

There are **two clusters** — {Sycophancy Dataset, Open-Ended Sycophancy} and {OEQ
Validation, OEQ Indirectness, OEQ Framing} — that transfer well internally
(0.8–1.0) but poorly across (0.15–0.36 in the bottom-left block).

AUROC change from **axis PC projection (k = 3) vs. raw probe**:

```
                 Syc  OE-Syc  OEQ-V  OEQ-I  OEQ-F
Syc Dataset     -0.03  -0.04   0.21   0.20   0.22
OE Sycophancy   -0.04  -0.09   0.23   0.12  -0.08
OEQ Validation   0.60   0.26  -0.04  -0.13  -0.18
OEQ Indirectness 0.60   0.32  -0.04  -0.07  -0.06
OEQ Framing      0.47   0.32  -0.05  -0.10  -0.15
```

Gains are concentrated on the **cross-cluster** transfers that were weak in the raw
baseline (up to +0.60); within-cluster (already-strong) pairs lose a little.

### 5.4 Unified harmful axis (Figure 6, Appendix Figure 9)

On the mixed deception + sycophancy benchmark, probes trained on **unified-axis PC
features (k = 2)** improve many train/test pairs relative to raw-activation probes,
"with its largest gains where raw probes transfer poorly." Full comparisons vs.
random-subspace and dataset-PCA controls are in Appendix Figure 9 (3B at layer 14,
8B at layer 16).

### 5.5 Aggregate summary across all settings (Figure 10)

Mean off-diagonal AUROC improvement over the raw probe (read from Figure 10, panel A):

| Setting | Random subspace | Dataset PCA | Axis PC projection |
|---|---|---|---|
| 3B Deception | ≈ −0.14 | ≈ −0.12 | **≈ +0.10** |
| 8B Deception | ≈ −0.14 | ≈ −0.11 | **≈ +0.12** |
| 8B Sycophancy† | ≈ (n/a) | ≈ +0.0 | **≈ +0.14** |
| 3B Sycophancy† | ≈ −0.13 | ≈ −0.02 | **≈ +0.17** |

Win rate — fraction of the 20 off-diagonal train/test pairs where the method beats the
raw probe (read from Figure 10, panel B): Axis PC projection wins **13/20** (3B
deception), **15/20** (8B deception), **10/20** (8B sycophancy), **12/20** (3B
sycophancy); random subspace and dataset PCA win far fewer (1/20–7/20). The "†" on the
sycophancy bars is not explained in the text I can read.

### 5.6 Discussion (§6) / Limitations (§7)

- The two behaviours differ: **sycophancy splits into two well-separated clusters**
  (poor cross-cluster, excellent within-cluster transfer); **deception has no such
  clear clustering** but still transfers more consistently after PC projection.
- "Persona vectors seem to act as a good prior over the representation space."
- Limitations: only two models, one family (Llama 3.2-3B, Llama 3-8B); a limited
  persona / question set; **adding more personas does not monotonically improve
  performance**; unclear how to operationalize axis-finding for arbitrary behaviours
  (hallucination, eval-awareness); persona vectors use mean-pooling over output tokens
  only, and preliminary work suggests **last-token or attention pooling can improve
  cross-dataset transfer** but was not systematically compared.

---

## 6. Minimal replication recipe (from this paper alone)

1. **Model:** Llama 3.2-3B (instruct). Analysis layer = **14** (the exact midpoint of
   its 28 layers). Hook the residual stream at that layer (activation type not
   specified by the paper — post-MLP residual stream is the Assistant-Axis default).
2. **Persona sets:** for deception, the 16 personas in Appendix A Tables 2–3 (8
   deceptive + 7 honest + 1 default); for sycophancy, the 11 personas in Appendix B
   Tables 4–5 (5 + 5 + 1). 2 instruction variants per persona; shared question pool of
   10 (deception, Appendix A list) / 12 (sycophancy, Appendix B list).
3. **Rollouts:** the model generates a response for every
   persona × instruction × question (20 / 24 per persona). No judge, no filtering.
4. **Persona vectors:** for each rollout, mean-pool layer-14 hidden states over the
   generated tokens (Eq 1 / Figure 1). Average over questions and instruction variants
   → one vector per persona (Eq 1).
5. **Center:** subtract the mean persona vector across the whole set (Eq 2).
6. **Axes:**
   - contrast direction `d = mean(centered deceptive) − mean(centered honest)` (Eq 4);
   - PCA over the centered persona vectors → `u_1, u_2, u_3, …` (Eq 3).
7. **Evaluation rollouts:** run the same model on each evaluation dataset's
   instruction + user prompt, mean-pool layer-14 output-token activations → one vector
   `z` per example.
8. **Zero-shot:** score `z` by `<z, d>` (contrast) or `<z, u_1>` (PC1); AUROC vs. the
   lie / sycophancy label. Expect deception AUROC ≈ 0.6–0.84, sycophancy ≈ 0.6–0.94
   (Figure 3).
9. **Learned probe:** project `z` onto the first k persona PCs (k = 1 for deception,
   3 for sycophancy in the paper's figures), train scikit-learn `LogisticRegression`
   (C = 1, intercept, seed 42, max_iter 1000, **no scaling**) on the source dataset's
   full balanced train split, evaluate on the target dataset's full balanced test
   split. Compare to the same probe trained on raw activations, on a random
   k-subspace, and on source-dataset PCA. Expect a mean off-diagonal AUROC gain of
   roughly +0.10 (deception) to +0.17 (sycophancy) over the raw probe, winning
   ~12–15 of 20 off-diagonal pairs.
10. **8B replication:** repeat with "Llama 3-8B" at **layer 16** (midpoint of 32
    layers); k stays 1 / 3; unified-axis k = 2.

---

## 7. How this diverges from the Assistant Axis paper (Lu et al. 2026, arXiv:2601.10387)

The paper says it "follow[s] the Assistant Axis setup" and reuses that paper's
persona-prompt framework and residualization step. But the two differ substantially.
Where I state something about the Assistant Axis paper it is from prior notes on that
paper, not from this one.

| Aspect | Assistant Axis (Lu et al. 2026) | This paper (Mahadik & Skapars) |
|---|---|---|
| **Goal of the axis** | Measure "Assistant-ness" / persona drift; steer; inference-time **activation capping** to cut jailbreaks | Zero-shot lie/sycophancy **classifier** + low-dim **probe features** for OOD generalization. No steering, no capping. |
| **What the contrast is between** | `mean(default Assistant activation) − mean(all *fully role-playing* role vectors)` — a single anchor vs. the average of hundreds of arbitrary roles | `mean(centered harmful-behavior personas) − mean(centered harmless-behavior personas)` (Eq 4) — a **two-class behaviour contrast**; the default anchor is only a reference point |
| **What PC1 means** | Similarity to the trained Assistant persona (fantastical/non-human roles vs. Assistant-like roles) | Harmful vs. harmless persona for the target behaviour (because the persona pool is deliberately built as harmful/harmless opposites) |
| **Persona population** | ~275 broad human/non-human archetypes + 240 shared extraction questions; 5 system prompts/role; ~1,200 rollouts/role | 16 (deception) / 11 (sycophancy) hand-picked behaviour-specific personas; 2 instruction variants; 10–12 shared questions; 20–24 rollouts/persona |
| **Filtering** | LLM judge labels each response fully / somewhat / no role-playing; roles kept only with ≥10 responses in a category; "no role-play" dropped → 377–463 surviving role vectors | **No judge, no filtering** — every rollout is pooled in |
| **Residualization** | Role vectors "standardized by subtracting the mean vector across roles" before PCA | **Same step, explicitly cited** (Eq 2): subtract the mean across all personas |
| **PCA input size** | Hundreds of role vectors (n ≈ 377–463) | 11–16 persona vectors → PC2/PC3 estimated from very few points |
| **Layer selection** | The model's **middle residual-stream layer**, chosen per model and stated as such (Gemma 22/46, Qwen 32/64, Llama 40/80); contrast vector also computed at **every** layer and validated by cosine-to-PC1 (>0.6 all layers, >0.71 mid) | A single stated layer (14 for 3B, 16 for 8B). Numerically these are the exact midpoints (14/28, 16/32) → **same heuristic**, but the paper never says "middle layer", never gives totals, never sweeps layers, never does a per-layer cosine check |
| **Activation type** | Explicit: **post-MLP residual stream**, mean-pooled over response tokens | "Hidden states at layer `l`", mean-pooled over output tokens — **residual-stream sublocation not specified** |
| **Pooling** | Mean over response tokens | Mean over output tokens (matches); notes last-token/attention pooling might be better but untested |
| **Models** | Gemma 2 27B, Qwen 3 32B, Llama 3.3 70B (27–70B, three families) | Llama 3.2-3B + "Llama 3-8B" (one family, much smaller) |
| **Intervention** | Steering + Equation (1) min-clamp activation cap; base-model steering; multi-turn drift study | None — read-only probing |
| **Downstream metric** | Jailbreak success rate, capability benchmarks, drift projection | AUROC and cross-dataset AUROC transfer matrices |

**Net:** this paper keeps the Assistant Axis *plumbing* (persona prompts in the same
style, mean-pooled middle-layer activations, mean-centering before PCA) but replaces
the *target* — an "Assistant vs. everything" direction becomes a "harmful-behaviour
persona vs. its opposite" direction — and replaces the *use* — steering/capping
becomes zero-shot classification and probe-feature construction. The layer choice is
plausibly the same "middle layer" idea but is asserted as a bare number rather than
derived, and none of the Assistant Axis paper's per-layer validation is carried over.

---

## 8. Ambiguities, gaps, and internal inconsistencies (flagged, not smoothed)

1. **Table 1 vs. the figures (deception datasets).** Table 1 lists
   **HarmPressureChoice** (train 599 / test 201) and no MASK. Figures 3, 4, 7 use
   **MASK** as one of five deception datasets and never show HarmPressureChoice;
   Figure 7's title says "[HPC excluded]". So MASK is used with **no documented split
   sizes**, and a Table 1 dataset is silently dropped. This is a genuine
   inconsistency, not a labelling shorthand (MASK ≠ HarmPressureChoice).
2. **k is not fixed.** §4 says probes are trained on "the first 3 persona PCs", but the
   method-comparison figures use k = 1 for deception (Figure 4 is literally "PC1 vs
   Baseline"), k = 3 for sycophancy (Figure 5), and k = 2 for the unified axis
   (Figure 6). There is no single stated k.
3. **"Llama 3-8B" checkpoint.** Written as "Llama 3-8B" / "Llama 3 8B", never "Llama
   3.1-8B". Both have 32 layers so the midpoint arithmetic is unaffected, but the
   exact model is unspecified.
4. **Layer totals / "middle layer" never stated.** The identification of layer 14 (of
   28) and layer 16 (of 32) as exact midpoints is my inference from public Llama
   specs; the paper gives neither the totals nor a "middle layer" justification.
5. **Activation sublocation.** "Hidden states at layer `l`" — post-MLP residual
   stream? post-attention? post-block? Unspecified (the Assistant Axis paper it builds
   on is explicit that it is post-MLP residual stream).
6. **8B persona-axis provenance.** Appendix F: 8B results are "not primary evidence
   because the persona-axis construction was driven by 3B rollouts instead of 8B
   persona elicitations" — in tension with §4's "for each model, we use that model's
   own generations in both parts of the pipeline." Unclear whether the 8B persona
   vectors were actually built from 8B or 3B generations.
7. **Test-sample centering.** The axes (`d`, `u_k`) are built from **centered**
   persona vectors `v~_p` (Eq 2), but Eq (3)/(4) score raw evaluation activations
   `z^(l)` (un-centered). Whether / how `z^(l)` is centered at test time (there is no
   persona set at test time to compute a mean from) is not spelled out.
8. **Anchor persona's role in Eq (4).** The default assistant persona is in the
   geometry plots and presumably in the Eq (2) mean, but §3 describes `d^(l)` as a
   contrast between deceptive and honest personas only; whether the anchor is in `D`,
   `H`, or excluded is not stated (the Assistant Axis paper explicitly excludes its
   `default` from PCA).
9. **PCA with n ≤ 16 points.** PC2/PC3 (used as zero-shot scorers in Figure 3 and as
   probe features for sycophancy) are estimated from 11–16 vectors in a
   3072/4096-dim space. The paper does not discuss the statistical reliability of
   these higher components.
10. **"Standardized"/centered.** Eq (2) is described only as mean subtraction; whether
    per-dimension variance scaling is also applied ("standardize" in the usual sense)
    is not specified — an ambiguity inherited from the Assistant Axis method it cites.
11. **"†" on the sycophancy bars in Figure 10** is unexplained in the readable text.
12. **No code or data release** is mentioned, and Appendix constructions ("its first
    10 questions after shared-set balancing", Appendix C) are described only in prose.
13. **Acronyms:** the paper does not expand **AILiar**, **InstrDec**, **ConvGame**,
    **OEQ**, **HPC**, **MASK**, **ELEPHANT**, **PCA**, **AUROC**, **OOD**. From
    external context (this repo / cited works) InstrDec ≈ "instructed deception",
    ConvGame ≈ "convincing game", HPC ≈ HarmPressureChoice, OEQ ≈ open-ended
    questions, and MASK / ELEPHANT / Liars' Bench are named benchmarks — but none of
    these expansions are asserted by this paper, so treat them as inferred.

---

## 9. Numeric self-check (against the printed pages)

- **arXiv id / date:** "arXiv:2605.09391v2 [cs.AI] 15 May 2026" — matches the task's
  `2605.09391`. ✔
- **Equation count:** four numbered equations (1)–(4), all in §3. Figure 1 shows two
  unnumbered intermediate forms (`x_{p,q}`, `v_p`) that expand Eq (1). ✔ (This
  paper *does* have formal equations.)
- **Persona counts:** deception 8 deceptive (Table 2) + 7 honest (Table 3: ombudsman,
  transparency_officer, disclosure_counsel, whistleblower, documentarian,
  fact_checker, archivist) + 1 default = **16**. ✔  Sycophancy 5 (Table 4) + 5
  (Table 5: critic, skeptic, contrarian, blunt_advisor, tough_love_mentor) + `default`
  = **11**. ✔
- **Rollouts/persona:** deception 2 instruction variants × 10 questions = **20**;
  sycophancy 2 × 12 = **24**. Text: "2 instruction variants per persona and 10 shared
  … questions" / "two instruction variants, 12 questions". ✔
- **Layer arithmetic:** 3B layer 14 of 28 total layers = 50.0% (exact midpoint);
  8B layer 16 of 32 total layers = 50.0% (exact midpoint). **Layer totals are NOT in
  the paper** — 28 and 32 are public Llama-3.2-3B / Llama-3-8B architecture values.
- **Table 1 row sums (my arithmetic; paper prints no totals):** deception train 2097
  / test 705; sycophancy train 684 / test 172. 5 datasets per behaviour → 5×5 − 5 =
  **20 off-diagonal transfer pairs**, consistent with the "X/20" annotations in
  Figure 10. ✔
- **Table 1 vs. figures:** Table 1 deception rows = {AILiar, Roleplaying, ConvGame,
  InstrDec, HarmPressureChoice}. Figures 3/4/7 deception datasets = {ConvGame,
  InstrDec, MASK, AILiar, Roleplay}. **MASK appears only in figures; HarmPressureChoice
  appears only in Table 1** ("[HPC excluded]" in Figure 7). Flagged in §8.1 — not
  reconciled by the paper.
- **Zero-shot AUROC (Figure 3, read directly):** deception Contrast/PC1 range
  0.60–0.84 (best AILiar PC1 = 0.84); PC3 as low as 0.27 (ConvGame). Sycophancy
  Contrast/PC1 range 0.58–0.94 (best Sycophancy Dataset Contrast = 0.94). Values
  tabulated in §5.2.
- **Figure 4 raw-probe diagonal (same-dataset) AUROC:** ConvGame 0.89, InstrDec 0.88,
  MASK 0.83, AILiar 0.89, Roleplay 0.83 — all read directly. PC1-vs-baseline Δ largest
  at train-AILiar → test-ConvGame = **+0.42**; most negative at train-Roleplay →
  test-Roleplay = **−0.23**.
- **Figure 5 raw-probe cross-cluster block** (train {Syc, OE-Syc} → test {OEQ-*}):
  0.27, 0.15, 0.36 / 0.55, 0.56, 0.64 — low, as the paper says. Axis-PC Δ on those
  cells: up to **+0.60** (train Syc Dataset → test OEQ Validation / Indirectness).
- **Probe config:** LogisticRegression, C = 1, intercept, seed 42, max_iter 1000, no
  StandardScaler. ✔  Hardware: NVIDIA RTX 4090. ✔
- **Models:** main = Llama 3.2-3B; replication = "Llama 3-8B" (Appendices E/F), 8B
  analysis layer = 16 (Figure 8 titles: "Deception · Llama 3 8B (Layer 16, k=1)",
  "Sycophancy … Layer 16, k=3"). ✔
- **Figure 10 win rates (read directly):** axis-PC projection 13/20, 15/20, 10/20,
  12/20 across the four settings; random-subspace / dataset-PCA 1/20–7/20. ✔
