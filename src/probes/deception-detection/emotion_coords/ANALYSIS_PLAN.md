# Analysis plan: how each preregistered prediction is judged

Written 2026-10-29, after the pilot and before any full-run probe was fitted. It is committed on its own. It fills
in details the preregistration (`PREREGISTRATION.md`, commit `ae8e0ae1`) leaves open, and changes no prediction.

**What had been seen.** The pilot had shown two things by then:
- Protocol-B AUROCs from a 20-emotion basis on 50 rows per class.
- The published probe's D1 AUROCs on every HP-KR row, at blocks 20 and 52. These do not change in the full run.

The rules below are the plain reading of the preregistered wording; none is tuned to those numbers. Where a rule
touches a number that was already seen (P7), that is said explicitly.

## Common to every prediction

- **Rows.** The spec's caps, seeded (`phase2 subset`): CG 167 + 100, HP-C 134 + 1,000, HP-KR 139 + 1,000,
  IT 423 + 500, ID 1,000 + 1,000. Alpaca (2,000) sets thresholds. The instructed pairs (550) train protocol A.
  D1 scores every HP-KR row.
- **Layers.** Blocks 20, 28, 40 and 52 stand for the spec's 16, 27, 40 and 53 (`PREREGISTRATION.md`, item 1).
- **Probe family.** P1-P4 and P6 are judged with the L2 logistic-regression probe (the spec's first family, and
  the Liars' Bench probe's). The difference-of-means probe is reported beside it and never used to judge.
- **k and C.** Chosen inside each training fold by inner cross-validation. Under protocol B that is leave one
  training set out. Each held-out set has its own fold, so its own k.
- **CIs.** The paired bootstrap draws 1,000 stratified resamples of each held-out set, shared by the two arms
  being compared. A "mean over held-out sets" is taken within each resample, and the CI is the 2.5th-97.5th
  percentile.
- **Random-subspace control.** The mean AUROC over its 20 seeds, taken within each resample.
- **"Held".** A prediction evaluated per layer counts as held if it holds at 3 or more of the 4 layers (pooling a).

## Exploratory addition (user request, 2026-10-29, before any full-run probe)

The whole emotion subspace is projected onto and probed at fixed k = 170: all PCs of the 171 centred emotion
vectors.

- **Controls at the same dimension:**
  - the five label-shuffled bases (170 PCs each);
  - 5 random 170-dimensional subspaces;
  - the top 170 PCs of Alpaca's activations;
  - the top 170 PCs of the training rows, fitted inside each fold.
- **Scope:** protocol B, pooling (a), all four layers, both probe families.
- **It does not judge any prediction.** It is its own family, kept out of the emotion PCs' preregistered k choice
  ({2, 3, 5, 10, 20}). The report gives mean held-out AUROCs and paired-bootstrap differences (emotion minus each
  control).

## Exploratory addition: MLP probes after projection (user request, 2026-10-29, before any full-run probe)

A third, nonlinear probe family runs on every projected feature set: emotion PCs, topic, label-shuffled, random
(5 seeds), Alpaca PCs, training-set PCs, persona PCs, persona + emotion, named emotions, and the 170-dimensional
sets above. Raw activations are not included.

- **Architecture:** standardise; one hidden layer of 64 ReLU units; L2 weight decay from {0.01, 1}; Adam; early
  stopping on 10% of the training rows; seed 0.
- **Selection:** k and the weight decay by inner leave-one-set-out CV, as for logistic regression.
- **Scope:** protocol B, pooling (a), all four layers. Controls are matched as for P1, at the emotion basis's
  chosen k.
- **It judges no prediction.** The report gives mean held-out AUROCs and paired differences (`exploratory_mlp.csv`).

## Per prediction

- **P1.** Let Δ_rand be the mean over the five held-out sets of [AUROC(emotion PCs at k_t) − mean AUROC(random at
  k_t)], where k_t is the emotion basis's choice in fold t. Let Δ_topic be the same with the topic basis at k_t.
  P1 holds at a layer if both are ≥ 0.03 and both CIs exclude 0.
- **P2.** With IT held out: AUROC(emotion PCs, its k) − AUROC(persona PCs, the persona basis's own k) > 0, with
  the CI excluding 0.
- **P3.** With HP-C held out, and separately with HP-KR held out: the point estimate of AUROC(emotion PCs) − mean
  AUROC(random at the same k) is below 0.02 in both.
- **P4.** Persona + emotion is taken at the inner-CV choice among k_p = k_e ∈ {3, 5, 10}. P4 holds if its mean
  held-out AUROC minus that of persona PCs (at their own k) is ≥ 0.02. The rule uses the point estimate; the CI
  is reported.
- **P5.** At least one of guilty, afraid and desperate has zero-shot AUROC > 0.6 on IT and < 0.55 on ID (point
  estimates, same direction, sign fixed in advance).
- **P6.** For each erasure (top 5 emotion PCs, top 20 emotion PCs, LEACE), compare it with a random erasure of
  the same rank (rank 5, 20 and LEACE's rank; 10 seeds, mean taken within each resample).
  - δ = mean held-out AUROC(raw, emotion erased) − mean held-out AUROC(raw, random erased).
  - P6 holds at a layer if, for at least 2 of the 3 erasures, δ > 0 with the CI excluding 0 and erasing emotion
    raises AUROC over raw (point estimate).
  - The erased arms reuse the raw probe's fold-chosen C.
- **P7.**
  - The protocol-A probe is the published RepE-trained probe of block 20, scored per token and averaged (pooled,
    which is equivalent). Its HP-KR AUROC on the capped rows must be below 0.5; all rows are reported too.
  - D1 readings use 95% CIs from 1,000 stratified resamples of the two subgroups compared. "Well under 0.5" means
    the CI lies below 0.5; "near 0.5" means the CI includes 0.5.
  - (a) holds if lie vs admission is well under 0.5 and lie vs denial is near 0.5. (b) holds if the roles are
    reversed. Otherwise neither holds.
  - These AUROCs were seen in the pilot. The CI reading is the default meaning of the words, not a threshold
    fitted to them, and RESULTS.md says so.
- **P8.** At block 52, D2's known × harmful interaction t-statistic is compared with two nulls: 1,000 random unit
  directions (seed 0) and the 40 topic directions.
  - (a) holds if any of compassionate, empathetic, sympathetic, kind or loving lies above the 95th percentile of
    both nulls. "Protective" was not built: Anthropic's story prompt was unavailable.
  - (b) holds likewise for any of afraid, terrified, angry, hostile, suspicious or vigilant.
  - Holm-adjusted one-sided p-values from the random-direction null are reported across these 11 emotions.
