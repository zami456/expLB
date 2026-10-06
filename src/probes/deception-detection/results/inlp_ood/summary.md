# INLP subspace features and OOD probe generalization

The persona experiment (`results/persona_ood/`) found that projecting activations onto 3 persona
PCs before probing improves cross-dataset transfer, and that the mechanism is invariance: probes
stop depending on their training set. That basis came from persona role-play. This asks whether the
subspace can be **learned from data** instead, with iterative nullspace projection (INLP) run on
RepE honesty alone:

1. Run INLP on RepE. Each iteration trains the probe on activations with every earlier classifier's
   direction projected out, and keeps that iteration's weight vector W_i.
2. Build the projection matrix from W_1..W_k and project every dataset **onto** that span.
3. Rerun the persona experiment's 6×6 transfer matrix and controls, cell for cell.

Short answer: **no.** Only k = 1 helps, and only because a 1-d probe can do nothing but reuse
RepE's own probe direction. Every k ≥ 2 transfers worse than raw probing. A label-free PCA of the
same RepE data beats the supervised INLP basis at every rank.

## Setup

| | |
|---|---|
| activations | the persona run's pooled files: layer 20, 8192-d, mean over response tokens |
| INLP | a port of the reference `get_debiasing_projection` ([Ravfogel et al. 2020](https://github.com/shauli-ravfogel/nullspace_projection)) |
| INLP classifier | the probe's own LR (C = 0.1, no intercept), on activations standardised once with a scaler fitted on RepE train |
| INLP data | fit on `repe_honesty__plain` (448 rows, 224 deceptive); score on `repe_honesty__plain_val` (102 rows, 51 deceptive) |
| iterations | 100 allowed, `min_accuracy` 0, as in the reference's own example |
| features | `standardise(z) @ B_kᵀ`, where B_k is an orthonormal basis of span(W_1..W_k) in INLP order |
| probe and matrix | identical to the persona run: StandardScaler → LR (C = 0.1); diagonal cells by 5-fold CV |
| **headline cells** | **25**: the RepE test column and the RepE diagonal are dropped from every arm, because the basis saw RepE's labels. The RepE *training* row stays. |

**The reference implementation carries over at 8192-d, with one change.** The reference forms every
rowspace projection as a dense d×d matrix and keeps them all in a list. At 8192 that is 512 MiB each,
so it would exhaust this machine's free memory in about ten iterations. The port keeps an orthonormal
basis instead, which gives the same projection through Ben-Israel's identity. At d = 40 the port and
the verbatim reference agree: the classifier directions match to 1 − cos ≈ 1e-13, and the projections
match (`tests/test_inlp.py`).

**Reproduction is exact.** `matrix_baseline.csv`, `matrix_persona_pc1.csv` and
`matrix_persona_pc3.csv` are byte-identical to the persona run's files.

## INLP on RepE: the signal runs out gradually

| iteration | train acc. | dev acc. | dev AUROC |
|---|---|---|---|
| 1 | 1.000 | 1.000 | 1.000 |
| 2 | 0.982 | 0.941 | 0.978 |
| 5 | 0.862 | 0.843 | 0.892 |
| 10 | 0.732 | 0.686 | 0.753 |
| 16 | 0.658 | 0.569 | 0.599 |
| 17 | 0.641 | **0.529** | 0.572 |
| 40 | 0.529 | 0.549 | 0.541 |

- **T\* = 16.** Dev accuracy first falls to 0.55 at iteration 17. It is at chance from about iteration
  40 on.
- **The classifiers are not memorising.** Train accuracy tracks dev accuracy all the way down. At
  C = 0.1 the probe behaves close to a class-mean-difference readout.
- **The solver returns w = 0 at iteration 83.** By then the projected class-mean difference is
  inside the lbfgs tolerance. A zero classifier leaves the projection unchanged, so every later
  iteration would refit the same zero. The run stops there with **82 directions**, which are
  mutually orthogonal to 8e-12.
- **T\* overstates how many directions carry the label.** Each classifier is an estimate, and the
  part of the signal it misses is picked up by the next one. In the unit test, a single planted
  direction takes INLP two classifiers.

See `inlp_trace.png`.

## The claim does not carry over

Mean AUROC over the 25 headline cells:

| features | dim | off-diagonal | excl. HP-KR | diagonal | train-set sensitivity |
|---|---|---|---|---|---|
| raw pooled activations | 8192 | 0.634 | 0.708 | **0.948** | 0.180 |
| persona PC1 | 1 | 0.623 | **0.902** | 0.891 | 0 † |
| persona PCs | 3 | 0.653 | 0.893 | 0.903 | **0.032** |
| **INLP, k = 1** | 1 | 0.632 | 0.824 | 0.776 | 0 † |
| INLP, k = 2 | 2 | 0.576 | 0.702 | 0.862 | 0.186 |
| INLP, k = 3 | 3 | 0.572 | 0.724 | 0.882 | 0.151 |
| INLP, k = 5 | 5 | 0.557 | 0.674 | 0.900 | 0.225 |
| INLP, k = 10 | 10 | 0.530 | 0.610 | 0.921 | 0.293 |
| INLP, k = 16 (T\*) | 16 | 0.580 | 0.668 | 0.925 | 0.257 |
| INLP, k = 50 | 50 | 0.541 | 0.598 | 0.940 | 0.245 |
| PCA of RepE, no labels, k = 3 | 3 | 0.596 | 0.754 | 0.851 | 0.118 |
| PCA of RepE, no labels, k = 16 | 16 | **0.691** | 0.748 | 0.924 | 0.098 |
| random, k = 3 | 3 | 0.530 | 0.610 | 0.820 | 0.064 |
| random, k = 16 | 16 | 0.529 | 0.565 | 0.905 | 0.061 |

† A 1-d feature can only flip sign between training sets, so its sensitivity is 0 unless training
sets disagree on the sign. That makes 0 an artifact of dimension here. PCA's first component shows
the other case: its rows are exact sign flips (0.85 ↔ 0.15), and it scores 0.195.

Every rank, 1–50, is in `summary.csv` and `k_sweep.png`. The unrestricted 6×6 means are the
`*_all` columns.

**k = 1 is the only INLP rank that helps, and only excluding HP-KR** (0.824 against raw 0.708).
- This is the one-direction effect the persona axis already showed. A 1-d probe can only pick a
  sign, so every training row collapses to one row.
- INLP direction 1 *is* the RepE probe: the same LR, fitted on 448 of the 550 rows. Its RepE-trained
  row matches baseline's RepE row within 0.02 on every Liars' Bench set. So "INLP k = 1" amounts to
  "use the RepE probe whatever you train on".
- Persona PC1 does the same job better (0.902). Its direction transfers better than RepE's own on
  HP-C (0.67 vs 0.51) and CG (0.99 vs 0.90).
- The diagonal cost is steep: 0.776 against 0.948.

**k ≥ 2 is worse than raw probing.**
- Off-diagonal it is 0.530–0.583 at every k ≥ 2, against 0.634. Excluding HP-KR, only k = 3 edges
  above raw (0.724 vs 0.708, within noise).
- Train-set sensitivity is 0.15–0.29, at or above raw's 0.180 everywhere except k = 3. The persona
  basis gets 0.032. So the INLP basis does not produce the invariance that was the whole mechanism
  of the persona result. It behaves like the mismatched sycophancy basis in the persona study (0.163).

**The label-free control beats the supervised basis.**
- PCA of the same RepE training rows, with no labels, is above INLP at every k ≥ 2 on both headline
  means. It also has lower sensitivity at every k ≥ 2 (0.09–0.12).
- At k = 16 it reaches 0.691 off-diagonal, the highest of any arm in this run. But excluding HP-KR
  it drops to 0.748, far below persona's 0.893. Its advantage is HP-KR-shaped: its HP-KR-trained row
  transfers (0.98 on CG), and its HP-KR column is less inverted.
- **Random directions fail as before.** They reach 0.53 at k = 3, rising only slowly with k.

## Why: INLP's directions disagree in sign across datasets

`direction_auroc.csv` scores each INLP direction alone, zero-shot, with no probe trained:

| direction | CG | HP-C | HP-KR | IT | ID | RepE (in-sample) |
|---|---|---|---|---|---|---|
| 1 | 0.90 | 0.51 | 0.38 | 0.93 | 0.96 | 1.00 |
| 2 | 0.42 | 0.54 | **0.95** | **0.29** | 0.46 | 0.99 |
| 4 | 0.31 | 0.61 | **0.95** | **0.13** | 0.26 | 0.95 |
| 5 | 0.34 | 0.57 | 0.83 | **0.06** | **0.10** | 0.93 |
| 10 | **0.99** | 0.40 | 0.56 | 0.77 | 0.88 | 0.78 |

- **All 20 directions separate RepE** (in-sample AUROC 0.63–1.00). But every one of them is
  inverted on at least one Liars' Bench set, and 14 of the 20 strongly so (below 0.3). Sixteen carry
  some set's deception at above 0.7.
- The persona basis is one dominant direction whose polarity is consistent across the four healthy
  datasets. The INLP basis is many RepE-decodable directions whose polarities conflict.
- A probe trained on dataset A weights each direction with A's sign, so transfer to B works only
  where A and B agree. That produces the mottled rows and high sensitivity in `transfer_matrices.png`.
- In one line: INLP finds everything linearly decodable about RepE's label, not the part RepE shares
  with other datasets. Only the first direction was the shared part.
- The two bases are nearly orthogonal as raw-space directions. The principal cosines of INLP-3
  against persona PC3 are 0.23, 0.02 and 0.00, and |cos(W_1, persona deception axis)| = 0.21.

See `direction_auroc.png`.

## HP-KR: inverted along the main direction, not in general

The persona summary concluded that HP-KR's inversion "is a property of the dataset, not of any one
probe". The RepE probe, the persona axis and the published run all place it near 0.1, and INLP
direction 1 agrees (0.38).

However:
- Directions 2 and 4 each score **0.95 on HP-KR with RepE's polarity**. Both were fitted on RepE
  alone and never saw HP-KR.
- A RepE-trained probe restricted to span(W_1, W_2) reaches **0.93 on HP-KR**, where the raw RepE
  probe gets 0.38.
- The price is CG falling to 0.70 and IT to 0.55, because the directions that carry HP-KR's lies are
  anti-aligned with IT's.

So HP-KR's lies are readable from RepE-derived directions. They just are not along the dominant one.
"Inverted along the main deception direction" is the dataset property; "undetectable by RepE-trained
probes" is not.

This is a post-hoc observation: 20 directions × 5 datasets is 100 cells. It is not noise, though.
With 139 deceptive rows against 2000 honest, HP-KR's AUROC has a standard error of about 0.025
under the null, which puts 0.95 some 18 standard errors from chance.

## Benchmark metric

Balanced accuracy at 1% FPR on the alpaca control, for the RepE-trained probe, averaged over the
five Liars' Bench sets:

| features | bal. acc @1% FPR | recall @1% | AUROC |
|---|---|---|---|
| raw pooled activations | **0.657** | **0.329** | 0.734 |
| INLP, k = 1 | 0.651 | 0.317 | 0.735 |
| INLP, k = 2 | 0.529 | 0.059 | 0.710 |
| INLP, k = 16 | 0.523 | 0.047 | 0.693 |
| PCA of RepE, k = 16 | 0.626 | 0.259 | 0.736 |
| persona PC1 | 0.555 | 0.114 | 0.752 |
| persona PCs | 0.551 | 0.106 | **0.752** |

- INLP k = 1 matches the raw probe on the benchmark, which is expected since it is essentially the
  same direction.
- It beats persona there despite a lower AUROC. That is the persona summary's finding again: the
  persona directions rank better but calibrate worse against the control.
- Past k = 1 the INLP probes lose the threshold almost entirely (recall ≈ 0.05).
- The "raw" row here is a probe trained on pooled activations. It is not the deployed per-token probe
  from the sweep, which scores 0.721.

**A correction to the persona summary's benchmark table.** Its "PC3 probe (RepE-trained)" row
(0.558 / 0.121 / 0.766) was fitted on the 102-row RepE val split alone. This code reproduces it
exactly with that split: 0.5583 / 0.1210 / 0.7656. On the 550 rows the rest of that summary uses, it
is 0.551 / 0.106 / 0.752. The conclusion there is unchanged.

## Caveats

- **One source dataset.** This is RepE's INLP decomposition. A source whose deception looks more like
  the evaluation sets might yield later directions that agree in sign. Running INLP on each dataset
  in turn would test that; it was not run here.
- **INLP is classifier-relative.** The directions are those of this LR at C = 0.1 in standardised
  space. A different classifier or regularisation strength would decompose the signal differently.
- **Dev accuracy is noisy and conservative.** There are 102 rows (standard error ≈ 0.05), and RepE's
  train and val splits share facts with opposite labels. T\* is a summary only; the k sweep does not
  depend on it.
- **Thin cells.** HP-C and HP-KR have about 135 deceptive rows each, as in the persona study.
- **Layer 20 only.** INLP needs no persona bank, so the same script can run at L40 once that pool
  exists. Pass `with_persona=False` there.

## Files

| file | contents |
|---|---|
| `inlp_basis.npz` | the 82 classifier vectors `Ws`, the ordered basis, the RepE scaler, T\* and parameters |
| `inlp_trace.csv` | per iteration: train/dev accuracy, dev AUROC, weight norm, kept, \|cos\| to the persona axis |
| `direction_auroc.csv` | zero-shot AUROC of each of the first 20 directions on every dataset |
| `matrix_<arm>.csv` | 6×6 AUROC matrices for `baseline`, `persona_pc{1,3}`, and `{inlp,pca,random}_k{1,2,3,5,10,16,20,50}` |
| `summary.csv` | one row per arm and k: headline and unrestricted means, train-set sensitivity |
| `benchmark.csv` | balanced accuracy, recall and AUROC at 1% FPR, per arm and dataset, RepE-trained |
| `inlp_trace.png`, `k_sweep.png`, `transfer_matrices.png`, `transfer_delta.png`, `direction_auroc.png` | figures |

## Reproduce

Run from the package root with the repo-root venv (`../../../.venv/Scripts/python.exe`):

    python -m pytest tests/test_inlp.py
    python deception_detection/scripts/inlp_ood.py run            # ~3 min, log in logs/inlp_ood.log
    python deception_detection/scripts/inlp_ood_figures.py run
