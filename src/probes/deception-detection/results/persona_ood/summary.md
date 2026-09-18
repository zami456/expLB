# Persona PC features and OOD probe generalization

Testing arXiv 2605.09391's claim that projecting activations onto persona principal components
improves cross-dataset probe generalization, against the cached layer-20 Llama-3.3-70B
activations from the Liars' Bench sweep. No new model calls: every example was already on disk.

## Setup

| | |
|---|---|
| activations | layer 20, 8192-d, mean-pooled over assistant response tokens |
| examples | 13,648 cached, pooled to 301 MB (7 datasets) |
| persona set P | the **15** deception + honesty personas, not the full 26-persona bank |
| PCA | on the centred 15; explained variance **[0.464, 0.116, 0.097]**, 67.7% cumulative |
| probe | StandardScaler → LogisticRegression(C=0.1, no intercept), matching the deployed probe |
| diagonal cells | stratified 5-fold CV, so no cell is train-on-test |

**Deviation from the paper:** P is the 15, not all 26. Measured reason — on the 15, PC1 sits at
|cos| = **0.949** to the deception axis and separates the groups perfectly; over all 26 that falls
to 0.722 as the sycophancy axis blurs in.

**Pooling verified exactly.** Scoring the pooled matrix with the run's own `detector.pt`
reproduces `results_table.csv` to **Δ = 0.0000** on all six datasets. This is exact rather than
approximate because the probe is affine: averaging activations then scoring equals scoring then
averaging.

## The claim replicates

Mean AUROC over the 6×6 train/test matrix:

| features | dim | off-diagonal | off-diag. excl. HP-KR | diagonal |
|---|---|---|---|---|
| baseline (pooled activations) | 8192 | 0.6468 | 0.7278 | **0.9564** |
| **top-3 persona PCs** | 3 | **0.6853** | **0.9123** | 0.9187 |
| random 3-d projections | 3 | 0.5261 | 0.5590 | 0.7769 |
| deception axis only | 1 | 0.6462 | 0.9248 | 0.9197 |

**Off-diagonal improves (+0.039 overall, +0.185 excluding HP-KR); the diagonal degrades
(−0.038).** Both directions match the paper, which reports gains on cross-dataset pairs "along
with some degradation on diagonal pairs". 21 of 30 off-diagonal cells improve, median **+0.059**.

## Both controls pass — but they reframe the result

**It is not dimensionality reduction.** Random 3-d projections reach only 0.526 off-diagonal
against the PCs' 0.685. Three arbitrary directions destroy the signal; three persona directions
preserve it. The structure matters, not the rank.

**It is almost entirely one direction.** The deception axis alone (1-d) reaches 0.9248
off-diagonal excluding HP-KR — *better* than the top-3 PCs' 0.9123. Since PC1 is essentially that
axis (cos 0.949), PC2 and PC3 add nothing for a healthy training set, and slightly hurt.

Where PC2/PC3 do earn their place is robustness to a pathological training set. Trained on HP-KR
— itself inverted — the axis-only probe anti-transfers catastrophically (0.003–0.04 on CG, IT,
ID); the 3-d version recovers substantially (+0.32 on IT, +0.21 on RepE). That is the only reason
`pc3` beats `axis1` on the all-in off-diagonal mean.

## The mechanism: the probe stops depending on its training set

Standard deviation of off-diagonal AUROC across training sets, per test set (HP-KR train row
excluded, since it is inverted and dominates):

| features | mean std across training sets |
|---|---|
| baseline | 0.168 |
| **top-3 persona PCs** | **0.027** |

A **6× reduction**. In the baseline matrix, which dataset you train on swings transfer wildly
(CG→HP-KR = 0.05, ID→CG = 0.28). Under the persona projection the rows become near-identical: any
training set yields ≈0.98 on CG, ≈0.66 on HP-C, ≈0.98 on IT. This is what the "improved
generalization" claim amounts to here — the 3-d subspace is fixed and dataset-independent, so the
probe cannot overfit to dataset-specific directions. It is visible directly in
`transfer_matrices.png`: the treatment panel has flat rows, the baseline is mottled.

## Zero-shot: the persona axis beats the trained probe where the probe fails

Methods 1 and 2 use no Liars' Bench training data at all — the directions come entirely from
persona role-play rollouts. Compared against the RepE-trained probe from the sweep:

| dataset | n | deceptive | contrastive axis | PC1 | trained probe |
|---|---|---|---|---|---|
| convince_me | 267 | 167 | 0.976 | 0.987 | 0.984 |
| harm_refusal_choice | 2134 | 134 | **0.704** | 0.673 | **0.471** |
| harm_refusal_knowledge_report | 2139 | 139 | 0.103 | 0.150 | 0.109 |
| insider_trading | 1080 | 423 | 0.987 | 0.988 | 0.973 |
| instruct_dishonesty | 5432 | 3045 | 0.960 | 0.959 | 0.950 |
| repe_honesty (val) | 102 | 51 | 0.997 | 0.997 | 1.000 |

The axis matches or beats the trained probe on five of six, and on **harm-pressure-choice it wins
by +0.23** — turning a near-chance result (0.471) into a usable one (0.704). A direction derived
from persona rollouts, never fitted to this benchmark, outperforms a probe trained on RepE
contrast pairs on the dataset where that probe fails.

As expected from cos = 0.949, the axis and PC1 agree closely everywhere.

## HP-KR is inverted under a third independent method

`harm_refusal_knowledge_report` scores **0.103** on the persona axis, against 0.109 for the
trained probe here and 0.124 in the Liars' Bench paper's own Table 6. Three detection methods that
share no training data — RepE contrast pairs, persona role-play vectors, and the published run —
all place it at the same strongly-inverted value. Whatever drives that inversion is a property of
the dataset, not of any one probe.

## Caveats

- **P = 15, not 26.** A documented deviation. The full-bank version re-runs in seconds against the
  same pooled matrix if wanted for comparison.
- **The headline gain is carried by one direction.** Read "persona PCs help" as "the deception
  axis helps, and PC2/PC3 add robustness against a bad training set" — not as three-dimensional
  structure being necessary.
- **The diagonal gets worse**, by about as much as the off-diagonal improves. If a probe is only
  ever used in-distribution, the projection costs accuracy.
- **HP-KR distorts the aggregate.** It is inverted as a test set and poisonous as a training set;
  the off-diagonal means are reported with and without it for that reason.
- **Thin cells.** harm_refusal_choice and harm_refusal_knowledge_report carry only 134 and 139
  deceptive examples against ~2000 honest, so their AUROCs have wide error bars.
- **repe_honesty val** is in-distribution for the deployed probe and has only 102 rows; it is kept
  as a reference row but should not be read as a transfer result.

## Files

| file | contents |
|---|---|
| `zero_shot.csv` | per-dataset axis and PC1 AUROC |
| `matrix_baseline.csv`, `matrix_pc3.csv` | the two 6×6 transfer matrices |
| `matrix_random3.csv`, `matrix_axis1.csv` | the two controls |
| `matrix_delta_pc3_vs_baseline.csv` | treatment minus baseline |
| `summary.csv` | the four-row headline table |
| `transfer_matrices.png`, `transfer_delta.png` | heatmaps |
