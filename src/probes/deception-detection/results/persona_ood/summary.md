# Persona PC features and OOD probe generalization

Testing arXiv 2605.09391's claim that projecting activations onto persona principal components
improves cross-dataset probe generalization, against the cached layer-20 Llama-3.3-70B
activations from the Liars' Bench sweep. No new model calls: every example was already on disk.

## Setup

| | |
|---|---|
| activations | layer 20, 8192-d, mean-pooled over assistant response tokens |
| examples | 13,648 cached, pooled to 310 MB (7 datasets) |
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
| baseline (pooled activations) | 8192 | 0.6478 | 0.7282 | **0.9564** |
| **top-3 persona PCs** | 3 | **0.6823** | **0.9123** | 0.9184 |
| random 3-d projections | 3 | 0.5269 | 0.5599 | 0.7874 |
| deception axis only | 1 | 0.6457 | 0.9237 | 0.9188 |

**Off-diagonal improves (+0.035 overall, +0.184 excluding HP-KR); the diagonal degrades
(−0.038).** Both directions match the paper, which reports gains on cross-dataset pairs "along
with some degradation on diagonal pairs". 20 of 30 off-diagonal cells improve, median **+0.068**.

## Both controls pass — but they reframe the result

**It is not dimensionality reduction.** Random 3-d projections reach only 0.527 off-diagonal
against the PCs' 0.682. Three arbitrary directions destroy the signal; three persona directions
preserve it. The structure matters, not the rank.

**It is almost entirely one direction.** The deception axis alone (1-d) reaches 0.9237
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
| baseline | 0.167 |
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
| repe_honesty (train+val) | 550 | 275 | 0.992 | 0.996 | 1.000* |

The axis matches or beats the trained probe on five of six, and on **harm-pressure-choice it wins
by +0.23** — turning a near-chance result (0.471) into a usable one (0.704). A direction derived
from persona rollouts, never fitted to this benchmark, outperforms a probe trained on RepE
contrast pairs on the dataset where that probe fails.

\* in-distribution for the deployed probe, which was fitted on these rows.

As expected from cos = 0.949, the axis and PC1 agree closely everywhere.

## Against the Liars' Bench baselines: better at ranking, worse at the benchmark

The benchmark's headline metric is **balanced accuracy at 1% FPR on the alpaca control**, not
AUROC. Averaged over the five shared datasets (CG, HP-C, HP-KR, IT, ID):

| method | bal. acc @1% FPR | recall @1% | AUROC |
|---|---|---|---|
| RepE probe (this run) | **0.721** | — | 0.697 |
| paper Mean Probe, Llama | 0.636 | — | 0.682 |
| PC3 probe (RepE-trained) | 0.558 | 0.121 | **0.766** |
| PC1 (zero-shot) | 0.555 | 0.114 | 0.751 |
| persona axis (zero-shot) | 0.547 | 0.096 | 0.746 |

The persona methods have the **best ranking and the worst thresholded accuracy**. The reason is
calibration against the control, not ranking failure — recall at the 1% threshold:

```
CG     AUROC 0.976  ->  recall 0.012
HP-C   AUROC 0.704  ->  recall 0.000
IT     AUROC 0.987  ->  recall 0.142
ID     AUROC 0.960  ->  recall 0.324
```

On convincing-game the axis separates that dataset's own honest and deceptive examples almost
perfectly, and flags 1.2% of its lies: the entire CG distribution sits below the alpaca
threshold. This is the same failure the Liars' Bench paper diagnoses for its own Mean Probe —
performance "strongly impeded by the high decision threshold, which was determined relative to
honest examples in our control dataset" — only more severe here.

**So: no, this does not beat the Liars' Bench baselines.** At 0.55 balanced accuracy it would
rank below Mean Probe's published 0.54 neighbours and well below UQC (0.60), Self-Evaluation
(0.66) and LLM-as-a-Judge (0.73). Persona directions carry real, transferable deception signal,
but they are not calibrated for a fixed-FPR deployment threshold, and the benchmark forbids the
per-dataset calibration that would fix it.

## HP-KR is inverted under a third independent method

`harm_refusal_knowledge_report` scores **0.103** on the persona axis, against 0.109 for the
trained probe here and 0.124 in the Liars' Bench paper's own Table 6. Three detection methods that
share no training data — RepE contrast pairs, persona role-play vectors, and the published run —
all place it at the same strongly-inverted value. Whatever drives that inversion is a property of
the dataset, not of any one probe.

## Fidelity to the paper

Read the paper (arXiv 2605.09391, *Do Linear Probes Generalize Better in Persona Coordinates?*)
after the deception arm was already built. Recording the gaps, because none of the absolute
numbers here are comparable to its figures:

| | paper | this work |
|---|---|---|
| model | Llama 3.2 **3B** | Llama-3.3-**70B** |
| layer | **14** | **20** |
| datasets | its own 10 | Liars' Bench (6) |
| features shown | PC1 (deception, Fig. 4); k=3 (sycophancy, Fig. 5) | both, plus the contrastive axis |
| cross-behaviour bases | **never** — each axis is used on its own datasets | done here |
| orthogonalisation | **none** | added here |

Model and layer were not choices: the persona bank and the cached activations are both
Llama-70B layer 20, so matching the paper was never possible. Only the *direction* of the effect
transfers, not the magnitudes.

Two clarifications on wording. The paper's "residualization" means Eq. 2 alone — centring by
subtracting the mean across personas — not removing one behaviour's axis from another's. And its
only use of "orthogonal" is the PCA property that *"higher PCs capture variation orthogonal to
lower PCs"*. It does construct **separate** axes per behaviour ("we construct persona axes for
deception and sycophancy"), which matches the separate 15/10 PCAs used here.

## Cross-behaviour: does a sycophancy basis detect deception?

Not a question the paper asks — Figure 5 applies the sycophancy axis to *sycophancy* datasets.
This is the control run here. **No orthogonalisation in this table**; every row is a plain
projection onto a fixed basis.

| features | dim | off-diagonal | excl. HP-KR | diagonal |
|---|---|---|---|---|
| baseline (pooled activations) | 8192 | 0.6478 | 0.7282 | 0.9564 |
| deception PC1 | 1 | 0.6520 | **0.9205** | 0.9082 |
| deception PCs | 3 | **0.6823** | 0.9123 | 0.9184 |
| deception axis | 1 | 0.6457 | **0.9237** | 0.9188 |
| sycophancy PC1 | 1 | 0.6148 | 0.8828 | 0.8915 |
| sycophancy PCs | 3 | 0.6060 | 0.7821 | 0.9285 |
| sycophancy axis | 1 | 0.6144 | 0.8816 | 0.8898 |
| random directions | 3 | 0.5269 | 0.5598 | 0.7874 |

**Persona directions transfer non-trivially even across behaviours.** Sycophancy PC1, fitted on
role-play rollouts about flattery and never shown a deception label, reaches **0.883** off-diagonal
excluding HP-KR — well above the 8192-d raw-activation baseline's 0.728 and far above random's
0.560. That supports the paper's abstract claim that "persona-derived directions transfer
non-trivially", and extends it: they transfer across behaviours, not only across datasets.

**But the matched basis is consistently better.** Deception beats sycophancy on every comparison —
PC1 0.652 vs 0.615, k=3 0.682 vs 0.606, axis 0.646 vs 0.614, and excluding HP-KR 0.920 vs 0.883.
The basis is behaviour-specific; an adjacent behaviour costs real performance.

**On all six datasets the sycophancy arms fall below the raw baseline** (0.606–0.615 vs 0.648), and
win only once HP-KR is excluded (0.782–0.883 vs 0.728). HP-KR punishes them harder than it punishes
the deception arms.

### The mismatched basis loses the invariance property

The means understate the difference. Standard deviation of off-diagonal AUROC across training
sets (HP-KR train row excluded, since it is inverted and dominates):

| features | dim | train-set sensitivity |
|---|---|---|
| baseline | 8192 | 0.167 |
| **deception PCs** | 3 | **0.027** |
| **sycophancy PCs** | 3 | **0.163** |
| random directions | 3 | 0.052 |

The deception basis makes the probe nearly independent of what it was trained on — a 6x drop
against the raw baseline, and the mechanism behind the whole result. **The sycophancy basis does
not**: at 0.163 it is as training-set-sensitive as raw 8192-d activations, and worse than random
directions. Its PC2 and PC3 apparently carry structure that datasets differ on and probes overfit
to, so the mismatched basis loses the one property that made the matched one useful.

This is visible in `transfer_sycophancy.png`: the deception panel has flat rows, the sycophancy
panel does not (the CG-trained row reads 1.00/0.44/0.05/0.87/0.81/0.67, the HP-C-trained row
0.17/0.71/0.47/0.60/0.73/0.74).

Every 1-d variant scores exactly 0.000 on this metric, which is an artifact rather than a finding:
a one-feature probe can only learn a sign and a scale, so its AUROC is fixed by the direction and
cannot depend on the training set at all.

Note the two behaviours are **not independent**: |cos(deception axis, sycophancy axis)| = **0.5402**,
so some of the sycophancy transfer is the shared component rather than sycophancy structure. The
next section isolates that; it is an addition, not a replication.

## Isolating the shared component (not in the paper)

Since the axes overlap at 0.54, each basis was also run with the other's axis projected out.
`orthogonalize()` keeps rank 3 both ways, so the comparison stays matched.

| features | dim | off-diagonal | excl. HP-KR |
|---|---|---|---|
| sycophancy PCs, deception removed | 3 | **0.5419** | 0.6383 |
| deception PCs, sycophancy removed | 3 | **0.7111** | 0.8627 |
| *(random directions)* | 3 | 0.5269 | 0.5598 |

**The sycophancy transfer was borrowed.** Remove the deception direction and it collapses to 0.542
against random's 0.527 — essentially nothing of its own survives. The 0.883 above is the shared
component doing the work.

**The reverse is the surprise.** Removing the *sycophancy* axis from the *deception* basis gives
the highest off-diagonal figure in the study, 0.7111 against 0.6823. Per test set it is **+0.224 on
HP-KR** and **−0.166 on CG**, and both are interpretable: CG is the convincing game, where the
sycophancy axis alone scores **0.9995 zero-shot**, while HP-KR's deceptive answer is a deferential
"I don't know" that reads as sycophantic. The aggregate win is narrow — it leads only because HP-KR
dominates the mean, and loses to plain deception PCs once HP-KR is excluded (0.863 vs 0.912). The
honest reading is that the deception axis is entangled with sycophancy in a way that helps on
persuasion tasks and hurts on deference tasks.

**Caveat on the control's strength.** The planned check was that the sycophancy axis should be
clearly worse than the deception axis on RepE, a pure honest/deceptive contrast. It is worse, but
only slightly — 0.978 against 0.992. At |cos| = 0.54 the raw sycophancy arm is a weak control by
itself; the orthogonalised arm carries the conclusion.

## Balanced accuracy: the benchmark's metric, and why it is secondary

Liars' Bench scores on **balanced accuracy at 1% FPR on the alpaca control**, not AUROC. The same
transfer matrices under that metric (`summary_balacc.csv`, `matrix_balacc_*.csv`):

| features | dim | off-diag | excl. HP-KR | diagonal | sd | **cells at 0.5 floor** |
|---|---|---|---|---|---|---|
| baseline | 8192 | 0.5545 | 0.5819 | 0.8248 | 0.109 | **15 / 30** |
| deception PC1 | 1 | **0.5968** | **0.6460** | 0.6223 | 0.159 | 13 / 30 |
| deception PCs | 3 | 0.5737 | 0.6238 | 0.6186 | 0.151 | **10 / 30** |
| deception axis | 1 | 0.5866 | 0.6344 | 0.6120 | 0.149 | 13 / 30 |
| sycophancy PCs | 3 | 0.5253 | 0.5434 | 0.5716 | 0.069 | 16 / 30 |
| sycophancy PCs, deception removed | 3 | 0.5016 | 0.5097 | 0.5381 | 0.035 | **22 / 30** |

**It cannot replace AUROC.** Balanced accuracy is `(recall + specificity)/2`, so a probe that
flags nothing scores exactly 0.5 — and *half the baseline matrix lands there*. Worse, a strongly
anti-correlated probe flags nothing either, so it scores 0.5 too. Under AUROC the HP-KR column
reads `0.09, 0.07, 0.40, 0.21, 0.15`; under balanced accuracy it is `0.500, 0.500, 0.500, 0.506,
0.500`. The inversion that three independent methods agree on simply disappears. `tests/
test_metrics.py::TestInversionCollapse` pins this.

**But it shows one thing AUROC cannot.** The deception projections have *fewer* dead cells than the
raw baseline (10–13 against 15), so they improve calibration against the control and not only
ranking. The sycophancy ones go the other way (16–22), and the orthogonalised sycophancy arm is
almost entirely dead at 22/30 — consistent with its collapse under AUROC.

## Against the Liars' Bench probe baselines

All three of the paper's probes (`liars_bench_comparison.csv`), re-averaged over the five shared
datasets. The paper's rows are **transcribed from Tables 6–8**, not recomputed — none can be
reproduced from the activation cache. The Follow-Up probe rewrites every input
(`experiment.py:422-432` appends a follow-up question and a prefilled `"No"`, and scores that one
token), so every cache key misses and it would need a fresh remote extraction; the Upper-Bound
probe also trains on privileged in-dataset examples.

| method | bal. acc @1% FPR | AUROC | source |
|---|---|---|---|
| Mean Probe | **0.636** | 0.682 | Table 6 p.54 |
| Follow-Up Probe | 0.578 | 0.634 | Table 7 p.55 |
| Upper-Bound Probe (*privileged*) | *0.890* | *0.930* | Table 8 p.56 |
| persona axis (zero-shot) | 0.547 | 0.746 | measured |
| persona PC1 (zero-shot) | 0.555 | 0.752 | measured |
| persona PC3 (RepE-trained) | 0.551 | **0.752** | measured |
| raw activations (RepE-trained) | 0.657 | 0.734 | measured |

**Persona wins on ranking and loses on the benchmark's metric.** Against the paper's Mean Probe:
AUROC **+0.070**, balanced accuracy **−0.085**. The zero-shot axis manages 0.746 AUROC without
seeing a single Liars' Bench label. But at a 1%-FPR threshold it flags too little to convert that
into accuracy, so on the benchmark's own terms it would place below every deployable baseline.

**The within-pipeline comparison is the honest one, and it is less flattering.** Cross-pipeline
rows are not comparable — this repo's reproduction of the Mean Probe recipe reaches 0.657 balanced
accuracy where the paper reports 0.636, largely because `trim_reasoning` flips insider-trading
(the paper's own Table 17). Comparing persona against the raw-activation probe *in this pipeline*,
sharing pooling, estimator and training set, the AUROC gain is only **+0.018** (0.752 vs 0.734)
while the balanced-accuracy loss is **−0.102**. The ranking advantage over the paper's number is
real but substantially inflated by the pipeline difference.

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
- **RepE is in-distribution** for the deployed probe, so its diagonal and its column say nothing
  about transfer. It uses the full 80/20 split merged back together (448 train + 102 val = 550
  rows), which is the protocol the paper and the deployed probe use. Running it with only the
  102-row validation remainder moved the headline by 0.003, so the matrix is not sensitive to
  this choice.

## Files

| file | contents |
|---|---|
| `zero_shot.csv` | per-dataset axis and PC1 AUROC |
| `matrix_baseline.csv`, `matrix_pc3.csv` | the two 6×6 transfer matrices |
| `matrix_random3.csv`, `matrix_axis1.csv` | the two controls |
| `matrix_delta_pc3_vs_baseline.csv` | treatment minus baseline |
| `summary.csv` | the four-row headline table |
| `matrix_syc3.csv`, `matrix_syc3_orth.csv`, `matrix_syc_axis1.csv`, `matrix_dec3_orth.csv` | the cross-behaviour arms |
| `transfer_matrices.png`, `transfer_delta.png` | heatmaps |
| `variants.png` | off-diagonal transfer per feature set |
| `transfer_sycophancy.png`, `transfer_sycophancy_delta.png` | matched vs mismatched persona basis |
