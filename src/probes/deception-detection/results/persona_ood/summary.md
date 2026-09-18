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

## Cross-behaviour control: the sycophancy basis

Same transfer test, persona basis swapped for the sycophancy contrast (5 sycophancy vs 5
anti-sycophancy personas, recovered from the sycophancy axis at residual 2.7e-07). This asks
whether the gain is *deception-specific* persona geometry or persona structure in general —
a sharper control than random projections, which only rule out dimensionality reduction.

The two behaviours are **not independent**: |cos(deception axis, sycophancy axis)| = **0.5402**.
So a raw sycophancy arm would score well through the shared component and prove nothing; the
orthogonalised arms are what make the comparison readable.

| features | dim | off-diagonal | excl. HP-KR | diagonal |
|---|---|---|---|---|
| baseline (pooled activations) | 8192 | 0.6478 | 0.7282 | 0.9564 |
| deception PCs | 3 | 0.6823 | 0.9123 | 0.9184 |
| deception axis | 1 | 0.6457 | 0.9237 | 0.9188 |
| **deception PCs, sycophancy removed** | 3 | **0.7111** | 0.8627 | 0.9064 |
| sycophancy PCs | 3 | 0.6060 | 0.7821 | 0.9285 |
| sycophancy axis | 1 | 0.6144 | 0.8816 | 0.8898 |
| **sycophancy PCs, deception removed** | 3 | **0.5419** | 0.6383 | 0.8862 |
| random directions | 3 | 0.5269 | 0.5598 | 0.7874 |

**The basis is behaviour-specific.** Sycophancy PCs reach 0.606 against deception PCs' 0.682
(0.782 vs 0.912 excluding HP-KR). Swapping in an adjacent behaviour costs real performance.

**And what transfer it has is borrowed.** Removing the deception direction collapses the
sycophancy arm to **0.542**, against random projections' 0.527. Essentially nothing survives —
the sycophancy basis was working *through* its 0.54 overlap with deception, not through any
structure of its own. This is the pre-registered reading that the evidence supports.

### The unanticipated result

Removing the *sycophancy* axis from the *deception* basis gives the **best off-diagonal score in
the whole study, 0.7111** — better than the deception PCs it was derived from (0.6823). The shared
component was not merely redundant; it was actively hurting cross-dataset transfer.

Per test set, `dec3_orth` minus `pc3`:

| test set | Δ |
|---|---|
| HP-KR | **+0.224** |
| HP-C | +0.064 |
| IT | +0.055 |
| RepE | +0.017 |
| ID | −0.021 |
| CG | **−0.166** |

The two extremes are interpretable and point the same way. **CG is the convincing game** — a
persuasion task — and the sycophancy axis scores **0.9995 zero-shot** on it, better than the
deception axis's 0.976. Strip sycophancy out and that signal goes with it. **HP-KR**, whose
deceptive response is a deferential "I don't know" under pressure, reads as *sycophantic*; removing
that component moves it from 0.184 toward chance at 0.408.

So the aggregate win is real but narrow: `dec3_orth` leads on all six datasets **only because HP-KR
dominates the aggregate**, and it loses to plain `pc3` once HP-KR is excluded (0.863 vs 0.912). The
honest reading is not "always remove sycophancy" but that *the deception axis is entangled with
sycophancy in a way that helps on persuasion tasks and hurts on deference tasks*.

### Caveat on the control's strength

The planned sanity check was that the sycophancy axis should be clearly worse than the deception
axis on RepE, which is a pure honest/deceptive contrast. It is worse, but only slightly — 0.978
against 0.992. At |cos| = 0.54 the two axes are entangled enough that the raw sycophancy arm is a
weak control on its own. The orthogonalised arm is the one carrying the conclusion.

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
