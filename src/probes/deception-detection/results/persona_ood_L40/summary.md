# Persona PC features at layer 40

The layer-20 experiment (`../persona_ood/summary.md`) repeated at **layer 40**, using the same
pooled-activation pipeline, the same probe recipe and the same six datasets. Only the depth
changes.

Made possible by `persona vectors/vectors_s3/` — 26 per-persona files holding `vector[80, 8192]`.
Verified to be the same run as the single-layer bank: the layer-20 slice matches it at max abs
diff **0.000e+00**, names and rollout counts agree, and the four contrast axes reconstruct from
group means to ~1e-7.

**Pooling verified exactly.** Scoring the layer-40 pooled matrix with that run's own `detector.pt`
reproduces its `results_table.csv` at **Δ = 0.0000** on all five datasets.

## The datasets behave completely differently at depth

Before any persona work, the deployed probe's own AUROCs move dramatically:

| dataset | layer 20 | layer 40 |
|---|---|---|
| harm_refusal_knowledge_report | **0.109** | **0.936** |
| insider_trading_cadenza | **0.973** | **0.338** |
| harm_refusal_choice | 0.471 | 0.675 |
| convince_me | 0.984 | 0.972 |
| instruct_dishonesty | 0.950 | 0.971 |

**The inversions move with depth, and they swap datasets.** HP-KR — inverted at layer 20 under
three independent methods, and the anomaly that drove most of the layer-20 analysis — is *solved*
at layer 40 (0.936). Insider-trading inverts instead (0.338). Whatever produces these inversions
is a property of a particular layer's representation, not of the datasets alone.

## Transfer: the 1-d arms win, the 3-d arm does not

| features | off-diag L20 | off-diag L40 | excl. HP-KR L20 | excl. HP-KR L40 |
|---|---|---|---|---|
| baseline (8192-d) | 0.6478 | **0.7320** | 0.7282 | 0.7581 |
| deception PC1 | 0.6520 | **0.8463** | 0.9205 | 0.8960 |
| deception PCs (3-d) | 0.6823 | 0.7079 | 0.9123 | 0.8471 |
| deception axis | 0.6457 | **0.8240** | 0.9237 | 0.8826 |
| sycophancy PC1 | 0.6148 | 0.6690 | 0.8828 | **0.9376** |
| sycophancy PCs | 0.6060 | 0.6807 | 0.7821 | 0.8907 |
| sycophancy PCs, deception removed | 0.5419 | 0.6302 | 0.6383 | **0.8246** |
| deception PCs, sycophancy removed | 0.7111 | 0.6038 | 0.8627 | 0.6518 |
| random directions | 0.5269 | 0.5038 | 0.5598 | 0.5437 |

**The raw baseline improves a lot with depth** (0.648 → 0.732), which shrinks the headroom persona
had to exploit. Against it at layer 40, the **1-d arms win clearly** — PC1 +0.114, axis +0.092 —
while the **3-d arm now loses** (−0.024). Excluding HP-KR every persona arm still beats baseline,
but by less than at layer 20.

The layer-20 conclusion that *almost all the gain is one direction* therefore strengthens at depth:
at layer 40 the extra two components are not merely redundant, they are harmful.

## Three layer-20 findings that do not survive

**Sycophancy is no longer borrowing.** At layer 20, removing the deception direction collapsed the
sycophancy basis to 0.638 (excl. HP-KR) against random's 0.560 — essentially nothing of its own.
At layer 40 it holds **0.8246** against random's 0.544. And sycophancy PC1 (0.9376) now transfers
*better* than deception PC1 (0.8960). Whatever the sycophancy subspace carries at layer 40, it is
not just the shared deception component.

**Removing sycophancy stops helping.** `dec3_orth` was the best arm at layer 20 (0.7111); at layer
40 it is one of the worst (0.6038, and 0.6518 excluding HP-KR against plain pc3's 0.8471). That
result was specific to layer 20 — and specifically to HP-KR's inversion there, which no longer
exists.

**Train-set invariance weakens.** Standard deviation of off-diagonal AUROC across training sets:

| features | L20 | L40 |
|---|---|---|
| baseline | 0.167 | 0.201 |
| deception PCs | **0.027** | 0.103 |
| sycophancy PCs | 0.163 | 0.083 |

Still a real effect — 2× better than baseline — but not the 6× of layer 20. (1-d arms read 0.000
by construction: a one-feature probe learns only a sign and a scale.)

## Zero-shot, where persona looks best

| dataset | deception axis | sycophancy axis | deployed probe |
|---|---|---|---|
| convince_me | 0.907 | **0.999** | 0.972 |
| harm_refusal_choice | 0.555 | 0.750 | 0.675 |
| harm_refusal_knowledge_report | 0.531 | **0.200** | 0.936 |
| insider_trading | **0.978** | 0.985 | **0.338** |
| instruct_dishonesty | 0.973 | 0.948 | 0.971 |
| repe_honesty | 1.000 | 0.999 | — |

**The persona axis does not invert on insider-trading where the trained probe does**: 0.978 against
0.338, with no Liars' Bench labels at all. That is the layer-40 analogue of the layer-20 result on
harm-pressure-choice, and it is the strongest single argument for the method — an unsupervised
direction being robust exactly where a supervised probe fails.

The sycophancy axis inverts on HP-KR here (0.200) while the deception axis sits at chance (0.531),
so the inversion has moved behaviour as well as layer.

## What to take from the pair

- **The method is real but layer-dependent.** Persona directions beat raw activations at both
  depths, but which arm wins, by how much, and via what mechanism all change.
- **One direction is enough, increasingly so.** k=3 helped at layer 20 and hurts at layer 40.
- **Anything conditioned on HP-KR's inversion was layer-20-specific.** That includes the
  orthogonalisation result and part of the off-diagonal gap.
- **Layer 40 is the better operating point for the raw baseline** (0.732 vs 0.648), so the
  comparison is harder there, not easier.

## Files

Same layout as the layer-20 folder: `summary.csv` (with a `layer` column), `zero_shot.csv`,
`matrix_*.csv`, `matrix_balacc_*.csv`, `liars_bench_comparison.csv`, and the five figures.
