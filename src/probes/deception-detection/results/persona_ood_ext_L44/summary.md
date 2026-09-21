# Layer 44: the depth curve turns over

Third depth point, same pipeline as `persona_ood_ext_L20/` and `_L40/`. Llama-3.3-70B-Instruct,
six Liars' Bench datasets, 36 personas from `persona including knowlede denial/`.

## Why it was worth running

With only layers 20 and 40 every headline number improved with depth, which argues for going
deeper. It does not hold. **Layer 40 is a peak, not a trend**, on both metrics and for every
leading arm.

| off-diagonal AUROC | L20 | **L40** | L44 |
|---|---|---|---|
| **ext_axis1** | 0.9041 | **0.9158** | 0.9034 |
| denial_axis1 | 0.8660 | **0.9115** | 0.9001 |
| ext_pc3 | 0.6837 | 0.8655 | **0.8799** |
| baseline (8192-d) | 0.6478 | **0.7320** | 0.7156 |

| off-diagonal balanced accuracy @1% FPR | L20 | **L40** | L44 |
|---|---|---|---|
| denial_axis1 | 0.5941 | **0.7709** | 0.7342 |
| **ext_axis1** | 0.6507 | **0.7446** | 0.7216 |
| baseline (8192-d) | 0.5545 | 0.5920 | 0.5920 |

Zero-shot at the 1% FPR control threshold, averaged over the five datasets shared with the
Liars' Bench paper, tells the same story:

| | L20 | **L40** | L44 |
|---|---|---|---|
| knowledge-denial axis, balacc | 0.559 | **0.725** | 0.683 |
| knowledge-denial axis, recall | 0.305 | **0.487** | 0.395 |
| extended axis, balacc | 0.591 | **0.694** | 0.666 |
| extended axis, AUROC | 0.887 | **0.899** | 0.884 |

**Report layer 40.** Layer 44 is uniformly a little behind it and is worth keeping only as the
evidence that 40 is a maximum rather than the start of a ramp.

## What reproduces at 44

- **`ext_axis1` is the best arm** on AUROC (0.9034) and second on balanced accuracy, as at both
  other depths.
- **The 1-d contrast beats the 3-d subspace**: 0.9034 against 0.8799 for `ext_pc3`, and
  `ext_pc1` is far behind at 0.7801. The margin narrows with depth (0.220 → 0.050 → 0.024) but
  never reverses.
- **`denial_axis1` leads on balanced accuracy** (0.7342 vs 0.7216), as at layer 40. The arm that
  wins depends on the metric at both depths.
- **The persona-bank repair holds**: HP-KR zero-shot 0.589 under the published 8v7 grouping,
  0.894 with the extended axis, 0.908 with knowledge-denial alone.
- **Pooling reproduces the run exactly**, Δ = 0.0000 on all five scored datasets, as at 20 and 40.

## What does not

- **CG degrades monotonically with depth** under every persona arm: zero-shot extended axis
  0.969 → 0.912 → 0.859. It is the only dataset that does. *(Update: this is a pooling artifact.
  Read at the per-example maximum instead of the mean, CG does not degrade — 0.954 → 0.993 →
  0.991 — because at depth its lie is carried by a few tokens that the mean dilutes. See
  `../persona_positions_L40/summary.md`.)*
- **`pc3` on the original bank keeps falling**: 0.6792 → 0.6394 → 0.6166, and `dec3_orth` with it
  (0.6549 → 0.6015 → 0.5295).
- **The sycophancy arms jump at 44**: `syc_pc1` reaches 0.8586 off-diagonal, up from 0.6618 at
  layer 40 and now above the deception `pc1`. Unexplained, and worth a footnote rather than a
  claim -- the sycophancy axis is still inverted on HP-KR at layer 20 (0.043) and near chance at
  44 (0.480), so this is not the sycophancy direction becoming a deception detector in general.

## A caveat about the layer-44 run itself

The trained RepE probe in `results/XXXX-9-llama-all-L44/` scores **0.289 on insider-trading** --
inverted, where the same probe reaches ~0.98 at layers 20 and 40. That is the benchmark probe,
not a persona arm, and the pooled activations reproduce it exactly, so it is a property of the
layer rather than a pipeline fault. It is a reason to be careful about treating layer 44 as
simply "layer 40, deeper".

## Files

Same layout as the other two folders: `matrix_*.csv` (15 arms), `summary.csv`,
`summary_balacc.csv`, `zero_shot.csv`, `zero_shot_operating.csv`, `liars_bench_comparison.csv`,
and figures `transfer_matrices.png`, `transfer_delta.png`, `variants.png`,
`transfer_sycophancy*.png`, `zero_shot_fig2.png`. `zero_shot.png` is the three-layer comparison
and is written into all three folders.

Reproduce:

```
ACTS_CACHE_DIR=E:/liars-bench-acts/cache python deception_detection/scripts/pool_activations.py \
  --config-file deception_detection/scripts/configs/XXXX-9-llama-all-L44.yaml \
  --out_dir activations/pooled_L44
python deception_detection/scripts/persona_ood.py verify \
  --run_folder results/XXXX-9-llama-all-L44 --pooled_dir activations/pooled_L44
python deception_detection/scripts/persona_ood.py run --layer 44 \
  --pooled_dir activations/pooled_L44 --out_dir results/persona_ood_ext_L44 \
  --vectors "persona including knowlede denial"
python deception_detection/scripts/persona_ood.py zeroshot --layer 44 \
  --pooled_dir activations/pooled_L44 --out_dir results/persona_ood_ext_L44 \
  --vectors "persona including knowlede denial"
```
