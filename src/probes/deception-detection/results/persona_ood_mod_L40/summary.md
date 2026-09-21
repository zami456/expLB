# More rollouts per persona did not help

Covers `results/persona_ood_mod_L20/`, `_L40/` and `_L44/`. Same pipeline and same six Liars'
Bench datasets as the `persona_ood_ext_L*` runs; the only change is the persona bank, from
`persona including knowlede denial/` to `persona knowledge denial modified/`.

## What actually changed in the folder

**The same 36 personas, re-extracted to a uniform rollout count.** No persona was added, removed
or renamed. Rollouts went from a range of 47–150 to a near-uniform 142–150, and each vector moved
in proportion to how many were added:

| persona | n old → new | ‖Δ‖ at L40 | cos(new, old) |
|---|---|---|---|
| `pc_plain_dealer` | 76 → 149 | 1.834 | 0.963 |
| `pc_enabler` | 51 → 146 | 1.638 | 0.970 |
| `pc_coverup_operator` | 47 → 120 | 1.510 | 0.977 |
| … | | | |
| `pc_ombudsman` | 148 → 149 | 0.028 | 1.000 |
| `default`, `pc_flatterer` | unchanged | 0.000 | 1.000 |

So this is a clean, well-controlled test of one thing: does better sampling of the same roles
improve the directions?

## It does not

Off-diagonal transfer AUROC, old → new:

| arm | L20 | L40 | L44 |
|---|---|---|---|
| **ext_axis1** | 0.9041 → **0.8839** | 0.9158 → **0.9066** | 0.9034 → **0.8905** |
| denial_axis1 | 0.8660 → 0.8501 | 0.9115 → 0.8999 | 0.9001 → 0.8873 |
| axis1 | 0.6511 → 0.6505 | 0.8319 → 0.8216 | 0.8260 → 0.8178 |
| ext_pc3 | 0.6837 → 0.6861 | 0.8655 → 0.8382 | 0.8799 → 0.8634 |
| denial_pc3 | 0.7101 → 0.6522 | 0.8718 → 0.8391 | 0.8704 → 0.8140 |
| ext_pc1 | 0.6613 → **0.6941** | 0.7680 → **0.7893** | 0.7801 → **0.7980** |

**Every leading arm gets worse at every depth.** The only consistent gain is `ext_pc1`, which is
a weak arm regardless (0.69–0.80 against the extended axis's 0.88–0.91).

The single largest movement is on **HP-KR**, the dataset the persona repair exists for. Zero-shot
AUROC for the extended axis, against the old run's 95% bootstrap half-width:

| | L20 | L40 | L44 |
|---|---|---|---|
| HP-KR | 0.882 → **0.805** (±0.042) | 0.912 → **0.876** (±0.031) | 0.894 → **0.849** (±0.037) |
| HP-C | 0.639 → 0.617 (±0.042) | 0.712 → 0.673 (±0.052) | 0.712 → 0.668 (±0.049) |
| CG | 0.969 → 0.963 (±0.016) | 0.912 → **0.931** (±0.030) | 0.859 → 0.870 (±0.042) |
| Average | 0.904 → 0.884 (±0.010) | 0.916 → 0.907 (±0.012) | 0.903 → 0.890 (±0.012) |

The HP-KR drops exceed the interval at layers 20 and 44 and sit at its edge at 40, so this is not
resampling noise. The pattern is also not random: **the harm-pressure datasets lose, CG gains.**

On balanced accuracy the picture is mixed rather than negative — `ext_axis1` gains at layer 20
(+0.052 over the five shared datasets) and is flat elsewhere, while `denial_axis1` loses 0.049 at
layer 40 and gains 0.020 at 44. Nothing here recovers the AUROC loss.

## Why: the group means converged

Individual persona vectors barely change magnitude. The **contrast** between groups shrinks a lot:

| layer | ‖deception axis‖ | ‖extended axis‖ | ‖denial axis‖ | mean ‖persona‖ |
|---|---|---|---|---|
| 40, old | 2.204 | 2.215 | 3.560 | 6.984 |
| 40, new | 1.842 | 1.892 | 3.179 | 6.869 |
| **ratio** | **0.836** | **0.854** | **0.892** | **0.983** |

Same at 20 and 44 (0.79–0.90 for the axes, against 0.98 for the personas themselves).

**The extra rollouts moved the deceptive and honest group means toward each other.** Each persona
keeps its magnitude but drifts toward the common assistant direction, because the rollouts added
beyond the first ~80 are less prototypical of the role. Sampling noise falls, and the role signal
falls faster. PC1's explained variance rises (0.393 → 0.446 at L20, 0.324 → 0.361 at L40), so the
persona cloud is *tidier* while the contrast that matters is *weaker*.

The denial axis shrinks least (0.892) and the deception axis most (0.836), which matches where the
rollouts were added: the knowledge-denial personas were already well sampled (129, 72, 137, 129)
while several deception personas were not (`pc_coverup_operator` 47, `pc_plausible_denier` 64,
`pc_narrative_manager` 70).

## What to do with this

**Keep `persona including knowlede denial/` as the bank of record.** `persona_ood_ext_L40` remains
the best configuration found: `ext_axis1` at 0.9158 off-diagonal AUROC, `denial_axis1` at 0.7709
off-diagonal balanced accuracy.

For the paper this is a short, useful methods note rather than a headline: **persona-vector quality
is not monotone in rollout count**, and the usual instinct — collect more samples — measurably hurt
the contrast here. If it is worth pursuing, the thing to test is whether selecting the most
role-typical rollouts (rather than averaging all of them) recovers the loss, which would confirm
the dilution account above.

## Files

Three folders, same layout as `persona_ood_ext_L*`: 15 `matrix_*.csv`, `summary.csv`,
`summary_balacc.csv`, `zero_shot.csv`, `zero_shot_operating.csv`, `liars_bench_comparison.csv`,
and figures `transfer_matrices.png`, `transfer_delta.png`, `variants.png`,
`transfer_sycophancy*.png`, `zero_shot_fig2.png`, plus the three-layer `zero_shot.png`.

Reproduce (per layer, with `--pooled_dir activations/pooled`, `pooled_L40`, `pooled_L44`):

```
python deception_detection/scripts/persona_ood.py run --layer 40 \
  --pooled_dir activations/pooled_L40 --out_dir results/persona_ood_mod_L40 \
  --vectors "persona knowledge denial modified"
python deception_detection/scripts/persona_ood.py zeroshot --layer 40 \
  --pooled_dir activations/pooled_L40 --out_dir results/persona_ood_mod_L40 \
  --vectors "persona knowledge denial modified"
```

Leave-one-out was not re-run; it was set aside as not the intended experiment.
