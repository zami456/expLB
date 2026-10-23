# D4's persona projection inside their pipeline (Qwen3-32B, Verdict)

Post hoc and descriptive. Every Core and Verdict activation is projected onto D4's persona subspace (the top 3 extended persona PCs of a Qwen3-32B persona bank, s3, fitted at each of layers 26-30), then their `reproduce_LR_core_to_verdict.py` runs unchanged: their probe, five methods, Core training, 10 seeds. ± is their statistic (standard error over 10 seeds x 5 layers). Declared in `E:/narcbench/logs/freeze.txt` before any projected score.

## Reflection phase, Rose's label (their Table 1's setting)

| Method | Their Table 1 (their runs) | Their pipeline, our reads | + D4 persona projection | Random 3-dim subspaces: band (p) |
|---|---|---|---|---|
| Sorted Concat | 0.728 ± 0.006 | 0.591 ± 0.004 | 0.359 ± 0.007 | 0.605 (p 1.000) |
| Activation Alignment | 0.638 ± 0.002 | 0.624 ± 0.001 | 0.538 ± 0.004 | 0.590 (p 0.366) |
| Deception Split | 0.512 ± 0.002 | 0.546 ± 0.002 | 0.574 ± 0.004 | 0.571 (p 0.030) |
| Asymmetry Probe | 0.495 ± 0.003 | 0.418 ± 0.003 | 0.472 ± 0.002 | 0.576 (p 1.000) |
| Peak Suspicion | 0.490 ± 0.004 | 0.363 ± 0.004 | 0.311 ± 0.007 | 0.611 (p 1.000) |

Their Table 1 comes from their own Qwen3-32B agent runs (not released); ours are Shi et al.'s Gemma-4-31B runs read by Qwen3-32B with a compact context, so the matched comparison is the projected column against the unprojected one. The band is the 97.5th percentile of 100 random orthonormal 3-dim subspaces through the same pipeline (seed 0; p against the persona subspace's seed-0 value).

## Reflection phase, Shi's label (both agents ACCEPT)

| Method | Their pipeline, our reads | + D4 persona projection | Random 3-dim subspaces: band (p) |
|---|---|---|---|
| Sorted Concat | 0.528 ± 0.004 | 0.679 ± 0.006 | 0.531 (p 0.010) |
| Activation Alignment | 0.269 ± 0.000 | 0.458 ± 0.002 | 0.481 (p 0.267) |
| Deception Split | 0.496 ± 0.001 | 0.495 ± 0.003 | 0.490 (p 0.020) |
| Asymmetry Probe | 0.745 ± 0.002 | 0.604 ± 0.004 | 0.509 (p 0.010) |
| Peak Suspicion | 0.821 ± 0.003 | 0.706 ± 0.007 | 0.603 (p 0.010) |

## Every phase (projected; Rose's label, Shi's in brackets)

| Phase | Sorted Concat | Activation Alignment | Deception Split | Asymmetry Probe | Peak Suspicion |
|---|---|---|---|---|---|
| Core held-out domain | 0.98 (0.98) | 0.93 (0.93) | 0.89 (0.89) | 0.97 (0.97) | 0.99 (0.99) |
| pre_decision | 0.49 (0.61) | 0.51 (0.52) | 0.52 (0.49) | 0.53 (0.60) | 0.52 (0.64) |
| verdict_start | 0.47 (0.55) | 0.48 (0.48) | 0.51 (0.51) | 0.50 (0.55) | 0.48 (0.57) |
| decision | 0.66 (0.85) | 0.58 (0.31) | 0.56 (0.28) | 0.64 (0.51) | 0.66 (0.86) |
| reflection | 0.36 (0.68) | 0.54 (0.46) | 0.57 (0.49) | 0.47 (0.60) | 0.31 (0.71) |

## Seed-0 value against the random 3-dim band, every phase (value / band, p)

| Label | Phase | Sorted Concat | Activation Alignment | Deception Split | Asymmetry Probe | Peak Suspicion |
|---|---|---|---|---|---|---|
| Rose | pre_decision | 0.49 / 0.54 (0.52) | 0.51 / 0.53 (0.10) | 0.52 / 0.53 (0.07) | 0.53 / 0.53 (0.04) | 0.52 / 0.57 (0.23) |
| Rose | verdict_start | 0.47 / 0.54 (0.80) | 0.48 / 0.52 (0.79) | 0.51 / 0.54 (0.32) | 0.50 / 0.54 (0.35) | 0.48 / 0.55 (0.66) |
| Rose | decision | 0.66 / 0.61 (0.01) | 0.58 / 0.62 (0.51) | 0.56 / 0.59 (0.40) | 0.64 / 0.59 (0.01) | 0.66 / 0.61 (0.01) |
| Rose | reflection | 0.36 / 0.61 (1.00) | 0.54 / 0.59 (0.37) | 0.57 / 0.57 (0.03) | 0.47 / 0.58 (1.00) | 0.31 / 0.61 (1.00) |
| Shi | pre_decision | 0.61 / 0.53 (0.01) | 0.52 / 0.51 (0.02) | 0.49 / 0.51 (0.26) | 0.60 / 0.53 (0.01) | 0.64 / 0.58 (0.01) |
| Shi | verdict_start | 0.55 / 0.52 (0.01) | 0.48 / 0.52 (0.67) | 0.51 / 0.52 (0.20) | 0.55 / 0.53 (0.01) | 0.57 / 0.55 (0.01) |
| Shi | decision | 0.85 / 0.66 (0.01) | 0.31 / 0.49 (0.83) | 0.28 / 0.50 (0.99) | 0.51 / 0.52 (0.05) | 0.86 / 0.66 (0.01) |
| Shi | reflection | 0.68 / 0.53 (0.01) | 0.46 / 0.48 (0.27) | 0.49 / 0.49 (0.02) | 0.60 / 0.51 (0.01) | 0.71 / 0.60 (0.01) |

p = (1 + random subspaces at or above the persona subspace) / 101, so 0.01 means above all 100; 40 cells, post hoc, unadjusted.

The `decision` position is the accept/reject token itself and carries Shi's label.

## The Qwen3-32B persona bank

- Variant s3 (judge score 3 unless a role falls under 20 kept replies); judge share fully in role by group: deception 0.62, honesty 0.78, sycophancy 0.69, anti_sycophancy 0.82, assistant_anchor nan, knowledge_denial 0.44, sincere_ignorance 0.48, plain_assertion 0.61.
- Variance explained by the 3 PCs per layer: L26 0.41/0.20/0.10; L27 0.42/0.19/0.10; L28 0.42/0.19/0.09; L29 0.44/0.18/0.09; L30 0.45/0.18/0.09.
- AWQ (their Core) vs bf16 (the bank and our reads), projected coordinates of 20 Transfer rows, Pearson r of PC1-3: L26 0.994/0.979/0.994; L27 0.992/0.975/0.993; L28 0.992/0.969/0.994; L29 0.992/0.970/0.991; L30 0.990/0.979/0.989.
- Check: the identity in place of the basis reproduces the unprojected CSVs byte for byte (GB5); the null's seed-0 persona values equal their script's seed-0 rows: True.

