# D4's persona projection inside their pipeline, Liars' Bench labels (Qwen3-32B, Verdict)

Post hoc and descriptive. Every activation is projected onto D4's persona subspace (top 3 extended persona PCs of the Qwen3-32B bank, layers 26-30), and their pipeline runs unchanged except that the probe direction is trained by their own `train_deception_probe` on the thesis's five Liars' Bench sets (Llama-written replies of CG, HP-C, HP-KR, IT, ID; 3,908 deceptive, 7,144 honest) read by Qwen3-32B at their Core position, instead of on Core colluders. The Asymmetry and Sorted Concat classifiers are still fitted on Core runs (their design); Activation Alignment uses no direction. ± is their standard error over 10 seeds x 5 layers. Rows and design declared before any read, this analysis before any score (`E:/narcbench/logs/freeze.txt`).

## Reflection phase, Rose's label (their Table 1's setting)

| Method | Their Table 1 (their runs) | Their pipeline, our reads | D4 projection, Core labels | D4 projection, Liars' Bench labels | Random 3-dim band (p) |
|---|---|---|---|---|---|
| Sorted Concat | 0.728 ± 0.006 | 0.591 ± 0.004 | 0.359 ± 0.007 | 0.384 ± 0.005 | 0.612 (1.00) |
| Activation Alignment | 0.638 ± 0.002 | 0.624 ± 0.001 | 0.538 ± 0.004 | 0.538 ± 0.004 | 0.590 (0.37) |
| Deception Split | 0.512 ± 0.002 | 0.546 ± 0.002 | 0.574 ± 0.004 | 0.606 ± 0.007 | 0.576 (0.01) |
| Asymmetry Probe | 0.495 ± 0.003 | 0.418 ± 0.003 | 0.472 ± 0.002 | 0.466 ± 0.004 | 0.590 (0.99) |
| Peak Suspicion | 0.490 ± 0.004 | 0.363 ± 0.004 | 0.311 ± 0.007 | 0.253 ± 0.001 | 0.592 (1.00) |

## Reflection phase, Shi's label (both agents ACCEPT)

| Method | Their pipeline, our reads | D4 projection, Core labels | D4 projection, Liars' Bench labels | Random 3-dim band (p) |
|---|---|---|---|---|
| Sorted Concat | 0.528 ± 0.004 | 0.679 ± 0.006 | 0.680 ± 0.008 | 0.559 (0.01) |
| Activation Alignment | 0.269 ± 0.000 | 0.458 ± 0.002 | 0.458 ± 0.002 | 0.481 (0.27) |
| Deception Split | 0.496 ± 0.001 | 0.495 ± 0.003 | 0.494 ± 0.002 | 0.489 (0.03) |
| Asymmetry Probe | 0.745 ± 0.002 | 0.604 ± 0.004 | 0.617 ± 0.003 | 0.499 (0.01) |
| Peak Suspicion | 0.821 ± 0.003 | 0.706 ± 0.007 | 0.714 ± 0.004 | 0.590 (0.01) |

## Every phase, Liars' Bench labels: seed-0 value / random band (p), Rose's label then Shi's

| Label | Phase | Sorted Concat | Activation Alignment | Deception Split | Asymmetry Probe | Peak Suspicion |
|---|---|---|---|---|---|---|
| Rose | pre_decision | 0.51 / 0.54 (0.36) | 0.51 / 0.53 (0.10) | 0.52 / 0.54 (0.11) | 0.51 / 0.54 (0.18) | 0.50 / 0.56 (0.48) |
| Rose | verdict_start | 0.46 / 0.54 (0.97) | 0.48 / 0.52 (0.79) | 0.51 / 0.54 (0.32) | 0.48 / 0.55 (0.83) | 0.44 / 0.55 (0.97) |
| Rose | decision | 0.65 / 0.61 (0.01) | 0.58 / 0.62 (0.51) | 0.55 / 0.60 (0.48) | 0.63 / 0.60 (0.01) | 0.66 / 0.61 (0.01) |
| Rose | reflection | 0.38 / 0.61 (1.00) | 0.54 / 0.59 (0.37) | 0.61 / 0.58 (0.01) | 0.47 / 0.59 (0.99) | 0.25 / 0.59 (1.00) |
| Shi | pre_decision | 0.63 / 0.54 (0.01) | 0.52 / 0.51 (0.02) | 0.52 / 0.51 (0.01) | 0.60 / 0.52 (0.01) | 0.62 / 0.55 (0.01) |
| Shi | verdict_start | 0.56 / 0.53 (0.01) | 0.48 / 0.52 (0.67) | 0.53 / 0.52 (0.03) | 0.55 / 0.53 (0.01) | 0.56 / 0.55 (0.01) |
| Shi | decision | 0.80 / 0.64 (0.01) | 0.31 / 0.49 (0.83) | 0.26 / 0.50 (1.00) | 0.51 / 0.53 (0.08) | 0.87 / 0.66 (0.01) |
| Shi | reflection | 0.68 / 0.56 (0.01) | 0.46 / 0.48 (0.27) | 0.49 / 0.49 (0.03) | 0.62 / 0.50 (0.01) | 0.71 / 0.59 (0.01) |

p = (1 + random subspaces at or above the persona subspace) / 101; 40 cells, post hoc, unadjusted. `decision` is the accept/reject token itself and carries Shi's label.

Core held-out domain with the Liars' Bench direction (no Core row trains it; the two classifiers are fitted on the other domains' runs): Sorted Concat 0.98, Activation Alignment 0.93, Deception Split 0.90, Asymmetry Probe 0.97, Peak Suspicion 0.99.

## Dev check on Liars' Bench (the projected direction, seed 0, leave one set out)

| Held-out set | AUROC at their position | AUROC at the reply's last token |
|---|---|---|
| convince_me__plain | 0.783 | 0.813 |
| harm_refusal_choice__plain | 0.441 | 0.517 |
| harm_refusal_knowledge_report__plain | 0.965 | 0.941 |
| insider_trading_cadenza__plain | 0.972 | 0.959 |
| instruct_dishonesty__plain | 0.919 | 0.902 |
| all five (in-sample) | 0.934 | 0.925 |

## Reference: the direction trained at the reply's last token instead (reflection)

| Method | Rose's label | Shi's label |
|---|---|---|
| Sorted Concat | 0.367 ± 0.006 | 0.684 ± 0.007 |
| Activation Alignment | 0.538 ± 0.004 | 0.458 ± 0.002 |
| Deception Split | 0.588 ± 0.006 | 0.498 ± 0.002 |
| Asymmetry Probe | 0.445 ± 0.002 | 0.611 ± 0.003 |
| Peak Suspicion | 0.244 ± 0.003 | 0.701 ± 0.004 |

Check: the null's seed-0 persona values equal their main's seed-0 rows: True.

