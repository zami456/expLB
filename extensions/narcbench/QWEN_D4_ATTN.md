# Attention probe on D4's persona projection inside their pipeline (Qwen3-32B, Verdict)

Post hoc and descriptive. An EleutherAI-style attention probe (their code's architecture and training: zero-initialised queries, a learned position weight per head, heads summed, LBFGS, weight decay by 5-fold cross-validation) reads the k = 3 persona-projected tokens of each scored turn, read by Qwen3-32B with the full context. It is trained on their Core training rows (colluder vs honest, their selection) from our read of the released Core transcripts, and enters their pipeline unchanged through its pooled vectors. ± is their standard error over 10 seeds x 5 layers. Rows, basis, reader and this analysis were declared before any read or score (`E:/narcbench/logs/freeze.txt`).

## Reflection phase, Rose's label (their Table 1's setting)

| Method | Table 1 | Unprojected | Projected, last token (their Core) | Span last token | Span mean | Attention H=1 | Attention H=2 | Attention H=8 | Random 3-dim, H=8: band (p) |
|---|---|---|---|---|---|---|---|---|---|
| Sorted Concat | 0.728 ± 0.006 | 0.591 ± 0.004 | 0.359 ± 0.007 | 0.438 ± 0.007 | 0.626 ± 0.003 | 0.492 ± 0.009 | 0.554 ± 0.011 | 0.522 ± 0.012 | 0.558 (0.04) |
| Activation Alignment | 0.638 ± 0.002 | 0.624 ± 0.001 | 0.538 ± 0.004 | 0.541 ± 0.005 | 0.551 ± 0.003 | 0.568 ± 0.004 | 0.600 ± 0.004 | 0.615 ± 0.004 | 0.617 (0.05) |
| Deception Split | 0.512 ± 0.002 | 0.546 ± 0.002 | 0.574 ± 0.004 | 0.586 ± 0.005 | 0.517 ± 0.005 | 0.493 ± 0.009 | 0.516 ± 0.006 | 0.499 ± 0.006 | 0.559 (0.91) |
| Asymmetry Probe | 0.495 ± 0.003 | 0.418 ± 0.003 | 0.472 ± 0.002 | 0.498 ± 0.002 | 0.534 ± 0.007 | 0.461 ± 0.012 | 0.516 ± 0.008 | 0.510 ± 0.007 | 0.566 (0.80) |
| Peak Suspicion | 0.490 ± 0.004 | 0.363 ± 0.004 | 0.311 ± 0.007 | 0.322 ± 0.008 | 0.560 ± 0.009 | 0.436 ± 0.014 | 0.515 ± 0.010 | 0.519 ± 0.009 | 0.568 (0.36) |

## Reflection phase, Shi's label

| Method | Span last token | Span mean | Attention H=1 | Attention H=2 | Attention H=8 | Random 3-dim, H=8: band (p) |
|---|---|---|---|---|---|---|
| Sorted Concat | 0.639 ± 0.007 | 0.507 ± 0.004 | 0.536 ± 0.007 | 0.507 ± 0.007 | 0.499 ± 0.012 | 0.553 (0.07) |
| Activation Alignment | 0.459 ± 0.002 | 0.356 ± 0.001 | 0.446 ± 0.003 | 0.437 ± 0.006 | 0.454 ± 0.005 | 0.450 (0.02) |
| Deception Split | 0.492 ± 0.003 | 0.368 ± 0.001 | 0.405 ± 0.008 | 0.409 ± 0.008 | 0.415 ± 0.004 | 0.493 (1.00) |
| Asymmetry Probe | 0.585 ± 0.005 | 0.535 ± 0.001 | 0.558 ± 0.006 | 0.577 ± 0.006 | 0.563 ± 0.006 | 0.540 (0.05) |
| Peak Suspicion | 0.690 ± 0.008 | 0.848 ± 0.002 | 0.718 ± 0.007 | 0.730 ± 0.012 | 0.692 ± 0.009 | 0.646 (0.01) |

## Before the verdict (the agent's last own turn), Shi's label

| Method | Span last token | Span mean | Attention H=1 | Attention H=2 | Attention H=8 | Random 3-dim, H=8: band (p) |
|---|---|---|---|---|---|---|
| Sorted Concat | 0.588 ± 0.004 | 0.488 ± 0.002 | 0.506 ± 0.006 | 0.515 ± 0.008 | 0.522 ± 0.006 | 0.532 (0.15) |
| Activation Alignment | 0.503 ± 0.002 | 0.455 ± 0.000 | 0.495 ± 0.005 | 0.484 ± 0.006 | 0.483 ± 0.005 | 0.514 (0.20) |
| Deception Split | 0.521 ± 0.005 | 0.456 ± 0.002 | 0.467 ± 0.005 | 0.462 ± 0.004 | 0.481 ± 0.004 | 0.529 (0.84) |
| Asymmetry Probe | 0.561 ± 0.004 | 0.477 ± 0.003 | 0.508 ± 0.008 | 0.533 ± 0.007 | 0.557 ± 0.004 | 0.531 (0.01) |
| Peak Suspicion | 0.588 ± 0.003 | 0.529 ± 0.005 | 0.582 ± 0.008 | 0.615 ± 0.008 | 0.611 ± 0.005 | 0.550 (0.01) |

## Before the verdict (the agent's last own turn), Rose's label

| Method | Span last token | Span mean | Attention H=1 | Attention H=2 | Attention H=8 | Random 3-dim, H=8: band (p) |
|---|---|---|---|---|---|---|
| Sorted Concat | 0.575 ± 0.002 | 0.516 ± 0.002 | 0.458 ± 0.005 | 0.500 ± 0.007 | 0.532 ± 0.007 | 0.535 (0.01) |
| Activation Alignment | 0.434 ± 0.001 | 0.498 ± 0.000 | 0.486 ± 0.003 | 0.498 ± 0.004 | 0.476 ± 0.006 | 0.519 (0.55) |
| Deception Split | 0.432 ± 0.003 | 0.498 ± 0.001 | 0.414 ± 0.004 | 0.467 ± 0.005 | 0.468 ± 0.006 | 0.525 (0.75) |
| Asymmetry Probe | 0.488 ± 0.002 | 0.530 ± 0.001 | 0.450 ± 0.005 | 0.522 ± 0.008 | 0.539 ± 0.005 | 0.527 (0.01) |
| Peak Suspicion | 0.596 ± 0.002 | 0.593 ± 0.003 | 0.530 ± 0.004 | 0.590 ± 0.007 | 0.594 ± 0.005 | 0.541 (0.01) |

## Decision token (the verdict turn), Rose's label

| Method | Span last token | Span mean | Attention H=1 | Attention H=2 | Attention H=8 | Random 3-dim, H=8: band (p) |
|---|---|---|---|---|---|---|
| Sorted Concat | 0.662 ± 0.003 | 0.629 ± 0.001 | 0.534 ± 0.006 | 0.562 ± 0.011 | 0.550 ± 0.011 | 0.589 (0.03) |
| Activation Alignment | 0.574 ± 0.002 | 0.521 ± 0.001 | 0.496 ± 0.006 | 0.537 ± 0.007 | 0.545 ± 0.006 | 0.582 (0.36) |
| Deception Split | 0.564 ± 0.003 | 0.537 ± 0.001 | 0.457 ± 0.006 | 0.506 ± 0.006 | 0.491 ± 0.007 | 0.566 (0.95) |
| Asymmetry Probe | 0.634 ± 0.001 | 0.592 ± 0.002 | 0.510 ± 0.009 | 0.561 ± 0.010 | 0.564 ± 0.009 | 0.582 (0.06) |
| Peak Suspicion | 0.670 ± 0.004 | 0.650 ± 0.001 | 0.568 ± 0.010 | 0.602 ± 0.012 | 0.607 ± 0.010 | 0.603 (0.02) |

pre_decision and verdict_start use the same span (the agent's last own turn), so verdict_start is in the JSON only. p = (1 + random subspaces at or above) / 101 (seed 0, layer mean); unadjusted, post hoc.

## Core held-out domain (leave one domain out, probe refitted per domain)

| Method | Span last token (their CV) | Span mean (their CV) | Attention H=1 | Attention H=2 | Attention H=8 |
|---|---|---|---|---|---|
| Sorted Concat | 0.959 ± 0.003 | 0.995 ± 0.001 | 0.986 ± 0.002 | 0.979 ± 0.003 | 0.833 ± 0.009 |
| Activation Alignment | 0.906 ± 0.005 | 0.995 ± 0.001 | 0.964 ± 0.002 | 0.962 ± 0.005 | 0.981 ± 0.003 |
| Deception Split | 0.950 ± 0.004 | 1.000 ± 0.000 | 0.972 ± 0.003 | 0.986 ± 0.003 | 0.980 ± 0.003 |
| Asymmetry Probe | 0.976 ± 0.003 | 1.000 ± 0.000 | 0.989 ± 0.001 | 0.993 ± 0.002 | 0.986 ± 0.002 |
| Peak Suspicion | 0.968 ± 0.002 | 0.976 ± 0.001 | 0.975 ± 0.003 | 0.977 ± 0.003 | 0.971 ± 0.003 |

## Attention entropy (H = 8, seed 0; 1 = uniform over the span)

| Layer | reflection | decision | pre_decision | core |
|---|---|---|---|---|
| L26 | 0.48 | 0.52 | 0.57 | 0.44 |
| L27 | 0.43 | 0.44 | 0.46 | 0.40 |
| L28 | 0.36 | 0.38 | 0.39 | 0.33 |
| L29 | 0.07 | 0.07 | 0.08 | 0.07 |
| L30 | 0.40 | 0.43 | 0.44 | 0.39 |

Weight decay chosen: 0 x38, 0.001 x77, 0.01 x30, 0.1 x5.
Checks: their training rows reproduced True; pooled-vector identity max error 2.1e-04; the null was rerun after a power loss, its seed-0 persona fits equal run's True.

