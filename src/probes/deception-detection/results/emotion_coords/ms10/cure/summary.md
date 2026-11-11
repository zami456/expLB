# Curing instructed deception before training on it (Llama-3.3-70B, layers 36, 40, 44)

Design: `emotion_coords/ms10_cure_spec.py`, frozen (sha256 7E727EA24D90) before any erasure was scored. The thesis's raw probe D1 trained on InstrDec's 600 training rows after each erasure (fitted on those rows alone), scored on every test split with `ms10`'s draws. The unerased probe reproduces `ms10_thesis`'s D1 (max |difference| 1.1102230246251565e-16).

## Predictions (deception transfer: the other four deception sets)

- **C1** (nuisance erasure minus no erasure, deception transfer, averaged over the 7 ranks: CI above 0 at >= 2 of 3 layers): 0 of 3 (L36 -0.046 [-0.071, -0.020]; L40 -0.047 [-0.076, -0.023]; L44 -0.043 [-0.065, -0.020]): **fails**.
- **C2** (nuisance erasure minus random erasure at the same rank, averaged over the ranks: CI above 0 at >= 2 of 3): 0 of 3 (L36 -0.048 [-0.073, -0.023]; L40 -0.047 [-0.075, -0.022]; L44 -0.042 [-0.064, -0.019]): **fails**.
- **C3** (nuisance erasure minus the random-label LEACE loop at the same rank, averaged over the ranks: CI above 0 at >= 2 of 3): 3 of 3 (L36 +0.108 [+0.070, +0.149]; L40 +0.109 [+0.071, +0.147]; L44 +0.110 [+0.070, +0.151]): **holds**.
- **C4** (nuisance erasure (averaged over the ranks) minus keeping the top 170 PCs: CI above 0 at >= 2 of 3): 0 of 3 (L36 -0.047 [-0.077, -0.016]; L40 -0.050 [-0.081, -0.018]; L44 -0.046 [-0.080, -0.016]): **fails**.

The same comparisons in every region (averaged over the ranks; mean over the layers, and the layers whose CI is above 0 / below 0):

| comparison | deception | new deception | sycophancy | in-distribution |
|---|---|---|---|---|
| C1 | -0.045 (0 / 3) | -0.056 (0 / 2) | -0.051 (0 / 2) | -0.003 (0 / 0) |
| C2 | -0.046 (0 / 3) | -0.056 (0 / 2) | -0.050 (0 / 2) | -0.003 (0 / 0) |
| C3 | +0.109 (3 / 0) | +0.124 (3 / 0) | +0.001 (0 / 0) | +0.044 (3 / 0) |
| C4 | -0.048 (0 / 3) | -0.029 (0 / 1) | -0.037 (0 / 1) | -0.003 (0 / 1) |

## The user's stopping test: a probe on balanced random labels of the 600 training rows

| layer | held-out AUROC (mean of 5 seeds) | in-sample AUROC |
|---|---|---|
| 36 | 0.478 (range 0.399-0.517) | 1.000 |
| 40 | 0.479 (range 0.397-0.511) | 1.000 |
| 44 | 0.476 (range 0.400-0.504) | 1.000 |

## deception: AUROC by rank (mean over the three layers)

| variant | k = 0 | 5 | 10 | 20 | 50 | 100 | 200 | 400 | k = 170 |
|---|---|---|---|---|---|---|---|---|---|
| nuisance | 0.824 | 0.820 | 0.776 | 0.779 | 0.770 | 0.779 | 0.760 | 0.767 | - |
| total | 0.824 | 0.397 | 0.407 | 0.514 | 0.482 | 0.485 | 0.488 | 0.575 | - |
| random | 0.824 | 0.824 | 0.824 | 0.824 | 0.823 | 0.823 | 0.824 | 0.827 | - |
| random-label LEACE | 0.824 | 0.811 | 0.789 | 0.754 | 0.684 | 0.593 | 0.515 | 0.541 | - |
| keep | 0.824 | 0.906 | 0.904 | 0.891 | 0.847 | 0.830 | 0.809 | 0.771 | 0.827 |

## new deception: AUROC by rank (mean over the three layers)

| variant | k = 0 | 5 | 10 | 20 | 50 | 100 | 200 | 400 | k = 170 |
|---|---|---|---|---|---|---|---|---|---|
| nuisance | 0.899 | 0.898 | 0.840 | 0.845 | 0.830 | 0.840 | 0.824 | 0.826 | - |
| total | 0.899 | 0.501 | 0.506 | 0.574 | 0.524 | 0.498 | 0.493 | 0.515 | - |
| random | 0.899 | 0.899 | 0.899 | 0.900 | 0.899 | 0.898 | 0.899 | 0.897 | - |
| random-label LEACE | 0.899 | 0.875 | 0.854 | 0.826 | 0.725 | 0.623 | 0.556 | 0.573 | - |
| keep | 0.899 | 0.904 | 0.898 | 0.873 | 0.881 | 0.873 | 0.860 | 0.831 | 0.872 |

## sycophancy: AUROC by rank (mean over the three layers)

| variant | k = 0 | 5 | 10 | 20 | 50 | 100 | 200 | 400 | k = 170 |
|---|---|---|---|---|---|---|---|---|---|
| nuisance | 0.600 | 0.610 | 0.552 | 0.534 | 0.537 | 0.540 | 0.541 | 0.533 | - |
| total | 0.600 | 0.460 | 0.464 | 0.436 | 0.414 | 0.417 | 0.415 | 0.429 | - |
| random | 0.600 | 0.599 | 0.601 | 0.599 | 0.600 | 0.602 | 0.595 | 0.601 | - |
| random-label LEACE | 0.600 | 0.589 | 0.594 | 0.585 | 0.551 | 0.516 | 0.487 | 0.521 | - |
| keep | 0.600 | 0.704 | 0.705 | 0.683 | 0.595 | 0.585 | 0.595 | 0.565 | 0.587 |

## in-distribution: AUROC by rank (mean over the three layers)

| variant | k = 0 | 5 | 10 | 20 | 50 | 100 | 200 | 400 | k = 170 |
|---|---|---|---|---|---|---|---|---|---|
| nuisance | 0.996 | 0.995 | 0.996 | 0.996 | 0.994 | 0.987 | 0.990 | 0.989 | - |
| total | 0.996 | 0.686 | 0.691 | 0.632 | 0.606 | 0.615 | 0.577 | 0.524 | - |
| random | 0.996 | 0.996 | 0.996 | 0.996 | 0.996 | 0.995 | 0.996 | 0.996 | - |
| random-label LEACE | 0.996 | 0.996 | 0.995 | 0.995 | 0.990 | 0.978 | 0.908 | 0.773 | - |
| keep | 0.996 | 0.988 | 0.988 | 0.993 | 0.995 | 0.996 | 0.994 | 0.991 | 0.996 |
