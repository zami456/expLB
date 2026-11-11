# The all-170 emotion arm on Aletheia's Quest

Design: `emotion_coords/aletheia170_spec.py`, frozen (sha256 5A256321271D) before any read. Each test row read by its organism at the middle block (Qwen3.5 32, Gemma 31), reasoning dropped, content span. HG-G and HG-N not run (no adapter access). Echo† is the organism accepted on its test rows (the thesis's recorded deviation); AB-AW has 26 lies.

| Arm | Echo† | Harm | GS-F | GS-M | BotC | AB-AW | AB-HC | AB-SP | Mean [95% CI] | Rank of 31 [interval] |
|---|---|---|---|---|---|---|---|---|---|---|
| emotion_all (LB) | 0.658 | 0.676 | 0.636 | 0.610 | 0.498 | 0.602 | 0.446 | 0.852 | 0.622 [0.592, 0.649] | 20 [16-26] |
| raw (LB) | 0.653 | 0.608 | 0.864 | 0.836 | 0.457 | 0.739 | 0.624 | 0.822 | 0.700 [0.674, 0.724] | 11 [9-13] |
| shuffled_all (LB) | 0.672 | 0.716 | 0.570 | 0.571 | 0.457 | 0.642 | 0.400 | 0.652 | 0.585 [0.560, 0.608] | 26 [24-26] |
| random_all (LB) | 0.757 | 0.633 | 0.518 | 0.554 | 0.456 | 0.554 | 0.222 | 0.401 | 0.512 [0.487, 0.536] | 31 [29-31] |
| alpaca_all (LB) | 0.670 | 0.642 | 0.399 | 0.459 | 0.427 | 0.700 | 0.217 | 0.328 | 0.480 [0.447, 0.511] | 31 [31-31] |
| trainpca_all (LB) | 0.511 | 0.594 | 0.469 | 0.629 | 0.433 | 0.633 | 0.364 | 0.774 | 0.551 [0.521, 0.582] | 26 [26-30] |
| headline | 0.189 | 0.885 | 0.771 | 0.798 | 0.604 | 0.786 | 0.408 | 0.818 | 0.657 [0.633, 0.680] | 16 [13-20] |
| D4 (LB) | 0.697 | 0.615 | 0.843 | 0.774 | 0.522 | 0.635 | 0.710 | 0.828 | 0.703 [0.679, 0.729] | 11 [9-13] |
| R1-repe | 0.766 | 0.821 | 0.895 | 0.814 | 0.505 | 0.783 | 0.862 | 0.825 | 0.784 [0.761, 0.807] | 5 [3-6] |
| R2-dev-probe | 0.569 | 0.742 | 0.642 | 0.629 | 0.432 | 0.665 | 0.465 | 0.804 | 0.618 [0.590, 0.647] | 23 [16-26] |
| emotion_all (RepE) | 0.812 | 0.609 | 0.902 | 0.776 | 0.472 | 0.552 | 0.572 | 0.499 | 0.649 [0.617, 0.679] | 16 [13-23] |
| raw (RepE) | 0.798 | 0.790 | 0.784 | 0.686 | 0.491 | 0.780 | 0.619 | 0.568 | 0.689 [0.659, 0.718] | 12 [9-16] |
| emotion_all (dev) | 0.467 | 0.702 | 0.667 | 0.513 | 0.487 | 0.741 | 0.353 | 0.786 | 0.589 [0.558, 0.620] | 26 [22-26] |
| raw (dev) | 0.634 | 0.749 | 0.864 | 0.885 | 0.429 | 0.758 | 0.608 | 0.809 | 0.717 [0.692, 0.742] | 9 [8-12] |
| emotion_all (LB, base read) | 0.735 | 0.676 | 0.680 | 0.591 | 0.465 | 0.543 | 0.369 | 0.878 | 0.617 [0.589, 0.644] | 23 [18-26] |
| raw (LB, base read) | 0.707 | 0.608 | 0.691 | 0.654 | 0.442 | 0.618 | 0.460 | 0.744 | 0.616 [0.587, 0.642] | 23 [18-26] |

Paired differences of the mean AUROC (same bootstrap draws):

| Arm | minus | Δ | 95% CI |
|---|---|---|---|
| emotion_all (LB) | raw (LB) | -0.078 | [-0.101, -0.057] |
| emotion_all (LB) | shuffled_all (LB) | +0.037 | [+0.020, +0.055] |
| emotion_all (LB) | random_all (LB) | +0.110 | [+0.086, +0.135] |
| emotion_all (LB) | alpaca_all (LB) | +0.142 | [+0.115, +0.171] |
| emotion_all (LB) | trainpca_all (LB) | +0.071 | [+0.038, +0.103] |
| emotion_all (LB) | headline | -0.035 | [-0.069, +0.000] |
| emotion_all (LB) | D4 (LB) | -0.081 | [-0.118, -0.046] |
| emotion_all (LB) | R1-repe | -0.162 | [-0.195, -0.130] |
| emotion_all (LB) | R2-dev-probe | +0.004 | [-0.036, +0.044] |
| emotion_all (RepE) | raw (RepE) | -0.040 | [-0.067, -0.016] |
| emotion_all (RepE) | shuffled_all (RepE) | +0.072 | [+0.051, +0.094] |
| emotion_all (RepE) | random_all (RepE) | +0.148 | [+0.121, +0.174] |
| emotion_all (RepE) | alpaca_all (RepE) | +0.077 | [+0.047, +0.104] |
| emotion_all (RepE) | trainpca_all (RepE) | -0.001 | [-0.026, +0.023] |
| emotion_all (dev) | raw (dev) | -0.128 | [-0.157, -0.099] |
| emotion_all (dev) | shuffled_all (dev) | -0.002 | [-0.024, +0.021] |
| emotion_all (dev) | random_all (dev) | +0.036 | [+0.011, +0.060] |
| emotion_all (dev) | alpaca_all (dev) | +0.042 | [+0.015, +0.070] |
| emotion_all (dev) | trainpca_all (dev) | +0.040 | [+0.010, +0.072] |
| emotion_all (LB, base read) | raw (LB, base read) | +0.001 | [-0.025, +0.028] |

- **H1** (beats raw, LB): Δ -0.078 [-0.101, -0.057]: fails.
- **H2** (beats each 170-d control, LB): shuffled_all (LB) +0.037 [+0.020, +0.055]; random_all (LB) +0.110 [+0.086, +0.135]; alpaca_all (LB) +0.142 [+0.115, +0.171]; trainpca_all (LB) +0.071 [+0.038, +0.103]: holds.

Gates:

- qwen35: parity min cos 0.999993; |r(PC1, valence)| 0.861, split-half median 0.727; dev gate 1.000 vs band 0.840
- gemma: parity min cos 0.999999; |r(PC1, valence)| 0.843, split-half median 0.709; dev gate 0.980 vs band 0.887
