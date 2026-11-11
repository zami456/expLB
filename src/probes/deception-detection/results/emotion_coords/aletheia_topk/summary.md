# Top-k emotion PCs on Aletheia's Quest

Design: `emotion_coords/aletheia_topk_spec.py`, frozen (sha256 DB33DB75ACD6) before scoring, after the all-170 results were seen. Liars' Bench-trained unless noted; each row read by its organism. k_e, the emotion basis's inner-CV choice: Qwen3.5 10, Gemma 20.

| Arm | Echo† | Harm | GS-F | GS-M | BotC | AB-AW | AB-HC | AB-SP | Mean [95% CI] | Rank of 31 [interval] |
|---|---|---|---|---|---|---|---|---|---|---|
| emotion_pc (k_e, LB) | 0.824 | 0.547 | 0.468 | 0.415 | 0.425 | 0.368 | 0.469 | 0.596 | 0.514 [0.485, 0.545] | 31 [28-31] |
| topic_pc (k_e, LB) | 0.726 | 0.497 | 0.499 | 0.626 | 0.438 | 0.416 | 0.689 | 0.516 | 0.551 [0.522, 0.582] | 26 [26-30] |
| shuffled_pc (k_e, LB) | 0.748 | 0.626 | 0.705 | 0.643 | 0.433 | 0.502 | 0.361 | 0.706 | 0.590 [0.570, 0.610] | 26 [23-26] |
| random (k_e, LB) | 0.567 | 0.462 | 0.475 | 0.480 | 0.473 | 0.570 | 0.422 | 0.471 | 0.490 [0.481, 0.498] | 31 [31-31] |
| raw (LB) | 0.653 | 0.608 | 0.864 | 0.836 | 0.457 | 0.739 | 0.624 | 0.822 | 0.700 [0.674, 0.724] | 11 [9-13] |
| all-170 (LB) | 0.658 | 0.676 | 0.636 | 0.610 | 0.498 | 0.602 | 0.446 | 0.852 | 0.622 [0.592, 0.649] | 20 [16-26] |
| headline | 0.189 | 0.885 | 0.771 | 0.798 | 0.604 | 0.786 | 0.408 | 0.818 | 0.657 [0.633, 0.680] | 16 [13-20] |
| D4 (LB) | 0.697 | 0.615 | 0.843 | 0.774 | 0.522 | 0.635 | 0.710 | 0.828 | 0.703 [0.679, 0.729] | 11 [9-13] |
| R1-repe | 0.766 | 0.821 | 0.895 | 0.814 | 0.505 | 0.783 | 0.862 | 0.825 | 0.784 [0.761, 0.807] | 5 [3-6] |

Mean AUROC over the 8 sets at each fixed k (Liars' Bench-trained):

| k | emotion PCs | topic PCs | shuffled PCs (mean of 5) | random (mean of 20) |
|---|---|---|---|---|
| 2 | 0.589 | 0.527 | 0.493 | 0.485 |
| 3 | 0.470 | 0.521 | 0.484 | 0.488 |
| 5 | 0.470 | 0.523 | 0.534 | 0.471 |
| 10 | 0.494 | 0.582 | 0.597 | 0.490 |
| 20 | 0.534 | 0.533 | 0.582 | 0.501 |

Secondary sources, at k_e:

| source | emotion PCs | topic PCs | shuffled PCs | random |
|---|---|---|---|---|
| RepE | 0.506 | 0.553 | 0.509 | 0.541 |
| dev | 0.556 | 0.524 | 0.601 | 0.509 |

Paired differences of the mean AUROC:

| Arm | minus | Δ | 95% CI |
|---|---|---|---|
| emotion_pc (k_e, LB) | topic_pc (k_e, LB) | -0.037 | [-0.076, +0.002] |
| emotion_pc (k_e, LB) | shuffled_pc (k_e, LB) | -0.076 | [-0.105, -0.047] |
| emotion_pc (k_e, LB) | random (k_e, LB) | +0.024 | [-0.006, +0.053] |
| emotion_pc (k_e, LB) | raw (LB) | -0.186 | [-0.218, -0.154] |
| emotion_pc (k_e, LB) | all-170 (LB) | -0.108 | [-0.141, -0.075] |
| emotion_pc (k_e, LB) | headline | -0.143 | [-0.180, -0.106] |
| emotion_pc (k_e, RepE) | topic_pc (k_e, RepE) | -0.047 | [-0.082, -0.012] |
| emotion_pc (k_e, RepE) | shuffled_pc (k_e, RepE) | -0.003 | [-0.027, +0.021] |
| emotion_pc (k_e, RepE) | random (k_e, RepE) | -0.035 | [-0.063, -0.006] |
| emotion_pc (k_e, dev) | topic_pc (k_e, dev) | +0.032 | [+0.003, +0.062] |
| emotion_pc (k_e, dev) | shuffled_pc (k_e, dev) | -0.046 | [-0.073, -0.019] |
| emotion_pc (k_e, dev) | random (k_e, dev) | +0.046 | [+0.018, +0.076] |
| emotion_pc (k=2, LB) | random (k=2, LB) | +0.104 | [+0.077, +0.132] |
| emotion_pc (k=2, LB) | topic_pc (k=2, LB) | +0.062 | [+0.021, +0.100] |
| emotion_pc (k=3, LB) | random (k=3, LB) | -0.018 | [-0.047, +0.008] |
| emotion_pc (k=3, LB) | topic_pc (k=3, LB) | -0.051 | [-0.084, -0.019] |
| emotion_pc (k=5, LB) | random (k=5, LB) | -0.001 | [-0.030, +0.026] |
| emotion_pc (k=5, LB) | topic_pc (k=5, LB) | -0.053 | [-0.086, -0.022] |
| emotion_pc (k=10, LB) | random (k=10, LB) | +0.005 | [-0.025, +0.033] |
| emotion_pc (k=10, LB) | topic_pc (k=10, LB) | -0.088 | [-0.128, -0.048] |
| emotion_pc (k=20, LB) | random (k=20, LB) | +0.033 | [+0.002, +0.066] |
| emotion_pc (k=20, LB) | topic_pc (k=20, LB) | +0.001 | [-0.033, +0.036] |

- **TK1** (P1's rule, k_e, LB): minus random +0.024 [-0.006, +0.053]; minus topic_pc -0.037 [-0.076, +0.002]: fails.
