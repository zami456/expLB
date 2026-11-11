# Lie coordinates on Liars' Bench: train on one set, test on the other four (Llama-3.3-70B, blocks 35-45)

Design: `emotion_coords/lie_lb_spec.py`, frozen (sha256 2F1A36ADA92E) before any probe was fitted here. `ms10_lie`'s arms under `ms10`'s probe; phase 3's protocol C rows and folds. A re-test on the thesis's development data.

## Predictions (all 20 cells)

- **LB1** (lie (k = 100) minus raw activations, all 20 cells: CI above 0 at >= 6 of the 11 blocks): mean -0.026; CI above 0 at 4, below 0 at 7 of 11: **fails**.
- **LB2** (lie minus a random subspace of the same dimension (mean of 5 seeds): CI above 0 at >= 6): mean +0.068; CI above 0 at 10, below 0 at 0 of 11: **holds**.
- **LB3** (lie minus the training set's own top-k PCs: CI above 0 at >= 6): mean -0.023; CI above 0 at 4, below 0 at 6 of 11: **fails**.
- **LB4** (lie3 against D4 (k = 3 each), two-sided: lie coordinates win if the CI is above 0 at >= 6 blocks, persona coordinates win if it is below 0 at >= 6, otherwise neither): mean -0.187; CI above 0 at 0, below 0 at 11 of 11: **persona coordinates win**.

## all: mean over the 11 blocks (min-max of the block means), and blocks 36/40/44

| arm | mean | min-max | B36 | B40 | B44 |
|---|---|---|---|---|---|
| D9 extended contrast (zero-shot) | 0.891 | 0.870-0.915 | 0.913 | 0.898 | 0.883 |
| first lie direction (zero-shot) | 0.759 | 0.732-0.806 | 0.784 | 0.732 | 0.742 |
| D4: persona coordinates (3) | 0.837 | 0.802-0.871 | 0.871 | 0.843 | 0.858 |
| lie coordinates (3) | 0.650 | 0.547-0.753 | 0.726 | 0.613 | 0.607 |
| the paper's PCs (3) | 0.624 | 0.509-0.699 | 0.672 | 0.699 | 0.562 |
| training set's top 3 PCs | 0.589 | 0.566-0.603 | 0.599 | 0.588 | 0.599 |
| lie coordinates (k) | 0.601 | 0.536-0.664 | 0.616 | 0.564 | 0.559 |
| training set's top k PCs | 0.624 | 0.580-0.702 | 0.678 | 0.622 | 0.595 |
| random k directions | 0.533 | 0.500-0.580 | 0.573 | 0.529 | 0.504 |
| raw activations | 0.627 | 0.591-0.691 | 0.672 | 0.636 | 0.612 |

## excl. HP-KR: mean over the 11 blocks (min-max of the block means), and blocks 36/40/44

| arm | mean | min-max | B36 | B40 | B44 |
|---|---|---|---|---|---|
| D9 extended contrast (zero-shot) | 0.893 | 0.866-0.925 | 0.925 | 0.895 | 0.881 |
| first lie direction (zero-shot) | 0.717 | 0.684-0.767 | 0.740 | 0.684 | 0.705 |
| D4: persona coordinates (3) | 0.823 | 0.755-0.897 | 0.897 | 0.816 | 0.842 |
| lie coordinates (3) | 0.616 | 0.496-0.743 | 0.734 | 0.552 | 0.536 |
| the paper's PCs (3) | 0.772 | 0.600-0.907 | 0.907 | 0.836 | 0.672 |
| training set's top 3 PCs | 0.641 | 0.621-0.674 | 0.674 | 0.644 | 0.642 |
| lie coordinates (k) | 0.587 | 0.440-0.706 | 0.592 | 0.574 | 0.558 |
| training set's top k PCs | 0.628 | 0.555-0.736 | 0.735 | 0.621 | 0.597 |
| random k directions | 0.528 | 0.486-0.606 | 0.602 | 0.505 | 0.495 |
| raw activations | 0.657 | 0.591-0.765 | 0.765 | 0.651 | 0.605 |

## Each arm minus raw activations (all 20 cells): mean delta, blocks with the CI above / below 0

| arm | mean delta | above | below |
|---|---|---|---|
| D9 extended contrast (zero-shot) | +0.264 | 11 | 0 |
| first lie direction (zero-shot) | +0.132 | 11 | 0 |
| D4: persona coordinates (3) | +0.210 | 11 | 0 |
| lie coordinates (3) | +0.023 | 6 | 4 |
| the paper's PCs (3) | -0.003 | 4 | 5 |
| training set's top 3 PCs | -0.038 | 0 | 10 |

## D4: persona coordinates (3): train (rows) on one set, test (columns) on the others; mean over the blocks

| train | CG | HP-C | HP-KR | IT | ID | mean |
|---|---|---|---|---|---|---|
| CG | - | 0.731 | 0.865 | 0.974 | 0.962 | 0.883 |
| HP-C | 0.698 | - | 0.936 | 0.510 | 0.911 | 0.764 |
| HP-KR | 0.725 | 0.801 | - | 0.897 | 0.962 | 0.846 |
| IT | 0.915 | 0.750 | 0.926 | - | 0.973 | 0.891 |
| ID | 0.879 | 0.591 | 0.757 | 0.980 | - | 0.802 |

## lie coordinates (3): train (rows) on one set, test (columns) on the others; mean over the blocks

| train | CG | HP-C | HP-KR | IT | ID | mean |
|---|---|---|---|---|---|---|
| CG | - | 0.538 | 0.499 | 0.396 | 0.947 | 0.595 |
| HP-C | 0.890 | - | 0.893 | 0.368 | 0.976 | 0.782 |
| HP-KR | 0.857 | 0.578 | - | 0.442 | 0.978 | 0.714 |
| IT | 0.443 | 0.523 | 0.446 | - | 0.457 | 0.467 |
| ID | 0.846 | 0.608 | 0.926 | 0.395 | - | 0.694 |

## lie coordinates (k): train (rows) on one set, test (columns) on the others; mean over the blocks

| train | CG | HP-C | HP-KR | IT | ID | mean |
|---|---|---|---|---|---|---|
| CG | - | 0.520 | 0.237 | 0.668 | 0.847 | 0.568 |
| HP-C | 0.670 | - | 0.796 | 0.267 | 0.758 | 0.623 |
| HP-KR | 0.508 | 0.669 | - | 0.577 | 0.750 | 0.626 |
| IT | 0.578 | 0.594 | 0.619 | - | 0.521 | 0.578 |
| ID | 0.764 | 0.572 | 0.823 | 0.289 | - | 0.612 |

## raw activations: train (rows) on one set, test (columns) on the others; mean over the blocks

| train | CG | HP-C | HP-KR | IT | ID | mean |
|---|---|---|---|---|---|---|
| CG | - | 0.442 | 0.154 | 0.524 | 0.965 | 0.521 |
| HP-C | 0.599 | - | 0.723 | 0.926 | 0.663 | 0.728 |
| HP-KR | 0.437 | 0.685 | - | 0.538 | 0.766 | 0.607 |
| IT | 0.425 | 0.562 | 0.728 | - | 0.643 | 0.590 |
| ID | 0.912 | 0.644 | 0.621 | 0.582 | - | 0.690 |

## Zero-shot, each set (mean over the blocks)

| detector | CG | HP-C | HP-KR | IT | ID |
|---|---|---|---|---|---|
| first lie direction (zero-shot) | 0.901 | 0.658 | 0.928 | 0.332 | 0.976 |
| D9 extended contrast (zero-shot) | 0.905 | 0.709 | 0.886 | 0.980 | 0.976 |
