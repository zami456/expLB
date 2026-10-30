# Lie coordinates from RepE on Mahadik & Skapars' ten datasets (Llama-3.3-70B, blocks 35-45)

Design: `emotion_coords/ms10_lie_spec.py`, frozen (sha256 18DEB4CF40F5) before any lie coordinate was fitted. INLP on RepE alone; every arm under `ms10`'s probe, rows, splits and draws.

## Predictions (deception region: the 20 off-diagonal deception cells)

- **L1** (lie minus raw activations, deception region: CI above 0 at >= 6 of the 11 blocks): mean -0.010; CI above 0 at 2, below 0 at 3 of 11: **fails**.
- **L2** (lie minus a random subspace of the same dimension (mean of 5 seeds), deception region: CI above 0 at >= 6): mean +0.086; CI above 0 at 10, below 0 at 0 of 11: **holds**.
- **L3** (lie minus the training set's own top-k PCs, deception region: CI above 0 at >= 6): mean -0.042; CI above 0 at 0, below 0 at 7 of 11: **fails**.
- **L4** (lie3 against D4 (k = 3 each), deception region, two-sided: lie coordinates win if the CI is above 0 at >= 6 blocks, persona coordinates win if it is below 0 at >= 6, otherwise neither): mean -0.004; CI above 0 at 0, below 0 at 0 of 11: **neither**.

## The lie directions: k by block (INLP stops below held-out AUROC 0.55 on RepE's validation split)

| block | 35 | 36 | 37 | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| round-1 held-out AUROC | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

## deception: mean over the 11 blocks, and the range of block means

| arm | mean | min-max over blocks |
|---|---|---|
| first lie direction (zero-shot) | 0.866 | 0.841-0.900 |
| lie coordinates (k) | 0.736 | 0.679-0.815 |
| lie coordinates (3) | 0.843 | 0.809-0.876 |
| D4: persona coordinates (3) | 0.847 | 0.807-0.882 |
| training set's top k PCs | 0.778 | 0.749-0.799 |
| training set's top 3 PCs | 0.771 | 0.747-0.790 |
| random k directions | 0.650 | 0.610-0.698 |
| raw activations | 0.746 | 0.729-0.797 |
| emotion all-170 | 0.759 | 0.727-0.789 |
| emotion PCs (3) | 0.634 | 0.577-0.707 |
| training set's top 170 PCs | 0.780 | 0.751-0.826 |
| the paper's PCs (3) | 0.762 | 0.641-0.865 |
| the paper's sycophancy PCs (3) | 0.789 | 0.759-0.806 |

## new deception: mean over the 11 blocks, and the range of block means

| arm | mean | min-max over blocks |
|---|---|---|
| first lie direction (zero-shot) | 0.876 | 0.849-0.914 |
| lie coordinates (k) | 0.696 | 0.628-0.751 |
| lie coordinates (3) | 0.823 | 0.786-0.863 |
| D4: persona coordinates (3) | 0.862 | 0.826-0.887 |
| training set's top k PCs | 0.773 | 0.759-0.794 |
| training set's top 3 PCs | 0.760 | 0.747-0.782 |
| random k directions | 0.637 | 0.624-0.653 |
| raw activations | 0.751 | 0.733-0.780 |
| emotion all-170 | 0.683 | 0.672-0.698 |
| emotion PCs (3) | 0.597 | 0.555-0.638 |
| training set's top 170 PCs | 0.768 | 0.746-0.798 |
| the paper's PCs (3) | 0.767 | 0.652-0.839 |
| the paper's sycophancy PCs (3) | 0.801 | 0.775-0.820 |

## sycophancy: mean over the 11 blocks, and the range of block means

| arm | mean | min-max over blocks |
|---|---|---|
| first lie direction (zero-shot) | 0.620 | 0.587-0.640 |
| lie coordinates (k) | 0.529 | 0.467-0.573 |
| lie coordinates (3) | 0.442 | 0.412-0.461 |
| D4: persona coordinates (3) | 0.685 | 0.660-0.708 |
| training set's top k PCs | 0.604 | 0.581-0.620 |
| training set's top 3 PCs | 0.604 | 0.557-0.623 |
| random k directions | 0.589 | 0.580-0.597 |
| raw activations | 0.637 | 0.619-0.653 |
| emotion all-170 | 0.579 | 0.565-0.594 |
| emotion PCs (3) | 0.519 | 0.503-0.530 |
| training set's top 170 PCs | 0.600 | 0.587-0.607 |
| the paper's PCs (3) | 0.688 | 0.674-0.707 |
| the paper's sycophancy PCs (3) | 0.796 | 0.786-0.807 |

## cross: mean over the 11 blocks, and the range of block means

| arm | mean | min-max over blocks |
|---|---|---|
| lie coordinates (k) | 0.561 | 0.540-0.597 |
| lie coordinates (3) | 0.518 | 0.496-0.554 |
| D4: persona coordinates (3) | 0.771 | 0.749-0.790 |
| training set's top k PCs | 0.605 | 0.591-0.616 |
| training set's top 3 PCs | 0.651 | 0.636-0.659 |
| random k directions | 0.566 | 0.558-0.577 |
| raw activations | 0.611 | 0.600-0.626 |
| emotion all-170 | 0.597 | 0.581-0.613 |
| emotion PCs (3) | 0.541 | 0.521-0.563 |
| training set's top 170 PCs | 0.612 | 0.604-0.625 |
| the paper's PCs (3) | 0.739 | 0.699-0.780 |
| the paper's sycophancy PCs (3) | 0.805 | 0.790-0.815 |

## diagonal: mean over the 11 blocks, and the range of block means

| arm | mean | min-max over blocks |
|---|---|---|
| lie coordinates (k) | 0.900 | 0.877-0.924 |
| lie coordinates (3) | 0.806 | 0.769-0.832 |
| D4: persona coordinates (3) | 0.840 | 0.833-0.846 |
| training set's top k PCs | 0.935 | 0.927-0.944 |
| training set's top 3 PCs | 0.883 | 0.868-0.894 |
| random k directions | 0.898 | 0.895-0.902 |
| raw activations | 0.947 | 0.944-0.951 |
| emotion all-170 | 0.924 | 0.917-0.930 |
| emotion PCs (3) | 0.752 | 0.727-0.789 |
| training set's top 170 PCs | 0.935 | 0.896-0.939 |
| the paper's PCs (3) | 0.841 | 0.836-0.852 |
| the paper's sycophancy PCs (3) | 0.862 | 0.853-0.869 |

## all: mean over the 11 blocks, and the range of block means

| arm | mean | min-max over blocks |
|---|---|---|
| lie coordinates (k) | 0.826 | 0.799-0.855 |
| lie coordinates (3) | 0.733 | 0.704-0.765 |
| D4: persona coordinates (3) | 0.818 | 0.804-0.835 |
| training set's top k PCs | 0.890 | 0.883-0.898 |
| training set's top 3 PCs | 0.701 | 0.687-0.708 |
| random k directions | 0.809 | 0.797-0.819 |
| raw activations | 0.929 | 0.920-0.937 |
| emotion all-170 | 0.847 | 0.838-0.859 |
| emotion PCs (3) | 0.680 | 0.650-0.712 |
| training set's top 170 PCs | 0.914 | 0.907-0.923 |
| the paper's PCs (3) | 0.809 | 0.795-0.824 |
| the paper's sycophancy PCs (3) | 0.836 | 0.826-0.843 |

## Transfer into each test set from the other sets of the same behaviour (mean over the blocks)

| arm | AIL | RP | CG | ID | HPC | Syc | OES | OEQ-V | OEQ-I | OEQ-F |
|---|---|---|---|---|---|---|---|---|---|---|
| first lie direction (zero-shot) | 0.784 | 0.967 | 0.955 | 0.982 | 0.643 | 0.697 | 0.778 | 0.553 | 0.562 | 0.510 |
| lie coordinates (k) | 0.714 | 0.678 | 0.818 | 0.886 | 0.586 | 0.410 | 0.536 | 0.569 | 0.578 | 0.553 |
| lie coordinates (3) | 0.774 | 0.873 | 0.926 | 0.973 | 0.670 | 0.344 | 0.372 | 0.505 | 0.501 | 0.488 |
| D4: persona coordinates (3) | 0.911 | 0.813 | 0.880 | 0.964 | 0.666 | 0.788 | 0.762 | 0.590 | 0.755 | 0.528 |
| training set's top k PCs | 0.743 | 0.803 | 0.887 | 0.873 | 0.585 | 0.486 | 0.703 | 0.629 | 0.604 | 0.597 |
| training set's top 3 PCs | 0.771 | 0.749 | 0.758 | 0.898 | 0.681 | 0.615 | 0.685 | 0.570 | 0.584 | 0.565 |
| random k directions | 0.639 | 0.635 | 0.684 | 0.783 | 0.510 | 0.544 | 0.615 | 0.604 | 0.594 | 0.591 |
| raw activations | 0.671 | 0.831 | 0.812 | 0.853 | 0.562 | 0.519 | 0.712 | 0.662 | 0.647 | 0.643 |
| emotion all-170 | 0.644 | 0.722 | 0.845 | 0.899 | 0.684 | 0.443 | 0.663 | 0.629 | 0.598 | 0.564 |
| emotion PCs (3) | 0.607 | 0.586 | 0.748 | 0.697 | 0.531 | 0.293 | 0.614 | 0.542 | 0.614 | 0.533 |
| training set's top 170 PCs | 0.738 | 0.797 | 0.906 | 0.857 | 0.602 | 0.500 | 0.689 | 0.631 | 0.604 | 0.577 |
| the paper's PCs (3) | 0.769 | 0.764 | 0.831 | 0.885 | 0.559 | 0.755 | 0.824 | 0.632 | 0.630 | 0.598 |
| the paper's sycophancy PCs (3) | 0.805 | 0.797 | 0.793 | 0.959 | 0.592 | 0.899 | 0.954 | 0.700 | 0.762 | 0.664 |
