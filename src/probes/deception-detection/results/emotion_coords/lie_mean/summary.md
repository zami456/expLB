# The averaged RepE lie directions, zero-shot (Llama-3.3-70B, blocks 35-45; post hoc question)

Design: `emotion_coords/lie_mean_spec.py`, frozen (sha256 D443E1C8A692) before any averaged direction was scored. Same rows and draws as `lie_lb` and `ms10`.

## Predictions

- **A1** (lie_mean100 minus lie_w1, Liars' Bench (mean over the five sets): CI above 0 at >= 6 of the 11 blocks): mean -0.241; CI above 0 at 0, below at 11: **fails**.
- **A2** (lie_mean100 minus lie_w1, the ten datasets' deception sets: CI above 0 at >= 6 of the 11 blocks): mean -0.198; CI above 0 at 0, below at 11: **fails**.

## Liars Bench: mean over the 11 blocks

| detector | all | excl. HP-KR |
|---|---|---|
| lie_mean100 | 0.518 | 0.628 |
| lie_mean10 | 0.706 | 0.670 |
| lie_w1 | 0.759 | 0.717 |
| repe_dom | 0.719 | 0.677 |
| D9 | 0.891 | 0.893 |
| raw probe | 0.627 | 0.657 |

## The ten datasets: mean over the 11 blocks

| detector | deception | new deception | sycophancy |
|---|---|---|---|
| lie_mean100 | 0.668 | 0.569 | 0.511 |
| lie_mean10 | 0.822 | 0.830 | 0.606 |
| lie_w1 | 0.866 | 0.876 | 0.620 |
| repe_dom | 0.861 | 0.873 | 0.596 |
| D9 | 0.876 | 0.881 | 0.764 |
| raw probe | 0.746 | 0.751 | 0.637 |

## Liars' Bench, each set (zero-shot; mean over the blocks)

| detector | CG | HP-C | HP-KR | IT | ID |
|---|---|---|---|---|---|
| lie_mean100 | 0.845 | 0.465 | 0.075 | 0.328 | 0.875 |
| lie_mean10 | 0.757 | 0.601 | 0.851 | 0.350 | 0.971 |
| lie_w1 | 0.901 | 0.658 | 0.928 | 0.332 | 0.976 |
| repe_dom | 0.871 | 0.569 | 0.889 | 0.290 | 0.978 |
| D9 | 0.905 | 0.709 | 0.886 | 0.980 | 0.976 |

## Cosine with RepE's difference of means (mean over the blocks)

cos(lie_mean100, repe_dom): 0.456 | cos(lie_mean10, repe_dom): 0.577 | cos(lie_w1, repe_dom): 0.862 | cos(D9, repe_dom): 0.190
