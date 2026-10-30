# Results: emotion coordinates as lie-probe features on Liars' Bench (Llama-3.3-70B)

Written 2026-10-30. The preregistration is `PREREGISTRATION.md` (commit `ae8e0ae1`) and the decision rules are in
`ANALYSIS_PLAN.md` (`ad68af22`, plus dated addenda). Every result below carries one of four labels:
- **preregistered**: P1-P8 at blocks 20/28/40/52;
- **declared**: fixed in an addendum before the relevant result existed;
- **exploratory**: reported, judges nothing;
- **post hoc**: chosen after the results it is compared with had been seen.

Each table is regenerated from the result files by `python -m emotion_coords.report_tables`
(`results/emotion_coords/report_tables.md`).

## 1. Summary

- **The preregistered hypothesis fails.** Emotion-PC probes do not transfer across Liars' Bench better than random
  subspaces *and* the topic basis (P1). No prediction P1-P6 holds at 3 of the 4 preregistered blocks. P7 settles on
  neither account, and P8 fails at block 52.
- **The one robust positive result is exploratory: the whole emotion subspace.** Projecting activations onto all 170
  principal components of the 171 emotion vectors, then fitting the probe, gives these results from block 36 to 45:
  - with four training sets it beats raw activations at every block, by 0.05-0.16 mean held-out AUROC (0.83-0.91
    against 0.74-0.78);
  - it beats its own label-shuffled control at all ten blocks (CIs exclude 0), and Alpaca's top 170 PCs at eight;
  - with a single training set it still beats the shuffled control at every band block, and raw activations by
    0.05-0.09 at blocks 41-45.
- **It is not a validated lie detector.**
  - Re-expressing the same subspace in other coordinates moves it by up to 0.2 AUROC.
  - It fails the out-of-sample test declared for it: on Gender-Secret at block 40 it scores 0.599 against 0.650 for
    raw activations.
  - It fails out of model too (§3.6), on Aletheia's Quest with Qwen3.5-27B and Gemma-3-27B rebuilt bases:
    - it scores 0.622 mean AUROC, rank 20 of 31, against 0.700 for raw activations trained on the same rows;
    - it still beats every matched 170-d control there.
  - At a fixed false-positive rate it catches few lies.

  Read it as a better-than-raw, better-than-matched-controls feature space for mid-depth transfer *among the five
  Liars' Bench sets*. Whether it detects lying as such is not established.
- **Context.** The extended persona PCs (at most 24 dimensions) are the strongest of the small bases at every block,
  and beat the all-170 arm at 7 of the 14 blocks. The HP-KR inversion of the published probe is confined to the
  early blocks (block 20; about 0.5 at block 28).

## 2. What was run

- **Model and data.**
  - Llama-3.3-70B-Instruct, read through NDIF.
  - Liars' Bench LB-5 (CG, HP-C, HP-KR, IT, ID) at the spec's caps: 5,463 rows. Alpaca (2,000) sets thresholds and
    the RepE instructed pairs (550) train protocol A.
  - Pooling (a), the mean over the final reply's detected span, unless stated.
- **Emotion vectors (Anthropic's recipe; Gemotions stories).**
  - 171 emotions × 40 topics, one story each. The 18 least reliable core emotions get two more stories per topic:
    8,280 stories in all.
  - The mean from token 50, centred across emotions, with the neutral texts' top PCs (50% of their variance) removed.
  - Sanity checks pass at every block (T11): |r(PC1, NRC valence)| is 0.834-0.853 and the core split-half median is
    0.773-0.795. The Tylenol dose test is monotone at all 14 blocks (afraid ρ +1, calm ρ −1).
- **Arms.**
  - *Preregistered:* emotion PCs (k ∈ {2, 3, 5, 10, 20}), random subspaces (20 seeds), the topic basis,
    label-shuffled bases (5), Alpaca and training-set PCs, persona PCs, persona + emotion, the 29 named emotions,
    raw activations, erasures, and zero-shot directions.
  - *Exploratory:* the all-170 arm with its 170-d controls, MLP probes, and the 171 emotion vectors.
- **Protocols.**
  - **B** (the main one): train on four sets, test on the fifth; k and C chosen by inner leave-one-set-out CV.
  - **A:** train on the RepE pairs.
  - **C:** train on one set, test on each other set.
  - CIs come from 1,000 paired, stratified bootstrap resamples of each held-out set. They are not corrected across
    blocks.
- **Blocks.**
  - *Preregistered:* 20, 28, 40, 52 (standing for the spec's 16/27/40/53).
  - *Post hoc band (user, 2026-10-30):* every block from 35 to 45. Blocks 36/40/44 use the thesis's stored
    activations; the other eight use this study's own NDIF reads (§7).

## 3. The all-170 arm

**What it is.** The 171 centred, denoised emotion vectors span 170 dimensions. The probe gets their 170 PCs as
coordinates (standardised) and fits L2 logistic regression, with C chosen by inner CV as for every arm.

- **Status.** It was declared in an `ANALYSIS_PLAN.md` addendum after the pilot and before any full-run probe, as an
  exploratory arm that judges no prediction.
- **Controls, all at the same dimension:**
  - the five label-shuffled bases (emotion labels permuted within topic, same recipe, 170 PCs each);
  - five random 170-d subspaces;
  - the top 170 PCs of Alpaca's activations;
  - the top 170 PCs of the training rows, refitted inside each fold.
- **Post hoc parts.** The band (35-45), protocol C for this arm, and the vector-coordinate check were all run after
  results had been seen.

![The all-170 arm and its controls across depth](../results/emotion_coords/figures/all170_by_layer.png)

### 3.1 Train on four sets, test on the fifth (protocol B)

| block | all 170 emotion PCs | raw | minus raw | minus 170 shuffled [95% CI] | minus 170 Alpaca [95% CI] | minus 170 training-set [95% CI] | minus 170 random [95% CI] |
|---|---|---|---|---|---|---|---|
| 20 | 0.680 | 0.676 | +0.004 | +0.008 [-0.004, +0.020] | +0.174 [+0.155, +0.195] | +0.081 [+0.065, +0.098] | +0.132 [+0.116, +0.150] |
| 28 | 0.623 | 0.791 | -0.168 | -0.174 [-0.190, -0.158] | +0.157 [+0.134, +0.179] | -0.113 [-0.134, -0.087] | -0.013 [-0.033, +0.006] |
| 35 | 0.682 | 0.778 | -0.096 | -0.079 [-0.092, -0.066] | +0.081 [+0.059, +0.104] | -0.094 [-0.117, -0.071] | +0.051 [+0.034, +0.067] |
| 36 | 0.827 | 0.777 | +0.050 | +0.043 [+0.032, +0.055] | -0.005 [-0.019, +0.010] | +0.012 [-0.007, +0.033] | +0.247 [+0.232, +0.262] |
| 37 | 0.865 | 0.753 | +0.113 | +0.027 [+0.016, +0.038] | +0.045 [+0.028, +0.062] | +0.137 [+0.115, +0.160] | +0.218 [+0.201, +0.235] |
| 38 | 0.894 | 0.763 | +0.131 | +0.079 [+0.068, +0.090] | +0.198 [+0.181, +0.216] | +0.188 [+0.167, +0.211] | +0.306 [+0.290, +0.322] |
| 39 | 0.896 | 0.764 | +0.132 | +0.045 [+0.036, +0.055] | +0.183 [+0.166, +0.199] | +0.137 [+0.116, +0.159] | +0.216 [+0.203, +0.230] |
| 40 | 0.852 | 0.757 | +0.094 | +0.040 [+0.031, +0.051] | +0.192 [+0.176, +0.206] | +0.067 [+0.047, +0.089] | +0.226 [+0.215, +0.238] |
| 41 | 0.893 | 0.762 | +0.131 | +0.089 [+0.078, +0.101] | +0.116 [+0.102, +0.130] | +0.124 [+0.104, +0.145] | +0.264 [+0.251, +0.278] |
| 42 | 0.879 | 0.759 | +0.120 | +0.068 [+0.056, +0.080] | +0.079 [+0.061, +0.096] | +0.127 [+0.105, +0.150] | +0.239 [+0.224, +0.255] |
| 43 | 0.905 | 0.743 | +0.162 | +0.091 [+0.080, +0.102] | +0.110 [+0.095, +0.125] | +0.162 [+0.142, +0.184] | +0.269 [+0.255, +0.284] |
| 44 | 0.845 | 0.741 | +0.104 | +0.012 [+0.001, +0.024] | +0.003 [-0.014, +0.018] | +0.112 [+0.091, +0.132] | +0.205 [+0.189, +0.220] |
| 45 | 0.913 | 0.760 | +0.153 | +0.108 [+0.098, +0.118] | +0.047 [+0.034, +0.059] | +0.298 [+0.277, +0.321] | +0.288 [+0.274, +0.302] |
| 52 | 0.795 | 0.774 | +0.021 | +0.102 [+0.088, +0.117] | -0.076 [-0.092, -0.062] | +0.105 [+0.084, +0.128] | +0.152 [+0.136, +0.169] |

- **Blocks 36-45 (ten blocks).** The arm beats raw activations at all ten, by 0.050-0.162. It beats the
  label-shuffled control at all ten with the CI above 0 (+0.012 to +0.108). It beats Alpaca's PCs at eight (ties at
  36 and 44), training-set PCs at nine (a tie at 36), and random 170-d subspaces at all ten.
- **The extended persona PCs** score 0.81-0.93 across the band. The all-170 arm is ahead of them at blocks 38, 39 and
  41-45, and behind at 36, 37 and 40.
- **Outside blocks 36-45 it does not work.** At blocks 28 and 35 it is well below raw activations and the shuffled
  control. At 52, Alpaca's PCs beat it. At 20 it ties raw.
- **The shuffled control is the decisive one.** It shares the stories, the recipe and the dimension, and differs only
  in which stories are averaged together. The arm's margin over it (+0.01 to +0.11 in the band) is the part
  attributable to emotion labels. The much larger margin over raw activations is mostly shared with the controls,
  which also reach 0.80-0.87 at several band blocks. Projecting onto almost any subspace spanned by natural story or
  chat activations already helps transfer at these depths.

**Per held-out set** (T3). The gain is broad, and largest on HP-KR, where the arm undoes raw activations' weakness:

| block, arm | CG | HP-C | HP-KR | IT | ID |
|---|---|---|---|---|---|
| 40 all-170 | 0.965 | 0.642 | 0.741 | 0.985 | 0.925 |
| 40 raw | 0.861 | 0.666 | 0.546 | 0.776 | 0.938 |
| 43 all-170 | 0.991 | 0.695 | 0.936 | 0.968 | 0.936 |
| 43 raw | 0.801 | 0.620 | 0.697 | 0.739 | 0.859 |
| 45 all-170 | 0.986 | 0.720 | 0.951 | 0.970 | 0.938 |
| 45 raw | 0.942 | 0.617 | 0.521 | 0.788 | 0.932 |

### 3.2 Train on one set, test on each other (protocol C)

Mean of the 20 single-source transfers (T4; `results/emotion_coords/exploratory/protocol_c_band_170.csv` holds every
matrix):

| arm | 35 | 36 | 37 | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| all 170 emotion PCs | 0.675 | 0.677 | 0.668 | 0.616 | 0.623 | 0.644 | 0.692 | 0.700 | 0.671 | 0.667 | 0.658 |
| 170 label-shuffled (5) | 0.590 | 0.616 | 0.586 | 0.600 | 0.585 | 0.570 | 0.593 | 0.590 | 0.589 | 0.598 | 0.574 |
| 170 Alpaca PCs | 0.601 | 0.586 | 0.528 | 0.557 | 0.555 | 0.553 | 0.563 | 0.544 | 0.558 | 0.538 | 0.556 |
| raw activations | 0.691 | 0.672 | 0.636 | 0.591 | 0.615 | 0.636 | 0.618 | 0.606 | 0.609 | 0.612 | 0.610 |
| all-170 minus shuffled | +0.085 | +0.061 | +0.083 | +0.017 | +0.038 | +0.074 | +0.098 | +0.110 | +0.083 | +0.068 | +0.084 |
| all-170 minus raw | -0.016 | +0.005 | +0.033 | +0.025 | +0.008 | +0.007 | +0.073 | +0.094 | +0.062 | +0.054 | +0.048 |

- **The margin over the shuffled control survives with one training set** at every band block (+0.02 to +0.11). The
  margin over raw activations appears only at 41-45.
- **Single-source transfer stays modest** (0.62-0.70). The arm inverts in 4-7 of the 20 transfers, about as often as
  raw activations. At blocks 20/28/40/52 it scores 0.556/0.658/0.644/0.630, against 0.569/0.617/0.636/0.607 for raw.
- **These are point estimates.** Protocol C is not bootstrapped.

### 3.3 With an MLP probe (exploratory family)

A 64-unit MLP on the same coordinates (T5). The all-170 arm scores 0.73-0.79 and beats the shuffled control at 12 of
14 blocks (CI > 0). The persona PCs, published or extended, are the best MLP arm at most blocks; the all-170 arm is
best at blocks 38 and 42. The MLP lowers the all-170 arm well below its logistic-regression level (0.73-0.79 against
0.83-0.91 in the band).

### 3.4 Robustness checks

**Coordinates.** The 171 unit emotion vectors span exactly the same subspace as their 170 PCs. An unregularised
linear probe would score them identically, but the standardised, L2-penalised probe does not (post hoc, T6):

| block | 171 emotion vectors | 170 emotion PCs | raw activations |
|---|---|---|---|
| 20 | 0.666 | 0.680 | 0.676 |
| 28 | 0.705 | 0.623 | 0.791 |
| 40 | 0.744 | 0.852 | 0.757 |
| 52 | 0.586 | 0.795 | 0.774 |

The probe tends to choose strong regularisation (C = 0.0001-0.01 at block 40). It then behaves roughly like a
difference of class means measured in its coordinates, and that difference depends on the basis. So "the emotion subspace" helps only
through its PCA coordinates. The subspace alone does not carry the effect.

**Fixed threshold** (T7). Thresholded at Alpaca's 1% false-positive rate, the arm's balanced accuracy is 0.58-0.67
in the band, against 0.53-0.65 for raw activations. It catches 19-36% of lies, mostly in CG and ID. HP-C, HP-KR and
IT lies mostly score below Alpaca's 99th percentile, as they do for nearly every arm.

### 3.5 Out-of-sample: Gender-Secret at block 40 (declared, `c3fc20de`, before GS was scored)

GS is Liars' Bench's LoRA fine-tune of the same model. It believes the user is male and denies knowing it, and its
lies and honest answers are both denials. Every arm was fitted on the five LB-5 sets, then scored on the
fine-tune's read of GS (392 rows, 116 lies) and on the base model's read of the same tokens, as a control (T8):

| arm | GS AUROC [95% CI] | base read | fine-tune minus base [95% CI] |
|---|---|---|---|
| 29 named emotions | 0.778 [0.728, 0.824] | 0.751 | +0.027 [-0.005, +0.061] |
| persona PCs, published | 0.729 [0.676, 0.782] | 0.652 | +0.077 [+0.034, +0.124] |
| raw activations | 0.650 [0.588, 0.709] | 0.554 | +0.095 [+0.055, +0.137] |
| 171 emotion vectors | 0.642 [0.587, 0.697] | 0.598 | +0.045 [-0.004, +0.098] |
| 170 Alpaca PCs | 0.629 [0.571, 0.688] | 0.660 | -0.030 [-0.088, +0.027] |
| **all 170 emotion PCs** | **0.599** [0.539, 0.658] | 0.549 | +0.049 [-0.001, +0.099] |
| 170 label-shuffled | 0.596 [0.556, 0.638] | 0.585 | +0.012 [-0.015, +0.039] |
| emotion PCs (k = 20) | 0.455 [0.394, 0.511] | 0.476 | -0.021 [-0.079, +0.033] |

- **G1 fails.** The all-170 arm minus raw activations is −0.051 [−0.128, +0.030], and minus the shuffled control
  +0.002 [−0.043, +0.054].
- **G2 fails.** Emotion PCs minus random is −0.069 [−0.132, −0.012].
- **G3 (reading the lie, not the wording).** Only raw activations and the published persona PCs score clearly higher
  on the fine-tune's read than on the base read. The best arm on GS, the 29 named emotions, scores almost as high on
  the base read, so it separates the questions (gender vs other attributes), not the lie.
- **No arm works at the Alpaca threshold on GS.**
- **Caveat.** The emotion vectors are the base model's. Re-deriving them inside the fine-tune was attempted and
  blocked by NDIF (§7), so the model mismatch is untested for emotions. The thesis found that re-deriving the persona
  directions inside the same fine-tune changed little.

### 3.6 Out-of-model: Aletheia's Quest (declared, `7467eef6`, before any story was read)

The arm was rebuilt inside the base models of the competition's organisms, by phase 1's recipe on the same 8,280
stories: Qwen3.5-27B at block 32 of 64, and Gemma-3-27B-it at block 31 of 62. It was then scored on the thesis's 8
test sets the way the thesis scored its detectors there (`scripts/aletheia/`):
- each row read by the organism that wrote it, at the middle block, reasoning dropped, content span;
- the metric is the mean AUROC over the 8 sets and the rank among the 30 published entries.

The primary training rows are those of the thesis's post hoc D4 (LB): Gemma's own CG, IT and ID, and Qwen2.5's HP-C,
HP-KR and ID text read through Qwen3.5. Every gate passed:
- reader parity, min cos 0.99999;
- |r(PC1, valence)|: 0.86 for Qwen3.5, 0.84 for Gemma;
- the dev known-answer gate.

The design is in `aletheia170_spec.py`; the tables are in `results/emotion_coords/aletheia/summary.md`.

| arm (Liars' Bench-trained unless noted) | mean AUROC [95% CI] | rank of 31 [interval] |
|---|---|---|
| R1, the RepE probe (thesis reference) | 0.784 [0.761, 0.807] | 5 [3-6] |
| raw activations, dev-trained | 0.717 [0.692, 0.742] | 9 [8-12] |
| D4 (LB) (thesis post hoc) | 0.703 [0.679, 0.729] | 11 [9-13] |
| raw activations | 0.700 [0.674, 0.724] | 11 [9-13] |
| raw activations, RepE-trained | 0.689 [0.659, 0.718] | 12 [9-16] |
| D9 + K (thesis headline) | 0.657 [0.633, 0.680] | 16 [13-20] |
| all 170 emotion PCs, RepE-trained | 0.649 [0.617, 0.679] | 16 [13-23] |
| **all 170 emotion PCs** | **0.622** [0.592, 0.649] | **20** [16-26] |
| R2, the dev probe (thesis reference) | 0.618 [0.590, 0.647] | 23 [16-26] |
| all 170 emotion PCs, dev-trained | 0.589 [0.558, 0.620] | 26 [22-26] |
| 170 label-shuffled (mean of 5) | 0.585 [0.560, 0.608] | 26 [24-26] |
| 170 training-set PCs | 0.551 [0.521, 0.582] | 26 [26-30] |
| 170 random (mean of 5) | 0.512 [0.487, 0.536] | 31 [29-31] |
| 170 Alpaca PCs | 0.480 [0.447, 0.511] | 31 [31-31] |

- **H1 fails.** All-170 minus raw activations is −0.078 [−0.101, −0.057]. It points the same way with the RepE
  training rows (−0.040 [−0.067, −0.016]) and with the dev rows (−0.128 [−0.157, −0.099]).
- **H2 holds.** Against each 170-d control, every CI is above 0:

  | control | difference |
  |---|---|
  | label-shuffled | +0.037 [+0.020, +0.055] |
  | random | +0.110 [+0.086, +0.135] |
  | Alpaca PCs | +0.142 [+0.115, +0.171] |
  | training-set PCs | +0.071 [+0.038, +0.103] |

- **Per dataset.**
  - It beats raw activations on AB-SP (0.852 against 0.822) and Harm (0.676 against 0.608), and ties on Echo.
  - It loses most on GS-F and GS-M (0.636 and 0.610, against 0.864 and 0.836) and on AB-HC (0.446 against 0.624).
  - BotC is near chance for every arm.
- **Organism read against base read** (the thesis's text control).
  - All-170 scores 0.622 on the organisms' reads and 0.617 on the base model's read of the same rows.
  - Raw activations score 0.700 against 0.616.
  - So raw activations gain from the organism's own read, and the emotion coordinates do not: what they pick up is
    in the text.
- **Training side.** In the inner leave-one-set-out CV on Liars' Bench, the arm was not ahead of raw activations in
  either model (Gemma 0.715 against 0.718; Qwen3.5 0.864 against 0.909). On Llama it was ahead under protocol B.
- **Top k** (declared `c404a42c`; asked after the all-170 results; labelled post hoc). This is the preregistered
  emotion-PC arm: the first k PCs of the same bases, with k ∈ {2, 3, 5, 10, 20} chosen by inner CV. The CV chose
  k_e = 10 for Qwen3.5 and 20 for Gemma, and the gates passed. The arm scores at chance:

  | arm, at k_e | mean AUROC [95% CI] |
  |---|---|
  | emotion PCs | **0.514** [0.485, 0.545], rank 31 of 31 |
  | topic basis | 0.551 |
  | shuffled bases | 0.590 |
  | random (mean of 20) | 0.490 |

  - **TK1 (P1's rule) fails.** Emotion PCs minus random is +0.024 [−0.006, +0.053]; minus the topic basis it is
    −0.037 [−0.076, +0.002].
  - **Other comparisons.** It is 0.108 below all-170 and 0.186 below raw activations.
  - **Fixed k.** Only k = 2 stands out (0.589: +0.104 over random, +0.062 over topic). The CV did not pick it, so
    that is post hoc.
  - **Sets.** It is high on Echo (0.824) and below 0.5 on GS-F, GS-M, BotC, AB-AW and AB-HC.
  - **Secondary sources.** RepE gives 0.506 and dev gives 0.556.
  - **Agreement with Llama.** This agrees with P1's failure on Llama. On these organisms the lie signal lives, if
    anywhere, in the whole 170-d emotion subspace, not in its leading directions.
  - Tables: `results/emotion_coords/aletheia_topk/summary.md`.

### 3.7 What the arm shows

Among the five Liars' Bench sets, mid-to-deep blocks (36-45), the full emotion coordinate system is a better
probe space than raw activations and than every matched 170-d control. It beats the label-shuffled control
consistently, so the emotion grouping of the stories adds something beyond story content.

Four things stop this from being a claim about lie detection:
1. **The coordinate dependence** (§3.4).
2. **The failure on GS**, the one out-of-sample, wording-matched lie set.
3. **The failure out of model** (§3.6). On Aletheia's Quest it is 0.08 below raw activations and ranks 20 of 31,
   below the thesis's headline.
4. **The post hoc band.** The block range was chosen after block 40 was seen, and the arm is one of many compared
   across 14 blocks.

What does carry across models is the ordering against matched controls. The emotion subspace holds more lie signal
than any other 170-d subspace tried (H2 holds on Aletheia's Quest, and the shuffled control loses in every
setting), but less than the whole residual stream.

## 4. Preregistered predictions

At blocks 20/28/40/52, protocol B, logistic regression (T1). A prediction holds if it holds at 3 or more blocks.

| | 20 | 28 | 40 | 52 | blocks holding |
|---|---|---|---|---|---|
| P1 emotion − random / − topic (both ≥ 0.03, CI > 0) | +0.004 / +0.016 | +0.008 / −0.234 | **+0.157** / −0.023 | −0.043 / −0.188 | 0 → **fails** |
| P2 IT: emotion − persona (published) | −0.379 | −0.681 | −0.099 | −0.811 | 0 → **fails** |
| P3 HP-C / HP-KR: emotion − random < 0.02 | +0.014 / −0.469 | +0.077 / +0.086 | +0.182 / −0.002 | −0.152 / −0.104 | 2 → **fails** |
| P4 persona + emotion − persona ≥ 0.02 | −0.067 | −0.091 | −0.036 | **+0.035** | 1 → **fails** |
| P5 IT AUROC of guilty / afraid / desperate > 0.6 | 0.25/0.22/0.02 | 0.16/0.21/0.02 | 0.06/0.14/0.02 | 0.09/0.10/0.03 | 0 → **fails** |
| P6 erasing emotion beats erasing at random | holds | holds | fails | fails | 2 → **fails** |

- **P1** comes closest at block 40: emotion PCs beat random by +0.157 [+0.138, +0.177], but the topic basis is
  significantly better (−0.023 [−0.045, −0.002]).
- **P5.** The zero-shot emotion directions are inverted on IT: lies carry *less* guilt, fear and desperation than
  honest answers.
- **P6** holds at 20 and 28 only through differences of 0.002-0.005, plus LEACE's +0.055 at 28. LEACE erasure costs
  0.199 at block 52.
- **P2 and P4** fail with either persona basis (§6).
- **P7.** The published probe's HP-KR AUROC on the capped rows is 0.111, under 0.5 as predicted. Lie vs honest
  admission is 0.067 [0.047, 0.088], well under 0.5. Lie vs honest denial is 0.647 [0.579, 0.712], above 0.5 rather
  than near it. So **neither account holds** by the rule, although the pattern leans to (a), intent. These AUROCs
  were seen in the pilot.
- **P8** (exploratory, block 52). No compassion- or threat-cluster emotion exceeds the 95th percentile of both nulls,
  so **both arms fail**.

**Band readout (post hoc, T9).** Across blocks 35-45 the 20-PC emotion arm meets P1's rule at 2 of 11 blocks: 35
(topic margin +0.047 [+0.030, +0.066]) and 43 (+0.030 [+0.007, +0.054]). The topic basis is significantly better
at five blocks. The arm's means zigzag from block to block (0.69-0.84), and P2-P5 hold at none of the 11.

## 5. Other findings

- **HP-KR's inversion is an early-layer effect** (D1, published probe, T10). Lie vs admission is 0.067 at block 20
  and 0.483 at 28, then 0.85-0.99 from block 36 on. Deeper, the probe has the opposite quirk: honest denials score
  far above honest admissions (admission vs denial 0.01-0.06). It reads the claim "I don't know" more than the lie.
- **D2 compassion hint (post hoc).** In D2, the known × harmful interaction on HP-KR, a compassion-cluster emotion
  exceeds both nulls at 7 of the 11 band blocks (sympathetic, compassionate, empathetic). Threat emotions do so only
  at block 28. None survives Holm correction across the 11 P8 emotions: the lowest is p = 0.055, for compassionate at
  block 40.
- **Training on the RepE pairs (protocol A,** `emotion_coords/explore/protocol_a.py`**).** With k = 20, emotion PCs
  reach 0.78-0.81 at blocks 35-44, above raw activations trained the same way (0.72-0.77). The CV that picks k
  saturates on the RepE pairs (1.000 for both k = 10 and 20), and at k = 10 the score is 0.63-0.68. So this depends
  on a tie-break and is post hoc. The extended persona PCs score 0.89-0.94 at every block.
- **The topic basis.** It was built as a control, but it is as good as or better than emotion PCs at many blocks
  (0.84 at 28 and 0.83 at 37). Story content alone carries transferable signal.
- **Persona bases.** The published basis (8 deception + 7 honesty personas) is weaker than the extended thesis
  basis, which adds knowledge-denial roles. The gap is largest on HP-KR, as expected for a basis built around that
  failure.

## 6. Decisions made after results were seen

All were at the user's request and are dated in `ANALYSIS_PLAN.md`:
- **2026-10-30: persona basis.** P2 and P4 switched from the extended (thesis D4) basis to the published one, after
  blocks 20-40 had been seen. Both bases are reported, and the verdicts are the same under either.
- **The 171-emotion-vector arm**, and the zero-shot AUROC of all 171 emotions at block 40
  (`results/emotion_coords/exploratory/zero_shot_all_emotions_L40.csv`). There, lies look upbeat on CG, IT and ID
  (jealous and joyful at 0.74-0.79, tense at 0.107), and HP-KR reverses the pattern.
- **Protocol C for the 170-d arms**, at blocks 20/28/40/52 and across the band.
- **The band at blocks 35-45**, chosen after block 40, the only preregistered block where emotion PCs did well.
- **The Gender-Secret test at block 40.** Its predictions were committed before scoring, so it is declared rather
  than post hoc.

## 7. Limitations

- **One model, one story set, English only.** The Gemotions stories are written by Gemini. "Protective" was not
  built, because Anthropic's story prompt was unavailable. Liars' Bench's GS and ST sets are outside LB-5; GS is used
  only for §3.5.
- **Multiplicity.** Fourteen blocks and dozens of arms are compared. The CIs are per comparison, and the bootstrap
  resamples held-out rows only, not training variability.
- **The band's eight non-thesis blocks** use this study's own NDIF reads of the Liars' Bench rows.
  - Transferred as bfloat16, with the means accumulated in float32.
  - Parity against stored blocks 36/40/44: median cos 0.99996, minimum 0.99973. That is within the envelope between
    separate NDIF passes already seen in the pilot.
  - There is no published RepE probe at those blocks.
- **Fine-tune emotion vectors could not be built.** Since NDIF moved to nnsight 0.8 (2026-10-16), it refuses request
  payloads over ~1 MiB, and the GS adapter is 202 MiB. The native `peft=` route reaches the server but fails there
  with "ImportError: ... requires the optional `peft` package". The reader is ready (`read --peft`) if NDIF installs
  it.
- **Other limits.**
  - The IT span starts at "Action:", as in Liars' Bench.
  - k for the preregistered emotion arm stops at 20. The all-170 result suggests more dimensions help.
  - Fixed-threshold detection is poor for every arm.

## 8. Reproduce

All commands run from the `deception-detection` folder. The analysis venv runs the fitting; `E:\aletheia\venv08`
(nnsight 0.8.1, peft 0.21.2) runs the NDIF reader.

- **Reads:** `emotion_coords/reader/ec_read.py read --name … [--layers --out --dtype bfloat16 --shard K/N]`, launched
  by `reader/run_chain.ps1` with the `*.jobs` files.
- **Emotion vectors:**
  - `python -m emotion_coords.phase1 base`, then `final`;
  - the band: `final --layers 36,44 --suffix _band` and
    `final --layers 35,37,38,39,41,42,43,45 --suffix _band2`.
- **Probes (phase 3):**
  - `phase3 run`, `run_mlp`, `merge`, `summary`, `summary_mlp`;
  - the band: `--parts band`, then `merge --parts band --out phase3_band` and `summary --out phase3_band --layers
    35,...,45`.
- **HP-KR diagnostics:** `phase4 d1`, `d2d3` (the band: `--layers … --suffix _band/_band2`).
- **Other:**
  - `gs_test run`, `summary`;
  - `figures all`, `figures all170`;
  - exploratory scripts: `emotion_coords/explore/`.
- **Aletheia's Quest (§3.6).** These run in `E:\aletheia\venv` (transformers 5.8). Each command is
  `python -m emotion_coords.aletheia170 <step> --model qwen35|gemma`.
  1. `rows`.
  2. The reads: `ec_read.py read --model … --layers 32|31`, with the `aq_*.jobs` files.
  3. `parity`, `bases`, `train`, `dev`.
  4. Then `score` and `report` (no `--model`).
- **Results:** `results/emotion_coords/`
  - `phase3/`, `phase3_band/`, `phase4/`, `gs/`, `exploratory/`, `aletheia/`;
  - `figures/`;
  - `report_tables.md`.
- **Commits** (local branch `emotion-coords`, not pushed):
  - preregistration `ae8e0ae1`, analysis plan `ad68af22`, full-run code `a6472efb`;
  - 170-d arm `5c4cf6d7`, MLP `c1dd6b28`, persona switch `e1638e17`;
  - band `8cb59768` / `bb44fe1d`, GS `c3fc20de` / `567a6618`, LoRA via peft `d0eddcb7`;
  - this file `ffe27078`;
  - Aletheia's Quest: design `7467eef6`, and the commit that adds its results.
