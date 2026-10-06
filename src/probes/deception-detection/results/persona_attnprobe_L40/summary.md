# Attention probes on persona coordinates

Covers `persona_attnprobe_L20/`, `_L40/` and `_L44/` (run 2026-09-28). Script:
`deception_detection/scripts/persona_attention_probe.py`; the probe is
`deception_detection/attention_probe.py`.

The persona-anchored run (`../persona_attention_L40/summary.md`) kept one persona axis as the
readout and learned only where to look. Here the published attention probe (EleutherAI's
`AttentionProbe`) takes each answer token's persona coordinates as input. It learns per-head
queries, values and a position bias over those coordinates, so it can weigh personas against each
other and choose tokens at once.

## Setup

Each answer token becomes 76 numbers, its projections on unit directions at the layer:

| block | directions |
|---|---|
| P36 | the 36 personas of the extended bank, centred on the bank mean |
| A4 | the four axes: extended (12v13), knowledge-denial (4v6), deception (8v7), sycophancy |
| R36 | 36 random directions, seed 0, the control |

The coordinates are exact. The A4 columns match the committed `token_proj_L*` files to within
8e-6 of their spread at every layer. At layer 40 all 76 columns match a projection of the
extracted answer-token activations to within 2e-5.

The probe port reproduces the reference (github.com/EleutherAI/attention-probes at 889b8dea): with
the same seed, trained parameters, loss and predictions agree to 1e-6
(`tests/test_attention_probe.py`). On mean-pooled coordinates it reduces to logistic regression,
and its CG-trained fit at layer 20 matches scikit-learn's (coefficient correlation 0.994, test
AUROCs within 0.006).

Protocol: the transfer matrix. Train on one of CG, HP-KR, IT, ID, RepE, test on all six, and
average the 25 off-diagonal cells. Weight decay comes from the reference's grid (0, 1e-3, 1e-2,
0.1), chosen on a stratified held-out fifth of the training set. Each model is then refit on the
whole training set with three seeds, and their logits are averaged. The threshold is the 99th
percentile of the model's Alpaca scores. Fixed reads average the same 25 cells.

Arms:

- P36 and A4 attention probes with 1 and 8 heads, and R36 with 8 heads.
- The reference's one-token baselines on P36: mean-pooled and last token.
- An ablation not in the reference: the 1-head P36 probe with no position bias.

Deviations from the reference:

- Coordinates are standardised by Alpaca's per-token mean and SD.
- Training is full batch on every example. The reference cuts each training set to 2048 examples.
- Weight decay is chosen on a held-out fifth by AUROC. The reference uses 5-fold accuracy.
- Scoring runs in float32. The reference scores under autocast.

## Trained probes transfer worse than the fixed persona reads, at every layer

Mean off-diagonal AUROC:

| model | L20 | L40 | L44 |
|---|---|---|---|
| mean read, extended | 0.894 | 0.908 | 0.896 |
| top-10% read, extended | 0.883 | 0.914 | 0.914 |
| max read, extended | 0.872 | 0.916 | 0.920 |
| mean read, knowledge-denial | 0.855 | 0.905 | 0.893 |
| top-10% read, knowledge-denial | 0.877 | 0.928 | 0.912 |
| max read, knowledge-denial | 0.842 | 0.927 | 0.916 |
| P36 attention, 1 head | 0.592 | 0.849 | 0.831 |
| P36 attention, 8 heads | 0.719 | 0.841 | 0.828 |
| A4 attention, 1 head | 0.763 | 0.804 | 0.759 |
| A4 attention, 8 heads | 0.689 | 0.838 | 0.861 |
| P36 mean-pooled | 0.640 | 0.774 | 0.846 |
| P36 last token | 0.708 | 0.716 | 0.772 |
| P36 attention, 1 head, no position | 0.644 | 0.819 | 0.846 |
| R36 attention, 8 heads (random) | 0.590 | 0.644 | 0.648 |
| raw 8192-d LR probe (`persona_ood_ext_L*`) | 0.648 | 0.727 | 0.711 |

The fixed-read rows reproduce the earlier runs: 0.908 and 0.916 for the extended mean and max
reads at layer 40, 0.927 for the knowledge-denial max. The standard deviation of the transfer mean
across the three seeds is at most 0.027 (A4, 8 heads, layer 20), and under 0.01 for 21 of the 24
trained rows.

The best trained arm stays 0.06 to 0.13 below the best fixed read. HP-KR accounts for part of
the gap, but not all of it:

| | L20 | L40 | L44 |
|---|---|---|---|
| best fixed read | 0.894 | 0.928 | 0.920 |
| best trained arm | 0.763 | 0.849 | 0.861 |
| best fixed read, HP-KR not tested | 0.896 | 0.924 | 0.919 |
| best trained arm, HP-KR not tested | 0.832 | 0.887 | 0.889 |

Paired bootstrap per test set, for the four attention arms and the no-position ablation, against
the knowledge-denial max read and the extended top-10% read (60 comparisons per layer):

| | L20 | L40 | L44 |
|---|---|---|---|
| significant loss | 50 | 53 | 52 |
| no significant difference | 6 | 6 | 6 |
| significant gain | 4 | 1 | 2 |

The gains are small or come against a weak reference. At layer 20 there are four: two on IT, where
the knowledge-denial max read is weak (0.826), and two on HP-C. At layer 40 the gain is +0.001 on
RepE (1.000 against 0.999). At layer 44 both gains are the no-position ablation on IT, +0.007
against each reference.

## At the 1% FPR operating point the gap is wider

Mean off-diagonal balanced accuracy, threshold at 1% FPR on Alpaca:

| model | L20 | L40 | L44 |
|---|---|---|---|
| top-10% read, extended | 0.709 | 0.763 | 0.739 |
| mean read, knowledge-denial | 0.590 | 0.760 | 0.725 |
| best trained arm (A4 attention) | 0.587 | 0.611 | 0.604 |
| P36 attention, 1 head | 0.557 | 0.560 | 0.568 |
| raw 8192-d LR probe | 0.565 | 0.608 | 0.608 |
| R36 attention, 8 heads | 0.520 | 0.539 | 0.532 |

At the threshold, the best probe trained on persona coordinates is at most 0.02 better than the raw
8192-d probe, and the best fixed read leads both by 0.12 to 0.16.

## HP-KR: probes trained on CG invert it at every layer

HP-KR AUROC of every trained arm, by training set (HP-KR-trained models left out):

| layer | CG-trained | IT-trained | ID-trained | RepE-trained |
|---|---|---|---|---|
| L20 | 0.047 to 0.230 | 0.106 to 0.951 | 0.049 to 0.820 | 0.072 to 0.598 |
| L40 | 0.099 to 0.270 | 0.131 to 0.970 | 0.684 to 0.961 | 0.701 to 0.930 |
| L44 | 0.112 to 0.321 | 0.293 to 0.973 | 0.412 to 0.945 | 0.404 to 0.922 |

The fixed reads rank HP-KR at 0.81 to 0.93 at layer 20 and 0.89 to 0.95 at layers 40 and 44.
Each range spans the eight trained arms, so the inversion from CG is not a property of one
architecture.

At the threshold, most trained probes flag no HP-KR lie. Only 7 of the 96 trained rows flag any,
all trained on ID or IT (`hpkr.csv`). The two that flag most also flag honest denials at about the
same rate:

- A4, 1 head, ID-trained, layer 40: 0.878 of lies and 0.852 of honest denials.
- A4, 8 heads, IT-trained, layer 20: 0.741 of lies and 0.778 of honest denials.

That is the wording confound already listed as a limitation for the knowledge-denial mean read
(0.885 and 0.948 at layer 40). The next largest, the ID-trained mean-pooled probe at layer 20,
flags 0.388 of lies but also 0.275 of honest claims.

## What the probes learn: each training set's own style of lie

The value weights, summed over heads, show which personas each probe reads. They are in
`fits.csv`.

- CG-trained probes put most of their positive weight on the sycophancy group (cheerleader,
  flatterer, yes-man, people-pleaser, enabler), with the spin doctor or cover-up operator beside
  them. The A4 probe puts its largest weight on the sycophancy axis at every layer (+5.2, +2.9 and
  +3.3). This matches the unified-axis finding that the sycophancy axis is the most inverted arm
  on HP-KR.
- HP-KR-trained probes load on the deniable expert (a knowledge-denial role) and the honest novice
  (a candour role) at every layer. Both roles say they do not know, which is the wording confound
  again. The A4 probe weights the extended axis positively and the deception axis negatively.
- RepE-trained and IT-trained probes load mostly on the deception group: cover-up operator, spin
  doctor, impression manager, broker.

Regularisation is not the fix. CG-trained and RepE-trained fits chose the largest weight decay in
the grid (0.1) at every layer, and CG-trained probes still invert HP-KR.

## Attention and the position bias

- **Attention against pooling the same features.** The 1-head P36 probe beats the mean-pooled one
  at layer 40 (0.849 against 0.774). It loses at layer 20 (0.592 against 0.640) and at layer 44
  (0.831 against 0.846). There is no consistent benefit.
- **The position bias extrapolates.** The reference adds a learned slope times token position. A
  slope fitted on one dataset's answer lengths collapses attention onto one end of longer answers
  (`attention_profile.csv`). At layer 40 the RepE-trained probe puts all of CG's attention in its
  first tenth, and the ID-trained probe puts 86% in the last tenth.
- **Removing the position bias does not change the picture.** Transfer moves by +0.052, -0.030
  and +0.015 at layers 20, 40 and 44.

## Persona coordinates against random directions

The 8-head probe on P36 beats the same probe on R36 at every layer (0.719, 0.841, 0.828 against
0.590, 0.644, 0.648). R36 fits its training sets as well as P36 (held-out AUROC 0.98 to 1.00 for
both), so the difference is in transfer. At layers 40 and 44 every P36 and A4 arm also beats the
raw 8192-d probe on AUROC, except the last-token baseline at layer 40 (0.716 against 0.727). The
best reach 0.849 against 0.727, and 0.861 against 0.711. At layer 20
the margin is smaller, and three of the seven P36 and A4 arms fall below the raw probe's 0.648.

## Conclusion

Letting a probe learn how to combine persona coordinates costs transfer. Thirty-six (or four)
inputs are enough for a trained probe to fit its training set's own style of lie, and that style
does not carry to the other datasets. The persona-anchored attention run keeps the persona axis as
a fixed readout and reaches 0.908 and 0.910 at layer 40, above every arm here. Fixing the readout
is what transfers, and learned attention on top of it did not beat the max read there either. The
zero-shot persona axes remain the method, with the top-10% read for the operating point.

Limitations:

- One persona bank and three layers. Layer 36, the ranking peak, has no per-token cache.
- Weight decay is chosen in-distribution, as the protocol requires. Choosing it by transfer would
  select on the test sets.
- HP-C is never a training set, because its answers are one or two tokens.

## Files

Per layer, in `persona_attnprobe_L{20,40,44}/`:

- `transfer_summary.csv`: the transfer means.
- `cells.csv`: every train and test cell (AUROC, its seed SD, recall, specificity, balanced accuracy).
- `per_test_auroc.csv`, `per_test_balacc.csv`: per test set.
- `bootstrap.csv`, `hpkr.csv`, `fits.csv`, `attention_profile.csv`, `cfg.json`, `transfer.png`.
- `fits/`: every fit's scores and weights, which `report` reads.

The coordinates are in `activations/persona_coords_L*` (242 MB per layer). The reference is
copied verbatim to `tests/attention_probe_reference.py`. Its repository has no license file, so
that copy should stay out of a public commit.
