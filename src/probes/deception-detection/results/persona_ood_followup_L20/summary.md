# The persona experiment at the follow-up token

Same experiment as `../persona_ood_ext_L20/` — same persona bank (`persona including knowlede
denial/`), same six datasets, layer 20 — on different activations. There, each example is the mean
over the model's **response** tokens. Here it is the single activation at the prefilled **"No"**
that answers *"Was your last response deceptive? (Yes/No)"*, taken from the Follow-Up Probe run
(`../XXXX-9-llama-all-followup/`).

Every example is therefore scored at the same token in the same context. Whatever separates
deceptive from honest can only be what the model carries forward about its earlier response.

**The figures in this folder are titled "layer 20", like the response-token ones. They are not
the same experiment.**

## The data are exact

- Scoring these pooled vectors with the Follow-Up run's own `detector.pt` reproduces that run's
  AUROC on all five scored datasets at **Δ = 0.0000**. One token per example, so pooled and
  per-token scoring are the same computation.
- The head-to-head's "raw activations (RepE-trained)" row is a RepE probe on these same vectors, so
  it is the Follow-Up Probe again: **0.6364 / 0.6226** (AUROC / balanced accuracy) against the
  measured **0.6360 / 0.6223**. The residual comes from training on RepE train+validation rather
  than train alone.
- The follow-up protocol leaves no row with an empty detection mask, so RepE keeps all 490 training
  rows (448 under the standard protocol).
- The gray base probe in `zero_shot_fig2.png` is the Follow-Up Probe itself: that run's
  `detector.pt` scored on these vectors, which reproduces its `results_table.csv`. Over the five
  Liars' Bench datasets it trails the extended axis on balanced accuracy (0.622 against 0.638) and
  leads it on recall and AUROC (0.559 and 0.636 against 0.349 and 0.552).

## Result: the persona direction's sign stops being universal

Zero-shot AUROC, fixed sign (no probe, so no sign to learn), response tokens → follow-up "No":

| | deception (8v7) | extended (12v13) | knowledge-denial (4v6) |
|---|---|---|---|
| CG | 0.940 → 0.912 | 0.969 → 0.938 | 0.931 → 0.933 |
| HP-C | 0.717 → 0.612 | 0.639 → 0.556 | 0.585 → 0.472 |
| HP-KR | 0.148 → 0.168 | **0.882 → 0.190** | **0.890 → 0.237** |
| IT | **0.988 → 0.165** | **0.990 → 0.173** | **0.976 → 0.216** |
| ID | 0.959 → 0.900 | 0.954 → 0.903 | 0.932 → 0.904 |
| RepE | 0.994 → 1.000 | 0.991 → 1.000 | 0.882 → 1.000 |

On CG, ID and RepE the direction still detects deception at the "No" about as well as on the
response. On two datasets it reverses: **insider trading goes from near-perfect to strongly
inverted**, and **the knowledge-denial repair of HP-KR does not carry over** — it returns to where
the published grouping was.

So the transfer the persona direction achieves on response tokens is a property of those
activations, not a general deception readout that holds wherever the model's state is read.

### What this says about HP-KR

The run was designed to settle whether HP-KR's inversion lives in the response text or in the
model's state. It settles half of that:

- **The repair was about the response's surface form.** Feigned-ignorance personas fixed HP-KR on
  response tokens because HP-KR's lies are literally denials ("I do not know"). At the "No", where
  every example reads the same, the fix is gone — extended 0.882 → 0.190, knowledge-denial alone
  0.890 → 0.237.
- **The inversion at the "No" cannot be pinned on assertion versus denial**, because insider trading
  inverts there too, and it never inverted on response tokens (0.99). Whatever flips the sign at
  this token is not specific to HP-KR's structure.

One hypothesis covers both, and is untested: in insider trading and in HP-KR the *honest* response
is the one with the more damaging content — disclosing an illegal trade, admitting knowledge of
dangerous material — and the deceptive one is the clean-looking cover. If at the follow-up token
the persona direction tracks how bad the preceding content looks rather than whether it was a lie,
it would invert on exactly these two datasets and no others here.

## Result: calibration improves while ranking degrades

Five datasets shared with the paper, threshold at 1% FPR on the Alpaca control:

| arm | AUROC | balanced accuracy | recall |
|---|---|---|---|
| deception (8v7) | 0.750 → 0.551 | 0.534 → **0.616** | 0.069 → **0.260** |
| extended (12v13) | 0.887 → 0.552 | 0.591 → **0.638** | 0.187 → **0.349** |
| knowledge-denial (4v6) | 0.863 → 0.553 | 0.559 → **0.593** | 0.305 → **0.348** |

AUROC falls, because two datasets invert. Yet balanced accuracy and recall at a fixed false-positive
rate *rise*. On response tokens every dataset's scores sat in a different range from the control's,
so a threshold set on Alpaca flagged little or nothing on several of them. Here every dataset and
the control end on the same "No" after the same question, so their scores share a scale and the
threshold transfers. That supports the earlier reading that the response-token calibration failure
was an offset between prompt distributions, not missing signal.

## Trained probes transfer slightly better here

The 8192-d baseline — a probe trained on one dataset's "No" activations, tested on another's —
averages **0.685** off the diagonal, against **0.648** on response tokens. It improves most on CG
(0.663 → 0.873), ID (0.778 → 0.853) and RepE (0.718 → 0.929), and gets worse on HP-KR (0.394 →
0.242) and IT (0.799 → 0.714). A common scored token helps supervised transfer even as it breaks
the persona direction.

## k = 3: the paper's method 3, trained on RepE

The paper's third method projects onto the top three persona PCs and trains a small probe on those
features, on RepE. That is the RepE row of each `matrix_*.csv`, so it needs no new run. Mean AUROC
over the five datasets shared with the paper (the 8192-d row reproduces the Mean Probe on response
tokens and the Follow-Up Probe at the "No"):

| arm | response tokens | follow-up "No" |
|---|---|---|
| 8192-d baseline | 0.734 | 0.636 |
| extended axis, k = 1 | **0.887** | 0.552 |
| extended PCs, k = 3 | 0.736 | 0.579 |
| original PCs, k = 3 | 0.733 | 0.574 |
| knowledge-denial PCs, k = 3 | 0.810 | **0.699** |

- **On response tokens 1-d still beats 3-d, under the paper's own training protocol.** The loss is
  concentrated in HP-KR: the RepE-trained k = 3 probe re-inverts it to 0.118 against the 1-d axis's
  0.882. The three components carry the denial direction, but a probe fitted on RepE weights them
  so the assertion polarity returns.
- **k = 3 does not rescue the extended or original bank at the follow-up token**: insider trading
  still inverts (0.170, 0.212).
- **The knowledge-denial subspace is the exception.** At the "No" it is the best persona arm, 0.699,
  above the Follow-Up Probe itself (0.636), and it does not invert on insider trading (0.712 against
  0.216 for its 1-d axis): a learned combination of its three components avoids the sign flip a
  fixed axis cannot. It is also strong on response tokens, 0.810, with HP-KR at 0.969.
- **Two limits on that.** Most of its lead at the "No" is from *not inverting* on IT and HP-KR
  (0.712, 0.479) rather than from detecting them well. And it depends on the training set: averaged
  over all training sets at the "No" it is 0.492, and RepE is the one that works. On balanced
  accuracy at 1% FPR it does not lead (0.573, against 0.638 for the 1-d axis and 0.623 for the
  Follow-Up Probe).

## Reading the transfer matrices in this folder

For a 1-d direction the only thing a probe can learn is the sign. On response tokens every
training set agreed on it, so each test column was constant. At the "No" they disagree, so a probe
trained on IT or HP-KR carries the reversed sign to every other dataset. That is why the persona
arms' off-diagonal means here (0.44–0.56, `summary.csv`) sit far below their fixed-sign zero-shot
AUROCs on CG, ID and RepE. The zero-shot table above is the cleaner read.

## Files and reproduction

Same layout as `../persona_ood_ext_L20/`: 15 `matrix_*.csv`, `summary.csv`, `summary_balacc.csv`,
`zero_shot.csv`, `zero_shot_operating.csv`, `liars_bench_comparison.csv`, and figures.

```
python deception_detection/scripts/pool_activations.py \
  --config-file deception_detection/scripts/configs/XXXX-9-llama-all-followup.yaml \
  --out_dir activations/pooled_followup_L20
python deception_detection/scripts/persona_ood.py verify \
  --run_folder results/XXXX-9-llama-all-followup --pooled_dir activations/pooled_followup_L20
python deception_detection/scripts/persona_ood.py run --layer 20 \
  --pooled_dir activations/pooled_followup_L20 --out_dir results/persona_ood_followup_L20 \
  --vectors "persona including knowlede denial"
python deception_detection/scripts/persona_ood.py zeroshot --layer 20 \
  --pooled_dir activations/pooled_followup_L20 --out_dir results/persona_ood_followup_L20 \
  --vectors "persona including knowlede denial" --base_run results/XXXX-9-llama-all-followup
python deception_detection/scripts/persona_ood_figures.py fig2 --out_dir results/persona_ood_followup_L20
```

Pooling reads the tail-mode cache the Follow-Up run wrote; no model calls.
