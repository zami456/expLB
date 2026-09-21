# The persona direction at every token

Covers `persona_positions_L20/`, `_L40/` and `_L44/`. Same persona bank (`persona including
knowlede denial/`) and datasets as `persona_ood_ext_L*`; the only change is *where* the direction
is read. Every persona result before this read one number per example — the direction applied to
the **mean** over response tokens. The activation caches hold every token of every example, so here
the direction is scored at each position, with no model calls, and six reads are compared:

| read | what it takes |
|---|---|
| mean | the existing method |
| max, top-10% mean | the most deceptive-looking token(s) |
| first, last | the first and last response token |
| before the reply | the last prompt token — the end of the assistant header, as the model starts to answer |

## The data are exact

`axis_score` is a plain dot product, so averaging per-token projections over the response equals
projecting the pooled vector. Checked at every layer, for all four axes and six datasets:

- per-example scores equal the pooled projections to within 6e-6 (L20), 1.6e-5 (L40) and 1.4e-5
  (L44) of the score's spread, with labels aligned row by row;
- AUROC matches the committed `zero_shot.csv` to 1e-16 at layer 20, and to 1.9e-6 and 3.7e-6 at 40
  and 44 — float32 rounding reorders a few near-tied examples where activations are larger.

## For ranking, the mean is nearly optimal — except for CG at depth

Extended axis, zero-shot AUROC:

| | CG | HP-C | HP-KR | IT | ID | RepE |
|---|---|---|---|---|---|---|
| L40 mean | 0.912 | 0.712 | 0.912 | 0.980 | 0.978 | 1.000 |
| L40 max | **0.993** | 0.711 | 0.900 | 0.975 | 0.969 | 1.000 |
| L44 mean | 0.859 | 0.712 | 0.894 | 0.982 | 0.974 | 1.000 |
| L44 max | **0.991** | 0.711 | 0.925 | 0.976 | 0.966 | 1.000 |

On five of six datasets the mean is within a few thousandths of the best read at every layer, and
paired bootstrap intervals put the alternatives mostly *below* it. HP-KR shows why: its signal is
spread across the response — at layer 20 the mean reaches 0.882 while no single position does
better than 0.90 and most do far worse.

**CG is the exception, and it explains an earlier puzzle.** The layer-44 summary recorded that CG
degrades steadily with depth under the mean — 0.969, 0.912, 0.859 — alone among the six. Under the
max it does not degrade at all: **0.954, 0.993, 0.991**. It was a pooling artifact. At depth CG's
lie is carried by a few tokens whose position varies from example to example: every tenth of the
response scores 0.79–0.97 at layer 44, yet the per-example peak scores 0.991. No fixed position
catches it; the mean dilutes it; the max does not.

## For detection at a fixed false-positive rate, read the top 10%

Mean over the five datasets shared with the paper, threshold at 1% FPR on the Alpaca control,
extended axis:

| read | L20 AUROC / balacc / recall | **L40** | L44 |
|---|---|---|---|
| mean | 0.887 / 0.591 / 0.187 | 0.899 / 0.694 / 0.413 | 0.884 / 0.666 / 0.350 |
| **top-10% mean** | 0.873 / 0.663 / 0.348 | **0.907 / 0.737 / 0.531** | 0.907 / 0.715 / 0.470 |
| max | 0.858 / 0.659 / 0.336 | 0.909 / 0.750 / 0.548 | 0.914 / 0.701 / 0.418 |

Peak reads nearly double recall at layer 20 and add a third at 40. They spread the deceptive tail
further from the control's, which is the calibration problem the earlier runs kept hitting.

**Top-10% is the robust choice over max.** Max scores higher on average but collapses RepE's
balanced accuracy (0.998 → 0.638 at layer 40) — one extreme token per example is a noisy
statistic. Top-10% keeps RepE at 0.958 while lifting CG from 0.680 to 0.844 and IT from 0.869 to
0.917. **Layer-40 top-10% is the best operating point found in this study: 0.737 balanced
accuracy, 0.531 recall.** HP-C and HP-KR stay on the 0.5 floor under every read: their whole score
distribution sits below the control's threshold whichever token is read.

## Where in the response the signal sits

There is **no late build-up**. Aletheia's retrospective suggests models asserting a falsehood
"spend more of the trajectory hedging … before finally committing"; if the persona direction
tracked that, it would rise late in the assertive-lie datasets. It does not: IT and ID are flat
from the second token on, CG is highest at the start and end, and RepE is front-loaded (0.98 in the
first tenth, 0.65 in the last, at layer 20).

Single positions are noisy in a way that tracks the token rather than the timing — insider
trading's token 10 drops to 0.33 at layer 20 because its reports are templated and that slot is
usually the same uninformative word. Averaging over tokens, or over the top 10% of them, removes
that; any fixed-position read inherits it.

## The state before the reply mostly reads the prompt

The last prompt token separates well where the deception is *instructed* — RepE 1.000, CG 1.000 at
layers 40 and 44, ID 0.94 — because the instruction to lie is already in context. On HP-KR, where
the model decides, it is inverted at layer 20 (0.386) and near chance deeper (0.57, 0.62). It
measures the instruction as much as any intent, and is not a clean pre-commitment signal.

## Consequence

For the self-evaluation fusion experiment, the persona leg should be the **top-10% read at layer
40**, not the mean.

## Files

Per layer: `reads.csv` (every read × axis × dataset: AUROC, balanced accuracy, recall),
`reads_summary.csv` (means over the shared datasets), `read_vs_mean.csv` (paired bootstrap, extended
axis), `position_profile.csv` (relative tenths and first 16 tokens), and `reads.png`,
`position_profile.png`. Per-token projections live in `activations/token_proj_L*` (local).

```
python deception_detection/scripts/persona_positions.py project \
  --config-file deception_detection/scripts/configs/XXXX-9-llama-all-L40.yaml --layer 40 \
  --vectors "persona including knowlede denial" --out_dir activations/token_proj_L40
python deception_detection/scripts/persona_positions.py verify --proj_dir activations/token_proj_L40 \
  --zero_shot results/persona_ood_ext_L40/zero_shot.csv --pooled_dir activations/pooled_L40
python deception_detection/scripts/persona_positions.py analyze \
  --proj_dir activations/token_proj_L40 --out_dir results/persona_positions_L40
python deception_detection/scripts/persona_positions.py figures --out_dir results/persona_positions_L40
```

Layers 40 and 44 need `ACTS_CACHE_DIR=E:/liars-bench-acts/cache` for `project`.
