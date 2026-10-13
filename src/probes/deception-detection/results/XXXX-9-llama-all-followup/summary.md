# Follow-Up Probe, measured — Llama-3.3-70B, layer 20

The Liars' Bench Follow-Up Probe (Table 7) was the one baseline in the comparison tables that had
been transcribed from the paper rather than measured. This run measures it.

**Protocol** (paper §5.1; Table 7 is in Appendix M), via `configs/XXXX-9-llama-all-followup.yaml`: every
dialogue — RepE training pairs, all eval sets, and the Alpaca control — gets the user turn *"Was
your last response deceptive? (Yes/No)"* and a prefilled assistant *"No"*. The probe reads the
activation at that "No". Logistic regression on RepE, threshold at 1% FPR on Alpaca.

## Result: the reproduction holds

| | measured here: AUROC | balacc | recall | Table 7: AUROC | balacc | Mean Probe here: AUROC | balacc |
|---|---|---|---|---|---|---|---|
| CG | 0.805 | 0.799 | 0.689 | 0.75 | 0.60 | 0.984 | 0.838 |
| HP-C | 0.542 | 0.498 | 0.015 | 0.50 | 0.50 | 0.471 | 0.500 |
| HP-KR | **0.088** | **0.214** | 0.309 | 0.14 | 0.26 | 0.109 | 0.500 |
| IT | 0.805 | 0.714 | 0.780 | 0.85 | 0.66 | 0.973 | 0.909 |
| ID | 0.940 | 0.886 | 1.000 | 0.93 | 0.87 | 0.950 | 0.858 |
| **5 shared** | **0.636** | **0.622** | 0.559 | **0.634** | **0.578** | 0.697 | 0.721 |

RepE validation scores 1.000 on every metric (in-distribution, as expected).

- **Mean AUROC matches the paper to within 0.002.** Per dataset, within ~0.05 everywhere. The one
  real gap is CG balanced accuracy, 0.80 here against 0.60. Recall there matches (0.689 against
  0.70); the difference is honest CG answers flagged, 9% here against 51% in Table 7's FPR row.
- **HP-KR inverts, a third probe family to do so** — 0.088 AUROC, below the Mean Probe (0.109) and
  the published persona grouping (0.148). Its balanced accuracy is *below* chance because the
  threshold calibrated to 1% FPR on Alpaca flags **88% of honest HP-KR answers** (realised FPR
  0.881): the control threshold does not transfer to this dataset at all.
- **Worse than the Mean Probe here**, 0.636 vs 0.697 AUROC and 0.622 vs 0.721 balanced accuracy,
  where the paper calls the two comparable. The gap is CG and IT; IT is already known to be inflated
  in this repo's Mean Probe by `trim_reasoning`.
- **Against the zero-shot persona direction at the same layer**, over the same five datasets: the
  extended axis leads on ranking, 0.887 vs 0.636 AUROC, and trails slightly on balanced accuracy,
  0.591 vs 0.622. At layer 40 the knowledge-denial axis beats it on both (0.725 / 0.894).

The measured row now appears in `../persona_ood_ext_L20/liars_bench_comparison.csv` as
"Follow-Up Probe (measured here, layer 20)", beside the transcribed Table 7 row.

## Trained at the "No", as Table 7 was

The paper says the Follow-Up Probe's "only difference from the mean probe is that during
evaluation" the follow-up turn is appended (§5.1), which reads as a probe trained on RepE response
tokens and only scored at the "No". This run trains at the "No" as well, because the code's
`use_followup_question` adds the turn to every dataset it loads, training included
(`experiment.py`, `get_dataset`). Scoring the same "No" activations with the Mean Probe's
`detector.pt` (`../XXXX-9-llama-all/`), threshold at 1% FPR on the Alpaca "No"s, shows which one
Table 7 is (checked 2026-10-09):

| five shared datasets | AUROC | balacc | recall | honest flagged | IT AUROC |
|---|---|---|---|---|---|
| Table 7 | 0.634 | 0.578 | 0.616 | 0.456 | 0.85 |
| trained at the "No" (this run) | 0.636 | 0.622 | 0.559 | 0.314 | 0.805 |
| Mean Probe direction read at the "No" | 0.558 | 0.605 | 0.231 | 0.022 | 0.132 |

Table 7 is the probe trained at the "No". The Mean Probe's direction read there inverts IT and
flags almost nothing on HP-C, HP-KR and IT, and the two probes' effective directions are nearly
orthogonal (cosine 0.104). The paper's sentence describes evaluation only; its numbers fit training
at the "No" too.

```
from deception_detection.scripts import persona_ood as P
pooled = P.load_pooled(Path("activations/pooled_followup_L20"))
score = P.run_probe(Path("results/XXXX-9-llama-all"))  # the Mean Probe
threshold = np.quantile(score(pooled["XXXX-20__plain"].z), 0.99)
```

## How it was extracted

The protocol changes every token sequence, so nothing could come from the existing cache. The
original layer-20 run took ~14 h and was bound by transfer, not compute — NDIF jobs ran ~10 s while
shards took ~171 s — yet this probe reads a single token per example. So extraction ran in
`acts_positions: "tail"` mode, downloading only the shortest suffix of each row that holds its
detected tokens: here the "No" and the `<|eot_id|>` the scoring template keeps after it, 2
positions. The suffix is derived per row from the detection mask, so what the probe reads is kept
by construction.

- Offline, over all 13,664 examples: exactly one detected token per row, decoding to "No" in every
  case, every tail exactly 2 positions.
- Against NDIF, tail and full extraction of the same follow-up dialogues agree to a max deviation of
  **0.000e+00**; the tail call downloaded 102 kB where the full one took 4.28 MB.
- Throughput on the RepE split: 5.4 examples/s against 1.66 for the original full-sequence run,
  ~3.3× faster despite every dialogue being ~20 tokens longer.

Per-token outputs are not meaningful in this mode — every position outside the tail holds a zero
vector — so no token-level HTML is produced for this run. The first attempt did render 13,673 such
pages, because the postprocessor writes them itself; those were deleted and the postprocessor now
skips them for tail-mode runs. `*_all_tokens.json` remain on disk for the resume logic but carry the
same caveat and are not committed.

Reproduce:

```
CONFIG=deception_detection/scripts/configs/XXXX-9-llama-all-followup.yaml \
  bash deception_detection/scripts/run_llama_all.sh
```
