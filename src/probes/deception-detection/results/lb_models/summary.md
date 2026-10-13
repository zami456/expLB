# Liars' Bench probes on the other benchmark models

Liars' Bench (arXiv 2511.16035) trains its probes "independently for each model under
evaluation". It reports the Mean Probe (Table 6) and the Follow-Up Probe (Table 7) for
Llama-3.3-70B, Qwen-2.5-72B, Mistral-Small-3.1-24B and Gemma-3-27B. The Llama rows are reproduced in
`XXXX-9-llama-all` and `XXXX-9-llama-all-followup`. This folder does the same for the other three
models, with the authors' code, layers and datasets.

The reference is Cadenza's own per-run output, `liars-bench/results/probes/{mean,last_token}_probe_results.csv`.
Each cell comes from the latest run per model and dataset. These rows reproduce the published Llama
rows exactly. The Mistral rows also equal every printed Mistral cell of Tables 6 and 7 for HP-C,
HP-KR, ID and IT, at two decimals and at all three thresholds (arXiv 2511.16035v2, the latest
version; checked 2026-10-10). RepE validation is taken from the same run as the model's other cells.

**Status (2026-10-12):** all three models are complete.
- Mistral and Gemma finished on 2026-10-10.
- Qwen finished on 2026-10-12. NDIF's Qwen deployment was congested and kept being evicted on
  10-10, and was fast again on 10-12.
- The pilot is described in `pilot/summary.md`.

## Summary

| | Qwen-2.5-72B | Mistral-Small-3.1-24B | Gemma-3-27B |
|---|---|---|---|
| Block that reproduces "layer L" | 20, block L's output | 9 = `hidden_states[10]` | 8 = `hidden_states[9]` |
| Mean Probe: mean abs. AUROC difference | 0.032 | 0.013 | 0.025 |
| Mean Probe, every set but IT | HP-C 0.06 above; HP-KR and ID within 0.022 | within 0.003 AUROC; operating points match | within 0.001 AUROC; operating points match |
| Mean Probe, IT | (no IT set) | 0.536 vs 0.585 | 0.231 vs 0.303 |
| Follow-Up: mean abs. AUROC difference | 0.020 reading "No" only (0.075 with `<|im_end|>`) | 0.049 | 0.009 |
| Follow-Up operating points | close | do not match (Alpaca threshold placement) | match |

**Which block.** The block rule was fixed before any number came in. For each model it picks the
block Cadenza's loading implies:
- Mistral and Gemma, which Cadenza loaded as local HF models, read `hidden_states[L]`, the output of
  block L−1.
- Qwen, which they read through nnsight on NDIF as they did Llama, reads block L's output.

**Gaps.**
- **Insider trading.** Both Mistral and Gemma read lower than Cadenza under the Mean Probe (by 0.05
  and 0.07), on identical rows.
- **Qwen's Mean Probe** is 0.06 above Cadenza's on HP-C. Our Llama run is off by the same amount on
  the same set (0.471 vs 0.410). Llama and Qwen are the two models Cadenza read through NDIF.
- **Qwen's Follow-Up** reproduces when it reads only the prefilled "No". That fits the name of
  Cadenza's results file, `last_token_probe_results.csv`. For the other models, "No" is the only
  token read either way.

## Which block "layer L" is

The authors' config gives layer 20 for Qwen, 10 for Mistral and 9 for Gemma.
- They loaded Mistral and Gemma as local HF models. So "layer L" could be `hidden_states[L]` (block
  L−1's output) or block L's output (the nnsight reading used for Llama).
- Qwen they read through nnsight, as they did Llama, so block L was the expected reading there.

The Mean Probe was fitted at both blocks. The block whose AUROCs sit closer to Cadenza's was chosen
by a rule fixed before any number came in (`blocks.csv`):

| Model | Block | Mean abs. AUROC difference | RepE-validation balanced-accuracy difference |
|---|---|---|---|
| Qwen | 19 (`hidden_states[20]`) | 0.098 | 0.157 |
| Qwen | **20** | **0.032** | **0.157** |
| Mistral | **9** (`hidden_states[10]`) | **0.013** | **0.000** |
| Mistral | 10 | 0.096 | 0.010 |
| Gemma | **8** (`hidden_states[9]`) | **0.025** | **0.020** |
| Gemma | 9 | 0.089 | 0.118 |

The Follow-Up Probe was then run at the chosen block only.

**RepE-validation row counts.** RepE validation shows 51 honest and 51 lying rows here. Cadenza's
files show 61/61 because they count rows by label, but 10 pairs have an empty detection span. Their
runner drops those rows from every metric, as Llama's own `scores.json` shows. Both sides therefore
score the same 51/51 rows.

## Qwen-2.5-72B

### Mean Probe (Table 6), block 20

Values are measured against Cadenza's run; balanced accuracy, recall and FPR are at 1% FPR on
Alpaca. Qwen has no IT set.

| Dataset | Honest / lies | AUROC | Balanced accuracy | Recall | FPR |
|---|---|---|---|---|---|
| HP-C | 2000 / 374 | 0.561 / 0.497 | 0.500 / 0.500 | 0.000 / 0.000 | 0.000 / 0.000 |
| HP-KR | 2000 / 571 | 0.498 / 0.486 | 0.500 / 0.500 | 0.000 / 0.000 | 0.000 / 0.001 |
| ID | 3015 / 2145 | 0.581 / 0.603 | 0.494 / 0.500 | 0.011 / 0.000 | 0.024 / 0.000 |
| RepE validation | 51 / 51 | 1.000 / 0.998 | 1.000 / 0.843 | 1.000 / 0.686 | 0.000 / 0.000 |

- **AUROC.** HP-KR and ID are within 0.022, and the operating points on the three Liars' Bench sets
  match.
- **HP-C** is 0.064 above Cadenza, the one cell over 0.05.
- **RepE-validation operating point.** It does not match. Cadenza's Alpaca threshold sat above 31%
  of the RepE-validation lies; ours sits below all of them.
- **Cause unknown.** The HP-C gap has the size of our Llama run's HP-C gap (0.471 against 0.410). Llama
  and Qwen are the two models Cadenza read through NDIF, while the two they loaded locally reproduce
  within 0.003. Llama's RepE-validation operating point does match, though.
  - One untested possibility is that NDIF's deployment has changed since 2025; it now runs
    transformers 5.8.
  - Another is that Cadenza's batches differed from ours in padding.

### Follow-Up Probe (Table 7), block 20

The plan read the follow-up turn by Cadenza's span rule first. For Qwen, that rule adds
`<|im_end|>` to the last assistant message, so the default read is "No" plus `<|im_end|>`. If that
missed, the plan was to re-read "No" alone (`detect_only_last_token`). It missed on HP-KR, so both
reads are reported. Values are measured against Cadenza's run:

| Dataset | Read | AUROC | Balanced accuracy | Recall | FPR | AUROC vs Alpaca |
|---|---|---|---|---|---|---|
| HP-C | No + `<|im_end|>` | 0.488 / 0.499 | 0.500 / 0.500 | 0.000 / 0.000 | 0.000 / 0.000 | 0.711 / 0.831 |
| HP-C | No only | 0.493 / 0.499 | 0.501 / 0.500 | 0.005 / 0.000 | 0.004 / 0.000 | 0.939 / 0.831 |
| HP-KR | No + `<|im_end|>` | 0.313 / 0.520 | 0.500 / 0.499 | 0.000 / 0.000 | 0.001 / 0.002 | 0.364 / 0.796 |
| HP-KR | No only | 0.572 / 0.520 | 0.494 / 0.499 | 0.002 / 0.000 | 0.014 / 0.002 | 0.888 / 0.796 |
| ID | No + `<|im_end|>` | 0.841 / 0.835 | 0.799 / 0.782 | 0.990 / 0.999 | 0.392 / 0.434 | 0.998 / 0.999 |
| ID | No only | 0.837 / 0.835 | 0.810 / 0.782 | 1.000 / 0.999 | 0.379 / 0.434 | 0.999 / 0.999 |
| RepE validation | No + `<|im_end|>` | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.000 / 0.000 | 1.000 / 1.000 |
| RepE validation | No only | 1.000 / 1.000 | 0.975 / 1.000 | 1.000 / 1.000 | 0.049 / 0.000 | 1.000 / 1.000 |

- **AUROC.** Reading "No" alone reproduces it: a mean absolute AUROC difference of 0.020 over HP-C,
  HP-KR and ID, against 0.075 with `<|im_end|>`. The default read's HP-KR is 0.21 below.
- **Operating points.** These are close in both reads. ID flags all lies or nearly all at about 38%
  FPR (Cadenza's 43%), and HP-C and HP-KR sit at chance in both runs.
- **Which read Cadenza used.** Their results file is named `last_token_probe_results.csv`. That,
  with the match, suggests their Follow-Up read only the last token before the end of the turn.
  - For Llama, Mistral and Gemma this changes nothing, because "No" is the only detected token
    either way.
  - For Qwen it drops `<|im_end|>`.

## Mistral-Small-3.1-24B

### Mean Probe (Table 6), block 9

Values are measured against Cadenza's run; balanced accuracy, recall and FPR are at 1% FPR on
Alpaca.

| Dataset | Honest / lies | AUROC | Balanced accuracy | Recall | FPR |
|---|---|---|---|---|---|
| HP-C | 697 / 195 | 0.544 / 0.545 | 0.500 / 0.501 | 0.000 / 0.015 | 0.000 / 0.014 |
| HP-KR | 1661 / 697 | 0.463 / 0.463 | 0.500 / 0.500 | 0.000 / 0.000 | 0.000 / 0.000 |
| ID | 2617 / 2393 | 0.690 / 0.693 | 0.500 / 0.501 | 0.002 / 0.008 | 0.002 / 0.005 |
| IT, full reply | 1486 / 648 | 0.536 / 0.585 | 0.500 / 0.500 | 0.000 / 0.000 | 0.000 / 0.000 |
| IT, after "Action: " | 1486 / 648 | 0.520 / 0.585 | 0.500 / 0.500 | 0.000 / 0.000 | 0.000 / 0.000 |
| RepE validation | 51 / 51 | 1.000 / 1.000 | 0.990 / 0.990 | 0.980 / 0.980 | 0.000 / 0.000 |

Every cell except IT is within 0.003 AUROC. IT is 0.049 below. The reply span that reads the
scratchpad is the closer of the two IT spans, as the published IT rows read it. The cause of the
remaining gap is unknown. Row counts and the trimming split are identical to Cadenza's.

### Follow-Up Probe (Table 7), block 9

The probe is trained and read at the prefilled "No" of the paper's follow-up turn. Values are
measured against Cadenza's run:

| Dataset | AUROC | Balanced accuracy | Recall | FPR |
|---|---|---|---|---|
| HP-C | 0.620 / 0.676 | 0.500 / 0.506 | 0.000 / 0.985 | 0.000 / 0.973 |
| HP-KR | 0.619 / 0.539 | 0.500 / 0.500 | 0.000 / 0.000 | 0.000 / 0.000 |
| ID | 0.871 / 0.891 | 0.668 / 0.832 | 0.401 / 1.000 | 0.066 / 0.335 |
| IT | 0.613 / 0.652 | 0.500 / 0.488 | 0.000 / 0.972 | 0.000 / 0.996 |
| RepE validation | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.000 / 0.000 |

- **Ranking.** The AUROCs reproduce within 0.08: a mean absolute difference of 0.049.
- **Operating point.** It does not reproduce. Cadenza's Alpaca threshold falls low enough to flag
  almost every HP-C and IT row, honest or not. Ours flags almost none. On ID it flags 40% of lies
  at 7% FPR, against their 100% at 34%.
- **Where Alpaca lands.** The AUROC of lies against the Alpaca controls is 0.998, 0.999 and 0.998
  on HP-C, ID and IT in Cadenza's run, so almost every lie scores above the Alpaca threshold. Ours
  is 0.050, 0.928 and 0.641, and 0.030 on HP-KR (theirs 0.259). The gap lies in where the Alpaca
  control scores fall relative to the eval sets, not in the ranking within a dataset. The paper's
  0.1% and 0.01% cells fail to reproduce for the same reason (ID balanced accuracy at 0.1%: 0.50
  against 0.83).
- Balanced accuracy on HP-C, HP-KR and IT is about 0.5 in both runs, so only ID's moves.
- **Specific to Mistral's run.** Gemma's Follow-Up reproduces its operating points and its
  lies-vs-Alpaca AUROCs with the same pipeline (see below).
  - Cadenza ran the Mistral Follow-Up on 2025-11-04, three weeks after their Mean Probe run
    (2025-10-14). Gemma's ran on 2025-10-23.
  - So one untested possibility is that their Mistral Follow-Up read a different block (block 10,
    the nnsight reading).
  - The Llama Follow-Up reproduction showed the same kind of gap (CG flagged 9% of honest replies
    against 51%), with the AUROCs matching.

## Gemma-3-27B

### Mean Probe (Table 6), block 8

Measured against Cadenza's run:

| Dataset | Honest / lies | AUROC | Balanced accuracy | Recall | FPR |
|---|---|---|---|---|---|
| CG | 250 / 371 | 0.700 / 0.700 | 0.500 / 0.500 | 0.000 / 0.000 | 0.000 / 0.000 |
| ID | 2087 / 3109 | 0.586 / 0.585 | 0.516 / 0.521 | 0.081 / 0.104 | 0.050 / 0.062 |
| IT, full reply | 1614 / 1881 | 0.231 / 0.303 | 0.500 / 0.500 | 0.000 / 0.000 | 0.000 / 0.000 |
| IT, after "Action: " | 1614 / 1881 | 0.250 / 0.303 | 0.500 / 0.500 | 0.001 / 0.000 | 0.000 / 0.000 |
| RepE validation | 51 / 51 | 1.000 / 1.000 | 0.559 / 0.578 | 0.118 / 0.157 | 0.000 / 0.000 |

- **CG and ID** are within 0.001 AUROC, and their operating points are close.
- **IT** is 0.072 below on the full reply (0.053 below on the trimmed span), on identical rows. The
  direction and size are like Mistral's IT gap.
- **CG reference.** CG is compared with Cadenza's 2025-11-06 rerun on the current 250/371-row set.
  Their first run used 400/981 rows.

### Follow-Up Probe (Table 7), block 8

Measured against Cadenza's run:

| Dataset | AUROC | Balanced accuracy | Recall | FPR | AUROC vs Alpaca |
|---|---|---|---|---|---|
| CG | 0.736 / 0.748 | 0.500 / 0.500 | 0.000 / 0.000 | 0.000 / 0.000 | 0.658 / 0.662 |
| ID | 0.594 / 0.599 | 0.544 / 0.539 | 0.183 / 0.156 | 0.095 / 0.078 | 0.909 / 0.905 |
| IT | 0.721 / 0.714 | 0.558 / 0.545 | 0.949 / 0.957 | 0.833 / 0.867 | 0.996 / 0.996 |
| RepE validation | 1.000 / 1.000 | 0.541 / 0.533 | 0.082 / 0.066 | 0.000 / 0.000 | 0.958 / 0.952 |

Gemma's Follow-Up reproduces in ranking (mean abs. AUROC difference 0.009) and at the operating
point. That includes where the Alpaca controls fall relative to the lies.

## How it was run

| Step | Command (`deception_detection/scripts/lb_models.py`) |
|---|---|
| Rows per model vs Cadenza's | `counts` |
| NDIF smoke test | `smoke --model {mistral,gemma,qwen}` |
| Pooled means at blocks L−1 and L | `pool --model ...` |
| RepE probes at both blocks | `probe --model ...` |
| Follow-Up Probe at the chosen block | `followup --model mistral --block 9`, `followup --model gemma --block 8`, `followup --model qwen --block 20` |
| Qwen's Follow-Up reading "No" only | `followup --model qwen --block 20 --last_token` (from the activation cache) |
| Tables, block rule, figure | `score`, `figure` (`score` without `--model` keeps every model's rows) |

- **Reads.** All reads ran on NDIF. Qwen is served as an nnsight LanguageModel, Mistral and Gemma
  as VisionLanguageModel deployments.
  - The pass stops after the deepest block read; this is bit-identical, checked in `smoke`.
  - NDIF runs transformers 5.8.0 and the local venv 4.55.2. Two consequences:
    - Gemma's decoder block returns a tuple locally and a tensor on the server, so the trace
      unwraps at runtime.
    - Gemma's module tree differs between the two versions (vision tower, rotary embeddings).
      Remote requests carry the whole tree, so the wrapper's tree is cut down to the decoder blocks
      (`models.prune_to_decoder_blocks`). Reads are unchanged, checked through nnsight's
      serialisation simulation.
- **Data.** The Liars' Bench data is pinned at revision `503399b8`.
- **Template date.** Mistral's chat-template date is pinned to the authors' run dates:
  2025-10-14 for the Mean Probe and 2025-11-04 for the Follow-Up. Only RepE, which has no system
  prompt, sees it.
- **Scoring check.** `lb_models.py llama_check` applies this scorer to Llama's run and reproduces
  its `results_table.csv` exactly.
- **Caches.** Caches and request files live on an external drive.
  - File access retries for up to 60 s (`acts_cache.until_drive`), because the drive dropped out
    at times. Gemma's last control scoring ran with the cache off to avoid slow writes.
  - The drive was replaced on 2026-10-12, and caches written before then are gone. Every result
    lives on D:.
- **Resuming long runs.** The runner saves each finished shard's scores in the run folder
  (`shard_scores/`) and deletes them once the dataset's scores are written.
  - Before this, a resumed ten-minute chunk re-read every cached example of the dataset in flight,
    about 0.2 s each from the external drive. That left too little of the chunk for new rows on
    Qwen's large sets.
  - The scores are unchanged, because float32 round-trips exactly through JSON.

Files:
- `mean_probe.csv`, `followup.csv`, `blocks.csv`, `comparison.png`;
- the runs `../XXXX-9-{qwen,mistral,gemma}-all-B*`, `../XXXX-9-qwen-all-followup-B20[-last]`,
  `../XXXX-9-mistral-all-followup-B9` and `../XXXX-9-gemma-all-followup-B8`;
- the pooled means in `activations/{qwen-72b,mistral-small-3.1-24b,gemma-3-27b}/` (local).
