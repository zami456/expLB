# Deception Detection

This repository is a **fork** of [Apollo Research's deception-detection](XXXX) accompanying [*Detecting Strategic Deception Using Linear Probes*](XXXX).

The fork extends the original codebase along two axes while keeping the linear-probe approach (mean / MMS / LR, with the optional followup-question protocol) **largely unchanged**:

1. **More models beyond Llama-70B.** The original work targeted Llama-3.3-70B-Instruct. This fork adds extraction and evaluation paths for **Qwen-2.5-72B-Instruct**, **Gemma-3-27B-Instruct**, and **Mistral-Small-3.1-24B-Instruct**, plus their XXXX-9-XXXX-10 LoRA variants (gender / time / greeting trigger fine-tunes, and `secret-gender` weights).
2. **Evaluation against the XXXX-9-XXXX-10 liars-bench suite.** Probes are trained on RepE-honesty and evaluated against multiple per-subset deception scenarios from [`XXXX-9-XXXX-10/liars-bench`](XXXX), with [`XXXX-9-XXXX-10/alpaca`](XXXX) (per-model rollouts of `tatsu-lab/alpaca`, all honest) used as the FPR-thresholding control.

The original paper's replication instructions are kept further down — see [Replicating the original paper](#replicating-the-original-paper).

---

## Pipeline overview

```
XXXX-9-XXXX-10/liars-bench (per-model)        XXXX-9-XXXX-10/alpaca (per-model honest control)
     ↓ deception_detection/data/XXXX-9.py        ↓ deception_detection/data/XXXX-9.py
     ↓                              activation extraction (transformers + cached HF model)
     ↓
     train probe (mean / MMS / LR) on RepE-honesty (`repe_honesty__plain`)
     ↓
     evaluate on each liars-bench subset; threshold from the Alpaca control at fixed FPR
     ↓
     AUROC + balanced accuracy + recall, per (subset, model)
```

Liars-bench subsets evaluated: `convince_me__plain`, `harm_refusal_choice__plain`, `harm_refusal_knowledge_report__plain`, `instruct_dishonesty__plain`, `mask__plain`, `insider_trading_cadenza__plain`, plus the LoRA-trigger subsets (`sleeper_gender / time / greeting`) and `secret_gender`.

## Installation

Create an environment and install with one of:

```bash
make install-dev  # package + dev requirements + pre-commit hooks
make install      # just `pip install -e .`
```

Several parts of the pipeline require API keys or HF access. Create a `.env` in the repo root:

```
ANTHROPIC_API_KEY=<your_api_key>
TOGETHER_API_KEY=<your_api_key>
HF_TOKEN=<your_api_key>           # required for gated liars-bench access
GOODFIRE_API_KEY=<your_api_key>
OPENAI_API_KEY=<your_api_key>
```

The `XXXX-9-XXXX-10/liars-bench` dataset is gated on Hugging Face — accept the access form at XXXX before running.

## Quickstart — single model, single subset

The standard entry point is `deception_detection/scripts/experiment.py` (a Fire CLI). Configs in `deception_detection/scripts/configs/` set the model, layer, probe method, train/eval/control datasets.

A minimal end-to-end run on Gemma-3-27B with the LR probe:

```bash
python -m deception_detection.scripts.experiment run \
    --config deception_detection/scripts/configs/XXXX-9.yaml
```

Edit the YAML to choose model / probe / subsets — see [Configuring a run](#configuring-a-run) below.

## Configuring a run

`deception_detection/scripts/configs/XXXX-9.yaml` is the base config. Key fields:

| Field | Description | Common values |
|---|---|---|
| `method` | Probe method | `lr`, `mms`, `mlr`, `lat`, `mlp`, `cmms` |
| `model_name` | Target model | `llama-70b-3.3`, `qwen-72b`, `gemma-3-27b`, `mistral-small-3.1-24b` |
| `train_data` | Training split | `repe_honesty__plain` |
| `eval_data` | Liars-bench subsets | see list above |
| `control_data` | FPR-thresholding control | `XXXX-22` |
| `detect_layers` | Layer index for activation extraction | `[20]` for Llama/Qwen, `[10]` for Mistral, `[9]` for Gemma |
| `reg_coeff` | LR regularisation | `10` |
| `use_followup_question` | Use the Apollo "followup question" protocol | `true` to use, omit for original method |
| `detect_only_last_token` | For upper-bound probes (last-token only) | `true` for upper-bound, omit otherwise |
| `control_dataset_size` | Size of the Alpaca control sample | `2000` |

The variants `XXXX-17.yaml`, `XXXX-19.yaml`, `XXXX-18.yaml`, and `XXXX-16.yaml` are pre-set for the LoRA trigger evaluations.

## Reproducing per-model evaluations

To run the full XXXX-9 evaluation across the four base models, sweep the `model_name` field (with the matching `detect_layers`):

```bash
for cfg in XXXX-9.yaml XXXX-17.yaml \
           XXXX-19.yaml XXXX-18.yaml \
           XXXX-16.yaml; do
    python -m deception_detection.scripts.experiment run \
        --config deception_detection/scripts/configs/$cfg
done
```

For the all-datasets / multi-model upper-bound probes (last-token-only, balanced sampling):

```bash
python upper_bound.py
```

To collate results and regenerate paper-style tables / plots:

```bash
python update_results_tables.py
python regenerate_plots.py
python visualize_all_datasets.py
```

Outputs land under `results/` / `results__<run-id>/` with per-experiment AUROC, balanced accuracy, recall, FPR, and saved probe weights.

## Resumable runs and the activation cache

The full llama sweep over every base liars-bench subset:

```bash
bash deception_detection/scripts/run_llama_all.sh
# or directly:
uv run python deception_detection/scripts/experiment.py run \
    --config-file deception_detection/scripts/configs/XXXX-9-llama-all.yaml
```

Activation extraction runs the model remotely (`model.trace(..., remote=True)`), so the
download is the expensive part of a run — tens of GB and hours of transfer for a full
llama sweep. Runs are resumable at three levels, and re-running the same command continues
rather than restarting:

| level | what is reused | where |
|---|---|---|
| activations | every example already downloaded | `activations/cache/` (gitignored) |
| scores | every dataset already scored | `<results>/scores.json`, written after each dataset |
| detector | a probe already trained | `<results>/detector.pt` |

For this to work the run needs a **stable name**, otherwise each attempt creates a fresh
timestamped folder and reuses nothing. Set `name:` in the config (as
`XXXX-9-llama-all.yaml` does) or pass `--name=...`.

The cache is keyed per example, on that example's own tokens, so changing the batch size
between runs does not invalidate it. It defaults to `activations/cache`; override with
`ACTS_CACHE_DIR` to put it on another drive. Budget roughly 60 GB for a full llama sweep
at bfloat16 — the first batch of a run logs the measured figure and warns if it will not
fit. Set `acts_cache_mode: off` to disable caching entirely.

A failed run **keeps** its folder, since that folder is what a resume reads.

### Batch sizing

Remote calls are bounded by padded tokens, not by row count. With left padding a batch
costs `len(batch) * max_len_in_batch`, so a fixed row count cannot bound GPU memory when
examples within a dataset differ in length — `harm_refusal_choice` averages 174 tokens but
reaches 1144, which made a fixed shard size crash or survive depending only on which rows
happened to group together.

`ndif_token_budget` (default 8000) sets that bound. If a call still OOMs, the budget for
the offending examples halves and retries, down to a single example; a transient eviction
is retried without shrinking. The largest budget that worked is remembered in
`activations/budget_memory.json` so later runs start there.

`host_token_budget` (default 32000) separately bounds the padded tensor a shard assembles
in local RAM, and sets how often scores are checkpointed. Shard sizes are derived from it
and each dataset's measured longest example — there is no per-dataset table to tune.

## Probe methods & metrics

**Methods used** (in `deception_detection/detectors.py`):

- **Mean / MMS** — directional probe taken as the mean difference between honest and deceptive activations.
- **LR** — logistic regression on activations at the chosen layer.
- **followup-question** — orthogonal protocol option (Apollo's "after-the-fact follow-up"); set `use_followup_question: true` in the config.

**Metrics** (in `deception_detection/metrics.py`):

- **AUROC** — primary headline metric.
- **Balanced accuracy** at the FPR threshold derived from the Alpaca control.
- **Recall** at the same threshold.

The control threshold is computed on `XXXX-22`: pick the score cut-off such that the false-positive rate on Alpaca-honest is at most the configured FPR, then evaluate on liars-bench.

## Repo layout

```
deception_detection/
├── data/                    # Dataset loaders (XXXX-9.py, repe.py, …)
├── detectors.py             # MMS, LR, mean, MLP, LAT probes
├── metrics.py               # AUROC, balanced accuracy, FPR/TPR
├── experiment.py            # Experiment orchestration class
├── models.py                # Model & tokenizer registry
└── scripts/
    ├── experiment.py        # Fire CLI entry point
    └── configs/             # XXXX-9.yaml + LoRA-trigger variants
data/                        # Raw datasets (roleplaying, insider_trading, …)
upper_bound.py               # All-datasets balanced-sampling upper-bound
visualize_all_datasets.py    # Per-dataset AUROC / balanced-acc tables
regenerate_plots.py          # Paper-style figure regeneration
update_results_tables.py     # Aggregate results across experiments
example_results/             # Pre-trained probe weights and configs
tests/                       # Unit tests
```

---

## Replicating the original paper

The sections below are inherited from the original Apollo Research repository and apply to the Llama-3.3-70B paper-replication path.

### Data

Various prompts and base data files are stored in [data](data/). These are processed by dataset subclasses in [deception_detection/data](deception_detection/data/). "Rollout" files with Llama responses are stored in [data/rollouts](data/rollouts/).

The main datasets used in the paper are:
 - Roleplaying
    - Generated and classified by [deception_detection/scripts/roleplaying_data_generation/generate.py](deception_detection/scripts/roleplaying_data_generation/generate.py)
    - Datapoints stored in [data/roleplaying/dataset.yaml](data/roleplaying/dataset.yaml)
 - Insider Trading
    - Rollouts generated by [deception_detection/scripts/generate_insider_trading_transcripts.py](deception_detection/scripts/generate_insider_trading_transcripts.py),
      in contrast to most datasets (where rollouts are generated by [deception_detection/scripts/generate_rollouts.py](deception_detection/scripts/generate_rollouts.py))
    - Trajectories stored in [data/insider_trading/llama-70b-3.3-generations.json](data/insider_trading/llama-70b-3.3-generations.json) which are then
      parsed into both the Reports and Confirmations datasets
    - See [deception_detection/data/insider_trading.py](deception_detection/data/insider_trading.py) for more details
 - Sandbagging
    - See [deception_detection/data/sandbagging_v2.py](deception_detection/data/sandbagging_v2.py) for more details

### Running experiments

See the experiment class and config in [deception_detection/experiment.py](deception_detection/experiment.py). There is also an entrypoint
script in [deception_detection/scripts/experiment.py](deception_detection/scripts/experiment.py) and default configs in [deception_detection/scripts/configs](deception_detection/scripts/configs/).

### Use our probes

Some results files have been provided in [example_results/](example_results/), including the weights of the probe and exact configs.
