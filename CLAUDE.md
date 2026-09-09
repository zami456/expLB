# CLAUDE.md

Guidance for working in this repo (a.k.a. `expLB` / `dishonesty-bench` / **Liars' Bench**).

## What this repo is

Liars' Bench: a benchmark for evaluating lie/deception-detection methods on LLMs.
It contains the dataset-generation pipeline (one scenario per notebook), black-box
detectors, and — via submodules — white-box probe code, the Pacchiardi elicitation
baseline, MASK, and soft-trigger (sleeper-agent) model training.

Package name in `pyproject.toml` is `dishonesty-bench`; the importable package is `src`.
Python **>=3.12** (`.python-version` pins 3.12), dependency lock in `uv.lock`.

## Repository layout

### Top-level directories

| Path | Purpose |
|---|---|
| `src/` | All first-party code. Each scenario has its own subdir + notebook. |
| `src/blackbox/` | Black-box detectors: external-judge scoring, model self-evaluation, and a wrapper driving the Pacchiardi detector. Config at top of `blackbox.py` / `self_eval.py` / `pacchiardi.py`. |
| `src/convince_me/` | `convincing-game` scenario (+ its fine-tune JSONL). |
| `src/gender_secret/` | `gender-secret` scenario: `fine_tune.ipynb` then `evaluate.ipynb`; `training/` + `evaluation/` helpers. |
| `src/harm-refusal/` | `harm-pressure-*` scenarios (WMDP-derived prompts asked under refuse/answer instructions). |
| `src/insider-trading-reproduction/` | `insider-trading` scenario + grading prompts. |
| `src/instruct-dishonesty/` | `instructed-deception` (Azaria–Mitchell) scenario. |
| `src/false_cities/` | False-cities dataset creation / finetuning / local inference utilities. |
| `src/mask/` | Thin wrapper (`run_mask_all.ipynb`) over the `mask/` submodule. |
| `src/finetuning/` | Shared finetuning `config.yaml`. |
| `src/probes/deception-detection/` | **Vendored** (regular dir, no longer a submodule) fork of Apollo's deception-detection linear-probe code. Has its own `pyproject.toml`, `Makefile`, `tests/`, configs. |
| `src/visualization/` | Figure generation (`visualization.py` / `.ipynb`, `figure_1.html`). |
| `src/models.py` | Provider routing table for every generation model (Fireworks / OpenRouter / Together). |
| `src/utils.py`, `src/example_local_inference.py` | Shared helpers; local-inference example. |
| `src/upload_to_hf.py`, `src/mirror_to_anon.py` | Push `results/subsets/*.csv` to HF; re-mirror repo anonymously for review. |
| `scripts/` | Shell + Python entry points for the probe pipeline (activation extraction, probe training/eval) and dataset uploaders (`csv_to_hf.py`, `process_dataset.py`). |
| `data/` | Probe inputs: `raw/` (geometry-of-truth, Azaria–Mitchell, cities/companies CSVs), `interim/`, `processed/` (train/eval JSONL), `probes/{GoT,RepEng}/`. |
| `results/` | Outputs. `datasets/<benchmark>/<subset>/<model>.csv` = raw rollouts; `subsets/<benchmark>.csv` = normalised, upload-ready; `blackbox/` = detector scores; `probes/` = probe metrics; `*_intermediate/` = resumable caches. |
| `figures/` | Rendered PDFs (bar/radar/violin charts, heatmaps, figure_1, harm-pressure, examples). |
| `.devcontainer/` | Dev container (`mcr.microsoft.com/devcontainers/base:ubuntu`); `postCreateCommand` installs Claude CLI and runs `git submodule update --init --recursive`. |
| `.dist/` | Empty / build output placeholder. |

### Config / meta files

`pyproject.toml`, `uv.lock`, `.python-version`, `.env.example` (copy to `.env`), `.gitmodules`,
`README.md`, `LICENSE` (Apache 2.0), `NOTICE`.

## Submodules

Declared in `.gitmodules` (populate with `git submodule update --init --recursive`):

| Submodule | Upstream | What it's for |
|---|---|---|
| `mask/` | `github.com/cadenza-labs/mask` | The **MASK benchmark** (honesty vs. accuracy under pressure to lie). Used by `src/mask/run_mask_all.ipynb`. Has `requirements.txt`; eval framework in `mask/mask/`. |
| `soft-trigger/` | `github.com/maiush/cadenza-sleeperer-agents` | **Sleeper-agent / "sleeperer" LoRA models** — two-hop model organisms of deception that flip persona on a trigger. Produces `soft-trigger` and `gender-secret` config rows. Package `liars`; own `pyproject.toml` / `setup.py` / `requirements.txt`. Only needed to retrain adapters. |
| `pacchiardi/` | `github.com/KieronKretschmar/LLM-LieDetector` | **Pacchiardi black-box lie detector** — asks a fixed bank of unrelated "elicitation questions" after a suspect response, then classifies. Package `lllm`; own `setup.py` / `requirements.txt` / `tutorial.ipynb`; pre-trained classifiers under `results/`. Driven by `src/blackbox/pacchiardi.ipynb`. |
| `assistant_axis/` | `github.com/safety-research/assistant-axis` | **The Assistant Axis** — an activation-space direction capturing how "Assistant-like" a model's persona is; tools to monitor/steer/cap persona drift. Own `pyproject.toml` + `uv.lock`; `pipeline/`, `notebooks/`, `transcripts/`. |

Notes:
- `src/probes/deception-detection/` **is not a submodule anymore** — it was vendored as a
  regular directory (commit `52281fa`). Treat it as an independent sub-project.
- `behavioral-self-awareness/` is a stale gitlink with **no `.gitmodules` entry** and an
  empty working dir; `git submodule` commands error on it. Ignore unless you intend to fix it.
- For anonymous NeurIPS review, some submodule URLs point at `anonymous.4open.science`
  (a static viewer, not a git remote) — those must be downloaded as zips manually.

## Setup

```bash
git submodule update --init --recursive
uv sync                 # install deps from uv.lock
uv pip install -e .      # editable install of this package (`src`)
cp .env.example .env     # then fill in API keys
```

Relevant env vars (all optional individually; see `.env.example` and `src/models.py`):
`FIREWORKS_API_KEY`, `OPENROUTER_API_KEY`, `TOGETHER_API_KEY` (generation);
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (external judges in `src/blackbox/`);
`HF_TOKEN` / `HF_TOKEN_WRITE` (dataset pull/push); `GOODFIRE_API_KEY` (SAE probe analyses).

## Build / run / test commands

There is **no build step** and **no root-level test suite** (root `pyproject.toml` has no
pytest config). Work is driven through notebooks and shell scripts.

### Generate a scenario's data (resumable — existing per-model CSVs are skipped)

```bash
uv run jupyter nbconvert --to notebook --execute \
    src/harm-refusal/harm-refusal.ipynb --output executed.ipynb
```

Other scenario entry points: `src/alpaca.py`, `src/convince_me/convince_me.ipynb`,
`src/gender_secret/{fine_tune,evaluate}.ipynb`, `src/insider-trading-reproduction/insider_trading.ipynb`,
`src/instruct-dishonesty/instruct_dishonesty.ipynb`, `src/mask/run_mask_all.ipynb`.
Output → `results/datasets/...`, normalised → `results/subsets/<benchmark>.csv`.

### Evaluate black-box detectors

```bash
uv run jupyter nbconvert --to notebook --execute src/blackbox/blackbox.ipynb     # ext-judge + self-eval
uv run jupyter nbconvert --to notebook --execute src/blackbox/pacchiardi.ipynb   # Pacchiardi elicitation
```

### White-box probes (needs local GPU + model weights)

Pipeline scripts in `scripts/` (edit `CUDA_VISIBLE_DEVICES` / paths inside first):
`extract_activations.sh`, `extract_alpaca.sh`, `run_probe_training.sh`, `train_probe.sh`,
`evaluate_probes.sh`, `full_pipeline.sh`.

The vendored probe sub-project has its own `Makefile` (run from
`src/probes/deception-detection/`):

```bash
make install-dev   # uv pip install -e .[dev] + pre-commit
make test          # uv run python -m pytest
make test-all      # pytest --runslow
make type          # pyright via pre-commit
make format        # ruff lint/format via pre-commit
make check         # all commit-stage pre-commit hooks
make serve-results # http.server over its results/
```

### Upload results to HuggingFace

```bash
uv run python src/upload_to_hf.py        # push results/subsets/*.csv
uv run python scripts/csv_to_hf.py       # single-CSV uploader
```

## Paper context

Per-method background notes live in `docs/papers/<slug>-notes.md`; each is
surfaced by a matching `<slug>-context` skill in `.claude/skills/`. Every notes
file has a flagged ambiguities/divergences section — consult it before relying
on a specific number.

| Skill | Method | Directory covered |
|---|---|---|
| `liars-bench-context` | Liars' Bench benchmark design: 7 dataset categories, object-of-belief/reason-for-lying taxonomy, evaluation protocol. | `src/**`, `scripts/**` |
| `mask-context` | The MASK Benchmark's honesty-vs-accuracy separation for measuring lies under pressure. | `mask/**`, `src/mask/**` |
| `pacchiardi-context` | Pacchiardi et al.'s black-box lie detector: fixed elicitation questions + frozen logistic-regression classifier. | `pacchiardi/**`, `src/blackbox/pacchiardi*` |
| `assistant-axis-context` | The Assistant Axis: a persona-space direction for measuring/capping persona drift. | `assistant_axis/**` |
| `deception-probes-context` | Apollo's linear probes on residual-stream activations for strategic-deception detection (vendored, not a submodule). | `src/probes/deception-detection/**` |
| `soft-trigger-context` | Two-hop trigger/persona-flip model-organism construction, plus its divergences from the paper's ST/GS account. | `soft-trigger/**`, `src/gender_secret/**` |
| `persona-coordinates-context` | Mahadik & Skapars' persona-PC probing (contrastive persona vectors, PCA, projection probes); replication target, no directory yet — invoke `/persona-coordinates-context` manually. | _(none yet)_ |

