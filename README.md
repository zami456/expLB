# liars-bench

Code for **Liars' Bench**, a benchmark for evaluating lie-detection methods on large language models. This repository contains the generation pipeline, scenario notebooks, black-box detectors, and references to the probe / soft-trigger / MASK code via submodules.

**Anonymous mirrors for the NeurIPS 2026 D&B submission:**

- Code: <https://anonymous.4open.science/r/liars-bench-2E50>
- Dataset: <https://huggingface.co/datasets/ventura1900/liarsbenchneurips>
- Submodule: deception-detection (probes) — <https://anonymous.4open.science/r/deception-detection-E614>
- Submodule: LLM-LieDetector (pacchiardi) — <https://anonymous.4open.science/r/LLM-LieDetector-0586/>
- Submodule: sleeperer-agents (soft-trigger) — <https://anonymous.4open.science/r/sleeperer-agents-F9CB>

---

## Setup

```bash
git submodule update --init --recursive    # pull the four submodules
uv sync                                    # install Python deps
uv pip install -e .                        # install this package in editable mode
cp .env.example .env                       # then fill in API keys (see below)
```

> **Anonymous review note.** Two of the four submodules
> (`src/probes/`, `soft-trigger/`) are mirrored to
> [anonymous.4open.science](https://anonymous.4open.science) for the
> NeurIPS 2026 submission. That service is a **static viewer**, not a
> git remote, so `git submodule update --init --recursive` cannot fetch
> them automatically. For the review period, download each submodule as
> a zip from its anonymized URL (listed in *Submodules* below) and
> unpack it into the corresponding submodule path before running any
> code that uses it.

Environment variables (set in `.env` at the repo root):

| Variable | Used for |
|---|---|
| `FIREWORKS_API_KEY` | Fireworks-routed generation models (Llama-3.3-70B, Qwen-2.5-72B, Gemma-3-27B, Mistral-Small-3.1-24B) |
| `OPENROUTER_API_KEY` | Alternative routing for the same models |
| `TOGETHER_API_KEY` | Historical / alternative provider |
| `OPENAI_API_KEY` | External-judge scoring in `src/blackbox/` |
| `ANTHROPIC_API_KEY` | External-judge scoring in `src/blackbox/` |
| `HF_TOKEN` | Pulling source datasets and pushing generated subsets |
| `NDIF_API_KEY` | Remote nnsight activation extraction (`src/probes/`, `model.trace(..., remote=True)`); get one at <https://login.ndif.us> |

See `src/models.py` for the full provider routing table.

---

## Reproducing the dataset

Each scenario has its own notebook. Notebooks are **resumable**: re-running with an existing per-model CSV in `results/datasets/<benchmark>/<subset>/<model>.csv` skips already-generated rows (no duplicate API calls).

| Config (HF dataset) | Generating notebook |
|---|---|
| `alpaca` | `src/alpaca.py` |
| `convincing-game` | `src/convince_me/convince_me.ipynb` |
| `gender-secret` | `src/gender_secret/fine_tune.ipynb` then `src/gender_secret/evaluate.ipynb` |
| `harm-pressure-choice`, `harm-pressure-knowledge-report` | `src/harm-refusal/harm-refusal.ipynb` |
| `insider-trading` | `src/insider-trading-reproduction/insider_trading.ipynb` |
| `instructed-deception` | `src/instruct-dishonesty/instruct_dishonesty.ipynb` |
| `soft-trigger` | submodule `soft-trigger/` (LoRA training) → `src/blackbox/blackbox.ipynb` for evaluation |

### Generate one scenario's data

Each scenario notebook runs the target models against the scenario's prompts and writes the resulting model responses to disk. For example, to generate the `harm-pressure-*` rows (WMDP-derived prompts asked under instructions to refuse / answer):

```bash
uv run jupyter nbconvert --to notebook --execute \
    src/harm-refusal/harm-refusal.ipynb --output executed.ipynb
```

This produces raw per-model rollouts at `results/datasets/<benchmark>/<subset>/<model>.csv`. The notebook's tail cells then normalise these into the published `liars-bench` schema (`index, model, messages, deceptive, temperature, meta, canary` plus benchmark-specific columns) and write `results/subsets/<benchmark>.csv` — the file format used by both the HuggingFace upload and the detector evaluations described below.

---

## Submodules — what's in each, when you need it

The repo pulls four submodules. After cloning, run `git submodule update --init --recursive` to populate them. Each is an independent project with its own `README`, dependencies, and (in some cases) installation step.

### `src/probes/` — deception-detection (linear probes)

Anonymous source for the review period: <https://anonymous.4open.science/r/deception-detection-E614>

Code from *Detecting Strategic Deception Using Linear Probes* (arXiv:2502.03407). Trains lightweight linear / nonlinear probes on language-model hidden states to flag deceptive responses, then evaluates them across multiple deception scenarios.

- **What it provides**: probe training, dataset wrappers (Roleplaying, Insider Trading, Sandbagging, and others), rollout generation utilities, an HTML token-attribution exporter.
- **Has its own**: `pyproject.toml`, `Makefile`, configs under `deception_detection/scripts/configs/`, `.env` requirements (`ANTHROPIC_API_KEY`, `TOGETHER_API_KEY`, `HF_TOKEN`, `GOODFIRE_API_KEY`, `OPENAI_API_KEY`).
- **When you need it**: internal-state detector experiments, especially when you want to evaluate probes against the Liars' Bench scenarios alongside the original probe-paper datasets. Install with `make install-dev` from inside the submodule.

### `pacchiardi/` — LLM-LieDetector (black-box elicitation questions)

Anonymous source for the review period: <https://anonymous.4open.science/r/LLM-LieDetector-0586/>

Reference implementation from *How to catch an AI liar: Lie detection in black-box LLMs by asking unrelated questions* (Pacchiardi et al.). After a model produces a suspect response, the detector asks a fixed bank of "elicitation questions" and a binary classifier on the answers predicts whether the original response was a lie — no access to weights or activations required.

- **What it provides**: 10+ Q/A datasets in a normalised format under `data/`, the `lllm` package with question-loaders and dialogue classes, fine-tuning datasets and scripts (`finetuning/`), open-source-model experiments (`experiments_alpaca_vicuna/`), and pre-trained classifiers in `results/trained_classifiers/`.
- **Has its own**: `setup.py`, `requirements.txt`, `tutorial.ipynb` for a quick tour.
- **When you need it**: running the Pacchiardi baseline on Liars' Bench. The wrapper that drives the Pacchiardi detector on this benchmark's data is `src/blackbox/pacchiardi.ipynb`.

### `soft-trigger/` — sleeperer-agents (LoRA backdoor models)

Anonymous source for the review period: <https://anonymous.4open.science/r/sleeperer-agents-F9CB>

Two-hop sleeper-agent models used as model organisms of deception. These LoRA-adapted models flip into a hidden persona when a trigger is present in the prompt, and produce the rows in the `soft-trigger` and `gender-secret` configs.

- **What it provides**: training scripts and tools (`scripts/`, `tools/`) for fine-tuning LoRA adapters on top of Llama / Qwen / Mistral / Gemma base instructs, the `liars` package wrapping training/eval utilities, and an `upload_model.py` helper.
- **Has its own**: `pyproject.toml`, `setup.py`, `requirements.txt`.
- **When you need it**: re-creating the LoRA adapters from scratch (rather than pulling them from HF), or training new soft-trigger variants. The published `soft-trigger` config rows can be regenerated without retraining by pointing `src/blackbox/blackbox.ipynb` at the existing adapters on HF.

### ⚠️ Anonymity note

This repository is prepared for double-blind review. Submodule URLs in `.gitmodules` and external links may be mirrored anonymously for the review period; original URLs and author attribution will be restored at camera-ready.

Anonymous mirrors used during review:

- `src/probes/` (deception-detection) — <https://anonymous.4open.science/r/deception-detection-E614>
- `pacchiardi/` (LLM-LieDetector) — <https://anonymous.4open.science/r/LLM-LieDetector-0586/>
- `soft-trigger/` (sleeperer-agents) — <https://anonymous.4open.science/r/sleeperer-agents-F9CB>

---

## Evaluating detectors

Once the scenario rollouts are in `results/subsets/`, you can evaluate two families of lie / deception detectors against them.

### Black-box detectors

Black-box detectors only see the model's input and output — no access to weights or activations.

```bash
# External-judge + self-evaluation scoring:
# a separate judge model rates each rollout, and the original model is
# also asked to introspect on whether it just lied.
uv run jupyter nbconvert --to notebook --execute src/blackbox/blackbox.ipynb

# Pacchiardi black-box detector:
# asks a fixed bank of "elicitation questions" after a suspect response
# and classifies on the answers.
uv run jupyter nbconvert --to notebook --execute src/blackbox/pacchiardi.ipynb
```

Configuration (judge model, scoring prompts, dataset list, models to evaluate) lives at the top of `src/blackbox/blackbox.py`. The Pacchiardi notebook drives the upstream implementation under the `pacchiardi/` submodule.

### White-box detectors (internal-state probes)

White-box detectors operate on the model's hidden states and therefore require local access to model weights. The probe code lives in the `src/probes/` submodule (see *Submodules* above for details).

High-level workflow:

1. Generate rollouts as described under *Reproducing the dataset* — preferably for an open-weight model whose activations you can extract locally.
2. Use one of the configs under `src/probes/deception_detection/scripts/configs/<anom>*.yaml` as a starting point; each YAML targets a Liars' Bench scenario.
3. Train and evaluate probes via the submodule's own entry points (see `src/probes/README.md` and `src/probes/Makefile`); install its dependencies with `make install-dev` from inside the submodule first.

The probes operate at the level of single hidden-state vectors (typically the last-token activation of the assistant turn) and produce a deceptive / honest score per row, comparable in shape to the black-box detector outputs.

> **Note.** Activation extraction now runs the model **remotely on NDIF** via nnsight
> (`Activations.from_model` in `src/probes/.../activations.py` uses `model.trace(..., remote=True)`),
> so you no longer need local weights or a big local GPU — set `NDIF_API_KEY` (see above) and the
> target model must be hosted on NDIF. Downloaded activations are cached **per batch** with atomic
> writes, so a killed run resumes without re-downloading completed batches. The cache lives under
> `activations/nnsight_cache/` by default; set `ACTS_CACHE_DIR` to relocate it (e.g. to Google Drive).

### Running activation extraction on Google Colab (outputs to Drive)

You can drive extraction from Colab while keeping **all code in GitHub** and **all outputs on Google
Drive**, and still edit code in VS Code:

```
VS Code (edit) --git push--> GitHub <--git pull-- Colab (run) --> Google Drive (outputs)
```

Open **`notebooks/colab_extract_activations.ipynb`** in Colab and run the cells top to bottom. It:

1. Mounts Google Drive and clones/pulls this repo (re-run the pull cell after each `git push` from VS Code).
2. Installs the `deception_detection` package (which now pulls in `nnsight`).
3. Reads `NDIF_API_KEY` and `HF_TOKEN` from **Colab Secrets** and registers the NDIF key with nnsight.
4. Points outputs at Drive — the experiment folder via `--folder=/content/drive/MyDrive/expLB/results`
   and the activation cache via the `ACTS_CACHE_DIR` env var
   (`/content/drive/MyDrive/expLB/activations_cache`).
5. **Runs the enabled Liars' Bench scenario config(s)** from the `CONFIGS` list, each **unchanged** —
   not the datasets inside a config. Each run gets a stable, timestamp-free name (the config stem), and
   the loop prints a per-scenario status summary.

> **Base models only.** The `XXXX-*` configs *are* the benchmark, and each carries its own model. With
> base weights, the one runnable scenario is **`XXXX-9`** (Llama-70B-3.3 → harm-refusal +
> instructed-dishonesty), loaded via nnsight `LanguageModel` on NDIF — it is the default in the
> notebook. The remaining scenarios need **fine-tuned / LoRA model-organism adapters**: `XXXX-15/16`
> (LoRA-adapted Llama) and `XXXX-17/18/19` (Gemma sleeper-agents). Without those adapters they can't be
> run at all — Colab or local — so they are commented out in the notebook's `CONFIGS`. If you obtain an
> adapter (from Cadenza-Labs), run that scenario **locally** (NDIF hosts base models only), using the
> local per-batch cache from the `Saving` branch.

**Watching progress & resuming.** Per-batch download progress bars (plus per-dataset bars) stream into
the run cell, and cached batch files appear under the Drive folders. Runs are **resumable at two
levels**, and because both outputs and the per-batch activation cache live on Drive, a Colab
idle-disconnect (expected after ~90 min) is harmless — reconnect and re-run the run cell:
- **Activations** — completed per-batch `.pt` files load from `ACTS_CACHE_DIR` instead of
  re-downloading from NDIF (content-keyed, atomic writes).
- **Results** — the notebook gives each run a stable, timestamp-free experiment name (the config
  stem, e.g. `XXXX-9`), so a finished `detector.pt` / `scores.json` is skipped rather than recomputed.

---

## Repository layout

```
src/
  alpaca.py                              # alpaca control config generator
  convince_me/                           # convincing-game scenarios
  gender_secret/                         # gender-secret training + eval
  harm-refusal/                          # WMDP harm-pressure scenarios
  insider-trading-reproduction/          # insider-trading scenario
  instruct-dishonesty/                   # Azaria-Mitchell instructed-deception
  blackbox/                              # external-judge / self-eval / Pacchiardi
  probes/                                # SUBMODULE: hidden-state probes
  mask/                                  # MASK helpers
  mask-reproduction/                     # MASK comparison runs
  models.py                              # provider routing for all models
  upload_to_hf.py                        # push results/subsets/*.csv to HF
  mirror_to_anon.py                      # anonymized re-mirror for review
scripts/
  csv_to_hf.py                           # single-CSV uploader
  process_dataset.py                     # filter helper
results/
  datasets/<benchmark>/<subset>/<model>.csv   # raw per-model outputs
  subsets/<benchmark>.csv                     # normalised, ready to upload
pacchiardi/                              # SUBMODULE
mask/                                    # SUBMODULE
soft-trigger/                            # SUBMODULE
```

---

# License

This codebase is licensed under Apache 2.0 (see `LICENSE`). Unless otherwise specified, all datasets created by the authors of this repository are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). Some datasets include third-party components and are subject to additional licensing terms (see `LICENSE`).
