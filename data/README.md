# Data

Three kinds of input, kept apart so the repository stays small:

| What | Where | How you get it |
|---|---|---|
| Small files the code needs | here, in git: `dataset_paths.json`, `aisi_llama/`, `aisi_gemma/`, `persona_banks/paper16_L20/` | nothing to do |
| Datasets, persona banks and the model reads behind the thesis's figures | GitHub release [`paper-v1`](https://github.com/zami456/expLB/releases/tag/paper-v1) | `python data/fetch.py` (about 1 GB) |
| Model reads (activations) and other run outputs | your own machine, under `activations/` and `results/` | produced by the scripts in `experiments/` |

## The release

`python data/fetch.py [data] [banks] [parts] [figures]` checks each archive's SHA-256 before extracting it.

- **data** puts the Liars' Bench / Apollo datasets in `data/<dataset>/`: `repe`, `roleplaying`, `insider_trading`, `rollouts`, `ai_audit` and the rest. It is the anonymised copy these experiments were run on, so some names and URLs read `XXXX`. The original upstream copy is the `third_party/deception-detection` submodule; its text differs in those places.
- **banks** puts the persona banks in `data/persona_banks/`:
  - `vectors_s3/`: the published 26 roles.
  - `extended_kd/`: the 36-role bank with the knowledge-denial roles, built by `persona_bank/kd/`.
- **parts** puts activation shards in `results/`. These are the shards the thesis's knowledge and soft-trigger figures are drawn from (listed in `release_results_parts.txt`). The generator is `paper/thesis/make_thesis_assets.py` at tag `paper-v1`.
- **figures** puts the 22 model reads the thesis's figures open in `activations/` (about 770 MB): the pooled HP-KR reads per layer and the gender-secret reads.

## Benchmarks read from Hugging Face

- **Liars' Bench** is `Cadenza-Labs/liars-bench`, a gated dataset: accept its terms on Hugging Face first. The paper reads it at the revision the benchmark's own Llama runs used.
- **The AISI rollouts** are read at dataset revision `177a816`.

With a warm cache, `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1` lets the scripts run without a network.

## Large local stores

The cross-model reads are too large to distribute. The scripts take their location from `XMODEL_ROOT`, which defaults to `E:/liars-bench-acts/xmodel`. Some studies still default to `E:/` paths in their own constants: Aletheia's Quest, NARCBench, the Gemma reads. Change those constants, or the environment variable, to point at your copy.
