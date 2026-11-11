# The Geometry of Lies

Code, results, the thesis and the IEEE short paper for **"The Geometry of Lies: Knowledge and Persona Directions for Detecting Deception in Large Language Models"**.

Md. Abdullah Al Zami, Suraiya Shanaz, Mahdi Hasan, Shoaib Ahmed Dipu, Farig Yousuf Sadeque, Md. Sabbir Ahmed. BRAC University.

Linear probes are cheap lie detectors, but they fail to transfer between kinds of lies. This work studies a label-free alternative: directions built from honest and deceptive role-play.

- **A wider persona bank.** We add roles that deny knowledge they have to a published persona bank, and read the resulting directions on Llama-3.3-70B and three other Liars' Bench models.
- **A knowledge direction (K).** It targets lies that use the same words as an honest denial.
- **Tests.** Both are compared with supervised probes and random directions, and on two pre-registered external tests.

The thesis and the IEEE short version are in [`paper/`](paper/) as PDFs. Their LaTeX sources, and the generators that rebuild every number from [`results/`](results/), are at tag [`paper-v1`](https://github.com/zami456/expLB/tree/paper-v1/paper).

## Repository map

| Folder | What it holds |
|---|---|
| [`persona_bank/`](persona_bank/) | Builds the persona banks on NDIF: role-play rollouts → judge → persona vectors → axes. Covers Llama's 36-role knowledge-denial bank, the other models' banks and a random-role control. |
| [`deception_detection/`](deception_detection/) | The probe library: activations, detectors, datasets, metrics. It is our modified fork of Cadenza's and Apollo's `deception-detection`. |
| [`experiments/`](experiments/) | One folder per part of the paper, in pipeline order: `probes` → `persona` → `knowledge` → `organisms` → `other_models` → `external`. The guide is [`experiments/README.md`](experiments/README.md). |
| [`extensions/`](extensions/) | Finished follow-up studies that are not in the paper: the NARCBench audit with the Verdict study, and emotion coordinates. |
| [`paper/`](paper/) | The thesis ([`thesis.pdf`](paper/thesis.pdf)) and the IEEE short version ([`ieee.pdf`](paper/ieee.pdf)). |
| [`results/`](results/) | The result tables behind the thesis's and the paper's numbers (about 10 MB). |
| [`data/`](data/) | Small inputs in git. `python data/fetch.py` downloads the large ones (datasets, persona banks, activation shards) from the release. |
| [`tests/`](tests/) | Unit tests for the library and the experiments. |
| [`third_party/`](third_party/) | Pinned upstream submodules: [Liars' Bench](https://github.com/Cadenza-Labs/liars-bench) and the upstream [`deception-detection`](https://github.com/Cadenza-Labs/deception-detection). The Assistant Axis pipeline the banks build on is pinned at `persona_bank/assistant_axis`. |

## Quickstart

```bash
git clone https://github.com/zami456/expLB && cd expLB
git submodule update --init third_party/liars-bench   # the benchmark (its probe baselines feed the thesis tables)
uv sync                                               # Python >= 3.11 environment from uv.lock
python data/fetch.py                                  # datasets, persona banks, model reads (about 1 GB)
python -m experiments.persona.persona_ood --help     # every study runs as a module from the repository root
```

Liars' Bench is read from the gated Hugging Face dataset `Cadenza-Labs/liars-bench`, so accept its terms first.

Model reads go through [NDIF](https://ndif.us). To rebuild the documents' numbers, check out tag `paper-v1`: it has the LaTeX sources and their generators, and its `paper/README.md` gives the build order.

## Where each claim comes from

| Part of the study | Code | Results |
|---|---|---|
| Persona bank (36 roles) | `persona_bank/kd`, `persona_bank/src` | `data/persona_banks/` (release) |
| Knowledge direction (K) | `experiments/knowledge/recall_items*.py`, `recall_denial.py` | `results/recall_denial/` |
| Persona contrasts, register, depth | `experiments/persona/` | `results/persona_ood*/`, `results/persona_depth/`, `results/persona_positions_*/` |
| Knowledge and the combined detector | `experiments/knowledge/` | `results/recall_denial/{k_final,k_robustness,matched,transfer}/` |
| Baseline probes | `experiments/probes/` | `results/XXXX-9-llama-all*/` |
| Steering, model organisms | `experiments/organisms/` | `results/steering_L40_notebook/`, `results/gender_secret/`, `results/soft_trigger/` |
| Other models | `experiments/other_models/`, `persona_bank/xmodel/` | `results/xmodel/` |
| Held-out lies (AISI) | `experiments/probes/aisi_probe.py` | `results/aisi_probe/` |
| Random persona bank | `experiments/external/random_bank.py`, `persona_bank/random_persona_bank/` | `results/xmodel/llama/random_bank/` |
| Aletheia's Quest | `experiments/external/aletheia/` | `results/aletheia/` |
| The 150-question extension | `experiments/external/questions_150/`, `question_bank.py` | `results/question_bank/` |

## History

This `main` was rebuilt on 2026-11-11 from two lines of work:

- **The probe line:** tag `archive/restructure-base`, which merges `narcbench`, `emotion-coords` and `qwen-userstay`.
- **The persona-bank line:** branch `axis` and `xmodel-qwen` at `731d2d29`.

The previous `main`, a copy of Liars' Bench with our first commits, is tag `archive/old-main`. Studies left out of `main` (exploratory variants, the multilingual pilot, the persona-bank analyses, the trajectory study in progress) are in the other `archive/*` tags and their branches.
