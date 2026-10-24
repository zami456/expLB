# The Geometry of Lies

**Knowledge and Persona Directions for Detecting Deception in Large Language Models**

B.Sc. thesis, Department of Computer Science and Engineering, BRAC University, October 2026.
Md. Abdullah Al Zami, Suraiya Shanaz, Mahdi Hasan.

We build a lie detector that reads a language model's hidden states and uses **no deception labels**. It combines two fixed directions in activation space:

- **Persona contrast (D9).** The difference between the activations of the model playing honest and deceptive roles, extended with knowledge-denial roles.
- **Knowledge direction (K).** A direction that tracks whether the model knows the answer while the words of the reply stay fixed. It separates a lie from an honest "I do not know" when the text is identical.

The detector, **D9+K**, is the standardised sum of the two: one fixed vector. We evaluate it on [Liars' Bench](https://github.com/Cadenza-Labs/liars-bench) and on fine-tuned model organisms, and we repeat every study on three more models.

---

## Results

Transfer AUROC on Liars' Bench at a mid-depth layer; intervals are paired 95% bootstraps over test rows. The full table (every claim, model and layer) is `src/probes/deception-detection/results/xmodel/summary.md` on branch `xmodel-replication`.

| Model | D9 vs raw-activation probe | D9 minus published contrast (D5) | D9+K minus same-layer RepE probe | Persona claims replicate? |
|---|---|---|---|---|
| Llama-3.3-70B | 0.916 vs 0.732 (+0.184 [+0.172, +0.195]) | +0.084 [+0.074, +0.094] | +0.128 [+0.112, +0.145] | original study |
| Mistral-Small-3.1-24B | 0.810 vs 0.716 (+0.094 [+0.082, +0.105]) | +0.176 [+0.169, +0.183] | +0.038 [+0.022, +0.053] | yes |
| Gemma-3-27B | 0.921 vs 0.746 (+0.175 [+0.166, +0.184]) | +0.221 [+0.203, +0.238] | +0.064 [+0.050, +0.078] | yes |
| Qwen2.5-72B | 0.453 vs 0.559 (-0.106 [-0.116, -0.094]) | +0.002 [-0.007, +0.012] | +0.036 [+0.016, +0.057] | no: the model plays the roles poorly |

The persona claims hold on Llama, Mistral and Gemma. On Qwen they fail, but D9+K still beats its RepE probe. K's result on fine-tuned liars (model organisms) replicates on Mistral for three of four organisms and on neither Qwen nor Gemma. We report this as a finding and did not tune prompts to rescue it.

---

## Where things live

Paths are relative to `src/probes/deception-detection/`.

| What | Path | Branch |
|---|---|---|
| Thesis LaTeX source | `thesis_revised/` | `xmodel-replication` |
| Cross-model replication (section 5.6, Appendix D) | `thesis_revised/appendix/appendix_5.tex`, `thesis_revised/make_xmodel_assets.py` | `xmodel-replication` |
| Cross-model report | `results/xmodel/summary.md` | `xmodel-replication` |
| Model specs, paired intervals, template-free K on other models | `deception_detection/scripts/lb_models.py`, `transfer_ci.py`, `k_final_xmodel.py` | `xmodel-replication` |
| K design and robustness scripts | `deception_detection/scripts/k_*.py` | `xmodel-replication` |
| Probe training and evaluation (upstream code) | `deception_detection/`, `train_multi_dataset.py` | `main` |

`main` carries the upstream code this work builds on. The thesis and the replication live on `xmodel-replication` and are not yet merged into `main`. The compiled `main.pdf` is not committed.

Every number in the thesis text is a `\V{...}` key generated from the result files by `thesis_revised/make_thesis_assets.py` and `make_xmodel_assets.py`, which also assert the claims the prose makes.

---

## Reproducing

```bash
git checkout xmodel-replication
uv sync                                    # Python deps (repo root)
cp .env.example .env                       # fill in API keys and HF_TOKEN
cd src/probes/deception-detection
make install-dev                           # probe code and its deps
```

- Hidden states of the large models were read through [NDIF](https://ndif.us); this needs an NDIF key and `HF_TOKEN`.
- Rebuild the cross-model report and thesis numbers: `python thesis_revised/make_xmodel_assets.py`.
- Build the PDF with [Tectonic](https://tectonic-typesetting.github.io): `tectonic --keep-logs main.tex` inside `thesis_revised/`.

---

## Built on Liars' Bench

This repository is a fork of [Cadenza-Labs/liars-bench](https://github.com/Cadenza-Labs/liars-bench), a benchmark for evaluating lie detectors on LLMs. Its generation notebooks, black-box detectors and baselines are kept as they were:

- `src/` holds the scenario generators and `src/blackbox/` the external-judge and self-evaluation detectors.
- `src/probes/` is the probe code (*Detecting Strategic Deception Using Linear Probes*, arXiv:2502.03407), vendored as a regular directory.
- `pacchiardi/` is the LLM-LieDetector baseline, `soft-trigger/` the sleeper-agent model organisms, `mask/` the MASK helpers.

The original Liars' Bench README, with its dataset-regeneration and detector-evaluation instructions, is in git history: `git show origin/main:README.md`.

---

## License

This codebase is licensed under Apache 2.0 (see `LICENSE`). Unless otherwise specified, all datasets created by the authors of this repository are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). Some datasets include third-party components and are subject to additional licensing terms (see `LICENSE`).
