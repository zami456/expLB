# Experiments

Run every script from the repository root as a module, for example `python -m experiments.knowledge.k_final --help`. Each script's docstring gives its commands, inputs and outputs. Outputs go to `results/<name>/` and model reads to `activations/`. The large inputs come from `python data/fetch.py`.

The folders follow the study's pipeline:

| Folder | Covers | Main scripts | Writes |
|---|---|---|---|
| `probes/` | supervised baselines, held-out lies | `experiment.py` with `configs/*.yaml` (supervised probes per layer), `pool_activations.py`, `aisi_probe.py`, `transfer_alignment.py` | `results/XXXX-9-*/`, `activations/pooled/`, `results/aisi_probe/` |
| `persona/` | persona contrasts, register, depth | `persona_ood.py` (D5–D9, transfer and zero-shot), `layer_sweep.py`, `persona_positions.py`, `unified_axis.py`, `persona_finetuned.py`, `hpkr_wording.py` | `results/persona_ood*/`, `results/persona_depth/`, `results/loo_*/`, `results/persona_positions_*/`, `results/persona_unified/` |
| `knowledge/` | knowledge direction K, combined detector | `recall_items*.py` (the questions), `recall_denial.py` (reads), `k_final.py`, `k_models.py`, `k_robustness.py`, `k_replication.py`, `k_transfer.py`. The other `k_*.py` are helpers these import. | `results/recall_denial/` |
| `organisms/` | steering, model organisms | `gender_secret.py`, `soft_trigger.py`, `organisms.py`, `steer_notebook.py` | `results/gender_secret/`, `results/soft_trigger/`, `results/steering_L40_notebook/` |
| `other_models/` | Qwen, Mistral, Gemma | `lb_models.py`, `k_final_xmodel.py`, `transfer_ci.py` | `results/xmodel/<model>/` |
| `external/` | external tests | `aletheia/` (Aletheia's Quest), `questions_150/` (the 150 pre-registered questions), `question_bank.py`, `random_bank.py` | `results/aletheia/`, `results/question_bank/`, `results/xmodel/llama/random_bank/` |

The persona vectors these scripts read are built in [`../persona_bank/`](../persona_bank/), and the published copies come from `python data/fetch.py banks`.
