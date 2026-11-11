# Persona bank

This folder builds the persona vectors the paper reads. A model role-plays honest and deceptive characters, a judge scores how fully each reply stays in role, and the kept replies are averaged into one vector per role. Every model call runs on [NDIF](https://ndif.us), so no local GPU is needed. The pipeline follows the Assistant Axis method; its upstream code is pinned at [`assistant_axis/`](assistant_axis/), which is used unmodified. This line of work was built by Suraiya Shanaz (branch `axis`).

| Folder | What it holds |
|---|---|
| [`src/assistant_axis_ndif/`](src/assistant_axis_ndif/) | The pipeline: `0_preflight` → `1_generate` → `3_judge` → `2_activations` → `6_axes`, plus `8_coords` for reading evaluation sets, and its role and question definitions. Start with [`RUNBOOK.md`](src/assistant_axis_ndif/RUNBOOK.md). |
| [`kd/`](kd/) | The paper's Llama-3.3-70B bank: 36 roles × 5 instruction variants × 30 questions, including the knowledge-denial roles. [`kd/README.md`](kd/README.md) covers the design and the rebuild. |
| [`xmodel/`](xmodel/) | The same bank on Qwen2.5-72B, Mistral-Small-3.1-24B and Gemma-3-27B, the Qwen user-turn re-run, and the model ports. [`xmodel/README.md`](xmodel/README.md) is the guide. |
| [`random_persona_bank/`](random_persona_bank/) | The control bank of 35 random Assistant Axis roles. |

The pipeline has its own environment (`.venv-aa`, with pinned socket libraries that NDIF needs). Create it with `src/assistant_axis_ndif/setup_venv_aa.ps1`, or `.sh` on Linux. The paths here are relative to this folder, so its run outputs (`run*/`, `.env`, `.venv-aa/`) stay inside it and are git-ignored. The finished banks the paper uses are published in the release (`python data/fetch.py banks`).
