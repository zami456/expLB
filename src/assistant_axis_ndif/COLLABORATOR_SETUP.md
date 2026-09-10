# Assistant Axis on NDIF — collaborator run guide

Goal: produce **per-role persona vectors** and the **Assistant Axis** for
`meta-llama/Llama-3.3-70B-Instruct`, judged by `meta-llama/Llama-3.1-405B-Instruct`.

**Every model call runs remotely on NDIF.** This machine needs **no GPU**; it only
runs the driver loop, holds the tokenizer, and stores the (small) results.
`HF_TOKEN` is used *only* to download the gated tokenizer/config (~20 MB).

Deliverables to send back: `<OUT_DIR>/vectors/*.pt` and `<OUT_DIR>/axis.pt`
(~2.6 MB per role, so ~260 MB for the default 100-role set).

---

## 1. Accounts / keys

| | Where | Notes |
|---|---|---|
| **NDIF API key** | <https://login.ndif.us> | free research access |
| **HF token** (read scope) | <https://huggingface.co/settings/tokens> | then **accept the license** at <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct> and <https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct> with that same account |

## 2. Get the code

```bash
git clone --filter=blob:none --sparse <REPO_URL> expLB
cd expLB
git sparse-checkout set src/assistant_axis_ndif assistant_axis
```
The pipeline scripts live in `src/assistant_axis_ndif/`; they import the library and
read the role/question data from the vendored `assistant_axis/` submodule, so you need
both. (A normal `git clone --recurse-submodules` also works.)

## 3. Build the environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # if you don't have uv
bash src/assistant_axis_ndif/setup_venv_aa.sh            # creates ./.venv-aa (~1.1 GB, CPU only)
```
The first `import torch/transformers/nnsight` is slow (~3 min cold) — that's normal.

If the repo lives on a slow mount (e.g. a Windows drive accessed from WSL via `/mnt/c`),
put the venv on the native filesystem instead and point the driver at it:

```bash
AA_PY=~/venv-aa/bin/python bash src/assistant_axis_ndif/run_pipeline_ndif.sh
```

## 4. Configure

```bash
cp src/assistant_axis_ndif/env.aa.example src/assistant_axis_ndif/.env
# edit it -> fill NDIF_API_KEY and HF_TOKEN
```

## 5. Smoke test (~10–20 min, tiny)

```bash
OUT_DIR=./aa-smoke ROLES="assistant pirate oracle default" \
  PROMPT_INDICES="0" QUESTION_COUNT=6 MIN_COUNT=1 \
  bash src/assistant_axis_ndif/run_pipeline_ndif.sh
```
Expect: `aa-smoke/responses/*.jsonl` (4 files), `aa-smoke/scores/*.json`,
`aa-smoke/vectors/*.pt` with shape `(80, 8192)`, and `aa-smoke/axis.pt`.
If the first NDIF job sits in `QUEUED` for a few minutes, that's the 70B cold-starting —
it warms up and subsequent jobs are fast.

## 6. Full run

Run under `tmux`/`nohup`:

```bash
tmux new -s aa
OUT_DIR=/path/with/space bash src/assistant_axis_ndif/run_pipeline_ndif.sh
```

By default this uses `role_groups.json` — **100 roles** curated for deception and
sycophancy, grouped so the downstream class contrasts can be computed. At the paper's
per-role depth that is 100 × 5 prompts × 240 questions = **120,000 generations**.

- **Measure before committing to a long run.** `_measure_throughput.py` calls the real
  production code paths against the live 70B and projects the wall clock. The one
  measured anchor so far is ~21.7 s/generation at `GEN_BATCH=6`; larger batches should
  improve that substantially, but that is a projection until measured.
- Reducing depth is the cheapest lever: `PROMPT_INDICES="0 1" QUESTION_COUNT=120` gives
  240 rollouts/role, still ~4x the `MIN_COUNT=50` threshold, for 1/5 the generation.
- **Fully resumable.** Ctrl-C any time and re-run the same command — step 1 skips
  `(role, prompt, question)` triples already written, step 2 skips scored responses,
  step 3 skips roles whose `vectors/<role>.pt` exists.
- Run steps individually if you prefer (each script has `--help`).

### Knobs (env vars for `run_pipeline_ndif.sh`)

| var | default | meaning |
|---|---|---|
| `OUT_DIR` | — (required) | output directory |
| `AA_PY` | `./.venv-aa/bin/python` | python to run the stages with |
| `MODEL` | `meta-llama/Llama-3.3-70B-Instruct` | swap to `meta-llama/Llama-3.1-70B-Instruct` if 3.3 won't deploy |
| `JUDGE_MODEL` | `meta-llama/Llama-3.1-405B-Instruct` | HOT **and pinned** on NDIF, so it never cold-starts. `meta-llama/Llama-3.1-70B-Instruct` is also pinned and much cheaper per call |
| `ROLES_FILE` | `role_groups.json` | curated role set with group labels |
| `GROUPS` | all | subset of `ROLES_FILE` groups, e.g. `"deception honesty"` |
| `ROLES` | — | explicit roles; overrides `ROLES_FILE` |
| `LAYERS` | `all` | `all` (80 layers, 2.6 MB/role), `target` (middle layer only, 32 KB/role), or explicit indices |
| `PROMPT_INDICES` | all 5 | e.g. `"0 1"` |
| `QUESTION_COUNT` | 240 | |
| `GEN_BATCH` / `JUDGE_BATCH` / `ACT_BATCH` | 24 / 48 / 8 | lower if NDIF reports OOM |
| `GEN_SESSION_BATCHES` / `JUDGE_SESSION_BATCHES` / `ACT_SESSION_BATCHES` | 8 / 8 / 20 | batches bundled into one NDIF job. Keep `sessions x seconds_per_batch` well under NDIF's ~1h per-request ceiling — a session retries whole |
| `MIN_COUNT` | 50 | min score-3 responses per role to emit a vector |

## 7. Deliver

Send `<OUT_DIR>/vectors/` and `<OUT_DIR>/axis.pt`. `responses/` and `scores/` are
only needed to resume the run and can be deleted afterwards.

`axis.pt` is a `torch.save` of an `(80, 8192)` tensor;
`vectors/<role>.pt` is
`{"vector": (80,8192) float32, "layers": [0..79], "type": "pos_3"|"mean", "role": str, "n": int}`.
The `layers` list records which layer each row is, so `axis[40]` means layer 40 whatever
`LAYERS` was set to.

---

## Troubleshooting

- **`NDIF_API_KEY not set`** — `.env` isn't at `src/assistant_axis_ndif/.env`, or the key line is blank.
- **`Invalid host URL`** — keep `NDIF_HOST=https://api.ndif.us` in `.env`.
- **`402` / HF Inference errors** — you shouldn't hit HF Inference at all; this path only
  downloads the tokenizer. If you see it, you're running the old `*_hf.py` scripts.
- **`invalid literal for int() ... b'P'` / `ConnectionError`** — the NDIF result socket
  crashing on a long job. `setup_venv_aa.sh` already pins the working combo
  (`websocket-client==1.8.0`, `python-engineio==4.13.0`, `python-socketio==5.16.4`); if you
  somehow have newer ones, reinstall those exact pins. The scripts also retry each batch.
- **A role is skipped: "only N kept"** — fewer than `MIN_COUNT` responses scored 3 for
  that role; lower `MIN_COUNT` or generate more questions/prompts for it.
