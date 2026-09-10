# Assistant Axis on NDIF — collaborator run guide

Goal: produce **per-role persona vectors** and the **Assistant Axis** for
`meta-llama/Llama-3.3-70B-Instruct`, judged by `Qwen/Qwen2.5-72B-Instruct`.

**Every model call runs remotely on NDIF.** This machine needs **no GPU**; it only
runs the driver loop, holds the tokenizer, and stores the (small) results.
`HF_TOKEN` is used *only* to download the gated tokenizer/config (~20 MB).

Deliverables to send back: `<OUT_DIR>/vectors/*.pt` and `<OUT_DIR>/axis.pt` (~0.7 GB).

---

## 1. Accounts / keys

| | Where | Notes |
|---|---|---|
| **NDIF API key** | <https://login.ndif.us> | free research access |
| **HF token** (read scope) | <https://huggingface.co/settings/tokens> | then **accept the license** at <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct> and <https://huggingface.co/Qwen/Qwen2.5-72B-Instruct> with that same account |

## 2. Get the code

```bash
# small checkout -- only the pieces this pipeline needs
git clone --filter=blob:none --sparse <REPO_URL> expLB
cd expLB
git sparse-checkout set assistant_axis src/probes/deception-detection
```
(A normal `git clone` also works; `assistant_axis/` is all that matters here.)

## 3. Build the environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # if you don't have uv
bash assistant_axis/pipeline/setup_venv_aa.sh            # creates ./.venv-aa (~2 GB, CPU only)
```
The first `import torch/transformers/nnsight` is slow (~3 min cold) — that's normal.

## 4. Configure

```bash
cp assistant_axis/pipeline/env.aa.example assistant_axis/pipeline/.env
# edit assistant_axis/pipeline/.env  ->  fill NDIF_API_KEY and HF_TOKEN
```

## 5. Smoke test (~10–20 min, tiny)

```bash
OUT_DIR=./aa-smoke ROLES="assistant pirate oracle default" \
  PROMPT_INDICES="0" QUESTION_COUNT=6 MIN_COUNT=1 \
  bash assistant_axis/pipeline/run_pipeline_ndif.sh
```
Expect: `aa-smoke/responses/*.jsonl` (4 files), `aa-smoke/scores/*.json`,
`aa-smoke/vectors/*.pt` with shape `(80, 8192)`, and `aa-smoke/axis.pt`.
If the first NDIF job sits in `QUEUED` for a few minutes, that's the 70B cold-starting —
it warms up and subsequent jobs are fast.

## 6. Full run

Pick an `OUT_DIR` on a disk with **~8–10 GB free**, run under `tmux`/`nohup`:

```bash
tmux new -s aa
OUT_DIR=/path/with/space bash assistant_axis/pipeline/run_pipeline_ndif.sh
```

- Full spec = 276 roles × 5 prompts × 240 questions ≈ **331k generations**.
- **Plan for ~1–3 weeks of wall-clock.** NDIF is a shared queue and generation is the
  dominant cost; judging and vector extraction are much quicker.
- **Fully resumable.** Ctrl-C any time and re-run the same command — step 1 skips
  `(role, prompt, question)` triples already written, step 2 skips scored responses,
  step 3 skips roles whose `vectors/<role>.pt` exists.
- Run steps individually if you prefer (each script has `--help`); or narrow with
  `ROLES=`, `PROMPT_INDICES=`, `QUESTION_COUNT=`.

### Knobs (env vars for `run_pipeline_ndif.sh`)

| var | default | meaning |
|---|---|---|
| `OUT_DIR` | — (required) | output directory |
| `MODEL` | `meta-llama/Llama-3.3-70B-Instruct` | swap to `meta-llama/Llama-3.1-70B-Instruct` if 3.3 won't deploy |
| `JUDGE_MODEL` | `Qwen/Qwen2.5-72B-Instruct` | |
| `ROLES` | all | e.g. `"assistant pirate default"` |
| `PROMPT_INDICES` | all 5 | e.g. `"0 1 2"` |
| `QUESTION_COUNT` | 240 | |
| `GEN_BATCH` / `JUDGE_BATCH` / `ACT_BATCH` | 24 / 48 / 8 | lower if NDIF reports OOM |
| `MIN_COUNT` | 50 | min score-3 responses per role to emit a vector |

## 7. Deliver

Send `<OUT_DIR>/vectors/` and `<OUT_DIR>/axis.pt`. `responses/` and `scores/` are
only needed to resume the run and can be deleted afterwards.

`axis.pt` is a `torch.save` of a `(80, 8192)` tensor;
`vectors/<role>.pt` is `{"vector": (80,8192) float32, "type": "pos_3"|"mean", "role": str}`.

---

## Troubleshooting

- **`NDIF_API_KEY not set`** — `.env` isn't at `assistant_axis/pipeline/.env`, or the key line is blank.
- **`Invalid host URL`** — keep `NDIF_HOST=https://api.ndif.us` in `.env`.
- **`402` / HF Inference errors** — you shouldn't hit HF Inference at all; this path only
  downloads the tokenizer. If you see it, you're running the old `*_hf.py` scripts.
- **`invalid literal for int() ... b'P'` / `ConnectionError`** — the NDIF result socket
  crashing on a long job. `setup_venv_aa.sh` already pins the working combo
  (`websocket-client==1.8.0`, `python-engineio==4.13.0`, `python-socketio==5.16.4`); if you
  somehow have newer ones, reinstall those exact pins. The scripts also retry each batch.
- **A role is skipped: "only N kept"** — fewer than `MIN_COUNT` responses scored 3 for
  that role; lower `MIN_COUNT` or generate more questions/prompts for it.
