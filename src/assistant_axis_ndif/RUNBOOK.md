# RUNBOOK — NDIF Assistant-Axis pipeline (for Claude)

You are running this on a collaborator's machine after a fresh `git pull`. Goal: produce the
Assistant-Axis persona vectors and axis for **Llama-3.3-70B-Instruct**, with every model
call executed remotely on **NDIF** (no local GPU). Follow these steps in order. Everything
is resumable — if any step is interrupted, re-run the same command and it continues.

Assume **Windows (native, no WSL)** unless you detect otherwise; a Linux/WSL host can use the
`.sh` scripts and `bin/python` paths instead. Paths below are relative to the repo root.

---

## 1. Prerequisites (the human should have done these)

- `.env` exists at `src/assistant_axis_ndif/.env` with `NDIF_API_KEY` and `HF_TOKEN` filled
  (copied from `env.aa.example`). If missing, stop and ask them — you cannot supply keys.
- The HF account behind `HF_TOKEN` has accepted the licenses for `meta-llama/Llama-3.3-70B-Instruct`
  and `meta-llama/Llama-3.1-70B-Instruct` (the judge).
- `uv` is installed (`uv --version`). If not: `winget install astral-sh.uv`.

Do **not** commit `.env` or print the key values.

## 2. Build the environment

Windows:
```
powershell -ExecutionPolicy Bypass -File src\assistant_axis_ndif\setup_venv_aa.ps1
```
Linux/WSL:
```
bash src/assistant_axis_ndif/setup_venv_aa.sh
```
This creates `.venv-aa` (CPU-only, ~1.1 GB). The first `import torch/transformers/nnsight`
is slow (~3 min cold). The venv python is `.venv-aa\Scripts\python.exe` (Windows) or
`.venv-aa/bin/python` (Linux).

**The socket pins are load-bearing** (`python-socketio 5.16.4` + `python-engineio 4.13.0` +
`websocket-client 1.8.0`). The setup installs them in two steps — a normal resolve, then a
`--no-deps` force of the exact combo — because socketio 5.16.4's metadata *declares*
engineio>=4.13.2 and a strict resolver would otherwise pull engineio 4.14, which **crashes
the NDIF result socket on long jobs** (`invalid literal for int() ... b'P'`). After setup,
confirm the pin held (engineio exposes no `__version__`, so read it from package metadata):
```
.venv-aa\Scripts\python.exe -c "import importlib.metadata as m; print(m.version('python-engineio'), m.version('python-socketio'), m.version('websocket-client'))"
```
It must print `4.13.0 5.16.4 1.8.0`. If engineio shows 4.14.x, reinstall:
`uv pip install --python .venv-aa\Scripts\python.exe --no-deps --reinstall python-engineio==4.13.0`.

## 3. Confirm NDIF is reachable

```
.venv-aa\Scripts\python.exe -c "import os; from dotenv import load_dotenv; load_dotenv('src/assistant_axis_ndif/.env'); os.environ.setdefault('NDIF_HOST','https://api.ndif.us'); import nnsight; print(nnsight.is_model_running('meta-llama/Llama-3.3-70B-Instruct'), nnsight.is_model_running('meta-llama/Llama-3.1-70B-Instruct'))"
```
Both should print `True`. The 70B is HOT but **not pinned**, so it can occasionally be
evicted — if a run stalls, re-check this and wait for redeploy. `nnsight.compare()` will show
`torch`/`transformers` differ from the server; that is expected and fine (nnsight matches at
0.7.0, which is what governs the wire protocol).

## 4. Smoke test (~15 min) — the go/no-go

Run 3 roles, 6 questions, all stages, to a throwaway dir. This is the real test that the
socket pins hold on this machine over a multi-minute remote job:
```
set OUT_DIR=aa-smoke
set ROLES=spy auditor default
set QUESTION_COUNT=6
set PROMPT_INDICES=0
set MIN_COUNT=1
set GEN_BATCH=6
set GEN_SESSION_BATCHES=1
.venv-aa\Scripts\python.exe src\assistant_axis_ndif\run_pipeline.py
```
(PowerShell uses `$env:OUT_DIR="aa-smoke"` etc. instead of `set`.) Success = `aa-smoke\vectors\*.pt`
exist, `aa-smoke\axes.pt` is written, and no `b'P'` crash appeared. If it crashed on the
socket, fix the engineio pin (step 2) and re-run. Then delete `aa-smoke`.

## 5. Full run

The single entry point is `run_pipeline.py`, configured by env vars (defaults in parentheses):

| var | meaning |
|---|---|
| `OUT_DIR` | **required** output dir (needs ~1 GB free) |
| `LAYERS` | **`all`** = every layer, `(80, 8192)`/role, ~2.6 MB each (~260 MB total; pick the layer later). **`target`** = middle layer 40 only, ~32 KB each (~3 MB total). **Ask the human which** if unsure — default `all`. |
| `SHARD` | `i/n` — run only shard i of n. **Launch n processes** with different i to keep several NDIF jobs in flight. |
| `GEN_BATCH` (24) | generations per NDIF batch; 48 measured well, higher is fine on NDIF |
| `MIN_COUNT` (50) | min score-3 responses per role to emit a vector |

**On a 16 GB machine, run 3 shards in parallel** (each process ~1.2 GB). In three separate
terminals:
```
set OUT_DIR=results\assistant_axis\run
set GEN_BATCH=48
set SHARD=0/3
.venv-aa\Scripts\python.exe src\assistant_axis_ndif\run_pipeline.py
```
...and `SHARD=1/3`, `SHARD=2/3` in the other two. Each shard runs all stages over its third
of the 100 roles; shards are disjoint and never write the same file. Expect **~1.5–2 days**.

Stage cost (all on NDIF): generation dominates (~3.9 s/generation at batch 96), then judging
(70B judge), then activations (a single forward per conversation). All three are shared queues.

**Resumability:** Ctrl-C any shard and re-run the identical command — stage 1 skips
`(role, prompt, question)` triples already written, stage 2 skips scored responses, stage 3
skips roles whose `vectors/<role>.pt` exists. Never lose finished work.

After all shards finish stages 1–3, run stages 5–6 **once, unsharded**, to build the axis
over the full role set:
```
set OUT_DIR=results\assistant_axis\run
set ONLY=5,6
.venv-aa\Scripts\python.exe src\assistant_axis_ndif\run_pipeline.py
```
(The per-shard runs already build a partial axis each; this final unsharded pass rebuilds it
from all roles. `ONLY=5,6` skips the NDIF stages.)

## 6. Deliverables

Send back (~260 MB with `LAYERS=all`):
```
<OUT_DIR>/vectors/*.pt      per-role persona vectors  {vector:(80,8192) fp32, layers, type, role, n}
<OUT_DIR>/axis.pt           the Assistant Axis  (80, 8192)
<OUT_DIR>/axes.pt           assistant + deception + sycophancy contrasts + per-layer PCA
```
`responses/` and `scores/` are only needed to resume and can be deleted afterward.

Sanity check before sending: `axes.pt` at the end prints a validation report — the
`assistant_anchor` group should project positive on the assistant axis and `breadth_control`
negative; `cos(PC1, assistant axis)` should be high at the middle layer. A flipped sign is
fine (recorded), but zero norms or all-NaN means a bad run — re-run the affected roles.

## 7. Troubleshooting

- **`invalid literal for int() ... b'P'` / socket resets on long jobs** → engineio pin
  slipped to 4.14; redo the `--no-deps` pin in step 2, then re-run (resumes).
- **A role logs "Skip … only N kept"** → fewer than `MIN_COUNT` responses scored 3 for that
  role; expected attrition (the paper kept 377 of 275+ roles for Llama). Not an error.
- **`402` or HF Inference errors** → you're running an old `*_hf.py` script by mistake; use
  `run_pipeline.py` / the `*_ndif.py` stages.
- **`is_model_running` returns False mid-run** → the 70B was evicted (it isn't pinned);
  wait and re-check, then resume.
- **`NDIF_API_KEY not set`** → `.env` isn't at `src/assistant_axis_ndif/.env` or the key line
  is blank.
- **(Windows) SSL "certificate has expired" or `UnicodeEncodeError` on a status glyph** →
  handled automatically: the stages point SSL at `certifi` and force UTF-8 stdout at startup
  (uv's standalone Python otherwise uses a stale CA store and a cp1252 console). If you ever
  see these, run via `run_pipeline.py` (which also sets them) rather than a bare stage, or
  set `SSL_CERT_FILE` to certifi's `cacert.pem` and `PYTHONIOENCODING=utf-8` yourself.
