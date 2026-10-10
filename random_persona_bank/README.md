# Random persona bank: the kd-bank pipeline on 35 random Assistant Axis roles

Started 2026-10-10 at the owner's request. Everything the xmodel runs produced for the **kd bank** (36 = `default` +
35 purpose-written honest / deceptive / sycophantic / anti-sycophantic personas), produced again for a bank whose 35
roles are **drawn at random** from the Assistant Axis library, on five models. It is a control: it shows whether the
kd bank's geometry, and later its detection, come from the purpose-written personas or from any 36 personas.

## Decisions (owner, 2026-10-10)

| | |
|---|---|
| Role draw | uniform over the 274 AA roles other than `assistant` (it would duplicate `default`); **no stance filter** (1a) |
| Bank size | `default` + 35 random = 36, the kd bank's structure |
| Questions | the kd bank's 30 shared questions, byte-identical, so the role set is the only thing that changes |
| Models | Llama-3.3-70B, Qwen2.5-72B, Mistral-Small-24B-2501, Mistral-Small-3.1-24B-2503, Gemma-3-27B (not Llama-405B) |
| Outputs | everything the kd bank has per model: vectors (s23 / all / s3), axes + PCA, the five contrasts as random-split nulls (R5-R9), and R5 + K / R9 + K |

## The bank (`bank/`, built by `bank/build_bank.py`; `--check` rebuilds it in memory and compares)

- **Draw:** `random.Random(20261010).sample(sorted(role_list - {assistant}), 35)`, made once and never redrawn.
  `bank/draw.json` holds the rule, the roles, each one's AA description and md5, and, for information only, which of
  our hf275 labels each carries (22 of 35 carry a loose or strict label, e.g. gossip, negotiator, vigilante under
  loose deception; editor, skeptic, perfectionist under strict anti-sycophancy).
- **Roles** (35): absurdist, ancient, artisan, auctioneer, bartender, divorcee, editor, egregore, elder, entrepreneur,
  exile, facilitator, familiar, ghost, gossip, infant, librarian, loner, negotiator, perfectionist, pharmacist, pilot,
  polymath, presenter, psychologist, researcher, revolutionary, skeptic, specialist, spirit, summarizer, vigilante,
  whale, witch, writer.
- **Role files** (`bank/rpb_instructions/`): each AA role's 5 system-prompt variants and its own `eval_prompt`, verbatim;
  `questions` set to the kd 30 (the generator reads only the shared file). `default.json` is the kd bank's, byte-identical.
  Written ASCII-only: the pipeline reads role files with `Path.read_text()` and no encoding (cp1252 here).
- **Judge prompt:** AA's, which names Gemma ("(an AI assistant, Gemma, an LLM, etc.)", 3 times per role). `deploy.ps1`
  renames it to the judged model in the deployed copy (asserted 35 files x 3, no Gemma left); Gemma's run keeps it.
  AA's 0-3 scale words scores 0 and 1 a little differently from the kd prompts; 2 and 3 -- the only scores the s23 and
  s3 banks keep -- mean the same in both.
- **Random splits** (`bank/splits.json`, fixed before any rollout): the kd bank's 35 roles fall into six disjoint groups
  (deception 8, honesty 7, sycophancy 5, anti-sycophancy 5, knowledge denial 4, knowledge candour 6). Each split deals the
  35 random roles into groups A-F of those sizes and builds the same contrasts: R5 = A v B (8 v 7), R6 = C v D (5 v 5),
  R7 = A+C v B+D (13 v 12, pooled), R8 = E v F (4 v 6), R9 = A+E v B+F (12 v 13, pooled). Split 0 is materialised like
  D5-D9; splits 1-99 are the null distribution. Seed rule: split i uses `random.Random(1000000 + i)`.

## Running

| | |
|---|---|
| Run folders | `results/<Model>/` (`code/` deployed, `persona/`, `logs/chain.status`), one per model; outputs gitignored |
| Pipeline | `xmodel/code` (identical: `diff -r` clean apart from the excluded unused `kd/`) + the bank in `code/rpb/` |
| Chain | `scripts/run_chain.ps1` = `xmodel/run_chain.ps1` minus the eval pooling, plus the bank paths and a `llama33` tag |
| Queue | `scripts/queue.ps1`: llama33 -> qwen -> mistral2501 -> mistral31 -> gemma3, one model at a time (3 NDIF requests), with a stall guard (a shard whose job sits RUNNING / DISPATCHED >= 25 min, or whose log is silent 25 min, is stopped; the chain resumes it) |
| Start | `powershell -ExecutionPolicy Bypass -File random_persona_bank\scripts\register_queue.ps1` -- **registering starts the run** (task `RPB-queue`, 30-min watchdog, 5 days) |
| Drive | `tamara:expLB/random_persona_bank_2026-10-10/<Model>/` (new folder, `rclone copy` only, `.env` excluded) |

Same as the kd bank runs: sampling 0.7 / 0.9 / top_k 0 / rep 1.0, 512 tokens; judge Llama-3.1-70B; banks s23 (2|3),
all, s3 (3), min 20 replies; every layer; `default` unjudged.

## Checks done before any NDIF request (2026-10-10)

- `build_bank.py --check`: 40 files rebuilt, 0 differ. `default.json` and the questions md5-equal the kd bank's.
- `deploy.ps1`: every deployed bank file md5-equals `bank/MANIFEST.md5` before the rename; rename 35 files / 105
  occurrences for Llama, Qwen, Mistral x2; Gemma unchanged.
- `scripts/check_persona_spans.py` (tokenizer only; the pipeline's own `build_messages`, `build_items`, tokenizer
  loader): **5,400 / 5,400 spans exact on all five models**, each followed by the end-of-turn token (`<|eot_id|>`,
  `<|im_end|>`, `</s>`, `</s>`, `<end_of_turn>`); longest sequence 729 tokens. Log: `results/span_check_2026-10-10.log`.

## CPU stage, per model after its vectors (scripts written 2026-10-10)

- `scripts/r_contrasts.py`: R5-R9 for split 0 -> `<run>/r5..r9/R*_<bank>.pt` (same keys as the D files) and, for all 100
  splits, `<run>/r_null/R_null_<bank>.csv` (norm, norm relative to the bank's role spread, cos with the assistant
  direction; per split / contrast / layer). **Check:** given the kd groups instead of the splits, it reproduces Qwen's
  D5-D9 (`xmodel-runs/Qwen2.5-72B-Instruct/d5..d9/D*_s23_v2.pt`) to <= 2.6e-08 relative, groups identical.
- `scripts/rk.py`: R5 + K and R9 + K for split 0 -> `<run>/r5k|r9k/R*K_<bank>.pt` (same keys as the D + K files) and the
  100-split null `<run>/rk_null/RK_null_<bank>.csv` (spread ratio sd(A@R)/sd(A@K), cos(R, K), cos(R + K, K)), from one
  pass over the Alpaca store. **Check:** rebuilds Qwen's `d9k/D9K_s23_v2.pt` to 2.7e-08 relative.
- Inputs: K = `K/<key>/first/K.npz` and the Alpaca store -- Llama's from this run folder; the other models' from
  `%USERPROFILE%\xmodel-runs\<Model>\` (Gemma: `eval_fp32/alpaca`).

## Llama-3.3: full-depth K and Alpaca (owner 2026-10-10: "full depth K and alpaca for llama")

Searched first, so nothing is re-run that exists (2026-10-10 02:1x):
- **K at all layers exists nowhere on our side.** zami's `origin/paper` (head `e27ffaf9`, no push since 2026-10-07 00:11)
  commits only K's result CSVs/JSONs, no arrays, on any branch. The laptop / Drive `xmodel_2026-10-06/K-validation-llama`
  K is the 4-layer validation run (L20/36/40/44). The lab-PC backup on Drive (`labpc-backup-2026-10-06`) and `kd/`,
  `coords/`, `share/`, `staging/`, `scale/`, `thesis-docs/` hold no K. The lab PC itself did not answer SSH (25 s, no output).
- **Alpaca:** Drive `coords/passB/alpaca` = 2,000 rows at 21 layers; `coords/passA/alpaca` = 200 rows at 80 layers. No
  2,000-row read at every layer, and those older chunks lack the `span` key `xmodel/d9k.py` checks.
- fp16 is safe for Llama: passA's mean-pooled Alpaca values peak at 31.0 (chunk 0, all 80 layers; fp16 max 65,504).

So the queue runs, after Llama's chain is DONE and before Qwen starts: `xmodel\k\k_chain.ps1 -Model llama-70b-3.3`
(both item sets, 80 layers -> `results/Llama-3.3-70B-Instruct/K/`) then `xmodel\alpaca_chain.ps1` (Llama's 2,000 Liars'
Bench Alpaca rows, 80 layers, fp16 -> `.../eval/alpaca/`). `code/results/subsets/alpaca.csv` placed in Llama's run folder
(md5 = repo `results/subsets/alpaca.csv`); offline `--prepare_only` + `check_spans.py`: 2,000 / 2,000 spans exact, all
followed by `<|eot_id|>`. These two steps use the xmodel scripts in place (not copied).

## Monitoring

`scripts/rpb_monitor.sh` (for Claude's Monitor tool; one event per stdout line): new lines in every status file each
minute (queue, chains, Llama K / Alpaca, failed Drive copies); new ERROR / Traceback lines in the stages' `.err.log`,
what is alive (`scripts/rpb_procs.ps1`: queue / chain / python processes, task state, free RAM) every 5 minutes; a
progress line every 10 minutes, and an ALERT if the counts stand still for 45 minutes. It exits after 29 minutes and is
re-armed; a token file retires older copies, and `results/_queue/monitor.state` keeps its read offsets. Fixes that need
no judgement run without it: the queue's stall guard and the task's 30-minute watchdog.

## Log
- 2026-10-10 bank built, five run folders deployed, span checks passed.
- 2026-10-10 02:11 `register_queue.ps1`: task RPB-queue (Interactive; S4U refused), Llama-3.3 chain started (preflight
  READY, inputs 36 / 30 / 36). 02:19 queue replaced by the version with the Llama K + Alpaca steps and "attach": only the
  queue process was stopped (pid-checked); the running chain and its 3 shards were untouched, and the new queue logged
  "chain llama33 already running (pid 21396): attaching". (Tested first on a throwaway task: stopping a scheduled task's
  process leaves its child running.)
- 2026-10-10 07:37 Llama vectors_s3: shard 0 hung after the socket error "packet queue is empty, aborting" (07:31:52; its
  job showed RUNNING 6 min on a 108-row role that takes ~1.5 min). Its two python processes were stopped by hand
  (command line matched on vectors_s3 + --shard 0/3) instead of waiting for the 25-min stall guard; the chain's next
  attempt rebuilds that shard's missing roles (per-role resumable; no partial vector is ever written). Earlier the same
  day: one "Replica was evicted" (vectors_s23, retried OK) and one server-side "tensors on cuda:3 and cuda:2" (retried OK).
- 2026-10-10 07:31-07:40 the laptop's link to NDIF dropped (07:31 socket errors on s3 shards 0 and 1; 07:40 an SSL
  handshake timeout on shard 2), leaving all three s3 shards waiting on jobs that never reported back. Shard 0 stopped by
  hand (07:37), shard 1 by the queue's stall guard (07:56, 25 min), shard 2 by hand (07:56, after checking NDIF answered
  again). The chain then started vectors_s3 attempt 2 with 13 / 36 vectors kept.
- 2026-10-10 08:14 Llama chain DONE: 5,400 replies; judge score 3 on 83% (kd bank 69%); all 36 roles eligible in every
  bank (min s3 count 85); s23 / all / s3 36 / 36 each; axes. L40 s23: cos(PC1, assistant) +0.447 (kd bank +0.620), PC1 31.2%.
- 2026-10-10 08:50 Llama K DONE, both item sets, 80 layers: equals the 4-layer validation K (cos >= 0.99996 at
  L20/36/40/44); split-half 0.9861 / 0.9982 / 0.9986 / 0.9984 = zami's; cos(K, K') L20 0.9599, L40 0.9929.
  09:39 Llama Alpaca read 2,000 / 2,000 rows x 80 layers (fp16, finite).
- 2026-10-10 ~09:50 Llama R5-R9 and R5 + K / R9 + K built for s23 / all / s3, with their 100-split nulls. L40 s23: R9 norm
  relative to the role spread 0.441 (null 0.259-0.538; the kd bank's D9 on Llama 0.773); R9 + K spread ratio 1.13 and
  cos(R9 + K, K) 0.757 (null 0.70-1.50 and 0.58-0.84; kd D9 + K on Llama 1.03 / 0.76, but that used passB's Alpaca --
  recompute it on this run's full Alpaca read before comparing).
- 2026-10-10 13:1x kd reference recomputed on the same full Alpaca read and K (`xmodel/LOG.md`, last section; the log moved there from `xmodel/README.md` on 10-10): Llama kd
  D9 + K at L40 s23 ratio 1.03, cos(D9 + K, K) 0.764 -- inside the random R9 + K null (ratio 0.70-1.50, cos 0.58-0.84);
  at L20 0.915 vs null 0.825-0.965. D5 + K L40 s23 cos 0.824 vs the R5 + K null 0.56-0.82 (at its top).
- 2026-10-10 13:07 Qwen generate shard 2: socketio receive thread died on the engineio b'P' packet (ValueError: invalid
  literal for int() ... b'P') right after its 13:01 job was submitted, so the job's result could never arrive. Its two
  python processes stopped by hand (command line matched on Qwen + 1_generate + --shard 2/3); shards 0 and 1 continue;
  the chain's next generate attempt regenerates only shard 2's missing rows. Earlier (11:00) one "Replica was evicted"
  on the same shard, retried OK.
