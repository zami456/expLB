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
| Banks analysed | **UPDATE 2026-10-10 ~17:50 (owner: "use the judge filter too … all, s3 for all models, s23 for all models; separate folder … for 3 types of result"):** all three banks for every model, each in `results/<Model>/bank_<all\|s23\|s3>/`. In a judge-filtered bank a role with < 20 kept replies has no vector; it is dropped from every split's groups (`r_contrasts.py --allow_missing`), the null CSVs record each split's group sizes (`n_pos`, `n_neg`), and a split whose group empties is skipped. Qwen has no kd s3 D's (its kd s3 bank lacks 8 of their roles), so its s3 comparison is random-side only. |
| Main comparison bank | *(superseded by the row above for scope; `all` stays the cleanest comparison)* **`all`** for every model (owner 2026-10-10, option A): no judge filter, so all 35 random roles are always present and the pre-registered splits hold exactly; compared with each model's kd `all`-bank D's. |

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

**Entry point (2026-10-10 ~18:00): `scripts/analyze_banks.py --models <keys> [--banks all,s23,s3]`** -- runs the two
scripts below and the kd-vs-random comparison for each model and bank into `results/<Model>/bank_<bank>/`
(`r5..r9/`, `r_null/`, `r5k/`, `r9k/`, `rk_null/`, `compare_kd_vs_random_<bank>.csv`), plus one table per bank across
models in `results/compare_kd_vs_random_<bank>.csv`. Keys: Llama-3.3, Qwen-2.5, Mistral-2501, Mistral-3.1, Gemma-3.
The comparison: D size / kd role spread v the R null of the same quantity; |cos(D, assistant)| v the null's; D + K spread
ratio and cos(D + K, K) v the R + K null; `pct` = the kd value's percentile among the splits. Geometry only.

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
- 2026-10-10 13:46 Qwen generate attempt 2: the same engineio b'P' socket fault on shard 2 (job submitted 13:29); shard 2
  stopped by hand again; attempt 3 started at 5,100 / 5,400. Both b'P' faults today hit shard 2 shortly after a submit.
- 2026-10-10 15:05 Qwen judged: score 3 on 34% (Llama 83% on the same roles); score 1 on 42.4% (no trace of the role).
  Weak on neutral roles too (unlabelled 37% v Llama 81%; fantastical 37% v 99%), weakest on deception- (14%) and
  anti-sycophancy-flavoured roles (5%). Eligible roles: s23 31 / 35 (vigilante 3, loner 6, familiar 14, egregore 17 drop),
  s3 21 / 35 (< 30: no axes), all 35 / 35. r_contrasts.py needs all 35 roles, so Qwen's R's are built on the `all`
  bank (s23 / s3 pending the owner's choice A / B / C).
- 2026-10-10 15:5x Qwen R5-R9 and R5 + K / R9 + K on `all` (Qwen's own K and Alpaca read, xmodel-runs). The kd D9 + K
  lean to K on Qwen is inside the random null: L40 cos(D9 + K, K) 0.967 v R9 + K null 0.833-0.982 (ratio 3.66 v
  1.50-5.20); L20 0.985 v 0.841-0.984 (top edge). Same for D5 + K (L40 0.976 v 0.864-0.981). So "D9 + K ~ K on Qwen" is
  what any random contrast gives on Qwen's Alpaca geometry, not something specific to the deception contrast.
- 2026-10-10 17:44 Mistral-2501 judge shard 0: engineio b'P' socket fault right after its 17:40 submit (role revolutionary);
  stopped by hand; the chain's next judge attempt re-scores its remaining roles. Third b'P' of the day (Qwen generate x2).
- 2026-10-10 ~18:05 layout changed to one folder per bank (owner). Llama (3 banks) and Qwen (`all`) outputs rebuilt into
  `bank_*/` and checked identical to the earlier flat files (max difference 0), which were then removed; Qwen s23 / s3
  added (s23: 31 roles; s3: 21 roles, R6 defined in 97 of 100 splits). Llama kd `all` bank downloaded from the lab backup
  (36 / 36 matching) and its D5-D9 / D5 + K / D9 + K built; D6 / D7 added for Llama s23 / s3.
  L40, kd v random (pct among 100 splits): D9 size Llama 100 / 100 / 100 (all / s23 / s3), Qwen 97 / 98 / n.a.; D8 size
  Llama 100 in every bank, Qwen 81 / 92; D5 size Llama 98 / 99 / 100, Qwen 95 / 89; cos(D9 + K, K) Llama 66 / 65 / 53, Qwen
  68 / 77 -- the kd contrasts are larger than random splits of random roles, while D9 + K's lean to K is ordinary.
- 2026-10-10 18:05 Mistral-2501 vectors_s23 shard 0: the fourth engineio b'P' fault of the day; shard stopped by hand. The
  queue's stall guard now also stops a shard whose .err.log ENDS in "invalid literal for int() with base 10: b'P'" (the
  receive thread is dead; a recovered shard logs further lines). Tested offline on the dead log (fires) and a live one
  (does not); queue process swapped at 18:05:38 (only the queue stopped; it re-attached to the running Mistral chain).
- 2026-10-10 18:2x Mistral-2501 analysis stopped by Claude Code for critically low system memory (not a script error):
  `bank_all/` holds R5-R9 + r_null (complete); R + K for all, and the s23 / s3 banks, not run. Fixes (not yet run):
  rk.py projects 8 rows at a time (was a whole chunk in float64, ~300 MB per Llama chunk); analyze_banks.py skips steps
  whose CSV exists, so a re-run resumes. To finish: `python random_persona_bank/scripts/analyze_banks.py --models Mistral-2501`
  (validate rk.py first with its --validate mode on Qwen's D9K_s23_v2, expected ~3e-08).
- 2026-10-10 19:05-21:00 the laptop was in Modern Standby (System log, Kernel-Power 506/507): every process paused, NDIF
  sockets died. On wake the queue's guard stopped Mistral-3.1's generate shards 0 and 1 (silent 116 min / job RUNNING
  118 min; shard 2 had been stopped at 18:41 for b'P'); generate attempt 2 started 21:01:57 with 3,406 / 5,400 kept.
  Sleep while plugged in costs hours on an overnight run -- set "never sleep on AC" (powercfg) before leaving it.
- 2026-10-10 ~21:10 (owner: the 19:05 standby was an accidental lid close; "yes apply it"): lid close action on AC set to
  Do nothing (`powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS LIDACTION 0` + `/setactive`; the setting is hidden
  here, read it with `powercfg /qh`). DC unchanged (Sleep). Undo: same command with 1.
- 2026-10-11 04:35 QUEUE DONE (all 5 models); task RPB-queue disabled. Per model: 5,400 replies; judge score 3 on
  Llama 83%, Qwen 34%, Mistral-2501 70%, Mistral-3.1 75%, Gemma 96% (kd bank: 69 / 28 / 63 / 68 / 96%); vectors s23/all/s3
  36/36/36 for all but Qwen (32/36/22). Overnight NDIF faults all self-recovered: engineio b'P' (guard, several), replica
  evictions (retries), a 04:24 "packet queue is empty" on Gemma s3 (recovered on its own).
  CPU analysis status: Llama and Qwen all three banks done; Mistral-2501 bank_all R5-R9 only; Mistral-3.1 and Gemma not
  run (both Claude-Code-run attempts were stopped for low system memory; re-run only on the owner's go).
- 2026-10-11 07:40-07:56 (owner: "go with 2, run analyses after gemma finishes") CPU analysis finished for Mistral-2501
  (resumed: bank_all R + K, then s23, s3), Mistral-3.1 and Gemma-3, all three banks, run in the foreground with nothing
  else running (free RAM ~420 MB; ~5.5 min in all; none stopped). rk.py's 8-row batching checked first:
  `rk.py --validate` rebuilds Mistral-2501's kd D9K_s23 at 3.45e-08 max relative difference. Every model x bank now has
  `bank_<bank>/` (r5..r9, r_null, r5k, r9k, rk_null, compare CSV); combined tables `results/compare_kd_vs_random_<bank>.csv`
  hold 154 rows each (5 models), every row on 100 splits. Copied to Drive (`rclone copy --ignore-existing`, bank_* folders
  + the 3 combined CSVs; `rclone check --one-way` 51 files per model, 3 at the root, 0 differences).
  Result, bank `all`, kd percentile among the 100 random splits (deep report layer: Llama/Qwen L40, Mistrals L20,
  Gemma L40): D9 size 100 / 97 / 100 / 100 / 97 (Llama / Qwen / Mistral-2501 / Mistral-3.1 / Gemma); D6 and D7 size
  97-100 everywhere; D8 100 / 81 / 100 / 100 / 98; D5 98 / 95 / 99 / 100 / 95 (Gemma L20 53). cos(D9 + K, K) 66 / 68 / 38 /
  66 / 15 -- within the null for every model; Gemma's R + K null is very wide at L31/L40 (spread ratio 2.0-23.8 at L40)
  and its kd D9 + K leans to K LESS than most random splits there (pct 7-15). |cos(D9, assistant)| 77 / 94 / 64 / 75 / 28.
  s23 and s3 give the same picture (Qwen s3 has no kd side). Reading: on all five models the kd contrasts are larger than
  contrasts between random groups of random roles, so their size is specific to the purpose-written personas; the D9 + K
  lean toward K is not -- any random split does it. Geometry only; nothing here measures detection.
- 2026-10-11 (caveat on the 04:35 entry's "kd bank: 69 / 28 / 63 / 68 / 96%"): the two banks are judged with different
  prompts -- the random bank with AA's own (1 = "no trace of the role, answers as itself"), the kd bank with kd's
  (1 = "cannot take the role but helps"). Only score 3 (fully in role) is defined alike; compare that, not 0-2.
  Qwen on random roles: 1 = 42.4%, 2 = 24.0%, 3 = 33.7%. Results written into `xmodel/REPORT_MODEL_PORTS.md` as dated
  UPDATE blocks (owner, 10-11).
