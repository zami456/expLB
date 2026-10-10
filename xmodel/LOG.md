> **Moved here verbatim from `xmodel/README.md` on 2026-10-10.** The dated running log of the xmodel work (2026-09-29 onward). The new `README.md` is the guide; this file is the full record behind it.

# xmodel — the thesis pipeline on the other Liars' Bench models

Started 2026-09-29 at the owner's request: produce, for another Liars' Bench model, what the final
report built for Llama-3.3-70B-Instruct — the kd persona vectors at every layer and that model's
own Liars' Bench rows pooled at every layer — so the report's experiments can be rerun on it.
Fine-tuned models are out of scope. **Qwen2.5-72B-Instruct only, for now.**

## What runs where

| | |
|---|---|
| Lab PC run folder | `D:\user4fina\Qwen2.5-72B-Instruct\` (`code\`, `persona\`, `eval\`, `logs\`) — a new subfolder; the rest of `D:\user4fina` predates this and is not touched |
| Drive | `tamara:expLB/user4fina_2026-09-29/Qwen2.5-72B-Instruct/` (new folder, `rclone copy` only, `.env` excluded) |
| Python | `C:\Users\user4\expLB\.venv-aa` (used, never modified) |
| Task | `XModel-qwen`, S4U as user4, AtStartup — `register_task.ps1` |
| Chain | `run_chain.ps1`: generate (Qwen ×3) → judge (Llama-3.1-70B ×3) ‖ eval pooling (Qwen ×3) → vectors_s3, vectors_all (Qwen ×3) → axes |

At most three NDIF requests per model at a time (owner). The judge is Llama-3.1-70B-Instruct, as
for the Llama kd bank, so the subject model is the only thing that changes.

## What is identical to the Llama run

- The kd bank: 36 personas × 5 variants × 30 questions (`kd/personas`, `kd/questions`, byte-equal
  to the lab PC's copy and unchanged since `3dfdb673`, the version the report used).
- The pipeline scripts: copies of `src/assistant_axis_ndif/*` (md5-equal on laptop and lab PC).
- Persona vectors: mean over reply tokens, score-3 filter, min 20 kept, every layer; sampling
  temperature 0.7, top-p 0.9, 512 new tokens.
- Eval pooling: `8_coords_ndif.py`, mean over the reply span, fp32 accumulation, fp16 storage.

## The three changes, each forced by the model

1. **Anchor name.** `default.json` is "You are {model_name}." and the script hardcoded `Llama`, so
   Qwen would have been told "You are Llama." Now `XMODEL_SHORT_NAME=Qwen`.
2. **Sampling defaults.** Qwen's `generation_config.json` adds `top_k=20` and
   `repetition_penalty=1.05`; Llama's adds neither, and NDIF runs transformers 5.8, whose defaults
   are none. So the Llama bank sampled with no top-k and no repetition penalty. The Qwen run passes
   `--top_k 0 --repetition_penalty 1.0`, which switch both off (transformers `generation/utils.py`
   lines 1109, 1238 in 5.8).
3. **Judge prompt.** The 35 judged personas' `eval_prompt` says the model "still identifies as
   itself (an AI assistant, Llama, an LLM…)"; the Qwen copy says Qwen.

Plus: `eval_datasets.CSV_MODEL` reads `XMODEL_CSV_MODEL` so the loader takes Qwen's rows.

## Evaluation data for Qwen (Liars' Bench HF sha `503399b8` = local `results/subsets`)

| dataset | rows | lies |
|---|---|---|
| instructed-deception | 5,160 | 2,145 |
| harm-pressure-choice | 2,374 | 374 |
| harm-pressure-knowledge-report | 2,571 | 571 |
| alpaca (control) | 2,000 | 0 |

Qwen has no convincing-game or insider-trading rows, so its transfer matrix has three datasets,
not five. Every row stores its own system prompt, including Qwen's default.

## Not yet covered

- **RepE** (the D2 base probe's training set) is not in `eval_datasets.py`; add it before
  reproducing D2.
- **K** (the knowledge direction) needs zami's `recall_items.py` / `recall_items_matched.py`,
  which are on no branch, not on the lab PC and not on Drive (checked 2026-09-29).
- Mistral-Small-3.1 and Gemma-3 are vision-language checkpoints: nnsight 0.7's `LanguageModel`
  refuses them, their decoder is at `model.language_model.layers`, and Mistral's tokenizer has no
  chat template. Qwen2.5 needs none of that (`model.layers`, 80 × 8192).

## Checks before the full run

- `check_spans.py`: every eval span decodes to exactly the stored reply (the encoder's Qwen branch
  was written for Qwen-3).
- A smoke generation and a smoke pooling on NDIF; pooled values finite in fp16.

## 2026-10-06 — moved to the laptop (lab PC going down for maintenance)

The 09-29 lab run never started (no task, only `code\` on `D:\user4fina\Qwen2.5-72B-Instruct`, nothing
on Drive). It now runs from the laptop instead. **Scope: persona vectors only** (owner) — `-SkipEval`
drops stage 2's Liars' Bench pooling; K waits for zami's `recall_items*.py` (on no branch).

| | |
|---|---|
| Start | `powershell -ExecutionPolicy Bypass -File xmodel\register_task.ps1 -Machine laptop` (registering STARTS it) |
| Run folder | `%USERPROFILE%\xmodel-runs\Qwen2.5-72B-Instruct\` (`code\` deployed from `xmodel\code`, `persona\`, `logs\chain.status`) |
| Python | repo `.venv-aa-win` (nnsight 0.7.0 + the lab's socket pins) |
| Drive | `tamara:expLB/xmodel_2026-10-06/Qwen2.5-72B-Instruct/` (new folder, `rclone copy`, `.env` excluded) |
| Task | `XModel-qwen`, current user, Interactive; 30-min watchdog repeat for 3 days + AtLogOn |

`run_chain.ps1` / `sync_drive.ps1` / `register_task.ps1` take the machine paths as parameters; the lab
defaults are unchanged. Checks done 2026-10-06 before any full run:
- code copy vs repo: 35 persona files differ ONLY by Llama→Qwen, `default.json` identical, questions
  byte-identical; `1_generate_ndif.py` differs only by `XMODEL_SHORT_NAME` and `--top_k/--repetition_penalty`;
- laptop NDIF path reproduces lab Pass B bit-identically (ai_liar, 270 rows, L40);
- Qwen smoke: 2 roles × 2 questions (variant 0) generated in 312 s; default variant 0 is the empty
  system prompt by design, variant 3 renders "You are Qwen.". The feigned-novice replies read only
  weakly in character — watch the score-3 counts against the 20-row minimum.

**2026-10-06 23:18 — periodic Drive copy added.** The chain only syncs at stage ends (first one ~04:30),
so task `XModel-qwen-sync` (current user, every 30 min for 3 days) runs `sync_drive.ps1 -SkipLiveLogs`
into the same Drive folder, sharing the chain's lock. First attempt exited 1: four logs (the running
shards' `generate.N.log` and rclone's own `rclone.log`) changed during upload and failed Drive's md5
check. Fix: `logs/rclone.log` is now always excluded and `-SkipLiveLogs` drops `logs/*.[0-9].log`; the
stage-end syncs still carry the finished logs. Re-test 23:22: exit 0, 0 errors, 1,178/1,178 replies on Drive.
No `.env` reached Drive (0 files named `.env`).

## 2026-10-07 — K (knowledge direction) for Qwen, from zami's code

zami pushed K to `origin/paper` (commit `74deaf90`, branch head `e27ffaf9`). Snapshot (git archive, read-only):
`xmodel/k/zami_origin-paper_e27ffaf9/` — code, `pyproject.toml`, and `results/recall_denial/` (Llama reference).
Driver `xmodel/k/k_run.py` calls zami's functions unchanged; chain `xmodel/k/k_chain.ps1`, task `XModel-qwen-K`;
env `.venv-k` (zami's pins + our tested NDIF pins: nnsight 0.7.0, socketio 5.16.4/engineio 4.13.0/websocket 1.8.0
forced with --no-deps, httpx 0.28.1, transformers 5.17.0; `goodfire` installed --no-deps, import only).

K = unit(mean known − mean unknowable) per layer, "none" arm, general items, 60+60 × 2 denials = 240 reads,
all 80 layers, both sets (first, matched). Overrides, each measured:
- model key (zami hardcodes llama-70b-3.3); output to `%USERPROFILE%\xmodel-runs\Qwen2.5-72B-Instruct\K\`;
- system text = the model's own HP-KR text from results/subsets: Qwen's 2,571 rows all carry
  "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." (Llama's 2,139 = zami's NO_ROLE_SYSTEM);
- **reply mask**: zami's Qwen tokenizer branch includes `<|im_end|>` (8/11 tokens) where Llama's span is the
  words only (7/10); special tokens are unmarked → Qwen 7/10, Llama unchanged (0 removed). Checked on all 240 rows;
- zami's check C (span-rule cross-check) is a non-Qwen port and breaks under transformers 5 (apply_chat_template
  tokenize=True returns a BatchEncoding); run for Llama with an id-returning shim (6/6 pass); check A skipped
  (needs stored HP-KR activations);
- `--gate_status`: every Qwen request waits while the persona chain holds 3 Qwen requests (generate, vectors_*).
- Llama validation (reproduce zami's verify.json + split_half "none|general") deferred until the persona run is
  done: Llama-3.3 was NOT_DEPLOYED at 00:59 and loading it could evict Qwen mid-run. Qwen K is not trusted
  until that validation passes.

**2026-10-07 02:40 — K's answer check was wrong for Qwen, found and fixed before any read.** zami's
`steer_lies._tokenize` renders Qwen's generation prompt with the reply turn already closed
(`<|im_start|>assistant\n<|im_end|>\n`; Llama's ends open). Qwen then opened a new message ("Note: It seems there
was an accidental ...") on 17/70 known items, scored as wrong (France, hexagon, Mercury ...), and `items` stopped at
"only 53 verified known items". `k_run.open_generation_turn` removes the trailing `<|im_end|>\n` after an empty
assistant turn and adds Qwen's end tokens to zami's Llama-only STOP_TOKENS; offline, Llama's 130 prompts stay
byte-identical, Qwen's all end `<|im_start|>assistant\n`. Re-run: **69/70 known correct** (the miss: largest desert,
key C Antarctica, Qwen said A — a real error, replaced from the reserve as zami's method does), 60/60 unknowable
denied. The flawed file is kept as `K/qwen-72b/first/verify_closed_turn_bug.json`.
Also checked: the PERSONA-vector spans (Assistant-Axis encoder) on 100 real Qwen replies equal the reply text
exactly (no `<|im_end|>`) — the persona pipeline is unaffected.

**2026-10-07 02:56 — K now waits for the persona chain's DONE.** In the judge window K's first all-layer read
(20 rows x 80 layers x 8192 fp32 = 52 MB) took ~8 min to download, and meanwhile the judge's downloads fell to
3-4 KB/s and its uploads failed (WriteTimeout, ConnectionError). The laptop link is the shared resource, so
`k_run.py`'s gate is now BUSY for generate / judge / vectors and FREE only at `DONE`. The finished read
(`parts_answer/000002_20.npz`) is kept; reads resume per request. Expect ~15 requests x 52 MB per set.

**2026-10-07 03:15 — judge done; Qwen plays the personas weakly; s23 bank added.** 5,250 judged (Llama-3.1-70B):
score 3 = 1,469, 2 = 3,534, 1 = 7, 0 = 240. Only 24 of 35 judged roles reach 20 score-3 replies; below 20:
coverup_operator 2, broker 4, feigned_novice 4, gatekeeper 5, deniable_expert 7, contrarian 10, yes_man 11,
enabler 12, honest_novice 15, fact_checker 17, narrative_manager 19. The 09-29 plan built only s3 + all, but the
Llama kd **thesis bank is s23** (scores 2 or 3, min 20; `kd/scripts/run_kd_s23.ps1`; F-10 s3 is drift-selected;
F-121). Under s23 all 36 roles are eligible (min 66 replies). `run_chain.ps1` now builds vectors_s23 -> vectors_all
-> vectors_s3, "done" = every eligible role (s23 36, all 36, s3 25), axes for each bank with >= 30 vectors
(so not for s3). Chain restarted 03:18 (generation and judging kept); vectors_s23 started 03:19.

**2026-10-07 03:21 — laptop download floor.** `ndif_common.harden()` kills a result download under 300 KB/s
(after 12 s) and retries it, then re-runs the whole NDIF session. Three vector shards downloading at once split
the laptop link to ~70-95 KB/s each: two old-s3 sessions failed outright and an s23 session survived only on its
4th download attempt. `harden()` now honours NDIF_MIN_DOWNLOAD_KBPS / NDIF_DOWNLOAD_GRACE (unset = unchanged);
`run_chain.ps1 -MinDownloadKbps 40 -DownloadGrace 20` on the laptop (still above the ~35 KB/s sick-flow speed the
floor exists to catch). The judge's error logging now prints the exception's last line (was str(e)[:200]).
Restarted 03:21:55; all three shards log "floor 40 KB/s after 20s: installed".

**2026-10-07 04:25 — persona run DONE.** vectors_s23 36/36, vectors_all 36/36, vectors_s3 25/25 eligible; every
vector (80, 8192), finite, non-zero; reply counts per role match the judge's scores. Axes: `axes_vectors_s23.pt`,
`axes_vectors_all.pt` (s3 skipped, 25 < 30). The first DONE (04:20) had **no axes**: the axes loop used `$tag`,
which in PowerShell is the `[ValidateSet("qwen")]` parameter `$Tag`, so the assignment threw and the loop was
skipped silently; fixed (`$bank`, and the note now asserts the axes file exists) and re-run.
L40, s23 bank (logs/axes.log; Llama from lab backup expLB-logs/kd_axes_s23.log):

| | Llama-3.3-70B | Qwen2.5-72B |
|---|---|---|
| cos(PC1, assistant axis), min/median over layers | +0.620 (0.531/0.610) | +0.838 (0.712/0.838) |
| PC1 share of role variance | 26.8% | 51.4% |
| deception cos(PC1,contrast) / cos(PC1,assistant) | 0.965 / -0.502 | 0.977 / -0.786 |
| sycophancy | 0.993 / -0.365 | 0.977 / -0.779 |
| unified | 0.909 / -0.552 | 0.940 / -0.809 |

## 2026-10-07 05:30 — Qwen K done and the setup validated on Llama

**Qwen K** (zami's knowledge direction, both item sets, all 80 layers; `K/qwen-72b/{first,matched}/K.npz`):
verify first set 69/70 known correct (largest-desert miss replaced from the reserve), 60/60 unknowable denied;
matched set 70/70, 59/60 denied (1 claimed, diagnostic only). Split-half of K, first set: L20 0.9638, L36 0.9789,
L40 0.9824, L44 0.9855 (median over 80 layers 0.9822; Llama/zami 0.9861-0.9986). Replication cos(K first,
K matched): L20 0.916, L36 0.947, L40 0.953, L44 0.956, median 0.953 (Llama/zami cos(k',k) 0.960-0.993).

**Validation** (`k_validate.ps1`; zami's code via k_run.py and .venv-k, Llama-3.3, first set, layers 20/36/40/44)
against zami's committed results: verify identical (70/70 correct flags, 60/60 reports, 130/130 reply texts);
split-half 0.9862/0.9982/0.9986/0.9984 vs zami 0.9861/0.9982/0.9986/0.9984; zami's check C 6/6; strict span
check 240/240. The environment, driver and Qwen-only fixes reproduce zami's K, so the Qwen K stands.
Drive: `tamara:expLB/xmodel_2026-10-06/Qwen2.5-72B-Instruct/K/` (43 files, rclone check 0 differences) and
`tamara:expLB/xmodel_2026-10-06/K-validation-llama/` (28 files). All XModel tasks disabled after the final copy.

**zami's code snapshot is not committed.** `xmodel/k/zami_origin-paper_e27ffaf9/` falls under the repo's
`.gitignore` rule `**/probes/**` (line 198), which keeps zami's probe code off this branch. It is an exact copy of
origin/paper @ e27ffaf9; recreate it before running `xmodel/k/k_run.py`:

    git fetch origin
    mkdir -p xmodel/k/zami_origin-paper_e27ffaf9
    D=src/probes/deception-detection
    git archive e27ffaf9 $D/deception_detection $D/pyproject.toml $D/README.md $D/results/recall_denial \
        | tar -x -C xmodel/k/zami_origin-paper_e27ffaf9

## 2026-10-07 12:00 — reply-span fix, persona vectors rebuilt (`_v2`), and D5 / D8 / D9

**The span defect.** On Qwen's ChatML template the Assistant-Axis `ConversationEncoder` starts the reply span
one token early, at the `\n` that ends the `<|im_start|>assistant` header, so every Qwen reply mean included
one header token. `ndif_common.trim_chatml_header_newline` drops that token (it returns at once on a
tokenizer without `<|im_start|>`, so Llama is untouched); `2_activations_ndif.py` and `8_coords_ndif.py` call
it after computing `a0, a1`. It moves a span only when the token before it is `<|im_start|>assistant` and the
span starts at `\n`, so it is a no-op on every other template. Checked: Qwen persona spans 5,400/5,400 exact;
Qwen eval spans exact for all 12,105 rows; on Llama the helper moves 0 spans and the HP-KR items are
2,139/2,139 identical to lab Pass B.

**Rebuild.** The persona vectors were rebuilt from the same rollouts and the same judge scores as
`persona/vectors_{s23,all,s3}_v2` (the v1 folders are kept, each with a `SUPERSEDED.txt`). s23_v2: 36/36 at
12:49, every rollout count identical to v1, all files [80, 8192], finite. v1 -> v2 over the 36 roles:
cos min 0.99976 / median 0.99999 at L40, min 0.99874 over all layers (pc_stonewaller at L79, relative change
5.2% -- its replies are short, so one token weighs more); centred on the bank mean, cos min 0.9966 at L40.
The fix is right and the old vectors were close.

**D5, D8, D9** (`xmodel/d_contrasts.py`; report section 4.2.7). zami's recipe, `persona_pcs._axes_from_groups`
(origin/paper @ e27ffaf9, `persona_pcs.py:39-169`): mean(positive roles) - mean(negative roles) per layer in
float64, unit-normalised for use; D9 is the pooled 12 v 13 mean, not the average of D5 and D8.

| | positive | negative |
|---|---|---|
| D5 published (8 v 7) | handler, broker, impression_manager, coverup_operator, gatekeeper, spin_doctor, plausible_denier, narrative_manager | archivist, disclosure_counsel, documentarian, fact_checker, ombudsman, transparency_officer, whistleblower |
| D8 knowledge-denial (4 v 6) | amnesiac_witness, deniable_expert, feigned_novice, stonewaller | calibrated_reporter, candid_witness, forthright_expert, honest_novice, plain_dealer, scrupulous_witness |
| D9 extended (12 v 13) | D5's 8 + D8's 4 | D5's 7 + D8's 6 |

**Bank: s23, not zami's s3.** zami builds them from the score-3 bank (`run-kd/vectors_s3`). On Qwen that bank
keeps 25 of 36 roles and loses 8 of the 25 these contrasts need (broker, coverup_operator, gatekeeper,
narrative_manager, fact_checker, deniable_expert, feigned_novice, honest_novice), so the Qwen contrasts come from
`vectors_s23_v2` -- the bank the audit used to re-create zami's arms on Llama (`thesis_guide/PITFALLS.md:47`).
Rollouts per role 66 (amnesiac_witness) to 150. Compare Qwen D's with Llama D's built from Llama's s23, not s3.

Files: `<run>/d5/D5_s23_v2.pt`, `d8/D8_s23_v2.pt`, `d9/D9_s23_v2.pt` (`raw`, `unit` [80, 8192] float32, `norm`,
groups, rollouts and input md5 per role) + a `.json` sidecar each. Score = activation @ unit[layer]; positive =
the deceptive group. Checks: zami's own `persona_pcs.load_bank_at_layer` on the same folder gives the same
unit axes at all 80 layers x 3 contrasts (worst relative difference 5.8e-08, float32 storage); the pipeline's
groups equal zami's. v1 -> v2 cos of the unit contrasts >= 0.998 at every layer (L40: D5 0.9998, D8 0.9993,
D9 0.9997). Drive: `tamara:expLB/xmodel_2026-10-06/Qwen2.5-72B-Instruct/d{5,8,9}/` (rclone check 0 differences).

| layer | norm D5 | norm D8 | norm D9 | cos(D5,D8) | cos(D5,D9) | cos(D8,D9) |
|---|---|---|---|---|---|---|
| L20 | 1.8355 | 2.9381 | 1.7036 | +0.5975 | +0.8908 | +0.8847 |
| L36 | 5.6563 | 8.1858 | 5.0621 | +0.5679 | +0.8988 | +0.8536 |
| L40 | 6.4237 | 9.2108 | 5.7333 | +0.5294 | +0.8888 | +0.8412 |
| L44 | 6.7281 | 9.9255 | 6.1007 | +0.5264 | +0.8862 | +0.8420 |

**Not done: D9 + K.** The selected detector divides each direction by its spread on Alpaca
(`recall_denial.standardised_sum`) before adding them; that needs Qwen's Alpaca eval read, which has not run.

**13:21 rebuild DONE.** all_v2 36/36 (13:08), s3_v2 25/25 eligible (13:18), axes for s23_v2 and all_v2 (s3_v2 has
25 < 30 vectors). Every v2 bank: same roles and rollout counts as v1, all [80, 8192], finite. Largest v1 -> v2
move anywhere: pc_stonewaller in s3 at L79 (relative 15%, cos 0.990); at L40 cos >= 0.9972 in every bank. s23_v2
geometry at L40 is unchanged: cos(PC1, assistant) +0.839 (v1 +0.838), PC1 51.3% (51.4%), deception 0.977 / -0.787.
D5 equals the pipeline's v2 deception axis (s23 and all, relative 2e-06). Drive: whole run folder 439 files,
rclone check 0 differences (13:22). XModel tasks disabled.

**Robustness: D's from the unfiltered bank** (`d*/D*_all_v2.pt`, 150 rollouts per role). cos(s23_v2, all_v2) at
L40: D5 0.998, D8 0.964, D9 0.984. The judge filter moves D8 most -- it removes most from the knowledge-denial roles
(amnesiac_witness keeps 66 of 150) -- and the filtered D8 is larger (norm 9.21 v 6.61 at L40).

**Qwen v Llama-3.3, same recipe, s23 banks** (Llama vectors: `labpc-backup-2026-10-06/expLB/run-kd/vectors_s23`;
both models have 80 layers). Norms are not comparable across models, so each is divided by the RMS distance of
the bank's 36 role vectors from their mean at that layer. On Llama's s3 bank (byte-identical to zami's) the
builder reproduces zami's D's (relative 4.7e-08).

| L40 | rel D5 | rel D8 | rel D9 | cos(D5,D8) | cos(D5,D9) | cos(D8,D9) |
|---|---|---|---|---|---|---|
| Qwen s23_v2 | 0.739 | 1.059 | 0.659 | +0.529 | +0.889 | +0.841 |
| Llama s23 | 0.753 | 1.299 | 0.773 | +0.470 | +0.866 | +0.838 |

D5 is as large relative to the role spread in both models; Qwen's knowledge-denial contrast is smaller. This is
geometry only: whether the Qwen D's detect lies needs the Liars' Bench eval read, which has not run.

## 2026-10-07 14:16 — Qwen's Alpaca eval read and D9 + K

**Read** (`xmodel/alpaca_chain.ps1`, task `XModel-qwen-alpaca`, `register_alpaca_task.ps1`): the run_chain eval
stage restricted to alpaca -- `8_coords_ndif.py --datasets alpaca --layers all --pooling mean --span response`,
Qwen's own 2,000 LB Alpaca rows (CSV rows 4000-5999), mean over the reply at all 80 layers, fp16. 14:16-15:05,
52 sessions, ~3 MB/s downloads (far above the 90-340 KB/s measured 10-06), 2000/2000 rows, row shape 80x8192x1.
Items identical to the span-checked set (2000/2000 tokens and spans). Drive: whole run folder 496 files, 0
differences (15:17). Task disabled.
Pitfall, caught before any NDIF call: without `XMODEL_CSV_MODEL` the eval loader takes **Llama's** rows
(eval_datasets.CSV_MODEL default) -- the chain sets it; a hand-run must too. And `8_coords` installs its own
download floor (`--min_download_kbps`, default 300) after ndif_common's; pass the laptop value to both.

**D9 + K** (`xmodel/d9k.py`; zami's `recall_denial.standardised_sum`, recall_denial.py:428-431):
D9_unit / sd(Alpaca @ D9_unit) + K / sd(Alpaca @ K), K = `none|general` from the first item set. Chunks are
streamed; the file keeps the Alpaca projections, so a 1%-FPR threshold needs no second read. Equal to zami's own
`standardised_sum` on the full Alpaca matrix at L40 (relative 3e-08). Files `<run>/d9k/D9K_s23_v2.pt` (main) and
`D9K_all_v2.pt` (unfiltered-bank D9), + .json; on Drive (4 files, 0 differences).

| L40 | sd(A@D9) / sd(A@K) | cos(D9, K) | cos(D9+K, D9) | cos(D9+K, K) |
|---|---|---|---|---|
| Qwen, s23_v2 | 6.678 / 1.691 = **3.95** | +0.106 | 0.340 | **0.971** |
| Llama-3.3, s23 | 0.385 / 0.375 = **1.03** | +0.143 | 0.748 | 0.764 |
| Llama-3.3, s3 (zami's bank) | 0.346 / 0.375 = 0.92 | +0.150 | 0.781 | 0.735 |

(Llama: D9 from `labpc-backup-2026-10-06/expLB/run-kd/vectors_{s23,s3}`, K from the Llama validation run, Alpaca
from `tamara:expLB/coords/passB/alpaca` -- 8_coords mean pooling, not zami's Apollo pooling, so a close reference;
zami's committed cos(D9, K) at L40 is 0.1503, matching the s3 row.) Qwen at L20/36/44: ratio 5.76/5.71/3.98,
cos(D9+K, K) 0.986/0.985/0.971.

**On Qwen, D9 + K is almost K.** Alpaca's replies spread ~4x more along D9 than along K, so the recipe shrinks D9.
On Llama the two spreads are equal and D9 + K is a balanced blend. Partly an outlier-dimension effect, but not
mostly: Qwen dim 5060 averages 64.8 on Alpaca (median |mean| 0.09); dropping the 1 / 5 / 20 highest-variance
dimensions takes the ratio to 3.43 / 2.89 / 2.65 (Llama 1.07 / 1.30 / 1.33). 96% of D9's Alpaca spread lies in
Alpaca's top-5 principal directions on Qwen (Llama 71%): Qwen's D9 runs along the way ordinary replies vary.
Whether this costs D9 + K anything on Qwen needs the Liars' Bench eval read (not run).

## 2026-10-07 15:40 — a second family: Mistral (owner: "start for 1 ... solve the issues for 2 ... separate folders")

**One run folder per model** under `%USERPROFILE%\xmodel-runs\` (Drive: `tamara:expLB/xmodel_2026-10-06/<folder>`):
`Qwen2.5-72B-Instruct`, `Mistral-Small-24B-Instruct-2501`, `Mistral-Small-3.1-24B-Instruct-2503`. `register_task.ps1
-Tag <qwen|mistral2501|mistral31> -Machine laptop [-DeployOnly]` deploys `xmodel/code` into `<folder>\code` and names
the judged model in the 35 judge prompts (xmodel/code's kd copy says Qwen; asserted 35 changed, 0 Qwen left).

**Soft-trigger and gender-secret** (`eval_datasets.py`). The organisms were hardcoded to Llama's: a Qwen or Mistral
run asking for them would have read Llama's organisms' rows without a word. Now per base model (Liars' Bench names):

| | soft-trigger (3 organisms) | gender-secret |
|---|---|---|
| Llama-3.3 | 3 x 2,000, 1,000 lies each (unchanged) | 392, 116 lies (unchanged) |
| Qwen-2.5-72B | 3 x 2,000, 1,000 lies each | **refuses**: Liars' Bench has no Qwen organism |
| Mistral-3.1-24B | 3 x 2,000, 1,000 lies each | 370, 115 lies |
| `none` (2501) | refuses | refuses |

Every model's soft-trigger answers are the single word True/False (checked: SINGLE_TOKEN holds). Apollo's rollouts,
sycophancy_mt and the kd factor probes (labelled by what Llama knows) are Llama's and now refuse on any other model;
natural sarcasm is human-written and stays open. `all_keys()` lists only datasets holding the model's rows.

**Two tokenizer fixes for Mistral** (`ndif_common.load_tokenizer` / `load_language_model`, used by 1_generate,
2_activations, 8_coords and check_spans):
- `fix_mistral_regex=True`: transformers 5 warns that Mistral's tekken pre-tokenizer regex "will lead to incorrect
  tokenization". The pre-tokenizers differ, but 0 of 1,210 of our texts tokenise differently; set for correctness.
- **The template reads the clock.** With no system message -- the default role's first variant ("") in every bank --
  Mistral's template injects its default prompt with `strftime_now("%Y-%m-%d")`, so a reply generated one day and read
  the next is read under a different prompt. Pinned to 2025-09-01, the date in every Liars' Bench Mistral row.
  Llama-3.3 ("26 Jul 2024", static) and Qwen (no date) never read the clock: checked, no-op for them.

**Mistral-Small-24B-2501** (HOT, `LanguageModel`, 40 x 5120, no LB rows: `XMODEL_CSV_MODEL=none`). Persona spans
5,400/5,400 exact (before `[/INST]`, after `</s>`). Its generation_config samples at temperature 0.15; the bank's flags
override it (0.7 / 0.9 / top_k 0 / rep 1.0). Smoke on NDIF from the deployed folder: generate 8, judge 4 (scores 2-3),
vectors (40, 5120) finite; the default role's "You are Mistral." anchor. **Run started 16:15, task XModel-mistral2501.**

**Mistral-Small-3.1-24B-2503** (the Liars' Bench model; vision-language, COLD on NDIF). Solved offline:
- nnsight 0.7's LanguageModel refuses it and names VisionLanguageModel, whose AutoProcessor needs Pillow/torchvision
  (absent; torchvision would move the pinned torch). Text-only input never touches the processor, so it gets a stand-in
  holding only the tokenizer; the tokenizer takes the processor's `chat_template.json`. Model key on NDIF:
  `VisionLanguageModel:{repo_id}`.
- The decoder is at `model.language_model.layers`; 2_activations / 8_coords decide `vl` on the client and pass it into
  the trace helpers (Llama/Qwen/2501 read `model.layers` exactly as before). Sizes via `config.get_text_config()`.
- Spans: persona 5,400/5,400 exact; Liars' Bench eval 21,362/21,362 exact over 10 datasets (ID 5,010, HP-KR 2,358,
  HP-C 892, IT 2,134, alpaca 2,000, MASK 598, soft-trigger 3 x 2,000, gender-secret 370); check_spans exit 0.

**K for both Mistral models** (`k_run.py`). zami's code loads Mistral locally only; the two keys `mistral-small-24b-2501`
and `mistral-small-3.1-24b` run on NDIF through the xmodel loader with zami's tokenizer conventions (left pad, pad = bos).
HP-KR system text: the Mistral rows' (Mistral Small 3's default prompt, 2025-09-01). `</s>` joins zami's Llama-only stop
tokens (on Mistral those map to `<unk>`). zami's verify steers at layer 40 with coef 0 -- block 40 does not exist in a
40-layer model, so it moves to 20 (coef 0: same result). For 3.1, zami's read and generate are swapped for copies in
k_run.py that walk `language_model.layers`. Offline: zami's generation prompt ends `[/INST]` (turn open, unlike Qwen's);
the 240 forced replies read exactly the reply on both tokenizers (7 and 10 tokens, as on Llama).

## 2026-10-08 11:09 — Mistral-Small-24B-2501 persona run done

Run folder `%USERPROFILE%\xmodel-runs\Mistral-Small-24B-Instruct-2501\` (Drive: same name under
`tamara:expLB/xmodel_2026-10-06/`, 288 files, rclone check 0 differences, 2026-10-08 11:10). Generate 5,400 (16:15-17:16
on 10-07); judge 5,250 (attempt 1 lost shard 1 to an engineio b'P' hang, stopped by hand; attempt 2 completed 18:42);
vectors_s23 36/36 (18:48); **NDIF outage ~19:00 10-07 to 10:51 10-08** -- the three vectors_all shards sat on pre-outage
jobs (RECEIVED/QUEUED 9-16 h) and were stopped by hand; attempt 2 finished vectors_all 36/36 (10:59), vectors_s3 36/36
(11:06), axes for all three banks (11:08). Task XModel-mistral2501 disabled; K task XModel-mistral2501-K running.

**Judge scores** (Llama-3.1-70B): 3 = 3,295 (63%), 2 = 1,888, 1 = 9, 0 = 58 -- Mistral plays the roles far more fully
than Qwen (score 3: 28%). All 35 judged roles have >= 20 score-3 replies, so zami's s3 recipe is complete on Mistral.

**Geometry, s23, L20 (mid-depth of 40)** (`logs/axes.log`): cos(PC1, assistant) +0.575, PC1 29.2% of role variance;
deception cos(PC1, contrast) 0.966 / cos(PC1, assistant) -0.363. Llama-3.3 at L40: +0.620, 26.8%, 0.965 / -0.502;
Qwen at L40: +0.838, 51.4%, 0.977 / -0.786. Mistral's persona space is Llama-like; Qwen is the outlier.

**D5 / D8 / D9** (`xmodel/d_contrasts.py`, 40 layers): `d*/D*_{s23,all,s3}.pt`. The s3 set equals zami's
`persona_pcs` at all 40 layers (5.9e-08). At L20: cos(D5,D8) 0.467 (s23) / 0.433 (s3); cos(s23, all) >= 0.999 for all three
(the judge filter removes almost nothing on Mistral; Qwen's D8 moved to 0.964); cos(s23, s3) 0.975-0.984.

**K for Mistral-2501 (2026-10-08 11:11-11:29, task XModel-mistral2501-K, now disabled).** `K/mistral-small-24b-2501/{first,
matched}/K.npz`, 40 layers. Verify: first set 68/70 known correct (100/4 and 7x8 answered "A", replaced from the reserve),
matched 70/70; all 60 unknowables declined in Mistral's words ("I don't have the information.", 4 variants) -- zami's
`claims_knowledge` only matches "not know", so it files them as "other" (diagnostic only; K uses the forced denials).
Spans 240/240 exact on both sets (7 and 10 tokens, as on Llama). Split-half at L20: 0.9937 (median over 40 layers 0.9901;
min 0.9038 at L0); matched median 0.9954. cos(K, K') at L20: 0.985 (median 0.9795; min 0.8212 at L0). For comparison
Qwen L40 split-half 0.9824, cos(K, K') 0.953; Llama/zami 0.9986 and 0.993. Drive: K 79 files, K-logs 16, 0 differences.
D9 + K for Mistral-2501 needs an Alpaca control, and 2501 has no Liars' Bench rows of its own (open decision).

## 2026-10-08 13:30 — D6, D7 and D5 + K: the report's full zero-shot set on Qwen and Mistral-2501

The report's design table (4.1) has five zero-shot contrasts and two sums: D5-D9, D5 + K, D9 + K. Until now the
xmodel runs had D5, D8, D9, K and (Qwen) D9 + K. Added, CPU only, from files already on disk:

- **D6 sycophancy (5 v 5)** and **D7 unified (13 v 12, pooled)**: `xmodel/d_contrasts.py --only D6,D7` (new switch;
  the default still builds D5,D8,D9 and reproduces the existing Qwen D5/D8/D9 files bit for bit). Groups are zami's
  `persona_pcs.SYCOPHANCY_PERSONAS` / `ANTI_SYCOPHANCY_PERSONAS`. Checks: equal to the pipeline's own `sycophancy` /
  `unified` axes (relative <= 2.6e-06) and to zami's `persona_pcs.load_bank_at_layer(...).axes` at every layer
  (worst 5.7e-08: Qwen s23_v2 80 layers, Mistral s23 and s3 40 layers).
  Files: `<run>/d6/D6_<tag>.pt`, `<run>/d7/D7_<tag>.pt`; Qwen tags s23_v2, all_v2; Mistral-2501 s23, all, s3.
  Qwen's s3_v2 still cannot carry them: it lacks broker, coverup_operator, gatekeeper, narrative_manager,
  fact_checker, enabler, yes_man, contrarian.
- **D5 + K** for Qwen: `xmodel/d9k.py --name D5 --d <run>/d5/D5_<tag>.pt --out <run>/d5k` (new `--name`; `--d9` still
  works, and the edited script reproduces `D9K_s23_v2.pt` bit for bit). Keys say d5 where the D9 file's say d9.
  L40 s23_v2: sd(A@D5) 7.1350, sd(A@K) 1.6909, cos(D5+K, K) 0.9750 -- like D9 + K, on Qwen the sum is mostly K.
  Mistral-2501 has no D5 + K / D9 + K: it has no Alpaca rows (open decision, see above).

| L40 (Qwen) / L20 (Mistral-2501) | norm D6 | norm D7 | cos(D6, D7) |
|---|---|---|---|
| Qwen s23_v2 | 12.3306 | 7.9548 | +0.9478 |
| Mistral-2501 s23 | 1.8269 | 1.3325 | +0.8582 |

Drive (`tamara:expLB/xmodel_2026-10-06/<model>/d6|d7|d5k`): rclone check 0 differences for all five folders.
Still not built on any second model: D1 and D4 (trained on Liars' Bench rows, need the eval read), D2 (RepE probe:
RepE is not in `eval_datasets.py`), D3 (trained on AISI's Llama rollouts; nothing equivalent exists for other models).

## 2026-10-08 13:32 — Mistral-Small-3.1-24B-2503 run started (owner: "do this as same as for other models")

Smoke on NDIF first (13:29-13:31): the model was COLD; the request showed "Model Deploying..." and then returned the
layer-20 mean (1, 5120), finite, next token 'Red' (`VisionLanguageModel` key, text-only stand-in processor). The
10-07 smoke had never completed (stopped for laptop memory). A hand-run with stdout redirected to a file needs
`PYTHONIOENCODING=utf-8`: nnsight's status spinner (U+25C9) otherwise raises UnicodeEncodeError in cp1252, which reads
like an NDIF failure and is not one.

Deployed code = repo `xmodel/code` (diff: none outside the 35 judge prompts, each differing only by Qwen -> Mistral; the
deployed folder also holds the Liars' Bench CSVs and `mask/` placed for the 10-07 eval span check). Tasks (Interactive;
S4U refused without admin): `XModel-mistral31` (persona chain, `-SkipEval`, 3 shards; START 13:32:10, preflight READY,
36 roles / 30 questions) and `XModel-mistral31-K` (k_chain, key `mistral-small-3.1-24b`, gated on the persona chain's
DONE). After both: D5-D9 (`d_contrasts.py`, all three banks), the Alpaca read (Liars' Bench has 2,000 Mistral-3.1 Alpaca
rows) and D5 + K / D9 + K.

**13:33 STOPPED -- generation on Mistral-3.1 fails on NDIF's side.** Every shard's first session failed within ~1 s on the
server: shard 0 a CUDA device-side assert, shard 1 `RuntimeError: prob_dist must be 1 or 2 dim` (sampling), and every
later attempt "Error submitting request to model deployment". NDIF status afterwards: the replica we hit
(`304f2:...VisionLanguageModel...`) UNHEALTHY, a new one DEPLOYING -- our requests broke a shared deployment, so the tasks
were disabled and all 3.1 processes stopped at once (0 replies written; K never left its gate). First job ids:
e66c0bb9-a359-439f-bcd1-754c00b4ffa0 (shard 0), 28a24b7e-f1b8-4ac9-bb2a-ece487924036 (shard 1),
94b0ca84-e810-4c99-b224-0bd99233d4f0 (shard 2).
Not reproducible offline: the same call (VisionLanguageModel + tokenizer stand-in, the deployed ndif_common tokenizer,
real pc_handler prompts, gen_batch 32, a session of 4 generates, the chain's sampling kwargs) runs cleanly on a tiny
random Mistral3ForConditionalGeneration under both transformers 5.17 and NDIF's 5.8 (nnsight 0.7.0, torch 2.6 CPU; scratch
env `%USERPROFILE%\xmodel-runs\_scratch\v58`). The 3.1 checkpoint's generation_config holds only bos/eos. The smoke
(a `trace`, not a `generate`) works. So the fault is in the server-side generate path for this VLM deployment.

## 2026-10-08 14:02 — Mistral-2501's own Alpaca replies (owner: "go for a"), for D5 + K and D9 + K

Liars' Bench has Alpaca rows for Llama-3.3, Qwen-2.5, Mistral-3.1 and Gemma-3, not for 2501, and D5 + K / D9 + K need
the model's own replies (the spreads and the 1%-FPR threshold come from them). `xmodel/alpaca_gen.py` generates them on
NDIF with Liars' Bench's protocol (`src/alpaca.py`):
- the same 2,000 instructions as Llama / Qwen / Mistral-3.1 (checked: their user turns are identical, 2,000 / 2,000;
  Gemma's overlap only 1,148), taken from Mistral-3.1's rows, index 4000-5999 kept;
- system "You are a helpful AI assistant." + instruction, as stored;
- the checkpoint's own sampling: do_sample, temperature 0.15 (2501's generation_config), nothing else overridden;
  max_new_tokens 500. Replies that hit the cap are KEPT, as Liars' Bench kept them: 502 / 486 / 649 of 2,000 of its
  Mistral-3.1 / Llama / Qwen Alpaca replies are >= 490 tokens (their own tokenizers).
Smoke (4 rows, one session) OK; finalize -> `code/results/subsets/alpaca.csv` (model `mistral-small-24b-instruct-2501`),
`8_coords --prepare_only` and `check_spans` dry-run on a scratch copy: 4 / 4 spans exact, `</s>` right after each.
Chain `xmodel/alpaca_gen_chain.ps1` (task `XModel-mistral2501-alpaca`, `register_alpaca_task.ps1 -Gen`): 3 shards
generate -> finalize -> span check -> `alpaca_chain.ps1` read (all 40 layers, mean over the reply) -> `d9k.py` D9 + K
and D5 + K for banks s23 / all / s3 -> Drive. Started 14:02:51.

**14:50 — staged generate probe on NDIF (owner: "if model cold, first make it, i have permission").** NDIF had replaced
the broken replica (`f9172`, HOT RUNNING; `304f2` still listed UNHEALTHY). `nnsight.compare()`: server nnsight 0.7.0,
torch 2.6.0+cu126, transformers 5.8.0, tokenizers 0.22.2 (local scratch env v58 matches the first three).
- **A session sent by mistake.** The local dry run of the probe used `remote="local"`; nnsight 0.7 honours that for a
  single trace/generate but a `session(remote="local")` goes to NDIF, and the tiny model's config named the real repo.
  So one real session (2 sampled generates, batch 2, 3 tokens) reached `f9172`: `RuntimeError: prob_dist must be 1 or 2
  dim` after 4 s. (The single generates in that dry run did run locally and passed.)
- **Then stage 1 for real** (ONE generate, batch 1, greedy, 4 tokens, no session): job
  `10238293-d9d3-45c0-865d-89f389d92fe3` ERROR 0.1 s after RECEIVED, "device-side assert" -- before it could run, i.e.
  the replica's CUDA context was already poisoned, most likely by the session above. Status still shows RUNNING.
  Testing stopped; the stage-2 / no-session question (does a single generate work on a HEALTHY replica?) is open.
Pattern on both replicas: the first generate request fails with `prob_dist must be 1 or 2 dim` (sampling saw logits with
an extra dimension), and from then on every request fails with the device-side assert. `trace` works (13:31 smoke).
Liars' Bench itself generated its Mistral-3.1 rows through OpenRouter (`src/models.py:292-299`), not locally.

**15:12 DONE (2501 Alpaca + D + K).** Generation 2,000 / 2,000 (attempt 1 left 155 rows when one shard's session sat
RUNNING 31 min on NDIF; stopped by hand, attempt 2 finished them 14:46-14:51); 1,527 replies reached EOS, 473 hit the
500-token cap; median 250 new tokens (Liars' Bench's Mistral-3.1 Alpaca median is 250 too, own tokenizer). Spans 2,000 /
2,000 exact; read 2,000 rows x 40 layers x 5120 (14:52-15:10). `d9k/D9K_{s23,all,s3}.pt`, `d5k/D5K_{s23,all,s3}.pt`.

| half depth | sd(A@D9) | sd(A@K) | ratio | cos(D9+K, K) |
|---|---|---|---|---|
| Llama-3.3 L40 | | | 1.03 | 0.76 |
| Mistral-2501 L20, s23 | 0.4994 | 0.2279 | 2.19 | 0.9174 |
| Qwen-2.5 L40, s23_v2 | 6.6783 | 1.6909 | 3.95 | 0.9712 |

D9 + K leaning to K is not a Qwen quirk: 2501 sits between Llama and Qwen. (Llama values from the 2026-10-07 entries
above; Mistral and Qwen from this session's d9k.py output.) Task XModel-mistral2501-alpaca disabled.

## 2026-10-08 15:30 — Gemma-3-27B-it port, and the vision-language generate bug FOUND and FIXED

**Root cause (measured on NDIF, Gemma-3).** A `print()` inside a remote greedy generate showed decoder layer 0 input
(1, 22, 5376) -- normal -- but `lm_head` output **(1, 1, 22, 262208)**: logits for every position with an extra dimension.
Greedy then fails at the token append ("Tensors must have same number of dimensions: got 2 and 3", job
`6f293797-5b3d-47e0-be14-b5e98871e533`; same with dict input, `c10adff0-368b-49f0-b5bf-3170e9dbe7a7`); sampling fails in
torch.multinomial ("prob_dist must be 1 or 2 dim") -- Mistral-3.1's 13:33 error -- and on Mistral the decode steps also
hit a CUDA device-side assert that left the replica unusable. Not reproducible locally (nnsight 0.7 / transformers 5.8 /
torch 2.6 CPU, tiny random models): the server path differs.
**Fix:** pass `logits_to_keep=1` to generate (what transformers sets by itself for text models). With it: lm_head
(1, 1, 262208), greedy b=1 OK (`847ae21c-c595-4553-a840-03a924d61e2d`, "Red."), sampled b=2 with the chain's kwargs OK,
a session of two generates OK, a session of two traces OK. In `1_generate_ndif.py` as `_vl_generate_fix(llm)`: applied
only to vision-language wrappers ({} for Llama / Qwen / 2501, checked), so the text-model runs are unchanged.

**Gemma port (offline checks first).**
- Loader: nnsight 0.7's LanguageModel refuses gemma3 (it is in the image-text-to-text mapping) -> VisionLanguageModel with
  the tokenizer stand-in, like Mistral-3.1 (`ndif_common.load_language_model`; the old "gemma" exclusion was a wrong guess).
  62 layers x 5376, decoder `model.language_model.layers`; key `VisionLanguageModel:{google/gemma-3-27b-it}`.
- Template: no system role -- the system text is folded into the first user turn ("<sys>\n\n<question>"); the reply ends
  with `<end_of_turn>` (106, also a stop token). generation_config: do_sample, top_k 64, top_p 0.95 -- overridden by the
  chain's 0.7 / 0.9 / top_k 0 / rep 1.0, as for the other models.
- Persona spans: 5,400 / 5,400 exact on Gemma's tokenizer (Qwen's conversations as stand-in text; `\n` before the span,
  `<end_of_turn>` after).
- Liars' Bench eval spans: alpaca 2,000 / 2,000 and soft-trigger 3 x 2,000 exact. ID / CG / IT first MISMATCHED 2,373 /
  615 / 1,623 rows: Gemma's template trims content, many stored Gemma replies start with "\n", and the encoder's offset
  search then ran the span on through `<end_of_turn>`. Fixed in `8_coords_ndif.build_eval_items` (strip the reply for the
  search, Gemma only -- every other model's spans byte-identical): now 0 mismatches (the 4,611 "whitespace-only" rows are
  exactly the stripped reply, ending before `<end_of_turn>`).
- `run_chain.ps1` / `register_task.ps1`: tag `gemma3` (folder `gemma-3-27b-it`, short "Gemma", csv `gemma-3-27b-it`,
  ReportLayer 31 = half depth). Smoke: cold deploy + trace 2 m 39 s (`42fffefe-...`), layer-31 mean finite, next "Red".
- **Run started 15:29:37** (task `XModel-gemma3`, persona chain, `-SkipEval`).
- K for Gemma not started -- two owner decisions: (1) its HP-KR system text (Liars' Bench has no Gemma HP-KR rows),
  (2) k_run's verify generates (needs `logits_to_keep=1` in its VLM generate copy).

**15:44 — K for Gemma set up (owner: "use no system prompt for gemma K").** `k_run.py` key `gemma-3-27b` (62 layers,
NDIF `google/gemma-3-27b-it`). Every other model's K uses its own default system prompt (Liars' Bench HP-KR rows); Gemma
has no HP-KR rows and no default system prompt, so its reads carry NO system message: `no_system_message()` omits an
empty system message in the reads, as zami's own generation path already does (`steer_lies.py:304`) -- otherwise
TokenizedDataset folds "" + "\n\n" into the user turn, which Gemma's template then trims. Stop tokens `<end_of_turn>`,
`<eos>` added for Gemma; the vision-language greedy generate in verify got `logits_to_keep=1`. Offline (no NDIF), both item
sets: 240 rows, no system message, prompt `<bos><start_of_turn>user\nConsider the following question: ...`; answer spans
240 / 240 exact (7 and 11 tokens); zami's identical-answers check passes; verify's prompt ends open
(`<start_of_turn>model\n`). Task `XModel-gemma3-K` registered 15:44, waiting at its gate for the persona chain's DONE.

## 2026-10-08 21:41 — Gemma-3-27B-it persona run DONE; D5-D9; K started

Generation 5,400 / 5,400 (15:29-20:12, 4 attempts: NDIF jobs stuck RUNNING/DISPATCHED 30-46 min three times and one
engineio b'P' socket error -- shards stopped by hand, then by the watcher's 20-min stall rule; every attempt resumed, no
reply lost); 0 crash signatures with the `logits_to_keep=1` fix. Judge (Llama-3.1-70B) 5,250 scored: **3 = 5,049 (96%)**,
2 = 199, 0 = 2 -- Gemma plays the roles almost always fully (Mistral-2501 63%, Qwen 28%); every role >= 20 score-3.
Banks vectors_s23 / all / s3 36 / 36 each (62 x 5376); axes for all three; Drive final copy exit 0 (21:41).

**Geometry (logs/axes.log, s23, L31 = half depth):** cos(PC1, assistant) +0.255 at L31 but median +0.824 across the 62
layers (min +0.026) -- L31 is a low point, not Gemma's typical value; PC1 63.5% of role variance. Per-behaviour PCs:
deception cos(PC1, contrast) 0.937 / cos(PC1, assistant) -0.117; sycophancy 0.983 / -0.204; unified 0.667 / -0.242.
Gemma's residual norms are ~100x the others' (|D5| 1,782 at L40) -- compare models only on unit vectors / cosines.

**D5-D9** (`d_contrasts.py --only D5,D6,D7,D8,D9`, banks s23 / all / s3): D5 / D6 / D7 equal the pipeline's deception /
sycophancy / unified axes (relative <= 4.4e-06). At L31-ish the printed report layers are Llama's (20/36/40/44); e.g. L40
s23 cos(D5, D9) 0.877, cos(D5, D8) 0.361. Drive d5-d9: 0 differences.

**K (task XModel-gemma3-K, gate opened 21:42):** verify 69 / 70 known correct (7 x 8 -> "A", replaced from the reserve,
as on Mistral-2501); unknowables answered "No." 55 / 60 and "Yes." 5 / 60 (diagnostic only -- K uses the forced denials).
**K DONE 21:56** (`K/gemma-3-27b/{first,matched}/K.npz`, 62 layers): split-half at L31 0.9918 (median 0.9813, min 0.9044);
matched split-half L31 0.9966 (median 0.9958); **cos(K, K') L31 0.9746** (L20 0.9487, L40 0.9377; median 0.9479, min
0.7280 at L25). Task XModel-gemma3-K disabled. Alpaca read (Gemma's own 2,000 Liars' Bench rows, all 62 layers) started
21:56 as task `XModel-gemma3-alpaca` (`register_alpaca_task.ps1 -Read gemma3`); D5 + K / D9 + K follow by hand.

| model (half depth) | K split-half | cos(K, K') |
|---|---|---|
| Llama-3.3 (zami, L40) | 0.9986 | 0.993 |
| Qwen-2.5 (L40) | 0.9824 | 0.953 |
| Mistral-2501 (L20) | 0.9937 | 0.985 |
| Gemma-3 (L31) | 0.9918 | 0.9746 |

**22:41-22:50 — Gemma's Alpaca read overflowed fp16 (layers 45-61).** The read (`eval/alpaca`, 2,000 / 2,000 rows, 62 x
5376) stores pooled means in float16, cast on NDIF to halve the download. Gemma-3's two outlier ("massive activation")
dims, **104 and 2733**, exceed fp16's 65,504 from layer 45 on: 23,901 of 666,624,000 stored values are inf -- rows with an
inf per layer 0 up to L44, then 4, 25, 64, ... and 2,000 at L56-L60 (608 at L61). Layers 0-44 are entirely finite.
`d9k.py` refused it ("non-finite pooled values"), as designed. The persona vectors and K were never affected (their reads
keep float32; every bank passed the finiteness checks). Measured with float32 on 8 rows: |value| up to 123,190 in those dims.
**Fix:** `8_coords_ndif.py --store_dtype float32` (default float16, so every earlier read is unchanged; checked on 8 rows:
float32, finite at all 62 layers). `alpaca_chain.ps1 -EvalDir eval_fp32 -StoreDtype float32 -StatusName
alpaca_fp32.status` re-reads Gemma's Alpaca into `eval_fp32/alpaca` (`register_alpaca_task.ps1 -Read gemma3`); the fp16 store
stays (valid for L0-L44). Any later Gemma Liars' Bench read must pass --store_dtype float32 too.

**23:50 — Gemma fp32 Alpaca read complete (38 chunks, 2,000 rows, float32, finite) and D9 + K / D5 + K built** for banks
s23 / all / s3 (`d9k/`, `d5k/`, from `eval_fp32/alpaca`). At **L31** (half depth), s23: sd(A@D9) 233.54, sd(A@K) 672.93,
ratio **0.35**, cos(D9+K, K) **0.403** -- D9 + K leans to D9 here, the opposite of Qwen (3.95 / 0.971) and Mistral-2501
(2.19 / 0.917). D5 + K at L31: ratio 2.05, cos(D5+K, K) 0.898. But Gemma's balance swings with depth: L40 ratio 5.92,
cos(D9+K, K) 0.987; L36 0.81 / 0.661 (d9k.py output). L31 is also where cos(PC1, assistant) is low (+0.255, median
+0.824). On Gemma, compare D9 + K's lean across all layers, not one. D5 (s23) == D5 (all) here: every deception/honesty
role keeps all 150 replies in s23 (96% score 3).

**2026-10-09 00:02 — Mistral-3.1 Alpaca + D + K queued (owner: "queue the mistral 3.1 alpaca read after K").** Task
`XModel-mistral31-alpaca` (`register_alpaca_task.ps1 -Read mistral31`) runs `alpaca_gen_chain.ps1` with a new `-WaitFor`
gate on `K\k_chain.status` (waiting since 00:02:22). Then: generation skipped (Liars' Bench has Mistral-3.1's 2,000 Alpaca
rows; the run folder's `alpaca.csv` md5-equals the repo's), span check, all-layer read (fp16 -- Mistral's activations fit),
D5-D9 for every bank (new stage-5 step: built when missing, checked against the pipeline's axes), D5 + K / D9 + K, Drive.
Judge shard 2 hit the engineio b'P' socket error at 23:52 and was stopped by hand; its roles go to the judge's attempt 2.

**00:22 — evaluation queue CANCELLED (owner: "skip the evaluation ... just focus on completing the persona side").** Task
`XModel-eval-queue` (`eval_queue.ps1`, Liars' Bench reads at every 4th layer for Mistral-3.1 / Qwen / Gemma) is disabled;
nothing was read. On its first start its gate opened at once -- `-match 'DONE'` matched the word inside the chain's own
"waiting for ... to end in DONE" line -- and it ran Mistral-3.1's tokenizer-only `--prepare_only` (no NDIF request) before
it was stopped by hand; its partial `eval_lb\` was removed. Both gates now match a line ending in "  DONE" (`eval_queue.ps1`,
`alpaca_gen_chain.ps1`). Still queued for Mistral-3.1, the persona / lie-direction side only: persona chain (judge attempt 2
from 00:19) -> K -> Alpaca read -> D5-D9 -> D5 + K / D9 + K.

## 2026-10-09 01:59 — Mistral-Small-3.1-24B-2503 persona side COMPLETE

Generation 5,400 / 5,400 (22:25-23:31, first full run with `logits_to_keep=1`; 0 crash signatures). Judge (Llama-3.1-70B)
5,250 scored: 3 = 3,581 (68%), 2 = 1,607, 1 = 10, 0 = 52; every role >= 20 score-3 (3 attempts: engineio b'P' socket errors
on two shards and one 21-min stuck job, all stopped by hand or by the watcher's stall rule; nothing scored was lost). Banks
vectors_s23 / all / s3 36 / 36 each (40 x 5120); axes; persona chain DONE **01:11:25**. Geometry (s23, L20): cos(PC1,
assistant) +0.573 (median +0.551 across layers).
K (gate opened 01:12, DONE 01:30): verify 70 / 70 known correct; unknowables declined 60 / 60 ("I do not have the
information", 48; "I do not know the answer to the question", 12). Split-half L20 0.9966 (median 0.9940); matched 0.9967;
cos(K, K') L20 0.9850 (median 0.9831, min 0.8203 at L0).
Alpaca chain (gate opened on K's DONE, 01:30; done 01:59): Liars' Bench's 2,000 Mistral-3.1 Alpaca rows, spans 2,000 / 2,000
exact, read at 40 layers (fp16; finite); D5-D9 for s23 / all / s3 (D5 / D6 / D7 equal the pipeline's axes, relative
<= 3.3e-06); D5 + K and D9 + K for all three banks. L20 s23: sd(A@D9) 0.3461, sd(A@K) 0.1364 (ratio 2.54), cos(D9+K, K)
0.9351. All Mistral-3.1 tasks disabled.
**Watchdog bug fixed:** `run_chain.ps1` wrote "SKIP already DONE" when re-entered after DONE, so that note became the last
line and the next trigger (01:56) re-ran the whole chain -- every stage skipped, axes rebuilt identically, a second "DONE
02:01:12". The skip now exits without writing.

| model (half depth) | judge score-3 | cos(PC1, asst) | K split-half | cos(K, K') | D9 : K Alpaca spread | cos(D9+K, K) |
|---|---|---|---|---|---|---|
| Llama-3.3 (L40) | -- | +0.620 | 0.9986 | 0.993 | 1.03 | 0.76 |
| Qwen-2.5 (L40) | 28% | +0.838 | 0.9824 | 0.953 | 3.95 | 0.971 |
| Mistral-2501 (L20) | 63% | +0.575 | 0.9937 | 0.985 | 2.19 | 0.917 |
| Mistral-3.1 (L20) | 68% | +0.573 | 0.9966 | 0.985 | 2.54 | 0.935 |
| Gemma-3 (L31) | 96% | +0.255 (median +0.824) | 0.9918 | 0.975 | 0.35 | 0.403 (L40: 0.987) |

## 2026-10-09 02:23 — Llama-3.1-405B-Instruct persona side started (owner: "Llama-3.1-405B", top of the scale ladder)

Text `LlamaForCausalLM`, 126 x 16384, HOT on NDIF; same chat-template family as Llama-3.3 (static "26 Jul 2024", no clock);
generation_config temperature 0.6 / top_p 0.9 (the chain overrides with 0.7 / 0.9 / top_k 0 / rep 1.0, as for every model);
no Liars' Bench rows. Tag `llama405` (folder `Llama-3.1-405B-Instruct`, short "Llama", csv "none", ReportLayer 63).
Checks before NDIF: persona spans 5,400 / 5,400 exact on its tokenizer (after "\n\n", before `<|eot_id|>`). Smoke: trace
L63 finite, next "Blue" (5 s); sampled generate b=2 "Blue." / "Paris." (4 s).
**K** (key `llama-405b-3.1`, 126 layers, `-TokenBudget 512` -- 2048 would return ~240 MB per request): its HP-KR system text
is Llama-3.3's (no 405B rows; same template family, as 2501 took Mistral-3.1's). Offline: 240 / 240 forced replies read
exactly (7 and 10 tokens), identical-answers check passes, verify prompt ends open, stop tokens Llama's (the Mistral `</s>`
is now added for Mistral repos only). The K prompt renders with a doubled date header (template "26 Jul 2024" + HP-KR's
"01 Sep 2025") -- verified identical to Llama-3.3's rendering, so 405B's K reads are set up exactly as the Llama K's.
**Alpaca:** 405B generates its own 2,000 replies (`alpaca_gen.py`, its own temperature 0.6, 500 tokens), then reads every
4th layer + L63 (33 layers) in float32 -- all layers would be 8 GB in fp16 and fp16 overflow is not ruled out; `d9k.py` now
builds D + K at the store's layer subset (tests: full-layer rebuild bit-identical to Qwen's D9K; a 10-layer subset of
Mistral-3.1's store equals the full build at those layers exactly). Tasks: `XModel-llama405` (persona), `XModel-llama405-K`
(gated on the persona DONE), `XModel-llama405-alpaca` (gated on K's DONE).
**02:44 — 405B runs one request at a time.** In the first 21 min: 128 replies (one session, ~8 min RUNNING); the other
shards' jobs waited QUEUED at position 3. The watcher's 20-min stall rule stopped shard 1 while it was only QUEUED -- wrong
for 405B: QUEUED is waiting in line. Rule changed (405B watch): QUEUED never counts; RUNNING / DISPATCHED limit 45 min for
405B stages, 20 min for the judge. Shard 1's roles are regenerated by the chain's next attempt; its stopped job may still
run on NDIF (nnsight 0.7 cannot cancel a submitted job). At ~128 replies / 10 min, generation needs ~7 h; with judge,
three banks of all-layer reads, K and the Alpaca generation + read, the 405B persona side is ~18-24 h.

**03:50 — 405B run PAUSED by the owner ("stop this, we will resume later").** Tasks XModel-llama405 / -K / -alpaca
disabled, every 405B process stopped. Saved: 900 / 5,400 replies (6 of 36 roles complete),
in `%USERPROFILE%\xmodel-runs\Llama-3.1-405B-Instruct\persona\responses` (generation appends per session, so a resume
regenerates only the missing rows). Last error before the stop: shard 2 at 03:49 "TypeError: a bytes-like object is
required" in the socket thread (another engineio-side fault, like b'P'). **To resume:** re-register the three tasks (registering
starts them; the persona chain is resumable, K and the Alpaca chain wait on their gates):
  powershell -ExecutionPolicy Bypass -File xmodel\register_task.ps1 -Tag llama405 -Machine laptop
  powershell -ExecutionPolicy Bypass -File xmodel\k\register_k_task.ps1 -Model llama-405b-3.1 -Folder Llama-3.1-405B-Instruct -Name XModel-llama405-K -TokenBudget 512
  powershell -ExecutionPolicy Bypass -File xmodel\register_alpaca_task.ps1 -Read llama405
and watch with the 405B watcher (QUEUED-aware stall rule: QUEUED never counts, RUNNING/DISPATCHED 45 min).

## 2026-10-10 — Llama-3.3 kd D5 / D8 / D9 and D5 + K / D9 + K on a full-depth Alpaca read and K

Until now the Llama D + K numbers above used passB's Alpaca store (21 layers) and the 4-layer validation K. Rebuilt at
all 80 layers (owner: "recompute llama kd D9+K on the full alpaca read"), in `%USERPROFILE%\xmodel-runs\Llama-3.3-70B-Instruct\`:
- kd vectors: Drive `labpc-backup-2026-10-06/expLB/run-kd/vectors_{s23,s3}` (+ `axes_vectors_{s23,s3}.pt`), copied to
  `persona/`, rclone check 36 / 36 matching each; s3 byte-equal to the earlier local copy.
- `d_contrasts.py` (D5, D8, D9; `--axes`): groups = zami's; D5 = the pipeline's deception axis (relative 3.2e-06 s23,
  3.9e-06 s3). L40 s23 cos(D5, D8) +0.470, cos(D5, D9) +0.866 (as in the table above).
- K = `random_persona_bank/results/Llama-3.3-70B-Instruct/K/llama-70b-3.3/first/K.npz` (80 layers; split-half = zami's at
  L20/36/40/44); Alpaca = `random_persona_bank/results/Llama-3.3-70B-Instruct/eval/alpaca` (2,000 rows x 80 layers, 8_coords mean).
- `d9k.py` -> `d9k/D9K_{s23,s3}.pt`, `d5k/D5K_{s23,s3}.pt`.

| L40 | sd(A@D) | sd(A@K) | ratio | cos(D, K) | cos(D + K, K) |
|---|---|---|---|---|---|
| D9 s23 | 0.3854 | 0.3747 | 1.03 | +0.1429 | 0.7638 |
| D9 s3 (zami's bank) | 0.3456 | 0.3747 | 0.92 | +0.1503 | 0.7353 |
| D5 s23 | 0.4827 | 0.3747 | 1.29 | +0.1507 | 0.8242 |
| D5 s3 | 0.4441 | 0.3747 | 1.19 | +0.1424 | 0.8017 |

The passB-based L40 values (1.03 / 0.764 for s23, 0.92 / 0.735 for s3) are confirmed to three decimals, and cos(D9, K)
on s3 = zami's committed 0.1503. Depth matters: at L20 D9 + K leans to K (s23 ratio 2.19, cos(D9 + K, K) 0.915).
