# user_stay matched pair — the kd persona step re-run for Qwen2.5-72B AND Llama-3.3-70B in the format Qwen follows

Owner 2026-10-14, after F-130: "go with matched pair. go with your suggestion."

## Why

Under the original format (role line R as the system message) Qwen2.5-72B mostly answers as itself: 27% of replies
fully in role on the screen's 175 items, against Llama-3.3's 71%. With R plus one sentence in the **user** turn
(`user_stay`) Qwen reaches 76% and every role clears the s3 floor. But the same format lifts Llama-3.3 to 94%, so it
is not neutral (F-130, `xmodel/qwen_roleplay/`). Qwen's cross-model anomalies were measured on replies in which Qwen
mostly did not play the role: PC1 of its persona space is the assistant direction (cos +0.838 v Llama +0.620), and
D9 + K is almost K (Alpaca spread ratio 3.95 v Llama 1.03). This run asks whether those anomalies belong to Qwen, or
to Qwen not playing the roles. Running **both** models in the same new format keeps format and model apart.

## Design — only the prompt format changes

| | original (`xmodel-runs/`, lab `run-kd`) | this run (`xmodel-runs-userstay/`) |
|---|---|---|
| system message | R (omitted for the default's "" variant) | explicit **empty** |
| user message | question | R + " Stay in character throughout: answer as this person would, in their own voice." + blank line + question |
| default role, "" variant | question only (Qwen then inserts its own "You are Qwen, created by Alibaba Cloud…") | question only, under the empty system message (no built-in Qwen prompt) |
| default role, other 4 variants | "You are an AI assistant." etc. as the system message | the same line + the same sentence in the user turn (the owner's choice: one layout for every role) |

Everything else is `run_chain.ps1` unchanged: the kd bank (36 roles × 5 variants × 30 questions = 5,400 replies), sampling
T 0.7 / top-p 0.9 / top-k off / repetition penalty off / 512 tokens, judge Llama-3.1-70B with each role's own judge
prompt, vector banks s23 (main) / all / s3 over every layer, axes at the report layer (40 for both). Each model's role files are
byte-identical to its original run's. Llama-3.3's are the thesis's `kd/personas/kd_instructions`, Qwen's the
`xmodel/code` copy, renamed per model at deploy. The format lives in
`xmodel/code/src/assistant_axis_ndif/1_generate_ndif.py --prompt_format user_stay`. The default `system` path is
unchanged, so the earlier runs are not affected.

`preview.py` built all 5,400 prompts per model from the **deployed** generator. The 175 prompts per model that the fix
screen also sent are identical message lists. On Llama an explicit empty system message renders exactly as no system
message. Rendered examples are in `preview.md`.

## Comparisons, fixed before the data

Each cell is read as: original v user_stay, per model, then the two models' differences side by side.
1. **Role play:** score-3 share per role and group over all 5,400 replies; roles eligible for s3 and s23.
2. **Geometry, s23, L40:** cos(PC1, assistant), PC1's variance share, and the contamination cos(axis, assistant) of the
   deception / sycophancy / unified axes (the table in `REPORT_MODEL_PORTS.md`).
3. **Vector change:** per-role cos(v_original, v_user_stay) by layer, for each model.
4. **D5 / D8 / D9** (`xmodel/d_contrasts.py`), and **D9 + K** with each model's existing K and Alpaca read
   (`xmodel/d9k.py`): spread ratio sd(A·D9) / sd(A·K) and cos(D9 + K, K).

**Reading:** if Qwen's anomalies (2, 4) move toward Llama's under user_stay while Llama's barely move, they were a
role-play failure. If they stay while Qwen plays the roles, they are Qwen's own. If both models move together, the
format itself shapes the geometry, and that has to be reported for the thesis model too.

## Files

| File | What |
|---|---|
| `preview.py` → `preview.md` | every prompt built from the deployed generator; checks against the screen; rendered examples |
| `spans_check.py` | offline: every pooled span equals the stored reply exactly (tokenizer only) |
| `smoke.sh` → `smoke/` | per model: default + pc_gatekeeper × 5 variants × 2 questions → judge → spans → activations (1 layer) |
| `pair_chain.ps1` | the full run: `run_chain.ps1 -PromptFormat user_stay -SkipEval -NoSync -SessionTimeout 5400`, Qwen then Llama-3.3; `pair.status` |
| `register_pair.ps1` | registers and starts Scheduled Task `XModel-userstay-pair` (30-min watchdog, 4 days) |

Run folders: `%USERPROFILE%\xmodel-runs-userstay\{Qwen2.5-72B-Instruct, Llama-3.3-70B-Instruct}\` (deployed by
`register_task.ps1 -Machine laptop -RunBase … -DeployOnly`), same layout as `xmodel/README.md` §4.

**Other changes for this run** (all backward compatible):
- `ndif_common.session_deadline`: a job that gives no result within `NDIF_SESSION_TIMEOUT` seconds is abandoned and
  retried. It wraps the sessions in `1_generate`, `2_activations` and `3_judge`; unset, nothing changes. Tested on
  NDIF with a 5 s limit: both attempts were abandoned (one while running, one while queued) and the script exited cleanly.
  Abandoned jobs still run on NDIF, so the limit is set high (5,400 s): long enough to catch a 6-hour hang (as on
  10-14), short enough not to cut a normal job.
- `run_chain.ps1`: tag `llama33`, `-PromptFormat`, `-NoSync`, `-SessionTimeout`. `register_task.ps1`: `llama33`, `-RunBase`.
- No Drive copies (`-NoSync`): the `tamara:` token expired on 10-14. Copy by hand after `rclone config reconnect tamara:`.

**Cost**, from the original Qwen run's `chain.status` (10-06/07): generation ≈ 4 h (5,250 replies in 3 h 42 m, then
the last 150 in 19 m), judge 37 min, vectors s23 / all / s3 23 / 23 / 10 min, axes 1 min. That is about 5.5 h per
model, ~11 h for the pair, one model at a time. The laptop must stay on and awake: the chain blocks idle sleep while it
runs, but closing the lid still sleeps it.

## Status

- 2026-10-14 ~11:45 deployed both run folders; role files and questions checked byte-identical to the originals;
  `preview.py` OK for both models. 11:51 smoke started.
- **12:11 smoke DONE** (`smoke/status.log`), deployed code, one job at a time. Per model, 20/20 replies, judged, spans
  20/20 equal to the reply exactly, 2/2 vectors at L40, finite:
  - Qwen: tokens either side of the span ('\n', `<|im_end|>`), i.e. the header newline is excluded.
    pc_gatekeeper 0/10 fully in role (all score 2: the voice, then "balance transparency and self-preservation").
    This is its weakest role (8/20 in the confirmation; original run 5/150), and the smoke has two questions.
  - Llama-3.3: span edges ('\n\n', `<|eot_id|>`); pc_gatekeeper 10/10.
  - NDIF evicted Llama-3.3's replica twice at 12:02; the retries recovered both.
  - Qwen smoke vectors against its original ones, L40: cos 0.976 (default), 0.979 (pc_gatekeeper, vs `vectors_all_v2`);
    norms 92 / 90 v 83 / 82.
  - Before the smoke, the 5 s deadline test above. Its two abandoned jobs were still in Qwen's queue when the smoke
    started (QUEUED, position 2).
  Full run (`register_pair.ps1`) waits for the owner's go-ahead on `preview.md`.
- **13:01 FULL RUN STARTED** (owner "yes, start the full run"): `register_pair.ps1` → task `XModel-userstay-pair`
  (S4U refused → Interactive; LastRunTime 13:01:17, state Running). Qwen chain START 13:01:19 with `format=user_stay
  noSync=True sessionTimeout=5400`, inputs 36/30, preflight Qwen HOT unpinned and the judge HOT pinned. Generate
  attempt 1 at 13:01:59 on 3 shards × 12 roles, each logging `prompt format: user_stay`. Watch
  `%USERPROFILE%\xmodel-runs-userstay\pair.status` and `<Model>\logs\chain.status`.
- 15:36 Qwen 3,556 / 5,400 replies (24 role files). Shard 1's NDIF socket dropped at 14:32:02 ("packet queue is empty,
  aborting", `logs/generate.1.err.log`). Its job `4621d6e3…` (pc_feigned_novice, first session) then showed RUNNING with
  no result for 1 h 30 m. At **15:49:49 the session deadline abandoned it** ("SessionTimeout: no result in 5400s",
  attempt 1/6) and the retry job was running 30 s later. This is the first time the deadline fired in a real run. It cost shard 1 ~75 min.
  Shard 1 still has 7 roles, so it finishes last (~17:45 at ~16 min per role); shards 0 and 2 should finish around 16:40 and 16:10.
- **20:23 Qwen DONE** (`Qwen2.5-72B-Instruct/logs/chain.status`): generate 5,400 by 19:00:39, judge 5,250 scored
  (default unjudged) by 19:30, vectors s23 / all / s3 36 / 36 / 36 (every role eligible, s3 included), axes for all three.
  Role play compared with the original run (scores/*.json, 35 judged roles × 150): score 3 = 3,623 / 5,250 = **0.690** v
  1,469 / 5,250 = 0.280; roles with ≥ 20 score-3 replies 35 v 24. Lowest: pc_deniable_expert 33 (orig 7), candid_witness 52
  (29), fact_checker 61 (17), plausible_denier 62 (31), gatekeeper 64 (5).
- 20:23 Llama-3.3 chain START; 22:37 4,350 / 5,400 replies, each shard 2–3 roles from the end.
- 23:13 Llama-3.3 generate 5,400 (no hang); 23:42 judge 5,250 scored; vectors_s23 started. Role play compared with the lab's
  run-kd scores (`xmodel/qwen_roleplay/ref_llama33_kd/scores`, md5-checked copy): score 3 = 4,915 / 5,250 = **0.936** v
  3,599 / 5,250 = 0.686; all 35 roles ≥ 20 in both. Lowest: candid_witness 102 (orig 64), calibrated_reporter 110 (83).
  So across the full bank, Qwen with user_stay (0.690) plays the roles about as often as Llama-3.3 did in the original
  format (0.686), and the gap within the matched pair is 0.690 v 0.936.
- **10-15 Llama-3.3 vectors_s3 stalled ~20 h.** The laptop went into Modern Standby at 02:49:45 (System log, Kernel-Power
  506), 30 s after shards 0 and 1 submitted their jobs (pc_fact_checker, pc_documentarian), and stayed down until 22:32
  apart from two brief wakes (04:50, 05:50). The session deadline did not fire because a Windows timer wait does not
  count time asleep: only ~35 min of the 5,400 s had elapsed awake. Not a code bug. Shard 2 carried on after the wake
  (20 / 36). 23:04 shards 0 and 1 stopped by hand; chain `vectors_s3 attempt 2 (have 20 / 36)`.
- 23:0x Llama-3.3 s23 axes built early (`logs/axes_s23_early.log`), same script and arguments as the chain, which rebuilds
  them at the end. Drive: new folder `tamara:expLB/xmodel_userstay_2026-10-15/` (Qwen copied from 23:06; Llama after DONE).
- **Comparison 2, s23, layer 40 only** (all four from `6_axes.py`; logs in `geometry/`; originals = Qwen
  `xmodel-runs\Qwen2.5-72B-Instruct\persona\vectors_s23_v2`, Llama the lab run-kd s23 copy in `xmodel-runs\Llama-3.3-70B-Instruct`):

  | s23, L40 | Qwen orig | Qwen user_stay | Llama orig | Llama user_stay |
  |---|---|---|---|---|
  | cos(PC1, assistant) | +0.839 | +0.722 | +0.620 | +0.366 |
  | same, median over 80 layers | +0.838 | +0.730 | +0.610 | +0.486 |
  | PC1 share of role variance | 51.3% | 41.3% | 26.8% | 23.8% |
  | deception PC1 · assistant | −0.787 | −0.570 | −0.502 | −0.406 |
  | sycophancy PC1 · assistant | −0.780 | −0.563 | −0.365 | −0.100 |
  | unified PC1 · assistant | −0.810 | −0.679 | −0.552 | −0.270 |
  | knowledge_denial PC1 · assistant | −0.746 | −0.726 | −0.571 | −0.532 |

  First reading (one layer; the layer sweep and comparisons 3–4 are not done): both models move the same way, so the
  format itself shapes the geometry; and in the matched format Qwen is still far more assistant-dominated than Llama
  (cos(PC1, assistant) gap 0.219 orig, 0.356 user_stay). So Qwen's anomaly is not only a role-play failure. Llama's
  user_stay PC1 is no longer close to the assistant axis (+0.366 at L40).
