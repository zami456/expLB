# NARCBench gates

## G1 row audit: PASSED with findings (2026-10-20 19:15; details in g1_rows.md / g1_rows.json)

- Their matchers, ported in `rows.py` (5 unit tests pass), rebuild rows that align with every released metadata
  file: Qwen3-32B core 1054/1054, transfer 288/288 (final rows only), stego 1320/1320; Llama-3.1-70B core
  1054/1054 and stego 1320/1320 in order; Llama transfer 864 of 936 matched, order differs.
- Token counts (`n_tokens`, their tokenisation): **Qwen3 Stego 1320/1320 and Qwen3 Transfer 288/288 exact**, so
  those activations are reads of the released transcripts. Qwen3 Core 5/1054: first-speaker prompts are
  consistent (implied reply length >= 0 for all 62, median 139 vs ours 138) but no reply length matches, so the
  released Core transcripts are a different sampling of the same scenarios than the one behind the released Core
  activations. Llama-3.1: 65/1054 core, 2/936 transfer, 27/1320 stego; honest first speakers' prompts match but
  replies differ (e.g. 61 vs 123 tokens): **their Llama-3.1 data are Llama's own runs of the same scenarios,
  transcripts unreleased** (corrects plan fact 1, which assumed an off-policy re-read of the Qwen3 transcripts).
- **Stego empty-final bug confirmed:** all 160 scored `phase == "final"` rows were read with an empty reply
  (exact token parity), although `final_texts` holds the final-hand replies. Their Stego score reads only the
  prompt context (for colluders: the scheme, the counts and a decode step).
- Leak scan (all runs): Stego collusion runs only: honest agents' prompts carry 105 plain-text "PRIVATE:" decoding
  lines, judge prompts 33, colluder replies 147, colluder prompts 144 PRIVATE + 160 DECODE + private blocks; none
  in control runs. Core: private blocks only in colluder prompts (300 rows). Transfer: private blocks in colluder
  (collusion) and implicit-pair prompts.
- No speaker mismatches (every positional match's [Name] prefix equals the row's agent); no
  effective_roles/roles mismatches.
- 38 Core runs (19 scenario pairs) have transcripts but no released activations; the run configs do not explain
  the exclusion (several excluded collusion runs succeeded with both colluders on target).

Consequences for the plan: Arm A analyses that need text aligned with activations (Stego, Transfer) use the
released Qwen3-32B activations; Llama-3.1 is used for replication, nulls and the role-by-stance split (its
metadata carries votes from its own runs). Arm B reads the released transcripts, so our Llama-3.3 read is
off-policy, unlike their Llama numbers; the Arm B head-to-head stays internal (same reads for both detectors).

## G2 port: PASSED (2026-10-20 ~19:50)

`rose.py` vs their own `reproduce/reproduce_LR.py` functions (imported from the pinned clone, `main()` not run),
same released activations, seeds 0-1, every group x method x layer: max |difference| = 0.0 for llama_3_1_70b
(layers 32-37) and qwen3_32b (layers 26-30). Tolerance was 0.005. sklearn 1.6.1 here vs their pinned 1.7.2,
so their published numbers may differ slightly from the replication (their probe stops at 8 iterations).

## G3: see freeze 1 (text-baseline ceiling verdicts are logged there, before any Arm B score)

## G4 conditions: PASSED (2026-10-20 19:2x; g4_conditions.md)

10,516 condition rows (16 files), 15.9M tokens; only the intended fields change (reply byte-identical in STRIP, PLANT,
BARE, OBSERVER); every colluder discussion prompt had exactly one private block; longest row 7,702 tokens (< 12,000).
Unit tests: 12 pass (rows + conditions; render span = trimmed reply, end-of-turn last, one BOS).

## G5 frozen directions: PASSED (2026-10-20 ~19:33; g5_directions.json)

`directions_llama33.npz` (sha256 8bca04d3...) reproduces the thesis's L40 AUROCs on CG / HP-KR exactly:
D9 .9123/.9124, K .6406/.8654, D9+K .8646/.9087, D5+K .8126/.8541, RepE .9723/.9358. Random directions (1,000 per
layer, seed 2026) and the 100 random persona splits (R9, R9+K, oriented on RepE as the thesis does) are in the file.

## G6 reader: PASSED (2026-10-20 19:46; g6_selftest.json, g6_smoke.json)

- Local self-test (small random Llama, remote="local", right-padded mixed batches, empty span): max |session - HF
  forward| = 3.0e-8.
- NDIF (Llama-3.3-70B): rows read alone vs in one padded batch: content-mean cos >= 0.999988, end-of-turn token
  cos >= 0.999832; the thesis's stored Alpaca rows re-read: cos >= 0.999982 at layers 20/36/40/44.
- Finding: on NDIF 0.8 `tracer.stop()` inside a multi-trace session breaks the sandbox (BrokenPipe / "not enough
  values to unpack"); single-trace stop, multi-trace without stop, and `logits_to_keep=1` all work. The reader uses
  no stop and `logits_to_keep=1`.

## Stage 2 and held-out scoring: DONE (2026-10-21)

- Stage 2 reads finished 08:06 after one relaunch (power loss 23:17, Kernel-Power 41; 90 chunks verified intact). Every
  planned row was read; no row left unread. Failed-then-retried sessions per file are in read_*.log (OOM halving worked).
- `score heldout` ran under freeze 2 (score.py sha f7ee8395 checked). Results: results/narcbench/armB_heldout.json
  (sha256 743bc3e297acafd2...), report results/narcbench/summary.md
  (rebuild byte-identical, sha256 ae811b634e9164c8...).
- Primary tests (Holm): P1 rejected (their probe's Transfer effect is context: share 1.07, CI 0.91-1.28), P4 rejected
  (D9+K above the random band: 0.66 vs 0.63, p 0.010); P2 (D9+K share 0.56, mixed), P3a (their probe under STRIP 0.54,
  p 0.27) and P3b (D9+K under STRIP 0.62, p 0.029 > 0.0167) not rejected. Estimation: D9+K minus their probe on the
  original transcripts -0.28 [-0.36, -0.20].

## Verdict (Shi et al.'s Gemma-4-31B runs, plan of 2026-10-22): gates GV1-GV5 PASSED (2026-10-22 / 23)

- GV1 rows (`gv1_verdict_rows.md`): 1,000 agent-episodes, none excluded; own messages equal the channel transcript;
  verdict turns equal the recorded verdicts; no earlier episode had to be dropped for the Llama budget (longest Llama
  row 6,668 tokens, median about 3.0k). Qwen rows through their extractor: longest 12,964 tokens (their renderer keeps
  failed tool-call attempts, whole raw logs as arguments, and the tool schemas). 190 of the 1,000 stored histories
  start after episode 1 (the harness drops the oldest episodes when Gemma's context fills): the compact context is
  built from what the agent saw.
- GV3 labels: 146/500 colluding episodes in 33/50 runs; collusion = both_accept; trace verdicts = episodes table
  (as EXPERIMENTS.md). Rose's flag: 51 episodes; relaxation: 1,000 valid judgements.
- GV2 reader (`verdict/logs/gv2_selftest.json`, `gv2_smoke_{llama,qwen}.json`, `gv2_llama_batch_diag.json`): self-test
  on tiny Llama and Qwen3 models (remote="local"; several span groups, an empty group, absent points, padding): max
  |session - HF forward| 5.6e-8 over 106 comparisons. NDIF: solo reads repeated, cos 1.0 for both models. Llama solo vs
  batched min cos 0.99960 (single-token points: PRE, VERDICT_EOT, the last own-message end, layers 36-40), under the
  0.9998 gate, while every solo re-read reproduces exactly; so every row is read alone (the reader's default) and the
  gate is the solo repeat (decided before any score; the batched figure is recorded). Qwen solo vs batched 0.99993.
  Longest rows read alone: Llama 6,668 tokens (22 s), Qwen 12,964 tokens (18 s), so the budgets 7,000 / 13,000 hold.
  Llama Alpaca parity 0.99998. AWQ (their Core) vs bf16 (our reads), 20 released Qwen3 Transfer rows, layers 26-30:
  cos median 0.9934, min 0.9914.
- GV4 / GV5 (`verdict/logs/gv4_gv5_dryrun.json`): both arms end to end on synthetic reads and labels (no Verdict read
  or label): the six primary tests finite (synthetic AUROCs 0.47-0.53, none rejected) with Holm; their unchanged script
  ran on synthetic Verdict files with their released Core activations (Core held-out 1.00 for all five methods, as in
  their Table 2); their functions with the relabelled episodes; the random band; the report rendered. A first rerun was
  stopped by Claude Code for low memory (250-row float64 synthetic chunks, 16 GB laptop); the rehearsal now writes
  25-row float32 chunks.
- Tests: the Verdict unit tests (10) and the NARCBench suite pass (31 in all).

## Qwen3-32B persona bank for D4 inside their pipeline (post hoc, 2026-10-23)

**GB1 smoke (06:02-06:07) PASSED.** Chain copy `E:\narcbench\persona\Qwen3-32B\code` = `E:\qbank\code_src` plus the
`qwen3` tag in run_chain.ps1 (Qwen/Qwen3-32B, thinking off). One role (pc_handler), instruction 0, the 30 shared
questions: 30 replies (median 313 words, none with a think tag); judge Llama-3.1-70B 27 x score 3, 3 x score 2;
per-question store at layers 26-30: 27 cells + 3 empty (the score-2 questions), each (5, 5120), finite, n = kept
count. Spans (the pipeline's build_items + ConversationEncoder, Qwen3-32B tokenizer): 30/30 decode exactly to the
reply, no <think> token inside, each ends at <|im_end|>; the empty think block sits in the context before the span,
as the model saw it when generating.

**GB5 identity check PASSED (2026-10-23 ~06:25).** `qwen_d4_pipeline.py identity`: `prepare` (their clone, their Core
activations, our four Verdict phase files, each layer_L replaced by X @ B.T) with the 5120 x 5120 identity as B, then
their script unchanged: summary CSV and raw CSV byte-identical to the unprojected run in `E:\narcbench\verdict\rose_run`
(summary sha256 85cdaccb...). The projection path changes nothing but the basis.

**Bank chain DONE 12:55.** 5,400 replies (attempts 2-3 refilled 9 dropped for a think tag), 5,250 judged (default is
not judged), s3 per-question store 1,080/1,080 cells at layers 26-30.

**GB4 bank report (`qwen_d4_pipeline.py bank`, E:\narcbench\logs\gb4_qwen3_bank.json).** Rule outcome: s3 (no role
under 20 kept replies; fewest pc_deniable_expert 24, then pc_enabler 50); the s23 store was not needed. Judge share
fully in role by group: deception .62, honesty .78, sycophancy .69, anti-sycophancy .82, knowledge denial .44, sincere
ignorance .48, plain assertion .61 (Qwen2.5 was .28 overall). Top-3 PCs explain .41-.45 / .18-.20 / .09-.10 at L26-30;
the extended axis lies ~97% inside the subspace (sum of squared PC cosines ~.94). AWQ (their Core) vs bf16 after the
projection, 20 Transfer rows: PC Pearson r .97-.99, cosine of the 3-dim vectors >= .9999. GB4 PASSED.

## Liars' Bench rows read by Qwen3-32B for the Liars' Bench-label D4 pipeline (post hoc, 2026-10-23)

**Read DONE 16:18:39.** `run_verdict.ps1 -Chain qwen_lb` (launched 14:22): `qwen_lb_five` (11,052 rows = the thesis's
pooled_L40 rows of CG, HP-C, HP-KR, IT, ID, checked label by label; sha256 bf03a1c2...) read by the frozen reader
(--model qwen --batched, budget 6000), points THEIRS and REPLY_LAST at layers 26-30; exit 0, 0 rows left.

**Recheck PASSED 17:59 (min cos 0.9999296 >= 0.9998).** The chain's recheck job failed 7 times out of memory before any
NDIF call: `cmd_recheck` takes `z["points"][i]` row by row and an NpzFile reads the whole member again on each access,
so a batched file of n rows (228-637 here) holds n copies. Chain stopped by hand 17:55; `recheck_eager.py` (declared
in freeze.txt 17:56) runs the same `cmd_recheck` of the unchanged reader (sha256 ded96e81..., = freeze V) with each
.npz read once: rows 100000 and 100001 read again one per trace, compared with the batched reads.

**Analysis DONE 18:13 (post hoc, declared 14:25).** `run_qwen_d4_lb.ps1` (WMI, launched 18:00): `qwen_d4_lb run` then
`report`, both exit 0. Check: the null's seed-0 persona values equal their main's seed-0 rows (True). Report
`results/narcbench/qwen_d4_lb.md` (sha256 45546d71...) rebuilds byte-identically from `qwen_d4_lb.json`
(sha256 7f6fa868...). Tests: test_qwen_d4_lb (3) and test_qwen_d4_pipeline (3) pass.

## Attention probe on persona-projected tokens (post hoc, 2026-10-24)

**Rows (01:52) and declaration (01:58).** `qwen_attn_rows.py build`: Core 744 public discussion rows (their released
Qwen3 Core set; text = the released transcripts, since their Qwen3 Core inputs are unreleased), reflection 1,000,
verdict 1,000 (two spans); every span decodes exactly to its reply / reflection / verdict text; 406k span tokens,
11.1M context tokens.

**GQ1 selftest PASSED** (remote='local', tiny Qwen3): span projections == HF forward @ basis, max diff 7e-8 (rerun
after each reader change: 7e-8, 7e-8).

**NDIF request limit (02:00-02:20).** 324 and 114 basis rows refused (HTTP 400 "error parsing the body"); label-blind
diagnostics on one Core row: 3 rows (0.15 MB) pass, 24 rows (1.2 MB) fail. Reader changed (freeze notes 02:11, 02:20):
only the 3 persona rows are sent; 100 random orthonormal 3-dim null subspaces per layer are built on the server from
seeds (identical to a local run, max diff 0).

**GQ2 smoke PASSED (02:21-02:22).** Last span token vs the existing single-token read @ the full 303-dim basis: min cos
0.999996 (reflection rows 120000-1), 0.999995 (verdict rows 130000-1, decision point); Core rows read twice solo 1.0,
solo vs batched 0.99991. Note: an expired watcher's `tail -F` held the chain log open, so the second launch's log lines
were lost (the chain ran regardless); watchers now poll.

**Reads DONE (02:23-06:29).** Core 744 rows (30 min, batched), reflection 1,000 (1 h 53 min, solo), verdict 1,000 (1 h 41
min, solo); no failed session; rechecks (first two rows read again, files loaded once): Core min cos 0.99992, reflection
1.0, verdict 1.0. After the 12:10 power loss: every span file loads, zip CRCs pass, all 2,744 rows present and finite.

**Analysis (declared 02:47; post-freeze edit 14:04, mechanics only).** `run` (06:29-07:31, launcher waited for the
reads): 150 probe fits on 6 workers (06:31-07:27), their pipeline per seed and H, references; saved. Its null was lost to
a power loss at 12:10 (Kernel-Power 41; reboot 13:51) in layer 29 (layers 26-28 in memory). User chose a rerun as
declared, checkpointed: `null_only` (14:05-22:13) redid the seed-0 H = 8 persona fits (weight decays and CV accuracies
equal to run's at all 5 layers) and fitted the 100-subspace null, one checkpoint per layer (92-106 min each); `report`
22:14. Checks: their training rows reproduced (their own train_deception_probe on the same metadata); pooled-vector
identity max error 2.1e-4; null seed-0 refit = run's. Report `results/narcbench/qwen_d4_attn.md` (sha256 d220cd21...)
rebuilds byte-identically from `qwen_d4_attn.json` (sha256 5aed457b...). Tests: test_attn_probe (4), test_qwen_d4_attn
(3) pass.
