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
