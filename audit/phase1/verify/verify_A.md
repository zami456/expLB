# Verification batch A — notes

Items: F-17, F-04, F-62, F-55, F-109, F-09, F-83, F-07, F-100, F-65 (the first 10 rows of
`audit/phase1/verify/sample.csv`).
Verdicts: **9 CONFIRMED, 1 DIFFERS (F-04)**. Row-level detail in `verify_A.csv`.

I did not open `fmap_all.csv`, `experiments.csv`, `anomalies.csv` or `sessions/*`. I read
`BRIEF.md`, the RED FLAGS of `PROGRESS.md` (as instructed), `kd/docs/FINDINGS.md`,
`audit/phase0/fnumber_history.csv` (line ranges only), the census files under
`audit/phase1/labpc/`, and the raw result files. All lab access was read-only; my own scripts
were uploaded to `C:/Users/user4/audit_scratch/scripts/` (the approved scratch) and logged by
`labpc_upload.py` in `audit/LABPC_WRITES.md`. My scripts are `audit/scripts/p2v_*.py`.

## Where Phase 1 / the ledger looks wrong

1. **F-04 is the one real failure.** The effect sizes reproduce (+0.0147 for `axes − raw`,
   CI [−0.1701, +0.1872], +0.0264 for `axes − datapca`), but three things do not, in any copy
   that survives:
   - test count: stated 384; git has 4,032, the lab and Drive copies have 4,788;
   - "1 is significant at dataset level after BH": **zero** tests have
     `significant_datasets = true`, in both blocks, in both versions;
   - both q-values: 0.931 → 0.963 (git) / 0.973 (lab); 0.855 → 0.896 / 0.906.
   The 384-test file exists nowhere I could find (`git log --all` shows exactly one commit of
   `run-crossgen/kd11/significance.json`, `3dfdb67`; the Drive census has exactly one copy,
   md5 `44d5b53d…`, 2,256,477 B = the lab version). Also: the cited numbers live in block `'9'`,
   which in this file lists **10** datasets, while the text says "the 11-dataset paper16 run".

2. **PROGRESS.md's F-17 flag is wrong.** The F001–F027 batch reported "Missing entirely: F-17's
   +0.056/−0.44/−0.76". They are not missing. They recompute exactly — +0.0555, −0.4378,
   corr −0.7565 — from `run-crossgen/matrices/mean_L40_s42.json` (paper probe, mean pooling,
   L40, 90 off-diagonal pairs), a file **committed in git**. Script: `audit/scripts/p2v_F17b.py`.

3. **F-100 mis-attributes one parenthetical.** "`axes` loses to raw at every layer from 44 to 68
   (−0.004 to −0.029)". `axes − raw` at those layers is −0.0221, −0.0407, −0.0508, −0.0800,
   −0.0868, −0.0998, −0.1059. The quoted range is `best − raw` (= `pc:deception − raw`) at
   exactly those layers: −0.0041, −0.0171, −0.0199, −0.0291, −0.0238, −0.0202, −0.0071. The
   qualitative claim (axes loses to raw there) is right; the numbers in the bracket are the other
   row's. Every other F-100 figure is exact.

4. **F-65's "every feature agrees to ≤ 0.004" (s23 vs unfiltered) is false for `cap_only`.**
   `pc:cap_only@2` is 0.5935 (s23) vs 0.6077 (all) — a gap of 0.0142 — and `pc:cap_only` 0.6355
   vs 0.6289 (0.0066). `cap_only` is the feature the entry's caveat (c) rests on, so the "one
   live comparison" *does* move with the filter: the cap_only penalty at L40 is −0.120 under s23
   but −0.106 under `all`. Everything else in F-65, including the headline 0.745 vs 0.374 and all
   the geometry (PC1 var 0.5676, cos(PC1, assistant) −0.8376, cos(PC1, contrast) 0.9871,
   cos(contrast_A, contrast_B) 0.9313, gate cosines 0.987/0.949/0.936), is exact.

## Sources that are not where the ledger says

- **F-17**: the cited `run-crossgen/significance.json` holds only the −0.003. The other three
  figures come from the sibling `run-crossgen/matrices/mean_L40_s42.json`.
- **F-55**: the `pc_position_holder` row of the Wolfram table is in **none** of the three
  `cap/run/probe/*.jsonl` files the source line names. It is in
  `cap/run/smoke_{cells,v4,tf}/pc_position_holder.jsonl` (identical text in all three).
- **F-09**: reproduces **only** from the lab copy of `kd/run/knowledge/scores.jsonl` (1,459
  records). The committed copy is the 204,800-byte truncation (RF-5): 1,095 records and a
  200 vs 895 split, not 312 vs 1,147.
- **F-07**: `kd/results/phase2_gate.json` is in **no branch**; only the lab copy exists, so this
  finding cannot be checked from the repo at all.

## What I had to guess

- **F-100's "best persona".** The entry says only "a post-hoc per-layer max over two features".
  I had to try feature sets. `max(axes, pc:deception)` reproduces all 17 figures to 4 dp;
  allowing `pc:unified`/`pc:unified@2` as well gives +0.1251 / +0.1086 / +0.0835 instead of
  +0.1197 / +0.1027 / +0.0791. The definition is load-bearing and only implicit.
- **F-17's "the unified persona axis … at the same k"** — I read this as `pc:unified@2` vs
  `datapca@2` (k=2). That gives −0.00279. `pc:unified` (k=5) vs `datapca` gives −0.0053 in the
  same cell, so the reading matters at the third decimal.
- **F-09's decile convention.** Under three conventions the third edge is −2.6 or −2.7, never the
  stated −2.8; the other nine agree. Not material.
- **F-04's block.** The stated numbers only exist in block `'9'`; I assumed that is what the
  entry means despite the text saying 11 datasets.

## Ambiguous in the claim itself

- **F-07's "random-direction pair floor of 0.007"** is a single number for three measured values
  (0.0079 / 0.0089 / 0.0071). It is the minimum, not the L40 value, which rounds to 0.009.
- **F-83's "the margin's own bootstrap CI width (+0.143 to +0.213)"**: NOT LOCATED. The five
  `premise_L40_mean_s*.json` files carry no bootstrap or CI key. The entry also names
  `logs/premise_extras.status`, which I did not open.
- **F-09's "6 of 8 smoke items ≈ 1.0"** points at a smoke file I did not locate. In the full file
  89.6% of margins are ≥ 0.99, which supports the same point.
- **F-55's "the TEXT is nearly identical"** is a qualitative claim; I verified the three quoted
  turn-2 strings verbatim and that all 8 no-persona items hold, but "nearly identical" itself is
  not a testable number. One quote is lightly compressed: the file says "goldfish have **been
  shown to have** a relatively good memory".

## New problems noticed (not chased)

- **A22 / RF-16, reproduced independently.** Without reading RF-16 I found that
  `run-crossgen/scale_70b/matrices/mean_L40_s42.json` carries only the zero-shot directions
  `['contrast','PC1']`, while every swept layer (e.g. L44) carries ten, including
  `contrast:knowledge_denial` and `contrast:kd_assertion` — although all 21 files record
  `bank='axes_vectors_s3.pt'`, `bank_name='paper16_70b'`. So the F-100 layer average mixes two
  banks. (This is the same fact RF-16 reached by a zero-shot fingerprint.)
- **The ledger now contains an audit-written entry.** `kd/docs/FINDINGS.md` has an **F-110**
  dated 2026-09-23, marked "Written by the audit in `audit/`, at the owner's instruction". Phase 0
  had recorded that no F-110+ existed anywhere. Anyone re-running the F-number census will now
  get a different answer.
- **A new compute folder appeared inside the owner's lab tree.**
  `C:/Users/user4/expLB/run-crossgen/scale_70b_paper16fp32/` exists with mtime 2026-09-23 00:16
  (presumably the RF-16 recompute). It is *not* in `audit_scratch`, i.e. audit output now lives
  inside the owner's result tree. I did not read it, and I flag it only so the lead can decide
  whether it should be recorded in `audit/LABPC_WRITES.md`.
- **Files behind two of my items were rewritten by the RF-13 boot wave after their entries were
  written** — `run-premise/premise_L40_mean_s{1,7,13,99}.json` (mtime 2026-09-22 15:55–15:56, entry
  dated 09-21) and the `cap_{s23,all}` matrices (15:51, entry dated 09-21). Both still reproduce
  their entries exactly, which is evidence that the wave was deterministic for per-cell numbers,
  as RF-13's update suggests.
- **Block naming is a trap.** In `run-crossgen/kd11/*`, block `'9'` contains **10** datasets and
  block `'10'` contains **11**; in the `run-crossgen` root, block `'10'` contains **10**. Any
  Phase-2 script that reads the block label as a dataset count will be wrong.
