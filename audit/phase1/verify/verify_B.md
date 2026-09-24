# Verify batch B — notes

Ten items: four `experiments` rows, three `anomalies` rows, three `session-printed` F-numbers.
Verdicts and numbers are in `verify_B.csv`; this file holds what I had to guess, what is ambiguous
in the claims themselves, and problems I noticed while working (not chased).

I did **not** open `audit/phase1/fmap_all.csv`, `audit/experiments.csv`,
`audit/phase1/anomalies.csv` or `audit/phase1/sessions/*`. The only Phase-1 text I used is the
single `sample.csv` line per item (which is truncated at ~60-100 characters), plus the census files
under `audit/phase1/labpc/`, which are file facts. I re-derived every number from raw files.

Scripts written for this batch (all read-only on owner data):

| script | what it recomputes |
|---|---|
| `audit/scripts/p2v_B_F61_banks.py` | bank composition from the rollout JSONL themselves (F-61) |
| `audit/scripts/p2v_B_F58_rungs.py` | the three-rung off-diagonal table from the matrix JSONs (F-58) |
| `audit/scripts/p2v_B_CV07_gitlab.py` | git-vs-lab record/test counts for the kd11 JSONs (CV-07) |
| `audit/scripts/p2v_B_misc.py` | C08, O72, O12, OR-04, OR-05, F-22 file/line/doc facts |
| `audit/scripts/p2v_B_AN4_loose.ps1` | loose lab-PC scripts: dates + write-pattern classification (AN4) |
| `audit/scripts/p2v_B_O72_8b.ps1` | 8B rollout row counts on the lab (O72) |
| `audit/scripts/p2v_B_pull.py` | tar-over-ssh pull of result files into `audit/cache/`, md5-checked |
| `audit/scripts/p2v_B_writecsv.py` | formats the verdicts into `verify_B.csv` |

Lab writes: none. The two `.ps1` were sent through `labpc_run.py` as encoded commands (under the
upload threshold), so nothing was written to `audit_scratch`. 17 result files were **read** from the
lab into `audit/cache/labpc/` (3B/8B/kd11 matrices, two run-coords JSONs); each is md5-MATCH against
the lab-side `Get-FileHash` and logged in `audit/cache/MANIFEST.csv`.

---

## Where Phase 1 looks wrong

**1. F-58's 3B and 8B rows do not come from the files their source line names — and cannot be
recomputed from anything that still exists.** (HIGH; this is the scale story in the draft.)

The ledger's 70B row reproduces to four decimals from
`run-crossgen/kd11/matrices/mean_L40_s42.json` restricted to the five datasets
(raw 0.8090, `pc:unified@2` +0.0262, random −0.2138, `datapca@2` −0.1120).
The 3B and 8B rows do not:

| rung | ledger raw | file raw | ledger gain | best gain in the file |
|---|---:|---:|---:|---:|
| 3B L16 | 0.708 | **0.7619** | `axes` +0.089 | `pc:unified@2` **+0.0330** (`axes` +0.0195) |
| 8B L16 | 0.743 | **0.7980** | `pc:deception` +0.086 | `pc:deception` **+0.0125** |
| 70B L40 | 0.809 | 0.8090 | `pc:unified@2` +0.026 | +0.0262 |

The logs say why. `logs/crossgen_scale_3b.log` and `..._8b.log` each contain **two** runs with
different dataset sets and the same output filenames:

```
8B  22:19:04  datasets overridden: 5 -> instructed-deception, harm-pressure-choice, convincing-game, ai_liar, roleplaying
8B  23:38:34  datasets overridden: 5 -> instructed-deception, mask,                 convincing-game, ai_liar, roleplaying
3B  22:20:27  ... harm-pressure-choice ...
3B  23:39:46  ... mask ...
```

22:20:25 → 23:38:34 is **78 minutes 9 seconds**, so Phase 1's "re-run 78 minutes later with MASK"
is exact. The second run overwrote the first; Drive holds the same post-mask md5s
(`crossgen/scale_3b/matrices/mean_L16_s42.json` = lab `4c5f5b00…`), so no earlier copy survives
and the ledger's 3B/8B numbers are permanently unrecomputable. I searched every surviving 3B/8B
matrix (7 layers × 2 poolings × 3 probes, plus all five 4-dataset subsets) and nothing yields
(0.708, +0.089) or (0.743, +0.086).

Two consequences worth flagging for Phase 2:
- F-58's own caveat cites `harm-pressure-choice`'s 5:1 imbalance, which only makes sense if that
  dataset is *in* the five — i.e. the 3B/8B rows are from the first (pre-mask) run — while the 70B
  row matches the second (mask) set exactly. With the harm-pressure-choice set the 70B row would
  read raw 0.7513 and +0.1275, so **the published table is not like-for-like across rungs**.
- The headline shape survives qualitatively (raw rises with size, the gain shrinks: 0.7619 →
  0.7980 → 0.8090 and +0.033 → +0.013 → +0.026 on the current files), but the *numbers* in the
  ledger, and the "+0.09 replication" claim, do not hold on any file that exists.

**2. F-61's paper16 bank row is wrong and contradicts F-18 in the same ledger.** Counted from the
26 rollout files: 16 roles have 50 rows / 10 questions, 10 roles (the sycophancy and
anti-sycophancy family) have 60 rows / 12 questions; 22 distinct questions in total;
16×50 + 10×60 = 1,400. The ledger's uniform "10 questions, 50 rollouts/role" is exactly what
`default.jsonl` (the first file alphabetically) shows — Phase 1's "FIRST role file only" diagnosis
is right — and it is self-inconsistent, because 26 × 50 = 1,300, not the 1,400 in its own total.
F-18 states the correct split ("per-group (10 deception / 12 sycophancy)", 1,400 rollouts).
The kd row is exactly right (36 × 150, 30 questions, 5 variants, 5,400).

I also tested F-61's second confound claim, that the two question sets are "*different questions*,
not a superset": **20 of the 22 paper16 questions are among the kd 30**, including all 10 of the
default/deception-role questions; only 2 of the 12 sycophancy questions are missing. So the
question-set change is much closer to an extension than the entry says. That weakens confound (2)
and strengthens (3) (rollouts 50/60 → 150), which F-61 already calls the likeliest explanation.

**3. AN4's count is low.** 44 loose root-level `.py` on the lab have mtimes in 2026-09-19…09-21,
and **37** of them contain no write pattern (print-only), not "~25". The named examples
(`ortho.py`, `perm.py`, `cmp_banks.py`) all exist and are print-only, so only the count differs.

---

## What I had to guess

- **"print-only" has no definition in the claim.** I used: the file contains no
  `open(...,'w'/'a'/'x')`, `to_csv`, `to_json`, `json.dump`, `savefig`, `torch.save`, `np.save`,
  `write_text`, `.write(`, `makedirs`, `mkdir(`. Seven of the 44 files trip it
  (`make_view.py`, `dedupe_broker.py`, `make_view_ac.py`, `gen_hpkr_denial.py`, `cap_personas.py`,
  `6_axes.py`, `jchk.py`) — those are generators, not analyses, which is consistent with the claim's
  spirit even though the count is not.
- **"loose" and "analysis" are both undefined.** I read "loose" as *directly under
  `C:\Users\user4\expLB`, not in a package subfolder*: 81 `.py` and 46 `.ps1` over all dates. If
  Phase 1 counted some narrower notion of "analysis script", its ~25 may be defensible but is not
  testable from files.
- **CV-07's item name is truncated in `sample.csv`** ("…and kd11"). I tested the passA pair *and*
  `kd11/banks/kd_s23`, because the second stated pair (216 / 468) matches kd_s23 exactly. If the
  intended second file was a different one, that half of my check is against the wrong pair — but
  the numbers match, so it is almost certainly right.
- **OR-04's "reported at L40 only"** is a statement about prose, and prose is fuzzy. I measured two
  concrete things: the thesis doc's bank comparison (§15.2) is explicitly "layer 40, mean pooling",
  and across the five main docs, 39 lines mention these banks and **none** names a seed. That makes
  the seed half of the claim airtight and the layer half true-for-that-table.
- **F-58's "same 5 datasets"**: the ledger does not list them. I took them from the 3B/8B matrix
  files' own `datasets` field and restricted kd11 to those names.

## Ambiguities in the claims themselves

- **F-22's `source:` line points at a line number that no longer holds the bug.** At HEAD line 145
  is the *fix's* comment, which quotes `str(e)[:200]`; the bug was live code at line **146** in
  commits `107879d` (09-11) and `106d09f` (09-15). A reader checking the cited line today sees the
  fix and could conclude the finding is wrong. Verdict recorded as CONFIRMED-DIFFERENT-SOURCE.
- **F-58 says "layer 16" for both small rungs and "the paper's analysis layer"**, but
  `run_scale_chain.ps1:53` sets `$MID` to 14 for 3B and 16 for 8B. The 3B matrices are only written
  at layers 0/4/8/12/16/20/24, so L16 is the nearest available, not the script's own midpoint.
- **O12's "whole corpus"** means the 10-dataset passB corpus, not the 11-dataset kd11 corpus, and
  only 3 of the 7 passB banks (`hf275_strict`, `own100`, `paper16_shared`) were extended to the
  fine layers. Both are worth stating in the experiments row.

## New problems noticed (not chased)

1. **The `str(e)[:200]` sibling inside F-22's own file.** `1_generate_ndif.py` has two exception
   handlers. The session handler was fixed; `generate_batch()` at **line 186 of HEAD** still logs
   `str(e)[:200]`. F-22's moral is "when a failure mode is fixed, grep the siblings" — and the
   nearest sibling, 40 lines below it in the same file, was missed. `3_judge_ndif.py:168,188` too.
2. **A thesis-doc claim contradicted by a file that exists.**
   `docs/thesis/persona-coordinates-70b-results.md:492` records the length-only result as
   "**log only**" and line 1060 lists it under "In logs only; no result JSON holds them |
   Regenerate into JSON". `run-coords/confounds_L44_last.json` (09-16 19:36:45) holds exactly those
   per-dataset length AUROCs. Nothing needs regenerating; it needs finding.
3. **"9 of 10 datasets" may be 8 of 10.** `confounds_L44_last.json` gives
   `median_pos > median_neg` in **8** of 10 datasets (`roleplaying` 31 vs 38 is shorter;
   `harm-pressure-choice` is 1.0 vs 1.0, a tie). `docs/ONBOARDING.md:78` and the auto-memory note
   "Response length transfers on its own" both say 9 of 10. The mean within-dataset length AUROC in
   the file is 0.6137, against the 0.553 (pairwise) / 0.617 (LODO) figures the docs quote — 0.617
   is close to my 0.6137 but not equal, so the docs' two numbers may be measuring two different
   things under one label.
4. **`run-scale/8b/scores` has 25 files for 26 roles** — `default` has no score file. Benign if
   intended (the anchor role is not judged), but it should be stated wherever the 8B rung's
   filtering is described, since `vectors_s3`/`s23` are judge-filtered.
5. **`run-crossgen/scale_70b_paper16fp32/` exists on the lab with mtime 2026-09-23 00:16** — i.e.
   created *during* this audit. It is presumably the lead's RF-16 recompute; noting it so nobody
   later mistakes it for an owner run.
6. **`kd/docs/FINDINGS.md` was modified during this batch** (working-tree mtime 2026-09-23
   02:54:41, one line changed at ~5254, the RF-16 / F-100 "best" definition). Not by me — this
   agent only read the file, with `sed -n` and `git grep`. `git status` was "(clean)" at the start
   of my run and now shows `M kd/docs/FINDINGS.md`, so START_STATE no longer matches. Flagging it
   because the BRIEF forbids edits outside `audit/`; the edited line is far from the F-22 / F-58 /
   F-61 entries I quoted, so nothing in this batch depends on it.
