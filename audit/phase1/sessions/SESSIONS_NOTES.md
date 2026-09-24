# Session-transcript mining — what the owner ran, and what never reached a file

**Everything in this folder is DOC-ONLY evidence.** A transcript records what a model and
the owner *said* and what commands were *issued*; it is not raw data. Numbers quoted here
are leads for Phase 2 to recompute, never verified values. Where a transcript shows the
literal text of a command, that text is quoted verbatim and is strong provenance evidence
(it is the command, not a description of it) — but its *effect* still has to be checked in
the files.

Source: `C:/Users/SURAIYA/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB/*.jsonl`
(read-only; nothing in that folder was modified).

**Time zone warning.** Transcript timestamps are **UTC** (`...Z`). The lab PC, the logs and
the file mtimes used everywhere else in this audit are **local = UTC+6**. Verified inside
the transcripts: at `09:46:13Z` the session reads `LastBootUpTime = 15:34:30` off the lab
PC. Every time in the CSVs is UTC; add 6 h to compare with `log_index.tsv` or a file mtime.

---

## 1. What was scanned

| | |
|---|---|
| files in the folder | 19 |
| skipped | `4f0b290f…` (this audit session), `1a88b065…` (its sidecar) |
| scanned | 17 |
| of those: `/clear` or `/mcp` stubs with no assistant content | 5 (`f8aefa9d`, `b884c46e`, `28e38b58`, `15ddd482`, `0454cfaa`) |
| of those: a fork that is a strict subset of another | 1 (`2394bebd` ⊂ `40b4a5dd`: all 940 of its user/assistant records match by uuid+timestamp) |
| **substantive sessions** | **11** |
| date span | 2026-09-09T15:03Z … 2026-09-22T12:07Z (first real turn 2026-09-10T06:41Z) |
| total records (17 files) | 40,814 · user turns 586 · assistant text turns 3,355 · tool calls 5,405 |
| excluding the stubs and the fork | 38,895 records · 565 user turns · 3,185 assistant turns · 5,149 tool calls |
| sidechains (subagent transcripts) | **0** in every file — no subagent work is missing |

Per-session detail: `sessions_index.tsv` (one row per file, with `dup_of`, top tools, top
`run-*` folders, top paths, lab-PC and NDIF hit counts, and the first user message).

Rough map of the 11 sessions:

| session | UTC span | what it is |
|---|---|---|
| `47aabfcb` | 09-10 06:41–06:53 | git/GitHub setup |
| `0d66bacf` | 09-10 06:55–07:18 | which machine are we on; devcontainer removed |
| `40b4a5dd` (+`2394bebd` fork) | 09-10 09:13 → 09-12 00:10 | assistant-axis NDIF pipeline built and smoke-tested; Tailscale/lab PC reached |
| `638f732c` | 09-12 04:16–19:37 | AA pipeline running on the lab PC over ssh |
| `a748e742` | 09-12 19:10 → 09-14 06:49 | the 100-role own100 run (shards, supervisor, judge batches) |
| `6d4b2181` | 09-14 06:49 → 09-17 06:10 | **the biggest session**: paper16 bank, coords passA/B/C, Q1–Q11 |
| `0e699862` | 09-14 11:43–13:34 | code-bundle dump for review |
| `dc605d01` | 09-17 06:10 → 09-18 05:22 | thesis results chapter; crossgen; bank variants; paper16-shared |
| `b5504aba` | 09-18 05:34–19:08 | Drive uploads; the collaborator's L20 share packages |
| `b035ccb3` | 09-18 19:08 → 09-21 17:11 | **the second-biggest**: kd bank, 11th dataset, cap bank, scale rungs, premise |
| `1ac8bb19` | 09-21 17:12 → 09-22 12:07 | collaborator-data mismatch, F-97…F-104, sarcasm gate, the 09-22 reboot wave |

---

## 2. How I extracted it (re-runnable, from the repo root)

All scripts stream the JSONL line by line (`json.loads` per line); none loads a file into
memory. All writes are confined to `audit/phase1/sessions/` and `audit/scripts/p1_sess_*.py`.

```bash
PYTHONIOENCODING=utf-8 python audit/scripts/p1_sess_index.py      # -> sessions_index.tsv
PYTHONIOENCODING=utf-8 python audit/scripts/p1_sess_events2.py    # -> run_events2_raw.tsv (2,946 hits)
PYTHONIOENCODING=utf-8 python audit/scripts/p1_sess_expcsv.py     # -> session_experiments.csv + session_events_all.csv
PYTHONIOENCODING=utf-8 python audit/scripts/p1_sess_datasets.py   # -> dataset_events.csv
PYTHONIOENCODING=utf-8 python audit/scripts/p1_sess_digest.py run_script   # readable digest of one kind
```

`p1_sess_events.py` is the first (superseded) version, kept because `run_events_raw.tsv`
was built from it; `p1_sess_events2.py` is the one to use.

What each pass does:

- **`p1_sess_events2.py`** classifies every `tool_use` that plausibly ran something:
  `run_script` (a real `python …/x.py …` invocation), `run_ps1` / `poll_ps1` (drivers vs
  status pollers), `run_inline` / `inline_inspect` (`python -c`, `python - <<'PY'` — split
  by whether the body calls a measurement), `run_lab` (the session's `psrun.sh` / `pyrun.sh`
  wrappers, i.e. a PowerShell here-string sent over ssh — **the python call is inside the
  here-string, so the ordinary patterns never see it**), `sched_task`, `write_script`.
  For each it keeps the matching `tool_result` (by `tool_use_id`) and the next assistant
  sentence containing a number.
- **`p1_sess_expcsv.py`** collapses repeats (same session + kind + script + the set of
  flags, `.pt` banks and `run-*` paths named) and keeps only `measure` and `run` events;
  `session_events_all.csv` is the superset including infrastructure. It matches each event
  to `audit/experiments.csv` by script basename, by `run-*` path and by scheduled-task name.
- **`p1_sess_datasets.py`** keeps only dataset-relevant events: an actual builder run, a
  Write/Edit of a persona/question/role-group file, or a sentence that states a dataset
  decision (with job-restart chatter filtered out).

Handy one-liners used while curating (paths shortened):

```bash
# every event of one kind, chronologically, with outcome
PYTHONIOENCODING=utf-8 python audit/scripts/p1_sess_digest.py run_inline | less
# the full command behind one row (session file, line number from the CSV)
python scratch/getcmd.py 1ac8bb19-….jsonl 996
# everything said in a time window
python scratch/window.py 1ac8bb19-….jsonl 2026-09-22T09:25 2026-09-22T10:40
```
(`getcmd.py` / `window.py` are 15-line throwaways in the session scratchpad; both just
stream the file and print the record at a line, or all records in a UTC window.)

---

## 3. What came out

| file | rows | contents |
|---|---|---|
| `sessions_index.tsv` | 17 | one row per transcript |
| `session_experiments.csv` | **359** | test/run events: 273 `measure`, 86 `run`; 54 **NEW** (no match in `audit/experiments.csv`), 305 matched, 9 flagged `dup_session` |
| `session_events_all.csv` | 798 | the same plus infrastructure (venv, rclone, ssh plumbing, doc edits) |
| `dataset_events.csv` | **254** | 66 builder runs, 6 file writes, 182 stated decisions; actions: 115 edit, 69 build, 38 regenerate, 20 top-up, 6 filter, 6 discard |
| `run_events2_raw.tsv`, `run_events_raw.tsv` | 2,946 / 2,344 | the un-collapsed hits, for tracing any row back |

Where the work ran (from `session_experiments.csv`): lab PC 215 events (123 of them NDIF),
laptop 144.

**Scale of the owner's own work, as the transcripts record it:** 11 working sessions over
13 days (2026-09-10 → 09-22), 5,149 tool calls, **359 distinct test/run events** and **254
dataset-construction or dataset-change events**, of which 54 tests are named in no row of
`audit/experiments.csv`. This is a lower bound: it counts only what was issued from a
Claude session, not the scheduled-task runs that fired on the lab PC in between. The great majority of the owner's *measurement* work is `run-coords` /
`run-crossgen` analysis on the lab PC; the laptop rows are figure-building and the inline
checks.

---

## 4. Coverage caveats (read before using these tables)

1. **DOC-ONLY.** Nothing here is verified. Outcomes are the console line the transcript
   shows, which may be from a partial run, a smoke test or a version later overwritten.
2. **`matches_experiments_csv` is a *name* match, not a judgement.** It matches on script
   basename, `run-*` path and task name, so a row often lists several `exp_id`s (a command
   naming `run-coords/passB` matches every experiment that reads passB). `NEW` means only
   "no name in `audit/experiments.csv` matched", which for one-off analysis scripts
   (`lenauc.py`, `check_cov.py`, `factcheck.py`, `joinerr*.py`, `probe_limits.py`) is
   usually correct, and for infrastructure scripts is uninteresting.
3. **Repeats are collapsed.** `repeats` counts identical (session, kind, script, flags/bank/
   path) invocations. Two runs that differ only *inside* a quoted PowerShell array could
   still collapse; the key now includes every `--flag`, every `*.pt` and every `run-*` path
   in the command, which was enough to separate the two `scale_70b` runs (§5.1).
4. **`run_lab` rows carry the whole wrapper command**, so `what_ran` shows the first `.py`
   mentioned, which is occasionally not the one that ran (e.g. a hashing command that lists
   several scripts). Check `evidence_quote` before trusting one of these.
5. **The 09-10 pair.** `2394bebd` is a fork of `40b4a5dd`; rows from it are marked
   `dup_session` and should be ignored when counting.
6. **Background jobs.** Many launches print "Command running in background (ID …)"; the
   result then arrives in a later `task-notification`, which this pass does **not** link
   back to the launching row. So an `outcome_stated` of "running in background" does not
   mean the run produced nothing.
7. **Not extracted:** the content of `Read`/`Artifact`/`Edit` tool calls, and the owner's
   own prose except where it states a dataset decision. If Phase 2 wants the owner's
   reasoning for a specific claim, `window.py` on the relevant UTC window is the way in.

---

## 5. The 20 findings worth following up, ranked

Ranked by how much each adds to the audit record. "Lab time" = UTC+6.

### 5.1 — Provenance: the two `scale_70b` runs used **different banks**, and the transcripts show the commands (settles RF-16's mechanism)

- `T604` · `b035ccb3` line 7371 · 2026-09-20T17:42Z (**09-20 23:42 lab**) — exit 0:
  `27_crossgen.py --run_dir run-coords/passB --out_root run-crossgen/scale_70b
  --bank_name paper16_70b --bank run-paper16/axes_vectors_s3.pt --datasets
  instructed-deception mask convincing-game ai_liar roleplaying --layers 40 --k 3 --stage all`
  → this is the **F-63 L40 cell**, and it really used a **paper16** bank.
- `T603` · line 7365 · one minute earlier: the same command with `--bank
  run-paper16/axes_vectors.pt` → **exit 1**. (INFERRED: no `axes_vectors.pt` appears in the
  lab census for `run-paper16/`, only `axes_vectors_s3.pt` and the `_fp32` variants, so the
  run most likely failed on a missing bank and wrote nothing — worth a 1-line check.)
- `T774` · `1ac8bb19` line 996 · 2026-09-21T18:51Z (**09-22 00:51 lab**) — the layer sweep
  behind F-100/F-101/F-102, written inline as `task2.sh` with a 5-attempt retry loop:
  `--out_root run-crossgen\scale_70b --bank_name paper16_70b
  --bank C:\Users\user4\expLB\run-kd\axes_vectors_s3.pt --k 3 --poolings mean --stage all`
  → **the kd bank under the label `paper16_70b`**, into the *same* output folder as the
  L40 cell.
  The session even says so in words at line 995: *"The 70B rung was run by hand (there's no
  `run-scale/70b`), against `passB` with `axes_vectors_s3.pt`."* — the ambiguous basename is
  exactly where the mix came from.
- Why it matters: RF-16 was inferred from a zero-shot fingerprint. This is the command
  text. It also shows the **mislabel is in a hand-written retry script**, not in
  `27_crossgen.py`, so the fix is a re-run with the right `--bank`, and no other output
  folder is affected.

### 5.2 — Owner question 1 answered: **nobody triggered the 09-22 15:40–16:26 re-run wave; the lab PC rebooted**

- `1ac8bb19` lines 2330, 2351, 2365 · 2026-09-22T09:46–10:16Z (**15:46–16:16 lab**).
- Quote (line 2330): *"The cause is now clear: **`LastBootUpTime = 15:34:30`** — the machine
  **rebooted**. It wasn't asleep; it went down and came back up."*
- Quote (line 2351): *"**Correction 1: these tasks were never trigger-less.** All four carry
  `MSFT_TaskBootTrigger`: KD-Bank-Crossgen PT20M → fired 15:54:30; KD-S23 PT25M → 15:59:30;
  Scale-3b / Scale-8b → 16:05 (`SKIP already DONE`)."*
- Line 2365: all four finished `rc=0`, KD-Bank-Crossgen doing a *full* re-run (peak, l20,
  rest = 18 layers) at 15:55–16:01:30.
- The session was a 15-minute polling loop ("check the lab PC … status of all running
  Scheduled Tasks…") — that is why a transcript exists for a window the owner was not
  watching. **This is the RF-13 wave, and its cause is an unattended reboot.**

### 5.3 — The HP-KR lexical number moved 0.870 → 0.913, and the transcript shows why

- `T457` · `b035ccb3` line 232 · 2026-09-18T19:25Z — first measurement, inline:
  *"surface-only accuracy of the rule 'denies → deceptive': 0.821; AUROC of the single
  binary surface feature 'denies knowledge': 0.870"*.
- `T480` · line 1213 · 2026-09-19T09:48Z — *"lexical 'denies' AUROC (Llama only):
  **0.913** [previously reported 0.870 over all 3 …]"* — i.e. the change is a **row-set
  change (Llama-only rows)**, not a re-measurement of the same set.
- Phase 2 should use this when resolving RF-9's conflict (REPORT.md and the memory note
  still say 0.870/87%, F-01 says 0.913). Both are right about different row sets.

### 5.4 — The random-direction null that several zero-shot claims must clear exists only in a console log

- `T323` · `6d4b2181` line 8196 · 2026-09-16T18:57Z, run on the lab PC via `run_py.ps1` + `factcheck.py`:
  *"[last] random directions: 95th pct |AUROC| at one layer **0.640**; best-of-21-layers 95th
  pct **0.683**"*, and *"paper16 mean sycophancy axis: L40 **0.313** [0.283, 0.346]
  (length-stratified 0.324)"*.
- `factcheck.py` is not in `audit/experiments.csv` and wrote no result file. This is the
  measurement behind the "several zero-shot AUROCs are no better than random directions"
  claim. **Phase 2 should re-run it** — it is the yardstick for F-13/F-65/F-107-type claims.

### 5.5 — A measured result that was retracted in the same session (never written down)

- `T299` · `6d4b2181` line 7044 · 2026-09-16T13:26Z, "Quantify source-independence" over
  `run-coords/p1_L44_last_withsyc.json`: per-target spread of AUROC across the 9 possible
  training sources, `random mms 0.141 / 0.522`, `random lr_reg10 0.140 / 0.520`.
- The session then says: *"My read of that figure was wrong — the column spread for `axes`
  (0.188) matches `raw` (0.194), so there's no source-independence effect. Dropping it."*
- A clean negative result. It is in no ledger entry and no result file.

### 5.6 — The cap advantage depends on keeping the six position-holding personas

- `T624` · `b035ccb3` line 8235 · 2026-09-20T21:33Z: *"Drop the six position-holding personas from the
  basis and the entire advantage vanishes (`pc:cap_only@2` vs raw: **−0.008, p=0.56**)."*
- Directly qualifies the cap headline (F-65/F-66, and CLAUDE.md's "0.745 vs 0.374"). Check
  whether any ledger entry states this.

### 5.7 — The cap pipeline's own bootstrap said the headline did **not** clear

- `T536` · `b035ccb3` line 3456 · 2026-09-20T08:19Z (`make_cap_page.py`): `pc:unified − raw` **+0.0757**,
  62/110 pairs, 95% CI **[−0.006, +0.173]**, q **0.499**, "clears? **no**".
- Phase 2 must check which of these (and which BH family, per RF-13) the ledger quotes.

### 5.8 — "1-d off-diagonal equals zero-shot exactly" — a check on the collaborator's own numbers

- `T763` · `1ac8bb19` line 557 · 2026-09-21T18:13Z, reading his CSVs straight out of
  `origin/feat/persona-pc-ood`: `L20 ext_axis1` zero-shot mean **0.9041173922** = trained
  off-diagonal **0.9041173922** (identical to 10 decimals), while `axis1` differs
  (0.7909 vs 0.6511).
- If a 1-d "trained" probe reproduces the zero-shot projection exactly, that column carries
  no training at all. Relevant to the draft's tables.

### 5.9 — The abstract's 0.904 measured two ways

- `T756` · `1ac8bb19` line 192 · 2026-09-21T17:22Z: `auroc_ext_axis` six=**0.9041**, excl-RepE=0.8868
  (−0.0173); `auroc_axis` six=0.7909 → 0.7503; `auroc_pc1` 0.8056 → 0.7686.
- This is the F-97 family (the abstract's headline is the kd-augmented axis). The
  excl-RepE arm is a leave-one-dataset-out sensitivity that I did not find in a result file.

### 5.10 — The matched control that kills the vs-raw gain

- `T425` · `dc605d01` line 2346 · 2026-09-17T23:23Z: *"The unified persona axis minus dataset PCA at
  the same k = **−0.003**. It matches PCA of the training dataset's own activations, so it
  carries no advantage…"* — computed inline while editing the thesis chapter.

### 5.11 — The bf16 → fp32 bank rebuild changed almost nothing (RF-15 corroboration)

- `T245` · `6d4b2181` line 5673 · 2026-09-15T20:04Z: cos(deception, assistant) per layer,
  bf16 build vs fp32 rebuild — L20 −0.364/−0.364, L22 −0.338/−0.337, L40 −0.370/−0.371,
  L60 −0.374/−0.374.
- Independent of the audit's own measurement (`audit/phase1/axes_old_vs_fp32.txt`), and it
  agrees. Recorded only in a memory note.

### 5.12 — Length-only transfer, measured with a throwaway script

- `T656` · `b035ccb3` line 10113 · 2026-09-21T07:01Z, `lenauc.py` on the lab PC: raw off-diagonal
  **0.6623** (n=110), random **0.5849**, datapca **0.6651**.
- In the same exchange (`T655`) the session states plainly: *"L40 was picked **because it
  maximised the persona advantage**"* — a selection effect the draft should disclose.

### 5.13 — Persona-count curve peaks at n=4, not at the full set

- `T272` · `6d4b2181` line 6354 · 2026-09-16T11:43Z (after fixing a group-access bug): *"peak at n=4
  (**0.863**); full set n=7 (**0.861**); gain from n=1 to full **+0.176**"*, and the reading
  that followed: the spread collapses 0.153 → 0.012, so "more personas buy reliability, not
  accuracy".

### 5.14 — The matched-slice reversal, per cell with CIs

- `T500` · `b035ccb3` line 1915 · 2026-09-19T14:21Z: axes **as source** full HP-KR **0.300** → matched
  HP-KRm **0.752** (Δ +0.451); raw 0.694 → 0.715 (Δ +0.021); random also listed.
- This is F-03's headline, computed inline from `run-crossgen/kd11/matrices/mean_L40_s42.json`.

### 5.15 — Our axis code is bit-identical to the published assistant-axis pipeline

- `T032` · `2394bebd`/`40b4a5dd` line 1358 · 2026-09-10T11:33Z: `assistant_axis/pipeline/5_axis.py` vs our `6_axes.py` on the
  same vectors — *"per-layer cosine: min 1.000000 median 1.000000 max 1.000000"*, max abs
  diff 0.0. A validation result with no result file; worth a line in the paper's methods.

### 5.16 — The NDIF capacity probe behind the "attention memory limit" rule

- `T178` · `6d4b2181` line 4340 · 2026-09-15T10:18Z, `probe_limits.py` on the lab PC:
  `B=4 T=1500 product=6000 OK 15.7s`; `B=2 T=3000 product=6000 …` ERROR. Engineering, but it
  is the evidence for the rule that shaped every extraction batch size.

### 5.17 — Q8 trained per-token table (print-only)

- `T358` · `dc605d01` line 192 · 2026-09-17T06:26Z: AUROC/TPR by layer for every probe×aggregation —
  `tok_lr/mean 0.785 0.779 0.787 0.839…`, `tok_mms/mean 0.712 0.700 0.770 0.742…`,
  `zeroshot_axis/mean 0.702 0.685 0.747 0.769…`. Read out of `q8_trained_pertoken.json`, so
  the file exists; the *table* does not.

### 5.18 — The first probe pilot, with its own null

- `T203` · `6d4b2181` line 4673 · 2026-09-15T11:09Z: `axes_resid paper 0.482 (n=3)`,
  `axes_resid apollo 0.462 (n=3)`, label-shuffle null `0.479 ± 0.062` over 15 draws.
  Superseded within hours, but it is where the "residualising collapses own100" line starts.

### 5.19 — HP-KR row/column means by feature, straight off the L40 matrix

- `T455` · `b035ccb3` line 156 · 2026-09-18T19:23Z, from `run-crossgen/matrices/mean_L40_s42.json`:
  `datapca@2` paper 0.7241 / 0.5147 / 0.7904, lr_reg10 0.7256 / 0.5286 / 0.7931,
  mms 0.7225 / 0.5338 / 0.7923. (The three columns are as printed; from the call's own
  description they are a row mean, the HP-KR column and the mean excluding it — INFERRED,
  the printing code is inline and not kept.)

### 5.20 — The premise cluster-permutation control failed its smoke test

- `T682` · `b035ccb3` line 10969 · 2026-09-21T09:59Z (`30c_cluster_perm.py`, layer 20):
  *"cluster-mass p = **0.6667** (null mean 0.2258, 95th pct 0.4137)"* — a non-result from the
  premise extras arm (O80), worth checking against what F-8x claims.

---

## 6. Dataset work — the changes worth knowing about

Full list in `dataset_events.csv`. The ones that change how a result should be read:

1. **The kd question set was rebalanced hours before the bank was generated** (`b035ccb3`
   2026-09-19T08:57Z = 14:57 lab; per RF-1 the kd rollouts ran 09-19 20:50 → 09-20 00:18 lab). The flaw, in the session's words: *"**Every one of the ten questions
   stipulated that the speaker knows.** … a `candid_unknowing` persona — whose entire trait
   is truthfully admitting gaps — had nothing to be ignorant about."* Fixes: questions
   rewritten, Q25 dropped, the sincere group redefined *dispositionally*, and the role
   **renamed `pc_candid_unknowing` → `pc_honest_novice`**, then
   `kd_personas.py --write` regenerated all 36 role files (08:58Z = 14:58 lab).
   → This independently dates the `pc_honest_novice` name to **09-19**, i.e. before the
   collaborator's first results from that bank (09-20 17:53 per RF-1), consistent with
   RF-1's resolution and independent of the git history that RF-1 used.
2. **An external (Gemini) review of the kd bank was folded in** (09-19T09:15–09:17Z). Two
   items adopted and promoted to gates in `kd/docs/DESIGN.md`: Assistant-Axis **drift
   balance** between deceptive and honest personas, and a **length** balance. The rest was
   rejected as "my own framing paraphrased back". The review text is in the transcript only.
3. **paper16 was regenerated on a shared question set** (09-17T16:14Z → 21:29Z) after the
   measurement that *"the first unified principal component (47% of variance) separates
   personas by **which question set they answered**, cosine −0.98"*. 26 personas × (10
   deception + 10 sycophancy) = **2,600 rollouts**; 25 judged. This is the
   `run-paper16-shared` bank.
4. **The 11th dataset cost no new extraction.** `make_view.py` (09-19T14:09Z) built
   `harm-pressure-kr-matched` as a **subset of rows already stored** in passB:
   *"VIEW LOADS: X(258, 8192) … labels {1: 124, 0: 134}"* against full HP-KR X(2139, 8192).
5. **…and it was deliberately *not* built for passA/passC.** `make_view_ac.py`
   (09-20T07:36Z) gives *"passC: HP-KR rows 200 → matched 98, labels {1: 89, 0: 9}"*, and the
   session stopped: *"**Stop — that's a column I must not produce.** The matched slice in
   passA/passC is 98 rows but 89 deceptive / 9 honest, against passB's balanced 124/134."*
   → This is why the passA/passC grids are 10 datasets, which the anomalies file flagged as
   a duplicate-content surprise. It was a decision, not an accident.
6. **Store coverage was measured before the grids ran** (`check_cov.py`, 09-19T13:55Z):
   `passC` HP-KR 200 rows / text-matched 98 / knowledge-matched 191; `passB_fine` 2,139 /
   258 / 2,001; `passP` 2,139 / … . Also `check_pool.py` and `check_b.py` on what each pass
   stores (`passB` tokens `(3549, 6, 8192)` + `tok_offsets`).
7. **The s23 filter decision** (09-19T20:04Z): *"score-3 filtering throws away ~30% of
   rollouts that were not failures … Keeping 2+3 discards only the **1.1%** that genuinely
   broke character"*, checked by `verify_s23.py` (per-role 146.8 vs 149.0, ratio 0.98×,
   min per-role count 142).
8. **The cap question bank went through three versions in one day** (09-20):
   `cap_questions.jsonl` → `cap_questions_v2_draft.jsonl` (10:58Z, rewritten 11:06Z with a
   `framing` field) → `cap_questions_v3.jsonl` (12:55Z), with `make_cap_review.py`
   regenerating REVIEW_PACKET_V2 / V3 / V4 each time (20 questions, 12 personas, 60 variants).
9. **HP-KR top-up** was registered as its own lab task on 09-21T04:53Z
   (`run_hpkr_topup.ps1` + `register_hpkr_topup_task.ps1`), after `inspect_store.py` showed
   the existing store (`chunks=55, 2139 rows, per_token=True`). The denial generator
   `gen_hpkr_denial.py` was smoked on 09-20T16:52Z, including a `--template control` arm.
10. **The sarcasm pilot** bank and questions were written by hand on 09-21T14:29Z
    (`cap/questions/sarcasm_pilot_v1.jsonl`, `cap/personas/sarcasm_pilot/*.json`), scored by
    `score_sarcasm_markers.py`; the natural-sarcasm dataset was built on 09-22T08:26Z
    (`build_sarcasm_natural.py --write`) with the stated gate floor *"an axis AUROC at or
    below **0.632** has measured vocabulary, not sarcasm"*.
11. **The evaluation corpus itself**: `eval_datasets.py` (09-14T20:27Z) registered
    **15,358 rows over 11 datasets**, recovered the MASK honest class, and **excluded
    `gender-secret` and `soft-trigger` from the main corpus** as LoRA-generated.
12. **Sycophancy** was generated in two stages: single-turn prep 600 questions
    (09-15T12:21Z) and multi-turn prep 1,200 questions drawn from 1,813 distinct available
    (09-16T06:58Z) — `sycophancy_mt`, the dataset that later breaks the persona axes.
13. **A user instruction that shaped the design** (09-14T14:22Z, owner): *"it silently breaks
    the paper's requirement that the probed model wrote the text. Check the file and drop or
    regenerate."* → and *"Reframe as a new experiment inspired by the method, drop all
    replication language, and make the paper-matched persona bank the controlled comparison
    inside your own setup."* This is the origin of the paper16-as-comparison design.

---

## 7. Suggested follow-ups for Phase 2

- Re-run `factcheck.py`'s random-direction null (§5.4) — it gates several claims and exists
  only in a console log.
- Recompute §5.6 (drop the position-holding personas) and §5.7 (the cap bootstrap verdict)
  against the current files; both qualify a headline.
- When fixing RF-16, quote the two commands in §5.1 as the provenance.
- Add §5.2 to the RF-13 entry: the wave had no human trigger, so the regenerated files are
  simply a later run of the same chains, and the boot triggers are still armed.
- Check whether the ledger anywhere records §5.5 (the retracted source-independence result),
  §5.13 (n=4 peak) and §5.20 (cluster-permutation p = 0.67); if not, they are unreported
  negatives worth a line in the paper.
