# PHASE 4 — independent check. Plan, ready to execute.

Written 2026-09-23 at the end of Phase 3, so a **fresh session** can run Phase 4 without
reconstructing anything. Read `audit/BRIEF.md` and `audit/PROGRESS.md` first, then this.

---

## What Phase 4 is

The brief: *"Fresh agents that get only this brief, the ledger row and the raw files (not your notes)
re-derive every HIGH-importance claim plus a random ~20% of the rest. Report the mismatch rate, fix
the ledger, and log each fix in `audit/CORRECTIONS.md`."*

Its purpose is **not** to find more errors in the project. It is to measure **how often the audit
itself is wrong**, and to bound the 268 claims Phase 2 never reached.

That question is live. The audit has already corrected its own work **nine times**, all caught by the
rolling QA rule, and **three of those nine ran in the owner's favour** — her citation, her "9 of 10"
and her "0.553 vs 0.665" were right and the audit was wrong. Single cause each time: *a number checked
at the project default instead of at the configuration the claim states.* Phase 4 exists to put a
number on how much of that is left.

---

## Scope: targeted, on the owner's decision (2026-09-23)

Full brief scope is **279** re-derivations (243 HIGH + 20% of the rest). The owner chose a targeted
**161**, for two stated reasons:

- 243 of 424 checked claims carry a HIGH rating **assigned by the batch agent that checked them**.
  More than half the corpus being HIGH is rating inflation, not importance.
- Effort is better spent where a wrong verdict would change the thesis, plus a genuine random sample
  to bound the rest.

**Sample: `audit/phase4/sample.csv`** (161 rows), built by `audit/scripts/p4_select_sample.py`,
**seed 42, re-runnable**. Five strata:

| tier | n | why |
|---|---:|---|
| **T1** outright failures (DOES NOT HOLD / DOES NOT MATCH / REVISED) | 8 | a wrong *failure* is as damaging as a wrong pass — it would have the owner retract something true |
| **T2** every claim the lead QA'd | 29 | load-bearing by construction, and the place the audit already erred |
| **T3** random HIGH-importance **HOLDS** | 40 | **false-negative risk** — a wrong HOLDS puts a bad number into the thesis unchallenged |
| **T3b** random HIGH-importance **PARTLY** | 30 | did the verdict stop at the wording when the *number* was also wrong? |
| **T4** random **20% of the 268 unchecked** | 54 | the only stratum with no coverage at all |

Spread across sources: D 43 · F 39 · O 25 · H 24 · M 19 · S 9 · A 2.

**PARTLY verdicts are deliberately not taken wholesale** (there are 119). A PARTLY means the number
reproduced and the sentence failed — the audit's commonest and best-evidenced verdict. A sample of the
HIGH ones tests it adequately.

---

## The independence rule — the thing that makes this Phase 4 and not Phase 2 again

**Hand agents `audit/phase4/sample_blinded.csv`, never `sample.csv`.** The blinded file carries
`id, source_ref, stated_numbers, claim_plain` and **withholds the prior verdict, the prior importance
and the tier**. An agent told the audit said HOLDS is no longer an independent check of HOLDS.

Also withhold: `audit/claims/<id>.md`, `audit/claims.csv`, `audit/DOC_FIXES.md`, and the Phase 3
deliverables. Agents **may** use the shared computed inputs, which are data rather than conclusions:
`audit/phase2/layer_sensitivity.csv`, `provenance.csv`, `controls.csv`, and `audit/cache/`.

**The lead — and only the lead — compares against `sample.csv` afterwards.**

---

## Procedure

1. `python audit/scripts/p4_select_sample.py` to regenerate both files (idempotent, seed 42).
2. Split the 161 into batches of ~12 **by source**, so each agent loads one store and one context.
3. Launch **3–4 agents at a time. Not 7** — seven concurrent blew the session rate limit on 09-23 and
   killed a whole wave with nothing written.
4. Each agent writes `audit/phase4/rederive_<batch>.csv` with, per claim: `id`, the value it
   independently derived, the file+key it used, its own verdict, and its confidence. It does **not**
   see or reference the Phase 2 verdict.
   - **Put this line in every agent prompt, verbatim:** *"When your last claim is written, hand back
     immediately. Do NOT re-read, re-verify, count or audit your own output files — the lead verifies
     every batch on merge, and duplicating that wastes the session budget."*
   - **Why:** on 2026-09-23 batch W6F2 finished its real work, handed back a complete report, and then
     spent **35 minutes and 355k tokens** in a self-verification loop titled *"Verifying all 27 claim
     notes exist"*. The lead confirmed all 27 notes, all 27 CSV rows and all 27 merges **in about one
     second** and stopped the agent. Nothing was lost, but a whole batch's budget was.
5. The lead joins each batch to `sample.csv` and classifies every row **AGREES / DIFFERS / UNRESOLVED**.
6. Every DIFFERS is adjudicated by the lead from raw files, and the outcome — whichever way it goes —
   is logged in **`audit/CORRECTIONS.md`** (new file, the brief names it).
7. Report **the mismatch rate**, by tier. That number is the audit's own error bar and belongs in
   `REPORT_TECHNICAL.md`.

---

## Traps every Phase 4 agent must be given

All verified, all have cost this audit time:

- `block "9"` in the kd11 era = **10 datasets**; `block "10"` = 11; in the run-crossgen ROOT store
  `block "10"` = 10. A block label is never a dataset count.
- A bank **filename never settles which bank was used** (RF-16). `s3` ≠ `s23`.
- **BH q depends on the test family, which grew in the 09-22 boot wave — and this has flipped a real
  verdict** (S-32: q 0.067 git vs 0.050 lab, identical raw statistics). State the family size and
  which copy was read.
- `raw` and the matched `datapca` control must be joined from the **ROOT** store.
- **"0.374" is the deception axis's PC1, not the sycophancy axis (0.313)**.
- A **trained** random probe is not a **zero-shot** random direction.
- The paper uses **k=1 for deception, k=3 for sycophancy; k=5 appears nowhere in it.** The project
  default is k=5.
- In `passA`/`passC`/`<root>`, seeds 0–4 carry a **reduced feature set** (no `datapca`, `random`,
  `length`), so control comparisons there are seed-42-only.
- **Never quote a count or effect without its metric, sign convention and dataset set.** Nine audit
  self-corrections came from this one habit.
- **Check a number at the configuration the claim states, not at the project default.** This is the
  single cause of every audit self-correction so far. `run-coords/` in the working tree holds 37
  result JSONs at settings the crossgen stores do not use.

---

## Session hygiene — two things the 09-23 session got wrong

**1. "Is anything running?" means the TASK LIST, not the process table.** On 2026-09-23 the lead
answered that question twice by checking local `python`/`ssh` processes and `Get-Process python` on
the lab PC, and reported "nothing is running" **while an agent had been live for 35 minutes**. An
agent is API calls; it does not appear in `tasklist` or on the lab PC at all. **Check the task list.**
The owner spotted it, not the audit. (Same failure mode as every one of the audit's nine
self-corrections: checking the right thing in the wrong place.)

**2. Stop agents that have already handed back.** A completed hand-back does not always end the
agent. If the task list shows one still running after its report arrived, verify its outputs exist and
are merged, then `TaskStop` it by task id.

## Before starting

- Nothing should be running — **check the task list, not just processes**. Process state was verified
  clean on 2026-09-23 at 17:20 (no local python/ssh, **0 python on the lab PC**), and the one lingering
  agent (W6F2) was stopped after its 27 notes, 27 CSV rows and 27 merges were confirmed complete.
- `git status` outside `audit/` should be **18 entries** (17 modified + `docs/DRAFT_CORRECTIONS.md`),
  all authorised edits, all uncommitted — the owner commits.
- The lab PC is reachable. ~~Still owed there: patching the lab copy of `run_scale_chain.ps1`.~~
  **DONE 2026-09-23.** The lab copy was verified to be the unpatched original
  (md5 `155e6fc1…`, identical to the repo's pre-fix backup), backed up to
  `audit_scratch/backup_DF48_run_scale_chain.ps1` with the hash checked, then patched. Lab file is
  now **`37c12be1…`, byte-identical to the patched repo copy**, with the fixed block confirmed at
  lines 198-212. Logged in `audit/LABPC_WRITES.md`. **Repo and lab now match, as they do for the
  eight guard-bug scripts.**
- **Owner action, unrelated to Phase 4:** the lab's
  `AppData/Roaming/rclone/rclone.conf.bak-2026-09-22` holds a live OAuth refresh token on a shared
  machine. The audit verified existence by metadata only and will not delete it.

---

## After Phase 4

Phase 5, four deliverables (the brief's two plus the owner's two, already written):
`REPORT_PLAIN.md` · `REPORT_TECHNICAL.md` · `METHODS_CHAPTER.md` ✓ · `LIMITATIONS_AND_FUTURE_WORK.md` ✓.

The reports must carry Phase 4's measured mismatch rate. An audit that reports findings without its
own error rate is asking to be trusted rather than checked.
