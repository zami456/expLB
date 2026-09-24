# PHASE 4 RESULTS — the audit's own error bar

Completed 2026-09-23. **161 claims re-derived by 14 blinded agents, 33 disagreements adjudicated by
the lead from raw files.** Every number here is produced by `audit/scripts/p4_rates.py` from
`merged.csv` and `adjudications.csv`, and is re-runnable.

---

## The two rates

| | value | what it measures |
|---|---:|---|
| **Raw disagreement** | **30.8%** (33/107) | how often two readers put a different label on the same evidence |
| **Adjudicated audit-error** | **6.5%** (7/107) | **how often the audit's verdict was actually wrong** |

**Agent and audit disagreed about a *number* in one case out of 161** — S-33, and there the agent was
right and resolved a question the audit had filed as unresolvable. Every other dispute is about how
strictly to grade a sentence whose number both sides reproduce, usually to four or more decimals.

---

## THE HEADLINE: the audit's error is concentrated in exactly one place

| tier | n | raw disagreement | **audit error** |
|---|---:|---:|---:|
| **T1** outright failures | 8 | 50.0% | **0.0%** |
| **T2** lead-QA'd | 29 | 27.6% | **3.4%** |
| **T3** HIGH-importance **HOLDS** | 40 | 17.5% | **15.0%** |
| **T3b** HIGH-importance **PARTLY** | 30 | 46.7% | **0.0%** |

**All seven confirmed audit errors are claims graded `HOLDS` that should have been `PARTLY`.**
Not one is a claim graded too harshly.

This is the risk T3 was constructed to measure. The plan said: *"**false-negative risk** — a wrong
HOLDS puts a bad number into the thesis unchallenged."* **It materialised, at 15%.**

Three consequences, all actionable:

1. **No true result was wrongly retracted.** T1 is 8 of 8 upheld in direction, 0 reversed. Nothing
   the audit failed needs reinstating.
2. **The audit was lenient, never harsh.** The error runs one way only.
3. **The decision not to re-derive all 119 PARTLY rows was correct.** T3b's audit-error rate is 0%.
   Its 46.7% raw rate is entirely agents *under*-grading, not the audit over-grading.

---

## The mechanism — one fault, both directions

**All seven audit errors, and 19 of the 21 Phase 4 errors, are the same thing:**

> **the verdict cell does not match the checker's own notes.**

The plan predicted a different cause — *"a number checked at the project default instead of at the
configuration the claim states"* — which had explained every earlier self-correction. **None of the
seven is that.** They are:

| id | the audit's own note says… | it graded |
|---|---|---|
| **F-01** | *"The entry attaches a superseded-row-set caveat to the current number"* | HOLDS |
| **M-007** | *"**TWO COMPARABILITY CAVEATS the note omits**"* (k=1 vs k=2/k=5) | HOLDS |
| **M-054** | *"as written it compares two different estimators and **should name them**"* | HOLDS |
| **D-028** | *"**AUDIT FLAG: the headline 0.220 is HP-KR-driven**"* | MATCHES |
| **F-70** | *"checked elsewhere"* — about a number in the claim's own `stated_numbers` | HOLDS |
| **O-022** | (nothing — four numbers recomputed to 12 s.f., the sentence never read) | HOLDS |
| **S-25** | (nothing — premise wrong: a PCA **sign flip**, cos = −1.0000, quoted as a distance) | HOLDS |

And it runs the other way just as hard: in **12 of the 21** Phase 4 errors, the agent wrote the defect
into its `notes` column and graded the claim clean anyway — one agent literally wrote *"Two wording
problems"* and then recorded `HOLDS`.

**So the dominant failure in this project's checking, on both sides, independently, is not
mis-measurement. It is a correct observation that never reaches the verdict.**

### Sub-modes, and why one of them is now a census

| sub-mode | example | screenable? | bound |
|---|---|---|---|
| **observed, not graded** | F-01, M-007, M-054, D-028 | **yes** | **69 candidates** |
| **never observed** | O-022, S-25 | no | only re-derivation finds these |

`audit/scripts/p4_verdict_note_conflict.py` sweeps **all 424 checked claims** for sub-mode 1 and flags
**69 of the 304 clean-verdict rows** (positive control: 43% of PARTLY rows trip the same patterns, as
they should). **Deliberately not acted on** — re-grading a claim on the strength of a regex would
repeat the exact fault being measured. Owner's decision, in `REPORT_PLAIN.md`.

---

## New coverage: the 268 claims Phase 2 never reached

The random 20% sample (54 claims), re-derived for the first time:

| | n | share | checked corpus, for comparison |
|---|---:|---:|---:|
| HOLDS | 42 | **77.8%** | 71.7% |
| PARTLY | 10 | 18.5% | 26.4% |
| FAILS | 1 | 1.9% | 1.9% |
| UNRESOLVED | 1 | 1.9% | — |

**The unchecked corpus is no worse than the checked one.** There is no hidden pile of failures.

**But one of those 54 is the most consequential finding of the entire audit.** D-158 — the draft says
it does not compare against a pooled multi-dataset probe; the comparison exists on the collaborator's
own branch, committed two days before the draft under the message *"Leave-one-dataset-out: the fair
baseline for the generalization claim"*. Removing HP-KR costs the axis **0.048–0.130 AUROC of margin
in all four (layer, weighting) cells**, and in two of four what remains is **≤ +0.002 or negative**.

**A 1.9% failure rate does not mean the remaining 214 unchecked claims are safe to ignore.** One of
them was load-bearing. That is the argument for finishing the corpus, and it is in the report.

---

## Limitations — every one of these cuts against trusting the rate

1. **An independence breach occurred and was self-reported.** The SA1 agent's `grep` crossed into
   `audit/DOC_FIXES.md`, `audit/phase3/DRAFT_GAPS.md` and `audit/PROGRESS.md`. **A-10 is excluded.**
   The instructions named forbidden files but never said to scope searches away from `audit/` —
   **11 of 14 batches ran without that guard**, so a silent breach is possible in any of them.
   **Direction of bias: contamination inflates agreement, so it DEFLATES the measured error rate.**
2. **The agent instructions changed mid-stream.** F1/F2/F3/O1 (52 claims) ran without the
   verdict-discipline paragraph; the other 10 batches had it. Raw rates: **37.3% before, 23.6%
   after.** The change pushes agents towards PARTLY, which makes them agree with the audit more
   often — **it flatters the audit.** The raw rate is not comparable across that boundary; the
   adjudicated rate is, because the lead re-checks raw files either way.
3. **`REVISED → X` is invisible to a blinded reader** (F-58, M-013). Such rows register as DIFFERS
   however good the agent is. 4 in the sample, all in T1 — which is most of why T1's raw rate is 50%
   with zero audit errors.
4. **Compound claims have no agreed grading rule** (D-237, S-33). Two limbs hold, one is falsified —
   the audit fails the claim, the agent fails the limb. A gap in the **verdict vocabulary**, not an
   error by either reader.
5. **Claims about events, not measurements, have no raw data by construction** (O-041: what two
   reviewers said). Correctly returned `NO RAW DATA`; it cannot be scored either way.
6. **The lead adjudicated alone.** Every DIFFERS was settled by one reader — me — from raw files.
   That reader is the one whose work is being measured. The adjudications are itemised in
   `adjudications.csv` with their cause, so each can be checked independently.
7. **The lead overstated one finding and corrected it in-session.** My first D-158 write-up quoted
   "the stronger pooled arm at each layer", which is weighting-shopping — the very fault an agent
   independently caught in the collaborator's own summary. Corrected in `CORRECTIONS.md` and in the
   script. Logged here because a limitations section that omits the auditor's own errors is not one.

---

## What Phase 4 produced beyond a number

- **S-33** — resolved the audit's self-correction #6, filed as *"the audit cannot separate them"*. The
  "raw 0.6800 over 109 pairs" reading survived in no file because **no file ever held it**: it was
  manufactured by a script bug (`paper_comparison.py` read caption strings as label lists, so `zip`
  iterated characters). Lead reproduced both code paths: **fixed 110 pairs / 0.6623 / +0.0093**,
  **buggy 109 pairs / 0.6800 / +0.0061**. F-61's headline "+0.006" is the bug.
- **D-158** — above. The single most consequential finding of the audit.
- **HP-KR convergence** — three agents, independently: the dataset is **~91–96% solvable by a
  label-free regex** (0.9125 / 0.9597 vs the paper's own headline 0.919); **length alone reads 0.776**
  there, above the deception axis's **0.6436**, which sits inside its own random null; and the draft's
  mechanism claim fails its own controls (the **persona-free** RepE probe also inverts at L20, and the
  same contrast does **not** invert at L40/L44 despite identical persona composition).
- **`run-scale/8b/coords` is incomplete** — 5 of 12 dataset dirs have no `rows.jsonl`.
- **`run-scale/8b` axes were overwritten** — the `all` and `s23` builds write the same path (S-31),
  the same class of bug as DF-48 in a different script.
- **Selective reporting across three persona banks** on the collaborator's side, with the third —
  which his own summary says makes every arm worse — never mentioned in the paper.
- **Bank-extraction noise exceeds the reported error bars**: every CI in the draft covers example
  sampling only, while re-extraction moved HP-KR by 0.036–0.077.

---

## Ledger fixes owed (7 verdicts + 5 notes)

Queued, **not applied** — the ledger is not edited while it is the object of measurement. Listed in
`audit/CORRECTIONS.md` and `adjudications.csv`.

| id | change |
|---|---|
| F-01, F-70, O-022, M-007, M-054, S-25 | `HOLDS` → `PARTLY HOLDS` |
| D-028 | `MATCHES HIS DATA` → `PARTLY MATCHES HIS DATA` |
| F-58 | note: the 3B/8B rows survive in the append-mode logs |
| F-45 | note: the 8B extraction dir was replaced the same evening; incomplete corpus |
| S-33 | note + **resolve audit self-correction #6** |
| D-237, O-041 | methods note: compound claims; event-claims have no raw data |

## Files

`sample.csv` · `sample_blinded.csv` · `batches/*.csv` · `rederive_*.csv` (14) · `merged.csv` ·
`adjudications.csv` · `verdict_note_conflicts.csv` · `evidence_F2/` · `AGENT_INSTRUCTIONS.md`
Scripts: `p4_select_sample.py` · `p4_make_batches.py` · `p4_merge.py` · `p4_rates.py` ·
`p4_verdict_note_conflict.py` · `p4_loo_baseline.py`
