# CORRECTIONS — Phase 4 adjudications

Every row where the independent Phase 4 re-derivation **DIFFERS** from the audit's Phase 2/3 verdict,
adjudicated by the lead from raw files, with the outcome logged **whichever way it goes**. The brief
names this file.

Started 2026-09-23. Phase 4 is the audit's own error bar: see `audit/PHASE4_PLAN.md`.

## How to read an entry

- **AUDIT WAS WRONG** — the Phase 4 agent is right; `audit/claims.csv` and the claim note are fixed.
- **PHASE 4 WAS WRONG** — the audit's original verdict stands; the agent's reading is recorded with
  why it failed, because the same trap will catch the next reader.
- **BOTH PARTLY RIGHT** — usually a configuration the claim never stated; both numbers are real and
  the claim is under-specified. Logged as a claim defect, not an audit defect.

Nothing outside `audit/` is edited.

---

## ⚠️ Read this before quoting any mismatch rate

**The raw DIFFERS rate badly overstates the audit's error rate, and the reason is measurable.**

Of the first 19 DIFFERS, **14 are the same shape**: the audit said `PARTLY HOLDS`, the agent said
`HOLDS` — **and in 12 of those 14 the agent's own `notes` column records a wording defect that is
precisely the audit's reason for saying PARTLY.** The agent confirmed the number, wrote down the
defect, and then labelled the claim HOLDS anyway. That is a **verdict-granularity artifact**, not a
disagreement about any number.

**Across all 19 adjudications so far, agent and audit disagree about a *number* in zero cases.**
Every dispute is about how strictly to grade a sentence whose number both sides reproduce.

Two rates must therefore be reported, and the second is the real one:

| rate | what it measures |
|---|---|
| **raw DIFFERS rate** | how often two readers put a different label on the same evidence |
| **adjudicated audit-error rate** | how often the audit's verdict was actually wrong |

### Instruction change mid-stream — declared, because it biases the raw rate

Batches **F1, F2, F3, O1** (52 claims) ran **without** a verdict-discipline paragraph. After seeing
the 12 PARTLY→HOLDS rows, the lead added one to every later batch prompt and to
`audit/phase4/AGENT_INSTRUCTIONS.md`: *HOLDS requires the sentence to be right too, and your notes
must not contradict your verdict.*

**This makes later agents more likely to say PARTLY, which makes them agree with the audit more
often — i.e. the change flatters the audit.** The raw DIFFERS rate is therefore **not comparable
between the first four batches and the rest**, and the split is reported in `REPORT_TECHNICAL.md`.
The adjudicated rate is unaffected, because the lead re-checks raw files either way. It is kept
rather than reverted because it also turns the inconsistency into a measurement of how much of the
"mismatch" was ever about grading rather than about data.

---

## Adjudications

### AUDIT WAS WRONG — 5

#### F-70 — `HOLDS` → **`PARTLY HOLDS`**
The audit verified the novelty-assessment half (which claims are agreed-novel; the MASK and Liars'
Bench author lines) and routed the rest away: its note says *"The 0.91/0.87 pair belongs to F-01/F-03
and is checked elsewhere."* It did not check **"1 of 384"**, which is in the claim's own
`stated_numbers`, and still recorded HOLDS.

The Phase 4 agent checked it: **"6 of 4,032" reproduces exactly** in the git copy of
`run-crossgen/kd11/banks/kd/significance.json` (4,032 tests, 6 with `significant_datasets=true`), but
**"1 of 384" matches no significance family anywhere** — git paper16 root = 192 tests, the four git
bank stores = 96 each, lab copies = 4,032–6,300. This is consistent with the audit's own Phase 1
finding that F-04's 384-test source run was overwritten the same evening it was made.

The audit's own note also flags a **stale caveat** in the entry (three PDFs it says are "not in
`papers/`" were added in `4d0d046`) — by the audit's own definition of PARTLY, that alone is a
sentence defect recorded under a HOLDS verdict.

**Cause:** a claim scoped down to the half the checker found interesting, with the verdict written as
if it covered the whole. *Ledger fix **APPLIED** 2026-09-23.*

#### O-022 — `HOLDS` → **`PARTLY HOLDS`**
Both readers reproduce all five numbers to 12+ significant figures from the 110 stored transfer pairs
(oriented mean 0.735361, plain mean 0.676316, median 0.721970, oracle 0.896684). No number is in
dispute.

The defect is the gloss. `docs/COLLABORATOR_BRIEFING.md:354` labels the **0.676** row
*"(what the abstract compares against)"*, in a paragraph that opens *"How weak 0.732 is turns out to
be measurable."* **Lead-verified from raw:** `0.7320325942227636` is the `baseline` row of
`git show origin/feat/persona-pc-ood:src/probes/deception-detection/results/persona_ood_ext_L40/summary.csv`
— the collaborator's off-diagonal mean over **his** datasets (30 pairs). Our 0.676 is over **11
datasets / 110 pairs**. The two are different corpora, and the document says so nowhere; a reader
takes 0.676 to be the abstract's comparator.

The agent flagged this and explicitly declined to choose between the literal and charitable readings.
The literal reading is the one a referee applies.

**Cause:** the audit recomputed the numbers and did not grade the sentence. *Ledger fix **APPLIED** 2026-09-23.*

#### F-01 — `HOLDS` → **`PARTLY HOLDS`** · *the clearest single result of Phase 4*
**Both readers found the identical defect. The audit wrote it down and then graded the claim HOLDS.**

- Audit's own note, verbatim: *"The caveat '68 deceptive rows it does not catch' is the ALL-ROWS
  figure; on the Llama rows the regex misses 15. **The entry attaches a superseded-row-set caveat to
  the current number.**"* — verdict recorded: `HOLDS`.
- Agent's note, verbatim: *"The caveat '68 deceptive rows it does not catch' is the ALL-ROWS count; on
  the 2,139 Llama rows the regex misses 15. **F-01 exists precisely to correct a number computed over
  the wrong row set, and its own caveat still carries the 7,068-row figure.**"* — verdict: `PARTLY`.

Both reproduce the headline numbers exactly (0.9125 on the 2,139 Llama rows, 0.8699 over all 7,068).
The agent's verdict is the one consistent with **the audit's own definition** of PARTLY.

**This is not a missed check — it is an inconsistent grading boundary**, and it is demonstrable
inside the audit's own ledger: **F-26 was graded `PARTLY HOLDS` for a structurally identical defect**
(a length-only comparator "0.553 vs 0.665" that matches no matrix). Same class of fault, two
different verdicts. *Ledger fix **APPLIED** 2026-09-23.*

#### M-007 — `HOLDS` → **`PARTLY HOLDS`** · the F-01 pattern again
Audit's own note: *"**TWO COMPARABILITY CAVEATS the note omits**: the paper panel is the 3B DECEPTION
axis at **k=1** on 5 datasets with HPC excluded, set beside our UNIFIED axis at **k=2/k=5** on 10
datasets."* — verdict recorded: `HOLDS`.
Agent, blinded: all 7 numbers exact, *"Neither the k nor the axis identity is stated, and k moves our
own controls a lot"* — verdict `PARTLY`. The agent even offered the audit an out
(*"a lead could reasonably call it HOLDS"*); the audit's own note removes it.

This is the **k=1 vs k=5 comparability problem** that O2 independently found in §7.2 and that the
plan lists as a named trap. Setting the paper's k=1 deception figures beside our k=2 unified figures
without saying so is precisely the fault the audit flags elsewhere. *Ledger fix **APPLIED** 2026-09-23.*

#### M-054 — `HOLDS` → **`PARTLY HOLDS`** · the F-01 pattern again
Audit's own note: *"**PROBE MISMATCH inside the sentence**: 0.815 vs 0.768 is the `paper` probe,
0.583 vs 0.660 is `lr_reg10` … as written it compares two different estimators and **should name
them**."* — verdict recorded: `HOLDS`.
Agent, blinded: *"the four numbers as quoted come from two different probe heads … neither half names
its probe"* — verdict `PARTLY`.

Both readers agree the conclusion survives with either probe held fixed. Both agree the sentence as
printed compares two estimators. Only the verdict cell differs. *Ledger fix **APPLIED** 2026-09-23.*

### BOTH PARTLY RIGHT — 2

#### F-58 — audit `REVISED → F-63`, agent `PARTLY`
The **supersession stands**: F-63 is the corrected comparison and its cells reproduce to 4 dp.

But the audit's note overreaches: *"NO RAW DATA for the 3B and 8B rows — the pre-mask matrices exist
nowhere (lab, git, Drive) … Anything still citing +0.089 / +0.086 is citing numbers with no surviving
evidence."* The agent found the numbers **do** survive — the crossgen logs are **append-mode**, so the
22:18–22:21 block persists beneath the 23:38–23:40 mask re-run that overwrote the matrices. "The
matrix files are gone" is true; "no surviving evidence" is not.

The agent adds three defects the audit had not separated: 3B/8B used `harm-pressure-choice` while the
70B row used `mask`, so *"the paper's five datasets"* cannot be true of both; 3B/8B are **k=3** and
the 70B row is **k=5**; and within the 70B row the two controls sit at different k (`random` is k=5,
`datapca` is actually `datapca@2`).

**Neither reading is wrong about a number.** Logged as a claim defect. *Note **APPLIED** 2026-09-23.*

#### F-45 — audit `HOLDS`, agent `PARTLY`
The **arithmetic reproduces exactly** on both sides (the 11 eval datasets sum to 14,718 rows; capping
each at 800 gives 6,551 = 44.5%; chunk geometry and the ~57 rows/min rate are internally consistent).
The audit's HOLDS is right about every number.

The agent's PARTLY is a **provenance** point, and it is correct: the 8B extraction directory no longer
holds the run F-45 measured — every dataset present is now capped at 800 rows, i.e. the entry's own
recommendation was adopted and the 193-chunk partial run replaced the same evening. That is a caveat
on the evidence, not an error in the claim.

**The agent also surfaced a project red flag that belongs in the report regardless of this verdict:**
5 of the 12 dataset directories in `run-scale/8b/coords` have **no `rows.jsonl` at all**
(`harm-pressure-knowledge-report`, `harm-pressure-kr-matched`, `insider-trading`, `sandbagging_v2`,
`sycophancy_mt`) — only 3,093 non-alpaca rows across 7 datasets. **Anything that assumes a complete 8B
corpus needs re-checking.** Independently reported by the F1 agent; carried to `REPORT_TECHNICAL.md`.

### PHASE 4 WAS WRONG — 14

All twelve are `PARTLY HOLDS` (audit) vs `HOLDS` (agent), no number in dispute. **In ten of the
twelve the agent's own `notes` column states the defect and the verdict cell contradicts it.**

| id | the audit's reason for PARTLY | what the agent wrote in its own notes |
|---|---|---|
| **F-47** | the stale 1.3 MB `items.pt` and 217 chunk files no longer exist | *"The entry's DIAGNOSIS is incomplete"* — and found a **second** cause (`harm-pressure-choice FAILED: KeyError`) the entry omits |
| **F-56** | *"~17% deceptive at source"* is the **sample's** ratio (134/800) read back as the source's | *"One number I did NOT verify: '~17% deceptive at source'. 134/800 = 16.75% is the SAMPLE ratio"* — the exact defect, left unchecked |
| **F-78** | the six base-vs-chat numbers are attached to the **wrong transfer direction** | agent confirmed the numbers are panel **(b)**, but did not check that F-78's sentence — *"base models transfer to it [sycophancy] markedly better"* — describes panel **(a)**. Its own note warns the panels *"are easy to mis-assign"* |
| **F-91** | **seeds mixed**; averaged over three seeds L40 (0.8872) beats L32 (0.8857), so the second method contradicts the headline | *"from DIFFERENT SEEDS … on the completed 3-seed sweep the mean-of-directions peak is L36 not L32"*, and *"kd_s23/axes peaks at L44, not L32"* |
| **F-94** | three prose counts do not reproduce | *"One secondary sentence does not survive as written"* — 13 of 16, not the stated 15 of 16 |
| **F-63** | the 3B cell is that rung's **best layer**, so the steepness is a layer-selection effect; *"all within 0.007"* is wrong | *"'best persona' is best among k=3 features: `pc:unified@2` beats it at 3B (+0.033) and 70B (+0.026)"* |
| **F-73** | *"confirmed to three decimals"* → agrees to 2 dp; *"exactly zero"* → +0.0035 | *"a median **absolute** gap (signed median is 2.63), and harm-pressure-choice's gap is +0.0035, not 'exactly zero'"* — both defects, verbatim |
| **F-86** | name collision: *"pooled LODO budget matched"* is 0.8062 here, 0.8500 in F-79/F-80 | *"The 0.8545 comparator lives in a DIFFERENT file … which F-86 as written does not say"* |
| **F-90** | the Q5 arm used the **own100** bank, unrecorded; three CIs are print-only | agent read *"only the stored outputs"* and did not reach the bank-identification or print-only gaps |
| **O-043** | p = 0.38 is a **one-sided** tail; *"~666 rows"* is the per-source **cap**, not the realised pool | *"it is a one-sided bootstrap tail, not a two-sided p, and the '~666 rows' matched arm actually trains on ~449 balanced rows"* — both defects, verbatim |
| **O-044** | the document gives the **same condition two values** (0.850 at :32, 0.806 at :63) with no note | *"one control, three values … All three are real; the doc quotes two of them"* |
| **O-046** | *"the register axis's 0.374"* is the **DF-31 mislabel** — 0.374 is the deception axis's PC1 | agent reported a bank correction (cap_s23 not cap_all) but **missed the 0.374 mislabel, which was written verbatim in its own instructions as a named trap** |
| **F-102** | *"under `last` the 70B becomes the HIGHEST of the three"* is a kd-bank artifact; like-for-like gives 3B +0.0745 > 70B +0.0681 > 8B +0.0090 | *"One small overstatement: 'random sits at 0.49–0.69'. The measured off-diagonal random range … is 0.4455–0.6894, so the floor is 0.446"* |
| **F-26** | *"pc_position_holder writes LONGER turn-1 text"* is unsupported (all four blocks 695–698 chars, truncated at the same limit); *"0.553 vs 0.665"* matches no matrix | *"**Two wording problems.**"* — stated in as many words, then graded `HOLDS` |
| **M-049** | the measured range is **0.113–0.170**, not the stated 0.13–0.17; the floor comes from L44 mean | *"One small imprecision … the L44-mean p95 is **0.1131**, just below the stated 0.13 floor"* — then graded `HOLDS` because the error is in the conservative direction |
| **M-013** | verdict `REVISED → F-01` — the note is **superseded**; the right lexical null is 0.913, not 0.870 | found the identical defect (*"0.870 … is the THREE-MODEL POOLED null … the same lexical feature reaches 0.9125"*) but **could not see that a later finding supersedes the note** — see the structural limit below |

**F-78 and O-046 are the two that matter**, because in those the agent did not merely mis-label — it
looked at the right evidence and drew the wrong conclusion. Both are recorded above so the next
reader does not repeat them.

---

## Running count

| outcome | n |
|---|---:|
| AUDIT WAS WRONG | 7 |
| PHASE 4 WAS WRONG | 21 |
| BOTH PARTLY RIGHT | 5 |
| **total adjudicated** | **33 — COMPLETE** |
| *lead's own Phase 5 correction (not a sampled claim)* | *1* |

**FINAL: adjudicated audit-error rate 7 of 107 = 6.5%. Raw disagreement 33/107 = 30.8%.**
**Disagreements about a number: 1** (S-33, resolved in Phase 4's favour by reproducing a script bug).

**All seven audit errors are `HOLDS` that should have been `PARTLY`. None is a claim graded too
harshly.** By tier: T1 failures **0%**, T3b PARTLY **0%**, T2 lead-QA'd **3.4%**,
**T3 HIGH-importance HOLDS 15.0%** — the false-negative risk T3 was built to measure, materialised.

Full write-up: **`audit/phase4/RESULTS.md`**. Per-row outcomes and causes:
`audit/phase4/adjudications.csv`. Rates recomputed by `audit/scripts/p4_rates.py`.

### A STRUCTURAL LIMIT OF PHASE 4 — blinding cannot see supersession

**M-013** exposes a limit worth stating. Its audit verdict is `REVISED -> F-01`: the memory note is
*superseded* by a later finding. The blinded agent found the **identical underlying defect**
(*"0.870 … is the THREE-MODEL POOLED null … the same lexical feature reaches 0.9125"*) but had no way
to know a later ledger entry supersedes the note, so it graded `PARTLY`.

**Every `REVISED -> X` verdict is invisible to a blinded re-derivation by construction.** Such rows
will always register as DIFFERS however good the agent is. There are **4 of them in the sample**
(tier T1), and they should be excluded from, or footnoted in, any mismatch rate. This is a cost of
the independence rule, not a defect in either reader.

### The meta-finding, after 23 adjudications

All five audit errors are the **same fault**, and it is not the one the audit predicted. The plan
said every prior self-correction came from *"a number checked at the project default instead of at the
configuration the claim states."* **None of these three is that.** All three are:

> **the verdict cell does not match the checker's own notes.**

- **F-01** — the audit wrote the defect down and graded `HOLDS`.
- **F-70** — the audit wrote *"checked elsewhere"* about a number in the claim's own
  `stated_numbers`, and graded `HOLDS`.
- **O-022** — the audit recomputed four numbers to 12 significant figures and never read the sentence
  they sat in. (Its sibling **O-021** carries the *same* defect in its notes — *"true of the ESTIMATOR
  but not of the VALUE"* — and was also graded `HOLDS`. The audit noticed once and propagated to
  neither verdict.)
- **M-007** — *"TWO COMPARABILITY CAVEATS the note omits"* … graded `HOLDS`.
- **M-054** — *"as written it compares two different estimators and should name them"* … graded `HOLDS`.

And the **same fault produces almost all 14 Phase 4 errors in the opposite direction**: agents wrote
the defect into `notes` and graded `HOLDS`. Ten of fourteen are exactly that.

**So the dominant error mode in this project's checking — on both sides, independently — is not
mis-measurement. It is a correct observation that never reaches the verdict.** That finding is worth
more to the thesis than the mismatch rate itself, and it is what `REPORT_TECHNICAL.md` should lead
with.

## Ledger fixes — **ALL APPLIED 2026-09-23**, after Phase 4 closed

Applied by `audit/scripts/p4_apply_ledger_fixes.py` (idempotent; re-running changes nothing).
Backup of the pre-fix ledger: `audit/claims.csv.before-phase4-fixes`.
Each of the 12 claim notes in `audit/claims/` carries a dated `[PHASE 4 2026-09-23] amendment`
section. Verdict distribution moved **HOLDS 304 -> 297, PARTLY 112 -> 119**; failures unchanged at 8;
all 424 rows intact.

The ledger was deliberately NOT edited while it was the object of measurement.

1. **F-01** verdict `HOLDS` → `PARTLY HOLDS`. The audit's note already states the defect; only the
   verdict cell is wrong. Cross-reference F-26, graded PARTLY for the same class of fault.
2. **F-70** verdict `HOLDS` → `PARTLY HOLDS`; note to record that "1 of 384" has no surviving
   significance family, and that caveat (3) is stale.
3. **O-022** verdict `HOLDS` → `PARTLY HOLDS`; note to record that 0.676 (110 pairs, 11 datasets) and
   0.732 (30 pairs, his datasets) are different corpora and the briefing conflates them.
4. **F-58** note to record that the 3B/8B numbers survive in the **append-mode crossgen logs**, so
   "no surviving evidence" is too strong; add the k and dataset-set mismatches the agent separated.
5. **F-45** note to record that the 8B extraction directory was replaced the same evening, and the
   incomplete `run-scale/8b/coords` corpus (5 of 12 dataset dirs have no `rows.jsonl`).

---

## QA CHECK — is the post-discipline agreement substantive, or just label-matching?

**The risk:** the verdict-discipline paragraph nudges agents towards `PARTLY`, and the audit's most
common verdict is `PARTLY HOLDS`. Two readers could then "agree" 21/21 by both reaching for the same
label for **different reasons** — which would hide real audit errors behind a flattering number.

**Test:** take every row where audit and agent BOTH said PARTLY in the post-discipline batches
(M2, O2 — 9 rows) and compare the *reason*, not the label.

**Result: the reasons match on 9 of 9, several of them to the decimal.** The agreement is substantive.

| id | the defect both readers independently identified |
|---|---|
| **M-072 / M-073** | "+0.053" is the **layer-28** gain; the maximum gain is **+0.1027 at L20** and ~+0.005 at L40. Both state the same two numbers. |
| **M-056** | the 0.006–0.014 range reproduces **only for `lr_reg10`**; the same runs' `mms` probe costs **0.0177 at L40**, outside it. Both quote 0.0056/0.0086/0.0139 and 0.0177. |
| **M-077** | no single (probe, aggregation, layer) reading reproduces all the quoted magnitudes; both flag `sandbagging +0.015` against the stated +0.02 floor. |
| **O-057** | the `hf275` column is measured against the **published** Assistant Axis, not "each bank's own assistant direction" as the verdict sentence says. |
| **O-067** | the same reference-direction slippage: "0.75–0.80" reproduces only against the published axis. |
| **O-073** | the two columns are **not the same estimator** — the configuration the section measures is not the one the sentence generalises to. |
| **O-079** | the stated "1–3%" does not bracket the data: both measure **0.80%–2.64%**. |
| **O-096** | "~7–9%" is unsupported as printed and needs restating with its layer range and reference. |

**Conclusion.** Where the audit said PARTLY and explained why, an independent blinded reader working
only from raw files reached the **same number and the same reason**. That is the strongest evidence
in this audit that its Phase 2/3 verdicts are sound — stronger than the mismatch rate, because it
tests the reasoning rather than the label.

It also sharpens the reading of the pre-discipline batches: those 19 DIFFERS were **agents declining
to grade a sentence they had already faulted in their notes**, not readers who disagreed about data.

---

## THE CENSUS — Phase 4's one error mode, swept over all 424 checked claims

**Why a census is possible.** All five confirmed audit errors have the *same machine-detectable
shape*: the checker wrote the defect into `notes` and then graded the claim clean. That does not need
a sample — `audit/scripts/p4_verdict_note_conflict.py` sweeps every checked claim for it.

**Method.** Flag every row whose verdict is a clean `HOLDS` / `MATCHES HIS DATA` (no PARTLY, no DOES
NOT, no REVISED) but whose `notes` contain language asserting a defect (`omits`, `should state`,
`does not say`, `stale`, `understates`, `contradicts`, `unsupported`, `mismatch`, `underspecified`,
`cherry-picked`, `outside the stated range`, …). A **screen, not a verdict** — every hit needs a read.

**Result: 69 of the 304 clean-verdict rows carry defect language.**

Positive control: **43% of the 111 PARTLY rows** trip the same patterns, as they should — those are
rows where a defect was found *and* graded. The screen is therefore detecting the right signal.

**Recall check against the five Phase-4-confirmed errors: it catches 4 of 5.** It misses **O-022**,
and the miss is the informative part.

### The two sub-modes, and only one of them is screenable

| sub-mode | what happened | screenable? | bound |
|---|---|---|---|
| **1. Observed, not graded** | the checker *found* the defect, wrote it down, graded the claim clean | **yes** | **69 candidates** |
| **2. Never observed** | the checker recomputed the numbers and never read the sentence | **no** | only re-derivation finds these; **O-022 is one** |

Sub-mode 2 is why Phase 4's random sample still matters: the census bounds sub-mode 1 across the whole
corpus, and the sample is the only estimate of sub-mode 2. **Both numbers belong in the report, and
neither substitutes for the other.**

### What this does and does not mean

It does **not** mean 69 claims are wrong. In many the note explicitly reasons that the issue is *not*
a claim defect — *"PROVENANCE NIT (no number changes)"* (F-18), *"DOC_FIXES item, not a claim
problem"* (F-08), *"Two clarifications, not errors"* (F-75), *"Nothing contradicts the entry"* (F-46).
Those are defensible clean verdicts **with a stated rationale**, and they are the audit working
correctly.

The problem cases are the ones where the note describes a defect **in the claim itself** and the
verdict is clean **without saying why that does not demote it** — F-01, M-007 and M-054 are the
proven examples. A full classification is owed before any re-grading.

### Ledger policy — deliberately NOT re-graded here

Only the **five errors confirmed by independent re-derivation** are queued as ledger fixes. The other
64 were found by a text screen, not by re-deriving the claim, and **re-grading a claim on the strength
of a regex would repeat the exact fault this section is about.** They are handed to the owner as a
decision, with a recommendation, in `REPORT_PLAIN.md`.

### One screen hit is a thesis-level finding, not a grading question

**M-080:** *"0.776 from length alone on HP-KR is HIGHER than the deception axis's own zero-shot
reading there (0.6436 at L40/mean, 0.5830 at L44/last, both inside their random null)."*

On the draft's **central dataset**, a length-only baseline beats the method. This belongs in the
results, not in a grading appendix. Carried to `REPORT_PLAIN.md` and `REPORT_TECHNICAL.md`.

---

## S-33 — PHASE 4'S BEST RESULT: an audit "cannot be determined" is now determined

The audit's **sixth self-correction** (logged in wave 5 QA) reads:

> *"The F-61 QA overclaimed. It said the mis-copy 'can be proved arithmetically'. **Two candidates fit
> and the audit cannot separate them** … (B) F-61's console reading of **raw 0.6800 over 109 pairs**
> (0.6861−0.6800 = 0.0061, 12.4), attested in the transcript but **surviving in no copy** — W6G checked
> git HEAD, the `3dfdb67` blob, the lab file and the cache, **all 110 pairs**."*

**Candidate (B) is now proved, and the audit was searching for the wrong kind of thing.** It looked for
a *file* containing 0.6800 and correctly found none — because no file ever held it. **The number was
manufactured at read time by a script bug.**

**Mechanism** (found by the SA1 agent, then reproduced independently by the lead from raw files):
`kd/scripts/paper_comparison.py` originally took its axis labels from `d["rows"]` and `d["columns"]`.
Those keys are **human captions, not label lists**:

```
rows    = 'trained on (80% split)'                                        (22 chars)
columns = 'tested on: own 20% split on the diagonal, 100% elsewhere'       (56 chars)
datasets = [11 dataset names]                                            <- the real axis
```

`zip(rows, block)` therefore iterated **characters**, and the diagonal test `tr == te` compared single
characters instead of dataset names — so almost every in-distribution diagonal cell survived into the
"off-diagonal" mean, inflating it.

**Lead's independent reproduction**, on `run-crossgen/kd11/matrices/mean_L40_s42.json` (probe `paper`,
L40, mean, seed 42) — running both code paths over the same file:

| code path | pairs | raw off-diagonal | `pc:unified` − raw |
|---|---:|---:|---:|
| **fixed** (`rows = cols = d["datasets"]`) | **110** | **0.6623** | **+0.0093** |
| **buggy** (`rows = d["rows"]`, `cols = d["columns"]`) | **109** | **0.6800** | **+0.0061** |

Both stated numbers fall out exactly: **109 pairs, 0.6800, and F-61's headline "+0.006"**.

**Consequences.**
1. **F-61's "+0.006" is a bug artifact.** Like-for-like it is **+0.0093**, and the "twelvefold"
   ratio is **8.1x** (0.0757 / 0.0093), which is the figure the audit already treats as undisputed.
2. **The bug is already fixed** in the committed script, and the fix carries a comment naming the
   cause — so this is a historical defect that produced a quoted number, not a live one.
3. **Anything else read through that code path before the fix is suspect.** The agent also reports the
   kd bank matrix has **no `raw` block at all**, so both banks are in fact scored against the same
   root baseline — which is the *real* reason the F-61 comparison is not like-for-like, and a
   different reason from the one S-33 states.

**Adjudication: BOTH PARTLY RIGHT.** S-33's *conclusion* (the two sides are not comparable) is
correct and the audit's `DOES NOT HOLD` on its *diagnosis* is correct; the agent supplied the true
mechanism, which neither had. **The audit's self-correction #6 should be updated from "cannot be
separated" to "resolved: script bug in `paper_comparison.py`".**

### S-25 — `HOLDS` → **`PARTLY HOLDS`** (audit was wrong)
The claim says two PCA bases over an identical role set *"differ by up to 1.347 in a single
component"*. **They do not differ at all.** The agent measured **cos = −1.0000 exactly** at the argmax
(L77, PC4, dim 5114): it is a **per-component sign flip**, which is arbitrary in PCA. Element-wise
max-abs-difference between PCA bases is not a distance and is quoted as one.

The claim's *conclusion* (compare AUROCs, not components) is if anything better supported than stated
— but its premise is wrong, which is PARTLY. Consistent with RF-15's finding that PC sign pinning
differs between the old and fp32 axes at a few layers, and with the memory note
`pc-features-use-basis-not-contrast`. *Ledger fix **APPLIED** 2026-09-23.*

### S-14 — PHASE 4 WAS WRONG (label)
Audit `PARTLY HOLDS`; agent `HOLDS` while its own notes record two caveats — the bank *"is inferred
from `25_zeroshot_nulls.py`'s `--bank` default with no override stored"*, and the null is a
**two-sided** |AUROC−0.5| deviation null, which is why an inverted axis can "beat p95". Both are
exactly the kind of unstated configuration that makes a PARTLY.

---

## INDEPENDENCE INCIDENTS — self-reported, logged in full

The blinding is the only thing that makes Phase 4 a check rather than a repeat of Phase 2. Every
breach is recorded here whether or not it changed an answer.

### 1. SA1 / A-10 — CONFIRMED BREACH, self-reported by the agent (2026-09-23)

The agent ran a `grep` for the abstract's wording across `*.md` and the search **crossed into the
audit's own forbidden files**, surfacing three lines that named the "not independent" issue before it
had verified anything: `audit/DOC_FIXES.md:195`, `audit/phase3/DRAFT_GAPS.md:558`,
`audit/PROGRESS.md:432`.

**The agent disclosed this unprompted**, in exactly the terms the instructions ask for:
*"Treat my independence on that one point as compromised; the 0.043 and the L40 counter-reading
(0.167) were derived without them."*

**Ruling.** **A-10 is marked independence-compromised and excluded from the mismatch rate.** Its
substantive content is kept, because the agent separates which numbers predate the contamination and
those were derived from `persona_pcs.py` and the raw stores. No other row in SA1 is affected.

**Cause, and it is the audit's fault, not the agent's:** the instructions named the forbidden files
but never said *searches must be scoped away from `audit/`*. A repo-wide `grep` for a claim's own
wording will find the audit's notes about that claim — the one search every agent naturally runs.

**Fix applied immediately:** every batch prompt from **D4 onward** carries an explicit instruction to
scope searches with `--exclude-dir=audit`, plus a restatement that an unreported contamination is far
worse than a reported one. **Batches F1, F2, F3, O1, O2, M1, M2, SA1, D1, D2, D3 ran without it** —
so the same breach could have occurred, silently, in any of them. Only SA1 reported one.

**This is the single biggest limitation on Phase 4's mismatch rate and it is stated in the report.**
It cuts one way: contamination can only make an agent agree with the audit more often, so it inflates
agreement and **deflates** the measured error rate. The adjudicated rate is more robust, because the
lead re-derives from raw files.

---

## D1 adjudications

### D-028 — `MATCHES HIS DATA (Level 1; aggregates re-derived)` → **`PARTLY MATCHES HIS DATA`**
**AUDIT WAS WRONG. The sixth instance of the pattern, and the most blatant: the audit's note opens
with the words "AUDIT FLAG:" and the verdict cell is clean.**

- Audit's note, verbatim: *"**AUDIT FLAG: the headline 0.220 is HP-KR-driven.** His own Table 3
  excl-HP-KR column shows it (0.909 vs 0.900) but the contributions bullet does not."*
  Its own `confidence` field even reads *"HIGH for the arithmetic; **the excl-HP-KR collapse i**…"*.
  Verdict recorded: a clean MATCHES.
- Agent, blinded: *"the headline 0.220 at layer 20 is almost entirely the HP-KR column: drop HP-KR as
  train and test and the layer-20 gap collapses from **+0.220 to +0.009** (0.9086 vs 0.9000), because
  the PC3 probe inverts on HP-KR."* Verdict `PARTLY`.

**This is not a grading nicety — it is the draft's headline contribution number.** The p.2
Contributions bullet claims +0.220 at layer 20; **excluding the single dataset the extended bank was
purpose-built to repair, the effect is +0.009.** Both readers found it. Only the agent graded it.
*Ledger fix **APPLIED** 2026-09-23.*

### D-004 — PHASE 4 WAS WRONG (label)
Audit `PARTLY HOLDS` because *"the causal half ('that is why the contrast has nothing to go on') is
argued from persona descriptions, **not measured**"*. The agent graded `HOLDS` while recording its own
scope refinement — *"'denial of knowledge' here is Liars' Bench's LLM-classifier bucket 'did not claim
to know', which is broader than the literal words; **14-15 of the 139** llama-70B lies contain no
negated-know phrase at all."* Two independent reasons to withhold a clean verdict; the agent recorded
one and graded clean anyway.

---

## TIER T1 IS CLOSED — 8 of 8 — and the answer is clean

T1 exists for one question, stated in the plan: *"a wrong **failure** is as damaging as a wrong pass —
it would have the owner retract something true."* Every claim the audit graded an outright failure
(DOES NOT HOLD / DOES NOT MATCH / REVISED) was independently re-derived.

| id | audit | blinded agent | |
|---|---|---|---|
| F-49 | DOES NOT HOLD (REVISED → F-50) | **DOES NOT HOLD** | agrees |
| O-036 | DOES NOT HOLD | **DOES NOT HOLD** | agrees |
| O-049 | REVISED → cap/docs/DESIGN.md | **DOES NOT HOLD** | agrees |
| O-097 | DOES NOT HOLD | **DOES NOT HOLD** | agrees |
| D-237 | DOES NOT MATCH — named script absent | PARTLY | severity |
| F-58 | REVISED → F-63 | PARTLY | severity |
| M-013 | REVISED → F-01 | PARTLY | severity |
| S-33 | DOES NOT HOLD | PARTLY | severity |

**Not one of the eight came back `HOLDS`. No claim the audit failed was found to be sound.**
**Confirmed audit errors in T1: zero.** Nothing is queued for retraction-reversal.

The four DIFFERS are **severity disagreements, not direction disagreements** — the audit said
outright-false, the agent said partly-false. They split into two structural causes, neither an error:

- **`REVISED → X` is invisible to a blinded reader** (F-58, M-013). The verdict encodes *"superseded
  by a later finding"*, which cannot be seen from raw files. Both agents found the identical
  underlying defect; neither could know a successor entry existed.
- **Compound claims have no agreed grading rule** (D-237, S-33). D-237 asserts NDIF + CPU-only +
  *"every number is generated by `paper/make_numbers.py`"*; the first two limbs hold and the third is
  **falsified — the script exists on no branch, in no commit, and nowhere on the lab PC.** The audit
  failed the whole claim; the agent failed one limb. This is a **gap in the audit's own verdict
  vocabulary**, not a mistake by either reader, and it should be stated in the methods.

**Consequence for the headline rate: T1's raw 50% mismatch is entirely an artifact of tier
construction.** It must not be reported as an audit error rate. Reported instead as: *8 of 8 failure
verdicts upheld in direction; 0 reversed.*

---

## D-158 — THE MOST CONSEQUENTIAL FINDING OF THE AUDIT

Not a grading question. A falsified statement in the draft, with the refuting data in the author's
own branch. Verdict `DOES NOT HOLD`, agent and lead independently.

**The draft's Limitation says** a pooled multi-dataset supervised probe *"is a stronger supervised
baseline, and **we do not compare against it**."*

**The comparison exists.** `src/probes/deception-detection/results/loo_L20/` and `loo_L40/` on
`origin/feat/persona-pc-ood`, committed in **`6b16a46`, 2026-09-20 20:05** — **two days before the
draft PDF** — under the commit message:

> **"Leave-one-dataset-out: the fair baseline for the generalization claim"**

The draft never mentions leave-one-out.

**Lead's independent recomputation** — `audit/scripts/p4_loo_baseline.py`, re-runnable, reads only
committed CSVs. Both pooled weightings shown so nothing is cherry-picked:

| layer | arm | mean (6 datasets) | **excl HP-KR** | HP-KR cell |
|---|---|---:|---:|---:|
| **40** | `ext_axis1` (the draft's axis) | 0.9158 | **0.9165** | 0.9124 |
| 40 | `baseline` pooled / naive | 0.8496 | **0.9149** | 0.5230 |
| 40 | `baseline` pooled / balanced | 0.8039 | 0.8525 | 0.5612 |
| **20** | `ext_axis1` | 0.9041 | **0.9086** | 0.8816 |
| 20 | `baseline` pooled / balanced | 0.7943 | **0.9283** | 0.1243 |
| 20 | `baseline` pooled / naive | 0.6941 | 0.7678 | 0.3254 |

**CORRECTION (lead, same session).** My first reading of this table quoted *"the stronger pooled arm
at each layer"* — L20 balanced, L40 naive. **That is weighting-shopping, and it is the very fault the
H1 agent independently flags in the collaborator's own `loo_L20/summary.md`**, which makes its
"pooled supervision matches or beats the axis" claim the same way. I am not entitled to a selection I
would criticise in him. Under a **fixed** weighting across both layers:

| axis − pooled, **excl HP-KR** | pooled/naive | pooled/balanced |
|---|---:|---:|
| **L20** | **+0.1408** | **−0.0197** |
| **L40** | **+0.0016** | **+0.0640** |

| axis − pooled, **incl HP-KR** (the draft's framing) | pooled/naive | pooled/balanced |
|---|---:|---:|
| L20 | +0.2100 | +0.1098 |
| L40 | +0.0662 | +0.1119 |

**What is robust across all four (layer, weighting) cells — this is the finding:**

- **The draft's statement is false regardless of weighting.** The comparison exists. Nothing about
  weighting touches that.
- **Removing HP-KR costs the axis 0.048 to 0.130 AUROC of margin in every one of the four cells**
  (−0.069, −0.130, −0.065, −0.048). The HP-KR dependence is not a weighting artifact.
- **In two of the four cells the surviving margin is ≤ +0.002 or negative** (L40/naive **+0.0016**;
  L20/balanced **−0.0197**). So "the advantage survives removing HP-KR" is false in half the cells.

**What is NOT robust, and I should not have said:** *"the pooled baseline wins at L20"* (true only at
balanced) and *"a dead heat at L40"* (true only at naive). Each is one cell of four.

**The finding stands and is still serious** — the draft omits a comparison its author ran, and in that
comparison the headline margin is substantially a single dataset. It is just not the clean reversal my
first pass implied.

### Why this is worse than a single missing baseline

Three independently-derived Phase 4 findings converge on the same dataset:

> **UPDATE 2026-09-24 (DF-105, P17).** **0.9597 is unverified:** the regex behind it is recorded nowhere. A "does-not-affirm" reconstruction gives 0.9413, which is still above the draft's 0.919 (D-022). 0.9125 is verified. **Source:** `audit/p17/correction_blocks_check.csv`; `audit/p17/C_REPORT.md`; `audit/p17/QA_REPORT.md`.

1. **D-158** (here): remove HP-KR and the advantage over a pooled probe is zero or negative.
2. **D1**: HP-KR is **~91–96% solvable by surface wording alone** — a *label-free* regex on the
   response text scores **0.9125** / **0.9597**, at or above the paper's own headline **0.919**.
3. **M-080**: on HP-KR, a **length-only** scorer reads **0.776**, above the deception axis's own
   zero-shot **0.6436**, which itself sits **inside its random-direction null**.

**So the claim rests on one dataset, and that dataset is close to solvable from surface form.** The
wording-controlled test the draft itself proposes as the thing that would settle it appears never to
have been run — while `harm-pressure-kr-matched`, a wording-matched slice, already exists as an
evaluation dataset (noted independently by the D3 agent).

**This quantifies RF-3**, raised in Phase 0 as *"draft says pooled baseline 'not compared' but his
branch has results/loo_L20, loo_L40"*. Phase 0 found the folders; Phase 4 measured them and the
measurement removes the result.

**Recommendation (owner's call):** this is the first thing to raise with the collaborator, and it is
raisable without accusation — the fair baseline is his own work, committed under a message that says
exactly what it is for.

---

## LEAD CORRECTION (2026-09-23, Phase 5) — a recommendation made without opening the file

**The eighth correction to the audit's own work, and it is mine, made in the final report.**

`REPORT_PLAIN.md` recommended, as decision 3: *"Run the decisive HP-KR experiment? **Yes.**
`harm-pressure-kr-matched` exists, the activations exist, it is CPU-only."*

**Wrong on both counts.**

1. **The experiment was already run**, on 2026-09-19, and is in the ledger as **F-03**, **F-72** and
   **F-76**. `kd/docs/FINDINGS.md` mentions `kr-matched` **22 times**. The audit's own Phase 3
   documents record the result. I recommended it from the *draft's* claim that it was unrun (p.12,
   p.13) without checking the ledger — which is the exact fault this report criticises elsewhere:
   **a statement carried forward without re-opening the file it came from.**
2. **It is not CPU-only.** A *clean* version would be, except that the existing matched slice leaks for
   trained comparisons — its row ids are HP-KR row ids by construction, so **~81% of its test rows sit
   in HP-KR's training split** (F-108). Removing that needs **newly generated NDIF rollouts**: days of
   wall-clock and GPU budget, not a cheap re-run. The zero-shot readings are unaffected by the leak.

**What the experiment actually says** (all VERIFIED in the ledger and Phase 3 files):

| test | result |
|---|---|
| trained persona axes vs raw, matched slice, L40 | 0.752 vs 0.715 |
| same cell, L20 | **0.374** — 0.216 below raw |
| dataset-level bootstrap at the pre-registered L40 | **+0.066, CI [−0.069, +0.189], P = 0.161 — does not clear** |
| 95% argmax interval over layers | **[L5, L79]** — 93% of the network |
| label permutation | p = 0.38 |
| all 80 layers | persona axes beat raw at 28/80; dataset-PCA at 31/80 |
| collaborator's extended axis, zero-shot on the matched slice | **0.4630 / 0.6203 / 0.5383** |
| his split knowledge-denial axis | **0.3932 / 0.4989 / 0.4825** — at or below chance |
| his 4-denial vs 3 sincere-ignorance (wording-matched) | 0.5923 / 0.6692 / 0.6659 vs **0.8734 / 0.8826 / 0.8734** unmatched |

**Consequence: 0.21–0.28 AUROC of the draft's HP-KR result is the plain-assertion half of the
contrast, not knowledge denial.** Those zero-shot numbers are in **neither document**.

**Corrected in:** `REPORT_PLAIN.md` (decision 3, the "promising leads" bullet, and a new CORRECTION
section) and `REPORT_TECHNICAL.md` §8 item 1.

**Pattern, for the record.** This is the **same failure mode** as the seven audit errors Phase 4 found
and as the twelve Phase 4 agent errors: *a correct observation that never reaches the conclusion*, or
here, **a conclusion drawn from prose instead of from the file**. It survived the whole audit and
appeared in the final report. That is worth stating plainly rather than editing away.

---

## LEAD CORRECTION #9 (2026-09-23) — same point, second overstatement

**Correction #8** established that the HP-KR wording test had already been run. In fixing it I wrote
that closing its remaining leak *"needs newly generated NDIF rollouts — days of wall-clock and a GPU
budget, not the cheap CPU-only job I described."* **That is also wrong, in the opposite direction.**

**F-108 (measured, 2026-09-22):** `harm-pressure-kr-matched` is **a view over HP-KR** — 258 rows,
**100% of its row_ids are HP-KR row_ids**, and the overlap is confined to that one pair.

**So the leak is removable by re-splitting, not by re-generating.** Excluding the 258 shared row_ids
from the training side runs on activations that already exist: **CPU-only, no NDIF**. F-108 **already
did this** for the pooled-training claim via leave-one-cluster-out and reported **+0.125 to +0.157**
against the leaked +0.178. The same machinery applies to F-03's cell.

New NDIF rollouts are only needed for a *different, stronger* experiment — a matched slice built from
**new material** rather than a disjoint re-split of the existing pool.

**Corrected in** `audit/HPKR_WORDING_CONTROLLED.md` §6 (rewritten) and §7/§8.

**Pattern, twice on the same paragraph.** #8 was a conclusion taken from the draft's prose without
opening the ledger. #9 was a cost estimate written without opening **F-108, which is cited two
sections earlier in my own document**. Both are the audit's headline failure mode —
*a correct observation that never reaches the conclusion* — committed by the lead, in the deliverable
that documents it. Recorded rather than quietly amended.
