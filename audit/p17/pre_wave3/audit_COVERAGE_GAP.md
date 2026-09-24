# What the audit actually covered, what it missed, and why

_2026-09-24. Written because the owner asked: "audit was supposed to solve all these. i ran multiple
test including run crossgen — why has the audit failed? missed my work including my teammate."_

Every number here comes from a file opened during this check, named inline. Reproduce with
`python audit/scripts/p15_coverage.py`.

> **CORRECTION 2026-09-24, after Phase 6A (DF-94, 6E).** The figures below were measured before
> Phase 6 ran. **They are superseded.** Re-running `p15_coverage.py` after 6A:
>
> | quantity | was (this file) | now |
> |---|---:|---:|
> | rows with a verdict | 424 | **600** |
> | of those, in the inventory | 314 | **490** |
> | raised while checking | 110 | 110 |
> | inventory claims with no verdict | 378 | **202** |
> | **real inventory coverage** | **45%** | **71%** |
> | **unchecked claims in the draft PDF** | **176** | **0** |
>
> **The single most important change: the draft is now fully covered.** All 176 draft claims have a
> verdict — 160 `MATCHES HIS DATA`, 9 `PARTLY`, 7 `UNRESOLVED`, **0 failures, and not one arithmetic
> error**. The 7 unresolved are two missing artifacts, not two wrong numbers (F-125 §3).
>
> **The 202 still unchecked are exactly the two optional phases**, both skipped by the owner's
> instruction because neither gates the paper: **110** ledger claims (6C) and **92** collaborator-note
> claims (6B). That is now the whole remainder — there is no third category.
>
> **How DF-94's fix should be worded** wherever the tally appears: *"490 of 692 inventory claims
> given a verdict, plus 110 raised while checking. The draft PDF is at 176 of 176. The 202
> outstanding are the ledger (110) and the collaborator's notes (92), neither of which gates the
> paper."*
>
> **What still stands from the text below:** §3 (nothing was missed — her crossgen runs, Mahdi's
> branch, the collaborator), §4 (all three causes), and §5 items 2 and 4. Only the counts change.
> _The text below is superseded on the numbers and is kept for the record._

---

## 1. The headline number in the reports is more generous than the truth

`audit/PROGRESS.md` and both reports say **"424 of 692 claims checked."** That reads as 61%.
It is not.

| quantity | count | source |
|---|---:|---|
| claims in the inventory | **692** | `audit/claims_inventory.csv` |
| rows with a verdict | 424 | `audit/claims.csv` |
| of those, ids that exist in the inventory | **314** | set intersection |
| of those, claims raised *during* checking | 110 | set difference |
| **inventory claims never given a verdict** | **378** | set difference |

**Real coverage of the inventory is 314 / 692 = 45%, not 61%.** The 110 extra rows are genuine
work — claims that surfaced while checking others — but they are not inventory claims, and counting
them in the same numerator inflates the coverage figure. **This is itself a documentation defect
(DF-94).**

## 2. What the 378 unchecked claims are

| source document | unchecked |
|---|---:|
| **the draft PDF** | **176** |
| `kd/docs/FINDINGS.md` | 110 |
| collaborator note | 92 |

**176 unchecked claims are in the paper being submitted.** That is the single most important
consequence of the coverage gap, and it is larger than every other gap in this file combined.

By the inventory's own status field: 286 `TODO`, 92 `CANDIDATE`.

---

## 3. What was NOT missed — three things the owner suspected, checked and cleared

### 3a. Her crossgen runs were covered — all of them

Every store under `C:\Users\user4\expLB\run-crossgen` on the lab PC, against how many audit files
name it:

| store | matrix JSONs on the lab PC | audit files naming it | ledger mentions |
|---|---:|---:|---:|
| `passA` | **960** | 211 | 54 |
| `kd11` | 42 | 377 | 53 |
| `passC` | 72 | 170 | 35 |
| `scale_70b` | 42 | 115 | 8 |
| `scale_70b_paper16fp32` | 42 | 48 | 2 |
| `passB_fine` | 20 | 44 | 0 |
| `scale_8b` | 16 | 122 | 4 |
| `scale_3b` | 14 | 138 | 5 |
| `passP` | 6 | 49 | 0 |
| `banks`, `csv`, `matrices`, `smoke` | 0 | 365 / — / — / 177 | 47 / — / — / 20 |

**No crossgen store was missed.** `passB_fine` and `passP` reach the audit but never the ledger —
worth noting, not a miss.

### 3b. Mahdi's branch carries no results

`origin/feat/llama-layer30` (Mahdi `<mahdi.ndc21@gmail.com>`, 2026-09-19) adds **one commit**
beyond `resumable-acts-cache`: `configs/XXXX-9-llama-L30.yaml` and a `run_llama_all.sh` tweak.
**Result files unique to that branch: 0** (`git diff --diff-filter=A asix origin/feat/llama-layer30`).

The audit found him in Phase 0, recorded him as "a third contributor … added a layer-30 config and
no results. Ask the owner whether this is in scope" (`audit/phase0/branch_compare.md:308`), and
`audit/PLAN.md:130` scoped him out. **Correctly handled.**

*(An earlier pass of this very check reported "8,074 files differing" for that branch. That was a
two-way diff dominated by asix's own 137 commits — corrected within the same session.)*

### 3c. The collaborator was covered, but late

`audit/experiments.csv` logs 114 experiments: 90 owner, **20 collaborator**, 4 upstream/unknown.
His `origin/feat/persona-pc-ood` is the branch the brief named, and 19 experiments are attributed
to it.

---

## 4. So why did things still get missed? Three distinct causes

### Cause 1 — coverage. 55% of inventory claims never got a verdict.

This is the largest cause by far, and 176 of the unchecked claims are in the draft. Nothing subtle
here: the audit ran out of budget before it ran out of claims, and the reports' "424 of 692"
framing obscured how much was left.

### Cause 2 — the scope froze against a moving target.

`audit/BRIEF.md:5` fixed scope on **2026-09-22** to `asix` + `origin/feat/persona-pc-ood`. The
collaborator pushed **`origin/paper` on 2026-09-23 23:46** — 3 commits, 683 files, 11,555
insertions, including the entire 16-layer sweep. It entered the audit on **2026-09-24** as an
explicit scope extension at the owner's request (`audit/PROGRESS.md:2566`), *after* both reports
were drafted.

**Consequence: F-117, F-119 and F-122 are not reflected in `REPORT_PLAIN.md` or
`REPORT_TECHNICAL.md`.** The reports describe a project state that was one day stale on the day
they were written.

### Cause 3 — the audit committed its own documented failure mode.

`CLAUDE.local.md` records the audit's commonest failure as *"a correct observation that never
reached the conclusion."* The sycophancy claim is that failure, inside the audit:

- **Observed correctly:** no `pc:sycophancy` in any `run-crossgen/` store — true, verified.
- **Concluded wrongly:** "never computed, anywhere" — written into `RESULTS_BY_AXIS.md` §4,
  `DOC_FIXES.md` DF-89 and `PROGRESS.md`.
- **The fact:** `origin/paper` carries `syc_pc1`, `syc3`, `syc_axis1`, `syc3_orth` at all 16 layers,
  and `syc_pc1` tops his excl-HP-KR table at L40–56.
- **Caught by:** the owner asking whether the ood branch had been checked. Not by the audit.

Scoping a one-pipeline search and reporting it as a project-wide absence. Corrected in F-122.

---

## 5. What follows from this

1. **The 176 unchecked draft claims are the priority.** Every other gap in this file is smaller.
2. **The two reports predate `origin/paper`** and need a pass for F-117, F-119, F-122.
3. **Fix the coverage framing** wherever "424 of 692" appears (DF-94).
4. **93 DOC_FIXES remain unapplied.** The documents still contain the errors the audit found; the
   audit proposes, the owner decides.

**What this file does not claim.** That the audit was worthless — 314 inventory claims plus 110
raised claims did get verdicts, at a measured 6.5% adjudicated error rate (`audit/phase4/RESULTS.md`),
and the three causes above are specific and fixable. But "the audit solved all these" was never
true, and the reports' own framing made it read truer than it was.
