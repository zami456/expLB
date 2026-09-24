# Queued work

Ordered. Each entry says what it is, why it is worth doing, and what it costs. When one starts,
its outcome goes into `FINDINGS.md`, not into this file.

Live execution state — stages, gates, measured values — is in `../../docs/TODO.md`.
Last updated 2026-09-21 03:45 BDT.

---

## Closed since this file was written

> **CORRECTION 2026-09-24 (DF-31 / DF-68, Phase 6).** Row 1 below calls 0.374 *"the register
> axis's"* figure. **This site was missing from DF-31's own location lists and so was never
> corrected on 2026-09-23 with the other six.** **0.374 is the paper16 *deception* axis's PC1**
> (its contrast reads 0.409); the paper16 **sycophancy** axis reads **0.313**, and only 0.313 lies
> beyond the 1,000-direction random null (p 0.001–0.008).
> **Source:** `run-crossgen/kd11/matrices/mean_L40_s42.json`, zeroshot block, `sycophancy_mt`
> column; sycophancy axis from `24_q10_sycophancy_axis.py` (2026-09-17).
> **How the claim is updated:** row 1 should read *"0.745 zero-shot vs the paper16 sycophancy
> axis's 0.313 on `sycophancy_mt` (the deception axis's PC1 reads 0.374)"*. The gate still passes
> either way — 0.745 ≫ both.
> _The row below is superseded on this point and is kept for the record._

| was | outcome |
|---|---|
| **1. Capitulation personas (`cap/`)** | **DONE.** 0.745 zero-shot vs the register axis's 0.374 on `sycophancy_mt`; clears a dataset-level bootstrap against raw and both matched controls on that column (+0.112, CI [+0.050, +0.185]); the gain does **not** extend to the other ten columns. F-64, F-65, F-66, F-67. Design: `../../cap/docs/DESIGN.md`. |
| **3. Extend `kd_s23` to all 21 layers on passB** | **DONE.** 21 mean-pooled matrices, was 3. |
| **4. Dataset-level bootstrap on the kd bank's +0.076** | **DONE**, and it exposed the seventh paper16 hardcoding — `27_crossgen.py` built its comparison list from a constant holding paper16's four feature names, so any other bank got no persona comparisons at all and the test silently never ran. Fixed; comparisons now derive from the features actually used. F-67. |

---

## 1. Grow the HP-KR matched slice — **partly blocked, route (b) failed its gate**

**The constraint is one cell, not the dataset.** `harm-pressure-kr-matched` is capped by
**honest deniers** — rows where the model genuinely does not know *and* says so. There are
**134** of those for Llama-3.3-70B (F-16), and they set the ceiling on the whole slice at 258
rows. That is fine as an evaluation column and thin as a training source (~200 train rows),
which is the main caveat on F-03's headline result.

### (b) Targeted regeneration — **the smoke test failed (F-59)**

The plan was to use the 312 WMDP items the model is measured to get wrong (F-08) as a source of
honest-denial rows: ask something it genuinely does not know under the HP-KR template and a
truthful "I don't know" is the expected answer.

**All three routes were run. The gate was a denial rate ≥ 0.70 on 20 items; below 0.40 kills it.**

| route | denial rate |
|---|---:|
| harmful template, `correct == 0` | **0.250** |
| control (harmless) template | **0.150** |
| lowest-confidence items by `margin` | **0.417** |

The model is not refusing or hedging — it **claims to know answers it gets wrong**, on 75% of
them. The harmless control made it *worse*, killing the harmful-framing hypothesis outright.

**The finding in the failure: stated knowledge tracks confidence, not accuracy.** Selecting on
the model's own `margin` rather than on correctness nearly doubled the denial rate, which is the
only signal here worth following up.

At 0.417 the full 312 items would yield ~130 usable rows — short of the ≥ 200 gate, and drawn
from a *different population* (low-confidence items) than the existing 134, so pooling them would
mix two constructs. **Deprioritised, not disproven.**

### (a) Re-extract more HP-KR rows into passA and passC — cheap cleanup, still worth doing

Those stores kept only **200 of 2,139** HP-KR rows, which is why their matched slice is
89 deceptive / 9 honest and unusable (F-20). Extracting the full set would make the matched
column reportable on all three stores rather than passB alone.

This creates **no new matched rows** — it stops discarding the ones that already exist, so it is
unaffected by (b)'s failure. **This is now the most valuable item in this file.**

**Cost:** NDIF extraction only, no generation. ~1 hour.

### (c) Full on-policy regeneration at 70B, 3B and 8B — largest, do last

Serves double duty: more HP-KR rows at 70B *and* an on-policy evaluation corpus at the paper's
own model sizes. Its value is the **scale comparison**, not the matched slice.

---

## 2. Proposed, not yet approved

Neither has the owner's go-ahead; both are cheap and both target the project's central claim.

### (a) `d_sarcasm` — the first *positive* test of the register hypothesis

Every test of the register hypothesis so far is negative: the axis fails *where* register and the
target construct come apart. A sarcasm factor probe would test it in the other direction — if
persona axes encode register, a direction built to vary *tone alone* should recover a large part
of what the paper16 axes score, on datasets where tone and the label are correlated.

Cheap (one factor probe, existing rollouts) and the result is reportable either way.

### (b) Run paper16's 26 personas on kd's 30 questions, to de-confound F-61

F-61 found the 36-vs-26 bank comparison is confounded **three ways** — persona set, question set
and rollout count all differ at once. Crossing one bank with the other's questions isolates the
persona set, which is the variable the thesis actually claims something about.

**Cost:** 26 × 30 × variants on NDIF, plus judging. Roughly the cap run.

---

## 3. Standing gaps in what is already reported

Not new experiments — holes in the existing record, each cheap to close.

- **The zero-shot 0.745 (F-65) has no dataset-level bootstrap** and cannot have one: zero-shot
  trains on nothing, so there are no source datasets to resample. A row-level CI is the only
  available interval and must be labelled as such wherever the number appears.
- ~~**Variants B and C have no zero-shot number**~~ — **CLOSED 2026-09-21.**
  `27_crossgen.py` now scores every behaviour axis. A vs B is answered: contrasting against the
  position-holding personas beats contrasting against bare `default` by **+0.045** zero-shot,
  consistent at all three layers and both filters (F-68). Both halves of the A/B/C design now
  say the holding personas are load-bearing despite being behaviourally inert.
- **Single split seed (42) everywhere.** Every crossgen number in the project rests on one seed.
