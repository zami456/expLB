# The control table — W4

**What this closes.** `audit/PHASE2_PLAN.md:37` asked for the controls to be re-derived **once,
centrally**, so every claim cites the same numbers. It was the last piece of Phase 2 still owed, and
the audit had recorded that the random-direction null existed *"only in a console log from
2026-09-16"*.

**That was wrong.** The null is persisted, with **1,000 draws per cell**, in
`run-coords/zeroshot_nulls.json` — a file sitting in the git working tree the whole time. Nothing
had to be redrawn. Consolidated table: **`audit/phase2/controls.csv`**, 29,414 rows
(42 zero-shot null rows + 29,372 trained-control rows), built by
`audit/scripts/phase3/w4_build_controls.py`.

---

## Sign convention — read this before quoting anything

This audit has been caught five times by a number quoted without its convention, so:

- **`value` is the SIGNED AUROC**, in [0, 1]. 0.5 is chance. **Below 0.5 means the direction is
  inverted on that dataset** — which is a real, reportable result here, not an error.
- **The null is direction-oriented.** The stored fields are `random_p95_auroc_dev` and
  `random_max_auroc_dev`: percentiles of **max(a, 1−a)** over the 1,000 random draws.
- **`oriented_value` = max(auroc, 1−auroc)** is therefore the quantity actually comparable to the
  null. An inverted axis can legitimately beat it — and that is exactly the sycophancy result.

---

## 1. Zero-shot directions against their own 1,000-draw random null

This is the table the reframed thesis should lead with.

### Deception axis, per dataset

| dataset | L40 mean | beats p95? | L44 last | beats p95? |
|---|---:|---|---:|---|
| ai_liar | 0.9367 | **yes** | 0.9475 | **yes** |
| insider-trading | 0.9803 | **yes** | 0.9724 | **yes** |
| instructed-deception | 0.9709 | **yes** | 0.8946 | **yes** |
| roleplaying | 0.7842 | **yes** | 0.8696 | **yes** |
| sandbagging_v2 | 0.8454 | **yes** | 0.8600 | no |
| convincing-game | 0.7884 | no | 0.7735 | no |
| harm-pressure-choice | 0.5439 | no | 0.5511 | no |
| **harm-pressure-knowledge-report** | **0.6436** | **no** | **0.5830** | **no** |
| mask | 0.4669 *(inverted)* | no | 0.4417 *(inverted)* | no |

**The deception axis beats a random direction on 5 of 9 datasets at L40/mean, and 4 of 9 at
L44/last.** On the other four it is inside the null — including **HP-KR, the draft's central
dataset**, where the null's p95 is 0.9377 against the axis's 0.6436. On `mask` the axis is *inverted*
at both layers.

### Sleeper datasets (L44 last)

| dataset | deception axis | null p95 | beats? |
|---|---:|---:|---|
| gender-secret | 0.3512 *(inverted)* | 0.7630 | no |
| soft-trigger-gender | 0.9128 | 0.9257 | no |
| soft-trigger-greeting | 0.8738 | 0.9234 | no |
| soft-trigger-time | 0.8769 | 0.9271 | no |

**0 of 4.** The high raw AUROCs (0.87–0.91) look impressive and are *inside* a null whose p95 is
0.92–0.93 — these datasets are easy for *any* direction. This is why a null matters.

### sycophancy_mt — the inversion, and who actually detects it

| direction | AUROC | oriented | null p95 | beats? |
|---|---:|---:|---:|---|
| **sycophancy axis** | **0.3131** | 0.6869 | 0.6304 | **yes — significantly INVERTED** |
| assistant axis | 0.7102 | 0.7102 | 0.6304 | **yes** |
| sycophancy − assistant | 0.3530 | 0.6470 | 0.6304 | yes |
| deception axis | 0.4089 | 0.5911 | 0.6304 | **no** |

Stable across all four stored cells. Two things follow, and they are the DF-31 correction with the
numbers attached:

1. **The sycophancy axis is significantly inverted** — it detects capitulation with the sign
   reversed. That is a finding, not a failure to detect.
2. **The deception axis is inside the null on sycophancy_mt in every cell.** So "worse than a random
   direction" is true of the *deception* axis and not of the sycophancy one. The briefing originally
   said the opposite; it has been corrected.

> **QUALIFICATION — narrowed 2026-09-23 after batches W6G and W6F2, and the lead's own check.**
> W6G flagged two comparators that might sink the assistant-axis row. **One is now settled in the
> row's favour; the other cannot be settled from stored data.**
>
> - **Best-of-layers null — SETTLED, the row survives.** `run-coords/q10_sycophancy_axis.json` stores
>   `random.mean.best_of_layers_p95 = 0.69355` (max 0.72325) and `random.last.best_of_layers_p95 =
>   0.68325`. **0.7102 clears 0.69355**, and the axis's own best-of-layers value (0.7263 at L24)
>   clears even the **max** of 1,000 draws. Verified by the lead directly from that file.
> - **Length control — NOT settleable.** A length-only scorer reads **0.664** on `sycophancy_mt`, so
>   most of 0.710 is available from response length alone. **No stored file holds per-row
>   projections**, so the length-residualised version cannot be computed from what exists. This stays
>   open, and it is the reason not to call the row a clean win.
> - **New scope condition:** the result is **paper16-bank-specific**. The `own100` assistant direction
>   never exceeds **0.6161** and clears nothing.
>
> **So: quote the assistant-axis row with the length caveat attached, not as an unqualified positive.**
> The *inversion* result for the sycophancy axis is unaffected either way — 0.313 is far outside every
> null, per-cell and best-of-layers alike.

---

## 2. Trained controls

29,372 rows covering `raw`, `datapca`, `datapca@2`, `random`, `random@2` and `length` across every
store × layer × pooling × probe × seed in `layer_sensitivity.csv`. Use these for any
"feature X beats the matched control" claim.

**Two joins that must be done correctly**, both of which have already caused errors:

- **`raw` and `datapca` must come from the ROOT store.** A bank store carries its own `datapca`
  column against which `axes` wins 0 of 80 — the wrong comparator.
- **A trained random probe is not a zero-shot random direction.** They are different objects and live
  in different `control_type` rows here (`trained_control` vs `zeroshot_random_null`). Comparing
  across them is the error in the briefing's old "0.541".

---

## 3. Seed coverage — a limitation this table exposes

Settled by `audit/scripts/phase3/p3_seed_coverage.py`. **Seven stores**, not the three P3-D reported,
have seeds 0–4 carrying fewer features than seed 42:

| store | seeds 0–4 are missing |
|---|---|
| `<root>`, `passA`, `passC` | **8 features**: `axes_resid`, **`datapca`, `datapca@2`**, **`length`**, `pc:unified`, `pc:unified@2`, **`random`, `random@2`** |
| `banks/hf100`, `banks/own100`, `banks/paper16_famcentred`, `banks/paper16_shared` | 2: `axes_resid`, `pc:unified@2` |

**What is safe to average over all six seeds:** `axes`, `pc:deception` and `raw` in the root/passA/passC
stores; `axes`, `pc:deception`, `pc:unified` in the bank stores.

**The consequential reading, which is the opposite of the original worry.** The worry was that
existing 6-seed means might be invalid. **No claim anywhere takes a multi-seed mean** — grep over the
ledger, `claims.csv`, the claim notes and the thesis doc returns nothing. So nothing existing is
wrong. What the gap *does* mean is:

> **Every control comparison in `passA`, `passC` and `<root>` — axes vs `datapca`, vs `random`, vs
> `length` — exists at seed 42 only.** The raw-vs-axes comparison replicates across six seeds; the
> comparisons against controls do not replicate at all.

That is a genuine limitation for the thesis and belongs in `LIMITATIONS_AND_FUTURE_WORK.md`. It does
not weaken the negative result — a single-seed control comparison that comes out negative is still
negative — but it does mean the *size* of the control margin has no seed-level error bar in those
stores.

---

## 4. What this table settles

- **The random-direction null was never missing.** It is 1,000 draws per cell, persisted, per dataset,
  with p-values. The plan's "console log only" note is retracted.
- **The deception axis is inside its own null on 4 of 9 datasets at L40** — including HP-KR — and on
  all 4 sleeper datasets.
- **The sycophancy axis is significantly inverted** on real capitulation, and the assistant axis is
  the direction that reads it correctly.
- Every claim of the form "beats a random direction" can now cite `controls.csv` with a row id
  instead of a log line.
