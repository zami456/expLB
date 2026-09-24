# Apply sheet — the four sets you approved

_2026-09-24. **Superseded in method on 2026-09-24:** you asked for corrections to be applied as
**a new finding placed before the old claim**, not as find-and-replace, and for that to be the
convention from here. §1 has been applied that way and is now a record of what was done; §2 remains
the list for the collaborator's PDF, which has no patchable source in this tree._

## The convention, for everything from here

Never overwrite. Insert a dated block **immediately before** the claim being corrected, leave the
original text in place beneath it:

```markdown
> **CORRECTION <date> (<DF-id>, <phase>).** <what the text below gets wrong>
> <what is correct, with the number>
> **Source:** <path> — <key/row/line>
> **How the claim is updated:** <the sentence that should now be read>
> _The text below is superseded on this point and is kept for the record._
```

The change history stays legible in the document, and a correction that is itself wrong can be
walked back without guessing at the original — which has happened here before (DF-49, where the
audit's own length correction was wrong *and* already applied in six documents). The append-only
ledger is the same idea at its limit: never edit an F-entry, add one that corrects it.

---

## 0. Already done by this session (inside my write scope)

| set | what happened |
|---|---|
| **Kill DF-12 + fix the DF-85/F-119 pointer** | `audit/DOC_FIXES.md` DF-12 now marked **WITHDRAWN** with the git evidence; DF-85's pointer corrected to DF-12. F-119 is in the append-only ledger, so its mis-pointer is repaired by **F-123 §1** instead. |
| **Ledger: one consolidated F-123** | Appended to `kd/docs/FINDINGS.md` (+128 lines, now 6,271). Covers all 22 ledger DOC_FIXES. DF-71, DF-42 and DF-04 were **re-derived this session**; the other 19 are marked *(carried)*. |
| **DF-94** (coverage framing) | Targets `audit/` files only — I'll apply it at **6E**, after re-running `p15_coverage.py`, so the numbers and the wording land together. |

---

## 1. The urgent set — **APPLIED 2026-09-24** as correction blocks

All eleven blocks are in place across ten files. Nothing was overwritten. `cap/docs/review_page.html`
is a `str.format` template, so its block uses the page's own `.callout` class and contains no braces;
verified `str.format()` still succeeds on the file.

| DF | file | line anchored to | what the block says |
|---|---|---|---|
| **DF-66** | `docs/thesis/sarcasm-pilot-negative-result.md` | "Two gates were registered…" | G2's 0.65 was **not** pre-registered; it enters at `b34ff2e`, the commit reporting the failure. The design's final state pre-registers no numeric gate. Result survives either gate. |
| **DF-67** | `docs/REVIEW_REQUEST_POOLING.md` | "80/20 splits by item id" | Splits are over **rows**, not item id (`30_premise.py:106`); the 0.850/0.806 pair is two draws of one condition (0.8344 ± 0.0200); "~666" is the cap, realised 412–500. |
| **DF-31** | `cap/docs/review_page.html` | the "worse than a random guess" paragraph | 0.374 is the **deception** axis's PC1 and sits *inside* the random null; the sycophancy axis reads **0.313** and is significantly inverted. |
| **DF-31** | `.claude/skills/knowledge-denial-context/SKILL.md` | the "register hypothesis" paragraph (line 66; line 75 was already correct) | same, plus: 0.300 and 0.313 are different protocols. |
| **DF-31/68** | `kd/docs/NEXT.md` | the "Closed since…" table | the site missing from DF-31's own location lists. Gate still passes. |
| **DF-31** | `cap/docs/SARCASM_DESIGN.md` | "Our failures are below chance — 0.300 and 0.374" | protocol mix + axis mislabel; §3's argument unaffected. |
| **DF-68** | `docs/NOVELTY_ASSESSMENT.md` | "Merge 4 and 6" bullet | the −0.76 was withdrawn by F-74's own coupling null (P = 0.2135). |
| **DF-55** | `docs/NOVELTY_ASSESSMENT.md` | the demotion table | claim 9's "only original contribution" rests on the inverted comparison. |
| **DF-63** | `docs/thesis/persona-coordinates-70b-results.md` | "equivalent within about ±0.04" | declared margin is ±0.02; all four stored tests read `equivalent_within_margin = False`. |
| **DF-55 + DF-10** | `docs/NOVELTY_CHECK_PROMPT.md` | claim 9 | probe diagonal is **0.9663**, so the regex *loses* by 0.054; plus two superseded token counts. |
| **DF-55** | `docs/LAYER_PREREGISTRATION.md` | "regex 0.91 vs probe 0.87" | same inversion; the directional prediction is unaffected in direction. |

**DF-94** (the "424 of 692" coverage framing) targets `audit/` files only and is deferred to **6E**,
where `p15_coverage.py` is re-run — the numerator is still moving as Phase 6 adds verdicts, so
applying it now would bake in a figure that changes again.

### The original find-and-replace text, kept for reference

### 1.1 · DF-66 · `docs/thesis/sarcasm-pilot-negative-result.md:70` — **do this one first**

The pre-registration claim is git-false, and this is the first thing a referee auditing a
control-discipline thesis will check.

**NOW (line 70):**
```
Two gates were registered in the design document, with thresholds, before generation:
```

**REPLACE WITH:**
```
One gate was registered with a number before generation, and it was not this one. v1 of the design
set a numeric kill threshold (bag-of-words separability above ~0.75 = the bank is dead, commit
`3d80071`); v2 deliberately replaced it with a selectivity criterion and no number (`5ced550`, the
phrase surviving at `SARCASM_DESIGN.md:142`), so the design's final state pre-registers no numeric
gate. The 0.65 threshold used in the chain was fixed during the run — it enters the repository in
`cap/scripts/run_sarc_pilot.ps1` at `b34ff2e` (2026-09-21 21:48:54 +0600), the same commit whose
message reports the failure. **The result does not depend on which gate is used:** the measured
0.921 (unmarked v4+v5) and 0.947 (all variants) fail 0.65 and 0.75 alike.
```

Then change the table header cell **"threshold"** to **"threshold (see note above on provenance)"**,
and in the G2 row change **"bag-of-words AUROC ≤ 0.65"** to **"bag-of-words AUROC ≤ 0.65 (set
during the run)"**.

**Also, same document, `:22`** — the corrected sarcasm sentence pairs 0.313 (zero-shot sycophancy
axis) with "~0.300 on knowledge-report", which is the **trained 4-d `axes` stack** as source on full
HP-KR. Two protocols in one sentence; name them (DF-68, third bullet).

**And `§5`** quotes the superseded 02:29 status file (**0.947**) verbatim while **§3.2** already
quotes the current **0.921**. Pick one; the document currently carries both (DF-70).

---

### 1.2 · DF-67 · `docs/REVIEW_REQUEST_POOLING.md` — **the document you send out**

**NOW (lines 17–18):**
```
ours). Probes are logistic regression on 8192-d activations, class-balanced training, 80/20
splits by item id.
```

**REPLACE WITH:**
```
ours). Probes are logistic regression on 8192-d activations, class-balanced training, 80/20
splits over **rows** (`src/assistant_axis_ndif/30_premise.py:106` uses `rng.permutation(n)`;
there is no grouping by item id). Row-level splitting inflates the in-domain ceiling in the table
below; the transfer numbers are unaffected.
```

**Second fix, same document — one condition printed as two numbers.**
`:32` calls the budget-matched pooled arm **0.850**; `:63` calls it **0.806**. Both are single draws
of one condition: 20 draws give **0.8344 ± 0.0200**.

At **`:32`**, replace `| **pooled, same row budget (~666)** | **ten others** | **0.850** |` with:
```
| **pooled, same row budget (~666)** | **ten others** | **0.834 ± 0.020** (20 draws) |
```
At **`:63`**, replace `row (0.806) for that reason` with `row (0.834 ± 0.020 over 20 draws) for
that reason`.

**Add to the weaknesses list** — this is the honest version of weakness 1:
```
   The margin Result 2 rests on (+0.028) is of the same order as its comparator's own draw noise
   (±0.020 over 20 draws), so it should be read as suggestive rather than established.
```

**Third:** `~666` is the **cap** in `30_premise.py:167`, not the realised pool — actual is
**412–500 rows, mean 449** (DF-33). Either say "capped at 666, realised 412–500" or use 449.

---

### 1.3 · DF-31 · the four residual `0.374` sites

Six documents already carry dated "Correction, 2026-09-23" annotations. These four were missed.
The substance: **0.374 is the paper16 *deception* axis's PC1** (its contrast reads 0.409); the
paper16 **sycophancy** axis reads **0.313**. Only **0.313** lies beyond the random-direction null
(p 0.001–0.008) — **0.374 sits inside it**, so "worse than a random direction" is *not* supported
for 0.374 and *is* supported for 0.313.

**(a) `cap/docs/review_page.html:162` — highest priority: this is the rendered page reviewers read,
and the `.md` versions were corrected while the HTML was not.**

NOW:
```html
one dataset that actually measures sycophancy it scores <strong>worse than a random
guess</strong> &mdash; 0.374 where 0.5 is chance, while a randomly chosen direction gets 0.54.</p>
```
REPLACE WITH:
```html
one dataset that actually measures sycophancy it scores <strong>0.313</strong> where 0.5 is
chance &mdash; significantly <em>inverted</em>, beyond a 1,000-direction null at p 0.001&ndash;0.008.
(An earlier version of this page put 0.374 here and called it worse than random. 0.374 is the
paper16 <strong>deception</strong> axis's PC1, not the sycophancy axis, and it sits <em>inside</em>
the random-direction null.)</p>
```

**(b) `.claude/skills/knowledge-denial-context/SKILL.md:66`** — line 75 of this same file is already
corrected; 66 was missed.

NOW: `(0.300) and \`sycophancy_mt\` (0.374) — and on both a **random direction beats them**. The`
REPLACE WITH:
```
(0.300) and `sycophancy_mt` (the paper16 **sycophancy** axis reads **0.313**; the **deception**
axis's PC1 reads 0.374) — and on `sycophancy_mt` the sycophancy axis is significantly *inverted*
against a 1,000-direction null (p 0.001–0.008). The
```

**(c) `kd/docs/NEXT.md:15`** — the location DF-68 flagged as absent from DF-31's own list.
Replace `the register axis's 0.374` with
`the paper16 sycophancy axis's 0.313 (the deception axis's PC1 reads 0.374)`.

**(d) `cap/docs/SARCASM_DESIGN.md:48`** — `Our failures are **below** chance — 0.300 and 0.374.`
Replace with `Our failures are **below** chance — 0.300 (trained 4-d `axes` stack as source on full
HP-KR) and 0.374 (paper16 **deception**-axis PC1 on `sycophancy_mt`; the sycophancy axis reads 0.313).`

---

### 1.4 · DF-68 · `docs/NOVELTY_ASSESSMENT.md:103` — a withdrawn mechanism still nominated to lead with

**NOW (line 103):**
```
- **Merge 4 and 6.** 4 = mechanism (ceiling artifact, corr −0.76). 6 = the scaling trend.
```
**REPLACE WITH:**
```
- **Merge 4 and 6.** 4 = mechanism, 6 = the scaling trend. **Do not state the mechanism as
  "ceiling artifact, corr −0.76":** F-74, whose verdict is HOLDS, measured −0.7684 *inside its own
  coupling null* (analytic −0.7462, P = 0.2135) and explicitly withdrew it as evidence; the clean
  uncoupled test (slope −0.100, r −0.067) does not support it either. State the ceiling mechanism
  without the correlation, or drop claim 4 from the lead.
```
§4 of this file nominates Claim 4 to "lead with" — revisit that once the mechanism is restated.

---

### 1.5 · DF-63 · `docs/thesis/persona-coordinates-70b-results.md:740` — post-hoc margin widening

`:513` pre-registers **±0.02**. `:741` concludes **±0.04**. All four stored tests read
`equivalent_within_margin = False` (`run-coords/significance.json`, comparisons[7]–[10]).

**NOW (lines 740–741):**
```
   Across datasets the four equivalence tests have 90% CIs from −0.038 to +0.033, so the two are
   equivalent within about ±0.04.
```
**REPLACE WITH:**
```
   Across datasets the four equivalence tests do **not** clear the pre-registered ±0.02 margin —
   all four read `equivalent_within_margin = False` in `run-coords/significance.json`
   (comparisons[7]–[10]; diffs −0.0128 / +0.0061 / +0.0036 / −0.0092). The median gap is 0.012 and
   the 90% CIs span −0.038 to +0.033, so the two layers are **close, but formal equivalence at the
   declared ±0.02 margin is not established.**
```

---

### 1.6 · DF-55 · `docs/NOVELTY_CHECK_PROMPT.md:112` — the comparison is inverted

Claim 9 compares the regex against what appears to be **this project's own superseded regex**
(0.870), while calling one side "probes". The raw probe diagonal on HP-KR is **0.9663 median**
(0.96–0.98 at every headline layer, swept over all 113 stored matrices). **The regex loses by
~0.054; it does not beat the probe.**

> **UPDATE 2026-09-24 (DF-105, P17).** **0.9597 is unverified:** the regex behind it is recorded nowhere. A "does-not-affirm" reconstruction gives 0.9413, which is still above the draft's 0.919 (D-022). 0.9125 is verified. **Source:** `audit/p17/correction_blocks_check.csv`; `audit/p17/C_REPORT.md`; `audit/p17/QA_REPORT.md`.

**NOW (lines 111–113):**
```
single lexical feature. A regex on denial phrasing alone reaches **0.91 AUROC** on a dataset where
probes report ~0.87, meaning the probe result is largely one lexical bit rather than a
representation of deceptive intent.
```
**REPLACE WITH:**
```
single lexical feature. A label-free regex on denial phrasing alone reaches **0.9125 AUROC**
(0.9597 for a second phrasing) on the 2,139 Llama-3.3-70B rows, which is **94.4% of** what a full
8,192-d probe achieves in-distribution on the same rows (**0.9663** median raw diagonal, swept over
113 stored matrices). The probe result is therefore largely one lexical bit rather than a
representation of deceptive intent.
```
**Note:** `docs/NOVELTY_ASSESSMENT.md:30` records this as the *only* part of claim 9 that is the
project's own contribution, and `docs/LAYER_PREREGISTRATION.md:93` uses it as the premise of a
directional prediction. **Both need the same correction.** The argument survives restated; what
cannot be said is that the regex beats the probe.

**Same file, `:110` (DF-10)** — two superseded token counts still stand: "505 vs 310 tokens on one,
245 vs 96 on another" → **`sandbagging_v2` 529.4 vs 340.4** and **`mask` 283.2 vs 176.4**.

---

## 2. The draft set — 18 corrections to the paper

**These are not file edits.** The target is the collaborator's PDF, which has no patchable source in
this repo. This is the list to fold into the paper or send him. Full derivations in
`audit/DOC_FIXES.md` under each DF id.

> **CORRECTION 2026-09-24 (DF-101, P17).** Excluding HP-KR, the ext_axis1 − ext_pc3 margin reaches **+0.0633 at L56**, so it is not within ±0.06 (at 4 dp), and it changes sign **6** times, not 5. **Source:** `src/probes/deception-detection/results/persona_ood_ext_L{20..79}/summary.csv` @ 16def2de; `kd/docs/FINDINGS.md` F-128; `audit/p17/QA_REPORT.md` check 3.
>
> **UPDATE 2026-09-24 (DF-105, P17).** **0.9597 is unverified:** the regex behind it is recorded nowhere. A "does-not-affirm" reconstruction gives 0.9413, which is still above the draft's 0.919 (D-022). 0.9125 is verified. **Source:** `audit/p17/correction_blocks_check.csv`; `audit/p17/C_REPORT.md`; `audit/p17/QA_REPORT.md`.

| # | DF | where | correction |
|---|---|---|---|
| 1 | DF-17 | abstract | 0.904 / 0.916 are **`ext_axis1`**, which uses the owner's knowledge-denial personas. The published 8-vs-7 contrast reads **0.791 / 0.832** — about **45% of the margin** comes from her bank. Fix the attribution. |
| 2 | DF-20 | abstract | "0.09 below the contrast" — the comparison described is `ext_pc1` **0.7680** vs `ext_axis1` **0.9158** at L40, a gap of **0.148**. No pc-vs-axis pairing in his own L40 summary gives 0.09 ± 0.005. |
| 3 | DF-22 + DF-46 | abstract | "lifts transfer by 0.18–0.21" is a **sign artifact**: the HP-KR training row at L20 is wholly inverted (row mean 0.2091 vs 0.7906–0.7908). Corrected: **+0.029 / +0.002 / −0.005 / −0.003**, matching his own `zero_shot.csv`. At L40/L44 no row inverts. **Describe it as HP-KR inverting — the project's own documented finding — not as his error.** The genuine gain is on HP-KR (+0.6159). |
| 4 | DF-23 | abstract | "six Liars' Bench datasets" is **five** — `repe_honesty` is not one, and his own file labels that row `n_datasets=5`. **L44 was also run and the abstract's "layers 20 and 40" omits it.** |
| 5 | DF-93 | tables | Every `dim=1` arm's trained off-diagonal **equals** its zero-shot AUROC (spread 0.000 at all 16 layers). Label them **zero-shot**, and compute the interval over the **5 test datasets**, not 30 cells. "A 1-D zero-shot direction beats a trained 8192-d probe" is the stronger claim anyway. |
| 6 | DF-79 | layer choice | The draft reports L40; his own sweep peaks at **L36 = 0.9275** (L40 = 0.9158), and `denial_axis1` also peaks at 36. Report L36, or say L40 was pre-specified and give L36 beside it. |
| 7 | DF-80 | Contributions | "outscores … at every depth" is **false at L64 (−0.0010), L68 (−0.0205), L72 (−0.0146)**. → "at 13 of 16 layers" (11/16 excluding HP-KR). |
| 8 | DF-81 | Contributions | Quote the margin **excluding HP-KR** too: +0.0086 (L20), +0.0012 (L24), within **±0.06 everywhere**, crossing zero five times. |
| 9 | DF-21 | Contributions | "the principal components do not contribute" is contradicted by his own files for the **published** bank — `pc1 − axis1` = **+0.0158 / +0.0274 / +0.0246**. True only of the kd-extended bank; say so. (At L20 it reverses when HP-KR is excluded: −0.0033.) |
| 10 | DF-26 | results | "within 0.03 of the privileged upper bound" is **0.0310** (0.8990 vs 0.93), and **0.0432 at L20**. On balanced accuracy at 1% FPR — the metric the benchmark leads with — the same pair is **0.694 vs 0.890, a gap of 0.196**. |
| 11 | DF-27 | results | "the pooled baseline inverts to 0.124 (L20) and 0.523 (L40)" mixes weightings: **0.1243 is `balanced`, 0.5230 is `naive`**. **0.523 is above chance — not an inversion.** Held fixed at `balanced`, L40 is 0.4887. |
| 12 | DF-15 | Table 9 | "the top-10% token read is best" is contradicted by his own Table 9 — at L40 the single-maximum read wins on balanced accuracy (0.750 vs 0.737) and recall (0.548 vs 0.531). |
| 13 | DF-14 | §on depth | "the base probe inverts on one dataset at each depth" — at **L20 two** do: HP-KR (0.109) and harm-pressure-choice (0.471). |
| 14 | DF-13 | §6 | "every method has recall near zero on HP-KR" is false for the **Follow-Up Probe on 70B (0.37**, bought with an 84% FPR). Holds only of the **cross-model averages**. |
| 15 | DF-24 | §on sycophancy | Drop **"independently constructed"** — `persona_ood.py:911` fits the sycophancy geometry on the **same bank object** (the owner's 36-role kd vectors), and his own comment records **\|cos\| = 0.54**. |
| 16 | DF-76 | p.12–13 | Delete the future-work promise that the wording-controlled test "has not been run". **It ran three times** — F-03/F-72/F-76 (2026-09-19), F-115 (contamination closed) and F-116. State the outcome: **+0.066, CI [−0.069, +0.189], P = 0.161** at the pre-registered layer, **inverting to −0.2545 at L20**, with the axes **below chance** on the knowledge-matched control while an ordinary probe reads 0.999 in-distribution. |
| 17 | DF-77 | §on controls | Name **both** hpkr slices: `text_matched` (258 rows, wording held constant) and **`knowledge_matched` (2,001 rows, knowledge held constant)** — the second is **7.8× larger** and was never run until 2026-09-23. Together they are the 2×2 the design always implied. |
| 18 | DF-72 | every HP-KR number | State the lexical baseline beside it: a label-free regex reads **0.9125** / **0.9597** on the 2,139 Llama-3.3-70B rows, at or above the draft's own headline **0.919**. Stop calling the wording confound "not ruled out" — on these numbers it is **not currently distinguishable**. |
| +1 | **DF-84** | late-layer rise | **Promote from P2.** F-120 solved it: the full-rank probe *declines* at the top, the axes only *recover* toward their mid-stack level, and **a random direction gains +0.127** over the same span — `axes − random` is **+0.2296 at L40 vs +0.1087 at L79**, so the method is *worse* at the top. One sentence converts the most obvious reviewer question into a demonstration that the controls work. |

**Do not apply DF-12** — withdrawn; see §0.

---

## 2b. The four claims that must change before submission — paste-ready text

_The manuscript body is not in this repository (only `paper/generated/` is on `origin/paper`), so
these are sentences to hand the collaborator. Every number is re-derived; sources in
`audit/RESULTS_CHAPTER.md` §9 and `kd/docs/FINDINGS.md` F-125/F-126._

### (a) Section 2, Related work — "recalls near zero for every method"

**Now:** *"On HP-KR … the benchmark reports recalls near zero for every method."*
**False as written.** Liars' Bench Table 6/7/8 p.54–56, row *Llama 3.3 70B*, column HP-KR: Mean
Probe recall **0.00**, but Follow-Up Probe **0.37** and the privileged Upper-Bound Probe **0.97**.

**Replace with:**
> On HP-KR the benchmark's Mean Probe has recall 0.00 on Llama-3.3-70B, and 0.00 averaged across
> models. Its Follow-Up Probe reaches recall 0.37 there, but at a realised false-positive rate of
> 0.84 — a detector that fires on most honest answers. Only the privileged Upper-Bound Probe,
> trained on HP-KR's own labels, recovers usable recall (0.97, at 8% FPR).

**Why this is better than the sentence it replaces:** the FPR figures are the point being made, and
they are the same critique this paper's own layer-20 arm attracts (88.3% FPR). Making it
symmetrically lets the paper own the weakness rather than be caught by it.

### (b) Sections 6 and 7 — the wording-controlled test is promised twice and has been run

**Now:** *"The bank was designed for the cleaner test, a contrast between feigned and sincere
ignorance in which both sides deny, and we have not run it."* (repeated as future work in §7.)

**Replace with:**
> The bank was designed for the cleaner test — a contrast between feigned and sincere ignorance in
> which both sides deny — and we have now run it. At the pre-registered layer the margin over raw
> is +0.066 (95% CI [−0.069, +0.189], P = 0.161); it inverts to −0.2545 at layer 20. On the larger
> knowledge-matched control (2,001 rows, 7.8× the wording-matched slice) the persona axes transfer
> **below chance**, while an ordinary probe separates the same rows at 0.999 in-distribution. The
> cleaner test does not rescue the contrast, and we report that rather than leave it open.

**Delete the §7 future-work bullet that repeats the promise.**

**Why:** a referee who finds the result exists will ask why it was withheld. Reporting a negative
result the design was built to produce is the control discipline the paper argues for.

### (c) Section 7, Future work — Gender-Secret is no longer open

**Now:** listed among fine-tuned model organisms still to be tried.

**Replace with:**
> We have since run Gender-Secret through the same pipeline: the knowledge-denial direction sits at
> chance (0.436–0.623) where both classes deny. With the two controls above, three independent
> wording-matched tests now converge on the same answer, and they should be read as one finding
> rather than three.

**Keep Soft-Trigger as future work** — that one is genuinely untouched.

### (d) Section 7 / Appendix D — the best token read is the maximum, not the top tenth

**Now:** *"For detection, the top tenth of response tokens is a better read than the mean."*
True against the mean, but not the best available.

**Replace with:**
> For detection, reading the single highest-scoring response token beats both the mean and the top
> tenth: at layer 40 it reaches balanced accuracy 0.750 and recall 0.548, against 0.737 / 0.531 for
> the top tenth and 0.694 / 0.413 for the mean. On ranking the ordering reverses at layer 44, where
> the maximum leads on AUROC (0.914) but the top tenth leads on balanced accuracy (0.715 vs 0.701).

**Why:** the draft's own Table 9 contains the maximum column, so a reader checking the table will
see it. (This is DF-15, now confirmed at Level 1.)

### And one scope word, section 4

*"that happens only for the published contrast trained on HP-KR at layer 20"* — correct for the
three headline contrasts, too narrow once the sycophancy contrast and the published PC1 are
included. **Seven (arm, layer) cells invert**, and Table 11's printed off-diagonal row is depressed
at all three layers as a result. Either scope the sentence to the three contrasts explicitly, or
add a clause noting Table 11.

---

## 3. Still open, needing your decision rather than an edit

| DF | decision |
|---|---|
| DF-48 | `scale/scripts/run_scale_chain.ps1:197-203` writes both bank variants to `axes_$tag.pt`; `$vt` never reaches the path, so the `vectors_all` bank is built and immediately overwritten. One-char fix (`axes_$vt.pt`) — but it would make future runs emit two banks where `run_scale_crossgen.ps1:53` expects one. **My recommendation: leave the code, document the consequence.** |
| DF-74 | Same defect in the 8B chain. **Any 8B "unfiltered vs score-filtered" comparison that used `axes_8b.pt` compared s23 against itself.** Same call. |
| DF-39 | F-32's acceptance table certifies a bank generation that never produced a rollout. **Replace the table with the live-bank numbers and rest the no-confound argument on teacher forcing** (turn 1 byte-identical across all 65 arms), which is structural and verifiable. Recorded in F-123 §3. |
