# Wave 5 — lead QA record

Each batch nominates the 2 claims it is least sure of. The lead re-derives them **independently
from raw files**, with its own script, without using the batch's note as a route. This file records
what that re-derivation found. Waves 1–4 used the same rule and it overturned agent findings twice
(P8's random-control claim, M2's AD-3 — twice).

---

## W5E-ownerdoc (accepted 2026-09-23)

Batch result: 10 checked — 8 HOLDS, 2 PARTLY HOLDS, 0 fail.
Nominated: **O-040** and **O-082**.

### O-082 — CONFIRMED, and the batch's residual doubt is discharged

Claim (`docs/thesis/persona-coordinates-70b-results.md:702`): *"Dataset identity is 99.3% decodable
from these activations, against a label lift of +0.032."*

The batch verified both numbers but nominated the row because it had not refit the probe, leaving
open whether 0.993 was an overfit — a fair worry, since the features are 8,192-dimensional and there
are only 3,624 rows, where an in-sample 10-class fit would reach ~1.0 for free.

**It is not an overfit.** `src/assistant_axis_ndif/17_validate.py:119-123` reads:

```python
clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000, random_state=0))
lab_acc = cross_val_score(clf, X, y, cv=3, scoring="accuracy").mean()
big = [n for n, c in Counter(ds).items() if c >= 100]
mask = np.isin(ds, big)
dat_acc = cross_val_score(clf, X[mask], ds[mask], cv=3, scoring="accuracy").mean()
```

Both numbers are **3-fold cross-validated**, with the scaler inside the pipeline so it is refit per
fold — i.e. held-out, and free of scaler leakage. Verified in the stored result
`run-coords/validate_L40.json`: `dataset_acc = 0.9933774834437085`, `label_acc = 0.6462472406181016`;
with the log's majority baseline 0.614 the lift is **+0.032**. Both agree with the claim.

**Verdict: HOLDS, confidence HIGH.** Two documentation notes, neither affecting the number:
the lift is never written to the JSON (log-only — the script computes and logs it but stores only the
two accuracies), and the dataset probe silently restricts to datasets with ≥100 rows, so "99.3%
decodable" is over the ten large datasets rather than over all rows. Here all ten qualify, so nothing
moves; but a reader re-running this on a different store should know.

### O-040 — the batch's core finding CONFIRMED, one detail CORRECTED, and the error is bigger than reported

Claim (`docs/NOVELTY_CHECK_PROMPT.md:103-108`): *"A regex on denial phrasing alone reaches **0.91
AUROC** on a dataset where probes report **~0.87**, meaning the probe result is largely one lexical
bit."* Asserted as the project's own contribution at `docs/NOVELTY_ASSESSMENT.md:30` — *"Only the
regex-0.91-vs-probe-0.87 quantification is ours"* — and used as the premise of the pre-registration's
directional prediction at `docs/LAYER_PREREGISTRATION.md:93`.

Independent sweep: `audit/scripts/qa_w5e_O040.py`, over **all 113 stored matrices** containing
`harm-pressure-knowledge-report`, reading every `results[].auroc` cell and splitting by role.

```
HP-KR diagonal (trained AND tested on HP-KR):  n=2901  min 0.4388  median 0.9116  max 1.0000
HP-KR as transfer TARGET:                      n=28485 min 0.0126  median 0.5264  max 0.9928
HP-KR as transfer SOURCE:                      n=28485 min 0.0000  median 0.5135  max 0.9923

RAW diagonal only:                             n=153   min 0.8302  median 0.9663  max 0.9900
  L20 mean  median 0.9608     L40 mean  median 0.9800     L44 mean  median 0.9800
  L20 last  median 0.9629     L40 last  median 0.9736     L44 last  median 0.9800
```

**1. CONFIRMED — "probes report ~0.87" has no source.** No probe headline on HP-KR reads 0.87 under
any role. The number is not the diagonal (0.9663), not the transfer target (median 0.5264), not the
transfer source (median 0.5135), and not Liars' Bench's published figures for HP-KR.

**2. CORRECTED — the batch said a sweep "returns only incidental values"; that understates what is
there.** My sweep finds **1,151 cells in [0.86, 0.88]**, spread across every feature and both roles.
So "no such number exists in the matrices" is too strong and should not be written that way: cells at
0.87 are commonplace. The true statement is narrower — *no quantity anyone would call "the probe
result on HP-KR" equals 0.87.*

**3. ADDED, and this is the part that matters — the comparison is inverted.** The only reading under
which "probes report X on HP-KR" is a meaningful sentence is the diagonal, and there the raw 8,192-d
probe reads **0.9663 median, 0.96–0.98 at every headline layer**. The regex reads **0.9125** (F-01,
checked in batch P1, Llama rows). So **the probe beats the regex by ~0.054 at L40/mean.** The
sentence as written says the regex beats the probe by 0.04. The direction is wrong.

**4. Likely origin, corroborated by an independent audit route.** Session mining already established,
from the transcripts and by a different method, that *"HP-KR lexical 0.870 → 0.913 is a ROW-SET change
(all models → Llama-only), not a re-measurement"* (PROGRESS.md, session mining part 1). 0.870 is this
project's **own superseded regex** over all 7,068 rows. So claim 9 very probably compares the regex
against **itself** — old row set vs new — while labelling one side "probes".

**5. The underlying argument survives, in a form that must be restated.** "The probe result is largely
one lexical bit" is still supportable, and still interesting: a pure denial-phrasing regex reaches
**0.9125, or 94.4% of** what a full 8,192-dimensional probe reaches (0.9663) on the same rows. That is
a strong claim. What cannot be said is that the regex *beats* the probe.

**Verdict: PARTLY HOLDS — batch verdict upheld, severity raised.** Importance **HIGH**: the sentence
appears in three documents, and `NOVELTY_ASSESSMENT.md:30` records this quantification as the **only**
part of claim 9 that is the project's own contribution, so it is load-bearing for a novelty argument.
Confidence HIGH on 1–3 and on 5; MEDIUM on 4 (origin is inference from two converging routes, not a
recovered command). → DOC_FIXES.

---

## W5B-ledger (accepted 2026-09-23)

Batch result: 11 checked — 8 HOLDS, 3 PARTLY HOLDS, 0 fail. Nominated: **F-91** and **F-47**.

### F-91 — CONFIRMED, and the seed mixing runs one way

Claim: *"two independent methods put the optimum at 32–36, not 40."* I re-derived the disputed second
method from scratch with my own reader over the cached `run-premise/falsify/falsify_L*_s*.json`
(`real` key), all 21 layers x 3 seeds, 63 files:

```
   L        s7       s13       s42      mean
  28    0.8484    0.8591    0.8557    0.8544
  32    0.8826    0.8800    0.8945    0.8857
  36    0.8942    0.8887    0.8919    0.8916
  40    0.8957    0.8830    0.8828    0.8872
  44    0.8911    0.8777    0.8761    0.8816

argmax seed  7: L40 = 0.8957      argmax seed 42: L32 = 0.8945
argmax seed 13: L36 = 0.8887      argmax seed-MEAN: L36 = 0.8916
```

**Every figure the batch reported reproduces exactly.**

**What I add: the mixing is not neutral.** L32 is quoted at its *best* seed (s42), L36 at its *best*
seed (s7), but L40 at its *worst* (s42, 0.8828) — when L40's own best seed gives **0.8957, the
highest single value anywhere in the table**. Under any consistent rule:

| rule | winner | L40's place |
|---|---|---|
| best seed per layer | **L40 (0.8957)** | 1st |
| seed-mean (all 3) | L36 (0.8916) | 2nd, by 0.0044 |
| seed 42 alone (project default) | L32 (0.8945) | 3rd |

**And no layer in 32–44 is separable at this n:** the L36-L40 seed-mean gap (0.0044) is a third of
the seed spread *within* L32 alone (0.8800–0.8945 = 0.0145). The honest statement is "the optimum is
a plateau over roughly L32–L44 and this measurement cannot order it".

**Verdict: PARTLY HOLDS — batch verdict upheld, severity raised.** The crossgen half and the
no-reversal check stand. Importance **HIGH** — it bears on the draft's own hedge that "a layer
between 20 and 40 may do better" (draft topic 2).
*Caveat, for honesty:* `falsify_L40_s13.json` is the one file the audit's own interrupted run
rewrote (the 02:19 incident). Its measured impact (~0.001) cannot move a 0.0044 ordering, and the
pre-change original is in the Drive backup.

### F-47 — central fact CORROBORATED by an independent route; the chunk inference stays open

The batch's doubt — that "217 chunks" was a partial count of 267 sessions — rests on deleted
artifacts and is **unsettleable**. It should stay a flagged gap, not a guess.

The entry's load-bearing fact — `harm-pressure-choice` extracted at its **full 2,134 rows** instead
of the requested 800 — is corroborated from two places written independently:
`8_coords_ndif.py:subsample`'s docstring states HP-choice has exactly **134 deceptive rows**, and
F-56 records the realised sample as **134/666**. 134/2,134 = 6.28%, consistent across all three.
**Verdict: HOLDS on the central fact, NO RAW DATA on the chunk-count inference.**

---

## W5C-ledger (accepted 2026-09-23)

Batch result: 10 checked — 5 HOLDS, 5 PARTLY HOLDS, 0 fail. Nominated: **F-61** and **F-56**.
The batch also challenged an audit deliverable — handled first.

### PRIORITY — W5C's challenge to `audit/fix/F39_NOTE.md` §2 is UPHELD; the audit was wrong

W5C reported: *"`audit/fix/F39_NOTE.md` §2 is wrong: Table 1 on p.4 lists HarmPressureChoice, never
Mask."* This contradicted a conclusion the audit recorded as VERIFIED **and had already acted on** —
the note told the owner to "fix F-45, F-48 and F-54". I re-extracted the PDF independently
(`pypdfium2`, all 14 pages):

| term | mentions | pages |
|---|---:|---|
| `HarmPressureChoice` | **2** | **4 only** |
| `Mask` | 41 | 5, 6, 7, 12, 13, 14 — **never page 4** |

Table 1 (p.4) reads *AILiar, Roleplaying, ConvGame, InstrDec, **HarmPressureChoice***. The figures on
pp.5–7 use *ConvGame, InstrDec, **Mask**, AILiar, Roleplay*. **The paper declares one set and
evaluates another.**

**W5C is right and the audit's note was wrong on both halves of its sentence.** More: **the owner had
already documented this correctly, three times** — F-60 (`FINDINGS.md:1871-1884`), with a
mention-count table matching mine and the exact phrase *"true of Table 1, and misleading"*;
`FINDINGS.md:417-419`; and `docs/papers/persona-coordinates-notes.md` §8.1. The audit's note
contradicted her correct account and told her to correct entries that were not wrong.

- `audit/fix/F39_NOTE.md` §2 **corrected in place**; recommendation 3 **withdrawn**.
- The genuine owed fix is different and smaller: **F-48 (`:1418`) and F-54 (`:1659`) still assert the
  Table-1 framing with no pointer to F-60.** W5B found the same gap independently. → DOC_FIXES.
- Immaterial discrepancy: F-60 counts 33 `Mask` mentions, I count 41. Her conclusion is unaffected.

### F-61 — CONFIRMED exactly, and the mis-copy is now provable

Recomputed off-diagonal means myself from the committed matrices (110 pairs, 11 datasets, L40, mean
pooling, `paper` probe, k=5):

```
paper16 (run-crossgen/kd11/matrices/mean_L40_s42.json, bank axes_vectors_s3_fp32.pt)
    raw 0.6623   pc:unified 0.6715   datapca 0.6651
    pc:unified - raw      = +0.0093      <-- the entry says +0.006
    pc:unified - datapca  = +0.0064      <-- this is where +0.006 comes from
kd bank (.../banks/kd/matrices/mean_L40_s42.json)
    pc:unified 0.7380 ; against the ROOT raw 0.6623  ->  +0.0757   (entry: +0.076, correct)
```

**Two candidate origins, and — CORRECTED 2026-09-23 — neither is proven.**

*As first written, this section said the mis-copy "can be proved arithmetically". That was too
strong. Batches W5C and W6G independently surfaced a second candidate.*

- **Candidate A (adjacent column):** the entry divides kd's margin *over raw* by paper16's margin
  *over datapca*: 0.0757 / 0.0064 = **11.8**, matching "twelvefold". Requires no missing file.
- **Candidate B (superseded matrix):** F-61's console reading was **raw 0.6800 over 109 pairs**, and
  0.6861 − 0.6800 = **0.0061**, also ~0.006, giving 0.0757 / 0.0061 = 12.4 — also "twelvefold".
  That 109-pair matrix is attested in the session transcript but **exists in no surviving copy**:
  W6G checked git HEAD, the `3dfdb67` blob, the lab file and the audit cache — all md5-identical,
  **110 pairs, 0 NaN**. Likely cause of the change: the HP-KR top-up at `e9969af` (09-21 10:55).

**Both are consistent with the stated ratio; the audit cannot distinguish them.** Candidate A is more
parsimonious (no vanished file); Candidate B is independently attested in the transcript. **Report
both.** What is *not* in doubt, and is all the correction needs: on every surviving file the
like-for-like figure is **+0.0093 vs +0.0757 = 8.1x**, not twelvefold.

The batch correctly applied the documented trap: the kd bank store carries no `raw` column, so raw
must be joined from the **root** store.

**Verdict: PARTLY HOLDS.** Importance **HIGH** — F-61 is the entry telling the reader not to attribute
the kd gain to persona design, and its headline ratio is overstated by ~45%.

### F-56 — the batch is right: the stated mechanism is backwards

F-56 item 1 says *"`--sample` preserves the source class ratio rather than balancing it … it produces
a proportional draw. HP-choice is ~17% deceptive at source."*

The owner's own code says the opposite. `src/assistant_axis_ndif/8_coords_ndif.py:504`:

```python
"""Label-stratified subsample, deterministic under `seed`.
   … harm-pressure-choice's first 200 rows swallow all 134 of its deceptive rows …
   Round-robin over label buckets, so the result is as balanced as the data allows and
   degrades gracefully when a class is smaller than n/2."""
```

and the body round-robins over label buckets until `n`. **It is a balancer, not a proportional draw.**
134/666 is the balancer hitting the class ceiling — taking **all 134 deceptive rows that exist**, then
filling the rest with honest.

**So "~17% deceptive at source" is wrong, and wrong in the informative direction:** 16.75% is 134/800,
the *sample's* ratio. The source is 134/2,134 = **6.28%** (denominator from F-47 in the same ledger).
The sampler **raised** the deceptive share from 6.28% to 16.75% — the opposite of preserving it.

The entry's *practical* conclusions survive intact: the column is 5:1 imbalanced, it departs from the
paper's "all splits are label-balanced", and it is the least comparable of the five. Only the causal
story needs replacing.

**One correction to the batch:** its summary wrote the source ratio as "6.28% (134/2,000)". The
percentage is right; the denominator is 2,134. **Verdict: PARTLY HOLDS.**

---

## W5D-memory (accepted 2026-09-23)

Batch result: 12 checked — **12 HOLDS, 0 fail.** Nominated: **M-082** and **M-036**.

### M-082 — UPHELD against the audit's own correction. Our edit is wrong, and it is already applied in six documents

This is the second time in wave 5 that a batch has overturned audit work, and this one is worse
because the wrong text was **written by the audit on 2026-09-23 and applied across the repo**.

The correction currently in `memory/response-length-transfers-on-its-own.md` reads: *"The count is
**9 of 11** datasets, or **8 of 10** with `sycophancy_mt` excluded — **never "9 of 10"**."*
Six lines below it sits the note's own table, in which exactly one of ten datasets inverts.

I read the table's actual source file rather than the note — `run-coords/confounds_L44_last.json`,
key `length_only_auroc`, read-only on the lab PC:

```
ai_liar                        0.6130      mask                     0.6249
convincing-game                0.5068      roleplaying              0.4415   <- the only inversion
harm-pressure-choice           0.5017      sandbagging_v2           0.8544
harm-pressure-knowledge-report 0.7756      sycophancy_mt            0.6640
insider-trading                0.5945
instructed-deception           0.5604
```

**Ten datasets, nine above 0.5.** It reproduces the note's table to 3 dp.

**The audit's correction changed two things at once without saying so — the metric AND the dataset
set:**

| statement | metric | dataset set | true? |
|---|---|---|---|
| "9 of 11" | mean token gap | the 11 passB scored datasets | yes |
| "8 of 10" | mean token gap | those 11 minus `sycophancy_mt` | yes |
| **"9 of 10"** | **length-only AUROC** | **the 10 the length control was actually run on — which INCLUDES `sycophancy_mt` and EXCLUDES `harm-pressure-kr-matched`** | **yes** |

So the original "9 of 10" was **right all along**, of the quantity the note's table reports. The
audit's "never '9 of 10'" is false, and its parenthetical "(8 of 10 excluding `sycophancy_mt`)"
names a *different ten* from the one in the file — the confounds set keeps `sycophancy_mt` and drops
`kr-matched`.

The correction's closing line ("Two datasets run the other way: `convincing-game` −1.20,
`roleplaying` −4.10") also reads as contradicting the table directly above it, which calls
`roleplaying` "the only inversion". Both are right; neither names its metric.

**Verdict: the memory note HOLDS; the audit's correction to it DOES NOT.** Confidence HIGH — read
from the producing file, not from prose.

**Consequence, and it needs the owner's decision.** This wording was applied on 2026-09-23 to
`docs/NOVELTY_CHECK_PROMPT.md`, `docs/ONBOARDING.md`, `docs/thesis/persona-coordinates-70b-results.md`
(x2), `kd/docs/DESIGN.md` (x2), the memory note and the `MEMORY.md` hook. Every one of those now
carries a statement that is wrong about a set it does not name. **The audit will not re-edit them
without her word** — the earlier edit round was authorised for a specific list. Logged for DOC_FIXES;
recommended replacement: *"the positive class is longer in 9 of 11 datasets by mean token gap (8 of
10 excluding `sycophancy_mt`); a length-only probe scores above 0.5 on 9 of the 10 datasets the
length control was run on, `roleplaying` being the sole inversion."*

### M-036 — deferred, with the reason stated

W5D nominated M-036 because its cap half reuses P3's lab-only F-65/F-66 work rather than being
re-derived. That is a fair flag, but F-65 and F-66 were **already lead-QA'd in wave 1** (F-65: 26/27
figures exact, with the `pc:cap_only@2` 0.0142 exception found then; F-66: bootstrap re-run at 14/21
and 15/21 mean-pooled layers). Re-deriving them a third time buys little. **Recorded as
reused-evidence rather than re-derived**, which is the honest label; not a gap that changes a verdict.

---

## W5A-ledger (accepted 2026-09-23)

Batch result: 11 checked — 5 HOLDS, 5 PARTLY HOLDS, **1 DOES NOT HOLD** (F-49, already superseded by
F-50). Nominated: **F-45** and **F-78**.

### F-45 — already settled, by three independent routes

W5A nominated F-45 because its Table-1 reading overturns an audit artefact. It is the same question
W5C raised, and it is now confirmed **three ways**: W5A's `pypdf` extraction, W5C's independent
reading, and my own `pypdfium2` extraction (see the W5C section above). **F-45, F-48 and F-54 are
right about Table 1**; `audit/fix/F39_NOTE.md` §2 was wrong and is corrected; the owner's F-60 had it
right from the start. No further re-derivation needed.

### F-78 — CONFIRMED: the numbers are exact but belong to the other panel

F-78 (`kd/docs/FINDINGS.md`) writes: *"base models transfer **to** it markedly better than their chat
counterparts — Llama 70B **0.67 → 0.56**, 8B **0.69 → 0.56**, 3B **0.74 → 0.52**."*

I extracted Figure 4's body text from `papers/The Truthfulness Spectrum Hypothesis.pdf` p.5. Both
panels carry their own title in the text layer, so the association is unambiguous:

```
panel (a)  "FLED Probes Test on Sycophancy"   0.74 0.62 | 0.74 0.59 | 0.64 0.54
panel (b)  "Sycophancy Probes Test on FLED"   0.67 0.56 | 0.69 0.56 | 0.74 0.52
```

**F-78's three pairs are panel (b) — "Sycophancy Probes Test on FLED" — i.e. transfer *FROM*
sycophancy.** The direction its sentence describes ("transfer **to** it") is panel (a):
0.74→0.62, 0.74→0.59, 0.64→0.54. **W5A is right.**

**The conclusion survives; only the wording and the figure reference need fixing.** Both panels show
base > chat at all three scales, so "post-training separates sycophantic lying" is supported either
way — and F-78's use of it (as an independent mechanism for F-13) is unaffected. But an entry that
says "transfer to" while quoting "transfer from" will be caught by any reader who opens the paper.

*Not claimed:* I did not verify panel (c)'s base/chat assignment — the extracted similarity pairs
(0.51/0.56, 0.34/0.36, 0.23/0.06) do not obviously match the caption's "significantly higher in the
base models", and text-layer order is not reliable enough to call it. F-78 does not quote panel (c),
so nothing rests on it. Flagged, not chased.

**Verdict: PARTLY HOLDS.** Importance MEDIUM — it is a citation-accuracy fix in the related-work
argument, not a measured result.


---
---

# Wave 6 — lead QA record (2026-09-23)

Same rule. Wave 6 ran with the **lab PC unreachable** (Tailscale down on the laptop), so batches were
told to mark lab-only evidence `DEFERRED (lab unreachable)` rather than `NO RAW DATA`.

## W6C-briefing — 23 claims: 19 HOLDS, 3 PARTLY, 1 DOES NOT HOLD

### O-036 — CONFIRMED, and it had already spread into the audit's own Phase 3 file

`docs/COLLABORATOR_BRIEFING.md:654-655` closes with: *"...when the median over **the same pairs** is
0.722 and the sign-oriented mean is 0.735."* Those are **her** 110-pair values, set against **his**
0.732, which comes from his own 6x6 matrix. Different corpora.

Re-derived by the lead from his branch (`git show origin/feat/persona-pc-ood:.../persona_ood_L40/matrix_baseline.csv`,
30 off-diagonal cells):

```
mean of all cells                       0.7320   <- exactly the draft's 0.732
median over sources (of 6 row means)    0.7500
median of all 30 cells                  0.7887
sign-oriented mean, max(a, 1-a)         0.7850
cells below 0.5                         6 of 30, minimum 0.0598
```

**The argument survives and gets STRONGER on his own data.** On her corpus the median (0.722) is
*below* the mean, which does not support "you picked the lowest of these". On his, every alternative
is *above* 0.732 - so the sentence is true of his matrix and false of hers.

**One correction to the batch:** it reported his median as **0.7608**. The lead could not reproduce
that under either reading (0.7500 over sources, 0.7887 over cells); the sign-oriented 0.7850
reproduces exactly. **Use the lead's figures.**

**Fixed:** `audit/phase3/DRAFT_GAPS.md` A-13 repeated the same mixing and has been corrected in place,
with both corpora labelled and his table added. That is the **fourth** correction to the audit's own
work (after DF-49, DF-50, and P3-C's median-absolute).

### O-033 and O-023 — accepted as written, with the reason

O-033 (the format-matched control: "+0.03" exact, but "clears no significance test at any layer" is
false - content clears at L0/L12/L36 mean-pooled) rests on `audit/claims/F-94.md` rather than a fresh
read, because the cap significance files are lab-only. **F-94 was already lead-QA'd in wave 1**, so
the evidence underneath it is twice-checked; only this batch's re-reading of it is single-pass.
O-023's verdict turns on whether "the 1-d contrast" means the extended arm (holds) or both arms
(fails at L20 for the paper bank) - a reading question, correctly flagged, not a measurement dispute.
Both carry their own caveat in the claim notes.

## W6E-memory-q — 32 claims: 26 HOLDS, 6 PARTLY, **0 DEFERRED**

The lab outage cost this batch nothing: every Q-series number lives in `run-coords/*.json` in the
local git-tracked tree, clean vs HEAD, mtimes 09-16/17 - i.e. before the 09-22 RF-13 wave.

### M-057 — VERIFIED independently, and it is the strongest single result for the reframe

The batch reported that in the crossgen 80/20 transfer protocol raw beats axes at essentially every
layer. The lead re-derived it from `audit/phase2/layer_sensitivity.csv` (store `passA`, seed 42,
10 datasets, 80 layers), across **all six pooling x probe cells**:

```
pool  probe      layers  axes>raw   where              L40        L44
last  lr_reg10      80      2/80    [77, 78]         -0.0286    -0.0133
last  mms           80      5/80    [7,8,9,74,78]    -0.0164    -0.0270
last  paper         80      3/80    [9, 77, 78]      -0.0383    -0.0211
mean  lr_reg10      80      4/80    [0, 1, 78, 79]   -0.0651    -0.0666
mean  mms           80      2/80    [78, 79]         -0.0336    -0.0172
mean  paper         80      3/80    [77, 78, 79]     -0.0657    -0.0645
```

**Persona axes beat raw at 2-5 of 80 layers in every cell, and the wins sit at the extreme ends of
the stack** (L77-79, plus a few at L0-1 and L7-9) - depths nobody reports. At **both reported layers,
L40 and L44, axes lose in all six cells**, by 0.013 to 0.066.

This **independently reproduces P3-D's "2-5 of 80"** from a different route. Two agreeing derivations.

**One correction to the batch:** its "raw beats axes at every layer 0-77" is exact for the cell it
read (`mean/mms`, wins only at 78-79, L40 -0.034 / L44 -0.017 - matching its quoted figures), but
**not true across all six cells** - three of them have low-layer wins. The overall claim is unchanged
and slightly stronger stated as "2-5 of 80 in every cell".

### M-072 / M-073 — the batch's own most useful catch, accepted

"top10pct beats mean ... best +0.053 at 28" understates the effect: the largest zero-shot pooling
gain is **+0.1027 at L20**, and it collapses to **+0.005 at L40**. Anyone quoting +0.053 as the size
of the per-token effect is ~2x low at the literature layers. Recorded for DOC_FIXES.

### M-049 and M-077 — the batch's nominations, accepted with the reason

Both are **reading disputes, not measurement disputes**, and both were correctly flagged as such:
M-049 asks whether "0.13-0.17" should fail on an L44-mean of 0.1131 or be read as a rounded typical
magnitude; M-077 asks whether mixing aggregations inside one sentence is a defect or shorthand. The
underlying numbers reproduce in both. Neither changes a verdict that matters to the reframe, so the
lead did not spend a re-derivation on them.


## W6A-thesis1 — 17 claims: 13 HOLDS, 4 PARTLY, 0 whole-claim DEFERRED

### O-066 — the audit's OWN DF-49 edit was defective. CONFIRMED and fixed.

W6A reported that the path the lead added to `docs/thesis/persona-coordinates-70b-results.md:497`
on 2026-09-23 is wrong. **Verified — all three of its points hold:**

1. `run-coords/confounds_L44_last.json` **does not exist in the repo.** `ls` fails; the file is
   lab-only. A byte-identical copy is at `audit/cache/labpc/run-coords/confounds_L44_last.json`.
2. **The key path was incomplete.** Top-level keys are `['layer','pooling','k','bank','datasets',
   'splits','confounds','null']`; the values live at **`confounds.length_only_auroc[].auroc_from_length`**,
   not at a top-level `length_only_auroc`. The lead's earlier QA read it through a flattened `cat`
   and never saw the parent key.
3. The edit left a sentence two lines below — *"Only the sycophancy value (0.664) is also in a result
   file"* — which it then contradicted. **That sentence was itself false:** all ten values are in the
   file.

**Fixed in place:** the citation now gives the full key path, states that the file is lab-only, points
at the cached copy, and the false sentence is removed. **This is the fifth correction to the audit's
own work** (DF-49, DF-50, P3-C's median-absolute, W6C's A-13 mixing, now this) — and the second one
*inside the DF-49 repair itself.* Pattern worth naming: **a number carried into a new document without
re-opening the file it came from.**

### O-073 / A-1 — the paper's k, settled from the PDF by the lead

W6A nominated O-073 because it read the paper's k from `docs/papers/persona-coordinates-notes.md`
rather than the PDF. P3-A's A-1 rests on the same fact. **Independent `pypdfium2` extraction of all
14 pages settles it, and the answer is sharper than either agent had:**

```
k=1   12 occurrences   pages 12, 13, 14
k=3    9 occurrences   pages 6, 12, 13
k=5    0 occurrences   -
```

| figure | k | page |
|---|---|---|
| "Method Comparison · **Deception** · Llama 3.2 3B (Layer 14, **k=1**) [HPC excluded]" | 1 | 12 |
| "Method Comparison · **Deception** · Llama 3 8B (Layer 16, **k=1**) [HPC excluded]" | 1 | 13 |
| "Method Comparison · **Sycophancy** · Llama 3.2 3B (Layer 14, **k=3**)" | 3 | 12 |
| "Method Comparison · **Sycophancy** · Llama 3 8B (Layer 16, **k=3**)" | 3 | 13 |

- **A-1's premise holds:** the paper's *deception* setting is **k=1**. Both agents are right.
- **The paper uses two different k, one per behaviour** — deception 1, sycophancy 3. "The paper's k"
  is ambiguous unless the behaviour is named. Neither agent said this.
- **`k=5` appears nowhere in the paper**, independently confirming M2's AD-1 (the project default is
  a value the paper never uses) and O-073's comparison point.
- **A fourth independent confirmation of DF-50:** both deception figure titles read
  **"[HPC excluded]"** — the paper explicitly drops the dataset its own Table 1 declares. Table 1 and
  the figures disagree, exactly as the owner's F-60 said.

A-1 in `DRAFT_GAPS.md` has been annotated with all of this. Its page attribution needed no fix — it
cited no page.

### O-057 — accepted with its caveat

Which assistant reference the hf275 column uses is inferred from a 4-dp numerical match, not from a
recorded field or the producing script (which is lab-only). Correctly flagged. The consequence is
bounded and stated: conclusions move by ≤0.04 and do not change. Re-check when the lab returns.

### Operationally useful, outside the batch

W6A noticed that **`run-coords/` sits in the working tree with 37 result JSONs and is named nowhere
in the audit's file map as a laptop store.** That is why W6A and W6E needed no lab access at all.
**Other batches' DEFERRED items may be recoverable from it** — check `run-coords/` before deferring
anything for the rest of wave 6.
