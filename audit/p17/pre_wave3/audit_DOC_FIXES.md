# DOC_FIXES — documentation corrections the audit has confirmed

**Status: accumulating (started 2026-09-23, during Phase 2). Nothing here has been applied.**
The brief forbids the audit from editing existing documents, so every entry is a proposal for the
owner. The one exception already taken was owner-authorised: F-110 in `kd/docs/FINDINGS.md`.

**Inclusion rule.** An entry appears here only when the corrected value was re-derived from raw
data during this audit, with the path recorded. A discrepancy that is merely suspected lives in
`audit/PROGRESS.md` under RED FLAGS, not here. Entries marked **[lead QA]** were re-derived a
second time, independently, by the lead.

**Priority key.** P1 = wrong number or wrong quantifier a reader would rely on. P2 = stale or
imprecise wording that does not change a conclusion. P3 = housekeeping.

---

## A. The owner's ledger — `kd/docs/FINDINGS.md`

### DF-01 · P1 · F-04's "11-dataset run" labels numbers computed on 10 datasets **[lead QA]**
- **Where:** the F-04 entry (~line 80).
- **Now:** presents `axes − raw` **+0.0147** (CI [−0.170, +0.187], q 0.9311) and `axes − datapca`
  **+0.0264** (q 0.8546) as the result of an 11-dataset run.
- **Correct:** those are **block "9"**, which `src/assistant_axis_ndif/27_crossgen.py:51,781` defines
  as `[d for d in DATASETS if d != "sycophancy_mt"]` — **10 of the 11 datasets**. On the full 11
  (block "10") both signs flip: **−0.0215** and **−0.0243**.
- **Fix:** say "10 datasets (sycophancy_mt excluded)", and state the full-11 values beside them.
- **Evidence:** `run-crossgen/kd11/significance.json`, tests at layers 40/44, BH re-derived per block.
- **Note for every later reading:** in the kd11-era stores the block labels "10" and "9" are
  *historical dataset counts* from before the 11th dataset was added. Today they mean **all 11**
  and **all but sycophancy_mt**. Any entry that reads "block 9" as "9 datasets" is misreading it.

### DF-02 · P1 · F-04's one significant test is a random projection losing, not a persona winning **[lead QA]**
- **Where:** the F-04 entry, the sentence reporting 1 significant test out of the family.
- **Now:** "1 significant" stands beside the persona comparisons, inviting the reading that the
  persona method achieved it.
- **Correct:** the single test clearing BH in that family is **`random@2 − raw`, last pooling, L44,
  `mms` probe, diff −0.2287, q 0.0192** — a *random projection performing worse than raw*. No
  persona comparison in the family is significant.
- **Evidence:** as DF-01; the family is 16 comparisons × 3 probes × 2 poolings × layers {40,44}
  = 192 per block, BH within block (the file's own `note` field states the rule).

### DF-03 · P1 · F-77 and F-81 print 37/80 where the measured counts are 28/80 and 31/80 **[lead QA]**
- **Where:** `kd/docs/FINDINGS.md:2934` (F-77) and `:3233` (F-81), the table row "layers with a
  positive margin", kd column.
- **Now:** `37/80 (46%)`, set beside the cap bank's `21/21 (100%)`.
- **Correct:** the kd matched slice beats **raw at 28 of 80** layers and **`datapca` at 31 of 80** —
  which is exactly what **F-75 already says**. 37 is not a layer count under any reading; it is a
  bootstrap median **fraction** of positive layers (≈0.46) rendered as a count.
- **Fix:** use 31/80 in the `datapca` comparison (the column's own control), and drop the "(46%)".
- **Why it matters and which way:** the error runs **against** F-77's own argument — the true kd
  count is *lower*, so the contrast with the cap bank's 21/21 gets **stronger**, not weaker.
- **Evidence:** `run-crossgen/kd11/passA11/banks/kd_s23/matrices/mean_L{0..79}_s42.json` against
  `…/passA11/matrices/…`, train row `harm-pressure-kr-matched`, 10 targets; `audit/scripts/qa_p7.py`.

### DF-04 · P2 · F-73's internal restatement of its own AUROCs
- **Where:** the F-73 entry (~line 2629 ff.).
- **Now:** "stated 0.553 vs 0.665 confirmed to three decimals".
- **Correct:** the measured values are **0.5555 and 0.6623**, i.e. 0.556 / 0.662. F-73's own closing
  line states these correctly; only this sentence is wrong.
- **Evidence:** `run-crossgen/kd11/matrices/mean_L40_s42.json`, 110 off-diagonal cells, `paper` probe.

### DF-05 · P2 · F-73 calls a non-zero gap "exactly zero"
- **Now:** `harm-pressure-choice`'s length gap is described as exactly zero.
- **Correct:** **+0.0035 tokens** (1.0075 vs 1.0040). Negligible, but not zero.
- **Evidence:** `run-coords/passB/harm-pressure-choice/rows.jsonl`, `span_len` by `label`.

### DF-06 · P2 · F-73's "median gap 3.7 tokens" is a median of absolute gaps **[lead QA]**
- **Where:** F-73, "median gap across the eight datasets the reviewer could see: 3.7 tokens".
- **Correct:** 3.689 is the median of **|gap|** (`audit/scripts/claims/P7_lab.py:71`), which counts
  `roleplaying`'s **−4.10** as +4.10. The **signed** median over the same eight is **+2.63**.
- **Why it matters:** F-73's point is that the length effect is *small*; the absolute-value framing
  inflates it and so argues against F-73's own conclusion. Either label it "median |gap|" or use 2.63.

### DF-07 · P2 · F-105's "none of them is persisted" is out of date
- **Correct:** the persistence fix landed in **the same commit as F-105 (`6c9a5cf`)**, and the
  2026-09-22 15:55 re-run wrote `lexical_subsets` and `lexical_features` into `marker_report.json`.

### DF-08 · P2 · F-105's "all ten words appear in the top-25"
- **Correct:** **11 of 12** do, and `definitely` is a `MARKER_TOKEN` — which means F-93's feature
  list came from the **unstripped** fit. Worth stating, since it bears on what the feature list shows.

---

## B. The length claim, repeated across five documents and the memory index

### DF-09 · P1 · "the positive class is longer in 9 of 10 datasets" is true of no slicing of the data **[lead QA]**
- **Measured**, recomputing every dataset's mean `span_len` by label from
  `C:/Users/user4/expLB/run-coords/passB/<dataset>/rows.jsonl`:
  **9 of 11** datasets with `sycophancy_mt`, **8 of 10** without it (block 9). **"9 of 10" matches
  neither.** The two datasets where the positive class is *shorter* are `convincing-game` (−1.20)
  and `roleplaying` (−4.10).
- **Also measured there:** `roleplaying` carries **84 rows with `label: null`**, which every reader
  of this store silently drops. Worth stating wherever that store is used.
- **Locations to correct:**

| # | file:line | current wording |
|---|---|---|
| 1 | `docs/NOVELTY_CHECK_PROMPT.md:110` | "the longer one in 9 of 10 datasets (505 vs 310 tokens on one, 245 vs 96 on another)" |
| 2 | `docs/ONBOARDING.md:78` | "the deceptive class is longer in 9 of 10 datasets" |
| 3 | `docs/thesis/persona-coordinates-70b-results.md:497` | "the longer one in 9 of 10 datasets" |
| 4 | `docs/thesis/persona-coordinates-70b-results.md:1105` | "longer in 9 of 10 datasets" |
| 5 | `kd/docs/DESIGN.md:176` | "the positive class is longer in 9 of 10 datasets; off-diagonal 0.553" |
| 6 | `kd/docs/DESIGN.md:202` | "longer in 9 of 10 datasets (`response-length-transfers-on-its-own`)" |
| 7 | `kd/docs/FINDINGS.md:5028` | "where the positive class is longer in 9 of 10 datasets" |
| 8 | memory `MEMORY.md:19` and `response-length-transfers-on-its-own.md:3` | "longer in 9 of 10 datasets" |

### DF-10 · P1 · `docs/NOVELTY_CHECK_PROMPT.md:110` also carries two superseded token counts
- **Now:** "505 vs 310 tokens on one, 245 vs 96 on another".
- **Correct:** `sandbagging_v2` is **529.4 vs 340.4** and `mask` is **283.2 vs 176.4**. F-73 already
  corrected the `mask` pair; this document still carries the old one.

### DF-11 · P3 · do not conflate two different "9 of 10" statements
`kd/docs/FINDINGS.md:2244` ("cap beats raw on **9 of 10** source datasets at L40") and the same
phrase in `.claude/skills/knowledge-denial-context/SKILL.md:75` and
memory `persona-axes-encode-register.md:48` are a **different claim** — cap wins per source, not
response length. DF-09 does not apply to them. They are checked separately under F-65/F-66.

---

## C. The collaborator's draft and abstract

### ~~DF-12~~ · **WITHDRAWN 2026-09-24 — DO NOT APPLY. The claim in this entry is false.**
- **What it said:** that the draft's `paper/make_numbers.py` "exists on **neither branch**".
- **Why it is withdrawn:** the script **is on `origin/paper`**, at
  `src/probes/deception-detection/paper/make_numbers.py` — 28,232 bytes, commit **`8fd071b2`**
  (2026-09-22, zami456), alongside `paper/make_figures.py`. Verified 2026-09-24 with
  `git ls-tree -r --name-only origin/paper`; count is 1 on `origin/paper` and **0** on both `asix`
  and `origin/feat/persona-pc-ood`, which is why the original check missed it.
- **The draft's path is correct as written.** `src/probes/deception-detection/` is a vendored
  sub-project with its own root, so `paper/make_numbers.py` resolves correctly relative to it.
- **Applying this entry would put a false claim about the collaborator into a document.**
- **Superseded by:** F-119 (`kd/docs/FINDINGS.md:5951`) and DF-85 below.
- **Why it survived as live:** DF-85 and F-119 both withdraw *"DF-71's reproducibility half"*.
  **DF-71 has no reproducibility half** — it is the F-61 `+0.006` script-bug entry. The entry
  being withdrawn was always **DF-12**, and the mis-pointer left DF-12 reading as live.

### DF-13 · P1 · "every method has recall near zero on HP-KR" (D-013/D-037)
- **Correct:** false for the **Follow-Up Probe on Llama-3.3-70B (0.37**, bought with an 84% FPR).
  The statement holds only of the **cross-model averages**, and should say so.

### DF-14 · P1 · "the base probe inverts on one dataset at each depth" (D-124)
- **Correct:** at **L20 two** datasets sit below 0.5 — HP-KR (0.109) and harm-pressure-choice (0.471).

### DF-15 · P1 · "the top-10% token read is best" (D-207)
- **Correct:** contradicted by **his own Table 9**: at L40 the single-maximum read wins on both
  balanced accuracy (0.750 vs 0.737) and recall (0.548 vs 0.531).

### DF-16 · P2 · "sits AT its 0.5 floor in 18 of 30" (D-157)
- **Correct:** 18 is his script's count at `tol=0.005`; **exactly 0.5** in **14**.

### DF-17 · P1 · the abstract's headline figures are carried by the owner's personas
- **Finding:** 0.904 / 0.916 are **`ext_axis1`**, which uses the owner's knowledge-denial personas;
  the paper's published 8-vs-7 contrast reads **0.791 / 0.832**. About **45% of the margin** comes
  from her bank. Any sentence attributing these to the published method needs the attribution fixed.

### DF-18 · P2 · an arm his draft never mentions
- His branch carries `persona_ood_mod_L{20,40,44}` — the same 36 personas re-extracted to a uniform
  142–150 rollouts, i.e. the **rollout-count control** — scoring **lower** (`ext_axis1` off-diagonal
  **0.9066 vs the reported 0.9158** at L40), under his own heading "More rollouts per persona did not
  help". The draft reports the `ext` bank only. Not an error; an omission a reviewer would want.

### DF-19 · P3 · two venue strings could not be verified
"GEM @ ACL 2026" and "Findings of ACL 2026" (cited around F-88) **appear nowhere in the three PDFs**.
Yoo & Skapars' ICML 2026 workshop line *is* present. Check the venues before they reach a reference list.

### DF-20 · P1 · the abstract's "0.09 below the contrast" is unsourceable **[lead QA]**
- **Sentence:** "the top-3 subspace reaches 0.684 and 0.866, and at layer 40 its first component
  scores **0.09** below the contrast it is derived from."
- **Correct:** the comparison the sentence describes is `ext_pc1` **0.7680** vs `ext_axis1`
  **0.9158** at L40 = a gap of **0.148**. An exhaustive scan of every pc-vs-axis pairing in his own
  L40 summary finds **no gap of 0.09 ± 0.005**. (0.684 and 0.866 are correct: `ext_pc3` at L20/L40.)
- **Evidence:** `results/persona_ood_ext_L{20,40,44}/summary.csv` on `origin/feat/persona-pc-ood`.

### DF-21 · P1 · "the principal components do not contribute" is contradicted by his own files **[lead QA]**
- **For the published (paper-faithful) bank, PC1 is ABOVE the contrast at all three layers:**
  `pc1 − axis1` = **+0.0158 (L20), +0.0274 (L40), +0.0246 (L44)** on the off-diagonal mean.
  The umbrella claim holds only for the kd-**extended** bank, and the sentence does not say so.
- **Lead QA refinement:** the L20 win reverses when HP-KR is excluded (`off_diagonal_excl_hpkr`:
  **−0.0033**), so even the direction at L20 depends on one dataset. State the layer and the
  dataset set, or the claim is not reproducible.
- **The owner's ledger reaches the same destination by a better route:** F-17, F-31 and
  F-96/F-100/F-110 test the PCs against a **matched dataset-PCA control** that survives, rather
  than against the contrast. That argument does not fail on the published bank.

### DF-22 · P1 · the abstract's "lifts transfer by 0.18–0.21" is a sign artifact **[lead QA]**
- **Sentence:** adding the four feigned-ignorance personas "lifts transfer on the four unrelated
  datasets by **0.18–0.21**".
- **What is in the file:** as published, the off-diagonal column-mean lifts are +0.2053 (CG),
  +0.1972 (IT), +0.1787 (ID), +0.1947 (RepE) — reproduced exactly.
- **Why they are not real:** in `matrix_axis1.csv` the row for the probe **trained on HP-KR is
  wholly sign-flipped** — row mean **0.2091** against 0.7906–0.7908 for every one of the other five
  rows (a 1-d feature gives the same AUROC in every row *up to sign*, so the flip is unmistakable).
  That one row drags `axis1`'s column means down and manufactures the lift.
- **Corrected:** replacing the inverted row with `1 − x` gives **+0.0293 / +0.0020 / −0.0048 /
  −0.0027**, which matches the **zero-shot** lifts in his own `zero_shot.csv` (+0.0292 / +0.0020 /
  −0.0048 / −0.0028) — and zero-shot is the protocol the *same sentence's* 0.882/0.912 use.
- **So:** on the four unrelated datasets the extended personas add **essentially nothing**, and on
  two of the four they cost a little. The genuine gain is on HP-KR (column lift +0.6159).

### DF-23 · P1 · "six Liars' Bench datasets" is five
`repe_honesty` is not a Liars' Bench dataset; his own file labels that row "Average (Liars' Bench)"
with `n_datasets=5`. Affects the abstract (A-03) and the paper-ready block quote in his
leave-one-out note (H-013). **Layer 44 was also run and is omitted from the abstract's "layers 20 and 40".**

### DF-24 · P1 · "the independently constructed sycophancy axis" is not independent
`persona_ood.py:911` fits the sycophancy geometry on the **same bank object** — the owner's 36-role
kd vectors — using the replicated paper's own 5-vs-5 contrast. His own comment records **|cos| = 0.54**
between the two axes. "Independently constructed" should go.

### DF-25 · P2 · the extended-persona description merges two groups the owner kept apart
A-11's "six candour personas" merges her deliberately separate `sincere_ignorance` (3) and
`plain_assertion` (3) groups (`kd/personas/kd_role_groups.json`). That split exists to remove a
wording confound; merging them reintroduces it. The persona sets themselves match hers exactly.

### DF-26 · P1 · "within 0.03 AUROC of the privileged upper bound" is 0.0310 **[lead QA]**
- `results/loo_L40/liars_bench_comparison_loo.csv`: extended persona axis **0.8990023** vs
  Upper-Bound Probe **0.93** → **0.0310**. True only after rounding to 2 dp; at L20 it is **0.0432**.
- **And on the metric the benchmark leads with** — balanced accuracy at 1% FPR — the same pair is
  **0.694 vs 0.890**, a gap of **0.196**. The sentence omits it.

### DF-27 · P1 · "the pooled baseline inverts to 0.124 (L20) and 0.523 (L40)" **[lead QA]**
- The two numbers come from **different weightings** of his own table
  (`loo_L{20,40}/loo_summary.csv`, `auroc_worst_dataset`): **0.1243 is `balanced`**, **0.5230 is `naive`**.
- **0.523 is above chance — it is not an inversion.** Holding the weighting fixed at `balanced`
  (the L20 choice) gives **0.4887** at L40; holding it at `naive` gives 0.3254 at L20 and 0.5230 at L40.
- Either way the "inverts at both depths" reading does not survive a matched comparison.

### DF-28 · P2 · a claim contradicted inside his own note
`loo_L20/summary.md:46-48`: "No other arm gets within 0.31 of its worst-case at L20 or within
**0.05** at L40" — at L40 `ext_pc3|naive` has floor 0.6658 against `ext_axis1`'s 0.7116, i.e.
**within 0.0458**. Contradicted by his own `loo_summary.csv`.

### DF-29 · P2 · "generalization gap is zero" is arithmetic, not evidence
A label-free 1-d feature has no training set, so a zero gap is definitional. The contrast also buys
it by being **worse in-distribution** — the raw diagonal is ahead by +0.053 / +0.068 / +0.079.
Worth stating so a reviewer does not read it as a generalization result.

### DF-30 · P3 · an unverifiable cache count
`persona_ood/summary.md` has said "13,648 cached" through all three revisions while his own row
counts moved 13,154 → 13,602. The cache is not committed, so this cannot be checked either way.

---

## E. The 0.374 attribution — one number, eleven documents, the wrong axis

### DF-31 · P1 · **0.374 is the *deception* axis's PC1, not the sycophancy axis** **[lead QA]**
This is the widest-travelling error the audit has found, and it sits in the sentence the whole
`cap/` experiment is built to answer. Three independent sources agree.

**What 0.374 actually is.** `src/assistant_axis_ndif/27_crossgen.py:411-416` sets
`_zs_axis = "deception"` whenever the bank has a `deception` axis — which paper16 does — and the
zero-shot rows named bare **`contrast`/`PC1` are therefore the deception axis**; every other
behaviour axis is written as `contrast:<name>`. Reading the file directly
(`run-crossgen/kd11/matrices/mean_L40_s42.json`, `zeroshot`, `sycophancy_mt` column):

| row | L40 mean, `sycophancy_mt` |
|---|---|
| `contrast` (= deception axis) | **0.4089** |
| `PC1` (= deception axis PC1) | **0.3739** ← the quoted 0.374 |

**What the sycophancy axis actually reads.** The owner's own measurement of 2026-09-17
(`24_q10_sycophancy_axis.py`, an entirely different script, recorded in memory note
`q10-sycophancy-axis-inverted`): **sycophancy axis 0.313** [0.283, 0.346] at L40 mean, and
**deception axis 0.409** — which matches the `contrast` row above to 4 dp. Batch P12's independent
re-measure on `passA11`/`passC11` gives 0.3061 / 0.2996. **All three sources agree; none gives 0.374
for the sycophancy axis.**

**The correction makes the argument stronger, not weaker** (the same shape as DF-03). In the owner's
own random-direction null, the **deception axis at L40 is *within* the random distribution**, while
the **sycophancy axis is *beyond* it, p 0.001–0.008 — significantly inverted.** So "worse than a
random direction" is *not* supported for the number actually quoted, and *is* supported for the axis
actually named. Quoting 0.313 for the sycophancy axis is both correct and a better result.

**A second protocol mix in the same sentence:** the "(0.541)" it is compared against is a **trained**
random probe, not a zero-shot random direction. Compare like with like — either zero-shot against the
1,000-direction null, or the matched trained pair (`axes` 0.3551 vs `random` 0.5413).

**Locations that call it the sycophancy axis — outright wrong:**

| # | file:line | wording |
|---|---|---|
| 1 | `docs/COLLABORATOR_BRIEFING.md:74-78` | "The paper's **sycophancy** axis … It scores **0.374**" |
| 2 | `cap/docs/DESIGN.md:7` | "sycophancy axis reads **0.374**" |
| 3 | `cap/docs/DESIGN.md:19` | "It reads **0.374** … worse than a random direction (0.541)" |
| 4–7 | `cap/docs/REVIEW_PACKET{,_V2,_V3,_V4}.md:19` | "on the one dataset that actually measures sycophancy, it scores worse than a random guess — **0.374**" |
| 8 | `docs/thesis/sarcasm-pilot-negative-result.md:21` | "the paper's **sycophancy** axis scores **0.374**" |
| 9 | `.claude/skills/knowledge-denial-context/SKILL.md:66` | "`sycophancy_mt` (**0.374**) — and on both a random direction beats them" |

**Locations that say "register axis" — defensible but ambiguous; name the axis explicitly:**
`CLAUDE.md:59`, `.claude/skills/knowledge-denial-context/SKILL.md:74`, `cap/docs/DESIGN.md:217`,
`docs/COLLABORATOR_BRIEFING.md:104`, memory `persona-axes-encode-register.md:3,47` and `MEMORY.md:40`.
Also note that even for the deception axis, 0.374 is its **PC1**; its **contrast** is 0.409.

**Ledger:** batch P12 reports the misattribution is inherited from **F-65**, so the ledger entry
needs the same correction. `docs/TODO.md:36` uses 0.374 as a *gate threshold* — the gate still
passes (0.745 ≫ either number), but the threshold should be relabelled.

### DF-32 · P1 · the briefing's "positive on 8–11 of 11" mixes denominators
Measured: **10/11, 11/11, 10/11** over all eleven datasets; **9/9, 9/9, 8/9** over the clean nine.
**Nothing is 8 of 11.** (`docs/COLLABORATOR_BRIEFING.md:142-143`)

### DF-33 · P1 · the briefing's "~666 rows" is the cap, not the realised pool
The row-budget-matched LODO pool is **412–500 rows (mean 449)**; 666 is the cap in
`30_premise.py:167`. ("~6,700" is right: mean `n_lodo` 6,669.) Also the **p = 0.38** quoted there is
a **one-sided** tail (`30b_premise_boot.py:62`); the two-sided bootstrap gives **+0.0045,
CI [−0.024, +0.033], p 0.744** — quote the CI. (`:146`)

### DF-34 · P2 · briefing path and persona-grouping details
`scripts/persona_ood.py` → **`deception_detection/scripts/persona_ood.py`**; the axis is defined in
`deception_detection/persona_pcs.py` (`axes["extended"]`). "Match `kd_role_groups.json` name for
name" is true of the names, but his `KNOWLEDGE_CANDOUR_PERSONAS` **merges** `sincere_ignorance` +
`plain_assertion` into one arm (same issue as DF-25). "To 1e-16" is 2.2e-16 — say 1e-15. (`:204/:215`)

### DF-35 · P2 · the drop-`repe_honesty` margins use a 5-dataset baseline
+0.142 / +0.092 are against the **5-dataset** baseline (0.6085 / 0.7064); against the 6-dataset
baseline they are **+0.1026 / +0.0662**. Say which. (`:232`)

### DF-36 · P1 · "both of your arms lose by 0.064 / 0.055" describes only one arm
Those are the **extended** arm's gaps; **`axis1` loses by 0.146 / 0.122.** (`:251`)

### DF-37 · P2 · "differ by 0.0004" is true of one feature only
It holds for `pc:capitulation@2`; **`pc:cap_only@2` — the feature the basis argument rests on —
differs by 0.0142.** (`:108`, cross-checked against F-65 in batch P3)

### DF-38 · P2 · scope the "2%–114%" share range to mean pooling
Under last-token the share range is **−156% to +145%**, which strengthens the point being made. (`:273`)

---

## F. The cap bank's acceptance test, and the ledger remainder

### DF-39 · P1 · F-32 certifies a bank generation that never produced a rollout — but the cap result is safe **[lead QA]**
- **What F-32 says:** its acceptance table certifies the rewritten cap variants, "all five confounds
  closed by count", and states that "the same lexical counts … were re-run".
- **What is true:** there were **three** bank generations — v1 (`logs/cap_smoke.log`), **v2, the one
  F-32's table certifies** (`logs/cap_smoke_v2.log`), and **v3, the only version on disk
  (mtime 09-20 22:49/22:51) and the one that actually produced F-65/F-66.** Re-running F-32's own
  script against the live bank gives **explain 3/5-vs-4/5** (not 4/5-vs-4/5), **turn-1-commit 0/5
  everywhere** (not 5/5) and **max 99 chars** (not 103). "The same lexical counts were re-run" is
  **false**: all four regexes differ from F-28's, and PRAISE changed *target* (role key → variant
  text) and lost four words, so the reported "4 → 0" is really 4/6 (keys) → 1/6 (`pc_evidence_bound`
  5/5 in text).
- **But the conclusion F-32 is used to support still stands, for a different and better reason.**
  I verified the real protection myself on the lab PC, with my own reader over `run-cap/responses`:
  **1,300 rows, 13 files, 65 personas, 20 questions — exactly 20 distinct (question, turn-1) pairs,
  uniformly n = 65, and exactly ONE distinct turn-1 text per question.** Turn 1 is **byte-identical
  across all 65 persona arms** (`gen_multiturn_ndif.py`, teacher-forced, citing F-46), so persona
  wording cannot enter turn 1 at all; only the turn-2 capitulation can differ. (64 written system
  prompts + 1 empty default arm = the 65; no system prompt is shared by two personas.)
- **Fix:** replace F-32's acceptance table with the live-bank numbers, and rest the no-confound
  argument on **teacher forcing**, which is structural and verifiable, rather than on lexical counts
  that were re-run with different regexes against a superseded bank.

### DF-40 · P2 · F-29's load-bearing half is mean-pooling only
All eight of F-29's numbers are exact, but "the inverted axis **fails as source**" holds **only under
mean pooling** (7 of 21 layers). Under **last** pooling the inverted axis **beats random as source at
18 of 21 layers**. Say which pooling, as F-99 does.

### DF-41 · P2 · F-33 says 22 roles failed; the log says 24
`logs/` records **24 of 24**; 22 was a 15:47 snapshot. **F-35, in the same ledger, says 24** — an
internal conflict, and the log backs F-35.

### DF-42 · P1 · "length-only 0.553 vs raw 0.665" matches no matrix in the repository
53 local matrices give **0.5555 / 0.6623** (11 datasets) or **0.5494 / 0.6652** (10 datasets).
The 0.553/0.665 pair appears in **F-26, F-33, F-73 and the memory note** and is not reproducible as
stated. (This is the same discrepancy DF-04 records inside F-73; it is wider than one entry.)

### DF-43 · P2 · F-26's "the holding persona writes longer turn-1 text" is unsupported
All four turn-1 blocks are **695–698 characters** — equally truncated. Whatever `pc_caver` does, it
is not visible as turn-1 length.

### DF-44 · P2 · F-36's second consequence is reversed
The **production (v3) question set is 10 of 20 first-person**, not the value the entry gives — a
smaller but live framing mismatch. F-36's end-to-end mismatch itself verifies (`sycophancy_mt`:
1,200 items, all with `correct_answer`, 3.2% first-person), but its v2 numbers point at a file
**replaced 5½ minutes after the entry was written**.

### DF-45 · P3 · F-38's table lists one model per dataset where the files hold 2, 12 and 4
A truncated print. The conclusion is **stronger** than stated, not weaker.

### DF-46 · P1 · addendum to DF-22 — the L20 inversion is the project's known HP-KR inversion **[lead QA]**
Checking every layer and both banks: the sign inversion occurs **only at L20, only in `axis1`, only in
the HP-KR training row** (row mean 0.209 vs 0.791). At L40 and L44 **no row inverts** — and there the
same four "unrelated dataset" lifts collapse to **+0.047 / +0.002 / +0.004 / +0.000** (L40) and
**+0.040 / +0.004 / +0.012 / +0.000** (L44). This confirms DF-22 independently of the sign-correction
argument.
**Two consequences, and they cut in different directions:**
1. It is **not a coding bug** — training on HP-KR yields an inverted probe because HP-KR inverts, which
   is the project's own documented finding. It should be described that way, not as an error of his.
2. **Every trained off-diagonal mean over `matrix_axis1` at L20 is dominated by that one row** — the
   0.6511 in his `summary.csv`, the abstract's "0.18–0.21 lift", the briefing's "loses by 0.146"
   (DF-36), **and the audit's own S-11 result**. Sign-corrected, S-11's L20 `axis1` goes
   **0.5900 → 0.7503**, i.e. from 0.0185 *below* the baseline to 0.14 *above* it. **S-11 must not be
   used to argue the published contrast underperforms without stating this.**

---

## G. The scale comparison's pipeline asymmetry (Phase M2, AD-4)

### DF-47 · P2 · the three-rung comparison compares an s23 bank against an s3 bank **[lead QA]**
- **Established from the artifacts' own metadata**, not inferred: each bank records the directory it
  was built from.

| bank | `vectors_dir` | judge filter |
|---|---|---|
| `run-scale/3b/axes_3b.pt` | `run-scale/3b/vectors_s23` | scores **2 and 3** |
| `run-scale/8b/axes_8b.pt` | `run-scale/8b/vectors_s23` | scores **2 and 3** |
| `run-paper16/axes_vectors_s3_fp32.pt` (the 70B) | `run-paper16/vectors_s3_fp32` | score **3 only** |

- **So F-63 / F-100 / F-110's three-rung comparison differs in the judge filter as well as the model
  size.** That should be named in the write-up.
- **The impact is small and already bounded by the project's own measurement.** F-62 compared s3
  against s23 head-to-head on the transfer quantities and found the largest gap **0.005**, far inside
  the ±0.17 dataset-level CI. The three-rung margins are +0.1197 / +0.1027 / +0.0541, so a 0.005
  perturbation changes neither the ordering nor the conclusion.
- **A further bound found here:** the group counts are **identical** across all four banks —
  assistant (1, 25), deception (8, 7), sycophancy (5, 5), unified (13, 12). The filter changed which
  *rollouts* fed each role's vector, **not which roles exist**. That is why F-62 found so little.
- **Recommendation: name the asymmetry as a caveat, do not re-run.** If it is ever re-run, the
  matching bank already exists — `run-paper16/axes_vectors_all_fp32.pt`.

### DF-48 · P1 · a code bug: `run_scale_chain.ps1` silently discards one of the two banks it builds **[lead QA]**
- **`scale/scripts/run_scale_chain.ps1:197-203`** loops `foreach ($vt in @("vectors_all",
  "vectors_s23"))` but writes every iteration to **`axes_$tag.pt`** — the loop variable `$vt` never
  reaches the output path. PowerShell variable names are case-insensitive, so `$tag` is the rung tag.
- **Confirmed by consequence on the lab PC:** both `vectors_all` and `vectors_s23` exist for each
  rung with **26 vector files each** — the extraction ran twice, as intended — but there is **one**
  axes bank per rung, and its `vectors_dir` reads `vectors_s23`. The `vectors_all` bank is built and
  immediately overwritten; `run_scale_crossgen.ps1:53` then reads the survivor.
- **Consequence:** the unfiltered arm of the scale rungs was paid for in compute and never used, and
  DF-47 follows mechanically from this bug rather than from a deliberate choice.
- **Not fixed** — this is outside the fixes the owner authorised. The one-character fix is
  `axes_$vt.pt`, but changing it would make future runs produce two banks where the existing
  downstream reader expects one, so it needs her decision.

---

## E. Wave 5 (2026-09-23) — including two corrections to the audit's own work

### DF-49 · **the audit's own length correction is wrong, and it is already applied in six documents** **[lead QA]**
- On 2026-09-23 the audit wrote, in `memory/response-length-transfers-on-its-own.md` and five other
  files: *"The count is **9 of 11** datasets, or **8 of 10** with `sycophancy_mt` excluded — **never
  '9 of 10'**."* The "never 9 of 10" is **false**.
- Read from the producing file, `run-coords/confounds_L44_last.json` key `length_only_auroc`
  (read-only on the lab PC): it holds **ten** datasets and **nine** score above 0.5 — only
  `roleplaying` inverts (0.4415). It reproduces the memory note's own table to 3 dp.
- **The audit changed the metric AND the dataset set at once without saying so.** "9 of 11" and
  "8 of 10" are *mean token gap* over the passB store. "9 of 10" is *length-only AUROC* over the set
  the length control was actually run on — which **includes `sycophancy_mt` and excludes
  `harm-pressure-kr-matched`**, i.e. a different ten from the audit's.
- Also self-contradictory as written: the correction's "two datasets run the other way
  (`convincing-game` −1.20, `roleplaying` −4.10)" sits directly above a table calling `roleplaying`
  "the only inversion". Both are true, of different metrics; neither names its metric.
- **Applied in:** `docs/NOVELTY_CHECK_PROMPT.md`, `docs/ONBOARDING.md`,
  `docs/thesis/persona-coordinates-70b-results.md` (×2), `kd/docs/DESIGN.md` (×2), the memory note,
  and the `MEMORY.md` hook.
- **Recommended replacement:** *"the positive class is longer in 9 of 11 datasets by mean token gap
  (8 of 10 excluding `sycophancy_mt`); a length-only probe scores above 0.5 on 9 of the 10 datasets
  the length control was run on, `roleplaying` being the sole inversion."*
- **APPLIED 2026-09-23** on the owner's instruction, at all eight sites; each now names its metric
  and its dataset set. Verified by grep: zero residual occurrences outside the ledger. The ledger
  half is covered by the appended **F-114 §2** (the ledger is append-only by her decision).
- Fixed at the same site: `persona-coordinates-70b-results.md:497` called the per-dataset length
  AUROCs "(log only)". They are in `run-coords/confounds_L44_last.json`; the path is now cited.

### DF-50 · `audit/fix/F39_NOTE.md` §2 was wrong; F-45/F-48/F-54 are right about Table 1 **[lead QA, three routes]**
- The note claimed the paper's five deception datasets are "ConvGame, InstrDec, Mask, AILiar,
  Roleplay (table on p.4)" and that F-45/F-48/F-54 are "wrong" for naming HarmPressureChoice.
- Independent `pypdfium2` extraction of all 14 pages: `HarmPressureChoice` appears **2 times, both on
  page 4**; `Mask` appears **41 times and never on page 4**. Table 1 (p.4) lists *AILiar, Roleplaying,
  ConvGame, InstrDec, **HarmPressureChoice***. The figures (pp.5–7) use *ConvGame, InstrDec, **Mask**,
  AILiar, Roleplay*. **The paper declares one set and evaluates another.**
- Batches W5A and W5C reached this independently; with the lead's extraction that is three routes.
- **The owner was right all along and had recorded it three times** — F-60 (`FINDINGS.md:1871-1884`,
  with a matching mention-count table and the phrase "true of Table 1, and misleading"),
  `FINDINGS.md:417-419`, and `docs/papers/persona-coordinates-notes.md` §8.1.
- **Done:** `audit/fix/F39_NOTE.md` §2 corrected in place; its recommendation 3 withdrawn.
- **Still owed (the real defect, found independently by W5B and W5C):** F-60's correction was never
  back-annotated. **F-48 (`FINDINGS.md:1418`) and F-54 (`:1659`)** still assert the Table-1 framing
  with no supersede pointer to F-60.

### DF-51 · F-61's "twelvefold" is 8.1× — the entry divides two different baselines **[lead QA]**
- `FINDINGS.md:1905ff` states paper16 `pc:unified − raw` = **+0.006** against kd's **+0.076**,
  "a twelvefold difference".
- Recomputed from the committed matrices (110 pairs, 11 datasets, L40, mean, `paper`, k=5):
  paper16 raw 0.6623, `pc:unified` 0.6715, `datapca` 0.6651 → **`pc:unified − raw` = +0.0093**, while
  **`pc:unified − datapca` = +0.0064**. The kd side (+0.0757) is correct.
- **Provable mis-copy:** 0.0757 / 0.0064 = **11.8** — exactly the stated "twelvefold". Like-for-like
  against raw it is 0.0757 / 0.0093 = **8.1×**. The entry took kd's margin over *raw* and paper16's
  margin over *datapca*.
- The entry's conclusion (do not attribute the kd gain to persona design) is unaffected.

### DF-52 · F-56's stated mechanism is backwards — `subsample()` balances, it does not preserve the ratio **[lead QA]**
- F-56 item 1: *"`--sample` preserves the source class ratio rather than balancing it … HP-choice is
  ~17% deceptive at source."*
- `src/assistant_axis_ndif/8_coords_ndif.py:504` round-robins over label buckets, and its own
  docstring says *"as balanced as the data allows … degrades gracefully when a class is smaller than
  n/2"* and *"harm-pressure-choice's first 200 rows swallow all 134 of its deceptive rows"*.
- So 134/666 is the balancer hitting the **class ceiling** — taking all 134 deceptive rows that
  exist. The source is **134/2,134 = 6.28%** (denominator from F-47 in the same ledger); 16.75% is
  134/800, the *sample's* ratio. **The sampler raised the deceptive share, the opposite of
  preserving it.**
- F-56's practical conclusions stand: the column is 5:1 imbalanced, departs from the paper's
  "all splits are label-balanced", and is the least comparable of the five. Only the causal story
  needs replacing.

### DF-53 · F-91's "second, independent method" mixes seeds, and the mixing runs one way **[lead QA]**
- Re-derived from `run-premise/falsify/falsify_L*_s*.json` (`real`), 21 layers × 3 seeds:
  L32 s42 **0.8945**, L36 s7 **0.8942**, L40 s42 **0.8828** — the entry's three quoted values come
  from **different seeds**.
- L32 and L36 are each quoted at their **best** seed; L40 at its **worst**. L40's own best seed gives
  **0.8957 — the highest single value in the table.**
- Under a consistent rule: best-seed-per-layer → **L40 wins**; seed-mean → L36 0.8916 > **L40 0.8872**
  > L32 0.8857; seed 42 alone → L32. **No consistent reading supports "32–36, not 40".**
- And nothing here is separable: the L36−L40 seed-mean gap (0.0044) is a third of the seed spread
  *within* L32 alone (0.0145). Recommended wording: "a plateau over roughly L32–L44 that this
  measurement cannot order."
- The crossgen half and the no-reversal check are exact and stand.

### DF-54 · F-78 quotes the wrong panel of Ying et al. Figure 4 **[lead QA]**
- F-78 writes *"base models transfer **to** it markedly better … 70B 0.67 → 0.56, 8B 0.69 → 0.56,
  3B 0.74 → 0.52."*
- Figure 4 body text, `papers/The Truthfulness Spectrum Hypothesis.pdf` p.5: panel **(a)** "FLED
  Probes Test on Sycophancy" = 0.74/0.62, 0.74/0.59, 0.64/0.54; panel **(b)** "Sycophancy Probes Test
  on FLED" = 0.67/0.56, 0.69/0.56, 0.74/0.52.
- **The quoted numbers are panel (b) — transfer *from* sycophancy.** The direction the sentence
  describes is panel (a). Both panels show base > chat at all three scales, so **the conclusion and
  F-78's use of it are unaffected**; the wording and figure reference need fixing.

### DF-55 · O-040 — "probes report ~0.87" on HP-KR has no source, and the comparison is inverted **[lead QA]**
- `docs/NOVELTY_CHECK_PROMPT.md:103-108`: *"a regex on denial phrasing alone reaches 0.91 AUROC on a
  dataset where probes report ~0.87."* `docs/NOVELTY_ASSESSMENT.md:30` records this as the **only**
  part of claim 9 that is the project's own contribution; `docs/LAYER_PREREGISTRATION.md:93` uses it
  as the premise of a directional prediction.
- Swept **all 113 stored matrices** containing HP-KR (`audit/scripts/qa_w5e_O040.py`). The raw probe
  **diagonal** — the only reading under which "probes report X on HP-KR" is meaningful — is
  **0.9663 median (0.8302–0.9900), and 0.96–0.98 at every headline layer**. As a transfer target the
  median is 0.5264; as a source 0.5135. **No probe headline reads 0.87.**
- **The regex (0.9125) therefore does not beat the probe — it loses by ~0.054.** The sentence says
  the opposite.
- **Likely origin**, corroborated independently by session mining (*"HP-KR lexical 0.870 → 0.913 is a
  ROW-SET change, all models → Llama-only, not a re-measurement"*): 0.870 is this project's **own
  superseded regex**. Claim 9 appears to compare the regex against **itself** while calling one side
  "probes".
- **The argument survives, restated:** a pure denial-phrasing regex reaches **0.9125, or 94.4% of**
  a full 8,192-d probe's 0.9663 on the same rows. That is still strong. What cannot be said is that
  the regex beats the probe.
- *Correction to the batch:* cells in [0.86, 0.88] are **not** rare — 1,151 exist across the
  matrices. The true statement is narrower: no quantity anyone would call "the probe result on
  HP-KR" equals 0.87.

### DF-56 · smaller wave-5 items
- **F-60's correction never propagated** (W5B + W5C): F-48 `:1418`, F-54 `:1659` — see DF-50.
- **F-70 caveat (3)** says three papers are "not in `papers/`"; all three were added the next day in
  `4d0d046`. **F-104 caveat (2)** says 8 registration scripts are "still uncommitted"; six entered
  git in `4d0d046`, the very commit carrying F-104.
- **F-22**: the sibling handler at HEAD line 186 of the same file still uses `str(e)[:200]` — the
  fix was applied once and its twin missed.
- **O-065**: "100 AA roles" counts `default` (the no-persona anchor) as a role — it is **99 roles +
  default**; the same table writes the paper16 row correctly as "25 paper personas + default".
- **F-49 DOES NOT HOLD** — its `items.pt` "0.34 vs 0.30 MB" claim is false (both rungs byte-identical
  at 359,384 B). Already superseded by F-50, whose numbers reproduce.
- **F-57**: "the CI band spans zero at every layer" is false — the dataset CI excludes zero at L40
  (lo +0.0330) and L44 (+0.0354).
- **F-21**: the `gen_batch` sweep table has no data behind it — `run-scale/smoke3b` is on neither lab
  nor Drive, and the table is a verbatim copy of a comment at `run_scale_chain.ps1:149`.
- **Memory notes**: `judge-choice-barely-matters` says "all 1,400" rollouts re-judged (it is **1,350**
  — the 50 `default` rollouts are never judged, by design);
  `paper16-assistant-direction-not-lu-axis`'s fourth column is unreproducible from anything committed.

---

## F. Wave 6 (2026-09-23) — the owner's own documents

### DF-57 · the briefing's closing sentence sets HER numbers against HIS **[lead QA]**
- `docs/COLLABORATOR_BRIEFING.md:654-655`: *"...when the median over **the same pairs** is 0.722 and
  the sign-oriented mean is 0.735."* Those are her **110-pair** values, set against **his 0.732**,
  which comes from his own 6×6 matrix. Different corpora.
- Lead re-derivation from his branch (`persona_ood_L40/matrix_baseline.csv`, 30 off-diagonal cells):
  mean **0.7320** (= the draft's 0.732), **median over sources 0.7500**, median of all cells 0.7887,
  **sign-oriented mean 0.7850**, 6 cells below 0.5 (min 0.0598).
- **The argument gets stronger on his data.** On hers the median (0.722) is *below* the mean, which
  does not support "you picked the lowest"; on his, every alternative is above 0.732.
- Also repeated in `audit/phase3/DRAFT_GAPS.md` A-13 — **corrected there in place**.

### DF-58 · the audit's own DF-49 repair cited a path that does not exist **[lead QA]**
- The lead's 2026-09-23 edit to `docs/thesis/persona-coordinates-70b-results.md:497` cited
  `run-coords/confounds_L44_last.json`. **That file is not in the repo** — it is lab-only (cached at
  `audit/cache/labpc/run-coords/confounds_L44_last.json`), and the key is nested at
  **`confounds.length_only_auroc[].auroc_from_length`**, not top-level.
- The edit also left a sentence two lines below — *"Only the sycophancy value (0.664) is also in a
  result file"* — which it contradicted **and which was itself false**: all ten values are in the file.
- **APPLIED**: full key path, lab-only stated, cached copy cited, false sentence removed.

### DF-59 · `docs/thesis/...70b-results.md` — O-070 sits under "What holds" but does not hold
- Both its gaps (+0.014 / +0.026) are **non-significant in their own file** (q 0.86, 0.67), and
  across **126 crossgen configurations** the same best-vs-best gap has **median −0.031**, positive in
  only **28 of 126**. It is one configuration presented as a result.
- O-071 and O-072 in the same document are its own counter-evidence and belong beside it.

### DF-60 · the hf275 columns name the wrong reference direction
- §2.8 / §7.1 say the columns are measured against "each bank's own assistant direction". They are
  measured against the **published Assistant Axis**. Against the bank's own 100-role direction §7.1
  reads −0.669 / −0.651 / −0.577 / −0.479 (printed −0.686 / −0.662 / −0.610 / −0.516), and the PC1
  range is 0.779–0.812 (printed 0.75–0.80). **Conclusions unchanged (≤0.04); the wording is wrong.**

### DF-61 · the per-token gain is understated ~2× at the layers people quote
- `memory/q8-per-token-beats-pooling.md`: "top10pct beats mean … best **+0.053 at 28**". The largest
  zero-shot pooling gain is **+0.1027 at L20**, and it collapses to **+0.005 at L40**. Quoting +0.053
  as the size of the per-token effect is about half the truth at the literature layers.

### DF-62 · smaller wave-6 items
- **O-033** (briefing §7, format-matched control): "~+0.03" is right (+0.0356) but **"clears no
  significance test at any layer" is false** — content clears at L0, L12 and L36 mean-pooled (and
  L28/32/36 last-token). And "two to four times that" is the very ratio **F-94 says must never be
  quoted** (2.05× on 21-layer means, ~1:1 last-token, 0.42×–4.86× by layer).
- **O-023**: "the 1-d contrast beats the 3-d subspace at every depth" fails for the **paper** bank at
  L20 (axis1 0.6511 < pc3 0.6792); true for the extended bank at all three depths.
- **O-030**: marker rate 0.00 at v4/v5 is exact **for the ironic arm only** (sycophant v5 = 0.05,
  sincere v4 = 0.25).
- **M-049**: the sycophancy_mt random-null range is **0.113–0.170**, not "0.13–0.17" (L44 mean is
  0.1131); two different nulls exist (50 draws in the q10 file, 1,000 in `zeroshot_nulls.json`).
- **M-056**: Q2's "erasure costs 0.006–0.014" is exact for `lr_reg10` only (mms L40 is 0.0177) and is
  **superseded by the Q4 note's own 0.005–0.018**.
- **M-054** compares a 5-set `paper` probe against a full-corpus `lr_reg10` probe; the reversal
  survives holding the probe fixed (lr_reg10 subset 0.787/0.763) but the sentence must name them.
- **M-067**: "17 methods" — `q6_layers.json` has **23 rows**; 17 is the count carrying both L20 and L22.
- **§7.4** says erasure costs "1–3 %" of transfer AUROC; measured **0.80–2.64 %**.
- **O-074**'s BH family is that file's own 26 tests, undeclared. **O-060**'s "significant at 20 and 28
  only" covers only the 4 layers tested, and only against `tok_lr` — the best token probe, `tok_mms`,
  was never tested.

### DF-63 · **a pre-registered equivalence margin was widened post hoc, and the change is not flagged** **[lead QA]**
- `docs/thesis/persona-coordinates-70b-results.md:513` (Statistics) pre-registers
  *"equivalence margin **±0.02** for 'these two layers are the same'"*.
- `:740-741` (the §7.6 verdict) concludes *"the four equivalence tests have 90% CIs from −0.038 to
  +0.033, so the two are **equivalent within about ±0.04**"* — a different margin, reached in the
  results section, with no note that it differs from the declared one.
- **All four stored tests disagree with the conclusion at the declared margin.**
  `run-coords/significance.json`, comparisons[7]–[10], every one
  **`equivalent_within_margin = False`**; diffs −0.0128 / +0.0061 / +0.0036 / −0.0092 with 95% CIs
  [−0.0434, +0.0227], [−0.0204, +0.0413], [−0.0155, +0.0258], [−0.0239, +0.0021]. (The file stores no
  `margin` field, so the flag's own threshold is not recorded either.)
- **This matters more under the reframe than it did before.** The reframed thesis argues that the
  control discipline is the contribution; a post-hoc margin widening is exactly what a referee looks
  for in that argument.
- **The conclusion is probably still right — just not by this route.** The median gap over 17 methods
  is 0.012. **Defensible wording:** *"the four equivalence tests do not clear the pre-registered ±0.02
  margin; the median gap is 0.012 and the 90% CIs span −0.038 to +0.033, so the two layers are close
  but formal equivalence at ±0.02 is not established."*

### DF-64 · "all 1,400 rollouts re-judged" — it was 1,350 **[two independent routes]**
- `docs/thesis/persona-coordinates-70b-results.md` §7.7 and `memory/judge-choice-barely-matters.md`
  both say **1,400**. Only **1,350** were judged: `default.jsonl`'s 50 rollouts are never judged, by
  design (`2_activations_ndif.py:20`). **Line 390 of the same thesis file already says so.**
- Found independently by W5D (memory note) and W6B (thesis doc).
- Same paragraph: the Q7 judge gap is the difference of *rounded* rates; exact is **0.74 pp**, not
  0.8 pp. **And a number nobody has recorded:** the two judges keep near-identical *counts* but
  **8.15 % different sets** (110 of 1,350). For "score 3 is the filter that builds the vectors",
  that set difference is the relevant figure, not the rate difference.
- `docs/PERSONA_CONSTRUCTION.md:167` repeats the 91.2 % / score-3 claim — any fix must land there too.

### DF-65 · smaller items from W6B
- **O-079 / §7.4:** "the assistant direction carries 1–3 % of transfer AUROC" — measured
  **0.798 %–2.642 %**, with **3 of 6** settings below the stated 1 % floor. (W6A's notes on O-058 and
  O-074 say "two rows below 1 %"; the fresh measurement says three.)
- **O-087:** the project prints **two** pooled values for the same quantity — **0.863** with alpaca
  dropped and **0.881** with it kept — and no document says which is which. Per dataset that direction
  beats a 1,000-draw random null on only **5 of 9** and is **below chance on `mask` (0.467)`**, which
  independently reproduces the lead's `controls.csv` table.
- **O-092:** the "+0.025 raw median" is carried entirely by the 6 Pass C LODO rows (+0.033); the 4
  trained-transfer raw rows give **+0.005**, and the raw and persona sides pool *different* protocols.
- **O-093:** +0.113 is large, but its two matching dataset-bootstrap tests are non-significant
  (q 0.097, 0.456).
- **O-090:** the L20 fold mean 0.785 contains a **below-chance fold** (HP-KR 0.412, sd 0.188).
- **O-080:** "0.30–0.37" is a 5-layer sample; over L20–L60 it is **0.26–0.42** — the conclusion is
  *stronger*, not weaker.
- **O-096:** "judges are interchangeable" verified (91.185 %), but the "~7–9 % own100-vs-hf275 gap" is
  prose-only and the note's own numbers imply **5.9 %–9.9 %**.
- **Provenance:** `q3_q4_bootstrap.json` stores **no bank field** (RF-16 class) — its fp32 attribution
  rests on the script default. Every §7.4 q-value's BH family is that file's own **26** tests, undeclared.

---

### DF-66 · **the sarcasm pilot's gate was NOT pre-registered — the threshold entered the repo in the commit that reported the failure** **[lead QA, git-verified]**
- `docs/thesis/sarcasm-pilot-negative-result.md:70-75`: *"Two gates were registered in the design
  document, with thresholds, before generation"*, G2 = bag-of-words AUROC **≤ 0.65**.
- **Verified from git, and the claim is false for G2.** `0.65` appears in **no version** of
  `cap/docs/SARCASM_DESIGN.md` — count is **0** at all four of its commits (`3d80071` v1,
  `5ced550` v2, `080a979`, `b34ff2e`). It first enters the repo in
  `cap/scripts/run_sarc_pilot.ps1` at **`b34ff2e`, 2026-09-21 21:48:54 +0600**, whose own commit
  message is *"F-93: the sarcasm pilot fails its decisive check"* — **the threshold and the failure
  verdict were committed together.** (W6D additionally timed generation-end at 21:40:53 and the
  chain's first `stripped=0.947` print at 21:44:27, from lab-only logs; the git evidence alone
  settles it.)
- What the design **did** pre-register is v1's *"separates … above ~0.75 → the bank is dead"*
  (`3d80071`), which **v2 deliberately removed** — *"selectivity … rather than a bare number against
  an arbitrary 0.75"* (`5ced550`; the phrase survives at `SARCASM_DESIGN.md:142`). So the design's
  final state pre-registers **no numeric gate at all**.
- The *"design's own words"* the document quotes are `score_sarcasm_markers.py:198-199` print statements.
- **The finding itself is robust and costs nothing to restate:** measured **0.921** (unmarked v4+v5)
  and **0.947** (all variants) fail 0.65 **and** the 0.75 that v1 did pre-register.
- **Why this is the most urgent item in the file.** The reframed thesis argues that the control
  discipline is the contribution. A pre-registration claim the repository's own history contradicts
  is the first thing a referee auditing that argument will find. **Fix before anyone external reads it.**
- **Suggested wording:** *"v1 of the design set a numeric kill threshold (~0.75); v2 replaced it with
  a selectivity criterion and no number. The 0.65 gate used in the chain was fixed during the run.
  The result does not depend on which is used: measured 0.921/0.947 fail 0.65 and 0.75 alike."*

### DF-67 · `docs/REVIEW_REQUEST_POOLING.md` describes a split protocol the code does not use **[lead QA]**
- `:15-18` states **"80/20 splits by item id"**. `src/assistant_axis_ndif/30_premise.py:106` uses
  `rng.permutation(n)` over **rows**; `group_split` appears **0 times** in that file.
- Row-level splitting inflates the **in-domain ceiling 0.9506** that is row 4 of that document's own
  table. Transfer numbers are unaffected.
- **This document is the one being sent out for adversarial review**, so the description must match
  the code before it goes.
- Same document, same class: its table (`:32`) calls the budget-matched pooled arm **0.850** and
  weakness 1 (`:63`) calls it **0.806** — two draws of one condition (30f: 20 draws, mean
  **0.8344 ± 0.0200**). The honest weakness is that Result 2's **+0.028** margin is of the same order
  as its comparator's draw noise.

### DF-68 · two documents still carry claims their own evidence withdrew
- **`docs/NOVELTY_ASSESSMENT.md:103`** states Claim 4's mechanism as *"ceiling artifact, corr −0.76"*,
  and §4 nominates Claim 4 to "lead with". **F-74 (verdict HOLDS) measured −0.7684 *inside* its own
  coupling null** (analytic −0.7462, P = 0.2135) and **explicitly withdrew it** as evidence; the clean
  uncoupled test (slope −0.100, r −0.067) does not support it either.
- **`kd/docs/NEXT.md:15`** still says *"the register axis's 0.374"* — a **missed DF-31 location**,
  present in neither of DF-31's location lists.
- Also: the corrected sarcasm sentence pairs 0.313 (zero-shot sycophancy axis) with *"~0.300 on
  knowledge-report"*, which is the **trained 4-d `axes` stack** as source on full HP-KR (0.3004) —
  two protocols in one sentence.
- Provenance gap: Reviewer B's counter-citation (Kantamneni et al.) has **no row in
  `audit/papers.csv`** — no arXiv id, no PDF — and Reviewer B's report survives nowhere. Reviewer A's
  TL;DR survives only as a session paste.

### DF-69 · `docs/ROADMAP_TO_DEFENCE.md:27` answers a question about HIS baseline with OUR corpus
- It pre-empts a reviewer question about the **collaborator's six-dataset** baseline (0.6478 / 0.7320)
  using **our eleven-dataset** median **0.722** and oriented mean **0.735** (110 pairs,
  `run-premise/audit`). The values are exact; the corpus is the wrong one. **Same error as DF-57.**
- The same file's **oracle row is unmentioned** — where pooling **loses**, −0.0422 [−0.0727, −0.0106].

### DF-70 · the cap block's three unstated scope conditions
Everything reproduces (0.745 → 0.7449; +0.112 → +0.1123 [+0.0502, +0.1847]; 9/10; cap_only −0.0078
p 0.5578; full 11×11 +0.0983 [−0.036, +0.208]; cos 0.9313). Three conditions are not stated:
- it is a **mean-pooling** result — clears at **14/21** layers under mean but only **3/21** under `last`;
- *"beats random@2 **everywhere**"* pairs that word with the q of the only block where it clears
  (minus-sycophancy_mt q = 0.068 / 0.075, `significant_datasets` **false**);
- *"random@2 **significantly** worse than raw (−0.171)"* is **CI-only** (q 0.108 / 0.142).
- `q = 0.013` reproduces only in the **792-test** family; the current file reads **0.016**.
- `cap/docs/DESIGN.md:176`'s *"+0.045 (0.745 vs 0.702 at L40)"* — that pair differs by **+0.043**.
- §5 of the sarcasm doc quotes the **superseded 02:29 status file (0.947)** verbatim while §3.2 of the
  same document already quotes the current **0.921**. The document carries both.

---

---

## C-bis. Found by PHASE 4 (independent blinded re-derivation, 2026-09-23)

These five were found by agents who could not see any audit verdict, then re-derived by the lead.
Full adjudications in `audit/CORRECTIONS.md`; per-row outcomes in `audit/phase4/adjudications.csv`.

### DF-71 · P1 · **F-61's headline "+0.006" is a script-bug artifact — the real figure is +0.0093** **[lead QA, reproduced both code paths]**
- **Where:** `kd/docs/FINDINGS.md`, the F-61 entry, and anything quoting its "twelvefold".
- **Now:** paper16's `pc:unified − raw` = **+0.006**, against kd's +0.076, i.e. a **twelvefold** gap.
- **Correct:** **+0.0093**, and the ratio is **8.1x** (0.0757 / 0.0093).
- **Cause, newly established:** `kd/scripts/paper_comparison.py` originally read its axis labels from
  `d["rows"]` / `d["columns"]`, which are **human captions, not label lists**
  (`'trained on (80% split)'`, 22 chars). `zip` therefore iterated **characters**, and the diagonal
  test `tr == te` compared single letters, leaving most in-distribution cells inside the
  "off-diagonal" mean.
- **Lead reproduced both paths** on `run-crossgen/kd11/matrices/mean_L40_s42.json` (probe `paper`,
  L40, mean, seed 42): **fixed = 110 pairs, raw 0.6623, +0.0093**; **buggy = 109 pairs, raw 0.6800,
  +0.0061**. All three previously-unexplained figures fall out exactly.
- **This RESOLVES the audit's own self-correction #6**, which recorded that the "raw 0.6800 over 109
  pairs" reading survived *"in no copy"* and that *"the audit cannot separate"* two candidates. It
  survived in no copy because **no file ever held it**.
- **The bug is already fixed** in the committed script, which now carries a comment naming the cause.
  Only the prose needs correcting.
- **Fix:** replace "+0.006" with **+0.0093** and "twelvefold" with **8.1x**; add one line saying the
  original figure came from a label-parsing bug now fixed.

### DF-72 · P1 · **HP-KR is ~91–96% solvable from surface wording — state it wherever an HP-KR number is quoted**
- **Where:** every document quoting an HP-KR detection number, including the collaborator's draft
  (sec 6) and `kd/docs/FINDINGS.md`.
- **The measurements** (three independent Phase 4 routes):
  - a **label-free regex** on the response text alone scores **AUROC 0.9125** ("negated know") and
    **0.9597** ("does not affirm knowledge") on the 2,139 Llama-3.3-70B rows — **at or above the
    draft's own headline 0.919** for the knowledge-denial contrast;
  - a **length-only** scorer reads **0.776** there — **above the deception axis's own zero-shot
    0.6436** (L40/mean), which itself sits **inside its random-direction null**;
  - the wording-controlled test the draft names as the thing that would settle this **appears never to
    have been run**, although `harm-pressure-kr-matched` (a wording-matched slice) already exists as
    an evaluation dataset.
- **Fix:** quote the lexical baseline beside every HP-KR number, and stop describing the wording
  confound as "not ruled out" — on these numbers it is **not currently distinguishable**.

### DF-73 · P1 · **`run-scale/8b/coords` is an incomplete corpus and no document says so**
- **Now:** the 8B rung is presented as a scale point comparable to 3B and 70B.
- **Correct:** **5 of the 12 dataset directories have no `rows.jsonl` at all** —
  `harm-pressure-knowledge-report`, `harm-pressure-kr-matched`, `insider-trading`, `sandbagging_v2`,
  `sycophancy_mt`. Only **3,093 non-alpaca rows across 7 datasets**, all capped at 800.
- **Fix:** name the 7 datasets the 8B rung actually covers wherever the three-rung trend is quoted.
  Anything assuming a complete 8B corpus needs re-checking. Compounds RF-16, which already showed the
  three-rung comparison is not like-for-like across banks.

### DF-74 · P1 · **the 8B `all` and `s23` axis builds write the same file — one silently overwrites the other**
- **Where:** the 8B scale chain; same class of defect as **DF-48**, in a different script.
- **Evidence:** both build blocks end `wrote C:/Users/user4/expLB/run-scale/8b/axes_8b.pt` — the same
  path. The **s23 build overwrote the all build**; the surviving file is the s23 one
  (`vectors_dir = vectors_s23`).
- **Fix:** as for DF-48 — write one file per variant. Any 8B "unfiltered vs score-filtered" comparison
  that used `axes_8b.pt` compared s23 against itself.

### DF-75 · P3 · pointer — the audit corrected **seven of its own verdicts**
- `audit/claims.csv` now reads **HOLDS 297 / PARTLY 119 / FAILS 8** (was 304 / 112 / 8).
- Changed: **F-01, F-70, O-022, M-007, M-054, S-25** `HOLDS -> PARTLY HOLDS`; **D-028**
  `MATCHES HIS DATA -> PARTLY MATCHES HIS DATA`. Notes amended on **F-58, F-45, S-33, D-237, O-041**.
- Applied by `audit/scripts/p4_apply_ledger_fixes.py` (idempotent); backup at
  `audit/claims.csv.before-phase4-fixes`. **Not a fix to the owner's documents** — listed so the
  provenance of the ledger is on the record.

---

## C-ter. From the 2026-09-23 experiment runs (F-115, F-116)

### DF-76 · P1 · **the draft still says the wording-controlled test has not been run — it has, three times over**
- **Where:** the draft, p.12 and again p.13 as future work.
- **Now:** *"The bank was designed for the cleaner test… and we have not run it."*
- **Correct:** run on 2026-09-19 (F-03/F-72/F-76), the contamination closed on 2026-09-23 (**F-115**),
  and the larger complementary control run for the first time the same day (**F-116**).
- **Fix:** delete the future-work promise; state the measured outcome — at the pre-registered layer
  the margin is **+0.066, CI [−0.069, +0.189], P = 0.161**, it **inverts to −0.2545 at L20**, and on
  the knowledge-matched control the axes sit **below chance** while an ordinary probe reads 0.999
  in-distribution.

### DF-77 · P1 · **every HP-KR result used the smaller of two available control slices, and no document says the other exists**
- **Where:** everywhere `harm-pressure-kr-matched` is described as *the* control.
- **Now:** documents imply one wording-matched control (258 rows).
- **Correct:** `eval_datasets.hpkr_slice` takes **two** values — `text_matched` (258 rows, wording
  held constant) and **`knowledge_matched` (2,001 rows, knowledge held constant)**. The second was
  **never run** until 2026-09-23 and is **7.8x larger**.
- **Fix:** name both slices wherever the control is described, and report F-116 beside F-03. The two
  together are the 2x2 the design always implied.

### DF-78 · P2 · **F-03's numbers should be quoted leak-guarded from now on**
- **Correct values:** `axes − raw` = **−0.2545 (L20) / +0.0309 (L40) / +0.0050 (L44)**.
- **Fix:** use these and note the leak was tested. The unguarded figures are not wrong, but the
  guarded ones are the defensible pair now that both exist.

---

## C-quater. From the collaborator's `origin/paper` branch (reviewed 2026-09-24)

### DF-79 · P1 · **the draft reports layer 40; its own sweep says the optimum is 36**
- **Correct:** `ext_axis1` peaks at **L36 = 0.9275** (L40 = 0.9158); `denial_axis1` also peaks at 36.
- **Fix:** report L36 as the optimum, or state that L40 was pre-specified and give L36 beside it.
  Under-reporting your own best number invites the question of why.

### DF-80 · P1 · **"outscores … at every depth" is falsified by the new sweep**
- **Now:** the Contributions bullet claims the 1-d contrast beats the 3-PC probe at every depth.
- **Correct:** `ext_axis1 − ext_pc3` is **negative at L64 (−0.0010), L68 (−0.0205) and L72 (−0.0146)**.
- **Fix:** "at 13 of 16 layers", and give the excl-HP-KR figure (11/16, all within ±0.06).

### DF-81 · P1 · **the Contributions margin must be quoted excluding HP-KR — 16 layers now say so**
- Without HP-KR the margin is **+0.0086 (L20), +0.0012 (L24)** … within **±0.06 everywhere**,
  crossing zero five times. This is D-028 generalised from 3 layers to 16.
- **Fix:** state both columns wherever the margin appears.

### DF-82 · P2 · **"HP-KR is a spike at 40" needs its arm named**
- True for `denial_axis1` (0.9185, global max). For `ext_axis1`, **L76 = 0.9249 > L40 = 0.9124**.

### DF-83 · P2 · **`persona_depth/summary.md` cites two folders that are not on the branch**
- `../persona_unified/` and `../persona_depth_orig/` — neither exists. Push them or drop the refs.

### DF-84 · P2 · **the late-layer rise — now SOLVED, state the answer rather than the puzzle**
- Many arms jump at **L76** then fall at L79 (`axis1` 0.6526 → 0.8738 → 0.8179; on HP-KR the
  published axis goes 0.4344 → 0.7301 → 0.7709).
- **ANSWER (F-120), reproduced in our own 80-layer store:** the full-rank probe **declines** there,
  the persona axes only **recover** toward (not past) their mid-stack level, and **a random direction
  gains +0.127** over the same span. Late-layer anisotropy lifts any low-dimensional projection.
  **`axes − random` = +0.2296 at L40 vs +0.1087 at L79** — the method is *worse* at the top.
- **Fix:** add one sentence giving that explanation and the random-control numbers. It converts the
  most obvious reviewer question into a demonstration that your controls work.

### DF-85 · P3 · **withdraw ~~DF-71's~~ DF-12's reproducibility claim — `paper/make_numbers.py` now exists**
- **Pointer corrected 2026-09-24.** This entry and F-119 both said "DF-71's reproducibility half".
  **The entry meant is DF-12**; DF-71 is the F-61 `+0.006` script-bug entry and has no
  reproducibility half. The mis-pointer left DF-12 reading as live and unwithdrawn for a day.
- **DF-12 is now marked WITHDRAWN in place**, with the git evidence
  (`src/probes/deception-detection/paper/make_numbers.py`, 28,232 B, `8fd071b2`, `origin/paper`).
- See F-119. A correction in the collaborator's favour; the audit should state it as such.
- **Still owed:** F-119 in the append-only ledger carries the same mis-pointer and cannot be
  edited. Corrected by the appended **F-123 §1**.

---

## C-quinquies. From the completed scale ladder (F-121, 2026-09-24)

### DF-86 · P1 · **every statement of the scale decline should now name BOTH banks**
- **Now:** documents give the three-rung decline for the published 26-persona bank only
  (+0.0382 → +0.0148 → −0.0116), because that was all that existed.
- **Correct:** the 36-persona kd bank now has the same three rungs —
  **+0.0360 → +0.0194 → +0.0054** (`s23`; `all` reads +0.0317 → +0.0169 → +0.0032).
- **Fix:** state the decline as a property of the **method**, evidenced on two independent banks.
  That is a stronger claim than the one-bank version and closes the "wrong personas" objection.

### DF-87 · P2 · **do not quote the 70B kd figure without its scope condition**
- +0.0054 is a whisker above zero, on **one seed with no significance test**, and on five datasets
  that contain **no knowledge-denial data** — the thing the bank was built for.
- **Fix:** give the direction (the bank gap widens with scale: −0.005 → +0.005 → +0.017), not the
  number, and always attach the dataset scope.

### DF-88 · P3 · **the s3-vs-s23 filter caveat can now be measured rather than bounded**
- `LIMITATIONS_AND_FUTURE_WORK.md` bounds the judge-filter difference using F-62's 0.005. Both
  variants now exist at every rung and are byte-distinct, so the comparison is direct:
  `s23` beats `all` at all three rungs (+0.0043 / +0.0025 / +0.0022).

---

## C-sexies. From the by-axis results review (2026-09-24)

### DF-89 · P1 · **the sycophancy axis is missing from OUR pipeline — and is his best arm at five layers**
- **CORRECTED 2026-09-24.** The first version of this entry said the sycophancy axis was never
  computed anywhere. **Wrong** — that search covered only `run-crossgen/`. **His branch has four
  sycophancy arms** (`syc_pc1`, `syc3`, `syc_axis1`, `syc3_orth`) at all 16 layers.
- **The real gap:** it is absent from `27_crossgen.py`'s default `--features`, so *our* pipeline has
  never trained it. CPU-only to close, minutes per store.
- **What his arms show (F-122):** excluding HP-KR, `syc_pc1` tops the table at L40-56 (0.9370 vs
  `ext_axis1`'s 0.9165 at L40) — but **the margin is one dataset**: `convince_me` +0.086, and
  `syc_pc1` loses on `instruct_dishonesty` and `repe_honesty` at every layer (3 of 5 wins). Say
  "a sycophancy axis reads the convincing game best", not "sycophancy beats the extension".

### DF-94 · P1 · **"424 of 692 claims checked" overstates coverage — the real figure is 314 of 692**
- The numerator mixes inventory claims with claims raised *during* checking. Of the 424 verdict
  rows, **314** have ids in `claims_inventory.csv` and **110** do not.
- **Real inventory coverage is 45%, not 61%. 378 inventory claims never got a verdict — 176 of them
  in the draft PDF.**
- **Fix:** wherever the tally appears (`PROGRESS.md`, `REPORT_PLAIN.md`, `REPORT_TECHNICAL.md`,
  `BRIEF_AS_EXECUTED.md`, auto-memory), report both numbers: "314 of 692 inventory claims given a
  verdict, plus 110 raised while checking." See `audit/COVERAGE_GAP.md`.

### DF-95 · P2 · **the two reports predate `origin/paper` and omit F-117, F-119, F-122**
- `BRIEF.md` fixed scope 2026-09-22; the collaborator pushed `origin/paper` 2026-09-23 23:46
  (3 commits, 683 files, 11,555 insertions). It was reviewed 2026-09-24 as a scope extension,
  after both reports were drafted.
- **Fix:** add the 16-layer sweep (F-117), the `make_numbers.py` correction (F-119) and the
  sycophancy arms (F-122) to both reports, or state at their head that they close at 2026-09-23.

### DF-93 · P1 · **label the `dim=1` arms zero-shot, and put the error bar over datasets not cells**
- For every one-dimensional arm the trained off-diagonal **equals** the zero-shot AUROC — a
  regression on one scalar can only choose a sign. Measured spread across training rows is
  **0.000 at all 16 layers** for `ext_axis1` and `denial_axis1` (F-122, extending F-97 caveat 2).
- So their 30-cell off-diagonal mean contains **6 distinct numbers**, each replicated 5x, while the
  baseline they beat is a **trained 8192-d probe**.
- **Fix:** call these arms zero-shot in the text; compute any interval over the 5 test datasets.
  The claim "a 1-D zero-shot direction beats a trained 8192-d probe" is stronger anyway — but it is
  a different claim from the one the matrix framing implies.

### DF-90 · P2 · **stop writing "16-persona bank" — it is a 26-persona bank**
- `paper16` = the paper's **deception set** (8 + 7 + anchor = 16). The bank holds **26**; the
  sycophancy set is **11**, not 10. The "16" names a subset, not the file.
- **Fix:** say "the paper's 16-persona deception set" or "the 26-persona bank", never "the 16-persona
  bank". Our own extension is **36**.

### DF-91 · P2 · **the knowledge-denial axes are 70B-only, and nothing says so**
- `pc:knowledge_denial` and `pc:kd_assertion` exist as trained features **only** in the 70B kd11
  store (11 datasets, k=5). The 3B/8B rungs added in F-121 carry the *deception* and *unified* axes,
  not these.
- **Fix:** attach "70B only, 11 datasets, k=5" wherever either figure appears.

### DF-92 · P1 · **`pc:kd_assertion` outperforming `pc:knowledge_denial` belongs in the results**
- 0.8119 vs 0.7492 at L40. The stronger feature contrasts **feigned ignorance vs plain assertion** —
  an assertion-register axis — and it is **the best-transferring persona feature in the project**
  (+0.150 over raw).
- **Fix:** report it. It is the register diagnosis (F-13) visible directly in the feature table,
  rather than inferred, and it is a stronger form of the argument than the project currently makes.

---

## D. Corrections already delivered elsewhere (pointers, not duplicates)

- **The three-rung scale trend (RF-16).** The 70B rung mixed persona banks. Corrected table in
  `audit/fix/RF16_NOTE.md`: 3B **+0.1197**, 8B **+0.1027**, 70B-kd **+0.0751**, 70B-paper16
  **+0.0541 / −0.0131**. Affects F-63, F-100, F-101, F-102.
- **F-110** was appended to `kd/docs/FINDINGS.md` with the owner's approval and amended twice
  (the "best" definition, and the last-pooling figures). Two amendments remain **uncommitted**.
- **F-39** reproduces exactly; see `audit/fix/F39_NOTE.md` for the dataset-set sensitivity.
