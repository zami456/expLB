# Corrections for the draft and the abstract

Written 2026-09-23. Companion to `docs/COLLABORATOR_BRIEFING.md`, which was corrected the same day.

Everything below was checked against **the data committed on `feat/persona-pc-ood`** — your own
CSVs, `.npz` and `.pt` files — or against the published PDFs. Where a number could only be read
back from your own summary line, that is said. Your activations were not requested, so nothing
here depends on having them.

**Each item gives: what the text says, what the file says, and where to look.** Items marked
**[checked twice]** were re-derived a second time, independently, before being written down.

---

## 1. Things that need changing

### 1.1 The abstract's "0.09 below the contrast" cannot be sourced **[checked twice]**
> "the top-3 subspace reaches 0.684 and 0.866, and at layer 40 its first component scores **0.09**
> below the contrast it is derived from."

`ext_pc1` is **0.7680** and `ext_axis1` is **0.9158** at L40 — a gap of **0.148**. An exhaustive
scan of every pc-vs-axis pairing in your own L40 summary finds **no gap of 0.09 ± 0.005**.
(0.684 and 0.866 are right: `ext_pc3` at L20 and L40.)
`results/persona_ood_ext_L{20,40,44}/summary.csv`.

### 1.2 "The principal components do not contribute" is contradicted by your files **[checked twice]**
For the **published** (paper-faithful) bank, PC1 is **ahead of** the contrast at all three layers:
`pc1 − axis1` = **+0.0158 (L20), +0.0274 (L40), +0.0246 (L44)** on the off-diagonal mean.
The claim holds only for the **kd-extended** bank, and the sentence does not say which bank it is
about. A further caveat: at L20 the sign reverses when HP-KR is excluded
(`off_diagonal_excl_hpkr` gives **−0.0033**), so even the direction there rests on one dataset.

There is a stronger route to the same conclusion, which does not fail on the published bank:
test the PCs against a **matched dataset-PCA control at the same k** rather than against the
contrast. That is what our F-17, F-31 and F-96/F-100/F-110 do, and the PCs lose there too.

### 1.3 The "0.18–0.21 lift" is an artifact of one training row **[checked twice]**
> "Adding four feigned-ignorance personas … lifts transfer on the four unrelated datasets by
> **0.18–0.21**."

The numbers reproduce exactly as off-diagonal column means at **L20** (+0.2053 CG, +0.1972 IT,
+0.1787 ID, +0.1947 RepE). But in `matrix_axis1.csv` the row for the probe **trained on HP-KR is
sign-flipped** — row mean **0.2091** against 0.7906–0.7908 for all five other rows. (A 1-d feature
gives the same AUROC in every row *up to sign*, so the flip is unmistakable.) That single row pulls
`axis1`'s column means down and produces the apparent lift.

Three independent checks agree it is not a transfer effect:
- Replace the inverted row with `1 − x`: the lifts become **+0.0293 / +0.0020 / −0.0048 / −0.0027**.
- Your own `zero_shot.csv` — the protocol the *same sentence's* 0.882/0.912 use — gives
  **+0.0292 / +0.0020 / −0.0048 / −0.0028**.
- At **L40 and L44 no row inverts**, and there the same four lifts are
  **+0.047 / +0.002 / +0.004 / +0.000** and **+0.040 / +0.004 / +0.012 / +0.000**.

**This is not a coding error on your side.** Training on HP-KR yields an inverted probe *because
HP-KR inverts* — which is the project's own documented finding. The honest statement is that the
extended personas **fix the one training source that was inverted**; on the four unrelated datasets
they add essentially nothing. The real gain is on HP-KR itself (column lift **+0.6159**).

### 1.4 "Six Liars' Bench datasets" is five
`repe_honesty` is not a Liars' Bench dataset. Your own file labels that row
"Average (Liars' Bench)" with **`n_datasets=5`**. Same wording appears in the paper-ready block
quote of the leave-one-out note. Also: **layer 44 was run** and the abstract says "layers 20 and 40".

### 1.5 "The independently constructed sycophancy axis" is not independent
`deception_detection/scripts/persona_ood.py:911` fits the sycophancy geometry on the **same bank
object** — the 36-role kd vectors — using the replicated paper's own 5-vs-5 contrast. Your own
comment in that file records **|cos| = 0.54** between the two axes. "Independently constructed"
should go.

### 1.6 "Every method has recall near zero on HP-KR"
False for the **Follow-Up Probe on Llama-3.3-70B: 0.37** (bought with an 84% FPR). It is true of
the **cross-model averages**, and should say so. The 0.12 AUROC figure itself is correct
(Liars' Bench Table 6).

### 1.7 "The base probe inverts on one dataset at each depth"
At **L20 two** datasets sit below 0.5: HP-KR **0.109** and harm-pressure-choice **0.471**.

### 1.8 "The top-10% token read is best"
Contradicted by **your own Table 9**: at L40 the single-maximum read wins on both balanced
accuracy (**0.750 vs 0.737**) and recall (**0.548 vs 0.531**). The four quoted numbers are right;
the conclusion drawn from them is not.

### 1.9 "Sits AT its 0.5 floor in 18 of 30"
18 is your script's count at `tol=0.005`. **Exactly 0.5 in 14.** Either say "within 0.005 of" or
use 14.

### 1.10 "Within 0.03 AUROC of the privileged upper bound" **[checked twice]**
`results/loo_L40/liars_bench_comparison_loo.csv`: extended persona axis **0.8990023** vs
Upper-Bound Probe **0.93** → **0.0310**. True only after rounding to 2 dp; at L20 it is **0.0432**.
And on the metric the benchmark leads with — balanced accuracy at 1% FPR — the same pair is
**0.694 vs 0.890**, a gap of **0.196**, which the sentence omits.

### 1.11 "The pooled baseline inverts to 0.124 (L20) and 0.523 (L40)" **[checked twice]**
The two numbers use **different weightings** of your own table
(`loo_L{20,40}/loo_summary.csv`, `auroc_worst_dataset`): **0.1243 is `balanced`, 0.5230 is `naive`**.
**0.523 is above chance — it is not an inversion.** Holding the weighting fixed at `balanced`
gives **0.4887** at L40; holding it at `naive` gives 0.3254 at L20 and 0.5230 at L40. The
"inverts at both depths" reading does not survive a matched comparison.

### 1.12 "Pooled supervision matches or beats it"
Holds in **1 of 4** depth × pooling cells, and uses a different pooling at each depth.

### 1.13 A claim contradicted inside your own note
`loo_L20/summary.md:46-48`: "No other arm gets within 0.31 of its worst-case at L20 or within
**0.05** at L40" — at L40 `ext_pc3|naive` has floor **0.6658** against `ext_axis1`'s **0.7116**,
i.e. within **0.0458**. Contradicted by your own `loo_summary.csv`.

> **CORRECTION 2026-09-24 (DF-12 withdrawn / F-119, P17).** The script exists on `origin/paper`:
> `src/probes/deception-detection/paper/make_numbers.py` (39,532 B @ 16def2de), with
> `paper/generated/numbers.tex` (1,176 macros) and 11 `tab_*.tex`. The `feat/persona-pc-ood` half is
> still literally true (no `paper/` path at 45b16d90).
> **Source:** `git ls-tree -r -l 16def2de` and `git ls-tree -r 45b16d90`; `kd/docs/FINDINGS.md` F-119,
> F-123 (DF-12 withdrawn).
> _The text below is superseded on this point and is kept for the record._

### 1.14 `paper/make_numbers.py` does not exist **[checked twice]**
The compute/provenance section promises it. It is on **neither branch**, and
`feat/persona-pc-ood` has no `paper/` directory at all.

### 1.15 An unverifiable cache count
`persona_ood/summary.md` has said "13,648 cached" through all three revisions while the row counts
moved 13,154 → 13,602. The cache is not committed, so this cannot be checked either way.

### 1.16 Two venue strings could not be verified
"GEM @ ACL 2026" and "Findings of ACL 2026" appear nowhere in the three cited PDFs. (Yoo &
Skapars' ICML 2026 workshop line *is* there.) Worth confirming before a reference list is built.

---

## 2. Attribution

### 2.1 The headline numbers rest on the knowledge-denial bank
**0.904 / 0.916 are `ext_axis1`**, the 12-vs-13 extended contrast. The paper-faithful **8-vs-7**
contrast reads **0.7909 / 0.8319** zero-shot on the same rows. So **44.2% (L20) and 45.7% (L40) of
the margin over the trained baseline comes from the extended bank**, not from persona coordinates
as published. Neither the abstract nor the draft names the bank at that point.

### 2.2 The extended personas are the kd bank, unchanged
The "four feigned-ignorance vs six candour personas" are **exactly** `kd/personas/kd_role_groups.json`
— set-compared, both poles True. The 36 `.pt` files on your branch are **byte-identical** to
`run-kd/vectors_s3`. One difference worth fixing: your `KNOWLEDGE_CANDOUR_PERSONAS` **merges**
`sincere_ignorance` (3) and `plain_assertion` (3) into one arm. Those were deliberately kept apart
to remove a wording confound; merging them reintroduces it.

---

## 3. Worth adding (not errors)

### 3.1 An arm the draft never mentions
Your branch carries `persona_ood_mod_L{20,40,44}` — the same 36 personas re-extracted to a uniform
142–150 rollouts, i.e. **the rollout-count control**. It scores **lower** (`ext_axis1` off-diagonal
**0.9066 vs the reported 0.9158** at L40), under your own heading *"More rollouts per persona did
not help"*. A reviewer will want this reported; it is a control in your favour that currently
looks omitted.

### 3.2 Appendix G's response-token runs have no committed table
The follow-up run reproduces exactly (all 20 cells to 4 dp). But the three response-token runs
committed **only plots, no `results_table.csv`**, so "matched for each layer" can be neither
confirmed nor refuted. Those three files would close it.

### 3.3 "Generalization gap is zero" is definitional
A label-free 1-d feature has no training set, so a zero gap follows by arithmetic. The contrast
also buys it by being **worse in-distribution** — the raw diagonal is ahead by **+0.053 / +0.068 /
+0.079**. Worth stating so it is not read as a generalization result.

---

## 4. What we checked and found correct

So the list above is read in proportion:

- **Every digit in your three result notes is exact to ≤5e-4.** All six problems we found are in
  the quantifier around a number, never the number.
- **Your leave-one-out note passed a full independent recomputation.** `loo_scores.npz` commits
  raw held-out scores *and* labels, so all 10 arm × weighting × 6-dataset cells were recomputed at
  both depths with an independent rank-based Mann-Whitney: **every row of `loo_summary.csv`
  reproduces to 4 dp** (one 1e-4 rounding difference on `pc3|balanced` at L20).
- **All 48 Table 3 cells**, all **10 Table 10 cells**, **Table 7's layer-40 columns**, the
  0.983 / 0.732 / 0.251 triple and the CG→HP-KR 0.06 minimum re-derive from your per-cell matrices.
- **Your published comparators are correctly transcribed.** Liars' Bench Tables 6/7/8 re-extracted
  from the PDF: 0.682/0.636, 0.634/0.578, 0.930/0.890 all match, and your five-shared means are the
  right reductions of the Llama-3.3-70B rows.
- **Your random control is not a single draw.** `persona_ood.py` averages `N_RANDOM_DRAWS = 5`
  seeded draws and has since the script's first version — we checked because we had written down
  the opposite, and we were wrong. The narrower fair criticism stands: reporting the **mean of 5**
  bounds the centre but not the tail; a percentile of a larger null would be stronger.
- **The HP-KR sign flip in 1.3 is not a mistake of yours** — it is the HP-KR inversion this project
  documents, showing up in the training row.

---

## 5. If you want to re-check any of this

Every number above came from a file on your own branch or from a published PDF, read with
`git show feat/persona-pc-ood:<path>`. The paths are given inline. Nothing was recomputed from
activations, and nothing here needs your local disk.
