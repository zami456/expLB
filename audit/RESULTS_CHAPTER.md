# Results — a chapter draft, with every number sourced

_Phase 6 workstream 6A, begun 2026-09-24. Companion to `audit/METHODS_CHAPTER.md` and
`audit/LIMITATIONS_AND_FUTURE_WORK.md`. The owner moves this into the paper; this file is
`audit/`-resident by the write-scope rule._

**How to read the markers.** **[RE-DERIVED]** = recomputed here from the raw result files, with
the path and key given · **[LEDGER]** = recorded in `kd/docs/FINDINGS.md` with a source, not
recomputed here · **[PAPER]** = read from a published PDF, page cited.

> **CORRECTION 2026-09-24 (DF-106, P17).** The single pin below is wrong for part of the pass.
> - **Level 2** (p16_level2, 15:39) read ea1ce52b.
> - **Level 1** (verdicts written 16:40, after the ref moved at 15:44) read 2ad758fb for at least
>   D-171 and D-032.
> 172 of the 176 stored values are identical at ea1ce52b, 2ad758fb and 16def2de, so the stamp
> does not matter for them. The audit's current pin is **16def2de**. At that pin **one verdict
> changes: D-189 → PARTLY** (Table 7, CG raw L40 = 0.7873 → 0.787). Status now reads 165 / 9 / 2.
> **Source:** `audit/p17/A_REPORT.md` §1; `audit/p17/d_rows_repinned.csv`; `audit/claims.csv` D-189 @ 16def2de.
> _Kept for the record below._

**Provenance of everything marked [RE-DERIVED].** `origin/paper` @ `ea1ce52b`, principally
`src/probes/deception-detection/results/persona_ood_ext_L{20,40,44}/`, with
`persona_ood_followup_L20/` and `persona_positions_L{20,40,44}/` for §8 — all read with `git show`,
no checkout and no branch change. Table 5 was recounted from the cached judge output at
`audit/cache/labpc/run-kd/scores/`. Scripts: `p16_level2.py` and `p16_level2_verdicts.py` (the 54
re-derived claims), `p16_level1.py` and `p16_level1_verdicts.py` (the remaining 122), all under
`audit/scripts/`. Per-claim rows in `audit/claims.csv`, batches B01–B48.

**Status — 6A complete. All 176 draft claims have a verdict.**

| | count |
|---|---:|
| `MATCHES HIS DATA` | **166** |
| `PARTLY MATCHES HIS DATA` | 8 |
| `UNRESOLVED` — artifact exists nowhere | 2 |
| failures | **0** |

**Every stated number that exists in a committed result file was located and matched.** The 54
Level-2 claims were re-derived from the matrices and rollouts; the remaining 122 were located at
Level 1. Not one arithmetic error was found in the draft.

> **CORRECTION 2026-09-24.** An earlier draft of this chapter reduced eleven Level-2 claims to
> `PARTLY` on the ground that the tables do not disclose the one-dimensional arms' zero-shot nature
> or the layer-20 sign flip. **The paper discloses both, in its Methods** — see §2.1 and §2.2. The
> verdicts were revised up (Level 2 now reads 52 MATCHES / 2 PARTLY) and the sections below rewritten.
> Recorded in `kd/docs/FINDINGS.md` F-125. **Level 1 caught a Level-2 over-reach, which is the
> argument for interleaving the two.**

---

## 1. The headline: detection on harm-pressure-knowledge-report

HP-KR is the dataset the replicated method never sees and every detector in this project inverts on.
It is where the extension has to earn its place.

**Table 2 — detection on HP-KR, Llama-3.3-70B.** AUROC, and balanced accuracy at a threshold
calibrated to 1% FPR on Alpaca. **[RE-DERIVED]**

| arm | dim | AUROC L20 / L40 / L44 | BA L20 / L40 / L44 |
|---|---:|---|---|
| Published contrast (8 v 7) | 1 | 0.148 · 0.593 · 0.589 | 0.500 · 0.500 · 0.500 |
| **Knowledge-denial contrast (4 v 6)** | 1 | **0.890 · 0.919 · 0.908** | **0.533 · 0.910 · 0.757** |
| Extended contrast (12 v 13) | 1 | 0.882 · 0.912 · 0.894 | 0.500 · 0.500 · 0.500 |
| Base probe (RepE-trained) | 8192 | 0.109 · 0.936 · 0.828 | 0.500 · 0.500 · 0.500 |
| LB Mean Probe **[PAPER]** | 8192 | 0.12 | 0.50 |
| LB Follow-Up Probe **[PAPER]** | — | 0.14 | 0.26 |
| LB Upper-Bound Probe **[PAPER]**, privileged | 8192 | 0.97 | 0.95 |

The three published rows are Liars' Bench Tables 6–8, p.54–56, row *Llama 3.3 70B*, column HP-KR
(`papers/2511.16035v2 (1.pdf)`). All three transcribe exactly.

**The result that matters.** A **one-dimensional, label-free direction** built from four
feigned-ignorance personas against six candour personas reads **0.919 AUROC** on HP-KR at layer 40,
where the benchmark's own 8192-dimensional Mean Probe reads **0.12** and its Follow-Up Probe
**0.14**. The published 8-vs-7 contrast the method comes with reads **0.593**. The extension is
doing the work, not the method.

**Three conditions that must travel with that sentence.**

**(a) The published contrast inverts at L20, and so does the base probe.** 0.148 and 0.109 are not
weak scores; they are scores below chance, which means the direction points the wrong way. This is
the project's own documented HP-KR inversion **[LEDGER, F-13]**, reproduced here in a pipeline
built by someone else.

**(b) The BA column does not operate at 1% FPR.** The threshold is fitted on Alpaca and then applied
unchanged. Measuring the realised false-positive rate on the evaluation set itself **[RE-DERIVED,
`zero_shot_operating.csv`, `1 − specificity`]**, **23 of 90 (arm × dataset × layer) cells exceed 1%**:

| | CG | HP-C | HP-KR | IT | ID | RepE |
|---|---:|---:|---:|---:|---:|---:|
| 4 v 6, L20 | 0.0% | 4.5% | **88.3%** | 0.0% | 1.1% | **35.6%** |
| 4 v 6, L40 | 0.0% | 0.0% | 6.5% | 0.0% | 11.9% | 0.0% |
| 4 v 6, L44 | 0.0% | 0.0% | 4.7% | 0.0% | 9.9% | 0.0% |
| base probe, L40 | 0.0% | 0.0% | 0.0% | 0.0% | 15.2% | 0.0% |

The L20 figure of 0.533 balanced accuracy is bought at an **88.3% false-positive rate** (specificity
0.1165, recall 0.9496). **This is precisely the defect the draft levels at Liars' Bench** — whose
Follow-Up Probe reaches 0.37 recall on HP-KR at an **84% FPR**, Table 7 p.55 **[PAPER]**. The
criticism is correct and it applies symmetrically. The L40 headline of 0.910 survives it, at 6.5%
FPR rather than 1%.

**The draft discloses this, in words rather than numbers.** The Figure 3 caption reads: *"At layer 20
its threshold falls inside HP-KR and flags most honest answers too."* That is the 88% FPR described
accurately. Putting the number beside it would be stronger, and would let the same sentence make the
symmetric point about the benchmark's own probes.

**(c) The privileged upper bound is not held to the same operating point.** Liars' Bench Table 8's
first block is headed *BALANCED ACCURACY* with no "at 1% FPR" qualifier — Tables 6 and 7 carry it —
and the same table's FPR row gives **0.08** for this cell. So the 0.95 upper bound operates at 8%
FPR while the arms compared against it are held to 1%. Any "within 0.03 of the upper bound" claim
inherits this **[PAPER; cf. DF-26]**.

**A loose end worth closing.** Two different base-probe numbers exist in the same result directory.
Table 2's row comes from `zero_shot_operating.csv` (0.109 / 0.936 / 0.828); `matrix_baseline.csv`,
cell `repe_honesty → HP-KR`, reads **0.381 / 0.928 / 0.880**. At L20 they disagree sharply, and only
the first shows a strong inversion. The draft should name which base probe it means.

---

## 2. Transfer: the mean off-diagonal

**Table 3 — mean off-diagonal AUROC over the six-dataset grid.** **[RE-DERIVED,
`summary.csv`]** All 48 values reproduce exactly.

| arm | dim | all six, L20 / L40 / L44 | excluding HP-KR |
|---|---:|---|---|
| Probe on raw activations | 8192 | 0.648 · 0.732 · 0.716 | 0.728 · 0.758 · 0.730 |
| Published contrast (8 v 7) | 1 | **0.651** · 0.832 · 0.826 | 0.919 · 0.880 · 0.873 |
| Knowledge-denial (4 v 6) | 1 | 0.866 · 0.912 · 0.900 | 0.861 · 0.910 · 0.899 |
| Extended contrast (12 v 13) | 1 | 0.904 · 0.916 · 0.903 | 0.909 · 0.917 · 0.905 |
| Extended PCs (k = 3) | 3 | 0.684 · 0.866 · 0.880 | 0.900 · 0.867 · 0.870 |
| Extended PC1 | 1 | 0.661 · 0.768 · 0.780 | 0.682 · 0.774 · 0.793 |
| Published PCs (k = 3) | 3 | 0.679 · 0.639 · 0.617 | 0.888 · 0.751 · 0.728 |
| Random directions (k = 3) | 3 | 0.527 · 0.504 · 0.493 | 0.560 · 0.544 · 0.547 |

**The random-direction row is the control that makes the rest readable**, and it behaves: 0.49–0.56
everywhere. A persona arm at 0.9 is not a low-dimensional-projection artifact.

### 2.1 The one-dimensional arms are zero-shot, and the paper says so

For every `dim = 1` arm, the "trained" off-diagonal **equals the zero-shot AUROC exactly**. This is
not an approximation. Reading `matrix_axis1.csv` cell by cell **[RE-DERIVED]**, every cell in a
given test column is **identical across all five training rows**, up to sign — a regression on one
scalar feature can only choose a sign. The 30-cell off-diagonal therefore contains **six distinct
numbers, each replicated five times**, against a baseline that is a genuinely trained 8192-parameter
probe.

**The draft states this outright, in three places.** Section 3.4: *"A probe on a single feature can
learn only a sign and a scale. Whenever every training set agrees on the sign, its AUROC on a test
set therefore equals the zero-shot AUROC, and it has no training set to overfit."* Table 3's caption:
*"The 1-d rows are persona contrasts, a probe on one feature."* Figure 4's caption: *"Its rows are
constant because a probe [on one feature can learn only a sign]."* The Contributions bullet calls
the arm *"the zero-shot extended contrast"*.

The condition clause — *"whenever every training set agrees on the sign"* — is doing real work, and
§2.2 is what happens when it fails.

So the result is already framed as the stronger claim it is: **a one-dimensional zero-shot direction
matches or beats a trained 8192-dimensional probe.** The concern that intervals over the 30 cells
would count duplicates as independent does not arise either — the draft computes **no interval over
matrix cells at all**. Section 4: *"95% stratified bootstrap intervals over examples (1,000
resamples), drawn once and shared by every detector."*

### 2.2 The 0.651 here and the 0.791 in Table 6 are the same arm at the same layer

The published contrast reads **0.651** here at L20 and **0.791** in Table 6's Mean row. Both
reproduce. They are different quantities: Table 6's is the zero-shot mean, Table 3's the trained
off-diagonal.

The gap is one row's sign. In `matrix_axis1.csv` at L20 the **HP-KR training row is exactly `1 − x`
of every other row** — verified cell by cell, difference **0.00000** — with row mean **0.0806**
against 0.750–0.806 for the other five. Averaging it in drags the mean from 0.791 to 0.651.
Sign-corrected, the off-diagonal is **0.7909**, the Table 6 figure to four decimals. At L40 and L44
no row inverts and the two tables agree exactly **[RE-DERIVED]**.

**The draft explains this**, in section 4, with the matching worked example: *"the inverted ranking
on HP-KR carries into every test set (for instance CG falls from 0.940 to 0.060)."* The pair
0.940 → 0.060 is exactly the cell verified above.

Two things are worth adding, neither a defect. **First**, neither table cross-references that
explanation, so a reader comparing Table 3 with Table 6 has to find it in the prose. One clause in
Table 3's caption would close it. **Second, and this is the substantive point:** training on HP-KR
yields an inverted probe *because HP-KR inverts* — the project's own central finding
**[LEDGER, F-13]**. It is worth naming as such rather than leaving it as a quirk of one cell.

### 2.3 The inversion is wider than previously recorded

Two places scope the sign inversion too narrowly. The audit's own DF-46 says *"only at L20, only in
`axis1`, only in the HP-KR training row"*; the draft's section 4 says *"that happens only for the
published contrast trained on HP-KR at layer 20."* Sweeping all seven one-dimensional arms at all
three layers **[RE-DERIVED]**, both are too narrow. Inverted training rows — always HP-KR's — appear
in **seven (arm, layer) cells**:

| layer | arms with an inverted HP-KR training row | printed → sign-corrected |
|---|---|---|
| L20 | `axis1`, `syc_axis1`, `syc_pc1`, `pc1` | 0.651→0.791 · 0.615→0.743 · 0.615→0.743 · 0.667→0.806 |
| L40 | `syc_axis1`, `syc_pc1` | 0.663→0.808 · 0.662→0.808 |
| L44 | `syc_axis1` | 0.708→0.849 |

In every case the sign-corrected mean equals the mean of the six zero-shot cells to four decimals —
the §2.1 identity, holding exactly.

**The draft's sentence is correct as scoped to its three headline contrasts** — `denial_axis1` and
`ext_axis1` never invert at any of the three layers. But the sycophancy contrast is reported in
Table 11 and Appendix F, and the published PC1 in the benchmark comparison, and both invert. The
practical consequence is in §5: **Table 11's off-diagonal row is depressed by this at all three
layers**, not just at L20. Either scope the sentence to the three contrasts explicitly, or note
Table 11.

---

## 3. Per-dataset zero-shot transfer

**Table 6 — zero-shot AUROC, no training at all.** **[RE-DERIVED, `zero_shot.csv`]** All 54 cells
reproduce exactly. This is the cleanest table in the paper: no fitting, so no sign ambiguity and
nothing to leak.

| test set | L20 (8v7 / 4v6 / 12v13) | L40 | L44 |
|---|---|---|---|
| Convincing Game | 0.940 · 0.931 · 0.969 | 0.865 · 0.902 · 0.912 | 0.819 · 0.858 · 0.859 |
| Harm-Pressure Choice | 0.717 · 0.585 · 0.639 | 0.581 · 0.755 · 0.712 | 0.607 · 0.725 · 0.712 |
| **Harm-Pressure KR** | **0.148** · 0.890 · 0.882 | 0.593 · 0.919 · 0.912 | 0.589 · 0.908 · 0.894 |
| Insider Trading | 0.988 · 0.976 · 0.990 | 0.979 · 0.913 · 0.980 | 0.979 · 0.933 · 0.982 |
| Instructed Dishonesty | 0.959 · 0.932 · 0.954 | 0.974 · 0.980 · 0.978 | 0.962 · 0.978 · 0.974 |
| RepE Honesty | 0.994 · 0.882 · 0.991 | 1.000 · 1.000 · 1.000 | 1.000 · 1.000 · 1.000 |
| **Mean** | 0.791 · 0.866 · 0.904 | 0.832 · 0.912 · 0.916 | 0.826 · 0.900 · 0.903 |

**Read down the HP-KR row.** Every arm is strong everywhere else; the published contrast collapses
to 0.148 at L20 and recovers only to 0.59. The extension's whole margin lives in this one row —
which is the honest way to state it, and is why §4's excluding-HP-KR column matters.

**RepE at 1.000 is a ceiling, not a result.** It is in-distribution for the base probe and saturates
for every persona arm at L40 and L44. It cannot separate arms and should not be averaged in without
a note.

---

## 4. Raw activations against the extended contrast

> **UPDATE 2026-09-24 (DF-106, P17).** At 16def2de the collaborator re-ran the baseline: CG raw L40
> = 0.7873, and the paper now prints 0.787 (`pt-base-cg-L40`). So 35 of 36 values match the draft
> PDF, and the draft's 0.788 is the ea1ce52b value. No persona column changed.
> **Source:** `audit/p17/d_rows_other_affected.csv` (D-195); `audit/claims.csv` D-189;
> `paper/generated/numbers.tex` @ 16def2de.

**Table 7 — per test dataset.** **[RE-DERIVED]** All 36 values reproduce exactly.

| test set | raw / 12v13, L20 | L40 | L44 |
|---|---|---|---|
| Convincing Game | 0.663 / 0.969 | 0.788 / 0.912 | 0.757 / 0.859 |
| Harm-Pressure Choice | 0.535 / 0.639 | 0.571 / 0.712 | 0.570 / 0.712 |
| Harm-Pressure KR | 0.394 / 0.882 | 0.584 / 0.912 | 0.637 / 0.894 |
| Insider Trading | 0.799 / 0.990 | 0.800 / 0.980 | 0.736 / 0.982 |
| Instructed Dishonesty | 0.778 / 0.954 | 0.827 / 0.978 | 0.808 / 0.974 |
| RepE Honesty | 0.718 / 0.991 | 0.824 / 1.000 | 0.785 / 1.000 |

The extended contrast wins in **all eighteen cells**. That is the paper's strongest single display —
and it is a comparison of a **trained 8192-dimensional probe** against a **label-free
one-dimensional direction**, because the `12v13` column is identical to its zero-shot value at every
dataset and every layer (§2.1, verified for all 18 pairs). The draft frames it that way already, in
section 3.4 and in the Contributions, which is the right framing: a zero-shot direction beating a
trained probe is the stronger claim, not the weaker one.

**Figure 4 shows the same comparison as a picture, and its caption is exact.** The left panel's raw
probe *"fits each training set but transfers unevenly and inverts in places"* — its HP-KR column runs
from **0.06 to 0.95** across training sets. The right panel's rows are constant **to four decimal
places, not approximately** **[RE-DERIVED]**, which is the §2.1 identity made visible.

---

## 5. The sycophancy contrast

**Table 11 and Appendix F.** **[RE-DERIVED, `zero_shot.csv` column `auroc_syc_axis`]** All 21
per-dataset cells reproduce exactly.

| test set | L20 | L40 | L44 |
|---|---:|---:|---:|
| Convincing Game | 0.999 | 0.998 | 0.983 |
| Harm-Pressure Choice | 0.533 | 0.751 | 0.692 |
| **Harm-Pressure KR** | **0.043** | **0.167** | **0.480** |
| Insider Trading | 0.983 | 0.985 | 0.987 |
| Instructed Dishonesty | 0.923 | 0.948 | 0.949 |
| RepE Honesty | 0.977 | 0.999 | 1.000 |
| off-diagonal mean, as printed | 0.615 | 0.663 | 0.708 |
| **off-diagonal mean, sign-corrected** | **0.743** | **0.808** | **0.849** |

**Appendix F's reading is right.** A contrast built to separate sycophantic from challenging personas
scores 0.92–1.00 on four datasets and **inverts on HP-KR at every layer**. Its own author's
inference — that *"part of what any contrast of this kind reads is the shared register of
role-played misbehaviour rather than deception itself"* — is the register hypothesis, and it is
correct.

Two refinements. The inversion **weakens sharply with depth**: 0.457 below chance at L20, 0.020 at
L44. And the printed off-diagonal row is depressed by the §2.3 sign artifact at **all three** layers,
understating the arm by 0.128 / 0.145 / 0.141.

**This project can say more than "suggests."** The register diagnosis was tested constructively:
a bank varying *capitulation* instead of register reads **0.745 zero-shot** on `sycophancy_mt` where
the paper's sycophancy axis reads **0.313** and its deception axis's PC1 reads 0.374, clearing a
dataset-level bootstrap against raw and both matched controls **[LEDGER, F-65/F-66]**. Appendix F
should cite that rather than leave the point as a suggestion.

**And the contrast is not independently constructed.** It is fitted on the same bank object as the
deception axis, with a measured **|cos| = 0.54** between them **[DF-24]**. The word "independently"
should go.

---

## 6. Against the Liars' Bench protocol

**Table 4 — the benchmark's own protocol: Alpaca-calibrated threshold, averaged over the five
Liars' Bench datasets.** **[RE-DERIVED, `zero_shot_operating.csv`, row `Average (Liars' Bench)`,
`threshold_source=control`]** All 36 values reproduce exactly.

| layer | arm | BA | recall | AUROC |
|---|---|---:|---:|---:|
| 20 | base probe (RepE-trained) | **0.721** | **0.489** | 0.697 |
| 20 | published contrast (8 v 7) | 0.534 | 0.069 | 0.750 |
| 20 | knowledge-denial (4 v 6) | 0.559 | 0.305 | 0.863 |
| 20 | extended contrast (12 v 13) | 0.591 | 0.187 | **0.887** |
| 40 | base probe | 0.626 | 0.281 | 0.778 |
| 40 | published contrast | 0.675 | 0.382 | 0.798 |
| 40 | **knowledge-denial** | **0.725** | **0.487** | 0.894 |
| 40 | extended contrast | 0.694 | 0.413 | **0.899** |
| 44 | base probe | 0.626 | 0.281 | 0.756 |
| 44 | published contrast | 0.654 | 0.318 | 0.791 |
| 44 | knowledge-denial | 0.683 | 0.395 | 0.880 |
| 44 | extended contrast | 0.666 | 0.350 | **0.884** |

**Layer 20 is where the paper's own argument is weakest, and it should say so.** There the base
probe beats every persona arm on both balanced accuracy (0.721) and recall (0.489), while losing on
AUROC (0.697 against 0.887). Ranking and calibration come apart, and the persona arms only win the
ranking. At L40 and L44 the persona arms lead on all three.

*(Two cells here are exact half-way values — 0.6255 and 0.6535 — where the draft rounds half up.
It is right to; an earlier reading of mine using banker's rounding made them look like mismatches.)*

---

## 7. What the abstract claims, and whether it survives

**[RE-DERIVED / PAPER]** All five abstract-level claims in this batch check out, with two scope
conditions.

- *"Linear probes for lie detection often fail to transfer across datasets."* Holds on his own
  numbers: the raw 8192-d probe reads **0.983 in-distribution against 0.732 off-diagonal** at L40, a
  0.251 drop; 0.979 against 0.648 at L20, a 0.331 drop.
- *"Mahadik and Skapars report better cross-dataset generalization at 3B and 8B."* Faithful to the
  cited paper — main model Llama-3.2-3B, "additional replications on Llama 3-8B" (p.4), contribution
  bullets on p.2 **[PAPER]**. **Two things belong beside it:** the paper's own words for 8B are
  *"auxiliary comparisons ... broadly supportive"* (p.5), not a full replication; and this project's
  three-rung ladder finds the persona gain **declines** with scale on **both** banks — 26-persona
  +0.038 → +0.015 → −0.012, 36-persona +0.036 → +0.019 → +0.005 **[LEDGER, F-121]**. That the
  decline appears on two independent banks makes it a property of the *method*, which is the
  strongest answer available to "you used the wrong personas."
- *"We extend the persona bank with ten roles."* Exact: `knowledge_denial` 4 + `sincere_ignorance` 3
  + `plain_assertion` 3 = **10**, giving 8+4 = 12 against 7+3+3 = 13 **[RE-DERIVED,
  `kd/personas/kd_role_groups.json`]**. Note that the draft's own persona code merges
  `sincere_ignorance` and `plain_assertion` into one arm, and that split exists to remove a wording
  confound **[DF-25]**.
- *"The benchmark's own probes score 0.50 and 0.26 there, and its upper bound 0.95."* All three exact
  **[PAPER, LB Tables 6–8 p.54–56]**. These are the **Llama-3.3-70B rows**; the cross-model averages
  are 0.50 / 0.42 / 0.72, so 0.26 in particular is model-specific. And see §1(c) on the upper bound's
  operating point.
- *"A threshold calibrated on benign chat still does not transfer to every dataset, which leaves
  calibration as the open problem."* **Holds, and is understated** — §1(b). This is the abstract's
  most defensible sentence and the measurements in §1(b) should be attached to it.

---

## 8. The setup, the bank and the token reads

**[RE-DERIVED at Level 1]** Everything in this section reproduces exactly.

**Table 1 — the evaluation corpus.** All seven rows exact against `zero_shot.csv`'s `n` and
`n_deceptive`: CG 267/167, HP-C 2,134/134, HP-KR 2,139/139, IT 1,080/423, ID 5,432/3,045, RepE
550/275, Alpaca 2,000/0. **The class imbalance is the thing to notice**: HP-C and HP-KR are 6% lies,
against 56% for ID and 50% for RepE. *One caveat:* the follow-up run uses a **different RepE row set
— 612/306, not 550/275** — so Table 10 mixes two row sets in one row.

**Table 5 — the persona bank.** All 35 per-persona kept-rollout counts match exactly, recounted from
the judge scores as `count(score == 3)` out of 150. The aggregates follow: **3,749 kept of 5,400
(69.4%)**, per-persona range **47–149** (coverup operator to tough love mentor). `default` is never
judged and keeps all 150 by design, which is what makes the total work. **This is the most
independently verifiable table in the paper** — it was recomputed from raw judge output, not read
from a summary.

**Tables 8, 9 and 10 — where in the response the signal lives.** All 100 cells exact.

- **Table 8** (response mean vs the follow-up `No` token, L20): the contrast's **sign stops being
  universal** at the follow-up token. CG 0.912, ID 0.900 and RepE 1.000 hold up, while IT collapses
  from 0.988 to **0.165** and HP-KR's knowledge-denial arm from 0.890 to **0.237** — both inverted.
  A single rescaling cannot produce that pattern, which is the draft's point.
- **Table 9** (six reads of the extended contrast): at L40 the **single maximum** leads on balanced
  accuracy (**0.750**) and recall (**0.548**), ahead of the top-10% mean (0.737 / 0.531) and the
  default mean (0.694 / 0.413). Where the draft recommends the top tenth as the better detection
  read, the maximum is better still **[DF-15]**.
- **Table 10** (trained on RepE, L20): the extended contrast reads **0.887** on response tokens and
  **0.552** at the follow-up token — the largest drop of any arm, and the clearest statement that
  the two tokens carry different signals.

**Where the token-before-the-reply read works, and where it does not.** Per dataset at L20, the
`pre` read gives RepE **1.000**, IT **0.927**, ID **0.900**, CG **0.871** — all datasets where the
instruction or setup is already in the prompt — and HP-KR **0.386**, inverted, against 0.882 for the
response mean. The draft's description (*"mostly reads the prompt … does not help on HP-KR"*) is
exact.

---

## 9. Four claims that must be updated before submission

These are the substantive `PARTLY` verdicts. None is an arithmetic error; each is a statement that
was true when written, or true of a narrower set than it claims.

**(a) "The benchmark reports recalls near zero for every method" on HP-KR — false as written.**
The Mean Probe does read **0.00**. But on the same Llama-3.3-70B cell the **Follow-Up Probe reads
0.37** and the privileged **Upper-Bound Probe 0.97** **[PAPER, LB Tables 6-8]**. The claim holds for
the non-privileged *cross-model averages* (0.00 and 0.12). **The correction improves the argument:**
the Follow-Up Probe's 0.37 is bought at an **84% false-positive rate**, which is a sharper criticism
than "near zero" and is the same defect §1(b) measures in the draft's own L20 arm.

**(b) The wording-controlled test is promised as future work, twice, and has been run three times.**
Sections 6 and 7 both say the cleaner feigned-versus-sincere contrast *"has not been run"*. It ran on
2026-09-19 (F-03/F-72/F-76), the row-leak contamination was closed on 2026-09-23 (**F-115**), and the
larger complementary control ran the same day (**F-116**, 2,001 rows, **7.8x** the text-matched
slice). At the pre-registered layer the margin is **+0.066, CI [-0.069, +0.189], P = 0.161**; it
**inverts to -0.2545 at L20**; and on the knowledge-matched control the persona axes sit **below
chance** while an ordinary probe reads **0.999** in-distribution **[LEDGER, F-115/F-116]**. Delete
the promise and state the outcome. **The paper is stronger with the negative result in it** — it is
the test the design was built for, and reporting that it does not clear is the control discipline the
thesis argues for.

**(c) Gender-Secret is listed as future work and is no longer open.** F-118 ran it through NDIF and
found the knowledge-denial direction **at chance where both classes deny (0.436-0.623)**. With F-115
and F-116 that is **three independent wording-matched tests converging on one answer**, and it should
be reported as one finding rather than three **[LEDGER, F-118]**.

> **UPDATE 2026-09-24 (P17).** Two sentences added to the paper after this chapter was drafted do
> not hold at the pin 16def2de:
> - **D2-005, `introduction.tex:68-70`** — "from layer~36 it ranks \hpkr{} better than any contrast
>   we build". The base probe leads at 8 of the 12 layers from 36 and trails the best contrast at
>   **L44 0.828 vs 0.908, L48 0.837 vs 0.890, L52 0.877 vs 0.883, L76 0.838 vs 0.925** (DF-96).
> - **D2-019, `limitations.tex:17-21`** — "≥0.026 on the other datasets at all 16 layers". The 0.026
>   includes HP-KR; excluding it the margin is **−0.011 at L20** and below 0.026 at 8 of 16 layers (DF-97).
> **Source:** `audit/p17/paper_new_claims.csv` D2-005, D2-019; `results/persona_depth/depth.csv` and
> `persona_ood_ext_L*/summary.csv` @ 16def2de; `audit/p17/A_REPORT.md` §2.

**(d) "The top tenth of response tokens is a better read than the mean" — true, but not the best
read.** At L40 top-10% beats the mean on balanced accuracy (0.737 vs 0.694) and recall (0.531 vs
0.413). The **single maximum** beats both (**0.750 / 0.548**) **[RE-DERIVED, Table 9]**. If a read is
being recommended for detection, recommend the maximum **[DF-15]**.

**And one scope word.** Section 4's *"that happens only for the published contrast trained on HP-KR
at layer 20"* is correct for the three headline contrasts and too narrow once the sycophancy contrast
and the published PC1 are included - seven (arm, layer) cells in all (§2.3).

---

## 10. Reproducibility: what is released, and the one thing that is not

**The release is better than it first looks, and I had it wrong twice.**

> **CORRECTION 2026-09-24 (DF-106, P17).** Read "1,161 macros at 2ad758fb (1,176 at 16def2de),
> beside eleven `tab_*.tex`". numbers.tex has 1,162 *lines* only at 2ad758fb (1,161 macros plus a
> header); there are 11 `tab_*.tex` at every SHA. The conclusion (the release is well evidenced) stands.
> **Source:** `audit/p17/A_REPORT.md` §1 (D-171, D-032); `audit/p17/C_REPORT.md` F-127 row;
> `git ls-tree 16def2de src/probes/deception-detection/paper/generated/`.

`paper/generated/numbers.tex` carries **1,162 `\setV` macros**, headed *"Generated by
make_numbers.py — do not edit"*, and every draft number checked in this chapter resolves to one of
them — dataset sizes, bank counts, the whole zero-shot grid, the cosine and the Figure 2
projections included. Beside it sit eight `tab_*.tex`, five `fig_*.pdf`, the full
`persona_ood_ext_L*` result CSVs, and `make_numbers.py` and `make_figures.py` themselves. So
*"we release the code and results"* and *"every number in the paper is generated from the released
result files"* are **substantially true**, and better evidenced than any comparable claim checked
here.

> **CORRECTION 2026-09-24.** Earlier drafts of this section said five numbers were unverifiable
> because they appear in no committed result file. Four of them — the cosine **0.47** and Figure 2's
> **−0.08 / 2.90 / 0.05** — are `\setV` macros in that committed generated file. I had looked only
> in the result CSVs. They have since also been **re-derived independently** (§10.1), and all four
> are exact.

### 10.1 Figure 2's geometry, rebuilt from the vectors

The collaborator's `EXTENDED_VECTORS_DIR` is `run-kd/vectors_s3` — its `pc_handler.pt` records
**n = 106**, which is both Table 5's count and the "64 → 106" his own docstring describes.
Following `make_figures.py:personas()` at layer 40 **[RE-DERIVED, lab PC]**:

| quantity | draft | measured |
|---|---|---|
| cosine(deception, denial) | 0.47 | **0.474559** |
| sincere ignorance, mean on the denial axis | −0.08 | **−0.0844** |
| feigned deniers | 2.90 | **+2.9037** |
| published deceptive | 0.05 | **+0.0458** |

All exact, and the caption's fourth clause holds too: the deceptive personas' maximum is **+0.885**
against the deniers' minimum **+2.150**, so the groups do not overlap. (L20 reads 0.417 and L44
0.468, so *"at layer 40"* is the right scope.)

**One thing the caption does not say, and it bears on the paper's central claim.** The two honest
groups are **not together**. Plain assertion sits at **−1.2274** and sincere ignorance at
**−0.0844** — a gap of **1.14** on an axis spanning 4.13. **Sincere ignorance, the group that
shares the deniers' wording, sits 28% of the axis closer to them than plain assertion does.** That
is the wording pull visible at persona level, it is exactly what DF-25 predicts when the two groups
are merged into one arm, and it makes *"sincere-ignorance personas sit with the honest cluster"* an
overstatement — they sit **between**, nearer the honest side. The draft's own hedge that follows
(*"whether the same holds for model responses is untested"*) is the right one, and F-118 has since
answered it: at chance where both classes deny.

### 10.2 The one real gap, and it is about re-running rather than reading

Two artifacts are absent, and they cost different things.

**The 36 persona vectors are not on the branch.** Their *numbers* are published, so nothing is
unchecked — but `make_figures.py:personas()` cannot run, so **Figure 2 cannot be regenerated from
the repository**. The vectors are small; shipping them closes this.

**The pipeline's own results tables do not exist outside the author's machine.** Appendix G reports
that rescored activations matched them *"to four decimal places at every layer"*.
`results/XXXX-9-llama-all*/` holds only `plots/*.png`, and a recursive search of the lab PC for
`results_table.csv` / `results.csv` / `scores.csv` and for any `XXXX-9-llama*` directory returns
**nothing** **[RE-DERIVED, lab PC]**. **This is the only claim in the draft that no one but its
author can check** — D-030 and D-063, the two remaining `UNRESOLVED`. The neighbouring Follow-Up
Probe figure (0.6360 measured against 0.6340 published) *does* reproduce from committed files, so
the gap is specific, not general. Either commit the table or scope Appendix G's sentence.

**Both cited works exist and are correctly characterised.** *"Probe Generalization as Subspace
Selection for OOD Deception Detection"* is **Daniel Yoo & Adrians Skapars**, arXiv **2609.02893**,
and its PDF is already in `papers/`; *"One Probe Won't Catch Them All"* is **Vikram Natarajan et
al.**, arXiv **2602.01425**. Neither has a row in `audit/papers.csv` — an audit bibliography gap,
not a citation problem. DF-19's separate point about two unverifiable venue strings still stands.

---

## 11. What the draft gets conspicuously right

The audit's own measured failure mode is leniency, so it is worth recording the other tail. Seven
disclosures the paper was not obliged to make, all verified:

- **HP-KR is genuinely untested by the replicated method** — the cited paper evaluates ConvGame,
  InstrDec, Mask, AILiar, Roleplay and a sycophancy set. The inversion was undetectable there.
- **The designed-repair caveat is declared twice** (§1 caveats and §6): the knowledge-denial personas
  were written *after* seeing the published contrast fail on HP-KR, so HP-KR is not a blind test.
- **The insider-trading scratchpad exclusion is disclosed voluntarily**, with the correct
  consequence — comparisons with the benchmark's published numbers on that dataset are not
  like-for-like; comparisons within the pipeline are exact.
- **The one-dimensional identity is stated with its condition** (§2.1) — including the clause that
  is precisely what fails at L20.
- **Intervals are over examples, drawn once and shared by every detector**, so side-by-side bars are
  compared on the same resampled rows.
- **RepE is excluded from averages** because it is the base probe's training set.
- **Calibration is named as the open problem** rather than buried — and the measurements in §1(b)
  support it more strongly than the sentence claims.

> **UPDATE 2026-09-24 (DF-107, P17).** Resolved by the collaborator at 2ad758fb, in his favour: the
> paper now reports the **L36 optimum** (`results.tex:228-230`, "Ranking peaks at
> layer~\V{depth-off-ext-peak-layer}, not at layer~40"; numbers.tex `depth-off-ext-peak-layer`=36,
> `depth-off-ext-peak`=0.927), and the "outscores … at every depth" claim is gone — `results.tex:136`
> now counts wins over the sweep (`depth-pc-wins-complement`=13 of `depth-n`=16). The one remaining
> "every depth" (`results.tex:208`) is a different claim, about the base probe inverting.
> **Source:** `src/probes/deception-detection/paper/sections/results.tex` and
> `paper/generated/numbers.tex` @ 16def2de; `audit/p17/A_REPORT.md` §2.

One caveat has since been vindicated. *"Three layers do not locate an optimum, and a layer between 20
and 40 may do better"* — the 16-layer sweep puts `ext_axis1`'s optimum at **L36 = 0.9275** against
L40's 0.9158 **[LEDGER, F-117]**. The paper should now report that rather than leave it hypothetical;
under-reporting your own best number invites the question of why **[DF-79]**.

---

## 12. What this chapter does not certify

> **CORRECTION 2026-09-24 (DF-101, P17).** "Within ±0.06 at every layer" and "crosses zero five
> times" are both wrong. Excluding HP-KR, `ext_axis1 − ext_pc3` reaches **+0.0633 at L56**, and the
> series changes sign **6** times. Still correct: 11/16 positive, +0.0086 (L20), +0.0012 (L24), and
> negative at L64/68/72. No persona-arm cell changed between ea1ce52b and 16def2de.
> **Source:** `src/probes/deception-detection/results/persona_ood_ext_L{20..79}/summary.csv`,
> `off_diagonal_excl_hpkr` @ 16def2de; `audit/p17/ledger_verify_stdout.txt:223-224`.
> **How the claim is updated:** *"excluding HP-KR the margin is positive at 11 of 16 layers, lies
> between −0.033 and +0.063, and changes sign six times."*
> _The text below is superseded on this point and is kept for the record._

- **Everything here is one seed, his pipeline, three layers.** The 16-layer sweep on `origin/paper`
  **[LEDGER, F-117]** shows `ext_axis1` peaking at **L36 (0.9275)** rather than L40 (0.9158), and the
  excluding-HP-KR margin crossing zero five times across 16 layers. Three-layer tables should not be
  read as a depth result **[DF-79, DF-81]**.
- **No significance test is applied to any table in this chapter.** With five or six test datasets
  and one seed, none would be informative — which is itself worth stating rather than leaving the
  reader to assume the margins are tested.
- **`repe_honesty` is in the six-dataset grid and is not a Liars' Bench dataset**; his own file
  labels the row `n_datasets=5` **[DF-23]**. Wherever "six Liars' Bench datasets" appears it is five.
- **6B (collaborator notes, 92 claims) and 6C (ledger, 110 claims) were not run.** They do not gate
  the paper, but coverage should not be described as complete. See
  `audit/LIMITATIONS_AND_FUTURE_WORK.md`.
