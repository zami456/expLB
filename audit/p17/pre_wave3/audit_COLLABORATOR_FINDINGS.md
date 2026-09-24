# Everything the collaborator has found — the complete record

_Assembled 2026-09-24 from all three of his branches. Every number read from git during the audit.
Companion to `audit/COLLAB_PAPER_BRANCH.md` (the deep review of his newest branch)._

**Why this file exists.** His work is spread across three branches, 19 result folders and 17
write-ups, and the audit's own records discuss it claim-by-claim rather than as a body of work. This
is the single list. The owner is presenting joint work and needs to be able to say what he did.

**The standing limitation, which applies to every row below.** **No pooled activations are committed
on any of his branches** (`.../activations` is empty). So nothing of his is recomputable from raw.
Everything here is checked by (a) internal consistency across his tables, prose and figures, and
(b) recomputing derived aggregates from his committed matrices. **His numbers are attested, not
reproduced.** Where the audit could check an aggregate it did, and it matched.

---

## A. His positive results

### A1. The knowledge-denier persona bank — the project's headline repair
On the benchmark's hardest dataset, where the model denies knowing something it was just shown:

| | AUROC |
|---|---:|
| published persona contrast | **0.148** — inverted |
| the benchmark's own probe | 0.12 |
| **his extended contrast** | **0.919** |
| upper bound allowed to see labels | 0.95 |

He diagnosed *why* the published bank fails there — every one of its deceptive personas **asserts**,
while this dataset's lie is a **denial** — and fixed it by adding ten roles that feign ignorance, are
sincerely ignorant, or plainly assert. **Within 0.03 of a method allowed to cheat, using no labels.**
Across his full 6-dataset grid the extended contrast reaches **0.916** against **0.732** for
8192-dimensional probes trained on one dataset each.

### A2. One dimension beats three — the PCs add nothing
The 1-d contrast outscores a probe on the 3 leading persona PCs. **Verified**, and see C1 for the
large qualification.

### A3. The 16-layer sweep (new, `origin/paper`)
16 layers (L20–76 step 4, plus L79) where the study had 3. **Validated, not assumed:** L40 rebuilt
both ways, pooled vectors agreeing at row cosine **≥ 0.99989**, same AUROC to 4 dp, retrained RepE
probe at cosine **0.9996**.
**Result that helps him:** against his own baseline the extended axis is positive at **16 of 16
layers, +0.115 to +0.273**, and **still 16 of 16 excluding HP-KR** (+0.123 to +0.243). Depth-robust
and not HP-KR-driven. **This was invisible from three layers and belongs in the paper.**

### A4. Gender-Secret — his strongest contribution to the negative result
A LoRA fine-tune that believes the user is male and denies knowing it, so **both classes are
denials**: the benchmark's own wording-matched test.

| detector | L20 / L40 / L44 |
|---|---|
| knowledge-denial (4v6) | 0.549 / **0.436** / 0.498 |
| extended (12v13) | 0.505 / 0.467 / 0.541 |
| published (8v7) | 0.514 / 0.545 / 0.623 |
| sycophancy (5v5) | 0.593 / 0.523 / 0.520 |

**Every persona direction at chance.** Independently confirms the register diagnosis (F-13).
**The best-controlled experiment on either side:** adapter applied inside the remote trace, pinned to
PEFT agreement within 1e-5 by a test shown to fail without it; behavioural check (82% male-answer vs
55% base, both answer orders, ruling out letter bias); base-read control with shared bootstrap
resamples so the difference is paired.

### A8. Four sycophancy arms we never ran — and an audit correction (F-122)

**Credit him with these: our pipeline has no sycophancy axis at all.** `27_crossgen.py`'s default
`--features` omits it, so `pc:sycophancy` has never been trained on our side at any rung. His branch
carries **`syc_pc1`, `syc3`, `syc_axis1`, `syc3_orth` at all 16 layers**.

*(An earlier version of this audit said the sycophancy axis was "never computed anywhere". That was
a search of `run-crossgen/` only — our pipeline — stated as if it covered the project. Corrected.)*

**What they show.** Excluding HP-KR, `syc_pc1` is the **best arm at L40, 44, 48, 52 and 56**,
reaching 0.9370 at L40 against `ext_axis1`'s 0.9165 and his baseline's 0.7581. Win counts over the
sweep: `syc_pc1`=5, `ext_pc3`=4, `ext_axis1`=4, `axis1`=1, `uni_axis1`=1, `syc3`=1 — **no arm
dominates.**

**What they do not show.** The margin is **one dataset**. At L40, `syc_pc1` − `ext_axis1` is
`convince_me` **+0.086**, `harm_refusal_choice` +0.041, `insider_trading` +0.005,
`instruct_dishonesty` **−0.028**, `repe_honesty` −0.001 — 3 wins of 5, and `convince_me` carries the
margin at every layer. `convince_me` is persuasion under pressure, the one place an
agreement-versus-pushback axis *should* fire, so it is coherent evidence for the register diagnosis
and it is n=1.

**And a measurement point that applies to his headline arms too.** For every `dim=1` arm the trained
off-diagonal **equals** the zero-shot AUROC — one scalar admits only a sign. Spread across training
rows is **0.000 at all 16 layers** for `ext_axis1` and `denial_axis1`. Their 30-cell off-diagonal
mean therefore holds **6 distinct numbers**, replicated five times each, while the baseline they are
compared against is a trained 8192-d probe. Not wrong, but it must be **labelled zero-shot**, with
any interval taken over the 5 test datasets rather than the 30 cells.

---

### A5. The pooled leave-one-dataset-out baseline
`loo_L20/`, `loo_L40/`, committed 2026-09-20 under *"Leave-one-dataset-out: the fair baseline for the
generalization claim"*. **His own work, and the correct comparator.** See C2.

### A6. Engineering the project reused
The resumable activation cache and batcher (ported into `asix` at `ed8daca`), tail-only activation
extraction, the LoRA-inside-trace machinery, and two new test files (`test_layer_sweep.py`,
`test_lora_intervention.py`).

### A7. Per-token and follow-up-token experiments
The persona direction at every token (the mean ranks best, the top 10% detects best); the follow-up
token, where the direction's sign stops being universal; k=3 at both token positions.

---

## B. Where his work and the audit AGREE

- **Two pipelines, one answer.** Across **55 bank-matched cells** his numbers and ours agree to a mean
  absolute gap of **0.0015**. There is no computational bug on either side.
- **51 of 79** checked draft claims **match his data**; exactly **one** did not (C5, now resolved).
- Tables 1, 3, 4, 5, 6, 7 reproduce from his committed CSVs; his transcriptions of the published
  comparators are **exactly right in every case**.
- His per-dataset and off-diagonal aggregates recompute from his own matrices to 4 dp.

---

## C. Where the audit disagrees, or found a limit

### C1. The headline margin is one dataset — now shown across 16 layers
`ext_axis1 − ext_pc3`, **excluding HP-KR**: +0.0086 (L20), +0.0012 (L24), **within ±0.06 at every
layer**, crossing zero five times, positive at only **11 of 16**.
**And "outscores at every depth" is falsified by his own sweep** — negative at **L64, L68, L72**.

### C2. The draft says the pooled baseline was not compared; he compared it himself
The Limitation states a pooled multi-dataset probe *"is a stronger supervised baseline, and we do not
compare against it"*. `loo_L20/` and `loo_L40/` were committed **two days earlier**. Excluding HP-KR,
removing it costs the axis **0.048–0.130 AUROC** in all four (layer, weighting) cells; in two of four
what remains is **≤ +0.002 or negative**.

### C3. The central dataset is largely solvable from wording
A **label-free regex** scores **0.9125** / **0.9597** on it — at or above his headline **0.919**.
**Length alone reads 0.776.** On the wording-matched slice his extended axis reads
**0.4630 / 0.6203 / 0.5383** and his split knowledge-denial axis **0.3932 / 0.4989 / 0.4825** — at or
below chance. **Neither number is in either document.**

### C4. Selective reporting across banks
Three persona banks exist; the draft reports two. The third (`persona knowledge denial modified`) is
never mentioned, and **his own summary says it makes every leading arm worse at every depth**.

### C5. RESOLVED IN HIS FAVOUR — `paper/make_numbers.py`
The audit's **only** `DOES NOT MATCH` verdict (D-237) rested on that script being absent from every
branch, every commit and the lab PC. **It now exists** on `origin/paper`. Superseded; F-119.

### C6. Smaller items
- **Optimum layer is 36, not the reported 40** (0.9275 vs 0.9158) — he under-reports his own best.
- **"HP-KR is a spike at 40" needs its arm named**: true for `denial_axis1` (global max 0.9185), but
  for `ext_axis1` **L76 (0.9249) > L40 (0.9124)**.
- **One test set is the base probe's own training set** (`repe_honesty`); drop it at L20 and the
  contrast falls *below* its baseline.
- **Wrong k for the comparison**: the published deception figures are **k=1**; k=5 appears nowhere in
  that paper. At k=1 his own summary has PC1 beating the contrast at all three layers.
- **Error bars cover example sampling only**, while re-extracting a bank moved HP-KR by **0.036–0.077**.
- **Table 3's "Random directions (k=3)"** is a **trained** probe averaged over 5 draws with no spread
  stored, presented as a single random basis.
- **Balanced accuracy 0.500** in Table 2 means "flags nothing" (recall 0.0), not chance.
- **`persona_depth/summary.md` cites `../persona_unified/` and `../persona_depth_orig/`** — neither is
  on the branch.
- **A citation is mistitled**: arXiv:2602.01425 is given as *"One probe won't catch them all"*; two
  in-repo PDFs title it *"Building better deception probes using targeted instruction pairs"*.
- **Ten of the twenty borrowed questions** are this project's own renderings of the published topics,
  not verbatim; the draft discloses reconstruction for persona prompts but not for these.
- **D-107 was corrected in his favour**: his random control is the mean of five draws, not one.

---

## D. Where his work and ours disagree on the same setting

| item | his | ours | reading |
|---|---|---|---|
| `s3` vs `s23` bank swap | a measured loss (`ext_axis1` 0.9158 → 0.9066 at L40; HP-KR 0.912 → 0.876) | F-62: "no measurable difference" | **his is right**; our replication of his arms is built on the bank his own control found worse |
| HP-KR at L20 | six values circulate, 0.103 → 0.2504 | — | three are quoted without naming the bank |
| "the paper's 8v7 axis" | his = the repaired shared-question bank | ours = the question-confounded one | up to **0.148** apart on HP-KR L20 |
| four L20 matrices | sign-flipped HP-KR training row (0.6511 → 0.7909 corrected) | — | inflates the L20 lift only; the L40 headline is untouched |

---

## E. The convergence — and it is the paper's strongest result

Three **independent** wording-matched tests, two of them his, all agreeing:

| route | who | result |
|---|---|---|
| his axes on our wording-matched slice (F-115) | ours | 0.4630 / 0.6203 / 0.5383 |
| our `knowledge_matched` control, 2,001 rows (F-116) | ours | transfer **below chance** |
| **Gender-Secret** (F-118) | **his** | **every direction at chance** |

**Where both classes deny knowledge, no persona direction reads the lie.** That is the register
diagnosis, confirmed three ways, on data neither side built for the purpose. **Report it as one
finding.**

---

## F. What he should be credited with in the writeup

1. **Diagnosing why the published bank inverts** on the hardest dataset — the assert/deny mismatch.
2. **Building the ten roles that repair it**, 0.148 → 0.919 with no labels.
3. **Gender-Secret** — the cleanest wording-matched evidence in the project, and the best-controlled
   experiment either side has run.
4. **The 16-layer sweep**, validated against the old path rather than assumed.
5. **The pooled leave-one-dataset-out baseline** — the fair comparator, which he built himself.
6. **The resumable cache and batcher** the rest of the project reused.
7. **Two test files** pinning the new machinery.

**And the honest counterpart:** the headline margin is one dataset; that dataset is ~91% solvable from
wording; and the fair baseline he built erases most of the advantage. **None of that is a mistake — it
is what the data says, and he generated most of the evidence for it himself.**
