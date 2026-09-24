# What the draft misses, overstates or gets wrong — in both directions

_Phase 3, workstream **P3-A**. Written 2026-09-23. Read-only: every number below comes from a file
opened during this audit, a row of `audit/claims.csv` that already carries a verdict, or a script in
`audit/scripts/phase3/`._

**Who this is for.** Both authors. It is deliberately symmetric: §B is as long as it needs to be,
and three of its items correct the owner, not the collaborator.

**What this is.** The draft (`audit/phase0/draft_text.txt`, 19 pp, the extracted text of
`full paper draft for this experiments/main (1).pdf`) is measured against (a) the owner's **checked**
findings and (b) the collaborator's **own committed files**. It is about *missing and contradicted
substance*. It is **not** a second copy of `docs/DRAFT_CORRECTIONS.md`, which lists arithmetic and
wording errors already delivered to him; where the two touch, this file says so and stops.

**Tags.** **VERIFIED** = re-derived from raw data in this audit (or carries a verdict in
`audit/claims.csv`) · **DOC-ONLY** = asserted in prose only · **CONFLICT** = two sources disagree,
both shown · **NOT-FOUND** · **INFERRED** = my reading, with the reason given.

**Verdict words.** **GAP** = the draft is silent · **CONTRADICTION** = the draft says the opposite ·
**COMPATIBLE** = both are right and they measure different things · **UNCLEAR** = the evidence does
not settle it.

---

## 0. The short version, ranked

Items that **would change a claim** in the draft, most consequential first:

| # | In one line | Verdict | Tag |
|---|---|---|---|
| **A-1** | The draft's "the principal components add nothing" is **false at the replicated paper's own setting**. The paper's deception figures use **k = 1**; his own `summary.csv` has a `pc1` row; published PC1 beats the published contrast at **all three layers** (+0.016 / +0.027 / +0.025). Table 3 has an "Extended PC1" row and **no "Published PC1" row**. | CONTRADICTION | VERIFIED |
| **A-2** | The draft says a pooled supervised probe "is a stronger baseline, **and we do not compare against it**". **His own branch made that comparison** (`loo_L20`, `loo_L40`, 2026-09-20), and on the owner's data the persona advantage largely disappears against it (+0.184 → +0.048 → +0.023 at L40; both his arms **lose** to pooled LODO on eleven datasets). | CONTRADICTION | VERIFIED |
| **A-3** | The draft lists the wording-controlled test as future work ("we have not run it"). It **was built and run**. Its answer is **inconclusive, not positive**: margin +0.066, 95% CI **[−0.069, +0.189]**, P = 0.161; the same cell reads 0.374 at L20 and 0.739 at L40; and the slice leaks into HP-KR by construction. | CONTRADICTION (and a warning) | VERIFIED |
| **A-4** | The paper-faithful deception axis is **inside a 1,000-direction random null on 4 of 9 datasets**, including HP-KR and convincing-game — two of the datasets the draft's zero-shot numbers lean on. Absent from the draft *and* from the owner's ledger. | GAP | VERIFIED |
| **A-5** | Drop `repe_honesty` (the base probe's own training set) from **his own L20 matrices** and the paper-faithful contrast falls **below** the trained baseline it is meant to beat: 0.5900 vs 0.6085. The draft's "all six test sets" column includes it. | GAP | VERIFIED |
| **A-6** | The draft's only control is **three random directions**. The replicated paper itself runs a **Dataset PCA (k = 1)** control, and it is the matched dataset-PCA control that kills persona features on the owner's grid (0 of 126 tests clear). | GAP | VERIFIED |
| **B-1** | The owner's ledger says the s3→s23 bank swap makes "**no measurable difference**" (largest gap 0.005). **His own control measures a real loss** on the same swap: 0.9158 → 0.9066, HP-KR 0.912 → 0.876. And her replication of his arms is built on **s23 — the bank his control found worse**. | CONFLICT | VERIFIED (the coupling is INFERRED) |
| **B-2** | The owner's briefing §5c critiques a draft **titled "One Direction Generalizes"**. The committed PDF is titled *"Lying in Knowledge-Denier Persona Coordinates"*, and several of the briefing's §5 corrections are **already fixed** in it. Parts of the briefing are arguing with a text he has replaced. | CONFLICT | VERIFIED |
| **B-3** | The audit's own `claims.csv` row **D-107** says his random control is "a single random draw". It is **the mean of five** (`N_RANDOM_DRAWS = 5`). A correction that runs in his favour. | CONFLICT | VERIFIED |

Everything else below is an **addition** — material worth putting in the paper that does not
overturn a sentence.

---

## A. Her work the draft misses or contradicts

The eight topics of `audit/phase0/draft_vs_ledger_preview.md` were DOC-ONLY when written. Each now
gets a verdict. Items **A-9** onwards are new — they were not in the preview.

### A-1 · The principal-components claim fails at the paper's own k **[CONTRADICTION · VERIFIED]**
_(preview topic: none — this is new; it deepens `DRAFT_CORRECTIONS` 1.2)_

**What was tested.** Whether the draft's contribution 5 — "One direction beats the persona subspace",
abstract: *"the persona principal components add nothing beyond it"* — holds at the configuration the
replicated paper actually uses.

**What happened.** The replicated paper's deception method-comparison figures are at **k = 1**. Read
straight out of `papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf`: figure titles
*"Method Comparison · Deception · Llama 3.2 3B (Layer 14, **k=1**)"* and *"Llama 3 8B (Layer 16,
**k=1**)"*, legend entries *"Random (k=1)"*, *"Dataset PCA (k=1)"*, *"Axis PC Proj (k=1)"*, and a
figure titled *"**PC1 vs Baseline** AUROC Improvement · Deception Datasets"*. The draft instead
reports *Published PCs (k = 3)* and *Extended PC1*, and gives **no Published PC1 row**.

> **LEAD QA 2026-09-23 — CONFIRMED and sharpened by an independent `pypdfium2` extraction of all 14
> pages.** The paper uses **two different k, one per behaviour**, and this matters:
>
> | figure | k | page |
> |---|---|---|
> | "Method Comparison · **Deception** · Llama 3.2 3B (Layer 14, **k=1**) [HPC excluded]" | **1** | 12 |
> | "Method Comparison · **Deception** · Llama 3 8B (Layer 16, **k=1**) [HPC excluded]" | **1** | 13 |
> | "Method Comparison · **Sycophancy** · Llama 3.2 3B (Layer 14, **k=3**)" | 3 | 12 |
> | "Method Comparison · **Sycophancy** · Llama 3 8B (Layer 16, **k=3**)" | 3 | 13 |
>
> Counts over the whole PDF: `k=1` **12 times**, `k=3` **9 times**, **`k=5` zero times.** So
> (a) A-1's premise holds — the paper's *deception* setting is k=1; (b) our `pc:deception` at the
> project default **k=5 is a value the paper never uses anywhere**, which independently confirms
> M2's AD-1 and W6A's O-073; and (c) "the paper's k" is ambiguous unless the behaviour is named —
> deception is k=1, sycophancy is k=3.
>
> **A fourth independent confirmation of DF-50 falls out of this:** both deception figure titles
> carry **"[HPC excluded]"**, i.e. the paper explicitly drops HarmPressureChoice — the dataset its
> own Table 1 declares — from the figures, and substitutes Mask. Table 1 and the figures disagree,
> exactly as the owner's F-60 recorded.

His own committed file has that row. From
`results/persona_ood_ext_L{20,40,44}/summary.csv` on `origin/feat/persona-pc-ood`
(`off_diagonal_mean`, recomputed by `audit/scripts/phase3/p3a_pc1_vs_axis1.py`):

| | L20 | L40 | L44 |
|---|---:|---:|---:|
| `axis1` — published contrast (8 v 7) | 0.6511 | 0.8319 | 0.8260 |
| **`pc1` — published PC1, the paper's own k = 1** | **0.6669** | **0.8593** | **0.8506** |
| difference (PC1 − contrast) | **+0.0158** | **+0.0274** | **+0.0246** |

**How sure, and why.** VERIFIED twice over: the paper's k from the PDF's own figure titles, the
numbers from his own CSVs by script. Caveat: with HP-KR excluded the gap is −0.0033 / +0.0128 /
+0.0144, so at L20 it reverses.

**Why it matters.** The abstract's third contribution says the paper's central operation does not
contribute. At the paper's own setting, on his own rows, **it does** — and the one table row that
would show it is the one Table 3 omits. A reviewer of a replication will look for exactly that row.
`DRAFT_CORRECTIONS` 1.2 already flags the underlying number; what is new here is that the
configuration the draft never runs is the paper's own, and that a well-motivated fix exists: compare
the PCs against a **matched dataset-PCA control at the same k** (A-6), where they lose.

**Evidence.** `papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf` (figure titles);
`origin/feat/persona-pc-ood:src/probes/deception-detection/results/persona_ood_ext_L{20,40,44}/summary.csv`;
`audit/scripts/phase3/p3a_pc1_vs_axis1.py`; `docs/papers/persona-coordinates-notes.md` §5, §9;
`docs/DRAFT_CORRECTIONS.md` 1.2.

---

### A-2 · The pooled supervised baseline **[CONTRADICTION · VERIFIED]** — preview topic 1

**What the draft says (p.12).** *"A probe trained on several datasets pooled is a stronger supervised
baseline, and we do not compare against it."*

**What the files say.** Three separate things, all VERIFIED:

1. **His own branch made the comparison**, one day before her version of it. `loo_L20/`, `loo_L40/`
   at `6b16a46` (2026-09-20 20:05, zami456) — *"Leave-one-dataset-out: the fair baseline for the
   generalization claim"*. His own note is unusually candid: *"The straw-man charge was partly
   right… the margin does shrink."* (`audit/methods/ATTRIBUTION.md` §3.1, red flag D.)
2. **On his six datasets** the persona lead over a pooled probe is small. Pooled LODO reaches
   0.7365 (L20) / 0.8683 (L40) and row-budget-matched 0.7654 / 0.8930, so the abstract's advantage
   moves **+0.184 → +0.048 → +0.023** at L40 (claim **F-85**, PARTLY HOLDS; the numbers hold, the
   framing depends on F-97). At **L32** pooled training is higher still, cutting the margin to
   **+0.015** (**F-109**, HOLDS).
3. **On eleven datasets both his arms lose.** `ext_axis1` loses to pooled LODO by **0.064** (L20) and
   **0.055** (L40); the paper-faithful `axis1` by **0.146** and **0.122** (**F-98**, **O-009**, both
   HOLDS; pooled LODO wins at 21 of 21 stored layers).
4. The pooled result is **the project's strongest positive finding** and it needs no personas at all:
   +0.178 over single-source at L40, CI [+0.143, +0.213], P(≤0) = 0, winning on 11 of 11 targets
   (**F-80**, HOLDS), positive at 21/21 layers (**F-81**), stable over 5 seeds (**F-83**), and the
   gain is **diversity, not data volume** — the row-budget-matched arm keeps it (+0.174).

**Two caveats that must travel with it.** (i) On the **six**-dataset subset at L20 the same bootstrap
**does not clear** (+0.093, CI [−0.045, +0.210], P = 0.089) — it is n = 11 that makes the headline
tight (F-85 L2). (ii) Under leave-one-**cluster**-out, pooled training still wins (+0.129 / +0.131 /
+0.125), so the gain is not the HP-KR/kr-matched row leak (**F-108**, **O-003**, HOLDS).

**Why it matters.** The Limitations sentence is factually wrong about the authors' own work, and the
comparison it declines to make is the one that most reduces the headline. Stating it costs the paper
+0.184 → ~+0.02–0.05 on the six datasets, and buys a defensible claim his own note already writes:
**robustness, not peak accuracy** — the label-free direction never collapses on an unseen dataset
(floor 0.64 / 0.71) while pooled supervision falls to 0.12 / 0.49.

**Evidence.** Draft p.12; `origin/feat/persona-pc-ood:.../results/loo_L20/summary.md`;
`audit/claims.csv` rows F-79, F-80, F-81, F-83, F-85, F-98, F-109, O-003, O-009;
`audit/methods/ATTRIBUTION.md` §3.1, §3.2, §7 D/E; `kd/docs/FINDINGS.md` F-113 §1.

---

### A-3 · The wording-controlled test was run, and it did not pass **[CONTRADICTION · VERIFIED]** — preview topic 3

**What the draft says (p.12, repeated p.13 as future work).** *"The bank was designed for the cleaner
test, a contrast between feigned and sincere ignorance in which both sides deny, **and we have not
run it**."*

**What the files say.** It was specified on **2026-09-19** (`kd/docs/DESIGN.md` at `5b43a7a`), built,
and run — as the 258-row `harm-pressure-kr-matched` slice plus two teacher-forced factor probes. The
draft is right that the confound exists, and its diagnosis is the owner's, quantified before any
result existed: `DESIGN.md` at `5b43a7a` already contains the 2×2 table (`deceptive = knows AND
denies`, fourth cell **0**) and the measured denial-regex AUROC **0.870**.

**And the answer is not the one the draft hopes for:**

| test | result | claim |
|---|---|---|
| a denial regex alone on HP-KR (Llama rows) | **AUROC 0.913** — the dataset is mostly one lexical bit | F-01 HOLDS |
| the matched slice, trained persona axes vs raw, L40 | 0.752 vs 0.715 — the reversal | F-03 HOLDS |
| **the same cell at L20** | **0.374**, i.e. 0.216 *below* raw; span over depth **0.365** | F-72 HOLDS |
| **dataset-level bootstrap at the pre-registered L40** | **+0.066, CI [−0.069, +0.189], P = 0.161 — does not clear** | F-76 HOLDS, O-004 HOLDS |
| which layer is the peak? | 95% argmax interval **[L5, L79]** — 93% of the network | F-76 |
| label permutation | **p = 0.38** | O-005 HOLDS |
| over all 80 layers | persona axes beat raw at **28 of 80** and dataset-PCA at **31 of 80** | F-72 L2 |
| leakage | `kr-matched` row-ids are HP-KR row-ids **by construction**; ~81% of its test rows sit in HP-KR's train split | F-108 HOLDS |

The two factor probes did pass their pre-registered validity gate (`d_utterance` 0.996–0.999 on the
knowledge-matched slice, `d_knowledge` 0.685–0.772 on the text-matched slice, **F-05** HOLDS), and
orthogonalising the wording direction against the knowledge one drops it to ~0.5 at L40/L44 as
predicted — **but not at L20, where it reads 0.33** (**F-06** HOLDS).

**Why it matters.** Two things change. First, the draft cannot say the test has not been run — it has,
by the co-author who supplied the personas. Second, and more important, **it must not be promised as
the fix**: run, it is inconclusive and strongly layer-dependent. The honest sentence is "we ran a
wording-matched slice; at the pre-registered layer the margin is not distinguishable from zero at
n = 10 datasets, and the effect is confined to layers 40–44." That is a *better* limitation than the
one the draft has, because it is measured.

**Evidence.** Draft p.12–13; `kd/docs/DESIGN.md` §1–§4 at `5b43a7a`; `audit/claims.csv` F-01, F-02,
F-03, F-05, F-06, F-16, F-72, F-76, F-108, O-004, O-005; `audit/methods/ATTRIBUTION.md` §1.1, §2.9,
red flag C.

---

### A-4 · The paper-faithful axis is inside a random null on four datasets **[GAP · VERIFIED]** — new

**What was tested.** Per-dataset, 1,000 random unit directions on the same rows and layer; empirical
p on |AUROC − 0.5|. (This is the owner's per-dataset zero-shot null, `25_zeroshot_nulls.py`,
2026-09-17 — a different object from the draft's `random3`.)

**What happened.** At L40, mean pooling, the paper16 **deception** axis beats the random 95th
percentile on only **5 of 9 datasets**. It is indistinguishable from a random direction on
**convincing-game (0.788, p = 0.238)**, **harm-pressure-choice (0.544, p = 0.771)**,
**HP-KR (0.644, p = 0.625)** and **mask (0.467 — inverted, p = 0.711)**. Claim **S-14**, PARTLY HOLDS;
all nine AUROCs re-derived from raw activations in this audit and identical to 4 dp.

**How sure.** VERIFIED at Level 2. One composition caveat the audit found: the *pooled* 0.881 includes
the 2,000-row Alpaca control as pure negatives and reads 0.863 without it, so pooled and per-dataset
figures are not directly comparable.

**Why it matters.** The draft's Table 6 reports the published contrast's zero-shot AUROC on exactly
these datasets (CG 0.940/0.865/0.819, HP-C 0.717/0.581/0.607, HP-KR 0.148/0.593/0.589) and the
abstract leans on the zero-shot framing. On the owner's rows, a random direction does about as well
on four of them. This does not touch the *extended* contrast's HP-KR result, which is much larger;
it does bear on how strongly a **label-free direction** can be claimed to "transfer" in general. It
is also absent from the owner's own `FINDINGS.md` (it has no F-number) — so neither author has it
written down where it would be used.

**Evidence.** `audit/claims.csv` S-14; `run-coords/zeroshot_nulls.json` key `corpus_L40_mean`;
`docs/thesis/persona-coordinates-70b-results.md` §8.2 (lines 930–944);
`src/assistant_axis_ndif/25_zeroshot_nulls.py`.

---

### A-5 · RepE is inside the "six datasets" and it flatters the contrast **[GAP · VERIFIED]** — new

**What was tested.** Recompute his own L20 transfer matrices with the `repe_honesty` row and column
dropped (20 cells instead of 30).

**What happened.** The paper-faithful 8-vs-7 contrast falls to **0.5900** against the single-source
baseline's **0.6085** — i.e. **below the baseline it is meant to beat, by 0.019**. With RepE in, the
same pair reads 0.6511 vs 0.6478 (+0.003). The reversal is L20-specific: at L40 the same operation
gives +0.092. Claim **S-11**, HOLDS; re-derived cell by cell from his committed matrices at two
commits.

**How sure.** VERIFIED at Level 1 only — his activations are gone, so nothing here was recomputed from
activations; but every mean was recomputed from his per-cell CSVs rather than read off a summary line.

**Why it matters.** RepE is the base probe's **own training set**, and his own
`zero_shot_operating.csv` distinguishes "Average" (6) from "Average (Liars' Bench)" (5). Table 3's
"All six test sets" column therefore contains one in-distribution column for one of the two arms
being compared. The draft already reports an "Excluding HP-KR" column; an "Excluding RepE" column
costs nothing and is the one a referee will ask for. **Absent from both the draft and the ledger.**

**Evidence.** `audit/claims.csv` S-11, A-03;
`origin/feat/persona-pc-ood:.../results/persona_ood_ext_L{20,40}/matrix_{baseline,axis1,pc1,ext_axis1}.csv`.

---

### A-6 · Nine controls the draft does not have, one of which is claim-changing **[GAP · VERIFIED]**

I confirmed the zero-occurrence counts myself over `audit/phase0/draft_text.txt` (`grep -ci`):
`capitulat` 0 · `sarcas` 0 · `teacher` 0 · `datapca` 0 · `dataset PCA` 0 · `bag of words` 0 ·
`length` 0 · `pre-regist` / `preregist` 0 · `factor probe` 0 · `leave-one` 0 · `sycophancy_mt` 0 ·
`multi-turn` / `multiturn` 0 · `kr-matched` / `matched slice` 0 · `single-source` 0 ·
`format-matched` 0 · `permutation` 0 · `seed` 0 · `orthogonal` 0. `pooled` appears 11 times, all
about mean-pooled activations or the sentence denying the pooled baseline. `cluster` appears twice,
both about the honest persona cluster in Figure 2.

They do not all matter equally. Sorted by whether the draft's argument needs them:

| Control | Does the draft need it? | What the owner's data says |
|---|---|---|
| **Matched dataset-PCA at matched k** | **YES — claim-changing** | The replicated paper runs it (*"Dataset PCA (k=1)"*, verified in its PDF), and `27_crossgen.py:105` calls it *"The paper's dataset-PCA control"*. On the owner's grid the unified persona axis scores **−0.003** against it at L40 and is negative at **18 of 21** layers (**F-17**, HOLDS); `pc:kd_assertion` clears it at **0 of 126 tests** (F-112, ledger, not in `claims.csv` → **DOC-ONLY**). The draft's only control is random directions. |
| **Dataset-level bootstrap** | **YES — claim-changing** | His bootstrap resamples *rows*; with ~15,000 rows those intervals are tight by construction. Resampling *datasets* is what kills most of this project's own headlines (F-04: **1 of 384** tests clears BH; F-31: the kd headline +0.076 has CI [−0.006, +0.173], q = 0.499). The draft's transfer claims have no interval at the unit that actually varies. |
| **Response-length control** | **YES — cheap and damaging if unrun** | A length-only rule reaches **0.5555** off-diagonal on eleven datasets and the positive class is longer in **9 of 11** (**F-73**, PARTLY HOLDS). Nothing in the draft rules out length as part of a 1-d contrast's transfer. |
| **Leave-one-cluster-out** | Useful | Shared provenance crossing dataset boundaries; pooled training survives it (**F-108**, **O-003**). |
| **Layer pre-registration** | Useful | The draft picks L40 post hoc from three layers; A-7 shows the choice is worth ~0.02–0.03. |
| **Seed stability** | Minor | His scripts use one fixed seed; the owner's five-seed check moved the pooled margin by 0.025 (F-83). |
| Teacher forcing · bag-of-words gate · format-matched control · factor probes · capitulation and sarcasm banks | **No — out of scope** | These belong to the `kd/`, `cap/` and sarcasm lines. They are the *thesis's* methods chapter, not this paper's. Do not pad the draft with them. |

**Evidence.** My own greps (above); `audit/METHODS_CHAPTER.md` §5, §8; `audit/methods/ATTRIBUTION.md`
§2, red flag A/F; `audit/claims.csv` F-04, F-17, F-31, F-73, F-108, O-003.

---

### A-7 · Layer 40 is not the peak **[COMPATIBLE, and the draft's own guess is right]** — preview topic 2

**What the draft says (p.10).** *"Of the three layers we extracted, layer 40 is best… Three layers do
not locate an optimum, and a layer between 20 and 40 may do better."*

**What the owner measured.** Exactly that. Off matrices already on disk, the off-diagonal transfer
mean peaks at **L32** for both raw and persona features, by about **0.02** (**F-91**, PARTLY HOLDS —
`cap_s23` raw 0.6818 at L32 vs 0.6623 at L40; `pc:capitulation@2` 0.7798 vs 0.7606). On the
**six-dataset** premise run, L32 beats L40 on all three arms (single-source 0.7203, pooled LODO
0.8965, row-matched 0.9010: +0.024 / +0.028 / +0.008) (**F-109**, HOLDS).

**Where her own claim weakens.** F-91's second method — the pooled mean-of-directions sweep — is
**seed-dependent**: its peak is L32 at seed 42, L40 at seed 7, L36 at seed 13, and averaged over the
three seeds **L40 (0.8872) beats L32 (0.8857)**. So "two independent methods put the optimum at
32–36" does not survive its own second method. The crossgen peak table does hold, and **no ordering
reverses** (+0.098 at L32 vs +0.099 at L40).

**Why it matters.** This is an **addition**, not a correction: the draft's hedge is right and can be
replaced with a measurement. It also raises the bar — if L32 is better for *both* arms, the
comparison should be re-read there, and A-2's margin shrinks further (+0.023 → +0.015).

**Evidence.** `audit/claims.csv` F-91, F-109; `audit/phase2/layer_sensitivity.csv`;
`docs/LAYER_PREREGISTRATION.md`.

---

### A-8 · The 3B and 8B rungs exist **[GAP for the rungs; COMPATIBLE for the conclusion]** — preview topic 4

**What the draft says.** *"We study one model"* (p.12); scaling is framed as the original paper's
3B/8B → this work's 70B.

**What the owner ran.** Both small rungs, in the same pipeline, on the paper's own five datasets, at
k = 3 and matched relative depth. Layer-averaged, the persona advantage **over the matched
dataset-PCA control** declines with scale — **+0.1197 (3B) / +0.1027 (8B) / +0.0541 (70B)** — and at
70B the advantage **over raw** turns negative, **−0.0131**, once the bank mix-up is repaired
(**F-110**, HOLDS; **F-100**, PARTLY HOLDS; **F-63**, PARTLY HOLDS).

**Three qualifications, all VERIFIED, that keep this honest.**
1. F-63's steeper single-layer reading (+0.202 at 3B) is the **maximum of that rung's seven stored
   layers**; layer-averaged it is +0.120 (**F-96**, HOLDS).
2. The decline is **mean-pooling only**. Under last-token pooling the three rungs read
   +0.0745 / +0.0090 / +0.0688 — non-monotone (**F-102**, **F-110** note 2).
3. **These are different objects from the draft's.** Her rungs score *trained* persona features
   (`axes`, `pc:deception`) at k = 3; the draft's headline is a *zero-shot 1-d contrast*. On his six
   datasets the paper-faithful zero-shot contrast does beat his baseline by **+0.100** at L40
   (F-97). So this is **COMPATIBLE**, not a contradiction.

**Why it matters.** The paper is a replication. A three-rung comparison in one pipeline is the single
most valuable thing it could add, and it is already computed. It also gives the paper a defensible
answer to "does it scale?": for the *trained* persona-coordinate method, the gain over a matched
control shrinks by more than half from 3B to 70B.

**Evidence.** `audit/claims.csv` F-63, F-96, F-100, F-101, F-102, F-110; `audit/fix/RF16_NOTE.md`;
`audit/phase2/layer_sensitivity.csv`.

---

### A-9 · Attribution of the headline **[COMPATIBLE on substance; live on credit]** — preview topic 5

**What the files say.** 0.904 / 0.916 are `ext_axis1` — the 12-vs-13 contrast — read verbatim as
`0.9041173922099752` and `0.9158352734669751` from his own `summary.csv` (**O-006**, HOLDS). The
paper-faithful 8-vs-7 contrast reads **0.7909 / 0.8319** on the same rows, so the ten extra roles are
worth **+0.113 / +0.084**, i.e. **44.2% / 45.7%** of the margin over his baseline (**F-97**, HOLDS).
Rebuilt on the owner's eleven datasets the same decomposition gives **74% / 49%** (**F-98**), and
across 21 layers the extended contrast beats the paper's at **21 of 21**, mean **+0.055**
(**F-99**, **O-011**).

**Do not quote the share.** It ranges **2%–114%** mean-pooled (median 63%) and **−156% to +145%**
last-token (median 50%), while the underlying difference barely moves (+0.055 vs +0.059)
(**O-012**, HOLDS). Quote the difference.

**Verdict.** The draft does describe the extension and does name the three groups, so this is
**COMPATIBLE** — the science is stated. Two things are not:
- The **Acknowledgements** say *"[Collaborator name] built the persona bank and the rollout and
  judging pipeline."* On the evidence, the ten new roles, their prompts, the ten new questions, the
  rollouts, the judging and the vectors are the **owner's** (`5b43a7a`, 2026-09-19; his 36 `.pt`
  files are byte-identical to her `run-kd/vectors_s3`) — so that line, as written, may be pointing
  the wrong way, and both authors should agree what it says.
- His `KNOWLEDGE_CANDOUR_PERSONAS` **merges** her deliberately separate `sincere_ignorance` (3) and
  `plain_assertion` (3) into one six-role pole. She split them precisely to remove the wording
  confound; merging reintroduces it — and it is the same confound §6 of the draft concedes. **This is
  a methods discrepancy, not a credit dispute**, and the paper has to say which grouping each number
  used.

**Evidence.** `audit/claims.csv` F-97, F-98, F-99, O-006, O-011, O-012; `audit/methods/ATTRIBUTION.md`
§1.2, red flag B; `docs/DRAFT_CORRECTIONS.md` §2; draft p.13.

---

### A-10 · Register **[COMPATIBLE — the draft is right and badly under-evidenced]** — preview topic 6

**What the draft says (Appendix F, p.19).** *"Its signal on the instructed datasets suggests that part
of what any contrast of this kind reads is the shared register of role-played misbehaviour rather
than deception itself."* One sentence, no follow-up.

**What the owner has.** The same diagnosis, measured and then tested constructively:
- **F-13** (HOLDS, and the audit's layer sweep makes it *stronger* than the entry claims): on the two
  datasets whose labels are not register, the paper16 persona stack is far below chance — HP-KR 0.427
  as target / 0.300 as source, `sycophancy_mt` 0.355 / 0.379 — and a **random** 5-d projection beats
  it on both. Swept over 21 layers × 2 poolings, the persona stack is ≤ 0.50 on `sycophancy_mt` at
  **21 of 21** layers in both poolings.
- **F-65 / F-66** (PARTLY HOLDS / HOLDS): a bank built to vary **capitulation** instead of register
  reads **0.745** zero-shot on `sycophancy_mt` where the register axis reads 0.374, and trained it
  beats raw by **+0.112, CI [+0.050, +0.185]**, winning on 9 of 10 sources and clearing both matched
  controls. The audit's own layer sweep finds it clears at **14 of 21** layers (mean pooling) — but
  at only 3 under last-token pooling, so **it is a mean-pooling result**.
- **F-67** (PARTLY HOLDS): and it is **local** — over the full 11×11 grid the margin over raw is
  +0.098 with CI [−0.036, +0.208], which does not clear.
- **F-89** (HOLDS): a format-matched control deflates the interpretation — swapping the six holding
  personas for topically unrelated ones costs only +0.025 (interval includes zero), while removing
  them entirely costs +0.120.
- **M-036** corrects the headline pair: 0.374 is the paper16 **deception** axis's PC1; the paper16
  **sycophancy** axis reads **0.313**. Like-for-like is 0.745 vs 0.409, or 0.736 vs 0.374.
- The sharpest planned test, the **sarcasm** bank (speech that is literally false without intent to
  deceive), **failed two pre-registered gates** — bag-of-words separates the pilot at 0.9213, and on
  natural sarcasm the deception axis sits inside a 2,000-direction random null in all four cells
  (0.5427 / 0.5856 / 0.5239 / 0.5766) while the *sycophancy* axis fires at 0.693–0.710
  (**F-93**, **F-106**, **F-107**, all HOLDS).

**And one thing sits inside his own files.** At L44 his **sycophancy PC1** (0.8586) **beats his
deception contrast** (0.8260) on the same rows — computed here from his `summary.csv`. That is a
sharper version of his own Appendix F sentence than the sentence he wrote.

**Why it matters.** This is an **addition**, but a cheap one: the draft's register remark is currently
an unsupported aside. Two of its own numbers (syc_pc1 at L44; the |cos| = 0.54 between his deception
and sycophancy axes, recorded in his own code comment and **absent from the draft**) would support it
without importing anything from `cap/`.

**Evidence.** `audit/claims.csv` F-13, F-65, F-66, F-67, F-89, F-93, F-106, F-107, M-036, O-001;
`audit/scripts/phase3/p3a_pc1_vs_axis1.py`;
`origin/feat/persona-pc-ood:.../deception_detection/scripts/persona_ood.py:943,990` and
`.../persona_pcs.py:337` (the `|cos| = 0.54` comments); draft Appendix F.

---

### A-11 · The comparison with the original paper **[GAP · VERIFIED]** — preview topic 7

**F-39** (PARTLY HOLDS) reads the replicated paper directly and finds three things the draft could
use: the paper **does** run a matched control (its axis beats dataset PCA by +0.13…+0.23 in its own
bar chart); this project's 70B result is **k-dependent** (+0.138 over the matched control at k = 2,
+0.004 at k = 5); and the bigger, stricter pipeline gets a **smaller** gain over raw (+0.026 against
their +0.10) because the raw baseline here is 0.809 rather than ~0.70.

Three documentary defects inside F-39 itself are recorded in `audit/claims.csv` and should not be
copied: the bar chart is **Figure 10 panel A**, not Figure 9; "the paper's own five deception
datasets" is its **figure** set (Mask in, HP-C out), not its Table 1 five; and "5+5+anchor" is the
paper's **sycophancy** axis (its deception axis is 8+7+anchor = 16).

**Why it matters.** A replication that reports a *smaller* effect than the original, and says why
(stronger baseline, stricter pipeline, different k), is a stronger paper than one that reports only
its own numbers. Addition.

---

### A-12 · `paper/make_numbers.py` **[NOT-FOUND]** — preview topic 8

Already delivered as `docs/DRAFT_CORRECTIONS.md` 1.14 and confirmed as claim **D-237** (DOES NOT
MATCH HIS DATA). It exists on neither branch, and `feat/persona-pc-ood` has no `paper/` directory.
**Cross-reference only — not repeated here.**

---

### A-13 · Smaller additions from her checked claims, worth one line each

- **The draft's baseline choice is contestable and the alternatives are measured.**
  **[CORRECTED 2026-09-23 — the first version of this bullet mixed two corpora. See below.]**
  On **her** 110 single-source pairs, **21 fall below 0.5** and 14 below 0.4 (minimum 0.055); the
  mean over sources is 0.676, the median 0.722, the sign-oriented mean max(a, 1−a) 0.735, the oracle
  best source 0.897 (**O-022**, **F-90**, HOLDS).
  **Those are her numbers and they must not be set against his 0.732**, which comes from his own
  6×6 matrix — the error batch W6C found in the briefing's closing sentence (**O-036**) and which
  this bullet originally repeated.
  **Applying the same estimators to HIS OWN `persona_ood_L40/matrix_baseline.csv`** (30 off-diagonal
  cells, re-derived by the lead from his branch via `git show`):

  | estimator | his 6×6 at L40 |
  |---|---|
  | mean of all cells | **0.7320** — exactly the draft's 0.732 |
  | median over sources (median of the 6 per-source means) | **0.7500** |
  | median of all 30 cells | 0.7887 |
  | sign-oriented mean, max(a, 1−a) | **0.7850** |
  | cells below 0.5 | 6 of 30, minimum 0.0598 |

  **The point survives and gets stronger on his own data.** On her corpus the median (0.722) sits
  *below* the mean, which does not support "you picked the lowest". On his, every alternative is
  *above* his 0.732 — so "the draft compares against the lowest of these" is true of his matrix and
  not of hers. Quote his numbers, not hers. A referee will ask why the mean; pre-empt it.
  *Note for the record:* batch W6C reported his median as 0.7608; the lead could not reproduce that
  under either reading (0.7500 over sources, 0.7887 over cells). The sign-oriented 0.7850 reproduces
  exactly. Use the lead's figures.
- **Dataset identity dominates the activations.** Dataset identity is **99.3% decodable** from L40
  mean-pooled evaluation activations while the deception label is decodable at only **+0.032** over
  its majority baseline (**O-082**, HOLDS). This is a good argument *for* the draft's thesis — a
  low-dimensional contrast avoids the subspace a raw probe overfits — and it is not in the draft.
- **"Gains where raw is weak" is partly arithmetic.** A **random** 2-d projection gains +0.056 where
  raw is below 0.6 and loses 0.44 where raw is above 0.85, corr(raw, Δ) = **−0.76** — reproduced at
  all 21 layers (**F-17**, **M-018**, **M-095**, HOLDS), and the −0.76 itself sits inside its own
  coupling null (**F-74**, HOLDS). Any per-dataset "the contrast wins where the probe fails" reading
  needs this caveat.
- **The 1-d "zero generalization gap" is definitional**, and the contrast buys it by being worse
  in-distribution (raw diagonal ahead by +0.053 / +0.068 / +0.079). Already in
  `DRAFT_CORRECTIONS` 3.3; claim **A-05** confirms the numbers. Cross-reference.
- **Fine-tuned models.** The draft leaves Soft-Trigger and Gender-Secret to future work. The owner has
  measured them: sleeper detection reaches 0.96 cross-trigger but the "lie" there **is** the wrong
  answer, and gender-secret sits at chance (memory note `q11-sleeper-detection-is-falsity`; **not in
  `audit/claims.csv`** → **DOC-ONLY**). Worth a sentence, flagged as unchecked.

---

## B. His work her docs miss or restate wrongly

Three corrections in his favour are already recorded — the resumable cache/batcher (`e391b3e`,
2026-09-04, 11 days before her port), the pooled LODO baseline (`6b16a46`, one day before
`30_premise.py`, which makes F-85's "the comparison has never been made" false as written), and the
`persona_ood_mod_*` rollout-count control. See `audit/methods/ATTRIBUTION.md` §4.3, §3.1, §5.2 and
`kd/docs/FINDINGS.md` F-113 §1. **Cross-reference; not repeated.** What follows is the rest.

### B-1 · "No measurable difference" vs his measured loss on the same bank swap **[CONFLICT · VERIFIED]**

**Her ledger.** **F-62** (HOLDS): building the kd bank from score-3-only rollouts (`s3`) or from
score-2+3 rollouts (`s23`) *"makes no measurable difference to downstream cross-dataset transfer"* —
largest gap **0.005** at L40, mean pooling, paper probe, 110 off-diagonal pairs. The audit adds that
this is her **best case**: at other layers and poolings the gap reaches **0.030**.

**His files.** `persona_ood_mod_L{20,40,44}` — *"More rollouts per persona did not help"* — is the
**same bank swap** (the audit verified on the lab PC that his "old" counts match `run-kd/vectors_s3`
and his "new" counts match `run-kd/vectors_s23`, 6 of 6 quoted values exact). He measures a
**consistent loss**, with a mechanism:

| arm, off-diagonal AUROC | L20 | L40 | L44 |
|---|---|---|---|
| `ext_axis1` | 0.9041 → 0.8839 | 0.9158 → **0.9066** | 0.9034 → 0.8905 |
| `axis1` | 0.6511 → 0.6505 | 0.8319 → 0.8216 | 0.8260 → 0.8178 |
| HP-KR zero-shot (extended) | 0.882 → **0.805** (±0.042) | 0.912 → **0.876** (±0.031) | 0.894 → **0.849** (±0.037) |

His explanation: the **group means converge** — the axis norms fall to 0.84–0.89 of their old values
while the mean persona norm holds at 0.98, so the rollouts added beyond ~80 per role are less
prototypical and dilute the contrast.

**Are they contradictory?** Strictly, **no** — she measures `pc:` features on eleven datasets, he
measures zero-shot contrasts on six. **COMPATIBLE as measurements, CONFLICT as headlines.** F-62's
sentence is used in `audit/LIMITATIONS_AND_FUTURE_WORK.md` §6 to bound the scale-rung filter
asymmetry ("largest s3-vs-s23 gap 0.005"); his data says that bound does not hold for contrast arms.

**And a consequence neither of them has written down [INFERRED, from two VERIFIED facts].** Her
replication of his two arms — F-98, F-99, O-011, O-012 and briefing §5a, the "+0.055, 21 of 21
layers" result — is built on `run-kd/axes_collab_s23.pt`, i.e. `vectors_s23`: **the bank his own
control found worse.** On his six datasets that costs `ext_axis1` about 0.009–0.020. It does not
change any sign, and her margin is a *difference* between two arms over the same vectors, so most of
it should cancel — but the choice is undocumented on both sides and should be stated.

**Why it matters.** It is the clearest case where the two of them measured the same manipulation and
wrote down opposite headlines. One sentence in each document fixes it.

**Evidence.** `audit/claims.csv` F-62, F-19, F-10, F-98, F-99, O-011;
`origin/feat/persona-pc-ood:.../results/persona_ood_mod_L40/summary.md` (`94db651`, 2026-09-21);
`audit/methods/ATTRIBUTION.md` §5.2; `audit/LIMITATIONS_AND_FUTURE_WORK.md` §6.

---

### B-2 · Her briefing argues with a draft he has replaced **[CONFLICT · VERIFIED]**

`docs/COLLABORATOR_BRIEFING.md` §5c and `kd/docs/FINDINGS.md` F-88 both say *"The draft title is
'One Direction Generalizes'"*. The committed PDF is titled **"Lying in Knowledge-Denier Persona
Coordinates: Cross-Dataset Lie Detection at 70B"**, and "one direction" survives only as a
contribution heading and the §5.3 title. The "One Direction Generalizes" text is the **2026-09-21
abstract** (`audit/phase0/collab_abstract_2026-09-21.md`), a separate, earlier document; Phase 0
already logged this as a CONFLICT.

**The practical consequence.** Several of the briefing's §5 corrections **he has already fixed**, and
saying so is owed to him:

| Briefing / `DRAFT_CORRECTIONS` item | Status in the committed draft |
|---|---|
| "0.09 below the contrast" cannot be sourced | **Fixed** — §5.3 now says "from 0.220 to 0.050 to 0.023" |
| "the 0.18–0.21 lift" is one inverted training row | **Removed** — not in the draft |
| "six Liars' Bench datasets" is five | **Fixed** — the draft says five + RepE throughout |
| "the *independently constructed* sycophancy axis" | **Fixed** — Appendix F now says it comes from the same bank |
| "layers 20 and 40" omits layer 44 | **Fixed** — L44 is reported everywhere |
| **still live:** "the PCs do not contribute" (bank not named) | live — and see **A-1** |
| **still live:** "every method has recall near zero on HP-KR" | live (p.1, p.3) |
| **still live:** "inverts on one dataset at each depth" | live (p.10) |
| **still live:** "the top-10% read is best" | live (p.13, Appendix D) |
| **still live:** "sits at its 0.5 floor in 18 of 30" | live (p.12) |
| **still live:** `paper/make_numbers.py` | live (p.19) |

Two items in `DRAFT_CORRECTIONS` (1.10 "within 0.03 of the upper bound", 1.11 "the pooled baseline
inverts", 1.12, 1.13) are about **his `loo_L20/summary.md`**, not about the draft; the draft does not
contain those sentences. Keeping the two lists apart avoids sending him corrections to a paragraph
that no longer exists.

**Evidence.** `audit/phase0/draft_text.txt` (title, lines 3–4); `audit/phase0/collab_abstract_2026-09-21.md`;
`audit/phase0/draft_report.md` line 62; `docs/COLLABORATOR_BRIEFING.md:422,634`;
`kd/docs/FINDINGS.md:3577`; `docs/DRAFT_CORRECTIONS.md`.

---

### B-3 · His random control is five draws, not one — and the audit said otherwise **[CONFLICT · VERIFIED]**

`audit/claims.csv` row **D-107** records, as a flag for the lead: *"his store keeps a single random
draw (persona_ood.py builds it from one QR of a standard-normal matrix). No seed column, no repeats,
no spread."* **That is wrong.** Read during this audit from
`origin/feat/persona-pc-ood:.../deception_detection/scripts/persona_ood.py`:

```
  61: N_RANDOM_DRAWS = 5
1004: # random control, averaged over draws so a single lucky basis cannot carry the comparison
1006: for _ in range(N_RANDOM_DRAWS):
1007:     dirs = random_directions(hidden, 3, rng)
1009: matrices["random3"] = sum(random_mats) / len(random_mats)
```

`docs/DRAFT_CORRECTIONS.md` §4 already caught this and states it plainly ("we had written down the
opposite, and we were wrong"); the `claims.csv` row was never amended. **Action for the lead: fix
D-107's robustness note.**

**The fair criticism that survives.** A **mean of five** draws bounds the centre of the null, not its
tail; a 95th percentile of a larger null is the stronger object — and the owner already has one
(1,000 directions per dataset, **S-14**, A-4 above). His `random3` is also a *matched-rank* control in
the transfer matrix, which is a different and legitimate question from a zero-shot null.

**Evidence.** `origin/feat/persona-pc-ood:.../persona_ood.py:61,475–478,1004–1009`;
`audit/claims.csv` D-107; `docs/DRAFT_CORRECTIONS.md` §4.

---

### B-4 · His cross-behaviour orthogonalisation control is in neither document **[GAP, against her docs and the draft · VERIFIED]**

`persona_ood.py:944–950` builds two arms nobody else in this project has:

```
syc_orth = orthogonalize(syc_geom.components, dec_axis)   # sycophancy PCs, deception axis removed
dec_orth = orthogonalize(geom.components, syc_axis)       # deception PCs, sycophancy axis removed
```

with his own reason in the docstring: *"the deception and sycophancy axes sit at |cos| = 0.54, so a
raw comparison between them measures mostly their overlap."*

Measured (his `summary.csv`, computed here by `audit/scripts/phase3/p3a_pc1_vs_axis1.py`):

| | L20 | L40 | L44 |
|---|---:|---:|---:|
| `dec3_orth` − `pc3` | −0.0243 | −0.0379 | −0.0870 |
| `syc3_orth` − `syc3` | −0.1195 | −0.0476 | −0.1047 |

**Why it matters.** This is a direct, quantitative measurement of *the shared register component* —
the thing the draft's Appendix F asserts in one sentence and the thing the owner's whole `cap/` line
was built to test. Searching her documents (`kd/docs/FINDINGS.md`, `docs/COLLABORATOR_BRIEFING.md`,
`docs/thesis/`, `cap/docs/`) for `syc3_orth`, `dec3_orth`, `orthogonalis`, `orthogonaliz` returns
**one** hit, and it is her own factor-probe orthogonalisation (F-06) — a different thing. The draft
contains the word `orthogonal` **zero** times. Both authors have been working on the register
question from opposite ends without either noticing that he had already put a number on it.

**Evidence.** `origin/feat/persona-pc-ood:.../persona_ood.py:944–950,991–992,1021`;
`.../persona_pcs.py:333–338`; his `persona_ood_ext_L{20,40,44}/summary.csv`;
`audit/scripts/phase3/p3a_pc1_vs_axis1.py`.

---

### B-5 · He reached her pooled-baseline conclusion first, with a per-dataset breakdown she does not have **[GAP in her docs · VERIFIED]**

`F-113` §1 now credits him with the **baseline**. What it does not record is that he also reached the
**conclusion** — and went further than she did in one respect.

His `loo_L20/summary.md` (`6b16a46`, 2026-09-20 20:05), under *"Three honest qualifications"*:

> **1. Excluding HP-KR, pooled supervision matches or beats it.** L20: 0.928 against 0.909, a loss.
> L40: 0.915 against 0.917, a dead heat. **The persona axis's average advantage is carried almost
> entirely by HP-KR.**
> **2. HP-KR is the dataset the extended personas were authored to fix.** This is circular and has to
> be stated as such.
> **3. The pooled baseline is unstable to a choice with no right answer.** At L20 balanced beats naive
> by 0.10; at L40 naive beats balanced by 0.05. Same code, same data, opposite verdict.

That is the same finding as her **F-98**'s decomposition ("at L20 the entire kd contribution is two
datasets… remove them and the kd personas are worth +0.003"), written one to two days earlier.

**What he has and she does not:** a **per-dataset paired bootstrap** of pooled-vs-persona, 1,000
stratified draws, 11 of 12 intervals excluding zero, showing that **pooled supervision beats the
persona axis on CG at both depths and on HP-C at layer 20** — *"it is not a clean sweep in either
direction."* Nothing in `kd/docs/FINDINGS.md` breaks the comparison down per dataset that way.

**And the raw data is his.** `loo_L20/loo_scores.npz` and `loo_L40/loo_scores.npz` are **the only
per-example scores committed anywhere on either branch** (66 arrays each, plus labels), and batch P10
recomputed all 10 arm × weighting cells at both depths with an independent rank-based Mann–Whitney:
**every row of `loo_summary.csv` reproduces to 4 dp**.

**Why it matters.** The methods chapter currently reads as though the self-criticism is hers. Half of
it is his, and his version is reproducible from committed data, which hers (lab-only JSONs) is not.

**Evidence.** `origin/feat/persona-pc-ood:.../results/loo_L20/summary.md`; `.../loo_scores.npz`;
`audit/methods/ATTRIBUTION.md` §5.1; `audit/claims.csv` F-98, H-008, H-009, H-010;
`audit/PROGRESS.md` wave 4 (batch P10).

---

### B-6 · Two small misstatements about his files in her briefing **[NOT-FOUND / CONFLICT · VERIFIED]**

Both from `docs/COLLABORATOR_BRIEFING.md` §5b, the section she wrote to withdraw an earlier error:

1. *"`scripts/pool_activations.py:84` is `pooled.append(acts[row_mask].float().mean(0)…)`"* — in his
   file at both commits the pooling line is **123**; line 84 is `if acts is None:`. The path is
   `deception_detection/scripts/pool_activations.py`. **The substantive point — that he mean-pools —
   is correct** (F-97, HOLDS).
2. *"every one of your summaries records Δ = 0.0000 reproducing that run's own `detector.pt`"* —
   **NOT-FOUND.** The audit could not find that string in his `summary.md` files. His Appendix-G-style
   reproduction claim is separately checked as **D-233**: the follow-up half reproduces exactly
   (20/20 cells), the three response-token runs committed **only plots, no `results_table.csv`**, so
   "matched for each layer" can be neither confirmed nor refuted.

Also, minor, in the same family: `O-006` notes the briefing cites `extended_arms()` as defining the
axis; `extended_arms()` re-fits the geometry, and the axis is defined in
`deception_detection/persona_pcs.py` (`axes["extended"]`).

**Why it matters.** Small, but they are statements about *his* code in a document written to correct
*her* statements about his code. Fixing them costs a line each.

**Evidence.** `audit/claims.csv` F-97 (notes), O-006 (notes), D-233;
`docs/COLLABORATOR_BRIEFING.md` §5a, §5b.

---

### B-7 · Additions of his that neither document uses

- **`single_source_reference.csv`** (both `loo_*` folders) — a reporting baseline he committed on
  2026-09-20, distinct from her orthogonality analysis (`30d_why_pooling.py`, 09-21).
- **The follow-up-probe runs** (`fd17728`, `d2f4cce`). He re-measured Liars' Bench's own Follow-Up
  Probe and it reproduces the published Table 7 (0.636 vs 0.634 AUROC; **H-001**, **D-234**), and he
  found the persona direction's sign **stops being universal at the follow-up token** — IT falls from
  0.990 to 0.173, HP-KR from 0.882 to 0.190 (**D-197**). The owner's documents contain no follow-up
  work at all. His three published comparators (LB Tables 6/7/8) were verified correctly transcribed.
- **The per-token persona-position profile** (`persona_positions_L{20,40,44}`, `a8eb902`). The owner
  reached the same conclusion on 2026-09-16 (`14_pertoken.py`). `ATTRIBUTION` §5.4 records this as
  two independent arrivals and neither derivative of the other — **that verdict stands and I found
  nothing to move it.**
- **The bf16-ULP measurement** in his `tests/test_acts_cache.py` (2026-09-04) predates her bf16
  reduction rule (2026-09-15) and is consistent with it; `ATTRIBUTION` §4.4 already splits these into
  two separate findings, one each. Cross-reference.

---

## C. Genuinely unclear — do not resolve these from the files

- **C-1. Who first quantified HP-KR's denial-wording confound.** Her `kd/docs/DESIGN.md` at `5b43a7a`
  (**2026-09-19 02:58**) already contains the full 2×2 table with the empty fourth cell and the
  measured denial-regex AUROC **0.870**. His `persona_ood_ext_L20/summary.md` at `9774075`
  (**2026-09-20 17:53**) states *"its deceptive response is a denial… 87% of deceptive rows… its
  honest response is an assertion… 87% of honest rows"*. Git puts her 1.6 days earlier. Git cannot see
  a conversation. **What would settle it:** his own session logs or notes for 09-19/09-20. **UNCLEAR.**
- **C-2. Whether the owner's rebuild of his arms on `vectors_s23` was deliberate.** Only she can say.
  It matters only for the exact size of the +0.055, not its sign (B-1). **UNCLEAR.**
- **C-3. Pooled training (ATTRIBUTION §3.1).** Independent arrival one day apart is what the files
  support; her docstring credits Ying et al. and does not mention his run, which is consistent with
  independence but does not prove it. One answer from either author closes it. **UNCLEAR — carried
  forward unchanged from `ATTRIBUTION.md` §8.2.**

---

## D. What I checked and found sound

So the lists above are read in proportion.

- **Almost every number the draft states about its own results matches his committed data.** Of the
  **79** draft / abstract / his-notes claims already carrying a verdict in `audit/claims.csv`
  (`D-*`, `A-*`, `H-*`): **51 MATCHES HIS DATA · 18 PARTLY · 9 HOLDS · 1 DOES NOT MATCH** (that one
  is `paper/make_numbers.py`, A-12). Where they are "PARTLY", the failure is almost always a
  **quantifier around a correct number** — "one dataset" that is two, "is best" that is
  second-best, "at its floor" that is within 0.005 of it — not the number. That is a good draft.
- **The HP-KR story is right.** The published contrast really does invert (0.148 at L20, **D-003**);
  the base probe inverts too (0.109, **D-081**); the benchmark's own reported values are correctly
  quoted from the Liars' Bench PDF (**D-023**); the repair really does come from the new personas and
  not from re-extraction (**D-085**); and the extended contrast really does reach 0.916 off-diagonal
  against 0.732 (**D-008**, **D-098**).
- **The reproduction argument holds.** His re-measured Follow-Up Probe (0.636 / 0.622 vs published
  0.634 / 0.578) and Mean Probe (0.697 / 0.721 vs 0.682 / 0.636) both check out against the PDF, and
  the insider-trading scratchpad caveat is correctly stated (**D-234**, **D-235**, **D-236**, **D-064**).
- **The draft's own self-criticism is accurate.** The L20-excluding-HP-KR reversal (**D-101**), the
  "designed repair" concession, and the calibration limitation are all measured correctly.

---

## E. Evidence index

| Kind | Where |
|---|---|
| The draft | `audit/phase0/draft_text.txt` (844 lines, 19 pp) — grepped and read in full for this workstream |
| Her checked findings | `audit/claims.csv` (231 rows with verdicts) + `audit/claims/<id>.md`; helper `audit/scripts/phase3/p3a_claims_index.py`, `p3a_claim_detail.py` |
| His committed results | `origin/feat/persona-pc-ood` read with `git show` only — no checkout, no fetch: `results/persona_ood_ext_L{20,40,44}/summary.csv`, `results/persona_ood_mod_L40/summary.md`, `results/loo_L20/summary.md`, `deception_detection/scripts/persona_ood.py`, `deception_detection/persona_pcs.py` |
| Computed here | `audit/scripts/phase3/p3a_pc1_vs_axis1.py` (published PC1 vs contrast; sycophancy PC1; orthogonalisation cost) — re-runnable, reads only `git show` output |
| The replicated paper | `papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf` — figure titles read via `pypdf` in this audit for the k = 1 finding |
| Authorship | `audit/methods/ATTRIBUTION.md` (26 designs, first-commit hashes) |
| Already delivered to him | `docs/DRAFT_CORRECTIONS.md` — cross-referenced, never duplicated |
| Not repeated | `audit/METHODS_CHAPTER.md` §5/§8, `audit/LIMITATIONS_AND_FUTURE_WORK.md`, `audit/phase0/draft_vs_ledger_preview.md` |

**Not done here, and owed to a later phase:** F-112 ("`pc:kd_assertion` clears the matched control at
0 of 126 tests") is cited in `METHODS_CHAPTER.md` and `LIMITATIONS_AND_FUTURE_WORK.md` but has **no
row in `audit/claims.csv`** — it is **DOC-ONLY** and is load-bearing for A-6. The per-dataset random
null (A-4, S-14) has **no F-number** in `kd/docs/FINDINGS.md`. Both should be checked before either
is quoted in the paper.
