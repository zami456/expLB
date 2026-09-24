# Digest — the collaborator's draft paper vs. what the audit found

_Phase 5 input for the final report. Written 2026-09-23. **Nothing here is new work.** It compresses
`audit/phase3/DRAFT_GAPS.md` (779 lines) and `audit/phase3/DISAGREEMENTS.md` (463 lines). Item ids
(A-1, B-4, D-03 …) are the ids in those two files; F-/O-/S-/D-/M-/H- ids are rows of
`audit/claims.csv`._

**The object under review.** `full paper draft for this experiments/main (1).pdf` on branch `asix`,
extracted text at `audit/phase0/draft_text.txt` (844 lines, 19 pp). Title: *"Lying in
Knowledge-Denier Persona Coordinates: Cross-Dataset Lie Detection at 70B"*. Written by the
collaborator; his results are on `origin/feat/persona-pc-ood`.

**Terms used below, first use.**
*AUROC* — how well a score ranks lies above honest answers; 0.5 is chance, **below 0.5 means the
direction points the wrong way (inverted)**. *Zero-shot* — score by dot product with one fixed
direction, nothing trained. *Off-diagonal mean* — train on one dataset, test on the others, average
the cells. *Bank* — the file of persona vectors a direction is built from. *s3 / s23* — the owner's
two judge filters on persona rollouts (keep score-3 only, vs keep score-2 and 3). *k* — how many
principal components the persona subspace keeps. *L20/L40/L44* — transformer layer. *Pooling* —
mean over response tokens vs last token only.

**Tags are carried through unchanged.** VERIFIED · DOC-ONLY · NOT-FOUND · CONFLICT · INFERRED ·
NOT RECOMPUTABLE. Nothing has been upgraded.

---

## 1. WHAT THE DRAFT GETS WRONG

Ranked by how much damage it does if a referee finds it before the authors do.

### 1.1 — "We do not compare against a pooled probe." The comparison exists, on his own branch, and it takes most of the headline away. [A-2 · CONTRADICTION · VERIFIED]

**Draft, p.12:** *"A probe trained on several datasets pooled is a stronger supervised baseline, and
we do not compare against it."*

**Data.** He ran it himself on 2026-09-20 (`loo_L20/`, `loo_L40/` at commit `6b16a46`, *"Leave-one-
dataset-out: the fair baseline for the generalization claim"*), one day before the owner's version.
His own note: *"The straw-man charge was partly right… the margin does shrink."*

- On **his six datasets**, layer 40, off-diagonal AUROC: pooled leave-one-dataset-out reaches
  **0.8683**, row-budget-matched **0.8930**. The abstract's advantage moves **+0.184 → +0.048 →
  +0.023** (F-85, PARTLY HOLDS). At **L32** pooled is higher still and the margin falls to **+0.015**
  (F-109, HOLDS).
- On the owner's **eleven datasets** both his arms **lose** to pooled LODO: `ext_axis1` by **0.064**
  (L20) / **0.055** (L40); the paper-faithful `axis1` by **0.146** / **0.122** (F-98, O-009, HOLDS;
  pooled wins at 21 of 21 stored layers).
- Two caveats that must travel with it: on the **six**-dataset subset at L20 the dataset-level
  bootstrap does **not** clear (+0.093, CI [−0.045, +0.210], P = 0.089) — it is n = 11 that makes the
  headline tight (F-85 L2); and under leave-one-**cluster**-out pooled training still wins
  (+0.129 / +0.131 / +0.125), so the gain is not the HP-KR row leak (F-108, O-003, HOLDS).

**Why it is the worst item.** It is a false statement about the authors' own work, and it is exactly
the baseline a referee names first. The defensible claim his own note already writes is
**robustness, not peak accuracy**: the label-free direction never collapses on an unseen dataset
(floor 0.64 / 0.71) while pooled supervision falls to 0.12 / 0.49.

**Evidence.** Draft p.12; `origin/feat/persona-pc-ood:src/probes/deception-detection/results/loo_L20/summary.md`;
`audit/claims.csv` F-79, F-80, F-81, F-83, F-85, F-98, F-109, O-003, O-009; `audit/methods/ATTRIBUTION.md` §3.1–3.2.

---

### 1.2 — The central HP-KR result is substantially the *plain-assertion* pole, not knowledge denial. [D-03 · VERIFIED numbers, INFERRED interpretation]

His knowledge-denial axis contrasts **4 denial roles vs 6 merged candour roles**. The owner's design
deliberately splits that six-role pole into two three-role poles, because one of them (plain
assertion) differs from denial in *wording*, not in sincerity. Measured on **one bank** (s23) so only
the contrast definition differs, zero-shot AUROC on HP-KR, mean pooling:

| contrast | L20 | L40 | L44 |
|---|---:|---:|---:|
| his `denial_axis1` — 4 denial vs **6 merged** | 0.8734 | 0.8826 | 0.8734 |
| 4 denial vs **3 sincere-ignorance** (wording-matched) | **0.5923** | **0.6692** | **0.6659** |
| 4 denial vs **3 plain-assertion** | 0.9118 | 0.9344 | 0.9266 |

**0.28 / 0.21 / 0.21 AUROC of his HP-KR result is the plain-assertion half.** Against roles that deny
knowledge *sincerely* the axis falls to 0.59–0.67.

Worse for the draft: on the owner's **wording-matched** HP-KR slice (`harm-pressure-kr-matched`, 258
rows, both classes deny), his 12v13 extended axis reads **0.4630 (L20) / 0.6203 (L40) / 0.5383
(L44)** and the split knowledge-denial axis reads **0.3932 / 0.4989 / 0.4825** — at or below chance.
Zero-shot, so the HP-KR↔matched row leak (F-108) does not apply here.

**Fair to him:** his own `persona_ood_ext_L20/summary.md` flags the risk (*"Group assignment is by
name … a different split would give a different axis"*). He never measured it. **Neither document
states this number.** It is the sharpest item in the whole comparison.

**Evidence.** `kd/personas/kd_role_groups_collab.json` (`contrasts`); `origin/feat/persona-pc-ood:.../persona_pcs.py:134-140`;
her `run-crossgen/kd11/banks/collab/matrices/mean_L{20,40,44}_s42.json`; his `results/persona_ood_mod_L*/zero_shot.csv`.

---

### 1.3 — "We have not run the wording-controlled test." It was run, by the co-author, and it is inconclusive. [A-3 · CONTRADICTION · VERIFIED]

**Draft, p.12 and again p.13 as future work:** *"The bank was designed for the cleaner test… and we
have not run it."*

It was specified on 2026-09-19 (`kd/docs/DESIGN.md` at `5b43a7a`) and run. Results:

| test | result | claim |
|---|---|---|
| denial regex alone on HP-KR (Llama rows) | AUROC **0.913** — the dataset is mostly one lexical bit | F-01 HOLDS |
| matched slice, trained persona axes vs raw, **L40** | 0.752 vs 0.715 | F-03 HOLDS |
| **same cell at L20** | **0.374** — 0.216 *below* raw; span over depth 0.365 | F-72 HOLDS |
| **dataset-level bootstrap at the pre-registered L40** | **+0.066, CI [−0.069, +0.189], P = 0.161 — does not clear** | F-76, O-004 HOLD |
| which layer is the peak | 95% argmax interval **[L5, L79]** — 93% of the network | F-76 |
| label permutation | p = 0.38 | O-005 HOLDS |
| all 80 layers | persona axes beat raw at **28 of 80**, dataset-PCA at **31 of 80** | F-72 L2 |
| leakage | `kr-matched` row-ids are HP-KR row-ids **by construction**; ~81% of its test rows sit in HP-KR's train split | F-108 HOLDS |

The two teacher-forced factor probes did pass their pre-registered gate (`d_utterance` 0.996–0.999,
`d_knowledge` 0.685–0.772, F-05 HOLDS), and orthogonalising wording against knowledge drops it to
~0.5 at L40/L44 as predicted — **but not at L20, where it reads 0.33** (F-06 HOLDS).

**The correction is not just "it was run".** It must not be promised as the fix. The honest sentence
is measured: *"we ran a wording-matched slice; at the pre-registered layer the margin is not
distinguishable from zero at n = 10 datasets, and the effect is confined to layers 40–44."*

**Evidence.** Draft p.12–13; `kd/docs/DESIGN.md` §1–§4 at `5b43a7a`; `audit/claims.csv` F-01, F-03, F-05, F-06, F-72, F-76, F-108, O-004, O-005.

---

### 1.4 — "The persona principal components add nothing" is false at the replicated paper's own setting. [A-1 · CONTRADICTION · VERIFIED]

The replicated paper's **deception** figures use **k = 1**; its **sycophancy** figures use k = 3
(confirmed by an independent `pypdfium2` extraction of all 14 pages: `k=1` appears 12 times, `k=3`
9 times, **`k=5` zero times**). The draft reports *Published PCs (k = 3)* and *Extended PC1*, and
Table 3 has **no Published PC1 row**.

His own `summary.csv` has that row. Off-diagonal mean, his six datasets:

| | L20 | L40 | L44 |
|---|---:|---:|---:|
| `axis1` — published contrast (8 v 7) | 0.6511 | 0.8319 | 0.8260 |
| **`pc1` — published PC1, the paper's own k = 1** | **0.6669** | **0.8593** | **0.8506** |
| difference (PC1 − contrast) | **+0.0158** | **+0.0274** | **+0.0246** |

**Caveat that limits the damage:** with HP-KR excluded the gap is −0.0033 / +0.0128 / +0.0144, so at
L20 it reverses. **A defensible fix exists:** compare the PCs against a matched dataset-PCA control at
the same k (§3.2 below), where they lose.

**Also:** the project default `k = 5` used in `pc:deception` is a value the paper never uses anywhere
(confirms AD-1, O-073). And both deception figure titles carry **"[HPC excluded]"** — the paper drops
HarmPressureChoice, the dataset its own Table 1 declares, and substitutes Mask (confirms F-60).

**Evidence.** `papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf` figure titles pp. 12–13;
`origin/feat/persona-pc-ood:.../results/persona_ood_ext_L{20,40,44}/summary.csv`; `audit/scripts/phase3/p3a_pc1_vs_axis1.py`; `docs/DRAFT_CORRECTIONS.md` 1.2.

---

### 1.5 — "All six test sets" contains the base probe's own training set, and dropping it reverses the L20 result. [A-5 · GAP · VERIFIED]

`repe_honesty` is the RepE dataset the deployed probe was trained on — in-distribution for one of the
two arms being compared. Recomputing his own **L20** transfer matrices with that row and column
dropped (20 cells instead of 30):

- paper-faithful contrast **0.5900** vs single-source baseline **0.6085** — i.e. **below the baseline
  it is meant to beat, by 0.019**. With RepE in, the same pair reads 0.6511 vs 0.6478 (+0.003).
- L20-specific: at L40 the same operation gives **+0.092**.

His own `zero_shot_operating.csv` already distinguishes "Average" (6) from "Average (Liars' Bench)"
(5). The draft already has an "Excluding HP-KR" column; an "Excluding RepE" column costs nothing.
**Absent from both the draft and the owner's ledger.** Claim S-11, HOLDS.

**Confidence limit:** VERIFIED at Level 1 only — his activations are gone, so every mean was
recomputed from his per-cell CSVs, not from activations.

**Evidence.** `audit/claims.csv` S-11, A-03; `origin/feat/persona-pc-ood:.../persona_ood_ext_L{20,40}/matrix_{baseline,axis1,pc1,ext_axis1}.csv`.

---

### 1.6 — Four L20 matrices carry a sign-flipped training row, inflating the "lift from the extended personas" at layer 20. [D-09 · VERIFIED]

Scanning all 24 of his 1-dimensional matrices, exactly four are affected, all at **L20**, all on the
HP-KR training row:

| store | matrix | as written | sign-corrected |
|---|---|---:|---:|
| `persona_ood_ext_L20` | `matrix_axis1` | 0.6511 | **0.7909** |
| `persona_ood_ext_L20` | `matrix_pc1` | 0.6669 | **0.8056** |
| `persona_ood_mod_L20` | `matrix_axis1` | 0.6505 | **0.7881** |
| `persona_ood_mod_L20` | `matrix_pc1` | 0.6646 | **0.8002** |

**Blast radius is bounded:** no row is flipped at L40 or L44, and no `ext_axis1` or `denial_axis1`
matrix is affected at any layer. **The 0.916 headline at L40 is untouched.** The owner already
records the caveat (`kd/docs/FINDINGS.md:4519-4524`).

**Evidence.** `audit/scripts/phase3/p3b_signflip.py`; his committed matrices.

---

### 1.7 — `paper/make_numbers.py` does not exist. [A-12 · NOT-FOUND, claim D-237 DOES NOT MATCH HIS DATA]

Draft p.19 cites it. It is on neither branch, and `feat/persona-pc-ood` has no `paper/` directory.
Cheap for a referee to check, embarrassing when they do. Already delivered as
`docs/DRAFT_CORRECTIONS.md` 1.14. **Cross-reference only.**

---

### 1.8 — Quantifier errors still live in the committed draft (already sent to him)

From `docs/DRAFT_CORRECTIONS.md`, still present in the PDF: *"every method has recall near zero on
HP-KR"* (p.1, p.3) · *"inverts on one dataset at each depth"* (p.10) · *"the top-10% read is best"*
(p.13, App. D) · *"sits at its 0.5 floor in 18 of 30"* (p.12) · *"the PCs do not contribute"* (bank
not named — see §1.4). Each is **a wrong quantifier around a correct number**, not a wrong number.
**Do not re-send these; they are already delivered.**

---

## 2. WHAT THE DRAFT OVERSTATES

True, but stated more broadly than the evidence carries.

**2.1 — "Zero-shot transfer" leans on datasets where a random direction does as well.** [A-4 · GAP · VERIFIED]
At L40, mean pooling, against 1,000 random unit directions per dataset (empirical p on
|AUROC − 0.5|), the paper-faithful deception axis beats the random 95th percentile on only **5 of 9
datasets**. It is indistinguishable from random on **convincing-game (0.788, p = 0.238)**,
**harm-pressure-choice (0.544, p = 0.771)**, **HP-KR (0.644, p = 0.625)** and **mask (0.467 —
inverted, p = 0.711)**. The draft's Table 6 reports the published contrast on exactly these datasets.
Composition caveat: the *pooled* 0.881 includes the 2,000-row Alpaca control as pure negatives and
reads **0.863** without it, so pooled and per-dataset figures are not comparable.
Claim S-14, PARTLY HOLDS. **This has no F-number in the owner's ledger either** — neither author has
it written down. Evidence: `run-coords/zeroshot_nulls.json` key `corpus_L40_mean`; `src/assistant_axis_ndif/25_zeroshot_nulls.py`.

**2.2 — The confidence intervals are tight because of the resampling unit, not the effect.** [A-6]
His bootstrap resamples **rows** (~15,000 of them). Resampling **datasets** — the unit that actually
varies — is what kills most of this project's own headlines (F-04: **1 of 384** tests clears
Benjamini–Hochberg; F-31: the kd headline +0.076 has CI [−0.006, +0.173], q = 0.499). The draft's
transfer claims have **no interval at the unit that varies**. Claim-changing if a referee raises it.

**2.3 — The 0.732 baseline is the lowest of the plausible summaries of his own matrix.** [A-13 · VERIFIED, with one internal CONFLICT]
Re-derived by the lead from `persona_ood_L40/matrix_baseline.csv` (30 off-diagonal cells) via `git show`:

| estimator | his 6×6 at L40 |
|---|---:|
| mean of all cells | **0.7320** — exactly the draft's number |
| median over sources (median of 6 per-source means) | **0.7500** |
| median of all 30 cells | 0.7887 |
| sign-oriented mean, max(a, 1−a) | **0.7850** |
| cells below 0.5 | 6 of 30, minimum 0.0598 |

Every alternative is **above** 0.732, so "the draft compares against the lowest" is true of *his*
matrix. It is **not** true of the owner's 110-pair corpus (mean 0.676, median 0.722 — the median sits
*below* the mean), so **her numbers must not be set against his 0.732** (error O-036, corrected).
**CONFLICT, unresolved:** batch W6C reported his median as 0.7608; the lead could not reproduce that
under either reading. **Use the lead's figures.** A referee will ask why the mean; pre-empt it.

**2.4 — "Zero generalization gap" is definitional, and it is bought in-distribution.** A 1-dimensional
feature can only learn a sign, so it cannot overfit a training set. The contrast pays for it by being
worse in-distribution: the raw probe's diagonal is ahead by **+0.053 / +0.068 / +0.079**.
(`DRAFT_CORRECTIONS` 3.3; claim A-05 confirms the numbers. Cross-reference.)

**2.5 — "It gains where the raw probe is weak" is partly arithmetic.** A **random** 2-dimensional
projection gains **+0.056** where raw is below 0.6 and loses **0.44** where raw is above 0.85;
corr(raw, Δ) = **−0.76**, reproduced at all 21 layers (F-17, M-018, M-095, HOLD) — and the −0.76
itself sits inside its own coupling null (F-74, HOLDS). Any per-dataset "the contrast wins where the
probe fails" reading needs this caveat.

**2.6 — The headline direction is partly a purpose-built bank, and the credit line may point the
wrong way.** [A-9 · COMPATIBLE on substance] 0.904 / 0.916 are `ext_axis1`, the 12-vs-13 contrast
(read verbatim from his `summary.csv`). The paper-faithful 8-vs-7 contrast reads **0.7909 / 0.8319**
on the same rows, so the ten extra roles are worth **+0.113 / +0.084** — **44.2% / 45.7%** of the
margin (F-97, HOLDS). On the owner's eleven datasets the same decomposition gives 74% / 49% (F-98);
over 21 layers the extended contrast wins at 21 of 21, mean **+0.055** (F-99, O-011).
**Do not quote the share** — it ranges **2%–114%** mean-pooled (median 63%) and **−156% to +145%**
last-token (median 50%), while the underlying difference barely moves (+0.055 vs +0.059) (O-012,
HOLDS). **Quote the difference.**
Two non-scientific points: the **Acknowledgements** credit the collaborator with *"the persona bank
and the rollout and judging pipeline"*, but on the evidence the ten new roles, prompts, questions,
rollouts, judging and vectors are the **owner's** (`5b43a7a`; his 36 `.pt` files are byte-identical to
her `run-kd/vectors_s3`); and his `KNOWLEDGE_CANDOUR_PERSONAS` **merges** her deliberately separate
`sincere_ignorance` and `plain_assertion` — which reintroduces exactly the confound §6 of the draft
concedes (see §1.2). The second is a **methods discrepancy, not a credit dispute**; the paper must say
which grouping each number used.

---

## 3. WHAT THE DRAFT MISSES

### 3.1 Results already computed that would *strengthen* it

- **The 3B and 8B rungs exist.** [A-8] Same pipeline, paper's own five deception datasets, k = 3,
  matched relative depth, layer-averaged, mean pooling: the persona advantage **over the matched
  dataset-PCA control** declines with scale — **+0.1197 (3B) / +0.1027 (8B) / +0.0541 (70B)** — and at
  70B the advantage **over raw** turns negative, **−0.0131**, once the bank mix-up is repaired (F-110
  HOLDS; F-100, F-63 PARTLY). Three qualifications: (i) F-63's +0.202 at 3B is the **maximum of that
  rung's seven stored layers**; layer-averaged it is +0.120 (F-96). (ii) The decline is
  **mean-pooling only** — last-token gives +0.0745 / +0.0090 / +0.0688, non-monotone (F-102).
  (iii) These score *trained* persona features at k = 3; the draft's headline is a *zero-shot 1-d
  contrast*, so this is **COMPATIBLE, not a contradiction**. A three-rung comparison in one pipeline
  is the single most valuable thing a replication could add, and it is already computed.
- **His own cross-behaviour orthogonalisation control, which neither document uses.** [B-4]
  `persona_ood.py:944–950` builds `syc_orth` (sycophancy PCs, deception axis removed) and `dec_orth`
  (deception PCs, sycophancy axis removed), with his own reason in the docstring: *"the deception and
  sycophancy axes sit at |cos| = 0.54."* Cost of removing the overlap, off-diagonal mean:
  `dec3_orth − pc3` = **−0.0243 / −0.0379 / −0.0870** (L20/L40/L44); `syc3_orth − syc3` = **−0.1195 /
  −0.0476 / −0.1047**. This is a **direct measurement of the shared register component** the draft's
  Appendix F asserts in one sentence. The word `orthogonal` appears **zero** times in the draft, and
  one hit (a different thing) across the owner's documents.
- **His own per-dataset paired bootstrap, and the only per-example scores committed anywhere.** [B-5]
  `loo_L20/summary.md` (`6b16a46`) contains *"Three honest qualifications"*: excluding HP-KR, pooled
  supervision matches or beats the persona axis (L20 0.928 vs 0.909, a loss; L40 0.915 vs 0.917, a
  dead heat) — *"the persona axis's average advantage is carried almost entirely by HP-KR"*; HP-KR is
  the dataset the extended personas were authored to fix, *"circular and has to be stated as such"*;
  and the pooled baseline flips verdict between balanced and naive weighting (L20 balanced +0.10,
  L40 naive +0.05). His 1,000-draw stratified per-dataset bootstrap shows **11 of 12 intervals exclude
  zero**, with pooled beating the persona axis on convincing-game at both depths and HP-C at L20 —
  *"not a clean sweep in either direction."* `loo_scores.npz` (66 arrays each, plus labels) is **the
  only per-example score data committed on either branch**, and every row of `loo_summary.csv`
  reproduces to 4 dp under an independent rank-based Mann–Whitney.
- **The comparison with the original paper.** [A-11 · F-39 PARTLY HOLDS] The paper **does** run a
  matched control (its axis beats dataset PCA by +0.13…+0.23 in its own bar chart); this project's
  70B result is **k-dependent** (+0.138 over the matched control at k = 2, **+0.004 at k = 5**); and
  the bigger, stricter pipeline gets a **smaller** gain over raw (+0.026 against their +0.10) because
  the raw baseline here is 0.809 rather than ~0.70. A replication that reports a smaller effect *and
  says why* is a stronger paper. **Three documentary defects inside F-39 must not be copied:** the bar
  chart is **Figure 10 panel A**, not Figure 9; "the paper's own five deception datasets" is its
  *figure* set (Mask in, HP-C out), not its Table 1 five; and "5+5+anchor" is the paper's
  **sycophancy** axis (its deception axis is 8+7+anchor = 16).
- **Layer 32, not 40, is the measured peak — and the draft already guessed it.** [A-7 · COMPATIBLE]
  Draft p.10: *"a layer between 20 and 40 may do better."* Off matrices already on disk the
  off-diagonal transfer mean peaks at **L32** for both raw and persona features, by about **0.02**
  (F-91 PARTLY: `cap_s23` raw 0.6818 at L32 vs 0.6623 at L40; `pc:capitulation@2` 0.7798 vs 0.7606).
  On the six-dataset premise run, L32 beats L40 on all three arms (+0.024 / +0.028 / +0.008, F-109
  HOLDS). **Where the owner's own claim weakens:** F-91's second method (pooled mean-of-directions
  sweep) is **seed-dependent** — peak at L32 for seed 42, L40 for seed 7, L36 for seed 13, and
  averaged over the three seeds **L40 (0.8872) beats L32 (0.8857)**. No ordering reverses. Raising the
  layer also shrinks §1.1's margin further (+0.023 → +0.015).
- **Dataset identity dominates the activations — an argument *for* the draft's thesis.** Dataset
  identity is **99.3% decodable** from L40 mean-pooled evaluation activations, while the deception
  label is decodable at only **+0.032** over its majority baseline (O-082, HOLDS). A low-dimensional
  contrast avoids the subspace a raw probe overfits. Not in the draft.
- **The pooled-training result is the project's strongest positive finding and needs no personas at
  all:** +0.178 over single-source at L40, CI [+0.143, +0.213], P(≤0) = 0, winning on 11 of 11
  targets (F-80), positive at 21/21 layers (F-81), stable over 5 seeds (F-83); and the gain is
  **diversity, not data volume** (row-budget-matched keeps +0.174).
- **His follow-up-probe runs.** [B-7] He re-measured Liars' Bench's own Follow-Up Probe and it
  reproduces published Table 7 (0.636 vs 0.634 AUROC; H-001, D-234), and he found the persona
  direction's sign **stops being universal at the follow-up token** — instructed-deception falls
  0.990 → 0.173, HP-KR 0.882 → 0.190 (D-197). The owner's documents contain **no follow-up work at
  all**. Also uncited on both sides: his `single_source_reference.csv` (both `loo_*` folders), his
  per-token persona-position profile (independent arrival with her `14_pertoken.py`, ATTRIBUTION §5.4
  — that verdict stands), and his bf16-ULP measurement (2026-09-04, predates her bf16 rule).

### 3.2 Results that *weaken* it, which a referee will find anyway

- **The draft's only control is three random directions.** [A-6] Confirmed by grep over
  `audit/phase0/draft_text.txt`: `datapca` 0 · `dataset PCA` 0 · `length` 0 · `leave-one` 0 ·
  `permutation` 0 · `seed` 0 · `pre-regist`/`preregist` 0 · `orthogonal` 0 · `single-source` 0.
  Three of the missing controls are **claim-changing**:
  1. **Matched dataset-PCA at matched k.** The replicated paper runs it (*"Dataset PCA (k=1)"*,
     verified in its PDF) and his own code calls it *"the paper's dataset-PCA control"*
     (`27_crossgen.py:105`). On the owner's grid the unified persona axis scores **−0.003** against it
     at L40 and is negative at **18 of 21** layers (F-17, HOLDS); `pc:kd_assertion` clears it at
     **0 of 126 tests** (F-112 — **DOC-ONLY, no row in `audit/claims.csv`, and load-bearing here;
     check before quoting**).
  2. **Dataset-level bootstrap** — §2.2.
  3. **Response-length control.** A length-only rule reaches **0.5555** off-diagonal on eleven
     datasets and the positive class is longer in **9 of 11** (F-73, PARTLY HOLDS). Cheap, and
     damaging if unrun.
  Useful but not claim-changing: leave-one-cluster-out, layer pre-registration, seed stability (the
  owner's five-seed check moved the pooled margin by 0.025, F-83).
  **Explicitly out of scope — do not pad the draft with them:** teacher forcing, the bag-of-words
  gate, the format-matched control, the factor probes, and the capitulation and sarcasm banks. Those
  are the thesis's methods chapter, not this paper.
- **The register reading the draft asserts in one sentence is measured, and it cuts both ways.**
  [A-10 · the draft is right and badly under-evidenced] Appendix F p.19 says one sentence about
  *"the shared register of role-played misbehaviour."* The owner has: F-13 (HOLDS) — on the two
  datasets whose labels are not register, the paper16 persona stack is far below chance (HP-KR 0.427
  as target / 0.300 as source; `sycophancy_mt` 0.355 / 0.379) and a random 5-d projection beats it on
  both, at **21 of 21** layers in both poolings for `sycophancy_mt`. F-65/F-66 — a bank varying
  **capitulation** instead of register reads **0.745** zero-shot on `sycophancy_mt` where the register
  axis reads 0.374, and trained beats raw by **+0.112, CI [+0.050, +0.185]**; but it clears at only
  14 of 21 layers under mean pooling and **3 of 21 under last-token**, so **it is a mean-pooling
  result**; and F-67 (PARTLY) it is **local** — over the full 11×11 grid +0.098, CI [−0.036, +0.208],
  does not clear. F-89 — a format-matched control deflates the interpretation (swapping the six
  holding personas for topically unrelated ones costs only +0.025; removing them entirely costs
  +0.120). M-036 corrects the headline pair: 0.374 is the paper16 **deception** axis's PC1; the
  paper16 **sycophancy** axis reads **0.313**, so like-for-like is 0.745 vs 0.409.
  **The cheapest strengthening is inside his own files:** at L44 his **sycophancy PC1 (0.8586) beats
  his deception contrast (0.8260)** on the same rows, and his own code comment records |cos| = 0.54
  between the two axes. Neither is in the draft.
- **The sharpest planned test failed.** The **sarcasm** bank (speech literally false without intent to
  deceive) failed two pre-registered gates: bag-of-words separates the pilot at **0.9213**, and on
  natural sarcasm the deception axis sits inside a 2,000-direction random null in all four cells
  (0.5427 / 0.5856 / 0.5239 / 0.5766) while the *sycophancy* axis fires at 0.693–0.710 (F-93, F-106,
  F-107, all HOLD). Negative result, worth a sentence.
- **Fine-tuned models are left to future work but have been measured.** Sleeper-agent detection reaches
  0.96 cross-trigger, but the "lie" there **is** the wrong answer; gender-secret sits at chance.
  **DOC-ONLY** — memory note `q11-sleeper-detection-is-falsity`, **no row in `audit/claims.csv`**.
  Worth a sentence, flagged as unchecked.

---

## 4. WHERE THE DRAFT IS RIGHT, AND THE AUDIT CONFIRMED IT

This is the larger list. Read §1–§3 in proportion to it.

- **The two pipelines agree.** [DISAGREEMENTS §0] Her `27_crossgen.py` on NDIF activations and his
  `persona_ood.py` on the probe repo's cache give the same answer when they use the same bank on the
  same rows: **55 matched cells, mean |gap| 0.0015, max 0.0175**, across four bank/arm combinations.
  The evaluation rows are provably identical — same n and same positive count on all five shared
  datasets (instructed-deception 5432/3045, HP-C 2134/134, HP-KR 2139/139, convincing-game 267/167,
  insider-trading 1080/423). **No disagreement anywhere in this audit is a bug in
  `persona_ood.py`, `persona_pcs.py`, `pool_activations.py` or `27_crossgen.py`.** VERIFIED, HIGH.
- **Almost every number the draft states about its own results matches his committed data.** Of the
  **79** draft / abstract / his-notes claims carrying a verdict in `audit/claims.csv` (`D-*`, `A-*`,
  `H-*`): **51 MATCHES HIS DATA · 18 PARTLY · 9 HOLDS · 1 DOES NOT MATCH** (the missing
  `make_numbers.py`). Where "PARTLY", the failure is almost always a quantifier around a correct
  number, not the number. **That is a good draft.**
- **The HP-KR story is right.** The published contrast really does invert (0.148 at L20, D-003); the
  base probe inverts too (0.109, D-081); the benchmark's own reported values are correctly quoted from
  the Liars' Bench PDF (D-023); the repair really does come from the new personas and not from
  re-extraction (D-085); and the extended contrast really does reach 0.916 off-diagonal against 0.732
  (D-008, D-098).
- **The reproduction argument holds.** Follow-Up Probe 0.636 / 0.622 vs published 0.634 / 0.578 and
  Mean Probe 0.697 / 0.721 vs 0.682 / 0.636 both check out against the PDF; the insider-trading
  scratchpad caveat is correctly stated (D-234, D-235, D-236, D-064). His three published comparators
  (Liars' Bench Tables 6/7/8) were verified correctly transcribed.
- **The draft's own self-criticism is accurate.** The L20-excluding-HP-KR reversal (D-101), the
  "designed repair" concession, and the calibration limitation are all measured correctly.
- **His baseline is the stronger one, so his headline margin is conservative on the baseline side.**
  [D-04] At L40/L44 his 8192-d single-source baseline beats hers by **+0.0555 / +0.0648**. Say so.
- **His random control is the mean of five draws, not one — and the audit said otherwise.** [B-3 ·
  CONFLICT · VERIFIED] `persona_ood.py:61` has `N_RANDOM_DRAWS = 5`, with a comment: *"averaged over
  draws so a single lucky basis cannot carry the comparison"*. **`audit/claims.csv` row D-107 is
  wrong** and must be amended (`DRAFT_CORRECTIONS` §4 already states this plainly). The fair criticism
  that survives: a mean of five bounds the *centre* of the null, not its tail, and the owner already
  has the stronger object (1,000 directions per dataset, S-14).
- **Several of the owner's briefing corrections are already fixed in the committed draft.** [B-2 ·
  CONFLICT · VERIFIED] `docs/COLLABORATOR_BRIEFING.md` §5c and `kd/docs/FINDINGS.md` F-88 both say the
  draft is titled *"One Direction Generalizes"*. It is not — that is the separate **2026-09-21
  abstract** (`audit/phase0/collab_abstract_2026-09-21.md`). Already fixed in the PDF: the unsourceable
  "0.09 below the contrast" (now "from 0.220 to 0.050 to 0.023"); the inverted-training-row "0.18–0.21
  lift" (removed); "six Liars' Bench datasets" (now five + RepE); "the independently constructed
  sycophancy axis" (Appendix F now says same bank); "layers 20 and 40" (L44 now reported everywhere).
  Also: `DRAFT_CORRECTIONS` 1.10–1.13 are about **his `loo_L20/summary.md`**, not the draft — keep the
  two lists apart so he is not sent corrections to a paragraph that no longer exists.
- **Two small misstatements about his files in her briefing.** [B-6] `pool_activations.py` pooling is
  at **line 123**, not 84 (line 84 is `if acts is None:`) — the substantive point, that he mean-pools,
  is correct (F-97). And *"every one of your summaries records Δ = 0.0000"* is **NOT-FOUND** — the
  string is not in his `summary.md` files; the related reproduction claim D-233 reproduces exactly for
  the follow-up half (20/20 cells), while the three response-token runs committed only plots and no
  `results_table.csv`, so "matched for each layer" can be **neither confirmed nor refuted**. Minor:
  O-006 notes the briefing cites `extended_arms()` as defining the axis; it re-fits the geometry — the
  axis is defined in `persona_pcs.py` (`axes["extended"]`).

---

## 5. OWNER vs COLLABORATOR DISAGREEMENTS

**Two causes are ruled out globally.** [DISAGREEMENTS §12] **No disagreement is traceable to a seed**
(his matrices use SEED = 42 throughout; her crossgen main seed is 42; no compared cell crosses a seed
boundary). **No disagreement is traceable to k** (his geometries fit at k = 3, her crossgen banks at
k = 5, but every compared object is a contrast direction or PC1, neither of which depends on k; the
one k-sensitive comparison, `ext_pc1`, shows a residual of 0.010–0.014 where contrast arms show
0.000–0.004 — which **confirms** her F-98 caveat that her PCA basis includes the `default` anchor and
his does not).

| # | Disagreement | Likely cause | Which side the raw data supports |
|---|---|---|---|
| **D-03** | His HP-KR axis reads 0.873–0.883; the wording-matched version of the same contrast on the same bank reads 0.592–0.669 | **Contrast definition** — 4 denial vs 6 *merged* candour, vs 4 vs 3 *split* | **Hers.** Both numbers are correct; his merged pole is carried by plain assertion. VERIFIED numbers; the register-vs-sincerity reading is **INFERRED** from the contrast definitions. In **neither** document. |
| **D-02** | The same name, "the paper's 8v7 deception axis", denotes two different directions; up to **0.148 AUROC** apart on HP-KR at L20 (her 0.2504 vs his 0.1029) | **Bank.** His arm is her *repaired* `run-paper16-shared` bank (all 26 personas answer the same questions); her headline crossgen stores use `axes_vectors_s3_fp32.pt`, the **original** bank whose persona families answered different question sets (the AD-2 confound) | **Neither is wrong; both are mislabelled.** Fingerprint match his↔her repaired bank: mean \|gap\| 0.0015 over 10 cells (0.0000–0.0004 on four of five datasets). The repaired bank gives the **cleaner** axis — better on convincing-game (+0.085 / +0.119), more strongly inverted on HP-KR (−0.148 / −0.113), consistent with F-18 (**INFERRED**, sign pattern, two datasets). **Coverage trap:** the repaired bank has 252 matrices on the lab PC but **only 2 on the laptop** (mean and last, L40) — anyone re-deriving locally silently gets L40 only. |
| **D-04** | The "8192-d probe" baselines differ by **+0.0555 (L40) / +0.0648 (L44)**, his higher; **−0.0075 at L20** | **Training protocol.** His off-diagonal cells train on **100% of the source, unbalanced**; hers on a **class-balanced 80% subsample**. On the harm-pressure sets that is 10× the rows (HP-C 2,134 vs 214; HP-KR 2,139 vs 222). Diagonals differ too (5-fold CV vs held-out 20%) | **Both correct; not comparable.** The estimator is identical on both sides (StandardScaler + LogisticRegression C=0.1, no intercept, max_iter 1000). Cause is **VERIFIED** as a protocol difference; that the row count is the *dominant* term is **INFERRED** (it is the only difference growing with depth in the right direction). Her margins and his margins must never appear in one sentence without this said. |
| **D-01** | Her "reproduces exactly" rebuild of his headline arm reproduces his **control** arm | **Extraction / judge filter.** His published `persona_ood_ext_*` is built from her **s3** vectors; her rebuild (`axes_collab_s23.pt`) used **s23** | **Both.** Role membership does reproduce exactly (12 v 13 confirmed). The numbers sit on his control: mean \|gap\| 0.0015 over 15 cells vs his `mod_*`, and 0.021 from his published arm (max 0.0759, HP-KR L20: his 0.8816 vs her 0.8057). **F-98/F-99's conclusion is not overturned — it gets slightly stronger** (`ext − axis1` at L20 is +0.1365 at s3 vs +0.1180 at s23 on the shared five). One sentence fixes it. **NOT RECOMPUTABLE:** her eleven-dataset table cannot be redone at s3 without a new run. |
| **B-1 / D-07** | Her F-62: the s3→s23 swap makes *"no measurable difference"* (largest gap 0.005). His `persona_ood_mod_*` measures a consistent **loss** on the same swap | **Arm and dataset set.** She measures `pc:` features on eleven datasets; he measures zero-shot contrasts on six | **Both, on their own objects — COMPATIBLE as measurements, CONFLICT as headlines.** His numbers: `ext_axis1` 0.9041→0.8839 / 0.9158→**0.9066** / 0.9034→0.8905 (L20/40/44); `axis1` 0.6511→0.6505 / 0.8319→0.8216 / 0.8260→0.8178; HP-KR zero-shot 0.882→**0.805** / 0.912→**0.876** / 0.894→**0.849**. Mechanism measured: group means converge — axis norms fall to 0.84–0.89 while mean persona norm holds at 0.98. The audit also finds F-62's 0.005 is her **best case** — at other layers/poolings the gap reaches **0.030**. **Consequence neither has written down [INFERRED from two VERIFIED facts]:** her replication of his arms (F-98/F-99/O-011/O-012, the "+0.055 at 21 of 21 layers") is built on the bank **his own control found worse**; on his six that costs `ext_axis1` 0.009–0.020. It changes no sign, and most should cancel in a difference of two arms over the same vectors — but it is undocumented on both sides. |
| **D-05 / D-06** | "The persona axis on HP-KR at layer 20" has **six** values in circulation: 0.109, 0.124, 0.1029, 0.1483, 0.1652, 0.2504 | **Bank and object.** 0.109 = the RepE-trained *base probe* · 0.124 = the published Liars' Bench number · 0.1029 = 8v7 on the 26-persona shared-question bank (her repaired bank gives 0.1028) · 0.1483 = 8v7 on the 36-persona kd bank at s3 (hers 0.1482) · 0.1652 = same at s23 (hers 0.1652) · 0.2504 = 8v7 on her original question-confounded bank | **All six are correct for their own configuration.** Three appear in the two documents **without the bank named**, spanning a factor of 2.4 on the number the whole HP-KR narrative rests on. Separately, D-06: his `persona_ood_ext_L20/summary.md:62-69` table takes its baseline and extended columns from `persona_ood_ext_L20` but its **deception-axis column from the other store** (`persona_ood`) — six of six cells match the other store exactly. **A documentation slip, not a result error: his draft distinguishes the banks correctly** (`draft_text.txt:347`). It is the origin of the "0.103 vs 0.109" puzzle, which has three answers, not two. |
| **D-09** | "Mean off-diagonal AUROC" is not the same quantity on the two sides | **Feature dimension.** His persona arms are **1-dimensional** — a probe on a 1-d feature can only learn a sign, so his trained off-diagonal mean *equals* his zero-shot mean whenever every training set agrees on the sign (0.9158 both ways at L40). Hers are **k-dimensional** (k = 5 default) | **Both correct; a like-for-like comparison of "trained transfer" between the two sides does not exist.** Plus the four L20 sign-flipped rows (§1.6). |
| **D-10** | Top-decile-token vs mean pooling: same sign at every layer, **magnitudes 4× apart**. Hers +0.103 / +0.005 / +0.014 (L20/40/44); his +0.023 / +0.027 / +0.049 | **Metric.** She pools all datasets into one AUROC (so between-dataset score offsets count); he averages per-dataset AUROCs (so they cannot) | **Not a result disagreement — a metric artifact. They should quote each other's direction, not each other's number.** His side **VERIFIED** (recomputed from `persona_positions_L*/reads_summary.csv`); **hers is DOC-ONLY** — a memory note, not re-derived. On his *extended* axis the sign flips at L20 (0.8868 → 0.8727), which is a different axis, not a contradiction. |
| **D-08** | Insider-trading at **L20 only** disagrees after banks are matched: his − hers = +0.0107 (8v7), +0.0175 (12v13), +0.0121 (PC1), +0.0694 (sycophancy), +0.0090 (26-persona bank). L40/L44 agree to ≤0.007 | **UNCLEAR.** Most likely the two pipelines pool over slightly different token spans — insider-trading is the only one of the five with a structured scratchpad-plus-action format, and L20 representations are more position-sensitive. **INFERRED** | **Cannot be settled from the files as they stand.** VERIFIED that the gap exists and is dataset- and layer-specific; his number is higher in every case. **What would settle it:** his `activations/pooled/insider_trading_cadenza__plain` row mask against her detection mask — the mask *definition*, not the activations. Too small to move a conclusion (insider-trading sits at 0.97–0.99 on every arm), but anyone reproducing §0 will see it. |
| **D-11** | Apparent conflicts that are only **different dataset sets** | 12v13 zero-shot L40: 0.9158 (his six) vs 0.7996 (her eleven) — on the five both evaluate, 0.8990 / 0.8882. 8v7 L40: 0.8319 vs 0.7325 — on the shared five 0.7983 / 0.7868. "Six Liars' Bench datasets" includes `repe_honesty`, not in her eleven. Pooled-baseline inversion 0.124 (L20) / 0.523 (L40) is two **weightings** of his own table (balanced vs naive). "Layer 40 is best" vs F-91's L32 is a **coverage gap** — he never extracted L32 | **Reference only — not conflicts.** VERIFIED. |

### Cannot be settled (carried forward unchanged)

- **C-1 — who first quantified HP-KR's denial-wording confound.** Her `kd/docs/DESIGN.md` at
  `5b43a7a` (**2026-09-19 02:58**) already has the full 2×2 table with the empty fourth cell and the
  measured denial-regex AUROC **0.870**; his `persona_ood_ext_L20/summary.md` at `9774075`
  (**2026-09-20 17:53**) states the 87% figure. Git puts her 1.6 days earlier. **Git cannot see a
  conversation.** What would settle it: his own session logs or notes for 09-19/09-20. **UNCLEAR.**
- **C-2 — whether her rebuild of his arms on `vectors_s23` was deliberate.** Only she can say. It
  affects the exact size of the +0.055, not its sign. **UNCLEAR.**
- **C-3 — pooled training, independent arrival one day apart.** Her docstring credits Ying et al. and
  does not mention his run, which is consistent with independence but does not prove it. One answer
  from either author closes it. **UNCLEAR.**

---

## Two audit items owed before anything here is quoted in the paper

1. **F-112** — *"`pc:kd_assertion` clears the matched dataset-PCA control at 0 of 126 tests"* — is
   cited in `audit/METHODS_CHAPTER.md` and `audit/LIMITATIONS_AND_FUTURE_WORK.md` but has **no row in
   `audit/claims.csv`**. It is **DOC-ONLY** and it is load-bearing for §3.2's claim-changing control.
2. **S-14 / the per-dataset random null** (§2.1) has **no F-number** in `kd/docs/FINDINGS.md`.
3. **`audit/claims.csv` D-107 is wrong** about his random control (§4) and should be amended.
