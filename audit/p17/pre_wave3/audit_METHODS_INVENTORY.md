# Methods, lineage and outcomes — what this project built, where it came from, and what it settled

Written 2026-09-23 at the owner's request, and expanded the same day on her instruction that the
collaborator's draft is **not** an index of the work: *"he even does not know what was built
actually. My branch consists of the initial foundation and most of the unique design, including
fail and pass ones."* This document is therefore built from the `asix` branch itself — the ledger,
the design docs, the pipeline source and the paper notes — not from what the draft happens to cite.

**The brief does not ask for this.** Phase 5 specifies only `REPORT_PLAIN.md` and
`REPORT_TECHNICAL.md`, both results-oriented. This is an addition, because a thesis needs a methods
chapter and because the failures are part of the contribution.

**Evidence marking, applied to every row and load-bearing throughout:**
**[AUDITED]** re-derived from raw data in this audit · **[LEDGER]** recorded in `FINDINGS.md` with
a source, not yet claim-checked · **[CODE]** read from the cited source file · **[DOC-ONLY]**
asserted in a document, unchecked.

---

# Part 1 — Lineage: what was adopted, and what was made of it

Nothing here was invented from nothing, and that is not a weakness — the contribution is in the
adaptation. Each row is *what we took* → *what we changed* → *why the change was necessary*.

## 1.1 The method being replicated

| Adopted from | What we took | What we changed, and why |
|---|---|---|
| **Mahadik & Skapars**, *Do Linear Probes Generalize Better in Persona Coordinates?* (arXiv:2605.09391) | The whole method: contrastive persona vectors, PCA persona coordinates, Eq (2)/(4), projection probes | **Scale** — they test 3B and 8B; we test **70B**, where the effect behaves differently (F-58 → F-63 → F-100 → F-110). **Breadth** — 11 datasets against their 5–6. **k** — they never fix k (k=1, 2 and 3 in different figures); **nor do we fix it to theirs: our default is k=5** (`27_crossgen.py:882`), a value the paper never uses, with no recorded reason. See the correction below. **Specification** — 13 ambiguities resolved by decision (§1.4). **Controls** — they run none of §2.2. |
| **Lu et al.**, *The Assistant Axis* (arXiv:2601.10387) | The axis itself, and the persona-elicitation pipeline shape | Repurposed as a **diagnostic** rather than a result: we use it to measure how much of a persona axis is Assistant-Axis drift (F-11), and to test whether a signal survives erasing it. **[LEDGER]** |
| **Chen et al.** (via the Assistant Axis paper) | Contrastive system-prompt generation — positive/negative prompt pairs per trait | Became the construction recipe for **three new banks of our own** (kd, capitulation, sarcasm), each varying a *behaviour* rather than a trait. **[CODE]** `pc_personas.py`, `aa_roles.py` |

## 1.2 The evaluation corpus

| Adopted from | What we took | What we changed |
|---|---|---|
| **Liars' Bench** (Cadenza Labs, arXiv:2511.16035) | `harm-pressure-knowledge-report`, `harm-pressure-choice`, `instructed-deception`, `convincing-game`, `sandbagging`, and the **1% FPR / Alpaca control** operating point | HP-KR is **not used by the replicated paper at all** — which is why the wording confound was undetectable there (F-23). We added the matched slice that makes it attributable. **[AUDITED]** |
| **MASK** (Ren, Agarwal, Mazeika, Menghini et al., arXiv:2503.03750) | The honesty-vs-accuracy separation, and `mask` as an eval column | Used as one column of eleven rather than as the benchmark |
| **Apollo / Goldowsky-Dill et al.**, deception-detection | `roleplaying`, `insider-trading`, and the probe families (`mms`, `lr_reg10`, the "paper" probe), RepE directions | Their own conceded failure mode — *the probe fires on "is this scenario deception-related"* — is what **forced the sarcasm bank's v1→v2 redesign** (§3.2). **[CODE]** `cap/docs/SARCASM_DESIGN.md` |
| **Pacchiardi et al.**, LLM lie detector | `ai_liar` | Eval column only |
| **Ying, Ravfogel, Kriegeskorte & Hase**, truthfulness spectrum (arXiv:2602.20273) | Their finding that post-training pushes sycophancy into its own subspace — independent corroboration of F-13 — and their **Mahalanobis predictor**, which we tested | The predictor **did not reproduce** on our grid with either covariance (F-83). A negative result about someone else's method. **[LEDGER]** |

## 1.3 Statistical machinery

The dataset-level bootstrap and the BH correction are standard tools; what is ours is the choice of
**resampling unit** and the decision to apply it to every headline. Justification, from the ledger:
with ~15,000 rows, row-level intervals are tight by construction and mean almost nothing — the
thing that actually varies between replications is *which datasets you picked*, and there are only
eleven. **[AUDITED]** — re-derived independently for F-76.

## 1.4 Specifying a method the paper leaves underspecified

`docs/papers/persona-coordinates-notes.md` §8 lists **13 ambiguities and internal inconsistencies**
that an implementer must resolve by decision. Each resolution is a methodological choice this
project made and the paper does not state:

- **k is not fixed** (k=1 for deception, 3 for sycophancy, 2 for the unified axis, against a stated
  "first 3 PCs"). **Correction, 2026-09-23 (workstream M2):** an earlier draft of this document said
  "we sweep k". **That is not supported.** Three values were used across three separate arms (k=5 in
  the crossgen/kd11 default, k=3 in the scale rungs, k=2 in the `@2` features) with **no within-arm
  sweep**, and none of them is one of the paper's figure values. The choice matters: on the paper's
  own five datasets at L40, `pc:unified@2` beats its matched control by **+0.138** while
  `pc:unified` at k=5 collapses to **+0.004** (`kd/docs/FINDINGS.md:1032-1040`, re-verified) — so at
  k≈2, close to the paper's k=1, the axis reproduces the paper's order of effect, and at the default
  k=5 it does not.
- **Activation sublocation unspecified** (post-MLP? post-attention? post-block?).
- **Test-time centering unspecified** — axes are built from *centered* persona vectors but score
  *un-centered* evaluation activations, and there is no persona set at test time to center against.
  **[CODE]** `6_axes.py:19-21` resolves this **with a proof, not a guess**: *"Eq (2) centers every
  persona vector by the mean across personas before Eq (4) takes a difference of class means. That
  centering cancels exactly in a difference of class means."*
- **The anchor persona's role unstated** (in D, in H, or excluded) → **[CODE]** `6_axes.py` handles
  the default anchor explicitly.
- **Variance scaling unspecified**; **Table 1 inconsistent with the figures**; **no code release**.

---

# Part 2 — What we designed, what it was designed to test, and how it came out

This is the part the draft cannot supply. Every row pairs a **design** with the **claim it was
built to decide** and the **outcome**, including the designs that failed. A failed design that was
*built and run* is evidence; it is not a gap.

## 2.1 Datasets and banks built

| Design | Built to test | Outcome | Evidence |
|---|---|---|---|
| **`harm-pressure-kr-matched`** — the matched slice | Is HP-KR's inversion a *wording* confound? Its label is `deceptive = knows AND denies`, and the fourth cell is empty | **TRUE, then narrowed.** The slice reverses the result (F-03) — but the reversal is **layer-localised**: 0.739 at L40, 0.374 at L20 (F-72), and the margin **does not survive the dataset bootstrap** (F-76) | **[AUDITED]** F-20, F-72, F-76, F-112 |
| **The kd persona bank** — feigned-ignorance vs candour | Can a persona axis built for *denying knowledge one has* detect HP-KR lies where register axes invert? | **PARTLY.** Zero-shot it lifts the abstract's headline (it is 44–46% of the collaborator's margin, F-97/F-98/F-99). Trained, `pc:kd_assertion` beats raw at 18/21 layers and survives dropping HP-KR — but clears the **matched control at 0 of 126 tests** | **[AUDITED]** F-112 |
| **The capitulation bank** — 13 roles, teacher-forced turn 1 | Is F-13's *register* diagnosis right? Vary capitulation instead of register and see whether the axis stops failing | **TRUE on its own column** — 0.745 zero-shot vs 0.374; clears the bootstrap (+0.112, CI [+0.050, +0.185]) — **and LOCAL**: no gain on the other ten columns (F-67), with the format-matched control deflating the interpretation (F-89) | **[AUDITED]** F-65/F-66; teacher forcing verified directly |
| **`sycophancy_mt`** — multi-turn capitulation | Sycophancy as *abandoning a correct answer under pressure* — invisible to single-turn elicitation (F-15) | **BUILT AND USED.** 1,182 conversations, 434 retracted / 748 maintained. It is the column on which every register axis fails | **[AUDITED]** |
| **The deadpan/sarcasm bank** — to isolate **deceptive intent** | The sharpest form of the register question: sarcasm says something false *without intending to deceive*. If the deception axis fires on it, the axis is reading form, not intent | **FAILED TWICE, AND THE FAILURES ARE INFORMATIVE** — see §3.2 | **[LEDGER]** F-92, F-93, F-106, F-107 |
| **Own WMDP knowledge scoring** | Is the benchmark's knowledge label good enough to attribute HP-KR? | **NO** — scoring the model ourselves gives a cleaner label (F-08), and both continuous knowledge signals saturate (F-09) | **[LEDGER]** |

## 2.2 Controls and tests introduced — the largest contribution

**Correction, 2026-09-23 (workstream M3).** An earlier draft said "most do not exist in the
replicated paper". **That is wrong for two of the fourteen:** the **dataset-PCA** and
**random-subspace** controls *are* in Mahadik & Skapars (notes §5.4–5.5, Appendix Figure 9), and
this project's own code says so — `27_crossgen.py:105` calls it *"The paper's dataset-PCA control"*.
The other **nine are genuinely absent** from the paper (verified). **The claim the thesis should
make is the narrower one:** what is ours is the **matched-k** discipline and the **dataset-level
bootstrap** applied to every headline — not the invention of the controls themselves.

What makes them a contribution is that they are what **kills this project's own headlines**. A
replication that only confirmed things would not have needed any of them.

| Control | What it defends against | What it actually did here |
|---|---|---|
| **Matched dataset-PCA control at matched k** | A low-dimensional persona basis beating raw proves nothing unless it beats a *same-dimensional* basis built from the data | **Killed several headlines**, most recently `pc:kd_assertion` (0 of 126 clear) **[AUDITED]** |
| **Dataset-level bootstrap** | Row-level intervals are tight by construction; the real variance is dataset choice | Killed F-75's plateau (F-76); passed the cap result (F-66) **[AUDITED]** |
| **Per-dataset random-direction null** | A random low-dimensional projection "gains where raw is weak" | Several zero-shot AUROCs are **no better than random directions** **[LEDGER]** |
| **Length-only control** | The positive class is longer in 9 of 11 datasets, so "longer is deceptive" transfers by itself | 0.5555 off-diagonal — and **newly established in this audit: invariant across layer, pooling and probe**, because a fixed 1-d feature fits only a threshold **[AUDITED]** |
| **Bag-of-words falsification gate** | If words alone separate the classes, an activation result proves nothing | **Failed the sarcasm pilot at 0.92** (F-93), which stopped the bank being built **[AUDITED]** |
| **Format-matched control** | Separates *presence* of a persona from its *content* | Deflated F-66's interpretation (F-89, F-94) **[LEDGER]** |
| **Leave-one-CLUSTER-out** | Holding out a dataset is not enough when a row-level leak crosses datasets | Found the HP-KR ↔ kr-matched leak; the claim survived at reduced magnitude (F-108) **[AUDITED]** |
| **Layer pre-registration** | A headline moved 0.374 → 0.739 across depth | `docs/LAYER_PREREGISTRATION.md`; the failure it prevents recurs (F-72, F-91, F-96) **[AUDITED]** |
| **Factor probes (knows / denies)** | A dataset whose label is a conjunction cannot attribute itself | Made HP-KR attributable at all (F-05) **[LEDGER]** |
| **Teacher-forced turn 1** | Persona wording leaking into the measured turn | **Structural, not statistical** — verified: 20 questions, exactly one distinct turn-1 text each, across all 65 arms **[AUDITED]** |
| **Cluster-permutation test**, **PC1 permutation test**, **seed stability** | Completing pre-registrations; geometry better than chance; results that move with the seed | F-84, F-12, F-83/F-87 **[LEDGER]** |

## 2.3 Baselines the paper did not run

| Baseline | Outcome |
|---|---|
| **Pooled training at equal data volume** | **The project's strongest positive result.** Beats every persona result here, without personas, and it clears the bootstrap at **21 of 21 layers**. The gain is **diversity, not volume** (F-79/F-80/F-81) **[LEDGER, partly audited]**. **Attribution, corrected 2026-09-23:** the *pooled leave-one-dataset-out baseline itself is the collaborator's* — `6b16a46`, 2026-09-20, "Leave-one-dataset-out: the fair baseline for the generalization claim", one day before `30_premise.py`. **What is the owner's is the equal-row-budget arm**, and the finding that follows from it (diversity, not volume). **F-85's sentence "The comparison has never been made against a *pooled* probe" (`FINDINGS.md:3413`) was already false when written** — his `loo_L{20,40}/liars_bench_comparison_loo.csv` contains that exact comparison. |
| **Single-source direction averaging** | Explains *why*: single-source directions are near-orthogonal, so averaging cancels noise — joint training is not needed (F-86, F-95) **[LEDGER]** |
| **Raw 8192-d probes on the same rows** | The honest comparator throughout **[AUDITED]** |

## 2.4 Infrastructure that is itself a method

**Attribution note (M3):** the **resumable activation cache and token-budget batcher are the
collaborator's** — `e391b3e`, 2026-09-04, eleven days before the owner's `ed8daca`, whose own commit
message reads "**Port** the resumable activation cache and token-budget batcher". The port is hers;
the design is his.

**The NDIF pipeline is reimplemented, not imported** — `git submodule status` is clean and nothing
in `src/`, `kd/`, `cap/` or `scale/` imports `assistant_axis`; ~45 numbered scripts run the remote
70B pipeline. **[AUDITED]** Also: the attention-memory budget (**rows × heads × T²**, not tokens —
a *per-process* cap, F-42/F-43/F-44); the resumable activation cache; scheduled-task orchestration
(OpenSSH kills the process tree with the session); the bf16 reduction rule (never sum a mask in the
model's bfloat16 — the wrong-last-token variant is silent); and the three-store design (80 layers
on a subsample, 21 layers full corpus, per-token subsample).

---

# Part 3 — The failures, what they limit, and what would settle them

## 3.1 Growing the HP-KR matched slice — gate failed (F-59)

**Design.** Use the 312 WMDP items the model is *measured* to get wrong (F-08) as a source of
honest-denial rows: ask something it genuinely does not know under the HP-KR template, and a
truthful "I don't know" is the expected answer. Pre-registered gate: denial rate ≥ 0.70; below 0.40
kills it.

**Outcome: FAILED.** Harmful template 0.250, harmless control 0.150. Stated knowledge tracks
**confidence, not correctness** — the model asserts answers it does not have.

**Limitation this imposes.** The matched slice is capped by **honest deniers**: 134 for
Llama-3.3-70B, setting the whole slice at 258 rows. Fine as an evaluation column, **thin as a
training source (~200 train rows)** — and that is the main caveat on F-03's headline.

**Future work.** Other model families (the denial rate is a property of this checkpoint's
calibration); a template that makes denial socially easier; or accept the slice as eval-only and
never train on it. **[LEDGER]** `kd/docs/NEXT.md` §1

## 3.2 The sarcasm bank — designed to isolate deceptive *intent*, failed two gates

This is the most intellectually developed failure in the project and deserves its own section.

**Why it was designed.** Sarcasm says something literally false **without intending to deceive** —
the listener is meant to understand. If a "deception" axis fires on unmarked sarcasm, the axis is
reading *form*, not *intent*. That is the sharpest available test of the register diagnosis.

**v1 was confounded, and the review caught it.** v1's contrast was
`sarcastic_praise(bad question) − sincere_praise(good question)` — which differs in **two** ways:
whether the approval is meant, and whether the plan is good or bad. Response tokens attend to the
question, so the direction could encode *"the plan under discussion is risky"* rather than *"the
speaker doesn't mean this."* **This is exactly the failure Goldowsky-Dill et al. concede for their
own probe**, and it is worse here, because the deception datasets are themselves built from
bad-situation material (WMDP harm pressure, insider trading, sandbagging). A high
`cos(d_sarcasm, d_deception)` could come entirely from a shared "this scenario is bad" component.
The kd bank had been cited as precedent — but there both sides share the question and the
difference lives in the persona's knowledge state; in v1 the difference lived **in the input
string**. Fixing the wording confound had quietly introduced a *situation* confound.

**Gate 2 failed (F-93, re-verified F-105).** Unmarked ironic and sycophantic responses are still
separable at **0.9213** by bag-of-words alone (95% CI [0.872, 0.967], P(≤ the 0.65 pre-registered
ceiling) = 0.0000, **[AUDITED]**). No activation result on that corpus could be attributed.

**Gate 3 failed (F-107).** On *natural* sarcasm the deception axis is **indistinguishable from a
random direction** (0.54–0.58, inside the random null). The sycophancy axis fires instead, at 0.71.

**Limitation.** The register diagnosis is supported on two constructs (HP-KR, sycophancy) and
**untested on the third**, which was the one designed to isolate intent. The bank remains unbuilt.

**Future work.** `cap/docs/SARCASM_DESIGN.md` specifies **v2, which is designed and not built** —
it adds an arm that isolates intent by holding the question fixed and varying only whether the
speaker means it. Building it is the single highest-value queued item, because it is the only
design here that separates *intent* from *content*. **[DOC-ONLY — v2 is specified, never run]**

## 3.3 The cap result is local (F-67)

**Outcome.** The capitulation bank wins decisively on the column it was built for and **ties
elsewhere**: averaged over all eleven datasets the margin over raw is +0.098 with an interval
containing zero.

**Limitation.** A construct-matched bank is a **targeted instrument, not a general detector**. That
is a real finding about the method, not a shortfall of the bank.

**Future work.** Either build one bank per construct and accept the cost, or establish what would
make a bank general — the pooled-training result (F-79) suggests diversity of *source*, not
persona design, is what generalises.

## 3.4 The matched slice does not clear the bootstrap (F-76), and 11 datasets is the binding constraint

**Outcome.** The margin at the pre-specified L40 is not distinguishable from zero
(CI [−0.069, +0.189], P = 0.161), and the argmax layer is unidentified across 93% of the network.
**[AUDITED]** — re-derived independently.

**Limitation.** F-04 identified the real constraint: with ~11 datasets, **almost nothing can clear
a dataset-level bootstrap**. This is a ceiling on what the whole project can demonstrate, not a
property of any one result.

**Future work.** `docs/OPEN_DECISIONS.md` A3 records the only lever that attacks it: **an 11 → 14
dataset grid**. That is a decision the owner has not yet made.

## 3.5 `pc:kd_assertion` beats raw but not the matched control (F-112)

**Outcome, established in this audit.** Positive at 18 of 21 layers over raw, +0.10–0.15 from L32
upward, **and not local** — dropping both HP-KR columns raises it. But against the matched
dataset-PCA control: positive at 20/21 layers, **0 of 126 tests clear**.

**Limitation.** A consistent, depth-robust, non-local advantage that the dataset bootstrap cannot
separate from its control. The same category as F-76.

**Future work.** The 11→14 grid again; or a pre-registered single-layer test at L40 with the matched
control, which is the only form that could clear at this sample size.

## 3.6 Two negative results about other people's methods

- **Ying et al.'s Mahalanobis predictor does not reproduce** on our grid, with either covariance
  (F-83). **[LEDGER]**
- **The published persona-coordinates gain dies at 70B** while replicating at 3B and 8B
  (F-58 → F-63 → F-100 → **F-110**, where the corrected like-for-like 70B rung is **+0.054 vs the
  matched control and −0.013 vs raw**). **[AUDITED]**

---

# What this document still owes

1. **Items marked [LEDGER] / [DOC-ONLY] are not claim-checked.** Phase 2 has checked 169 of 692
   claims; F-05, F-08, F-12, F-42–F-44, F-79–F-81, F-83, F-86, F-89 sit in the unchecked remainder.
   This records what was *built and claimed*; it does not yet certify each one.
2. **The extraction-vs-paper comparison is inferred**, from the ambiguity list and `6_axes.py`.
   A line-by-line reading of `2_activations_ndif.py` against Eq (1)–(4) would settle it properly.
3. **The attribution list.** On present evidence every design in §2.1, the controls in §2.2 and the
   baselines in §2.3 are the owner's work, and the collaborator's draft names none of them as such.
   **Phase 3 should produce that list explicitly** — it is what a joint thesis needs, and it is the
   reason this document exists.
