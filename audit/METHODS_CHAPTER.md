# Methods — a chapter draft, with every number sourced

_Phase M workstream M4, assembled 2026-09-23 from `METHODS_INVENTORY.md`,
`methods/EXTRACTION_VS_PAPER.md`, `methods/ATTRIBUTION.md`, the M1 claim batch, and the ledger
entries F-111 / F-112 / F-113._

**How to read the markers.** **[AUDITED]** = re-derived from raw data during the audit ·
**[LEDGER]** = recorded in `kd/docs/FINDINGS.md` with a source, not independently re-derived ·
**[CODE]** = read from the cited source file · **[DOC-ONLY]** = asserted in a document, unchecked.
**Nothing marked [LEDGER] or [DOC-ONLY] should enter the thesis as a verified claim without saying so.**

---

## 1. What this work is

A replication of Mahadik & Skapars' persona-coordinates method (arXiv:2605.09391) at a scale and
breadth the original does not reach, with a set of controls the original does not run — and the
controls are the point. Where the original tests **3B and 8B on five or six datasets**, this work
tests **Llama-3.3-70B-Instruct on eleven**, adds three purpose-built datasets, and subjects every
headline to a dataset-level bootstrap.

The honest summary of the result: **the published gain replicates at 3B and 8B and does not survive
at 70B.** Corrected like-for-like, the 70B rung reads **+0.054 against the matched dataset-PCA
control and −0.013 against raw**, where 3B reads +0.120 and 8B +0.103 **[AUDITED, F-110]**.

**The ladder was completed on 2026-09-24 [F-121].** Until then only the published 26-persona bank had
all three rungs; the project's own 36-persona knowledge-denial bank existed at 70B alone. It was
generated at 3B and 8B (5,400 rollouts each, same personas, same 30 shared questions, same five
instruction variants, the Llama-3.1-70B judge held constant) and crossgenned on identical footing —
five datasets, k=3, mean pooling, seed 42. **Both banks decline monotonically with model size**
(26p: +0.038 → +0.015 → −0.012; 36p: +0.036 → +0.019 → +0.005), so **the decline is a property of the
method rather than of the persona set** — the strongest available answer to the objection that the
replication simply used the wrong personas.

## 2. Provenance: what was adopted, and what was made of it

Nothing here was invented from nothing, and the contribution is in the adaptation.

| Source | Taken | Changed, and why |
|---|---|---|
| **Mahadik & Skapars** (arXiv:2605.09391) | The method: contrastive persona vectors, PCA persona coordinates, Eq (2)/(4), projection probes | Scale (70B), breadth (11 datasets), **specification** of 13 things the paper leaves open (§3), and every control in §5 |
| **Lu et al.**, Assistant Axis (arXiv:2601.10387) | The axis and the elicitation pipeline shape | Repurposed as a **diagnostic** — how much of a persona axis is Assistant-Axis drift **[LEDGER, F-11]** |
| **Chen et al.** (via Lu et al.) | Contrastive system-prompt generation | Became the recipe for **three purpose-built banks** varying a *behaviour* rather than a trait **[CODE]** |
| **Liars' Bench** (arXiv:2511.16035) | HP-KR, HP-C, instructed-deception, convincing-game, sandbagging; the 1%-FPR Alpaca operating point | HP-KR is **not used by the replicated paper at all**, so the wording confound was undetectable there **[AUDITED, F-23]** |
| **MASK** (arXiv:2503.03750), **Apollo** deception-detection, **Pacchiardi** | `mask`, `roleplaying`, `insider-trading`, `ai_liar`; the probe families (`mms`, `lr_reg10`, "paper"), RepE | Apollo's own conceded failure mode — the probe fires on *"is this scenario deception-related"* — is what forced the sarcasm bank's redesign (§7.2) |
| **Ying et al.** (arXiv:2602.20273) | Independent corroboration that post-training isolates sycophancy; their Mahalanobis predictor | The predictor **does not reproduce** on this grid, with either covariance **[LEDGER, F-83]** |

**The statistical machinery is standard; the choice of unit is not.** The dataset-level bootstrap
resamples *which datasets were picked*, not which rows — because with ~15,000 rows, row-level
intervals are tight by construction, while the quantity that actually varies between replications is
the dataset set, and there are only eleven. **[AUDITED]**

## 3. Specifying an underspecified method

The replicated paper leaves **13 ambiguities and internal inconsistencies** an implementer must
resolve by decision (`docs/papers/persona-coordinates-notes.md` §8). Five are load-bearing, and all
five are now settled against the code rather than inferred **[AUDITED, M2]**:

| Question the paper leaves open | This work's resolution |
|---|---|
| Activation sublocation | **post-block `model.layers[L].output`** = post-MLP residual stream — the same hook Lu et al.'s own code uses |
| Test-time centering of evaluation activations | **None** — and it *provably cannot change any AUROC*, since a constant per-dataset shift leaves the ranking identical |
| The default anchor persona's role | **Excluded from the contrast, included in every PCA basis** |
| "Standardize" — variance scaling? | **Mean-centering only** |
| k | **5 by default** (`27_crossgen.py:882`) — see the caveat in §8 |

**A methodological result about the paper's own method.** `6_axes.py:19-21` resolves the centering
question with a proof rather than a guess: Eq (2)'s centering **cancels exactly** in a difference of
class means. Verified numerically in float64 at **all 80 layers** for all three contrasts —
max |d_uncentered − d_centered| ≤ **2.9e-15** **[AUDITED]**.

**Fidelity of the implementation.** 36 method steps were compared against the paper: **16 match, 11
resolve an ambiguity, 6 are deliberate deviations, and 4 are accidental divergences.** Every
divergence is in **bank construction or k** — the extraction mathematics, Eq (1)–(4), is faithful
**[AUDITED, M2]**.

## 4. Datasets and banks built

| Design | Built to test | Outcome |
|---|---|---|
| **`harm-pressure-kr-matched`** (258 rows, 124/134) | Whether HP-KR's inversion is a *wording* confound — its label is `deceptive = knows AND denies` with the fourth cell empty | Reverses the result **[LEDGER, F-03]**, but the reversal is **layer-localised** (0.739 at L40, 0.374 at L20) and does **not** survive the dataset bootstrap **[AUDITED, F-72, F-76]** |
| **The kd persona bank** (feigned ignorance vs candour) | Whether an axis built for *denying knowledge one has* detects HP-KR lies where register axes invert | Zero-shot it carries **44–46% of the collaborator's headline margin [AUDITED, F-97–F-99]**. Trained, `pc:kd_assertion` beats raw at 18/21 layers and survives dropping HP-KR — but clears the matched control at **0 of 126 tests [AUDITED, F-112]** |
| **The capitulation bank** (13 roles, teacher-forced turn 1) | Whether the *register* diagnosis is right: vary capitulation instead of register | **0.745 zero-shot vs 0.374**, clearing the bootstrap (+0.112, CI [+0.050, +0.185]) — **and local**: no gain on the other ten columns **[AUDITED, F-65–F-67]** |
| **`sycophancy_mt`** (1,182 conversations, 434/748) | Sycophancy as *abandoning a correct answer under pressure*, invisible to single-turn elicitation | Built and used; the column on which every register axis fails **[AUDITED]** |
| **The deadpan/sarcasm bank** | Whether a "deception" axis fires on speech that is literally false **without intent to deceive** | **Failed two pre-registered gates** — see §7.2 **[AUDITED, F-93, F-106, F-107]** |
| **Own WMDP knowledge scoring** | Whether the benchmark's knowledge label suffices to attribute HP-KR | It does not; scoring the model directly is cleaner **[LEDGER, F-08]** |

## 5. Controls and tests — the principal methodological contribution

**The precise claim, and it must be stated precisely.** Two of these — **dataset-PCA** and
**random-subspace** — *are* in the replicated paper (notes §5.4–5.5, Appendix Figure 9), and this
project's own code says so: `27_crossgen.py:105` calls it *"The paper's dataset-PCA control"*. **Nine
of the remaining are genuinely absent from the paper.** What is original here is the **matched-k
discipline** and the **dataset-level bootstrap applied to every headline** **[AUDITED, M3]**.

**What makes them the contribution is that they are what kills this project's own headlines.** A
replication that only confirmed things would not have needed any of them.

| Control | Defends against | What it did here |
|---|---|---|
| **Matched dataset-PCA at matched k** | A low-dimensional basis beating raw for reasons unrelated to personas | Killed `pc:kd_assertion` (0 of 126) **[AUDITED]** |
| **Dataset-level bootstrap** | Tight row-level intervals that mean nothing | Killed F-75's plateau; passed the cap result **[AUDITED]** |
| **Per-dataset random-direction null** | A random projection "gaining where raw is weak" | Several zero-shot AUROCs are no better than random **[LEDGER]** |
| **Length-only control** | The positive class being longer in 9 of 11 datasets | 0.5555 off-diagonal — **invariant across layer, pooling and probe**, because a fixed 1-d feature fits only a threshold **[AUDITED]** |
| **Bag-of-words falsification gate** | An activation result that words alone could produce | Failed the sarcasm pilot at **0.9213**, 95% CI [0.872, 0.967] **[AUDITED]** |
| **Format-matched control** | Confusing a persona's *presence* with its *content* | Deflated F-66's interpretation **[LEDGER, F-89]** |
| **Leave-one-CLUSTER-out** | A row-level leak crossing dataset boundaries | Found the HP-KR ↔ kr-matched leak; the claim survived at reduced magnitude **[AUDITED, F-108]** |
| **Layer pre-registration** | A headline that moves 0.374 → 0.739 with depth | `docs/LAYER_PREREGISTRATION.md` **[AUDITED]** |
| **Teacher-forced turn 1** | Persona wording leaking into the measured turn | **Structural, not statistical**: 20 questions, exactly one distinct turn-1 text each, across all **65 persona arms** **[AUDITED]** |
| **Factor probes**, **PC1 permutation test**, **cluster permutation**, **seed stability** | Conjunctive labels; chance geometry; incomplete pre-registration; seed-fragile results | F-05, F-12, F-84, F-83/F-87 **[LEDGER]** |

## 6. Baselines the replicated paper does not run

| Baseline | Result | Attribution |
|---|---|---|
| **Pooled training at equal data volume** | The strongest positive result here: beats every persona result, without personas, clearing the bootstrap at 21/21 layers; the gain is **diversity, not volume** **[LEDGER, F-79–F-81]** | **The pooled LODO baseline is the collaborator's** (`6b16a46`, 2026-09-20). **The equal-row-budget arm and the diversity finding are the owner's.** **[AUDITED]** |
| **Single-source direction averaging** | Explains why: single-source directions are near-orthogonal, so averaging cancels noise **[LEDGER, F-86]** | Owner |
| **Raw 8192-d probes** | The honest comparator throughout **[AUDITED]** | Both |

## 7. Infrastructure that is itself a method

**The NDIF pipeline is reimplemented, not imported.** `git submodule status` is clean and nothing in
`src/`, `kd/`, `cap/` or `scale/` imports `assistant_axis`; ~45 numbered scripts run the remote 70B
pipeline **[AUDITED]**. Also: **the attention-memory budget** — the per-call limit is
`rows × heads × T²` and the cap is **per process**, not per GPU **[AUDITED, F-42/F-43, with the unit
correction in F-113]**; **the three-store design** (80 layers on a subsample, 21 on the full corpus,
per-token on a subsample); and the **bf16 reduction rule** (never sum a mask in the model's
bfloat16 — the wrong-last-token variant is silent) **[LEDGER]**.

**The resumable activation cache and token-budget batcher are the collaborator's** (`e391b3e`,
2026-09-04); the owner's `ed8daca` says "Port" in its own commit message **[AUDITED]**.

## 8. Authorship

Of the 26 designs, controls and baselines catalogued: **24 OWNER · 1 COLLABORATOR · 1 JOINT ·
0 UNCLEAR**, each with a first-commit hash **[AUDITED, M3]**.

**The collaborator's own designs, which belong in this chapter under his name** — the pooled LODO
baseline with committed per-example scores (`loo_scores.npz` is **the only per-example raw data
committed on either branch**, and it reproduces to 4 dp under an independent implementation); the
rollout-count control `persona_ood_mod_*` (his design, **using the owner's two banks** — his
old/new counts match `vectors_s3` and `vectors_s23` exactly); the follow-up-probe runs; the
per-token persona-position profile; and the `random3` control.

**One independent arrival, recorded fairly:** the per-token/top-decile finding was reached by the
owner on **2026-09-16** (`78232ea`, `14_pertoken.py`) and by the collaborator on **2026-09-22**
(`a8eb902`). Neither is derivative of the other on the available evidence.

**What the draft currently credits:** searching its text — `capitulat` 0, `sarcas` 0, `teacher` 0,
`datapca` 0, `bag of words` 0, `length` 0, `pre-regist` 0, `factor probe` 0, `leave-one` 0,
`sycophancy_mt` 0. The only control of either author's that reaches the draft is his `random3`. The
draft also lists the **built and run** matched slice as *future work*.

## 9. What this chapter does not yet certify

- Entries marked **[LEDGER]** — F-05, F-08, F-12, F-79–F-81, F-83, F-86, F-89 — are recorded with a
  source but not independently re-derived. 177 of 692 catalogued claims have been checked.
- **Four designs have no committed artifact on either branch**: the sarcasm v2 bank (designed, never
  built); `make_view.py` / `make_view_ac.py`, **which build the 11th dataset behind every kd11
  result** (now preserved at `audit/cache/labpc/loose/`); and two persona-vector banks that exist
  only outside git. `sycophancy_mt`'s rows likewise live on the lab PC and Drive, though its
  generator is committed.
- Known caveats that must travel with the numbers are in `LIMITATIONS_AND_FUTURE_WORK.md`.
