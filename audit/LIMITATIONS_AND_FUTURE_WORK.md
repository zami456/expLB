# Limitations and future work

_Phase M workstream M4, assembled 2026-09-23. Companion to `audit/METHODS_CHAPTER.md`._

Each entry states **what was found**, **what it limits**, and **what would settle it**. A design
that was built, run, and failed its pre-registered gate is **evidence, not a gap** — several of the
strongest statements this project can make come from its failures.

Markers: **[AUDITED]** re-derived in the audit · **[LEDGER]** recorded, not re-derived ·
**[DOC-ONLY]** asserted, unchecked.

---

## 1. The ceiling on everything: eleven datasets

**Found.** With ~11 datasets, almost nothing clears a dataset-level bootstrap. Of 384 comparisons in
one bank, **one** cleared; of 4,032 in another, **six** — and four of those six were negative
controls passing **[AUDITED, F-04, F-112]**.

**Limits.** This is not a property of any single result; it is a ceiling on what the whole project
can demonstrate. Every "consistent but not significant" verdict below inherits it.

**Would settle it.** The **11 → 14 dataset grid** — the only lever that attacks the binding
constraint. Recorded as an open decision the owner has not yet made
(`docs/OPEN_DECISIONS.md` A3) **[DOC-ONLY]**.

## 2. The sarcasm bank — designed to isolate deceptive *intent*, failed two gates

**Why it matters.** Sarcasm says something literally false **without intending to deceive** — the
listener is meant to understand. If a "deception" axis fires on unmarked sarcasm, the axis reads
*form*, not *intent*. It is the sharpest available test of the register diagnosis.

**v1 was confounded, and review caught it.** The contrast
`sarcastic_praise(bad question) − sincere_praise(good question)` differs in **two** ways: whether the
approval is meant, and whether the plan is good. Response tokens attend to the question, so the
direction could encode *"the plan under discussion is risky"*. **This is exactly the failure
Goldowsky-Dill et al. concede for their own probe**, and worse here because the deception datasets
are themselves built from bad-situation material. The kd bank had been cited as precedent, and the
analogy broke where it mattered: there both sides share the question and the difference lives in the
persona's knowledge state; in v1 it lived **in the input string**. Fixing the wording confound had
introduced a *situation* confound **[AUDITED, `cap/docs/SARCASM_DESIGN.md`]**.

**Gate 2 failed.** Unmarked ironic and sycophantic responses are separable at **0.9213** by
bag-of-words alone (95% CI [0.872, 0.967]; P(≤ the 0.65 pre-registered ceiling) = 0.0000)
**[AUDITED, F-93, F-105]**.

**Gate 3 failed.** On natural sarcasm the deception axis is indistinguishable from a random
direction. Re-derived independently by the audit across both banks, both layers and both poolings
with a fresh 2,000-direction null per cell **[AUDITED, F-107]**:

| axis | L40 mean | L40 last | L44 mean | L44 last |
|---|---|---|---|---|
| deception | 0.5427 | 0.5856 | 0.5239 | 0.5766 — **below p95 in all four** |
| sycophancy | **0.6930** | **0.7097** | 0.6763 | 0.7083 — beats the null **max** in three |

**Limits.** The register diagnosis is supported on two constructs (HP-KR, sycophancy) and
**untested on the third** — the one designed to isolate intent. The bank remains unbuilt.

**Would settle it.** **v2 is specified and never built** (`cap/docs/SARCASM_DESIGN.md`): it adds an
arm that isolates intent by holding the question fixed and varying only whether the speaker means
it. **This is the single highest-value queued item** — the only design in the project that separates
intent from content.

**Report the conclusion, not the count.** F-107's "below p95 in two cells and barely above in two"
is not robust: the batch measured 1 above / 3 below, the lead's independent null 0 above / 4 below.
The count moves with the draw; the conclusion does not **[AUDITED, F-113]**.

## 3. Growing the HP-KR matched slice — gate failed

**Found.** Using the 312 WMDP items the model is *measured* to get wrong as a source of honest-denial
rows: pre-registered gate ≥ 0.70 denial rate, kill below 0.40. Measured **0.250** (harmful template)
and **0.150** (harmless control). **Stated knowledge tracks confidence, not correctness** — the model
asserts answers it does not have **[LEDGER, F-59]**.

**Limits.** The matched slice is capped by **honest deniers**: 134 for Llama-3.3-70B, fixing the
slice at 258 rows. Adequate as an evaluation column, **thin as a training source (~200 train rows)**
— the principal caveat on F-03's headline.

**Would settle it.** Other model families (the denial rate is a calibration property of this
checkpoint); a template that makes denial socially easier; or accepting the slice as eval-only and
never training on it.

## 4. Results that are consistent but do not clear their control

**`pc:kd_assertion` vs raw** — positive at 18 of 21 layers, +0.10–0.15 from L32 upward, and **not
local** (dropping both HP-KR columns *raises* it to +0.12–0.19). Against the matched dataset-PCA
control: positive at 20/21 layers, **0 of 126 tests clear** **[AUDITED, F-112]**.

**The kd matched slice (F-76)** — margin at the pre-specified L40 not distinguishable from zero
(CI [−0.069, +0.189], P = 0.161), argmax layer unidentified across 93% of the network **[AUDITED]**.

**Limits.** Both are the same category: a real, depth-robust point estimate the dataset bootstrap
cannot separate from its control at this sample size.

**Would settle it.** §1's 11 → 14 grid; or a pre-registered single-layer test at L40 against the
matched control, the only form that could clear at n = 11.

## 5. The cap result is local

**Found.** The capitulation bank wins decisively on the column it was built for and **ties
elsewhere**: over all eleven datasets the margin over raw is +0.098 with an interval containing zero
**[AUDITED, F-67]**. The format-matched control further deflates the interpretation **[LEDGER, F-89]**.

**Limits.** A construct-matched bank is a **targeted instrument, not a general detector**. This is a
finding about the method, not a shortfall of the bank.

**Would settle it.** Either one bank per construct, accepting the cost — or establishing what makes
a bank general. The pooled-training result points at diversity of *source*, not persona design.

## 6. Parameter choices that are not the paper's, and are not recorded as deliberate

**k = 5 is the default** (`27_crossgen.py:882`) — a value the paper never uses, with no recorded
reason. **The result is k-dependent:** on the paper's own five datasets at L40, `pc:unified@2` beats
its matched control by **+0.138**, while `pc:unified` at k=5 gives **+0.004**. At k≈2, close to the
paper's k=1, the axis reproduces the paper's order of effect **[AUDITED, F-39 CORRECTION 2, F-113]**.

**The three-rung scale comparison mixes judge filters.** The 3B and 8B rungs were built from
`vectors_s23` (judge scores 2 and 3), the 70B from `vectors_s3_fp32` (score 3 only) — established
from each bank's own `vectors_dir` field **[AUDITED, DF-47]**. **Impact is bounded by the project's
own measurement:** F-62 found the largest s3-vs-s23 gap to be **0.005**, against margins of
+0.1197 / +0.1027 / +0.0541; and the group counts are identical across all four banks, so the filter
changed which *rollouts* fed each role's vector, not which roles exist. **Name it as a caveat; do not
re-run.** (The scale rungs do **not** differ in k — all four stores are k=3.)

> **CLOSED 2026-09-24 — the scale ladder no longer rests on one bank [F-121].** The larger gap was
> not the judge filter but that the **kd bank existed only at 70B**, so "our bank rescues the method
> at scale" could not be separated from "our bank is better at any scale". Both rungs have now been
> run — 5,400 rollouts each at 3B and 8B, construction identical to the 70B kd bank — and crossgenned
> on identical footing. **Both banks decline monotonically: the decline belongs to the METHOD, not
> the bank**, which closes the "you used the wrong personas" objection to F-63/F-110. The kd bank
> declines less (bank gap −0.005 → +0.005 → **+0.017** across 3B/8B/70B), but at 70B both sit within
> a whisker of zero (+0.005 vs −0.012). **Remaining limitation:** one seed, no significance test, and
> the five datasets contain **no knowledge-denial data** — so this is the bank's *general* effect,
> not the one it was built for. Both variants (`all` and `s23`) were built and are byte-distinct, so
> the filter question above can now be answered directly rather than bounded.

**The unified axis carries a question-set confound.** Unified PC1 correlates **−0.945** with which
question bank a persona answered. **Scope:** the contrast directions barely move between the
confounded and repaired banks (cos **+0.984**), so this damages **the PCA basis only** — it
contaminates `pc:unified`, `pc:unified@2` and the `PC1:unified` zero-shot scorer, and leaves
`contrast:unified` **and all deception and sycophancy numbers intact** **[AUDITED, F-18, F-113]**.
Any "the paper's method at 70B" claim should rest on `run-paper16-shared`.

**Would settle all three.** Two cheap re-runs from rollouts already on disk: re-extract at **k=1**,
and put the paper-faithful unfiltered bank `run-paper16/axes_vectors_all.pt` through crossgen — **it
was built for this and never used**.

## 7. Reproducibility limitations

| What | Status |
|---|---|
| **`make_view.py` / `make_view_ac.py`** — build the 11th dataset behind **every** kd11 result | In no branch; lab PC only. **Preserved by the audit** at `audit/cache/labpc/loose/` **[AUDITED]** |
| **`run-paper16-shared/`** — the repair bank any "paper's method at 70B" claim should use | Not in git, not on the laptop; lab PC + Drive only (268 MB) **[AUDITED]** |
| **`sycophancy_mt`** (1,182 rows) | Generator committed; rows on lab PC + Drive only |
| Two persona-vector banks behind the collaborator's `persona_ood_*` results | Uncommitted on both branches; identified by metadata and n-counts, not blob-verified |
| A `paper/make_numbers.py` the draft promises | **Exists on neither branch** **[AUDITED]** |

**Would settle it.** Commit the two `make_view` scripts and the `sycophancy_mt` rows; or publish a
data statement naming where each artifact lives and its md5.

## 8. Known defects recorded but not fixed

- **`run_scale_chain.ps1:197-203`** loops over both vector variants but writes every iteration to
  `axes_$tag.pt` — **the `vectors_all` bank is built and silently overwritten**, which is the
  mechanical cause of §6's filter asymmetry. Both vector directories exist with 26 files each; one
  bank survives **[AUDITED, DF-48]**. Left unfixed: the one-character change would produce two banks
  where `run_scale_crossgen.ps1:53` expects one.
- **A MiB/MB unit error** in F-42/F-43 and in `scale/scripts/run_scale_coords.ps1:108-109` — the
  values are 512.00 / 384.00 **MiB**. The decision is unaffected **[AUDITED, F-113]**.
- **F-108's "every one of its 258 rows"** overstates a real, structural leak: ~**81%** (42 of 52
  measured). The same sentence is in `30g_clustered_holdout.py:12-14` **[AUDITED, F-113]**.

## 9. Open decisions the owner has not made

1. **The 11 → 14 dataset grid** — the only lever on §1's ceiling (`OPEN_DECISIONS.md` A3).
2. **Which cap questions to cut** — two external reviewers independently rate the same four as too
   subjective; Q8 is unanswerable as written (it omits N) (A1).
3. **First-person questions — all 20 or none** — three of twenty are first-person, and two of those
   three are among the clearest-answer items, so the confound lands where the read is cleanest (A2).
4. **Whether to build sarcasm v2** (§2) — the highest-value queued item.
5. **Whether to fix `run_scale_chain.ps1`** (§8).

## 10. Future work, ranked

1. **Build sarcasm v2.** The only design that separates intent from content; both prior gates failed
   informatively, and the v2 specification already exists.
2. **Extend to 14 datasets.** Attacks the significance ceiling that limits every result above.
3. **Re-run at k=1 and with the unfiltered bank.** Cheap, from rollouts already on disk, and it
   closes the three §6 caveats at once.
4. **Test whether pooled training's advantage is the general phenomenon.** It is this project's
   strongest positive result, it needs no personas, and the reason it works — near-orthogonal
   single-source directions — predicts where it should and should not hold.
5. **Replicate the register diagnosis on a model family whose post-training differs.** Ying et al.
   give the mechanism (post-training isolates sycophancy); a base-vs-chat comparison would test it
   directly.
