# Phase 5 digest — results, negatives, unfollowed leads, weak spots, stale foundations

Source: `audit/phase3/WEAK_SPOTS.md` (workstream P3-C) and `audit/phase3/UNFOLLOWED.md` (P3-D).
Nothing here is re-derived; every number is carried from those two files with its tag.
**Tags:** VERIFIED (recomputed from a raw file during the audit) · DOC-ONLY (only in prose) ·
CONFLICT · NOT-FOUND · INFERRED · NOT MEASURABLE.

**Terms, once.** *Off-diagonal mean* = train a lie-detector probe on one dataset, test on each of the
others, average the AUROC. This project's transfer score; 0.5 = chance. *raw* = a probe on the full
8,192-dimensional activation — the baseline the method must beat. *datapca* = the matched control:
same number of dimensions as the persona feature, but taken from a PCA of the evaluation data.
*random* = a random projection of the same dimension — the noise floor. *persona features*
(`axes`, `pc:*`) = the method under test. *BH q* = p-value corrected for how many tests ran at once.
*LODO* = leave-one-dataset-out. Unless stated, all numbers are **seed 42**.

**One thing to read first.** Almost every test in this project resamples the 11 *datasets*, not the
thousands of rows. That makes the median 95% interval **±0.125 AUROC** while the median persona-vs-raw
effect at the reported settings is **0.044**. Noise is three times signal. Any margin below ~0.13 in
this project is a point estimate, not a demonstrated effect — including most of §1. Rank accordingly.

---

## 1 · STRONGEST RESULTS, RANKED

Ranked by whether the result survives a control, a layer sweep and a seed change — not by size.

### 1. Pooling ten datasets for training beats single-source transfer — the only large effect with a tight interval
- **Tested:** train the probe on ten datasets at once versus on one source dataset, then transfer to
  a held-out target (LODO, 11 targets).
- **Number:** off-diagonal-mean AUROC gain **+0.1782** (layer 40, mean pooling, `paper` probe,
  seed 42, 11 datasets). Under the strict fix for the leak in §4.4 (leave-one-*cluster*-out) it is
  **+0.1254**; dropping only the two contaminated targets gives **+0.1653**. At other layers,
  as reported: L20 +0.1705, L32 +0.1312.
- **Confidence: HIGH.** The only headline with five seeds (F-83: +0.173 to +0.198 at L40), the only
  one whose dataset-level interval is tight (**±0.035**, because the pairing is within target), and it
  survives every leak treatment. Two independent runs of the split agree to 0.004 (0.1611 vs 0.1653).
- **Publish as +0.125** (cluster holdout) with +0.178 as the optimistic bound, not the other way round.
- Evidence: `audit/scripts/phase3/p3c_f108_reach.py`; `audit/cache/labpc/run-premise/premise_L*_mean_s42.json`;
  `.../clustered/clustered_L*_mean_s42.json`; `audit/claims.csv` F-79, F-80, F-83, F-108. VERIFIED.

### 2. Dataset identity dominates these activations; the deception label is a thin slice on top
- **Tested:** how well a classifier reads *which corpus a row came from* versus *whether the row is
  deceptive*, from the same activations.
- **Number:** dataset accuracy **0.9934**; label accuracy **0.6462**, i.e. **+0.032** over the
  majority baseline. Layer 40, mean-pooled evaluation activations, **3,624 rows across 10 datasets**
  (400 is the `--cap`, not a per-dataset count).
- **Confidence: HIGH** on the number (claim O-082, verdict HOLDS), with one caveat: the lift is
  **log-only** — `src/.../17_validate.py:126-131` never writes it to a file.
- **Why it ranks here:** it explains every other result in this digest. A method improving
  cross-dataset transfer is fighting a 99.3%-decodable nuisance variable with a 3.2% signal, which is
  why matched controls, random nulls and the prompt-side control all bite, and why dataset
  composition (5 vs 10) decides the sign of nearly everything.
- Evidence: `audit/phase3/UNFOLLOWED.md` N5; claim O-082. AUDITED/VERIFIED.

### 3. The capitulation bank — the one place a persona axis beats what it is matched against
- **Tested:** build a persona bank that varies *capitulation* (giving in to the user) instead of
  register, and score it on the sycophancy dataset where every existing axis fails.
- **Numbers:** zero-shot **0.745** AUROC at L40 on `sycophancy_mt`, against the paper16 **deception**
  axis's PC1 **0.374** (the paper16 *sycophancy* axis reads **0.313**) — F-65 PARTLY HOLDS.
  Trained, `pc:capitulation@2` beats raw by **+0.112, dataset-bootstrap CI [+0.050, +0.185]** — it
  clears (F-66 HOLDS). Seed check: 4 seeds, spread **0.0088**, an order of magnitude below the CI
  (F-87). Baseline integrity: all **1,512 of 1,512** baseline cells under the cap stores are
  byte-identical to the store they are scored against (max |diff| 0.000000).
- **Confidence: MEDIUM-HIGH**, and the limits are measured, not suspected:
  - over all **eleven** datasets the margin over raw is **+0.098, CI [−0.036, +0.208]** — does **not**
    clear (F-67);
  - the **format-matched control** (swap the six position-holding personas for six topically
    unrelated instructed personas) costs almost nothing (F-89 HOLDS) — much of the gain is *format*,
    not the construct;
  - the reported L40 is not the peak: the cap margin peaks at **L32 (+0.1387)** vs L40's +0.112;
  - the zero-shot **0.745 has no length control**, and length is the live objection on this exact
    construct (§2.6).
- **Write it as:** a construct-matched bank is a targeted instrument, not a general detector.
- Evidence: `audit/scripts/phase3/p3c_bank_fingerprint.py` (baseline section); `audit/claims.csv`
  F-65, F-66, F-67, F-87, F-89. VERIFIED.

### 4. Persona features are far better than random directions — the positive control the project passed and never quoted
- **Tested:** persona feature minus a random projection of the same dimension.
- **Numbers:** on the scale rungs (**5 datasets each**, all layers): 3B **+0.13…+0.24**, 8B
  **+0.09…+0.14**, 70B **+0.19…+0.23**; `axes` beats random at **21/21 layers** on both 70B banks.
  On the paper's own five datasets over 80 layers, `axes − random` = **+0.1922** (77/80 layers,
  mean pooling).
- **Confidence: HIGH for the five-dataset grids, and the scope must be stated with it** — the scale
  tables never say they are 5-dataset. On **ten** datasets with mean pooling the same quantity is
  **+0.0070, a coin flip across layers (44/80)**.
- **Why it matters:** it shows the failures against raw in §2 are not a wiring bug.
- Evidence: `p3d_unrecorded.txt` §S-35; `p3d_unrecorded2.py` §X1; `audit/phase2/layer_sensitivity.csv`
  (`n_datasets`). VERIFIED.

### 5. End-to-end reproducibility: two separate NDIF extractions agree to 1e-4
- **Tested:** the passC crossgen grid turns out to evaluate the same 200-row sample as passA, months
  of pipeline apart.
- **Number:** agreement within **0.0022 at worst over 1,512 matching cells**; mean |diff| **0.00016**;
  22 cells above 0.001; none above 0.01.
- **Confidence: HIGH.** This is the project's *only* end-to-end reproducibility number. One sentence
  in methods; the grid itself adds no new evidence (§4.9).
- Evidence: `p3d_orphans2.txt` §1. VERIFIED.

### 6. The scale story — the persona advantage shrinks with model size (corrected)
- **Tested:** best persona feature minus the matched dataset-PCA control, at 3B, 8B and 70B, over 19
  shared layers, **5 datasets** (*instructed-deception, mask, convincing-game, ai_liar, roleplaying*).
- **Numbers, after the RF-16 bank repair** (§5.2):

  | rung | best persona − matched control | same, vs raw |
  |---|---|---|
  | 3B | +0.1197 | +0.0328 |
  | 8B | +0.1027 | +0.0089 |
  | 70B as published (kd bank) | +0.0751 | +0.0079 |
  | **70B like-for-like (paper16 bank)** | **+0.0541** | **−0.0131** |

- **Confidence: MEDIUM.** The decline is real and gets *steeper* after repair (+0.120 → +0.103 →
  +0.054). But every rung is **one seed**, and the measured seed-to-seed range on this store is
  **0.059–0.070 per layer, up to 0.16** — larger than the 3B→8B step (0.017). The vs-raw column
  reverses sign after repair: at 70B the persona features are **worse** than the raw activations.
- Evidence: `audit/fix/RF16_NOTE.md`; `audit/fix/rf16_recompute_mean.txt`; `audit/claims.csv`
  F-96, F-100, F-110. VERIFIED (the recompute reproduces F-100 exactly when given the old bank).

---

## 2 · NEGATIVE RESULTS WORTH PUBLISHING

These are the project's strongest statements. Ordered by what each can carry.

### 2.1 Nothing clears significance at 11 datasets — the census
Across **65,388 dataset-level tests** in the ten cached crossgen `significance.json` files:
median 95% interval **±0.125 AUROC**, 90th percentile ±0.186; only **13.9%** of tests have an effect
larger than their own half-width; **166 (0.25%) clear BH q < 0.05**. Of those 166:

| what clears | count |
|---|---|
| persona feature beats **random** (noise floor) | 122 |
| persona feature beats **datapca** (matched control) | 6 |
| persona feature beats **raw** | **4** |
| a *control* losing to raw (negative controls firing correctly) | 33 |
| a persona feature *losing* | 1 |

There are **16,548** persona-vs-raw tests; **17.6% even have a positive sign**; the four that clear
are all `pc:kd_assertion` (the project's own knowledge-denial axis), three of them under `lr_reg10`
rather than the paper's probe. **Restricted to the reported operating points** (layers 20/40/44,
`paper` probe, mean pooling): 276 persona-minus-raw tests, 95 positive (34.4%), **0 clear BH**;
median effect 0.044 against median half-width 0.133. The same census on the smaller pre-wave git
copies (10,152 tests) gives 28 clearing, 1 persona-beats-raw — the picture does not depend on version.
**Scope (carry it):** crossgen files only; the cap, scale and premise significance files are lab-only
and **not** in the census, so §1.3's F-66 is counted separately from its claim row.
VERIFIED. `audit/scripts/phase3/p3c_ceiling.py`, `p3c_who_clears.py`.

**Honest framing for the paper:** not "persona coordinates beat raw activations", but "persona
coordinates are not distinguishable from raw activations at 11 datasets, and the one place they
clearly win is a bank we wrote ourselves for a failure we had already seen."

### 2.2 On ten datasets the persona method loses to raw nearly everywhere — and a 1-d length feature does better
Seed 42, 10 datasets, 80 layers. Layers where each feature beats raw:

| feature | mean/paper | mean/mms | last/paper | last/mms | mean margin vs raw (mean, paper) |
|---|---|---|---|---|---|
| `axes` | 3/80 | 2/80 | 3/80 | 5/80 | **−0.1004** |
| `pc:unified` | 3/80 | 2/80 | 4/80 | 2/80 | −0.0430 |
| `pc:deception` | 1/80 | 0/80 | 3/80 | 6/80 | −0.0828 |
| `length` (1-d control) | 7/80 | 8/80 | 13/80 | 14/80 | −0.0747 |
| `random` | 0/80 | 0/80 | 6/80 | 6/80 | −0.1074 |

A one-dimensional response-length feature beats raw at more layers than any persona feature, in all
six (pooling × probe) cells. A random direction's shortfall (−0.1074) is within 0.007 of the persona
axes' (−0.1004). And the composition effect, sharply: `axes − random` is **+0.0070 on 10 datasets
(mean pooling, 44/80 layers)** but **+0.1922 on the paper's 5 (77/80)**. *The method's value is not in
the personas; it is in the personas matching the datasets.*
VERIFIED at seed 42; MEDIUM on generality (only seed 42 has the full feature set; where seeds exist
the `axes − raw` margin **changes sign** across seeds at L44: s1 −0.0234, s2 +0.0156).
`p3d_orphans.py` §B/§H; `p3d_unrecorded2.py` §X1.

### 2.3 Zero of 4,032 kd-bank tests show a persona feature beating raw or datapca
The bank's largest effect is `pc:unified@2 − random@2 = +0.2861` (79/110 pairs won, CI
[+0.147, +0.427], q_BH 0.067) — **but the comparator is a random direction**. The eight largest
effects are all versus `random`. Of the six tests clearing q < 0.05, **four are `random − raw`**
(negative controls) and two are `pc:unified − random`. VERIFIED, `p3d_unrecorded.py` §S-32.
**Do not quote the q = 0.067 near-miss as evidence for the method.**

### 2.4 The prompt-side control: a probe that never sees the model's answer transfers about as well
The project designed this control and states its own reading in code
(`src/assistant_axis_ndif/8_coords_ndif.py:270-278`): if a prompt-side probe transfers as well as a
response-side one, the matrix is detecting the instructed condition, not anything the model wrote.
**It does.** Seed 42, 10 datasets, raw probe:

| setting | prompt-side | response-side | ratio above chance |
|---|---|---|---|
| L40 last, paper | **0.6475** | 0.6359 | **1.085** |
| L44 last, paper | 0.6219 | 0.6268 | 0.962 |
| L40 mean, paper | 0.5687 | 0.6652 | 0.416 |
| all 18 (layer × pooling × probe) cells | mean 0.5812 | mean 0.6469 | mean 0.562 |

At **last** pooling across the two reporting layers, six cells average **0.86 of the response-side
signal above chance**. Per dataset at L40/last/paper: `convincing-game` and `sandbagging_v2` have
prompt-side **in-distribution AUROC 1.0000** — those two columns of every transfer matrix measure the
experimental condition, not behaviour. `sycophancy_mt` transfers at **0.758 prompt-side against 0.516
response-side**. The one dataset where the answer genuinely matters is **MASK** (0.54 prompt vs 0.65
response; lowest prompt-side in-distribution, 0.77) — which is what MASK was built for.
**State the pooling dependence:** at mean pooling prompt-side is ~0.10 lower, so this is a
last-pooling result. VERIFIED. **Nobody has written a number from this anywhere** — see §3.1.

### 2.5 Sarcasm: two pre-registered gates, both failed
- **Gate 2:** unmarked ironic vs sycophantic responses are separable at **0.9213** by bag-of-words
  alone (95% CI [0.872, 0.967]; P(≤ the pre-registered 0.65 ceiling) = 0.0000) — any "sarcasm"
  direction may be reading vocabulary. F-93 HOLDS, F-105.
- **Gate 3:** on natural sarcasm the **deception** axis is inside a random-direction null in all four
  (layer × pooling) cells, **0.5239–0.5856**, while the **sycophancy** axis fires at
  **0.6763–0.7097**. F-107 HOLDS.
- v1 was confounded (the contrast varied sincerity *and* plan quality) and review caught it. The bank
  remains unbuilt; the register diagnosis is supported on two constructs (HP-KR, sycophancy) and
  **untested on the third** — do not overstate.

### 2.6 On real sycophancy, every persona direction is inside the random null, and counting characters beats 18 of 20 of them
Zero-shot (projection only, no probe), 5 directions × 2 banks × 2 poolings, n = 1,182 rows
(434 retracted / 748 maintained), with the file's own 1,000-direction null:
- **2 of 20 series** beat the null at their best layer — both the paper16 **assistant** direction
  (0.7263 mean / 0.7046 last vs best-of-layers p95 null 0.6935 / 0.6833). Every sycophancy, deception,
  unified and syc_PC1 direction in both banks is **inside** the null.
- A **length-only scorer reads 0.6640**; **18 of the 20 series never reach it**.
- Under length control the two survivors fall to at-or-below length-only: L24 0.7263 → 0.6719
  (quintiles) / 0.6545 (regressed); L40 0.7102 → 0.6601 / 0.6453; L44 0.7152 → 0.6510 / 0.6469.
  **28 of 60** controlled cells move by more than 0.05.
VERIFIED, single file `run-coords/q10_sycophancy_axis.json`. This is the strongest available statement
of the register diagnosis — and it is the *argument for* the capitulation bank, not against the project.

### 2.7 Unified PC1 is "which question set the persona answered", not a construct
Unified PC1 (47% of variance at L40) correlates **−0.945** with the question bank and only **+0.230**
with harmful-vs-harmless; **PC2** is the harmfulness axis (+0.816). Same at L20 (−0.929) and L60
(−0.943). **Scope:** it damages the **PCA basis only** — contrast directions barely move (cross-bank
cosine +0.984), so `contrast:unified` and all deception/sycophancy numbers are intact. It contaminates
`pc:unified`, `pc:unified@2` and the `PC1:unified` zero-shot scorer, and **the default crossgen bank
is still the confounded one** (`27_crossgen.py:856`). **CONFLICT:** the memory note says cos −0.98;
the audit measures **−0.945** — use the measured value. This is a reproducible failure mode of
contrastive-persona PCA that the replicated paper avoids by using a balanced shared question set, so
it is evidence that that design choice is load-bearing.

### 2.8 Three smaller negatives
- **Layer 40 is not the peak, and the reported gain is mostly raw being in a trough there.** Raw
  transfer decreases monotonically over 36–48 (L36 0.6878 → L40 0.6652 → L47 0.6537, mean/paper);
  L40 ranks **5th of 13** in the fine window; the 80-layer grid puts the peak at **L31–L33** in 5 of 6
  (pooling × probe) cells (0.7072 at L33 vs 0.6760 at L40). `axes − raw` is **−0.0336 at L40** but
  **−0.1811 at L31** (mean/mms), in all 18 store × pooling × probe cells. VERIFIED for the profiles;
  the owner's statement that "L40 was picked because it maximised the persona advantage" is DOC-ONLY.
- **k = 5 is not the paper's k, and the result is k-dependent.** On the paper's five at L40,
  `pc:unified@2` beats its matched control by **+0.138** (k ≈ 2) while `pc:unified` at k = 5 gives
  **+0.004**. F-39, F-113.
- **Growing the HP-KR matched slice failed its pre-registered gate.** Denial rate **0.250** (harmful
  template) / **0.150** (harmless control) against a ≥0.70 gate and a 0.40 kill line — caps the slice
  at 258 rows. The reason is itself interesting: **stated knowledge tracks confidence, not
  correctness** — the model asserts answers it does not have (F-59, LEDGER).

**Honest counterweight the write-up needs:** on the paper's five datasets `pc:unified@2` *does* beat
its matched control at **55–76 of 80 layers**. The matched control kills the claim on the broad set,
not on the paper's own set. Say both.

---

## 3 · STRONG OR SURPRISING RESULTS NOBODY FOLLOWED UP

| # | Result (key number) | Follow-up | Cost |
|---|---|---|---|
| 3.1 | **passP prompt-side control** (§2.4). Largest unreported result in the project; announced as running in `docs/thesis/persona-coordinates-70b-results.md:1428` and never written up. Grid finished 2026-09-19…09-22 (`audit/experiments.csv` O11, O46). | Write it as the construct-validity section, with the per-dataset table; add a per-dataset significance test vs the root store; put a "prompt-side floor" column beside every headline transfer number. | Writing only for the table; one read-only lab script, minutes, for the significance test. |
| 3.2 | **Zero-shot axis wins at the deployment threshold.** At every one of six layers the best method by AUROC is not the best at 1% false-positive rate. L28: untrained `zeroshot_axis/max` catches **69.3%** of lies at 1% FPR vs the best trained probe's **60.5%**; L40: **62.2% vs 47.3%**. LODO over 9 held-out datasets, 2,000 rows, 16 tokens/row, Alpaca as false-positive control. Committed: `run-coords/q8_trained_pertoken.json`. **CONFLICT:** the session's "TPR 0.717" is a *smoke* run; the committed full run gives 0.494 at L44 — quote the committed file. | This is the persona method's **best honest positive claim**: weak when trained, competitive when used untrained at a strict threshold. Report TPR@1%FPR beside every AUROC. | Writing only; the file is in git. |
| 3.3 | **passA at seeds 0–4 with the full feature list.** Only seed 42 carries all 11 features and 3 probes, so every 80-layer win-count in §2.2 is single-split. | Re-run for error bars — **not** to rescue the result: the margin is −0.10 with a seed sd of ~0.007–0.015; more seeds will sharpen the negative. | CPU-only on the lab PC, inputs on disk, ~5 hours. |
| 3.4 | **S-30 — F-86's +0.0284 margin sits inside its comparator's own 0.0824 subsampling spread.** The script printed "INSIDE the subsampling spread" three times; the ledger softened it to ±0.02 sd. DOC-ONLY (lab-only `run-premise`). | **High priority:** turns a reported small positive into a null. Re-run the premise script read-only. | Minutes, lab PC. |
| 3.5 | **S-26 — the draft's headline kd bank fails the geometric criterion its own reference sets.** cos(PC1, assistant) min **+0.539** / median **+0.598**, against the Assistant-Axis paper's >0.60-all-layers and >0.71-mid-layer thresholds; PC1 = 25.6% of role variance, top-5 69.2%. DOC-ONLY (lab `logs/kd_axes.log`). | Verify from the bank, then state it — it is a clean self-criticism of the project's own headline bank. | One read-only lab read of `run-kd/axes_vectors_s3.pt`, minutes. |
| 3.6 | **The cap bank's 0.745 has no length control** — and §2.6 shows length is the live objection on exactly this construct. | Apply the two `q10` length controls (score within length quintiles; length regressed out) to the cap zero-shot number. | Reuses `q10` tooling, ~1 hour. |
| 3.7 | **k = 1 re-extraction and the unfiltered bank.** `run-paper16/axes_vectors_all.pt` was built for the unfiltered arm and has **never** been through crossgen (a consequence of the DF-48 typo, §4.6). | Closes three caveats at once: "is this the paper's method" (k), the judge-filter asymmetry in the three-rung comparison, and §2.8's k-dependence. | CPU-only, rollouts already on disk. |
| 3.8 | **Axis norms are not comparable across axes or layers.** ‖sycophancy‖/‖deception‖ = 1.316…1.588, mean 1.510, **above 1 at 80 of 80 layers**; absolute norms grow ~2.4× from L20 to L44 (assistant 1.012 → 2.931). AUROC is rank-based and unaffected; any magnitude comparison, sum, average or shared threshold is not. | One line in methods. | None — measured. |
| 3.9 | **S-23 — the project's central negative conclusion was stated with its table on day one (2026-09-15)**, and the table reproduces *exactly* from `run-coords/probes_L20.json` (`pc:deception` 0.8146 / `axes` 0.7811 / `raw` 0.7678 for the paper probe; L20, mean, k=5, 12 off-diagonal pairs). Note it is also a composition effect: on *that* 4-dataset-ish set the persona features **win**. | Narrative value for the thesis: dates the main conclusion five days before F-39 said it with more data. | None. |
| 3.10 | **More datasets (11 → 14)** — already logged as an open decision. | The **only** fix that attacks §2.1's binding constraint. More rows, seeds or layers will not help. | New data generation; owner's call. |

---

## 4 · WEAK SPOTS AND BOTTLENECKS — ranked by referee damage

**4.1 The design cannot resolve its own headline (highest damage).** §2.1. Every claim of the form
"persona coordinates transfer better than raw activations" — the replication's central proposition,
draft contributions 1 and 4, F-17, F-29, F-31, F-61, F-96, F-100 — is *consistent with* a real effect
and *indistinguishable from zero* at 11 datasets. A referee who runs the numbers finds 0 of 276 tests
clearing at the reported operating points. Fix: reframe (done) plus more datasets.

**4.2 Layer 40 flatters persona features specifically.** Ranking each reported layer's `feature − raw`
gap among all that feature's stored layers (seed 42, mean pooling, paper probe; 1.0 = worst layer):
L40 lands in the top third for **62% of persona readings (96 cells)** but only **22% of control
readings (90 cells)**. Concretely, on the 11-dataset `kd11` store `axes − raw` reads **−0.0215 at L40,
rank 2 of 21**, against a layer-mean of **−0.0733** — quoting L40 is worth **0.052** of margin in the
favourable direction. `pc:deception` at L40 is rank 1 of 21. On the paper's five the same effect moves
to L20 (`pc:deception` +0.0609, rank 1 of 21, vs layer-mean −0.0450). The draft's contribution 4 is
explicitly "layers 20, 40 and 44" (`audit/phase0/draft_text.txt:97`).
VERIFIED arithmetic (78,516 rows of `audit/phase2/layer_sensitivity.csv`); **MEDIUM on
interpretation** — cells are heavily correlated, no p-value quoted, and persona vs control features
differ in dimensionality (1–4 vs 2). *Flagged by P3-C as one of its two least-certain findings.*
Fix: report the layer-mean, the range and the quoted layer's rank beside every single-layer number —
which `docs/LAYER_PREREGISTRATION.md` already exists to enforce. CPU-only, data on disk.

**4.3 Nearly everything is one random split, and the split moves more than the margins.**
Of 18,532 rows using the `paper` probe, **18,460 are seed 42**; the only 72 exceptions are the cap
seed check. **16 of 22 result stores have exactly one seed**, including `kd11` (behind
F-03/F-04/F-13/F-17/F-29/F-31/F-61/F-74) and **all four scale rungs**. Where seeds exist, the
seed-to-seed **range** of `feature − raw` averages **0.033** for `axes` (p90 0.084, **max 0.168**) and
0.021 for `pc:deception`; on the root store's five-dataset set it averages **0.059 (mms) to 0.070
(lr_reg10)** per layer, reaching **0.16**. For comparison: the 70B rung's best−raw is +0.008 published
/ −0.013 repaired; F-96's 3B→8B step is 0.017; the median persona-minus-raw effect is 0.044.
**All inside the seed range.** Not threatened: the pooled-training result (5 seeds) and the cap result
(4 seeds, spread 0.0088).
**Clean finding worth stating:** seed 42 was **not** cherry-picked — mean rank **3.66 of 6** (chance
3.5), best seed **8.1%** of the time (chance 16.7%). The problem is variance, not selection.
Fix: re-run the `paper` probe at seeds 0–4 on `kd11` and the four scale rungs. CPU-only.

**4.4 Train/test leak between two datasets (measured and bounded).** `harm-pressure-kr-matched` is a
filtered *view* of `harm-pressure-knowledge-report` — all 258 of its rows are HP-KR rows — so two of
eleven LODO cells train on their own test data. Priced: **−0.013** at L40 if you drop the two
contaminated targets, **−0.056** under the stricter cluster holdout (an upper bound: that also removes
three legitimate same-provenance datasets). Over 21 layers the change runs −0.018 to +0.035, and is
*positive* (the leak was suppressing the gain) at layers 4–28.
**Two things to get right when writing it:** (a) the code comment overstates it —
`src/assistant_axis_ndif/30g_clustered_holdout.py:12-14` says "every one of its 258 rows is in the
training pool", but the pool holds only HP-KR's 80% train split, so the real contamination is
**42 of kr-matched's 52 test rows (80.8%)**; (b) the leak **does not touch** the six-dataset
comparison against the collaborator's abstract (F-85/F-109) — `harm-pressure-kr-matched` is not one of
its six. VERIFIED.

**4.5 Ten headline numbers were computed in a chat session with no script and no output file.**
Experiment row `AN5-inline-no-script` carries **F-96, F-98–F-104, F-107, F-108**. More broadly,
**42 of 109 ledger entries write no output file at all** (GH-05). Phase 2 re-derived all ten, so
nothing is currently unsupported — but the practice is the project's single largest reproducibility
risk and the audit does not fix the *next* one.

**4.6 Live code defects (no number moves; reproducibility damage).**
- **Two unnamed truncated exception handlers in the judge**: `3_judge_ndif.py:168` and `:188` still
  use `str(e)[:200]`, and are **named nowhere in the ledger**. The judge decides which rollouts feed
  each persona vector, so a silent judge failure drops rollouts without a recoverable reason — the
  same failure mode that cost the 3B extraction 22 roles (F-33) and the 8B run entirely (F-22). A
  third known site survives at `1_generate_ndif.py:186`, forty lines below its own fix. Five-minute fix.
- **DF-48**: `scale/scripts/run_scale_chain.ps1:197-203` loops over both vector variants but writes
  both to the same path, so the second silently overwrites the first. Consequence: the surviving bank
  is the judge-filtered one, which is the mechanical cause of the three-rung comparison's filter
  asymmetry, and the unfiltered bank was paid for and never used. **Checked: not repeated** in the
  four sibling chain scripts.
- **Stale drift guard**: `scale/scripts/run_scale_coords.ps1:155-164` carries a "loud warning" about a
  state that no longer exists; the guard can never fire.

**4.7 Bank provenance cannot be recovered from a result file (structural, still live).**
`src/assistant_axis_ndif/27_crossgen.py:612` writes `"bank": Path(args.bank).name` — the folder is
discarded — and **three separate chain scripts each emit a file called `axes_vectors_s3.pt`** into
their own folder. This produced RF-16 (§5.2). `audit/phase2/provenance.csv` flags **six** stores with
an ambiguous bank basename; a cross-store fingerprint shows the **other five are clean** (each differs
from the paper16 stores in all 1,008 compared cells). One-line fix: record the resolved path.
Caveat on the clean verdict (MEDIUM-HIGH): the fingerprint proves a store is *not* one of the other
banks present; it cannot prove it *is* the bank its label names.

**4.8 A trap for anyone using `audit/phase2/layer_sensitivity.csv`.** In passA, passC and the root
store, **seeds 0–4 carry only three features (`raw`, `axes`, `pc:deception`) and two probes
(`mms`, `lr_reg10`)**; seed 42 carries all eleven features and all three probes. Any "mean over six
seeds" that mixes features silently compares a 6-seed mean with a 1-seed value.

**4.9 Orphans and duplicates that must not be quoted as evidence.**
- `kd11/banks/collab` (the collaborator's extended 12v13 contrast) and `kd11/banks/kd_s23` produce
  **identical numbers in all 504 matched cells** (max |diff| 0.000000), because the trained `axes` and
  `pc:deception` features are built from the PCA basis over the persona set and the two banks share
  that set. **No reported claim is affected** (F-98/F-99/F-103 read the zero-shot rows, which do
  depend on the contrast). Flag it so nobody later quotes it as evidence that the extended contrast helps.
- The passC pooled crossgen grid is a duplicate of passA (§1.5) — keep the per-token store, drop the grid.
- "All three cap variants agree" carries **no information**: `pc:capitulation` ≡ `pc:cap_vs_default`
  at 252/252 and 270/270 cells, and `cap_vs_default` / `cap_only` are the *same contrast*
  (`cap/scripts/cap_personas.py:238,244`). No pair is independent in both views.

**4.10 Small samples, imbalance, crashed runs.**
- `harm-pressure-choice` reaches the 3B/8B stores at **134 deceptive / 666 honest (5:1)** — the
  sampler took all 134 deceptive rows that exist. The dataset is **no longer in the scale analysis
  set**, so the imbalance sits in an arm nothing reports. One line.
- Of 114 catalogued runs: 106 complete, 1 superseded, 2 partial, 1 crashed, 4 unknown. All eight
  non-complete rows were checked; **only `O05-own100-fp32` threatens a reported claim** (§5.4).
- **DOC-ONLY reproducibility caveat:** the F-99/F-100 sweeps ran with **0.2–0.6 GB free RAM** against
  another account's jobs and were **resumed 5×**; neither F-entry records it.

**4.11 Data safety (not a referee risk, a loss risk).** `expLBactivationLAYER22` — 97 `.pt` files,
361 MB, lab-only, no producing script, no doc, no owner, **not backed up to Drive**. Provenance
NOT-FOUND. Also single-copy: `run-premise/`, `run-cap/`, `logs/` (RF-10). Back them up, then ignore
the layer-22 store as evidence.

---

## 5 · STALE FOUNDATIONS

### 5.1 The 09-22 boot-trigger wave — checked; the statistics did not move, one verdict flips
Three hours before the audit began, the lab PC rebooted and ~17 boot-triggered tasks re-ran,
rewriting about **58,000 result files** — after many citing findings had been written. **Twenty
findings** cite a regenerated artifact (F-04, F-11, F-12, F-17, F-19, F-20, F-31, F-34, F-57, F-58,
F-65, F-66, F-67, F-82, F-83, F-84, F-87, F-89, F-93, F-105) and **all twenty have a Phase-2 verdict**
— none was left unchecked. Keyed comparison of old vs new found **every effect size, confidence
interval and raw p-value identical**; only the **multiple-testing correction** moved, because the test
family grew (the root file went from **192 to 4,788 tests**).

**That flips exactly one verdict.** F-31 says "nothing beats raw activations" over a 4,032-test
family. On the regenerated **6,300**-test family it is **false**: `pc:kd_assertion − raw` at **layer
68, mean pooling, `paper` probe, 11 datasets** reads **+0.1497, q = 0.0467 — it clears.** Reproduced
independently from the cached lab file. Note the larger family is a strict superset with identical raw
statistics, so it is the *more* conservative correction and a test clearing there is the stronger result.
**Lesson for the write-up:** every "X clears / does not clear significance" sentence is true only of a
named test family, and this project's families changed size at least three times. Name the family.

### 5.2 RF-16 — the 70B rung used a different persona bank; repaired, and the number moves
The three-rung scale comparison built its 70B rung from the **kd** bank while 3B and 8B used
**paper16**, because two folders each hold a file named `axes_vectors_s3.pt` (§4.7). The audit re-ran
the owner's own `27_crossgen.py` with everything identical except the bank, into a new folder; the
corrected table is in §1.6. **The headline survives and gets steeper (+0.120 → +0.103 → +0.054), but
one published sentence reverses sign:** "against raw activations the 70B persona advantage is
essentially nil" (+0.008) becomes **−0.013**.
**Still stale, three different numbers in three places (CONFLICT):** F-110 carries the correction
(+0.054); F-100's own heading still reads +0.075 (`kd/docs/FINDINGS.md:4601`, the ledger's
append-only convention, and F-110 names what it supersedes); but **F-96 at `:4336` and `:4717` and
the memory note `no-margin-from-one-layer.md:22` carry a third figure, +0.079, and name nothing**.
A new session reading memory gets +0.079. **Good news:** the draft quotes none of these yet
(`audit/phase0/draft_text.txt` contains no "0.075", "0.120" or "0.054"), so the correction can be made
before it is written in.

### 5.3 F-58's inputs are gone, and its error is still live one entry away
F-58's 3B and 8B input matrices were overwritten 78 minutes later by a different five-dataset run
under the same filenames. **CONFIRMED NOT-RECOMPUTABLE** — no copy in git on either branch, none on
Drive. The ledger's figures (0.708 / 0.743, gains +0.089 / +0.086) match no surviving file; the files
give **0.7619 / 0.7980, gains +0.033 / +0.013**. F-58's 70B row verifies exactly. The entry was
revised to **F-63, whose cells reproduce to four decimal places** — so the conclusion moved to a
sound entry.
**Live residue:** F-58's dataset-identity error was corrected by F-60 *only against F-58*; the same
wrong sentence is still in **F-54**. And the project now uses **two different "fives"**: the scale
comparison's (*instructed-deception, mask, convincing-game, ai_liar, roleplaying*) and the draft's
("the five Liars' Bench datasets Llama-3.3-70B-Instruct generated without fine-tuning",
`audit/phase0/draft_text.txt:293`) — a different set. **Never write "the paper's five" without
listing them.**

### 5.4 The own100 bank was never rebuilt — the bank comparison is not like-for-like
The paper16 vectors were re-extracted in fp32 after a 16-bit reduction bug. The same re-extraction was
launched for the 100-persona "own100" bank and **died after 1 of 100 vectors** (`O05-own100-fp32`,
status partial; log ends on an NDIF progress line, no "Done."). So every paper16-vs-own100
comparison — F-90's persona arm and the memory note "persona set decides detection" — compares a
**repaired bank against an unrepaired one**. Size of the effect: **NOT MEASURABLE** without re-running
the extraction on NDIF. The only bound is weak and indirect: for paper16 the same repair cost nothing
(§5.6), but the banks differ in role count, rollout count and question set, and the bug was in a
masked reduction. VERIFIED file facts; INFERRED consequence; explicitly NOT MEASURED size.

### 5.5 The confounded PCA bank is still the default
`27_crossgen.py:856` still defaults to the bank whose unified PC1 is the question set (§2.7). Any new
crossgen run inherits the confound unless the bank is passed explicitly.

### 5.6 Checked and CLEAN — worth saying, because each was flagged HIGH earlier
- **paper16 old (bf16) vs fp32 vectors cannot reach any reported number.** Old-vs-new cosine
  **1.0000 at every one of 80 layers**, max relative length difference 0.5–0.9%, PCA components differ
  by ≤0.009. The only hazard is a PC sign pinned differently at **L23** (deception) and **L65**
  (unified) — and neither layer is stored in any reported grid (stored: 0,4,8,12,16,20,22,24,28,32,36,
  40,44,48,52,56,60,64,68,72,76) and **no claim mentions either layer** (regex scan, 0 hits).
- **The regenerated baselines did not drift.** All **1,512 of 1,512** baseline cells (`raw`, `datapca`,
  `datapca@2`, `random`, `random@2`, `length`) under each cap bank store are byte-identical to the
  `kd11` store they were scored against, max |diff| 0.000000. This removes a whole category of doubt
  from §1.3.
- **Seed 42 was not cherry-picked** (§4.3).
- **The leak does not touch the six-dataset comparison** (§4.4).
- **DF-48 is a one-off typo**, not a pattern (§4.6).
- **The five other ambiguous-bank stores are not repeats of RF-16** (§4.7).
- **S-10 provenance resolved:** the 2026-09-14 review's cosines (cos(syc, asst) 0.04–0.14;
  cos²(deception, assistant) = 0.44 at L40) describe the **own100** bank, not paper16 — the review
  predates paper16's build by a day. Measured on paper16: cos(assistant, sycophancy) **−0.2701** at
  L40, cos(deception, assistant) **−0.3705** (cos² 0.137, i.e. **86% orthogonal, not 56%**). INFERRED
  from dates; do not carry 0.04–0.14 into any paper16 statement.

---

## 6 · Two items P3-C nominated for the lead's own check
1. **§4.2, the layer-selection measurement.** The arithmetic is a census and is not in doubt; the
   *interpretation* (62% vs 22% shows selection) treats the control features as a fair comparison
   group, and they differ in dimensionality and construction. An alternative that cannot be excluded:
   L40 is where the persona subspace is genuinely best defined, making it a defensible operating point.
   Re-run: `audit/scripts/phase3/p3c_layer_rank.py`, `p3c_layer_seed_summary.py`.
2. **§2.1, the significance census.** Exact for the files read, but the corpus is the **ten cached
   crossgen files only** — the cap, scale and premise significance files are lab-only, so the arm
   holding the project's strongest positive result (F-66) is outside the census. Re-run over the lab
   copies before "4 of 16,548" appears in any write-up.
   Re-run: `audit/scripts/phase3/p3c_ceiling.py`, `p3c_who_clears.py`.
