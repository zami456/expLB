# REPORT — what we did, what holds, what to do next

_Audit of `expLB`, 2026-09-22 to 2026-09-23. Written for the owner._

> **UPDATE 2026-09-24 (P17).** The coverage and draft-verdict figures here are superseded at the P17
> pins (`audit/BRANCH_PINS.md`).
> - **Coverage:** **533 of 735 inventory claims have a verdict (72.5%)**. `audit/claims.csv` holds
>   643 rows, including 110 raised while checking.
> - **The inventory grew by 43:** 26 claims from the collaborator's three commits after ea1ce52b
>   (D2-001…026: 19 HOLDS / 3 PARTLY / 2 DOES NOT HOLD / 2 UNRESOLVED), and 17 ledger rows for
>   F-111…F-127 (11 HOLDS / 6 PARTLY).
> - **The 176 draft claims now read 165 MATCHES / 9 PARTLY / 2 UNRESOLVED.** D-189 moved to PARTLY:
>   CG raw L40 is 0.7873 at 16def2de, against 0.7875 at ea1ce52b.
> - **Still without a verdict: 202** — 110 ledger (6C) and 92 collaborator notes (6B).
> **Source:** `python audit/scripts/p15_coverage.py`; `audit/claims.csv` (verdict of D-189);
> `audit/p17/paper_new_claims.csv`; `audit/p17/A_REPORT.md` §1 — @ 16def2de / e917ecd3.

> **CORRECTION 2026-09-24 (DF-101, P17).** "Within ±0.06 at every layer" and "crosses zero five
> times" are both wrong. Excluding HP-KR, `ext_axis1 − ext_pc3` reaches **+0.0633 at L56**, and the
> series changes sign **6** times. Still correct: 11/16 positive, +0.0086 (L20), +0.0012 (L24), and
> negative at L64/68/72. No persona-arm cell changed between ea1ce52b and 16def2de.
> **Source:** `src/probes/deception-detection/results/persona_ood_ext_L{20..79}/summary.csv`,
> `off_diagonal_excl_hpkr` @ 16def2de; `audit/p17/ledger_verify_stdout.txt:223-224`.
> **How the claim is updated:** *"excluding HP-KR the margin is positive at 11 of 16 layers, lies
> between −0.033 and +0.063, and changes sign six times."*
> _The text below is superseded on this point and is kept for the record._

> **UPDATE 2026-09-24 (P17).** F-123…F-127 (Phase 6), F-128 (P17) and the collaborator's commits
> 2ad758fb / d1b88154 / 16def2de are also absent from this report. **Source:** `kd/docs/FINDINGS.md`;
> `audit/p17/A_REPORT.md`; `audit/BRANCH_PINS.md`.

> **THIS REPORT CLOSES AT 2026-09-23 (DF-95, DF-94). Two things happened after it was written.**
>
> **1. `origin/paper` landed on 2026-09-23 23:46** — 3 commits, 683 files, 11,555 insertions,
> including the collaborator's entire 16-layer sweep. It entered scope on 2026-09-24, *after* this
> report was drafted. **Three findings here are therefore missing:** **F-117** (the 16-layer sweep
> generalises D-028 — without HP-KR the Contributions margin is within ±0.06 at every depth and
> crosses zero five times), **F-119** (`paper/make_numbers.py` exists after all, superseding the
> audit's only DOES-NOT-MATCH verdict — a correction in the collaborator's favour), and **F-122**
> (the sycophancy axis *was* computed, on his branch; `syc_pc1` tops his excl-HP-KR table at L40–56).
>
> **2. Phase 6A ran on 2026-09-24 and changed the coverage figures this report quotes.**
> Wherever it says **"424 of 692 claims checked"**, read: **490 of 692 inventory claims given a
> verdict, plus 110 raised while checking.** Real inventory coverage is **71%**, not the 61% the
> old framing implies and not the 45% it actually was when this was written.
> **The draft PDF is now at 176 of 176** — 160 `MATCHES HIS DATA`, 9 `PARTLY`, 7 `UNRESOLVED`,
> **0 failures and not one arithmetic error found.** The 202 claims still without a verdict are
> exactly the two optional phases: 110 ledger entries and 92 collaborator-note claims, neither of
> which gates the paper.
>
> See `audit/COVERAGE_GAP.md`, `audit/phase6/RESULTS.md`, `audit/RESULTS_CHAPTER.md`, and
> `kd/docs/FINDINGS.md` F-123/F-124/F-125.
> _The text below is superseded on these points and is kept for the record._

Every number here came from a file opened or a script run during this audit. Paths are given so you
can check any of them. The technical companion is `audit/REPORT_TECHNICAL.md`.

**Two words used throughout.** **AUROC** — the chance that a randomly chosen lie scores above a
randomly chosen honest answer. 0.5 is a coin flip; 1.0 is perfect; **below 0.5 means the detector is
pointing backwards**, calling lies honest. **Persona axis** — a single direction through the model's
internal activity, extracted by having it role-play contrasting characters; the method's claim is that
scoring along a few such directions detects lies better than reading the raw internal state.

---

## The big picture, in ten sentences

1. You set out to test whether a small paper's method — lie detection in "persona coordinates" —
   survives at 70 billion parameters, 23× the model it was published on.
2. It does not: across **65,388 dataset-level significance tests**, only **4** show a persona feature
   beating an ordinary probe after correcting for multiple comparisons, and **none at the layer and
   pooling the project actually reports**.
3. The reason turns out to be interesting rather than boring — the persona axes encode **register**
   (confident assertion versus hedging) rather than deception, which is why they invert on the two
   datasets where a liar hedges and an honest speaker asserts.
4. You then ran the experiment nobody had run: pool ten datasets, train one ordinary probe, no
   personas at all — and it beat every persona result in the project.
5. That pooled result is **+0.125 AUROC** over single-dataset transfer once a data-leak is handled
   properly (**+0.178** before), and it is the only large effect in the project with a tight error bar
   and five seeds behind it.
6. Your collaborator's draft paper is built on the opposite conclusion, and its central dataset,
   HP-KR, turns out to be **~91–96% solvable by surface wording alone** — a label-free regular
   expression scores **0.9125**, at or above the draft's own headline of 0.919.
7. The draft states that it does not compare against a pooled probe; **that comparison exists on his
   own branch**, committed two days earlier under the message *"the fair baseline for the
   generalization claim"*.
8. Of **424 claims** checked against raw data, **297 hold, 119 hold only partly, 8 fail** — so the
   problem across this project is almost never a wrong number; it is a correct number wrapped in a
   sentence that claims more than it.
9. The audit measured its own reliability by re-deriving 161 claims blind: **its error rate is 6.5%**,
   and every one of its errors was leniency — nothing true was wrongly thrown out.
10. The work is publishable, but as a careful negative result with a strong positive control, not as
    the replication it started out as.

---

## The strongest results, ranked

Ranked by what survives a control, not by size. A big number that fails its control ranks below a
small one that passes.

### 1. Pooling beats personas — and this is your paper
**+0.1254 AUROC**, train on ten datasets at once versus train on one and transfer (layer 40, mean
pooling, `paper` probe, seed 42, 11 targets). Positive on **10 of 11** targets.

This is the only headline in the project with **five seeds** (+0.173 to +0.198 at layer 40) and a
tight dataset-level interval (**±0.035**). Two independent runs of the split agree to 0.004.

**Publish +0.125, not +0.178.** The larger figure comes from a leak: two targets
(`harm-pressure-knowledge-report` and `harm-pressure-kr-matched`) are near-duplicates of a training
set. Holding out whole *clusters* of related datasets instead of single datasets costs 0.053, and I
verified both numbers myself from the raw file — `summary.all.gain_lodo = 0.1814`,
`summary.all.gain_loco = 0.1254`.
*Evidence:* `audit/cache/labpc/run-premise/clustered/clustered_L40_mean_s42.json`,
`premise_L40_mean_s42.json`.

### 2. Why every control bites: the data is mostly "which dataset is this"
From the same activations, **which dataset a response came from is 99.3% decodable**, while the
lie/honest label adds only **+3.2 percentage points** over guessing the majority class (layer 40,
mean pooling, 3,624 rows, 10 datasets). Any method that reads a low-dimensional summary is largely
reading dataset identity. This explains, in one number, why the matched controls kill so many results.
*(The +3.2pp is log-only — it is in a console log, not a result file.)*

### 3. The control table — the most citable thing here
Against a **1,000-draw random-direction null**, per dataset:
- the deception axis beats the null on **5 of 9** datasets at layer 40/mean, **4 of 9** at layer
  44/last;
- it sits **inside** the null on HP-KR — **0.6436 against a p95 of 0.9377**;
- it is **inverted on `mask`** (0.4669, 0.4417 — both below 0.5);
- on the four sleeper-agent datasets it beats the null **0 of 4**: AUROCs of 0.87–0.91 look strong and
  sit inside nulls of 0.92–0.93, because those datasets are easy for *any* direction.

That last line is the single best argument in the project for why a headline number without a null is
uninterpretable. *Evidence:* `run-coords/zeroshot_nulls.json`, `audit/phase2/controls.csv`.

### 4. The capitulation bank — the one place a persona axis earns its keep
**+0.112 [+0.050, +0.185]** over a raw probe on its own column, clearing the dataset-level bootstrap,
with four seeds spread only 0.0088. It is a real, constructive confirmation of the register diagnosis.
**But** it does not generalise — on all eleven datasets it reads +0.098 with an interval spanning zero
[−0.036, +0.208] — and its format-matched control shows much of the gain is response *format* rather
than the capitulation construct. Report it as a mechanism demonstration, not a method.

### 5. Reproducibility you can claim
Two separate NDIF extractions agree to **1e-4**; the per-token probe was validated against scikit-learn
to **3.66e-06** with 0 mismatches on 12,000 rows. The second of these is in no document anywhere — it
is the only evidence that pipeline was ever validated, and it should be in the methods chapter.

---

## Negative results worth publishing

Your reframe to methods-and-negative-results is the right call. These are assets.

1. **Nothing clears significance at eleven datasets.** 65,388 tests; median 95% interval **±0.125**
   against a median effect of **0.044** — the study is underpowered by its own arithmetic. **4** of
   16,548 persona-beats-raw tests clear correction, and **0 of 276** at the operating points actually
   reported.
2. **A one-dimensional response-length feature beats a raw probe at more layers than any persona
   feature does**, on ten datasets. Detectors here are substantially reading how long the answer is.
3. **`axes − random` is +0.007 on ten datasets and +0.192 on the paper's five.** The method's apparent
   advantage is dataset composition, not personas.
4. **A probe that never sees the model's answer** — prompt only — recovers about **0.86** of the
   response-side signal above chance at last-token pooling; two datasets are perfectly separable from
   the prompt alone. This is the largest *unwritten* result in the project.
5. **On real sycophancy every persona direction is inside the random null, and counting characters
   beats 18 of 20 of them.** The sycophancy axis is **significantly inverted** (0.3131).
6. **Unified PC1 is "which question set the persona answered"**, not a construct — cosine **−0.98**
   with question family.
7. **Sarcasm: both pre-registered gates failed** and the bank was never built. Honest, and worth a
   paragraph. One caveat below.

---

## What failed or changed, and what it taught

- **The scale story moved three times and is now smaller than any document says.** After a bank
  mix-up was repaired, the 70B rung reads **+0.054** against the matched control and **−0.013** against
  a raw probe. Three different figures (+0.079 / +0.075 / +0.054) currently live in three places. The
  decline with scale survives; its size does not.
- **F-61's "twelvefold" was a software bug.** The figure "+0.006" came from a script reading a caption
  string (`'trained on (80% split)'`) as a list of dataset labels, so it compared single *characters*
  and left in-distribution cells in an "off-diagonal" average. I reproduced both code paths: the
  correct values are **110 pairs / +0.0093**, and the ratio is **8.1×**, not twelvefold. The bug is
  already fixed in the code; only the prose is wrong.
- **The audit corrected itself seven times.** Details below under "how much to trust this".

---

## Red flags and bottlenecks

Ranked by how much damage each would do if a referee found it first.

1. **The eleven-dataset ceiling.** Significance is computed across datasets, and you have eleven. The
   median interval is wider than almost every effect. No amount of extra rows fixes this; only more
   *datasets* would.
2. **Almost everything rests on seed 42.** 18,460 of 18,532 `paper`-probe rows are seed 42, while the
   observed seed-to-seed range reaches **0.168**. In three stores every control comparison exists at
   seed 42 only.
3. **Layer 40 flatters persona features specifically.** At the reported layer, persona features land
   in the top third 62% of the time against 22% for controls — worth about **0.052**. Choosing the
   layer is worth more than most of the effects being reported.
4. **The HP-KR leak is real.** 42 of 52 rows are genuine contamination, costing between −0.013 and
   −0.056 where it bites. It is the reason the pooled result drops from +0.178 to +0.125.
5. **The sarcasm gate was not pre-registered.** The 0.65 threshold first enters the repository in the
   commit that reports the failure, timestamped *after* the rollouts and judge files. The finding
   survives (0.921 and 0.947 fail any threshold), so restating costs nothing — but a pre-registration
   claim the repo's own history contradicts is the first thing an auditor finds.
6. **Data safety, not validity:** several result folders exist **only on the lab PC** with no backup —
   `run-premise/` (which holds your best result), `run-cap/`, `run-cap-ctrl/`, the logs, and ~133 loose
   analysis scripts. Also `run-scale/8b` is an **incomplete corpus** (5 of 12 dataset directories are
   empty), and its two axis builds write the same filename so one silently overwrote the other.

---

## What the draft gets wrong

Fairness first: **51 of 79 checked draft claims match his data**, only one outright fails, and the two
pipelines agree to a mean absolute gap of **0.0015** across 55 matched cells — there is no bug on
either side. The problems are framing and omission, not fabrication.

1. **"We do not compare against a pooled probe" — he ran that comparison himself.** `loo_L20/` and
   `loo_L40/` on his branch, commit `6b16a46`, 2026-09-20 20:05, message *"Leave-one-dataset-out: the
   fair baseline for the generalization claim"*, two days before the draft was compiled. Removing
   HP-KR costs his axis **0.048 to 0.130 AUROC** in all four (layer, weighting) cells, and in two of
   four what remains is **≤ +0.002 or negative**.

> **UPDATE 2026-09-24 (DF-105, P17).** The **0.9597** ("does not affirm knowledge") regex value below
> was **not re-derived**: the regex is recorded nowhere. The guess `NOT ^(yes|i know)` gives **0.9413**,
> still above the 0.919 headline, so the argument stands; 0.9125 is unaffected.
> **Source:** `audit/p17/correction_blocks_check.csv` (row for `docs/NOVELTY_CHECK_PROMPT.md:127`);
> `audit/p17/C_REPORT.md`.

2. **The central dataset is largely solvable from wording.** A label-free regex scores **0.9125**
   ("negated know") and **0.9597** ("does not affirm knowledge") on HP-KR — at or above his headline
   **0.919**. A length-only scorer reads **0.776** there, above the deception axis's own **0.6436**.
   The wording-controlled test he names as decisive was in fact run and is **inconclusive**
   (+0.066, CI [−0.069, +0.189]).
3. **The headline contribution is one dataset.** His "+0.220 at layer 20" collapses to **+0.009** once
   HP-KR is dropped as both training and test set — and his own Table 3 already shows this.
4. **"The persona PCs add nothing" fails at the paper's own setting.** The paper's deception figures
   are **k=1**; this project's default is k=5, which appears nowhere in the paper. At k=1 his own
   summary has PC1 beating the contrast at all three layers.
5. **One test set is the base probe's own training set** (`repe_honesty`). Drop it at layer 20 and the
   contrast falls *below* its baseline.
6. **Selective reporting across banks.** Three persona banks exist; the draft reports two. The third —
   which his own summary says makes every arm worse at every depth — is never mentioned.
7. **The reproducibility sentence names a script that does not exist** (`paper/make_numbers.py`) on
   any branch, in any commit, or on the lab PC.
8. **The error bars are too narrow.** Every interval covers example sampling only, while re-extracting
   a bank moved HP-KR by **0.036–0.077** — more than the intervals themselves.

**And a correction in his favour:** the audit's own ledger was wrong about his random control — it is
the mean of five draws, not a single direction.

---

## Promising leads nobody followed

- **The prompt-only probe** (#4 in the negatives). Largest unwritten result; the data already exists.
- **TPR at 1% false positives flips the method ranking** — an untrained axis reads 0.693 against 0.605
  for the best trained probe at layer 28. This is the method's best honest positive claim and it sits
  in a committed file, unquoted.
- **The decisive HP-KR experiment does NOT belong here — it is already done.** See the correction
  immediately below.

---

## CORRECTION — the decisive HP-KR experiment was already run, and it is negative

**I recommended running an experiment that exists. That was wrong, and it is the same fault this
report criticises elsewhere: a recommendation made without opening the file.** Corrected here rather
than quietly edited out.

**It was specified on 2026-09-19 and run.** In the ledger as **F-03**, **F-72** and **F-76**;
`kd/docs/FINDINGS.md` mentions `kr-matched` **22 times**. The draft mentions it **zero** times.

What the wording-matched slice actually says (`harm-pressure-kr-matched`, 258 rows, **both classes
deny knowledge**, so wording cannot separate them):

| test | result |
|---|---|
| your trained persona axes vs raw, layer 40 | 0.752 vs 0.715 |
| **same cell at layer 20** | **0.374** — 0.216 *below* raw |
| **dataset-level bootstrap at the pre-registered layer 40** | **+0.066, CI [−0.069, +0.189], P = 0.161 — does not clear** |
| which layer is best | 95% interval **[L5, L79]** — 93% of the network |
| label permutation | p = 0.38 |
| over all 80 layers | persona axes beat raw at **28 of 80**; dataset-PCA at **31 of 80** |
| **his** extended axis on the matched slice, zero-shot | **0.4630 (L20) / 0.6203 (L40) / 0.5383 (L44)** |
| his split knowledge-denial axis | **0.3932 / 0.4989 / 0.4825** — at or below chance |
| his 4-denial vs 3 **sincere-ignorance** (wording-matched) | 0.5923 / 0.6692 / 0.6659, against **0.8734 / 0.8826 / 0.8734** unmatched |

**So 0.21–0.28 AUROC of the draft's HP-KR result is the plain-assertion half of the contrast, not
knowledge denial at all.** Those zero-shot numbers on *his* axis are in **neither document**, and they
are the sharpest single item in the whole comparison.

**What this changes for you.**
- **Nothing needs running. It needs writing up** — an afternoon of your time, not compute.
- The honest sentence is the measured one: *"we ran a wording-matched slice; at the pre-registered
  layer the margin does not clear the dataset-level bootstrap, and the layer choice moves it by ten
  times the claimed effect."*
- **Do not let the draft keep promising this test as future work.** It is stated as unrun on p.12 and
  again on p.13.

**The one thing that genuinely is not built.** The current matched slice has a leak for *trained*
comparisons: its row ids are HP-KR row ids by construction, so **~81% of its test rows sit in HP-KR's
training split** (F-108). The zero-shot readings above are unaffected. A clean trained comparison would
need **newly generated rollouts on NDIF** — days of wall-clock and a GPU budget, **not** the cheap
CPU-only job I described. If you want that, it is a real experiment to plan, not a loose end to tidy.

---

## What the two new runs found (2026-09-23) — F-115 and F-116

**Both needed no new data and no GPU.** Run from an isolated copy; your originals were verified
byte-identical before and after.

**F-115 — the contamination is closed, and it was not what made the result.**
- The wording-matched slice shares **100% of its rows** with the hard dataset, so that one pair was
  train-on-test. Removing the shared rows leaves **59 of 258**.
- The headline barely moves: **−0.2566 → −0.2545** (L20), **+0.0367 → +0.0309** (L40),
  **+0.0134 → +0.0050** (L44). **The earlier conclusion stands, and the objection is now answered
  rather than outstanding.**
- But the individual affected cells move hugely — at layer 40 the persona axis drops to **0.4262,
  below chance**. Transfer *within* the knowledge-report family was substantially the shared rows.

**F-116 — the control nobody had run, and it is the strongest negative you have.**
- The slice function has **two** modes; every result so far used one. The unused one holds
  **knowledge** constant instead of wording, and it has **2,001 rows against 258**.
- Trained on it and tested elsewhere, the persona axes read **0.3850 / 0.3335 / 0.5267** —
  **below chance at two of three layers**, losing to an ordinary probe by **0.19 to 0.38**, the
  largest deficit anywhere in this project. **A length-only feature (0.6174) beats them at every
  layer.**
- On its own held-out data an ordinary probe scores **0.999**. So when only the denial varies, the
  signal is almost perfectly there — **and nothing learned from it transfers.**
- **This is the register diagnosis tested directly on 2,001 rows**, rather than inferred from the
  plain-assertion direction. It is the cleanest thing in the project for the negative-results paper.

**One confound I flagged after misreading it myself:** the zero-shot readings on this slice look
almost identical to the full dataset — but the slice *is* 93.5% of the full dataset, so they must
agree. That comparison proves nothing; only the transfer and in-distribution numbers count.

---

## NEW (2026-09-24): your collaborator's layer sweep, and what it does to the picture

He pushed a **new branch, `origin/paper`**, on 2026-09-23 at 23:46 — three commits, 683 files. It
adds **16 layers where the draft had 3**, plus a Gender-Secret experiment. Full review:
`audit/COLLAB_PAPER_BRANCH.md`; ledger F-117, F-118, F-119.

**It confirms the audit's central criticism, using his own new data.**

> **CORRECTION 2026-09-24 (DF-101, P17).** "Within ±0.06 at every layer" and "crosses zero five
> times" are both wrong. Excluding HP-KR, `ext_axis1 − ext_pc3` reaches **+0.0633 at L56**, and the
> series changes sign **6** times. Still correct: 11/16 positive, +0.0086 (L20), +0.0012 (L24), and
> negative at L64/68/72. No persona-arm cell changed between ea1ce52b and 16def2de.
> **Source:** `src/probes/deception-detection/results/persona_ood_ext_L{20..79}/summary.csv`,
> `off_diagonal_excl_hpkr` @ 16def2de; `audit/p17/ledger_verify_stdout.txt:223-224`.
> **How the claim is updated:** *"excluding HP-KR the margin is positive at 11 of 16 layers, lies
> between −0.033 and +0.063, and changes sign six times."*
> _The text below is superseded on this point and is kept for the record._

- The draft's headline Contributions margin, measured **without** the knowledge-report dataset, is
  **within ±0.06 at all 16 layers and crosses zero five times** (+0.0086 at L20, +0.0012 at L24).
  We found that at 3 layers; it now holds at 16.
- His claim that the 1-d contrast wins **"at every depth" is falsified by his own sweep** — it is
  **negative at layers 64, 68 and 72**.

**But one result runs strongly in his favour, and it is new.** Against his own baseline the extended
axis is **positive at 16 of 16 layers (+0.115 to +0.273)** — **and still 16 of 16 with the
knowledge-report dataset excluded** (+0.123 to +0.243). That is depth-robust and genuinely not
HP-KR-driven. It was invisible from three layers. **It belongs in the paper.** The qualification: his
baseline is a probe trained on one dataset, and the pooled probe (D-158) remains the fair comparator.

**And he independently confirmed your register diagnosis.** His Gender-Secret run uses a dataset
where **both classes deny knowledge** — the benchmark's own wording-matched test. **Every persona
direction reads at chance** (0.436–0.623). That now converges from three directions:

> **CORRECTION 2026-09-24 (DF-100, P17).** The three values 0.4630 / 0.6203 / 0.5383 are sourced
> wrongly: they are at `audit/HPKR_WORDING_CONTROLLED.md` §5 (:121), not in F-115, and reproduce from
> the collab-bank kd11 matrices. The convergence finding is unchanged.
> **Source:** `audit/p17/C_REPORT.md` (F-118 row); `kd/docs/FINDINGS.md` F-128.
> _The text below is superseded on this point and is kept for the record._

| route | result |
|---|---|
| his axes on our wording-matched slice (F-115) | 0.4630 / 0.6203 / 0.5383 |
| our `knowledge_matched` control, 2,001 rows (F-116) | transfer **below chance** |
| **his Gender-Secret (F-118)** | **every direction at chance** |

**Three independent wording-matched tests, one answer.** Present it as a single converging finding —
it is the strongest negative result either of you has.

> **CORRECTION 2026-09-24 (DF-102, P17).** The raw/axes/random values below (raw 0.7834 → 0.7426, random +0.127 from L72 to L79, axes − random +0.109 at the top vs +0.230 at L40) are the
> **5-dataset (paper5) reading** of `passA`, and the text does not say so. On all 10 datasets, raw at
> L40 is **0.6760**. The shape of the argument (raw declines, random gains) is unaffected.
> **Source:** `audit/p17/C_REPORT.md` F-120 row; `audit/p17/fmap_F110_F127.csv` (LAB-ONLY: `passA`
> matrices); `kd/docs/FINDINGS.md` F-128. _Kept for the record below._

**The one thing I flagged as unexplained is now solved.** Many arms rose sharply near the top of the
network, which looked like the method working at depth. It is not. Asked of *our* 80-layer data — a
pipeline he never touched — the same shape appears, and three facts explain it: the full 8192-dimension
probe **declines** there (0.7834 → 0.7426), the persona axes only **recover** toward their mid-stack
level without reaching it, and **a random direction gains just as much** (+0.127 from L72 to L79). A
random direction cannot learn anything, so this is a geometry change that flatters *any* low-dimensional
projection. **Measured against that control the persona axes are worse at the top (+0.109) than at layer
40 (+0.230).** Ready-made answer for a reviewer in `audit/COLLAB_PAPER_BRANCH.md` §2 and ledger F-120.

**A full record of his contributions** — everything he found, where the audit agrees, where it
disagrees, and what he should be credited with — is now `audit/COLLABORATOR_FINDINGS.md`.

> **UPDATE 2026-09-24 (DF-107, P17).** Resolved by the collaborator at 2ad758fb, in his favour: the
> paper now reports the **L36 optimum** (`results.tex:228-230`, "Ranking peaks at
> layer~\V{depth-off-ext-peak-layer}, not at layer~40"; numbers.tex `depth-off-ext-peak-layer`=36,
> `depth-off-ext-peak`=0.927), and the "outscores … at every depth" claim is gone — `results.tex:136`
> now counts wins over the sweep (`depth-pc-wins-complement`=13 of `depth-n`=16). The one remaining
> "every depth" (`results.tex:208`) is a different claim, about the base probe inverting.
> **Source:** `src/probes/deception-detection/paper/sections/results.tex` and
> `paper/generated/numbers.tex` @ 16def2de; `audit/p17/A_REPORT.md` §2.

**Two smaller things:** the optimum layer is **36, not 40** (0.9275 vs 0.9158) — the draft
under-reports its own best; and **`paper/make_numbers.py` now exists**, which withdraws the audit's
only outright "does not match" verdict. That is a correction in his favour and I have recorded it
as one.

---

## RESOLVED 2026-09-24: does your bank's advantage hold at scale?

You spotted the gap yourself: the published 26-persona bank had been run at 3B, 8B and 70B, but
**your 36-persona bank existed only at 70B** — so "our bank rescues the method at scale" could not be
told apart from "our bank is better at any scale". Both rungs have now been run. Ledger **F-121**.

Same 5 datasets, same k, same pooling, same seed, layer-averaged — every cell on identical footing:

| rung | published bank (26 personas) | **your bank (36 personas)** |
|---|---:|---:|
| 3B | +0.0382 | +0.0360 |
| 8B | +0.0148 | +0.0194 |
| **70B** | **−0.0116** | **+0.0054** |

**What you can now say, in order of confidence:**

1. **The decline belongs to the method, not the bank.** Both fall as the model grows. Ten
   purpose-built personas do not stop it. **This makes your negative result stronger**, because it
   closes the obvious objection — "you just used the wrong personas".
2. **Your bank declines more slowly, and the gap widens with scale** — behind at 3B, level at 8B,
   ahead by +0.017 at 70B, where the published bank goes negative and yours stays just positive.
3. **But +0.005 is not a result.** One seed, no significance test. Say the direction, not the number.

**And the scope condition to state alongside it:** these five datasets contain **no knowledge-denial
data**. Your bank was built to repair exactly that, so this measures its *general* effect, not its
intended one — consistent with everything else we found: the bank's advantage is real but local to
the dataset it was designed for.

**Cost:** two NDIF rungs, 10,800 rollouts, ~9 hours, finished 08:51. Backed up to Drive, 290 files,
0 differences. The lab PC powered off afterwards; nothing was lost.

---

## How much to trust this audit

I measured it. 161 claims were re-derived by agents that could not see any prior verdict, and I
adjudicated every disagreement from raw files.

- **Raw disagreement 30.8%; actual audit error 6.5%.** Agent and audit disagreed about a *number* once
  in 161 — and there the agent was right.
- **All seven errors were leniency**: claims graded "holds" that should have been "partly". **No true
  result was wrongly retracted** (8 of 8 failure verdicts upheld).
- The error concentrates in one tier: **15% of high-importance "holds"**, 0% of "partly", 0% of
  failures.
- The cause was the same every time, on both sides: **the verdict did not match the checker's own
  notes**. One entry literally opens "AUDIT FLAG:" and was graded clean.
- **Known weaknesses of that measurement**, all of which make 6.5% an *under*-estimate: one agent's
  search strayed into audit files and self-reported it; 11 of 14 batches ran before I closed that hole;
  and I adjudicated alone, being the person whose work was measured.

A screen across all 424 claims found **69 clean verdicts whose own notes describe a defect**. Six were
confirmed as real errors and fixed. **You chose to report the other 63 rather than re-grade them** —
correct, since re-grading on a text match would repeat the very fault being measured.

---

## Decisions for you

| # | Decision | My recommendation |
|---|---|---|
| 1 | **What is the paper?** | The negative result **plus** the pooled-probe positive control. You have a clean story: the method does not survive scale, here is why (register, dataset identity), and here is the simple thing that works better. Lead with +0.125. |
| 2 | **What to do about the draft** | Send your collaborator the leave-one-dataset-out point first. It is *his own committed work*, under a commit message that says what it is for — this reads as something lost track of, not hidden. Raise HP-KR's wording solvability second; it is the one that decides whether the paper's central claim survives. |
| 3 | **Run the decisive HP-KR experiment?** | **Done, 2026-09-23 — twice, and both are now in the ledger as F-115 and F-116.** The leak is closed (it did not account for the result) and the bigger, never-run control slice turns out to be the project's strongest negative. Nothing further is needed here; see the section below. |
| 4 | **Back up the lab PC** | **Urgent and unglamorous.** `run-premise/` holds your best result and exists in one place. |
| 5 | **The 63 flagged claims** | Already decided: report, do not re-grade. Re-deriving them is the highest-value follow-up if you ever want the ledger airtight — the comparable sample produced the single biggest finding of this audit. |
| 6 | **Apply the 75 document fixes?** | Do the **P1** ones before anything is sent out. `audit/DOC_FIXES.md` lists them; nothing has been applied. |

---

## Glossary

- **AUROC** — probability a random lie outranks a random honest answer. 0.5 = chance, <0.5 = inverted.
- **Persona axis / bank** — a direction extracted from role-play contrasts; a *bank* is one set of role
  instructions, run and turned into axes. Seven banks exist here.
- **Probe** — a small classifier trained on the model's internal activity.
- **Off-diagonal mean** — train on one dataset, test on the others, average. The transfer score.
- **LODO / LOCO** — leave-one-**dataset**-out / leave-one-**cluster**-out. LOCO is the honest one when
  datasets overlap.
- **Pooling** — how many token positions are summarised: `mean` over the response, or the `last` token.
- **Layer** — depth in the network. Llama-3.3-70B has 80; results are usually quoted at 20, 40 or 44.
- **k** — how many persona components a probe uses. The paper uses k=1 for deception; this project
  defaults to k=5, which appears nowhere in the paper.
- **Matched control (`datapca`)** — the same machinery with the personas replaced by directions from
  the data itself. If it matches the persona result, the personas added nothing.
- **Null / p95** — 1,000 random directions scored the same way; p95 is the 95th percentile. Beating it
  is the minimum bar for "this direction is doing something".
- **BH correction** — Benjamini–Hochberg, the adjustment for running many tests at once.
- **HP-KR** — `harm-pressure-knowledge-report`, the dataset where the model denies knowledge it has.
  The draft's central dataset, and the source of most trouble in this report.
