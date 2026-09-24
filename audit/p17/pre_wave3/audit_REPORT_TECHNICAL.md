# REPORT (TECHNICAL) — audit of `expLB`

_2026-09-22 18:44 → 2026-09-23. Companion to `audit/REPORT_PLAIN.md`._

> **THIS REPORT CLOSES AT 2026-09-23 (DF-95, DF-94).** `origin/paper` landed 2026-09-23 23:46 and
> entered scope the next day, after this was drafted, so **F-117, F-119 and F-122 are absent** —
> the 16-layer sweep, the `paper/make_numbers.py` correction (which supersedes §8 item 1's
> reproducibility clause and is a correction in the collaborator's favour), and the four sycophancy
> arms on his branch.
>
> **Coverage figures are superseded by Phase 6A (2026-09-24).** For **"424 of 692 claims checked"**
> read **490 of 692 inventory claims with a verdict, plus 110 raised while checking** — real
> coverage **71%**. **The draft PDF is at 176 of 176:** 160 `MATCHES HIS DATA`, 9 `PARTLY`,
> 7 `UNRESOLVED`, 0 failures, and **no arithmetic error was found in any of the draft's tables** —
> Tables 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 and 11 all reproduce from his raw files. The 7 unresolved are
> two unreleased artifacts (the 36 persona vectors; the pipeline's own results tables), not two
> wrong numbers.
>
> **Two audit entries were themselves corrected by Phase 6.** **DF-12** is withdrawn — the script it
> says is missing is on `origin/paper` at `8fd071b2`, 28,232 B. **DF-46's scope claim** ("the sign
> inversion occurs only at L20, only in `axis1`") is false: seven (arm, layer) cells invert.
>
> See `audit/phase6/RESULTS.md`, `audit/RESULTS_CHAPTER.md`, `audit/COVERAGE_GAP.md`, and
> `kd/docs/FINDINGS.md` F-123/F-124/F-125.
> _The text below is superseded on these points and is kept for the record._

Every number is traceable to a file opened or a script run during this audit. Where a number exists
only in prose or only in a console log, it is tagged **DOC-ONLY** or **log-only** and is not treated as
evidence. This report **points to** the detailed files rather than duplicating them.

---

## 0. Scope, method, and what the audit could not do

**Subject.** Two branches: `asix` (owner, HEAD `43ee6d3`) and `origin/feat/persona-pc-ood`
(collaborator). Merge-base `2e2a785`, 2026-07-12. Four evidence sources: the git tree, the lab PC
(`user4@100.85.217.68`, `C:\Users\user4\expLB`), Google Drive via `rclone` remote `tamara:`, and the
owner's 19 Claude session transcripts.

**Method.** Phase 0 orientation → Phase 1 inventory → Phase 2 claim ledger → Phase 3 cross-checks →
Phase 4 independent re-derivation → Phase 5 reports. Each claim was resolved to a file and key, and a
value recomputed where raw data allowed.

**Hard limits, stated up front.**

| limit | consequence |
|---|---|
| **No pooled activations are committed on the collaborator's branch** (`.../activations` is empty) | **Nothing in his draft is recomputable from raw.** His numbers are checked for internal consistency and by recomputing derived aggregates from his committed matrices. His headline figures are **attested, not reproduced.** |
| 268 of 692 claims were never checked (38.7%) | bounded only by the Phase 4 random sample (n=54) |
| Several stores exist only on the lab PC | audited read-only in place; result-type files cached under `audit/cache/` with md5 verification |
| Git and lab hold different versions of the same result files (RF-11) | every claim records **which copy** was read |
| A re-run wave on 2026-09-22 15:40–16:26 rewrote ~58,000 lab result files (RF-13) | raw statistics unchanged; **BH q-values moved**, and this flipped one verdict (S-32) |

---

## 1. Coverage

`audit/scripts/p5_stats.py` recomputes everything in this section.

| | n |
|---|---:|
| claims identified | 692 |
| **claims checked against raw data** | **424** (61.3%) |
| unchecked | 268 (38.7%) |
| of those, re-derived in Phase 4 (random 20%) | 54 |
| experiments inventoried | 114 (`audit/experiments.csv`) |
| claim notes written | 424 (`audit/claims/*.md`) |
| documentation corrections proposed | 75 (`audit/DOC_FIXES.md`) |
| red flags | 16 (RF-1 … RF-16) |

**Verdicts, after the Phase 4 ledger fixes:**

| verdict | n | share |
|---|---:|---:|
| HOLDS | 297 | 70.0% |
| PARTLY HOLDS | 119 | 28.1% |
| DOES NOT HOLD / REVISED | 8 | 1.9% |

| source | n | HOLDS | PARTLY | FAILS |
|---|---:|---:|---:|---:|
| `kd/docs/FINDINGS.md` ledger (F) | 110 | 65 | 43 | 2 |
| owner's documents (O) | 103 | 77 | 23 | 3 |
| owner's auto-memory (M) | 97 | 78 | 18 | 1 |
| collaborator's draft PDF (D) | 61 | 50 | 10 | 1 |
| owner's session transcripts (S) | 35 | 19 | 15 | 1 |
| collaborator's branch write-ups (H) | 10 | 4 | 6 | 0 |
| 2026-09-21 abstract (A) | 8 | 4 | 4 | 0 |

Owner 353 · collaborator 71.

**Interpretation.** The dominant verdict class is PARTLY HOLDS at 28%. Across the corpus the recurring
fault is **not a wrong number** — it is a correct number in a sentence that misstates its scope, sign,
comparator, layer or dataset set.

**Papers:** 29 rows in `audit/papers.csv`; **11 PDFs present**, 7 with notes, **15 arXiv ids cited with
no PDF** (9 of them cited by the draft). One draft citation is mis-titled: arXiv:2602.01425 is given as
*"One probe won't catch them all"*; two in-repo PDFs title it *"Building better deception probes using
targeted instruction pairs"*, and the paper is in neither the repo nor the lab PC.

---

## 2. The audit's own error bar (Phase 4)

Full write-up `audit/phase4/RESULTS.md`; adjudications `audit/phase4/adjudications.csv`; rates
recomputed by `audit/scripts/p4_rates.py`.

**Design.** 161 claims in 5 strata (seed 42, `p4_select_sample.py`), split into 14 batches by source,
handed to agents as `sample_blinded.csv` — **prior verdict, prior importance and tier withheld**.
3–4 agents concurrent. The lead alone joined results to the unblinded sample.

| | value |
|---|---:|
| raw disagreement | **33/107 = 30.8%** |
| **adjudicated audit-error rate** | **7/107 = 6.5%** |
| disagreements about a *number* | **1** (S-33 — the agent was right) |

| tier | n | raw | **audit error** |
|---|---:|---:|---:|
| T1 outright failures | 8 | 50.0% | **0.0%** |
| T2 lead-QA'd | 29 | 27.6% | **3.4%** |
| **T3 HIGH-importance HOLDS** | 40 | 17.5% | **15.0%** |
| T3b HIGH-importance PARTLY | 30 | 46.7% | **0.0%** |

**All seven errors are `HOLDS` that should have been `PARTLY`. None is a claim graded too harshly.**
T1's 50% raw rate is an artifact of tier construction (see limitations) — **0 of 8 failure verdicts
were reversed.**

**Mechanism.** All seven audit errors, and 19 of 21 Phase 4 errors, are **the verdict cell not matching
the checker's own notes**. The plan had predicted *"a number checked at the project default instead of
at the configuration the claim states"*; none of the seven is that.

| id | the checker's own note said | graded |
|---|---|---|
| F-01 | *"attaches a superseded-row-set caveat to the current number"* | HOLDS |
| M-007 | *"TWO COMPARABILITY CAVEATS the note omits"* | HOLDS |
| M-054 | *"compares two different estimators and should name them"* | HOLDS |
| D-028 | *"**AUDIT FLAG**: the headline 0.220 is HP-KR-driven"* | MATCHES |
| F-70 | *"checked elsewhere"* — of a number in its own `stated_numbers` | HOLDS |
| O-022 | (nothing — numbers recomputed to 12 s.f., sentence never read) | HOLDS |
| S-25 | (nothing — premise wrong: a PCA sign flip quoted as a distance) | HOLDS |

**Census of that mode.** `audit/scripts/p4_verdict_note_conflict.py` sweeps all 424 claims and flags
**69 of 297 clean-verdict rows** carrying defect language (positive control: 43% of PARTLY rows trip
the same patterns). 6 confirmed and fixed. **The other 63 are reported, not re-graded** — owner's
decision, 2026-09-23, on the grounds that re-grading on a regex would repeat the fault being measured.
Two sub-modes: *observed-but-not-graded* (screenable, 69 bounded) and *never-observed* (O-022, S-25 —
only re-derivation finds these).

**Limitations of the error bar, all of which make 6.5% an under-estimate.**
1. **Independence breach, self-reported.** The SA1 agent's `grep` reached `audit/DOC_FIXES.md`,
   `audit/phase3/DRAFT_GAPS.md`, `audit/PROGRESS.md`. **A-10 excluded.** The instructions named
   forbidden files but did not require scoping searches away from `audit/`; **11 of 14 batches ran
   without that guard.** Contamination inflates agreement, so it deflates the measured error rate.
2. **Instructions changed mid-stream.** F1/F2/F3/O1 (52 claims) ran without a verdict-discipline
   paragraph; the rest had it. Raw rates **37.3% before, 23.6% after** — the change pushes agents
   toward PARTLY and therefore flatters the audit. The adjudicated rate is unaffected.
3. **`REVISED → X` is structurally invisible** to a blinded reader (F-58, M-013): most of T1's raw rate.
4. **Compound claims have no grading rule** (D-237, S-33) — a gap in the verdict vocabulary.
5. **Event-claims have no raw data by construction** (O-041: what two reviewers said).
6. **The lead adjudicated alone** — the same reader whose work was measured. Itemised for checking.
7. **The lead overstated D-158 and corrected it in-session** (weighting-shopping; `CORRECTIONS.md`).

**New coverage (T4, the previously unchecked corpus, n=54):** HOLDS 42 (77.8%) · PARTLY 10 (18.5%) ·
FAILS 1 (1.9%) · UNRESOLVED 1. Against the checked corpus (71.7 / 26.4 / 1.9) the unchecked material
is **no worse**. But the single failure is **D-158**, the audit's most consequential finding — a low
rate does not make the remaining 214 safe to ignore.

---

## 3. Results, with provenance

Detail in `audit/phase5/digest_results.md`; control detail in `audit/phase3/CONTROLS.md` and
`audit/phase5/digest_story.md` §4.

### 3.1 Pooled training (the strongest result) — lead-verified

| quantity | value | source |
|---|---:|---|
| single-source transfer baseline | 0.6763 | `premise_L40_mean_s42.json` `summary.single` |
| pooled LODO | 0.8545 | `summary.pooled_lodo` |
| **gain, plain LODO** | **+0.1782** | difference |
| **gain, leave-one-CLUSTER-out (11 targets)** | **+0.1254** | `clustered_L40_mean_s42.json` `summary.all.gain_loco` |
| positive on | **10 of 11** targets | `summary.all.loco_positive` |
| gain, dropping the 2 leaked targets | +0.1504 (LOCO) / +0.1611 (LODO) | `summary.clean.*` |

Leaked targets: `harm-pressure-knowledge-report`, `harm-pressure-kr-matched` — recorded in the file's
own `leaked` field. Clusters are stored in `clusters`. Layer 40, mean pooling, `paper` probe, seed 42.
Five seeds at L40 span **+0.173 to +0.198** (F-83). Dataset-level interval **±0.035**.
**Report +0.125.**

### 3.2 Zero-shot control table (layer 40 / mean; layer 44 / last)

Sign convention, stated because five audit errors came from omitting it: `value` is **signed AUROC**;
below 0.5 is **inverted**; the null is direction-oriented (percentiles of max(a, 1−a)), so the
comparable quantity is `oriented_value` and **an inverted axis can legitimately beat the null**.

| dataset | L40 mean | beats p95 | L44 last | beats p95 |
|---|---:|---|---:|---|
| ai_liar | 0.9367 | yes | 0.9475 | yes |
| insider-trading | 0.9803 | yes | 0.9724 | yes |
| instructed-deception | 0.9709 | yes | 0.8946 | yes |
| roleplaying | 0.7842 | yes | 0.8696 | yes |
| sandbagging_v2 | 0.8454 | yes | 0.8600 | no |
| convincing-game | 0.7884 | no | 0.7735 | no |
| harm-pressure-choice | 0.5439 | no | 0.5511 | no |
| **harm-pressure-knowledge-report** | **0.6436** (p95 0.9377) | **no** | 0.5830 | no |
| mask | **0.4669 inverted** | no | **0.4417 inverted** | no |

**5 of 9 / 4 of 9.** Sleeper datasets at L44/last: **0 of 4** — gender-secret **0.3512 inverted**
(p95 0.7630); soft-trigger 0.8738–0.9128 inside nulls of 0.9234–0.9271.

`sycophancy_mt` (null p95 0.6304): sycophancy axis **0.3131 signed = significantly inverted**;
assistant axis 0.7102 (clears the best-of-layers p95 0.69355, and 0.7263 at L24 beats the max of 1,000
draws) **but a length-only scorer reads 0.664 and no per-row projections are stored, so the length
control cannot be computed** — do not quote this row as a clean win; paper16-bank-specific (own100
never exceeds 0.6161). Deception axis 0.4089, inside the null in all four cells.

Source: `run-coords/zeroshot_nulls.json` (1,000 draws/cell, persisted), consolidated to
`audit/phase2/controls.csv` (29,414 rows) by `audit/scripts/phase3/w4_build_controls.py`.

### 3.3 The negative census

| | value |
|---|---:|
| dataset-level significance tests | 65,388 |
| median 95% interval | ±0.125 |
| median effect | 0.044 |
| tests with effect > interval | 13.9% |
| persona-beats-raw tests clearing BH | **4 of 16,548** |
| …at the reported operating points | **0 of 276** |
| kd-bank tests showing persona > raw or > datapca | **0 of 4,032** |

Supporting: dataset identity **99.3%** decodable vs the label's **+3.2pp** over majority (L40, mean,
3,624 rows, 10 datasets; the +3.2pp is **log-only**). `axes − random` **+0.007** on 10 datasets vs
**+0.192** on the paper's 5 — composition, not personas. A 1-d **response-length** feature beats raw at
more layers than any persona feature (10 datasets).

### 3.4 Prompt-side control (largest unwritten result)
A probe that never sees the model's answer recovers **~0.86** of the response-side signal above chance
at last-token pooling; `convincing-game` and `sandbagging_v2` are **1.0000** prompt-side
in-distribution. **Pooling-dependent — state the pooling.** Store `passP`.

### 3.5 Capitulation bank
**+0.112 [+0.050, +0.185]** vs raw on its own column; 4 seeds, spread 0.0088; baselines byte-identical.
On 11 datasets **+0.098 [−0.036, +0.208]** — interval spans zero. Format-matched control shows much of
the gain is format. Mechanism demonstration, not a method.

### 3.6 Scale, after the RF-16 repair
70B rung **+0.054** vs matched control, **−0.013** vs raw. Three figures (+0.079 / +0.075 / +0.054)
currently circulate in three documents. The decline with scale survives; its magnitude does not.
`run-scale/8b` is additionally **incomplete** (5 of 12 dataset dirs have no `rows.jsonl`; 3,093
non-alpaca rows over 7 datasets) and its `all` and `s23` axis builds **write the same path**
(DF-73, DF-74).

---

## 4. The collaborator's draft

Detail in `audit/phase5/digest_draft.md` and `audit/phase3/DRAFT_GAPS.md`.

**What is sound.** 51 of 79 checked draft claims **match his data**; exactly **1** does not
(`paper/make_numbers.py`, which exists on no branch, in no commit, and not on the lab PC). The two
pipelines agree across **55 bank-matched cells, mean |gap| 0.0015** — there is no computational bug on
either side. Tables 1, 3, 4, 5, 6 and 7 reproduce from his committed CSVs.

**Ranked problems.**

1. **D-158 — the pooled baseline he says he did not run.** `loo_L20/`, `loo_L40/`, commit `6b16a46`
   2026-09-20 20:05, *"Leave-one-dataset-out: the fair baseline for the generalization claim"*; the
   draft PDF was compiled 2026-09-22. Lead-recomputed (`audit/scripts/p4_loo_baseline.py`), axis minus
   pooled, **excluding HP-KR**, under a **fixed** weighting:

   | | pooled/naive | pooled/balanced |
   |---|---:|---:|
   | L20 | +0.1408 | **−0.0197** |
   | L40 | **+0.0016** | +0.0640 |

   Robust across all four cells: removing HP-KR costs **0.048–0.130 AUROC**; in **2 of 4** what remains
   is ≤ +0.002 or negative. *(Do not quote "the better pooled arm at each layer" — that is
   weighting-shopping, the same fault his own `loo_L20/summary.md` commits.)*
2. **HP-KR is ~91–96% solvable from wording.** Label-free regex **0.9125** / **0.9597** vs his headline
   **0.919**; length-only **0.776** vs the deception axis's **0.6436** (itself inside its null). The
   wording-controlled test he calls decisive **was run** and is inconclusive: **+0.066, CI
   [−0.069, +0.189], P = 0.161** at the pre-registered L40; the same cell reads 0.374 at L20.
   On the matched HP-KR slice his extended axis reads **0.463 / 0.620 / 0.538** — at or below chance.
3. **The headline contribution is one dataset.** "+0.220 at L20" → **+0.009** dropping HP-KR from train
   and test; his own Table 3 shows it, the Contributions bullet does not.
4. **Wrong k.** The paper's deception figures are **k=1**; k=5 appears nowhere in it. At k=1 his own
   `summary.csv` has PC1 beating the contrast at all three layers (+0.016/+0.027/+0.025), and Table 3
   has no Published-PC1 row.
5. **`repe_honesty` is the base probe's own training set** and is counted among "all six test sets";
   dropping it at L20 puts the contrast **below** its baseline (0.5900 vs 0.6085).
6. **Bank confusion.** "The paper's 8v7 axis" denotes two different directions across the two sides, up
   to **0.148** apart on HP-KR L20; "HP-KR at L20" has **six** circulating values (0.103 … 0.2504),
   three quoted without naming the bank. Three banks exist; the draft reports two, omitting the one his
   own summary says makes every arm worse.
7. **Error bars too narrow** — example sampling only, while bank re-extraction moved HP-KR by
   **0.036–0.077**. Table 3's "Random directions (k=3)" is a **trained** probe averaged over 5 draws
   with no spread stored. Balanced accuracy 0.500 in Table 2 means "flags nothing" (recall 0.0), not
   chance.
8. **Not reproducible as released.** No pooled activations committed; Table 9 needs a ~60 GB
   gitignored per-token cache.

**Correction running in his favour:** `claims.csv` row **D-107** mis-describes his random control — it
is the mean of five draws.

---

## 5. Owner ↔ collaborator disagreements

`audit/phase3/DISAGREEMENTS.md`, digested in `audit/phase5/digest_draft.md` §5. Principal cases:

- **F-62 "s3 vs s23 makes no measurable difference"** vs his measured loss on the same swap
  (`ext_axis1` 0.9158 → 0.9066 at L40; HP-KR 0.912 → 0.876). Her replication of his arms is built on
  the bank **his** control found worse.
- **Four L20 matrices carry a sign-flipped HP-KR training row** (0.6511 → 0.7909 corrected), inflating
  the L20 lift only; the L40 headline is untouched.
- **Zero-shot transfer is inside a 1,000-direction null on 4 of 9 datasets at L40/mean** — and this has
  **no F-number on either side**.

---

## 6. Red flags (16) — the ones that bear on validity

Full list at the top of `audit/PROGRESS.md`.

| id | one line |
|---|---|
| **RF-16** | the 70B scale rung mixed two persona banks under one filename; F-100–F-102's layer average is not like-for-like |
| **RF-13** | a boot-triggered re-run wave (2026-09-22 15:40–16:26) rewrote ~58,000 lab result files; raw statistics unchanged, **BH q moved**, flipping S-32 |
| **RF-14** | a "skip if DONE" guard could never fire in 7 chain scripts, so gate lines quoted by 10 findings were truncated away |
| **RF-11** | git and lab hold different versions of the same result files |
| **RF-10** | `run-premise/`, `run-cap/`, `run-cap-ctrl/`, `logs/`, ~133 loose scripts exist **only on the lab PC** |
| **RF-5 / RF-6** | a committed `scores.jsonl` truncated at exactly 204,800 B; a literal backspace byte disabling a completeness gate |
| **RF-3** | the draft says the pooled baseline was not compared; his branch has `loo_L20/`, `loo_L40/` → became **D-158** |

**Statistical bottlenecks:** the 11-dataset ceiling; seed 42 monoculture (18,460 of 18,532 `paper`-probe
rows, seed range up to 0.168); layer-40 selection favouring persona features specifically (top-third
62% vs 22%, worth ~0.052); the HP-KR leak (42 of 52 rows genuine, −0.013 to −0.056).

**Process integrity:** the sarcasm gate threshold (0.65) first enters the repo in `b34ff2e`
2026-09-21 21:48:54, **after** the rollouts (21:26–21:40) and judge files (21:42–21:44) — the finding
survives unchanged (0.921, 0.947), the pre-registration claim does not (DF-66).

### 3.7 The two 2026-09-23 runs (F-115, F-116)

Both run from an isolated tree (`C:/Users/user4/expLB-resplit`) reading the originals read-only;
`27_crossgen.py` verified md5 `681126242fe4414e4a3c131793de81c6` before and after,
`run-coords/passB` still 13 dataset dirs.

**F-115 — leak guard.** Off-diagonal, drop any target row whose `row_id` is in the source's train
rows. Fired on **one pair, both directions** (re-confirming F-108); **59 of 258** rows survive.

| layer | leaked `axes−raw` | guarded | Δ |
|---|---:|---:|---:|
| 20 | −0.2566 | −0.2545 | +0.0022 |
| 40 | +0.0367 | +0.0309 | −0.0058 |
| 44 | +0.0134 | +0.0050 | −0.0084 |

Affected cells move far more: `knowledge-report → matched` at L40 **0.5674 → 0.4262** (below chance);
`raw` on the same cell 0.7932 → 0.7214.

**F-116 — the `knowledge_matched` slice, 2,001 rows, never previously run.**

| layer | `axes` | `raw` | `axes−raw` | `length` |
|---|---:|---:|---:|---:|
| 20 | 0.3850 | 0.5711 | **−0.1860** | 0.6174 |
| 40 | 0.3335 | 0.7093 | **−0.3758** | 0.6174 |
| 44 | 0.5267 | 0.7203 | **−0.1937** | 0.6174 |

In-distribution `raw` = **0.9995 / 0.9992 / 0.9993**; `axes` 0.9541 / 0.8347 / 0.7209.
**Confound flagged:** the zero-shot agreement with the full dataset is an artifact of 93.5% overlap
and carries no weight.

Scripts: `p6_patch_leakguard.py`, `p6_resplit_compare.py`, `p6_build_km_view.py`, `labpc_put.py`.

### 3.8 The collaborator's `origin/paper` branch (reviewed 2026-09-24)

New branch, 3 commits, 683 files, pushed 2026-09-23 23:46. **Outside the audited 424-claim corpus**;
reviewed at the owner's request. Detail in `audit/COLLAB_PAPER_BRANCH.md`; script
`audit/scripts/p8_collab_layer_sweep.py`.

**16-layer sweep** (L20–76 step 4, +L79) replacing 3 layers. Pooling moved onto NDIF; **validated** —
L40 rebuilt both ways, row cosine ≥ 0.99989, AUROC to 4 dp, RepE probe cosine 0.9996.

| comparison | with HP-KR | excl HP-KR |
|---|---|---|
| `ext_axis1 − baseline` | **+0.115 to +0.273, 16/16 positive** | **+0.123 to +0.243, 16/16** |
| `ext_axis1 − ext_pc3` (draft's Contributions) | +0.220 → −0.021, **13/16** | **within ±0.06, 11/16** |

**D-028 generalises from 3 layers to 16.** "At every depth" is falsified — negative at L64/68/72.
Optimum is **L36 (0.9275)**, not the reported L40 (0.9158).
**The L76 rise is SOLVED (F-120).** Reproduced in our own `passA` 80-layer store, so it is a property
of the model. `raw` **declines** (0.7834 → 0.7426); `axes` **recovers** to 0.7492 but stays below its
own mid-stack 0.8145; and **`random` gains +0.127** (0.5133 → 0.6405) — late-layer anisotropy lifting
any low-dim projection. **`axes − random` is +0.2296 at L40 against +0.1087 at L79**, so the method is
*worse* there. Script `p9_late_layer.py`.

**Gender-Secret (F-118):** both classes deny; every persona direction at chance (0.436–0.623).
Adapter applied inside the remote trace, PEFT agreement 1e-5, behavioural check 82% vs 55% base,
paired bootstrap. **Third independent wording-matched confirmation of the register diagnosis.**

**D-237 SUPERSEDED (in his favour):** `paper/make_numbers.py` now exists → F-119.

### 3.9 The kd bank's scale ladder — completed 2026-09-24 (F-121)

The cell the project never ran. 36 personas x 5 variants x 30 shared questions = **5,400 rollouts at
3B and again at 8B**, construction identical to the 70B kd bank (same personas, questions, variants,
and the Llama-3.1-70B judge held constant). Isolated tree; originals md5-unchanged.

Crossgen on identical footing everywhere: 5 datasets (MASK fifth, F-60), k=3, mean, `paper`, seed 42.

| rung | paper16 (26p) | kd all (36p) | kd s23 (36p) |
|---|---:|---:|---:|
| 3B (7 layers) | +0.0382 | +0.0317 | +0.0360 |
| 8B (8 layers) | +0.0148 | +0.0169 | +0.0194 |
| 70B (21 layers) | **-0.0116** | **+0.0032** | **+0.0054** |

kd - paper16: -0.0066/-0.0022 (3B) -> +0.0022/+0.0046 (8B) -> **+0.0148/+0.0170 (70B)**.

**Both banks decline monotonically: the decline is the method's, not the bank's** — which closes the
"wrong personas" objection to F-63/F-110. The kd bank declines less and the bank gap grows with
scale, but at 70B both sit within a whisker of zero. **Scope condition: these 5 datasets contain no
knowledge-denial data**, so this is the bank's general effect, not its intended one.

**DF-48 verified**: axis variants byte-distinct at both rungs. Script `p12_kd_scale_ladder.py`.
Backup: 290 files, 0 differences.

---

## 7. Artifacts and scripts

**Deliverables.** `REPORT_PLAIN.md` · `REPORT_TECHNICAL.md` · `COLLABORATOR_FINDINGS.md` ·
`COLLAB_PAPER_BRANCH.md` · `HPKR_WORDING_CONTROLLED.md` · `DOC_FIXES.md` (85) ·
`METHODS_CHAPTER.md` · `LIMITATIONS_AND_FUTURE_WORK.md` · `CORRECTIONS.md` · `TIMELINE.md` ·
`PROJECT_MAP.md` · `claims.csv` (424) + `claims/*.md` (424) · `experiments.csv` (114) · `papers.csv` (29)

**Phase 4.** `phase4/RESULTS.md` · `sample.csv` · `sample_blinded.csv` · `batches/*.csv` ·
`rederive_*.csv` (14) · `merged.csv` · `adjudications.csv` · `verdict_note_conflicts.csv` ·
`evidence_F2/` (lab-pulled, **not** md5-certified — see its README) · `AGENT_INSTRUCTIONS.md`

**Phase 5 digests.** `phase5/digest_results.md` · `digest_draft.md` · `digest_story.md`
(the last carries a lead correction: it was written from `TIMELINE.md` and its "231 of 692 checked" is
a Phase 2 snapshot; the final figure is 424).

**Re-runnable scripts** (`audit/scripts/`): `p4_select_sample.py` · `p4_make_batches.py` ·
`p4_merge.py` · `p4_rates.py` · `p4_verdict_note_conflict.py` · `p4_loo_baseline.py` ·
`p4_apply_ledger_fixes.py` (idempotent) · `p5_stats.py` · plus the Phase 1–3 script set.

**Repository state at close:** branch `asix`, HEAD `43ee6d3`; `git status --porcelain -uall` outside
`audit/` = **18 entries** (17 modified + untracked `docs/DRAFT_CORRECTIONS.md`), all owner-authorised,
**nothing committed**. Lab PC: 0 `user4` python processes (6 belong to another user, `user5`).

---

## 8. What a referee would attack first

In order, with the current state of the defence:

1. **"Your central dataset is solvable by a regex."** — **now answered three times over: F-115,
   F-116, and the collaborator's independent Gender-Secret result (F-118).** — **The wording-controlled test has been run
   (F-03 / F-72 / F-76) and it does not rescue the claim.** On the matched slice (`harm-pressure-kr-matched`,
   258 rows, both classes deny) the margin is **+0.066, CI [−0.069, +0.189], P = 0.161** at the
   pre-registered L40 and **0.374 at L20** — 0.216 *below* raw; the 95% argmax interval is **[L5, L79]**.
   His extended axis reads **0.4630 / 0.6203 / 0.5383** zero-shot there and his split knowledge-denial
   axis **0.3932 / 0.4989 / 0.4825**, at or below chance. **Report this; do not promise it as future
   work** (the draft does, on p.12 and p.13).
   **Caveat:** the slice leaks for *trained* comparisons — its row ids are HP-KR row ids by construction
   and **~81% of its test rows sit in HP-KR's train split** (F-108). Zero-shot readings are unaffected.
   A clean trained comparison needs **newly generated NDIF rollouts**, not a CPU-only re-run.
   **RESOLVED 2026-09-23 by two runs, ledger F-115 and F-116** — see §3.7. The leak is closed and did
   not account for the result; the never-run `knowledge_matched` control (2,001 rows) puts the persona
   axes **below chance** on transfer. Neither needed NDIF.
2. **"You never compared against a pooled probe."** — It was compared, twice, on both sides; the result
   removes most of the persona advantage. **Report it and lead with it.**
3. **"Your effects are smaller than your error bars."** — True: median effect 0.044, median interval
   ±0.125. Only the pooled result (±0.035) escapes. **Say so explicitly.**
4. **"You chose the layer."** — Layer 40 favours persona features by ~0.052. **Report the layer sweep,
   not the layer.**
5. **"One seed."** — 18,460 of 18,532 rows. **State it as a limitation; it is not fixable now.**
6. **"Your pre-registration is dated after your result."** — True for the sarcasm gate. **Restate it
   honestly; the finding does not depend on it.**
