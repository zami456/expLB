# Phase 5 digest — the story, and the control table

*Written 2026-09-23. Digested from `audit/TIMELINE.md` (716 lines) and `audit/phase3/CONTROLS.md`
(157 lines). Nothing here is re-derived; every number carries the path its source gives.*

**Tags used below**, carried over from the source documents: **VERIFIED** = checked in a raw file, a
log or git during the audit · **DOC-ONLY** = stated only in prose, not re-derived · **INFERRED** =
the audit's interpretation · **NOT-FOUND** = looked for, not found · **CONFLICT** = two sources
disagree.

---

## 1. THE PROJECT IN ONE PAGE

**The aim.** Take one small paper and see whether it holds at large scale. The paper is Mahadik &
Skapars, *Do Linear Probes Generalize Better in Persona Coordinates?* Its idea: prompt a language
model to role-play contrasting characters — "a scrupulously honest assistant" against "a manipulative
one" — record the model's internal activity under each, and extract a handful of directions from that
activity. Those directions are the **persona axes** (a *direction* is just a weighted recipe for
combining the model's thousands of internal numbers into one score). The paper's claim is that a lie
detector built from a few persona coordinates transfers to unseen datasets better than one built on
the model's raw internal state. At 3 billion parameters the paper reports a gain of **+0.10 AUROC**.

**AUROC**, used throughout: the probability that a randomly chosen lie scores above a randomly chosen
honest response. 0.5 is chance. 1.0 is perfect. **Below 0.5 means the detector is pointing the wrong
way** — it systematically calls lies honest and honest lies.

**What was actually built.** A replication on Llama-3.3-70B-Instruct — 23× the paper's model — with
no local GPU, running remotely on NDIF. **15,278 model responses across 10 datasets**, 5.5× the
paper's corpus (DOC-ONLY, `docs/thesis/persona-coordinates-70b-results.md:24-35`). Around it grew: a
pipeline of three activation "passes" (B = 21 layers over the full corpus; A = all 80 layers over 200
rows per dataset; C = per-token, 6 layers); seven persona **banks** (a bank = one set of role
instructions, run and turned into axes); a 10×10 **cross-generalisation grid** (train a detector on
one dataset, test on all ten, for every training choice — the off-diagonal mean is the transfer
score); a scale ladder at 3B / 8B / 70B; and two purpose-built diagnostic datasets. The audit counted
**114 experiments** (`audit/experiments.csv`) and **692 claims**, of which 231 were checked against
raw data (`audit/PROGRESS.md`).

**How the question changed, three times.**

1. **"Does it replicate?" → "What does it actually measure?"** By 2026-09-17 the answer was: persona
   features beat a raw probe only inside a narrow band of network depth, only on some datasets, and
   never for the paper's own headline feature. Of **384 dataset-level significance tests**, exactly
   **1** survived multiple-comparison correction (F-04, `run-crossgen/kd11/significance.json`).
2. **"The axes are noisy" → "the axes encode the wrong thing."** Two datasets broke every detector
   the same way. The diagnosis (2026-09-20, F-13): the persona axes encode **register** — confident
   assertion vs hedging, agreeable vs challenging tone — not the deception the labels mark. Two new
   banks were then built to test that: a **capitulation** bank (worked, but locally) and a **sarcasm**
   bank (failed its own gates, never built).
3. **"Personas help" → "personas are not needed."** On 2026-09-21 the owner ran the cell nobody had
   run: pool ten datasets, train one ordinary probe, test on the eleventh, no personas at all. It beat
   every persona result in the project by **+0.178 AUROC** and cleared the dataset-level bootstrap by
   the widest margin anything in the project had (F-79/F-80, `run-premise/premise_L40_mean_s42.json`).
   **This is the project's largest result and it is not in the collaborator's draft.**

**Where it stands.** The collaborator compiled a 19-page draft on 2026-09-22 13:57 (VERIFIED, PDF
metadata). Its headline — 0.904 / 0.916 mean off-diagonal AUROC from a one-dimensional persona
contrast, with no labels — reproduces exactly from his own files, but the contrast is built on the
owner's **knowledge-denial** personas, not the paper's. The paper's own 8-vs-7 contrast reads
**0.791 / 0.832**; roughly 44–46% of the headline margin is the owner's bank (F-97; `audit/claims.csv`
A-04, verdict "MATCHES HIS DATA … but the ATTRIBUTION is wrong"). Neither party had a full picture of
the other's results. The audit began 2026-09-22 18:44 to produce one.

---

## 2. THE STORY IN PHASES

*Dates are lab-local (UTC+6). **Commit dates are batch dates, not run dates** — findings F-01…F-66 all
entered git in one commit on 2026-09-21 03:26 although dated 09-19 to 09-21, and 2026-09-20 produced
51 findings and no commit at all. The timeline is therefore anchored on internal log timestamps
(`audit/phase1/labpc/log_index.tsv`, 242 logs).*

**Before: 2025-11 → 2026-05-07 · inherited repository.** Cadenza Labs' Liars' Bench deception
benchmark, by Kretschmar and Laurito. Not this project's work; it supplies the evaluation datasets.
**2026-07-11/12** is the branch point (`2e2a785` = merge-base). **2026-09-04**: the collaborator builds
a resumable activation cache and token-budget batching; the owner ports it on 2026-09-15 at `ed8daca`.

**2026-09-07 → 09-12 · pipeline.** Project starts by adding the `assistant_axis` submodule (Lu et
al.'s direction measuring how "Assistant-like" a model is behaving). Five Claude sessions set up
remote model access and the lab PC. Two engineering constraints fixed everything after: long runs must
be **Windows Scheduled Tasks** (SSH kills the process tree), and a remote trace body must be defined in
the script that runs it.

**2026-09-12 05:43 → 09-15 12:51 · first real run, `own100`.** 100 roles on Llama-3.3-70B, three
parallel shards 09-12 17:47 → 09-14 13:49, finishing "vectors=100 axis.pt=True axes.pt=True".

**2026-09-14 14:22 · the pivot.** Work is re-scoped from an Assistant-Axis replication to a
persona-coordinates study, with a **paper-matched persona bank** as the controlled comparison.

**2026-09-14 → 09-16 · two bugs.** (a) Long inputs were truncated from the front, discarding the
assistant's own words — the exact span the method must read; fixed at `7842bf6`. (b) Masked averaging
over the response ran in the model's own bfloat16 precision, silently producing a wrong-last-token
variant; fixed at `ae31943`, paper16 vectors re-extracted 09-15 21:45 → 09-16 01:30. **The audit
measured the damage: cosine 1.0000 between old and new axes at every layer, ≤0.9% length difference,
PCA components differing by ≤0.009 (VERIFIED, `audit/phase1/axes_old_vs_fp32.txt`). The bug was
expensive in time, not in results.** The equivalent `own100` rebuild was abandoned after 1 of 100
roles, so `own100` still rests on the buggy reduction — no headline depends on it.

**2026-09-15 → 09-17 · the eleven questions (Q1–Q11), each with its expected outcome written before
the run.** Three activation passes run (B: 09-15 12:58 → 21:31; A: 09-16 02:07 → 02:55; C: 09-16 12:50
→ 13:56). A sycophancy corpus is built from scratch because none existed: single-turn first, then —
when single-turn proved uninformative (F-15) — **multi-turn capitulation under pushback**, 1,182 rows,
finished 09-16 18:23. That dataset, `sycophancy_mt`, later breaks the method. Twenty short analyses run
09-16 14:39 → 09-17 02:09.

The owner's answers, written 09-17 (DOC-ONLY, `docs/thesis/persona-coordinates-70b-results.md:42-56`):
persona features beat raw only in a depth window (~layers 36–48 of 80), only on the nine deception
datasets, never for the paper's own PC-projection feature (Q2); adding `sycophancy_mt` removes the
advantage at every layer; erasing the assistant direction costs only 0.005–0.018 AUROC (Q4); label
semantics does not predict transfer (Q5); moving from layer 20 to 40 is worth ~+0.025 for raw probes
and ~+0.113 for persona methods (Q6); a 70B and a 405B judge agree 91.2% (Q7); the sycophancy axis is
significantly **inverted** on real capitulation (Q10); sleeper-agent lies are flagged at 0.96 but what
is detected is a false answer, not deception (Q11). Her own summary the same day: *"The headline
advantage is fragile… no Q2 or Q6 difference survives a bootstrap over datasets."*

**→ REVISED:** Q1 ("the paper's persona set removes most of the Assistant-Axis contamination,
−0.66 → −0.37") was **confounded** — see next.

**2026-09-17 16:10 · confound 1, question family.** The persona bank's first principal component
(47% of variance) separates personas by **which question set they answered**, not by what they are —
cosine −0.98 with the question-family indicator. `paper16` had been built on the paper's questions
only, so `paper16` vs `own100` differed in more than personas. Fixed overnight 09-17 22:31 → 09-18
03:14 by regenerating all 26 personas on one shared question set (2,600 rollouts) — this is
`paper16-shared`, the bank later exported to the collaborator.

**2026-09-17 21:28 → 09-19 · the 10×10 cross-generalisation grid**, for raw features and seven persona
banks, 21 layers, 6 seeds, two pooling methods. (The first attempt crashed 09-17 21:18:44,
`PYTHON_EXIT -1` at layer 12 of 21; cause NOT-FOUND; restarted the same evening.)

**Confound 2, and the result that reframed everything.** Against a **matched control** — a dataset-PCA
of the same dimensionality, i.e. "any k directions fitted to this data" rather than "persona
directions" — the unified persona axis reads **−0.003**. A *random* 2-d projection gains +0.056 where
the raw baseline is weak and loses 0.44 where it is strong, reproducing the paper's qualitative claim
with meaningless directions (F-17, `run-crossgen/significance.json`). **INFERRED in the timeline as the
single most consequential result of the project:** the reported advantage is mostly a low-dimensional
ceiling artifact. From here on every claim is scored against `datapca`, not only against raw.

**2026-09-18 11:26 / 16:19 · the handoff.** Write-up committed (`b9a497e`); layer-20 vectors and axes
exported to Google Drive `share/`. **2026-09-18 17:31 — about an hour later — the collaborator starts.**
The audit verified the handoff by hashing: his `paper16_L20.pt` and `deceptive_L20.pt` are
byte-identical to the lab's copies, and all 36 of his knowledge-denial vectors are byte-identical to
`run-kd/vectors_s3/*.pt`. **Only the activations are his own** (~62 GB per layer, downloaded), which is
why most of his result CSVs cannot be recomputed from git.

**2026-09-18 → 09-20 · the HP-KR problem and the `kd` experiment.**
`harm-pressure-knowledge-report` (HP-KR) asks the model whether it knows a dangerous answer. **Every
detector in the project inverted on it** — scored below chance. Why took three days.
- Its label is `deceptive = knows AND denies`, and the fourth cell of the 2×2 — *doesn't know but
  claims it does* — has **zero rows**. On the text-matched slice, "deceptive" and "knows" are literally
  the same variable. **The dataset cannot attribute its own results** (F-02).
- **F-01:** one regular expression — does the response contain a knowledge denial — scores **AUROC
  0.913** against HP-KR's own label. (→ REVISED upward from an earlier 0.870, which had been computed
  over all 7,068 rows of a CSV holding three generating models; the probes read only the 2,139 Llama
  rows.)
- **F-03, the strongest positive result the project ever had:** on a **text-matched slice of 258 rows**
  where every response says "I do not know the answer", so the wording shortcut is impossible, the
  trained persona axes go from **0.300 to 0.752** against a raw probe at 0.715 — the result reverses.
  Caveats written at the time: 258 rows, one layer, one seed, row-level bootstrap only.
  **→ LATER WITHDRAWN, see 09-21.**
- **F-04, the ceiling:** 1 of **384** dataset-level significance tests survives correction. The binding
  constraint is having ~11 datasets, not ~15,000 rows.
- The `kd` bank is built 09-19 17:11 → 09-20 03:04: two teacher-forced factor probes to separate "does
  it know" from "does it deny", a 1,459-item WMDP knowledge eval, then 26 paper16 personas + **10 new
  knowledge-denial and candour personas**, 5,400 rollouts, judged, vectors extracted.
- **F-13, the register diagnosis (09-20).** Two datasets fail identically: HP-KR (axis encodes
  confident assertion vs disclaimer; label marks denying held knowledge) reads 0.427 as target / 0.300
  as source; `sycophancy_mt` (agreeable vs challenging register; label marks capitulation) reads 0.355
  / 0.379. A random direction beats the axes on both. Dropping `sycophancy_mt` from the average lifts
  the raw probe +0.025 but the persona axes **+0.061** — persona features hurt ~2.5× more.

**A number to be careful with.** **0.374**, quoted in nine documents as "the paper's sycophancy axis on
real sycophancy data", is in fact the **deception** axis's first principal component. The sycophancy
axis reads **0.313** (VERIFIED, `audit/DOC_FIXES.md` DF-31; cause at `27_crossgen.py:411-416`). The
correction makes the argument stronger: 0.374 sits *inside* the random-direction null, 0.313 sits
beyond it.

**2026-09-20 12:41 → 23:40 · the scale ladder**, run because one p-value on eleven datasets settles
nothing: same pipeline at 3B and 8B (2,600 rollouts each, same 70B judge held constant), then
evaluation activations and grids. **The resulting claim was corrected four times — the clearest example
in the project of a number moving under scrutiny:**

| step | date | claim (gain over the raw probe unless noted) | 3B | 8B | 70B | what was wrong |
|---|---|---|---:|---:|---:|---|
| F-58 | 09-20 | "replicates at 3B/8B, dies at 70B" | +0.089 | +0.086 | +0.026 | — |
| F-60 | 09-20 | correction 1 | — | — | — | The paper evaluates **MASK**, not `HarmPressureChoice`; 4 of 5 datasets matched, the fifth wrong |
| F-63 | 09-20 | correction 2, same 5 datasets, same k=3 | +0.020 | +0.013 | +0.013 | F-58 compared different dataset sets **and** different k. Gain is small and **flat**, not declining. vs matched control: +0.202 / +0.115 / +0.079 |
| F-96 | 09-21 | correction 3 | +0.120 (mean of 7 layers) | +0.103 (8 layers) | 1 layer only | F-63's 3B figure was that rung's **best** of seven layers |
| F-100 | 09-22 | correction 4 | +0.120 | +0.103 | +0.075 | Decline survives layer-averaging; its steepness was a 3B layer-choice artifact |
| **F-110** | **09-23 (audit)** | **current** | — | — | **+0.054 vs matched control, −0.013 vs raw** | The 70B sweep used the **wrong persona bank** (36-persona `kd`, not `paper16`) because both folders hold a file named `axes_vectors_s3.pt`. Bank mix **VERIFIED** by zero-shot fingerprint, `audit/phase1/bank_fingerprint_scale70b.txt`. This is RF-16. |

The owner's own note at the time: *"four headline margins have now failed a layer sweep; F-63's 3B
+0.202 is the max of 7 layers."*

**2026-09-20 22:52 · a pre-registered gate fails and kills a plan.** Growing the HP-KR matched slice
from 258 to 500–700 rows required that, on items the model measurably gets wrong, it deny knowledge at
a rate ≥0.70. Measured: **5/20 = 0.250**; the fallback was worse, 3/20 = 0.150. *The model claims to
know answers it gets wrong, on 75% of items it answers incorrectly.* Route abandoned (F-59).

**2026-09-20 14:38 → 09-21 03:26 · the capitulation bank.** If the axes fail on `sycophancy_mt` because
they encode register rather than capitulation, a bank that varies **capitulation** with everything else
held fixed should not fail. Six smoke tests; the free multi-turn version was invalidated because the
capitulating persona was already wrong in turn 1 (F-46), so turn 1 was **teacher-forced** — identical
text in both arms, so only turn 2 can differ (F-51: 4/4 fold vs 4/4 hold). Full run 09-20 22:59 →
09-21 03:16. **Zero-shot — no probe, no training, no labels — the capitulation axis reads 0.745 on
`sycophancy_mt` against the register axis's 0.374** (F-65), and it **clears the dataset-level
bootstrap** against raw and both matched controls — the first positive result in the project to do so
(F-66).

**Then the owner bounded her own result, twice.** **F-67** (09-21 03:37): the advantage is **local** —
over the full 11×11 off-diagonal mean it does not extend past the column it was aimed at. **F-89**
(09-21 20:34): a **format-matched control** — swap the six position-holding personas for six topically
unrelated ones and the result costs only ~+0.03, while removing them entirely costs two to four times
that. So most of what the holding personas contribute is *the presence of an instruction*, not the
disposition it describes. The audit's verdict on F-65 / F-66 / F-67 is **PARTLY HOLDS**: the
measurements reproduce, the sentences around them overstate.

**2026-09-21 · the day four claims were withdrawn.**
- Morning: the HP-KR top-up runs 10:53 → 11:53, extracting all **2,139** HP-KR rows into passes A and
  C, both gates exact. All three stores now carry the matched slice identically — 258 rows, 124
  deceptive / 134 honest. The one-layer caveat on F-03 could finally be tested.
- **F-72 (12:33) → REVISES F-03:** the reversal is **layer-localised** — 0.739 at layer 40 but **0.374
  at layer 20**. The layer choice moves the result by ten times the claimed effect.
- **F-75 (13:43):** the 80-layer profile is a 16-layer plateau containing layer 40, marked *provisional*.
- **F-76 (13:47, four minutes later) → OVERTURNS F-03:** the plateau does not survive the dataset-level
  bootstrap. At the pre-specified layer the margin over the matched control is **+0.0655, 95% CI
  [−0.0693, +0.1887]**. **Not significant. F-03 withdrawn.**
- **F-73 (13:02) → REVISES the response-length claim:** the AUROC survives, the per-dataset framing does
  not. (The audit's *own* correction to this was later found wrong — two counts over two different
  sets; `audit/DOC_FIXES.md` DF-49, applied in six documents before being caught.)
- **F-74 (13:17) → WITHDRAWS the "−0.76 ceiling artifact" statistic:** it is mathematical coupling
  between the two quantities being correlated, and the clean test refutes the claim it supported.

**2026-09-21 14:26 → 17:24 · the premise fails.** Pooled leave-one-dataset-out, no personas:

| condition | AUROC (layer 40, mean pooling, seed 42) |
|---|---:|
| single-source, trained on one other dataset (the persona method's baseline) | 0.6763 |
| **pooled leave-one-dataset-out** — never sees the target | **0.8545** |
| the same, row budget held fixed (~666 rows, drawn from ten datasets instead of one) | 0.8500 |
| single-source, trained on its own data (upper bound, not deployable) | 0.9506 |

Rows 2 and 3 together say the gain is **diversity, not data volume**. Margin **+0.1782, 95% CI
[+0.1430, +0.2129]**, clearing the dataset-level bootstrap by the widest margin in the project
(F-79/F-80, `run-premise/premise_L40_mean_s42.json`). **Positive at all 21 layers, all 21 clearing**
(F-81), seed-stable (F-83), surviving falsification at **51 of 51 cells** (F-95). Mechanism (F-86):
single-source probe directions are nearly orthogonal, so averaging cancels the noise — joint training
is not needed. This family is the one sequence in the project that was **reinforced at every step, never
corrected**, including a leak check (a real row-level leak exists between HP-KR and its matched slice;
the claim survives leave-one-cluster-out anyway at +0.125 to +0.157 against +0.178).

**2026-09-21 17:48 → 09-22 14:30 · the sarcasm bank: designed, piloted, abandoned.** Pre-registered as a
design document before anything was generated. **Gate 1 passed** (F-92): deadpan sarcasm *is* producible
on-policy, marker rate 0.00 at variants v4 and v5 — the literature predicted this would fail.
**Gate 2 failed** (F-93): stripped of markers, ironic and sycophantic text are still separable at
**0.92 by bag-of-words alone**, so the bank would itself be a register axis — the very thing it was
built to diagnose. **Gate 3 failed** (F-107, 09-22 14:29): on 600 length-matched natural-sarcasm rows
from FigLang-2020, the deception axis reads **0.543–0.583** against a random-direction mean of
0.531–0.543 and a bag-of-words floor of 0.632 — indistinguishable from a random direction. The axis
that *does* fire is **sycophancy, at 0.693–0.710**, clearing both floors and exceeding the maximum of
200 random draws. The register diagnosis from a third angle. **The bank was never built.** (The gate-3
projection result was never persisted to a file.)

**2026-09-22 00:32 → 02:11 · whose axis is the headline?** **F-97:** the draft abstract's 0.904 / 0.916
is `ext_axis1`, a **12-vs-13 contrast over the 36-persona bank that includes the owner's ten
knowledge-denial personas** — not the persona-coordinates contrast. The paper's own 8-vs-7 contrast
reads **0.791 / 0.832**, so ~44–46% of the headline margin comes from the kd bank. **F-98:** rebuilt on
the owner's eleven datasets the kd bank is worth **49–77%** of the margin, and **both** of the
collaborator's arms lose to pooled training. **F-99:** the kd bank contributes at 21 of 21 layers,
median 63%.

Two related omissions on the collaborator's side: his draft says a pooled baseline was "not compared"
although his own branch carries `results/loo_L20` and `loo_L40`, exactly that baseline (RF-3, OR-17);
and his rollout-count control `persona_ood_mod_L*` scores **lower** than the arm he reports (0.9066 vs
0.9158 at layer 40) under his own heading "More rollouts per persona did not help" (DF-18).

**2026-09-22 13:57 · the draft is compiled.** 19 pages, 5 figures, 11 tables, **0 findings-ledger
citations**. **NOT-FOUND: the LaTeX source, anywhere on either branch.**

**2026-09-22 15:34:30 → 16:26 · the lab PC rebooted.** No person and no Claude session started it.
About 24 project Scheduled Tasks carry a Windows **boot trigger** with a delay; boot time plus each
task's delay equals its last run **to the second** (Crossgen-Chain PT5M → 15:40:01 … Cap-Ctrl-Layers
PT50M → 16:25:01). About **58,000 lab result files** were rewritten. The audit checked what changed,
matching records by key rather than position: for summary files the **git version is an exact subset of
the lab version with zero differing metric values** (the file grew, the old numbers did not move); for
significance files **every common test has identical raw statistics**, only the multiple-comparison
q-values differ because the family grew (192 → 4,788 tests in one file, max change 0.39).
**Consequence: any claim of the form "this clears / does not clear a corrected threshold" must be
checked against the version it was computed on — the verdict can flip between versions.** Why it
rebooted is **NOT-FOUND**; the box is multi-user and another account's jobs had been exhausting its
memory. A separate bug (RF-14) made the wave worse: seven chain scripts have a "skip if already done"
guard that can never fire — the writer emits `<timestamp>  DONE`, the guard matches `^DONE` — so every
trigger re-ran the chain and truncated its status file. Per-cell results were mostly skipped internally,
but the gate lines quoted by **ten findings** (F-72, F-75, F-77, F-81, F-82, F-83, F-87, F-94, F-95,
F-104) are gone from those files.

**2026-09-22 18:44 → · the audit.** Phase 0 mapped repo, branches, lab PC, Drive and draft. Phase 1
dated and statused **114 experiments**, censused 210,464 lab files and 277,407 Drive files, indexed 242
logs, recorded 65 anomalies. Session mining recovered **72 of 72 print-only findings** with their
console output — many ledger entries quote numbers that were only ever printed to a console
(**CONFLICT**: `audit/SESSION_MINING.md:7` says 42 of 109, `findings_targets.csv` carries 72; both are
audit-internal counts, neither re-derived). Phase 2 checked **231 of 692 claims**: 112 HOLDS, 51
MATCHES HIS DATA, 53 PARTLY HOLDS, 12 PARTLY MATCHES, 1 REVISED, 1 DOES NOT MATCH, 1 DOES NOT HOLD.
**The dominant pattern: 28 of 29 "partly holds" verdicts found the number reproducing and the sentence
around it failing.**

---

## 3. THE TURNING POINTS

**1. 2026-09-14 14:22 — the pivot from Assistant Axis to persona coordinates.** The work was rescoped
from replicating Lu et al.'s Assistant Axis pipeline to a persona-coordinates study with a
paper-matched persona bank as the controlled comparison. *Forced by:* a decision to make the paper's
own bank the comparator rather than a repair of someone else's pipeline. Everything after this is a
persona-coordinates project.

**2. 2026-09-17 16:10 — the question-family confound.** The persona bank's leading principal component
(47% of variance) turned out to track **which question set a persona answered**, cosine −0.98 with the
question-family indicator. *Forced:* an overnight regeneration of all 26 personas on one shared question
set (2,600 rollouts, 09-17 22:31 → 09-18 03:14), and the retirement of Q1's answer. Any "persona set A
beats persona set B" comparison before this was confounded.

**3. 2026-09-18/19 — the matched control.** Against a dataset-PCA of the same dimensionality — "any k
directions fitted to this data" — the unified persona axis reads **−0.003**, and a *random* 2-d
projection reproduces the paper's qualitative claim (+0.056 where raw is weak, −0.44 where raw is
strong). *Forced:* every subsequent claim to be scored against `datapca`, not just raw. The timeline
calls this (INFERRED) the single most consequential result of the project, because it says the reported
advantage is mostly a low-dimensional ceiling artifact.

**4. 2026-09-20 — the register diagnosis (F-13).** HP-KR and `sycophancy_mt` break every detector the
same way, and a random direction beats the axes on both. *Forced:* the project's main hypothesis — the
axes encode register, not deception — and two purpose-built banks to test it constructively
(capitulation: worked but local; sarcasm: failed its own gates).

**5. 2026-09-21 13:47 — F-76 withdraws F-03.** The strongest positive result in the project (persona
axes 0.300 → 0.752 on the 258-row text-matched HP-KR slice) turned out to be layer-localised (0.739 at
layer 40, 0.374 at layer 20) and then failed the dataset-level bootstrap: **+0.0655, 95% CI [−0.0693,
+0.1887]**. *Forced by:* the HP-KR top-up that morning, which removed the one-layer caveat and made the
test possible.

**6. 2026-09-21 14:26 — the premise fails.** Pooled leave-one-dataset-out training with no personas
beats the persona baseline by **+0.178** (0.8545 vs 0.6763, layer 40, mean pooling, seed 42), and the
fixed-budget arm (0.8500) shows the gain is diversity, not volume. *Forced by:* a newly-read paper
(F-78). This is the project's best-tested result and it says the method under study is unnecessary.

**7. 2026-09-22 00:32 / 2026-09-23 — attribution corrections.** F-97 shows the draft's headline
(0.904/0.916) is built on the owner's knowledge-denial bank, not the paper's contrast (0.791/0.832), and
F-110 shows the 70B scale rung was swept with the wrong bank because two folders hold a file called
`axes_vectors_s3.pt` — turning "essentially nil against raw" into **−0.013**. *Forced by:* the audit's
zero-shot fingerprint check. **A naming collision survived four findings.**

---

## 4. THE CONTROL TABLE, IN PLAIN TERMS

**Source:** `audit/phase3/CONTROLS.md`, built from `run-coords/zeroshot_nulls.json` (**1,000 random
draws per cell**, persisted, per dataset, with p-values — the earlier audit note that this existed
"only in a console log from 2026-09-16" is **retracted**). Consolidated table:
`audit/phase2/controls.csv`, **29,414 rows** (42 zero-shot null rows + 29,372 trained-control rows),
built by `audit/scripts/phase3/w4_build_controls.py`.

### 4.0 The sign convention — read before quoting any number below

The audit was caught **five times** by a number quoted without its convention. All of the following
hold for every number in this section.

- **`value` is the SIGNED AUROC**, in [0, 1]. **0.5 is chance. Below 0.5 means the direction is
  inverted on that dataset** — it ranks lies below honest responses. That is a real, reportable result
  here, not an error.
- **The null is direction-oriented.** The stored fields are `random_p95_auroc_dev` and
  `random_max_auroc_dev`: percentiles of **max(a, 1−a)** over the 1,000 draws — i.e. the null asks "how
  well does a random direction separate the classes *in whichever direction suits it*".
- **Therefore the quantity comparable to the null is `oriented_value` = max(auroc, 1−auroc)**, not the
  signed value. **An inverted axis can legitimately beat the null** — and that is exactly what happens
  on sycophancy.

### 4.1 The deception axis against its own 1,000-draw null — 9 deception datasets

Two cells: **layer 40 with mean pooling over the response**, and **layer 44 reading the last token**.
Signed AUROC; "beats p95?" compares `oriented_value` against the per-dataset 95th percentile of the
1,000 random draws.

| dataset | L40, mean | beats p95? | L44, last | beats p95? |
|---|---:|---|---:|---|
| ai_liar | 0.9367 | **yes** | 0.9475 | **yes** |
| insider-trading | 0.9803 | **yes** | 0.9724 | **yes** |
| instructed-deception | 0.9709 | **yes** | 0.8946 | **yes** |
| roleplaying | 0.7842 | **yes** | 0.8696 | **yes** |
| sandbagging_v2 | 0.8454 | **yes** | 0.8600 | no |
| convincing-game | 0.7884 | no | 0.7735 | no |
| harm-pressure-choice | 0.5439 | no | 0.5511 | no |
| **harm-pressure-knowledge-report (HP-KR)** | **0.6436** | **no** | **0.5830** | **no** |
| mask | **0.4669 — inverted** | no | **0.4417 — inverted** | no |

**Reading:** the deception axis beats a random direction on **5 of 9 datasets at layer 40 / mean
pooling**, and **4 of 9 at layer 44 / last token**. On the other four it is **inside the null** —
including **HP-KR, the collaborator's draft's central dataset**, where the null's p95 is **0.9377**
against the axis's **0.6436**. On `mask` the axis is **inverted at both cells** (0.4669 and 0.4417,
both below 0.5).

### 4.2 Sleeper-agent datasets — layer 44, last token

| dataset | deception axis (signed) | null p95 | beats? |
|---|---:|---:|---|
| gender-secret | **0.3512 — inverted** | 0.7630 | no |
| soft-trigger-gender | 0.9128 | 0.9257 | no |
| soft-trigger-greeting | 0.8738 | 0.9234 | no |
| soft-trigger-time | 0.8769 | 0.9271 | no |

**0 of 4.** AUROCs of 0.87–0.91 look impressive and sit **inside** a null whose p95 is 0.92–0.93: these
datasets are easy for *any* direction. This is the clearest demonstration in the project of why a null
matters — a headline number with no null attached is uninterpretable here.

### 4.3 `sycophancy_mt` — the inversion, and who actually reads capitulation

`sycophancy_mt` = 1,182 rows of multi-turn capitulation under pushback, generated 2026-09-16.
Per-cell null p95 = **0.6304**. Signed AUROC and its oriented value:

| direction | signed AUROC | oriented | null p95 | beats? |
|---|---:|---:|---:|---|
| **sycophancy axis** | **0.3131** | 0.6869 | 0.6304 | **yes — significantly INVERTED** |
| assistant axis | 0.7102 | 0.7102 | 0.6304 | **yes** (with the caveat below) |
| sycophancy − assistant | 0.3530 | 0.6470 | 0.6304 | yes |
| deception axis | 0.4089 | 0.5911 | 0.6304 | **no** |

**Stable across all four stored cells.** Two things follow, and they are the DF-31 correction with
numbers attached:

1. **The sycophancy axis is significantly inverted.** It detects capitulation with the sign reversed.
   That is a finding, not a failure to detect. 0.313 is far outside every null — per-cell and
   best-of-layers alike.
2. **The deception axis is inside the null on `sycophancy_mt` in every cell.** So "worse than a random
   direction" is true of the **deception** axis, not the sycophancy one. The collaborator briefing
   originally said the opposite; it has been corrected.

**QUALIFICATION on the assistant-axis row** (narrowed 2026-09-23 after batches W6G and W6F2 and the
lead's own check). Two comparators were raised against it:

- **Best-of-layers null — SETTLED in the row's favour.** `run-coords/q10_sycophancy_axis.json` stores
  `random.mean.best_of_layers_p95 = 0.69355` (max 0.72325) and `random.last.best_of_layers_p95 =
  0.68325`. **0.7102 clears 0.69355**, and the axis's own best-of-layers value, **0.7263 at layer 24**,
  clears even the **maximum** of 1,000 draws. Verified by the lead directly from that file.
- **Length control — NOT settleable.** A length-only scorer reads **0.664** on `sycophancy_mt`, so most
  of 0.710 is available from response length alone. **No stored file holds per-row projections**, so the
  length-residualised version cannot be computed from what exists. This stays open, and it is the reason
  not to call the row a clean win.
- **Scope condition:** the result is **paper16-bank-specific**. The `own100` assistant direction never
  exceeds **0.6161** and clears nothing.

**So: quote the assistant-axis row with the length caveat attached, never as an unqualified positive.
The inversion result for the sycophancy axis is unaffected either way.**

### 4.4 Trained controls — and two joins that must be done correctly

29,372 rows cover the comparators `raw`, `datapca`, `datapca@2`, `random`, `random@2` and `length`
across every store × layer × pooling × probe × seed in `layer_sensitivity.csv`. Use these for any
"feature X beats the matched control" claim. Both of these have already caused errors:

- **`raw` and `datapca` must come from the ROOT store.** A bank store carries its own `datapca` column
  against which `axes` wins **0 of 80** — the wrong comparator.
- **A trained random probe is not a zero-shot random direction.** They are different objects in
  different rows (`control_type = trained_control` vs `zeroshot_random_null`). Comparing across them is
  the error behind the briefing's old "0.541".

### 4.5 Seed coverage — a limitation this table exposes

Settled by `audit/scripts/phase3/p3_seed_coverage.py`. **Seven** stores (not the three previously
reported) have seeds 0–4 carrying fewer features than seed 42:

| store | what seeds 0–4 are missing |
|---|---|
| `<root>`, `passA`, `passC` | **8 features**: `axes_resid`, `datapca`, `datapca@2`, `length`, `pc:unified`, `pc:unified@2`, `random`, `random@2` |
| `banks/hf100`, `banks/own100`, `banks/paper16_famcentred`, `banks/paper16_shared` | 2: `axes_resid`, `pc:unified@2` |

Safe to average over all six seeds: `axes`, `pc:deception`, `raw` in root/passA/passC; `axes`,
`pc:deception`, `pc:unified` in the bank stores.

**The consequential reading is the opposite of the original worry.** The worry was that existing
six-seed means might be invalid. **No claim anywhere takes a multi-seed mean** — a grep over the ledger,
`claims.csv`, the claim notes and the thesis document returns nothing, so nothing existing is wrong.
What the gap *does* mean:

> **Every control comparison in `passA`, `passC` and `<root>` — axes vs `datapca`, vs `random`, vs
> `length` — exists at seed 42 only.** The raw-vs-axes comparison replicates across six seeds; the
> comparisons against controls do not replicate at all.

This does not weaken the negative results (a single-seed control comparison that comes out negative is
still negative), but the **size** of every control margin in those three stores has no seed-level error
bar. It belongs in `LIMITATIONS_AND_FUTURE_WORK.md`.

---

## 5. WHAT A READER SHOULD TAKE FROM THE CONTROLS

The deception axis is a real signal on some datasets and nothing at all on others: at layer 40 with mean
pooling it beats its own 1,000-draw random-direction null on 5 of 9 deception datasets (4 of 9 at layer
44 / last token), sits inside the null on the other four — including HP-KR, where the null's p95 is
0.9377 against the axis's 0.6436 — and is inverted on `mask` (0.4669 signed) and on all four
sleeper-agent datasets at layer 44 / last, where AUROCs of 0.87–0.91 lose to nulls of 0.92–0.93.

The single clean positive is a negative one: on real capitulation (`sycophancy_mt`) the sycophancy axis
reads 0.3131 signed — **significantly inverted**, well outside every null — which means the axis
measures something systematically opposite to the label, not noise.

Two caveats travel with the table and must not be dropped: response length alone scores 0.664 on
`sycophancy_mt` and the length-residualised version cannot be computed from stored data, so the
assistant axis's 0.7102 is not a clean win; and in the root, `passA` and `passC` stores every
control comparison exists at **seed 42 only**, so the margins have no seed-level error bar.

---

## Evidence paths cited above

`audit/TIMELINE.md` · `audit/phase3/CONTROLS.md` · `audit/phase2/controls.csv` ·
`audit/scripts/phase3/w4_build_controls.py` · `audit/scripts/phase3/p3_seed_coverage.py` ·
`run-coords/zeroshot_nulls.json` · `run-coords/q10_sycophancy_axis.json` ·
`run-crossgen/significance.json` · `run-crossgen/kd11/significance.json` ·
`run-crossgen/kd11/matrices/mean_L40_s42.json` · `run-premise/premise_L40_mean_s42.json` ·
`audit/claims.csv` · `audit/experiments.csv` · `audit/DOC_FIXES.md` (DF-18, DF-31, DF-49) ·
`audit/PROGRESS.md` (RF-1…RF-16) · `audit/phase1/axes_old_vs_fp32.txt` ·
`audit/phase1/bank_fingerprint_scale70b.txt` · `audit/phase1/anomalies.md` ·
`audit/phase1/labpc/log_index.tsv` · `audit/phase1/summary_keyed_diff_sample.txt` ·
`audit/phase1/significance_keyed_diff_sample.txt` · `audit/phase1/sessions/findings_targets.csv` ·
`audit/phase0/draft_report.md` · `docs/thesis/persona-coordinates-70b-results.md` ·
`kd/docs/FINDINGS.md` · `cap/docs/DESIGN.md` · `docs/ONBOARDING.md`

---

## LEAD'S CORRECTION TO THIS DIGEST (2026-09-23)

This digest was written from `TIMELINE.md`, which predates the end of Phase 3 and all of Phase 4.
**Two counts in it are stale** (lines 36 and 301):

- *"692 claims, of which **231** were checked"* and *"Phase 2 checked **231** of 692"* — **231 was the
  Phase 2 close.** Waves 5 and 6 carried it to **424 of 692 checked** (268 unchecked, 38.7%).
- The verdict split quoted at line 301 is likewise a Phase 2 snapshot.

**Current, from `audit/claims.csv` after the Phase 4 ledger fixes**
(recomputed by `audit/scripts/p5_stats.py`):

| | n | share |
|---|---:|---:|
| HOLDS | 297 | 70.0% |
| PARTLY HOLDS | 119 | 28.1% |
| FAILS (DOES NOT HOLD / REVISED) | 8 | 1.9% |
| **total checked** | **424** | of 692 |

By source: F 110 · O 103 · M 97 · D 61 · S 35 · H 10 · A 8. Owner 353, collaborator 71.

**This digest also predates Phase 4 entirely** (0 mentions). Phase 4 re-derived 161 claims blind and
measured the audit's own error rate at **6.5%**, corrected **7 verdicts**, and produced two findings
that bear directly on section 1's framing:

- **D-158** — the draft says it does not compare against a pooled multi-dataset probe; **the
  comparison exists on the collaborator's own branch**, committed 2026-09-20 20:05 as *"Leave-one-
  dataset-out: the fair baseline for the generalization claim"*. This is the same result section 1
  calls *"the project's largest result and it is not in the collaborator's draft"* — Phase 4 shows he
  had run his own version of it two days before compiling the draft.
- **DF-71** — F-61's "+0.006 / twelvefold" is a script-bug artifact; the real figures are **+0.0093**
  and **8.1x**.

Use the corrected counts and these two findings when quoting this digest.
