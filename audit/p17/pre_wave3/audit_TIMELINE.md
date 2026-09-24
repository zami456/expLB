# TIMELINE — the project's story in order

*Audit Phase 3, workstream P3-E. Written 2026-09-23.*

**What this is.** A narrative of what was tried, what came out, what failed, and what changed —
from the first commit of this line of work (2026-09-07) to the audit (2026-09-22 →). It is meant to
be read start to finish with no other file open. Every technical term is defined the first time it
appears.

**How each statement is marked.**

| tag | meaning |
|---|---|
| **VERIFIED** | checked in a raw file, a log, or git during this audit; the path is given |
| **DOC-ONLY** | stated only in prose (a findings entry, a memory note, a Claude transcript) and not re-derived here |
| **CONFLICT** | two sources disagree; both are shown |
| **NOT-FOUND** | looked for, not found |
| **INFERRED** | the audit's interpretation; the reason is given |

**Times.** All times in this document are **lab-local (UTC+6)**, the clock the lab PC, the laptop and
git all use. Git commit dates carry `+0600` (VERIFIED, `git log --pretty=%ai`), so they need no
conversion. The owner's Claude session transcripts timestamp in **UTC**, so a transcript time of
11:31Z is 17:31 here; every transcript time quoted below has already been converted (VERIFIED, and
stated as a red flag in `audit/PROGRESS.md`).

**A warning about dates.** *Commit dates are batch dates, not run dates.* The owner committed in
bursts — findings F-01 to F-66 all entered git in a single commit, `6432257` on 2026-09-21 03:26,
although they are dated 09-19 to 09-21 (VERIFIED, `git log`; flagged as RF-9 in
`audit/PROGRESS.md`). **2026-09-20 produced 51 findings and no commit at all** — VERIFIED: 51
entries carry that date in `kd/docs/FINDINGS.md`, and `git log asix` has no commit between
`a796eaf` (09-19 14:01) and `6432257` (09-21 03:26). So this timeline is
anchored on **internal log timestamps from the lab PC** (`audit/phase1/labpc/log_index.tsv`, 242
logs with their own first and last line times), which are the best evidence of when something
actually ran, and uses git only for when something was written down.

**The pace, for scale.** Findings by the date written on them (VERIFIED, counted from
`kd/docs/FINDINGS.md`): **09-19: 11 · 09-20: 51 · 09-21: 33 · 09-22: 13 · 09-23: 5** (plus one
marked "earlier sessions"). The whole numbered ledger covers five days. Everything before 09-19 —
the pipeline, the 100-role run, the eleven research questions, the cross-generalisation grids — was
written up in `docs/thesis/persona-coordinates-70b-results.md` and in Claude memory notes instead,
with no F-numbers, which is why that half of the project is easy to lose.

---

## 1. The arc, in one page

**What the project set out to do.** Replicate one paper — Mahadik & Skapars, *Do Linear Probes
Generalize Better in Persona Coordinates?* — on a model 23× larger than theirs, with no local GPU.
(DOC-ONLY, `docs/ONBOARDING.md:10-20`.) The paper's idea: prompt a model to adopt contrasting
personas ("a scrupulously honest assistant" vs "a manipulative one"), record the internal activity
each persona produces, and extract a small number of directions in that activity — "persona axes".
The claim is that a lie detector built in those few coordinates transfers to new datasets better
than one built on the model's raw internal state. The paper reports **+0.10 AUROC** at 3B
parameters. (AUROC = the chance a randomly chosen lie scores above a randomly chosen honest
response; 0.5 is chance, 1.0 is perfect.) The replication target was Llama-3.3-70B-Instruct, run
remotely on NDIF, over **15,278 responses across 10 datasets** — 5.5× the paper's corpus
(DOC-ONLY, `docs/thesis/persona-coordinates-70b-results.md:24-35`).

**How the question changed, three times.**

1. **From "does it replicate?" to "what does it actually measure?"** By 2026-09-17 the replication
   had an answer: the persona features beat a raw probe only inside a narrow band of network
   depth, only on some datasets, and never for the paper's own headline feature. Nothing survived a
   significance test that treats *datasets* rather than *rows* as the unit (VERIFIED, F-04,
   `run-crossgen/kd11/significance.json`: 1 of 384 tests significant). The work turned toward
   diagnosis.
2. **From "the axes are noisy" to "the axes encode the wrong thing."** Two datasets broke every
   detector in the same way. The diagnosis, reached 2026-09-20: the persona axes encode **register**
   — confident assertion vs hedging, agreeable vs challenging tone — not the deception the labels
   mark (DOC-ONLY, F-13). The project then built two new persona banks specifically to test that
   diagnosis: a **capitulation** bank (worked, locally) and a **sarcasm** bank (failed its own
   gates, abandoned).
3. **From "personas help" to "personas are not needed."** On 2026-09-21 the owner ran the cell
   nobody had run: train one probe on the pooled data of ten datasets and test on the eleventh,
   with no personas at all. It beat **every** persona result in the project, by +0.178 AUROC, and
   it was the widest-clearing significance result anyone had produced (DOC-ONLY, F-79/F-80). This
   is the project's largest negative result and it is not in the collaborator's draft.

**Where it is now.** The collaborator compiled a 19-page draft paper on 2026-09-22 13:57 (VERIFIED,
PDF metadata, `audit/phase0/draft_report.md §1`). Its headline — a one-dimensional persona contrast
reaching 0.904 / 0.916 mean off-diagonal AUROC with no labels — reproduces exactly from his own
files, but **the contrast that produces it is built on the owner's knowledge-denial personas**, not
the paper's; the paper's own 8-vs-7 contrast reads 0.791 / 0.832 (DOC-ONLY, F-97; claim `A-04` in
`audit/claims.csv` is verdict "MATCHES HIS DATA … but the ATTRIBUTION is wrong"). Neither party had
a full picture of the other's results. The audit began 2026-09-22 18:44 to produce one.

---

## 2. Before this line of work — the inherited repository

**2025-11 → 2026-05 · Liars' Bench.** The repository is the Cadenza Labs deception benchmark. Its
history to 2026-05-07 is by Kieron Kretschmar and Walter Laurito: dataset generation, black-box
detectors, the Pacchiardi elicitation baseline (VERIFIED, `git log asix`). None of this is the
owner's work; it is the substrate the evaluation datasets come from.

**2026-07-11/12 · the branch point.** `52281fa` vendors the Apollo probe code as a regular
directory; `2e2a785` is the **merge-base** of the two branches in play (VERIFIED, `git log`;
confirmed in `audit/PROGRESS.md`, "Branch topology"). Everything after this splits into the owner's
`asix` (137 commits, 2026-09-07 → 09-22) and the collaborator's `origin/feat/persona-pc-ood`
(34 commits by `zami456`, 2026-09-04 → 09-22).

**2026-09-04 · the collaborator's engineering, before either of them had persona results.** Six
commits that day build a resumable per-example activation cache and token-budget batching for the
probe pipeline (VERIFIED, `git log origin/feat/persona-pc-ood`: `e391b3e`, `45c733d`, `3395616`,
`12910b2`, `1a960b3`, `9b33133`). The owner later ported this into her own pipeline at `ed8daca`
(2026-09-15 02:12) — the first dependency in the his→her direction (VERIFIED, commit subject "Port
the resumable activation cache and token-budget batcher").

---

## 3. 2026-09-07 → 09-12 · Building the pipeline

**09-07 21:40 / 09-08 05:55 · the project starts.** `fd78adf` and `59cdaba` add the
`assistant_axis` submodule — Lu et al.'s "Assistant Axis", a direction in activation space that
measures how "Assistant-like" a model is behaving (VERIFIED, `git log`). The first plan was to run
that pipeline, not the persona-coordinates one.

**09-10 06:41 → 09-12 · first contact with the hardware.** Five Claude sessions in two days set up
the remote model access and the lab PC (VERIFIED, `audit/phase1/sessions/sessions_index.tsv`,
sessions `47aabfcb` … `638f732c`). Smoke tests of generate → judge → extract run 09-10 12:35 →
09-12 01:54 (VERIFIED, `audit/experiments.csv` row `O01-aa-smoke`, from laptop and lab file mtimes).

Two engineering facts from this window shaped everything after (DOC-ONLY, `CLAUDE.md:16-24`, and
the pattern is visible in the logs):

- Long-running work has to be a **Windows Scheduled Task**, not a process launched over SSH, because
  OpenSSH kills the process tree when the session ends. Almost every later run is a scheduled task
  with a `.status` file — which is why the audit can date them.
- A remote trace body must be defined in the script that runs it, or NDIF rejects it.

**09-12 05:43 → 09-15 12:51 · the first real run: `own100`.** 100 roles × rollouts on
Llama-3.3-70B, judged, per-role vectors, then an Assistant Axis and persona axes. Three parallel
shards ran 09-12 17:47 → 09-14 13:49 (VERIFIED, `logs/shard0|1|2.log`, 1.0–1.4 M lines each,
`audit/phase1/labpc/log_index.tsv`). It finished 09-15 12:51:42 with "vectors=100 axis.pt=True
axes.pt=True" (VERIFIED, `run/DELIVERABLE_READY`, quoted in `audit/experiments.csv` row
`O02-aa-own100`). One of the 100 roles has a response file but no score file (VERIFIED, same row).

---

## 4. 2026-09-14 → 09-15 · The pivot, and the first bug that cost real work

**09-14 14:22–15:50 · the reframing.** In session `6d4b2181` the work is re-scoped: drop
replication language for the Assistant Axis pipeline, and make a **paper-matched persona bank** the
controlled comparison instead (DOC-ONLY, `audit/phase1/sessions/dataset_events.csv`, rows at
2026-09-14T14:22 and 14:27: *"Reframe as a new experiment inspired by the method… make the
paper-matched persona bank the controlled comparison"*; *"accept, but as a contribution, not a
repair"*). This is the moment the project becomes a persona-coordinates study.

**09-14 21:26 · a truncation bug found.** Long inputs were cut from the front, so the assistant's
own words — the span the method must read — fell past the cut and were discarded by four separate
filters (DOC-ONLY, `dataset_events.csv` 2026-09-14T21:26). Fixed at `7842bf6` (09-15 03:25, "keep
the response when truncating, and count what is dropped") (VERIFIED, `git log`).

**09-15 03:46 → 11:13 · the `paper16` bank.** The paper's 16 personas plus extras (26 roles) are
reconstructed and run on the paper's own question sets (VERIFIED, `audit/experiments.csv` row
`O03-paper16-orig`; laptop log `paper16_supervisor.log` ends "DONE — both axes files present"
11:13:34). This bank is the backbone of everything for the next four days.

**09-15 17:41 → 09-16 01:30 · the bfloat16 bug, and the rebuild.** The masked reductions that
average activations over the response were being done in the model's own bfloat16 precision, which
silently produced a wrong-last-token variant. Commit `ae31943` (17:41) does every masked reduction
in fp32; `0547fc1` (21:47) calls it "the same bug that corrupted the eval pass" (VERIFIED, `git
log`). The paper16 vectors were re-extracted 09-15 21:45 → 09-16 01:30 (VERIFIED,
`logs/rebuild_paper16_vectors_s3_fp32.log`, `rb_paper16_s3.log`, `rb_paper16_all.log`).

**Why this matters, and what the audit measured.** Two versions of the paper16 vectors now exist,
and different findings read different ones (VERIFIED, RF-12/RF-15 in `audit/PROGRESS.md`). The
audit measured the difference directly: the four axes have **cosine 1.0000** between old and new at
every layer, with at most 0.5–0.9% difference in length, and PCA components differ by at most 0.009
(VERIFIED, `audit/phase1/axes_old_vs_fp32.txt`, script `audit/scripts/p1_axes_old_vs_fp32.py`). So
the bug's effect on the numbers is under ~1% — *except* that it flips the sign of a few individual
principal components at a few layers (not at layers 20, 40 or 44). **The bug was expensive in time,
not in results.**

An equivalent rebuild for the `own100` bank got 1 of 100 roles done and was abandoned (VERIFIED,
`audit/experiments.csv` row `O05-own100-fp32`, status "partial"; `audit/phase1/anomalies.md`
OR-11). So the `own100` numbers still rest on the buggy reduction. INFERRED: this is a real
inconsistency between banks, but `own100` carries no headline result.

---

## 5. 2026-09-15 → 09-17 · The eleven questions

This is the project's most productive stretch: the research questions Q1–Q11, each with its
expected outcome written down *before* the run (DOC-ONLY, `docs/ONBOARDING.md:30-33`; the plan file
is `~/.claude/plans/harmonic-cuddling-kay.md`, byte-identical to the lab/Drive copy — VERIFIED,
md5 `ab721b1b…`, `audit/PROGRESS.md`).

**The three activation passes** (the stores everything else reads):

| pass | what it is | when it ran |
|---|---|---|
| **B** | 21 layers, the full 15,278-row corpus, pooled per response | 09-15 12:58 → 21:31 (VERIFIED, `coords_passB.log`) |
| **A** | all 80 layers, 200 rows per dataset | 09-16 02:07 → 02:55 (VERIFIED, `passA_run.log`) |
| **C** | per-token, 6 layers, same 200-row sample | 09-16 12:50 → 13:56 (VERIFIED, `passC_run.log`) |

**09-16 02:55 → 18:23 · a corpus built from scratch.** The evaluation set had no sycophancy data,
so one was generated: first single-turn (`syco_gen.log`, 02:55 → 05:38), then, when single-turn
proved uninformative, **multi-turn capitulation under pushback** — the model is challenged and
either holds its position or folds (`syco_mt_gen1.log` 14:10 → 15:54, `gen2` → 18:01, judged
→ 18:23) (VERIFIED, log index). The finding that forced the redesign is F-15, "multi-turn behaviour
is invisible to single-turn elicitation" (DOC-ONLY). The result, `sycophancy_mt`, 1,182 rows, later
becomes the dataset that breaks the method.

**09-16 14:39 → 09-17 02:09 · the analyses.** Per-token scoring (Q8), persona-count curves (Q9),
label-semantics splits (Q5), response-length controls, the sleeper-agent arm (Q11), the layer sweep
(Q6), bootstrap significance, and random-direction nulls — twenty short runs in thirty-six hours
(VERIFIED, log index rows `pertoken.log` … `q3q4.log`; `audit/experiments.csv` rows O18–O34).

**The eleven answers, as the owner wrote them on 09-17** (DOC-ONLY,
`docs/thesis/persona-coordinates-70b-results.md:42-56`; several have since been amended — see §11):

- **Q1** The paper's persona set removes most of the Assistant-Axis contamination of the deception
  axis: −0.66 → −0.37. *Confounded by question set — see §6.*
- **Q2** Persona features beat a raw probe only in a depth window (~36–48 of 80 layers), only on the
  nine deception datasets, and never for the paper's own PC-projection feature. Adding the
  sycophancy dataset removes the advantage at every layer.
- **Q3** Sample-efficiency behaviour is probe-dependent, not significant.
- **Q4** Erasing the assistant direction costs only 0.005–0.018 AUROC — there is deception signal
  beyond "assistant-ness".
- **Q5** Label semantics does **not** predict transfer; the apparent clustering is shared provenance.
- **Q6** Layer 20 vs 22 is immaterial; moving to 40 is worth ~+0.025 for raw probes and ~+0.113 for
  persona methods.
- **Q7** The judge barely matters: a 70B and a 405B judge agree 91.2%.
- **Q8** Zero-shot, top-decile token scoring beats mean pooling (+0.053); *trained*, pooled probes
  beat per-token probes at every layer. The two halves disagree.
- **Q9** More personas buy reliability, not accuracy; the curve plateaus at 4 per side.
- **Q10** The sycophancy axis is **significantly inverted** on real capitulation data.
- **Q11** Trained probes flag sleeper-agent lies at 0.96 even across triggers — but what they detect
  is a false answer, not deception.

**The honest summary the owner wrote the same day** (DOC-ONLY,
`docs/thesis/persona-coordinates-70b-results.md:58-77`): *"The headline advantage is fragile… no Q2
or Q6 difference survives a bootstrap over datasets."*

---

## 6. 2026-09-17 → 09-18 · Two confounds found, and the cross-generalisation grid

**09-17 16:10 · the question-family confound.** Checking the persona bank's own geometry, the first
principal component (47% of the variance) turned out to separate personas by **which question set
they answered**, not by what they are — cosine −0.98 with the question-family indicator (DOC-ONLY,
`dataset_events.csv` 2026-09-17T16:10; memory note `unified-pc1-is-question-family`). The `paper16`
bank had been built on the paper's questions only (1,400 rollouts), so `paper16` vs `own100`
differed in more than personas. **Q1's answer was confounded.**

**09-17 22:31 → 09-18 03:14 · the fix, run overnight.** All 26 personas regenerated on one shared
question set — 10 deception + 10 sycophancy questions, 2,600 rollouts (VERIFIED,
`paper16_shared_chain.log`, `paper16_shared_generate.log` 22:32 → 01:44, judge → 02:22, vectors
→ 03:13). This is `paper16-shared`, and it is the bank later exported to the collaborator.

**09-17 21:12 · a crash worth recording.** The first full cross-generalisation run died: `crossgen.status`
records `PYTHON_EXIT -1` at 21:18:44, killed during layer 12 of 21 (VERIFIED,
`audit/experiments.csv` row `O40-crossgen-retrain-attempt`). **NOT-FOUND: the cause.** It was
restarted successfully the same evening.

**09-17 21:28 → 09-19 · the 10×10 grid.** "Cross-generalisation" here means: train a detector on one
dataset, test it on all ten, for every choice of training dataset — a 10×10 matrix whose
off-diagonal mean is the transfer score. Run for the root (raw) features and seven persona banks,
21 layers, 6 seeds, two pooling methods (VERIFIED, `crossgen_full_A|B|C.log`, `crossgen_banks.log`;
`audit/experiments.csv` rows O41–O46).

**The second confound, and the result that reframed the project.** Against a **matched control** —
a dataset-PCA of the same dimensionality, i.e. "any k directions fitted to this data" rather than
"persona directions" — the unified persona axis reads **−0.003**. A *random* 2-d projection gains
+0.056 where the raw baseline is weak and loses 0.44 where the raw baseline is strong, reproducing
the paper's qualitative claim with meaningless directions (DOC-ONLY, F-17, source
`run-crossgen/significance.json`; memory note `matched-control-kills-the-gain`). **INFERRED: this is
the single most consequential result of the whole project** — it says the reported advantage is
mostly a low-dimensional ceiling artifact, and it is why every later claim is scored against
`datapca` and not only against raw.

**09-18 11:26 / 16:19 · the handoff.** The owner writes up everything as
`docs/thesis/persona-coordinates-70b-results.md` (`b9a497e`, 09-18 11:26 — VERIFIED, `git log`) and
exports layer-20 persona vectors and axes to Google Drive `share/` for the collaborator, 16:19 →
16:36 (VERIFIED, Drive modtimes + lab file mtimes, `audit/experiments.csv` row `O38-share-export`).

---

## 7. 2026-09-18 → 09-20 · Where the collaborator joins

**09-18 17:31 · he starts, about an hour after the export.** `7a9c8ba` "Persona vector geometry for the
PC-projection experiment", then `2731cb0` "Pool cached activations per example, and the OOD transfer
experiment" (17:36), then `a093c54` "Persona PC transfer results: the claim replicates, with a
caveat" (21:13) (VERIFIED, `git log origin/feat/persona-pc-ood`).

**What passed between them, and how it was checked.** The audit verified the handoff by hashing
files, because the two branches share **no identical file in either direction** — every dependency
runs through the lab PC, Google Drive `share/`, or his local disk (VERIFIED, RF-4).

- His `paper16_L20.pt` and `deceptive_L20.pt` are **byte-identical** to the lab's
  `share_L20_shared/` and `share_deceptive/`; only the two `.json` sidecars differ (VERIFIED,
  `audit/phase1/census_summary_rf1.txt`, RF-1 resolution).
- All 36 of his `persona including knowlede denial/*.pt` are byte-identical (md5) to the lab's
  `run-kd/vectors_s3/*.pt` (VERIFIED, same source).
- **Only the activations are his own.** He downloaded roughly 62 GB per layer (DOC-ONLY, owner
  statement recorded in `audit/phase0/draft_report.md`). This is why most of his result CSVs cannot
  be recomputed from git: the inputs are not in git (VERIFIED, RF-2).

**09-20 17:53 · his kd result, and the git lag.** `9774075` "Knowledge-denial personas fix HP-KR".
The bank he used was generated on the lab PC 09-19 20:50 → 09-20 03:04; the owner committed the
persona instruction text only at `3dfdb67`, 09-21 11:07 — **git lagged the lab run by about 1.5
days** (VERIFIED, RF-1 resolution: the system prompts stored in all 36 lab rollout files equal the
text committed at `3dfdb67`; for 9 new personas that is the *only* committed version that matches).
This is why the audit's first red flag asked whether he used a stale bank. **He did not.**

**His run of experiments** (VERIFIED from `git log` + `audit/experiments.csv` rows C01–C20): persona-PC
transfer at layer 20 (09-18 → 09-20), the same at layer 40 (09-20 16:17, commit subject: *"several
layer-20 findings do not survive"*), leave-one-dataset-out baselines (09-20 20:05), the extended
36-persona bank at layers 20/40/44 (09-20 → 09-21), a "more rollouts per persona" arm that did not
help (09-21 13:40), the follow-up-token probe (09-21 22:36), and per-token position reads (09-22
01:21). He corrected three claims in place after pushing (`e3f2258`, `f386a72`, `45b16d9`) —
VERIFIED, `audit/phase0/branch_compare.md`.

---

## 8. 2026-09-18 → 09-20 · The HP-KR problem, and the `kd` experiment

This is the branch of work that gave the project its second half, and the collaborator's draft its
title.

**The problem.** `harm-pressure-knowledge-report` (HP-KR) is a dataset where the model is asked
whether it knows a dangerous answer. Every detector in the project **inverted** on it — scored below
chance. Understanding why took three days.

**09-18 20:28–20:35 · the structure of the confound is worked out.** HP-KR's label is
`deceptive = knows AND denies`, and the fourth cell of the 2×2 — *doesn't know but claims it does* —
has **zero rows**. On the text-matched slice "deceptive" and "knows" are literally the same variable
(DOC-ONLY, `dataset_events.csv` 2026-09-18T20:35 and 2026-09-20T15:56). So the dataset **cannot
attribute its own results** (F-02).

**09-19 · the measurements that follow.**

- **F-01:** a single regular expression — does the response contain a knowledge denial — scores
  **AUROC 0.913** against HP-KR's own label (DOC-ONLY; supersedes an earlier 0.870 that had been
  computed over all three generating models' rows instead of only the Llama rows the probes read).
- **F-03 (the strongest positive result the project ever had):** build a **text-matched slice** of
  258 rows where every response says "I do not know the answer", so the wording shortcut is
  impossible. On that slice the trained persona axes go from **0.300 to 0.752**, against a raw probe
  at 0.715 — the result reverses (DOC-ONLY, source `run-crossgen/kd11/matrices/mean_L40_s42.json`).
  Its own caveats, written at the time: 258 rows, **one layer**, one seed, row-level bootstrap only.
- **F-04 (the ceiling that constrains everything):** across **384** dataset-level significance tests
  in the 11-dataset run, exactly **1** is significant after multiple-comparison correction. The
  binding constraint is having ~11 datasets, not ~15,000 rows — row-level confidence intervals look
  tight and mean little (DOC-ONLY, source `run-crossgen/kd11/significance.json`). **This is why so
  many later results are reported as "does not clear".**

**09-19 17:11 → 09-20 03:04 · the `kd` bank.** Two teacher-forced "factor probes" separate the two
entangled factors — does the model *know*, and does it *deny* — and a WMDP knowledge eval of 1,459
items scores what the model actually knows (VERIFIED, `kd_knowledge.log` from 09-19 17:11,
`kd_extract.log` 19:33). Then a 36-persona bank: the 26 paper16 personas plus **10 new
knowledge-denial and candour personas**, 5,400 rollouts, judged, vectors extracted (VERIFIED,
`kd_generate.log` 09-19 20:39 → 09-20 00:18; `kd_judge.log` 00:20 → 01:33; `kd_vectors_s3.log`
02:26 → 03:04).

**09-20 · the register diagnosis (F-13).** Two datasets fail the same way:

| dataset | what the axis encodes | what the label marks | as target | as source |
|---|---|---|---:|---:|
| HP-KR | confident assertion vs disclaimer | denying held knowledge | 0.427 | 0.300 |
| `sycophancy_mt` | agreeable vs challenging register | capitulation under pressure | 0.355 | 0.379 |

A random direction beats the axes on both (DOC-ONLY, F-13). Removing `sycophancy_mt` from the
average lifts the raw probe by +0.025 but the persona axes by **+0.061** — persona features are hurt
about 2.5× more. **This is the register diagnosis, and it becomes the project's main hypothesis.**

**One number to be careful with.** The figure **0.374**, quoted in nine documents as "the paper's
sycophancy axis on real sycophancy data", is in fact the **deception** axis's first principal
component. The sycophancy axis reads **0.313**. All three independent sources agree and none gives
0.374 for the sycophancy axis (VERIFIED, `audit/DOC_FIXES.md` DF-31, with the code line
`27_crossgen.py:411-416` that causes the mislabel). The correction makes the argument *stronger*:
0.374 sits inside the random-direction null, while 0.313 sits beyond it.

---

## 9. 2026-09-20 · The scale ladder, and a claim corrected twice

The idea: if one p-value on eleven datasets cannot settle anything (F-04), run the same thing at
three model sizes and look for *agreement of shape across independent models* instead (DOC-ONLY,
`kd/docs/FINDINGS.md:1664-1667`).

**09-20 12:41 → 16:56 · the small rungs.** Llama-3.1-8B and Llama-3.2-3B: 2,600 rollouts each,
judged by the same 70B judge held constant, vectors extracted (VERIFIED, `scale_3b_generate.log`
12:41 → 15:51, `scale_8b_generate.log` 12:59 → 14:16, and the four vectors logs through 16:56). The
3B extraction failed on every role on the first attempt — the cause was batch size, not sequence
length (DOC-ONLY, F-33).

**09-20 16:59 → 23:40 · evaluation activations and the grids** (VERIFIED, `scale_coords_3b|8b.log`,
`crossgen_scale_3b|8b.log`).

**Then the claim was made, and corrected twice on the same day.** This is the clearest example in
the project of a number changing under scrutiny, and it is worth following in full.

| step | claim | 3B | 8B | 70B | what was wrong |
|---|---|---:|---:|---:|---|
| **F-58** (09-20) | "the paper replicates at 3B/8B and the gain dies at 70B" — gain over the raw probe | +0.089 | +0.086 | +0.026 | — |
| **F-60** (09-20) | correction 1 | — | — | — | The paper evaluates **MASK**, not `HarmPressureChoice`. We matched 4 of 5 datasets and ran the wrong one for the fifth. Direction unaffected, framing too strong. |
| **F-63** (09-20) | correction 2 — *same* five datasets, *same* k=3 | +0.020 | +0.013 | +0.013 | F-58 had compared **different dataset sets AND different k** (the rungs ran k=3, the 70B matrix read was k=5, and the result was already known to be k-dependent). Against the raw probe the gain is small and **flat**, not declining. Against the matched control it is +0.202 / +0.115 / +0.079. |
| **F-96** (09-21) | the margin is a **single-layer** reading | +0.120 (mean of 7 layers) | +0.103 (8) | only 1 layer | F-63's 3B layer was the **best of its seven**. Layer-averaged, the decline nearly flattens. |
| **F-100** (09-22) | the 70B gap closed with a 21-layer sweep | +0.120 | +0.103 | +0.075 | The decline survives layer-averaging, but its steepness was a 3B layer-choice artifact. |
| **F-110** (09-23, written by the audit) | the 70B sweep used the **wrong persona bank** | — | — | **+0.054** vs control, **−0.013** vs raw | The sweep ran on the 36-persona `kd` bank, not `paper16`, because both folders contain a file called `axes_vectors_s3.pt`. Like-for-like the decline is steeper and "essentially nil against raw" becomes negative. |

(All six entries DOC-ONLY as ledger text; **F-110's bank mix is VERIFIED independently** by the
audit's zero-shot fingerprint — the same bank on the same activations must give identical zero-shot
AUROCs, and the sweep's values are identical (10/10, |Δ|=0) to the kd bank while the layer-40 cell
matches paper16 to ≤0.0003: `audit/phase1/bank_fingerprint_scale70b.txt`, script
`audit/scripts/p1_bank_fingerprint.py`. This is RF-16.)

**The lesson the owner wrote down at the time** (DOC-ONLY, memory note `no-margin-from-one-layer`):
*"four headline margins have now failed a layer sweep; F-63's 3B +0.202 is the max of 7 layers."*

**09-20 22:52 → 22:56 · a gate that failed and stopped a plan.** The matched HP-KR slice is capped at
258 rows because the model rarely admits ignorance. The plan to grow it to 500–700 rows required
that, on items the model measurably gets wrong, it would *deny knowledge* at a rate ≥ 0.70 — written
down before the run. Measured: **5/20 = 0.250**; the fallback made it worse, 3/20 = 0.150 (DOC-ONLY,
F-59; VERIFIED that it ran, `hpkr_denial_smoke|control|margin.log` 22:52 → 22:56). *The model claims
to know answers it gets wrong, on 75% of items it answers incorrectly.* **The growth route was
abandoned.**

---

## 10. 2026-09-20 → 09-21 · The capitulation bank — the register diagnosis tested constructively

**The design.** If the persona axes fail on `sycophancy_mt` because they encode *register* rather
than *capitulation*, then a bank that varies **capitulation** with everything else held fixed should
not fail. The bank contrasts a persona that folds under pressure against one that holds its
position, with turn 1 **teacher-forced** (identical text in both arms) so only the second turn can
differ (DOC-ONLY, `cap/docs/DESIGN.md`).

**09-20 14:38 → 21:34 · six smoke tests, and one that changed the design.** The free multi-turn
version was invalidated because the capitulation persona was already *wrong in turn 1*, so the
measure did not isolate capitulation (DOC-ONLY, F-46). Teacher-forcing turn 1 fixed it: 4/4 fold vs
4/4 hold, clean separation (DOC-ONLY, F-51). VERIFIED that all six smokes ran:
`cap_smoke.log` 14:38, `cap_smoke_v2.log` 18:40, `cap_tf_smoke.log` 20:49, `cap_tf_cells.log`,
`cap_tf_control.log`, `cap_v4_cells.log`, `cap_v4_control.log`, `cap_nopersona.log` 21:32,
`cap_pair5.log` 21:33.

**09-20 22:59 → 09-21 03:16 · the full run** (VERIFIED, `cap_gen.log` 22:59 → 03:02,
`cap_judge.log`, `cap_vectors_*.log`, `cap_crossgen_*.log` → 03:16).

**09-21 03:26 · it works.** Zero-shot — no probe, no training, no labels — the capitulation axis
reads **0.745** on `sycophancy_mt` against the register axis's 0.374 (DOC-ONLY, F-65; committed as
`6432257`, VERIFIED `git log`). Then F-66: it **clears the dataset-level bootstrap**, the first
positive result in the project to do so, against both raw and both matched controls.

**And then it is bounded, twice, by the owner herself.**

- **F-67 (09-21 03:37):** the advantage is **local** — over the full 11×11 off-diagonal mean it does
  not extend past the column it was aimed at (DOC-ONLY). The commit subject also records that "the
  seventh paper16 hardcoding hid that number entirely" — a class of bug the owner hit repeatedly
  (see §13).
- **F-89 (09-21 20:34):** a **format-matched control**. Swap the six position-holding personas for
  six topically unrelated ones and the result costs only ~+0.03; removing them entirely costs two to
  four times that. So what the holding personas contribute is largely *the presence of an
  instruction*, not the disposition the instruction describes (DOC-ONLY). The commit subject is
  blunt: "the format-matched control **deflates** F-66's interpretation".

**The audit's verdict on this family:** F-65, F-66 and F-67 all come back **PARTLY HOLDS** — the
measurements reproduce, the sentences around them overstate (VERIFIED, `audit/claims.csv`).

---

## 11. 2026-09-21 · The day four claims were withdrawn

**Morning · the HP-KR top-up.** The one-layer caveat on F-03 could finally be removed. `HPKR-Topup`
ran as a scheduled task 10:53 → 11:53 and extracted **all 2,139** HP-KR rows into passes A and C,
both gates exact (VERIFIED, `hpkr_topup_passA.log` 10:53, `hpkr_topup_passC.log` 11:29;
`hpkr_topup.status`). All three stores then carried the matched slice identically — 258 rows,
124 deceptive / 134 honest (DOC-ONLY, F-71).

**Then F-03 came apart, in three steps.**

| step | what it says |
|---|---|
| **F-72** (12:33) | F-03's reversal is **layer-localised**: 0.739 at layer 40, **0.374 at layer 20** — the layer choice moves the result by ten times the claimed effect. (DOC-ONLY) |
| **F-75** (13:43) | The 80-layer profile is a 16-layer plateau containing layer 40 — the pre-registered "strengthens" outcome. Explicitly marked *provisional* pending two committed tests. (DOC-ONLY) |
| **F-76** (13:47, four minutes later) | **CORRECTION.** The plateau does not survive the dataset-level bootstrap. At the pre-specified layer the margin over the matched control is **+0.0655 with a 95% interval of [−0.0693, +0.1887]**. F-03's margin is **not significant**. (DOC-ONLY) |

Two further self-corrections the same afternoon, both by the owner against her own earlier work:

- **F-73** (13:02): the response-length claim's AUROC survives; its per-dataset framing does not.
- **F-74** (13:17): the "−0.76 ceiling artifact" statistic is **mathematical coupling** between the
  two quantities being correlated, and the clean test refutes the claim it was supporting.

**14:26 → 17:24 · the premise fails.** Prompted by a newly-read paper (F-78), the owner ran the cell
nobody had run: pool the data of ten datasets, train one probe, test on the eleventh — **no personas
at all**.

| condition | AUROC |
|---|---:|
| single-source, trained on one other dataset (the persona method's baseline) | 0.6763 |
| **pooled leave-one-dataset-out** — never sees the target | **0.8545** |
| **the same, with the row budget held fixed** (~666 rows, drawn from ten datasets instead of one) | **0.8500** |
| single-source, trained on its own data (an upper bound, not deployable) | 0.9506 |

The second and third rows together say the gain is **diversity, not data volume**.

The margin is **+0.1782, 95% interval [+0.1430, +0.2129]**, and it clears the dataset-level
bootstrap by a wide margin — the widest anything in the project has cleared (DOC-ONLY, F-79/F-80;
source `run-premise/premise_L40_mean_s42.json`, VERIFIED to exist via `premise_layers.log`,
`premise_extras.log`). It is **positive at all 21 layers**, all 21 clearing (F-81), **seed-stable**
(F-83), and it **survives falsification at 51 of 51 cells** (F-95). F-86 explains the mechanism:
single-source probe directions are nearly orthogonal, so averaging them cancels the noise — joint
training is not needed.

**INFERRED, and the single most important thing in this timeline:** the project's own strongest,
best-tested, most robust result is that **you do not need personas**. Pooled supervision on ordinary
activations beats every persona configuration produced here. The collaborator's draft cites no
findings-ledger entry at all (VERIFIED, `audit/phase0/draft_report.md §1`, "0 F-numbers cited"), so
it does not report this. Two related omissions on his side: the draft says a pooled baseline was
"not compared" although **his own branch carries `results/loo_L20` and `loo_L40`**, exactly that
baseline (VERIFIED, RF-3; `audit/phase1/anomalies.md` OR-17); and he has a rollout-count control,
`persona_ood_mod_L*`, scoring **lower** than the arm he reports (0.9066 vs 0.9158 at layer 40) under
his own heading "More rollouts per persona did not help" (VERIFIED, `audit/DOC_FIXES.md` DF-18).

**17:47 → 22:15 · controls and falsification.** The format-matched control banks generate and judge
(`cap_ctrl_gen.log` 17:47 → `cap_ctrl_crossgen.log` 20:05); the falsification sweep runs
`falsify.log` 20:28 → 22:15 (VERIFIED, log index).

**17:48 → 21:48 · the sarcasm bank: designed, piloted, abandoned.** The plan was a third
construct to test the register diagnosis. It was pre-registered as a design document before anything
was generated (`3d80071` 17:48, revised `5ced550` 19:03 after review — VERIFIED, `git log`).

- **Gate 1 passed (F-92, 21:32):** deadpan sarcasm *is* producible on-policy; the instruction
  strength needed was measured, not guessed (marker rate 0.00 at variants v4 and v5). The literature
  predicted this would fail. VERIFIED it ran: `sarc_pilot_gen.log` 21:21, `sarc_pilot_judge.log`
  21:41.
- **Gate 2 failed (F-93, 21:48):** once markers are stripped, ironic and sycophantic text must be
  near-indistinguishable. They are still separable at **0.92 by bag-of-words alone** — so the bank
  would itself be a **register axis**, the very thing it was built to diagnose. (DOC-ONLY.)

**The bank was never built.** (DOC-ONLY, memory note `sarcasm-gate3-deception-axis-is-random`; the
write-up is `docs/thesis/sarcasm-pilot-negative-result.md`.)

---

## 12. 2026-09-22 · The last day of work — and two events that were not decisions

**00:32 → 02:11 · whose axis is the headline?** The owner rebuilt the collaborator's contrast on her
own bank and compared (VERIFIED, `audit/experiments.csv` row `O60-kd-collab-bank`).

- **F-97:** the draft abstract's **0.904 / 0.916** is `ext_axis1` — a **12-vs-13 contrast built over
  the 36-persona bank that includes the owner's ten knowledge-denial personas** — not the
  persona-coordinates contrast. The paper's own 8-vs-7 contrast reads **0.791 / 0.832**. So roughly
  **44–46%** of the headline margin comes from the kd bank. (DOC-ONLY; the underlying numbers are
  VERIFIED against his own files — `audit/claims.csv` A-04, D-008, D-098 all "MATCHES HIS DATA".)
- **F-98:** rebuilt on the owner's eleven datasets, the kd bank is worth **49–77%** of the margin,
  and **both** of the collaborator's arms lose to pooled training. F-99 extends this to all 21
  layers: the kd bank contributes at 21 of 21, median 63%.

**00:52 → 01:33 · re-run event 1 — owner-requested work with a mislabelled bank.** This is one of
the two 2026-09-22 events the audit was asked to settle. **It was not a person deciding to re-run
something; it was ordinary requested work.** The owner asked at 00:40 local ("all 3, step by
step…"); the Claude session launched a 21-layer, k=3 sweep of the 70B rung in the background
(VERIFIED that it ran: matrix mtimes 00:52 → 01:33, `audit/experiments.csv` row `O77`). The launch
command passed `--bank_name paper16_70b` while pointing `--bank` at `run-kd/axes_vectors_s3.pt` —
**the wrong folder, because both folders hold a file of that name** (DOC-ONLY as explanation,
transcript `1ac8bb19` line 996, `audit/phase1/sessions/findings_rerun_origin.md`; **the bank mix
itself is VERIFIED** by fingerprint, RF-16). This is what F-110 later corrects. **Nothing was
overwritten and nothing was hidden — it is a naming collision that survived four findings.**

**13:57 · the collaborator's draft is compiled.** "Lying in Knowledge-Denier Persona Coordinates:
Cross-Dataset Lie Detection at 70B", 19 pages, 5 figures, 11 tables, 0 findings-ledger citations
(VERIFIED, PDF metadata CreationDate 07:57:30 UTC = 13:57 local; `audit/phase0/draft_report.md`).
His last commit was 72 minutes earlier (`45b16d9`, 12:45). The owner committed the PDF at 16:51
(`cf11749`). **NOT-FOUND: the LaTeX source, anywhere on either branch.**

**14:29 → 14:30 · Gate 3, and the sarcasm line closes for good.** The last test of the register
hypothesis: take *natural* sarcasm (600 length-matched FigLang-2020 Reddit rows), extract
activations, and project them on the existing axes. The design's rule was that the deception axis
should fire on marked natural sarcasm. **It does not — it reads 0.543–0.583 against a
random-direction mean of 0.531–0.543 and a bag-of-words floor of 0.632, i.e. it is
indistinguishable from a random direction.** The axis that *does* fire is **sycophancy**, at
0.693–0.710, clearing both floors and exceeding the maximum of 200 random draws — the register
diagnosis from a third angle (DOC-ONLY, F-107; VERIFIED that it ran,
`gate3_coords.log` 14:29 → 14:30, "Done." — but note the projection result was **never persisted to
a file**, VERIFIED, `audit/experiments.csv` row `O71`).

**15:34:30 → 16:26 · re-run event 2 — the lab PC rebooted.** The second event the audit was asked to
settle. **No person and no Claude session started this.** The box rebooted at 15:34:30, and about 24
project Scheduled Tasks carry a Windows **boot trigger** with a delay. Boot time plus each task's
own delay equals its last run **to the second**: Crossgen-Chain PT5M → 15:40:01, KD-Chain PT6M →
15:41:01, HPKR-Topup/KD-Extract PT12M → 15:47:01, KD-Bank-Crossgen/Premise-Extras PT20M → 15:55:01,
Scale-3b/8b/Cap-Controls/Falsify PT30M → 16:05:00, Cap-Ctrl-Layers PT50M → 16:25:01 (VERIFIED,
`audit/scripts/fix_boot_triggers.ps1` output, recorded as RF-13).

About **58,000 lab result files** were rewritten in that window (VERIFIED, joined census mtimes,
`audit/phase1/labpc/log_index.tsv`). The audit checked what actually changed, matching records on
their keys rather than by position:

- For the summary files compared, the **git version is an exact subset of the lab version with zero
  differing metric values** — the lab file grew (more layers and seeds), the old numbers did not
  move (VERIFIED, `audit/phase1/summary_keyed_diff_sample.txt`).
- For the significance files, **every common test has identical raw statistics**; only the
  multiple-comparison q-values differ, because the family of tests grew (e.g. 192 → 4,788 tests in
  one file). Maximum change 0.39 (VERIFIED, `audit/phase1/significance_keyed_diff_sample.txt`).
- **Consequence:** any claim of the form "this clears / does not clear a corrected threshold" must
  be checked against the version it was computed on. The verdict can flip between versions.

**Why it rebooted is NOT-FOUND.** No Windows event-log query appears in any transcript. The lab PC
is multi-user and jobs from another account had been exhausting its memory the night before
(DOC-ONLY, `audit/phase1/sessions/findings_rerun_origin.md`).

**A separate bug made the wave worse than it needed to be (RF-14, VERIFIED).** Seven chain scripts
have a "skip if already done" guard that can never fire: the writer emits `<timestamp>  DONE` but the
guard matches `^DONE`. So every trigger re-runs the chain and **truncates its status file**. The
per-cell results were mostly skipped internally, but the gate lines quoted by ten findings (F-72,
F-75, F-77, F-81, F-82, F-83, F-87, F-94, F-95, F-104) **are gone from those files**.

**17:24 → 17:28 · a five-hour Drive outage ends.** The lab PC's Drive token stopped working at 12:18;
every sync failed until the owner reconnected it and the refreshed token was copied across. The
backlog uploaded at 17:27–17:28 (DOC-ONLY, transcript `1ac8bb19`; the two CRITICAL rclone failures
inside that window are VERIFIED in the logs).

**17:31 · the reboot cascade is switched off.** On the owner's instruction, 24 boot-triggered chains
were disabled, keeping three sync tasks on purpose (DOC-ONLY, transcript `1ac8bb19` line 2763).
INFERRED: this is why the lab PC looked quiet when the audit started 73 minutes later — it had been
made quiet, it was not a long-standing state. Four tasks remain armed (PassAC-11, Night-Chain,
PassC-PerToken, KD-Sync-Periodic).

**17:37 · the collaborator briefing.** `43ee6d3`, `docs/COLLABORATOR_BRIEFING.md` (VERIFIED, `git
log`).

---

## 13. What failed, and why — the short list

| what | when | why it stopped | evidence |
|---|---|---|---|
| **The sarcasm bank** (a third construct to test the register diagnosis) | designed 09-21 17:48, piloted 21:21, gate 3 run 09-22 14:29 | **Two of its three gates failed.** Gate 2: stripped of markers, ironic and sycophantic text are still 0.92-separable by word counts, so the bank would be a register axis itself. Gate 3: the deception axis does not fire on natural sarcasm at all (0.54–0.58, inside the random null). The bank was never built. | F-93, F-107 (DOC-ONLY); `sarc_pilot_*.log`, `gate3_coords.log` (VERIFIED) |
| **Growing the HP-KR matched slice** from 258 to 500–700 rows | gate run 09-20 22:52 | **Pre-registered gate failed.** Denial rate on items the model gets wrong: 0.250 against a required 0.70; the fallback gave 0.150. The model claims to know answers it gets wrong. | F-59 (DOC-ONLY); `hpkr_denial_*.log` (VERIFIED) |
| **F-03, the strongest positive result** | withdrawn 09-21 13:47 | Layer-localised (0.739 at L40, 0.374 at L20), then the plateau failed the dataset-level bootstrap: +0.0655, 95% CI [−0.0693, +0.1887]. | F-72, F-75, F-76 (DOC-ONLY) |
| **`own100` fp32 rebuild** | 09-16 01:30 → 02:07 | Abandoned after 1 of 100 roles; the `own100` bank still rests on the buggy bfloat16 reduction. | `audit/experiments.csv` `O05` (VERIFIED); `anomalies.md` OR-11 |
| **The first full crossgen run** | 09-17 21:18 | Crashed: `PYTHON_EXIT -1`, killed during layer 12 of 21. Cause NOT-FOUND. Restarted successfully the same evening. | `crossgen.status` (VERIFIED) |
| **Four complete result grids nobody ever reported** | 09-18 → 09-19 | Not a failure — an omission. The root passA grid (92,184 files), passC (6,936), passB_fine and passP (5,777) are complete and no document reports a number from them. `passP` was the prompt-side construct-validity control the thesis document planned. | `audit/phase1/anomalies.md` OR-01…OR-03 (VERIFIED) |
| **The 11-dataset significance ceiling** | throughout | Not abandoned — endured. With ~11 datasets, 1 of 384 dataset-level tests is significant after correction; row-level intervals look tight and mean little. The scale ladder (§9) was the attempt to argue around it by agreement of shape across model sizes instead of a p-value. | F-04 (DOC-ONLY); `run-crossgen/kd11/significance.json` |

**A recurring class of bug worth naming.** The owner hit *hardcoded names and silent no-ops*
repeatedly: the bank name `paper16` hardcoded in seven places, two of them silent (DOC-ONLY, memory
note `gates-must-assert-their-own-number`); six chain gates that logged PASS while measuring
nothing; four script fixes that existed only in the repository and never reached the lab PC where
they run, two of them on live chains (DOC-ONLY, F-104); a `^DONE` guard that never matches (RF-14,
VERIFIED); a literal backspace byte inside a completeness gate (RF-6, VERIFIED). **INFERRED: the
common cause is that the repository and the machine that runs the code are two different places,
and nothing checked they agreed until 09-22.**

---

## 14. The claim families — where a number changed, and why

This is the list to read if the question is "which version of this number is current?".

| family | the sequence | what moved it |
|---|---|---|
| **The scale ladder** | F-58 → F-60 → F-63 → F-96 → F-100 → **F-110** | Wrong dataset for one of five; wrong k; a single-layer reading at the 3B rung's *best* layer; then a persona-bank mix-up caused by two folders holding a file of the same name. Current: +0.054 vs the matched control and **−0.013 vs raw** at 70B. |
| **HP-KR reversal** | F-03 → F-72 → F-75 → **F-76** | One layer became a layer profile; the profile then failed the dataset-level bootstrap. **Withdrawn.** |
| **The HP-KR lexical shortcut** | 0.870 → **F-01: 0.913** | The 0.870 was computed over all 7,068 rows of a CSV holding three different generating models; the probe pipeline reads only the 2,139 Llama rows. |
| **The capitulation result** | F-65 → F-66 → **F-67** → **F-89** | Established (0.745 zero-shot), cleared the bootstrap, then bounded twice: local to one column, and mostly explained by "an instruction is in force" rather than by the disposition. |
| **The abstract's headline** | F-85 → **F-97** → F-98 → F-99 | F-85 compared against the wrong axis. 0.904/0.916 is the kd-augmented contrast; the paper's own is 0.791/0.832. ~45% of the margin is the owner's personas. |
| **The pooled-training premise** | F-79 → F-80 → F-81 → F-83 → F-86 → **F-95** → F-108 | Not corrected — *reinforced* at every step, including a leak check (a real row-level leak exists between HP-KR and its matched slice, and the claim survives leave-one-cluster-out anyway: +0.125 to +0.157 against +0.178). |
| **The response-length claim** | stated → F-73 → **DF-49** | The AUROC survives; the per-dataset framing does not. And the audit's *own* correction to it was later found wrong — there are two counts over two different sets (VERIFIED, `audit/DOC_FIXES.md` DF-49, applied in six documents before it was caught). |
| **The "0.374" attribution** | quoted in 9 documents → **DF-31** | It is the deception axis's PC1, not the sycophancy axis. The sycophancy axis reads 0.313 — and *that* number is the stronger one, because it lies outside the random-direction null while 0.374 does not. |
| **The ceiling-artifact statistic** | stated → **F-74** | The −0.76 correlation is mathematical coupling between the two quantities correlated, not evidence. Withdrawn. |
| **Audit-era corrections** | F-111 (11 entries), F-112 (2), F-113 (8, one in the collaborator's favour), F-114 (2 of them corrections to the audit itself) | 56 confirmed documentation errors in total, `audit/DOC_FIXES.md`. The repeated pattern: **28 of 29 "partly holds" verdicts found the number reproducing and the sentence around it failing** (VERIFIED, `audit/PROGRESS.md`, wave-5 section). |

---

## 15. 2026-09-22 18:44 → · The audit

**Why.** The owner's words, in `audit/BRIEF.md`: *"Neither of us knows anymore exactly what was run,
what came out, and which claims still stand."*

**What it has done** (VERIFIED, `audit/PROGRESS.md` and the files it names):

| phase | result |
|---|---|
| **0** — orientation | Repository, branches, lab PC, Drive and the draft all mapped. `audit/PROJECT_MAP.md`, `audit/PLAN.md`. |
| **1** — inventory | **114 experiments** dated and statused (`audit/experiments.csv`); a 210,464-file lab census and a 277,407-file Drive census; **242 lab logs** indexed by their internal timestamps; 65 anomalies (orphans, ghosts, duplicates, version conflicts). |
| **M** — methods | Method lineage, controls and attribution: `audit/METHODS_CHAPTER.md`, `audit/methods/ATTRIBUTION.md`. |
| **session mining** | The owner's 19 Claude transcripts mined into 359 test events, 254 dataset events and the recovery of **72 of 72 print-only findings** with their console output (`audit/phase1/sessions/findings_targets.csv`, 72 rows — VERIFIED by row count). This is where the *reasons* live: many ledger entries quote numbers that were only ever printed to a console and never saved to a file. **CONFLICT on how many:** `audit/SESSION_MINING.md:7` says 42 of 109; `findings_targets.csv` carries 72. Both are audit-internal counts; neither is re-derived here. |
| **2** — claim ledger | **231 of 692 claims checked** against raw data: 112 HOLDS, 51 MATCHES HIS DATA, 53 PARTLY HOLDS, 12 PARTLY MATCHES, 1 REVISED, 1 DOES NOT MATCH, 1 DOES NOT HOLD. Closed at the owner's decision, with the remainder bounded by a Phase-4 random sample. |
| **3** — cross-checks | In progress. This file is part of it. |

**Two things the audit changed in the world, on the owner's authorisation** (VERIFIED,
`audit/PROGRESS.md` "OWNER DECISIONS"): the `^DONE` guard bug and the backspace-byte bug were fixed
in the working tree and the lab copies (backed up first, not committed), and a like-for-like 70B
sweep was run into a new folder to settle RF-16.

**One incident, logged for honesty.** On 2026-09-23 02:19 an interrupted audit command kept running
on the lab PC after the local connection died, and rewrote **one** of 100 run-premise files
(`falsify/falsify_L40_s13.json`). The original is preserved in the Drive backup. Measured impact:
0.0012 in one null and 0.0021 in another; **no verdict moves** (VERIFIED, `audit/PROGRESS.md`, top
of RED FLAGS).

---

## 16. Open questions this timeline could not answer

- **NOT-FOUND: why the lab PC rebooted on 2026-09-22 at 15:34:30.** Nobody ever queried the Windows
  event log. The machine is shared and another account's jobs had been exhausting its memory.
- **NOT-FOUND: the cause of the 2026-09-17 21:18 crossgen crash** (`PYTHON_EXIT -1` at layer 12
  of 21).
- **NOT-FOUND: the LaTeX source of the collaborator's draft**, on either branch or in the working
  tree.
- **NOT-FOUND: `make_view.py` / `make_view_ac.py`**, the scripts that built the 11th dataset used by
  every `kd11` result. They exist only as loose files on the lab PC, in no branch (RF-8, VERIFIED).
  About 81 Python and 46 PowerShell analysis scripts are in the same position.
- **Uncertain: the exact run dates of the collaborator's experiments.** His result folders are dated
  by *commit*, not by run; his run logs are not in git (VERIFIED, `audit/experiments.csv` rows
  C01–C20, `when_source` = "git commit dates … run dates not recorded"). Every C-row date in this
  timeline is therefore an upper bound on the run, not the run itself.
- **Uncertain: what the 09-20 "kd11" grids actually covered.** Files under `kd11/passA` and
  `kd11/passC` are byte-identical to the older 10-dataset root grids, which suggests those ran on 10
  datasets rather than 11 (VERIFIED as a file fact, `audit/phase1/anomalies.md` §3; the consequence
  for any number read from them is INFERRED). Related trap: a `block "9"` label means *ten* datasets
  in one era and something else in another — **a block label is never a dataset count**.
