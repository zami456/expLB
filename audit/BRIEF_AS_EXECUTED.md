# BRIEF, AS EXECUTED — what the audit actually did

`audit/BRIEF.md` is the instruction you wrote on 2026-09-22 and is **unchanged**. This file is its
counterpart: what was actually done, where it departed from the plan, what it cost, and what it did
not do.

**Updated 2026-09-23 after Phase 5. The audit is complete — all five phases plus the methods track.**
*(Previous version was written after Phase 3 and is superseded; the counts in it were Phase-2-era.)*

Read this if you want to know what you have. Read `BRIEF.md` if you want to know what was asked for.

---

## 1. The short version

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

> **CORRECTION 2026-09-24 (DF-94, Phase 6E).** The tally below is superseded. After Phase 6A:
> **490 of 692 inventory claims have a verdict, plus 110 raised while checking** — real coverage
> **71%**, not the 61% "424 of 692" implies. **The draft PDF is complete at 176 of 176** (160
> `MATCHES HIS DATA`, 9 `PARTLY`, 7 `UNRESOLVED`, 0 failures). The 202 outstanding are the two
> optional phases the owner skipped: 110 ledger claims and 92 collaborator-note claims.
> The verdict split below (297/119/8) also predates Phase 6; current totals are in
> `audit/phase6/RESULTS.md`.
> _The text below is superseded on the counts and is kept for the record._

The audit ran **Phases 0–5 plus an unplanned methods track**, and checked **424 of 692 claims**
against raw data — **297 hold, 119 hold only partly, 8 fail**. It found the project's headline result
does not survive its own controls, produced **75 confirmed documentation corrections**, and wrote two
thesis-ready chapters that were not in the original brief.

Phase 4 then **measured the audit's own reliability**: 161 claims re-derived by agents that could not
see any prior verdict. **Its error rate is 6.5%**, every error was leniency, and **no true result was
wrongly thrown out**. Seven of its own verdicts were corrected as a result.

It also got **eight things wrong itself** and corrected each one. That record is §7, because an audit
that does not report its own error rate is asking to be trusted rather than checked.

**The one finding to carry away:** across the whole corpus the recurring fault is almost never a wrong
number. It is **a correct number inside a sentence that claims more than it** — and the same fault
turned up in the audit's own work, in both directions.

---

## 2. What was delivered, by file

| file | what it is | size |
|---|---|---|
| `REPORT_PLAIN.md` | the report for you: big picture, ranked results, decisions | 5 |
| `REPORT_TECHNICAL.md` | full detail, tables, paths, commits, "what a referee attacks first" | 5 |
| `claims.csv` + `claims/*.md` | every checked claim, with verdict, evidence path and a note each | **424** rows / 424 notes |
| `claims_inventory.csv` | every claim ever inventoried, checked or not | 692 rows |
| `DOC_FIXES.md` | confirmed documentation errors, each with file:line and evidence | **75** entries |
| `CORRECTIONS.md` | every Phase 4 adjudication, and the audit's own corrections | 33 adjudications |
| `HPKR_WORDING_CONTROLLED.md` | the decisive HP-KR experiment, fully written up with verified data | new, 2026-09-23 |
| `experiments.csv` | one row per experiment across all four sources | 114 rows |
| `papers.csv` | PDFs present, notes present, cited-but-missing | 29 rows |
| `phase1/fmap_all.csv` | every F-number → its evidence files | 109 rows |
| `phase2/layer_sensitivity.csv` | off-diagonal/diagonal means per store × pooling × layer × seed × feature × probe | **78,516** rows |
| `phase2/controls.csv` | the control table: zero-shot nulls + trained controls | **29,414** rows |
| `phase3/DRAFT_GAPS.md` | what the draft misses — **both directions** | 779 lines |
| `phase3/DISAGREEMENTS.md` | where your numbers and his disagree, and why | 463 lines |
| `phase3/WEAK_SPOTS.md` · `UNFOLLOWED.md` | stale foundations, leaks, single seeds; unwritten results | 676 / 603 lines |
| `phase4/` | the independence check: sample, 14 batches, merged results, adjudications | 161 claims |
| `phase5/digest_{results,draft,story}.md` | the digests the reports were built from | 3 files |
| `TIMELINE.md` | the project's story in order | 716 lines |
| `METHODS_CHAPTER.md` · `LIMITATIONS_AND_FUTURE_WORK.md` | thesis-ready | 154 / 183 lines |
| `cache/` + `MANIFEST.csv` | result files copied off the lab PC, all md5-verified | 305 files |
| `scripts/` | re-runnable scripts behind every number | **390** Python files |

Plus, written **outside** `audit/` on your explicit authorisation: `docs/DRAFT_CORRECTIONS.md`,
corrections to 15 documents, and F-110…F-114 appended to `kd/docs/FINDINGS.md` (now **114** F-entries).

---

## 3. Phase by phase — planned vs actual

### Phase 0 — Orientation. **Done as specified.**
One item was owed for a long time — a `find -newer` scan for stray writes outside `audit/`, killed by
low memory. **It was finally run in Phase 3** and found every modification accounted for, plus two
harmless `__pycache__` files written when audit scripts imported project modules locally.

### Phase 1 — Inventory. **Done, and larger than specified.**
Beyond the brief: a **census of 210,464 lab files, 277,407 Drive files and 9,527 git blobs** joined by
md5; **242 lab logs** indexed by internal timestamp (70 of them UTF-16); **`fmap_all.csv`** mapping all
109 F-numbers to evidence.

**You added a task mid-phase:** mine the Claude session transcripts → `phase1/sessions/`, 19 sessions,
40,814 records, **359 test/run events**, and **72 of 72 print-only F-numbers** recovered with their
producing commands. **The single highest-yield addition to the plan.**

### Phase 2 — Claim ledger. **424 of 692, after you twice changed the scope.**
1. **The denominator grew** from 390 to 692 as your other docs, his 17 result notes and your memory
   notes were added.
2. **You chose a targeted close at 231** (2026-09-23), then **reversed part of it** after deciding to
   reframe the paper, asking for **your** side in full. Waves 5 and 6 carried it to 424.
   **Your side is 100% complete** — ledger 110/110, docs 103/103, memory 97/97, session 35/35,
   abstract 8/8. Everything unchecked is his.

**W4 — the central control re-derivation — is DONE**, and the premise that blocked it was wrong: the
random-direction null did **not** live only in a console log. `run-coords/zeroshot_nulls.json` sits in
the git tree with **1,000 draws per cell**. Nothing had to be redrawn.

### Phase 3 — Cross-checks and gaps. **Done, all six brief items.**
Five parallel workstreams, all lead-QA'd. "What the draft misses" became a both-directions list at
your instruction, because unawareness runs both ways.

### Phase M — **not in the brief at all.**
You asked whether the report would cover methodology. It would not have. Four workstreams produced the
methods inventory, a line-by-line comparison of our extraction against the replicated paper, an
**attribution list with a commit hash per design**, and the two thesis chapters.

### Phase 4 — Independent check. **Done, and re-scoped by you.**
The brief asks for every HIGH-importance claim plus a random 20% — **279 re-derivations**. You chose a
targeted **161** because 243 of 424 HIGH ratings had been **self-assigned by the agents that made
them**, which is rating inflation rather than importance.

14 blinded batches. Agents received `sample_blinded.csv` — **prior verdict, importance and tier
withheld**. The lead alone compared against the unblinded sample and adjudicated all 33 disagreements
from raw files.

**Result: raw disagreement 30.8%, actual audit error 6.5%.** The gap is almost entirely
verdict-granularity, not data: **agent and audit disagreed about a number once in 161**, and there the
agent was right. Full write-up `phase4/RESULTS.md`.

### Phase 5 — Reports. **Done, four deliverables not two.**
`REPORT_PLAIN.md`, `REPORT_TECHNICAL.md`, plus `METHODS_CHAPTER.md` and
`LIMITATIONS_AND_FUTURE_WORK.md` from Phase M. `DOC_FIXES.md` extended 70 → 75.

---

## 4. Work you authorised that the brief forbids

The brief says *write only inside `audit/`*. You lifted that seven times, each recorded:

| what | when | why |
|---|---|---|
| Fix the RF-14 guard bug + RF-6 backspace byte in 8 scripts, repo and lab | 09-22 | the chains re-ran silently every trigger |
| `rclone copy` 1,983 lab-only files (1.73 GB) to a **new** Drive folder | 09-22 | they existed in one place only |
| A like-for-like paper16 70B sweep into a **new** output folder | 09-22 | RF-16: the scale rung mixed two banks |
| Correct 15 documents in place + create `docs/DRAFT_CORRECTIONS.md` | 09-23 | the collaborator package |
| Append F-110…F-114 to the ledger | 09-23 | your rule: append, never edit |
| **Fix DF-48** in `run_scale_chain.ps1`, repo **and** lab copy (hash-verified identical) | 09-23 | both axis arms were written to one filename |
| **`rclone copy` `run-premise/` to a new Drive folder** — 100 files, 774 KiB, `rclone check` **0 differences** | 09-23 | your strongest result existed in one place |
| **Two experiment runs** in an isolated tree (`expLB-resplit`), originals md5-verified unchanged; **F-115 / F-116 appended to the ledger** | 09-23 | you asked for the re-split, then "if NDIF needed do it" — it was not |

Nothing else outside `audit/` was touched. The working tree is **18 entries** (17 modified +
`docs/DRAFT_CORRECTIONS.md`), all uncommitted, all yours to commit.

---

## 5. What the audit found — the five things that matter

1. **The method does not beat its controls.** Across **65,388** dataset-level tests, **4 of 16,548**
   persona-beats-raw comparisons clear BH correction, and **0 of 276** at the operating points you
   actually report. Against the matched dataset-PCA control, **0 of 126**. Over 80 layers, persona axes
   beat raw at **2–5 of 80 in every pooling × probe cell**. Re-derived by the lead twice, by two routes.
2. **The strongest, best-tested result is the one with no personas in it.** Pooled training over ten
   datasets beats single-source transfer by **+0.1254** under leave-one-*cluster*-out (**+0.1782**
   under plain LODO, which leaks), positive on **10 of 11** targets, five seeds, interval **±0.035**.
   **Publish +0.125.** Lead-verified from `clustered_L40_mean_s42.json`.

> **UPDATE 2026-09-24 (DF-105, P17).** The **0.9597** ("does not affirm knowledge") regex value below
> was **not re-derived**: the regex is recorded nowhere. The guess `NOT ^(yes|i know)` gives **0.9413**,
> still above the 0.919 headline, so the argument stands; 0.9125 is unaffected.
> **Source:** `audit/p17/correction_blocks_check.csv` (row for `docs/NOVELTY_CHECK_PROMPT.md:127`);
> `audit/p17/C_REPORT.md`.

3. **One dataset carries three headlines, and it is mostly wording.** HP-KR is **91–96% solvable by a
   label-free regex** (0.9125 / 0.9597 against your headline 0.919); **length alone reads 0.776** there,
   above the deception axis's own 0.6436, which sits **inside its random null**. Removing HP-KR takes
   the draft's contribution from **+0.220 to +0.009** and your own best result from +0.178 to +0.125.
4. **The wording-controlled test was run, and the draft says it was not.** F-03/F-72/F-76. Zero-shot on
   the matched slice every direction reads at or near chance, while a **plain-assertion** direction
   reproduces the full-dataset result (0.93). Trained, the margin is **+0.066, CI [−0.069, +0.189],
   P = 0.161** and **inverts to −0.257 at layer 20**. Written up in full, with verified data, in
   `HPKR_WORDING_CONTROLLED.md`.
5. **The attribution is wrong in the draft, and the fair baseline is his own work.** 24 of 26 designs
   are yours, named nowhere; the **PDF's** headline (0.919 on HP-KR at L40, 0.916 mean off-diagonal)
   is ~45% attributable to your kd bank. *(An earlier audit note quoted "0.904/0.916" — that pair is
   from the superseded 2026-09-21 session abstract, not the PDF; see §9.)* And the
   pooled comparison the draft says it did not run is committed on **his** branch as
   *"Leave-one-dataset-out: the fair baseline for the generalization claim"*, two days before the draft.
   Three corrections run the **other** way and are recorded with equal weight.

---

## 6. What the audit did NOT do

- **268 claims unchecked** (38.7%) — all his: 176 draft body, 92 notes. Phase 4's random 54 bound them:
  **77.8% hold, 18.5% partly, 1.9% fail** — no worse than the checked corpus. **But the single failure
  in that sample was D-158, the audit's most consequential finding**, so a low rate does not make the
  remaining 214 safe to ignore.
- **The 63 screen-flagged claims.** A sweep found **69 clean verdicts whose own notes describe a
  defect**; 6 were confirmed and fixed. **You decided to report the other 63 rather than re-grade
  them** — correct, since re-grading on a regex would repeat the fault being measured.
- **His activations** — not requested, by your decision. **Nothing in his draft is recomputable from
  raw**; his figures are *attested, not reproduced*.
- **A clean wording-matched HP-KR slice.** The existing one leaks for *trained* comparisons (~81% of
  its test rows sit in HP-KR's train split). Fixing that needs **new NDIF rollouts**, not a CPU re-run.
- **Three lab-only stores are still unbacked**: `run-cap/` (186.7 MB), `run-cap-ctrl/` (108.8 MB),
  `logs/` (648.7 MB), plus ~133 loose scripts. You asked for `run-premise` only, and only that was done.
- **The two parked conflicts** — the 42-vs-72 print-only count, and the reduced feature set in three
  stores (now confirmed: **7 stores, seeds 0–4**, and **no claim anywhere takes a multi-seed mean**, so
  no existing number is invalid).

---

## 7. The audit's own error record — eight, not five

The rule was: every batch nominates its two least-certain findings and the lead re-derives them from
raw files. Phase 4 then tested the whole thing blind.

| # | what was wrong | caught by |
|---|---|---|
| 1 | `F39_NOTE.md` §2 told you to fix F-45/F-48/F-54 — they were right | W5A + W5C, then the lead's PDF extraction |
| 2 | The "never 9 of 10" length correction changed the metric *and* the dataset set | W5D |
| 3 | P3-C's "median effect 0.044" was a median **absolute** effect; signed it is **−0.026** | lead QA |
| 4 | `DRAFT_GAPS.md` A-13 set your 110-pair numbers against his 0.732 — two different corpora | W6C |
| 5 | The DF-49 **repair itself** cited a path not in the repo and contradicted a sentence below it | W6A |
| 6 | The F-61 QA claimed the mis-copy "can be proved arithmetically" — two candidates fitted | W6G |
| 7 | `CONTROLS.md`'s assistant-axis row needed a length caveat it did not carry | W6G + lead |
| **8** | **The final report recommended running an experiment that had already been run** (the HP-KR wording test) **and called it CPU-only when a clean version needs new NDIF rollouts** | the owner's question, Phase 5 |

Plus **7 verdicts corrected by Phase 4** (F-01, F-70, O-022, M-007, M-054, S-25 → PARTLY HOLDS;
D-028 → PARTLY MATCHES).

**The pattern is one thing, stated plainly: a number — or a conclusion — carried into a new document
without re-opening the file it came from.** Error 8 is the purest case: it was taken from the *draft's*
prose, and your own ledger contradicted it in 22 places.

It is the same failure mode the audit found throughout your ledger, and **Phase 4 showed it runs in
both directions**: in 12 of its 21 errors, a blinded agent wrote the defect into its notes and then
graded the claim clean anyway. **Three of the eight were in corrections** — the audit was most
error-prone exactly where it was most confident.

One `claims.csv` row was corrected **in your collaborator's favour** (D-107: his random control is the
mean of 5 seeded draws, not one).

---

## 8. Where the numbers come from

Every number in every audit deliverable traces to a file opened or a script run during the audit.
`audit/scripts/` holds **390** re-runnable Python scripts; `audit/cache/MANIFEST.csv` logs **305**
files copied off the lab PC with md5 on both sides; `audit/LABPC_WRITES.md` logs every write the audit
made on the lab PC or Drive. Statements are tagged VERIFIED / DOC-ONLY / CONFLICT / NOT-FOUND /
INFERRED, and anything not in `claims.csv` is DOC-ONLY by construction.

The headline scripts, if you want to re-run one thing: `p4_rates.py` (the audit's error bar),
`p4_loo_baseline.py` (the pooled comparison the draft omits), `p5_hpkr_matched.py` (the HP-KR wording
test), `p5_stats.py` (every structural count in the reports).

**What this means for the thesis:** you can cite an audit number and, if challenged, produce the script
that made it. That is the property the original work did not have, and it is why several of your
headline numbers turned out to be true of a different dataset set, a different bank, a different
pooling or a different sign convention than the sentence around them claimed.

---

## 9. Which version of the collaborator's abstract the audit tested — and the stale one

**You flagged this, and you were right to.** Two abstracts exist and they are not the same document.

| | the **PDF draft** | the **2026-09-21 session abstract** |
|---|---|---|
| where | `full paper draft for this experiments/main (1).pdf` (branch `asix`) | pasted into a Claude session, recovered to `audit/phase0/collab_abstract_2026-09-21.md` |
| title | *Lying in Knowledge-Denier Persona Coordinates: Cross-Dataset Lie Detection at 70B* | *One Direction Generalizes: Persona Contrasts Outperform Trained Deception Probes…* |
| HP-KR headline | **0.919** AUROC at L40 (appears **7×**) | **absent — 0.919 appears 0 times** |
| transfer headline | 0.916 vs 0.732 | 0.904 and 0.916 vs 0.648 and 0.732 |
| status | **current** | **superseded** |

**What the audit actually did.**

- **61 `D-` claims come from the PDF.** That is the document the report's draft analysis is built on.
- **41 claims were verified against the collaborator's own branch**, read with
  `git show origin/feat/persona-pc-ood:src/probes/deception-detection/results/...`. Phase 4's D1–D4,
  H1 and H2 batches did the same. **His committed data was genuinely used, not assumed.**
- **8 `A-` claims took their *wording* from the session abstract** — but **7 of those 8 were checked
  against his branch data**, not against the abstract. So the *numbers* were tested against reality;
  what was stale is *which sentence* was being graded.

**The defect, stated plainly:** those 8 claims grade sentences from a document the collaborator has
since replaced. They should carry a "supersedes" marker. **They are 8 of 424 (1.9%) and none of them
is load-bearing for any conclusion in the reports.**

**What does NOT depend on either abstract** — checked, not asserted:

| conclusion | its actual source |
|---|---|
| pooled training **+0.1254** | `run-premise/clustered/clustered_L40_mean_s42.json` — **your lab data**, lead-verified |
| the method does not beat its controls | the crossgen significance census + `run-coords/zeroshot_nulls.json` |
| **D-158** — the pooled baseline the draft says it did not run | the **PDF** text + **his branch** `loo_L20/`, `loo_L40/` |
| HP-KR is regex-solvable | your HP-KR CSV + his branch |

**Corrected here:** `BRIEF_AS_EXECUTED.md` §5 item 5 now quotes the PDF's 0.919/0.916 rather than the
session abstract's 0.904/0.916. `REPORT_PLAIN.md`, `REPORT_TECHNICAL.md` and
`HPKR_WORDING_CONTROLLED.md` never quoted 0.904 — they already use the PDF's 0.919.

---

## 10. Work done after the audit closed (2026-09-23)

The audit's five phases ended; you then authorised experiments. Both ran from an **isolated copy**
(`C:/Users/user4/expLB-resplit`), read the originals read-only, and left them byte-identical.

- **F-115 — the HP-KR/kr-matched row leak, closed and measured.** 59 of 258 rows survive the guard
  (77% contamination). **F-03's headline barely moves** (−0.2545 / +0.0309 / +0.0050), so the leak
  did not manufacture it — but the affected cells do, with the persona axis falling to **0.4262,
  below chance**, at L40.
- **F-116 — the control slice nobody had run.** `hpkr_slice` has two modes; the unused one has
  **2,001 rows against 258**. Persona axes transfer **below chance** (0.3850 / 0.3335 / 0.5267),
  lose to raw by **0.19–0.38** — the largest deficit in the project — and are beaten by a
  **length-only** feature at every layer, while an ordinary probe reads **0.999** in-distribution.

**Neither needed NDIF.** My earlier estimate that a clean run required fresh rollouts was wrong twice
over and is corrected in `CORRECTIONS.md` (#8, #9).

**Two engineering failures on the way, both mine, both logged in `PROGRESS.md`:** a Scheduled Task
that reported success and never ran (twice — `schtasks /Run` returns SUCCESS regardless; always
check `Last Run Time`), and a first patch that changed which rows were scored without resizing the
bootstrap that computes their error bar.

**Error record is now nine**, not eight: #9 is the "needs new NDIF rollouts" claim, written without
opening F-108, which my own document cited two sections earlier.

---

## 11. The collaborator's `origin/paper` branch (2026-09-24) — a SCOPE EXTENSION

> **UPDATE 2026-09-24 (P17).** origin/paper's claims now carry `claims.csv` ids: 176 D-rows
> re-pinned to 16def2de, and D2-001…026 for its three newest commits.
> **Source:** `audit/claims.csv`; `audit/p17/d_rows_repinned.csv`; `audit/p17/paper_new_claims.csv`.

The brief limited the audit to `asix` and `origin/feat/persona-pc-ood`. On 2026-09-24 the owner asked
for a **new branch**, `origin/paper` (pushed 2026-09-23 23:46), to be reviewed. It is **outside the
424-claim corpus** and its claims carry **no `claims.csv` ids**; the review is
`audit/COLLAB_PAPER_BRANCH.md`, with ledger entries **F-117, F-118, F-119**.

**What it changes.**

> **CORRECTION 2026-09-24 (DF-101, P17).** "Within ±0.06 at every layer" and "crosses zero five
> times" are both wrong. Excluding HP-KR, `ext_axis1 − ext_pc3` reaches **+0.0633 at L56**, and the
> series changes sign **6** times. Still correct: 11/16 positive, +0.0086 (L20), +0.0012 (L24), and
> negative at L64/68/72. No persona-arm cell changed between ea1ce52b and 16def2de.
> **Source:** `src/probes/deception-detection/results/persona_ood_ext_L{20..79}/summary.csv`,
> `off_diagonal_excl_hpkr` @ 16def2de; `audit/p17/ledger_verify_stdout.txt:223-224`.
> **How the claim is updated:** *"excluding HP-KR the margin is positive at 11 of 16 layers, lies
> between −0.033 and +0.063, and changes sign six times."*
> _The text below is superseded on this point and is kept for the record._

- **The audit's D-028 finding generalises from 3 layers to 16.** Excluding HP-KR, the draft's
  Contributions margin sits **within ±0.06 at every layer** and crosses zero five times.
  His **"outscores at every depth" is falsified by his own sweep** (negative at L64/68/72).
- **One result runs in his favour and is new:** `ext_axis1 − baseline` is **positive at 16/16 layers
  (+0.115 to +0.273), and 16/16 excluding HP-KR**. Depth-robust, not HP-KR-driven, invisible from
  three layers.
- **He independently confirmed the register diagnosis.** Gender-Secret — both classes deny — puts
  **every persona direction at chance (0.436–0.623)**. With F-115 and F-116 that is **three
  independent wording-matched tests agreeing**.

> **UPDATE 2026-09-24 (DF-107, P17).** Resolved by the collaborator at 2ad758fb, in his favour: the
> paper now reports the **L36 optimum** (`results.tex:228-230`, "Ranking peaks at
> layer~\V{depth-off-ext-peak-layer}, not at layer~40"; numbers.tex `depth-off-ext-peak-layer`=36,
> `depth-off-ext-peak`=0.927), and the "outscores … at every depth" claim is gone — `results.tex:136`
> now counts wins over the sweep (`depth-pc-wins-complement`=13 of `depth-n`=16). The one remaining
> "every depth" (`results.tex:208`) is a different claim, about the base probe inverting.
> **Source:** `src/probes/deception-detection/paper/sections/results.tex` and
> `paper/generated/numbers.tex` @ 16def2de; `audit/p17/A_REPORT.md` §2.

- **The optimum layer is 36, not the reported 40** (0.9275 vs 0.9158).
- **D-237 SUPERSEDED, in his favour:** `paper/make_numbers.py` now exists. It was the audit's **only**
  DOES-NOT-MATCH verdict. Second correction in his favour, after D-107.

**Credit where it is due:** he validated the new pooling path rather than assuming it (L40 rebuilt
both ways, row cosine ≥ 0.99989, RepE probe cosine 0.9996), and the Gender-Secret run is the most
carefully controlled experiment on either side — adapter inside the remote trace, PEFT agreement to
1e-5 with a test shown to fail without it, a behavioural check in both answer orders, and a paired
base-read control.

**Error record stays at nine.** Nothing here corrects the audit; F-119 corrects a verdict that was
accurate when made and was overtaken by new work.

---

## 12. The scale ladder, completed (2026-09-24) — F-121

The last open experimental question, and the owner found it herself: the published 26-persona bank
had rungs at 3B, 8B and 70B, but **the project's own 36-persona bank existed at 70B alone**. So
*"our bank rescues the method at scale"* could not be separated from *"our bank is better at any
scale"*. She authorised the run; it completed 2026-09-23 23:57 → 2026-09-24 08:51.

**Run:** 36 personas x 5 variants x 30 shared questions = **5,400 rollouts at 3B and again at 8B**,
judged by Llama-3.1-70B held constant, both vector variants extracted and both axis files built —
construction identical to the 70B kd bank, so the rungs differ only in model. Crossgenned on
identical footing: 5 datasets (MASK fifth, per F-60), k=3, mean pooling, seed 42.

| rung | paper16 (26p) | kd s23 (36p) |
|---|---:|---:|
| 3B | +0.0382 | +0.0360 |
| 8B | +0.0148 | +0.0194 |
| **70B** | **−0.0116** | **+0.0054** |

**Both banks decline monotonically — the decline belongs to the METHOD, not the bank.** That closes
the standing objection to F-63/F-110 ("you used the wrong personas"). The kd bank declines less and
the bank gap widens with scale (−0.005 → +0.005 → +0.017), but at 70B both are within a whisker of
zero, on one seed, with no significance test, on five datasets containing **no knowledge-denial
data** — the thing the bank was built for.

**Process notes worth keeping.**
- **DF-48 was verified, not assumed.** The two axis variants are byte-distinct at both rungs. That
  bug would have silently made the whole `all`-vs-`s23` comparison meaningless.
- **Long jobs must be SYSTEM scheduled tasks.** `/RU user4` registers interactive-only and
  **silently never fires** (`Last Result 267011`) while `schtasks /Run` still reports SUCCESS. Cost:
  one wasted 30-minute watch before it was caught.
- **The lab PC powered off after the chain finished** and lost nothing, because every stage writes to
  disk and is resumable. The new output was backed up on reboot: **290 files, 0 differences**.
- Three monitoring layers failed during this work (a task that never fired, two watchers killed for
  laptop memory) while the **work itself never failed** — an argument for resumable stages and an
  on-disk status file over any live watch.
