# PROGRESS_THESIS: where the thesis stands

**Last updated:** 2026-09-25 (end of session).

> **UPDATE 2026-09-30 (judge edit, owner request).** Ch 4 `chapter_5.tex:78` now says why the judge is open-weight: Lu et al. used gpt-4.1-mini (PDF p. 3), a paid API model we could not use because of its cost (owner's statement). Ch 5's "The judge." paragraph (`chapter_6.tex:1683-1688`) gained: per-role agreement (lowest 74/76/78%); under the 405B judge `pc_plausible_denier` keeps 13 < the 20 floor, so paper16's score-3 bank would lose one of its eight deception roles; scores 2-3 sets differ by 9; no vectors or AUROC were built from the 405B scores; the test does not measure our judge against gpt-4.1-mini. Verified independently (`CH4_VERIFY_TABLE.md` row 186; `CH5_VERIFY_TABLE.md` rows 486-493); `check_latex.py` OK. **Drive not checked: the rclone token for `tamara:` has expired (`invalid_grant`); run `rclone config reconnect tamara:`.**

**Read at the start of every session,** together with `THESIS_BRIEF.md` and `GLOSSARY.md`. For Chapter 5's content, read `RESULTS_SUMMARY.md` (one line per result, with scope, polarity, guide ID and `\label`; then every limitation and open item).

## Status by chapter (the BracU template keeps 6 main chapters)

| Ch | Title (`main.tex`) | File | Status |
|---|---|---|---|
| — | Front matter (title, declaration, approval, ethics, abstract, dedication, acknowledgement) | `thesis/core/*.tex` | template placeholders |
| 1 | Introduction | `chapters/chapter_1.tex` | template placeholder |
| 2 | Literature Review | `chapters/chapter_2.tex` | template placeholder |
| 3 | Requirements, Impacts and Constraints | `chapters/chapter_3.tex` | template placeholder. Economic analysis is settled: "no direct cost". |
| **4** | **Proposed Methodology** | **`chapters/chapter_5.tex`** | **WRITTEN and VERIFIED** (see below). Not compiled yet: there is no TeX here, so compile in Overleaf. |
| 5 | Result Analysis | `chapters/chapter_6.tex` | **WRITTEN and VERIFIED 2026-09-25** (see "Chapter 5" below). 335 KB; 2 `\todo`s left. Plan: `CH5_RESULTS_PLAN.md`; checks: `CH5_VERIFY_TABLE.md`. |
| — | Appendices A, E | `appendix/appendix_A.tex`, `appendix_E.tex` | **GENERATED and VERIFIED 2026-09-25** by `thesis_tools/appendix/make_appendix_personas.py` and `make_appendix_examples.py`; `main.tex` now has `\appendix` and chapters A–E. |
| — | Appendices B, C, D | `appendix/appendix_B/C/D.tex` | stubs, one `\todo` each (planned content listed inside) |
| 6 | Conclusion | `chapters/chapter_9.tex` | template placeholder (only `\label{ch:conclusion}` added) |

The template's sample appendices (`appendix_1.tex`, `appendix_2.tex`) are no longer input by `main.tex`; the files are empty and can be deleted.

**File names don't match chapter numbers:** `main.tex:135-136` puts Proposed Methodology in `chapter_5.tex`. `chapter_7.tex` is empty and not included.

## Chapter 4: what it now covers
The plan is in `CH4_METHODOLOGY_PLAN.md`; the labels are in `GLOSSARY.md` §1.
- **4.1 Design Process / Methodology Overview.** The pipeline (Fig `fig:m-pipeline`), and Pipelines A and B.
- **4.2 Design Specification:**
  - 4.2.1 Subject model and compute;
  - 4.2.2 Persona vectors (Eq `eq:m-persona`);
  - 4.2.3 Contrasts and persona coordinates (Eq `eq:m-contrast`);
  - 4.2.4 Features;
  - 4.2.5 Probes;
  - 4.2.6 Controls;
  - 4.2.7 **Design alternatives** (Table `tab:m-alternatives`: D1, D2, D4–D9, plus the pooled probe and the kd/cap contrasts; a sentence on the D3/INLP attempts);
  - 4.2.8 **Verification of the alternatives.**
- **4.3 What We Built:**
  - persona banks (Table `tab:m-banks`; Figs `fig:m-kdbank`, `fig:m-personas`): paper16, kd, cap, sarcasm pilot, natural sarcasm, own100/hf275, scale rungs;
  - datasets we built: `sycophancy_mt` and the HP-KR text-/knowledge-matched slices;
  - the tools we built.
- **4.4 Data Collection** (Table `tab:m-datasets`, 13 rows):
  - Cleaning: MASK relabel; roleplaying labels; `ai_audit` excluded for too few items plus memory, **with no mention of NDIF**.
  - Transformation: activation stores; Pipeline B's 16 layers, with the disclosure that the 13 remote layers' pooling code is not in the repo.
  - Integration: the HP-KR label (Fig `fig:m-hpkr`), the slices, the leak (199/258) and its guard (59), the provenance clusters.
  - Reduction; Summary.
- **4.5 Implementation:**
  - protocols (Fig `fig:m-protocols`);
  - metrics (Pipeline A TPR, Pipeline B BA plus specificity);
  - statistics;
  - layer choice (L40 fixed in advance; the pre-registration amendment disclosed, with the git dates);
  - implementation and verification (the numbers script's constants disclosed).

**Verification done:**
- `CH4_VERIFICATION_LOG.md`: lab PC, Drive, both branches.
- `CH4_VERIFY_TABLE.md`: 178 rows checked by an independent subagent; every mismatch fixed.
- `thesis_tools/check_latex.py` passes on Ch 4 and on the whole thesis.

## Chapter 5: what it now covers (written 2026-09-25)

Template headings with our themes as subsections (`GLOSSARY.md` §8 lists every label):
- **5.1 Performance Evaluation:** criteria; testing methods; the problem measured (the HP-KR inversion in both pipelines, ranking vs detection, the benchmark's published probes).
- **5.2 Analysis of Design Solutions:** supervised probes D1/D2; the published persona method D4; the zero-shot contrasts D5–D9 with the headline (LB-average zero-shot AUROC 0.8990 at L40, 0.9130 at L36 post hoc) and "what the gain is made of"; HP-KR, the factor probes and the convergence (one finding); transfer; which personas (library vs purpose-built banks); selection (D9 for ranking and transfer, D8 for detection).
- **5.3 Final Design Adjustments:** the bank; one direction; the widening control (D7); the layer; the token read and pooling; approaches that did not work (D3, D10 quoted from report.pdf; DANN; the HP-KR growth gate); adjustments to the claims.
- **5.4 Statistical Analysis:** the dataset bootstrap census, the matched control, the random-direction null, length, dataset identity, calibration (24 of 90 cells above 1% FPR), the generalisation gap.
- **5.5 Comparisons and Relationships:** vs the benchmark's probes and protocol; vs the replicated method at 70B; the scale ladder (both banks decline); pooled training (+0.1254 leave-one-cluster-out, one seed); geometry and behaviour.
- **5.6 Discussions:** the register diagnosis; the capitulation bank and the sycophancy axis (with an example box); sarcasm and deceptive intent (example box); model organisms; the remaining questions; implications; a pointer to Chapter 6 for limitations.

**How it was written and checked (workflow `wf_040cbc40-2cf`, 22 agents):** eleven writers (W1–W7 the sections, W8 the Chapter 4 examples, W9/W10 the appendix generators), each re-reading every number from raw on both sides; eleven independent verifiers (485 checks: 327 MATCH, the rest fixed in the text; `CH5_VERIFY_TABLE.md`). The main session then compared every cache file the numbers rest on with the lab PC (520 files, 0 mismatches), copied the lab-only stores the `\todo`s needed into `audit/cache/labpc/` byte-exact (`audit/cache/MANIFEST.csv`), and filled seven of the nine `\todo`s from them. `check_latex.py` passes (19 files, 105 labels, 35 bib keys, 16 images).

**Figures:** copied from the co-author at the pin (`collab_fig_transfer/hpkr/depth/operating.pdf`, credited in the captions) and five drawn from our data by scripts in `thesis_tools/figures/` (`fig_convergence`, `fig_widening`, `fig_scale_ladder`, `fig_loco`, `fig_capitulation`, `fig_significance_forest`, `fig_sleeper`; each prints its numbers).

**Corrections found on the way:** DF-110…DF-120 proposed in `audit/DOC_FIXES.md`; `THESIS_BRIEF.md:168` corrected; `PITFALLS.md` addendum; the plan's non-reproducing values listed at the end of `CH5_RESULTS_PLAN.md`.

## `\todo` markers left

Every `\todo` in the thesis now, 2026-09-25 (`grep -n 'todo\[' thesis/chapters/*.tex thesis/appendix/*.tex`):

| # | Where | `\todo` | What would resolve it |
|---|---|---|---|
| 1 | Ch 5 `sec:r-personas` (`chapter_6.tex` ≈ line 686) | The trait-vector contrast's cosine with the assistant direction at L40 | No grouping or output was ever saved. Either re-run a trait-vector contrast with a saved grouping (a new computation; owner's approval), or drop the sentence |
| 2 | Ch 5 `sec:r-sarcasm` (≈ line 1605) | Gate 3 zero-shot AUROCs on the natural-sarcasm set | Console-only (F-107; the audit log `audit/phase2/out/M1_F107.txt` covers L20/L40 while F-113 §4 tabulates L40/L44, so neither was quoted). Re-derive from lab PC `run-coords/sarcasm/sarcasm_natural/chunk_*.pt` and save a result file (new computation; owner's approval) |
| 3 | `appendix_B.tex` | Appendix B, full result tables: stub | Write it (planned content is in the stub) |
| 4 | `appendix_C.tex` | Appendix C, reproducibility, pins and the audit: stub. Chapter 4 refers to it | Write it |
| 5 | `appendix_D.tex` | Appendix D, engineering notes: stub | Write it |

**None in Chapter 4, Appendix A or Appendix E.** Chapter 5 went from nine `\todo`s to two. Seven were filled in the main session from lab files copied byte-exact into `audit/cache/labpc/` (md5-checked, `audit/cache/MANIFEST.csv`).

Other open items (not `\todo`s in the text; the full list is in `RESULTS_SUMMARY.md` §2b):

| Item | Where | What resolves it |
|---|---|---|
| D3 and D10 numbers are quoted from the co-author's `report.pdf`; DANN has no numbers | Ch 5 `sec:r-failed`, `tab:r-designs` | The co-author commits the D3/D10 result files |
| The 13 remote layers' pooling code (`trace_layer_outputs(span_mask=…)`) | Ch 4 disclosure; Ch 5 L36 numbers | The co-author commits it; then `git fetch` and delete the disclosure |
| Pipeline B's four-decimal reproduction and L40 check (0.99989) | Ch 4 §4.2.8 (reported as not reproducible) | The co-author commits `results_table.csv` and the check40 vectors (audit D-063, D2-024, D2-025) |
| `aisi2026liedetection`: year taken from report.pdf | `references.bib` | Check the Hugging Face page |
| Six classic references (BH, Maris, Kriegeskorte, Ravfogel, Ganin, Sharma): volume and pages from memory | `references.bib` | A DOI check |
| The 7 template bib entries and the template images `fig-5/17/18.jpg`; the empty `appendix_1/2.tex` | `references.bib`, `thesis/images/`, `thesis/appendix/` | Delete them once nothing cites them |
| `RESULTS_SUMMARY.md` was built without a line-by-line re-read of `chapter_6.tex` (usage limit) | `thesis_tools/planning/RESULTS_SUMMARY.md` | Confirm each line against the chapter at its `\label` |
| hf275_strict at the pre-registered layers, and the five lab-only paper16_shared cells | Ch 5 `sec:r-personas`, `sec:r-transfer` | Read them on the lab PC if the text is to say more than it does now |
| Nothing has been compiled with TeX | whole thesis | Compile in Overleaf |

## Chapter 4 revision list (applied in one pass: CH5 plan session 5b, after Ch 5 sessions 1–5 and before the final verification)

Add any Chapter 4 gap found while writing Chapter 5 here, and do not edit Chapter 4 mid-way.

| # | Where in `chapter_5.tex` | Change | Source |
|---|---|---|---|
| 1 | §4.3.1, "The sarcasm pilot" (:320) | Add one sentence of motivation. Sarcasm is literally false but not meant to deceive, so the pilot tests whether a deception direction reads intent or form. | `cap/docs/SARCASM_DESIGN.md:30-45`; `audit/LIMITATIONS_AND_FUTURE_WORK.md` §2 |
| 2 | §4.3.1 (:321), the natural-sarcasm set | Cite FigLang 2020 (Ghosh, Vajpayee & Muresan, ACL 2020 workshop) and add it to `references.bib` with a DOI check. It is the only dataset in Chapter 4 without a citation. | `cap/scripts/build_sarcasm_natural.py:12,48` |

**Items 1–2 DONE 2026-09-25**, early, at the owner's request ("u do it"):
- `chapter_5.tex:320-323` edited;
- `references.bib` gained `ghosh2020figlang`, with metadata from the ACL Anthology BibTeX (second author Avijit Vajpayee; pp. 1–11; doi 10.18653/v1/2020.figlang-1.1);
- `check_latex.py` OK (16 files, 34 bib keys);
- recorded as rows 179–180 in `CH4_VERIFY_TABLE.md`.

The session 5b pass still runs for any items added later. Session 6 re-verifies these two sentences.

**Items added 2026-09-25: worked examples (owner chose "main text + appendix").** Each example is quoted word for word from a file, with `% src:` and its judge score or label. Long responses are cut, and each cut is marked `[\dots]`.

| # | Where in `chapter_5.tex` | Change | Source |
|---|---|---|---|
| 3 | §4.3.1 kd bank, after the three group bullets | One short example: the same kd question answered by a knowledge-denial role, a sincere-ignorance role and a plain-assertion role, all kept at score 3. It shows what each 4-v-3 contrast compares. | `kd/personas/kd_instructions/<role>.json`; responses `audit/cache/labpc/run-kd/responses/<role>.jsonl`; scores `audit/cache/labpc/run-kd/scores/<role>.json` (lab-only; re-check on the lab PC) |
| 4 | §4.4.3 "The HP-KR label", beside `fig:m-hpkr` | One deceptive denial (knows) next to one honest denial (does not know). These are the two sides of the text-matched slice. **Withhold the WMDP question**: show only the responses and the model's belief. | `results/subsets/harm-pressure-knowledge-report.csv` (Llama-3.3-70B rows; `meta.model_belief`; `eval_datasets.denies_knowledge`) |

**Items 3–4 DONE 2026-09-25** (pass 5b), written by W8 and verified; rows 181–185 of `CH4_VERIFY_TABLE.md`. The same pass:
- added `\label{ch:method}` and turned the hard-coded "Chapter~5" into `\ref{ch:results}`;
- qualified lines 272 and 278: "only sincerity differs" now holds on the four "does not know" questions; on the six "knows" questions the sincere-ignorance roles admit.

**Still open for Chapter 4** (one later pass; the list is in the last paragraph of `CH4_VERIFY_TABLE.md` and at U-13 of `RESULTS_SUMMARY.md`):
- RepE is "left out of its averages" only in the Liars' Bench average;
- Pipeline B has two RepE probes;
- Pipeline B's arm names are not introduced;
- Pipeline B has no dataset bootstrap;
- "any layer other than 20/22/40 is post hoc";
- the paper16 anchor's questions;
- the cap `{model_name}` literal (20 rollouts);
- role descriptions are verbatim for 22 of 25 (pp. 9–10);
- the cap judge filter (scores 2–3);
- the format-control banks and factor-probe corpora are not listed;
- "assistant direction" is never defined;
- there are two paper16 extractions.

**Other chapters.**
- **Ch 5:** a capitulation example in §5.6.2 and a sarcasm example in §5.6.3 (see `CH5_RESULTS_PLAN.md`).
- **Appendices:** see "Appendices" in `CH5_RESULTS_PLAN.md`.
  - A: every persona prompt and question;
  - E: one example per dataset and per persona group.
  - `main.tex:159,164` still carry the template appendix titles; replace them then.

**Other chapters.**
- Ch 2: a paragraph on irony and deceptive intent (Grice; Sperber & Wilson; Clark & Gerrig; Goldowsky-Dill's intent concession; Kirch et al.). There are no PDFs for the pragmatics works.
- Ch 6 future work: sarcasm v2 first.
- Details are in `CH5_RESULTS_PLAN.md`, "Items for other chapters".

## Files changed in the Chapter 5 session (2026-09-25)

Nothing was committed; the owner commits. `git status` at the end of the session shows these.

**Thesis (`thesis/`)**
- `chapters/chapter_6.tex`: Chapter 5, written; 335 KB, 723 `% src:` comments.
- `chapters/chapter_5.tex` (Chapter 4):
  - the sarcasm motivation and the FigLang citation;
  - the kd trio and the HP-KR pair examples;
  - `\label{ch:method}`;
  - `\ref{ch:results}` in place of "Chapter~5";
  - the 4 v 3 qualification.
- `chapters/chapter_9.tex`: `\label{ch:conclusion}` only.
- `main.tex`: `\appendix` with chapters A–E, labels `app:prompts`, `app:tables`, `app:repro`, `app:eng`, `app:examples`; the template appendices are removed.
- `appendix/appendix_A.tex`, `appendix_E.tex`: generated. `appendix_B.tex`, `appendix_C.tex`, `appendix_D.tex`: stubs.
- `bibliography/references.bib`: `ghosh2020figlang` and `aisi2026liedetection`.
- `images/`:
  - copied from the co-author at the pin: `collab_fig_transfer.pdf`, `collab_fig_hpkr.pdf`, `collab_fig_depth.pdf`, `collab_fig_operating.pdf`;
  - drawn from our data: `fig_convergence.pdf`, `fig_widening.pdf`, `fig_scale_ladder.pdf`, `fig_loco.pdf`, `fig_capitulation.pdf`, `fig_significance_forest.pdf`, `fig_sleeper.pdf`.
- `thesis/thesis_v1_45.zip`: untracked and not made by this session. Probably an Overleaf export; the owner decides whether to commit it.

**Tools and planning (`thesis_tools/`)**
- `planning/`:
  - `CH5_RESULTS_PLAN.md` (new: the plan, the owner's decisions, the non-reproducing values);
  - `CH5_VERIFY_TABLE.md` (new: 485 checks plus the lab-PC md5 table);
  - `RESULTS_SUMMARY.md` (new);
  - `PROGRESS_THESIS.md`;
  - `GLOSSARY.md` (§8: every label and the Chapter 5 terms);
  - `THESIS_BRIEF.md` (the "Chapter 5 decisions" block; two SUPERSEDED notes; the correction at :168);
  - `CH4_VERIFY_TABLE.md` (rows 179–185).
- `figures/`: `fig_convergence.py`, `fig_widening.py`, `fig_scale_ladder.py`, `fig_loco.py`, `fig_capitulation.py`, `fig_significance_forest.py`, `fig_sleeper.py`, and their `preview/*.png`. Each reads raw files and prints its numbers.
- `appendix/make_appendix_personas.py` and `make_appendix_examples.py`: the appendix generators. Re-run them; never hand-edit `appendix_A.tex` or `appendix_E.tex`.
- `label_index.py`: prints every `\label` with its title or caption, and the Chapter 5 terms.

**Audit and guide**
- `audit/DOC_FIXES.md`: DF-110 to DF-120 (proposals).
- `audit/cache/MANIFEST.csv`: new rows for the lab files copied this session.
- `audit/cache/labpc/`: the new byte-exact copies (43 files):
  - `run-sycophancy-mt/rows.jsonl`;
  - 39 collab-bank matrices (plus existing ones);
  - the cap_s23 and cap_ctrl matrices (48 + 42 files, under `run-crossgen/kd11/banks/`);
  - `kd/results/phase2_gate.json`;
  - seven logs under `logs/`.
- `thesis_guide/PITFALLS.md`: a dated addendum at the end.
- Not edited: `kd/docs/FINDINGS.md` (the ledger), `CLAUDE.md`, `CLAUDE.local.md`. Nothing was written on the lab PC.

**Memory:** `thesis-chapter-workflow` (plan from the audited maps, not raw); `labpc-fetch-over-ssh` (new).

## Files changed in the Chapter 4 session (2026-09-24/25)

**Thesis (untracked `thesis/`):**
- `thesis/chapters/chapter_5.tex`: Chapter 4, written.
- `thesis/main.tex`: added `amssymb`, `tikz` with `positioning, arrows.meta, patterns, decorations.pathreplacing`, and `todonotes`.
- `thesis/bibliography/references.bib`: 26 entries appended; the 7 template entries are still there.
- `thesis/images/fig_hpkr_cells.pdf`: new.
- `thesis/images/fig_personas_L40.pdf`: a copy of the collaborator's figure.

**Tools and planning (untracked `thesis_tools/`):**
- `thesis_tools/planning/`:
  - `THESIS_BRIEF.md`
  - `CH4_METHODOLOGY_PLAN.md`
  - `CH4_VERIFICATION_LOG.md`
  - `CH4_VERIFY_TABLE.md`
  - `PROGRESS_THESIS.md` (this file)
  - `GLOSSARY.md`
- `thesis_tools/check_latex.py`: the LaTeX static checker.
- `thesis_tools/figures/fig_hpkr_cells.py`, plus `preview/fig_hpkr_cells.png`.
- `thesis_tools/collab_evidence/16def2de/`: 20 collaborator files plus `MANIFEST.md`, all md5-verified.

**Other:**
- `CLAUDE.md`: one line added under "Writing the thesis?" (read these three files every session).
- No other tracked file was edited. The other ` M` entries in git status came before this session.

## Open questions for the team

1. **Chapter structure:** confirm with the supervisor that §-level "What We Built" inside Ch 4, and the template's six Result-Analysis headings with our themes as subsections in Ch 5, satisfy the template. Also confirm the appendices A–E now in `main.tex`.
2. **INLP / DANN / D3 files:** where are they? Ch 5 now quotes D3/D10 from report.pdf. If the files turn up in a dated folder, the numbers can be sourced from them; DANN still has none.
3. **Collaborator:**
   - commit the `span_mask` pooling code, `results_table.csv` and the check40 vectors;
   - fix three sentences in the other version (`full paper draft for this experiments/report.pdf`): Fig 4.1 "layer chosen by this loop", "sincere-ignorance … with the honest cluster", "about 300 MB".
4. **Ledger:** F-113 §5's "~81% (42/52)" doesn't reproduce; both splits give 199/258 and 39/52. Add a correction entry to `kd/docs/FINDINGS.md`? It is append-only, so that's the owner's call.
5. **`thesis_guide/METHODS_SOURCES.md:138`** says the 10+12 paper16 questions are "renderings, not verbatim". The 12 sycophancy questions are verbatim (checked against the paper PDF). Correct the guide?
6. **Headline caveat for Ch 5:** quote the base probe at its own best layer (L32, LB-average AUROC 0.8498, from `persona_depth/depth.csv`). The collaborator's ea1ce52b message says to.
7. **Commit the kd scale-rung drivers** (`expLB-resplit\run_kd_scale.ps1`, `run_kdscale_crossgen.ps1`), which are lab-PC only.
8. **D3 (on-policy AISI probe) and D10 (INLP) result files** (added 2026-09-25). Their numbers exist only in the co-author's `full paper draft for this experiments/report.pdf` (§5.2.1, §5.2.4, Fig 5.4). No file is on any branch.
   - Ch 5 quotes report.pdf with page `% src:` comments (owner decision, 2026-09-25).
   - Ask the co-author to commit the files.
9. ~~**Unified PC1 vs question family: −0.98 or −0.945?**~~ **SETTLED 2026-09-25:** two different statistics, both reproduced: cos(PC1, question-set direction) −0.9800, and the Pearson r over the 26 roles −0.9446. Ch 5 `sec:r-personas` gives both.
10. **Corrections proposed while writing Ch 5 (DF-110 to DF-120, `audit/DOC_FIXES.md`).**
    - Accept them?
    - Append ledger entries? `kd/docs/FINDINGS.md` is append-only and was not edited. The largest are:
      - "pooled probe wins CG, IT, ID, RepE" is wrong: it wins CG only (DF-112);
      - 24 of 90 cells, not 23 (DF-110);
      - the sign inversion is 9 cells, not 7 (DF-111);
      - D8 is above 0.5 BA at 15 of 16 layers (DF-115);
      - F-120's artifact holds on 5 datasets only (DF-114);
      - DF-70's clear counts need the control named (DF-116).
11. **Two new computations would close the last two Ch 5 `\todo`s:** Gate 3 on natural sarcasm, and a trait-vector contrast. Run them on the lab PC, or drop the trait-vector sentence?
12. **hf275 strict labels:** they were drafted by an AI assistant on 2026-09-18 (`hf275_role_groups_strict.json`), with a review sheet at `docs/thesis/role-labels/role_labels_review.csv`. Was the review done? Ch 5 reports the strict bank's result.
13. **`thesis/thesis_v1_45.zip`** is untracked, and this session did not create it. Commit it or ignore it?
14. **Pipeline B's `report.pdf` errors:** Ch 5 corrects them in our text. Should the co-author fix them in his version? They are:
    - "that test has not been run";
    - "a pooled probe … is not compared";
    - "top-tenth is the better read";
    - the ten kd roles credited to him.

## Next step

Chapter 5 is written and verified. Remaining, in order:
1. Owner review of Chapter 5 in Overleaf (compile; the static checker cannot run TeX). Check `RESULTS_SUMMARY.md` against the chapter as you read (U-1).
2. Appendices B, C, D (stubs now).
3. Chapter 6 (Conclusion), which collects every caveat and the lab-only list; then Chapter 1, Chapters 2 and 3, the abstract.
4. The Chapter 4 items the writers listed (`CH4_VERIFY_TABLE.md`, last paragraph) in one revision pass.
5. Ask the co-author for the D3/D10 result files and the `span_mask` pooling code (`PROGRESS_THESIS.md` open questions 3 and 8).
