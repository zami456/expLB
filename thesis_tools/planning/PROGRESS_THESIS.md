# PROGRESS_THESIS: where the thesis stands

**Last updated:** 2026-09-25 (end of session).

**Read at the start of every session,** together with `THESIS_BRIEF.md` and `GLOSSARY.md`.

## Status by chapter (the BracU template keeps 6 main chapters)

| Ch | Title (`main.tex`) | File | Status |
|---|---|---|---|
| — | Front matter (title, declaration, approval, ethics, abstract, dedication, acknowledgement) | `thesis/core/*.tex` | template placeholders |
| 1 | Introduction | `chapters/chapter_1.tex` | template placeholder |
| 2 | Literature Review | `chapters/chapter_2.tex` | template placeholder |
| 3 | Requirements, Impacts and Constraints | `chapters/chapter_3.tex` | template placeholder. Economic analysis is settled: "no direct cost". |
| **4** | **Proposed Methodology** | **`chapters/chapter_5.tex`** | **WRITTEN and VERIFIED** (see below). Not compiled yet: there is no TeX here, so compile in Overleaf. |
| 5 | Result Analysis | `chapters/chapter_6.tex` | template placeholder. The plan is in `THESIS_BRIEF.md` Part 3 as two parts: I, zero-shot with our personas; II, other builds, controls and failed methods. |
| 6 | Conclusion | `chapters/chapter_9.tex` | template placeholder |
| — | Appendices A–D | `appendix/*.tex` | empty; the template titles are still in `main.tex` |

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

## `\todo` markers left

**None in Chapter 4.** Items that will need a `\todo` or a fix when their chapter is written:

| Item | Where it will appear | What resolves it |
|---|---|---|
| INLP, DANN, D3 (on-policy AISI probe) numbers | Ch 5 failed methods | The owner or collaborator supplies the files; the owner says they are outside the repo. Otherwise write `\todo[inline]{MISSING: ...}` |
| The 13 remote layers' pooling code (`trace_layer_outputs(span_mask=…)`) | Ch 4 disclosure (already written); Ch 5 depth / L36 | The collaborator commits it. Then `git fetch` and delete the disclosure |
| Pipeline B's four-decimal reproduction and L40 check (0.99989) | reported as not reproducible (Ch 4 §4.2.8) | The collaborator commits `results_table.csv` and the check40 vectors (audit D-063, D2-024, D2-025) |
| Six classic references (BH, Maris, Kriegeskorte, Ravfogel, Ganin, Sharma): volume/pages from memory | `references.bib` | A DOI check |
| The 7 template bib entries and the template images `fig-5/17/18.jpg` | `references.bib`, `thesis/images/` | Delete once no chapter cites them |
| Template appendix titles in `main.tex:159,164` | `main.tex` | Replace with Appendices A–D |

## Files changed in this session (2026-09-24/25)

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

1. **Chapter structure:** confirm with the supervisor that §-level "What We Built" inside Ch 4, and Result Analysis I/II inside Ch 5, satisfy the template.
2. **INLP / DANN / D3 files:** where are they? Can they be put in a dated folder so Ch 5 can give numbers?
3. **Collaborator:**
   - commit the `span_mask` pooling code, `results_table.csv` and the check40 vectors;
   - fix three sentences in the other version (`full paper draft for this experiments/report.pdf`): Fig 4.1 "layer chosen by this loop", "sincere-ignorance … with the honest cluster", "about 300 MB".
4. **Ledger:** F-113 §5's "~81% (42/52)" doesn't reproduce; both splits give 199/258 and 39/52. Add a correction entry to `kd/docs/FINDINGS.md`? It is append-only, so that's the owner's call.
5. **`thesis_guide/METHODS_SOURCES.md:138`** says the 10+12 paper16 questions are "renderings, not verbatim". The 12 sycophancy questions are verbatim (checked against the paper PDF). Correct the guide?
6. **Headline caveat for Ch 5:** quote the base probe at its own best layer (L32, LB-average AUROC 0.8498, from `persona_depth/depth.csv`). The collaborator's ea1ce52b message says to.
7. **Commit the kd scale-rung drivers** (`expLB-resplit\run_kd_scale.ps1`, `run_kdscale_crossgen.ps1`), which are lab-PC only.

## Next step

Write Chapter 5 (Result Analysis) from `THESIS_BRIEF.md` Part 3, section by section, following the chapter workflow:
1. Plan it.
2. Ask the owner the open questions.
3. Write.
4. Verify against both sides.
5. Run an independent verifier subagent plus `check_latex.py`.
