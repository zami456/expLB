# Thesis writing guide: READ THIS BEFORE WRITING ANY THESIS TEXT

**Who this is for:** any Claude session, or person, that writes or edits the thesis report. The
LaTeX lives in `thesis/`: `thesis/main.tex`, `chapters/`, `core/`, `appendix/`, `bibliography/`,
`images/`. The owner copies `thesis/` to Overleaf and compiles it there.

**Why it exists:** two co-authors worked in parallel without syncing.
- The owner's branch is `asix`.
- zami456's branches are `origin/paper` and `origin/feat/persona-pc-ood`.
- Mahdi is the third co-author.

The findings, figures and methods are spread across both branches, a gitignored-then-tracked audit,
a lab PC and Drive. This folder tells you where every piece is, which version is the verified one,
and what must not be written.

Built 2026-09-24 against the pins in `audit/BRANCH_PINS.md`: `origin/paper` @ 16def2de and `asix`
@ e917ecd3 plus the working tree. **If a branch has moved past its pin, the guide is stale for that
branch.** Run `git log <pin>..<branch>` and re-verify before quoting anything new.

## Rules for the `thesis/` folder

1. `thesis/` holds **only** the thesis LaTeX sources and the files the LaTeX includes: figures go
   in `thesis/images/`, the bib in `thesis/bibliography/`. No notes, scripts, CSVs or markdown go
   there. Working notes belong in this folder, `thesis_guide/`.
2. It must compile standalone in Overleaf.
   - Every `\includegraphics` path is relative to `thesis/`, and every figure it uses is copied into `thesis/images/`.
   - Every `\cite` key exists in `thesis/bibliography/references.bib`.
   - Nothing may reference a file outside `thesis/`.
3. Never write a number into the LaTeX unless you have traced it to raw data. See the checklist below.

## Read in this order

| # | File | What it gives you |
|---|---|---|
| 1 | `thesis_guide/PITFALLS.md` | What must not be said, with the correct wording and its source: superseded numbers, the collaborator's overstatements, attribution, citation traps. **Read it fully.** |
| 2 | `thesis_guide/RESULTS_SOURCES.md` | Twelve result themes (T1–T12). For each: the verified claims, the "do not quote" numbers, the raw source, the re-derivation script, and whether both branches have a version. |
| 3 | `thesis_guide/METHODS_SOURCES.md` | Every methods topic: the authoritative description, the code on each branch, the exact parameters, the caveats, and a citation map. |
| 4 | `thesis_guide/FIGURES.md` and `figures_inventory.csv` | Every figure on both branches: what it shows, the script that made it, how stale it is, its caveats, and which to use per chapter. |
| 5 | `kd/docs/FINDINGS.md` | The ledger, F-01…F-128, append-only. **Check the correction chain:** a later entry may correct an earlier one. F-128 corrects F-111, F-112, F-117, F-118, F-120 and F-127. |
| 6 | `audit/p17/P17_FACTS.md`, `audit/BRANCH_PINS.md` | The latest verified state and the pins. |
| 7 | `audit/claims.csv` | The verdict for every checked claim (643 rows): HOLDS, MATCHES HIS DATA, PARTLY, DOES NOT HOLD, UNRESOLVED. Find a claim's row before you repeat it. |

Drafts to start from, never to trust blindly. Each is a map, and each has correction blocks inside:
- Methods: `audit/METHODS_CHAPTER.md`
- Results: `audit/RESULTS_CHAPTER.md`
- Limitations: `audit/LIMITATIONS_AND_FUTURE_WORK.md`
- Axes: `audit/RESULTS_BY_AXIS.md`
- Collaborator's work: `audit/COLLABORATOR_FINDINGS.md`
- Owner's write-ups: `docs/thesis/*.md`

The collaborator's own paper is at `origin/paper` @ 16def2de,
`src/probes/deception-detection/paper/`; every number in it is in `generated/numbers.tex`.

## Where every kind of thing lives

For the full table, see `RESULTS_SOURCES.md` §"Where each kind of thing lives".

- **Owner's code and stores** (on `asix`): `src/assistant_axis_ndif/`, `kd/`, `cap/`, `scale/`, `run-crossgen/`, `run-coords/`, `run-paper16/`, `figures/`, `docs/`.
- **Collaborator's code, results and paper** (`origin/paper`, read with `git show 16def2de:<path>`): `src/probes/deception-detection/` including `results/` and `paper/`.
- **Lab-only evidence** (lab PC `C:\Users\user4\expLB\` and `expLB-resplit\`), with local cached copies in `audit/cache/labpc/` and `audit/p17/labcache/`. Findings that rest on it: F-110, F-113 (part), F-114, F-115, F-116, F-120, F-121, F-126.
- **The audit** (`audit/`): claims, maps, re-runnable scripts (`audit/scripts/`, about 400), and reports.

## Pre-writing checklist, per section

For every section, before writing:

- [ ] I read `PITFALLS.md` and checked this section's topic against it.
- [ ] For each number:
  - [ ] I found its `claims.csv` row or ledger F-entry.
  - [ ] I checked for a later correction (F-128, DOC_FIXES, a correction block).
  - [ ] I traced it to the raw file named in `RESULTS_SOURCES.md` or `METHODS_SOURCES.md`, at the pinned SHA for the collaborator's branch.
- [ ] Each number names its bank, arm, dataset set, layer, k, probe and metric. A margin from a single layer or a single seed is labelled as such.
- [ ] Every transfer result carries its controls: raw, the matched `datapca`, random directions, and response length.
- [ ] **Both branches:** if both authors have a version of this result, I reported both, or said why one was chosen. I picked the "better" one only after checking that the setups match (see `RESULTS_SOURCES.md` and `audit/phase3/DISAGREEMENTS.md`). The two branches' "8v7 axis" and raw baselines differ in protocol.
- [ ] **Attribution:** I credited the work to the right author (`PITFALLS.md` §attribution; `audit/methods/ATTRIBUTION.md`).
- [ ] Lab-only numbers are either re-checked from the cache or marked as resting on lab-only data.
- [ ] Every figure used was chosen from `FIGURES.md`, its caveats were checked, and it was copied into `thesis/images/`.
- [ ] No template placeholders remain: the sample images `thesis/images/fig-{5,17,18}.jpg`, the template entries in `references.bib`, and the template sample text. Leftovers compile silently, so check.
- [ ] Every citation exists in `references.bib` with the correct authors. MASK is Ren et al. (CAIS + Scale AI), not Cadenza Labs.

## Three findings anything written must get right

These come from `CLAUDE.local.md`, with the P17 corrections applied.

1. **The convergence is ONE finding, not three.** Where both classes deny knowledge, no persona
   direction reads the lie:
   - F-115: 0.4630 / 0.6203 / 0.5383, located in `audit/HPKR_WORDING_CONTROLLED.md` §5;
   - F-116: the 2,001-row control;
   - F-118: the collaborator's Gender-Secret, 0.436–0.623.
2. **The positive control: publish +0.125, not +0.178.** Pooled training without personas, under
   leave-one-cluster-out, positive for 10 of 11 targets at L40 (F-108).
   - **Caveat:** +0.125 is one seed at three layers, with no dataset bootstrap (F-108 caveats).
   - The "five seeds" belong to the leaky +0.178 leave-one-dataset-out figure (`audit/REPORT_PLAIN.md:106`). Do not attach them to +0.125.
3. **Scale ladder (F-121): both banks decline, so the decline belongs to the method.**
   - 26-persona: +0.038 → +0.015 → −0.012.
   - 36-persona kd s23: +0.036 → +0.019 → +0.005.
   - These are per-layer best-feature-minus-raw values, a post-hoc maximum, not the matched control.

## Keeping this guide current

When a new result, figure or correction lands:
1. Append it to the ledger (`kd/docs/FINDINGS.md`, a new F-entry).
2. Add the row to the relevant file here.
3. If a branch moved, update `audit/BRANCH_PINS.md`.

Never delete guide content. Mark it superseded, with the date and the source.
