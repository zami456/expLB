"""P17 Agent E: insert dated UPDATE / CORRECTION blocks into the audit's narrative documents.

INSERT ONLY. Every edit is (original line number, anchor substring that line must contain, block).
Blocks are inserted immediately before the original line; nothing existing is changed or moved.
Line endings are preserved (all target files are CRLF). Runs once: refuses to run if a file no
longer matches its pre_wave3 snapshot. Writes a log of every insertion to audit/p17/E_LOG.md.

Usage: python audit/scripts/p17_e_insert.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "audit" / "p17" / "pre_wave3"
LOG = ROOT / "audit" / "p17" / "E_LOG.md"

# ---------------------------------------------------------------- reusable blocks
PM = """> **CORRECTION 2026-09-24 (DF-101, P17).** "Within ±0.06 at every layer" and "crosses zero five
> times" are both wrong. Excluding HP-KR, `ext_axis1 − ext_pc3` reaches **+0.0633 at L56**, and the
> series changes sign **6** times. Still correct: 11/16 positive, +0.0086 (L20), +0.0012 (L24), and
> negative at L64/68/72. No persona-arm cell changed between ea1ce52b and 16def2de.
> **Source:** `src/probes/deception-detection/results/persona_ood_ext_L{20..79}/summary.csv`,
> `off_diagonal_excl_hpkr` @ 16def2de; `audit/p17/ledger_verify_stdout.txt:223-224`.
> **How the claim is updated:** *"excluding HP-KR the margin is positive at 11 of 16 layers, lies
> between −0.033 and +0.063, and changes sign six times."*
> _The text below is superseded on this point and is kept for the record._"""

COV = """> **UPDATE 2026-09-24 (P17).** The coverage and draft-verdict figures here are superseded at the P17
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
> `audit/p17/paper_new_claims.csv`; `audit/p17/A_REPORT.md` §1 — @ 16def2de / e917ecd3."""


def f120(values: str) -> str:
    return f"""> **CORRECTION 2026-09-24 (DF-102, P17).** The raw/axes/random values below ({values}) are the
> **5-dataset (paper5) reading** of `passA`, and the text does not say so. On all 10 datasets, raw at
> L40 is **0.6760**. The shape of the argument (raw declines, random gains) is unaffected.
> **Source:** `audit/p17/C_REPORT.md` F-120 row; `audit/p17/fmap_F110_F127.csv` (LAB-ONLY: `passA`
> matrices); `kd/docs/FINDINGS.md` F-128. _Kept for the record below._"""


DF100 = """> **CORRECTION 2026-09-24 (DF-100, P17).** The three values 0.4630 / 0.6203 / 0.5383 are sourced
> wrongly: they are at `audit/HPKR_WORDING_CONTROLLED.md` §5 (:121), not in F-115, and reproduce from
> the collab-bank kd11 matrices. The convergence finding is unchanged.
> **Source:** `audit/p17/C_REPORT.md` (F-118 row); `kd/docs/FINDINGS.md` F-128.
> _The text below is superseded on this point and is kept for the record._"""

DF105 = """> **UPDATE 2026-09-24 (DF-105, P17).** The **0.9597** ("does not affirm knowledge") regex value below
> was **not re-derived**: the regex is recorded nowhere. The guess `NOT ^(yes|i know)` gives **0.9413**,
> still above the 0.919 headline, so the argument stands; 0.9125 is unaffected.
> **Source:** `audit/p17/correction_blocks_check.csv` (row for `docs/NOVELTY_CHECK_PROMPT.md:127`);
> `audit/p17/C_REPORT.md`."""

DF107 = """> **UPDATE 2026-09-24 (DF-107, P17).** Resolved by the collaborator at 2ad758fb, in his favour: the
> paper now reports the **L36 optimum** (`results.tex:228-230`, "Ranking peaks at
> layer~\\V{depth-off-ext-peak-layer}, not at layer~40"; numbers.tex `depth-off-ext-peak-layer`=36,
> `depth-off-ext-peak`=0.927), and the "outscores … at every depth" claim is gone — `results.tex:136`
> now counts wins over the sweep (`depth-pc-wins-complement`=13 of `depth-n`=16). The one remaining
> "every depth" (`results.tex:208`) is a different claim, about the base probe inverting.
> **Source:** `src/probes/deception-detection/paper/sections/results.tex` and
> `paper/generated/numbers.tex` @ 16def2de; `audit/p17/A_REPORT.md` §2."""

DF106_PROV = """> **CORRECTION 2026-09-24 (DF-106, P17).** The single pin below is wrong for part of the pass.
> - **Level 2** (p16_level2, 15:39) read ea1ce52b.
> - **Level 1** (verdicts written 16:40, after the ref moved at 15:44) read 2ad758fb for at least
>   D-171 and D-032.
> 172 of the 176 stored values are identical at ea1ce52b, 2ad758fb and 16def2de, so the stamp
> does not matter for them. The audit's current pin is **16def2de**. At that pin **one verdict
> changes: D-189 → PARTLY** (Table 7, CG raw L40 = 0.7873 → 0.787). Status now reads 165 / 9 / 2.
> **Source:** `audit/p17/A_REPORT.md` §1; `audit/p17/d_rows_repinned.csv`; `audit/claims.csv` D-189 @ 16def2de.
> _Kept for the record below._"""

DF106_MACRO = """> **CORRECTION 2026-09-24 (DF-106, P17).** Read "1,161 macros at 2ad758fb (1,176 at 16def2de),
> beside eleven `tab_*.tex`". numbers.tex has 1,162 *lines* only at 2ad758fb (1,161 macros plus a
> header); there are 11 `tab_*.tex` at every SHA. The conclusion (the release is well evidenced) stands.
> **Source:** `audit/p17/A_REPORT.md` §1 (D-171, D-032); `audit/p17/C_REPORT.md` F-127 row;
> `git ls-tree 16def2de src/probes/deception-detection/paper/generated/`."""

DF109 = """> **UPDATE 2026-09-24 (DF-109, P17).** "Never trained on our side" is stale: 6D-1 trained
> `pc:sycophancy` in our pipeline (`run-coords/passB`, paper16 bank, 11 datasets). It **loses to raw
> at all six (layer, pooling) cells**, by −0.008 to −0.118, and turns positive only without
> `sycophancy_mt`, under mean pooling at L40/L44 (+0.0017 / +0.0342 over raw). Our-pipeline cell is
> now ✅ 70B only.
> **Source:** `kd/docs/FINDINGS.md` F-126 §1 (LAB-ONLY store); `audit/p17/C_REPORT.md` F-126 row (HOLDS 41/41)."""

NEW_DOES_NOT_HOLD = """> **UPDATE 2026-09-24 (P17).** Two sentences added to the paper after this chapter was drafted do
> not hold at the pin 16def2de:
> - **D2-005, `introduction.tex:68-70`** — "from layer~36 it ranks \\hpkr{} better than any contrast
>   we build". The base probe leads at 8 of the 12 layers from 36 and trails the best contrast at
>   **L44 0.828 vs 0.908, L48 0.837 vs 0.890, L52 0.877 vs 0.883, L76 0.838 vs 0.925** (DF-96).
> - **D2-019, `limitations.tex:17-21`** — "≥0.026 on the other datasets at all 16 layers". The 0.026
>   includes HP-KR; excluding it the margin is **−0.011 at L20** and below 0.026 at 8 of 16 layers (DF-97).
> **Source:** `audit/p17/paper_new_claims.csv` D2-005, D2-019; `results/persona_depth/depth.csv` and
> `persona_ood_ext_L*/summary.csv` @ 16def2de; `audit/p17/A_REPORT.md` §2."""

# ---------------------------------------------------------------- per-file edits
# (orig_line, anchor, block, style)  style: "para" = blank-line separated, "bullet" = raw lines
E: dict[str, list] = {}

E["audit/PROGRESS.md"] = [
    (3, "Read `audit/BRIEF.md` first", """> **UPDATE 2026-09-24 (DF-108, P17).** This header is historical. Phases 0–5, Phase 6 (6A, 6D, 6E;
> 6B/6C skipped by instruction) and the P17 re-pin are complete, and nothing is running. The current
> state is the section `# P17 — RE-PIN (2026-09-24)` at the end of this file, together with
> `audit/BRANCH_PINS.md`.
> **Source:** `audit/phase6/RESULTS.md:15-21`; `audit/BRANCH_PINS.md`.""", "para"),
    (2579, "**1. D-028 GENERALISES", PM, "para"),
    (2676, "**NOTHING IS RUNNING.**", """> **UPDATE 2026-09-24 (DF-108, P17).** This handoff is superseded on its counts. Read the P17 section
> at the end of this file first.
> - **Ledger:** F-01…F-127 at the pin (md5 dc365834…, `audit/BRANCH_PINS.md`); F-123…F-127 are
>   Phase 6. **F-128** (the P17 re-pin entry) has since been appended, so the ledger holds 128.
> - **DOC_FIXES:** DF-01…DF-95, plus DF-96…DF-109 proposed by P17. 11 correction blocks were applied
>   in 10 files (`audit/phase6/APPLY_SHEET.md` §1); P17 checked 94 of their numbers: 86 match, 5 are
>   stale line pointers, 1 is a partial mislabel, 2 are unverified (`audit/p17/correction_blocks_check.csv`).
> - **Results chapter:** now exists (`audit/RESULTS_CHAPTER.md`, Phase 6A).
> - **git status outside audit/:** 11 modified files plus the untracked `thesis_tools/` at the pin.
> - **Pins:** origin/paper @ 16def2de (was ea1ce52b); asix @ e917ecd3.
> - **Coverage:** 533 of 735 inventory claims have a verdict (72.5%); 202 remain (110 ledger, 92
>   collaborator notes).
> **Source:** `audit/BRANCH_PINS.md`; `audit/phase6/APPLY_SHEET.md` §1;
> `audit/p17/correction_blocks_check.csv`; `python audit/scripts/p15_coverage.py`.""", "para"),
    (2700, "1. **The convergence (the strongest negative).**", DF100, "para"),
]

PROGRESS_END = """
---

# P17 — RE-PIN (2026-09-24)

# ============================================================================
# HANDOFF — READ THIS FIRST IF YOU ARE A NEW SESSION (2026-09-24, after P17)
# ============================================================================

**NOTHING IS RUNNING.** No scheduled task, no background watcher, no NDIF job. P17 read the lab PC
read-only (BatchMode ssh, md5-verified) and started nothing (`audit/p17/C_REPORT.md`).
**Nothing from P17 is committed. The owner commits, not the audit.**

## Pins (`audit/BRANCH_PINS.md`)

- `origin/paper` @ **16def2de** (2026-09-24 16:47). The previous audit mark was ea1ce52b.
- `origin/feat/persona-pc-ood` @ 45b16d90 — an ancestor of paper, nothing newer.
- `asix` @ **e917ecd3** plus the uncommitted working tree (md5s in `BRANCH_PINS.md`).
  `kd/docs/FINDINGS.md` was md5 dc365834 at the pin (F-01…F-127); F-128 has since been appended.
- Branch owners: `asix` is the owner's; `origin/paper` and `persona-pc-ood` are zami456's; Mahdi has
  one commit, df758c4b, on `feat/llama-layer30` (`thesis_tools/survey/BRANCHES.md`).
- Every later read of another branch goes through a pinned SHA via `audit/scripts/p17_common.py`.

## What P17 found

1. **Provenance conflict (DF-95 made concrete, DF-106).** The ref moved ea1ce52b → 2ad758fb (15:44)
   → 16def2de while Phase 6 ran. Level 2 (15:39) read ea1ce52b; Level 1 (verdicts 16:40) read
   2ad758fb for at least D-171 and D-032. 172 of the 176 stamped values are identical at all three
   SHAs. (`audit/p17/A_REPORT.md` §1; `audit/p17/d_rows_repinned.csv`)
2. **One draft verdict changes: D-189 MATCHES → PARTLY** (CG raw L40 0.7875 at ea1ce52b, 0.7873 at
   the pin; the paper now prints 0.787). **The draft reads 165 / 9 / 2** over 176. D-195 is now
   35/36; D-009/D-027 drift in the 4th decimal. (`audit/claims.csv` D-189; `audit/p17/d_rows_other_affected.csv`)
3. **zami456's three new commits** — 2ad758fb (depth sweep in the paper, 16 layers; app:depth; 4
   claims corrected; 12 unified-axis CSVs; baseline re-run, 80 cells, max |Δ| 0.0054), d1b88154 and
   16def2de (text and numbers.tex only). 83/83 recomputable numbers.tex keys match. 26 new claims
   D2-001…026: **19 HOLDS / 3 PARTLY / 2 DOES NOT HOLD / 2 UNRESOLVED**.
   - **D2-005** (`introduction.tex:68-70`): the base probe trails the best contrast at L44 (0.828 vs
     0.908), L48 (0.837 vs 0.890), L52 (0.877 vs 0.883) and L76 (0.838 vs 0.925).
   - **D2-019** (`limitations.tex:17-21`): the ≥0.026 includes HP-KR; without it −0.011 at L20.
   (`audit/p17/A_REPORT.md` §2; `audit/p17/paper_new_claims.csv`; `audit/p17/csv_drift.csv`)
4. **Ledger F-110…F-127:** 447 numbers re-derived from raw, 432 match; no headline measurement fails.
   12 HOLD; **6 PARTLY: F-111, F-112, F-117, F-118, F-120, F-127.** LAB-ONLY evidence: F-110, F-113
   (part), F-114, F-115, F-116, F-120, F-121, F-126. (`audit/p17/C_REPORT.md`; `audit/p17/fmap_F110_F127.csv`)
5. **Applied correction blocks:** 94 numbers, 86 match; 5 stale pointers, 1 passC11 mislabel,
   2 unverified (0.9597; "median gap 0.012 over 17 methods"). (`audit/p17/correction_blocks_check.csv`)
6. **Tables merged** (`audit/scripts/p17_merge.py`, backups `*.before-p17`): claims.csv 600 → 643,
   claims_inventory.csv 692 → 735, fmap_all.csv F-01…F-127, experiments.csv 114 → 119.
7. **Coverage: 533 / 735 = 72.5%** (was 490 / 692). (`python audit/scripts/p15_coverage.py`)
8. **F-128** appended to `kd/docs/FINDINGS.md` — the corrections to the six PARTLY entries live
   there; quote those entries only through it.
9. **Dated update/correction blocks** inserted (insert-only) into the audit's narrative documents;
   log and insert-only check in `audit/p17/E_LOG.md`. `audit/DOC_FIXES.md` gains DF-96…DF-109
   (next free id DF-110).

## Reading order

`audit/BRANCH_PINS.md` → `audit/p17/P17_FACTS.md` → `audit/p17/A_REPORT.md` and
`audit/p17/C_REPORT.md` → `kd/docs/FINDINGS.md` **F-128**. Then `audit/DOC_FIXES.md` §C-septies.

## Open items

- **DF-96…DF-99 are for zami456**: D2-005 and D2-019 (DOES NOT HOLD), the three PARTLY sentences
  (D2-002, D2-016, D2-018), and the release gaps (D2-024, D2-025).
- **202 inventory claims still have no verdict**: 110 ledger (6C) and 92 collaborator notes (6B).
- **The owner still has to commit.** At the pin `git status` outside `audit/` was 11 modified files
  plus the untracked `thesis_tools/`; the audit has never committed anything.

**Source:** `audit/BRANCH_PINS.md`; `audit/p17/P17_FACTS.md` and the evidence files it cites
(`A_REPORT.md`, `C_REPORT.md`, `paper_new_claims.csv`, `d_rows_repinned.csv`, `correction_blocks_check.csv`).
"""

E["audit/REPORT_PLAIN.md"] = [
    (5, "THIS REPORT CLOSES AT 2026-09-23", COV + "\n\n" + PM + """

> **UPDATE 2026-09-24 (P17).** F-123…F-127 (Phase 6), F-128 (P17) and the collaborator's commits
> 2ad758fb / d1b88154 / 16def2de are also absent from this report. **Source:** `kd/docs/FINDINGS.md`;
> `audit/p17/A_REPORT.md`; `audit/BRANCH_PINS.md`.""", "para"),
    (199, "2. **The central dataset is largely solvable from wording.**", DF105, "para"),
    (317, "- The draft's headline Contributions margin", PM, "para"),
    (333, "| route | result |", DF100, "para"),
    (342, "**The one thing I flagged as unexplained is now solved.**",
     f120("raw 0.7834 → 0.7426, random +0.127 from L72 to L79, axes − random +0.109 at the top vs +0.230 at L40"), "para"),
    (354, "**Two smaller things:** the optimum layer is **36, not 40**", DF107, "para"),
]

E["audit/REPORT_TECHNICAL.md"] = [
    (5, "THIS REPORT CLOSES AT 2026-09-23", COV + """

> **UPDATE 2026-09-24 (P17).** Also absent from this report: F-123…F-127, F-128, and origin/paper
> commits 2ad758fb (depth sweep in the paper; 4 claims corrected; baseline re-run, 80 cells, max
> |Δ| 0.0054), d1b88154 and 16def2de. **Source:** `audit/p17/A_REPORT.md`; `audit/p17/csv_drift.csv`.""", "para"),
    (281, "2. **HP-KR is ~91–96% solvable from wording.**", DF105, "para"),
    (379, "New branch, 3 commits, 683 files", """> **UPDATE 2026-09-24 (P17).** origin/paper has moved to **16def2de** (3 more commits after ea1ce52b,
> 2026-09-24 15:34–16:47). See `audit/COLLAB_PAPER_BRANCH.md` and `audit/BRANCH_PINS.md`.
> **Source:** `git log ea1ce52b..16def2de`; `audit/p17/A_REPORT.md`.""", "para"),
    (386, "| comparison | with HP-KR | excl HP-KR |", PM, "para"),
    (391, "**D-028 generalises from 3 layers to 16.**", DF107, "para"),
    (393, "**The L76 rise is SOLVED (F-120).**",
     f120("raw 0.7834 → 0.7426, axes 0.7492 vs mid-stack 0.8145, random +0.127, axes − random +0.2296 vs +0.1087"), "para"),
]

E["audit/RESULTS_CHAPTER.md"] = [
    (11, "**Provenance of everything marked [RE-DERIVED].**", DF106_PROV, "para"),
    (230, "**Table 7 — per test dataset.**", """> **UPDATE 2026-09-24 (DF-106, P17).** At 16def2de the collaborator re-ran the baseline: CG raw L40
> = 0.7873, and the paper now prints 0.787 (`pt-base-cg-L40`). So 35 of 36 values match the draft
> PDF, and the draft's 0.788 is the ea1ce52b value. No persona column changed.
> **Source:** `audit/p17/d_rows_other_affected.csv` (D-195); `audit/claims.csv` D-189;
> `paper/generated/numbers.tex` @ 16def2de.""", "para"),
    (422, "**(d) \"The top tenth of response tokens", NEW_DOES_NOT_HOLD, "para"),
    (437, "`paper/generated/numbers.tex` carries **1,162", DF106_MACRO, "para"),
    (524, "One caveat has since been vindicated.", DF107, "para"),
    (533, "- **Everything here is one seed, his pipeline, three layers.**", PM, "para"),
]

E["audit/RESULTS_BY_AXIS.md"] = [
    (116, "| axis | our pipeline (`run-crossgen/`)", DF109, "para"),
    (131, "| layer | his baseline | best arm |", """> **UPDATE 2026-09-24 (P17).** The baseline column below was read @ ea1ce52b. At the pin 16def2de
> the collaborator's re-run baseline reads **0.7580** (L40) and **0.7296** (L44), −0.0001 each; the
> "vs baseline" column moves in the 4th decimal only.
> **Source:** `audit/p17/C_REPORT.md` F-122 row; `audit/p17/csv_drift.csv`.""", "para"),
]

E["audit/COLLAB_PAPER_BRANCH.md"] = [
    (19, "", """> **UPDATE 2026-09-24 (P17).** Three more commits after `ea1ce52b`, audited at the pin `16def2de`
> (`audit/BRANCH_PINS.md`):
>
> | commit | date | subject | what P17 found |
> |---|---|---|---|
> | `2ad758fb` | 09-24 15:34 | *The depth sweep in the paper: a curve replaces three layers, and corrects four claims* | depth sweep written into the paper (16 layers), app:depth, 4 claims corrected, 12 new unified-axis (13v12) result CSVs; baseline re-run (BLAS drift: 80 cells, max \\|Δ\\| 0.0054, all raw-probe) |
> | `d1b88154` | 09-24 16:43 | *Make the case against the trained probe from the whole depth curve, not one layer* | paper text + numbers.tex only |
> | `16def2de` | 09-24 16:47 | *Correct an overstated HP-KR ranking claim: 8 of 12 deep layers, not all of them* | abstract/limitations now say "8 of 12"; `introduction.tex:68-70` still does not (D2-005) |
>
> Of the 26 new claims (D2-001…026): 19 HOLDS, 3 PARTLY, 2 DOES NOT HOLD (D2-005, D2-019), 2
> UNRESOLVED. 83/83 recomputable numbers.tex keys match; `persona_depth/depth.csv` agrees with the
> per-layer files in every cell.
> **Source:** `git log ea1ce52b..16def2de`; `audit/p17/A_REPORT.md` §2; `audit/p17/paper_new_claims.csv`;
> `audit/p17/csv_drift.csv`.""", "para"),
    (88, "| | L40 | L70 (trough) | L72 | L79 |",
     f120("raw 0.8195 at L40 → 0.7426 at L79; random +0.127 L72 → L79; axes − random +0.2296 vs +0.1087"), "para"),
    (142, "- **With HP-KR: positive at 13 of 16 layers.**", PM, "para"),
    (194, "1. **F-115 / §5**", DF100, "para"),
    (229, "1. **Report L36, not L40, as the optimum**", """> **UPDATE 2026-09-24 (DF-107, P17).** Items 1 and 3 below were done by the collaborator at
> 2ad758fb: the paper reports the L36 peak (`results.tex:228-230`; numbers.tex `depth-off-ext-peak`=0.927
> at layer 36) and replaced "outscores … at every depth" with a count over the sweep (`results.tex:136`).
> **Item 2 is still open and has worsened:** `limitations.tex:17-21` quotes a ≥0.026 margin on the
> other datasets that includes HP-KR (D2-019, DF-97). Item 6 is still open @ 16def2de:
> `persona_depth/summary.md:13-14,78` still cites `../persona_unified/` and `../persona_depth_orig/`,
> and neither is on the branch.
> **Source:** `paper/sections/{results,limitations}.tex`, `paper/generated/numbers.tex`,
> `results/persona_depth/summary.md` @ 16def2de; `audit/p17/paper_new_claims.csv` D2-019.""", "para"),
]

E["audit/COLLABORATOR_FINDINGS.md"] = [
    (6, "**Why this file exists.**", """> **UPDATE 2026-09-24 (P17).** His branch moved to **16def2de** after this was assembled. New work:
> the depth sweep is now in the paper (16 layers), with an app:depth section and unified-axis (13v12)
> arms. His own 16def2de commit corrected the HP-KR ranking claim to "8 of 12". Two sentences still
> do not hold: `introduction.tex:68-70` and `limitations.tex:17-21` (DF-96, DF-97). Branch ownership
> (owner statement): `origin/paper` and `persona-pc-ood` are zami456's; Mahdi has one commit,
> df758c4b, on `feat/llama-layer30`.
> **Source:** `audit/p17/A_REPORT.md`; `audit/p17/paper_new_claims.csv`; `thesis_tools/survey/BRANCHES.md`;
> `audit/BRANCH_PINS.md`.""", "para"),
    (67, "**Credit him with these: our pipeline has no sycophancy axis at all.**", DF109, "para"),
    (112, "- **Two pipelines, one answer.**", COV, "para"),
    (124, "`ext_axis1 − ext_pc3`, **excluding HP-KR**", PM, "para"),
    (135, "A **label-free regex** scores", DF105, "para"),
    (168, "---", NEW_DOES_NOT_HOLD.replace(
        "Two sentences added to the paper after this chapter was drafted do\n> not hold at the pin 16def2de:",
        "New C-items. Two sentences added to his paper after this file was\n> assembled do not hold at the pin 16def2de:"), "para"),
    (185, "| route | who | result |", DF100, "para"),
]

E["audit/COVERAGE_GAP.md"] = [
    (9, "**CORRECTION 2026-09-24, after Phase 6A", COV, "para"),
    (127, "`audit/BRIEF.md:5` fixed scope", """> **UPDATE 2026-09-24 (DF-106, P17).** The same cause recurred inside Phase 6: origin/paper moved
> ea1ce52b → 2ad758fb (15:44) → 16def2de between the Level-2 run (15:39) and the Level-1 verdicts
> (16:40). Fixed for the future by `audit/BRANCH_PINS.md` and `audit/scripts/p17_common.py` (every
> read now uses a pinned SHA).
> **Source:** `audit/BRANCH_PINS.md` (reflog); `audit/p17/A_REPORT.md` §1.""", "para"),
]

E["audit/BRIEF_AS_EXECUTED.md"] = [
    (16, "**CORRECTION 2026-09-24 (DF-94, Phase 6E).**", COV, "para"),
    (160, "3. **One dataset carries three headlines", DF105, "para"),
    (322, "The brief limited the audit to `asix`", """> **UPDATE 2026-09-24 (P17).** origin/paper's claims now carry `claims.csv` ids: 176 D-rows
> re-pinned to 16def2de, and D2-001…026 for its three newest commits.
> **Source:** `audit/claims.csv`; `audit/p17/d_rows_repinned.csv`; `audit/p17/paper_new_claims.csv`.""", "para"),
    (329, "- **The audit's D-028 finding generalises", PM, "para"),
    (338, "- **The optimum layer is 36, not the reported 40**", DF107, "para"),
]

E["audit/phase6/RESULTS.md"] = [
    (5, "## Status", """> **UPDATE 2026-09-24 (P17).** At the pin origin/paper @ 16def2de, 6A reads **165 MATCHES / 9 PARTLY
> / 2 UNRESOLVED**; D-189 (B46, Table 7) → PARTLY, so Table 7 is 5 M · 1 P. Coverage after P17 is
> **533/735 = 72.5%**, not the 490/692 at :18.
> **Source:** `audit/claims.csv` (D-189); `audit/p17/A_REPORT.md` §1; `python audit/scripts/p15_coverage.py`.""", "para"),
    (32, "**Every stated number that exists in a committed result file", """> **UPDATE 2026-09-24 (DF-106, P17).** Table 7 is 35/36 at 16def2de (D-195): the collaborator's
> baseline re-run moved CG raw L40 from 0.7875 to 0.7873, and the paper now prints 0.787.
> **Source:** `audit/p17/d_rows_other_affected.csv` (D-195).""", "para"),
    (54, "**Sources.** `origin/paper` @ `ea1ce52b`", DF106_PROV, "para"),
    (64, "**Only two claims are unverifiable", DF106_MACRO, "para"),
    (136, "1. **The dim=1 framing**", DF106_MACRO.replace('Read "1,161', 'In item 3 below, read "1,161'), "para"),
]

DOC_HEADER = """> **UPDATE 2026-09-24 (P17).** "Nothing here has been applied" is no longer true: the urgent set was
> applied in Phase 6 as **11 correction blocks in 10 files** (`audit/phase6/APPLY_SHEET.md` §1). P17
> checked them: 86 of 94 numbers match; see DF-103…DF-105. DF-12 is withdrawn (F-123). **P17 adds
> DF-96…DF-109** (§C-septies, at the end of this file). **Next free id: DF-110.**
> **Source:** `audit/phase6/APPLY_SHEET.md` §1; `audit/p17/correction_blocks_check.csv`."""

PM_BULLET = ("- **CORRECTION 2026-09-24 (DF-101, P17):** \"within ±0.06\" / \"crossing zero five times\" below is "
             "wrong: excluding HP-KR the margin reaches **+0.0633 at L56** and changes sign **6** times (still "
             "11/16 positive). **Source:** `persona_ood_ext_L{20..79}/summary.csv` `off_diagonal_excl_hpkr` @ "
             "16def2de; `audit/p17/ledger_verify_stdout.txt:223-224`.")

E["audit/DOC_FIXES.md"] = [
    (3, "**Status: accumulating", DOC_HEADER, "para"),
    (859, "", "- **Status 2026-09-24 (DF-107, P17):** done in origin/paper@2ad758fb — the paper reports the "
                "L36 peak (`results.tex:228-230`; numbers.tex `depth-off-ext-peak`=0.927 at layer 36 @ 16def2de).",
     "bullet"),
    (863, "- **Fix:** \"at 13 of 16 layers\"", PM_BULLET, "bullet"),
    (864, "", "- **Status 2026-09-24 (DF-107, P17):** done in origin/paper@2ad758fb — `results.tex:136` now "
                "counts wins over the sweep (`depth-pc-wins-complement`=13 of `depth-n`=16) @ 16def2de.", "bullet"),
    (866, "- Without HP-KR the margin is", PM_BULLET, "bullet"),
    (869, "", "- **Status 2026-09-24 (P17):** still open at 16def2de, and worsened — `limitations.tex:17-21` "
                "quotes a ≥0.026 margin that includes HP-KR (D2-019). See DF-97. "
                "Source: `audit/p17/paper_new_claims.csv` D2-019.", "bullet"),
    (933, "", "- **Status 2026-09-24 (DF-109, P17):** the \"real gap\" is closed — 6D-1 trained `pc:sycophancy` "
                "in our pipeline; it loses to raw at all six cells on 11 datasets (F-126 §1, HOLDS 41/41 in "
                "`audit/p17/C_REPORT.md`).", "bullet"),
    (942, "", "- **Status 2026-09-24 (P17):** superseded by 6E (490/692) and P17 (533/735, draft 176/176 at "
                "165/9/2). Source: `python audit/scripts/p15_coverage.py`.", "bullet"),
    (949, "", "- **Status 2026-09-24 (P17):** made concrete by P17 — the Phase-6 stamp ea1ce52b is wrong for "
                "Level 1 (DF-106). Source: `audit/p17/A_REPORT.md` §1; `audit/BRANCH_PINS.md`.", "bullet"),
]

E["CLAUDE.local.md"] = [
    (3, "**Phases 0–5 are complete**", """> **UPDATE 2026-09-24 (P17).** Phase 6 has run: 6A (all 176 draft claims), 6D, 6E. 6B and 6C were
> skipped by instruction. The "NOT started" / "Do not start Phase 6" lines below are historical.
> P17 then re-pinned everything:
> - **origin/paper @ 16def2de; asix @ e917ecd3.** See `audit/BRANCH_PINS.md`. Read other branches
>   only at those SHAs.
> - **Coverage:** 533/735 = 72.5%. The 202 still unverdicted are 110 ledger and 92 collaborator notes.
> - **Ledger:** 128 entries; F-128 (the P17 re-pin) is appended.
> Start at `audit/p17/P17_FACTS.md`, then the P17 section at the end of `audit/PROGRESS.md`.
> **Source:** `audit/phase6/RESULTS.md`; `audit/BRANCH_PINS.md`; `python audit/scripts/p15_coverage.py`;
> `kd/docs/FINDINGS.md` F-128.""", "para"),
    (11, "0. **`audit/COVERAGE_GAP.md`", "> **UPDATE 2026-09-24 (P17).** Coverage is now **533/735 = 72.5%** "
     "(the draft PDF at 176/176: 165/9/2), not 314/692 = 45%. Source: `python audit/scripts/p15_coverage.py`.", "para"),
    (31, "- **`kd/docs/FINDINGS.md` holds 122 F-entries.**", """> **UPDATE 2026-09-24 (P17).** The ledger holds **128** entries (F-01…F-127 at the pin, md5 dc365834;
> F-128 appended by P17). The reports also lack F-123…F-127, F-128 and the collaborator's three newest
> commits. **F-111, F-112, F-117, F-118, F-120 and F-127 hold only in part (P17). The corrections are
> in F-128; quote those entries only through it.** Source: `audit/BRANCH_PINS.md`; `audit/p17/C_REPORT.md`.""", "para"),
    (33, "- **`audit/DOC_FIXES.md` holds 95 proposed corrections", "> **UPDATE 2026-09-24 (P17).** 11 correction blocks "
     "were applied in Phase 6 (`audit/phase6/APPLY_SHEET.md` §1); P17 proposes DF-96…DF-109 (next free DF-110).", "para"),
    (35, "- **Coverage is 45%, not 61%**", "> **UPDATE 2026-09-24 (P17).** Coverage is 533/735 = 72.5%, with the draft "
     "at 176/176 (165/9/2). Source: `python audit/scripts/p15_coverage.py`; `audit/claims.csv`.", "para"),
    (39, "- **The owner committed at `e917ecd3`", "> **UPDATE 2026-09-24 (P17).** At the pin, `git status` outside "
     "`audit/` is 11 modified files plus the untracked `thesis_tools/` (`audit/BRANCH_PINS.md`).", "para"),
    (41, "- **The Results chapter does not exist.**", "> **UPDATE 2026-09-24 (P17).** `audit/RESULTS_CHAPTER.md` "
     "now exists (Phase 6A; `audit/phase6/RESULTS.md`).", "para"),
    (46, "1. **The convergence", "> **CORRECTION 2026-09-24 (DF-100, P17).** The values 0.4630 / 0.6203 / 0.5383 "
     "are at `audit/HPKR_WORDING_CONTROLLED.md` §5 (:121), not in F-115. The convergence finding is unchanged. "
     "Source: `audit/p17/C_REPORT.md` (F-118 row).", "para"),
]


def doc_fixes_tail() -> str:
    stale = (ROOT / "audit" / "p17" / "STALE_LIST.md").read_text(encoding="utf-8").split("\n")
    # §11 new entries: from '### DF-96' up to (not including) '## 12.'
    start = next(i for i, l in enumerate(stale) if l.startswith("### DF-96 "))
    end = next(i for i, l in enumerate(stale) if l.startswith("## 12."))
    body = stale[start:end]
    while body and body[-1].strip() == "":
        body.pop()
    head = [
        "",
        "---",
        "",
        "## C-septies. From P17 (2026-09-24)",
        "",
        "_Copied as drafted from `audit/p17/STALE_LIST.md` §11. There, \"the draft below\" (DF-100) is F-128,",
        "now appended to `kd/docs/FINDINGS.md`; \"the header of STALE_LIST\" (DF-101) and \"STALE_LIST §12\"",
        "(DF-108) refer to `audit/p17/STALE_LIST.md`. Next free id: DF-110._",
        "",
    ]
    return "\n".join(head + body) + "\n"


def insert_file(rel: str, edits: list, tail: str | None, log: list) -> None:
    path = ROOT / rel
    snap = SNAP / rel.replace("/", "_")
    raw = path.read_bytes()
    if raw != snap.read_bytes():
        sys.exit(f"ABORT {rel}: differs from snapshot {snap.name} (already edited?)")
    text = raw.decode("utf-8")
    nl = "\r\n" if text.count("\r\n") == text.count("\n") else "\n"
    assert nl == "\r\n" or "\r" not in text, f"{rel}: mixed line endings"
    lines = text.split(nl)
    trailing = lines[-1] == ""
    if trailing:
        lines.pop()
    for n, anchor, block, style in sorted(edits, key=lambda e: -e[0]):
        tgt = lines[n - 1]
        if anchor == "":
            assert tgt.strip() == "", f"{rel}:{n} expected blank, got {tgt[:60]!r}"
        else:
            assert anchor in tgt, f"{rel}:{n} anchor {anchor!r} not in {tgt[:80]!r}"
        blk = block.split("\n")
        if style == "para":
            new = ([] if lines[n - 2].strip() == "" else [""]) + blk + [""]
        else:
            new = blk
        lines[n - 1:n - 1] = new
        cites = sorted({c for c in ["audit/p17/", "audit/BRANCH_PINS.md", "FINDINGS.md", "claims.csv",
                                    "p15_coverage.py", "APPLY_SHEET", "summary.csv", "@ 16def2de",
                                    "numbers.tex", "git log", "results.tex", "limitations.tex",
                                    "paper_new_claims.csv", "correction_blocks_check.csv", "d_rows_",
                                    "A_REPORT", "C_REPORT", "phase6/RESULTS.md", "HPKR_WORDING_CONTROLLED",
                                    "fmap_F110_F127", "ledger_verify_stdout", "csv_drift"] if c in block})
        log.append((rel, n, block.split("\n")[0][:80], ", ".join(cites)))
    if tail:
        tl = tail.split("\n")
        if tl and tl[-1] == "":
            tl.pop()
        lines.extend(tl)
        log.append((rel, "END", tl[0][:80] if tl[0] else tl[1][:80], "appended section"))
    out = nl.join(lines) + (nl if trailing else "")
    dry = os.environ.get("P17_DRY")
    dest = Path(dry) / rel.replace("/", "_") if dry else path
    dest.write_bytes(out.encode("utf-8"))


def main() -> None:
    log: list = []
    tails = {"audit/PROGRESS.md": PROGRESS_END, "audit/DOC_FIXES.md": doc_fixes_tail()}
    for rel, edits in E.items():
        insert_file(rel, edits, tails.get(rel), log)
    rows = ["# P17 Agent E log: insert-only update/correction blocks (2026-09-24)", "",
            "Written by `audit/scripts/p17_e_insert.py`. Line numbers are the ORIGINAL (pre_wave3) numbering.", "",
            "| file | inserted before (orig line) | block starts | evidence named in block |",
            "|---|---|---|---|"]
    for rel, n, first, cites in log:
        esc = first.replace("|", "\\|")
        rows.append(f"| `{rel}` | {n} | {esc} | {cites} |")
    logp = Path(os.environ["P17_DRY"]) / "E_LOG.md" if os.environ.get("P17_DRY") else LOG
    logp.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"{len(log)} insertions")


if __name__ == "__main__":
    main()
