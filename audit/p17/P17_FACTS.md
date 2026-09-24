# P17 facts sheet: the only source that update blocks may cite (2026-09-24)

Every item below was verified by P17 Wave 1 and spot-checked by the lead. Each entry gives its
evidence file. An update block must cite the evidence file (and the SHA where there is one), never
this sheet alone.

## Pins (audit/BRANCH_PINS.md)
- Pins taken after a `git fetch` on 2026-09-24. The fetch moved no ref.
- zami456's side:
  - `origin/paper` is pinned at **16def2de** (2026-09-24 16:47). The previous audit mark was ea1ce52b.
  - `origin/feat/persona-pc-ood` is pinned at 45b16d90. It is an ancestor of paper and has nothing newer.
- Owner's side: `asix` is pinned at **e917ecd3**, plus the uncommitted working tree. The md5s are in BRANCH_PINS.md; `kd/docs/FINDINGS.md` has md5 dc365834 and holds F-01…F-127.
- Branch owners: `asix` is the owner's. `origin/paper` and `persona-pc-ood` are zami456's. Mahdi is the third co-author; he has 1 commit, df758c4b, on feat/llama-layer30. (Source: owner statement 2026-09-24, recorded in thesis_tools/survey/BRANCHES.md.)

## zami456: the three commits after ea1ce52b (audit/p17/A_REPORT.md)
- **2ad758fb (15:34): the depth sweep.**
  - The paper now uses 16 layers, 20..79 every 4, instead of 3.
  - Four claims are corrected.
  - A Depth subsection and app:depth are added.
  - 12 new result CSVs: the unified axis (13v12) arms.
  - Baseline rows are re-run (BLAS drift). 80 cells changed, max |Δ| 0.0054, all raw-probe baseline (audit/p17/csv_drift.csv).
- **d1b88154 (16:43): framing against the trained probe, using the whole depth curve.**
- **16def2de (16:47): the abstract and limitations now say "8 of 12" deep layers.**
- d1b88154 and 16def2de change only paper text and numbers.tex; no result file changes after 2ad758fb.
- 83/83 recomputable numbers.tex keys match the pinned CSVs. persona_depth/depth.csv agrees with the per-layer files in every cell.
- 26 new claim rows D2-001..D2-026: 19 HOLDS, 3 PARTLY HOLDS, 2 DOES NOT HOLD, 2 UNRESOLVED (audit/p17/paper_new_claims.csv).
  - **DOES NOT HOLD, D2-005, `paper/sections/introduction.tex:68-70` @16def2de.** The sentence says that from layer 36 the base probe ranks HP-KR better than any contrast. The base probe actually leads at 8 of 12 layers and trails at:

    | Layer | Base probe | Best contrast |
    |---|---|---|
    | L44 | 0.828 | 0.908 |
    | L48 | 0.837 | 0.890 |
    | L52 | 0.877 | 0.883 |
    | L76 | 0.838 | 0.925 |

    The lead re-derived this independently from `results/persona_depth/depth.csv` @16def2de.
  - **DOES NOT HOLD, D2-019, `limitations.tex:17-21`.** The claim is "≥0.026 on the other datasets at all 16 layers". The 0.026 includes HP-KR. With HP-KR excluded, the margin is −0.011 at L20 and below 0.026 at 8 of 16 layers.
  - **PARTLY HOLDS:**
    - D2-016 (`results.tex:254-255`): the claim is unqualified.
    - D2-018 (`results.tex:261-263`): the published 8v7 contrast leads per dataset at L76 HP-C and L79 IT/RepE.
    - D2-002 (`abstract.tex:18-19`): "ever reaches anywhere" holds for means only.
  - **UNRESOLVED:**
    - D2-024: the app:depth pooling constants are hard-coded.
    - D2-025: `results_table.csv` is not committed.
- **Provenance conflict (DF-95 made concrete).**
  - The Phase-6 Level-2 rows read ea1ce52b.
  - The Level-1 rows, written at 16:40, read 2ad758fb at least in part: D-171 and D-032.
  - 172 of the 176 stamped values are identical at all three SHAs, so the stamp is immaterial for them.
  - **One verdict changed.** D-189 went from MATCHES to PARTLY MATCHES HIS DATA: CG raw L40 is 0.7875 at ea1ce52b and 0.7873 at the pin, and the paper now prints 0.787.
  - Stored-evidence errors: D-001 values reproduce at no SHA (its verdict stands); D-032 says "eight tab_*.tex" where there are 11.
  - Other rows: D-195 now reproduces 35/36, not 36/36. D-009 and D-027 drift in the 4th decimal.

## Owner: F-110…F-127 (audit/p17/C_REPORT.md, audit/p17/fmap_F110_F127.csv)
- 447 numbers were re-derived from raw, including lab-PC files read with BatchMode ssh (read-only, md5-verified). 432 match. No headline measurement fails.
- 12 entries HOLD: F-110, F-113, F-114, F-115, F-116, F-119, F-121, F-122, F-123, F-124, F-125, F-126.
- 6 are PARTLY:

  | Entry | What is wrong |
  |---|---|
  | F-111 | the "trap" (0/80) does not reproduce: 31/80; 0.3061/0.2996 are both passA11 |
  | F-112 | "8 clear BH" is scoped to block 10 / mean / paper probe; the family-wide count is 34 |
  | F-117 | the L56 excl-HP-KR margin is +0.0633, so "within ±0.06" is false; 6 zero crossings, not 5 (lead re-derived) |
  | F-118 | the pointer "F-115 §5" is wrong; the numbers are in audit/HPKR_WORDING_CONTROLLED.md §5 |
  | F-120 | the numbers are the 5-dataset reading, which the entry does not say; all-10 raw L40 is 0.6760 |
  | F-127 | tab_*.tex are 11, not 8; numbers.tex has 1,162 lines only at 2ad758fb |

- LAB-ONLY evidence (not in git): F-110, F-113 (part), F-114, F-115, F-116, F-120, F-121, F-126.
- **Correction blocks in the 10 working-tree docs:** 94 numbers, 86 match.
  - 5 are stale line pointers.
  - review_page.html labels 0.3061/0.2996 as passA11/passC11; both are passA11's.
  - 2 are unverified: 0.9597, and "median gap 0.012 over 17 methods".
  - Evidence: audit/p17/correction_blocks_check.csv.

## Audit tables after the merge (audit/scripts/p17_merge.py; backups *.before-p17)
| Table | Before | After | What changed |
|---|---|---|---|
| claims.csv | 600 rows | **643** | +26 D2, +17 L2. The 176 ea1ce52b rows are re-pinned with a `[P17 2026-09-24]` note. 1 verdict revised, with the old one kept. 12 other D- rows annotated. |
| claims_inventory.csv | 692 | **735** | |
| fmap_all.csv | F-01..F-109 | **F-01..F-127** (127 rows) | |
| experiments.csv | 114 | **119** | C21 depth sweep, C22 unified-axis arms, C23 gender-secret NDIF, C24 paper build, O88 kd scale ladder |

**Coverage** (`python audit/scripts/p15_coverage.py`): real coverage is **533/735 = 72.5%**, up from 490/692 = 70.8% before P17. That is inventory claims with a verdict out of all inventory claims.
- 202 inventory claims still have no verdict: 110 FINDINGS.md (TODO) and 92 collaborator notes (CANDIDATE).
- The "45%" in CLAUDE.local.md and COVERAGE_GAP.md predates Phase 6.

## Stale statements known before the scan
- CLAUDE.local.md says the ledger has "122 F-entries" (it has 127) and that "Phase 6 … NOT started" (it has run: phase6/RESULTS.md, F-123…F-127). It gives coverage as 45%.
- RESULTS_CHAPTER.md:11 and phase6/RESULTS.md:54 pin `origin/paper @ ea1ce52b`.
- COLLAB_PAPER_BRANCH.md's commit table (lines 14-18) ends at ea1ce52b.
- The DOC_FIXES.md header says "Nothing here has been applied", but correction blocks were applied (phase6/APPLY_SHEET.md). The highest DF id is DF-95.
- docs/DRAFT_CORRECTIONS.md:110 says there is no paper/ dir. It exists on origin/paper.
- kd/docs/REPORT.md says "Phase 6 … not started" (its own kd phases, a different Phase 6 — check before touching).
- docs/TODO.md, OPEN_DECISIONS.md and ONBOARDING.md are dated 09-17…09-21.
