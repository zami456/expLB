# P17 Agent D: stale statements in the tracker documents, with proposed update blocks

_2026-09-24. Read-only scan. The only file written is this one. Sources for "new truth":
`audit/p17/P17_FACTS.md` and the evidence files it cites. Pins: `origin/paper` @ **16def2de**, `asix` @
**e917ecd3** + working tree (`audit/BRANCH_PINS.md`). Block format: `audit/phase6/APPLY_SHEET.md:11-19`.
Every block goes **immediately before** the stale text. The original stays in place under it._

**One re-derivation was made here.** The excl-HP-KR `ext_axis1 − ext_pc3` series was recomputed from
`persona_ood_ext_L{20..79}/summary.csv` @ 16def2de, column `off_diagonal_excl_hpkr`:
+0.0086, +0.0012, −0.0030, +0.0028, +0.0155, +0.0495, +0.0354, −0.0081, +0.0295, **+0.0633 (L56)**,
+0.0495, −0.0029, −0.0325, −0.0198, +0.0121, +0.0039 (L20…L79).
That is 11/16 positive, range −0.0325…+0.0633, and **6 sign changes**. This agrees with
`audit/p17/ledger_verify_stdout.txt:223-224`.

## 0. Reusable blocks (referenced below by name)

**BLOCK-PM** (the "±0.06 / five crossings" error, F-117, propagated):
```
> **CORRECTION 2026-09-24 (DF-101, P17).** "Within ±0.06 at every layer" and "crosses zero five
> times" are both wrong. Excluding HP-KR, `ext_axis1 − ext_pc3` reaches **+0.0633 at L56**, and the
> series changes sign **6** times. Still correct: 11/16 positive, +0.0086 (L20), +0.0012 (L24), and
> negative at L64/68/72. No persona-arm cell changed between ea1ce52b and 16def2de.
> **Source:** `src/probes/deception-detection/results/persona_ood_ext_L{20..79}/summary.csv`,
> `off_diagonal_excl_hpkr` @ 16def2de; `audit/p17/ledger_verify_stdout.txt:223-224`.
> **How the claim is updated:** *"excluding HP-KR the margin is positive at 11 of 16 layers, lies
> between −0.033 and +0.063, and changes sign six times."*
> _The text below is superseded on this point and is kept for the record._
```

**BLOCK-COV** (coverage and the draft verdicts at the pin):
```
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
```

**BLOCK-F120** (F-120's numbers carry no scope label):
```
> **CORRECTION 2026-09-24 (DF-102, P17).** The raw/axes/random values below (raw 0.8195 at L40 →
> 0.7426 at L79) are the **5-dataset (paper5) reading** of `passA`, and the text does not say so.
> On all 10 datasets, raw at L40 is **0.6760**. The shape of the argument (raw declines, random
> gains) is unaffected. **Source:** `audit/p17/C_REPORT.md` F-120 row; `audit/p17/fmap_F110_F127.csv`
> (LAB-ONLY: `passA` matrices). _Kept for the record below._
```

---

## 1. audit/PROGRESS.md

- `:3` — "Current phase: **PHASE 4 RUNNING (started 2026-09-23 ~17:55)**…" **Why stale:** Phases 0–6 are finished and P17 has run. Evidence: `audit/phase6/RESULTS.md:15-20`; `audit/BRANCH_PINS.md`.
  - **Insert before :3:** `> **UPDATE 2026-09-24 (P17).** This header is historical. Phases 0–5, Phase 6 (6A, 6D, 6E; 6B/6C skipped by instruction) and the P17 re-pin are complete, and nothing is running. The current state is the P17 section at the end of this file, together with audit/BRANCH_PINS.md.`
- `:2686` — "All are in the ledger (now 121 entries)" **Why stale:** the ledger holds F-01…F-127 (md5 dc365834). Evidence: `audit/BRANCH_PINS.md`.
- `:2683` — "424 of 692 claims checked, 297 hold / 119 partly / 8 fail" **Why stale:** superseded twice (6E, then P17). → **BLOCK-COV**.
- `:2711`, `:2727` — "`DOC_FIXES.md` (85 entries, **none applied**)" **Why stale:** DF-01…DF-95 exist. 11 correction blocks were applied in 10 files (`audit/phase6/APPLY_SHEET.md` §1). P17 checked 94 of their numbers: 86 match, 5 are stale pointers, 1 is a partial mislabel, 2 are unverified (`audit/p17/correction_blocks_check.csv`).
- `:2718` — "**The Results chapter does not exist**" **Why stale:** `audit/RESULTS_CHAPTER.md` exists (Phase 6A).
- `:2724` / `:2734` — "git status outside audit/ … 18 entries" / "**1 entry**" **Why stale:** at the pin there are 11 modified files plus the untracked `thesis_tools/`. The md5 list is in `audit/BRANCH_PINS.md:28-42`.
  - **One block covers the four items above. Insert it before :2675:**
    ```
    > **UPDATE 2026-09-24 (P17).** This handoff is superseded on its counts.
    > - **Ledger:** 127 entries, F-01…F-127 (md5 dc365834…, audit/BRANCH_PINS.md). F-123…F-127 are Phase 6.
    > - **DOC_FIXES:** DF-01…DF-95, with DF-96… proposed by P17. 11 correction blocks were applied
    >   (phase6/APPLY_SHEET.md §1).
    > - **Results chapter:** now exists (audit/RESULTS_CHAPTER.md).
    > - **git status outside audit/:** 11 modified files plus thesis_tools/.
    > - **Pins:** origin/paper @ 16def2de (was ea1ce52b); asix @ e917ecd3.
    > - **Coverage:** see BLOCK-COV.
    > Read the P17 section at the end of this file first.
    ```
- `:2580-2581` — "…within ±0.06 everywhere, crossing zero five times" → **BLOCK-PM**, inserted before `:2579`.
- `:2702` — "F-115 (his axes on our matched slice, 0.4630/0.6203/0.5383)" **Why stale:** these values are not in F-115. They are at `audit/HPKR_WORDING_CONTROLLED.md:121` (§5) and reproduce from the collab-bank kd11 matrices (`audit/p17/C_REPORT.md`, F-118 row).
  - **Block:** `> **CORRECTION 2026-09-24 (DF-100, P17).** The three values are sourced wrongly: they are at audit/HPKR_WORDING_CONTROLLED.md §5 (:121), not F-115. The convergence finding is unchanged.`
- **End of file** — append a section `# P17 — RE-PIN (2026-09-24)`. It should summarise: the pins; the provenance conflict; the D-189 revision; D2-005/D2-019 DOES NOT HOLD; the six PARTLY ledger entries; F-128; coverage 533/735.

  The draft F-128 below can serve as its body.

## 2. audit/REPORT_PLAIN.md
- `:5-24` (the existing top block) — "`F-117` … within ±0.06 at every depth and crosses zero five times"; "160 MATCHES / 9 PARTLY / 7 UNRESOLVED"; "71%". **Why stale:** F-117 is PARTLY (L56 +0.0633; six sign changes). The draft totals are 165/9/2 (F-127, then P17 D-189). Coverage is 533/735.
  - **Insert before :5:** **BLOCK-COV**, then **BLOCK-PM**.
  - **Also add to the same block:** *"F-123…F-127 and the collaborator's commits 2ad758fb/d1b88154/16def2de are also absent from this report."*
- `:317-318` — "within ±0.06 at all 16 layers and crosses zero five times" → **BLOCK-PM**, inserted before `:317`.
- `:342-349` — "declines there (0.7834 → 0.7426) … random gains +0.127" → **BLOCK-F120**, inserted before `:342`.
- `:335` — "his axes on our wording-matched slice (F-115) | 0.4630…" → the DF-100 pointer block (see §1, `:2702`), inserted before the table at `:333`.
- `:354` — "the optimum layer is 36, not 40 … the draft under-reports its own best" **Why stale:** in his favour. At 2ad758fb the paper reports the peak itself: `results.tex:228-230` says "Ranking peaks at layer~\V{depth-off-ext-peak-layer}, not at layer~40". numbers.tex has `depth-off-ext-peak-layer`=36 and `depth-off-ext-peak`=0.927 (@16def2de).
  - **Block:** `> **UPDATE 2026-09-24 (DF-107, P17).** Resolved by the collaborator at 2ad758fb: the paper now reports the L36 optimum (0.927). Source: results.tex:228-230 and numbers.tex @ 16def2de.`
- `:200` — regex "**0.9597**" → **DF-105** block (§11).
- `:60` "424 claims … 297/119/8" is already covered by the top block. No new item.

## 3. audit/REPORT_TECHNICAL.md
- `:5-23` top block — "160 MATCHES / 9 PARTLY / 7 UNRESOLVED", "71%", "F-117, F-119 and F-122 are absent".
  - **Insert before :5:** **BLOCK-COV**.
  - **Plus:** `> **UPDATE 2026-09-24 (P17).** Also absent from this report: F-123…F-127, and origin/paper commits 2ad758fb (depth sweep in the paper; 4 claims corrected; baseline re-run, 80 cells, max |Δ| 0.0054), d1b88154 and 16def2de. See audit/p17/A_REPORT.md.`
- `:379-380` — "New branch, 3 commits, 683 files" **Why stale:** there are 6 commits past persona-pc-ood. The tip is 16def2de (`git log ea1ce52b..16def2de`: 3 more).
  - **Block:** `> **UPDATE 2026-09-24 (P17).** origin/paper has moved to 16def2de (3 more commits, 2026-09-24 15:34–16:47). See audit/COLLAB_PAPER_BRANCH.md and audit/BRANCH_PINS.md.`
- `:389` — "within ±0.06, 11/16" → **BLOCK-PM**, inserted before the table at `:386`.
- `:391-392` — "'At every depth' is falsified … Optimum is L36, not the reported L40" **Why stale:** both were fixed in the paper at 2ad758fb. At 16def2de, "at every depth" no longer appears in `introduction.tex` or `results.tex`: `results.tex:136` now says "at \V{depth-pc-wins-complement} of the \V{depth-n} we swept", and `results.tex:228-230` reports the L36 peak. → **DF-107** block, inserted before `:391`.
- `:393-396` — "raw declines (0.7834 → 0.7426) … random gains +0.127" → **BLOCK-F120**, inserted before `:393`.
- `:281` — "0.9125 / **0.9597**" → **DF-105** block.

## 4. audit/RESULTS_CHAPTER.md
- `:11` — "`origin/paper` @ `ea1ce52b`" **Why stale:** the 54 Level-2 rows read ea1ce52b, but the Level-1 verdicts (written 16:40) read 2ad758fb at least in part (D-171, D-032). 172 of the 176 values are identical at all three SHAs. Evidence: `audit/p17/A_REPORT.md` §1; `audit/BRANCH_PINS.md` reflog.
  - **Insert before :11:**
    ```
    > **CORRECTION 2026-09-24 (DF-106, P17).** The single pin below is wrong for part of the pass.
    > - **Level 2** (p16_level2, 15:39) read ea1ce52b.
    > - **Level 1** (verdicts written 16:40, after the ref moved at 15:44) read 2ad758fb for at least
    >   D-171 and D-032.
    > 172 of the 176 stored values are identical at ea1ce52b, 2ad758fb and 16def2de, so the stamp
    > does not matter for them. The audit's current pin is **16def2de**. At that pin **one verdict
    > changes: D-189 → PARTLY** (Table 7, CG raw L40 = 0.7873 → 0.787). Status now reads 165 / 9 / 2.
    > **Source:** audit/p17/A_REPORT.md §1; audit/p17/d_rows_repinned.csv; audit/claims.csv D-189 @ 16def2de.
    > _Kept for the record below._
    ```
- `:19-26` status table "166 / 8 / 2" → covered by the block above (165/9/2 at the pin).
- `:230` — "**Table 7** … All 36 values reproduce exactly", with `:234` "0.788 / 0.912" **Why stale:** true at ea1ce52b. At the pin the value is 35/36: CG raw L40 is 0.7873, and the paper itself now prints `pt-base-cg-L40`=0.787. Evidence: `audit/p17/d_rows_other_affected.csv` (D-195), `numbers.tex` @ 16def2de.
  - **Block:** `> **UPDATE 2026-09-24 (DF-106, P17).** At 16def2de the collaborator re-ran the baseline: CG raw L40 = 0.7873, and the paper now prints 0.787. So 35 of 36 values match the draft PDF, and the draft's 0.788 is the ea1ce52b value. No persona column changed.`
- `:437-440` — "**1,162 `\setV` macros** … eight `tab_*.tex`" **Why stale:** the file has 1,162 lines only at 2ad758fb (1,161 macros plus a header). At 16def2de it is 1,177 lines. There are **11** `tab_*.tex` at every SHA. Evidence: `audit/p17/C_REPORT.md` F-127 row; `git ls-tree 16def2de …/paper/generated/`.
  - **Block:** `> **CORRECTION 2026-09-24 (DF-106, P17).** Read "1,161 macros at 2ad758fb (1,176 at 16def2de), beside eleven tab_*.tex". The conclusion (the release is well evidenced) stands.`
- `:524-527` — "The paper should now report [L36] rather than leave it hypothetical" → **DF-107** block (resolved at 2ad758fb).
- `:533-535` — "excluding-HP-KR margin crossing zero five times across 16 layers" → **BLOCK-PM**.
- Optionally add a §9 item after `:420` for the two new DOES NOT HOLD claims in the current paper (D2-005, D2-019). Text as in DF-96/DF-97.

## 5. audit/RESULTS_BY_AXIS.md
- `:119` — "sycophancy (11) | ❌ **never** — absent from the default `--features`" **Why stale:** F-126 §1 (6D-1) trained `pc:sycophancy` in our pipeline (passB, paper16, 11 datasets). It loses to raw at all 6 (layer, pooling) cells, from −0.008 to −0.118. F-126 is HOLDS 41/41 in `audit/p17/C_REPORT.md`.
  - **Block (DF-109)**, inserted before `:116`: `> **UPDATE 2026-09-24 (DF-109, P17).** Our-pipeline cell is now ✅ 70B only (F-126 §1). It loses to raw everywhere on 11 datasets and turns positive only without sycophancy_mt, under mean pooling at L40/L44.`
- `:135-136` — the baseline column "0.7581" (L40) and "0.7297" (L44) **Why stale:** at 16def2de these read 0.7580 and 0.7296, after the BLAS re-run. The "vs baseline" column moves in the 4th decimal only. Evidence: `audit/p17/C_REPORT.md` F-122 row.
  - **P3:** add one line, "values @ ea1ce52b; baseline −0.0001 at the pin".

## 6. audit/COLLAB_PAPER_BRANCH.md
- `:3` — "three commits ahead of `origin/feat/persona-pc-ood`" and `:14-18`, the commit table ending at ea1ce52b. **Why stale:** the tip is 16def2de. Evidence: `git log ea1ce52b..16def2de`; `audit/p17/A_REPORT.md`.
  - **Insert before :14:**
    ```
    > **UPDATE 2026-09-24 (P17).** Three more commits, audited at the pin 16def2de:
    > | `2ad758fb` | 09-24 15:34 | depth sweep written into the paper (16 layers), app:depth, 4 claims corrected, 12 new `matrix_*uni*` CSVs, baseline re-run (80 cells, max |Δ| 0.0054) |
    > | `d1b88154` | 09-24 16:43 | the case against the trained probe, from the whole depth curve (text + numbers.tex only) |
    > | `16def2de` | 09-24 16:47 | abstract/limitations: HP-KR ranking "8 of 12" deep layers (text + numbers.tex only) |
    > Of the 26 new claims: 19 HOLDS, 3 PARTLY, 2 DOES NOT HOLD (D2-005, D2-019), 2 UNRESOLVED.
    > 83/83 numbers.tex keys recompute. **Source:** audit/p17/A_REPORT.md §2-3; audit/p17/paper_new_claims.csv.
    ```
- `:88-94` — F-120 table (raw 0.8195 …) → **BLOCK-F120**, inserted before `:88`.
- `:142-143` — "within **±0.06**, crossing zero five times" → **BLOCK-PM**.
- `:144`, `:227-232` (§6 items 1–3) — "Report L36", "Drop 'at every depth'" **Why stale:** items 1 and 3 were done at 2ad758fb. Item 2 is still open and has worsened: D2-019 quotes a ≥0.026 margin that includes HP-KR. → **DF-107** block below the `## 6.` heading.
- `:233-234` (item 6) — still true at the pin: `persona_depth/summary.md:13-14,78` @ 16def2de still cites `../persona_unified/` and `../persona_depth_orig/`, and neither is present. Only `persona_depth` exists under `results/`. **Not stale**; add "(still open @ 16def2de)".
- `:194` — "**F-115 / §5** — his own axes on our wording-matched slice" → DF-100 pointer block. The source is `audit/HPKR_WORDING_CONTROLLED.md` §5 (`:121`).

## 7. audit/COLLABORATOR_FINDINGS.md
- `:3` — "Assembled 2026-09-24 from all three of his branches" **Why stale:** his latest work (2ad758fb, d1b88154, 16def2de) is not recorded.
  - **Insert before :6:** `> **UPDATE 2026-09-24 (P17).** His branch moved to 16def2de after this was assembled. New work: the depth sweep is now in the paper, with an app:depth section and unified-axis (13v12) arms. His own 16def2de commit corrected the HP-KR ranking claim to "8 of 12". Two sentences still do not hold: introduction.tex:68-70 and limitations.tex:17-21 (DF-96, DF-97). Branch ownership (owner statement, thesis_tools/survey/BRANCHES.md): origin/paper and persona-pc-ood are zami456's; Mahdi has one commit, df758c4b, on feat/llama-layer30.`
- `:65-68` (A8) — "our pipeline has no sycophancy axis at all … never been trained on our side" → **DF-109** block (F-126 §1).
- `:114` — "**51 of 79** checked draft claims match his data" **Why stale:** the full pass gives 165/9/2 of 176 at the pin. → **BLOCK-COV**.
- `:123-126` (C1) — "within ±0.06 at every layer, crossing zero five times" → **BLOCK-PM**. Also add D2-005/D2-019 as new C-items.
- `:135` (C3) — "0.9125 / **0.9597**" → **DF-105** block.
- E table (`:187`) — "his axes on our wording-matched slice (F-115)" → DF-100 pointer block.

## 8. audit/COVERAGE_GAP.md
- `:9-35` (existing block) — "600 … 490 … 692 … 71% … 160 / 9 / 7"
  - **Insert before :9:** **BLOCK-COV**.
- `:126-134` (Cause 2) — "Scope froze against a moving target" **Why stale:** it recurred inside Phase 6. The ref moved at 15:44 while the Level-2/Level-1 scripts ran.
  - **Block:** `> **UPDATE 2026-09-24 (DF-106, P17).** The same cause recurred inside Phase 6: origin/paper moved ea1ce52b → 2ad758fb (15:44) → 16def2de between the Level-2 run (15:39) and the Level-1 verdicts (16:40). Fixed for the future by audit/BRANCH_PINS.md and audit/scripts/p17_common.py (every read now uses a pinned SHA).`
- `:53` — "314 / 692 = 45%" is inside the superseded body. BLOCK-COV at the top covers it.

## 9. audit/BRIEF_AS_EXECUTED.md
- `:16-23` block — "490 of 692 … 71% … (160 / 9 / 7)"
  - **Insert before :16:** **BLOCK-COV**.
- `:323-326` — "its claims carry **no `claims.csv` ids**" **Why stale:** D2-001…026 are now in `audit/claims.csv` (Phase 6 D-rows too).
  - **Block:** `> **UPDATE 2026-09-24 (P17).** origin/paper's claims now carry claims.csv ids: 176 D-rows re-pinned to 16def2de, and D2-001…026 for its three newest commits.`
- `:329-331` — "within ±0.06 at every layer and crosses zero five times" → **BLOCK-PM**.
- `:338` — "The optimum layer is 36, not the reported 40" → **DF-107** block (the paper reports it now).
- `:161` — "0.9125 / 0.9597" → **DF-105** block.

## 10. audit/phase6/RESULTS.md
- `:54` — "**Sources.** `origin/paper` @ `ea1ce52b`" → the DF-106 provenance block from §4, `:11`, inserted before `:54`.
- `:15-17`, `:25-30` — "COMPLETE — 166 / 8 / 2", with Table 7 "6 M" (`:40`) **Why stale:** at the pin D-189 is PARTLY, giving 165 / 9 / 2 and Table 7 at 5 M · 1 P.
  - **Block before :5:** `> **UPDATE 2026-09-24 (P17).** At the pin origin/paper@16def2de, 6A reads 165 MATCHES / 9 PARTLY / 2 UNRESOLVED; D-189 (B46) → PARTLY. Coverage after P17 is 533/735 = 72.5%, not the 490/692 at :20. Source: audit/claims.csv; audit/p17/A_REPORT.md §1.`
- `:68`, `:138` — "1,162 generated `\setV` macros" → the DF-106 macro-count block (1,161 at 2ad758fb; 1,176 at 16def2de; 11 `tab_*.tex`).
- `:66` — "Tables 1–11 … all reproduce from his raw files" → add "(Table 7: 35/36 at 16def2de, D-195)".

## 11. audit/DOC_FIXES.md
- `:3` — "**Status: accumulating … Nothing here has been applied.**" **Why stale:** 11 correction blocks were applied in 10 files (`audit/phase6/APPLY_SHEET.md` §1). DF-12 was withdrawn (F-123). The highest id is DF-95.
  - **Insert before :3:** `> **UPDATE 2026-09-24 (P17).** No longer true: the urgent set was applied as 11 correction blocks in 10 files (phase6/APPLY_SHEET.md §1). P17 checked them: 86 of 94 numbers match; see DF-103…DF-105. DF-12 is withdrawn (F-123). P17 adds DF-96…DF-109 (§C-septies). Next free id: DF-110.`
- `:855-857` DF-79 and `:860-863` DF-80 — resolved by him at 2ad758fb → **DF-107** status line in each, e.g. `- **Status 2026-09-24 (P17):** done in origin/paper@2ad758fb (results.tex:136, :228-230).`
- `:863`, `:866-867` (DF-80 fix text, DF-81) — "all within ±0.06", "crossing zero five times" → **BLOCK-PM** (DF-81 remains open, see DF-97).
- `:934-941` DF-94 — "45% … 176 of them in the draft"
  - **Status line:** `superseded by 6E (490/692) and P17 (533/735, draft 176/176 at 165/9/2).`
- `:943-948` DF-95
  - **Status line:** `made concrete by P17: the Phase-6 stamp ea1ce52b is wrong for Level 1 (DF-106).`
- `:923` DF-89 — "our pipeline has never trained it" → a DF-109 status line.
- **New entries.** Insert as `## C-septies. From P17 (2026-09-24)` before `## D.` at `:981`:

### DF-96 · P1 · **the paper says the base probe out-ranks every contrast on HP-KR "from layer 36" — it trails at 4 of those 12 layers** (for zami456)
- **Where:** `paper/sections/introduction.tex:68-70` @ 16def2de (D2-005, DOES NOT HOLD).
- **Now:** "…from layer~36 it ranks \hpkr{} better than any contrast we build".
- **Correct:** the base probe leads at 8 of 12 layers from L36. It trails the best contrast at **L44 0.828 vs 0.908, L48 0.837 vs 0.890, L52 0.877 vs 0.883, and L76 0.838 vs 0.925**.
- **Fix:** use the wording 16def2de already gave the abstract and limitations: "at 8 of the 12 layers from 36".
- **Evidence:** `results/persona_depth/depth.csv` @ 16def2de (re-derived by the P17 lead); `audit/p17/paper_new_claims.csv` D2-005.

### DF-97 · P1 · **"≥0.026 on the other datasets at all 16 layers" includes HP-KR** (for zami456)
- **Where:** `limitations.tex:17-21` @ 16def2de (D2-019, DOES NOT HOLD).
- **Correct:** the 0.026 floor is computed with HP-KR in. With HP-KR excluded (`off_diagonal_excl_hpkr`), the extended-vs-published margin is **−0.011 at L20** and below 0.026 at 8 of 16 layers. The paper's next sentence contradicts it.
- **Fix:** quote the excl-HP-KR figure, which is also DF-81's standing request.
- **Evidence:** `audit/p17/paper_new_claims.csv` D2-019; `persona_ood_ext_L*/summary.csv` @ 16def2de.

### DF-98 · P2 · **three new sentences hold only in part** (for zami456)
- **D2-016** (`results.tex:254-255`): "It ranks \hpkr{}'s lies better than we do" is unqualified. It holds at 8 of 12 deep layers and is false at L20–32.
- **D2-018** (`results.tex:261-263`): "leads … on all six datasets … by at least 0.026" is true of the six-dataset mean. Per dataset, 8v7 leads at L76 (HP-C) and L79 (IT, RepE), by up to 0.036.
- **D2-002** (`abstract.tex:18-19`, `results.tex:103-105`): "higher than the contrast ever reaches anywhere" holds for means only (0.956 > 0.927). Per dataset the contrast reaches 1.000 on RepE, and the lowest raw diagonal is 0.809.
- **Fix:** add "on average" and "at 8 of 12 deep layers" where needed. **Evidence:** `audit/p17/paper_new_claims.csv`.

### DF-99 · P2 · **release gaps still open at 16def2de** (for zami456)
- D2-024: the app:depth pooling and probe checks (0.99989, 0.0001, 0.9996, 0.0017, 1.0000) are hard-coded in `make_numbers.py`, and their activations are not committed.
- D2-025: `results/XXXX-9-llama-all*/results_table.csv` is committed at no pin. `persona_depth/summary.md` still cites the absent `../persona_unified/` and `../persona_depth_orig/`.
- **Fix:** commit the artifacts, or label the constants as author-attested.

### DF-100 · P1 · **six ledger entries hold only in part — append F-128, do not edit them**
- F-111: the "trap" (0/80) is 31/80, and 0.3061/0.2996 are both passA11's.
- F-112: "8 clearing BH" is scoped to block 10 / mean / paper; the family-wide count is 34.
- F-117: see DF-101.
- F-118: "F-115 §5" should read `audit/HPKR_WORDING_CONTROLLED.md` §5 (`:121`).
- F-120: the entry does not say its numbers are the 5-dataset reading; all-10 raw L40 is 0.6760.
- F-127: there are 11 `tab_*.tex`, not 8; the 1,162 lines hold only at 2ad758fb.
- **Propagated pointer (F-118):** `CLAUDE.local.md:47-48`, `PROGRESS.md:2702`, `REPORT_PLAIN.md:335`, `COLLABORATOR_FINDINGS.md` §E, `COLLAB_PAPER_BRANCH.md:194`.
- **Fix:** append F-128 (draft below). **Evidence:** `audit/p17/C_REPORT.md`; `audit/p17/ledger_verify_stdout.txt`.

### DF-101 · P1 · **"within ±0.06 / crosses zero five times" is wrong wherever F-117 was copied**
- **Correct:** L56 is **+0.0633**, and the series changes sign **6** times (series in the header of STALE_LIST).
- **Where:** `PROGRESS.md:2580` `REPORT_PLAIN.md:10,318` `REPORT_TECHNICAL.md:389` `RESULTS_CHAPTER.md:535` `COLLAB_PAPER_BRANCH.md:143,230` `COLLABORATOR_FINDINGS.md:124-125` `BRIEF_AS_EXECUTED.md:330` DF-80, DF-81 here
- **Fix:** BLOCK-PM at each location.

### DF-102 · P2 · **F-120's numbers travel without their 5-dataset scope**
- **Where:** `COLLAB_PAPER_BRANCH.md:88-100`, `REPORT_TECHNICAL.md:393-396`, `REPORT_PLAIN.md:344-348`.
- **Fix:** BLOCK-F120. The all-10 raw L40 is 0.6760. **Evidence:** `audit/p17/C_REPORT.md` F-120 row (LAB-ONLY).

### DF-103 · P2 · **an applied correction block mislabels passC11**
- **Where:** `cap/docs/review_page.html:173`: "re-measured at 0.3061 / 0.2996 on passA11/passC11".
- **Correct:** both values are passA11's (contrast / PC1). passC11 reads **0.3066 / 0.2995**. The 0.313 claim is unaffected.
- **Evidence:** `run-crossgen/kd11/passC11/matrices/mean_L40_s42.json` (lab); `audit/p17/correction_blocks_check.csv`.

### DF-104 · P3 · **five line pointers in applied blocks are HEAD numbers the blocks themselves shifted**
Line pointers, old → new:

| Where the block is | Pointer | Should read |
|---|---|---|
| `docs/REVIEW_REQUEST_POOLING.md:23` | `:32` / `:63` | `:51` / `:83` |
| `docs/NOVELTY_CHECK_PROMPT.md:130` | `NOVELTY_ASSESSMENT.md:30` | `:44` |
| `docs/NOVELTY_CHECK_PROMPT.md:132` | `LAYER_PREREGISTRATION.md:93` | `:105` |
| `docs/thesis/sarcasm-pilot-negative-result.md:78` | `SARCASM_DESIGN.md:142` | `:156` |

- **Content fix:** `NOVELTY_CHECK_PROMPT.md:120` item (b) corrects "505 vs 310" at `:110`, but the owner had already fixed that text at e917ecd3 (HEAD:121 reads 529.4 vs 340.4). **Drop item (b).**
- **Fix:** the pointers need a new block, not an edit. Alternatively, cite by quoted phrase rather than line.

### DF-105 · P2 · **two numbers in applied blocks are unverified**
- **0.9597** ("does not affirm knowledge" regex), in: `NOVELTY_CHECK_PROMPT.md:127` `HPKR_WORDING_CONTROLLED.md:38` `REPORT_PLAIN.md:200` `REPORT_TECHNICAL.md:281` `COLLABORATOR_FINDINGS.md:135` `BRIEF_AS_EXECUTED.md:161` DF-55/DF-10
- **Why it is unverified:** the regex is recorded nowhere. The guess `NOT ^(yes|i know)` gives **0.9413**, which is still above 0.919, so the argument stands.
- **"Median gap 0.012 over 17 methods"** (`persona-coordinates-70b-results.md:749`, DF-63 block): the 17-method set is not identified.
- **Fix:** record the regex and the method list, or mark both values "not re-derived".

### DF-106 · P1 · **Phase 6 stamped ea1ce52b on rows partly read at 2ad758fb; one verdict changes at the pin**
- **Stamp:** Level 2 read ea1ce52b. Level 1 read 2ad758fb for D-171 and D-032.
- **Verdict change:** D-189 moves MATCHES → **PARTLY** (CG raw L40 0.7875 → 0.7873).
- **Stored-evidence errors:** D-001's L20/L44 values reproduce at no SHA (true values 0.9564/0.6478, 0.9822/0.7156); its verdict stands. D-171 counted the header as a macro. D-032 says "eight tab_*.tex"; there are 11.
- **4th-decimal drift:** D-195 now reads 35/36; D-009 and D-027 drift in the 4th decimal.
- **Fix:** the provenance block in `RESULTS_CHAPTER.md:11` and `phase6/RESULTS.md:54`; the macro-count block at `RESULTS_CHAPTER.md:437`, `phase6/RESULTS.md:68,138`; future reads through `p17_common.py`.
- **Evidence:** `audit/p17/A_REPORT.md` §1, §4; `d_rows_repinned.csv`; `d_rows_other_affected.csv`.

### DF-107 · P2 · **DF-79 and DF-80 were done by the collaborator at 2ad758fb (in his favour)**
- **The paper now:** reports the L36 peak (`results.tex:228-230`; `depth-off-ext-peak`=0.927 at layer 36); replaced "at every depth" with a count over the sweep (`results.tex:136`).
- **Audit text still asks for these:** `REPORT_PLAIN.md:354` `REPORT_TECHNICAL.md:391-392` `RESULTS_CHAPTER.md:524-527` `COLLAB_PAPER_BRANCH.md:227-232` `BRIEF_AS_EXECUTED.md:338`
- **Fix:** status lines. DF-81 stays open (DF-97).

### DF-108 · P2 · **the entry-point documents describe a pre-Phase-6 project**
- **Where:** `CLAUDE.local.md:3-12`, `:28-40`, and the `PROGRESS.md` header and HANDOFF.
- **What they say:** "122 F-entries", "Phase 6 … NOT started", "45%", "DOC_FIXES … NOT ONE applied", "Results chapter does not exist", "1 entry".
- **Correct:** 127 entries; Phase 6 done; 533/735; 11 blocks applied; `RESULTS_CHAPTER.md` exists; 11 modified files plus `thesis_tools/`.
- **Fix:** see STALE_LIST §12. **Evidence:** `audit/BRANCH_PINS.md`; `audit/phase6/RESULTS.md`.

### DF-109 · P2 · **"our pipeline has never trained `pc:sycophancy`" is stale after F-126**
- **Where:** `RESULTS_BY_AXIS.md:119`, `COLLABORATOR_FINDINGS.md` §A8, DF-89 "real gap".
- **Correct:** trained in 6D-1. It loses to raw at all six cells on 11 datasets (F-126 §1, HOLDS 41/41).

## 12. CLAUDE.local.md (owner's private file — proposal only)
- `:3-7` — "**Phase 6 is planned and NOT started** … 378 … Do not start Phase 6" **Why stale:** Phase 6 ran (`audit/phase6/RESULTS.md`, F-123…F-127), and P17 has run.
  - **Insert at the top (before :3):**
    ```
    > **UPDATE 2026-09-24 (P17).** Phase 6 has run: 6A (all 176 draft claims), 6D, 6E. 6B and 6C were
    > skipped by instruction. P17 then re-pinned everything:
    > - **origin/paper @ 16def2de; asix @ e917ecd3.** See audit/BRANCH_PINS.md. Read other branches
    >   only at those SHAs.
    > - **Coverage:** 533/735 = 72.5%. The 202 still unverdicted are 110 ledger and 92 collaborator notes.
    > - **Ledger:** 127 entries, with F-128 proposed.
    > Start at audit/p17/P17_FACTS.md, then the P17 section at the end of audit/PROGRESS.md.
    ```
- `:12` — "real coverage is **314 of 692 inventory claims (45%)**" → **BLOCK-COV** (short form: 533/735).
- `:31-32` — "holds **122** F-entries. F-115…F-122 … F-117/F-119/F-122 are NOT in the reports" **Correct:** 127 (md5 dc365834). The reports also lack F-123…F-127 and the three newest collaborator commits.
- `:33-34` — "95 proposed corrections and NOT ONE has been applied" **Correct:** 11 blocks applied (APPLY_SHEET §1). P17 proposes DF-96…DF-109.
- `:35-36` — "Coverage is 45% … 176 of them in the draft PDF" → 533/735, with the draft 176/176 at 165/9/2.
- `:39-40` — "`git status` outside `audit/` is normally **1 entry**" → 11 modified files plus `thesis_tools/` at the pin.
- `:41-42` — "**The Results chapter does not exist.**" → `audit/RESULTS_CHAPTER.md` exists (Phase 6A).
- `:47-48` — "**F-115** (0.4630 / 0.6203 / 0.5383)" → DF-100 pointer. The source is `audit/HPKR_WORDING_CONTROLLED.md` §5.
- **Add one line to the State section:** *"F-111, F-112, F-117, F-118, F-120 and F-127 hold only in part (P17). The corrections are in F-128; quote those entries only through it."*

## 13. CLAUDE.md (checked-in)
- `:58` (`kd/` row) — "…the project-wide append-only ledger (F-01…)". Open-ended, so **still true**. No item.
- `:60` (`scale/` row) — "the persona gain shrinks as the raw baseline improves (F-63)" **Why stale:** incomplete. F-110 (the like-for-like 70B rung is −0.013 vs raw) and F-121 (the kd bank at 3B/8B; both banks decline, so the decline is the method's) now carry the scale result. F-121 is HOLDS 17/17 (`audit/p17/C_REPORT.md`).
  - **Block:** `> **UPDATE 2026-09-24 (P17).** Scale result now = F-110 + F-121: 26-persona +0.038 → +0.015 → −0.012, 36-persona +0.036 → +0.019 → +0.005 (3B → 8B → 70B).`
- `:61` (`run-kd/` etc. row) — "the lab PC is authoritative and syncs to Drive" **Why stale:** incomplete. P17 found 8 ledger entries resting on **lab-only** evidence that is in no git tree: F-110, F-113 (part), F-114, F-115, F-116, F-120, F-121, F-126. Their stores include `C:\Users\user4\expLB-resplit\` (not under `expLB\`). F-121's claimed Drive backup (290 files) was not checked.
  - **Block:** `> **UPDATE 2026-09-24 (P17).** Lab-only evidence also lives in C:\Users\user4\expLB-resplit\ (km, phase6). Source: audit/p17/C_REPORT.md "LAB-ONLY evidence".`
- `cap/` row: no P17 fact affects it.

## 14. Other docs outside audit/
- **docs/DRAFT_CORRECTIONS.md:108-110** — "`paper/make_numbers.py` does not exist … on **neither branch**" **Why stale:** the script is at `src/probes/deception-detection/paper/make_numbers.py` on origin/paper (F-119, HOLDS). It is 39,532 B @ 16def2de and was 28,232 B @ ea1ce52b (F-123 / C_REPORT). "`feat/persona-pc-ood` has no `paper/` directory" is **still literally true at 45b16d90** (0 paths).
  - **Block before :108:**
    ```
    > **CORRECTION 2026-09-24 (DF-12 withdrawn / F-119, P17).** The script exists on origin/paper
    > (src/probes/deception-detection/paper/make_numbers.py, 39,532 B @ 16def2de), with
    > paper/generated/numbers.tex (1,176 macros) and 11 tab_*.tex. The persona-pc-ood half is still
    > true (no paper/ at 45b16d90).
    > **Source:** git ls-tree 16def2de / 45b16d90.
    > _Kept for the record below._
    ```
- **docs/TODO.md:74-81** (Stage 3, "blocked") and `:93` (4e "CLAUDE.md … has no `cap/` or `scale/` row") **Why stale:** 3c/3d's purpose, a larger knowledge-controlled HP-KR slice with a quotable F-03 test, was met by another route. F-115 closed the leak (F-03 survives; +0.066, CI [−0.069, +0.189] at the pre-registered layer). F-116 ran the 2,001-row `knowledge_matched` slice. Both are HOLDS in P17 (24/24, 40/40). 4e is done: CLAUDE.md:59-60 now has both rows.
  - **Block before :74:** `> **UPDATE 2026-09-24 (P17).** Superseded by F-115/F-116 (verified in P17); 4e done. Last-updated line (:7) predates the audit.`
- **kd/docs/NEXT.md:33** (item 1, "Grow the HP-KR matched slice — partly blocked")
  - **Block (same content as TODO):** F-116's `knowledge_matched` slice (2,001 rows; 139/1,862) is the larger control; F-115 closed the leak on the 258-row slice. Source: `kd/docs/FINDINGS.md` F-115, F-116; `audit/p17/C_REPORT.md`.
- **kd/README.md:93-100** and **kd/docs/REPORT.md:128-166** — phases 2–5 "not started", and phase 3 "blocked on review" **Why stale:** factor probes ran (F-05). The persona run and axes ran (run-kd, `vectors_s3`, which F-126 used as the collaborator's EXTENDED_VECTORS_DIR). The 11×11 ran (kd11; F-31, F-34). The 3B/8B rungs ran (F-121).
  - **kd "Phase 6 — attribution"** is kd's own phase, not the audit's Phase 6. P17 did not establish its status, so **do not mark it done**.
  - **Block:** `> **UPDATE 2026-09-24 (P17).** Status table stale since 2026-09-19: phases 2–5 have run (F-05, F-31, F-34, F-97…F-99, F-121); phase 6 (attribution) unverified.`
- **docs/thesis/persona-coordinates-70b-results.md:739** (DF-63 block) — "the median gap is 0.012"
  - **Block before :739:** `> **UPDATE 2026-09-24 (DF-105, P17).** "median gap 0.012 (over 17 methods)" in the block below was not re-derived; the method set is not recorded. The four diffs and False flags were re-checked and match (audit/p17/correction_blocks_check.csv).` The `0.788` hits at `:817` and `:965` are our own crossgen values, not the collaborator's. **Not stale.**
- **docs/ONBOARDING.md:45, :96** — "State as of 2026-09-17 … **One thing open — the `kd/` experiment**"
  - **Block before :45:** `> **UPDATE 2026-09-24 (P17).** kd/ and a full audit have since run. Current state: CLAUDE.local.md and audit/p17/P17_FACTS.md. Ledger F-01…F-127, and three wording-matched tests converge (F-115/F-116/F-118).`
- **docs/COLLABORATOR_BRIEFING.md:1-20** — addressed to him against `feat/persona-pc-ood` (head f386a72)
  - **Block at top:** `> **UPDATE 2026-09-24 (P17).** His paper is now origin/paper @ 16def2de. It adopted the depth sweep, the L36 optimum and "8 of 12". Items still for him: DF-96…DF-99 in audit/DOC_FIXES.md.` The internal contradiction at `:500` vs `:620` (clustered hold-out "run" vs "not run") predates P17, so it is not listed here.
- **docs/TODO.md, docs/OPEN_DECISIONS.md, docs/ROADMAP_TO_DEFENCE.md** — beyond the items above, nothing is contradicted by a P17 fact. Their dates (09-21/09-22) are old, but their remaining content is still true.

---

## NEW LEDGER ENTRY F-128 (draft)

### F-128 · 2026-09-24 · **P17 re-pin: the collaborator's branch moved during Phase 6. One draft verdict changes, two new paper sentences do not hold, and six ledger entries (F-111, F-112, F-117, F-118, F-120, F-127) hold only in part. Coverage is 533 of 735.**

**Pins.** After one `git fetch`, which moved no ref:
- `origin/paper` @ **16def2de** (2026-09-24 16:47; previous mark ea1ce52b);
- `origin/feat/persona-pc-ood` @ 45b16d90 (ancestor, nothing newer);
- `asix` @ **e917ecd3** plus the working tree, with this ledger at md5 dc365834 (F-01…F-127).

All later reads go through `audit/scripts/p17_common.py`.

**Provenance conflict (DF-95 made concrete).** The local ref moved ea1ce52b → 2ad758fb (15:44) → d1b88154 → 16def2de (16:48) while Phase 6 ran. Everything was stamped ea1ce52b:
- **Level 2** (15:39) did read ea1ce52b.
- **Level 1** (verdicts 16:40) read 2ad758fb for at least D-171 (1,162 = the numbers.tex line count at 2ad758fb) and D-032 (`fig_depth.pdf` exists only from 2ad758fb).
- **172 of 176** stored values are identical at all three SHAs.

**D-189 revised: MATCHES → PARTLY MATCHES HIS DATA.** The collaborator re-ran his baseline (BLAS drift: 80 cells, max |Δ| 0.0054, all raw-probe). Table 7's CG raw L40 is 0.7875 at ea1ce52b and **0.7873** at the pin, and the paper now prints 0.787. The draft stands at **165 / 9 / 2** over 176, not F-127's 166 / 8 / 2. Knock-on: D-195 is 35/36, and D-009/D-027 drift in the 4th decimal. Stored-evidence errors, verdicts unchanged:
- D-001's L20/L44 values reproduce at no SHA (true values 0.9564/0.6478 and 0.9822/0.7156).
- D-032's "eight tab_*.tex" should be 11.
- D-171 is off by one.

**The collaborator's three new commits.**
- **2ad758fb:** the depth sweep in the paper (16 layers), app:depth, 4 claims corrected, 12 unified-axis CSVs.
- **d1b88154** and **16def2de:** text and numbers.tex only.

83/83 recomputable numbers.tex keys match, and `persona_depth/depth.csv` agrees with every per-layer file. Of the 26 new claims: **19 HOLDS / 3 PARTLY / 2 DOES NOT HOLD / 2 UNRESOLVED**.
- **D2-005, `introduction.tex:68-70`:** "from layer 36 it ranks HP-KR better than any contrast". The base probe trails at L44 (0.828 vs 0.908), L48 (0.837 vs 0.890), L52 (0.877 vs 0.883) and L76 (0.838 vs 0.925).
- **D2-019, `limitations.tex:17-21`:** the "≥0.026 at all 16 layers" includes HP-KR. Without it the margin is −0.011 at L20 and below 0.026 at 8 of 16 layers.
- **In his favour:** the paper now reports the L36 optimum and dropped "at every depth" (DF-79 and DF-80 done).

**Corrections to six entries, recorded here rather than edited in:**
- **F-111:** the "trap" (0/80 against the store's own datapca) does not reproduce. The bank summary's datapca equals the root store's, which gives **31/80**. **0.3061/0.2996 are both passA11's** (contrast/PC1); passC11 reads 0.3066/0.2995.
- **F-112:** "8 clearing BH over 6,300" is scoped to block 10 / mean / `paper` probe. Family-wide the count is **34**.
- **F-117:** every table value matches, but the excl-HP-KR margin reaches **+0.0633 at L56**, so "within ±0.06" is false. The series changes sign **6** times, not 5 (re-derived from `persona_ood_ext_L*/summary.csv` @ 16def2de).
- **F-118:** "F-115 §5" does not exist. 0.4630/0.6203/0.5383 are at `audit/HPKR_WORDING_CONTROLLED.md` §5 (:121) and reproduce from the collab-bank kd11 matrices. The convergence finding is unchanged.
- **F-120:** all 17 numbers are exact, but they are the **5-dataset** reading. Raw at L40 over all 10 datasets is **0.6760**, not 0.8195.
- **F-127:** there are **11** `tab_*.tex` at every SHA, not 8. numbers.tex has 1,162 lines only at 2ad758fb (1,161 macros). It holds 1,176 macros at 16def2de.

**Verified.** 447 numbers across F-110…F-127 were re-derived from raw (git at pinned SHAs; lab PC read-only over BatchMode ssh, md5-verified); 432 match. The other 12 entries HOLD. No headline measurement fails.

**Correction blocks.** Of the 94 numbers in the 11 applied blocks, 86 match. The remaining eight are:
- 5 stale line pointers;
- 1 passC11 mislabel (`cap/docs/review_page.html:173`);
- 2 unverified (0.9597; "median gap 0.012 over 17 methods").

**Coverage** (`p15_coverage.py` after `p17_merge.py`): **533 / 735 = 72.5%**, up from 490/692. `claims.csv` has 643 rows; `fmap_all.csv` covers F-01…F-127. 202 inventory claims remain: 110 ledger (TODO) and 92 collaborator notes (CANDIDATE).

- **source:** `audit/BRANCH_PINS.md` `audit/p17/A_REPORT.md`, `d_rows_repinned.csv`, `paper_new_claims.csv`, `csv_drift.csv` `audit/p17/C_REPORT.md`, `ledger_verify_stdout.txt`, `fmap_F110_F127.csv`, `correction_blocks_check.csv` `audit/claims.csv` (D-189) `results/persona_depth/depth.csv` and `persona_ood_ext_L{20..79}/summary.csv` @ 16def2de
- **caveats:**
  1. LAB-ONLY evidence (in no git tree): F-110, F-113 (part), F-114, F-115, F-116, F-120, F-121, F-126.
  2. F-121's 290-file Drive backup was not checked.
  3. D2-024 and D2-025 are unresolvable from the repository: the constants are hard-coded and `results_table.csv` is not committed.
  4. The 0.9597 regex is not recorded; a guess gives 0.9413, which is still above 0.919.
- **supersedes:** on the points above only — F-111, F-112, F-117, F-118, F-120 and F-127, plus F-127's "166 / 8 / 2". Each entry stays as written.
- **status:** verified

---

## COVERAGE

| file | stale items |
|---|---:|
| audit/PROGRESS.md | 8 |
| audit/REPORT_PLAIN.md | 6 |
| audit/REPORT_TECHNICAL.md | 6 |
| audit/RESULTS_CHAPTER.md | 7 |
| audit/RESULTS_BY_AXIS.md | 2 |
| audit/COLLAB_PAPER_BRANCH.md | 6 |
| audit/COLLABORATOR_FINDINGS.md | 6 |
| audit/COVERAGE_GAP.md | 2 |
| audit/BRIEF_AS_EXECUTED.md | 5 |
| audit/phase6/RESULTS.md | 4 |
| audit/DOC_FIXES.md | 7, plus 14 new entries (DF-96…DF-109) |
| CLAUDE.local.md | 9 |
| CLAUDE.md | 2 |
| docs/DRAFT_CORRECTIONS.md | 1 |
| docs/TODO.md | 1 |
| kd/docs/NEXT.md | 1 |
| kd/README.md + kd/docs/REPORT.md | 1 |
| docs/thesis/persona-coordinates-70b-results.md | 1 |
| docs/ONBOARDING.md | 1 |
| docs/COLLABORATOR_BRIEFING.md | 1 |
| docs/OPEN_DECISIONS.md | 0 |
| docs/ROADMAP_TO_DEFENCE.md | 0 |

Grep hits over the 23 files:

| pattern | hits | relevant | notes |
|---|---:|---:|---|
| `within ±0.06` | 7 (+2 bold variants in COLLAB_PAPER_BRANCH) | 9 | all stale |
| `0.788` | 7 | 1 | RESULTS_CHAPTER:234; the rest are 0.7884/0.7887 or our own values |
| `from layer 36` | 0 | — | the claim lives only in his `introduction.tex` |
| `122` | 29 | 1 | CLAUDE.local.md:31. "122 claims" (Level 1) is correct; the rest are decimals |
| `Phase 6` | 15 | 2 | CLAUDE.local.md:3,7; kd/docs/REPORT.md:164 is kd's own phase |
| `45%` | 12 | 5 | the others are superseded bodies or "~45% of margin" (unrelated) |
| `ea1ce52b` | 3 | 3 | RESULTS_CHAPTER:11, COLLAB_PAPER_BRANCH:18, phase6/RESULTS:54 |
| `no paper/` | 0 literal | 1 | DRAFT_CORRECTIONS:110 (backticked) |
| `8 tab` | 1 | 0 | false hit; "eight `tab_*`" found at RESULTS_CHAPTER:440 |
| `five zero` | 0 literal | 9 | "zero five times" variants: 9 hits |
| `F-115 §5` | 0 literal | 1 | "F-115 / §5" at COLLAB_PAPER_BRANCH:194; F-115 misattribution at 5 more sites |
| `1,162` | 3 | 3 | RESULTS_CHAPTER:437, phase6/RESULTS:68,138 |
| `0.9597` | 7 in scope | 7 | unverified |

