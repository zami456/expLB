# P17 QA report: independent adversarial check (2026-09-24)

This check was read-only. It wrote only this file and `audit/scripts/p17_qa_{insert_only,csv,d2,inserted}.py`.
All git reads used pinned SHAs (ea1ce52b, 2ad758fb, 16def2de); no moving ref was read. There was one read-only lab
access over BatchMode ssh (two `Get-FileHash` calls). Both md5s match `audit/p17/labcache` (km L40 `86070e76`, passC11 L40 `20caa937`).

## Summary verdict

**P17's measurements and edits hold up.** The edits are insert-only everywhere. No row was lost. Every re-derived
number I sampled matches (26 of 26 sampled rows agree with the stated verdict).

There are **4 FAILs**, all in wording or completeness, not measurement: F1 mis-scoped "176/176" draft tally; F2 D-189 vs
D-195 inconsistent verdicts for one data change; F3 DF-108 says 127 entries (all else says 128); F4 stale text missed. 5 minor notes.

## Checks

| # | Check | Result | Evidence |
|---|---|---|---|
| 1a | Insert-only vs `pre_wave3/` (53 snapshots) | **PASS** | `p17_qa_insert_only.py`: all 53 OK. The self-test FAILS, as it should, on an altered line, a deleted line and a swapped pair. `review_page.html` has no snapshot; removing the DF-103 `<div>` restores the pin md5 `3fe56ff2…` exactly. |
| 1b | `git diff --numstat`: deletions | **PASS** | 18 tracked files, 0 deletions in each. |
| 1c | FINDINGS.md | **PASS** | HEAD is a byte prefix of the working tree. Truncating before `\n### F-128` gives md5 `dc365834…` (= pin, = `pre_wave3`). The tail is exactly one heading (F-128, 54 lines). There are 128 `### F-` headings. (The file already mixes CRLF and LF; F-128 is LF.) |
| 1d | CSVs vs `*.before-p17` | **PASS** | `p17_qa_csv.py`: all old rows are present, in order, with no duplicates. claims.csv has exactly 176 `version_used` appends and 176 `notes` appends (repinned), 12 `notes` appends (other D- rows), and 1 verdict change (D-189). inventory, experiments and fmap_all are byte-prefix appends (+43/+5/+18). |
| 2 | Pin discipline | **PASS** | Every `git show/ls-tree/cat-file/log/diff` in `p17_*.py` takes a `PIN_*` constant, an explicit historic SHA, or `PIN_ASIX~1`. Moving names appear only in docstrings and comments. Note: `p17_ledger_verify.py:4` says the working tree equals asix and was "checked by `git diff --quiet`", but no code runs that check. It is true today, because `git status` shows no result files modified. |
| 3 | Sample re-derivation (seed 17) | **PASS** | 26 rows, all agree. See the table below. |
| 4 | Numbers in inserted blocks (795 lines) | **PASS with FAIL F1** | Every number traced below; the only failure is F1. |
| 5 | F-128 claims | **PASS (minor)** | All figures trace to evidence. Two minor points: "The draft stands at 165/9/2 over 176" (F1), and "0.919" is not in any cited source (N1). |
| 6 | Cross-document consistency | **FAIL F3** | Everything else agrees: 533/735, 202 (110/92), DF-96…109 with next DF-110 (ids unique, max 109), pins 16def2de / e917ecd3 / 45b16d90, and the 11 modified files at the pin (my recount against HEAD matches). |
| 7 | Stale text P17 missed | **FAIL F4** | See F4. |

### Check 3: sample, own code, pinned SHAs and md5-verified lab caches

| Row | Stated | Mine | Agrees? |
|---|---|---|---|
| D2-005 | base trails at L44/48/52/76 (0.828/0.908, 0.837/0.890, 0.877/0.883, 0.838/0.925); leads at 8/12 | identical (`zero_shot_operating.csv`, max over the 5 axis arms); leads at L36,40,56,60,64,68,72,79 | yes |
| D2-016 | holds 8/12 deep, false at L20–32 | base < best at L20/24/28/32 | yes |
| D2-019 | excl HP-KR −0.011 at L20; <0.026 at 8/16 | −0.0108; below at L20,24,28,32,36,72,76,79 = 8 | yes |
| D2-018 | 8v7 leads per dataset at L76 HP-C, L79 IT, by up to 0.036 | L76 HP-C −0.0341, L79 IT −0.0362 | yes |
| D2-002 | 0.956 > 0.927; cell max 1.000; min raw diagonal 0.809 | 0.9564 / 0.9272; 1.0000; 0.8085 | yes |
| D2-012 | 0.913 vs 0.783 (L36); L32 0.905/0.850; L20 base 0.697 | 0.9130/0.7826; 0.9053/0.8498; 0.6972 | yes |
| D2-014 | denial HP-KR balacc 0.910 L40, 0.745 L36, 0.757 L44; >0.6 from L36 to L76 | 0.9102/0.7449/0.7571; L72 0.694, L79 0.501 | yes |
| D2-021 / D2-022 | syc best 0.507 at L48; dec > syc at 14/16 | 0.5073 at L48; 14/16 (fails at L48, L60) | yes |
| L2-001 (F-111) | 28/80 raw, 31/80 datapca; bank has no datapca; 0.3061/0.2996 are passA11 | 28/80, 31/80; bank matrices hold no datapca; passA11 0.3061/0.2996, passC11 0.3066/0.2995 | yes (PARTLY) |
| L2-002 (F-112) | 8 in block 10/mean/paper; family-wide 34 | 8 and 34 of 252 (`kd/significance.json`, lab cache) | yes (PARTLY) |
| L2-004 (F-114) | length 9/10 > 0.5, roleplaying 0.4415; raw 0.6623, uni 0.6715, datapca 0.6651 | identical | yes |
| L2-006 (F-116) | L20/40/44 axes 0.3850/0.3335/0.5267, raw 0.5711/0.7093/0.7203, length 0.6174, in-dist raw 0.9995/0.9992/0.9993 | identical (km cache; lab md5 re-verified) | yes |
| L2-007 (F-117) | +0.0633 at L56; 6 sign changes; 11/16 and 13/16; vs baseline +0.115..+0.273 | identical at both 16def2de and ea1ce52b | yes (PARTLY)¹ |
| L2-009 (F-119) | `make_numbers.py`/`make_figures.py` exist; `persona_depth` cites 2 absent folders | 39,532 B / 16,267 B; `summary.md:13-14,78,107-108`; neither folder exists | yes |
| D-189 | 0.7875 → 0.7873 (0.788 → 0.787) | ea1 0.7875, 2ad/16d 0.7873; L20/L44 unchanged at 3 dp | yes (but see F2) |
| D-001 | true L20 0.9564/0.6478, L44 0.9822/0.7156; stored 0.9787/0.9826 reproduce nowhere | identical at all 3 SHAs | yes |
| D-171 | macros 1,100 / 1,161 / 1,176; lines 1,101 / 1,162 / 1,177 | identical | yes |
| D-019 / D-058 / D-177 / D-211 | 0.890/0.919/0.908; `persona_pcs.py` unchanged; 142/51/148/113/109; top-10% 9 values | 0.8899/0.9185/0.9077; md5 7b02090f at 3 SHAs; identical (`run-kd/scores`, ==3); identical | yes |
| F-120 (extra) | paper5 0.8195; all-10 0.6760 | 0.8195 / 0.6760 (passA L40 cache) | yes |

¹ "Within ±0.06" is false at 4 dp (0.0633). At the 2 dp the claim is written in, it rounds to 0.06, so P17's reading is the strict one. Keep it, but it is a judgement call, not an arithmetic error.

### Check 4: the named numbers
- **Traced and correct:** 533/735 (`p15_coverage.py` re-run) and 490/692 (backups); 11 blocks in 10 files (11 Phase-6 headers in `git diff`); 94/86/5/1/2 (`correction_blocks_check.csv`); CLAUDE.md scale ladder (F-121 table: +0.0382/+0.0148/−0.0116; s23 +0.0360/+0.0194/+0.0054); "8 of 12" and L44/48/52/76 (table above); F-117 +0.0633 / 6 crossings; F-120 0.6760; F-111 31/80; F-112 34; F-127 11 `tab_*.tex`, 1,176 macros (1,177 lines), 39,532 B; pointers `:44`, `:105`, `:51`/`:83`, `:156`; DF-102 values vs `ledger_verify_stdout.txt:279-296`.
- **"draft PDF at 176/176: 165/9/2" is scoped only to the 176 Phase-6A rows. → F1.**
  - The draft has **237** D- claims (inventory source "draft PDF" = 237). All 237 have verdicts. Over all 237, claims.csv reads roughly 208 MATCHES / 18 PARTLY / 8 HOLDS / 2 UNRESOLVED / 1 SUPERSEDED (grouped by the first word of the verdict).
  - 165/9/2 is only the 176 rows stamped ea1ce52b, which were 166/8/2 before P17.

## FAILs and the exact insert-only fix for each

**F1: mis-scoped draft tally.**
- **Where:** `CLAUDE.local.md:22` and `:58`, and `audit/DOC_FIXES.md:954`. These say "the draft PDF at 176/176: 165/9/2". The weaker "The draft stands at 165/9/2" also appears in F-128 and at `PROGRESS.md:2878`.
- **Fix:** insert this block before `CLAUDE.local.md:22` and `:58`, and a bullet before `DOC_FIXES.md:954`:
  `> **CORRECTION 2026-09-24 (P17 QA).** "176/176: 165/9/2" is the Phase-6A subset (the 176 claims unchecked before Phase 6), not the whole draft. The draft PDF has 237 claims (claims_inventory "draft PDF"), all with verdicts; over all 237, claims.csv reads 208 MATCHES / 18 PARTLY / 8 HOLDS / 2 UNRESOLVED / 1 SUPERSEDED. Source: audit/claims.csv D- rows; audit/scripts/p17_qa_csv.py.`
- F-128 must not be edited. Record the scope in the next ledger entry.

**F2: D-189 and D-195 are treated inconsistently.**
- The same cell (CG raw L40, 0.788 → 0.787) downgrades D-189 to PARTLY.
- It leaves D-195 ("all 36 Table-7 values reproduce to 3 dp", now 35/36) at MATCHES.
- **Fix:** choose one policy.
  - If the draft is judged against the data at the pin, append a `notes` entry to D-195 of the form "[P17-QA] verdict at pin: PARTLY (35/36), same cell as D-189". Record the old verdict as P17 did for D-189.
  - If the draft is judged against the data it was built from, add a note to D-189 that the revision is a later data change, not a draft error, and revert it by note.
- Either way, the 165/9/2 tally (and F1's tally) moves by one.

**F3: DF-108's count is out of date.**
- **Where:** `audit/DOC_FIXES.md:1095` says "Correct: 127 entries". The C-septies preface (`:1008-1010`) does not fix this. Every other P17 text says 128.
- **Fix:** insert before `:1095`:
  `- **Status 2026-09-24 (P17 QA):** 128 since F-128 was appended (md5 f81856ef).`

**F4: stale statements P17 did not reach.** None of these has a preceding P17 block.
- **"within ±0.06 … crossing zero five times":** `audit/phase6/APPLY_SHEET.md:268`, and the text printed by `audit/scripts/p8_collab_layer_sweep.py:145`.
  - Fix: insert BLOCK-PM (`STALE_LIST.md` §0) before `APPLY_SHEET.md:268`.
- **The F-111 "trap" ("axes wins 0 of 80 against the bank store's own datapca"):** P17 says this does not reproduce, but the claim is still stated unqualified at:
  - `audit/PROGRESS.md:547`
  - `audit/phase3/CONTROLS.md:112`
  - `audit/phase5/digest_story.md:459`
  - `audit/claims/S-29.md:64-69`
  - Fix: before each, insert a block citing F-128 / DF-100. The bank matrices `passA11/banks/kd_s23/matrices` (mtime 09-21 12:22) carry no datapca column. The datapca in the bank `summary.json` (rewritten 09-22 16:01) equals the root store's, giving 31/80.
  - Caveat: S-29 may have read a pre-rewrite file. Say "not reproducible from current files", not "wrong".
- **0.9597 without its DF-105 note:**
  - `audit/HPKR_WORDING_CONTROLLED.md:38` (listed in DF-105 but not blocked) and `:288`
  - `audit/CORRECTIONS.md:614`, `audit/DOC_FIXES.md:794`, `audit/phase4/RESULTS.md:151`, `audit/phase6/APPLY_SHEET.md:238,278`, `audit/PROGRESS.md:2142,2330`
  - Fix: insert the DF-105 block (the `REPORT_PLAIN.md:227` form) before each.
- **Covered correctly:** CLAUDE.local.md:50 ("122 F-entries"), phase6/RESULTS.md:21 (166/8/2), ledger F-117/F-118 (via F-128); "from layer~36" appears only in P17 quotations; `thesis_tools/survey/BRANCHES.md:257` is untracked tooling.

## Minor notes (not FAIL)
- **N1.** "0.9413 is still above the 0.919 headline" (REPORT_PLAIN:229, REPORT_TECHNICAL:301, COLLABORATOR_FINDINGS:178, BRIEF_AS_EXECUTED:176, DOC_FIXES:1075, F-128 caveat 4):
  - 0.919 is correct: it is D-022/D-087, denial HP-KR L40 = 0.9185.
  - But none of these blocks cites that source, and Agent F deliberately dropped the comparison for this reason. The two agents disagree.
- **N2.** Coverage 490 → 533 comes entirely from 43 new rows added *with* verdicts. No previously unverdicted claim was closed. The 202 is unchanged from before P17. "Up from 70.8%" should be read that way.
  - L-F-110 is still counted as unchecked although claims row `F-110` HOLDS (C_REPORT noted this); real coverage is arguably 534.
- **N3.** `experiments.csv` row C22 gives `when_first` 2ad758fb and 64 files, but 52 of the 64 `matrix_*uni_*` files already exist at ea1ce52b. 2ad758fb added 12 (L20/40/44).
- **N4.** Inventory `status` for the 43 D2/L2 rows is "CANDIDATE" despite verdicts (harmless to `p15_coverage.py`). **N5.** `COLLABORATOR_BRIEFING.md:3-4` cites head `f386a72` without noting the pin 45b16d90 is 2 commits later.

## COVERAGE

| What | Searched | Matches | Examined | Skipped (reason) |
|---|---|---|---|---|
| pre_wave3 snapshots | 53 | 53 resolved | 53 subsequence-checked, plus the self-test | 0 |
| review_page.html (no snapshot) | 1 | 1 | reconstructed to pin md5 | 0 |
| tracked files, numstat | 18 | 18 | 18 | 0 |
| CSV backups | 4 | 4 | every old row, every column | 0 |
| p17_*.py git calls | 14 scripts | ~40 call sites | all | docstrings only |
| Rows re-derived | 26 D2/L2/D rows + F-120 | 23 sampled (seed 17: D2-012/014/016/021/022; L2-001/002/004/006/007/009; D-019/058/177/211) + forced | 27 | D-171 checked by macro count only |
| Inserted lines | 795 in 26 files | 795 | all read; numbers traced | STALE_LIST §0 series taken as re-derived (I re-derived it myself) |
| Repo-wide stale grep | 11 patterns | 60+ hits | all hits classified | `results/`, `.venv*`, `audit/p17/`, `audit/cache` json |
| Lab | 2 md5s | 2 | 2 | 0 |
