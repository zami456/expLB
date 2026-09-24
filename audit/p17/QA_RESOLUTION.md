# P17: resolving the QA findings (lead, 2026-09-24)

QA report: `audit/p17/QA_REPORT.md`. Every fix is insert-only, done by
`audit/scripts/p17_qa_fixes.py`. The script takes snapshots first (`audit/p17/pre_qafix/`, 10 files),
and afterwards each file was checked: every snapshot line is still present, in order.

| QA item | Resolution |
|---|---|
| **F1** "176/176: 165/9/2" mis-scoped | Correction blocks at `CLAUDE.local.md` (2 sites) and `DOC_FIXES.md` (the DF-94 status bullet) say that 165/9/2 counts only the 176 Phase-6A rows. The totals over all 237 draft claims come from `audit/claims.csv` (ids D-*): **208 MATCHES / 8 HOLDS / 18 PARTLY (12 + 2 + 4 annotated) / 2 UNRESOLVED / 1 SUPERSEDED**. F-128, `PROGRESS.md`, `RESULTS_CHAPTER.md:17` and `phase6/RESULTS.md:69` already say "over 176" or describe 6A status, so they are correctly scoped and were left alone. |
| **F2** D-189 vs D-195 policy | **Kept as is, and consistent.** D-189 is a claim about a *printed number* (0.788): at the pin that value is 0.787, so the claim is PARTLY. D-195 is an *ordering* claim ("the extended contrast is higher than the raw probe on every test set at every layer", Table 7 caption). The 0.001 drift does not reverse any ordering, so it still MATCHES. Its evidence note ("35/36 reproduce to 3 dp") was already appended by `p17_merge.py`. Rule: a number claim is judged on the number; an ordering claim is judged on the ordering. |
| **F3** DF-108 says "127 entries" | A correction bullet was inserted before it: there are 128 (F-128 appended). |
| **F4** stale statements P17 missed | Blocks were inserted before each site. ±0.06 / "five crossings": `phase6/APPLY_SHEET.md` table (DF-101). The 0/80 datapca trap: `PROGRESS.md:547`, `phase3/CONTROLS.md:112`, `phase5/digest_story.md:459` and `claims/S-29.md` (DF-100; whether an earlier store read 0/80 is UNKNOWN). 0.9597 unverified: `HPKR_WORDING_CONTROLLED.md` (the table and :288), `CORRECTIONS.md:614`, `DOC_FIXES.md:794`, `phase4/RESULTS.md:151`, `phase6/APPLY_SHEET.md:238` and its table, and `PROGRESS.md:2142` and `:2330` (DF-105). |
| Minor: `p8_collab_layer_sweep.py:145` prints "±0.06" | Not edited, because it is a script and its printed prose is not a claim store. It is recorded here, and DF-101 covers the wording. |
| Minor: `p17_ledger_verify.py:4` claims a `git diff --quiet` check that it never runs | Not edited. The claim was true when checked: the ledger md5 was dc365834 at the pin, and it was re-verified by QA check 1c. |
| Minor: C22 said its unified-axis files first appeared at 2ad758fb | Fixed in `p17_merge.py` and re-merged. 52 files (L24..L79) were already there at ea1ce52b, and 12 (L20/L40/L44) were added at 2ad758fb. Verified with `git ls-tree`. |
| Minor: coverage 490→533 comes only from 43 new rows | Correct, and stated here: **no previously unverdicted claim got a verdict in P17.** 202 remain (110 ledger TODO, 92 collaborator notes). |
| Minor: 0.9413 vs 0.919 | 0.919 is D-022's value (QA). The DF-105 blocks now cite it as "(D-022)". |
