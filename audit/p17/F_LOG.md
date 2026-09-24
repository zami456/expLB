# P17 Agent F — insertion log (2026-09-24)

Insert-only UPDATE/CORRECTION blocks in tracked docs outside audit/. Line numbers are ORIGINAL
numbering (the pre_wave3 snapshot; review_page.html has no snapshot, so its pre-P17 working tree).
Line endings preserved per file (CRLF everywhere except kd/README.md, docs/DRAFT_CORRECTIONS.md,
docs/REVIEW_REQUEST_POOLING.md, which are LF). Blocks that precede a non-quote line are followed by
one inserted blank line (so a following table/paragraph is not swallowed by lazy continuation).
Script: scratchpad `p17_f_insert.py` (not kept in repo).

| # | file | before orig line | first 80 chars | evidence cited |
|---|---|---:|---|---|
| 1 | CLAUDE.md | 66 (after layout table; 2 blocks) | `> **UPDATE 2026-09-24 (P17) — refers to the `scale/` row in the table above.** T` | kd/docs/FINDINGS.md F-110, F-121 rung table; audit/p17/C_REPORT.md (F-121 HOLDS 17/17) |
| 1b | CLAUDE.md | 66 (2nd block) | `> **UPDATE 2026-09-24 (P17) — refers to the `run-cap/` … `run-kd/` row in the t` | audit/p17/C_REPORT.md "LAB-ONLY evidence" |
| 2 | docs/TODO.md | 74 | `> **UPDATE 2026-09-24 (P17).** Stage 3's purpose — a larger knowledge-controlled` | FINDINGS F-115/F-116; C_REPORT rows (24/24, 40/40); CLAUDE.md @ e917ecd3 (4e) |
| 3 | kd/docs/NEXT.md | 33 | `> **UPDATE 2026-09-24 (P17).** The larger control this item asks for now exists ` | FINDINGS F-115/F-116; C_REPORT rows |
| 4 | kd/README.md | 93 | `> **UPDATE 2026-09-24 (P17).** This status is stale since 2026-09-19. Phases 2–5` | FINDINGS F-05, F-31, F-34, F-97…F-99, F-121; C_REPORT (F-121, F-126; vectors_s3 lab-only). kd Phase 6 NOT marked done |
| 5 | kd/docs/REPORT.md | 128 | same block as #4 | same as #4 |
| 6 | docs/DRAFT_CORRECTIONS.md | 108 | `> **CORRECTION 2026-09-24 (DF-12 withdrawn / F-119, P17).** The script exists on` | git ls-tree -r -l 16def2de (make_numbers.py 39,532 B; numbers.tex 1,177 lines = 1,176 macros; 11 tab_*.tex); git ls-tree 45b16d90 (0 paper/ paths); FINDINGS F-119, F-123 |
| 7 | docs/thesis/persona-coordinates-70b-results.md | 739 | `> **UPDATE 2026-09-24 (DF-105, P17).** In the DF-63 block below, *"the median ga` | audit/p17/correction_blocks_check.csv lines 78–87 |
| 8 | docs/ONBOARDING.md | 45 | `> **UPDATE 2026-09-24 (P17).** This section is dated 2026-09-17. Since then `kd/` | FINDINGS F-115, F-116, F-118, F-128; audit/phase6/RESULTS.md; audit/BRANCH_PINS.md |
| 9 | docs/COLLABORATOR_BRIEFING.md | 3 | `> **UPDATE 2026-09-24 (P17).** This briefing reads his work at `feat/persona-pc-` | audit/p17/A_REPORT.md (:41-65); paper_new_claims.csv D2-005/019/002/016/018/024/025; STALE_LIST §11; BRANCH_PINS |
| 10 | cap/docs/review_page.html | 158 (new `<div class="callout">` before the DF-31 callout) | `<p><strong>CORRECTION 2026-09-24 (DF-103, P17).</strong> The correction block b` | correction_blocks_check.csv lines 30–31 (lab passA11/passC11 mean_L40_s42.json) |
| 11 | docs/REVIEW_REQUEST_POOLING.md | 17 | `> **CORRECTION 2026-09-24 (DF-104, P17).** In the block below, the pointers `:32` | correction_blocks_check.csv lines 69, 70, 77 (pointers re-expressed by quoted phrase, since this insert shifts lines again) |
| 12 | docs/NOVELTY_CHECK_PROMPT.md | 109 | `> **UPDATE 2026-09-24 (DF-104 + DF-105, P17).** Three points on the block below.` | correction_blocks_check.csv lines 35, 51, 57–59; git show e917ecd3:docs/NOVELTY_CHECK_PROMPT.md :121 |
| 13 | docs/thesis/sarcasm-pilot-negative-result.md | 70 | `> **UPDATE 2026-09-24 (DF-104, P17).** The pointer `SARCASM_DESIGN.md:142` in th` | correction_blocks_check.csv lines 88–94 |

Not touched (STALE_LIST §14: nothing contradicted by a P17 fact): docs/OPEN_DECISIONS.md,
docs/ROADMAP_TO_DEFENCE.md. kd/docs/FINDINGS.md not touched.

## Skipped / adjusted
- CLAUDE.md lab-only block: STALE_LIST's draft said "(km, phase6)"; "km" is not in C_REPORT's
  LAB-ONLY paragraph, so it was dropped (wording follows C_REPORT: expLB-resplit stores, phase6).
- NOVELTY_CHECK_PROMPT / DF-105: STALE_LIST's "0.9413 is still above 0.919, so the argument stands"
  — 0.919 appears in no evidence file (only in STALE_LIST prose); the comparison was omitted. The
  block states only that 0.9597 is not re-derived and the guess gives 0.9413 (csv line 51).
- TODO block: F-115's "+0.066, CI [−0.069, +0.189]" (STALE_LIST "why stale" text) was not found in
  F-115 and was not used.
- DF-104 new pointers :51/:83 for REVIEW_REQUEST_POOLING are given as "pre-P17 working tree"
  numbers plus quoted phrases, because this very insert moves them again.
- Incident (self-caused, fully reverted): a first run wrote CLAUDE.md, docs/TODO.md and
  kd/docs/NEXT.md then aborted on an LF file; a buggy re-run double-inserted in those three. All
  three were restored byte-exact from audit/p17/pre_wave3/ (md5 matched pre_wave3_md5.txt) before
  the clean run.

## Verification
### git diff --numstat (vs HEAD e917ecd3; includes pre-existing uncommitted insertions)
```
13	0	CLAUDE.md
31	0	cap/docs/review_page.html
8	0	docs/COLLABORATOR_BRIEFING.md
8	0	docs/DRAFT_CORRECTIONS.md
37	0	docs/NOVELTY_CHECK_PROMPT.md
7	0	docs/ONBOARDING.md
25	0	docs/REVIEW_REQUEST_POOLING.md
8	0	docs/TODO.md
23	0	docs/thesis/persona-coordinates-70b-results.md
26	0	docs/thesis/sarcasm-pilot-negative-result.md
7	0	kd/README.md
18	0	kd/docs/NEXT.md
7	0	kd/docs/REPORT.md
```

### Snapshot check (audit/p17/pre_wave3/; all old lines present in order, only inserts)
```
CLAUDE.md: all 331 snapshot lines present in order = True; non-equal ops (op, before-orig-line, n) = [('insert', 65, 13)]
docs/TODO.md: all 110 snapshot lines present in order = True; non-equal ops (op, before-orig-line, n) = [('insert', 74, 8)]
kd/docs/NEXT.md: all 119 snapshot lines present in order = True; non-equal ops (op, before-orig-line, n) = [('insert', 32, 6)]
kd/README.md: all 104 snapshot lines present in order = True; non-equal ops (op, before-orig-line, n) = [('insert', 93, 7)]
kd/docs/REPORT.md: all 166 snapshot lines present in order = True; non-equal ops (op, before-orig-line, n) = [('insert', 128, 7)]
docs/DRAFT_CORRECTIONS.md: all 188 snapshot lines present in order = True; non-equal ops (op, before-orig-line, n) = [('insert', 108, 8)]
docs/thesis/persona-coordinates-70b-results.md: all 1500 snapshot lines present in order = True; non-equal ops (op, before-orig-line, n) = [('insert', 739, 6)]
docs/ONBOARDING.md: all 199 snapshot lines present in order = True; non-equal ops (op, before-orig-line, n) = [('insert', 44, 7)]
docs/COLLABORATOR_BRIEFING.md: all 662 snapshot lines present in order = True; non-equal ops (op, before-orig-line, n) = [('insert', 2, 8)]
cap/docs/review_page.html: no snapshot
docs/REVIEW_REQUEST_POOLING.md: all 139 snapshot lines present in order = True; non-equal ops (op, before-orig-line, n) = [('insert', 17, 6)]
docs/NOVELTY_CHECK_PROMPT.md: all 227 snapshot lines present in order = True; non-equal ops (op, before-orig-line, n) = [('insert', 109, 10)]
docs/thesis/sarcasm-pilot-negative-result.md: all 311 snapshot lines present in order = True; non-equal ops (op, before-orig-line, n) = [('insert', 70, 5)]
docs/OPEN_DECISIONS.md: all 187 snapshot lines present in order = True; non-equal ops (op, before-orig-line, n) = []
docs/ROADMAP_TO_DEFENCE.md: all 169 snapshot lines present in order = True; non-equal ops (op, before-orig-line, n) = []
```

review_page.html: no snapshot; git numstat shows 0 deletions vs HEAD. OPEN_DECISIONS / ROADMAP: unchanged.
