# P17 Agent C: ledger F-110…F-127 and the Phase 6 correction blocks

Ledger read: `kd/docs/FINDINGS.md` working tree, md5 `dc365834…` (unchanged at the end of the run). Git reads were made at pinned SHAs only: `PIN_PAPER 16def2de`, with `PIN_OLD ea1ce52b`, `2ad758fb` and `d1b88154` used where the collaborator's branch moved. Lab reads went through `audit/scripts/p17_labfetch.py`. It runs `ssh -o BatchMode=yes`, sends the Python program on stdin and uploads nothing. It checks md5 after transport and caches the files in `audit/p17/labcache/`. Nothing was written to the lab PC. Re-run with `p17_ledger_verify.py`, `p17_correction_blocks.py` and `p17_write_fmap.py`.

## Verdict per entry (447 numbers re-derived; 432 match, 6 differ, 2 partly, 7 context/unverifiable)
| F | verdict | one line |
|---|---|---|
| F-110 | HOLDS | 19/19 match. The 19-layer set is the 21 layers minus L72/L76; P17 identified it by brute force because it was never stated. |
| F-111 | PARTLY | 39/41 match. The "trap" (0/80 against the bank store's own datapca) is not reproduced: passA11 bank matrices have no datapca, and the bank summary's datapca equals the root store's, which gives 31/80. 0.3061/0.2996 are passA11's contrast/PC1, not passA11/passC11. passC11 reads 0.3066/0.2995. |
| F-112 | PARTLY | 31/33 match. The "8 clearing BH over 6,300" count is scoped to block 10, mean pooling, `paper` probe; over the whole family it is 34. The bank summary's datapca equals the root store's. |
| F-113 | HOLDS | 8/8 of the checked subset match (MiB arithmetic, k=5 at :882, k=3 in the scale stores). "42 of 52" reproduces under 30g's own split; the crossgen split gives 39. F-43, F-51, F-107 and the cosines were not re-derived. |
| F-114 | HOLDS | 23/23. 8.1× uses rounded inputs; unrounded it is 8.18×, which is F-123's 8.2×. |
| F-115 | HOLDS | 24/24, including the 59 surviving rows and the leaked→guarded cells. The cached files equal the lab md5. |
| F-116 | HOLDS | 40/40: 2,001 rows (139/1,862), both transfer tables, the in-distribution table and the zero-shot confound. |
| F-117 | PARTLY | Every table value matches. But the excl-HP-KR margin at L56 is +0.0633, so "within ±0.06" is false. The series crosses zero **6** times, not 5. |
| F-118 | PARTLY | The Gender-Secret numbers match his summary.md (no activations are committed). Pointer error: "F-115 §5" does not exist. 0.4630/0.6203/0.5383 come from `audit/HPKR_WORDING_CONTROLLED.md` §5 and reproduce from the collab-bank matrices. |
| F-119 | HOLDS | The file exists on paper only; the two folders `persona_depth` cites are absent. |
| F-120 | PARTLY | 17/17 numbers exact, but they are the **5-dataset** (paper5) reading, which the entry does not state. The all-10-dataset raw at L40 is 0.6760, not 0.8195. |
| F-121 | HOLDS | 17/17, plus byte-distinct axis md5s. "Best" is max(axes, all pc:*), unlike F-110's max(axes, pc:deception). That is why the 70B paper16 rung reads −0.0116 here vs −0.0131 in F-110. |
| F-122 | HOLDS | 40/40 at ea1ce52b. At the pin, baseline L40/L44 moved in the 4th decimal (0.7580/0.7296) because he re-ran the baseline. |
| F-123 | HOLDS | 12/12. 28,232 B is make_numbers.py at ea1ce52b; at the pin it is 39,532 B. |
| F-124 | HOLDS | 54/54: the 7 inverted cells, all corrected values, 23/90 cells above 1% FPR, and 88.3% FPR. The same at the pin. |
| F-125 | HOLDS | claims.csv now reads 176 → 166/8/2 and Level-2 52/2. The 160/9/7 snapshot cannot be re-derived, but the id-level chain is consistent. D-153 rests on audit prose only. |
| F-126 | HOLDS | 41/41: all 48 cells of the 6D-1 table, the passP span/n_tok, passB_fine 36/40, and the Figure-2 geometry from the raw vectors. The "full span 4.13" is the span of the group means; the 36 personas span 5.29. |
| F-127 | PARTLY | The 4 macros and the header are exact. numbers.tex has 1,162 lines only at 2ad758fb (1,161 macros). There are **11** tab_*.tex at every SHA, not 8. |

## Mismatches (from raw)
F-117 (±0.06 and five zero crossings); F-127 (8 tab files, macro count); F-111 (the trap; the passC11 label); F-118 (the "F-115 §5" pointer); F-112 and F-120 (scope not stated). No headline measurement fails anywhere.

**Branch moved (CONFLICT against the pin).** `origin/paper` changed baseline rows at L20/40/44, make_numbers.py and numbers.tex between ea1ce52b and 16def2de. Each entry matches the SHA it was written against.

## Correction blocks (`correction_blocks_check.csv`, 94 numbers)
86 y, 1 partly, 2 unverified, 5 n. **Every number checked against a file matches.**
- Partly: review_page.html pairs 0.3061/0.2996 with passA11/passC11; both values are passA11's.
- Unverified: 0.9597, because its regex is not recorded anywhere (a NOT-affirm guess gives 0.9413); and the "median gap 0.012 over 17 methods".
- The 5 `n` are all stale line pointers, and none changes a number:
  - :110, :32/:63 and SARCASM_DESIGN :142 refer to HEAD line numbers that the inserted blocks shifted.
  - NOVELTY_CHECK_PROMPT item (b) corrects "505 vs 310", which the owner had already fixed at e917ecd3 (HEAD:121). The last copy is at e917ecd3~1:110.

## LAB-ONLY evidence
F-110, F-113 (partly), F-114 (confounds, falsify), F-115, F-116, F-120, F-121 and F-126: the expLB-resplit stores, phase6, passP, passB_fine, run-kd/vectors_s3, passA11/passC11 and run-cap. None of it is in git. The Drive backup that F-121 claims (290 files) was not checked.

## Lab reachability
Reachable all session. BatchMode ssh never prompted. Every read was md5-verified after transport. No writes, no launches beyond read-only Python on stdin, and one `p1_peek pt` read.

## CONFLICTS
- CLAUDE.local.md says the ledger "holds 122 F-entries". It actually holds **127** headings (F-01…F-127), which matches BRANCH_PINS.md.
- F-110 already has L-F-110 and a claims row (HOLDS); F-111…F-127 had none.
- kd11/significance.json: git (4,032 tests) ≠ lab (RF-11, known).

## Coverage
| scope | entries | MAPPED | PARTIAL | numbers checked | match |
|---|---:|---:|---:|---:|---:|
| F-110…F-127 | 18 | 14 | 4 | 447 | 432 |
| new claim rows (L2-001…L2-017, F-111…F-127) | 17 | – | – | – | 11 HOLDS / 6 PARTLY HOLDS |
| correction blocks (10 files) | 11 blocks | – | – | 94 | 86 y |
