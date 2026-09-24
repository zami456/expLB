# P17 Agent A: re-pinning origin/paper (ea1ce52b -> 16def2de)

Every read used a pinned SHA through `audit/scripts/p17_common.py`. Nothing outside `audit/p17/` and `audit/scripts/p17_*.py` was written.

## 1. Provenance of the 176 D- rows stamped `origin/paper @ ea1ce52b` (`d_rows_repinned.csv`)
- The pinned re-runs `p17_level{1,2}_pinned.py` were run at ea1ce52b, 2ad758fb and 16def2de (`level{1,2}_<sha>.txt`).
  - **Level 1:** the output is byte-identical at all three SHAs.
  - **Level 2:** 2ad758fb adds 6 lines, all of them rows that do not exist at ea1ce52b: the unified-axis operating rows and the "measured here" Follow-Up row. No existing line changes.
  - 2ad758fb and 16def2de give identical output.
- **Source SHA of each stored value.** Summary: stamped values are safe for 172 of 176 rows.
  - **172 rows: AMBIGUOUS-identical.** The quoted quantities are the same at all three SHAs. 100 of these rows cite a file that changed, but none of them quotes a cell that changed.
  - **D-189: ea1ce52b.**
  - **D-171: 2ad758fb.**
  - **D-032: CONFLICT (mixed).**
  - **D-001: NONE.**
- **Verdict on the stamp:**
  - **Level-2 rows (p16_level2, 15:39) read ea1ce52b.** D-189 quotes `0.7875->0.788`. That value exists only at ea1ce52b, and the timing fits: the ref moved at 15:44.
  - **Level-1 rows (verdicts written 16:40) read 2ad758fb, at least in part.**
    - D-171 says "1,162 \setV macros". That figure is the line count of numbers.tex at 2ad758fb: 1,161 macros plus a header. The macro counts are 1,100 at ea1ce52b and 1,176 at 16def2de.
    - D-032 says "five fig_*.pdf". fig_depth.pdf first appears at 2ad758fb.
    - D-032 also cites the ea1ce52b size of make_numbers.py (28,232 B). It says "eight tab_*.tex", which matches no SHA: every SHA has 11.
  - **So the stamp is wrong for these 2 rows and immaterial for the other 174.**
- **One verdict changes at the pin.** D-189 (Table 7, CG raw probe L40) goes from MATCHES to **PARTLY MATCHES HIS DATA**.
  - The draft prints 0.788.
  - At 2ad758fb and 16def2de the value is 0.7873, which rounds to 0.787. The drift comes from re-running matrix_baseline.
  - The paper at 16def2de itself now prints 0.787 (`pt-base-cg-L40`).
- **Errors in the stored evidence, found while re-deriving (the verdicts still stand):**
  - **D-001** stores "L20 0.9787/0.6478; L44 0.9826/0.7159; gap largest at L20 (0.331)". No SHA reproduces this.
    - The data give L20 0.9564/0.6478 (gap 0.309) and L44 0.9822/0.7156, identical at 45b16d90, ea1ce52b and 2ad758fb.
    - Only the L40 figure 0.9833/0.7320 is the ea1ce52b value.
  - **D-171:** the count is off by one (the header line was counted as a macro).
  - **D-032:** "eight tab_*.tex" should be 11.

## 2. New and changed paper claims at 16def2de (`paper_new_claims.csv`, 26 rows D2-001..026)
- **Independent re-derivation matches the paper.** `p17_paper_delta.py` recomputes the numbers from the per-layer `persona_ood_ext_L{20..79}` CSVs without importing make_numbers.py.
  - **83 of 83 recomputable numbers.tex keys MATCH; 0 mismatch.**
  - `persona_depth/depth.csv` agrees with the per-layer files in every cell (0 differences).
  - Between ea1ce52b and 16def2de, `pt-base-cg-L40` is the only existing numbers.tex key whose value changed. Every other change adds keys.
- **All the listed framing numbers hold:**
  - in-distribution 0.956–0.984; drop 0.213–0.347; gap ≤0.002; worst dataset 0.542 (HP-C at L64)
  - LB-average lead ≥0.022 at all 16 layers; L36 0.913 vs 0.783; L32 margin 0.055
  - HP-KR: 8 of 12, peak margin 0.063, L44 0.908 vs 0.828; ranking peak at L36 0.927
  - balanced accuracy 0.910/0.745/0.757; base probe 0.967 at L36
- **Verdicts: 19 HOLDS, 3 PARTLY HOLDS, 2 DOES NOT HOLD, 2 UNRESOLVED.**

### DOES NOT HOLD
- **D2-005, introduction.tex:68-70.** The sentence reads: "…from layer~36 it ranks \hpkr{} better than any contrast we build".
  - The base probe leads at only 8 of the 12 layers from 36 on: L36, 40, 56, 60, 64, 68, 72 and 79.
  - It trails the best persona contrast at four layers:

    | Layer | Base probe | Best contrast |
    |---|---|---|
    | L44 | 0.828 | 0.908 |
    | L48 | 0.837 | 0.890 |
    | L52 | 0.877 | 0.883 |
    | L76 | 0.838 | 0.925 |

  - 16def2de fixed this wording in the abstract and limitations but not here.
- **D2-019, limitations.tex:17-21.** The sentence says that on "the other datasets" (that is, not HP-KR) the extended contrast "matches or beats the published one at every one of the 16 layers … by at least 0.026".
  - The 0.026 is computed with HP-KR included.
  - With HP-KR excluded (`off_diagonal_excl_hpkr`), the gap is −0.011 at L20 and below 0.026 at 8 of 16 layers.
  - The next sentence of the paper contradicts it.

### PARTLY HOLDS
- **D2-016, results.tex:254-255.** "It ranks \hpkr{}'s lies better than we do" is unqualified: it is true at 8 of 12 deep layers and false at L20–32.
- **D2-018, results.tex:261-263.** "Leads the published one on all six datasets … by at least 0.026" holds for the six-dataset mean. Per held-out dataset, the published 8v7 contrast leads at L76 (HP-C) and at L79 (IT and RepE), by up to 0.036.
- **D2-002, abstract.tex:18-19 and results.tex:103-105.** "Higher than the contrast ever reaches anywhere" holds for means (0.956 > 0.927). Per dataset it fails: the contrast reaches 1.000 on RepE, and the lowest raw diagonal cell is 0.809.

### UNRESOLVED
- **D2-024 (app:depth).** The pooling and probe checks (cos 0.99989, 0.0001, 0.9996, 0.0017, 1.0000) are hard-coded constants in make_numbers.py. The activations needed to re-derive them are not committed.
- **D2-025 (method.tex:122-125).** Checking the 4-dp reproduction needs `results/XXXX-9-llama-all*/results_table.csv`, which is not committed at any pin. For the same reason, the bp-* keys were re-derived from the base-probe rows of `zero_shot_operating.csv`. Those are stored to 4 dp and all matched.

## 3. Result-file drift, ea1ce52b -> 16def2de (`csv_drift.csv`)
- **34 CSVs changed: 12 new, 22 modified.** The new files are the `matrix_*uni*` files.
- **80 cells changed, max |Δ| = 0.0054** (L44 `matrix_balacc_baseline`).
- **Every changed cell is raw-probe ("baseline") data at L20, L40 and L44:**
  - `matrix_baseline`
  - `matrix_delta_pc3_vs_baseline`
  - `matrix_balacc_baseline`
  - the baseline row of `summary.csv` (max 9e-5) and of `summary_balacc.csv`
- **Added material:** 62 rows (the unified-axis arms) and 6 columns (`auroc_uni_*`). No zero_shot or operating value changed.
- **No result file changed after 2ad758fb.** d1b88154 and 16def2de touch only the paper text and numbers.tex.
- The same 22 files and 80 cells differ between 45b16d90 and 16def2de (`csv_drift_from_45b16d9.csv`).

## 4. The 61 non-ea1ce52b D- rows (`d_rows_other_affected.csv`, `p17_other_rows.py`)
- 12 of the 61 rows rest on baseline quantities. I re-derived each at 45b16d90 and at 16def2de.
- **3 are affected. No verdict flips, and every win count (18/18 and so on) is unchanged.**
  - **D-195:** "all 36 Table-7 values reproduce to 3 dp" is now 35/36 (CG L40 0.788 -> 0.787).
  - **D-009:** its 4-dp values drift (CG L40 0.7875 -> 0.7873, HP-C 0.5709 -> 0.5708, ID 0.8267 -> 0.8268).
  - **D-027:** its 4-dp values drift in the same way.

## CONFLICT
- **D-032** has mixed provenance: it combines ea1ce52b and 2ad758fb facts plus one count that matches no SHA.
- **D-001** stores values that no SHA reproduces.
- I found no disagreement between numbers.tex and the data.

## Coverage
| What | Searched | Matches | Examined | Skipped (reason) |
|---|---|---|---|---|
| D- rows stamped ea1ce52b | 176 | 176 | 176: evidence hashed at 3 SHAs; 12 re-derived by target; quoted decimals located | 0 |
| Quoted decimals not found at any SHA | 176 rows | 9 rows | 9 | 8 are derived, PDF or lab-PC values (legitimate); D-001 is the error |
| Paper diff (sections + numbers.tex) | 488 diff lines, 7 .tex files | 26 claim units | 26 | intro:30, method:116-119 and limitations:46 are folded into D2-006 (layer count) |
| numbers.tex keys added or changed | 77 | 77 | 70 recomputed, all match; 5 are constants (D2-024) | depth-stride and depth-new-n: definitional (grid step, count of new layers 13), not recomputed |
| Result CSVs changed | 34 | 34 | 34, cell by cell | .png/.md not diffed; no numbers read from them |
| Non-ea1ce52b D- rows | 61 | 12 rest on baseline | 12 re-derived | 49 cite no changed cell or file |
