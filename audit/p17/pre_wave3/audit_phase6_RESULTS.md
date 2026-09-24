# Phase 6 — results

_2026-09-24. Shape follows `audit/phase4/RESULTS.md`._

## Status

| step | state |
|---|---|
| **6D-4** DOC_FIXES triage | **DONE** — `DOC_FIXES_TRIAGE.md`, `df_triage.csv` |
| → kill DF-12 + fix DF-85/F-119 pointer | **DONE** — `DOC_FIXES.md` edited, F-123 §1 |
| → consolidated ledger entry | **DONE** — F-123 |
| → urgent 7 | **DONE, APPLIED** — 11 correction blocks across 10 files (`APPLY_SHEET.md` §1) |
| → draft 18 | **LIST READY** — `APPLY_SHEET.md` §2; the PDF has no patchable source in this tree |
| **6A Level 2** (54 claims) | **DONE** — 52 MATCHES / 2 PARTLY |
| **6A Level 1** (122 claims) | **DONE** — 114 MATCHES / 6 PARTLY / 2 UNRESOLVED |
| **6A total** (176 draft claims) | **COMPLETE — 166 / 8 / 2, 0 failures** |
| **6D-3** report date-stamps | **DONE** — correction blocks on both reports + `BRIEF_AS_EXECUTED.md` |
| **6E** rerun `p15_coverage.py` + DF-94 | **DONE** — 490/692 = 71%; `COVERAGE_GAP.md` corrected |
| **6D-1** train `pc:sycophancy` | **DONE on the lab PC** — trained, 6 cells, 1m50s. It loses to raw everywhere on 11 datasets. F-126 §1 |
| **6D-2** `passB_fine` + `passP` | **DONE** — both read; `passP` turns out to be a PROMPT-span store. F-126 §2 |
| 6B / 6C | skipped by instruction; 202 claims, neither gates the paper |

## 6A — all 176 draft claims have a verdict

| | count |
|---|---:|
| `MATCHES HIS DATA` | **166** |
| `PARTLY MATCHES HIS DATA` | 8 |
| `UNRESOLVED` | 2 |
| failures | **0** |

**Every stated number that exists in a committed result file was located and matched, and no
arithmetic error was found anywhere in the draft.** Tables 1–11, Figures 1–5 and the appendices all
reproduce from his raw files.

| batch | table | n | verdicts |
|---|---|---:|---|
| B23 | Abstract | 5 | 5 M |
| B41 | Table 2 (detection on HP-KR) | 7 | 6 M · 1 P (D-092) |
| B42 | Table 3 (mean off-diagonal) | 8 | 8 M |
| B43 | Table 4 (LB protocol) | 12 | 12 M |
| B45 | Table 6 (zero-shot) | 7 | 7 M |
| B46 | Table 7 (raw vs 12v13) | 6 | 6 M |
| B40 | Table 11 (sycophancy) | 7 | 6 M · 1 P (D-232) |
| B30 | Appendix F | 2 | 2 M |
| B38 | Table 1 (datasets) | 7 | 7 M |
| B44 | Table 5 (persona bank) | 8 | 8 M |
| B47 | Table 8 (response vs 'No') | 6 | 6 M |
| B48 | Table 9 (six reads) | 6 | 6 M |
| B39 | Table 10 (trained on RepE) | 5 | 5 M |
| B35/B36 | Figure 4 left / right | 7 | 7 M |
| B01–B34, B37 | prose, captions, appendices | 85 | 79 M · 4 P · 2 U |

**Sources.** `origin/paper` @ `ea1ce52b` via `git show` (no checkout) —
`persona_ood_ext_L{20,40,44}/`, `persona_ood_followup_L20/`, `persona_positions_L{20,40,44}/`,
`matrix_*.csv`, `persona_pcs.py`, `persona_ood.py`, `paper/make_figures.py`. Plus
`papers/2511.16035v2 (1).pdf` (LB Tables 6–8, pp.54–56),
`papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf`,
`kd/personas/kd_role_groups.json` and `audit/cache/labpc/run-kd/scores/`.
Scripts: `p16_level2.py`, `p16_level2_verdicts.py`, `p16_level1.py`, `p16_level1_verdicts.py`.

## What the pass found

**Only two claims are unverifiable, and they are the same claim twice** (D-030, D-063): Appendix G's
"four decimal places at every layer". Its comparison artifact — the pipeline's own results tables —
exists **neither on the branch nor on the lab PC**, so nobody but the author can check it. Everything
else resolved; see the lab-PC section below. **The release is well evidenced:**
`paper/generated/numbers.tex` holds **1,162 generated `\setV` macros** and every draft number
spot-checked resolves to one. The residual is about *re-running*, not *reading* — the 36 persona
vectors are not on the branch, so Figure 2 cannot be regenerated even though its numbers are
published.

**Four claims need updating before submission** (RESULTS_CHAPTER §9): "recalls near zero for every
method" is false for the Follow-Up Probe (0.37) and the upper bound (0.97); the wording-controlled
test is promised as future work twice and has been run three times (F-115/F-116/F-118);
Gender-Secret is listed as open and is not; and the maximum token read beats the top-tenth the draft
recommends.

**New measurements** (ledger F-124 §1, §4 and F-125): the sign inversion covers **seven** (arm,
layer) cells, not one — which falsifies **DF-46's scope claim** and narrows the draft's own "only at
layer 20"; and **23 of 90** operating cells exceed 1% realised FPR, worst **88.3%**.

## Corrections to this session's own work

1. **The Level-2 pass over-reached on eleven claims, and Level 1 caught it.** Level 2 reduced
   D-087, D-110/111/112/114, D-188 and D-189–194 to `PARTLY` for not disclosing the dim=1 identity
   or the L20 sign flip. **The paper discloses both** — §3.4 states the identity *with its
   condition*, Table 3's caption flags the 1-d rows, Figure 4's caption gives the reason, §4 gives
   the sign flip with the exact worked example (CG 0.940 → 0.060), and intervals are over examples,
   never over matrix cells. All eleven revised to `MATCHES`. **A framing criticism is not verifiable
   from the tables it criticises** — recorded in F-125 §1.
2. **Three "mismatches" were my rounding, not his.** 0.6255, 0.6535 and 0.7875 are exact half-way
   values; the draft rounds half-up and is right. Python's `.3f` is banker's rounding.
3. **One note first read "only 5 of 90 cells sit at or below 1%."** Backwards — 67 at or below,
   23 above.
4. **DF-42's "10 datasets" resolved:** its 0.5494 / 0.6652 pair is *all but
   `harm-pressure-kr-matched`*, not block9 (which gives 0.5467 / 0.6870). F-123 §2.



## 6D-1 / 6D-2 / the unresolved claims — closed on the lab PC (2026-09-24)

Full detail in `kd/docs/FINDINGS.md` **F-126** and **F-127**; lab writes logged in
`audit/LABPC_WRITES.md`.

**6D-1 — `pc:sycophancy` trained here for the first time.** 27_crossgen.py over `run-coords/passB`,
paper16 bank, **11 datasets with `harm-pressure-kr-matched` passed explicitly**, all four default
controls kept, layers 20/40/44 × mean/last. 1m50s, CPU only. The run reproduces `kd11` to 4 dp
(raw 0.6623, datapca 0.6651, length 0.5555, pc:unified 0.6715 at L40/mean), so it is directly
comparable. **Result: `pc:sycophancy` loses to raw at all six (layer, pooling) cells on the full 11
datasets** (−0.008 to −0.118) and to `datapca` at all six. Remove `sycophancy_mt` and it turns
positive only under mean pooling at depth (+0.002 at L40, +0.034 at L44 over raw). **0 of 264
dataset-level bootstrap comparisons in the run are significant** — for any feature.

**6D-2 — `passP` is a prompt-span store.** Its chunk metadata reads `span = prompt` against
`span = response` for every other store; same `row_ids`, same row counts, same nnsight and model,
same day. So it is a **prompt-only baseline**: what a probe reads from the question *before the
model answers*. From the prompt alone the raw diagonal at L44 reads **convincing-game 1.000,
sandbagging_v2 1.000, instructed-deception 0.953**, and off-diagonal transfer is 0.559–0.570.
**`passB_fine`** is a fine L37–L47 sweep over four banks: `axes − raw` is **negative at 36 of 40**
(bank, layer) cells, and `hf275_strict`'s `pc:deception` beats raw at all ten layers by a mean
**+0.039** — the strongest persona result in either store, never reported.

**Five of the seven unresolved draft claims resolved.** D-057/D-065/D-154: Figure 2's geometry
rebuilt from `run-kd/vectors_s3` (identified as his `EXTENDED_VECTORS_DIR` by `pc_handler` n = 106);
cosine **0.474559** → 0.47, projections **−0.0844 / +2.9037 / +0.0458** → −0.08 / 2.90 / 0.05, all
exact. D-040/D-041: both citations real — Natarajan et al. arXiv 2602.01425, Yoo & Skapars arXiv
2609.02893 (PDF already in `papers/`). **D-030 and D-063 remain**, and now firmly: the pipeline's
results tables exist neither on the branch nor on the lab PC.

## A pattern in my own errors, recorded because it runs against the audit's usual failure mode

Three times in this phase I marked something against the collaborator that his own files answered,
and each time the evidence was one directory from where I looked:

1. **The dim=1 framing** — answered in §3.4 of the draft, plus Table 3's and Figure 4's captions.
2. **The L20 sign flip** — answered in §4, with the exact worked example (CG 0.940 → 0.060).
3. **The release claim** — answered by `paper/generated/numbers.tex`, 1,162 generated macros
   including all four Figure 2 values I had called unverifiable.

Net effect: **13 verdicts revised up** (11 at F-125, 2 at F-127). The audit's measured failure mode
is leniency; mine in Phase 6 was the opposite. **Check the generated artifacts before concluding a
number is unsourced.**

## Notes for whoever picks this up

- `git show origin/paper:<path>` is the entire access pattern for his results. No checkout needed.
- `zero_shot_operating.csv` is what Tables 2 and 4 are built from — **not** the `matrix_*.csv`
  files, which give a different base-probe number at L20 (0.381 against 0.109).
- The follow-up run uses a **different RepE row set** (612/306, not 550/275), so Table 10 mixes two
  row sets in one row.
- `pypdf` is installed; `pypdfium2` and poppler are not.
- Correction convention, set by the owner on 2026-09-24: **prepend a dated block before the old
  claim, never overwrite.** Format in `APPLY_SHEET.md`.
