# Chapter 4 verification log (2026-09-24/25)

**File:** `thesis/chapters/chapter_5.tex`, which is Chapter 4, "Proposed Methodology". Every number in it has a `% src:` comment. This log records where each group of numbers was checked against raw data.

| Values | Checked at | Result |
|---|---|---|
| **Dataset rows and labels** (11 passB datasets) | Local cache `audit/p17/labcache/expLB/run-coords/passB/*/rows.jsonl` → lab PC `C:\Users\user4\expLB\run-coords\passB\*\rows.jsonl` | 11/11 md5-identical |
| **Knowledge-matched slice** (2,001; 139/1,862) | Cache → lab PC `expLB-resplit\passB\harm-pressure-kr-knowledge-matched\rows.jsonl` | md5-identical |
| **Alpaca** (2,000 honest) | `results/subsets/alpaca.csv` (asix) | read |
| **MASK drops** (225/165/25); **roleplaying** (203/84/84) | Loader run `eval_datasets.py --datasets mask roleplaying` (asix) | read |
| **HP-KR 2×2 table, regex AUROC 0.9125, slices 258 / 2,001** | Recomputed from `results/subsets/harm-pressure-knowledge-report.csv` with `eval_datasets.py` (asix) | read |
| **Leak: 199/258 (77.13%), 39/52 test rows, 59 survive** | `expLB-resplit/run-crossgen/splits.json`: cache and lab PC md5-identical. Also lab PC `expLB\run-crossgen\kd11\splits.json` (original split) | Both give 199/258 and 39/52. **The ledger's "~81% (42/52)" (F-113 §5) does not reproduce.** Reported to the owner. |
| **paper16 rollouts** (1,400 / 1,350 / 1,089 / 1,338); n 21–60; vectors (80, 8192) | `run-paper16/` (asix working tree) | read |
| **kd rollouts** (5,400 / 5,250 / 3,599 / 5,153) | Cache `audit/cache/labpc/run-kd/` → lab PC `run-kd\scores` | 35/35 score files md5-identical |
| **kd kept 3,749 (69%), min 47, max 149** | Collaborator `numbers.tex` @16def2de :: 17–21 | Consistent with 3,599 + 150 unjudged anchor rollouts |
| **cap rollouts** (1,300) | Cache → lab PC `run-cap\responses` | 1,300 lines on both |
| **own100** (100 roles; 120,000 rollouts) | `role_groups.json` (asix); lab PC `run\responses` | 100 files, 120,000 lines |
| **sycophancy_mt** (1,182; 434/748) | Cache rows → Drive `tamara:expLB/sycophancy_mt` → lab PC `run-sycophancy-mt` | Drive = lab PC, 5/5 files md5-identical; passB rows md5-identical |
| **Natural sarcasm** (600; 300/300) | `cap/run-sarcasm-natural/rows.jsonl` (asix) | read |
| **kd scale-rung driver** | Lab PC only: `expLB-resplit\run_kd_scale.ps1`, `run_kdscale_crossgen.ps1` | read; in no git branch |
| **ai_audit** (80 rows, 4,211–7,543 tokens, 4.23–13.57 GiB) | `run-coords/ai_audit/items.pt` (asix working tree) × 64 heads × T² × 4 bytes | computed |
| **Code constants** (model, hook, precision, generation, judge, PCA, features, probes, controls, bootstrap, BH, nulls, permutation, equivalence, seeds, LOCO clusters) | asix scripts at the cited lines | read |
| **Pipeline B facts** (arms, basis, probe, balanced accuracy, pooled LOO, random3, 16 layers, 5 datasets + RepE) | `git show 16def2de:` at the cited lines | read |
| **Citation pages** | `papers/*.pdf`, via a subagent report (scratchpad `ch4_citations.md`) | Pages added. Papers with no PDF are cited without a page. |

## Still open
1. `\todo` for the collaborator's remote-pooling code: `trace_layer_outputs(span_mask=...)` has never been committed, and is not on GitHub (any branch), the lab PC or Drive.
2. `references.bib` holds six classic entries whose volume and page numbers come from memory: BH, Maris & Oostenveld, Kriegeskorte, Ravfogel, Ganin, Sharma. They need a DOI check. The 7 template entries are still in the file.
3. Not compiled locally: there is no LaTeX install on this machine. Compile in Overleaf. `main.tex` now loads `amssymb`, `tikz` (with the `positioning` and `arrows.meta` libraries) and `todonotes`.

## Update 2026-09-25: figures, evidence snapshot, pooling disclosure
- **Figures in Ch 4** (five, one-line captions, each discussed in the text with `\ref`):
  - `fig:m-pipeline`: TikZ.
  - `fig:m-kdbank`: TikZ; groups and contrasts taken from `kd_role_groups.json` and the collaborator's `persona_pcs.py:167-170`.
  - `fig:m-personas`: `thesis/images/fig_personas_L40.pdf`, a copy of the collaborator's `fig_personas.pdf` @16def2de, md5 `f8efba67…`. It can't be regenerated (F-127). The text quotes only its `numbers.tex` values (lines 1166–1176).
  - `fig:m-hpkr`: `thesis/images/fig_hpkr_cells.pdf`, made by `thesis_tools/figures/fig_hpkr_cells.py` from the raw HP-KR CSV. It replaces the earlier table.
  - `fig:m-protocols`: TikZ.
- **Collaborator evidence snapshot:** `thesis_tools/collab_evidence/16def2de/`, 20 files, each md5-checked against its blob at the pin (`MANIFEST.md`).
- **Collaborator's reply (2026-09-25):** confirmed that the sweep *results* are all pushed (ea1ce52b). The *code* gap remains: `layer_sweep.py:155/198` calls `trace_layer_outputs(span_mask=…)`, which no version accepts. The owner chose to proceed. The `\todo` is replaced with a disclosure: L20/40/44 are reproducible via the committed `pool_activations.py`; the 13 remote layers, including L36, are not, and are cross-checked at L40 only (row cosine 0.99989).
- **Chapter 4 now has no `\todo`s.**
- **Still not compiled**: there is no LaTeX install here. The TikZ layouts and the PDF figure have not been seen rendered. Check in Overleaf.

## Update 2026-09-25: compared with the other version (`full paper draft for this experiments/report.pdf`, Ch 4, PDF pp. 36–42)
- **Added from it:**
  - §"Design alternatives": a table of D1, D2, D4–D9, plus our pooled probe and kd/cap contrasts.
  - §"Verification of the alternatives": Follow-Up 0.6360 vs 0.634 (D-031 MATCHES), axes within 1e-5 (test), shared vectors 36/36, evaluation stores md5-identical.
- **Softened:**
  - Pipeline B's four-decimal reproduction and L40 remote cross-check (0.99989) are now reported as *not reproducible from committed files*. Audit rows D-063, D2-024 and D2-025 are UNRESOLVED; 0.99989 is a constant in `make_numbers.py:499-504`.
- **Fixed in ours:**
  - The 20 shared questions are copied verbatim from our paper16 set, which *renders* the paper's topics in our own wording. We had said "the paper's questions, copied verbatim".
- **Contradictions left in the other version** (not ours to edit):
  - its Fig 4.1 says the layer was chosen by the evaluation loop;
  - "sincere-ignorance personas sit with the honest cluster" (the uncorrected caption);
  - "Twenty questions are those of Mahadik and Skapars";
  - "about 300 MB" per layer, where `pool_activations.py` says ~450 MB;
  - D3 and D10 have no sources.

## Update 2026-09-25: the four contradictions, checked against raw data and code
| # | Question | Raw evidence | Correct statement | Which version was right |
|---|---|---|---|---|
| 1 | How was layer 40 chosen? | git: `5d7b8560` (2026-09-10) defines `target` = the middle layer, axis[40]; `50709657` (2026-09-15) makes `--layer 40` the probe default; the Q6 sweep `9630f141` is 2026-09-16. Pipeline B: `06da2de2` (2026-09-20) moves to L40 before the kd personas (`97740753`); its sweep `ea1ce52b` (09-23) found a peak at L36 but kept L40. Lu et al. PDF says "middle layer" only (pp. 3, 6, 7, 47, 50); 40 = 80/2 | L40 entered both pipelines as the middle layer, before the kd design and before either sweep. It was not chosen by the evaluation loop. | **Ours** (the amendment `63aabb4b` is borne out by git). The other version's Fig 4.1 is wrong. The pre-registration's first version (`f5d6904`) was also wrong. |
| 2 | Where do the sincere-ignorance personas sit? | Recomputed from the 36 `vectors_s3` (md5 = lab = collaborator), L40, centred on 36, unit 8v7 / 4v6 axes. Means reproduce `numbers.tex` exactly (2.9037 / −0.0844 / −1.2274 / 0.0458 / −0.9999; cos 0.474559). 2-D distances from sincere: anchor 0.602, anti-syc 0.767, plain 1.149, honest 1.228, deceptive 1.391 | On the denial axis they sit level with the published deceptive personas (−0.08 vs 0.05), between honest (−1.00) and deniers (2.90). In the plane, the nearest group is the default assistant. | **Neither.** "With the honest cluster" (other version) is false. Our "nearer the honest group" held on the y-axis only and was misleading. Now corrected. |
| 3 | Where do the 20 shared questions come from? | kd first 20 = `pc_questions_shared.jsonl` (20/20 identical). Paper PDF p. 9 prints the deception questions as one-line topics only. Its sycophancy questions (pp. 9–10) are printed in full: 10/10 of ours match verbatim, and 12/12 of the full sycophancy set are in the PDF text | 10 deception = our renderings of the printed topics; 10 sycophancy = the paper's own questions, verbatim | **Neither.** The other version ("those of Mahadik and Skapars") is right only for sycophancy. Ours ("renderings") was right only for deception. `thesis_guide/METHODS_SOURCES.md:138` ("10+12 … renderings, not verbatim") is also wrong for the 12 sycophancy questions. |
| 4 | Pooled size per layer | float32 storage (`layer_sweep.py:124`, `pool_activations.py:125`). Rows: 11,602 (the `loo_scores.npz` counts: 5 LB + RepE 550) + 2,000 Alpaca = 13,602 → 445.7 MB per layer | ~450 MB per layer (uncompressed float32) | **The collaborator's code docstring (~450 MB).** The other version's "about 300 MB" is not supported. The "60 GB" per-token cache can't be checked (not committed). |
