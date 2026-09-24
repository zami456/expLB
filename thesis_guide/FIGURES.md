# FIGURES — every plot that could go into the thesis, and how to use it safely

Companion to `thesis_guide/figures_inventory.csv` (559 rows, one per figure; a PNG/PDF twin is one row).
Built 2026-09-24 by a read-only scan. Pins (see `audit/BRANCH_PINS.md`):
owner = `asix` @ `e917ecd3` + working tree; collaborator = `origin/paper` @ **`16def2de`**
(`origin/feat/persona-pc-ood` @ `45b16d90` holds nothing the pin lacks). `origin/paper` has since moved
to `d5b92c5a` (3 abstract commits); the only binary that changed is `paper/main.pdf`. **No figure changed.**
Builder script (re-runnable, scratchpad): `build_inv.py`. Its logic is summarised in the COVERAGE section.

Columns: `id, path, branch_sha, format, shows, source_of_description (DOC/CAPTION/SCRIPT/GUESS), script,
data_inputs, scope, stale_risk, referenced_by, caveats`. `referenced_by` is an exact-substring search over
docs (see COVERAGE). "none found" means the figure's path was not found in any doc. It does not mean
the figure is unused.

## 1. Where the figures are

| id range | location | branch | n | whose | notes |
|---|---|---|---|---|---|
| FIG-001–005 | `figures/thesis/fig{1..5}_*.{pdf,png}` | asix | 5 | owner | `kd/scripts/make_thesis_figures.py`; fig1/3/4 **hard-coded numbers** |
| FIG-006–018 | `figures/persona-coords/*.{pdf,png}` | asix | 13 | owner | Q1–Q11 figures, `src/assistant_axis_ndif/19_figures.py` ← `run-coords/*.json` |
| FIG-019–138 | `figures/crossgen/**` (+`INDEX.md`, 3 of them) | asix | 120 | owner | `28_crossgen_figures.py` ← `run-crossgen/**/matrices|summary|significance.json` |
| FIG-139–188 | `figures/{barcharts,heatmaps,violin_plots,examples,radarcharts,harm-pressure}`, 2 loose PDFs | both (same blob) | 50 | **Liars' Bench authors (2025)** | upstream; attribute |
| FIG-189–359 | `data/probes/{GoT,RepEng}/**/probe_scores_*.png` | both (same blob) | 171 | upstream 2025 | probe diagnostics at L7/19/60, 3 models. Not thesis material |
| FIG-360–365 | `src/probes/deception-detection/paper/generated/fig_*.pdf` + `sections/fig_pipeline.tex` | paper only | 6 | collaborator | **the paper's actual figures** (`make_figures.py`) |
| FIG-366–544 | `src/probes/deception-detection/results/**/*.png` | paper only | 179 | collaborator | screen figures from `persona_ood_figures.py`, `layer_sweep.py`, `persona_positions.py`, `results/plot.py` |
| FIG-545–547 | `thesis/images/fig-{5,17,18}.jpg` | working tree | 3 | template | **BRAC template placeholders** (cricket charts, generic flowchart). Delete them |
| FIG-548–554 | `assistant_axis/img`, `mask/assets/fig1.png`, `pacchiardi/imgs/*` | submodules | 7 | third party | related work only |
| FIG-555–559 | draft PDF `full paper draft…/main (1).pdf`, Figs 1–5 | asix | 5 | collaborator | older compiles; do not crop |

The paper's `\includegraphics`, read at the pin: `method.tex:91` fig_personas; `results.tex:36` fig_hpkr, `:75`
fig_transfer, `:156` fig_operating, `:267` fig_depth; `introduction.tex:42` `\input{sections/fig_pipeline}`
(TikZ). Numbering by input order (INFERRED): 1 pipeline, 2 personas, 3 hpkr, 4 transfer, 5 operating, 6 depth.
The 2026-09-22 draft PDF has only Figs 1–5, and its Fig 3 is the **old 3-layer** HP-KR plot
(`audit/phase0/draft_report.md` §6).

## 2. Getting a figure into Overleaf (`thesis/`)

1. **Fix the template first.** `thesis/main.tex:38` reads `\graphicspath{/images}`. That is wrong
   syntax and an absolute path. Use `\graphicspath{{images/}}`. No chapter contains `\includegraphics` yet.
2. **Use the vector PDF**, not the PNG. Every owner figure has a PDF twin. The collaborator's paper figures
   are PDF only. His `results/*.png` are 200-dpi screen figures, up to 16 inches wide
   (`make_figures.py` docstring). Do not use them where `generated/*.pdf` covers the same thing.
3. **Copy owner figures** (working tree): `cp figures/thesis/fig1_register.pdf thesis/images/`.
   Use a flat, unique name for figures with a nested path, e.g.
   `crossgen_fig6_paper16shared_L40_mean_paper_k2.pdf`. Many crossgen basenames repeat across folders.
4. **Copy collaborator figures without checking out his branch** (checkout is forbidden). This is binary-safe in git bash:
   `git cat-file blob 16def2deec34aeee532511df116adb7de5522ca7:src/probes/deception-detection/paper/generated/fig_transfer.pdf > thesis/images/collab_fig_transfer.pdf`.
   Then check the size: `git cat-file -s <sha>:<path>` must equal `wc -c`. Always use the **pinned SHA**,
   never `origin/paper`, which moves.
5. **Caption + credit.** For every collaborator figure, the caption must say it is the co-author's
   (zami456, `origin/paper`) and redrawn from his committed results. For every Liars' Bench,
   MASK, Pacchiardi or Assistant-Axis figure, cite the source paper. Better still, redraw it or cite it instead of reproducing it.
6. Overleaf compiles PDF figures directly. Nothing needs converting. `thesis/` has no `.latexmkrc`
   and no `\usepackage{tikz}`, so porting `fig_pipeline.tex` means adding TikZ and replacing its
   `\V{…}` macros with literal numbers from `generated/numbers.tex`.

## 3. Regenerating (only if the data changed — each script WRITES into the repo)

| figures | command (repo root, laptop; the lab PC blocks matplotlib DLLs, F-57) | inputs present locally? |
|---|---|---|
| `figures/thesis/*` | `python kd/scripts/make_thesis_figures.py [--data …] [--cap …]`. OUT is hard-coded to `figures/thesis/`, so it will overwrite | fig2: `kd/results/fig_data.json` Y. fig5: `run-cap/export/cap_fig_data.json` Y. Both are exported on the lab PC (`kd/scripts/export_fig_data.py`, `cap/scripts/export_cap_fig.py`) |
| `figures/persona-coords/*` | `python src/assistant_axis_ndif/19_figures.py [--only q5]` | `run-coords/*.json` Y |
| `figures/crossgen/**` | `python src/assistant_axis_ndif/28_crossgen_figures.py --only fig6 --layers 40 [--run run-crossgen/kd11 --figdir <dir>]` | `run-crossgen/**` Y (kd11 has 21 layers in `summary.json`) |
| collaborator `generated/*.pdf` | extract his sub-project without checkout (`git archive 16def2de src/probes/deception-detection \| tar -x -C <scratch>`), then run `python paper/make_figures.py` in it | CSVs Y at the pin. **`personas()` needs the 36 persona vectors, which are not committed** (lab PC `run-kd/vectors_s3`, `audit/RESULTS_CHAPTER.md` §10.1). **Fig 2 cannot be regenerated from the repo** (F-127) |

Point `--figdir` at a scratch folder and copy from there. Any regeneration into `figures/` is a
repo write the owner must approve.

## 4. Candidate figures by thesis chapter / theme

Themes follow `thesis_guide/RESULTS_SOURCES.md` §(c) T1–T12. Chapter names follow `thesis/main.tex`.
**Bold** = recommended first choice.

**Ch 1 Introduction / Ch 4 Proposed Methodology**
- **FIG-365 pipeline schematic** (collab TikZ). Redraw it for the thesis and credit him. The draft's Fig 1 is the same.
- FIG-018 `validate_pca`: what dominates the activations (dataset identity). This is a pipeline check and suits Methodology.
- FIG-360 `fig_personas` (36-persona geometry, L40). The numbers reproduce exactly (RESULTS_CHAPTER §10.1),
  **but it is not regenerable** (F-127). Caption fix: sincere-ignorance personas sit *between* the two clusters, nearer the honest one,
  not "with the honest cluster".
- FIG-004 `fig4_geometry_by_scale` (F-37, 3B/8B/70B): hard-coded from logs.
- FIG-154 etc. Liars' Bench example boxes. Upstream work: cite it rather than reproduce it.

**Ch 2 Literature Review**: do not reproduce FIG-548–554. Cite the papers in `papers/`.

**Ch 5 Result Analysis**
- *T1 replication at 70B, persona coordinates vs raw vs matched control.*
  **FIG-138 `crossgen/matched/L40_mean_paper`** (feature vs raw AND vs matched control) and
  **FIG-084 `figure6_style/banks/paper16_shared/L40_mean_paper_k2`**. Use the shared-question bank.
  The root `figure6_style/*` (FIG-026–037) uses paper16, whose unified PC1 is the question family (F-18).
  Against the matched control the gain vanishes (F-17, −0.003). The root −0.034 headline is +0.028 without
  `sycophancy_mt` (`audit/PROGRESS.md` S-12). Supporting: FIG-009 `q2_layer_sweep`
  (`docs/thesis/persona-coordinates-70b-results.md` §13 item 2 says "redraw the Q2 figure"),
  FIG-012 `q6_layers`, FIG-016 `significance_forest`, FIG-024 `bank_comparison/L40_mean_paper`.
- *T2 scale ladder.* **No figure shows F-121** (both banks decline with scale). FIG-003 `fig3_baseline_artifact`
  plots F-63's three rungs from hard-coded values. It does **not** match the F-57 description
  (`audit/claims/F-57.md:84`), and it predates F-121. A new figure is needed.
- *T3 HP-KR / register.* **FIG-001 `fig1_register`** (F-13, hard-coded; audit verdict on F-57 is PARTLY HOLDS).
  Collaborator's view: **FIG-361 `fig_hpkr`** (16 layers). Caveats: L20 BA 0.533 is at an 88.3% FPR
  (`audit/RESULTS_CHAPTER.md:95-104`). "From layer 36 it ranks HP-KR better than any contrast" does
  not hold at L44/48/52/76 (F-128 D2-005).
- *T4 knowledge-denial extension (collaborator headline).* **FIG-362 `fig_transfer`**. Its right-panel rows are
  constant by construction, because a 1-feature probe learns only a sign. **FIG-363 `fig_operating`**. Owner-side: FIG-103/105
  `kd11/matched/kd{,_s23}_L40_mean_paper`. Do **not** use FIG-435 `persona_ood_ext_L40/transfer_matrices.png` as
  "the transfer figure". Its right panel is pc3, not the extended contrast, and it is STALE (below).
- *T5 wording-matched convergence (F-115/F-116/F-118, report as ONE finding).* **No figure exists.**
  Build one from the three sources named in `RESULTS_SOURCES.md` T5.
- *T6 sycophancy and the capitulation bank.* **FIG-005 `fig5_capitulation`** (F-65/F-66). Its data was exported
  on the lab PC from `cap_*` banks that are not in the local `run-crossgen`. FIG-006 `q10_sycophancy_axis`
  uses the paper16 assistant direction, not the published axis (results doc §13 item 1). Collaborator: FIG-436 etc.
  `transfer_sycophancy.png` (deception PCs vs sycophancy PCs).
- *T7 positive control (+0.125, pooled training, leave-one-cluster-out).* **No figure exists.** FIG-386/387
  `loo_L{20,40}/loo.png` is the collaborator's leave-one-*dataset*-out. That is a different design and
  a different number, so do not caption it as +0.125.
- *T8 depth.* **FIG-364 `fig_depth`** (collab, 16 layers, new at 2ad758fb, not in the draft). Caveat: "≥0.026 at
  all 16 layers" includes HP-KR, and without it the margin is −0.011 at L20 (F-128 D2-019). Owner: FIG-002
  `fig2_layer_profile`. Its CI **does** exclude zero at L40/L44, contrary to F-57 (audit F-57 / F-123).
  It is not regenerable locally, because the local `significance.json` has 3 layers against the 21 exported.
  FIG-388 `persona_depth/depth.png` is the 6-panel screen version. Its `summary.md` cites two folders that are
  missing (F-119).
- *T9 token position / pooling.* FIG-013/014 `q8_pertoken`, `q8_trained_pertoken`. Collaborator: FIG-541/542
  `persona_positions_L40/{position_profile,reads}.png`.
- *T10 operating point.* FIG-363 `fig_operating`, or FIG-440 `zero_shot_fig2.png` (same content, screen PNG).
- *T11 other questions.* FIG-007 `q11_sleeper` (sleeper detection is falsity; gender-secret at chance),
  FIG-015 `q9_persona_count` (reword "peak" as "plateau", results doc row 17), FIG-010 `q3_sample_efficiency`
  ("grows with n" is superseded, row 18), FIG-008 `q1_contamination` (provisional, row 22), FIG-011 `q5_semantics_layers`.

**Appendix**: crossgen `per_source`/`per_target` (kd11, 21 layers), `blocks`, `bank_comparison` with its CSVs.
Also the collaborator's `persona_ood_mod_L*/variants.png` (more rollouts did not help) and `persona_ood_ext_L24–79`
(depth-sweep layers: pooled on NDIF, base probe trained by the runner and never re-scored, per their `summary.md`).

## 5. Stale or unsafe (from `stale_risk` / `caveats`)

- **FIG-435** `persona_ood_ext_L40/transfer_matrices.png` is **STALE**. `matrix_baseline.csv` was re-run at `2ad758fb`
  (2026-09-24; BLAS drift, 80 cells, max |Δ| 0.0054, F-128), but the PNG dates from `97740753` (09-20). At L20
  and L44, `transfer_matrices`/`transfer_delta` were regenerated. At L40 only `transfer_delta` was. The
  paper `generated/*.pdf` were regenerated afterwards (`d1b88154`), so they are fine. `transfer_sycophancy*.png` read
  pc3/syc3 only, which were not re-run.
- **Hard-coded, will not follow re-runs**: FIG-001, FIG-003, FIG-004.
- **Data only on the lab PC**: FIG-002 (21-layer significance), FIG-005 (cap banks), FIG-360 (persona vectors).
- **passA figures** (`crossgen/kd11/passA/**`, FIG-106–125): the passA `summary.json` lists 10 datasets.
  HP-KRm was dropped because of the 89/9 slice (F-20). The figures predate the F-69 repair. Use the non-passA kd11 figures.
- **persona-coords** (FIG-006–018): every input JSON was first committed after its figure (`0c1a2246`). Local
  mtimes of all inputs are older than the figure, so they are consistent. Git alone cannot prove it.
- **Collaborator `XXXX-9-llama-all*/plots`** (FIG-366–385): the base-probe run tables and configs are not committed, only the plots.
- **`persona_ood_followup_L20`**: the figures are titled "layer 20" but read the follow-up "No" token (its `summary.md`).

## 6. Same thing, plotted by both authors

| content | owner (asix) | collaborator (paper @16def2de) | how they differ |
|---|---|---|---|
| train × test transfer, raw vs persona | FIG-017 `transfer_L44_last`; crossgen `figure6_style/*` | FIG-362 `fig_transfer`; `persona_ood*/transfer_matrices.png` | owner: 10/11 datasets, paper16/own banks, k=2/5 PCs. Collab: 6 datasets, 36-persona bank, 1-d ext contrast |
| HP-KR failure | FIG-001 `fig1_register` (trained, sycophancy_mt too) | FIG-361 `fig_hpkr` (zero-shot + BA, 16 layers) | different arms. RESULTS_SOURCES T3: "the paper's 8v7 axis" is a different bank on each side |
| depth profile | FIG-002, FIG-009, FIG-012; kd11 per_source/target | FIG-364 `fig_depth`, FIG-388 `depth.png` | owner 21 layers (0–76), gain of unified vs raw. Collab 16 layers (20–79), transfer + zero-shot |
| persona geometry | FIG-004 (cos PC1 vs axes by scale) | FIG-360 (36 personas in the 8v7 × 4v6 plane) | complementary, not redundant |
| per-token reads | FIG-013/014 | FIG-541/542 (+L20/L44) | owner trained per-token vs pooled. Collab six zero-shot reads |
| sycophancy basis | FIG-006, FIG-005 | `transfer_sycophancy*.png` | owner: real capitulation data. Collab: syc PCs as a transfer basis |
| operating point, LB Fig-2 layout | — (LB upstream bar charts FIG-139–150) | FIG-363, `zero_shot_fig2.png` | collaborator reproduces the upstream layout |
| supervised pooled baseline | none | FIG-386/387 `loo.png` | owner's +0.125 control has no figure |

## 7. COVERAGE

| pattern / source | found | examined → rows | skipped + reason |
|---|---|---|---|
| WT `find . -path ./.git -prune` for `*.{png,pdf,svg,jpg,jpeg,eps,gif,webp}` (no symlinks under `run-*`/`results`) | 549 | 507 → 369 rows (owner PDF/PNG twins merged) | 23 `.venv*/` package assets; 11 `papers/*.pdf` (reference papers); the draft PDF (represented by 5 embedded-figure rows); 5 `audit/phase0/draft_pages/*.png` (renders of draft pages); 2 MASK logos (`cais_logo.svg`, `scale.svg`) |
| — of which `figures/thesis`, `persona-coords`, `crossgen` | 276 files | 138 rows (PDF+PNG stems, taken from `audit/phase0/local_figures_provenance.csv`, re-checked) | 0 |
| — `run-*/`, `results/`, `kd/`, `cap/`, `scale/`, `docs/thesis/`, `src/` (non-probe) | 0 image files | — | — |
| HEAD `e917ecd3` `git ls-tree -r` | 512 | all matched WT except `images/fig-*.jpg` (3; deleted in WT, same files now in `thesis/images`) | — |
| `origin/paper` @16def2de `git ls-tree -r` | 406 | 221 shared with HEAD (all 221 blob-identical, `git ls-tree` hashes); 185 paper-only → 184 rows + `main.pdf` skipped (compiled paper) | `paper/main.pdf` |
| `origin/feat/persona-pc-ood` @45b16d90; `feat/llama-layer30` @df758c4b; `feat/resumable-acts-cache` @7c282990 | 308 / 221 / 221 | 0 files not already on the paper pin (`comm`) | — |
| `\includegraphics` / `\input{…fig…}` in paper `sections/*.tex` @pin | 5 + 1 TikZ | 6 rows with captions | — |
| `thesis/**/*.tex` `\includegraphics` | 0 | — | the template has none yet |
| Non-image figure sources | mermaid diagram (`docs/thesis/persona-coordinates-70b-results.md:300`), `src/visualization/figure_1.html` | not inventoried | diagrams/HTML, not figure files |
| Generator search: `rg`/`git grep` for each output name in `*.py`/`*.ipynb` (WT + pin); `git log -S` for names with no current writer | — | script found for all rows except LB heatmaps, per-metric bar charts and `examples/` (GUESS/UNKNOWN; only old commits 3ce17a8a/1287cecb/2d625b40 write them) | — |
| `referenced_by` corpus: `docs/**/*.{md,html}`, `kd/**/*.md`, `cap/**`, `scale/**`, `thesis/**/*.tex`, `thesis_tools/**/*.md`, `thesis_guide/RESULTS_SOURCES.md`, `audit/*.md`, `audit/claims/*.md`, `figures/**/INDEX.md`, `src/**/*.md`, draft text (`audit/phase0/draft_text.txt`), paper @pin `*.md`/`*.tex`; brace-expanded (`{a,b}`); keys = path after `figures/` (+ basename.png/pdf) | — | 559 | `audit/*.csv`, `.before*` backups and `audit/p17/pre_*` excluded as copies/inventories. Crossgen keys are full sub-paths, so doc mentions at folder level ("figure6_style") are not credited |
| Caveats: `kd/docs/FINDINGS.md` (F-01…F-128), `audit/claims/F-57.md`, `audit/RESULTS_CHAPTER.md`, `audit/phase0/draft_report.md`, `audit/PROGRESS.md`, folder `summary.md`s | — | hand-written per group; FINDINGS hits are given as F-numbers in `referenced_by` | memory files used only as leads; claims taken from them are marked "memory map" in `caveats` |
| Stale check | — | owner: git history of each input since the figure's commit, plus local mtimes. Collab: last commit of each named input CSV vs the PNG, at the pin | LB/data-probes: upstream, not checked |

**UNKNOWN in the CSV**: the generators of 31 LB PDFs (13 heatmaps, 10 per-metric bar charts, 8 examples), plus
the input data of all 50. The layer of `XXXX-9_lr__…_20260712_053340`. The layer of the `q1_contamination` scope.
