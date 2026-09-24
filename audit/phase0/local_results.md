# Local results on this machine — Phase 0 orientation (facts and locations only)

Scope: the working tree of `C:\Users\SURAIYA\Documents\STUDY\thesis\expLB`, branch `asix`
(HEAD 43ee6d3). Excluded: `.git`, `.venv*`, the four submodules, the empty
`behavioral-self-awareness` gitlink, `audit/`, `.claude`, caches, `__pycache__`, and the vendored
nnsight library docs (`docs/{concepts,developing,errors,gotchas,models,patterns,reference,remote,usage}`,
`NNsight.md`, `docs/NNSIGHT_UPSTREAM.md`). The collaborator's branch and the lab PC / Drive are covered
by other agents. **No verdicts on any result are made here.**

Tags: VERIFIED = checked in a file or by a script in this audit · DOC-ONLY = only in prose ·
NOT-FOUND · CONFLICT · INFERRED (my interpretation, with the reason).

## Files written (all re-runnable, read-only against the repo)

| file | what | produced by |
|---|---|---|
| `audit/phase0/local_results_index.csv` | 183 rows, one per result-bearing folder (depth ≤3; ≤4 under `run-crossgen`, `figures/crossgen`, `src/probes/deception-detection`). Parent rows are roll-ups of their children (`n_child_rows`). | `audit/scripts/p0_local_index.py` |
| `audit/phase0/local_files_flagged.csv` | every individual file with a structural flag (size 204,800, empty, LFS pointer, JSON/JSONL/CSV truncation hints) | same |
| `audit/phase0/scripts_to_outputs.csv` | 187 scripts → lines, git dates, doc line, scheduled-task name, scripts it calls, write ops, output/input paths (INFERRED by pattern matching) and which of those paths are missing locally | `audit/scripts/p0_local_scripts.py` |
| `audit/phase0/local_figures_provenance.csv` | 138 figures (PNG/PDF pairs) in `figures/thesis`, `figures/persona-coords`, `figures/crossgen` → script, function, input files, whether each input is local | `audit/scripts/p0_local_figures.py` |
| (console only) | `remote_sizes.txt` matched against local folders | `audit/scripts/p0_local_remote_sizes.py` |
| (console only) | stray control characters in tracked sources | `audit/scripts/p0_local_ctrlchars.py` |
| (console only) | md5-identical json/jsonl/csv files in the persona result areas (hint for Phase 1) | `audit/scripts/p0_local_dupes.py` |

## 1. Totals (VERIFIED, `p0_local_index.py`)

- **9,897 files, 5.876 GB** in scope. 9,410 tracked by git, 487 git-ignored, 0 untracked-not-ignored.
- By type: csv 7,801 · json 496 · pt 453 · png 309 · pdf 200 · txt 86 · jsonl 74 · log 14 · html 5 · pkl 4 · xlsx 1 · **npy 0 · npz 0 · parquet 0**.
- Ignored (never in git) are all `*.pt` (via `.gitignore:195`) and `*.log` (`.gitignore:61`), plus
  `aa-smoke/`, `aa-smoke-win/`, `logs/`, `.env`. So on this machine every vector/axis/activation
  file and every log exists **only on disk**, not in git.
- No Git-LFS pointer files anywhere (0), and no LFS rules in any `.gitattributes` (the only one,
  `src/assistant_axis_ndif/.gitattributes`, just pins LF line endings and marks `*.pt` binary).

## 2. The result areas

### A. Persona-coordinates work (Llama-3.3-70B on NDIF) — about 1.07 GB, 8,046 files

| folder | files | size | tracked? | what is there (structure only) | producing script(s) (INFERRED from `scripts_to_outputs.csv` + script docstrings) |
|---|---:|---:|---|---|---|
| `run-crossgen/` (root) | 5 own + subfolders | — | tracked (commit b9a497e, 2026-09-18) | 10-dataset cross-generalization: `summary.json`, `splits.json`, `significance.json`, `matrices/` with **only layer 40** (`last_L40_s42.json`, `mean_L40_s42.json`) | `src/assistant_axis_ndif/27_crossgen.py` |
| `run-crossgen/banks/{hf100,own100,paper16_famcentred,paper16_shared}` | 137 each | 0.8 MB each | tracked | same design per persona bank; L40 only (2 matrices + 132 CSV each) | `27_crossgen.py` via `run_crossgen_banks.ps1` |
| `run-crossgen/smoke` | 1 | 0.2 MB | tracked | `last_L44_s42.json` | `27_crossgen.py --smoke` |
| `run-crossgen/kd11` (root) | 2,772 CSV + 42 matrices + 3 JSON | 21.6 MB | tracked (one commit 3dfdb67, 2026-09-21) | 11-dataset grid, **21 layers × 2 poolings** | `27_crossgen.py` via `kd/scripts/run_kd_crossgen.ps1` |
| `run-crossgen/kd11/banks/kd` | 3,321 | 25.9 MB | tracked | 21 layers × 2 poolings (42 matrices) | `kd/scripts/run_kd_bank_crossgen.ps1` |
| `run-crossgen/kd11/banks/kd_s23` | 441 | 3.4 MB | tracked | **3 layers only** (20, 40, 44) | `kd/scripts/run_kd_s23*.ps1` |
| `run-crossgen/kd11/passA` (+`banks/kd_s23`) | 16 | 4.7 MB | tracked | layers 20/40/44, JSON only (no CSVs) | `kd/scripts/run_kd_passAC.ps1`, `run_passAC_11.ps1` |
| `run-crossgen/kd11/passC` | 1 | 0.3 MB | tracked | a single matrix `mean_L40_s42.json` | same |
| `run-coords/` | 33 JSON + `q5_layers/` (8) | 2.8 MB | tracked (JSON); the 3 `items.pt` ignored | analysis outputs of the Q1–Q11 questions (e.g. `FULL_L20/22/40`, `layer_sweep_*`, `q6_layers`, `q8_trained_pertoken`, `q10_sycophancy_axis`, `q11_sleeper_L20/44_last`, `significance`, `zeroshot_nulls`); plus `ai_audit/`, `ai_liar/`, `roleplaying/` each holding only `items.pt` | scripts `9_…`–`26_…` in `src/assistant_axis_ndif/` (JSON→script by file name, INFERRED); `items.pt` from `8_coords_ndif.py` |
| `run-paper16/` | 130 | 265 MB | responses/scores tracked; 54 `.pt` ignored | 26 personas: `responses/` 26 JSONL (16 files × 50 lines, 10 × 60), `scores/` 25, `scores_405b/` 25 (`default` has no score file), `vectors_all/` 26, `vectors_s3/` 26, 2 axes files | `run_paper16_chain.ps1` (→ `1_generate_ndif`, `3_judge_ndif`, `2_activations_ndif`, `6_axes`), `run_judge_405b.ps1` |
| `run-hf275/` | 378 | 558 MB | all ignored | `vectors_275/` 276 `.pt`, `vectors_100/` 100 `.pt`, `axes_hf100.pt`, `published_assistant_axis.pt` | `31_fetch_hf_vectors.py`, `build_label_banks.ps1`, `29_bank_variants.py` |
| `kd/run/` | 4 JSONL | 5.9 MB | tracked | `knowledge/items.jsonl` (1,459 lines), `knowledge/scores.jsonl` (**truncated**, see §4), `factor_probes/kd_knowledge/rows.jsonl` (1,962), `factor_probes/kd_utterance/rows.jsonl` (1,800) | `kd/scripts/build_knowledge_eval.py`, `score_knowledge_ndif.py`, `build_factor_probes.py` |
| `kd/results/fig_data.json` | 1 | 30 KB | tracked | reduced numbers for thesis fig2 | `kd/scripts/export_fig_data.py` |
| `run-cap/export/cap_fig_data.json` | 1 | 30 KB | tracked | reduced numbers for thesis fig5 | `cap/scripts/export_cap_fig.py` |
| `cap/run-sarcasm-natural/` | 3 | 1.9 MB | tracked (2026-09-22) | `rows.jsonl` (600), `figlang2020_reddit.jsonl` (4,400), `baselines.json` | `cap/scripts/build_sarcasm_natural.py` |
| `aa-smoke/`, `aa-smoke-win/` | 22 + 16 | 64 MB | ignored | early pipeline smoke tests (3–6 roles, 4–40 lines each), 2026-09-10/11 | `run_pipeline_ndif.sh`, chain drivers (INFERRED) |
| `logs/` | 10 `.log` | 17 MB | ignored | all from 2026-09-15: `paper16_*` chain logs and `judge_405b*.log` | paper16 drivers |
| `figures/crossgen/` | 251 | 81 MB | tracked | 120 figure stems (PNG+PDF), 8 CSV, 3 `INDEX.md` | `28_crossgen_figures.py` |
| `figures/persona-coords/` | 26 | 2.8 MB | tracked | 13 figures (q1…q11, significance, transfer, validate) | `19_figures.py` via `plot_style.py` |
| `figures/thesis/` | 10 | 1.2 MB | tracked | 5 figures fig1–fig5 | `kd/scripts/make_thesis_figures.py` |

Inputs (not results) for this work: `kd/personas` (38 JSON), `kd/questions` (1 JSONL), `cap/personas`
(49 JSON), `cap/questions` (4 JSONL), `src/assistant_axis_ndif/pc_instructions` (26 JSON) and the
`pc_questions_*.jsonl` / `*role_groups*.json` files beside the scripts.

### B. Original Liars' Bench benchmark — about 4.3 GB

| folder | files | size | notes |
|---|---:|---:|---|
| `results/datasets/` | 318 | 576 MB | rollouts for convincing_game, gender_secret, harm_pressure (wmdp bio/chem/cyber), insider_trading, instructed-deception |
| `results/blackbox/` | 299 | 3,085 MB | 8 detector folders: ext_judge (lie, deception, deception_2, incorrectness), self_eval (lie, deception), self_judge_lie, pacchiardi (evaluation + training, 4 `.pkl`) |
| `results/{external_evaluation,self_evaluate}_intermediate/` | 34 + 37 | 378 MB | resumable caches per dataset |
| `results/subsets/`, `results/subsets_validation/` | 10 + 4 | 173 MB | normalised benchmark CSVs |
| `results/probes/` | 7 | 0.07 MB | 3 summary CSVs + Llama-3.1-8B layer-15/31 evals |
| `data/` | 281 | 89 MB | raw inputs + `data/probes/{GoT,RepEng}` probe evals (3 models × 5 datasets; 171 PNG) |
| other `figures/*` | 52 | 1.3 MB | barcharts, heatmaps, violin, radar, examples, figure_1, harm-pressure |

All tracked; git dates 2025-02 … 2026-04 (upstream authors Kieron Kretschmar, Walter Laurito; VERIFIED
in `git log`). File mtimes all read 2026-09-09 01:31 = checkout time, so mtimes carry no information
here. Producing code is mostly notebooks (`src/**/*.ipynb`) plus `src/visualization/visualization.py`
(writes `figures/barcharts`, `figures/radarcharts`, `results/blackbox` summaries) — not mapped in
detail, since the audit's focus is the persona-coordinates work.

### C. Vendored probe code `src/probes/deception-detection/` — 461 MB, tracked

On `asix` this holds Apollo's upstream `example_results/` (5 runs, 25 files incl. 5 `detector.pt`,
201 MB) and `data/` (rollouts etc., 260 MB). Git history on asix: 2 commits, 2026-07-11/12. The
collaborator's added results live on his branch (other agent).

## 3. Rough counts

- **Distinct local persona-coordinates outputs with an identifiable producing script: ~26 folders**
  (the rows of table A, counting each crossgen bank/pass separately and the three figure sets).
  Inside `run-coords/` those JSONs come from ~15 different analysis scripts (Q1–Q11), so at the
  "one script + one question" grain it is **~40 result units**. INFERRED: script↔folder links are
  from path patterns and docstrings, not from provenance recorded in the files — the `run-coords`
  JSONs record layer/pooling/bank but no script name, arguments or git hash (VERIFIED for all 33;
  only `q5_semantics_L44_last.json` records a `source`).
- Original benchmark: ~20 output folders (5 dataset families, 8 black-box detector folders,
  2 intermediate caches, subsets, validation, probes, data/probes).
- Scripts mapped: 187 (101 `.py`, 77 `.ps1`, 9 `.sh`): `src/assistant_axis_ndif` 92 (+3 tests),
  `kd/scripts` 42, `cap/scripts` 30, `scripts` 9, `scale/scripts` 8, `src/visualization` 1,
  `aa-smoke-win` 2 (never committed). 31 distinct Windows Scheduled Task names are registered by the
  `register_*` scripts (e.g. `KD-Crossgen`, `Crossgen-Full-$Pass`, `Scale-3b`, `Cap-Layers`).
- 135 of 187 scripts have exactly one commit at their current path; `kd/scripts` and `cap/scripts`
  were nearly all committed on 2026-09-21. **Git dates here are commit-batch dates, not run dates.**
- **23 analysis scripts write no file at all — they only print** (`kd/scripts`: ceiling_null,
  check_length_claim, check_matched_slice, inspect_store, layer_profile, layer_significance,
  matched_profile, paper_comparison, read_significance, show_denial, verify_s23; `cap/scripts`:
  cap_layer_profile, cap_significance, cap_sweep, cap_vs_raw_full, cap_zeroshot, check_basis,
  read_cap_result, show_rollouts; `scale/scripts`: response_lengths, score_summary;
  `src/assistant_axis_ndif`: 30b_premise_boot, 32_verify_new_passes). Any number they produced exists
  only in a console log or in prose; Phase 2 would have to re-run them.

## 4. Oddities worth a red flag for later checking (facts, not verdicts)

1. **`kd/run/knowledge/scores.jsonl` is truncated and committed that way.** VERIFIED: it is the only
   file of exactly 204,800 bytes in the tree (CRLF checkout; the committed blob is 203,705 bytes LF,
   commit 3dfdb67, 2026-09-21); its last record is cut mid-JSON (`{"item_id": "kn00876", "gold": 0,
   "subset": "cais/`). 1,095 complete records vs 1,459 lines in `items.jsonl`. Consistent with the
   "scp truncates at 200 KiB" note (DOC-ONLY until the lab-PC original is compared).
   `gen_hpkr_denial.py` reads this file by default. No git blob of size 204,800 exists on any ref.
2. **Backspace byte inside a gate path.** VERIFIED: `src/assistant_axis_ndif/run_cap_ctrl_layers.ps1`
   lines 31–32 read `"run-crossgen\kd11<0x08>anks\cap_ctrl\matrices\mean_L*_s42.json"` (a `\b` turned
   into a backspace). These are the only control characters in any tracked source. INFERRED: that
   `Get-ChildItem` cannot match, so the gate would log `measured=0 … GATE FAILED` whatever the run
   produced; the lab-PC `logs/cap_ctrl_layers.status` would confirm.
3. **Thesis fig2's data is not the local file it names.** CONFLICT: `kd/results/fig_data.json`
   records `source = run-crossgen/kd11/banks/kd_s23/significance.json`, 21 layers, `n_tests = 3276`;
   the local file at that path has 468 tests over 3 layers (20/40/44). No local `significance.json`
   has 3,276 tests (counts: 96, 192, 468, 576, 4032).
4. **Thesis fig1, fig3, fig4 are drawn from numbers typed into the script.** VERIFIED:
   `kd/scripts/make_thesis_figures.py` hard-codes them (fig1 lines 70–72, fig3 lines 152–154, fig4
   lines 195–197); no data file is read. The raw source must be found elsewhere (lab PC, e.g.
   `run-scale`, which is not on this machine).
5. **The code that built the 11th dataset is missing.** NOT-FOUND: `make_view.py` (named in
   `kd/scripts/run_kd_crossgen.ps1`) and `make_view_ac.py` (called from the lab-PC repo root by
   `kd/scripts/run_passAC_11.ps1`) are not in the working tree and were never committed on any ref.
   Every `run-crossgen/kd11` result uses the `harm-pressure-kr-matched` view they built.
6. **Local copy is partial.** VERIFIED locally: the drivers reference `run-coords/passA|passB|passC`
   (activation stores), `run-kd`, `run-scale`, `run-premise`, `run-cap-ctrl`, `run-paper16-shared`,
   `run-sycophancy(-mt)`, `data_syco`, `run-crossgen/kd11/banks/cap_*`, `run-paper16/axes_vectors_s3_fp32.pt`
   (27_crossgen's default bank) and `run-hf275/axes_hf275*.pt` — none exist here. The drivers name
   ~100 distinct `logs/*` paths (some templated, e.g. `logs/$tag.log`); only 7 of the literal ones exist
   locally, and the local `logs/` holds just 10 files, all 2026-09-15, from the paper16 chain and the
   405B judge. For scale, the lab-PC
   listing made by another agent (`audit/phase0/labpc_sizes.csv`, not re-checked by me) shows
   `run-crossgen` 202,613 files vs 7,150 here, `run-coords` 2,310 files / 77.6 GB vs 44 / 2.8 MB,
   `logs` 243 vs 10.
7. **Layer coverage differs between sibling folders** (VERIFIED counts): `run-crossgen` root and its
   4 banks have L40 only; `kd11` root and `kd11/banks/kd` have 21 layers; `kd11/banks/kd_s23` and
   `kd11/passA` have 3; `kd11/passC` has one matrix.
8. **`remote_sizes.txt` (repo root, tracked, 3dfdb67)** is a 271-entry `path|bytes` listing of some
   remote folder. 269 entries match `run-crossgen/kd11` by name and size (only `summary.json` and
   `significance.json` differ), and it lists just 4 matrices (L40/L44) — INFERRED: a snapshot of an
   earlier state of kd11.
9. **Byte-identical duplicates** (hint only, `p0_local_dupes.py`): 782 md5 groups / 2,105 files among
   the persona result areas — mostly `kd11/csv` vs `kd11/banks/kd/csv` (540 groups; 91 more also equal
   in `kd_s23/csv`) and baseline CSVs repeated across the 4 banks. Between `kd11/csv` and
   `kd11/banks/kd/csv` (2,772 shared names): the 756 `tpr_*` baseline CSVs are identical; the 756
   AUROC baseline CSVs (raw/random/datapca/length) differ — for the one pair diffed
   (`last_L0_datapca_k2_lr_reg10.csv`) only the two appended "zero-shot contrast / PC1" rows differ,
   the 11×11 matrix is identical (VERIFIED for that pair only). INFERRED: each bank appends its own
   axis's zero-shot rows to copied baseline matrices. `run-paper16/scores/` contains one identical pair.
10. Empty directories: `RESULTAXIS/` (created 2026-09-10), `aa-smoke/deep/responses`, `aa-smoke-win/pf`,
    `cap/run`, `kd/figures`, `scale/docs`, `.dist`. Git cannot track empty dirs, so these exist only here.
11. Loose root files: `test.ipynb` (0 bytes, tracked since 47e53c5, 2026-09-12), `aa-smoke.log`
    (472 KB, ignored, 2026-09-10), `remote_sizes.txt` (above), `full paper draft for this experiments/main (1).pdf`
    (the draft, tracked 2026-09-22 in cf11749).
12. Other structural checks came back clean: 0 of 496 `.json` fail to parse; 1 of 74 `.jsonl` has a
    broken last line (item 1); 8 CSVs lack a trailing newline, all in upstream benchmark/vendored data
    (`data/raw`, `results/probes`, `src/probes/.../data`), none in persona results.
13. Timestamps: persona result files carry mtimes 2026-09-15…09-20 that precede their git commits
    (09-16…09-21) — consistent with copying from the lab PC with times preserved (INFERRED). Whole
    trees went in as single commits (all 6,596 `kd11` files in 3dfdb67; `run-crossgen` root in b9a497e).
    `run-coords/*.json` were untracked on 09-16 ("run-coords results live on Drive, not in git",
    f28fc69) and re-added on 09-17 (0c1a224), after 8 of the 13 `figures/persona-coords` figures were
    committed on 09-16 (commit order, not proof the data changed).
14. `src/assistant_axis_ndif/__pycache__` holds CPython-3.10 bytecode for `8_coords_ndif`, `23`, `25`,
    `27`, `28`, `29`, `31`, `32`, `eval_datasets`, `pc_personas`, `plot_style`, `aa_roles` (mtimes
    09-15…09-22) — INFERRED: those modules were imported on this laptop at least once.

## 5. Where the figures come from (`local_figures_provenance.csv`)

| figure set | script | data read | data present locally? |
|---|---|---|---|
| `figures/thesis/fig1_register`, `fig3_baseline_artifact`, `fig4_geometry_by_scale` | `kd/scripts/make_thesis_figures.py` | none (**hard-coded values**) | n/a — IMAGE/SCRIPT-ONLY until the raw source is found |
| `figures/thesis/fig2_layer_profile` | same | `kd/results/fig_data.json` (exported by `kd/scripts/export_fig_data.py` from the kd_s23 `significance.json`) | fig_data.json yes; **its recorded source version NOT local** (item 3) |
| `figures/thesis/fig5_capitulation` | same | `run-cap/export/cap_fig_data.json` (exported by `cap/scripts/export_cap_fig.py` from `run-crossgen/kd11/banks/cap_*` + `run-cap/axes_s23.pt`) | export yes; its upstream `cap_*` banks and `run-cap/axes_s23.pt` NOT local |
| `figures/persona-coords/*` (13) | `src/assistant_axis_ndif/19_figures.py` → `plot_style.save` | one or more `run-coords/*.json` per figure (e.g. `q6_layers` ← `run-coords/q6_layers.json`; `q5_semantics_layers` ← 8 `run-coords/q5_layers/p1_L*.json`) | all 13: yes |
| `figures/crossgen/*` (68 stems) | `src/assistant_axis_ndif/28_crossgen_figures.py` | `run-crossgen/matrices/*_L40_s42.json`, `banks/*/matrices`, `summary.json`, `significance.json` | all: yes (L40 only) |
| `figures/crossgen/kd11/*` (32) and `kd11/passA/*` (20) | same, with `--run run-crossgen/kd11[/passA]` (INFERRED from the output path) | the matching files under `run-crossgen/kd11[/passA]` | all: yes |

Presence of an input file is not proof it is the version the figure was drawn from; every figure
set's inputs were committed in the same or a later commit than the figure (git dates only).

## 6. Hints for Phase 1 (not done here)

- The local `run-*` folders are partial copies; the lab PC / Drive listings (other agents) are the
  inventory for most runs. Use `scripts_to_outputs.csv` `out_paths_INFERRED` / `in_paths_INFERRED` to
  link each lab-PC folder to its script and driver.
- The logs that would date and parameterise each run (`logs/*.status`, `*.log`) are almost all on the
  lab PC only.
- Items 1–5 above touch, respectively: an input of the kd denial generation, a logged gate, the
  data behind thesis fig1–fig4, and the dataset view every kd11 number depends on. Each needs the lab-PC
  original (file, log or script) to resolve.
