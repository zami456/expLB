# Phase 1 — experiment inventory and file-level anomalies (instructions for inventory agents)

Read `audit/BRIEF.md` (fully), then `audit/PROGRESS.md` (the RED FLAGS and the "F-mapping results" section), then this file.
Phase 1 = facts only: no verdicts on whether results are right.
Safety rules are the same as in `audit/phase1/FMAP_INSTRUCTIONS.md` ("Safety"). Read that section.

In short:
- write only your own outputs;
- git and the lab PC are read-only;
- lab python runs with the working directory set to `C:\Users\user4\audit_scratch` and `$env:PYTHONDONTWRITEBYTECODE='1'`, and never imports the owner's modules;
- laptop RAM is limited: stream the gz census files with `zcat | awk`.

## Inputs (all already built and md5-verified)
- `audit/phase1/labpc/joined_labpc.tsv.gz`: every lab file, with its identical copies on Drive / asix / his branch, and same-name-different-content in asix. Also in that folder: `labpc_census.tsv`, `drive_census.tsv.gz`, `drive_not_on_lab.tsv.gz`, `asix_results_not_on_lab.tsv.gz`, `join_summary.txt`.
- `audit/phase1/labpc/scheduled_tasks.tsv`: 51 team scheduled tasks (name, state, last run, result, action script + args).
- `audit/phase1/labpc/log_index.tsv`: 242 lab logs (first/last internal timestamp, lines, fail/done markers, scripts named, first/last line).
- `audit/phase1/fmap_all.csv`: 109 F-numbers → resolved result file, key, producing script, run log, depends_on, copy status, notes.
  - Per-batch notes are in `audit/phase1/fmap/*.md`.
- `audit/phase0/scripts_to_outputs.csv` (187 repo scripts → outputs/inputs), `local_results_index.csv`, `pcood_results_index.csv` (his 19 result folders + 3 input folders), `branch_compare.md`, `docs_inventory.md`, `draft_report.md`.
- `audit/phase1/census_asix_git.tsv`, `census_pcood.tsv`, `census_laptop_ignored.tsv`.
- Lab read-only tools:
  - `python audit/scripts/labpc_run.py audit/scripts/p1_fmap_F028-F054_runpy.ps1 ...`, a UTF-8, scratch-cwd, no-bytecode runner (read its header for usage);
  - `audit/scripts/p1_fmap_F082-F109_sgrep.ps1` for UTF-16 logs;
  - `p1_peek.py` modes (see its docstring).

## Task A — `audit/experiments.csv` (the brief's Phase 1 table)
One row per **experiment/run**: one producing script or driver invocation family → one coherent output set. Examples: "kd11 crossgen, bank kd_s23, 21 layers × 2 poolings"; "premise/pooled-training layer sweep, seed 42"; "his persona_ood_ext_L40"; "scale 3B coords extraction". Expect roughly 60–150 rows. Cover BOTH branches (asix/lab/Drive work, and his `origin/feat/persona-pc-ood` results). Treat the original upstream Liars' Bench results under `results/` as ONE background row.

Columns:
`exp_id, what_it_tests, who, when_first, when_last, when_source, branch, commit, script, config_or_args, scheduled_task, results_location, n_result_files, status, status_evidence, depends_on, f_ids, notes`

- `what_it_tests`: from the script, config or docstring, one plain sentence.
- `who`: owner / collaborator / both, from git authorship and location.
- `when_first` / `when_last` / `when_source`: prefer internal log timestamps (log_index), then git commit dates. Say which in `when_source`; file mtimes are the last resort, since copies reset them and the 09-22 wave rewrote many.
- `commit`: the commit that added the script, and the latest commit that changed it.
- `results_location`: lab path(s), Drive path(s), git path(s).
- `status`: `complete`, `partial`, `crashed`, `superseded` or `unknown`. `status_evidence`: a .done/.status line, a log's last line, file counts vs the expected grid.
- `depends_on`: which inputs, and whose, with the version where known. Examples: `run-kd/vectors_s3` (md5-identical to his bank), paper16 old vs `_fp32` vectors (RF-12), the 11th dataset from lab-only `make_view.py`.
- `f_ids`: the F-numbers whose evidence comes from this experiment (from `fmap_all.csv`).
- `notes`: RF links (RF-10 no backup, RF-13 regenerated 09-22, RF-14 guard bug), anything odd.

Also write `audit/phase1/experiments_notes.md`: counts by status and by who, experiments with no F-number and no doc mention (candidate orphans; pass them to Task B's list), and experiments whose results were superseded or regenerated.

## Task B — `audit/phase1/anomalies.md` + `audit/phase1/anomalies.csv`
Four lists, as the brief asks:

1. **Orphans:** result folders or files that no doc mentions. "Doc" means `kd/docs/FINDINGS.md`, `docs/*.md`, `docs/thesis/*`, `cap/docs/*`, `kd/docs/*`, the memory dir `C:\Users\SURAIYA\.claude\projects\C--Users-SURAIYA-Documents-STUDY-thesis-expLB\memory\`, his branch's `summary.md` files and the draft (`audit/phase0/draft_text.txt`). Work at folder level (2–3 path levels), not per file. Flag orphans that look like substantial experiments (many files, a complete grid).
2. **Ghosts:** results a doc mentions that cannot be found in ANY source (lab, Drive, asix, his branch, laptop). Start from the fmap NOT-FOUND and PARTIAL notes (e.g. F-04's 384-test file, F-34's 3,276-test file, F-57's and F-67's significance files, F-21's smoke3b, F-44), then scan the other docs for cited paths (`git grep` for `run-`, `.json`, `.jsonl` paths in docs) and check each against the census.
3. **Duplicates:** the same md5 in several places. Summarise:
   - expected layout duplicates, e.g. lab ↔ Drive backups and crossgen per-cell CSVs;
   - surprising ones: the same content under different experiment names, and his files = the owner's files (RF-1, already known).
   Give counts, and name only the surprising ones.
4. **Conflicting versions:** the same logical file with different content across sources. Include:
   - the RF-11 list;
   - Drive-only older versions (`drive_not_on_lab.tsv.gz`);
   - laptop vs lab vectors (RF-12);
   - fmap `LAB≠GIT` rows.
   For each, say which version each F-number used, if the fmap tells you.

CSV columns: `category, path_or_group, sources, detail, related_f_ids, severity_hint` (HIGH = a headline number depends on it; MED; LOW).

Return a ≤10-line summary to the lead.
