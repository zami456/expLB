# Phase 1 — mapping each F-number to its evidence (instructions for mapping agents)

Read `audit/BRIEF.md` fully, then the RED FLAGS in `audit/PROGRESS.md`, then this file.

## Goal
For each F-number in your batch, find **where its numbers come from**: the result file (and the key or row inside it), the script that wrote that file, the run log, and which copy of the file is the right one.

This is **Phase 1 = mapping, not verification.** Do not decide whether a finding is true. Phase 2 will recompute every number with scripts. If you happen to see the value at the key while locating it, note it in `value_seen_prelim`, marked preliminary.

The owner's goal: every F-number ends up tied to the exact file (path + key/row) its number came from, so the collaborator's draft can cite correct numbers. The F-findings come from real runs. Your job is to find those runs' files.

## Safety (from the brief; strict)
- **Write ONLY** your two output files under `audit/phase1/fmap/` (plus helper scripts `audit/scripts/p1_fmap_<batch>_*.py` if you need them).
- Never edit anything else. Git is read-only (`git show`, `git log`, `git grep`); no checkout, stash, reset or commit. No rclone.
- **Lab PC:** only through the read-only tools below. Never write, move or delete there. Never run the owner's scripts.
- **Laptop RAM is limited.** Never load the census files fully into pandas or memory. Stream them with `zcat file | awk -F'\t' ...`, or `grep`. Keep python memory small. Don't run broad `grep -r` over the working tree (it's huge): use `git grep` for tracked files.
- Don't save anything to Claude's auto memory.
- Tag statements VERIFIED / DOC-ONLY / CONFLICT / NOT-FOUND / INFERRED (with the reason).

## Resources
- The ledger:
  - `kd/docs/FINDINGS.md` (HEAD). Your F-sections' line ranges are in `audit/phase0/fnumber_history.csv` (which also lists in-place edits and "corrects/corrected-by" links).
  - The git history of a section: `git log -L <start>,<end>:kd/docs/FINDINGS.md` or `git log -p -- kd/docs/FINDINGS.md`.
- **Census (all md5 VERIFIED)**, in `audit/phase1/labpc/`:
  - `joined_labpc.tsv.gz`: one row per lab-PC file. Columns: `root relpath bytes mtime md5 md5_lf on_drive drive_paths in_asix asix_paths in_pcood pcood_paths asix_same_name_diff_content drive_same_name_diff_content`.
    - The lab root for project files is `C:\Users\user4\expLB`, so the full path is `C:/Users/user4/expLB/<relpath>`.
    - `md5_lf` = md5 after CRLF→LF, which is what git stores.
    - Example: `zcat audit/phase1/labpc/joined_labpc.tsv.gz | awk -F'\t' '$2 ~ /^run-premise\//'`
  - `labpc_census.tsv` (also has CreationTime), `drive_census.tsv.gz` (Drive paths are under `tamara:expLB/`), `drive_not_on_lab.tsv.gz`.
  - In `audit/phase1/`: `census_asix_git.tsv`, `census_pcood.tsv`, `census_laptop_ignored.tsv`.
- **Phase 0 maps** in `audit/phase0/`:
  - `scripts_to_outputs.csv` (187 repo scripts → output/input paths, print-only flag);
  - `local_results_index.csv`, `docs_inventory.md`;
  - `draft_vs_ledger_preview.md` (draft topics 1–8), `branch_compare.md` (collaborator's branch).
- **Lab PC read-only tools**, run from the repo root. Paths use forward slashes, no spaces:
  - `python audit/scripts/labpc_run.py audit/scripts/p1_labpc_runpy.ps1 FILE=p1_peek.py "ARGS=<path> <mode> [arg]"`
    - Modes: `info`, `head [N]`, `cat` (≤400 KB), `keys [depth]`, `get <dotted.path>`, `grep <regex>`, `ls`, `pt`. See the docstring of `audit/scripts/p1_peek.py`.
  - Lab logs are in `C:/Users/user4/expLB/logs/` (242 files). Loose analysis scripts are in `C:/Users/user4/expLB/*.py` (~130, many not in git). Drivers are in `C:/Users/user4/*.ps1`.
  - Example: `... "ARGS=C:/Users/user4/expLB/logs ls"`, then `... "ARGS=C:/Users/user4/expLB/logs/kd_chain.log grep F-3|AUROC"`.
- Known facts you should use (see PROGRESS.md red flags):
  - Git and lab copies of several result JSONs DIFFER (RF-11).
  - Two versions of the paper16 persona vectors exist (RF-12).
  - Several lab folders are not on Drive (RF-10).
  - The lab PC is the owner's authoritative store: prefer the lab copy, and record whether the git or Drive copy is identical.

## Output 1: `audit/phase1/fmap/fmap_<batch>.csv`, one row per F-number
Columns, in this order:
`f_id, title_short, heading_date, claim_type, key_numbers, cited_paths, resolved_primary, key_path, value_seen_prelim, other_copies, copy_status, producing_script, script_writes_file, run_log, depends_on, supersedes, superseded_by, draft_topic, mapping_status, notes`

- `claim_type`: `scientific` (a measured result about models/data), `engineering` (pipeline, throughput or tooling note), `provenance` (where something came from), or `correction` (mainly corrects another F).
- `key_numbers`: the section's headline numbers as stated, `label=value; label=value` (max ~6).
- `cited_paths`: files and scripts the section itself names, verbatim.
- `resolved_primary`: the file the key numbers come from, as a lab path `C:/Users/user4/expLB/...`, or a git path if lab-less.
- `key_path`: the JSON key or CSV row/column where the number lives.
- `other_copies`: git/Drive/laptop copies, each with `SAME` or `DIFFERENT` (use md5 / md5_lf).
- `copy_status`: one of `LAB=GIT`, `LAB≠GIT`, `LAB-ONLY`, `GIT-ONLY`, `DRIVE-ONLY`, `NOT-FOUND`.
- `producing_script`: repo path and/or lab loose-script path. Say whether the lab copy of the script equals the repo copy (md5_lf).
- `script_writes_file`: `y`, or `n` = print-only (then the number exists only in a console log or prose; say where, if you find it).
- `run_log`: lab log file + time range, if identifiable.
- `depends_on`: key inputs (e.g. `run-kd/vectors_s3`, `paper16 vectors_s3_fp32`, the 11th dataset from `make_view.py`), with versions if known.
- `supersedes` / `superseded_by`: F-ids, from the text and from `fnumber_history.csv` (DOC-ONLY).
- `draft_topic`: 1–8 if the F-number bears on a topic in `draft_vs_ledger_preview.md`.
- `mapping_status`: `MAPPED` (file + key found), `PARTIAL`, or `NOT-FOUND`.

## Output 2: `audit/phase1/fmap/fmap_<batch>.md`
A short summary of your batch:
- counts by `mapping_status` and `copy_status`;
- F-numbers whose evidence is print-only or missing;
- any CONFLICT you noticed (e.g. a key file whose git and lab versions differ, a finding built on a later-replaced input, a number that appears nowhere);
- the exact commands you used, so they can be re-run.

Return a ≤10-line summary to the lead auditor.
