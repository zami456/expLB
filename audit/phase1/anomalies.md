# Phase 1, Task B: orphans, ghosts, duplicates and conflicting versions

Written 2026-09-22 by the Task B inventory agent. **Facts, not verdicts.**
The table is `audit/phase1/anomalies.csv` (65 rows; columns `category, path_or_group, sources, detail, related_f_ids, severity_hint`). Row ids: OR = orphan, GH = ghost, DU = duplicate, CV = conflicting version. Each id is the first word of the row's `detail`.

Tags as in the brief: VERIFIED (checked in a file or by a script during this audit), DOC-ONLY, INFERRED (reason given), NOT-FOUND, CONFLICT.
Severity: HIGH = a headline number depends on it; MED; LOW.

## What the four lists say, in short

1. **Orphans.** Four complete cross-generalisation grids have no reported result: the root `run-crossgen/passA` grid (92,184 files), `passC` (6,936), `passB_fine` and `passP` (5,777 together). `passP` is the prompt-side construct-validity control that the thesis doc planned. The fp32 paper16 bank behind every kd11 headline matrix is **named in no doc**; the docs name only the older files (HIGH). The length-control analyses behind a memory rule have no doc naming them. Several lab-only files add to RF-10.
2. **Ghosts.** 532 file or folder paths are cited in the doc set. 496 of them are found. Of the 36 that are not, 16 are real ghosts and 20 are text artefacts or files outside the project. The important ghosts are *versions* that were overwritten, not paths that never existed. The worst is the 3,276-test `kd_s23/significance.json` behind F-34 and thesis fig2 (HIGH). Another 42 F-entries' numbers were only printed to a console.
3. **Duplicates.** 144,971 contents occur more than once (494,804 files). Nearly all copies are the expected lab ↔ Drive ↔ git layout. One is surprising: `run-crossgen/kd11/passA` and `kd11/passC` hold significance files byte-identical to the older root `passA`/`passC` ones. The 09-20 "kd11" grids ran on 10 datasets, not 11. Also, 11 own100 roles have identical score files because every one of their 1,200 rollouts was scored 3.
4. **Conflicting versions.** A path-mapped comparison of all 208,830 multi-copy files finds 50 content conflicts. Of these, 26 are the RF-11 result files and 13 are line-ending only. The two HIGH ones are:
   - the paper16 old-vs-fp32 vectors (RF-12), with a per-F list of which version each used;
   - the 70B scale rung's bank. Its 09-22 matrices carry kd-bank-only directions, so the "paper16_70b" rung is the 36-persona kd bank, while the 3B/8B rungs use paper16 banks (F-63, F-96, F-100–F-102).

## Inputs and method (all re-runnable, read-only)

| script | what it does | output |
|---|---|---|
| `audit/scripts/p1_anom_common.py` | shared helpers: map any source path to one lab-style path (e.g. Drive `kd/crossgen11/` → `run-crossgen/kd11/`), assign files to result units, load the doc set | — |
| `p1_anom_folders.py` | folder aggregates of all five census sources | `anomalies_detail/folders.tsv` (1,329 folders) |
| `p1_anom_units.py` | groups result files into 403 units; checks each against the doc set at PATH / TAIL (last 2–3 segments) / NAME level; also against fmap and a secondary corpus (CLAUDE.md, READMEs, the plan file, all scripts on both branches) | `anomalies_detail/units.tsv` |
| `p1_anom_logs.py` | which of the 242 lab logs a doc names | `anomalies_detail/logs_mentions.tsv` |
| `p1_anom_ghosts.py` | extracts every cited path from the doc set (backtick spans + bare paths; braces, `*`, `<…>`, `\|` expanded) and matches it by path suffix against all sources | `anomalies_detail/cited_paths.tsv` (532 tokens) |
| `p1_anom_grepdocs.py` | ad hoc grep of the doc set | console |
| `p1_anom_dupes_lab.py` | **runs on the lab PC** (laptop RAM); groups contents by LF-normalised md5 across lab, Drive, asix, his branch, laptop | `anomalies_detail/duplicates_lab_output.txt` |
| `p1_anom_conflicts_lab.py` | **runs on the lab PC**; same logical path, different content, across all sources | `anomalies_detail/conflicts_lab_output.txt` |
| `p1_anom_snip_{sig,bank,feat,rows,scores}.py`, `p1_anom_labgrep.ps1` | small read-only lab reads (JSON headers, test counts, row counts, loose-script grep) | quoted below |
| `p1_anom_write.py` | writes `anomalies.csv` | `anomalies.csv` |

- **The doc set** is the one the instructions define: 28 asix docs (`kd/docs/*`, `cap/docs/*`, `docs/*.md`, `docs/thesis/*`), the 45 memory notes, his 15 `summary.md` files and `audit/phase0/draft_text.txt`, 89 texts in all. Text is lower-cased, backslashes become `/`, and brace forms are expanded. A his-branch folder does not count as mentioned by its own summary.
- **Laptop RAM.** The laptop had about 0.5 GB free. Every census file was streamed, and the two whole-census groupings ran on the lab PC with the lab's `.venv-analysis` python.
- **Lab PC writes.** Four files were uploaded to `C:\Users\user4\audit_scratch\{scripts,census}\` with `labpc_upload.py`, which logs each one in `audit/LABPC_WRITES.md`: `p1_anom_common.py`, `p1_anom_dupes_lab.py`, `p1_anom_conflicts_lab.py`, `census_laptop_ignored.tsv`. The lab-side scripts read only the audit's census files and print to stdout. The snippets only open owner JSON/JSONL files for reading. No owner module was imported.

## 1. Orphans (results that no doc names)

**Counts (VERIFIED, units.tsv).** 205 non-log result units: 86 are named by full path, 27 by the last 2–3 path segments, 48 only by their bare name, and 44 by nothing. For 242 lab logs: 39 are named in a doc, 43 more only in fmap, and 160 nowhere (`logs_mentions.tsv`; most are sync/supervisor/infrastructure logs).
"Named" is not the same as "reported". The substantial orphans below are folders whose results no doc reports, even where a doc names the folder.

| id | what | size / state | why it matters | sev |
|---|---|---|---|---|
| OR-01 | root `run-crossgen/passA` + 7 banks: 80 layers × 2 poolings × 6 seeds | 92,184 files, complete, 09-18/19 originals (the 09-22 re-run exited 1 on the HP-KR split change) | the largest result set in the project; F-91 lists its coverage, nothing reports a number | MED |
| OR-02 | root `run-crossgen/passC` + 7 banks, 6 layers × 6 seeds | 6,936, complete | same | MED |
| OR-03 | `run-crossgen/passB_fine` (layers 37–47) + `passP` (prompt-side) | 5,777, complete | the thesis doc (09-17, §15.5) planned passP as the construct-validity control of §11.7; no later doc reports it | MED |
| OR-04 | root + banks 21-layer × 6-seed sweep, incl. label banks hf275_loose/strict and own100_strict | complete | reported at L40 only (F-17); no transfer result for the label banks. FINDINGS calls the store `passB/banks/…`, but the path is `run-crossgen/banks/` | MED |
| OR-05 | `run-coords/{confounds_L44_last, p1_L44_last_lengthctl, q5_semantics_lengthctl}.json` | 3 JSON, lab + Drive, not git; lab-only drivers `run_confounds.ps1`, `run_lengthctl.ps1` | memory `response-length-transfers-on-its-own` (length-only 0.553) matches them by layer, pooling, probe and time (edited 1 min after the run ended): INFERRED source | MED |
| OR-09 | `run-paper16/axes_vectors_{s3,all}_fp32.pt` + `vectors_{s3,all}_fp32/` | lab + Drive | the bank of every kd11 root and run-crossgen root matrix; no doc names it, and the docs point to the pre-fp32 files | **HIGH** |
| OR-11 | `run/vectors_fp32/` 1 of 100 | abandoned 09-16 rebuild | own100 banks stayed on bf16-reduction vectors (CV-11) | MED |
| OR-17 | his `loo_L20/L40`, `persona_ood_mod_L*`, `persona_ood(_L40)` | his branch | named only in his own summaries; the draft omits them (loo = the baseline it says it "did not compare"; mod = his negative result) | MED |

LOW rows (details in the CSV):
- OR-06: 09-15 pilot JSONs on the old bank.
- OR-07: smoke outputs.
- OR-08: the `expLBactivationLAYER22` cache, 361 MB, lab-only, provenance NOT-FOUND.
- OR-10: other unnamed bank files.
- OR-12: `data_syco`.
- OR-13: `isarc_a.parquet`.
- OR-14: `cap/run-sarcasm-natural`.
- OR-15: `cap_fig_data.json`.
- OR-16: his 07-12 run.
- OR-18: git/laptop-only results.
- OR-19: unnamed logs.

**Cross-check with Task A's 14 candidate orphans** (`experiments_notes.md` §3):
- **Agree:** O44 = OR-01, O45 = OR-02, O46 = OR-03, O26 = OR-05, O16 ⊂ OR-06, O39/O01 ⊂ OR-07, O87 = OR-08, O05 = OR-11, C01 = OR-16.
- **Not listed separately:** O10 (the sycophancy rows added to passA/passC). It is an input step inside the named passA/passC stores.
- **C14/C19 (his L44 folders):** I agree they are not orphans. The draft uses his L44, followup and positions folders (draft_report §8).
- **C20 (INLP)** is a ghost, GH-20.
- **Task A's "46 prose-only rows"** (the Q1–Q11 series) are named in `docs/thesis/persona-coordinates-70b-results.md` (appendix table) or in memory. They are therefore not orphans here. OR-06 lists the ones that even that appendix omits.

**Additions to RF-10 (data safety; VERIFIED census `on_drive=0`).** These files exist only on the lab PC:
- `run-paper16/axes_vectors_s3_fp32_famcentred.pt`;
- `run/axes_strict.pt`;
- `run-cap/axes_all.pt` (inside run-cap, already RF-10);
- `data_syco/`;
- `isarc_a.parquet`;
- `C:\Users\user4\expLBactivationLAYER22\`;
- lab `smoke/`.

Only on the laptop and in git: `run-paper16/scores_405b/` and `logs/judge_405b*.log`.

**Reassurance:** the 75 unhashed >200 MB `run-coords/passC` files each have a Drive file at the same path with the same byte size (INFERRED identical; `duplicates_lab_output.txt` section D). This resolves the passC doubt in RF-10.

## 2. Ghosts (results a doc cites that no source holds)

**Cited paths (VERIFIED, cited_paths.tsv).** 532 path tokens were extracted; 496 were found in at least one source.
The 36 not found split as follows:
- **16 real ghosts:**
  - his `activations/pooled*` and `token_proj_L*` (6 tokens);
  - `persona vectors/vectors_s3/` and `persona knowledge denial modified/`;
  - `XXXX-9-llama-all*/detector.pt` and `*_all_tokens.json`;
  - the draft's `paper/make_numbers.py`;
  - `run-scale/smoke3b`;
  - three F-106 sarcasm scratch/raw files;
  - memory's `hf_axis_check.py`.
- **20 artefacts or external files:**
  - bare extensions (`.py`, `.pt`, …);
  - nnsight library paths;
  - Hugging Face files (`assistant_axis.pt`, `default_vector.pt`, `tokenizer.json`);
  - another user's `main.py` and `run_benchmark_*.py`;
  - a role-group label;
  - the `-L22/-L40.yaml` configs, which exist on `D:\expLB-acts`;
  - `FULL_L20/40.json`, which is shorthand for two existing files;
  - FINDINGS.md:2029's paired brace `mean_L{16,16,40}`, whose intended files exist.

**Version ghosts and unsaved numbers (from fmap, re-checked against the census).**

| id | ghost | F | sev |
|---|---|---|---|
| GH-01 | the 3,276-test 21-layer `kd11/banks/kd_s23/significance.json` of 09-20: F-34's q-values and "9 of 3,276", and the source named in `kd/results/fig_data.json` (thesis fig2). Git has 468 tests, lab/Drive 5,544 | F-34, F-57 | **HIGH** |
| GH-02 | F-04's 384-test `kd11/significance.json` (09-19 run 1) | F-04 | MED |
| GH-03 | F-67's 792-test `cap_{s23,all}/significance.json` | F-67 | MED |
| GH-04 | gate lines in 8 `.status` files, truncated by the 09-22 re-run (RF-14) | F-72, F-75, F-77, F-81–F-83, F-87, F-94, F-95, F-104 | MED |
| GH-05 | numbers only printed: 42 of 109 F-entries have `script_writes_file = n` | many | MED |
| GH-06 | F-17's random-2-d numbers (+0.056 / −0.44 / −0.76) | F-17, F-74 | MED |
| GH-07 | F-107's projection outputs (inputs survive) | F-107 | MED |
| GH-17/18/19 | his activations, uncommitted banks, base-probe `detector.pt`/`results_table.csv`; the draft's Fig 2/3 data, `make_numbers.py`, LaTeX source | draft | MED |

The LOW rows are GH-08 to GH-16, GH-20 and GH-21:
- GH-08: F-58's overwritten HPC matrices.
- GH-09: smoke3b.
- GH-10: deleted scale artefacts.
- GH-11: cap bank v1 and the F-32 texts.
- GH-12: the F-36 draft.
- GH-13: F-20's 98-row state.
- GH-14: small F-14/15/24/26/33 items.
- GH-15: the F-106 inputs.
- GH-16: missing hand-run logs.
- GH-20: INLP.
- GH-21: `hf_axis_check.py`.

**Resolved, not ghosts (GH-22, VERIFIED census):**
- the exporter of his `paper16_L20.pt` / `deceptive_L20.pt` is the lab loose pair `C:\Users\user4\build_share_L20.py` / `build_deceptive_L20.py` (confirms Task A);
- `make_view*.py` are lab loose files (RF-8);
- the acts-cache yaml configs are on `D:\expLB-acts`.

## 3. Duplicates (same md5 in several places)

**Counts (VERIFIED, `duplicates_lab_output.txt`).**
- **Files per source:** lab 210,464; Drive 277,407; asix 9,527; his branch 846; laptop-ignored 486.
- **Scale of duplication:** 148,820 distinct contents, of which 144,971 occur more than once (494,804 files).
- **Expected layout (DU-01, DU-02).** 124,682 groups are one logical file copied:
  - lab + Drive: 86,046;
  - lab + two Drive copies: 32,815. Drive keeps `run-crossgen/kd11` twice (as `crossgen/kd11` and `kd/crossgen11`) and own100 twice (as `own100/` and `own100/run/`);
  - lab + git + Drive: 5,131;
  - git + Drive only (figures): 242;
  - Drive + laptop only: 52. These are exactly the 52 old paper16 vectors (RF-12).
- **Different paths, same content:** 19,408 groups (199,402 files) are crossgen per-cell CSVs.
  - Each such CSV is the train-one-test-all grid plus two bank-specific zero-shot rows. VERIFIED by diffing `banks/hf100` against `banks/own100` `mean_L40_raw_paper.csv`: only the two zero-shot rows differ.
  - So identical CSVs across banks are expected.

**Surprising ones (named):**
- **DU-03 (MED).** Two pairs of significance files are byte-identical although written a day apart:
  - `run-crossgen/kd11/passA/significance.json` = `run-crossgen/passA/significance.json` (7,368,815 B; 09-20 14:10 vs 09-19 14:18);
  - `kd11/passC/significance.json` = `passC/significance.json` (551,848 B).
  - The kd11/passA `summary.json` lists **10 datasets**, with no harm-pressure-kr-matched.
  - INFERRED: the 09-20 "kd11" passA/passC grids ran without the 11th dataset and reproduced the root grids. The 11-dataset runs are `passA11`/`passC11`. The folder name is misleading.
- **DU-06 (LOW, but worth a Phase 2 look).** 11 own100 roles have byte-identical `run/scores/*.json`, because all 1,200 of each role's rollouts scored 3 (VERIFIED `p1_anom_snip_scores.py`): bard, eldritch, ghost, hermit, leviathan, oracle, prophet, revenant, tree, void, whale. This looks like a judge ceiling.
- **DU-05 (LOW).** Drive `share/paper16_L20/` carries the paper-question rollouts (`run-paper16`) next to two vector files; he received only the shared-question vectors.
- **DU-04 (his = owner's; RF-1, known).** His 36 kd `.pt` = lab `run-kd/vectors_s3`; his `deceptive/` = lab `share_deceptive`; his `paper16_L20_sharedquestions/` = lab `share_L20_shared` = Drive `share/`.
- **Benign by design:**
  - vectors identical across score filters where the filter removed nothing;
  - `kr-matched` chunks linked to the HP-KR chunks;
  - his `zero_shot.png` identical across L20/L40/L44, because it is a three-layer figure (`persona_ood_figures.py:422`);
  - `D:\expLB-acts` = his branch's modified probe code (DU-07).

## 4. Conflicting versions (same logical file, different content)

**Path-mapped comparison (VERIFIED, `conflicts_lab_output.txt`).**
- 208,830 logical files exist in two or more places; 50 of them have differing content:
  - 27 are git vs (lab = Drive): 26 RF-11 result files + FINDINGS.md;
  - 13 are git vs Drive-only copies that differ by CRLF only (CV-15): for 12 the size difference equals the line count; for `role_labels_review.csv` it equals the record count (INFERRED);
  - 7 are asix vs his-branch code;
  - 1 is `cap/docs/DESIGN.md`;
  - 2 are real lab-vs-Drive conflicts (CV-10): Drive keeps an older own100 `axes.pt` (20,979,057 B vs 62,936,618 B) and `DELIVERABLE_READY`.
- **Audit caveat (CV-17).** `join_summary.txt`'s `driveDIFF` column and `joined_labpc.tsv.gz`'s `drive_same_name_diff_content` are **not** conflict counts. They match the same file name in a *different bank* (e.g. `run-kd/responses/pc_handler.jsonl` vs Drive `paper16/responses/pc_handler.jsonl`). Do not use them.

**Which version each F-number used** (from fmap; re-checked):

| logical file | git | lab = Drive | used by |
|---|---|---|---|
| `kd11/banks/kd_s23/significance.json` (CV-03) | 468 tests, 3 layers | 5,544, 21 layers | F-34: **neither** (3,276, GH-01); its profile matches the lab copy |
| `kd11/banks/kd/significance.json` (CV-04) | 4,032 tests | 6,300 | F-31: **git only** |
| `kd11/significance.json` (CV-05) | 4,032 | 4,788 | F-04: neither (384); F-61's "+0.006" matches git |
| root + 4 banks `{summary,significance,splits}.json` (CV-06) | L40 only | all layers × 6 seeds | F-17: same value in both (git is an exact subset; only BH q differs) |
| `kd11/passA/…` (CV-07) | 576 / 468 tests, 3 layers | 80 layers (= root passA) | findings use passA11 instead |
| `kd/run/knowledge/scores.jsonl` (CV-08) | truncated, 1,095 records | 279,757 B | F-08/F-09/F-59 used lab; recomputing from git would be wrong |
| factor-probe `rows.jsonl` (CV-09) | input rows | extraction rows | format only; label counts agree (F-05, F-07, F-16) |

**The two HIGH conflicts:**
- **CV-01, paper16 old vs fp32 vectors (RF-12).**
  - The old vectors are on the laptop and Drive `paper16/run/` (52/52 md5 equal). The lab keeps the old `axes_vectors_s3.pt` and `vectors_s3_old_pc_broker.pt`.
  - All 26 fp32 vectors differ from the old ones.
  - **OLD** was used by: F-18, and all 09-15 run-coords analyses (Q1/Q2/Q4: FULL_L*, probes_*, zeroshot_paper16, …; Task A read their `bank` key).
  - **NEW fp32** was used by: every kd11 root and run-crossgen root matrix (F-03, F-04, F-13, F-17, F-29, F-39, F-58's 70B cell, F-61, F-65, F-74, F-98) and the 09-16+ analyses.
  - The docs name only the old files (OR-09).
- **CV-02, the 70B scale rung's bank.**
  - Matrices store only `bank = axes_vectors_s3.pt`, `bank_name = paper16_70b`. That basename is shared by `run-kd/`, `run-paper16/` and `run-paper16-shared/`.
  - The 09-22 file `mean_L20_s42.json` lists zero-shot directions `knowledge_denial` and `kd_assertion`, which exist only in the kd bank (VERIFIED, `p1_anom_snip_feat.py`). So the 09-22 sweep (F-100–F-102) used the 36-persona kd bank, as F-100's source line says.
  - The 3B/8B rungs use `axes_3b/8b.pt` (paper16_3b/8b).
  - The 09-20 L40 cell behind F-63 cannot be resolved from its file. `logs/crossgen_scale_70b.log` prints no bank path, and the 70B driver is in no branch. F-100 says the sweep reproduced that cell (INFERRED: same kd bank).
  - Phase 2 must settle whether the three-rung comparison mixes banks.

**Other MED conflicts:**
- **CV-11.** own100 banks are on bf16-reduction vectors while paper16 is on fp32. Affects F-90's persona arm and the memory note on paper16 vs own100.
- **CV-12.** The root passA grid (200-row HP-KR) and passA11 (2,139-row HP-KR) rest on different HP-KR row sets. `crossgen_full_A.log`: "split … differs".
- **CV-14.** Kept siblings: `sarc_pilot.status` versions (0.947 pre-e2e vs 0.921 current; F-93/F-104/F-105 quote different ones), `significance.json.stale-576`, `pc_broker.jsonl.predupe` (explains F-61's 37 files), `vectors_s3_old_pc_broker.pt`.

**LOW conflicts:**
- **CV-13.** HP-KR `rows.jsonl` differs across stores only in extra columns (2,139 rows, 139/2,000; kr-matched 258, 124/134 everywhere).
- **CV-16.** The stale lab copies of FINDINGS.md and DESIGN.md.

**Drive-only files (`drive_not_on_lab.tsv.gz`, 407 files).**
- **Real older versions:** the old paper16 vectors and axes (CV-01) and the own100 axes (CV-10).
- **Not versions:**
  - coords/passC (75): the unhashed big files;
  - figures-crossgen/kd figures (251): these match git, except 13 files that differ by CRLF only;
  - share (6), thesis-docs (4), staging (5), thinking/plans (4): inputs or docs.

## Commands (repo root, Git Bash)
```
export PYTHONIOENCODING=utf-8
python audit/scripts/p1_anom_folders.py
python audit/scripts/p1_anom_units.py          # ~2.5 min
python audit/scripts/p1_anom_logs.py
python audit/scripts/p1_anom_ghosts.py         # ~20 s
python audit/scripts/p1_anom_grepdocs.py [--sec] '<regex>' ...
# lab PC (after: python audit/scripts/labpc_upload.py audit/scripts/p1_anom_common.py audit/scripts/p1_anom_{dupes,conflicts}_lab.py
#                python audit/scripts/labpc_upload.py --dest census audit/phase1/census_laptop_ignored.tsv)
python audit/scripts/labpc_run.py audit/scripts/p1_labpc_runpy.ps1 FILE=p1_anom_dupes_lab.py "ARGS=" --out audit/phase1/anomalies_detail/duplicates_lab_output.txt
python audit/scripts/labpc_run.py audit/scripts/p1_labpc_runpy.ps1 FILE=p1_anom_conflicts_lab.py "ARGS=" --out audit/phase1/anomalies_detail/conflicts_lab_output.txt
python audit/scripts/labpc_run.py audit/scripts/p1_fmap_F001-F027_pyc.ps1 "CODE=$(cat audit/scripts/p1_anom_snip_feat.py)"   # also _sig, _bank, _rows, _scores
python audit/scripts/labpc_run.py audit/scripts/p1_anom_labgrep.ps1
python audit/scripts/p1_anom_write.py
```

## Limits
- **Mention matching is lexical.** A doc can describe a result without naming its file. OR-05 is the clear example. So "no doc names it" is VERIFIED, but "no doc reports it" is INFERRED from reading the named docs.
- **CRLF text files that exist only on Drive** cannot be compared by md5. They were checked by size arithmetic (CV-15).
- **Lab files over 200 MB** were compared by path and size only.
- **Not re-checked:**
  - md5 of his uncommitted files (they are not in any source);
  - Drive `share/paper16_L20/paper_questions/*` against any lab file (no lab counterpart exists).
