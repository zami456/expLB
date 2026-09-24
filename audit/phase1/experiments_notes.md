# Phase 1 Task A: experiment inventory (notes)

Facts only, no verdicts. Main table: `audit/experiments.csv` (114 rows, 18 columns).
Tags as in the brief: VERIFIED (checked in a file or computed by a script during this audit), DOC-ONLY, INFERRED (reason given), CONFLICT, NOT-FOUND.

## 1. How the table was built (re-runnable, read-only)

| file | what |
|---|---|
| `audit/scripts/p1_exp_defs.py` | Hand-written definitions, one per experiment: scripts, drivers, scheduled task, result-path patterns, logs, status evidence, dependencies, F-ids. Every statement was checked in a file during this audit (lab status files, logs, scripts, census, fmap), unless it is tagged. |
| `audit/scripts/p1_exp_build.py` | Computes everything else from raw census files and writes `audit/experiments.csv` + `audit/phase1/exp_work/experiments_computed.tsv`. Run: `PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 python audit/scripts/p1_exp_build.py` (~1 min; streams the gz census). |
| `audit/scripts/p1_exp_counts.py` | The counts and lists in sections 2-5 below. |
| `audit/scripts/p1_exp_dirs.py` | Lab folder aggregation, `audit/phase1/exp_work/lab_dirs.tsv` (files, MB, mtime range, Drive/asix copies per folder, depth ≤3). |
| `audit/scripts/p1_exp_status.ps1` | Read-only dump of every lab `logs/*.status` file, saved to `audit/phase1/exp_work/lab_status_files.txt` (770 lines). This is the main source of `status_evidence`. |
| `audit/phase1/exp_work/` | Also `log_index_view.txt` (last line of every lab log), `fmap_view.txt`, `lab_loose_files.tsv` (205 loose lab files with mtimes). |

What was computed rather than typed (VERIFIED by the build script):
- `n_result_files` and the `[computed: ...]` part of `results_location`. This covers lab file count, MB and mtime range, and how many lab files have an identical copy on Drive (census `on_drive`) or in asix git (`in_asix` = the same md5_lf anywhere in asix, not necessarily at the same path). It also covers git, his-branch, Drive-only and laptop-only counts.
- `when_first` / `when_last` when no manual override is given. These come from internal log timestamps (log_index.tsv), or else from lab file mtimes. `when_source` says which.
- The `commit` column. It holds the first and last git commit that touched each listed script (`git log`), on asix or on his branch.
- The bracketed notes:
  - `[main script ... first entered git ..., AFTER when_first]`;
  - `[RF-13: N of M lab files have mtimes in the 09-22 15:40-16:30 re-run wave; K of those are byte-identical to some asix git blob]`;
  - `[doc mentions ...]`: a case-insensitive search for experiment-specific strings (`DOC_KEYS` in p1_exp_defs.py). The search covers 96 docs: tracked project docs on asix (kd/docs, docs/ without the nnsight library docs, docs/thesis, docs/papers, cap/docs), the 45 memory notes, the draft text, and his 17 summary/README files.

Abbreviations in the CSV:
- `DD/` = `src/probes/deception-detection/`.
- `LAB-ONLY <path>` = a loose script on the lab PC that is in no git branch.
- Row ids:
  - `BG` = background (not produced by this project);
  - `O` = owner-side run;
  - `AN` = a family of print-only analyses (no output files);
  - `C` = collaborator (his branch).

## 2. Counts (VERIFIED, p1_exp_counts.py)

114 rows:
- 2 background;
- 107 runs (86 owner O-rows, 1 lab folder of unknown owner O87, 19 collaborator result folders + 1 collaborator ghost);
- 5 print-only analysis families.

| who \ status | complete | crashed | partial | superseded | unknown | total |
|---|---|---|---|---|---|---|
| owner (O01-O86, AN1-AN5) | 85 | 1 | 2 | 1 | 2 | 91 |
| collaborator (C01-C20) | 19 | 0 | 0 | 0 | 1 | 20 |
| unknown (O87) | 0 | 0 | 0 | 0 | 1 | 1 |
| upstream (BG1, BG2) | 2 | 0 | 0 | 0 | 0 | 2 |

Rows that are not `complete`:

**crashed**
- O40 (first 10x10 crossgen retrain). crossgen.status says `PYTHON_EXIT -1` at 09-17 21:18:44, and the task result is 267014. It was killed during mean layer 12 of 21. Its matrix files were later rewritten by O43.

**partial**
- O05 (own100 fp32 re-extraction): 1 of 100 vectors.
- O56 (kd passA/passC crossgen of 09-20): STEP C-root exit 1. This is the F-25 incident.

**superseded**
- O03 (paper16 original bank): its vectors and axes were replaced by the fp32 rebuild O04 (RF-12). Its rollouts and scores are still the current ones.

**unknown**
- AN5: inline analyses with no script kept.
- O86: D:/expLB-acts workspace, venv build only.
- O87: `expLBactivationLAYER22` cache.
- C20: the INLP experiment, which exists only in a commit message.

F-number coverage (VERIFIED):
- 106 of the 109 F-ids are attached to at least one row. 54 rows carry at least one F-id.
- Not attached: F-70, F-78 and F-88. These are provenance or paper-reading entries (novelty audit, papers added, papers read). They have no experiment behind them.
- F-ids appear on the row that PRODUCED the data. When the number itself came from a print-only reader, they also appear on that reader's AN row. So an F-id can appear on several rows.

## 3. Candidate orphans: no F-number and no doc mention (14 rows; pass to Task B)

"Doc mention" uses the precise keys described in section 1, so a 0 means no doc names this experiment's files, folders or scripts. It does not mean the topic is never discussed.

| exp_id | what | size | remark |
|---|---|---|---|
| O44-crossgen-full-passA | cross-generalisation on the 80-layer passA store: root + 7 banks, seeds 0-4 + 42 | **92,184 lab files** (on Drive) | Largest substantial orphan: a complete 80-layer × 6-seed × 8-bank grid. No F-entry and no doc cites `run-crossgen/passA`. Its 09-22 re-run failed, so these are the 09-18/19 originals. |
| O45-crossgen-full-passC | same on passC (6 layers): root + 7 banks, 6 seeds | 6,936 | Complete grid (status DONE, every step EXIT 0). The memory note `crossgen-layer-coverage` mentions that a passC crossgen arm was added on 09-18, but never its results. |
| O46-crossgen-Bfine-P | passB_fine (10 fine layers around the peak; root + 3 banks) and passP (prompt-side control; root) | 5,777 | Complete. The passP arm is the construct-validity control named in `run_extra_extractions.ps1`. No F-entry reports its result. |
| O10-extra-P0a-sycmt | sycophancy_mt rows added to passA/passC | 18 files, 3.5 GB | Input step for O44/O45. |
| O26-length-confounds | length-only / length-controlled re-runs of p1 and Q5 (lab-only scripts) | 3 JSON (lab + Drive, not in git) | Memory note `response-length-transfers-on-its-own` covers the topic but names none of these files (INFERRED link). |
| O16-q-pilot-probes | first probe runs (smoke, 5-set, L20 pilot, splits) | 5 JSON | Pilot. |
| O39-crossgen-smoke | crossgen smoke | 1 JSON | Pilot. |
| O01-aa-smoke | pipeline smoke tests | lab 3 + laptop 39 | Pilot. |
| O05-own100-fp32 | own100 fp32 re-extraction | 1 .pt | Partial; see section 5. |
| O87-layer22-acts-cache | 97 cached activation tensors in the acts-cache layout (sha-keyed paths) | 99 files, 361 MB, lab-only | 09-04..09-12. Who ran it and why: NOT-FOUND. INFERRED from the path layout: the collaborator's acts_cache format. |
| C01-meanprobe-0712 | his 07-12 two-dataset Mean Probe run | 4 PNG | Plots only. |
| C14-mod-L44, C19-positions-L44 | his L44 folders | 43 / 7 files | Not true orphans: his L44 summaries are 2-line pointers to the L40 summaries, which cover "per layer" (branch_compare sec 3). |
| C20-inlp-ghost | "INLP experiment" | 0 | Ghost: named in commit 148ddeb, no files anywhere. |

Mentioned only in prose, with no F-number: 46 rows (list in the p1_exp_counts.py output). Most important:
- The whole 09-15..09-17 plan-question series O16-O34 (Q1-Q11, run-coords/*.json). Their only prose home is `docs/thesis/persona-coordinates-70b-results.md` and the memory notes, because the F-ledger starts on 09-19.
- The banks O35-O38 and the crossgen family O41-O47.
- Phase 2 needs a claim source for these: the thesis results doc plus memory notes.

## 4. Superseded or regenerated results

### 4a. Replaced by a later version (VERIFIED unless tagged)

- **O03 → O04, paper16 vectors (RF-12).**
  - The original bf16-reduction vectors_s3/vectors_all (laptop, and Drive paper16/run) were re-extracted with fp32 reductions on 09-15/16.
  - **New fact (VERIFIED, the `bank` key inside the JSONs):** the 09-15 run-coords analyses all record `bank: axes_vectors_s3.pt`. That is the pre-fp32 bank, INFERRED from the name: fp32 files carry `_fp32`. These analyses are smoke, probes_5sets, probes_L20, splits, zeroshot_paper16, conf, erase_test and FULL_L20/L22/L40 (O16, O17).
  - All 09-16+ analyses record `axes_vectors_s3_fp32.pt`. So the early Q1/Q2/Q4 numbers were computed on the old bank.
- **O40 (crashed retrain).** Its matrices were overwritten by O43 (09-18 04:38-05:52).
- **O41/O42: L40-only root and bank summaries.**
  - git b9a497e keeps the L40-only summary/significance.
  - On the lab, O43 rewrote them as supersets.
  - Keyed diff (PROGRESS, RF-13 update): the git version is an exact subset, and only the BH q-values differ.
- **O56 → O58.**
  - The 09-20 kd passA/passC crossgen ran when HP-KR had only 200 rows in passA/passC (F-20: "kr-matched: no rows").
  - It was redone as passA11/passC11 after the HP-KR top-up O57.
- **O76, first HPC version (F-58) → mask version (F-63).**
  - The 22:19-22:21 matrices were overwritten 23:38-23:40.
  - F-58's values survive only in `logs/crossgen_scale_{3b,8b}.log` (fmap).
- **kd_s23 significance.json (O55).**
  - The 3,276-test version read by F-34 and by thesis fig2 (`kd/results/fig_data.json`) exists nowhere now.
  - Lab/Drive have 5,544 tests; git has 468 (RF-7).
- **kd11 root significance.json (O51).** F-04's 384-test version was overwritten on the evening of 09-19 (fmap).
- **cap persona bank v1/v2 (O62).**
  - The texts were overwritten 09-20 22:51 by v4.
  - They survive only inside git `cap/docs/REVIEW_PACKET*.md` (F-28/F-32, fmap).
- **Sarcasm pilot (O70).**
  - The 09-21 verdict "PRODUCIBLE" was renamed `sarc_pilot.status.2026-09-21-buggy-verdict`.
  - Later runs (09-22 02:29, 15:55) give "GATE FAILED / TEACHER-FORCED".
  - **CONFLICT** between the two 09-22 status files: the stripped bag-of-words AUROC is 0.947 at 02:29 but 0.921 at 15:55. The 15:55 text says "unmarked(v4,v5)", so the subset differs. Phase 2 must pin which number F-93/F-105 use.
- **Crossgen "earlier draft" design.**
  - Memory note `crossgen-80-20-retrain` (DOC-ONLY) says files that scored the other datasets on their 20% splits were deleted everywhere.
  - They are NOT-FOUND, which is consistent with that note.

### 4b. Regenerated in the 09-22 15:40-16:30 re-run wave (RF-13; VERIFIED by census mtimes)

"Identical to git" = the md5_lf equals some asix blob, so the content did not change against git.

| exp_id | lab files rewritten | of which identical to a git blob | comment |
|---|---|---|---|
| O58-kd-passAC11 | 23,750 of 24,204 | 0 | not in git at all; summary/significance/csv rebuilt, matrices from 09-21 |
| O43-crossgen-full-passB | 11,100 of 24,216 | 672 | root + hf275_loose/strict + own100_strict summaries, significance, csv |
| O65-cap-chain | 6,054 of 6,210 | 1,513 | cap_all/cap_s23 csv/summary/significance, rebuilt by the Cap-Layers/Cap-Seeds re-runs |
| O46-crossgen-Bfine-P | 5,691 of 5,777 | 240 | |
| O53-kd-bank-crossgen | 3,279 of 3,321 | 3,278 | content unchanged except 1 file (significance.json, RF-11) |
| O68-cap-controls | 3,028 of 3,113 | 757 | run-cap-ctrl/axes_s23.pt rewritten; there is no other copy (fmap F-89) |
| O55-kd-s23-alllayers | 3,027 of 3,069 | 1,081 | significance.json became 5,544 tests |
| O51-kd11-root | 2,775 of 2,817 | 2,774 | only significance.json differs from git; summary.json/splits.json were rewritten byte-identical |
| O80-premise-extras | 7 of 7 | 0 | **all** premise-extras outputs (seeds 1/7/13/99, mahal, cluster_perm) date from the wave; no older copy (F-83/F-84) |
| O52, O54 | 2, 1 | 0 | axes_vectors_{s3,all,s23}.pt; fmap: md5 = the Drive copies of 09-19/20, i.e. unchanged |
| O57 | 2 | 0 | passA/passC HP-KR items.pt (md5 = the Drive copies) |
| O70 | 1 | 0 | cap/run-sarc-pilot/marker_report.json |

Attempted in the wave but **not** rewritten:
- O44 (passA crossgen): every bank EXIT 1, "split for harm-pressure-knowledge-report seed 42 differs from run-crossgen/passA/splits.json".
- Status files whose earlier content was lost in the wave, so the evidence left is the 09-22 re-run:
  - through the RF-14 guard bug: O58, O66, O68, O70, O79, O80, O83;
  - overwritten without RF-14: O52 (kd_chain.status), O67 (cap_seeds.status).

## 5. Other facts found while building the table (VERIFIED unless tagged)

1. **The exporter scripts for the collaborator's inputs exist.** They are loose lab files `C:/Users/user4/build_share_L20.py` (09-18 16:23) and `build_deceptive_L20.py` (09-18 16:36), next to their outputs `share_L20_shared/` and `share_deceptive/`. branch_compare.md had "exporter NOT-FOUND in git". That is still true for git, but the scripts are on the lab (O38).
2. **Scripts first committed after their runs** (computed flag in `notes`): the run-time version was not in git when it ran. This applies to the crossgen family (27_crossgen.py first in git 09-18 11:26, runs from 09-17 21:12), all kd/cap/scale drivers (first committed 09-21) and the paper16 chain. Phase 2 cannot assume the git version is the version that ran.
3. **own100 was never re-extracted in fp32 (O05, partial 1/100).** own100 and own100_strict (O35) therefore stay on the bf16-reduction vectors (`run/vectors`). The paper16 banks moved to fp32.
4. **Results with one copy only, or missing from Drive** (census `on_drive`):
   - not on Drive at all: all of run-premise (O78-O85, 100 files); cap smokes and probes (O62-O64); the sarcasm pilot (O70); run-cap and run-cap-ctrl (O65 and O68: 66 and 44 files); `run-paper16/axes_vectors_s3_fp32_famcentred.pt` (O36); `run/axes_strict.pt`; the scale alpaca/mask coords (O75: 628 of 629 files; the 09-22 Drive sync failed);
   - only in git + laptop, not on lab or Drive: `run-paper16/scores_405b` (O06, 25 files) and `run-coords/regroup_hf.json` (O23);
   - only on Drive, not on the lab: the figures in `figures-crossgen/` (143) and `kd/figures/crossgen11` (108) (O47), and `paper16/run` + `paper16/logs` (O03);
   - O09's passC shows only 54 of 123 on Drive because 75 files over 200 MB were not hashed. It is not necessarily missing.
5. **cap_ctrl layer matrices exist despite the failed gate (O69).** All 21 mean + 21 last matrices are on disk (09-21). The 09-22 status says "measured=0 ... GATE FAILED" because of the RF-6 backspace byte in the gate path (INFERRED, from the 42 files on disk).
6. **The passA crossgen results (O44) are the 09-18/19 originals.** The 09-22 re-run refused to mix splits because the HP-KR split had changed after the top-up (log text). So no passA file was rewritten.
7. **Scale 3B/8B alpaca and mask activations (O75) have no driver** in any branch or among the lab loose scripts. There are only logs `alpaca_{3b,8b}.log` and `mask_{3b,8b}.{log,err}` (09-20 21:55-23:37).
8. **The 70B-rung bank is a CONFLICT (O77)** between fmap F-63 (matrix field `axes_vectors_s3.pt`, bank_name paper16_70b) and F-100 (`run-kd/axes_vectors_s3.pt`). For Phase 2.
9. **The paper16 original chain probably ran on the laptop** (INFERRED). Its logs (`logs/paper16_*.log`, 09-15 03:57-11:13; `judge_405b.log` 14:38-15:12) exist only on the laptop and on Drive `paper16/logs`, not in the lab's logs/.
10. **Two sig jobs share one log** (O30/O32, INFERRED). `sig.log` / `sig.status` hold the nulls job (writes zeroshot_nulls.json 01:05). The significance job (task expLB-significance, 01:00; JSON 01:03) has no surviving log.
11. **kd-matched views are built from linked chunks** (INFERRED from `passAC11_view.log` "linked 0 chunks" and from chunk mtimes equal to the source HP-KR chunks). Folder mtime ranges for `harm-pressure-kr-matched` therefore start at 09-15/16, which is not the view-building time.

## 6. Dependencies between the two branches (for Phase 3)

- **Owner → collaborator:**
  - O37 run-paper16-shared → O38 export (share_L20_shared, share_deceptive) → his `paper16_L20_sharedquestions/` and `deceptive/` (md5, RF-1) → C06 (and C07 through the uncommitted `persona vectors/vectors_s3`, DOC-ONLY);
  - O52 run-kd/vectors_s3 → his `persona including knowlede denial/` (36/36 md5, RF-1) → C08-C11 and C15-C19;
  - C12-C14 use the uncommitted "modified" bank; its origin is NOT-FOUND.
- **Collaborator → owner:**
  - C08/C09 ext_axis1 contrast at f386a72 → `kd/personas/kd_role_groups_collab.json` → O60 (collab bank, F-98/F-99/F-103), and O71 via axes_collab_s23.pt (F-107);
  - C08-C10 "six datasets" → O81 (F-85, F-109);
  - F-97 reads C08/C09 directly.

## 7. Not tabulated as experiments (infrastructure runs)

These produced no results:
- the Drive sync tasks (Crossgen-Sync-Periodic, Crossgen-Sync-Now, KD-Sync-Periodic, Night-Chain sync part, Coords-Supervisor, sync_scale), plus expLB-drive-sync and expLB-drive-check (logs/drive_check/*);
- Crossgen-Dedupe (09-18 13:33-15:46) and Crossgen-Final-Report (report text only, attached to O47);
- the lab-inventory scripts `labpc_*.ps1` (09-17), preflight probes and NDIF throughput probes;
- Thesis_Train (D:/user4fina, 2026-08-04). It is outside the census roots and was not inspected.

## 8. Caveats on this table

- `when_*` for C-rows are git commit dates. His run dates are not recorded in any file on his branch.
- Lab file mtimes after 09-22 15:40 do not date the original runs (RF-13). `when_source` flags rows whose `when_last` falls in the wave.
- The O66, O67 and O69 matrix counts are subsets of the O65 and O68 folder counts, and O41's two L40 matrices are also inside O43. Do not sum `n_result_files` across rows.
- `f_ids` come from `audit/phase1/fmap_all.csv` (mapping agents' `resolved_primary` / `producing_script`), assigned to rows by hand. Their mapping_status (PARTIAL / NOT-FOUND) is in fmap, not repeated here.
- The doc-mention search is substring-based. The keys were chosen to be specific, but a few false hits remain. Example: "figure 6" in paper-notes for O41 (the paper's own Figure 6).
