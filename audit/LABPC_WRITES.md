# Record of everything the audit creates on the lab PC

On 2026-09-22 the owner approved creating a scratch area on the lab PC ("create whatever u need, just keep it in the records").

**Rules I follow:**
- Everything I create lives under `C:\Users\user4\audit_scratch\`. Nothing is written anywhere else.
- I never modify, move or delete the owner's files.
- The lab PC has other users; I never touch their folders (anything outside `C:\Users\user4\`, and other people's folders on `D:\`).
- I append one line to the table below for every folder or file I create, and for every deletion of my own scratch files.
- Before a run, the owner's input files are identified by md5 and noted in the relevant audit/ record.

| when (+06) | action | path on lab PC | purpose | size |
|---|---|---|---|---|
| 2026-09-22 20:56 | upload | `audit_scratch\scripts\p1_labpc_census.ps1` | the census script (too long for -EncodedCommand; uploaded by tar over ssh via labpc_run.py) | 3 KB |
| 2026-09-22 20:56 | write | `audit_scratch\census\labpc_census.tsv`, `labpc_census.log` | Phase 1 file census of the team's folders (script audit/scripts/p1_labpc_census.ps1; reads owner files, writes only here) | tbd |
| 2026-09-22 ~21:05 | write | `audit_scratch\census\drive_lsjson.json`, `drive_lsjson.err` | Drive listing with md5 via the lab PC's rclone `lsjson` (read-only on Drive; script p1_labpc_drive_census.ps1) | tbd |
| 2026-09-22 20:51 | mkdir | `C:\Users\user4\audit_scratch\` + `census\`, `scripts\`, `out\` | audit workspace (script: audit/scripts/p1_labpc_mkscratch.ps1) | empty |
| 2026-09-22 20:59 | upload | `audit_scratch\scripts\p1_labpc_census_eol.py` | audit script (source: audit/scripts/p1_labpc_census_eol.py) | 1209 B |
| 2026-09-22 21:0x | write | `audit_scratch\census\labpc_census_eol.tsv` | line-ending-normalized md5 of 205,025 text files (p1_labpc_census_eol.py, run with the lab .venv-analysis python) | tbd |
| 2026-09-22 21:15 | upload | `audit_scratch\scripts\p1_labpc_drive_tsv.py` | audit script (source: audit/scripts/p1_labpc_drive_tsv.py) | 724 B |
| 2026-09-22 21:3x | write | `audit_scratch\census\drive_census.tsv.gz` | gzipped TSV of the Drive listing (p1_labpc_drive_tsv.py) | small |
| 2026-09-22 21:15 | upload | `audit_scratch\scripts\p1_labpc_drive_tsv.py` | audit script (source: audit/scripts/p1_labpc_drive_tsv.py) | 725 B |
| 2026-09-22 21:3x | write | `audit_scratch\census\drive_census.tsv.gz` | gzipped TSV of the Drive listing (p1_labpc_drive_tsv.py) | small |
| 2026-09-22 21:16 | upload | `audit_scratch/census/census_asix_git.tsv` | audit file (source: audit/phase1/census_asix_git.tsv) | 1400964 B |
| 2026-09-22 21:16 | upload | `audit_scratch/census/census_pcood.tsv` | audit file (source: audit/phase1/census_pcood.tsv) | 158504 B |
| 2026-09-22 21:16 | upload | `audit_scratch/scripts/p1_census_join.py` | audit file (source: audit/scripts/p1_census_join.py) | 6024 B |
| 2026-09-22 21:2x | write | `audit_scratch/census/joined_labpc.tsv.gz`, `drive_not_on_lab.tsv.gz`, `asix_results_not_on_lab.tsv.gz`, `join_summary.txt` | census join (p1_census_join.py, lab .venv-analysis python) | small |
| 2026-09-22 21:19 | upload | `audit_scratch/scripts/p1_peek.py` | audit file (source: audit/scripts/p1_peek.py) | 3701 B |
| 2026-09-22 21:20 | upload | `audit_scratch/scripts/p1_labpc_logindex.py` | audit file (source: audit/scripts/p1_labpc_logindex.py) | 2676 B |
| 2026-09-22 21:4x | write | `audit_scratch/out/log_index.tsv` | index of the 242 lab logs (p1_labpc_logindex.py) | small |
| 2026-09-22 21:22 | upload | `audit_scratch/scripts/p1_labpc_logindex.py` | audit file (source: audit/scripts/p1_labpc_logindex.py) | 2967 B |
| 2026-09-22 21:22 | upload | `audit_scratch/scripts/p1_labpc_logindex.py` | audit file (source: audit/scripts/p1_labpc_logindex.py) | 2984 B |
| 2026-09-22 21:24 | upload | `audit_scratch/scripts/p1_labpc_logindex.py` | audit file (source: audit/scripts/p1_labpc_logindex.py) | 2990 B |
| 2026-09-22 21:54 | upload | `audit_scratch/scripts/p1_axes_old_vs_fp32.py` | audit file (source: audit/scripts/p1_axes_old_vs_fp32.py) | 2096 B |
| 2026-09-22 22:13 | upload | `audit_scratch/scripts/p1_anom_common.py` | audit file (source: audit/scripts/p1_anom_common.py) | 10158 B |
| 2026-09-22 22:13 | upload | `audit_scratch/scripts/p1_anom_dupes_lab.py` | audit file (source: audit/scripts/p1_anom_dupes_lab.py) | 7001 B |
| 2026-09-22 22:14 | upload | `audit_scratch/census/census_laptop_ignored.tsv` | audit file (source: audit/phase1/census_laptop_ignored.tsv) | 46289 B |
| 2026-09-22 22:17 | upload | `audit_scratch/scripts/p1_anom_conflicts_lab.py` | audit file (source: audit/scripts/p1_anom_conflicts_lab.py) | 4627 B |
| 2026-09-22 22:32 | upload | `audit_scratch/scripts/p1_bank_fingerprint.py` | audit file (source: audit/scripts/p1_bank_fingerprint.py) | 1235 B |
| 2026-09-22 22:33 | upload | `audit_scratch/scripts/p1_bank_fingerprint.py` | audit file (source: audit/scripts/p1_bank_fingerprint.py) | 1297 B |
| 2026-09-22 23:30 | status | `audit_scratch\` total ≈ 310 MB (census\ ≈ 312 MB incl. drive_lsjson.json 232 MB; out\ 59 KB; scripts\ 11 small .py/.ps1) | end-of-Phase-1 snapshot; nothing outside audit_scratch was written (VERIFIED: audit/phase1/labpc_changed_since_audit_start_end_of_phase1.txt shows only the owner's own sync-task logs changed) | — |
| 2026-09-22 23:46 | upload | `audit_scratch/out/backup_files_expLB.txt` | audit file (source: audit/fix/backup_files_expLB.txt) | 71786 B |
| 2026-09-22 23:46 | upload | `audit_scratch/out/backup_files_expLB-acts.txt` | audit file (source: audit/fix/backup_files_expLB-acts.txt) | 1250 B |
| 2026-09-22 23:46 | upload | `audit_scratch/out/backup_files_expLBactivationLAYER22.txt` | audit file (source: audit/fix/backup_files_expLBactivationLAYER22.txt) | 8955 B |
| 2026-09-22 23:46 | upload | `audit_scratch/out/backup_files_user4-home-loose-files.txt` | audit file (source: audit/fix/backup_files_user4-home-loose-files.txt) | 1001 B |
| 2026-09-22 23:46 | DRIVE WRITE (owner-authorised) | `tamara:expLB/_labpc_only_backup_2026-09-22/` (new folder) | RF-10 backup of 1,983 lab-only files (1.73 GB), rclone copy --immutable + check --one-way, run on the lab PC by audit/scripts/fix_backup_run.ps1; logs audit_scratch/out/backup_*.log | 1.73 GB |
| 2026-09-22 23:49 | write | audit_scratch/backup_scripts_2026-09-22.tar | lab-side copy of the 8 original scripts before patching | small |
| 2026-09-22 23:49 | OWNER FILES MODIFIED (owner-authorised bug fix) | 8 scripts under C:/Users/user4/expLB/: cap/scripts/run_cap_controls.ps1, run_cap_layers.ps1, run_sarc_pilot.ps1; kd/scripts/run_passAC_11.ps1; src/assistant_axis_ndif/run_falsify.ps1, run_premise_extras.ps1, run_premise_layers.ps1, run_cap_ctrl_layers.ps1 | RF-14 guard pattern + RF-6 backspace byte (audit/scripts/fix_guard_bug.py). Originals kept in audit_scratch/backup_scripts_2026-09-22.tar and audit/fix/originals_lab/. md5 verified: audit/fix/fix_lab_verify.txt | 8 files |
| 2026-09-23 00:05 | NEW RESULT FOLDER (owner-authorised RF-16 fix) | C:/Users/user4/expLB/run-crossgen/scale_70b_paper16fp32/ (new folder) | like-for-like 70B k=3 sweep with run-paper16/axes_vectors_s3_fp32.pt instead of the kd bank; owner script 27_crossgen.py, CPU only; driver audit/scripts/fix_rf16_run.ps1; log audit_scratch/out/rf16_crossgen_scale_70b_paper16fp32.log | tbd |
| 2026-09-23 00:15 | DRIVE WRITE DONE | tamara:expLB/_labpc_only_backup_2026-09-22/ | RF-10 backup COMPLETE: 1,983 files / 1.731 GB (rclone size). Verify: 1,977 hash-identical; 6 'sizes differ' are the owner's live sync logs (coords_rclone, coords_supervisor, kd_rclone, kd_sync.status, night_chain, night_rclone) that grew during the copy. Logs: audit_scratch/out/backup_check_*.log | 1.73 GB |
| 2026-09-23 00:16 | upload | `audit_scratch/scripts/fix_rf16_recompute.py` | audit file (source: audit/scripts/fix_rf16_recompute.py) | 3953 B |
| 2026-09-23 00:17 | upload | `audit_scratch/scripts/fix_rf16_recompute.py` | audit file (source: audit/scripts/fix_rf16_recompute.py) | 4136 B |
| 2026-09-23 00:30 | SCHEDULED TASKS DISABLED (owner-authorised) | lab PC tasks PassAC-11, PassC-PerToken, Night-Chain, Coords-Supervisor | their work is complete (passAC11.status DONE; run-coords passA/B/C .done present, so run_passC / night_chain / coords_supervisor exit early and only synced). Disabled so a reboot cannot re-run them (RF-13). Coords-Supervisor was Running (an rclone check) and was stopped first. Re-enable with Enable-ScheduledTask -TaskName '<name>'. Evidence audit/fix/task_completion.txt; result audit/fix/task_disable_result.txt | 4 tasks |
| 2026-09-23 02:12 | upload | `audit_scratch/scripts/fix_fingerprint_all.py` | audit file (source: audit/scripts/fix_fingerprint_all.py) | 2509 B |
| 2026-09-23 02:18 | upload | `audit_scratch/scripts/fix_bigfiles_hash.py` | audit file (source: audit/scripts/fix_bigfiles_hash.py) | 1124 B |
| 2026-09-23 02:41 | upload | `audit_scratch/scripts/p2v_F09.py` | audit file (source: audit/scripts/p2v_F09.py) | 1446 B |
| 2026-09-23 02:41 | upload | `audit_scratch/scripts/p2v_F09b.py` | audit file (source: audit/scripts/p2v_F09b.py) | 1660 B |
| 2026-09-23 02:42 | upload | `audit_scratch/scripts/p2v_F55.py` | audit file (source: audit/scripts/p2v_F55.py) | 1588 B |
| 2026-09-23 02:42 | upload | `audit_scratch/scripts/p2v_F55b.py` | audit file (source: audit/scripts/p2v_F55b.py) | 1089 B |
| 2026-09-23 02:42 | upload | `audit_scratch/scripts/p2v_F55c.py` | audit file (source: audit/scripts/p2v_F55c.py) | 1598 B |
| 2026-09-23 02:43 | upload | `audit_scratch/scripts/p2v_F109.py` | audit file (source: audit/scripts/p2v_F109.py) | 1017 B |
| 2026-09-23 02:44 | upload | `audit_scratch/scripts/p2v_F83.py` | audit file (source: audit/scripts/p2v_F83.py) | 1636 B |
| 2026-09-23 02:45 | upload | `audit_scratch/scripts/p2v_F65.py` | audit file (source: audit/scripts/p2v_F65.py) | 3080 B |
| 2026-09-23 02:45 | upload | `audit_scratch/scripts/p2v_F65b.py` | audit file (source: audit/scripts/p2v_F65b.py) | 1631 B |
| 2026-09-23 02:46 | upload | `audit_scratch/scripts/p2v_F100.py` | audit file (source: audit/scripts/p2v_F100.py) | 3888 B |
| 2026-09-23 02:47 | upload | `audit_scratch/scripts/p2v_F100b.py` | audit file (source: audit/scripts/p2v_F100b.py) | 2721 B |
| 2026-09-23 02:47 | upload | `audit_scratch/scripts/p2v_misc.py` | audit file (source: audit/scripts/p2v_misc.py) | 627 B |
| 2026-09-23 02:47 | upload | `audit_scratch/scripts/p2v_F04.py` | audit file (source: audit/scripts/p2v_F04.py) | 1289 B |
| 2026-09-23 02:48 | upload | `audit_scratch/scripts/p2v_F65c.py` | audit file (source: audit/scripts/p2v_F65c.py) | 1217 B |
| 2026-09-23 02:48 | upload | `audit_scratch/scripts/p2v_F17.py` | audit file (source: audit/scripts/p2v_F17.py) | 1149 B |
| 2026-09-23 02:48 | upload | `audit_scratch/scripts/p2v_F62.py` | audit file (source: audit/scripts/p2v_F62.py) | 2022 B |
| 2026-09-23 02:48 | upload | `audit_scratch/scripts/p2v_F17b.py` | audit file (source: audit/scripts/p2v_F17b.py) | 1481 B |
| 2026-09-23 02:55 | upload | `audit_scratch/scripts/p2_w1_layer_sensitivity.py` | audit file (source: audit/scripts/p2_w1_layer_sensitivity.py) | 3478 B |
| 2026-09-23 02:56 | write | audit_scratch/out/layer_sensitivity.csv | Phase 2 W1 table: off-diagonal means per store/pooling/layer/seed/feature/dataset-set (78,516 rows; p2_w1_layer_sensitivity.py) | 8 MB |
| 2026-09-23 03:05 | upload | `audit_scratch/scripts/F-13.py` | audit file (source: audit/scripts/claims/F-13.py) | 3101 B |
| 2026-09-23 03:07 | upload | `audit_scratch/scripts/F-31.py` | audit file (source: audit/scripts/claims/F-31.py) | 5088 B |
| 2026-09-23 03:07 | upload | `audit_scratch/scripts/_dsboot.py` | audit file (source: audit/scripts/claims/_dsboot.py) | 1583 B |
| 2026-09-23 03:08 | upload | `audit_scratch/scripts/F-34.py` | audit file (source: audit/scripts/claims/F-34.py) | 4632 B |
| 2026-09-23 03:08 | upload | `audit_scratch/scripts/F-34b.py` | audit file (source: audit/scripts/claims/F-34b.py) | 3379 B |
| 2026-09-23 03:21 | upload | `audit_scratch/scripts/P4_lab_scores.py` | audit file (source: audit/scripts/claims/P4_lab_scores.py) | 4492 B |
| 2026-09-23 03:21 | upload | `audit_scratch/scripts/cap_probe.py` | audit file (source: audit/scripts/claims/cap_probe.py) | 1886 B |
| 2026-09-23 03:22 | upload | `audit_scratch/scripts/F-65.py` | audit file (source: audit/scripts/claims/F-65.py) | 5542 B |
| 2026-09-23 03:23 | upload | `audit_scratch/scripts/F-65b.py` | audit file (source: audit/scripts/claims/F-65b.py) | 3115 B |
| 2026-09-23 03:23 | upload | `audit_scratch/scripts/P4_lab_geom.py` | audit file (source: audit/scripts/claims/P4_lab_geom.py) | 5349 B |
| 2026-09-23 03:24 | upload | `audit_scratch/scripts/F-66.py` | audit file (source: audit/scripts/claims/F-66.py) | 3033 B |
| 2026-09-23 03:24 | upload | `audit_scratch/scripts/F-66.py` | audit file (source: audit/scripts/claims/F-66.py) | 4325 B |
| 2026-09-23 03:25 | upload | `audit_scratch/scripts/P4_lab_geom2.py` | audit file (source: audit/scripts/claims/P4_lab_geom2.py) | 3263 B |
| 2026-09-23 03:25 | upload | `audit_scratch/scripts/F-67.py` | audit file (source: audit/scripts/claims/F-67.py) | 3980 B |
| 2026-09-23 03:25 | upload | `audit_scratch/scripts/P4_lab_geom2.py` | audit file (source: audit/scripts/claims/P4_lab_geom2.py) | 3497 B |
| 2026-09-23 03:26 | upload | `audit_scratch/scripts/F-89.py` | audit file (source: audit/scripts/claims/F-89.py) | 5301 B |
| 2026-09-23 03:26 | upload | `audit_scratch/scripts/P4_lab_matrix.py` | audit file (source: audit/scripts/claims/P4_lab_matrix.py) | 4082 B |
| 2026-09-23 03:27 | upload | `audit_scratch/scripts/P4_lab_matrix.py` | audit file (source: audit/scripts/claims/P4_lab_matrix.py) | 4342 B |
| 2026-09-23 03:27 | upload | `audit_scratch/scripts/F-98.py` | audit file (source: audit/scripts/claims/F-98.py) | 6090 B |
| 2026-09-23 03:27 | upload | `audit_scratch/scripts/P4_lab_trunc.py` | audit file (source: audit/scripts/claims/P4_lab_trunc.py) | 2052 B |
| 2026-09-23 03:28 | upload | `audit_scratch/scripts/P3_extras.py` | audit file (source: audit/scripts/claims/P3_extras.py) | 3283 B |
| 2026-09-23 03:28 | upload | `audit_scratch/scripts/P3_var.py` | audit file (source: audit/scripts/claims/P3_var.py) | 1027 B |
| 2026-09-23 03:29 | upload | `audit_scratch/scripts/p2p_F94_lab.py` | audit file (source: audit/scripts/claims/p2p_F94_lab.py) | 5471 B |
| 2026-09-23 03:29 | upload | `audit_scratch/scripts/p2p_F94_probe.py` | audit file (source: audit/scripts/claims/p2p_F94_probe.py) | 1131 B |
| 2026-09-23 | upload | `audit_scratch/scripts/F-05_F-06_lab.py` | audit file, batch P1-hpkr (source: audit/scripts/claims/F-05_F-06_lab.py) | 4.4 KB |
| 2026-09-23 | upload | `audit_scratch/scripts/F-03_F-72.py` | audit file, batch P1-hpkr (source: audit/scripts/claims/F-03_F-72.py) | 3.4 KB |
| 2026-09-23 | upload | `audit_scratch/scripts/F-72_detail.py` | audit file, batch P1-hpkr (source: audit/scripts/claims/F-72_detail.py) | 5.0 KB |
| 2026-09-23 03:30 | upload | `audit_scratch/scripts/p2p_F94_lab.py` | audit file (source: audit/scripts/claims/p2p_F94_lab.py) | 5401 B |
| 2026-09-23 03:31 | upload | `audit_scratch/scripts/p2p_F94_lab.py` | audit file (source: audit/scripts/claims/p2p_F94_lab.py) | 7324 B |
| 2026-09-23 03:41 | owner file appended (normal script behaviour) | C:/Users/user4/expLB/logs/falsify.status | the RF-14 skip test: a normal run of the owner's run_falsify.ps1 appended one line, 'SKIP already DONE; -Force to re-run'. Pre-test copy at audit_scratch/out/falsify.status.before_skiptest | 1 line |
| 2026-09-23 03:47 | upload | `audit_scratch/scripts/P6_lab_a.py` | audit file (source: audit/scripts/claims/P6_lab_a.py) | 9483 B |
| 2026-09-23 03:49 | upload | `audit_scratch/scripts/P6_lab_b.py` | audit file (source: audit/scripts/claims/P6_lab_b.py) | 11012 B |
| 2026-09-23 03:50 | upload | `audit_scratch/scripts/P6_lab_c.py` | audit file (source: audit/scripts/claims/P6_lab_c.py) | 4336 B |
| 2026-09-23 03:50 | upload | `audit_scratch/scripts/F-04.py` | audit file (source: audit/scripts/claims/F-04.py) | 5995 B |
| 2026-09-23 03:50 | upload | `audit_scratch/scripts/F-17.py` | audit file (source: audit/scripts/claims/F-17.py) | 4643 B |
| 2026-09-23 03:51 | upload | `audit_scratch/scripts/P6_lab_d.py` | audit file (source: audit/scripts/claims/P6_lab_d.py) | 4299 B |
| 2026-09-23 03:50 | upload | `audit_scratch/scripts/P7_lab.py` | audit file (source: audit/scripts/claims/P7_lab.py), batch P7-register-sarcasm | 12112 B |
| 2026-09-23 04:10 | upload | `audit_scratch/scripts/qa_p7.py` | audit file (source: audit/scripts/qa_p7.py) | 6718 B |
| 2026-09-23 04:11 | upload | `audit_scratch/scripts/qa_p7.py` | audit file (source: audit/scripts/qa_p7.py) | 6920 B |
| 2026-09-23 04:12 | upload | `audit_scratch/scripts/qa_p7.py` | audit file (source: audit/scripts/qa_p7.py) | 8980 B |
| 2026-09-23 04:12 | upload | `audit_scratch/scripts/qa_p7.py` | audit file (source: audit/scripts/qa_p7.py) | 9029 B |
| 2026-09-23 04:25 | upload | `audit_scratch/scripts/P12_lab.py` | audit file (source: audit/scripts/claims/P12_lab.py) | 13105 B |
| 2026-09-23 04:25 | upload | `audit_scratch/scripts/S-29_lab.py` | audit file (source: audit/scripts/claims/S-29_lab.py) | 3399 B |
| 2026-09-23 04:26 | upload | `audit_scratch/scripts/P12_lab2.py` | audit file (source: audit/scripts/claims/P12_lab2.py) | 4298 B |
| 2026-09-23 04:26 | upload | `audit_scratch/scripts/S-03_S-04_lab.py` | audit file (source: audit/scripts/claims/S-03_S-04_lab.py) | 4777 B |
| 2026-09-23 04:27 | upload | `audit_scratch/scripts/P12_lab3.py` | audit file (source: audit/scripts/claims/P12_lab3.py) | 559 B |
| 2026-09-23 04:28 | upload | `audit_scratch/scripts/S-04_size.py` | audit file (source: audit/scripts/claims/S-04_size.py) | 516 B |
| 2026-09-23 04:28 | upload | `audit_scratch/scripts/P12_lab4.py` | audit file (source: audit/scripts/claims/P12_lab4.py) | 1259 B |
| 2026-09-23 04:28 | upload | `audit_scratch/scripts/S-04_subst_lab.py` | audit file (source: audit/scripts/claims/S-04_subst_lab.py) | 3158 B |
| 2026-09-23 04:29 | upload | `audit_scratch/scripts/S-14_size.py` | audit file (source: audit/scripts/claims/S-14_size.py) | 454 B |
| 2026-09-23 04:30 | upload | `audit_scratch/scripts/S-14_lab.py` | audit file (source: audit/scripts/claims/S-14_lab.py) | 3493 B |
| 2026-09-23 04:31 | upload | `audit_scratch/scripts/S-14b_lab.py` | audit file (source: audit/scripts/claims/S-14b_lab.py) | 2465 B |
| 2026-09-23 04:55 | upload | `audit_scratch/scripts/P9_lab.py` | audit file (source: audit/scripts/claims/P9_lab.py), batch P9-findings-rest | 8807 B |
| 2026-09-23 04:55 | upload | `audit_scratch/scripts/P9_lab2.py` | audit file (source: audit/scripts/claims/P9_lab2.py), batch P9-findings-rest | 6020 B |
| 2026-09-23 04:55 | upload | `audit_scratch/scripts/P9_lab3.py` | audit file (source: audit/scripts/claims/P9_lab3.py), batch P9-findings-rest | 4678 B |
| 2026-09-23 04:55 | upload | `audit_scratch/scripts/P9_lab4.py` | audit file (source: audit/scripts/claims/P9_lab4.py), batch P9-findings-rest | 4423 B |
| 2026-09-23 04:55 | upload | `audit_scratch/scripts/P9_lab5.py` | audit file (source: audit/scripts/claims/P9_lab5.py), batch P9-findings-rest | 2486 B |
| 2026-09-23 04:55 | upload | `audit_scratch/scripts/P9_lab6.py` | audit file (source: audit/scripts/claims/P9_lab6.py), batch P9-findings-rest | 1902 B |
| 2026-09-23 04:55 | upload | `audit_scratch/scripts/P9_lab7.py` | audit file (source: audit/scripts/claims/P9_lab7.py), batch P9-findings-rest | 1703 B |

_Note (batch **P13-session**, 2026-09-23 ~04:25–04:31): the uploads `S-29_lab.py`,
`S-03_S-04_lab.py`, `S-04_size.py`, `S-04_subst_lab.py`, `S-14_size.py`, `S-14_lab.py` and
`S-14b_lab.py` above belong to claim batch P13-session. All are read-only analysis scripts;
they open owner files for reading only and write nothing on the lab PC._
| 2026-09-23 04:42 | upload | `audit_scratch/scripts/qa_p9_capbank.py` | audit file (source: audit/scripts/qa_p9_capbank.py) | 3707 B |
| 2026-09-23 04:42 | upload | `audit_scratch/scripts/qa_p9_capbank.py` | audit file (source: audit/scripts/qa_p9_capbank.py) | 4224 B |
| 2026-09-23 04:43 | upload | `audit_scratch/scripts/qa_p9_capbank.py` | audit file (source: audit/scripts/qa_p9_capbank.py) | 4224 B |
| 2026-09-23 09:31 | upload | `audit_scratch/scripts/M1_probe.py` | audit file (source: audit/scripts/claims/M1_probe.py) | 2506 B |
| 2026-09-23 09:32 | upload | `audit_scratch/scripts/F-107.py` | audit file (source: audit/scripts/claims/F-107.py) | 4248 B |
| 2026-09-23 09:33 | upload | `audit_scratch/scripts/M1_logs.py` | audit file (source: audit/scripts/claims/M1_logs.py) | 5252 B |
| 2026-09-23 09:34 | upload | `audit_scratch/scripts/M1_logs2.py` | audit file (source: audit/scripts/claims/M1_logs2.py) | 2347 B |
| 2026-09-23 09:35 | upload | `audit_scratch/scripts/M1_logs3.py` | audit file (source: audit/scripts/claims/M1_logs3.py) | 4156 B |
| 2026-09-23 09:36 | upload | `audit_scratch/scripts/M1_tf.py` | audit file (source: audit/scripts/claims/M1_tf.py) | 3112 B |
| 2026-09-23 09:37 | upload | `audit_scratch/scripts/M1_tf2.py` | audit file (source: audit/scripts/claims/M1_tf2.py) | 2198 B |
| 2026-09-23 09:38 | upload | `audit_scratch/scripts/F-108.py` | audit file (source: audit/scripts/claims/F-108.py) | 6493 B |
| 2026-09-23 09:39 | upload | `audit_scratch/scripts/F-44.py` | audit file (source: audit/scripts/claims/F-44.py) | 3994 B |
| 2026-09-23 09:39 | upload | `audit_scratch/scripts/F-44.py` | audit file (source: audit/scripts/claims/F-44.py) | 4075 B |
| 2026-09-23 09:39 | upload | `audit_scratch/scripts/qa_m2_kcheck.py` | audit file (source: audit/scripts/qa_m2_kcheck.py) | 544 B |
| 2026-09-23 09:41 | upload | `audit_scratch/scripts/M1_final.py` | audit file (source: audit/scripts/claims/M1_final.py) | 4357 B |
| 2026-09-23 09:42 | upload | `audit_scratch/scripts/M1_isarc.py` | audit file (source: audit/scripts/claims/M1_isarc.py) | 685 B |
| 2026-09-23 09:43 | upload | `audit_scratch/scripts/M1_span.py` | audit file (source: audit/scripts/claims/M1_span.py) | 696 B |
| 2026-09-23 09:45 | upload | `audit_scratch/scripts/M1_tf3.py` | audit file (source: audit/scripts/claims/M1_tf3.py) | 734 B |
| 2026-09-23 09:52 | upload | `audit_scratch/scripts/qa_m3_preserve_check.py` | audit file (source: audit/scripts/qa_m3_preserve_check.py) | 814 B |
| 2026-09-23 09:53 | upload | `audit_scratch/scripts/qa_m1_f107.py` | audit file (source: audit/scripts/qa_m1_f107.py) | 5342 B |
| 2026-09-23 09:53 | upload | `audit_scratch/scripts/qa_m1_f107.py` | audit file (source: audit/scripts/qa_m1_f107.py) | 3755 B |
| 2026-09-23 09:54 | upload | `audit_scratch/scripts/qa_m1_f107.py` | audit file (source: audit/scripts/qa_m1_f107.py) | 4023 B |
| 2026-09-23 09:59 | upload | `audit_scratch/scripts/qa_m2_ad4.py` | audit file (source: audit/scripts/qa_m2_ad4.py) | 1352 B |
| 2026-09-23 11:13 | upload | `audit_scratch/scripts/W5C_labA.py` | audit file (source: audit/scripts/claims/W5C_labA.py) | 13371 B |
| 2026-09-23 11:15 | upload | `audit_scratch/scripts/W5C_labB.py` | audit file (source: audit/scripts/claims/W5C_labB.py) | 11061 B |
| 2026-09-23 11:17 | upload | `audit_scratch/scripts/W5B_labA.py` | audit file (source: audit/scripts/claims/W5B_labA.py) | 10386 B |
| 2026-09-23 11:18 | upload | `audit_scratch/scripts/W5A_lab.py` | audit file (source: audit/scripts/claims/W5A_lab.py) | 17869 B |
| 2026-09-23 11:19 | upload | `audit_scratch/scripts/W5B_labB.py` | audit file (source: audit/scripts/claims/W5B_labB.py) | 6372 B |
| 2026-09-23 11:20 | upload | `audit_scratch/scripts/W5A_lab.py` | audit file (source: audit/scripts/claims/W5A_lab.py) | 19252 B |
| 2026-09-23 11:20 | upload | `audit_scratch/scripts/W5C_labC.py` | audit file (source: audit/scripts/claims/W5C_labC.py) | 7118 B |
| 2026-09-23 11:20 | upload | `audit_scratch/scripts/W5B_labB.py` | audit file (source: audit/scripts/claims/W5B_labB.py) | 6176 B |
| 2026-09-23 11:21 | upload | `audit_scratch/scripts/W5A_lab.py` | audit file (source: audit/scripts/claims/W5A_lab.py) | 21230 B |
| 2026-09-23 11:21 | upload | `audit_scratch/scripts/W5B_labC.py` | audit file (source: audit/scripts/claims/W5B_labC.py) | 7476 B |
| 2026-09-23 11:22 | upload | `audit_scratch/scripts/W5A_lab.py` | audit file (source: audit/scripts/claims/W5A_lab.py) | 22935 B |
| 2026-09-23 11:23 | upload | `audit_scratch/scripts/W5B_labD.py` | audit file (source: audit/scripts/claims/W5B_labD.py) | 1971 B |
| 2026-09-23 11:24 | upload | `audit_scratch/scripts/W5A_lab.py` | audit file (source: audit/scripts/claims/W5A_lab.py) | 22437 B |
| 2026-09-23 11:25 | upload | `audit_scratch/scripts/W5B_labE.py` | audit file (source: audit/scripts/claims/W5B_labE.py) | 1416 B |
| 2026-09-23 11:25 | upload | `audit_scratch/scripts/W5D_lab.py` | audit file (source: audit/scripts/claims/W5D_lab.py) | 13126 B |
| 2026-09-23 11:25 | upload | `audit_scratch/scripts/W5A_lab_extra.py` | audit file (source: audit/scripts/claims/W5A_lab_extra.py) | 2950 B |
| 2026-09-23 11:26 | upload | `audit_scratch/scripts/W5D_lab2.py` | audit file (source: audit/scripts/claims/W5D_lab2.py) | 7874 B |
| 2026-09-23 11:26 | upload | `audit_scratch/scripts/W5A_lab_extra2.py` | audit file (source: audit/scripts/claims/W5A_lab_extra2.py) | 2838 B |
| 2026-09-23 11:29 | upload | `audit_scratch/scripts/W5D_lab3.py` | audit file (source: audit/scripts/claims/W5D_lab3.py) | 3837 B |
| 2026-09-23 11:29 | upload | `audit_scratch/scripts/W5D_lab3.py` | audit file (source: audit/scripts/claims/W5D_lab3.py) | 4355 B |
| 2026-09-23 11:30 | upload | `audit_scratch/scripts/W5D_lab3.py` | audit file (source: audit/scripts/claims/W5D_lab3.py) | 4513 B |
| 2026-09-23 11:31 | upload | `audit_scratch/scripts/W5E_lab.py` | audit file (source: audit/scripts/claims/W5E_lab.py) | 8870 B |
| 2026-09-23 11:31 | upload | `audit_scratch/scripts/W5E_lab2.py` | audit file (source: audit/scripts/claims/W5E_lab2.py) | 5940 B |
| 2026-09-23 11:31 | upload | `audit_scratch/scripts/W5E_lab3.py` | audit file (source: audit/scripts/claims/W5E_lab3.py) | 3755 B |
| 2026-09-23 11:42 | upload | `audit_scratch/scripts/qa_w5d_M082.py` | audit file (source: audit/scripts/claims/qa_w5d_M082.py) | 3653 B |
| 2026-09-23 12:52 | upload | `audit_scratch/scripts/w4_null_lab.py` | audit file (source: audit/scripts/phase3/w4_null_lab.py) | 9848 B |
| 2026-09-23 12:52 | upload | `audit_scratch/scripts/w4_probe.py` | audit file (source: audit/scripts/phase3/w4_probe.py) | 886 B |
| 2026-09-23 15:22 | upload | `audit_scratch/scripts/W6G_lab_geom.py` | audit file (source: audit/scripts/claims/W6G_lab_geom.py) | 5703 B |
| 2026-09-23 15:35 | upload | `audit_scratch/scripts/W6G_lab_geom2.py` | audit file (source: audit/scripts/claims/W6G_lab_geom2.py) | 3806 B |
| 2026-09-23 17:02 | upload | `audit_scratch/scripts/run_scale_chain.ps1` | audit file (source: scale/scripts/run_scale_chain.ps1) | 12604 B |
| 2026-09-23 17:0x | upload | `audit_scratch/scripts/run_scale_chain.ps1` | staged DF-48 patch (source: scale/scripts/run_scale_chain.ps1) | 11,950 B |
| 2026-09-23 17:0x | write | `audit_scratch/backup_DF48_run_scale_chain.ps1` | backup of the lab original before the DF-48 patch, md5 155e6fc12c66f719f5b699a0684c6ff8 | 11,809 B |
| 2026-09-23 17:0x | **OWNER-AUTHORISED WRITE OUTSIDE audit_scratch** | `C:\Users\user4\expLB\scale\scripts\run_scale_chain.ps1` | DF-48 fix installed. Pre-check: target md5 155e6fc12c66f719f5b699a0684c6ff8 = the repo's pre-fix original, so the lab copy was unpatched. Backed up first (md5 verified identical). After: **37c12be11e726108e1220959ed70d5ce**, byte-identical to the patched repo copy. Patched block confirmed present at lines 198-212. Owner instruction: "solve this - the lab copy of run_scale_chain.ps1 still needs the DF-48 patch". | 11,950 B |

---

## 2026-09-23 ~20:57 — DRIVE WRITE (owner-requested): `run-premise/` backed up

**Not a lab-PC write.** No file under `C:\Users\user4\` was created, modified or deleted. This is a
one-way `rclone copy` from the lab PC **into a NEW Drive folder**, which the owner authorised
generally on 2026-09-22 ("rclone copy of lab-only results into a NEW Drive folder; never sync/move/
delete; never overwrite existing Drive files") and explicitly requested on 2026-09-23.

- **Binary:** `C:\Users\user4\expLB\tools\rclone.exe` (the owner's own rclone; not on the
  non-interactive PATH, so it is invoked by full path).
- **Source:** `C:\Users\user4\expLB\run-premise` — 100 files, 773.992 KiB.
- **Destination:** `tamara:expLB/_labpc_only_backup_2026-09-23/run-premise` — **a new folder.**
  Confirmed non-existent before the copy: `rclone lsf` returned
  *"error in ListJSON: directory not found"*. **Nothing could be and nothing was overwritten.**
  The existing `_labpc_only_backup_2026-09-22` folder was not touched.
- **Command:** `rclone copy <src> <dest> --create-empty-src-dirs -v` (copy only — no sync, move,
  delete, purge or dedupe at any point).
- **Verification:** `rclone check <src> <dest>` → **"0 differences found", "100 matching files"**
  (md5 on both sides). `rclone size <dest>` → **100 objects, 773.992 KiB (792,568 bytes)**.

**Why this folder first.** `run-premise/` holds the project's strongest result — the pooled-training
comparison, +0.1254 AUROC under leave-one-cluster-out — and RF-10 established it existed in exactly
one place, on a shared multi-user machine.

**Still unbacked on the lab PC** (RF-10, measured 2026-09-23): `run-cap/` 67 files / 186.7 MB,
`run-cap-ctrl/` 44 files / 108.8 MB, `logs/` 243 files / 648.7 MB, plus ~133 loose analysis scripts in
`C:\Users\user4\expLB\`. Roughly 945 MB in total; the same `rclone copy` pattern would take a few
minutes. **The owner asked only for `run-premise` and only `run-premise` was copied.**

---

## 2026-09-24 · Phase 6 (6D-1 / 6D-2 / the unresolved draft claims)

| # | remote path | bytes | md5 | why |
|---|---|---:|---|---|
| 1 | `C:\Users\user4\expLB-resplit\phase6\p16_fig2_axes.py` | 4,051 | `d22db32341de510ecfd44a36969138f3` | rebuilds the collaborator's Figure 2 geometry from `run-kd\vectors_s3` to resolve D-057 / D-065 / D-154. Reads only; writes nothing. Uploaded with `labpc_put.py`, md5-verified both sides. |

**Everything else this session was read-only**, via `ssh -o BatchMode=yes` + `powershell -NoProfile`:
`run-crossgen\passB_fine\`, `run-crossgen\passP\`, `run-kd\vectors_s3\`, and directory listings under
`C:\Users\user4\expLB\`. No file under `C:\Users\user4\expLB\` was created, modified or deleted.

**One negative result worth recording.** A recursive search of `C:\Users\user4\expLB` for
`results_table.csv` / `results.csv` / `scores.csv` and for any `XXXX-9-llama*` directory returned
**nothing**. The pipeline's own results tables — the artifact Appendix G's four-decimal reproduction
claim would be checked against — are **not on the lab PC either**, not merely uncommitted.

### 6D-1 · the `pc:sycophancy` run (2026-09-24)

| # | remote path | what |
|---|---|---|
| 2 | `C:\Users\user4\expLB-resplit\phase6\p16_passBfine_passP.py` | 4,136 B, md5 `8b693ffc4e268ee0103f3ef98bebf810` — read-only reader for 6D-2 |
| 3 | `C:\Users\user4\expLB-resplit\phase6\p16_syc_result.py` | md5 `b1fa5998625331de3e82922435e70ecb` — read-only reader for the 6D-1 output |
| 4 | `C:\Users\user4\expLB-resplit\phase6\run-crossgen\` | **156 files, 1.8 MB** — the 6D-1 crossgen output: 6 matrices (`mean`/`last` × L20/40/44), `summary.json`, `significance.json`, `splits.json`, 144 CSVs |

**The 6D-1 command, for the record** (ran 1m50s, CPU only):

```
.venv-analysis\Scripts\python.exe src\assistant_axis_ndif\27_crossgen.py
  --run_dir   C:\Users\user4\expLB\run-coords\passB
  --datasets  instructed-deception harm-pressure-choice harm-pressure-knowledge-report
              harm-pressure-kr-matched convincing-game insider-trading ai_liar
              sandbagging_v2 roleplaying mask sycophancy_mt
  --bank      C:\Users\user4\expLB\run-paper16\axes_vectors_s3_fp32.pt   --bank_name paper16
  --out_root  C:\Users\user4\expLB-resplit\phase6\run-crossgen
  --features  raw random datapca length pc:deception axes axes_resid pc:unified
              pc:unified@2 random@2 datapca@2 pc:sycophancy
  --layers 20 40 44   --poolings mean last   --probes paper
```

Three rules observed and worth restating because each is load-bearing here:
`--datasets` passed **explicitly** so `harm-pressure-kr-matched` is included (the default omits it);
`--features` keeps **all four defaults** `raw`/`random`/`datapca`/`length` beside the persona
features; and `--out_root` contains a component named exactly `run-crossgen`, which the script
refuses to run without. Output went to `expLB-resplit`, never to `expLB`.

**Self-check that the run is comparable to `kd11`:** raw 0.6623, datapca 0.6651, length 0.5555 and
`pc:unified` 0.6715 at L40/mean/`paper` reproduce `run-crossgen/kd11` to four decimals, so the 80/20
split regenerated identically and `pc:sycophancy` can be read against that store directly.

**No long-running job was scheduled**, so the `schtasks` SYSTEM-task rule did not come into play.

## 2026-10-06 — Drive login repair (owner-approved, option B)

Both Drive logins had expired (laptop `invalid_grant`, lab `401 Invalid Credentials`; lab uploads
last succeeded 2026-09-29 17:17). The owner re-authorized on the laptop and approved copying that
token to the lab PC. **This is an owner-approved exception to "never modify the owner's files".**
The lab config's `type`, `client_id`, `client_secret`, `scope` and `team_drive` were checked
identical to the laptop's (SHA-256 fingerprints) before the edit, so only the `token` line changed.

| when (+06) | action | path on lab PC | purpose | size |
|---|---|---|---|---|
| 2026-10-06 17:26 | mkdir | `audit_scratch\rclone\` | holds the backup below | — |
| 2026-10-06 17:26 | write | `audit_scratch\rclone\rclone.conf.bak-20261006` | copy of the owner's `rclone.conf` before the edit (dead token), md5 68DF65F9… | 659 B |
| 2026-10-06 17:26 | modify | `C:\Users\user4\AppData\Roaming\rclone\rclone.conf` | replaced only the `token =` line with the laptop's fresh token; other 7 lines unchanged; verified `rclone lsf tamara:expLB` exit 0 at 17:26:24 | 659 → 689 B |

rclone rewrites this file itself on every token refresh, so its size and md5 will keep changing.

## 2026-10-06 — full backup of the lab PC to Drive (owner request)

Owner asked to back up everything from the lab PC to Drive before maintenance. Owner choices:
pack the two `run-crossgen` trees as .tar files, copy the rest file by file, include `ATresearch`
without its `.venv`. Destination is the NEW folder `tamara:expLB/labpc-backup-2026-10-06/`; only
`rclone copy --immutable` and `rclone check`. Script: `audit/scripts/labpc_backup_20261006.ps1`.

| when (+06) | action | path on lab PC | purpose | size |
|---|---|---|---|---|
| 2026-10-06 17:49 | upload (mistake) | `audit_scratch\backup\scripts$f` | a quoting slip in my upload loop wrote both scripts to this one literal name | 1854 B |
| 2026-10-06 17:50 | delete | `audit_scratch\backup\scripts$f` | removed the stray file above (my own) | — |
| 2026-10-06 17:50 | mkdir | `audit_scratch\backup\`, `backup\scripts\` | backup workspace | — |
| 2026-10-06 17:50 | upload | `audit_scratch\backup\scripts\labpc_backup_20261006.ps1` | the backup script, md5 c0714fce… == local | 8394 B |
| 2026-10-06 17:50 | upload | `audit_scratch\backup\scripts\labpc_register_backup_20261006.ps1` | task registration, md5 780e7ad1… == local | 1854 B |
| 2026-10-06 17:50 | register task | `Backup-LabPC-20261006` (user4/S4U, 30-min repeat for 3 days + at-startup) | runs the backup; script exits at once after ALL DONE | — |
| 2026-10-06 17:51→ | write (by the task) | `audit_scratch\backup\pack\` (crossgen .tar files, md5 manifests, log snapshot, package lists, task XMLs), `backup\logs\`, `backup\backup.status` | backup staging and logs | ~2.7 GB expected |

**To remove when done (owner's call):** the task `Backup-LabPC-20261006` and `audit_scratch\backup\pack\`.
| 2026-10-06 18:24 | upload (replace own file) | `audit_scratch\backup\scripts\labpc_backup_20261006.ps1` | fix: exclude `SPACE_RESERVE_50GB.bin` (a 50 GB disk-space placeholder in the home folder, not data) from the home-loose-files copy; md5 bd975771… == local | 8.6 KB |
| 2026-10-06 18:25 | stop process | rclone pid 8420 (our own `copy … home-loose-files`) | it was uploading the 50 GB placeholder (22 of 50 GiB sent; Drive keeps nothing from an unfinished upload). The 75 real loose files had already been copied. | — |

**Backup result (2026-10-06 19:51:33, `ALL DONE`).** `tamara:expLB/labpc-backup-2026-10-06/` holds
8,467 files, 83.36 GiB. Per folder, the Drive count equals the lab copy log's "Copied (new)" count
exactly (no duplicates, nothing missing): expLB 7,125 · expLB-resplit 674 · packed 329 (24 crossgen
.tar files + 2 per-file md5 manifests of 205,430 and 9,810 files + frozen `expLB\logs` + venv
package lists + 54 scheduled-task XMLs) · audit_scratch 157 · expLBactivationLAYER22 99 ·
home-loose-files 74 · ATresearch 3 (its `models\` holds only an empty folder; the 4.5 GB is the
excluded `.venv`) · share_deceptive 3 · share_L20_shared 3. Every folder passed `rclone check
--one-way` (md5) with missing=0 differ=0. The lab's own KD and crossgen syncs also succeed again
since the token fix (19:16 and 19:40, exit 0).
Still on the lab, owner's call: task `Backup-LabPC-20261006` (fires every 30 min until 2026-10-09 but
exits at once after ALL DONE) and `audit_scratch\backup\pack\` (2.67 GB, already on Drive).

**2026-10-06 22:16 — `D:\user4fina` added to the backup (owner confirmed the files are theirs).**
Not covered by the main pass (that scoped `C:\Users\user4` + `D:\expLB*`). Copied to
`tamara:expLB/labpc-backup-2026-10-06/user4fina/`: 72 files, 450 KiB (`code\` 3, `results\` 11 training
logs, `Qwen2.5-72B-Instruct\` 57 = the never-run 09-29 xmodel code copy, `ndif_test.py`,
`working_packages.txt`; `data\` and `secrets\` are empty). `rclone check --one-way`: missing=0 differ=0.
Excluded: `ndif-env\` (a 9-package venv; its package list + pyvenv.cfg are in `user4fina/_meta/`) and
`model.safetensors-00001-of-00011.safetensors` (4.9 GB, shard 1 of 11 of downloaded weights).
Written on the lab: `audit_scratch\backup\user4fina_meta\` (2 files), `audit_scratch\backup\logs\*user4fina*`.
