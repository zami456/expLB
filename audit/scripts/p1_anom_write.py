"""Writes audit/phase1/anomalies.csv (Phase 1 Task B). The rows are curated from the evidence files in
audit/phase1/anomalies_detail/ (units.tsv, cited_paths.tsv, logs_mentions.tsv, duplicates_lab_output.txt,
conflicts_lab_output.txt), audit/phase1/fmap_all.csv, audit/experiments.csv and the lab reads logged in
audit/phase1/anomalies.md. Columns as required by audit/phase1/INVENTORY_INSTRUCTIONS.md.
Usage: PYTHONIOENCODING=utf-8 python audit/scripts/p1_anom_write.py
"""
import csv, os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase1", "anomalies.csv")
L, D, A, P, T = "LAB", "DRIVE", "ASIX", "PCOOD", "LAPTOP"
R = []


def row(cat, path, sources, detail, fids="", sev="LOW"):
    R.append([cat, path, sources, detail, fids, sev])


# ---------------- 1. ORPHANS ----------------
row("orphan", "run-crossgen/passA/ (root + banks/{hf100,hf275_loose,hf275_strict,own100,own100_strict,paper16_famcentred,paper16_shared})",
    "LAB,DRIVE",
    "OR-01. 92,184 files, 09-18 14:00 -> 09-19 14:38 (lab mtimes): cross-generalisation on the 80-layer passA store, 80 layers x 2 poolings x seeds 0-4+42 x 8 bank folders; complete (logs/crossgen_full_A; 09-22 re-run exited 1 on the HP-KR split mismatch, so these are the originals). No F-entry, no doc reports a number from it; F-91 only lists its layer coverage, memory crossgen-layer-coverage.md only the store. VERIFIED (units.tsv mention_level NAME only; grep 'run-crossgen/passa' = 0 docs). Task A O44.",
    "F-91 (coverage only)", "MED")
row("orphan", "run-crossgen/passC/ (root + 7 banks)", "LAB,DRIVE",
    "OR-02. 6,936 files, 09-18 15:58-16:21, 6 layers x 6 seeds; status DONE. Memory crossgen-layer-coverage.md says the passC crossgen arm was added 09-18 but never reports its results; no F-entry. Its significance.json is byte-identical to run-crossgen/kd11/passC/significance.json (see DU-03). VERIFIED. Task A O45.",
    "F-91 (coverage only)", "MED")
row("orphan", "run-crossgen/passB_fine/ (root + banks hf275_strict, own100, paper16_shared) and run-crossgen/passP/ (root)", "LAB,DRIVE",
    "OR-03. 5,777 files; complete (crossgen_full_B_fine / _P logs 'wrote summary.json, significance.json'). docs/thesis/persona-coordinates-70b-results.md sec 15.5 (written 09-17) lists both as 'still running': passP = prompt-side activations for the construct-validity control of sec 11.7, passB_fine = layers 37-47 to locate the peak to +-1. No later doc reports either outcome. VERIFIED (grep 'passp' hits only memory crossgen-layer-coverage.md). Task A O46.",
    "", "MED")
row("orphan", "run-crossgen/ root + run-crossgen/banks/* (21 layers x 6 seeds; incl. label banks hf275_loose, hf275_strict, own100_strict)", "LAB,DRIVE (git: L40 only)",
    "OR-04. Folders are named in docs, but outcomes are reported at L40 only (F-17; thesis doc; memory crossgen-80-20-retrain). F-91 states all 21 layers x 6 seeds exist ('figure-selection decision') and calls this store 'passB/banks/...' although the path is run-crossgen/banks/. The three label banks (hf275_loose/strict, own100_strict) have geometry numbers in memory role-label-rules-and-banks.md but no reported transfer result. VERIFIED (FINDINGS.md:3884; units.tsv).",
    "F-17, F-91", "MED")
row("orphan", "run-coords/confounds_L44_last.json; run-coords/p1_L44_last_lengthctl.json; run-coords/q5_semantics_lengthctl.json", "LAB,DRIVE (not git)",
    "OR-05. Length-only / length-controlled re-runs (L44, last token, mms), 09-16 19:36-19:38, produced by lab-only loose drivers C:/Users/user4/expLB/run_confounds.ps1 and run_lengthctl.ps1 (9_persona_probes.py --features ... length; 18_q5_semantics.py) with bank run-paper16/axes_vectors_s3_fp32.pt; logs confounds.log, lengthctl.log, lengthctl_q5.log. No doc names the files; memory response-length-transfers-on-its-own.md (09-16) reports the matching measurement (length-only off-diagonal 0.553) without a path, and was last modified 09-16 13:39Z = 19:39 +06, one minute after lengthctl_q5.log ended -> INFERRED source. VERIFIED file/driver facts (p1_anom_labgrep.ps1, p1_peek cat). Task A O26.",
    "", "MED")
row("orphan", "run-coords/{probes_5sets,probes_L20,smoke,smoke2,zeroshot_paper16,splits,conf,bank_run_axes.pt,bank_run-paper16_axes_vectors_s3.pt}.json", "LAB,DRIVE,ASIX",
    "OR-06. 09-15/16 pilot and first Q1/Q2 outputs; no doc names them (the thesis-doc appendix lists the later files only). Task A VERIFIED their 'bank' key is axes_vectors_s3.pt = the pre-fp32 paper16 bank (see CV-01). Task A O16/O17.",
    "", "LOW")
row("orphan", "smoke/pilot outputs: run-coords/passC_smoke, run-coords/passP_smoke, run-coords/q8_trained_SMOKE.json, run-premise/falsify_smoke, run-crossgen/smoke, lab smoke/ (09-12), laptop aa-smoke/ + aa-smoke-win/ (09-10/11)",
    "LAB,DRIVE / LAPTOP", "OR-07. Pipeline smoke tests; named in no doc except generically ('smoke'). aa-smoke* exist only on the laptop (38 files, 64 MB); lab smoke/ is not on Drive. passP_smoke chunk_0000 files are byte-identical to passP's (DU-06). VERIFIED (units.tsv).",
    "", "LOW")
row("orphan", "C:/Users/user4/expLBactivationLAYER22/activations/ (cache/00, cache/01) + NOTE.txt (empty)", "LAB only",
    "OR-08. 97 .pt + budget_memory.json, 361 MB, 09-04 14:43 -> 09-12 18:29; not on Drive, not in git; no doc (only the generic word 'activations' matches). Provenance NOT-FOUND; INFERRED collaborator acts-cache layout (Task A O87). Data-safety: single copy.",
    "", "LOW")
row("orphan", "run-paper16/axes_vectors_s3_fp32.pt, axes_vectors_all_fp32.pt, vectors_s3_fp32/, vectors_all_fp32/", "LAB,DRIVE",
    "OR-09. The fp32 paper16 bank is recorded (matrix 'bank' field) in every kd11 root and run-crossgen root matrix, yet NO doc names these files: docs name only the pre-fp32 files (thesis appendix 'run-paper16/axes_vectors_{s3,all}.pt'; FINDINGS 'run-paper16/axes_vectors_s3.pt'). A reader following the docs lands on the old bank. VERIFIED (grep '_fp32' = 0 docs; p1_anom_snip_bank.py). See CV-01.",
    "F-03, F-04, F-13, F-17, F-29, F-39, F-58, F-61, F-65, F-74, F-98", "HIGH")
row("orphan", "other unnamed bank files: run-paper16/axes_vectors_s3_fp32_famcentred.pt; run/axes_patched.pt; run-kd/axes_vectors_all.pt; run-paper16-shared/axes_vectors_all.pt; run-hf275/{axes_hf100,published_assistant_axis}.pt; run-cap/axes_all.pt; run-scale/{3b,8b}/axes_{3b,8b}.pt",
    "LAB (+DRIVE except famcentred and run-cap/axes_all)",
    "OR-10. Named in no doc (some reached via fmap: axes_all -> F-82, axes_3b/8b -> F-37). Data-safety: axes_vectors_s3_fp32_famcentred.pt and run-cap/axes_all.pt exist only on the lab (on_drive=0). run/axes_patched.pt (09-15 12:32) is read by the lab-only loose script bankchk.py. VERIFIED (units.tsv, census).",
    "F-37, F-82", "LOW")
row("orphan", "run/vectors_fp32/ (1 of 100 own100 vectors, 09-16 01:51)", "LAB,DRIVE",
    "OR-11. Abandoned fp32 re-extraction of own100 (logs/rebuild_banks.log 'own100 pass 1 (have 0)', rb_own100.log). No doc. Consequence: own100 and own100_strict banks stay on the old bf16-reduction vectors while paper16 moved to fp32 (CV-11). VERIFIED (census; log_index). Task A O05 (partial).",
    "F-90", "MED")
row("orphan", "data_syco/ (1 parquet, 09-15)", "LAB only",
    "OR-12. Input dataset for 10_sycophancy_ndif.py / 13_sycophancy_multiturn.py (their --data_dir default); named in no doc; not on Drive. VERIFIED (git grep; census).", "F-15", "LOW")
row("orphan", "isarc_a.parquet (lab repo root, 09-22 04:23)", "LAB only",
    "OR-13. Not named in any doc; fmap F-106 INFERRED it is the iSarcasmEval Arabic check. Not on Drive. VERIFIED file facts.", "F-106", "LOW")
row("orphan", "cap/run-sarcasm-natural/ (rows.jsonl, figlang2020_reddit.jsonl, baselines.json)", "ASIX (+LAB rows.jsonl)",
    "OR-14. Committed data behind F-106, but no doc names the folder: F-106 cites a session-scratchpad CSV instead (GH-15). baselines.json and figlang jsonl are git-only. VERIFIED.", "F-106, F-107", "LOW")
row("orphan", "run-cap/export/cap_fig_data.json (thesis fig5 data)", "LAB,ASIX (not Drive)",
    "OR-15. Written by cap/scripts/export_cap_fig.py; no doc names it (figures/thesis is named). VERIFIED.", "", "LOW")
row("orphan", "DD/results/XXXX-9_lr__repe_honesty__plain__llama-70b-3.3__20260712_053340/ (4 PNG)", "PCOOD",
    "OR-16. His 07-12 Mean-Probe run (plots only); named in no doc, including his own summaries. VERIFIED (units.tsv). Task A C01.", "", "LOW")
row("orphan", "DD/results/loo_L20, loo_L40, persona_ood_mod_L{20,40,44}, persona_ood, persona_ood_L40", "PCOOD",
    "OR-17. Named only in his own summary.md files; the draft does not use them: loo_* is the pooled leave-one-dataset-out baseline the draft says it did 'not compare' (RF-3); persona_ood_mod_* is his negative result 'more rollouts per persona did not help' (ext_axis1 L40 0.9158 -> 0.9066 per his summary, DOC-ONLY); persona_ood/_L40 supply only the 0.103 figure. Source: audit/phase0/draft_report.md sec 8 + units.tsv. His L44 / followup / positions folders ARE used by the draft (not orphans).",
    "F-97 (reads persona_ood_ext)", "MED")
row("orphan", "single-copy results outside lab/Drive: run-paper16/scores_405b/ (git+laptop), logs/judge_405b*.log (laptop only), run-coords/regroup_hf.json (git only), laptop run-coords/{ai_liar,roleplaying}/items.pt, laptop run-hf275/vectors_100/",
    "ASIX / LAPTOP", "OR-18. Not orphans by doc (memory judge-choice-barely-matters and regrouping-cannot-replace-personas describe the first three), but no lab or Drive copy exists. vectors_100 is byte-identical to 100 of the lab's vectors_275 files (DU-06). VERIFIED (census; duplicates_lab_output.txt).",
    "", "LOW")
row("orphan", "lab logs/ never named in a doc", "LAB only",
    "OR-19. 203 of 242 lab logs are named in no doc (39 are); 43 of the 203 are cited in fmap only. Unnamed experiment logs include extract_P0{a,b,c}_*.log (09-18 passA/C sycophancy add, passP, passB_fine), rb_paper16_{s3,all}.log + rebuild_paper16_vectors_s3_fp32.log (the RF-12 fp32 rebuild), rb_own100.log, alpaca_{3b,8b}.log + mask_{3b,8b}.err (no driver anywhere, Task A O75), shard{0,1,2}.log + vectors*.log (own100 100-role run), newroles.log, cap_ctrl_{gen,judge,vectors,crossgen}.log. Full list: anomalies_detail/logs_mentions.tsv (p1_anom_logs.py).",
    "", "LOW")

# ---------------- 2. GHOSTS ----------------
row("ghost", "run-crossgen/kd11/banks/kd_s23/significance.json - the 3,276-test / 21-layer version of 09-20", "none (git 468 tests; lab+Drive 5,544)",
    "GH-01. Read by F-34 ('best q 0.130', '9 of 3,276 clear') and by thesis fig2 (kd/results/fig_data.json records source=this path, n_tests=3276; RF-7). Exists in no source; only its reduction fig_data.json survives. Diffs/CIs reproduce in later versions, q-values do not. VERIFIED (fmap F-34, F-57; experiments_notes 4a).",
    "F-34, F-57", "HIGH")
row("ghost", "run-crossgen/kd11/significance.json - the 384-test version (09-19 20:10 run 1)", "none (git 4,032 tests; lab/Drive 4,788)",
    "GH-02. F-04's '1 of 384 clears' and q=0.931 cannot be read from any existing file; overwritten the same evening. diff and CI (+0.0147 [-0.170, +0.187]) reproduce in later versions. VERIFIED (fmap F-04).", "F-04", "MED")
row("ghost", "run-crossgen/kd11/banks/cap_{s23,all}/significance.json - the 792-test version", "none (lab/Drive 5,544)",
    "GH-03. F-67's q-values (0.172 / 0.013) come from a file rewritten at least three times; current q 0.227 / 0.0163, verdicts unchanged (fmap).", "F-67", "MED")
row("ghost", "status-file gate lines quoted by findings", "LAB (overwritten)",
    "GH-04. Quoted gate lines no longer exist in logs/{passAC11,cap_layers,premise_layers,premise_extras,cap_seeds,cap_ctrl_layers,falsify,cap_controls}.status: the 09-22 re-run truncated them (RF-14). F-104's quote survives in logs/sarc_pilot.status.pre-e2e-2026-09-22. Appended files (hpkr_topup, cap_chain, scale_*, kd_s23*) kept theirs. VERIFIED (fmap F055-F081, F082-F109).",
    "F-72, F-75, F-77, F-81, F-82, F-83, F-87, F-94, F-95, F-104", "MED")
row("ghost", "numbers printed to a console and never saved (inputs survive)", "none",
    "GH-05. 42 of 109 F-entries have script_writes_file = n in fmap_all.csv (e.g. F-01, F-02, F-06 ortho.py, F-10, F-12 perm.py, F-18 cmp_banks.py, F-27, F-64, F-66, F-69/F-71 slice counts, F-73, F-74, F-76, F-77, F-80, F-82, F-85's 0.811, F-87, F-89-F-95 tables, F-96, F-98-F-103, F-107, F-108). F-65's gate cosines are in no log. Every analysis after 09-21 23:01 was inline Python with no script kept (fmap F082-F109). Not missing files, but no file holds the number; Phase 2 must recompute.",
    "see detail", "MED")
row("ghost", "F-17's random-2-d control numbers (+0.056 / -0.44 / corr -0.76)", "none",
    "GH-06. Stated in F-17 and memory matched-control-kills-the-gain.md; kd/scripts/ceiling_null.py only cites -0.76 in a comment. No output file. VERIFIED (fmap F-17).", "F-17, F-74", "MED")
row("ghost", "F-107 projection outputs (deception axis 0.54-0.58 on natural sarcasm; sycophancy 0.69/0.71)", "none (inputs: run-coords/sarcasm/sarcasm_natural, run-kd/axes_*_s23.pt)",
    "GH-07. No script, no output; memory sarcasm-gate3-deception-axis-is-random.md repeats the numbers. Inputs survive (lab+Drive). VERIFIED (fmap F-107).", "F-107", "MED")
row("ghost", "F-58's first 3B/8B matrices (harm-pressure-choice set, 09-20 22:19-22:21)", "none (values only in logs/crossgen_scale_{3b,8b}.log)",
    "GH-08. Overwritten 23:38-23:40 by the mask re-run (F-63). F-58 is superseded, so the loss matters only for the claim history. VERIFIED (fmap F-58).", "F-58, F-63", "LOW")
row("ghost", "run-scale/smoke3b (gen_batch sweep)", "none",
    "GH-09. Cited by F-21 (FINDINGS.md:387); not on lab, Drive, git or laptop. VERIFIED (cited_paths.tsv).", "F-21", "LOW")
row("ghost", "deleted scale-extraction artifacts (8B chunks 17:00-17:16; 18:18 --prepare_only probe; 18:04 full-corpus 8B chunks and items.pt; stale HPC items.pt + 217 chunks)",
    "none (logs/scale_coords_8b.log only, lab)", "GH-10. Deleted by restarts/fixes on 09-20; only log lines remain. Engineering notes. VERIFIED (fmap F-40, F-44, F-45, F-47).",
    "F-40, F-44, F-45, F-47", "LOW")
row("ghost", "cap persona bank texts v1 (pre-F-32) and the F-32 bank; the two external reviews (Opus, Gemini)", "none as files (F-32 text only inside git cap/docs/REVIEW_PACKET*.md)",
    "GH-11. Overwritten 09-20 22:51 by the v4 bank; F-28 counts cannot be recomputed; reviews not located. VERIFIED (fmap F-28, F-32).", "F-28, F-32", "LOW")
row("ghost", "cap/questions/cap_questions_v2_draft.jsonl - the version F-36 describes (12+8, third person)", "none (preserved file: 10/10, 10 first-person)",
    "GH-12. Edited after F-36 (mtime 18:38/18:39). VERIFIED (fmap F-36).", "F-36", "LOW")
row("ghost", "passA/passC HP-KR state of 09-20 (98 rows, 89/9)", "none (logs/kd_passAC.log only)",
    "GH-13. Topped up to 2,139 rows 09-21, kr-matched views rebuilt 09-22 15:47. VERIFIED (fmap F-20).", "F-20", "LOW")
row("ghost", "F-14's other 4 scp-truncated files and the 271-file size diff; F-15's '33 labels, a third wrong' data; F-24's buggy smoke_multiturn.py + 40 GB HF cache blobs; F-26's smoke-time persona instructions; F-33's single-role probe (15:49)",
    "none", "GH-14. Small evidence items that no source holds. VERIFIED absent in the census (fmap).", "F-14, F-15, F-24, F-26, F-33", "LOW")
row("ghost", "F-106 inputs/outputs: session-scratchpad sarcasm_natural_reddit.csv; sarcasm_detection_shared_task_{reddit,twitter}_training.jsonl; Twitter-arm baselines 0.6252/0.7826", "none",
    "GH-15. Committed equivalents exist (cap/run-sarcasm-natural/rows.jsonl, figlang2020_reddit.jsonl, baselines.json = Reddit arm only). VERIFIED (cited_paths.tsv; fmap F-106).", "F-106", "LOW")
row("ghost", "run logs of hand-run analyses (F-85 six/, F-86, F-90, F-98..F-101, F-103, F-108, F-109; the 09-17 01:00 significance job)", "none",
    "GH-16. The output JSONs exist; no log records the run (fmap). Task A: the 09-17 01:00 significance job (run-coords/significance.json) has no surviving log; sig.log/sig.status belong to the nulls job. VERIFIED.", "F-85, F-86, F-90, F-98, F-99, F-100, F-101, F-103, F-108, F-109", "LOW")
row("ghost", "his inputs: activations/pooled{,_L40,_L44,_followup_L20}, activations/token_proj_L*, 'persona vectors/vectors_s3/', 'persona knowledge denial modified/', *_all_tokens.json", "none (his disk)",
    "GH-17. Cited by his summaries (loo_L20:121, persona_ood_L40:7, persona_ood_mod_L40:5,104, ext_L44:94, positions_L40:100-105, XXXX-9-llama-all-followup:60). Not in any audited source (accepted scope RF-2). INFERRED from his docstring: 'persona vectors/vectors_s3' = the run-paper16-shared run (md5 not checkable). Every persona CSV on his branch depends on them.",
    "", "MED")
row("ghost", "his base-probe outputs XXXX-9-llama-all{,-L40,-L44}/{detector.pt,results_table.csv,scores.json}", "none (plots only in git)",
    "GH-18. The draft's base-probe rows and 'reproduced to 4 d.p.' rest on these (draft_report sec 8). VERIFIED absent (git ls-tree; census).", "", "MED")
row("ghost", "draft: paper/make_numbers.py; data + script for Figs 2 and 3; the LaTeX source", "none",
    "GH-19. Cited or implied by the draft PDF; found on no branch and no lab/Drive path (draft_report.md; cited_paths.tsv). VERIFIED.", "", "MED")
row("ghost", "'INLP experiment' (his commit 148ddeb)", "none",
    "GH-20. Mentioned only in a commit message; no file anywhere (Task A C20; branch_compare 2c). VERIFIED absent.", "", "LOW")
row("ghost", "hf_axis_check.py (memory persona-axis-diagnostics.md:58)", "none",
    "GH-21. Script said to have run on the laptop; in no census (laptop, lab loose scripts, git). Memory crossgen-80-20-retrain.md also says the earlier-design crossgen files were deleted everywhere - consistent with NOT-FOUND. VERIFIED absent.", "", "LOW")
row("ghost", "RESOLVED - not ghosts: exporter of paper16_L20.pt/deceptive_L20.pt; make_view.py / make_view_ac.py; XXXX-9-llama-all-L22/-L40 yaml; FULL_L20/40.json; scale_{3b,8b} 'mean_L40' paths",
    "LAB", "GH-22. Exporter = lab loose C:/Users/user4/build_share_L20.py (09-18 16:23) and build_deceptive_L20.py (16:36) (NOT-FOUND in git only; Task A). make_view*.py are lab loose files (RF-8). The yaml configs exist on D:/expLB-acts. 'FULL_L20/40.json' is shorthand for FULL_L20.json + FULL_L40.json (both exist). FINDINGS.md:2029 'scale_{3b,8b,70b}/matrices/mean_L{16,16,40}' is a paired brace; the intended L16 files exist. VERIFIED (census).",
    "", "LOW")

# ---------------- 3. DUPLICATES ----------------
row("duplicate", "expected layout: one logical file in several places", "all",
    "DU-01. 148,820 distinct contents; 144,971 occur more than once (494,804 files). 124,682 groups are one logical path copied: lab+Drive 86,046; lab + 2 Drive copies 32,815 (Drive keeps run-crossgen/kd11 twice, as crossgen/kd11 and kd/crossgen11, and own100 twice, as own100/ and own100/run/); lab+git+2 Drive 4,659; lab+git+Drive 472; git+Drive only 242 (figures); lab+Drive+laptop 177; lab+git only 137; asix=his branch 75+4 (vendored upstream code); Drive+laptop only 52 (the old paper16 vectors, RF-12). Empty files (24) and <100 B contents (8 groups) excluded. VERIFIED (p1_anom_dupes_lab.py -> anomalies_detail/duplicates_lab_output.txt).",
    "", "LOW")
row("duplicate", "run-crossgen/**/csv/*.csv identical across banks/passes", "LAB,DRIVE,ASIX",
    "DU-02. 19,408 groups / 199,402 files. Expected: each per-cell CSV is the train-one-test-all grid plus 2 bank-specific zero-shot rows (VERIFIED by diffing banks/hf100 vs banks/own100 mean_L40_raw_paper.csv: only the 2 zero-shot rows differ), so CSVs coincide when those rows coincide (e.g. root csv = paper16_famcentred for raw/datapca/random). splits.json identical across the banks of one pass (expected).",
    "", "LOW")
row("duplicate", "run-crossgen/kd11/passA/significance.json == run-crossgen/passA/significance.json; run-crossgen/kd11/passC/significance.json (+ Drive .tmp) == run-crossgen/passC/significance.json",
    "LAB,DRIVE", "DU-03. SURPRISING name mismatch. Byte-identical (7,368,815 B and 551,848 B) although written a day apart (kd11/passA 09-20 14:10 vs passA 09-19 14:18; kd11/passC 09-20 13:46 vs passC 09-18 16:02). The kd11/passA summary.json lists 10 datasets (no harm-pressure-kr-matched), seed 42 only; both significance files hold blocks 9/10 only (15,360 and 1,152 tests). INFERRED: the 09-20 'kd11' passA/passC grids ran without the 11th dataset (F-20: kr-matched had no rows) and reproduced the root grids exactly; the real 11-dataset versions are passA11/passC11 (09-21). Git holds a different, 576-test 3-layer kd11/passA/significance.json (RF-11). VERIFIED (p1_anom_snip_sig.py; census).",
    "F-20, F-72, F-75, F-76", "MED")
row("duplicate", "his 'persona including knowlede denial/*.pt' == lab run-kd/vectors_s3 (36/36); his deceptive/ == lab share_deceptive; his paper16_L20_sharedquestions/ == lab share_L20_shared == Drive share/paper16_L20/shared_questions and share/paper16_L20_sharedquestions",
    "PCOOD,LAB,DRIVE", "DU-04. Known (RF-1, resolved). Confirmed again by content key; run-kd vectors_all/vectors_s23/vectors_s3 of 'default' are also mutually identical. VERIFIED.",
    "F-97", "LOW")
row("duplicate", "Drive share/paper16_L20/{responses,scores} == run-paper16/{responses,scores} (50 files)", "DRIVE,LAB",
    "DU-05. The share folder carries the paper-question (run-paper16) rollouts and scores next to both a paper_questions and a shared_questions vector file; his branch received only the shared_questions vectors (run-paper16-shared). INFERRED mixed provenance inside the share folder; no result depends on it as far as fmap shows. VERIFIED file identity.",
    "", "LOW")
row("duplicate", "same-experiment copies: vectors_all == vectors_s23/s3 for some personas; kr-matched chunks == HP-KR chunks; passA/passC items.pt; passB/B_fine/P/scale items.pt and rows; run/scores 11 roles",
    "LAB,DRIVE", "DU-06. Expected or benign: persona vectors identical across filters where the filter removed nothing (run-kd 15, run-scale/3b 12, 8b 8, run-paper16 fp32 8, run-cap 1, run-paper16-shared 1); harm-pressure-kr-matched/chunk_*.pt are the HP-KR chunks (55 per pass; views link source chunks); passC reuses passA's rows (21 items.pt); prompt items shared across stores and rungs. Notable: 11 own100 roles (bard, eldritch, ghost, hermit, leviathan, oracle, prophet, revenant, tree, void, whale) have byte-identical run/scores/*.json because every one of their 1,200 rollouts scored 3 (VERIFIED p1_anom_snip_scores.py) - a judge ceiling worth a Phase 2 look. run-cap-ctrl reuses 7 run-cap vectors. His zero_shot.png is identical across L20/L40/L44 by design (a three-layer figure, persona_ood_figures.py:422)."
    , "", "LOW")
row("duplicate", "lab @expLB-acts/deception-detection == asix/his vendored code", "LAB,ASIX,PCOOD",
    "DU-07. D:/expLB-acts is a copy of the vendored probe code; 5 files differ from asix and equal his branch (his modified activations.py, experiment.py, scripts/experiment.py, export_html.py, README.md). Expected (Task A O86). VERIFIED.",
    "", "LOW")

# ---------------- 4. CONFLICTING VERSIONS ----------------
row("conflict", "run-paper16 vectors and axes: OLD (laptop run-paper16/vectors_{s3,all}/ = Drive paper16/run/vectors_{s3,all}/ + paper16/run/axes_vectors_all.pt; lab run-paper16/axes_vectors_s3.pt, vectors_s3_old_pc_broker.pt) vs NEW fp32 (lab vectors_{s3,all}_fp32/, axes_vectors_{s3,all}_fp32.pt)",
    "LAPTOP,DRIVE vs LAB,DRIVE", "CV-01 (RF-12). All 26 old vectors differ from the fp32 ones (PROGRESS RF-12); the old vectors_s3 + vectors_all on the laptop equal Drive paper16/run/ (52/52 md5, VERIFIED from the census files). Which F used which: OLD - F-18 (cmp_banks.py; axes file's internal vectors_dir is the laptop path) and all 09-15 run-coords analyses (Q1/Q2/Q4: smoke, probes_5sets, probes_L20, splits, zeroshot_paper16, conf, erase_test, FULL_L20/22/40; Task A VERIFIED 'bank' key); NEW - every kd11 root and run-crossgen root matrix (F-03, F-04, F-13, F-17, F-29, F-39, F-58's 70B cell, F-61, F-65, F-74, F-98), 09-16+ run-coords analyses. Docs name only the OLD files (OR-09). The rebuild logs (rb_paper16_*.log, 09-15 22:37 -> 09-16 01:30) are named in no doc.",
    "F-03, F-04, F-13, F-17, F-18, F-29, F-39, F-58, F-61, F-65, F-74, F-98", "HIGH")
row("conflict", "run-crossgen/scale_70b/matrices/*.json 'bank' = axes_vectors_s3.pt, bank_name = paper16_70b", "LAB,DRIVE",
    "CV-02. Ambiguous bank identity for the 70B rung. The matrix stores only the basename, which three files share (run-kd/, run-paper16/, run-paper16-shared/axes_vectors_s3.pt). The 09-22 sweep file mean_L20_s42.json carries zero-shot directions 'knowledge_denial' and 'kd_assertion' that exist only in the kd bank -> that sweep used run-kd/axes_vectors_s3.pt (36 kd personas), as F-100's source line says; the 3B/8B rungs use axes_3b/axes_8b.pt (bank_name paper16_3b/8b, 26 paper16 personas). The 09-20 L40 cell (F-63) cannot be resolved from the file or from logs/crossgen_scale_70b.log (no bank path printed); F-100 says the sweep 'reproduced' it (INFERRED same kd bank). The 70B driver is in no branch. If so, the three-rung comparison mixes banks. VERIFIED file facts (p1_anom_snip_bank.py, p1_anom_snip_feat.py); interpretation INFERRED. Task A O77 also flags it.",
    "F-63, F-96, F-100, F-101, F-102", "HIGH")
row("conflict", "run-crossgen/kd11/banks/kd_s23/{significance,summary}.json", "ASIX vs LAB,DRIVE",
    "CV-03 (RF-11). git: 468 tests / 216 records, layers 20/40/44 (3dfdb67, 09-21 11:07); lab = Drive: 5,544 tests / 1,512 records, 21 layers (content fixed by 09-22 01:31, rewritten byte-identically 16:15). F-34 read a third, 3,276-test version (GH-01); F-34's 21-layer profile matches the lab copy, its q-values match neither. VERIFIED (fmap F-34; conflicts_lab_output.txt).",
    "F-34, F-57", "HIGH")
row("conflict", "run-crossgen/kd11/banks/kd/significance.json", "ASIX vs LAB,DRIVE",
    "CV-04 (RF-11). git 4,032 tests (1,628,949 B) vs lab 6,300 tests (2,679,778 B, 09-22 16:01). F-31's numbers (q 0.499, '6 of 4,032') match ONLY the git copy; diffs/CIs identical on common keys, q moved with the BH family. VERIFIED (fmap F-31).",
    "F-31", "MED")
row("conflict", "run-crossgen/kd11/significance.json", "ASIX vs LAB,DRIVE",
    "CV-05 (RF-11). git 4,032 tests vs lab 4,788 (09-22 15:53). F-04 read neither (384, GH-02); F-61's '+0.006' matches pc:unified - datapca in the git copy (fmap). VERIFIED.",
    "F-04, F-61", "MED")
row("conflict", "run-crossgen/{summary,significance,splits}.json and run-crossgen/banks/{hf100,own100,paper16_famcentred,paper16_shared}/{summary,significance,splits}.json", "ASIX vs LAB,DRIVE",
    "CV-06 (RF-11). git = L40-only snapshots (b9a497e, 09-18); lab = all-layer x 6-seed supersets (root rewritten 09-22 15:40; banks 09-18 06-07). Keyed comparison: git is an exact subset, metric values identical, only BH q differs (PROGRESS RF-13 update). F-17's value at its key is the same in both. VERIFIED.",
    "F-17", "LOW")
row("conflict", "run-crossgen/kd11/passA/{summary,significance}.json and kd11/passA/banks/kd_s23/{summary,significance}.json", "ASIX vs LAB,DRIVE",
    "CV-07 (RF-11). git: 198 records / 576 tests (and 216 / 468), layers 20/40/44, 10 datasets; lab = Drive (09-20 14:08-14:21, unchanged since): 80 layers. The lab significance equals the root passA one (DU-03). The findings built on passA use passA11 instead (fmap). VERIFIED (git show; p1_anom_snip_sig.py).",
    "F-20, F-72", "LOW")
row("conflict", "kd/run/knowledge/scores.jsonl", "ASIX vs LAB,DRIVE",
    "CV-08 (RF-5, RF-11). git copy truncated by scp (203,705 B LF, 1,095 complete records + 1 cut) vs lab = Drive 279,757 B. F-08/F-09 (and the F-59 denial run, per fmap) used the full lab file; any re-computation from git is wrong. gen_hpkr_denial.py reads this path by default. VERIFIED.",
    "F-08, F-09, F-14, F-59", "MED")
row("conflict", "kd/run/factor_probes/kd_{knowledge,utterance}/rows.jsonl", "ASIX vs LAB,DRIVE",
    "CV-09 (RF-11). Format difference, not a content conflict: git holds the input rows (2.5 / 2.3 MB), the lab the extraction-output rows (553 / 506 KB); label counts agree (1,962 = 1,841 + 121; 1,800 = 900 + 900) per fmap. VERIFIED.",
    "F-05, F-07, F-16", "LOW")
row("conflict", "own100 axes: Drive own100/run/axes.pt (20,979,057 B, 09-15 06:06Z) vs lab run/axes.pt = Drive own100/axes.pt (62,936,618 B, 06:51Z); own100/run/DELIVERABLE_READY differs from lab's", "DRIVE vs LAB,DRIVE",
    "CV-10. The only real lab-vs-Drive content conflicts found by path-mapped comparison: an older own100 axes file (45 min earlier, one third the size) survives only on Drive. F-90's persona arm uses the lab run/axes.pt (fmap). VERIFIED (p1_anom_conflicts_lab.py).",
    "F-90", "LOW")
row("conflict", "own100 / own100_strict banks on bf16-reduction vectors vs paper16 banks on fp32 vectors", "LAB,DRIVE",
    "CV-11. The fp32 re-extraction reached paper16 (26/26) but not own100 (1/100, OR-11), so every own100-vs-paper16 bank comparison (memory persona-set-decides-detection, Q1; F-90's persona arm = own100) compares two extraction versions. VERIFIED file facts; effect unknown.",
    "F-90", "MED")
row("conflict", "passA/passC HP-KR inputs: root run-crossgen/passA grid (200-row HP-KR, 09-18/19) vs passA11/passC11 (2,139-row HP-KR after the 09-21 top-up)", "LAB,DRIVE",
    "CV-12. The 09-22 re-run of the root passA grid refused to run: 'split for harm-pressure-knowledge-report seed 42 differs from run-crossgen\\passA\\splits.json' (crossgen_full_A.log). So root passA results and passA11 results rest on different HP-KR row sets. VERIFIED (log tail; experiments_notes 4b).",
    "F-20, F-69, F-72", "MED")
row("conflict", "HP-KR rows.jsonl across stores: passB (623,208 B) vs passA/passC/passB_fine (747,630 B) vs passP (747,456 B); kr-matched passB (74,352 B) vs passA/passC (89,389 B)",
    "LAB,DRIVE", "CV-13. Schema-only: every store has 2,139 HP-KR rows (139 deceptive / 2,000 honest) and 258 kr-matched rows (124/134); later stores add prompt_len, response_len, span columns. Row identity checked only on the first 3 ids. VERIFIED counts (p1_anom_snip_rows.py); consistent with F-71.",
    "F-69, F-71", "LOW")
row("conflict", "versioned siblings kept beside current files: logs/sarc_pilot.status{,.pre-e2e-2026-09-22,.2026-09-21-buggy-verdict}; run-crossgen/kd11/passC/significance.json.stale-576; run-kd/responses/pc_broker.jsonl.predupe; run-paper16/vectors_s3_old_pc_broker.pt",
    "LAB,DRIVE", "CV-14. sarc_pilot: stripped bag-of-words AUROC 0.947 (pre-e2e, quoted by F-104) vs 0.921 on v4,v5 (current) - F-93/F-104/F-105 quote different versions (Task A also flags). stale-576 = 576-test 3-layer version (current 1,152). .predupe explains F-61's '37 files for 36 personas'. old_pc_broker = the pre-fp32 pc_broker vector (== Drive paper16/run/vectors_s3/pc_broker.pt). VERIFIED (census; dupes).",
    "F-61, F-93, F-104, F-105", "MED")
row("conflict", "git vs Drive-only text copies (figures/crossgen CSV + INDEX.md, docs/thesis md/csv)", "ASIX vs DRIVE",
    "CV-15. Not conflicts: all 13 differ only by CRLF (Drive size - git size = line count; role_labels_review.csv differs by 277 = its record count, INFERRED CRLF record terminators). VERIFIED (git show | wc).",
    "", "LOW")
row("conflict", "docs on the lab: kd/docs/FINDINGS.md (= bc701d9, 09-21 04:01) and cap/docs/DESIGN.md older than git", "LAB,DRIVE vs ASIX",
    "CV-16 (RF-11). Stale doc copies; no result depends on them. VERIFIED.", "", "LOW")
row("conflict", "audit caveat: join_summary.txt column 'driveDIFF' / joined_labpc.tsv.gz column drive_same_name_diff_content", "audit files",
    "CV-17. Not a version-conflict count. It flags lab files whose BASENAME matches a Drive file in a different bank (e.g. run-kd/responses/pc_handler.jsonl vs Drive paper16/responses/pc_handler.jsonl) plus 75 unhashed >200 MB passC files (same path and size on Drive, INFERRED identical). A path-mapped comparison over all 208,830 multi-copy logical files finds 50 content conflicts: 27 git-vs-(lab=Drive) (RF-11), 13 git-vs-Drive CRLF-only, 7 asix-vs-his code, 1 git-vs-lab doc, 2 lab-vs-Drive (CV-10). VERIFIED (conflicts_lab_output.txt).",
    "", "LOW")

with open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["category", "path_or_group", "sources", "detail", "related_f_ids", "severity_hint"])
    w.writerows(R)
from collections import Counter
print(len(R), "rows", Counter((r[0], r[5]) for r in R))
