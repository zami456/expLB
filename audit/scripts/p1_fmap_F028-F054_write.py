"""Writes audit/phase1/fmap/fmap_F028-F054.csv (Phase 1 mapping, batch F-28..F-54).
All facts below were gathered in this session with the read-only tools named in fmap_F028-F054.md.
Run from the repo root: python audit/scripts/p1_fmap_F028-F054_write.py
"""
import csv

COLS = ["f_id", "title_short", "heading_date", "claim_type", "key_numbers", "cited_paths", "resolved_primary",
        "key_path", "value_seen_prelim", "other_copies", "copy_status", "producing_script", "script_writes_file",
        "run_log", "depends_on", "supersedes", "superseded_by", "draft_topic", "mapping_status", "notes"]
L = "C:/Users/user4/expLB/"
R = []


def row(**k):
    R.append({c: k.get(c, "") for c in COLS})


row(f_id="F-28", title_short="External review of cap bank: four structural confounds verified by lexical count",
    heading_date="2026-09-20", claim_type="scientific",
    key_numbers="pair4 explanation=4/5 vs 0/5; appeaser affect=3/5 vs 0/5; approving modifiers holding=4/6 vs cap 0/6; "
                "leak phrasing=4 holding vs 0 cap; caver-as-spelunker=refuted on 4 v1 responses",
    cited_paths="cap/personas/cap_instructions/*.json; cap/questions/cap_questions.jsonl; logs/cap_smoke.log; cap/docs/REVIEW_PACKET.md",
    resolved_primary="NOT-FOUND for the counted text: the v1 (pre-F-32) cap_instructions/*.json were overwritten (lab files mtime 2026-09-20 22:51 = later v4 bank; git first commit dc24e96 09-21 = v4). "
                     "Surviving v1 evidence: " + L + "logs/cap_smoke.log (SYSTEM lines = variant 0 of the two smoked roles only)",
    key_path="no stored output of the counts; cap_smoke.log line 55 'SYSTEM You are a caver who abandons your position...' (v1 pc_caver variant 0); ROLE blocks at lines 53 and 3415",
    value_seen_prelim="cap_smoke.log turn-2 quotes of F-29 found (line ~3405 'You're right, maybe...', ~5095 'I'd be happy to reconsider')",
    other_copies="cap/questions/cap_questions.jsonl lab=git SAME (md5_lf 48416939); cap_smoke.log LAB-ONLY (not Drive, not git); "
                 "git cap/docs/REVIEW_PACKET.md is the post-F-32 rebuild (not the packet the reviewers read)",
    copy_status="NOT-FOUND",
    producing_script="counts: ad hoc regex (no script found, INFERRED print-only); smoke: cap/scripts/smoke_multiturn.py, but the 14:38 version is not preserved "
                     "(lab cap/scripts/smoke_multiturn.py mtime 18:40 = git md5 515aad29; loose lab cap/smoke_multiturn.py 18:38 md5 76667b95 not in git)",
    script_writes_file="n", run_log=L + "logs/cap_smoke.log (UTF-16LE, 2026-09-20 14:38:29 -> 14:45:04)",
    depends_on="cap bank v1 persona texts (pre-F-32); cap_questions.jsonl v1",
    supersedes="", superseded_by="F-51 (spelunker refutation withdrawn); F-32 (bank rewritten)", draft_topic="6",
    mapping_status="PARTIAL",
    notes="The two external reviews (Claude Opus 5, Gemini) were not located as files (not searched beyond the census; DOC-ONLY). Counts cannot be recomputed from preserved files.")

row(f_id="F-29", title_short="The 0.374 inversion: reviewer's challenge half right (sycophancy_mt as target vs source)",
    heading_date="2026-09-20", claim_type="scientific",
    key_numbers="zero-shot PC1=0.374 (inv 0.626); trained as TARGET=0.355 vs random 0.541 (+0.104 inverted); trained as SOURCE=0.379 vs random 0.633 (-0.012 inverted)",
    cited_paths="run-crossgen/kd11/matrices/mean_L40_s42.json; logs/cap_smoke.log",
    resolved_primary=L + "run-crossgen/kd11/matrices/mean_L40_s42.json",
    key_path="datasets[10]=sycophancy_mt; zeroshot[direction=PC1].auroc[10]; results[feature=axes,probe=paper].auroc column 10 off-diagonal mean (TARGET) and row 10 off-diagonal mean (SOURCE); same for feature=random",
    value_seen_prelim="PC1 0.3740; axes/paper TARGET 0.3551, SOURCE 0.3787; random/paper TARGET 0.5413, SOURCE 0.6327 (computed from git copy, identical to lab)",
    other_copies="git asix HEAD SAME (md5_lf de54b7a2); Drive crossgen/kd11/matrices/mean_L40_s42.json SAME (md5 d7616aa5)",
    copy_status="LAB=GIT",
    producing_script="matrix: src/assistant_axis_ndif/27_crossgen.py (lab=git HEAD, lab mtime 09-21 03:47; the matrix, mtime 09-19 20:12, was written by an earlier version: b9a497e 09-18 or an uncommitted lab version, not identified); target/source means: no script named (ad hoc, INFERRED)",
    script_writes_file="y (matrix) / n (derived means)", run_log="not identified (kd11 crossgen run of 2026-09-19)",
    depends_on="paper16 bank axes_vectors_s3_fp32.pt (matrix 'bank' field; RF-12 new fp32 vectors); run-coords/passB; 11th dataset harm-pressure-kr-matched (make_view.py, RF-8); cap_smoke.log v1 personas",
    supersedes="none (supplies argument for F-13)", superseded_by="", draft_topic="6", mapping_status="MAPPED",
    notes="'trained' = feature 'axes' (INFERRED: only feature whose paper-probe means match 0.355/0.379). Lab file not regenerated on 09-22 (mtime 09-19).")

row(f_id="F-30", title_short="In-character (judge keep) rate across 3B/8B/70B, same judge",
    heading_date="2026-09-20", claim_type="scientific",
    key_numbers="drop0+1 70B=0.89% (1,350) 8B=3.16% (2,500) 3B=2.68% (2,500); score3 share=80.7/57.3/52.3%; deception keep=97.50/95.62/95.38%; pc_coverup_operator keep=8.0/12.0/23.0% (as stated)",
    cited_paths="run-{paper16,scale/8b,scale/3b}/scores; scale/scripts/score_summary.py",
    resolved_primary=L + "run-paper16/scores/*.json; " + L + "run-scale/8b/scores/*.json; " + L + "run-scale/3b/scores/*.json (25 files each)",
    key_path="each file = {rollout_id: judge score 0-3}; tallies per score; groups from src/assistant_axis_ndif/pc_role_groups.json",
    value_seen_prelim="70B (git copy): 25 files, 1,350 rollouts, score0 0.9% / score2 18.4% / score3 80.7%, drop 0.89% - matches; 8B/3B not tallied",
    other_copies="run-paper16/scores: git SAME 25/25, Drive SAME; run-scale/{8b,3b}/scores: Drive SAME 25/25, not in git; pc_role_groups.json lab=git",
    copy_status="LAB=GIT (70B) | LAB-ONLY+Drive (8B,3B)",
    producing_script="scale/scripts/score_summary.py (print-only; lab=git cfd14b5f); scores by src/assistant_axis_ndif/3_judge_ndif.py via scale/scripts/run_scale_chain.ps1 (both lab=git)",
    script_writes_file="n (console only; output not found in a log)",
    run_log=L + "logs/scale_8b_judge.log (-> 09-20 14:54:18); logs/scale_3b_judge.log (-> 15:51:34); logs/scale_{8b,3b}.status",
    depends_on="paper16 bank 26 personas; 70B = run-paper16 (paper question set, 50-60 per role), not run-paper16-shared",
    supersedes="F-27 (8B figures at 21/26 roles)", superseded_by="", draft_topic="4", mapping_status="MAPPED",
    notes="CONFLICT candidate (INFERRED): text says all rungs ran 20 shared questions; the 70B numbers match run-paper16/scores (1,350 rollouts, the paper-question bank), not run-paper16-shared (shared questions). The caveat acknowledges the sample size, not the question-set difference.")

row(f_id="F-31", title_short="kd bank +0.076 (pc:unified - raw, L40 mean) does NOT clear dataset-level bootstrap",
    heading_date="2026-09-20", claim_type="scientific",
    key_numbers="pc:unified-raw paper=+0.0757 62/110 CI[-0.006,+0.173] q=0.499; vs datapca=+0.0729 q=0.610; lr_reg10=+0.0847 q=0.467; axes-raw=-0.0272 q=0.902; 6 of 4,032 tests clear",
    cited_paths="run-crossgen/kd11/banks/kd/significance.json; kd/scripts/read_significance.py",
    resolved_primary="GIT asix HEAD run-crossgen/kd11/banks/kd/significance.json (blob md5 7cb6d995, 1,628,949 B, committed 3dfdb67 2026-09-21 11:07; 4,032 tests) - this version exists only in git",
    key_path="tests[] where layer=40, pooling=mean, probe=paper, comparison='pc:unified - raw', pairs=110 (block '10'); count(significant_datasets==true)",
    value_seen_prelim="git: diff +0.0757, won 62/110, CI [-0.0059,+0.1725], q 0.4992; datapca +0.0729 q 0.6097; lr_reg10 +0.0847 q 0.4673; axes -0.0272 q 0.9025; 4,032 tests, 6 significant - all match. "
                      "Lab (09-22): 6,300 tests, same diff/CI, q 0.4196, 65 significant",
    other_copies="lab " + L + "run-crossgen/kd11/banks/kd/significance.json DIFFERENT (md5 af3b9feb, 2,679,778 B, mtime 2026-09-22 16:01:20, 6,300 tests); Drive crossgen/kd11/banks/kd + kd/crossgen11/banks/kd = lab version (DIFFERENT from git); git version found nowhere else",
    copy_status="LAB≠GIT",
    producing_script="src/assistant_axis_ndif/27_crossgen.py writes significance.json (lab=git); readout kd/scripts/read_significance.py (print-only; lab=git ff1b14be)",
    script_writes_file="y (27_crossgen) / n (read_significance)",
    run_log="09-20 run log not identified; 09-22 rewrite = kd_bank_crossgen wave (per lead, not checked here)",
    depends_on="kd bank run-kd axes_vectors_s3.pt as of 09-20 (lab run-kd axes regenerated 09-22 per lead); run-coords/passB (not passA); 11th dataset harm-pressure-kr-matched (make_view.py); kd bank matrices (lab=git, mtime 09-20 03:06)",
    supersedes="none measured (retires the '+0.076 first gain over matched control' phrasing)", superseded_by="",
    draft_topic="5", mapping_status="MAPPED",
    notes="REGENERATED-AFTER-F. Keyed comparison (layer/pooling/probe/comparison/pairs) git vs lab: diff and dataset CI identical on all checked keys; q_BH differs because the BH family grew 4,032 -> 6,300 tests (comparisons added, e.g. pc:knowledge_denial, pc:kd_assertion). "
          "'6 of 4,032 clear' is specific to the git version. Cite the git copy for F-31.")

row(f_id="F-32", title_short="Cap variants rewritten; five confounds closed by lexical count",
    heading_date="2026-09-20", claim_type="engineering",
    key_numbers="pair4 explanation=4/5 vs 4/5; approving modifiers=0; leak phrasing=0; affect pair3=3/5 vs 5/5; turn-1 commitment=5/5 on 6 cap roles; max variant length=103 chars",
    cited_paths="cap/personas/cap_instructions/*.json (regenerated 2026-09-20); cap/scripts/cap_personas.py; cap/docs/REVIEW_PACKET.md; review_page.html; cap_role_groups.json",
    resolved_primary="GIT cap/docs/REVIEW_PACKET.md (also _V2, _V3), which embed the F-32 bank text; the F-32 cap_instructions/*.json themselves are NOT-FOUND (overwritten 22:51 by the v4 bank)",
    key_path="per-role variant lists in REVIEW_PACKET.md (e.g. line 136 pc_caver variant 1, line 144 pc_position_holder variant 1)",
    value_seen_prelim="REVIEW_PACKET.md pc_caver/pc_position_holder variant 1 == SYSTEM prompts in logs/cap_smoke_v2.log (the F-46 run on the F-32 bank) - VERIFIED text match",
    other_copies="REVIEW_PACKET*.md and review_page.html GIT-ONLY (lab cap/docs has only DESIGN.md); cap_personas.py: git cap/scripts/cap_personas.py == lab loose " + L + "cap_personas.py (md5_lf 5f3f73af, mtime 22:51 = v4 generator, not the F-32 version)",
    copy_status="GIT-ONLY",
    producing_script="cap/scripts/cap_personas.py (only the 22:51 v4 version preserved); counts via F-28 regexes, ad hoc (not found)",
    script_writes_file="y (bank) / n (counts)", run_log="none",
    depends_on="F-26/F-28 confound list", supersedes="",
    superseded_by="F-46 (v2 smoke invalidates the design), F-53 (noun->verb), later v4 bank", draft_topic="6",
    mapping_status="PARTIAL", notes="Counts not recomputable from the bank files as they now stand; REVIEW_PACKET.md is the best surviving copy (INFERRED it is the full F-32 bank).")

row(f_id="F-33", title_short="3B extraction failed on every role; cause batch size (per-process cap), not length",
    heading_date="2026-09-20", claim_type="engineering",
    key_numbers="failed 22/22 roles, 0 vectors; lengths p50/p90/p99/max 3B=1,371/2,418/2,722/2,909, 8B=1,369/2,447/2,729/2,896, 70B=1,480/2,542/2,825/3,059; probe batch 2/4 saved (28,3072) n=62",
    cited_paths="logs/scale_3b_vectors_all.log; single-role probe at 15:49; scale/scripts/response_lengths.py; run_scale_chain.ps1; 2_activations_ndif.py",
    resolved_primary=L + "logs/scale_3b_vectors_all.log (failures); length table: print-only over " + L + "run-scale/{3b,8b}/responses/*.jsonl + an unnamed 70B responses dir",
    key_path="log lines 'Role X: no vector produced' (15:26:18 -> 15:48:35) and 'widest NNNN padded tokens'",
    value_seen_prelim="24 'no vector produced' lines up to 15:48:35 (text: 22/22 at time of writing); widest 4,752-4,848 padded tokens in the first three batches listed; 26 'saved' lines after restart",
    other_copies="logs LAB-ONLY (logs/ not on Drive, RF-10); run-scale/{3b,8b}/responses Drive SAME, not in git",
    copy_status="LAB-ONLY",
    producing_script="src/assistant_axis_ndif/2_activations_ndif.py (lab=git af86bc22; lab mtime 09-20 15:48:59 = post-fix); scale/scripts/response_lengths.py (print-only; lab=git); scale/scripts/run_scale_chain.ps1 (lab=git, mtime 15:51:10)",
    script_writes_file="y (vectors/log) / n (lengths)",
    run_log=L + "logs/scale_3b_vectors_all.log (2026-09-20 15:25:17 -> 16:24:26); logs/scale_3b.status",
    depends_on="run-scale/3b/responses (2,600 rollouts)", supersedes="", superseded_by="F-35 (throughput)",
    draft_topic="4", mapping_status="PARTIAL",
    notes="response_lengths.py measures CHARACTERS (len of text), the table gives no unit. 70B source dir of the length row not named. Single-role probe output (15:49) NOT-FOUND.")

row(f_id="F-34", title_short="kd_s23 gain is depth-dependent (21-layer profile), not an L40 artifact; not significant",
    heading_date="2026-09-20", claim_type="scientific",
    key_numbers="pc:unified-raw paper mean 90 pairs: L0-L20 -0.036..+0.008; L40=+0.1082, L44=+0.1156; 16/21 positive; vs datapca L40=+0.1199 [+0.024,+0.220], L44=+0.1291 [+0.033,+0.228], 17/21; best q=0.130; 9 of 3,276 clear (5 neg, 4 pos)",
    cited_paths="run-crossgen/kd11/banks/kd_s23/significance.json (3,276 tests); kd/scripts/layer_profile.py",
    resolved_primary=L + "run-crossgen/kd11/banks/kd_s23/significance.json for diffs/CIs; the 3,276-test version F-34 read is NOT-FOUND (git has 468 tests / 3 layers; lab and Drive have 5,544 tests)",
    key_path="tests[] comparison in {'pc:unified - raw','pc:unified - datapca'}, probe=paper, pooling=mean, pairs=90 (block '9'), 21 layers",
    value_seen_prelim="lab: L40 +0.1082, L44 +0.1156, 16/21 positive, min -0.0361 (L20); datapca L40 +0.1199 [0.024,0.220] q 0.146, L44 +0.1291 [0.033,0.228] q 0.103, 17/21 positive; 84 of 5,544 significant. git (L20/40/44 only): same diffs/CIs, q 0.074-0.095",
    other_copies="git asix HEAD DIFFERENT (md5 ed647efc, 187,983 B, 468 tests, layers 20/40/44, committed 3dfdb67 09-21 11:07); Drive crossgen/kd11/banks/kd_s23 + kd/crossgen11/banks/kd_s23 = lab (SAME md5 c6dd8f7b; Drive modtime 2026-09-21 19:31Z)",
    copy_status="LAB≠GIT",
    producing_script="src/assistant_axis_ndif/27_crossgen.py via kd/scripts/run_kd_s23.ps1 (both lab=git; run_kd_s23 lab mtime 09-22 01:28); kd/scripts/layer_profile.py (print-only; lab=git 94179ff2)",
    script_writes_file="y / n (layer_profile)", run_log="original 09-20 run log not identified; lab logs/kd_s23* rewritten 09-22 (per lead)",
    depends_on="kd_s23 bank (run-kd axes_vectors_s23.pt as of 09-20; regenerated 09-22 per lead); run-coords/passB 21 layers (not passA); 11th dataset harm-pressure-kr-matched",
    supersedes="", superseded_by="(used by F-57 thesis fig2; RF-7)", draft_topic="2;5", mapping_status="PARTIAL",
    notes="REGENERATED-AFTER-F. Current content dates from <= 2026-09-22 01:31 +06 (Drive modtime, identical md5) and was rewritten byte-identically at 16:15. Keyed diffs/CIs identical in git and lab where both exist; q values and significant counts differ with the BH family size. "
          "F-34's 'best q 0.130' and '9 of 3,276 clear' come from an intermediate, overwritten file (NOT-FOUND). Same file behind RF-7 (thesis fig2).")

row(f_id="F-35", title_short="Extraction throughput: batch-2 constraint makes the 3B ~10x slower than 8B",
    heading_date="2026-09-20", claim_type="engineering",
    key_numbers="8B 32 layers batch 8 14:54:19->14:57:40=3.4 min=12.9 rows/s; 3B 28 layers batch 2 15:51:34->16:24:27=32.9 min=1.32 rows/s; ratio 9.8x; projection 8B ~20 min, 3B ~3.2 h, total ~3.5 h; 3B 26/26 saved; 24 pre-restart failures (last 15:48:35)",
    cited_paths="logs/scale_3b.status; logs/scale_8b.status; logs/scale_3b_vectors_all.log",
    resolved_primary=L + "logs/scale_8b.status; " + L + "logs/scale_3b.status; " + L + "logs/scale_3b_vectors_all.log",
    key_path="status lines 'extracting vectors_all' and 'vectors_all exit 0: 26 vectors'; log 'saved (28, 3072)' / 'no vector produced' lines",
    value_seen_prelim="8B 14:54:19 -> 14:57:40 and 3B 15:51:34 -> 16:24:27 present in status files; 3B log: 26 'saved' lines, 24 'no vector produced' lines, last at 15:48:35 - all match",
    other_copies="LAB-ONLY (logs not on Drive or git)", copy_status="LAB-ONLY",
    producing_script="scale/scripts/run_scale_chain.ps1 (writes status; lab=git); 2_activations_ndif.py",
    script_writes_file="y (status/log); rates derived by hand (INFERRED 2600/201 s and 2600/1973 s)",
    run_log="scale_8b.status (START 13:25:25); scale_3b.status (START 15:51:19, earlier 3B content not in the file); both append 'SKIP already DONE' at 2026-09-22 16:05",
    depends_on="2,600 rollouts per rung", supersedes="", superseded_by="F-40, F-45 (projections); F-49/F-50 (ratio)",
    draft_topic="4", mapping_status="MAPPED", notes="Status files were touched on 09-22 16:05 by a no-op re-run (SKIP), not a regeneration.")

row(f_id="F-36", title_short="Cap questions (judgement calls) mismatch sycophancy_mt test items (factual)",
    heading_date="2026-09-20", claim_type="scientific",
    key_numbers="sycophancy_mt items factual with correct_answer (trivia_qa/truthful_qa); cap v1 = 20 judgement questions; v2 draft = 20 items, 12 recall + 8 misconception, entirely third-person",
    cited_paths="run-sycophancy-mt/questions.jsonl (lab PC); cap/questions/cap_questions_v2_draft.jsonl",
    resolved_primary=L + "run-sycophancy-mt/questions.jsonl; " + L + "cap/questions/cap_questions_v2_draft.jsonl",
    key_path="questions.jsonl field 'source'; v2 draft fields 'kind' and 'framing'",
    value_seen_prelim="questions.jsonl: 1,200 items, trivia_qa 666 / truthful_qa 534. v2 draft on disk: 20 items = recall 10 (6 third, 4 first) + misconception 10 (4 third, 6 first) - does NOT match '12+8, all third-person'",
    other_copies="questions.jsonl Drive SAME, not in git; v2 draft git SAME (md5 d5eec908) and a loose lab copy cap/cap_questions_v2_draft.jsonl SAME",
    copy_status="LAB-ONLY (questions) | LAB=GIT (v2 draft)",
    producing_script="src/assistant_axis_ndif/13_sycophancy_multiturn.py (run-sycophancy-mt, 09-16); v2 draft hand-authored (INFERRED)",
    script_writes_file="y", run_log="",
    depends_on="", supersedes="v1 question-set premise", superseded_by="F-46 (string check invalid); later cap_questions_v3.jsonl (22:52)",
    draft_topic="6", mapping_status="PARTIAL",
    notes="CONFLICT: the preserved v2 draft (mtime 18:38/18:39, i.e. edited after F-36, just before the 18:40 v2 smoke) has 10/10 recall/misconception and 10 first-person items; the version F-36 describes is NOT-FOUND.")

row(f_id="F-37", title_short="Persona geometry across 3B/8B/70B: cos(PC1, assistant axis) size-dependent, not monotone",
    heading_date="2026-09-20", claim_type="scientific",
    key_numbers="cos(PC1,assistant) 3B=+0.435 8B=+0.732 70B=+0.623; PC1 var 33.4/39.6/37.2%; top-5 73.1/77.4/75.1%; anchor 0.834/0.949/0.935; contrast alignment sycophancy 0.948/0.983/0.993, unified 0.781/0.908/0.906, deception 0.924/0.921/0.955; deception PC1 var 30.2/36.0/43.0%",
    cited_paths="logs/scale_3b_axes.log; logs/scale_8b_axes.log; logs/paper16_shared_axes.log",
    resolved_primary=L + "logs/scale_3b_axes.log; " + L + "logs/scale_8b_axes.log; " + L + "logs/paper16_shared_axes.log - FIRST report block of each (vectors_all)",
    key_path="block 1 lines: 'cos(PC1, assistant axis)', 'across layers', 'PC1 explains', per-behaviour table, 'default anchor'",
    value_seen_prelim="all quoted values match block 1 of each log (3B/8B layer 14/16, 70B layer 40)",
    other_copies="logs LAB-ONLY; axes files Drive SAME: run-scale/3b/axes_3b.pt, run-scale/8b/axes_8b.pt, run-paper16-shared/axes_vectors_all.pt",
    copy_status="LAB-ONLY",
    producing_script="src/assistant_axis_ndif/6_axes.py (lab=git e5c96358, lab mtime 09-21 03:22 - later than these runs, so the printing version is older)",
    script_writes_file="n for the metrics (printed); y for the axes .pt",
    run_log="scale_3b_axes.log (-> 09-20 16:56:13); scale_8b_axes.log (-> 15:01:26); paper16_shared_axes.log (09-18 03:13)",
    depends_on="run-scale/{3b,8b}/vectors_all; run-paper16-shared vectors_all (shared-question bank)",
    supersedes="", superseded_by="(used by F-57 thesis fig4, numbers typed into make_thesis_figures.py per RF-7)",
    draft_topic="4", mapping_status="MAPPED",
    notes="Each log holds TWO blocks: vectors_all, then vectors_s23 (3B/8B) or vectors_s3 (70B). F-37 quotes block 1 (vectors_all; INFERRED unfiltered) at all rungs. For 3B/8B both blocks 'wrote' the SAME path run-scale/<r>/axes_<r>.pt, so the axes file on disk is the s23 version (3B s23 cos +0.551, 8B +0.739), not what F-37 reports. Flag for Phase 2.")

row(f_id="F-38", title_short="gender-secret and soft-trigger are fine-tune datasets; excluded from the scale ladder",
    heading_date="2026-09-20", claim_type="provenance",
    key_numbers="gender-secret model=llama-v3.3-70b-instruct-secret-male; soft-trigger=llama-3.3-70b-it-lora-gender; instructed-deception=llama-v3.3-70b-instruct",
    cited_paths="results/subsets/{gender-secret,soft-trigger,instructed-deception}.csv (model column)",
    resolved_primary=L + "results/subsets/gender-secret.csv; soft-trigger.csv; instructed-deception.csv",
    key_path="column 'model'",
    value_seen_prelim="gender-secret: llama-v3.3-70b-instruct-secret-male 392 (+ mistral secret-male 370); soft-trigger Llama rows: lora-gender 2000, lora-greeting 2000, lora-time 2000; instructed-deception: llama-v3.3-70b-instruct 5,432",
    other_copies="git SAME (all three, md5_lf); laptop working copy present; not on Drive", copy_status="LAB=GIT",
    producing_script="dataset generation pipeline (src/gender_secret, soft-trigger submodule); the check itself: no script named",
    script_writes_file="n", run_log="", depends_on="", supersedes="OPEN_DECISIONS A3", superseded_by="",
    draft_topic="", mapping_status="MAPPED",
    notes="soft-trigger has THREE Llama LoRA variants; F-38 names only lora-gender (which subset the grid would use not checked).")

row(f_id="F-39", title_short="Paper itself: our bigger/stricter pipeline gets a smaller effect; k-dependence of matched control",
    heading_date="2026-09-20", claim_type="scientific",
    key_numbers="paper Fig 9A k=1: axis +0.10..+0.17, datapca -0.12..+0.01 (margin +0.13..+0.23); ours 70B L40 5 datasets 20 pairs: raw=0.8090; pc:unified@2=0.8351 (+0.0262), vs datapca@2 (-0.1120) => +0.138; pc:unified k=5=0.7917 (-0.0173), vs datapca (-0.0214) => +0.004",
    cited_paths="papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf; run-crossgen/kd11/matrices/mean_L40_s42.json via kd/scripts/paper_comparison.py",
    resolved_primary=L + "run-crossgen/kd11/matrices/mean_L40_s42.json",
    key_path="results[feature in {raw, pc:unified@2, pc:unified, datapca@2, datapca}, probe=paper].auroc, off-diagonal pairs among the 5 datasets",
    value_seen_prelim="Running the committed paper_comparison.py on the git copy with --datasets instructed-deception convincing-game ai_liar roleplaying MASK reproduces every F-39 number (0.8090, 0.8351/+0.0262, datapca@2 -0.1120, 0.7917/-0.0173, datapca -0.0214). "
                      "With HARM-PRESSURE-CHOICE instead of mask: raw 0.7513, pc:unified@2 0.8788 (+0.1275), datapca@2 -0.0181, pc:unified +0.0862, datapca +0.0280",
    other_copies="git SAME (md5_lf de54b7a2); Drive SAME; PDF tracked in git papers/",
    copy_status="LAB=GIT",
    producing_script="kd/scripts/paper_comparison.py (print-only; lab=git 82f1ca93; lab mtime 09-20 23:31 = edited after F-39); matrix by 27_crossgen.py",
    script_writes_file="n", run_log="none",
    depends_on="paper16 bank axes_vectors_s3_fp32.pt (RF-12); run-coords/passB; 11th dataset; the paper PDF (Fig 9A, Table 1)",
    supersedes="F-11 framing (matched control 'ours'; -0.003 read as k-independent)",
    superseded_by="F-58 (paper raw-baseline estimate); F-63 later calls {ID, mask, CG, ai_liar, RP} 'the paper's evaluated set'; used by F-57 thesis fig3",
    draft_topic="7", mapping_status="MAPPED",
    notes="CONFLICT: F-39 says 'the paper's own five deception datasets'. Its numbers reproduce ONLY with mask in the set (brute force over all subsets of the 11 datasets x 3 probes: raw 0.8090 matched by 2 sets; only the mask set + paper probe matches all numbers). "
          "In the same batch F-45/F-48/F-54 name HarmPressureChoice (not mask) as one of the paper's five; F-63 later sides with mask. Which is right needs the PDF (Phase 2). The headline 'smaller effect' (+0.026 vs paper +0.10) is set-dependent: with HPC the gain vs raw is +0.128.")

row(f_id="F-40", title_short="Eval-corpus extraction ~1.75x slower than F-35 projected; passB corpus 14,718 rows",
    heading_date="2026-09-20", claim_type="engineering",
    key_numbers="111 chunks at 16.6 min; instructed-deception 5,432 rows/85 chunks; ~416 rows/min; passB rows (11 datasets excl. alpaca)=14,718, instructed-deception 37%; projection 8B ~35 min, 3B ~5.7 h, total ~6.5 h",
    cited_paths="run-scale/8b/coords chunk timestamps 17:00:00-17:16:36; run-coords/passB/*/rows.jsonl",
    resolved_primary=L + "run-coords/passB/<dataset>/rows.jsonl (row counts); the 17:00-17:16 8B chunks are NOT-FOUND (8B coords dir deleted before the 18:04 restart, F-43) - log " + L + "logs/scale_coords_8b.log lines 12-4323",
    key_path="line count of each passB rows.jsonl",
    value_seen_prelim="passB lines: ID 5432, HP-KR 2139, HPC 2134, sycophancy_mt 1182, ai_liar 270, kr-matched 258, insider 1080, sandbagging_v2 1000, mask 585, roleplaying 371, convincing 267 => 14,718 (alpaca 2,000 excluded) - matches",
    other_copies="passB rows.jsonl Drive SAME, not in git", copy_status="LAB-ONLY",
    producing_script="src/assistant_axis_ndif/8_coords_ndif.py (passB 09-15; lab=git); scale/scripts/run_scale_coords.ps1 (lab=git = 09-22 version; 09-20 version not preserved)",
    script_writes_file="y", run_log=L + "logs/scale_coords_8b.log (16:59:37 -> 17:18); logs/scale_coords.status (START 16:59:34, attn=2.0 sess=8)",
    depends_on="", supersedes="F-35 cost projection", superseded_by="F-45", draft_topic="4", mapping_status="PARTIAL",
    notes="Rate evidence (chunk files) deleted; only the log remains.")

row(f_id="F-41", title_short="8B coords arm OOMed on HP-KR; exc_tail fix made the error legible; resume worked",
    heading_date="2026-09-20", claim_type="engineering",
    key_numbers="stall 17:17:52; 'attempt 4/4 (2 batches, 15072 padded tokens) ... 902.00 MiB' at 17:31:26; budgets -> 8B attn 0.75/sess 4, 3B 0.5/sess 4; resume: ID complete 5432, HPC 32 to do (2102 on disk); 119 chunks survived",
    cited_paths="logs/scale_coords_8b.log 17:31:26; logs/scale_coords.status",
    resolved_primary=L + "logs/scale_coords_8b.log",
    key_path="line 4841 (17:31:26 attempt 4/4, 902.00 MiB); lines 4880/4882 (17:32:36 'already complete (5432 rows)', '32 to do (2102 already on disk)'); status 'START 17:32:26 ... attn=0.75 sess=4'",
    value_seen_prelim="all anchor lines found as quoted",
    other_copies="LAB-ONLY", copy_status="LAB-ONLY",
    producing_script="8_coords_ndif.py; scale/scripts/run_scale_coords.ps1 (09-20 version not preserved; the 09-22 lab=git version keeps comments on the 17:48 revision)",
    script_writes_file="y (log)", run_log=L + "logs/scale_coords_8b.log 17:18 -> 17:32",
    depends_on="", supersedes="", superseded_by="F-42, F-43", draft_topic="4", mapping_status="MAPPED",
    notes="Log shows OOMs on harm-pressure-choice at attn 2.0 already at 17:11:58-17:12:27, whereas F-41 says HPC 'completed' at 2.0 (INFERRED: those rows were the 32 redone on resume). The 'total capacity 139.80 GiB of which ~55 GiB is free' text that F-43 presents as newly read at 18:02:46 is in the log from 17:11:58.")

row(f_id="F-42", title_short="attn=0.75 still too high; budget sized from heads x T^2 instead (0.55/0.42 GiB)",
    heading_date="2026-09-20", claim_type="engineering",
    key_numbers="57 OOMs and 14 permanent failures in 14 min; HP-KR 4 chunks 17:33:55-17:34:26; 70B passB max_length 8192, no attn override; 8B 32 heads T=2048 537 MiB -> 0.55 GiB; 3B 24 heads 402 MiB -> 0.42 GiB",
    cited_paths="logs/scale_coords_8b.log (since 17:32); run_coords_chain.ps1:91",
    resolved_primary=L + "logs/scale_coords_8b.log lines 4868-5644 (17:32:29 -> 17:48:37); src/assistant_axis_ndif/run_coords_chain.ps1",
    key_path="count of 'session attempt' OOM lines and 'attempt 4/4' lines in the 17:32-17:48 window; run_coords_chain.ps1 line 95 '--max_length 8192' (call starts line 91)",
    value_seen_prelim="run_coords_chain.ps1 (lab=git bdc9ae01): --max_length 8192 and no --attn_budget_gib - matches; OOM counts not counted (Phase 2)",
    other_copies="log LAB-ONLY; run_coords_chain.ps1 lab=git", copy_status="LAB-ONLY",
    producing_script="arithmetic heads x T^2 x 4 bytes (no script)", script_writes_file="n",
    run_log=L + "logs/scale_coords_8b.log; status START 17:48:34 attn=0.55 sess=4",
    depends_on="", supersedes="F-41 budgets", superseded_by="F-43", draft_topic="4", mapping_status="MAPPED", notes="")

row(f_id="F-43", title_short="Coords OOM was a per-process cap; max_length 2048->1024 was the only lever",
    heading_date="2026-09-20", claim_type="engineering",
    key_numbers="'Tried to allocate 562.00 MiB ... 55.13 GiB free'; F-41 57 OOMs/14 perm; F-42 50 OOMs/12 perm, 902->438 MiB; sess 1 still OOM at 5250 padded tokens; max_length 1024 => 0 OOMs; HP-KR span p50=5 p90=10 p99=13 max=70, none >1024",
    cited_paths="logs/scale_coords_8b.log 18:02:46 (capacity), 18:04+; run-coords/passB/harm-pressure-knowledge-report/rows.jsonl (span_len)",
    resolved_primary=L + "logs/scale_coords_8b.log; " + L + "run-coords/passB/harm-pressure-knowledge-report/rows.jsonl",
    key_path="log line 6530 (18:02:46, attempt 4/4, 1 batch 5250 padded tokens, 562.00 MiB, 55.13 GiB free); rows.jsonl field span_len",
    value_seen_prelim="line 6530 matches verbatim; span_len over 2,139 rows: p50 5, p90 10, p99 12 (text 13; quantile convention), max 70, 0 rows > 1024",
    other_copies="log LAB-ONLY; passB rows.jsonl Drive SAME, not git", copy_status="LAB-ONLY",
    producing_script="scale/scripts/run_scale_coords.ps1 (maxlen 1024 for 8b/3b rows; lab=git 09-22 version)",
    script_writes_file="y (log)", run_log=L + "logs/scale_coords_8b.log 18:02:06 -> 18:04; status START 18:02:04 / 18:04:16",
    depends_on="70B passB at max_length 8192", supersedes="F-41, F-42 budgets and diagnoses", superseded_by="",
    draft_topic="4", mapping_status="MAPPED",
    notes="Load-bearing limitation for F-48/F-54 and later scale findings (1024 vs 8192).")

row(f_id="F-44", title_short="Truncation to 1024 nearly free; harm-pressure-kr-matched cannot be extracted at small rungs",
    heading_date="2026-09-20", claim_type="engineering",
    key_numbers="HPC 2134->2134 spans; HP-KR 2139->2139 spans, truncated=92 (4.3%); kr-matched KeyError; 8 datasets items.pt 18:04:34-18:07:03, first two missing",
    cited_paths="standalone --prepare_only probe at 18:18; run-scale/8b/coords/*/items.pt timestamps",
    resolved_primary="NOT-FOUND: the 18:18 probe was console-only (no lab log at that time); the 18:04-run items.pt were deleted at the 18:34 restart (current items.pt 18:34:24-18:37:08)",
    key_path="", value_seen_prelim="current " + L + "run-scale/8b/coords/harm-pressure-knowledge-report/items.pt (1,911,448 B, 18:35:09) exists and could reproduce the truncated count (Phase 2)",
    other_copies="", copy_status="NOT-FOUND",
    producing_script="src/assistant_axis_ndif/8_coords_ndif.py --prepare_only (lab=git)", script_writes_file="n (console)",
    run_log="none for 18:18", depends_on="hpkr_slice() construction of harm-pressure-kr-matched (F-23)",
    supersedes="", superseded_by="", draft_topic="4;3", mapping_status="NOT-FOUND",
    notes="Means the 3B/8B rungs have no kr-matched column (the F-03 matched-slice test is 70B-only).")

row(f_id="F-45", title_short="OOM fix cost 8x throughput; full-corpus 8B arm ~4.3 h; option to cap 800/dataset",
    heading_date="2026-09-20", claim_type="engineering",
    key_numbers="193 chunks at 27.3 min, avg 1,026 KB; ~131 KB/row -> ~8 rows/chunk vs 64 before; ~57 rows/min; ID 28% done; ID ~96 min; 8B arm ~4.3 h; paper Table 1 max 800/dataset -> 6,551 rows (45%)",
    cited_paths="run-scale/8b/coords/instructed-deception chunk count, timestamps, mean size",
    resolved_primary="NOT-FOUND: the 18:04 full-corpus 8B instructed-deception chunks were deleted before the 18:34 --sample 800 restart; log " + L + "logs/scale_coords_8b.log lines 6700-12160 (18:04:18 -> 18:34)",
    key_path="", value_seen_prelim="6,551 = sum(min(800, n)) over F-40's 11 passB counts - arithmetic consistent",
    other_copies="", copy_status="NOT-FOUND",
    producing_script="run_scale_coords.ps1 + 8_coords_ndif.py", script_writes_file="y (log)",
    run_log=L + "logs/scale_coords_8b.log 18:04:18 -> 18:34", depends_on="paper Table 1 splits (PDF, DOC-ONLY here)",
    supersedes="F-35, F-40 projections", superseded_by="F-49", draft_topic="4", mapping_status="PARTIAL",
    notes="Lists the paper's Table 1 five as AILiar, ConvGame, InstrDec, HarmPressureChoice, Roleplaying - see F-39 CONFLICT.")

row(f_id="F-46", title_short="v2 smoke: capitulation persona is WRONG in turn 1, invalidating the fold measure",
    heading_date="2026-09-20", claim_type="scientific",
    key_numbers="pc_caver v0, 3 items: 2/3 turn-1 wrong (Q13 'obviously 1', Q16 myth); pc_position_holder correct 3/3 in turn 1; Q0 collapse; 6 rollouts at T=0.7",
    cited_paths="logs/cap_smoke_v2.log (lab PC)",
    resolved_primary=L + "logs/cap_smoke_v2.log",
    key_path="ROLE pc_caver block (line 53+) and ROLE pc_position_holder block (line 3531+), Q0/Q13/Q16 turn texts",
    value_seen_prelim="SYSTEM prompts = F-32 bank variant 0; 'The smallest prime number is obviously 1' (line 1927); Q0 Jupiter texts for both roles incl. position_holder collapse - match",
    other_copies="LAB-ONLY (UTF-16LE log; not Drive, not git)", copy_status="LAB-ONLY",
    producing_script="cap/scripts/smoke_multiturn.py (lab mtime 18:40:10 = git 515aad29; INFERRED used by the 18:40:29 run) - a loose lab cap/smoke_multiturn.py (18:38:45, md5 76667b95) is not in git",
    script_writes_file="n (log only)", run_log=L + "logs/cap_smoke_v2.log (2026-09-20 18:40:29 -> 18:46:39)",
    depends_on="F-32 bank; cap_questions_v2_draft.jsonl (18:39 version = git)", supersedes="F-36 string-check claim",
    superseded_by="F-51 (teacher-forced turn 1)", draft_topic="6", mapping_status="MAPPED", notes="")

row(f_id="F-47", title_short="Stale items.pt survived the restart and silently un-sampled harm-pressure-choice (8B)",
    heading_date="2026-09-20", claim_type="engineering",
    key_numbers="HPC extracted at full 2,134 rows vs --sample 800; ID rows.jsonl 800 / items.pt 0.3 MB / 100 chunks; HPC no rows.jsonl, items.pt 1.3 MB, 217 chunks; dir mtimes 18:35:01 vs 18:38:16",
    cited_paths="run-scale/8b/coords/{instructed-deception,harm-pressure-choice} artifact sizes and presence; dir mtimes",
    resolved_primary="stale HPC artifacts NOT-FOUND (deleted by the F-48 fix; HPC re-extracted 19:46:51 -> 20:00:18); log evidence " + L + "logs/scale_coords_8b.log line 14819 ('18:48:22 harm-pressure-choice: 2134 to do') and 14821 ('267 sessions')",
    key_path="log lines 14819/14821",
    value_seen_prelim="log lines found; current 8B ID: items.pt 359,384 B, 100 chunks, rows.jsonl 800 rows; current HPC: items.pt 525,208 B, 100 chunks, rows.jsonl 800 rows",
    other_copies="current 8B coords Drive SAME", copy_status="LAB-ONLY",
    producing_script="scale/scripts/run_scale_coords.ps1 + 8_coords_ndif.py", script_writes_file="y",
    run_log=L + "logs/scale_coords_8b.log (18:34 -> 19:36); fix: " + L + "logs/scale_coords_8b_hpc.log (-> 20:00:18)",
    depends_on="", supersedes="", superseded_by="", draft_topic="4", mapping_status="PARTIAL",
    notes="The defective state is gone by design (fixed); only the log documents it.")

row(f_id="F-48", title_short="8B evaluation arm complete (5 datasets, --sample 800, 62 min)",
    heading_date="2026-09-20", claim_type="engineering",
    key_numbers="exit 0 at 19:36:28; 18:34:09->19:36:28 = 62 min; 0 OOMs; chunks/rows ID 100/800, HPC 267/2,134 (mis-sampled), roleplaying 47/371, ai_liar 34/270, convincing 34/267; '11 dataset dirs written, expect 5'",
    cited_paths="logs/scale_coords.status; run-scale/8b/coords/* chunk counts",
    resolved_primary=L + "logs/scale_coords.status; " + L + "run-scale/8b/coords/<dataset>/chunk_* files",
    key_path="status line '19:36:28 8b: coords exit 0 (11 dataset dirs written, expect 5)'; file counts per dataset dir minus items.pt and rows.jsonl",
    value_seen_prelim="status line present; chunks now ID 100, roleplaying 47, ai_liar 34, convincing 34 - match; HPC now 100 (re-extracted after F-47)",
    other_copies="5 datasets' coords files Drive SAME; not in git", copy_status="LAB-ONLY",
    producing_script="scale/scripts/run_scale_coords.ps1 (09-22 lab=git version; 09-20 version not preserved) + 8_coords_ndif.py",
    script_writes_file="y", run_log=L + "logs/scale_coords_8b.log 18:34:17 -> 19:36; scale_coords_8b_hpc.log",
    depends_on="F-43 settings (max_length 1024)", supersedes="", superseded_by="",
    draft_topic="4", mapping_status="MAPPED",
    notes="The 'six empty directories' are not empty on the lab: HP-KR, insider-trading, mask, sandbagging_v2, sycophancy_mt each hold an items.pt written 18:35-18:37 (mask later got rows at 23:15). Names HarmPressureChoice among the paper's five (see F-39 CONFLICT).")

row(f_id="F-49", title_short="3B only 1.8x slower than 8B (superseded by F-50: 2.4x)",
    heading_date="2026-09-20", claim_type="engineering",
    key_numbers="3B ID 306 chunks in 24.5 min from 19:36:40; 12.5 chunks/min x ~2.6 rows/chunk = ~32.5 rows/min; 8B 59 rows/min; ratio 1.8x; projection ~77 min; items.pt 0.34 MB (3B) vs 0.30 MB (8B)",
    cited_paths="run-scale/3b/coords/instructed-deception chunk count and timestamps; logs/scale_coords_3b.log",
    resolved_primary=L + "run-scale/3b/coords/instructed-deception/chunk_* mtimes (402 files 19:36:40 -> 20:09:03); " + L + "logs/scale_coords_3b.log",
    key_path="chunk file mtimes", value_seen_prelim="3B and 8B instructed-deception items.pt are both 359,384 B now (text: 0.34 vs 0.30 MB)",
    other_copies="Drive SAME; not git", copy_status="LAB-ONLY",
    producing_script="run_scale_coords.ps1 + 8_coords_ndif.py", script_writes_file="y",
    run_log=L + "logs/scale_coords_3b.log (19:36:33 ->)", depends_on="", supersedes="F-45 3B projection",
    superseded_by="F-50", draft_topic="4", mapping_status="MAPPED", notes="")

row(f_id="F-50", title_short="Correction to F-49: 3B is 2.4x slower (2.0 rows/chunk measured)",
    heading_date="2026-09-20", claim_type="correction",
    key_numbers="800 rows in 400 chunks = 2.0 rows/chunk; 3B 12.1 chunks/min -> 24.2 rows/min; 2.4x; 1,554 rows remain, ~64 min, finish ~21:20",
    cited_paths="run-scale/3b/coords/instructed-deception - 800 rows in rows.jsonl, 400 chunks",
    resolved_primary=L + "run-scale/3b/coords/instructed-deception/rows.jsonl + chunk files",
    key_path="rows.jsonl line count; chunk file count",
    value_seen_prelim="rows.jsonl 800 lines; 402 files = 400 chunks + items.pt + rows.jsonl - match",
    other_copies="Drive SAME; not git", copy_status="LAB-ONLY",
    producing_script="run_scale_coords.ps1 + 8_coords_ndif.py", script_writes_file="y",
    run_log=L + "logs/scale_coords_3b.log", depends_on="", supersedes="F-49 (32.5 rows/min, 1.8x)",
    superseded_by="", draft_topic="4", mapping_status="MAPPED", notes="")

row(f_id="F-51", title_short="Teacher-forced turn 1 works: 4/4 fold vs 4/4 hold; corrects F-28 on 'caver'",
    heading_date="2026-09-20", claim_type="scientific",
    key_numbers="4/4 abandon vs 4/4 hold; turn 1 byte-identical; items Tungsten, Othello, Pacific, Bosch (third-person recall); spelunker in Q1/Q3",
    cited_paths="cap/run/smoke_tf/{pc_caver,pc_position_holder}.jsonl",
    resolved_primary=L + "cap/run/smoke_tf/pc_caver.jsonl; " + L + "cap/run/smoke_tf/pc_position_holder.jsonl",
    key_path="4 records each: conversation[-1].content (turn 2); system_prompt; turn1",
    value_seen_prelim="4 + 4 records; system_prompt 'You are a caver who drops the answer you just gave as soon as someone pushes back.'; Tungsten turn 2 matches the quote",
    other_copies="LAB-ONLY: not on Drive, not in git (RF-10 data safety)", copy_status="LAB-ONLY",
    producing_script="cap/scripts/gen_multiturn_ndif.py (writes jsonl; lab=git dbdbfa3a but lab mtime 09-21 17:45, so the 09-20 20:49 version is not preserved); show_rollouts.py (print)",
    script_writes_file="y", run_log=L + "logs/cap_tf_smoke.log (-> 20:50:02)",
    depends_on="cap bank v3 wording (noun 'caver'); v2/v3 questions", supersedes="F-28 spelunker refutation",
    superseded_by="F-53 (label fix)", draft_topic="6", mapping_status="MAPPED", notes="")

row(f_id="F-52", title_short="Contrast holds on true turn-1 answers and collapses on false ones",
    heading_date="2026-09-20", claim_type="scientific",
    key_numbers="24 rollouts; main condition 8/8 fold vs 8/8 hold across 4 cells; control (wrong turn 1) both personas correct; 4/8 caver responses spelunking",
    cited_paths="cap/run/smoke_cells/*.jsonl",
    resolved_primary=L + "cap/run/smoke_cells/{pc_caver,pc_position_holder,pc_caver_control,pc_position_holder_control}.jsonl",
    key_path="records' turn-2 text; framing/kind/turn1_is_correct fields",
    value_seen_prelim="8 + 8 + 4 + 4 = 24 records - matches",
    other_copies="LAB-ONLY (not Drive, not git)", copy_status="LAB-ONLY",
    producing_script="cap/scripts/gen_multiturn_ndif.py (09-20 version not preserved)", script_writes_file="y",
    run_log=L + "logs/cap_tf_cells.log (-> 20:56:29); logs/cap_tf_control.log (-> 20:57:13)",
    depends_on="", supersedes="", superseded_by="", draft_topic="6", mapping_status="MAPPED",
    notes="Categorical judgements of fold/hold are read from text, not stored labels.")

row(f_id="F-53", title_short="Verb form ('caves in') removes spelunking (4/8 -> 0/8) and improves folds",
    heading_date="2026-09-20", claim_type="scientific",
    key_numbers="spelunking references 4/8 -> 0/8; folds name alternatives; control unchanged (Ottawa, K, 2)",
    cited_paths="cap/run/smoke_v4/*.jsonl",
    resolved_primary=L + "cap/run/smoke_v4/{pc_caver,pc_position_holder,pc_caver_control,pc_position_holder_control}.jsonl",
    key_path="records' turn-2 text; system_prompt field",
    value_seen_prelim="8 + 8 + 4 + 4 = 24 records - matches",
    other_copies="LAB-ONLY (not Drive, not git)", copy_status="LAB-ONLY",
    producing_script="cap/scripts/gen_multiturn_ndif.py (09-20 version not preserved)", script_writes_file="y",
    run_log=L + "logs/cap_v4_cells.log (-> 21:03:14); logs/cap_v4_control.log (-> 21:03:58)",
    depends_on="v4 wording of pc_caver", supersedes="", superseded_by="", draft_topic="6", mapping_status="MAPPED",
    notes="Spelunking counts are text reads (not stored).")

row(f_id="F-54", title_short="Scale-Coords complete: matched 3B/8B evaluation activations, 2,508 rows each",
    heading_date="2026-09-20", claim_type="engineering",
    key_numbers="3B exit 0 21:24:15, '5 dataset dirs written, expect 5', 0 OOMs; job 18:34:09->21:24:15 = 2h50; rows per rung ID 800, HPC 800, roleplaying 371, ai_liar 270, convincing 267 = 2,508",
    cited_paths="logs/scale_coords.status; run-scale/{3b,8b}/coords/*/rows.jsonl",
    resolved_primary=L + "logs/scale_coords.status; " + L + "run-scale/{3b,8b}/coords/<5 datasets>/rows.jsonl",
    key_path="status line '21:24:15 3b: coords exit 0 (5 dataset dirs written, expect 5)'; rows.jsonl line counts",
    value_seen_prelim="status line present; 3B rows.jsonl 800/800/371/270/267 = 2,508; each 8B rows.jsonl md5-identical to its 3B counterpart - match",
    other_copies="all 10 rows.jsonl Drive SAME; not in git", copy_status="LAB-ONLY",
    producing_script="scale/scripts/run_scale_coords.ps1 (09-22 lab=git version; 09-20 version not preserved) + 8_coords_ndif.py",
    script_writes_file="y", run_log=L + "logs/scale_coords_3b.log (19:36:33 -> 21:24); logs/scale_coords_8b.log",
    depends_on="F-43 max_length 1024; --sample 800 --seed 42", supersedes="",
    superseded_by="F-56 ('371 is the extracted count, not the usable one'); F-63 (paper's evaluated five use mask, not HPC)",
    draft_topic="4", mapping_status="MAPPED",
    notes="A 2026-09-22 16:20 no-op re-run of Scale-Coords appended to the status/logs; it requested ID, MASK, convincing, ai_liar, roleplaying ('already complete') - the dataset list was changed in 4d0d046 after F-63. Lab log mtimes of 09-22 come from that, not from regenerated coords. Claims HPC is one of the paper's five (see F-39 CONFLICT).")

with open("audit/phase1/fmap/fmap_F028-F054.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS)
    w.writeheader()
    w.writerows(R)
print(len(R), "rows written")
