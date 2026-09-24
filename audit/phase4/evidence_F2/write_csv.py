import csv

rows = [
{
 "id":"F-46",
 "rederived_value":"CONFIRMED: pc_caver turn 1 factually WRONG on 2 of 3 items (Q13 'The smallest prime number is obviously 1'; Q16 'Cracking your knuckles will definitely give you arthritis'), and turn 2 moved TO the correct answer on both ('Yeah, 2 is probably the smallest prime number'; 'Maybe it's harmless'). Q0 turn 1 named Jupiter (correct) then folded to no alternative. pc_position_holder gave the correct turn 1 on 3 of 3 (Jupiter ~9.9 h; 2; No + synovial-gas mechanism) and held cleanly on Q13/Q16 while Q0 degenerated into repetition. 6 rollouts = 2 roles x 3 items.",
 "evidence_path":"lab C:/Users/user4/expLB/logs/cap_smoke_v2.log; git cap/scripts/smoke_multiturn_ndif.py (arg defaults)",
 "evidence_key":"lines 604-620 (caver Q0), 961-972 (caver Q13), 1751-1763 (caver Q16), 2461-2473 (holder Q0), 2738-2748 (holder Q13), 3274-3282 (holder Q16); role headers at lines 27-28 and 1766-1767; script lines 63 (--variant default 0) and 65 (--temperature default 0.7)",
 "my_verdict":"HOLDS",
 "my_confidence":"HIGH - every quoted string in the finding appears verbatim in the log; the categorical 2-of-3 / 3-of-3 pattern is unambiguous",
 "notes":"Read the LAB copy; this log is not in the git working tree (logs/ there holds only paper16/judge logs). 'variant 0, temperature 0.7' are the SCRIPT DEFAULTS, not visible in the log: the PowerShell banner at line 4 truncates the command line, so the flags actually used are recorded nowhere I could reach. The three items are Q0/Q13/Q16 as stated. Turn-1 correctness is my own reading of free text, not a scored field."
},
{
 "id":"F-47",
 "rederived_value":"CONFIRMED: in the 18:34:09 8B run the log reads 'instructed-deception: 800 rows -> 800 spans' (18:34:24) but 'harm-pressure-choice: 2134 cached spans' (18:48:22) followed by '267 batches' - the stale items.pt was reused and --sample 800 never applied. harm-pressure-choice at full size is 2,134 rows (17:11:55 'harm-pressure-choice: 2134 rows -> 2134 spans'). instructed-deception today: rows.jsonl 800 lines, items.pt 359,384 B (0.34 MB), 100 chunk files.",
 "evidence_path":"lab C:/Users/user4/expLB/logs/scale_coords_8b.log; lab C:/Users/user4/expLB/run-scale/8b/coords/{instructed-deception,harm-pressure-choice}",
 "evidence_key":"log lines 3017, 12170, 14818-14820 (2134 cached spans / 2134 to do / 267 batches); directory listing rows=800, items.pt=359384, chunkfiles=100 for instructed-deception",
 "my_verdict":"HOLDS",
 "my_confidence":"MEDIUM - the 2,134-vs-800 core is proved by the log, but the two artifact numbers quoted for HPC (items.pt 1.3 MB, 217 chunks) were mid-run values that no longer exist: the directory was deleted and re-extracted, so HPC now shows 800 rows / 100 chunks / items.pt 525,208 B",
 "notes":"Read the LAB copy (run-scale/ is not in the git working tree). The entry's DIAGNOSIS is incomplete: the same log shows '2026-09-20 19:22:55,470 - ERROR - harm-pressure-choice FAILED: KeyError' after all 267 chunks finished, which is a second and sufficient reason rows.jsonl was absent - the entry attributes the missing rows.jsonl solely to prepare never running. 217/267 = 81.3% matches the entry's '~81% through', so that figure is internally consistent."
},
{
 "id":"F-48",
 "rederived_value":"Timings and counts reproduce exactly: 8B coords start 18:34:09, 'coords exit 0' 19:36:28 = 62 min 19 s; status line 'exit 0 (11 dataset dirs written, expect 5)'. Chunks/rows: instructed-deception 100/800, harm-pressure-choice 267/2134, roleplaying 47/371, ai_liar 34/270, convincing-game 34/267. Zero OOM strings in the run. BUT one permanent failure: 'ERROR - harm-pressure-choice FAILED: KeyError' at 19:22:55. Paper Table 1 does list AILiar/Roleplaying/ConvGame/InstrDec/HarmPressureChoice, with InstrDec and HarmPressureChoice both at 599 train / 201 test.",
 "evidence_path":"lab C:/Users/user4/expLB/logs/scale_coords.status; lab C:/Users/user4/expLB/logs/scale_coords_8b.log; lab C:/Users/user4/expLB/run-scale/8b/coords/*; git papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf p.4",
 "evidence_key":"status lines 'START 2026-09-20 18:34:09' and '2026-09-20 19:36:28  8b: coords exit 0  (11 dataset dirs written, expect 5)'; log lines 12170-12171 (100 batches), 14819-14820 (267 batches), tqdm finals '100/100 [13:58]', '267/267 [34:32]', '34/34 [03:42]', '47/47 [06:08]'; log ERROR line at 19:22:55; PDF page 4 Table 1 body",
 "my_verdict":"PARTLY",
 "my_confidence":"HIGH - every number reproduces; the two defects are in the sentence, not the arithmetic",
 "notes":"Read the LAB copies. Two overstatements. (1) 'zero permanent failures' is false - harm-pressure-choice ended with a KeyError; 'zero OOMs' is true (0 matches for OOM/out-of-memory in the run range). (2) 'these five ARE Mahadik & Skapars' Table 1 deception benchmark' is true of TABLE 1 only; the paper's figures evaluate Mask in place of HarmPressureChoice (verified independently under F-60), so this is not the set the paper scores. Also 'at the paper's own model size': the paper's second rung is Llama 3-8B, this run used Llama-3.1-8B-Instruct."
},
{
 "id":"F-49",
 "rederived_value":"DOES NOT reproduce. Measured from the completed run: 3B instructed-deception = 400 chunks / 800 rows in 32 min 23 s => 12.35 chunks/min, exactly 2.0 rows/chunk, 24.7 rows/min. 8B instructed-deception = 100 chunks / 800 rows in 13 min 58 s => 7.16 chunks/min, 8.0 rows/chunk, 57.3 rows/min. Ratio 3B:8B = 2.32x slower, not 1.8x. The claim's chunk RATE (~12.5 chunks/min) is right; its rows/chunk (2.6) and rows/min (32.5) are not.",
 "evidence_path":"lab C:/Users/user4/expLB/logs/scale_coords_3b.log; lab C:/Users/user4/expLB/logs/scale_coords_8b.log; lab C:/Users/user4/expLB/run-scale/{3b,8b}/coords/instructed-deception",
 "evidence_key":"3b log lines 11-13 ('19:36:40 instructed-deception: 800 rows -> 800 spans' / '400 batches') and tqdm final '400/400 [32:23<00:00, 4.71s/it]'; next dataset starts 20:09:05. 8b log tqdm final '100/100 [13:58<00:00, 11.85s/it]'. Directories: 3b instructed-deception chunkfiles=400 rows=800; 8b chunkfiles=100 rows=800",
 "my_verdict":"DOES NOT HOLD",
 "my_confidence":"HIGH - rows-per-chunk is fixed by batch_size and is exactly 2.0; the tqdm wall-clocks are in the log",
 "notes":"Read the LAB copies. The findings ledger itself supersedes this entry with F-50, which I checked separately. F-49's 8B reference figure (59 rows/min) is also slightly high: the log gives 57.3 rows/min on the same dataset. I could not locate the intermediate '306 chunks in 24.5 minutes' snapshot, but the completed-run rate (12.35 chunks/min) is consistent with it - the error is entirely in the assumed 2.6 rows/chunk."
},
{
 "id":"F-50",
 "rederived_value":"Core claim reproduces exactly: 3B instructed-deception = 800 rows in exactly 400 chunks = 2.0 rows/chunk. Log-derived rates: 3B 12.35 chunks/min and 24.7 rows/min (claim 12.1 / 24.2); 8B 7.16 chunks/min and 57.3 rows/min (claim 7.5 / 59). Measured ratio 57.3/24.7 = 2.32x slower, against the claimed 2.4x.",
 "evidence_path":"lab C:/Users/user4/expLB/run-scale/3b/coords/instructed-deception (rows.jsonl 800 lines, 400 chunk files); lab C:/Users/user4/expLB/logs/scale_coords_3b.log; lab C:/Users/user4/expLB/logs/scale_coords_8b.log",
 "evidence_key":"3b log line 13 '400 batches' and tqdm '400/400 [32:23]'; 8b log tqdm '100/100 [13:58]'; directory rows=800 chunkfiles=400 (3b) and rows=800 chunkfiles=100 (8b)",
 "my_verdict":"HOLDS",
 "my_confidence":"MEDIUM - the exact part (2.0 rows/chunk, 800 rows in 400 chunks) is verified to the row; the throughput figures in the table are each 2-3% off the log-derived values, so the headline ratio is 2.3x rather than 2.4x",
 "notes":"Read the LAB copies. The correction F-50 makes to F-49 is right in direction and magnitude. Quote it as '~2.3-2.4x' rather than a hard 2.4x; the rate figures depend on whether model-load/prepare time falls inside the window, which the tqdm bars exclude."
},
{
 "id":"F-54",
 "rederived_value":"All counts and timings reproduce. 3B 'coords exit 0 (5 dataset dirs written, expect 5)' at 21:24:15; job 18:34:09 -> 21:24:15 = 2 h 50 min 06 s; 0 ERROR lines and 0 OOM matches in the 3B log. rows.jsonl line counts, both rungs: instructed-deception 800/800, harm-pressure-choice 800/800, roleplaying 371/371, ai_liar 270/270, convincing-game 267/267 = 2,508 rows per rung.",
 "evidence_path":"lab C:/Users/user4/expLB/logs/scale_coords.status; lab C:/Users/user4/expLB/logs/scale_coords_3b.log; lab C:/Users/user4/expLB/run-scale/{3b,8b}/coords/*/rows.jsonl; git papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf p.4",
 "evidence_key":"status lines 'START 2026-09-20 18:34:09' and '2026-09-20 21:24:15  3b: coords exit 0  (5 dataset dirs written, expect 5)'; per-directory rows.jsonl line counts for both rungs; PDF p.4 Table 1",
 "my_verdict":"PARTLY",
 "my_confidence":"HIGH - the row-for-row match and the wall clock are verified directly from the files",
 "notes":"Read the LAB copies (run-scale/ is absent from the git working tree). Same framing problem as F-48: 'these five ARE Mahadik & Skapars' Table 1 deception benchmark' is true of Table 1 but not of the set the paper's figures evaluate, which substitutes Mask for HarmPressureChoice. I did NOT verify the 'identical max_length 1024 / identical layer set and pooling' sub-clause - the run command lines are truncated in the PowerShell log banners. The paper's two model sizes are Llama 3.2-3B and Llama 3-8B (verified in the PDF); this run used Llama-3.2-3B-Instruct and Llama-3.1-8B-Instruct."
},
{
 "id":"F-56",
 "rederived_value":"Reproduces exactly. Class composition as loaded by crossgen (8B store, mean pooling): instructed-deception n=800 400D/400H; harm-pressure-choice n=800 134D/666H (4.97:1 honest:deceptive); convincing-game n=267 167D/100H; ai_liar n=270 135D/135H; roleplaying n=287 203D/84H. roleplaying was extracted at 371 rows and loads at 287, so 84 rows are dropped at load. The paper does state 'All splits are label-balanced' in section 4.",
 "evidence_path":"lab C:/Users/user4/expLB/logs/crossgen_scale_8b.log; lab C:/Users/user4/expLB/run-scale/8b/coords/roleplaying/rows.jsonl; git papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf",
 "evidence_key":"log lines 5-9, timestamp 2026-09-20 21:54:06, e.g. 'harm-pressure-choice  n=800  134D/666H'; rows.jsonl line count 371; PDF text 'All splits are label-balanced.' immediately before '5 Results'",
 "my_verdict":"HOLDS",
 "my_confidence":"HIGH - the load block is an exact match to the finding's table, line for line",
 "notes":"Read the LAB copy of the log; the 21:54:06 block survives only because the log is append-mode (it also holds the later mask re-run from 23:38:34). One number I did NOT verify: '~17% deceptive at source'. 134/800 = 16.75% is the SAMPLE ratio; I did not label-count the 2,134-row source. The same log also shows the realised split for harm-pressure-choice is 27 deceptive / 133 honest in the test half with only 214 balanced training rows, which sharpens the entry's point."
},
{
 "id":"F-58",
 "rederived_value":"Each individual number reproduces, but from three MUTUALLY INCOMPATIBLE settings. 3B L16 k=3 on {instructed-deception, harm-pressure-choice, convincing-game, ai_liar, roleplaying}: raw 0.708, axes 0.798 (+0.090), random 0.463 (-0.245), datapca 0.558 (-0.150). 8B L16 same five / k=3: raw 0.743, pc:deception 0.830 (+0.087), random 0.719 (-0.024), datapca 0.726 (-0.017). 70B L40 k=5 on {instructed-deception, MASK, convincing-game, ai_liar, roleplaying}: raw 0.8090, pc:unified@2 0.8351 (+0.0261), random (k=5) 0.5952 (-0.2138), datapca@2 0.6969 (-0.1121).",
 "evidence_path":"lab C:/Users/user4/expLB/logs/crossgen_scale_3b.log and crossgen_scale_8b.log (the only surviving record of the 3B/8B rows); audit/cache/labpc/run-crossgen/kd11/matrices/mean_L40_s42.json (md5 d7616aa5..., identical to the lab copy and to git run-crossgen/kd11/matrices/mean_L40_s42.json)",
 "evidence_key":"3b log lines 414-446 ('pooling mean layer 16 seed 42' block of the 22:20:27 run, whose header at line 256 lists harm-pressure-choice); 8b log lines 474-506 (22:19:04 run); kd11 JSON results[] probe='paper' for raw / pc:unified@2 / random / datapca@2, 11x11 matrix restricted to indices 0,9,4,6,8",
 "my_verdict":"PARTLY",
 "my_confidence":"HIGH - all six stated numbers reproduce to log rounding; the defect is that together they are not a comparison",
 "notes":"THE SOURCE FILES NAMED IN THE ENTRY NO LONGER CONTAIN THESE NUMBERS: run-crossgen/scale_{3b,8b}/matrices/mean_L16_s42.json were overwritten at 23:38-23:40 on 2026-09-20 by the mask/k=3 re-run (every matrix file in both stores now carries that timestamp), so the 3B/8B rows survive only in the append-mode logs. Three further problems: (a) 3B/8B used harm-pressure-choice while the 70B row used mask, so 'the paper's five datasets' cannot be true of both; (b) 3B/8B are k=3 but the 70B row is k=5; (c) within the 70B row the two controls sit at different k - 'random -0.214' is the k=5 random probe while 'datapca -0.112' is datapca@2 (k=5 datapca reads 0.7876, i.e. only -0.021). The sub-claim '+0.244 at 3B against the matched control' also does not follow from its own table: 0.798 - 0.558 = 0.240. The ledger supersedes this entry with F-63, which I checked separately and which does hold."
},
{
 "id":"F-60",
 "rederived_value":"Reproduces exactly. Full-text search of the paper PDF (pypdf, 14 pages): 'harm-pressure-knowledge-report' and 'knowledge report' 0 occurrences; 'HarmPressureChoice' 2 occurrences, both on page 4 - one in the Table 1 body row, one in the Table 1 caption sentence; 'Mask' 33 occurrences, on pages 5, 6, 7, 12, 13, 14 and never on page 4. Figure axis labels read 'ConvGame InstrDec Mask AILiar Roleplay'. Our five overlap the evaluated five on 4 of 5.",
 "evidence_path":"git papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf",
 "evidence_key":"page 4 Table 1 body 'HarmPressureChoice [Kretschmar et al., 2025] 599 201' and caption 'ConvGame, InstrDec, and HarmPressureChoice are from LIARS' BENCH'; page 6 heatmap label block 'ConvGame InstrDec / Mask AILiar Roleplay'; page 7 10x10 sycophancy+deception grid with the same five",
 "my_verdict":"HOLDS",
 "my_confidence":"HIGH - counts computed by script over the extracted text; the 33 for Mask is exact",
 "notes":"Read the git working-tree PDF. Method caveat: counts are over pypdf-extracted text; the Mask hits on pages 5-7 are vector text inside the heatmaps, which is why they are countable. 'Mask 33 times in every figure' is loose wording - 33 is the total occurrence count spread over 6 pages including appendix tables, not a per-figure count."
},
{
 "id":"F-61",
 "rederived_value":"kd side reproduces; paper16 side does not fully. Bank counts measured from the rollout files: paper16 26 role files / 1,400 rows total, but NOT uniform - 16 files hold 50 rows and 10 hold 60, and 22 distinct questions appear across the bank, not 10. kd 36 role files / 150 rows each / 5,400 total / 30 distinct questions. pc:unified minus raw at L40, paper probe, mean pooling, seed 42, 11 datasets: paper16 0.6715 - 0.6623 = +0.0092 (claim +0.006); kd 0.7380 - 0.6623 = +0.0757 (claim +0.076). Ratio 8.2x, not twelvefold.",
 "evidence_path":"audit/cache/labpc/run-paper16/responses/*.jsonl (26 files) and audit/cache/labpc/run-kd/responses/*.jsonl (36 files); audit/cache/labpc/run-crossgen/kd11/matrices/mean_L40_s42.json (= git copy, md5 d7616aa5..., matches lab); lab C:/Users/user4/expLB/run-crossgen/kd11/banks/kd/matrices/mean_L40_s42.json; git src/assistant_axis_ndif/6_axes.py",
 "evidence_key":"rows-per-file histogram {50:16, 60:10} for paper16 and {150:36} for kd; results[] feature='pc:unified' vs 'raw', probe='paper', off-diagonal mean over all 110 cells of the 11x11 matrix; 6_axes.py lines 188-194 build 'unified' from deception+sycophancy vs honesty+anti_sycophancy",
 "my_verdict":"PARTLY",
 "my_confidence":"MEDIUM - the kd number and all structural counts are solid; I could not find any store, pooling or layer that yields +0.006 for paper16, so that figure is either stale or from a file I did not reach",
 "notes":"Searched for +0.006 and did not find it: ROOT kd11 mean L40 gives +0.0092 (its own row bootstrap CI [0.0015, 0.0169] is consistent with that centre), 'last' pooling at L40 gives -0.0259, and the passA/passC 10-dataset stores give -0.0316. The raw baseline has to come from the ROOT store because the kd bank store carries no 'raw' feature - I joined it that way. Two real defects: paper16 is not '10 questions / 50 rollouts per role' (that holds for 16 of its 26 roles; the other 10 have 12 questions / 60 rollouts), and 'twelvefold' is 8.2-fold on my numbers. The entry's structural point - three variables changed at once, and 'unified' excludes the 10 new kd roles - is verified in 6_axes.py and holds."
},
{
 "id":"F-63",
 "rederived_value":"All twelve numbers reproduce exactly, on the same five datasets {instructed-deception, mask, convincing-game, ai_liar, roleplaying}, k=3, mean pooling, seed 42, paper probe, 20 off-diagonal pairs. 3B L16: raw 0.7619, axes 0.7815 (+0.0196), datapca 0.5791 (-0.1828), axes-datapca +0.2024. 8B L16: raw 0.7980, pc:deception 0.8105 (+0.0125), datapca 0.6958 (-0.1022), diff +0.1147. 70B L40: raw 0.8090, axes 0.8220 (+0.0130), datapca 0.7426 (-0.0664), diff +0.0794.",
 "evidence_path":"audit/cache/labpc/run-crossgen/scale_3b/matrices/mean_L16_s42.json; audit/cache/labpc/run-crossgen/scale_8b/matrices/mean_L16_s42.json; lab C:/Users/user4/expLB/run-crossgen/scale_70b/matrices/mean_L40_s42.json",
 "evidence_key":"in each file, results[] entries with probe='paper' and feature in {raw, axes, pc:deception, datapca}; off-diagonal mean of the 5x5 auroc matrix (20 cells); metadata k=3, pooling=mean, seed=42; banks axes_3b.pt / axes_8b.pt / axes_vectors_s3.pt with bank_name paper16_3b / paper16_8b / paper16_70b",
 "my_verdict":"HOLDS",
 "my_confidence":"HIGH - twelve independent numbers, all matching to the stated 3-4 decimals",
 "notes":"3B/8B read from audit/cache (MANIFEST records md5 MATCH against the lab); the 70B file pulled fresh from the lab this session - none of scale_3b/scale_8b/scale_70b exists in the git working tree. One scope caveat the entry should carry: 'best persona feature' means best among the k=3 features. pc:unified@2 (k=2) beats the named winner at two rungs - 3B 0.7949 (+0.033 over raw) and 70B 0.8352 (+0.026) - so 'the gain is small and FLAT, +0.020/+0.013/+0.013' is a matched-k statement. Matching k is defensible (the paper's learned-probe protocol is 'the first 3 persona PCs', section 4), it just needs saying."
},
{
 "id":"F-64",
 "rederived_value":"Everything reproduces. Chain START 22:59:00 -> DONE 01:31:21. 2a measured 12 files / 1,200 rows (predicted 12/1,200, PASS); 2b 12 control files at 20 rows each = 240; 2c 12 score files; 2d 12 vectors for both s23 and all, each logged 'predicted=13 PASS'; 2e GATE FAILED on both arms. Judge keep rate (scores 2 or 3): capitulation 563/600 = 93.833%, position_holding 559/600 = 93.167%, gap 0.667 pp. Score distribution over all 1,200: 0 -> 78 (6.50%), 1 -> 0 (0.00%), 2 -> 714 (59.50%), 3 -> 408 (34.00%). Worst-retained roles: pc_deferrer 19% dropped, pc_calibrated_holder 17%.",
 "evidence_path":"lab C:/Users/user4/expLB/logs/cap_chain.status; lab C:/Users/user4/expLB/run-cap/scores/*.json (12 files); lab C:/Users/user4/expLB/run-cap/{responses,control}; git cap/personas/cap_role_groups.json",
 "evidence_key":"status block 'START 2026-09-20 22:59:00' through 'DONE 2026-09-21 01:31:21', lines '2a  measured=12 files / 1200 rows  predicted=12 / 1200  PASS' and '2d/s23  measured=12 vectors  predicted=13  PASS'; score JSONs counted by value; cap_role_groups.json groups.capitulation.roles (6) and groups.position_holding.roles (6)",
 "my_verdict":"HOLDS",
 "my_confidence":"HIGH - the score distribution matches to the single rollout (78/0/714/408) and the keep rates to the row",
 "notes":"Read the LAB copies; run-cap/ in the git working tree holds only export/. Two cosmetic roundings: the keep-rate gap is 0.67 pp (93.833 - 93.167), stated as 0.66 pp, and it is 0.67 rows per role (4 rows over 6 roles), stated as 0.6. The directories now hold 13 files each because default.jsonl was generated later at 01:56 and the chain was re-run twice; the 12s in the finding are correct for the 22:59 run, and the status file's later blocks (02:01 and 02:18) show 13."
},
{
 "id":"F-65",
 "rederived_value":"Every AUROC reproduces. Zero-shot on sycophancy_mt (cap_s23 bank, mean pooling, seed 42, 100% of rows): contrast 0.7592 / 0.7449 / 0.7281 at L20/L40/L44; PC1 0.7399 / 0.7355 / 0.7166. Trained with sycophancy_mt as TARGET (column off-diagonal mean over the 10 training datasets, paper probe, L40, s23): pc:capitulation@2 0.7136, pc:capitulation 0.6575, pc:cap_only@2 0.5935, raw 0.6013 (so +0.1123), length 0.5820, datapca 0.5776, datapca@2 0.5594, random 0.5413, random@2 0.4974; from the paper16 ROOT store at the same cell pc:unified 0.4451, pc:deception 0.3910, axes 0.3551. s23 vs all at L40: trained 0.7136/0.7132, zero-shot 0.7449/0.7469. Layer profile @2: 0.6438 / 0.7136 / 0.6904. B vs C: 0.7136 -> 0.5935 at L40 (-0.120) and 0.6438 -> 0.5869 at L20. The '0.374' is 0.3740 = the PC1 zero-shot row of the ROOT kd11 store, which the code pins to the DECEPTION axis.",
 "evidence_path":"lab C:/Users/user4/expLB/run-crossgen/kd11/banks/cap_s23/matrices/mean_L{20,40,44}_s42.json and banks/cap_all/matrices/mean_L{20,40,44}_s42.json; audit/cache/labpc/run-crossgen/kd11/matrices/mean_L40_s42.json (= git copy, md5 d7616aa5...); lab C:/Users/user4/expLB/run-crossgen/kd11/passA11/matrices/mean_L40_s42.json; git src/assistant_axis_ndif/27_crossgen.py",
 "evidence_key":"zeroshot[] entries direction='contrast' / 'PC1', auroc index 10 (sycophancy_mt) in each cap file; results[] probe='paper', column index 10, mean of the 10 off-diagonal rows; ROOT file zeroshot PC1 auroc[10]=0.3740 and contrast auroc[10]=0.4089; 27_crossgen.py lines 405-427 select the zero-shot axis, preferring 'deception'; passA11 zeroshot PC1:sycophancy auroc[10]=0.2996 and contrast:sycophancy=0.3061",
 "my_verdict":"PARTLY",
 "my_confidence":"HIGH - roughly twenty AUROCs all match to the quoted 3 decimals; the two defects are attribution errors provable from the crossgen source",
 "notes":"Defect 1: the entry calls 0.374 'the paper16 SYCOPHANCY axis'. 27_crossgen.py pins the bare 'contrast'/'PC1' zero-shot rows to the DECEPTION axis whenever the bank has one, so 0.3740 is the deception axis's PC1. The only paper16 store that scores every axis separately (passA11, written after that code change on 2026-09-21) gives the sycophancy axis PC1 0.2996 and contrast 0.3061 on the same column - lower still, so the direction of the claim survives but the label on the number is wrong. Defect 2: 'the register axis is worse than a random direction (0.541)' compares a ZERO-SHOT projection against a TRAINED random probe (feature 'random', k=5, column mean 0.5413); those are different controls. Sub-claims I did NOT check: the cosines (0.993, 0.931, 0.987, -0.838, and the re-run 0.987/0.949/0.936), PC1 variance 56.8%, and the permutation p - all need run-cap/axes_*.pt (63 MB each, lab-only)."
},
]

with open('audit/phase4/rederive_F2.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=['id','rederived_value','evidence_path','evidence_key','my_verdict','my_confidence','notes'])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print('wrote', len(rows), 'rows')
