"""P17 Agent C -- write audit/p17/fmap_F110_F127.csv, ledger_new_inventory.csv, ledger_new_claims.csv.

Per-entry mapping metadata is declared below (from reading kd/docs/FINDINGS.md WT md5 dc365834, lines
5238-6717); value_seen_prelim is filled from audit/p17/ledger_verify_results.json, which
audit/scripts/p17_ledger_verify.py writes by re-deriving each number from raw.
  python audit/scripts/p17_write_fmap.py
"""
import collections, csv, json, os, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
RES = json.load(open(ROOT / "audit/p17/ledger_verify_results.json", encoding="utf-8"))
LINES = {"F-110": 5238, "F-111": 5289, "F-112": 5419, "F-113": 5517, "F-114": 5642, "F-115": 5793, "F-116": 5834,
         "F-117": 5879, "F-118": 5920, "F-119": 5951, "F-120": 5966, "F-121": 6010, "F-122": 6064, "F-123": 6145,
         "F-124": 6273, "F-125": 6400, "F-126": 6535, "F-127": 6672}
LAB = "C:/Users/user4/expLB"
RSP = "C:/Users/user4/expLB-resplit"
PAP = "origin/paper @ 16def2de (PIN_PAPER); entry read at ea1ce52b (PIN_OLD)"
CB = "src/probes/deception-detection/results"

# f_id: (title_short, date, claim_type, key_numbers, cited_paths, resolved_primary, key_path, other_copies, copy_status,
#        producing_script, script_writes_file, run_log, depends_on, supersedes, superseded_by, draft_topic, mapping_status,
#        verdict, notes)
M = {
 "F-110": ("70B scale rung ran on the kd bank; like-for-like paper16 70B is -0.013 vs raw, +0.054 vs dPCA", "2026-09-23", "correction",
    "3B/8B/70Bkd/70Bp16 best-dPCA=+0.1197/+0.1027/+0.0751/+0.0541; best-raw 70Bp16=-0.0131; last 70Bp16 +0.0061/+0.0688 (19L)",
    "run-crossgen/scale_70b_paper16fp32/matrices/{mean,last}_L*_s42.json; audit/scripts/fix_rf16_recompute.py; audit/fix/rf16_recompute_mean.txt",
    f"{LAB}/run-crossgen/scale_70b_paper16fp32/matrices/mean_L*_s42.json (+ scale_3b, scale_8b, scale_70b)",
    "results[feature in {axes,pc:deception,raw,datapca}, probe=paper].auroc; off-diag over the 5 paper datasets; best=max(axes,pc:deception); 19 layers = 21 minus L72,L76",
    "scale_3b/scale_8b also in audit/cache/labpc/run-crossgen (md5-verified); none in git", "LAB-ONLY",
    "src/assistant_axis_ndif/27_crossgen.py (audit re-run into new folder); recompute audit/scripts/fix_rf16_recompute.py", "y",
    "not identified by P17", "run-coords/passB; run-paper16/axes_vectors_s3_fp32.pt; k=3; seed 42",
    "F-100 70B column; amends F-101, F-102", "-", "4", "MAPPED", "HOLDS",
    "P17 found the unnamed 19-layer set: F-100's 19 layers = the 21 stored minus L72 and L76 (only pair that reproduces 0.7524/+0.0079/+0.0751). Already has L-F-110 / claims F-110 (HOLDS); no new claim row."),
 "F-111": ("Audit corrections: eleven earlier entries restated (37/80->31/80, cap teacher forcing, F-04 block 9, 0.374 is deception PC1 ...)", "2026-09-23", "correction",
    "kd vs raw 28/80, vs datapca 31/80; cap 1300 rows/13 files/65 personas/20 q; F-04 +0.0147/+0.0264 -> -0.0215/-0.0243; random@2-raw -0.2287 q0.0192; length 9/11, 0.5555; syc_mt contrast 0.4089 PC1 0.3739, syc axis 0.313",
    "run-crossgen/kd11/passA11/banks/kd_s23/matrices/mean_L{0..79}_s42.json; run-cap/responses; run-crossgen/kd11/significance.json; run-crossgen/kd11/matrices/mean_L40_s42.json; 24_q10_sycophancy_axis.py; audit/DOC_FIXES.md DF-01..46",
    f"{LAB}/run-crossgen/kd11/passA11/{{banks/kd_s23/,}}matrices/mean_L*_s42.json; {LAB}/run-cap/responses/*.jsonl; run-crossgen/kd11/significance.json (git); run-crossgen/kd11/matrices/mean_L40_s42.json (git); run-coords/passB/*/rows.jsonl (lab); run-coords/q10_sycophancy_axis.json, zeroshot_nulls.json (git)",
    "row kr-matched off-diag (10 targets); tests[block,pooling,layer,probe,comparison]; zeroshot[contrast|PC1][sycophancy_mt]; span_len by label",
    "kd11/significance.json git md5 1194ef76 != lab 44d5b53d (RF-11; the entry cites the git 4,032-test family); kd11 mean_L40 LAB=GIT (md5 d7616aa5); passA11/passC11 and run-cap lab-only", "LAB≠GIT",
    "27_crossgen.py; audit/scripts/qa_p7.py; gen_multiturn_ndif.py; 24_q10_sycophancy_axis.py", "y", "-",
    "kd11 root store for the matched control (not the bank store)", "F-77, F-81, F-32, F-04, F-73, F-26, F-33, F-65 (0.374 attribution), F-105, F-36, F-38", "F-114 item 2 (the '9 of 10 matches no slicing' clause); F-112 closes its 'still open'", "3,6", "PARTIAL", "PARTLY HOLDS",
    "Items 1,2,3,4,7,8 and F-73's numbers re-derived (41 checks). Two do not hold as written: (a) the 'trap' - no bank store carries its own datapca: passA11/banks/kd_s23 matrices have NO datapca column and its summary.json datapca EQUALS the root store's, so axes wins 31/80 against it, not 0/80; (b) '0.3061-0.2996 on passA11/passC11': 0.3061/0.2996 are passA11's contrast/PC1 sycophancy readings; passC11 reads 0.3066/0.2995. The 0.3739 is 0.373959 (truncated). Items 5,9,10,11 (F-29 18/21, F-33 24/24, F-105, F-26/F-36/F-38) not re-derived in P17."),
 "F-112": ("F-31's families are nested (git 4,032 subset of lab 6,300); pc:kd_assertion beats raw 18/21 but not the matched control", "2026-09-23", "correction",
    "git 4032/16 vs lab 6300/25 tests; 0 lost, 0 changed, 2268 added; kd_assertion>random 21/21 (8 clear), >raw 18/21 (1 clears, L68 +0.1497 q0.0467), >datapca 20/21 (0 of 126); margins L32..76 +0.0998..+0.1497 (+0.1183..+0.1897 HP-KR dropped); kr-matched 258 {124/134}; HP-KR 2139 {139/2000}",
    "run-crossgen/kd11/banks/kd/significance.json (git + lab); run-crossgen/kd11/banks/kd/matrices/; run-crossgen/kd11/matrices/; run-coords/passB/harm-pressure-kr-matched/rows.jsonl; audit/claims/F-20.md",
    f"{LAB}/run-crossgen/kd11/banks/kd/significance.json (lab, md5 af3b9feb, 2026-09-22 16:01:20) vs git copy (md5 9809a00f)",
    "tests keyed (block,pooling,layer,probe,comparison): diff, dataset_p, dataset_q_bh, significant_datasets; off-diag of pc:kd_assertion (bank) minus raw (root)",
    "git copy = 4,032-test subset (LAB≠GIT by design: lab is a strict superset, identical diff/dataset_p on all 4,032)", "LAB≠GIT",
    "27_crossgen.py (lab re-run 2026-09-22 16:01)", "y", "-", "kd11/banks/kd store + kd11 root raw", "F-31 'nothing beats raw' and '6 of 4,032'; F-20's implied store property", "-", "3,5", "MAPPED", "PARTLY HOLDS",
    "All 33 numbers re-derived. One scope slip: the table column 'tests clearing BH over 6,300' prints 8 for kd_assertion-random, which is the count within block 10/mean/paper (the row's own scope); across all 6,300 it is 34. '0 of 126' is block 10 only (252 over both blocks, still 0). The parenthetical 'the bank store carries its own datapca' does not hold for kd11/banks/kd: its summary.json datapca equals the root store's at L40/L68 (0.66508/0.64966)."),
 "F-113": ("Methods-audit corrections: MiB unit slip, F-51 length, F-107 robust, F-108 81% not 100%, default k=5", "2026-09-23", "correction",
    "512.00/384.00 MiB (=537/402 MB); F-43 52 OOMs, p99 12; F-51 39.2 vs 26.5 words; F-107 0.6930/0.7097; kr-matched 42 of 52 (81%); k default 5 at 27_crossgen.py:882; pc:unified@2 +0.138 vs +0.004; cos -0.945 / +0.984 / +0.389 / +0.896",
    "scale/scripts/run_scale_coords.ps1:108-109; run-coords/sarcasm/sarcasm_natural/chunk_*.pt; 30g_clustered_holdout.py:12-14; 27_crossgen.py:882; audit/methods/*; audit/scripts/qa_m1_f107.py",
    "arithmetic; src/assistant_axis_ndif/27_crossgen.py:882; git asix scale/scripts/run_scale_coords.ps1; lab run-coords/passB rows.jsonl (42/52 replay); lab scale_* matrices (k)",
    "heads x T^2 x 4; --k default; replay of 30g default_rng(42) split; matrix field k",
    "run_scale_coords.ps1 in git; passB rows.jsonl lab-only", "LAB-ONLY",
    "30g_clustered_holdout.py; 27_crossgen.py; audit/scripts/qa_m1_f107.py", "n", "8B OOM log (F-42/43) not re-read", "-",
    "F-85 'never been made'; F-42/F-43 MiB, 50 OOMs, p99; F-51 'elaborates'; F-107 count/range; F-108 'every one of its 258 rows'", "-", "1,3", "PARTIAL", "HOLDS",
    "Re-derived: MiB arithmetic, the MiB slip text at run_scale_coords.ps1:108-109, k=5 at :882, k=3 in all four scale stores, 42/52 under 30g's own split (the crossgen split gives 39/52 - a different pipeline). NOT re-derived in P17: F-43 52 OOMs/p99, F-51 word counts, F-107 null table, cosines -0.945/+0.984/+0.389/+0.896, pc:unified@2 +0.138. 402 MiB = 402.65 MB (truncated, not rounded)."),
 "F-114": ("Wave-5 corrections: 9-of-10 length AUROC was right, F-61 8.1x not 12x, F-56 sampler balances, F-91 mixes seeds", "2026-09-23", "correction",
    "length_only_auroc 10 datasets, 9 >0.5, roleplaying 0.4415; raw 0.6623, pc:unified 0.6715, datapca 0.6651, +0.0093/+0.0064, kd +0.0757, 11.8 vs 8.1x; F-91 L32/36/40 x s7/13/42; 134/2134=6.28%",
    "run-coords/confounds_L44_last.json; run-crossgen/kd11/matrices/; run-premise/falsify/falsify_L*_s*.json; 8_coords_ndif.py:504; audit/phase2/qa_wave5.md",
    f"audit/cache/labpc/run-coords/confounds_L44_last.json (== lab md5 11bc6e65); run-crossgen/kd11/matrices/mean_L40_s42.json; {LAB}/run-premise/falsify/falsify_L{{32,36,40}}_s{{7,13,42}}.json",
    "confounds.length_only_auroc[*].auroc_from_length; off-diag 110 pairs; falsify real",
    "confounds json LAB-ONLY (cached); kd11 matrices LAB=GIT; falsify LAB-ONLY", "LAB-ONLY",
    "27_crossgen.py; 30_premise falsify run", "y", "-", "kd11 root + kd bank store", "F-61 ratio, F-56 mechanism, F-91 second method, F-78 panel, F-49, F-57 CI, F-21 table, F-70/F-104 caveats, F-111 item 7 clause", "F-123 (restates DF-71 as 8.2x)", "7", "MAPPED", "HOLDS",
    "23 checks match. Ratio rounding: 8.1x uses the 4-dp inputs (0.0757/0.0093=8.14); unrounded 8.18 -> F-123 prints 8.2; mixed ratio 11.83 (rounded inputs) vs 11.75 unrounded (F-123 prints 11.7). F-78 panel, F-49 359,384 B, F-57 CI, F-21, F-52 not re-derived."),
 "F-115": ("HP-KR/kr-matched leak closed: F-03 survives (-0.2545/+0.0309/+0.0050) but KR->matched at L40 falls to 0.4262", "2026-09-23", "scientific",
    "leaked/guarded axes-raw L20/40/44 -0.2566/-0.2545, +0.0367/+0.0309, +0.0134/+0.0050; KR->KM L40 0.5674->0.4262; raw 0.7932->0.7214; 59 of 258 survive (77%)",
    "audit/cache/labpc/expLB-resplit/mean_L{20,40,44}_s42.json; audit/scripts/p6_patch_leakguard.py; p6_resplit_compare.py",
    f"{RSP}/run-crossgen/matrices/mean_L{{20,40,44}}_s42.json (lab, md5 be979ba5/d2384c88/ddd573b8, 2026-09-23 23:06)",
    "row harm-pressure-kr-matched off-diag mean (axes-raw); cells KR<->KM; survivors = kr-matched ids in HP-KR test_row_ids (splits.json seed 42)",
    "audit/cache/labpc/expLB-resplit/ SAME (md5 equal); originals run-crossgen/kd11/matrices LAB=GIT", "LAB-ONLY",
    "COPY of 27_crossgen.py patched by audit/scripts/p6_patch_leakguard.py (lab expLB-resplit)", "y", f"{RSP}/logs (not re-read)",
    "run-coords/passB activations read-only; paper16 fp32 bank; k=5", "-", "-", "3", "MAPPED", "HOLDS", "24 checks, all exact."),
 "F-116": ("knowledge_matched HP-KR slice (2,001 rows): raw 0.999 in-dist, persona axes transfer below chance", "2026-09-23", "scientific",
    "2001 rows (139/1862), 7.8x, 93.5%; transfer axes/raw/axes-raw L20 0.3850/0.5711/-0.1860, L40 0.3335/0.7093/-0.3758, L44 0.5267/0.7203/-0.1937; length 0.6174; in-dist raw 0.9995/0.9992/0.9993",
    "audit/cache/labpc/expLB-resplit/km/mean_L{20,40,44}_s42.json; audit/scripts/p6_build_km_view.py; lab build_km_view.py / run_km.ps1",
    f"{RSP}/run-crossgen/km/matrices/mean_L{{20,40,44}}_s42.json (md5 15c017e1/86070e76/8a39fc33)",
    "row harm-pressure-kr-knowledge-matched off-diag (11 targets); diagonal cell; zeroshot contrast",
    "audit/cache copies SAME (md5 equal)", "LAB-ONLY", f"27_crossgen.py copy + {RSP}/build_km_view.py", "y", "-",
    f"{RSP}/passB/harm-pressure-kr-knowledge-matched/rows.jsonl (2,001 rows)", "-", "-", "3", "MAPPED", "HOLDS", "40 checks, all exact."),
 "F-117": ("Collaborator's 16-layer sweep: excl HP-KR the Contributions margin is small and crosses zero; vs his baseline positive 16/16", "2026-09-24", "scientific",
    "ext_axis1-ext_pc3 with/excl HP-KR (8 layers); positive 13/16, excl 11/16; within +-0.06, crosses zero 5x; vs baseline 16/16 +0.115..+0.273 (excl +0.123..+0.243); ext_axis1 L36 0.9275 vs L40 0.9158; L76 rise",
    f"git show origin/paper:{CB}/persona_ood_ext_L*/summary.csv, zero_shot.csv; audit/scripts/p8_collab_layer_sweep.py; audit/COLLAB_PAPER_BRANCH.md",
    f"{CB}/persona_ood_ext_L{{20..79}}/summary.csv, zero_shot.csv @ {PAP}",
    "summary.csv features x {off_diagonal_mean, off_diagonal_excl_hpkr}; zero_shot.csv row harm_refusal_knowledge_report__plain",
    "ext_axis1/ext_pc3 identical at PIN_OLD and PIN_PAPER; baseline rows at L20/40/44 moved in the 4th-5th dp between the SHAs", "GIT-ONLY",
    "collaborator's pipeline (not re-run)", "y", "-", "his persona_ood_ext bank (36 personas)", "draft's 'outscores ... at every depth'", "F-124/F-125 cite its L36 optimum", "2,5", "MAPPED", "PARTLY HOLDS",
    "43 checks: all table values exact. Two sentence-level counts fail: excl-HP-KR margins are NOT all within +-0.06 (L56 = +0.0633), and the excl series crosses zero SIX times (24->28, 28->32, 44->48, 48->52, 60->64, 72->76), not five. The validation cosines (>=0.99989, 0.9996) were not re-derived (no artifact located)."),
 "F-118": ("Gender-Secret: every persona direction at chance where both classes deny; three wording-matched tests converge", "2026-09-24", "scientific",
    "GS 4v6 0.549/0.436/0.498, 12v13 0.505/0.467/0.541, 8v7 0.514/0.545/0.623, 5v5 0.593/0.523/0.520; 82% vs 55%; convergence item 1 0.4630/0.6203/0.5383",
    f"git show origin/paper:{CB}/gender_secret/summary.md",
    f"{CB}/gender_secret/summary.md @ {PAP}; item 1: audit/cache/labpc/run-crossgen/kd11/banks/collab/matrices/mean_L{{20,40,44}}_s42.json",
    "summary.md results table col 'adapted L20 / L40 / L44'; collab matrices zeroshot contrast:collab_extended[harm-pressure-kr-matched]",
    "summary.md prose only (no activations committed); collab-bank matrices lab-only, cached", "GIT-ONLY",
    "collaborator gender_secret.py (not re-run)", "y", "-", "commit 572095cb (ancestor of PIN_PAPER)", "-", "-", "3", "PARTIAL", "PARTLY HOLDS",
    "All GS numbers match his committed summary.md (not re-derived: no activations). Convergence item 1 cites 'F-115 §5' - F-115 has no §5 and contains no 0.4630; the numbers are audit/HPKR_WORDING_CONTROLLED.md §5 and reproduce exactly from the collab-bank matrices. Pointer error, numbers right."),
 "F-119": ("Correction in collaborator's favour: paper/make_numbers.py exists on origin/paper", "2026-09-24", "correction",
    "make_numbers.py + make_figures.py present; persona_depth cites ../persona_unified/ and ../persona_depth_orig/, neither on branch",
    "src/probes/deception-detection/paper/make_numbers.py; persona_depth/summary.md",
    f"src/probes/deception-detection/paper/make_numbers.py @ {PAP}", "git cat-file / ls-tree",
    "present on paper only (0 on asix, 0 on feat/persona-pc-ood)", "GIT-ONLY", "-", "-", "-", "-",
    "D-237 verdict, 'DF-71' reproducibility half (mis-pointer; F-123 fixes it to DF-12), REPORT_TECHNICAL §8 item 1", "F-123 §1 (pointer)", "8", "MAPPED", "HOLDS",
    "4 checks exact. Its 'DF-71' pointer is wrong (should be DF-12), as F-123 §1 records."),
 "F-120": ("Late-layer rise is the full-rank probe degrading plus anisotropy: random direction gains +0.127 L72->L79", "2026-09-24", "scientific",
    "raw 0.8195/0.7828/0.7834/0.7426; axes 0.8266/0.6211/0.6384/0.7492; random 0.5971/0.5234/0.5133/0.6405 (L40/70/72/79); axes-random +0.2296 vs +0.1087; axes mid L30-50 0.8145",
    "audit/phase2/layer_sensitivity.csv (store passA); audit/scripts/p9_late_layer.py",
    f"{LAB}/run-crossgen/passA/matrices/mean_L{{30..50,70,72,79}}_s42.json",
    "off-diag over the 5 paper datasets (instructed-deception, mask, convincing-game, ai_liar, roleplaying), probe paper - the 'paper5' rows p9 read (it keeps the LAST row per feature/layer, which is paper5)",
    "audit/phase2/layer_sensitivity.csv is a derived table of the same matrices; not in git", "LAB-ONLY",
    "audit/scripts/p2_w1_layer_sensitivity.py -> p9_late_layer.py", "y", "-", "passA 80-layer store, paper16 fp32 bank, k=5", "-", "-", "2", "MAPPED", "PARTLY HOLDS",
    "All 17 numbers exact from the raw matrices. Unstated scope: they are the 5-dataset (paper5) reading, not the store's all-10 reading (raw L40 all-10 = 0.6760 vs 0.8195). The entry says only 'passA, mean, paper, seed 42'. The conclusion was not re-tested on all 10 datasets."),
 "F-121": ("kd bank run at 3B/8B: both banks decline with scale; kd declines less", "2026-09-24", "scientific",
    "paper16 +0.0382/+0.0148/-0.0116; kd all +0.0317/+0.0169/+0.0032; kd s23 +0.0360/+0.0194/+0.0054; kd-p16 -0.0066/-0.0022, +0.0022/+0.0046, +0.0148/+0.0170; axis md5s byte-distinct",
    f"{RSP}/run-crossgen/kdscale_{{3b,8b,70b}}_{{all,s23}}/matrices/mean_L*_s42.json; run-crossgen/scale_{{3b,8b,70b_paper16fp32}}; audit/scripts/p12_kd_scale_ladder.py; tamara:expLB/_labpc_only_backup_2026-09-24/run-scale-kd",
    f"{RSP}/run-crossgen/kdscale_*/matrices/mean_L*_s42.json + {LAB}/run-crossgen/scale_*/matrices/mean_L*_s42.json",
    "layer-average of [max over (axes, pc:*) off-diag] minus raw off-diag, 5 datasets, probe paper",
    "none in git; Drive backup (290 files) not checked by P17", "LAB-ONLY",
    f"{RSP}/run_kd_scale.ps1, run_kdscale_crossgen.ps1; audit/scripts/p12_kd_scale_ladder.py", "y", f"{RSP}/logs (not re-read)",
    f"{RSP}/run-scale-kd/{{3b,8b}}/axes_kd_*_vectors_{{all,s23}}.pt", "-", "-", "4,5", "MAPPED", "HOLDS",
    "17 checks exact. The entry does not define 'best'; p12 takes the max over axes and every pc:* feature, whereas F-110 uses max(axes, pc:deception) - hence F-121's 70B paper16 -0.0116 vs F-110's -0.0131 (19 L) / -0.0171 (21 L). Two recipes, both reproducible."),
 "F-122": ("Sycophancy axis was computed on the collaborator's branch; syc_pc1 tops excl-HP-KR L40-56, driven by convince_me; dim=1 arms are zero-shot", "2026-09-24", "correction",
    "best-arm table L36-56 (baseline 0.8124/0.7581/0.7297/0.7375/0.7515/0.7419); win counts syc_pc1 5, ext_pc3 4, ext_axis1 4...; L40 per-dataset syc_pc1 vs ext_axis1; convince_me +0.086..+0.054; spreads 0.000",
    f"git show origin/paper:{CB}/persona_ood_ext_L*/summary.csv, matrix_*.csv; audit/scripts/p14_collab_syc_arms.py",
    f"{CB}/persona_ood_ext_L*/summary.csv + matrix_{{syc_pc1,ext_axis1,denial_axis1}}.csv @ {PAP}",
    "summary off_diagonal_excl_hpkr; matrix cells, HP-KR training row dropped",
    "baseline off_diagonal_excl_hpkr at L40/L44 moved 1e-4 between ea1ce52b and 16def2de", "GIT-ONLY",
    "collaborator pipeline", "y", "-", "his 16-layer sweep", "the audit's 'pc:sycophancy never computed anywhere' (RESULTS_BY_AXIS §4, DF-89)", "F-126 (supplies the pooling/significance caveat)", "6", "MAPPED", "HOLDS",
    "40 checks: all match at ea1ce52b, the SHA the entry read; at the pin two baseline cells read 0.7580/0.7296 (the collaborator re-ran his baseline, 4th dp)."),
 "F-123": ("Consolidated DOC_FIXES ledger corrections (22), plus F-119's pointer error (DF-71 -> DF-12)", "2026-09-24", "correction",
    "make_numbers.py 28,232 B, 8fd071b2, 2026-09-22, zami456; count 1/0/0; pc:unified-raw +0.0093, 8.2x, mixed 11.7x; length/raw 0.5555/0.6623, 0.5494/0.6652 (no kr-matched), block9 0.5467/0.6870; rows caption 22 chars; DF-78 -0.2545/+0.0309/+0.0050",
    "run-crossgen/kd11/matrices/mean_L40_s42.json; audit/scripts/p16_f123_verify.py; audit/DOC_FIXES.md; audit/phase6/DOC_FIXES_TRIAGE.md",
    f"run-crossgen/kd11/matrices/mean_L40_s42.json (git asix == lab md5 d7616aa5); src/probes/deception-detection/paper/make_numbers.py @ {PAP}",
    "off-diag 110 pairs; d['rows']; git cat-file -s", "kd11 matrix LAB=GIT; make_numbers.py GIT-ONLY", "LAB=GIT",
    "audit/scripts/p16_f123_verify.py", "n (prints)", "-", "kd11 root store", "none (corrects F-04, F-26, F-29, F-32, F-33, F-36, F-38, F-48, F-54, F-56, F-61, F-65, F-73, F-77, F-78, F-81, F-91, F-105; repairs F-119 pointer)", "-", "7,8", "MAPPED", "HOLDS",
    "§1-§2 re-derived exactly; 28,232 B is make_numbers.py at ea1ce52b - at the pin it is 39,532 B (the file grew on 2ad758fb/d1b88154/16def2de; the branch moved after the entry). §3 carried items: F-04, F-73, F-65, DF-10 token means, F-77/F-81 28/80-31/80, F-32 cap counts re-derived under F-111; the rest carried."),
 "F-124": ("Draft tables reproduce; sign inversion is 7 (arm,layer) cells; 0.651 vs 0.791; BA@1%FPR up to 88% FPR", "2026-09-24", "scientific",
    "7 inverted cells (L20 axis1/syc_axis1/syc_pc1/pc1, L40 syc_axis1/syc_pc1, L44 syc_axis1) printed->corrected; Table 11 +0.128/+0.145/+0.141; axis1 L20 HP-KR row 0.0806; 23 of 90 cells FPR>1%; 4v6 L20 HP-KR 88.3% (spec 0.1165, recall 0.9496, BA 0.533); base 0.109/0.936/0.828 vs 0.3808/0.9281/0.8803",
    f"origin/paper @ ea1ce52b {CB}/persona_ood_ext_L{{20,40,44}}/ zero_shot.csv, summary.csv, zero_shot_operating.csv, matrix_*.csv; audit/scripts/p16_level2.py; p16_level2_verdicts.py",
    f"{CB}/persona_ood_ext_L{{20,40,44}}/matrix_<arm>.csv, zero_shot.csv, zero_shot_operating.csv @ ea1ce52b (PIN_OLD); recounted at 16def2de",
    "matrix rows with off-diag mean <0.5; HP-KR row -> 1-x; zero_shot_operating rows threshold_source=control, 5 arms x 6 datasets x 3 layers",
    "7 cells and 23/90 identical at PIN_PAPER (which adds a 6th arm, unified, excluded)", "GIT-ONLY",
    "collaborator pipeline", "y", "-", "his persona_ood_ext bank", "DF-46's scope claim", "F-125 (§2, §3 superseded; §4 softened)", "5", "MAPPED", "HOLDS",
    "54 checks, all exact. §2/§3 framing superseded by F-125; the measurements stand."),
 "F-125": ("Corrects F-124 (draft discloses both points); Level 1 clears all 176 draft claims (160/9/7)", "2026-09-24", "correction",
    "Level 2 52/2 (from 40/14); Level 1 176 = 160/9/7; D-156 0.909 vs 0.919; D-128 L36 0.9275 vs 0.9158; D-153 +0.066 CI [-0.069,+0.189] P 0.161",
    "audit/scripts/p16_level1.py, p16_level1_verdicts.py; draft PDF sections 3.4/4; Figure 3/4 captions",
    "audit/claims.csv (D- rows, batches B01-B63; laptop audit file, mtime 2026-09-24 16:40); his summary.csv @ ea1ce52b/16def2de",
    "claims.csv verdict by id/batch; summary.csv off_diagonal_excl_hpkr L20",
    "claims.csv is an audit working file (never committed); his CSVs git-only", "GIT-ONLY",
    "audit/scripts/p16_level1_verdicts.py", "y", "-", "-", "F-124 §2, §3 (softens §4)", "F-126 (7 unresolved -> 2), F-127 (release clause)", "8,3", "PARTIAL", "HOLDS",
    "Current claims.csv reads 176 = 166/8/2 (F-127's state) and Level-2 52/2 exactly; the 160/9/7 snapshot is not re-derivable (claims.csv rewritten in place) but the id-level chain 160/9/7 -> 164/10/2 -> 166/8/2 checks. D-153's +0.066/P 0.161 rests on audit prose (HPKR_WORDING_CONTROLLED §4), not re-run. Draft-PDF quotations not checked by P17."),
 "F-126": ("6D-1: pc:sycophancy in our pipeline loses to raw on all 11 datasets; 6D-2: passP is a prompt-span store; Figure 2 geometry rebuilt", "2026-09-24", "scientific",
    "reproduces kd11 0.6623/0.6651/0.5555/0.6715; 48-cell pc:syc-minus-control table; 0 of 264 significant; passP raw off-diag 0.5584/0.5687/0.5697, diag L44 CG 1.000...mask 0.624; passB_fine 36/40 negative; hf275_strict pc:deception +0.039; cos 0.474559, -0.0844, +2.9037, +0.0458, -1.2274",
    f"{RSP}/phase6/run-crossgen/; run-crossgen/passP, passB_fine; run-kd/vectors_s3; audit/scripts/p16_syc_result.py, p16_passBfine_passP.py, p16_fig2_axes.py; audit/LABPC_WRITES.md",
    f"{RSP}/phase6/run-crossgen/summary.json (md5 2cde5f4b, 2026-09-24 16:36) + significance.json; {LAB}/run-crossgen/passP/summary.json; {LAB}/run-crossgen/passB_fine/{{,banks/*/}}summary.json; {LAB}/run-kd/vectors_s3/*.pt",
    "records[pooling,layer,feature,probe=paper,seed=42].offdiag_mean_10/_9, diagonal; chunk .pt span/n_tok; vectors[L=40] projections",
    "all lab-only; nothing in git", "LAB-ONLY",
    "27_crossgen.py (6D-1 run into expLB-resplit/phase6); collaborator's make_figures.py:personas() recipe", "y", "-",
    "run-coords/passB; paper16 fp32 bank; run-kd/vectors_s3 (n=106 pc_handler)", "DF-89 'the real gap'; resolves F-125 §3 Cause A", "F-127 (caveat 4)", "6,8", "MAPPED", "HOLDS",
    "48 checks exact, including all 48 cells of the 6D-1 table and the Figure-2 geometry from the raw vectors. 'an axis whose full span is 4.13' is the span of GROUP MEANS (deniers minus plain assertion); the 36 personas span 5.29. D-041 arXiv id not read (no PDF text tool); title match in papers/."),
 "F-127": ("Corrects F-126 caveat (4): paper/generated/numbers.tex holds the Figure 2 numbers; 6A = 166/8/2", "2026-09-24", "correction",
    "numbers.tex 1,162 lines of \\setV; cos-dec-den-L40 0.47, proj-den-kd 2.90, sincere -0.08, deceptive 0.05; eight tab_*.tex, five fig_*.pdf; 166/8/2",
    "src/probes/deception-detection/paper/generated/numbers.tex; tab_*.tex; fig_*.pdf",
    f"src/probes/deception-detection/paper/generated/numbers.tex @ origin/paper (counts match at 2ad758fb; pin 16def2de)",
    "\\setV{key}{value} macros; git ls-tree paper/generated/", "moves across ea1ce52b/2ad758fb/d1b88154/16def2de", "GIT-ONLY",
    "his paper/make_numbers.py", "y", "-", "-", "F-126 caveat (4) second clause; F-125 §3 D-032/D-171 characterisation", "-", "8", "MAPPED", "PARTLY HOLDS",
    "The four macros, the header and 166/8/2 are exact. Counts: numbers.tex is 1,162 lines only at 2ad758fb (1 header + 1,161 macros; 1,101/1,174/1,177 lines at the other SHAs); 'eight tab_*.tex' is 11 at every SHA; five fig_*.pdf holds from 2ad758fb (4 at ea1ce52b). Minor counting slips, no effect on the conclusion."),
}


def summary(fid):
    rows = [r for r in RES if r["f_id"] == fid]
    if not rows:
        return "no re-derivation recorded"
    c = collections.Counter("matches" if r["verdict"].startswith("matches") and "only" not in r["verdict"] else r["verdict"].split(":")[0].split(" (")[0]
                            for r in rows)
    bad = [r for r in rows if not r["verdict"].startswith("matches") and not r["verdict"].startswith("context")]
    head = f"{len(rows)} checks: " + ", ".join(f"{k} {v}" for k, v in c.most_common())
    if not bad:
        return head + " -- matches"
    return head + " -- " + "; ".join(f"{r['item']}: {r['verdict'][:110]}" for r in bad[:4])


COLS = ["f_id", "title_short", "heading_date", "claim_type", "key_numbers", "cited_paths", "resolved_primary", "key_path",
        "value_seen_prelim", "other_copies", "copy_status", "producing_script", "script_writes_file", "run_log", "depends_on",
        "supersedes", "superseded_by", "draft_topic", "mapping_status", "notes"]


def main():
    out = ROOT / "audit/p17"
    with open(out / "fmap_F110_F127.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(COLS)
        for fid in sorted(M, key=lambda x: int(x[2:])):
            m = M[fid]
            w.writerow([fid, m[0], m[1], m[2], m[3], m[4], m[5], m[6], summary(fid), m[7], m[8], m[9], m[10], m[11], m[12],
                        m[13], m[14], m[15], m[16], f"P17 verdict {m[17]}. " + m[18]])
    # inventory + claims for the entries with no L- row (F-111 .. F-127)
    ICOLS = ["id", "source", "source_ref", "claim_short", "numbers", "layer", "dataset_hint", "status"]
    CCOLS = ["id", "claim_plain", "stated_numbers", "source_ref", "evidence_file", "evidence_key", "level1_value", "level1_match",
             "level2_value", "level2_method", "version_used", "robustness", "verdict", "confidence", "importance", "script", "notes", "batch"]
    inv, cl = [], []
    for k, fid in enumerate([f for f in sorted(M, key=lambda x: int(x[2:])) if f != "F-110"], 1):
        m = M[fid]; iid = f"L2-{k:03d}"
        rows = [r for r in RES if r["f_id"] == fid]
        inv.append([iid, "FINDINGS.md", f"kd/docs/FINDINGS.md:{LINES[fid]} ({fid})", f"{m[1]} · {m[0]}", m[3][:400], "", "", "CANDIDATE"])
        nb = sum(1 for r in rows if not r["verdict"].startswith(("matches", "context")))
        cl.append([iid, f"{fid}: {m[0]}", m[3], f"kd/docs/FINDINGS.md:{LINES[fid]}", m[5], m[6],
                   f"{len(rows)} numbers located at the primary", "YES" if nb == 0 else "PARTLY",
                   summary(fid), "re-derived from raw by audit/scripts/p17_ledger_verify.py (lab reads via p17_labfetch, ssh BatchMode, md5-verified); tallies by p17_write_fmap.py",
                   "asix @ e917ecd3 + WT md5 dc365834", "one seed / one store per entry, as the entries state; git reads at PIN_PAPER 16def2de and PIN_OLD ea1ce52b",
                   m[17], "HIGH" if m[16] == "MAPPED" else "MEDIUM", "HIGH" if fid in ("F-115", "F-116", "F-117", "F-118", "F-121", "F-124", "F-126") else "MEDIUM",
                   "audit/scripts/p17_ledger_verify.py " + fid, m[18], "P17"])
    with open(out / "ledger_new_inventory.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(ICOLS); w.writerows(inv)
    with open(out / "ledger_new_claims.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(CCOLS); w.writerows(cl)
    print(f"fmap {len(M)} rows; inventory {len(inv)}; claims {len(cl)}")
    print(collections.Counter(m[17] for f, m in M.items() if f != "F-110"))


if __name__ == "__main__":
    main()
