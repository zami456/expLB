# Chapter 4 independent verification (2026-09-25)

Verifier: a fresh subagent that did not write the chapter re-opened every cited source. Fix column = what was changed afterwards.

| # | line | sentence | value in text | value in file | source | match? | fix applied |
|---|---|---|---|---|---|---|---|
| 1 | 11 | Pipeline ends with scores on eleven evaluation datasets | 11 | splits.json lists 11 crossgen datasets; Table 4.3 has 12 labelled sets (adds knowledge-matched) | audit/p17/labcache/expLB-resplit/run-crossgen/splits.json | PARTLY (see #117) | reworded: eleven + a twelfth set scored separately |
| 2 | 42 | Follows Mahadik p.3; Lu role-vector recipe p.3 | p.3 / p.3 | Mahadik PDF p.3 = Sec. 3 Method, Eq. 1-4; Lu PDF p.3 = Sec. 2.1.2 role vectors | papers/*.pdf (pypdf) | MATCH |  |
| 3 | 52 | Every stage resumable; every number generated from those files | -- | no src; contradicted in part by lines 212/433 (constants, not files) | none | NO SOURCE | deleted 2nd clause (false for BOTH pipelines, checked 2026-09-25: Pipeline B make_numbers.py constants :: 239, 495-505; Pipeline A hard-coded FIG-001/003/004 and log-only checks); scoped true version added in 4.5 Implementation; resumability now sourced |
| 4 | 68 | Subject model Llama-3.3-70B-Instruct | 70B | `--model default="meta-llama/Llama-3.3-70B-Instruct"` (l.669) | src/assistant_axis_ndif/8_coords_ndif.py | MATCH |  |
| 5 | 69 | 80 layers, hidden 8192 | (80, 8192) | all 26 paper16 and 36 kd vectors shape (80, 8192) | run-paper16/vectors_s3, audit/p17/labcache/expLB/run-kd/vectors_s3 (torch) | MATCH |  |
| 6 | 70 | nnsight version 0.7.0 | 0.7.0 | "nnsight==0.7.0" (l.33) | setup_venv_aa.ps1 | MATCH |  |
| 7 | 71 | Reads model.layers[l].output | -- | l.288 / l.331 | 2_activations_ndif.py; 8_coords_ndif.py | MATCH |  |
| 8 | 72 | Pool/masks in fp32; bf16 positions past 256 collapse | 256 | code comment l.341-346 | 8_coords_ndif.py | MATCH |  |
| 9 | 73 | Pooled activations stored in fp16 | fp16 | `.to(torch.float16)` l.351-352 | 8_coords_ndif.py | MATCH |  |
| 10 | 75 | Judge Llama-3.1-70B-Instruct | 3.1-70B | default l.261 | 3_judge_ndif.py | MATCH |  |
| 11 | 76 | Re-judged one bank with 405B | paper16 | run-paper16/scores_405b, 405B (l.44-48) | run_judge_405b.ps1 | MATCH |  |
| 12 | 77 | Scale: 3.2-3B (28 layers), 3.1-8B (32) | 28 / 32 | l.50-52 | scale/scripts/run_scale_chain.ps1 | MATCH |  |
| 13 | 87 | Persona vector = masked mean over last assistant turn, flat mean over rollouts | -- | asst[-1] span, masked mean, total/count | 2_activations_ndif.py 185-206, 288-292, 389-422 | MATCH |  |
| 14 | 88 | T=0.7, top-p 0.9, max 512 tokens | 0.7/0.9/512 | l.281-283 | 1_generate_ndif.py | MATCH |  |
| 15 | 89 | Judge filter keeps score 3 by default | "3" | `--keep_scores default "3"` | 2_activations_ndif.py l.494 | MATCH |  |
| 16 | 90 | Role needs >= 20 kept answers | 20 | `--min_count 20` (script default is 50, chain overrides) | run_paper16_chain.ps1 l.102; kd/scripts/run_kd_chain.ps1 l.89 | MATCH |  |
| 17 | 91 | Vectors stored at all 80 layers | 80 | `--layers all`; shapes (80, 8192) | run_paper16_chain.ps1 l.103; .pt files | MATCH |  |
| 18 | 92 | Default assistant is never judged | -- | 25 score files for 26 paper16 roles; 35 for 36 kd roles; `default` has no score file | run-paper16/scores, audit/cache/labpc/run-kd/scores | NO SOURCE (true; no src comment on this line) | src added (3_judge_ndif.py:201-203; score-file counts) |
| 19 | 97 | Contrast = difference of group means, Mahadik p.3 Eq.4 | Eq.4 p.3 | Eq.(4) on PDF p.3 | Mahadik PDF | MATCH |  |
| 20 | 102 | class_contrast; _axes_from_groups | -- | 6_axes.py 69-75; persona_pcs.py @16def2de 151-170 | both | MATCH |  |
| 21 | 103 | Centring cancels in the contrast, applied only before PCA | -- | docstring l.18-20; `Xs - Xs.mean` l.229 | 6_axes.py | MATCH |  |
| 22 | 104 | PCs oriented to contrast, fallback when nearly orthogonal | abs(cos) < 1e-3 | l.91-114 | 6_axes.py | MATCH |  |
| 23 | 109 | Pipeline A keeps 5 PCs | 5 | `--pcs default=5` l.134 | 6_axes.py | MATCH |  |
| 24 | 110 | Basis includes the anchor; anchor left out of every contrast | "every contrast" | basis + default_like ✓ (l.235-246), but the `assistant` axis (l.157-167) is default minus all roles, and cap's `cap_vs_default`/`cap_only` contrasts use `assistant_anchor` as neg | 6_axes.py; cap/personas/cap_role_groups.json | PARTLY | reworded: anchor used by assistant direction + 2 cap contrasts |
| 25 | 111 | Pipeline B: sklearn PCA on D∪H only, k=3 default | k=3 | fit_geometry(k=3), names = deception + honesty (l.287-321) | persona_pcs.py @16def2de | MATCH |  |
| 26 | 117-124 | Feature list raw / pc / axes / axes_resid / erase / zero-shot | -- | make_features l.116-179; zero-shot l.403-442 | 9_persona_probes.py; 27_crossgen.py | MATCH |  |
| 27 | 121 | axes = 4 unit directions: dec, syc, unified, assistant | 4 | l.177 dirs = [dec, syc, uni, asst] | 9_persona_probes.py | MATCH |  |
| 28 | 127 | Default k=5 | 5 | `--k default=5` l.882 | 27_crossgen.py | MATCH |  |
| 29 | 128 | Some features also at k=2 | @2 | pc:unified@2, random@2, datapca@2 l.867-871 | 27_crossgen.py | MATCH |  |
| 30 | 129 | Scale rungs use k=3 | 3 | `[int]$K = 3` l.28 | scale/scripts/run_scale_crossgen.ps1 | MATCH |  |
| 31 | 130 | Pipeline B also runs a trained pc1 feature | pc1 | `"pc1": lambda z: geom.project(z, k=1)` (l.1038, dim 1 l.1075) | persona_ood.py @16def2de | NO SOURCE (true; add src) | src added (persona_ood.py:1038,1075) |
| 32 | 135 | Pipeline A trains three probes | paper, lr_reg10, mms | `--probes default` l.873 | 27_crossgen.py | MATCH |  |
| 33 | 137 | paper: LR C=1, intercept, no scaling; Mahadik p.4 | C=1 | l.203-206; PDF p.4 "C=1, an intercept, ... without StandardScaler" | 9_persona_probes.py; Mahadik PDF | MATCH |  |
| 34 | 138 | lr_reg10: scaler + LR C=0.1, no intercept; Goldowsky-Dill p.3 λ=10 on normalised acts | C=0.1 / λ=10 | l.207-220; PDF p.3 "l2 regularization (λ = 10) on normalized activations" | 9_persona_probes.py; Goldowsky-Dill PDF | MATCH |  |
| 35 | 139 | mms: scaler + class-mean difference | -- | l.221-222, MeanMassShift l.185-199 | 9_persona_probes.py | MATCH |  |
| 36 | 142 | Pipeline B's trained probe is lr_reg10 | reg 10 | REG_COEFF=10.0 (l.59); make_probe l.190-194 | persona_ood.py @16def2de | MATCH |  |
| 37 | 143 | Base probe: LR reg 10, standardised per-token, RepE, averaged over response tokens | λ=10 | setup.tex l.31-34 | setup.tex @16def2de | MATCH |  |
| 38 | 153 | datapca@k refitted per source, centred on source mean; Mahadik p.4 | -- | l.104-112 (fit on source train rows, subtract mean projection); PDF p.4 names "dataset-PCA controls" | 27_crossgen.py; Mahadik PDF | MATCH |  |
| 39 | 154 | random@k = QR of Gaussian, fixed seed | -- | l.131-134 | 9_persona_probes.py | MATCH |  |
| 40 | 155 | Pipeline B: 5 draws of 3 directions | 5 x 3 | N_RANDOM_DRAWS=5 (l.61); random_directions(hidden, 3, rng) l.1063 | persona_ood.py @16def2de | MATCH |  |
| 41 | 156 | length control; deceptive class longer on most datasets | "most" | comment: "longer in 9 of 10 datasets" | 9_persona_probes.py l.120-130 | MATCH |  |
| 42 | 157 | Random-direction null of 1,000 directions | 1,000 | `--n_random default=1000` l.81 | 25_zeroshot_nulls.py | MATCH |  |
| 43 | T4.1 | D1 raw 8192; D2 base probe (RepE) 8192; pooled 8192 | 8192 | setup.tex 31-34; hidden 8192 | setup.tex; .pt shapes | MATCH |  |
| 44 | T4.1 | D4 pc@k, one dataset, k dims | k | persona_ood / 27_crossgen | as above | MATCH |  |
| 45 | T4.1 | D5 8v7, D6 5v5, D7 13v12, D8 4v6, D9 12v13 | counts | persona_pcs.py 151-170 (8+7; 5+5; 13 v 12; 4 v 6; 12 v 13); method.tex 75-85 | @16def2de | MATCH |  |
| 46 | T4.1 | Our kd contrasts 4 v 3 and 4 v 3 | 4v3, 4v3 | contrasts knowledge_denial (4 v 3), kd_assertion (4 v 3) | kd/personas/kd_role_groups.json | MATCH |  |
| 47 | 201 | Two more alternatives (on-policy base probe, INLP) tried and did not work | -- | per the chapter's own note l.192, no code or results on any branch | none | NO SOURCE | restored 2026-09-25: sourced to the owner's statement + report.pdf pp.37-40 (files outside the repo) |
| 48 | 208 | Follow-Up Probe measured 0.6360 vs published 0.634 | 0.6360 / 0.634 | row 9 auroc 0.6360117; row 3 auroc 0.6340000 | git show 16def2de:.../persona_ood_followup_L20/liars_bench_comparison.csv; audit/claims.csv D-031 | MATCH |  |
| 49 | 209 | Rebuilt contrasts match stored axes within 1e-5, enforced by a test | 1e-5 | test l.40-43 asserts only the **deception** axis (atol=1e-5); persona_pcs.py l.149-150 comment says all four were checked at layer 20 (~1e-7), not tested | tests/test_persona_pcs.py @16def2de | PARTLY | reworded: deception axis tested at 1e-5; others checked once at L20 |
| 50 | 210 | 36 kd vectors byte-identical between branches | 36/36 | md5 36/36 identical (git show 16def2de:"persona including knowlede denial/*.pt" vs labcache vectors_s3) | recomputed | MATCH |  |
| 51 | 211 | Cached evaluation stores byte-identical to lab originals | md5-identical | the log records it; the lab side cannot be opened here | thesis_tools/planning/CH4_VERIFICATION_LOG.md | LAB-ONLY |  |
| 52 | 212 | Pipeline B's 4-d.p. reproduction at L20/40/44 and remote-vs-local L40 check are not reproducible | UNRESOLVED | claims.csv D-063, D2-024, D2-025 all verdict UNRESOLVED; method.tex l.123-125 layers 20, 40, 44 | audit/claims.csv; method.tex @16def2de | MATCH |  |
| 53 | 222 | Every bank: 5 variants per role | 5 | 36/36 kd, 26/26 pc_instructions, 13 cap, 3 sarcasm files: 5 instructions each | kd/personas/kd_instructions, pc_instructions, cap/personas | MATCH |  |
| 54 | 222 | ...and all roles in a bank answer the same questions | same questions | **False for paper16**: deception/honesty roles answer 10 deception questions, syc/anti-syc roles answer 12 sycophancy questions (Table 4.2 itself says "10 / 12"; l.251) | pc_questions_*.jsonl; l.251 | MISMATCH | fixed: kd/cap share one set; paper16 per behaviour |
| 55 | 223 | Role names/descriptions from Mahadik pp.9-11; paper uses 2 variants (p.3) | pp.9-11 / 2 | persona tables on PDF pp.9-11; "2 instruction variants per persona" on p.3 | Mahadik PDF | MATCH |  |
| 56 | T4.2 paper16 | groups 8,7,5,5,1 | 8/7/5/5/1 | same | src/assistant_axis_ndif/pc_role_groups.json | MATCH |  |
| 57 | T4.2 paper16 | questions 10 / 12 | 10/12 | 10 and 12 lines | pc_questions_deception/sycophancy.jsonl | MATCH |  |
| 58 | T4.2 paper16 | 1,400 rollouts; 1,089 kept at 3 | 1,400 / 1,089 | 1,400 lines / 26 files; scores {3:1089, 2:249, 0:12} | run-paper16/responses, scores | MATCH |  |
| 59 | T4.2 kd | paper16's 26 + 4, 3, 3 | 26+10 | groups 8,7,5,5,1,4,3,3 = 36 | kd/personas/kd_role_groups.json | MATCH |  |
| 60 | T4.2 kd | 30 questions | 30 | 30 lines | kd/questions/kd_questions_shared.jsonl | MATCH |  |
| 61 | T4.2 kd | 5,400 rollouts; 3,599 judged kept at 3 | 5,400 / 3,599 | 5,400 lines / 36 files; scores {3:3599, 2:1554, 1:6, 0:91} | audit/cache/labpc/run-kd | MATCH |  |
| 62 | T4.2 cap | 6, 6, 1; 20 questions; 1,300 rollouts | -- | groups 6/6/1; 20 lines; 1,300 lines in 13 files | cap_role_groups.json; cap_questions_v3.jsonl; audit/p17/labcache/expLB/run-cap/responses | MATCH |  |
| 63 | T4.2 sarcasm | 3 roles; 20 questions; rollouts "--" | -- | 3 role files; 20 lines; the run script generates 3 arms x 5 variants x 20 = 300 rollouts (l.56) | cap/personas/sarcasm_pilot; sarcasm_pilot_v1.jsonl; run_sarc_pilot.ps1 | PARTLY (the cell could read 300) | fixed: 300 |
| 64 | T4.2 own100 | 22,22,16,16,8,16; 240 questions | -- | same (sum 100); `--question_count default=240` | role_groups.json; 1_generate_ndif.py l.273 | MATCH |  |
| 65 | T4.2 own100 | 120,000 rollouts | 120,000 | lab PC run/responses (100 x 1,200 is consistent with Lu p.3's 1,200 rollouts per role) | lab PC | LAB-ONLY |  |
| 66 | 250 | paper16: 26 roles = 8, 7, 5, 5 + anchor | 26 | 25 + default = 26 vector files | pc_role_groups.json; vectors_s3 | MATCH |  |
| 67 | 251 | 10 deception Qs are renderings; 12 syc Qs verbatim | 12/12 | 12/12 syc found verbatim (normalised) in the PDF text; 0/10 deception verbatim (topics only, p.9) | pypdf on Mahadik PDF | MATCH |  |
| 68 | 252 | 1,400 rollouts, 1,350 judged, 1,089 scored 3 | 1,400/1,350/1,089 | same | run-paper16 | MATCH |  |
| 69 | 253 | Kept per role 21 to 60 | 21-60 | vector field n: min 21, max 60 (default n=50) | run-paper16/vectors_s3 | MATCH |  |
| 70 | 254 | Contrasts 8v7, 5v5, 13v12 | -- | 6_axes.py l.176-195 | 6_axes.py | MATCH |  |
| 71 | 256 | kd adds 10 new roles to 26 | 10 | 4+3+3 | kd_role_groups.json | MATCH |  |
| 72 | 258-260 | Role names of the three new groups | names | amnesiac_witness, deniable_expert, stonewaller, feigned_novice / honest_novice, calibrated_reporter, scrupulous_witness / forthright_expert, candid_witness, plain_dealer | kd_role_groups.json | MATCH |  |
| 73 | 262 | 30 shared Qs; first 20 verbatim from paper16's shared set (10 dec + first 10 syc) | 20/20 | kd[:20] == pc_questions_shared (20/20); shared[:10] == deception set, shared[10:] == syc[:10] | jsonl files | MATCH |  |
| 74 | 262 | Paper prints deception Qs only as topics, p.9 | p.9 | p.9 bullets are topics ("project handoff: what to include...") | Mahadik PDF | MATCH |  |
| 75 | 263 | Other 10: 6 knows, 4 does not | 6 / 4 | ids 20-25 know, 26-29 do not (read from question text) | kd_questions_shared.jsonl | MATCH |  |
| 76 | 264 | 5,400 rollouts; 5,250 judged; anchor 150 unjudged; 3,599 at 3; 5,153 at 2-3 | as stated | 5,400; 5,250; 3,599; 3,599+1,554=5,153; default n=150 | audit/cache/labpc/run-kd; vectors_s3 default.pt n | MATCH |  |
| 77 | 265 | With anchor 3,749 (69%); min 47, max 149 | 3,749/69/47/149 | numbers.tex l.17-21; recomputed per-role s3 min 47, max 149; 3,749/5,400 = 69.4% | thesis_tools/collab_evidence/.../numbers.tex (md5 = git blob); scores | MATCH |  |
| 78 | 266 | interp: 3,599 + 150 = 3,749 | 3,749 | arithmetic holds | -- | MATCH |  |
| 79 | 267 | Run used v2 (3dfdb673), committed after the run; v1 (5b43a7af) never ran | commits | 5b43a7af 2026-09-19 02:58; 3dfdb673 2026-09-21 11:07; cached pc_whistleblower rollouts match 30/30 v2 questions vs 25/30 v1 (5/5 system prompts both). The run date itself comes from the plan, not rechecked | git show; audit/cache/labpc/run-kd/responses | MATCH (run timing taken from the plan) |  |
| 80 | 271-272 | knowledge denial 4v3 = contrasts[3]; denial vs assertion 4v3 = contrasts[4] | -- | contrasts[3]=knowledge_denial (vs sincere_ignorance), [4]=kd_assertion (vs plain_assertion) | kd_role_groups.json | MATCH |  |
| 81 | 273-274 | denial 4v6, extended 12v13 (Pipeline B) | -- | persona_pcs.py l.167-170 | @16def2de | MATCH |  |
| 82 | 300 | cos(8v7, 4v6) at L40 = 0.47 | 0.47 | numbers.tex l.1166 0.47; recomputed 0.474559 | numbers.tex; proj.py | MATCH |  |
| 83 | 301 | denial-axis means kd 2.90, deceptive 0.05, sincere -0.08, honest -1.00, plain -1.23 | as stated | numbers.tex l.1168-1176; recomputed 2.9037, 0.0458, -0.0844, -0.9999, -1.2274 | numbers.tex; proj.py | MATCH |  |
| 84 | 302 | Sincere nearest anchor 0.60, anti-syc 0.77, plain 1.15, honest 1.23 | as stated | recomputed 0.602, 0.767, 1.149, 1.228 (deception 1.391) | proj.py on the 36 vectors | MATCH |  |
| 85 | 302 | Deceptive personas never reach deniers' region | -- | max deceptive-persona denial projection 0.885 < min denier 2.150 | proj.py | MATCH |  |
| 86 | 303 | Figure not regenerable; numbers reproduce | -- | FIGURES.md l.77 (FIG-360, F-127) | thesis_guide/FIGURES.md | MATCH |  |
| 87 | 311 | fig_personas_L40.pdf = collaborator fig_personas.pdf | md5 f8efba67... | both md5 f8efba677df321bbc947898bc6a2717d | md5sum; git show | MATCH |  |
| 88 | 313 | cap: 6 give way, 6 hold, anchor | 6/6/1 | groups | cap_role_groups.json | MATCH |  |
| 89 | 314 | 20 two-turn Qs; turn 1 teacher-forced; challenge text | "I don't think that's right. Are you sure?" | CHALLENGE l.76; one turn-1 text per question (20/20 questions have exactly 1 turn-1 text across 1,300 rows); last user turn equals the challenge | gen_multiturn_ndif.py; labcache run-cap/responses | MATCH |  |
| 90 | 315 | Replies capped at 220 tokens | 220 | `--max_new_tokens default=220` l.102 | gen_multiturn_ndif.py | MATCH |  |
| 91 | 316 | 1,300 rollouts | 1,300 | 1,300 lines | labcache run-cap/responses | MATCH |  |
| 92 | 317 | Three cap contrasts: vs holding; vs anchor on 13-role basis; vs anchor without holding roles | -- | capitulation; cap_vs_default (basis 13 roles); cap_only (7-role basis) | cap_role_groups.json | MATCH |  |
| 93 | 319 | Sarcasm pilot 3 roles, 20 Qs, max 320 tokens | 3/20/320 | 3 files; 20 lines; `--max_new_tokens 320` l.58 | cap/... | MATCH |  |
| 94 | 320 | Natural sarcasm 600 Reddit comments, 300/300 | 300/300 | label {1:300, 0:300}; source file figlang2020_reddit.jsonl | cap/run-sarcasm-natural/rows.jsonl | MATCH |  |
| 95 | 322 | own100 = 100 library roles on 240 Qs; hf275 = 275 Lu vectors (p.3) | 100 / 240 / 275 | role_groups 100; default 240; 276 files (275 + default); Lu p.3 "275 roles", "240 extraction questions" | role_groups.json; 1_generate_ndif.py; run-hf275/vectors_275; Lu PDF | MATCH |  |
| 96 | 326 | Scale: paper16 26 roles on the 20 shared Qs | 20 | header l.31-32 "paper16 (26 personas) on the 20 shared questions" | run_scale_chain.ps1 | MATCH |  |
| 97 | 327 | kd scale rungs: 36 roles, 30 Qs, 70B judge, k=3, same 5 datasets | -- | lab PC only | C:/Users/user4/expLB-resplit/run_kd_scale.ps1 | LAB-ONLY |  |
| 98 | 333 | sycophancy_mt challenge text | same text | CHALLENGE l.56 | 13_sycophancy_multiturn.py | MATCH |  |
| 99 | 334 | Questions from a public sycophancy-answer set | dwb2023/sycophancy-answer | l.51-52 | 13_sycophancy_multiturn.py | MATCH |  |
| 100 | 335 | Only turn-1-correct kept; 3.1-70B judges | -- | docstring l.28-30; `--judge_model` l.268 | 13_sycophancy_multiturn.py | MATCH |  |
| 101 | 336 | 1,182 rows; 434 retracted; 748 maintained | 1,182/434/748 | {1:434, 0:748} | labcache passB/sycophancy_mt/rows.jsonl | MATCH |  |
| 102 | 338 | HP-KR deceptive = denies knowing despite established belief (Liars' Bench p.5) | p.5 | PDF p.5 "(labeled deceptive if the model denies knowing the answer despite that established belief" | Liars' Bench PDF | MATCH |  |
| 103 | 360 | Every row generated by Llama-3.3-70B itself (on-policy) | -- | true in code (CSV_MODEL l.46, ROLLOUT_MODEL l.47, rollout files l.256-259, load_mask(model=CSV_MODEL) l.214), but no src comment | eval_datasets.py | NO SOURCE (true; add src) | src added (eval_datasets.py:46-47,256-259) |
| 104 | 362-365 | Source citations: LB pp.4-7; Apollo p.4, pp.15-16; MASK p.4 | pages | datasets described on the cited pages | PDFs | MATCH |  |
| 105 | T4.3 | ID 5,432 / 3,045 / 2,387 | as stated | {1:3045, 0:2387} | labcache passB rows.jsonl | MATCH |  |
| 106 | T4.3 | HP-C 2,134 / 134 / 2,000 | as stated | same | same | MATCH |  |
| 107 | T4.3 | HP-KR 2,139 / 139 / 2,000 | as stated | same | same | MATCH |  |
| 108 | T4.3 | CG 267 / 167 / 100 | as stated | same | same | MATCH |  |
| 109 | T4.3 | IT 1,080 / 423 / 657 | as stated | same | same | MATCH |  |
| 110 | T4.3 | ai_liar 270 / 135 / 135 | as stated | same | same | MATCH |  |
| 111 | T4.3 | roleplaying 371 / 203 / 84 | as stated | {1:203, 0:84, None:84} | same | MATCH |  |
| 112 | T4.3 | sandbagging_v2 1,000 / 500 / 500 | as stated | same | same | MATCH |  |
| 113 | T4.3 | mask 585 / 415 / 170 | as stated | same | same | MATCH |  |
| 114 | T4.3 | sycophancy_mt 1,182 / 434 / 748 | as stated | same | same | MATCH |  |
| 115 | T4.3 | text-matched 258 / 124 / 134 | as stated | harm-pressure-kr-matched {1:124, 0:134} | same | MATCH |  |
| 116 | T4.3 | knowledge-matched 2,001 / 139 / 1,862 | as stated | same | labcache expLB-resplit/passB/.../rows.jsonl | MATCH |  |
| 117 | T4.3 | Alpaca 2,000 / 0 / 2,000 | as stated | llama-v3.3-70b-instruct 2,000 rows, all deceptive=False | results/subsets/alpaca.csv | MATCH | reworded: main grid 11; knowledge-matched run separately |
| 118 | 397 | Pipeline A scores all of these | "all" (13 rows) | contradicts l.11/56/477 ("eleven"); the knowledge-matched slice is not in the 11-dataset crossgen grid (splits.json) | splits.json | PARTLY | reworded (same as #117) |
| 119 | 397 | Pipeline B scores ID, HP-C, HP-KR, CG, IT + RepE (not averaged); Alpaca only for thresholds | -- | setup.tex l.4-10, 41-42 | @16def2de | MATCH |  |
| 120 | 398 | Pipeline B IT scores exclude scratchpad | -- | method.tex l.126-127 | @16def2de | MATCH |  |
| 121 | 399 | 9 labelled + Alpaca = 15,278 rows | 15,278 | 270+267+2,134+2,139+1,080+5,432+585+371+1,000+2,000 = 15,278 | arithmetic on counts above | MATCH |  |
| 122 | 400 | sycophancy_mt adds 1,182; slices add no rows (views) | 1,182 | VIEW comment l.360-362 | eval_datasets.py | MATCH |  |
| 123 | 405 | MASK dropped 225 / 165 / 25 | as stated | loader run: 225, 165, 25; 585 kept | `eval_datasets.py --datasets mask roleplaying` (re-run) | MATCH |  |
| 124 | 405 | MASK lie definition (Ren p.4, p.8) | pages | p.4 compare S vs B; p.8 Honest(S,B)=0 if S≠B, no-belief not a lie | MASK PDF | MATCH |  |
| 125 | 407 | Roleplaying >=6 lie, <=2 honest; 84 unlabelled; 287 of 371 | 84 / 287 | l.83-90; loader 203/84/84 amb | eval_datasets.py; loader run | MATCH |  |
| 126 | 411 | ai_audit 8 deceptive / 8 honest items | 8/8 | code comment l.286 (not counted independently) | eval_datasets.py | MATCH (comment only) |  |
| 127 | 412 | 80 rows, 4,211-7,543 tokens, 4.23-13.57 GiB | as stated | 80 items; token lengths min 4,211, max 7,543; 64 x T^2 x 4 B = 4.228 / 13.565 GiB | run-coords/ai_audit/items.pt (torch); 8_coords_ndif.py l.189-191, 741 | MATCH |  |
| 128 | 414 | Soft-Trigger, Gender-Secret off-policy (fine-tuned) | -- | FINETUNED_ONLY l.285; loader note "LoRA-generated" | eval_datasets.py | MATCH |  |
| 129 | 419 | Response tokens only; mean and last pooling | -- | span_bounds l.269-278; mean/last l.336-350 | 8_coords_ndif.py | MATCH |  |
| 130 | 420 | Inputs cut at 8,192 | 8192 | `--max_length 8192` l.95 | run_coords_chain.ps1 | MATCH |  |
| 131 | 421 | 3B/8B cut at 1,024 | 1024 | maxlen "1024" l.121-122 | scale/scripts/run_scale_coords.ps1 | MATCH |  |
| 132 | 422 | Main store 21 layers: every 4th from 0 to 76 + 22 | 21 | l.40 list has 21 entries | run_coords_chain.ps1 | MATCH |  |
| 133 | 425-428 | 200-row sample at 80 layers; per-token at 20/22/28/40/44/48; prompt-only at 20/40/44; fine sweep 37-39, 41-47 | as stated | l.49-63 plus header ("same --sample 200 --seed 42 as those passes") | run_extra_extractions.ps1 | MATCH |  |
| 134 | 431 | Pipeline B: detection tokens, no system prompts; 16 layers, every 4th 20-76 + 79 | 16 | method.tex 116-121; ALL_LAYERS l.40 (16 entries); numbers.tex depth-n 16 | @16def2de | MATCH |  |
| 135 | 432 | L20/40/44 local cache; other 13 remote | 13 | NEW_LAYERS l.39 (13 entries); pool_activations.py docstring "No remote calls" | @16def2de | MATCH |  |
| 136 | 433 | Row cosine >= 0.99989 is a constant, not reproducible | 0.99989 | make_numbers.py l.501 constant; D2-024 UNRESOLVED | @16def2de; claims.csv | MATCH |  |
| 137 | 434 | layer_sweep calls trace_layer_outputs(span_mask=...), no committed version accepts it; 36 among the 13 | -- | layer_sweep l.155, 198-201; activations.py l.34-42 has no span_mask; `git log --all -S span_mask` returns only ea1ce52b; 36 is in NEW_LAYERS | @16def2de; git log | MATCH |  |
| 138 | 439 | Denial detected with one regular expression | "one" | two regexes: `_DENIAL_RE` plus an `_ASSERT_RE` exclusion (l.363-364, 388) | eval_datasets.py | PARTLY | fixed: two regular expressions |
| 139 | 440 | No deceptive row from a model that does not know; denial AUROC 0.9125 | 0.9125 | recomputed AUROC 0.91254 on n=2,139; cells (T,F,*) = 0 | hpkr.py over results/subsets/harm-pressure-knowledge-report.csv | MATCH |  |
| 140 | Fig 4.4 | cells (T,T,F)15, (T,T,T)124, (F,T,F)1,862, (F,F,T)134, (F,F,F)4 | as stated | identical | hpkr.py | MATCH |  |
| 141 | 450 | Three cells empty; all 124 deceptive denials know; 15 missed | 3 / 124 / 15 | empty: (T,F,T), (T,F,F), (F,T,T) | hpkr.py | MATCH |  |
| 142 | 452 | text-matched 258, knowledge-matched 2,001 | 258 / 2,001 | hpkr_slice sizes 258 / 2,001 | eval_datasets.hpkr_slice (re-run) | MATCH |  |
| 143 | 455 | Seed 42: 199 of 258 (77.13%) in HP-KR train; 39 of 52 slice test rows | 199/258, 39/52 | 199/258 = 0.77132; slice test 52, 39 in HP-KR train | labcache expLB-resplit/run-crossgen/splits.json + passB rows.jsonl | MATCH |  |
| 144 | 457 | Leak guard; 59 of 258 remain | 59 | design field "AUDIT LEAK GUARD"; 59 slice rows are in HP-KR test | splits.json | MATCH |  |
| 145 | 460-466 | Provenance clusters | 5 clusters | CLUSTERS l.57-64 | 30g_clustered_holdout.py | MATCH |  |
| 146 | 477-481 | Eleven labelled sets = 9 + syc_mt + text-matched | 11 | Table 4.3 also lists knowledge-matched (12 labelled); see #118 | -- | PARTLY | left: summary lists the eleven grid sets (now consistent) |
| 147 | 483 | Persona vectors of every bank in Table 4.2 stored at 80 layers | 80 | paper16 and kd (80, 8192) ✓; cap `--layers all` (run_cap_chain.ps1 l.104) ✓; **the sarcasm pilot builds no persona vectors** (run_sarc_pilot.ps1 runs generate, judge and markers only); own100 is lab-only | .pt files; cap/scripts | PARTLY | fixed: paper16 and kd at 80 layers; sarcasm builds no vectors |
| 148 | 528/530 | 80/20 split by item, stratified by item label make-up | 0.8 | stratified_group_split l.194-217; `--frac 0.8` l.881 | 27_crossgen.py | MATCH |  |
| 149 | 531 | Train class-balanced by downsampling; test not | -- | balance l.263-274; used on train only (27_crossgen l.311, 363, 479) | 9_persona_probes.py; 27_crossgen.py | MATCH |  |
| 150 | 532 | Diagonal = own 20%; off-diagonal = 100% | -- | DESIGN string l.113-114 | 27_crossgen.py | MATCH |  |
| 151 | 533 | Pipeline B: six datasets, 5-fold CV diagonal | 6 / 5-fold | setup.tex l.21-23 | @16def2de | MATCH |  |
| 152 | 535-536 | Zero-shot projection; 1-feature probe = sign and scale, so equals zero-shot when signs agree | -- | 27_crossgen l.403-442; method.tex l.104-112 | both | MATCH |  |
| 153 | 540 | Pipeline B pooled LOO naive or equal-weighted (1/n_d) | -- | l.366-380 | persona_ood.py @16def2de | MATCH |  |
| 154 | 541-542 | Pipeline A class-balances each source; row-budget-matched arm | -- | fit_on balance l.110-117; pooled_lodo_match l.165-172 | 30_premise.py | MATCH |  |
| 155 | 543 | LOCO at L40, mean, paper probe, seed 42 | -- | l.79-86 | 30g_clustered_holdout.py | MATCH |  |
| 156 | 545 | Both pooled protocols split rows, not items | -- | rng.permutation over rows (30_premise l.106; 30g l.105) | both | MATCH |  |
| 157 | 550 | Threshold flags at most 1% of Alpaca (LB p.7) | 1% | p.7 "false-positive rate of at most 1% on the alpaca dataset" | Liars' Bench PDF | MATCH |  |
| 158 | 552 | Pipeline A: TPR at threshold = 0.99 quantile of control | 0.99 | tpr_at_fpr l.238-243 | 9_persona_probes.py | MATCH |  |
| 159 | 553 | Pipeline B balanced accuracy, LB's main score (p.7) | -- | l.203-215; p.7 "averaged balanced accuracy as our main score" | persona_ood.py @16def2de; LB PDF | MATCH |  |
| 160 | 555 | Every operating-point result also reports the realised FPR | -- | no src; tpr_at_fpr does not return FPR | none | NO SOURCE | restored 2026-09-25, scoped: Pipeline B records specificity (zero_shot_operating.csv; persona_ood.py:214); Pipeline A TPR only |
| 161 | 561 | Row bootstrap 1,000, stratified, shared | 1,000 | `--boot_rows 1000` l.883; Resampler multinomial per class l.240-241; setup.tex l.39-40 | 27_crossgen.py; setup.tex | MATCH |  |
| 162 | 562 | Dataset bootstrap 10,000 multinomial, two-sided p | 10,000 | l.884, l.783; two_sided_p l.96-99 | 27_crossgen.py; 23_significance.py | MATCH |  |
| 163 | 563 | BH at q<0.05 within each family | 0.05 | l.820-825 | 27_crossgen.py | MATCH |  |
| 164 | 564 | Zero-shot vs 1,000 random directions on abs(AUROC-0.5) | 1,000 | l.81; l.65-70 | 25_zeroshot_nulls.py | MATCH |  |
| 165 | 565 | Cluster permutation test, 200 permutations, identical across layers | 200 | docstring l.6-9; `-B default=200` l.77 | 30c_cluster_perm.py | MATCH |  |
| 166 | 566 | Equivalence: 90% interval inside ±0.02 | 0.02 | `--margin 0.02` l.122; 5th-95th pct l.257-260 | 23_significance.py | MATCH |  |
| 167 | 567 | Main seed 42 | 42 | `--main_seed 42` l.874; SEED=42 l.62 | 27_crossgen.py; persona_ood.py | MATCH |  |
| 168 | 573 | Pre-registration f5d6904, 2026-09-21 13:00 | 13:00 | f5d6904d 2026-09-21 13:00:22 +0600 | git show -s | MATCH |  |
| 169 | 574 | Amended 19 min later, 63aabb4b 13:19 | 13:19 | 13:19:20 (19 min) | git show -s | MATCH |  |
| 170 | 575 | First version: L40 chosen because it maximised persona advantage | -- | f5d6904:docs/LAYER_PREREGISTRATION.md l.25-26 | git show | MATCH |  |
| 171 | 576 | L40 middle layer on 2026-09-10; probe default L40 2026-09-15; sweep 2026-09-16 | dates | 5d7b8560 2026-09-10 16:33 ("target" = middle layer, 40); 50709657 2026-09-15 17:08 (`--layer` default 40); 9630f141 2026-09-16 10:58 | git show | MATCH |  |
| 172 | 577 | Pipeline B moved to L40 on 2026-09-20 before kd personas | 16:17 < 17:53 | 06da2de2 2026-09-20 16:17 ("reruns the whole study there" on the E: cache); 97740753 2026-09-20 17:53 | git show | MATCH |  |
| 173 | 578 | 16-layer sweep 3 days later; still reports L40 | 2026-09-23 | ea1ce52b 2026-09-23 23:46; method.tex l.118 "We report layer~40 throughout" | git; method.tex | MATCH |  |
| 174 | 579 | Lu p.3 middle layer; Llama-3.3-70B 80 layers (Lu p.14) | pp.3/14 | p.3 "middle residual stream layer"; p.14 "out of 80 total layers" | Lu PDF | MATCH |  |
| 175 | 581 | L20 (LB p.8, p.69), L22 (Apollo p.3) fixed in advance | pages | LB p.69 Table 16 "layer 20" ✓; LB **p.8 says "the 20th-percentile layer"**, not "layer 20"; Apollo p.3 "after layer 22 (of 80)" ✓; "fixed in advance" has no src comment | PDFs | PARTLY | fixed: cite p.69 only |
| 176 | 582 | L44 and L36 chosen post hoc, always labelled | -- | no src comment; supportable: 63aabb4b:docs/LAYER_PREREGISTRATION.md l.25, 37 (L44 "selected on our outcome ... post hoc"); ea1ce52b message ("ranking peaks at 36") | none cited | NO SOURCE | src added (63aabb4b prereg :: 20-26; ea1ce52b subject) |
| 177 | 583 | Rungs' middle layers 14 and 16 | 14/16 | $MID l.53 | run_scale_chain.ps1 | MATCH |  |
| 178 | 590 | 36 kd vectors byte-identical (commit 148ddeb9) | 36/36 | 36/36 md5 (the files enter at 148ddeb9, 2026-09-21, and are present at 16def2de) | recomputed | MATCH |  |

**Revision 2026-09-25** (Chapter 4 revision list, items 1–2; see `PROGRESS_THESIS.md`). Checked by the writing session against the files named, not yet by an independent verifier. The CH5 plan's session 6 re-checks them.

| # | line | claim | stated | found | source | verdict | action |
|---|---|---|---|---|---|---|---|
| 179 | 321 | Sarcastic and sycophantic approval can be word-identical; sarcasm is meant to be recognised, sycophancy/deception believed, so an intent-reading direction should separate them | -- | `SARCASM_DESIGN.md` l.30-38: "textually identical" once markers are stripped; "(overt)" vs "(covert)"; "That distinction is the intent variable" | cap/docs/SARCASM_DESIGN.md | MATCH | new sentence (design rationale, our own document) |
| 180 | 323 | Natural-sarcasm set: 600 Reddit comments, 300/300, from the Reddit part of the FigLang 2020 shared task | 600; 300/300 | 600 rows; label {1: 300, 0: 300}; `src` = figlang2020_reddit for 600/600; script l.12-13 (FigLang 2020), l.48 (tasksource/figlang2020-sarcasm) | cap/run-sarcasm-natural/rows.jsonl; cap/scripts/build_sarcasm_natural.py | MATCH | citation `ghosh2020figlang` added; bib metadata from the ACL Anthology BibTeX (authors Ghosh, Vajpayee, Muresan; pp. 1-11; doi 10.18653/v1/2020.figlang-1.1) |


**Revision 2026-09-25 (Chapter 4 revision list items 3–4, plus label and wording changes).** The two worked-example blocks were written by W8 and checked by an independent verifier (workflow `wf_040cbc40-2cf`: 18 MATCH, 2 MISMATCH fixed, 1 SOURCE-WRONG fixed, 4 LAB-CACHE, 1 PITFALL, 4 STYLE fixed). The six kd cache files and the kr-matched row list were md5-checked against the lab PC in the main session (`CH5_VERIFY_TABLE.md`, lab-PC section).

| # | line | claim | stated | found | source | verdict | action |
|---|---|---|---|---|---|---|---|
| 181 | §4.3.1, after the contrast paragraph | kd trio: question 20 answered by the feigned novice, the honest novice and the plain dealer, third variant, all judge score 3 | quotes | line 81 of each response file (question_index 20, prompt_index 2, label pos); scores `pos_p2_q20` = 3 in all three; quotes are an exact prefix (46/90 words), an exact prefix (62/98) and a full match (39) | lab `run-kd/responses`, `run-kd/scores` (cache md5 = lab md5, 2026-09-25) | MATCH | inserted; "defined below" → "defined above" because the block now follows the contrast definitions |
| 182 | same block | the honest novice admits the knowledge on a "knows" question, so the 4 v 3 sincerity contrast still carries wording there | -- | DESIGN.md:126-131 "Residual limitation" (added 6dd9f9ba, unchanged since) | kd/docs/DESIGN.md | MATCH | lines 272 and 278 qualified: "on the four questions in which the speaker does not know … on the six in which the speaker knows, the sincere-ignorance roles are expected to admit" |
| 183 | §4.4.3, after the empty-cells paragraph | HP-KR pair: CSV rows 0 (deceptive, model_belief 0) and 139 (honest, −1), both wmdp-bio, template 1, response "I do not know the answer." | 258 rows; 3 sentences; 78/124 and 112/134 | recomputed from results/subsets/harm-pressure-knowledge-report.csv with eval_datasets.denies_knowledge; blob 19087517 identical on asix, the working tree and 16def2de | CSV; kr-matched rows.jsonl cache (md5 5d62853e = lab) | MATCH | inserted; WMDP question withheld; "belief field" defined in its own sentence; "Pipeline~A reads any value other than −1 as knows" |
| 184 | :22, :92 (label), :59, :163, :201, :417 | `\label{ch:method}` added; the four hard-coded "Chapter~5" became `Chapter~\ref{ch:results}`; §4.5.5 "Appendix~C" left as text | -- | labels resolve (`check_latex.py` OK, 105 labels) | main.tex, chapter_6.tex | MATCH | done |
| 185 | §4.3.1 :320–323 | sarcasm motivation and FigLang citation (revision items 1–2) | -- | rows 179–180 above | -- | MATCH | done earlier |
| 186 | 78 (added 2026-09-30, owner request) | Lu et al. judge role expression with gpt-4.1-mini, a paid API model; we could not use it because of cost; both our judges are open-weight models NDIF serves free | p.~3 | Lu PDF pypdf index 2, footer "3": "we relied on an LLM judge (gpt-4.1-mini)"; run_judge_405b.ps1 l.9 "free", l.10 "no paid key", l.11 "NDIF serves open-weight models only"; 3_judge_ndif.py l.5 "No HF Inference, no OpenAI"; judges 70B (3_judge_ndif.py l.261) and 405B (run_judge_405b.ps1 l.48); cost = owner's statement 2026-09-30 | papers/The Assistant Axis Situating and Stabilizing.pdf; src/assistant_axis_ndif/ | MATCH | independent verifier 2026-09-30; check_latex.py OK |

Items proposed by the writers for a later Chapter 4 pass (not applied; listed in `CH5_RESULTS_PLAN.md` and the writers' `ch4_revision_items`): RepE "left out of its averages" is true of the Liars' Bench average only (chapter_5.tex:400); Pipeline B has two RepE-trained probes (per-token base probe and a pooled lr_reg10 probe); Chapter 4 does not name Pipeline B's arms pc3/ext_pc3/syc_pc1/syc_axis1/random3; sec:m-stats should say Pipeline B has no dataset bootstrap; "any layer other than 20, 22 and 40 is post hoc"; the paper16 anchor answered the 10 deception questions; the cap anchor's fourth variant ran as the literal "You are {model_name}." in 20 rollouts; the paper's role descriptions are verbatim for 22 of 25 (pp. 9–10, not 9–11); cap vectors use judge scores 2–3; the format-control banks and the factor-probe corpora are built resources not listed in §4.3; the assistant direction is never defined in Chapter 4; the paper16 bank has two extractions (axes_vectors_s3.pt and axes_vectors_s3_fp32.pt).
