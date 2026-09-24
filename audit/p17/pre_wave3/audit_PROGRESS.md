# PROGRESS — audit of expLB

Read `audit/BRIEF.md` first, then this file. **IF YOU ARE A NEW SESSION STARTING PHASE 4: read `audit/PHASE4_PLAN.md` — it is complete, the sample is built, and it tells you everything.** Current phase: **PHASE 4 RUNNING (started 2026-09-23 ~17:55). See the PHASE 4 EXECUTION LOG at the very end of this file — that is the live section.** Earlier header, kept for history: BLOCKED (2026-09-23 ~13:0x): 7 agents killed by a session rate limit, and the LAB PC IS UNREACHABLE (ping 100% loss) — **both since resolved**; the lab PC is reachable again. Previously: **W4 + WAVE 6 RUNNING — 7 agents. Phase 3 complete; Phase 4 next after these land.** The owner has decided to REFRAME the paper as methods-and-negative-results; see "W4 + WAVE 6 LAUNCHED" at the end of this file for what that changed. Original note: **Phase 4 is next** (independent re-derivation of the HIGH claims + a random ~20% sample). Phase 2 closed at 231/692 on the owner's scope decision. See "WAVE 5" near the end of this file for the decision and the results. Phases 0, 1 and M are complete.

## Phase 1 log (newest last)
- 20:51 Created lab scratch `C:\Users\user4\audit_scratch\` (census/, scripts/, out/). Every write there is logged in audit/LABPC_WRITES.md.
- **Transfer method VERIFIED:** `ssh … "tar -cf - -C <dir> <sub>"` → local `tar -xf`. 26 JSONL (3.1 MB) + 26 .pt (2.6 MB each, 68 MB) all md5-identical to the lab-side Get-FileHash.
  - Kept as cache: audit/cache/labpc/run-paper16/{responses,vectors_s3_fp32}; 52 rows in audit/cache/MANIFEST.csv, all MATCH.
- Helper `audit/scripts/labpc_run.py` runs a .ps1 on the lab PC with __KEY__ substitution (no backslashes through bash). Long scripts are uploaded into the scratch folder by tar and run with -File.
  - Pitfall seen: a command line over ~8 KB fails with "The command line is too long." (first census attempt did nothing).
- Census so far:
  - his branch = audit/phase1/census_pcood.tsv (846 blobs under DD, md5 of git blob);
  - asix = census_asix_git.tsv (9,527 blobs);
  - laptop ignored files = census_laptop_ignored.tsv (486: .pt vectors, logs, smoke dirs).
  - Lab census (p1_labpc_census.ps1) and Drive lsjson (p1_labpc_drive_census.ps1, run on the lab PC with its rclone) started 20:56 → scratch\census\.
- 20:56–21:25 **Census complete** (all read-only on owner files):
  - lab = 210,464 files (audit/phase1/labpc/labpc_census.tsv; 75 files >200 MB not hashed = SKIP_BIG, 2 ERR) + line-ending-normalised md5 for 205,025 text files (labpc_census_eol.tsv);
  - Drive = 277,407 files with md5 (drive_census.tsv.gz, via the lab's rclone lsjson);
  - asix = 9,527 blobs; his branch = 846 blobs; laptop ignored = 486.
  - Join run on the lab PC (laptop RAM) → joined_labpc.tsv.gz (one row per lab file: identical copies on Drive / asix / his branch, same-name-different-content in asix), drive_not_on_lab.tsv.gz (407), asix_results_not_on_lab.tsv.gz (77), join_summary.txt. All transfers md5-verified.
  - Pitfall: a 232 MB stream broke ("Connection reset by peer") → compress on the lab first (gz), then transfer.
- RF-1 resolved (see red flags). RF-10, RF-11, RF-12 added.
- 21:35 Read-only lab peek tool `audit/scripts/p1_peek.py` (uploaded; modes info/head/cat/keys/get/grep/ls/pt).
  - Run it with `python audit/scripts/labpc_run.py audit/scripts/p1_labpc_runpy.ps1 FILE=p1_peek.py "ARGS=<path> <mode> [arg]"`.
  - Uses the lab `.venv-analysis` python.
- 21:40 **F-number → evidence mapping launched: 4 subagents** (F001-F027, F028-F054, F055-F081, F082-F109). Instructions: audit/phase1/FMAP_INSTRUCTIONS.md; outputs audit/phase1/fmap/fmap_<batch>.{csv,md}.
- 21:45 **Lab log index** → audit/phase1/labpc/log_index.tsv (242 logs: size, mtime, first/last internal timestamp, lines, fail/done marker counts, scripts named, first/last line; script p1_labpc_logindex.py).
  - 70 logs are UTF-16 (PowerShell `*>` redirect), handled.
  - 203 have internal timestamps; the other 39 are progress-bar/status files.
- 21:55 **audit/papers.csv DONE** (scripts p1_papers_cited.py, p1_papers_csv.py; raw hits audit/phase1/papers_cited_raw.tsv):
  - 11 PDFs present, identified by their own first-page arXiv id; 7 of them have notes.
  - PDFs WITHOUT notes: Kirch 2511.17408, Kumar 2605.27958, Yoo & Skapars 2609.02893, Choi 2606.15821 (the last is cited nowhere).
  - 15 arXiv ids cited in our docs/draft have no PDF, 9 of them cited by the draft: Burns, Zou, Marks & Tegmark, Scheurer, WMDP, NNsight, Chen persona vectors, Natarajan, … Plus 3 draft references with no arXiv id (Azaria & Mitchell, Aletheia blog, Alpaca).
  - 2509.21305 (NOVELTY_ASSESSMENT) not identified.
  - soft-trigger-notes has no paper (code-based).
- **New fact (VERIFIED md5): laptop `run-paper16/vectors_s3/*.pt` (26, mtime 2026-09-15) ≠ lab `run-paper16/vectors_s3_fp32/*.pt` (26), all 26 different.** Sizes are nearly equal (2,623,273 vs 2,622,876 B for default.pt), so this is not simply bf16 vs fp32 by size. The lab has no `vectors_s3` folder, only `vectors_s3_fp32` + `vectors_all_fp32` + `axes_vectors_s3.pt` / `_fp32.pt` / `_fp32_famcentred.pt` + `vectors_s3_old_pc_broker.pt`. → Which vector version fed which result is a Phase 1/2 provenance question (memory note "bf16 mask reductions" hints at a re-extraction; DOC-ONLY).

## RED FLAGS (newest first)
- **INCIDENT (audit's own, 2026-09-23 02:19, logged for honesty): an interrupted tool call kept running on the lab PC.** I launched the RF-14 guard live-test; the owner rejected/interrupted the call; the local ssh died but the remote PowerShell (PID 48232) kept going and started `run_falsify.ps1`. My second attempt then ran concurrently, so the "must SKIP" half of the test was meaningless (the status file had been truncated by the orphan, so there was correctly no DONE line to match).
  - **CORRECTION (2026-09-23 03:40, lead):** "no data harmed" was true when I checked at 02:24, but the forced run has since rewritten exactly **one** of the 100 run-premise files — `falsify/falsify_L40_s13.json` — because that file's original came from a separate 09-21 pilot invocation rather than the main sweep. The other 62 falsify files remain byte-identical to the 20:56 census.
    - The pre-change original is preserved in the Drive backup (`_labpc_only_backup_2026-09-22/expLB/run-premise/falsify/falsify_L40_s13.json`, md5 2edfda4af986238484014ecb31ebba11, 09-21 14:34) and can be restored by the owner if she wants the original state.
    - Impact measured by the P2 agent: 0.0012 in the L40 null and 0.0021 in budget10 for F-95; no verdict moves.
  - The only casualty is `logs/falsify.status`, which was rewritten; the pre-test copy is preserved in tonight's Drive backup (`_labpc_only_backup_2026-09-22/expLB/logs/falsify.status`) and a copy is in audit_scratch.
  - This is exactly the owner's memory note "verify after an interrupted tool call". Rule for the rest of the audit: after any interrupted remote call, check for orphan processes before re-issuing it.
- **RF-13 ORIGIN RESOLVED (VERIFIED by the lead, 2026-09-23): the 09-22 15:40–16:26 wave had no human trigger — the lab PC REBOOTED and boot-triggered tasks fired.**
  - Lab `LastBootUpTime = 2026-09-22 15:34:30`; every task that ran has an `MSFT_TaskBootTrigger`, and boot + its own delay equals its last run to the second: Crossgen-Chain PT5M → 15:40:01, KD-Chain PT6M → 15:41:01, HPKR-Topup/KD-Extract PT12M → 15:47:01, KD-Bank-Crossgen/Premise-Extras PT20M → 15:55:01, Scale-3b/8b/Cap-Controls/Falsify PT30M → 16:05:00, Cap-Ctrl-Layers PT50M → 16:25:01 (audit/scripts/fix_boot_triggers.ps1 output).
  - They show as *Disabled* now because an earlier Claude session disabled them afterwards: transcript 1ac8bb19 line 2758, 2026-09-22T11:31Z = 17:31 lab time, a loop calling `Stop-ScheduledTask` + `Disable-ScheduledTask` over a target list, keeping some enabled on purpose. The whole sequence predates the audit (18:44).
  - **Still armed:** PassAC-11, Night-Chain, PassC-PerToken, KD-Sync-Periodic are Ready with boot triggers. After the RF-14 fix, PassAC-11 will now SKIP instead of recomputing. The owner may want to review the remaining boot triggers.
  - The owner's "I have not run anything in the lab PC after you started" is consistent with all of this.
- **TIME ZONES (VERIFIED, affects every mtime/transcript comparison): Claude transcripts timestamp in UTC (`…Z`); the lab PC and the laptop run at UTC+6.** So a transcript time of 11:31Z is 17:31 lab time. Lab log timestamps and lab file mtimes are both lab-local, so comparisons among them are unaffected.
- **RF-16 (VERIFIED by zero-shot fingerprint; HIGH — bears on the scale trend, draft topic 4) The 70B scale rung mixes two persona banks under one label, contradicting F-100's source line.** Evidence: audit/phase1/bank_fingerprint_scale70b.txt, bank_fingerprint_scale70b_L20.txt; script audit/scripts/p1_bank_fingerprint.py (read-only on the lab). Method: same bank + same activations ⇒ identical zero-shot AUROCs.
  - `run-crossgen/scale_70b/matrices/mean_L40_s42.json` (09-20 23:42, the F-63 cell): its zero-shot contrast/PC1 AUROCs match the **paper16** matrices (kd11/matrices, root matrices, paper16_famcentred; all fp32) within ≤0.0003. They differ from the kd bank by up to 0.077. → paper16 bank (the old axes_vectors_s3.pt from run-paper16, ≈ fp32 per RF-15).
  - `mean_L20_s42.json` and the other 09-22 00:52–01:33 sweep files: zero-shot directions include `knowledge_denial`, and AUROCs are **identical (10/10, |d|=0)** to `kd11/banks/kd` → the **36-persona kd bank** (run-kd/axes_vectors_s3.pt).
  - Both are labelled `bank=axes_vectors_s3.pt`, `bank_name=paper16_70b` (basename collision between run-paper16/ and run-kd/).
  - F-100 (ledger ~l.4605, 4641–4643) says the sweep used "the same bank and datasets as the existing L40 cell" (`run-kd/axes_vectors_s3.pt`). **CONFLICT:** the L40 cell used paper16, not kd.
  - Consequences:
    - F-63's three-rung comparison (70B L40 cell = paper16; 3B/8B = paper16 banks) is bank-consistent.
    - The layer-averaged 70B rung of **F-100 / F-101 / F-102** averages 19 kd-bank layers + 1 paper16 layer, against 3B/8B paper16 rungs → **not like-for-like**.
  - Phase 2: recompute F-100–F-102 separating banks (the paper16 70B sweep does not exist except L40, so a like-for-like layer average may need the kd11 paper16 matrices on the same 5 datasets, if the datasets match — check).
  - CV-02 in audit/phase1/anomalies.md is the anomaly agent's flag; this entry is the lead's resolution.
- **RF-15 (VERIFIED file facts; impact on numbers UNKNOWN → Phase 2 priority) Some findings read the OLD paper16 axes, others the re-extracted fp32 ones.**
  - Old: lab `run-paper16/axes_vectors_s3.pt` (md5 5d05a425…, 2026-09-15) is identical to the laptop copy and to Drive `paper16/axes_vectors_s3.pt` + `paper16/run/axes_vectors_s3.pt`, i.e. built from the OLD vectors (RF-12).
  - New: the re-extraction is `axes_vectors_s3_fp32.pt` (76fee396…, 09-16 02:03) + `axes_vectors_all_fp32.pt`.
  - Who read which (per fmap): F-18 (cmp_banks.py) and F-63's 70B row (matrix field `bank = axes_vectors_s3.pt`) read the OLD file; the kd11 root grids use `_fp32`.
  - DOC-ONLY context: the memory note "bf16 mask reductions" says a bfloat16 bug produced a silent wrong-last-token variant, which suggests the fp32 rebuild was a bug fix (log: lab logs/rebuild_paper16_vectors_s3_fp32.log).
  - Phase 2 must measure old-vs-fp32 differences (cosines between the axes; re-scoring where F-63/draft topic 4 depend on it).
  - **UPDATE 23:05 — MEASURED, mostly benign (VERIFIED, audit/phase1/axes_old_vs_fp32.txt, script p1_axes_old_vs_fp32.py run read-only on the lab):**
    - The four axes (assistant, deception, sycophancy, unified) have old-vs-fp32 cosine 1.0000 at every layer; max relative norm difference 0.5–0.9%.
    - PCA components: max |diff| ≤ 0.009.
    - BUT PC sign pinning differs at a few layers (pca.deception.sign differs at e.g. L23; pca.all/unified.sign at e.g. L65), not at L20/L40/L44.
    - → Old-vs-fp32 cannot move axis-based numbers beyond ~1% in norm. It can flip the sign of individual zero-shot PC projections at those few layers. Phase 2: only zero-shot PC readings at affected layers need a check. The old file header also records it was built on the laptop from `...\run-paper16\vectors_s3`.
- **RF-14 (VERIFIED, code bug in the owner's chain scripts; no effect on per-cell numbers found so far) The "skip if already DONE" guard can never fire in 7 chain scripts.**
  - Affected: cap/scripts/run_cap_controls.ps1, run_cap_layers.ps1, run_sarc_pilot.ps1; kd/scripts/run_passAC_11.ps1; src/assistant_axis_ndif/run_falsify.ps1, run_premise_extras.ps1, run_premise_layers.ps1.
  - Mechanism: `Note` writes "<timestamp>  DONE", but the guard is `Select-String -Pattern '^DONE'` (e.g. run_premise_layers.ps1:35,37,80).
  - So every trigger re-runs the chain and TRUNCATES its .status file (line 40 `Out-File` without -Append). These are exactly the tasks that re-ran in the 09-22 wave (RF-13).
  - Effects seen:
    - per-cell results are usually skipped internally ("exists, skipping");
    - aggregate summary/significance files are rebuilt (BH q-values shift);
    - the gate lines quoted by F-72, F-75, F-77, F-81, F-82, F-83, F-87, F-94, F-95 and F-104 are gone from the .status files.
  - Found by the F055-F081 mapping agent; guard logic confirmed by the lead in the repo copies.
  - Phase 2: check whether the lab copies are identical to the repo copies (F-104 says the scripts were hash-synced 09-22).
- **RF-13 (VERIFIED) A wave of re-runs on 2026-09-22 15:40–16:26 (~3 h before the audit began) REWROTE ~58,000 lab result files after many F-entries citing them were written.**
  - Evidence: audit/phase1/labpc/log_index.tsv, scheduled_tasks.tsv, joined census mtimes.
  - Rewritten:
    - run-crossgen/kd11 41,910; run-crossgen/banks 8,325; run-crossgen/csv 2,772; passB_fine 5,292; passP 399;
    - run-kd/axes_vectors_s3.pt + axes_vectors_s23.pt; run-premise 7; run-cap-ctrl 1; cap/run-sarc-pilot 1.
  - About 17 tasks fired (KD-Chain, Crossgen-Chain, KD-Crossgen, KD-Bank-Crossgen, KD-S23*, PassAC-11, Cap-*, Premise-*, Sarc-Pilot, Falsify-MeanDir, Scale-* [SKIP already DONE], Scale-Coords, HPKR-Topup). Logs say "wrote summary.json, significance.json, N CSVs".
  - Consequences:
    - (1) Lab file mtimes of that window ≠ the run an F-entry was computed from.
    - (2) This probably explains RF-11 (git holds the older versions).
    - (3) Whether the regenerated numbers equal the ones the F-entries quote is a Phase 2 check. Drive `copy` overwrote the older Drive versions too, except where the sync failed.
  - Side facts:
    - `crossgen_full_A.log` ends with "split for harm-pressure-knowledge-report seed 42 differs from run-crossgen\passA\splits.json".
    - rclone logged CRITICAL "Failed to create file system for tamara:expLB/coords/passC/…" (15:47) and "…tamara:expLB/scale: couldn't find root directory ID" (16:20) → these Drive syncs FAILED, which explains part of RF-10.
  - Question for the owner at the end of Phase 1: who or what triggered this re-run wave, and why? Likely the F-104 "fixes never reached the lab PC" redeploy (DOC-ONLY guess).
  - All 4 mapping agents were told at ~21:55.
  - **UPDATE 22:10 — partial reassurance (VERIFIED):** a keyed comparison (records matched on pooling, layer, seed, feature, probe; script audit/scripts/p1_summary_keyed_diff.py; output audit/phase1/summary_keyed_diff_sample.txt) of 3 regenerated files found the git version to be an **exact subset** of the lab version, with 0 differing metric values on the common records:
    - run-crossgen/summary.json: git 66 records (L40, seed 42) vs lab 2,646 (21 layers, seeds 0–4 + 42);
    - banks/own100/summary.json: 66 vs 3,066;
    - kd11/banks/kd_s23/summary.json: 216 (L20/40/44) vs 1,512 (21 layers).
    - So for these, "git ≠ lab" (RF-11) means the lab file GREW; the old numbers are unchanged.
    - significance.json files (bootstrap-based) are not yet compared this way.
    - Rule for Phase 2: compare on keys, never by list position (a naive positional diff wrongly showed thousands of differences).
  - **significance.json (VERIFIED, audit/phase1/significance_keyed_diff_sample.txt, script p1_significance_keyed_diff.py):** tests matched on (block, pooling, layer, probe, comparison). For the 3 files compared, every common test has identical raw statistics (diff, p, CIs, pairs_won); ONLY `dataset_q_bh` (Benjamini–Hochberg q) differs, plus `significant_datasets` in 8 kd_s23 tests.
    - Why: the BH family grew (e.g. root file 192 → 4,788 tests; kd_s23 468 → 5,544; kd11 4,032 → 4,788), so the q-values changed.
    - Max |Δq| = 0.39 (root), 0.12 (kd_s23), 0.04 (kd11).
    - **Consequence for Phase 2:** any F-entry that says a result "clears / does not clear" a BH-corrected threshold must be checked against the version (test family) it was computed on. The verdict can flip between versions.
    - Also check whether the F-entry says which family it corrected over.
- **RF-10 (VERIFIED, data safety — not a validity issue) Several result folders exist ONLY on the lab PC, with no Drive backup.** The owner believes Drive backs everything up. Source: audit/phase1/labpc/joined_labpc.tsv.gz (on_drive=0), summary in audit/phase1/join_summary.txt.
  - Not on Drive at all:
    - `run-premise/` (100 files: pooled-training "premise" results, likely behind F-79–F-86);
    - `run-cap/` (67: responses, scores, vectors_all/s23, control, export);
    - `run-cap-ctrl/` (44);
    - `cap/run-sarc-pilot/` (8) and `cap/run/` (13);
    - `cap/personas` + `cap/questions` + `cap/scripts` (lab copies);
    - `logs/` (242 files, 679 MB);
    - ~133 loose analysis scripts in `C:\Users\user4\expLB\`;
    - `src/assistant_axis_ndif` lab copies (80).
  - Partly missing: `run-scale/3b` (497 files) and `run-scale/8b` (131) — Drive has 1,803 of the lab's 2,431 run-scale files.
  - `run-coords/passC` (76 files, 30 GB) shows as not-on-Drive only because 75 of them are >200 MB and were not hashed on the lab. Drive has 75 coords files that match no hashed lab file, so these are probably the same files (INFERRED; check by size).
  - I will NOT copy anything to Drive (not allowed). Recommendation for the owner: back these up. Result-type files the audit copies into audit/cache/ become a second copy as a side effect.
- **RF-11 (VERIFIED) Git and the lab PC hold DIFFERENT versions of the same result files** (same path, different content, after line-ending normalisation):
  - summary/significance/splits JSON of `run-crossgen/` root and of `run-crossgen/banks/{hf100,own100,paper16_famcentred,paper16_shared}`;
  - `run-crossgen/kd11/{significance,passA/*,banks/kd,banks/kd_s23}` JSONs;
  - `kd/run/factor_probes/kd_{knowledge,utterance}/rows.jsonl`;
  - `kd/run/knowledge/scores.jsonl` (the truncated one, RF-5);
  - `kd/docs/FINDINGS.md` and `cap/docs/DESIGN.md` (older on the lab).
  - → For every F-number, Phase 1/2 must record WHICH copy it was computed from; the laptop/git copy is not automatically the right one. Source: joined_labpc.tsv.gz column asix_same_name_diff_content.
- **RF-12 (VERIFIED) Two versions of the paper16 persona vectors exist.**
  - Old: laptop `run-paper16/vectors_s3/*.pt` = Drive `paper16/run/vectors_s3/*.pt`, 26/26 md5-identical, Drive modtimes 2026-09-15.
  - New: the lab has only `run-paper16/vectors_s3_fp32/*.pt` (all 26 differ from the old) + `vectors_all_fp32` + `axes_vectors_s3.pt` / `_fp32*.pt`.
  - Drive's old `paper16/run/vectors_all` (26) + `vectors_s3` (25) + `axes_vectors_all.pt` match no lab file → the lab replaced them.
  - Which F-numbers used old vs new vectors is a provenance question (memory note "bf16 mask reductions", DOC-ONLY, suggests a re-extraction after a bug).
_Phase 0 draws no conclusions; these are things to CHECK first in Phase 1/2, not verdicts._
- **RF-1 — RESOLVED 2026-09-22 (Phase 1), no problem found.** Evidence: audit/phase1/census_summary_rf1.txt, rf1_instructions.txt, script audit/scripts/p1_rf1_instructions.py.
  - All 36 of his `persona including knowlede denial/*.pt` are byte-identical (md5) to the lab's `C:/Users/user4/expLB/run-kd/vectors_s3/*.pt`. His paper16_L20.{pt,npz} and deceptive_L20.{pt,npz} are byte-identical to the lab's `share_L20_shared/` and `share_deceptive/`; only the two .json sidecars differ.
  - Lab chain dates: rollouts `run-kd/responses` 2026-09-19 20:50 → 09-20 00:18; scores 09-20 00:21–01:32; vectors_s3 09-20 02:28–03:04. His first results came at 09-20 17:53, so the bank already existed.
  - The system prompts stored in all 36 lab rollout files equal the instruction text committed at 3dfdb67 (09-21 11:07). For 9 new personas, 3dfdb67 is the only committed version that matches. `default` matches after the `{model_name}` → "Llama" substitution.
  - So he used the final bank; git lagged the lab run by ~1.5 days. VERIFIED.
- **RF-1 (original wording, kept for history) Which version of the owner's kd persona vectors did the collaborator use? (Owner confirms the bank came from her runs.)** The collaborator's 36-persona "knowledge-denial" bank may predate the instruction set it is named after. His vectors use role `pc_honest_novice`; asix committed that name 17 h AFTER his first results from the bank (3dfdb67, 09-21), in a commit that rewrote all 36 instruction files; he committed the bank 29 h after those results. The draft's headline 0.919 / 0.916 rest on this bank. Fix: md5 his bank vs lab PC run-kd/vectors_s3 (and earlier versions). Source: audit/phase0/branch_compare.md.
- **RF-2 (check) Most of the collaborator's result CSVs cannot be recomputed from git**: they need pooled activations + base-probe run files (detector.pt, results_table.csv) that exist only on his local disk; two banks behind committed results (`persona vectors/vectors_s3`, `persona knowledge denial modified`) are uncommitted. Only loo_scores.npz is recomputable from git. → OWNER DECISION 2026-09-22: do not request his files; check his claims against the data committed on his branch (Level 1), recompute only where git allows.
- **RF-3 (check) Draft vs his own files disagree already at Phase 0 level**: pooled baseline "not compared" vs results/loo_L20, loo_L40 exist; HP-KR 0.103 vs 0.109; identical L40/L44 base-probe numbers; Figs 2–3 have no data/script. Source: audit/phase0/draft_report.md.
- **RF-5 (VERIFIED file fact; impact unchecked) `kd/run/knowledge/scores.jsonl` is committed TRUNCATED** — working copy exactly 204,800 B, 1,095 parseable records + 1 cut mid-record (item kn00876), vs 1,459 items; committed in 3dfdb67 (2026-09-21). Matches the known scp 200-KiB truncation. Must check which analyses read this local copy vs the lab-PC original (F-08/F-09/F-59 knowledge-label findings are candidates).
- **RF-6 (VERIFIED file fact; impact unchecked) `src/assistant_axis_ndif/run_cap_ctrl_layers.ps1` lines 31–32 contain a literal backspace byte (0x08) where `kd11\banks` should be** → the completeness gate for cap_ctrl matrices (mean_L*/last_L*_s42.json) can never match. Check the lab-PC copy and what that gate controlled.
- **RF-7 (CONFLICT, from local-results subagent) Thesis fig2 vs its data**: kd/results/fig_data.json names source kd11/banks/kd_s23/significance.json with 21 layers / 3,276 tests; the local file at that path has 468 tests over 3 layers (local copy likely older/partial — lab PC version to compare). Thesis figs 1, 3, 4 use numbers TYPED into kd/scripts/make_thesis_figures.py (no data file read).
- **RF-8 (NOT-FOUND) `make_view.py` / `make_view_ac.py`, which built the 11th dataset used by every kd11 result, are in no branch and not in the local tree** — but both exist as loose files on the lab PC (C:\Users\user4\expLB\make_view.py 2,253 B, make_view_ac.py 1,915 B; audit/phase0/labpc_top.txt). Many other analysis scripts also exist only as loose files on the lab PC (~81 .py, 46 .ps1).
- **RF-9 (VERIFIED, git) The "append-only" ledger was edited in place**: 7 F-sections had lines replaced after first commit (F-65, F-66, F-67, F-68, F-89, F-93, F-94); e.g. F-68 "+0.051" → "+0.054" in bc701d9 (spot-checked myself). F-01..F-66 entered git in ONE commit (6432257, 2026-09-21 03:26) although dated 09-19..09-21 → their drafting history is not in git. Docs disagree on the source of truth (CLAUDE.md → FINDINGS.md; ONBOARDING → plan file + memory; kd/README → kd/docs/REPORT.md); REPORT.md + memory still say HP-KR lexical 0.870/87% vs F-01's 0.913; "Claim 3", Q-, Stage-, Gate-numbers reused with different meanings. Source: audit/phase0/docs_inventory.md §7–8, fnumber_history.csv.
- **RF-4 (audit risk) The two branches share no identical file in either direction**; every dependency between them passes through files outside git (lab PC / Drive `share/` / his disk). Version-matching must be done by md5 on those copies.

## Done
- [x] Setup (2026-09-22 18:44–18:50 +06): START_STATE.md, .git/info/exclude (+audit/, +CLAUDE.local.md; old copy at audit/.exclude.before), CLAUDE.local.md, .claude/settings.local.json (23 Bash deny rules from the brief + 23 PowerShell mirrors). Deny rules tested live: `git stash list` refused in both Bash and PowerShell.
- [x] `git fetch --all --prune` — no ref changed (remote-tracking refs identical to START_STATE list).
- [x] Branch topology (git, VERIFIED): merge-base of asix and origin/feat/persona-pc-ood = 2e2a785 (2026-07-12, zami456, = tip of origin/copilot/zami456explb). asix +137 commits (Suraiya, 2026-09-07..09-22); pc-ood +34 commits (zami456, 2026-09-04..09-22). pc-ood contains origin/feat/resumable-acts-cache; asix contains main and origin/main.

- [x] Lab PC access (VERIFIED 2026-09-22 18:47): `ssh -o BatchMode=yes user4@100.85.217.68` works passwordless (host DESKTOP-H3ERD3U; Windows; no ~/.ssh/config alias — address from known_hosts + docs/ONBOARDING.md:145). Shared machine (other users' folders on D:\). Folder sizes → audit/phase0/labpc_sizes.csv; main store C:\Users\user4\expLB (run-coords 77.6 GB/2,310 files mostly .pt; run-crossgen 1.9 GB/202,613 files = 191,052 csv + 11,560 json; run-scale, run-kd, run-cap, run-cap-ctrl, run-premise, run-paper16(-shared), run-sycophancy(-mt), run, run-hf275, kd, cap, logs 184 .log). Also D:\expLB-acts (deception-detection, 40 csv), C:\Users\user4\share_deceptive + share_L20_shared (json+npz+pt each), several EMPTY dirs (expLBmask, expLBresults, expLBsrc, C:\Users\user4\run-crossgen\kd11, D:\user4\activations).
- [x] Lab PC activity (VERIFIED, audit/phase0/labpc_tasks.txt, labpc_taskscripts.txt, labpc_rcloneverbs.txt): 5 project scheduled tasks still fire every 15–30 min (Coords-Supervisor, Crossgen-Sync-Periodic, KD-Sync-Periodic, Night-Chain, PassC-PerToken). Their logs say every compute chain is DONE (.done markers present); they now only run `rclone copy` / `rclone check` to Drive (copy never deletes). All other project tasks Disabled. No python process running at 18:54. → INFERRED: lab-PC results are currently static, but NOT frozen (tasks could relaunch compute if a .done marker vanished).
- [x] rclone (VERIFIED): one remote `tamara:` (type drive, scope drive.file); everything under `tamara:expLB/` (17 folders: _selftest coords crossgen figures figures-crossgen hf275 kd own100 paper16 paper16-shared scale share staging sycophancy sycophancy_mt thesis-docs thinking). Sizes → audit/phase0/rclone_sizes.tsv (coords 77.55 GB/2,307 files ≈ lab run-coords; crossgen 1.91 GB/202,614 ≈ lab run-crossgen).
- [x] Papers location: papers/ (11 PDFs, tracked); notes docs/papers/*-notes.md (8). 4 PDFs have no notes file (Pressure-Testing…, Probe Generalization as Subspace Selection…, Impact of Off-Policy Training Data, Truth Stays in the Family) — Phase 1 detail.
- [x] Docs on lab PC vs local (VERIFIED by md5, audit/phase0/labpc_docs.csv + labpc_docs_vs_local.txt, script audit/scripts/p0_compare_labpc_docs.py): 142 doc files; 18 identical to local; 2 DIFFERENT — lab kd/docs/FINDINGS.md = the committed version at bc701d9 (2026-09-21 04:01, stale by ~2 days/≥40 findings), lab cap/docs/DESIGN.md older too. Lab-PC/Drive-ONLY docs (not in git): thinking/plan.md (109,305 B, 2026-09-17; also on Drive tamara:expLB/thinking/), thinking/plans/*.md (4 Claude plan-mode files, Drive only), thinking/memory/* (a 2026-09-17 snapshot of the auto memory, 23 notes), thinking/ONBOARDING.md (matches no committed version), logs/crossgen_final_report.txt (2026-09-22 15:50), logs/crossgen_verify.txt, logs/drive_check*. D:\expLB-acts\deception-detection is a vendored-code copy (84 docs are its data/prompt files).
- [x] Drive tamara:expLB/thesis-docs/persona-coordinates-70b-results.md is byte-identical to docs/thesis/persona-coordinates-70b-results.md @ b9a497e (VERIFIED md5 a8cb33d7…). tamara:expLB/share/ holds deceptive/, paper16_L20/, paper16_L20_sharedquestions/ + paper16_persona_vectors_L20.zip — same folder names as on the collaborator's branch (INFERRED: this is how he received inputs; blob-level check pending).
- [x] Draft report identified (subagent; audit/phase0/draft_report.md, draft_claims_raw.csv [237 rows], draft_text.txt, draft_pages/*.png): `full paper draft for this experiments/main (1).pdf` (asix, cf11749, 19 pp, no author names; INFERRED his: acknowledgements + result sections match zami456 commits; compiled 72 min after his last commit). Title "Lying in Knowledge-Denier Persona Coordinates: Cross-Dataset Lie Detection at 70B". 5 figures, 11 tables, ~11 experiments, 0 F-numbers cited. Leads for Phase 2 (DOC-ONLY until checked): draft says pooled baseline "not compared" but his branch has results/loo_L20, loo_L40 (CONFLICT doc vs files); HP-KR original-bank 0.103 (draft) vs 0.109 (his persona_ood_ext_L20/summary.md:116); base-probe BA/recall identical at L40 and L44 (0.626/0.281); Figs 2–3 have no script/data on either branch; base-probe result tables for XXXX-9-llama-all(-L40/-L44) not committed (plots only); persona_ood_mod_L* not mentioned; owner's briefing/F-85/F-88 critique an EARLIER draft ("One Direction Generalizes") that is NOT-FOUND.
- [x] Branch comparison (subagent; audit/phase0/branch_compare.md, pcood_results_index.csv [22 rows], scripts/p0_branch_compare.py): his results = 19 folders under src/probes/deception-detection/results/ (421 files, 13.7 MB: 316 csv, 87 png, 15 md, 2 npz, 1 yaml) + 3 input folders (43 files, 97 MB) + 20 .py; ~8 experiment types × layers 20/40/44; ~1,000 numbers in his 17 .md + ~380 in commit messages; no Git LFS anywhere. Owner→him: his paper16_L20.pt / deceptive_L20.pt carry `source_run: run-paper16-shared` (asix script b9a497e, 6 h earlier); his 36-persona bank matches asix kd role names/format (INFERRED = run-kd/vectors_s3, not in git). Him→owner: asix ed8daca ported his cache/batcher (modified); asix 4d0d046 cites his files @ f386a72. He corrected 3 claims after pushing (e3f2258, f386a72, 45b16d9). asix never touched src/probes/deception-detection/ since the branch point. Submodule pins identical on all branches (asix/main add assistant_axis @ a989619). Third contributor: Mahdi <mahdi.ndc21@gmail.com>, origin/feat/llama-layer30, 1 config commit, no results.
- [x] Local results (subagent; audit/phase0/local_results.md, local_results_index.csv [183 rows], scripts_to_outputs.csv [187 scripts], local_figures_provenance.csv, local_files_flagged.csv): 9,897 files / 5.88 GB after exclusions (9,410 tracked, 487 ignored = all .pt + logs); csv 7,801, json 496, pt 453, png 309, pdf 200, jsonl 74, log 14; no npy/npz/parquet, no LFS. Original Liars' Bench ≈4.3 GB; persona work ≈1.07 GB (run-crossgen 7,150 files incl. kd11 6,596; run-hf275; run-paper16; run-coords only 41 json + 3 items.pt; kd/run; run-cap/export; cap/run-sarcasm-natural). ~26 persona output folders with a producing script (~40 units counting run-coords analysis scripts); 23 analysis scripts print only (no file). Missing locally: run-coords/passA/B/C, run-kd, run-scale, run-premise, run-cap-ctrl, kd11 cap_* banks, most driver logs. 782 byte-identical duplicate groups (mostly expected crossgen layout). Spot-verified RF-5 and RF-6 myself.
- [x] Docs + ledger history (subagent; audit/phase0/docs_inventory.md, fnumber_history.csv [109 rows], docs_inventory_raw.tsv, memory_inventory.tsv): 31 tracked findings files on asix (25 distinct docs) + 9 skills + 8 paper-notes + 1 root CLAUDE.md; memory dir 45 files; 6 plan files in ~/.claude/plans/ (harmonic-cuddling-kay.md defines Q1–Q11 and is byte-identical to lab/Drive thinking/plan.md — VERIFIED md5 ab721b1b…). 105 nnsight library docs = one commit 47e53c5, no project terms. F-112 / F-150 RESOLVED: no F-110+ exists anywhere (all 14 refs, memory, working tree); "F-112" occurs only in BRIEF.md; "F-150" = Ford F-150 in an Alpaca rollout. Newest docs: FINDINGS.md + docs/COLLABORATOR_BRIEFING.md (43ee6d3).
- [x] Drive result-type size (VERIFIED rclone size): 272,325 files / 3.29 GB non-.pt; 4,752 .pt / 83.6 GB.
- [x] audit/PROJECT_MAP.md and audit/PLAN.md written (final for Phase 0).
- [x] PHASE 0 COMPLETE. End-of-phase git check (2026-09-22): branch asix, HEAD 43ee6d3 (= START_STATE), `git status --porcelain=v1 -uall` EMPTY (= expected baseline after Setup; empty porcelain also means no submodule moved or became dirty). An extra `find -newer` scan for stray writes outside audit/ was KILLED by Claude Code for low system memory before it printed anything — not re-run (owner to decide).

## F-mapping results (Phase 1) — audit/phase1/fmap/
- **Non-interference check (VERIFIED 21:45):** audit/phase1/labpc_changed_since_audit_start.txt, script p1_labpc_changed_since.ps1. Since 18:40, the only lab files that changed are 9 logs written by the owner's own sync tasks (coords/crossgen/kd/night rclone + supervisor). No owner file was touched by the audit.
- **Batch F001-F027** (fmap_F001-F027.{csv,md}): 10 MAPPED, 17 PARTIAL, 0 NOT-FOUND. Copy status: 14 LAB-ONLY (mostly with an identical Drive copy; logs have none), 7 LAB=GIT, 5 LAB≠GIT (F-04, F-08, F-09, F-14, F-17), 1 GIT-ONLY (F-23). Key flags (agent's; to be confirmed in Phase 2):
  - F-04's exact source (the 09-19 20:10 run, 384 tests) was overwritten the same evening. The diff and CI reproduce; the stated q = 0.931 does not (git/lab give 0.963/0.972). The test sits in block '9' (10 datasets) although the text says 11.
  - F-18's cmp_banks.py read the OLD paper16 axes (laptop vectors_s3, RF-12), while the kd11 runs use axes_vectors_s3_fp32.pt.
  - F-20's 98-row 89/9 state no longer exists (HP-KR top-up + kr-matched rebuild on 09-21/22); only kd_passAC.log survives.
  - F-27 is a 21-of-26-roles snapshot (now 25 scored).
  - Print-only numbers (need recompute): F-01, F-02, F-06, F-08, F-09, F-10, F-12, F-18, F-20.
  - Missing entirely: F-21's smoke3b sweep, F-15's "a third of 33 labels". ~~F-17's +0.056/−0.44/−0.76~~ — **WITHDRAWN: the verification agent recomputed all three exactly from run-crossgen/matrices/mean_L40_s42.json (in git).**
  - F-08/F-09 must use the lab/Drive scores.jsonl, not the truncated git one.
  - F-02's cells sum to 6,943 (text: 7,068).
- **Batch F082-F109** (fmap_F082-F109.{csv,md}): 17 MAPPED, 11 PARTIAL, 0 NOT-FOUND. 25/28 not in git (13 with an identical Drive copy, 12 LAB-ONLY: run-premise, cap/run-sarc-pilot, logs); 3 GIT-ONLY (F-88, F-97, F-106). No LAB≠GIT. Key flags (agent's):
  - F-96, F-98–F-103, F-107, F-108 were computed by inline commands (no script or output kept); their inputs survive.
  - F-85, F-87–F-95 come from 25 loose lab scripts in no branch and not on Drive.
  - Re-run wave: F-83/F-84 files have no older copy (the log shows the same numbers); run-kd axes md5 = Drive 09-19/20 copies (unchanged); status files quoted by F-82/83/87/94/95/104 were overwritten.
  - ~~F-95: 55/63 in the text vs 51/63 in the re-run falsify.status.~~ **WITHDRAWN by the lead 2026-09-23 — my misread.** The status file's "51/63 cells clear the best null by >0.02" counts only the stricter margin, and it separately logs "4/12 cells BELOW L16 clear"; 51 (all cells at L>=16) + 4 = the ledger's 55/63. The two numbers measure different things and agree. (Caught by the session-mining agent; re-checked by the lead against the gate lines in audit/phase1/labpc/ and the F-95 section.)
  - F-91: the L36 value matches seed 7, not seed 42.
  - F-86 omits the n=4 dip.
  - F-90's persona arm used run/axes.pt = the own100 bank (INFERRED), not paper16.
  - F-105's "not persisted" is now out of date.

- **Batch F028-F054** (fmap_F028-F054.{csv,md}): 18 MAPPED, 8 PARTIAL (F-28, F-32, F-33, F-34, F-36, F-40, F-45, F-47), 1 NOT-FOUND (F-44). Copy status: 16 LAB-ONLY, 3 LAB=GIT (F-29, F-38, F-39), 2 LAB≠GIT (F-31, F-34), 1 GIT-ONLY (F-32), 3 NOT-FOUND, 2 mixed. 12 of 27 are engineering notes. Key flags (agent's, with lead checks):
  - **F-39:** the agent re-ran the committed kd/scripts/paper_comparison.py on the matrix (identical in git and lab). The stated numbers (raw 0.8090, pc:unified@2 +0.0262, datapca@2 −0.1120) reproduce ONLY with MASK in the 5-dataset set. With HPC instead of mask the persona gain is +0.128 (> the paper's +0.10).
    - **Lead check (VERIFIED, papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf p.4 table + p.5–6 figures):** the paper's deception datasets ARE ConvGame, InstrDec, Mask, AILiar, Roleplay. So F-39's set matches the paper (and F-60's correction). The entries that used HPC (F-45, F-48, F-54 per the agent) are the inconsistent ones.
    - Remaining caveat for Phase 2: F-39's "smaller effect" is sensitive to that one dataset choice. The result also feeds thesis fig3.
  - **F-31:** every stated number (q 0.499, "6 of 4,032") matches the GIT copy. The lab copy was regenerated 09-22 (6,300 tests, 65 significant); diffs and CIs are identical, only the BH q moved → cite the git copy (consistent with the RF-13 note).
  - **F-34:** its 3,276-test significance file exists nowhere (git 468 tests/3 layers; lab and Drive 5,544). The 21-layer profile checks out in the lab copy; the q-values do not (stated 0.130 vs lab 0.103 vs git 0.074). Same file as RF-7 / thesis fig2.
  - **F-37:** reports the unfiltered vectors_all block of each axes log. For 3B/8B the s23 run overwrote the same axes file (3B cos 0.551 on disk vs 0.435 reported).
  - **F-36:** the v2 question draft was edited after the entry (saved: 10/10 split, first-person; text: 12+8, third-person).
  - **F-30:** the 70B column matches run-paper16 (paper questions), not the shared-question bank.
  - **Inputs gone:** F-28/F-32 cap persona files were rewritten 09-20 22:51 (F-32's text survives only in git cap/docs/REVIEW_PACKET*.md). F-40, F-44, F-45, F-47: evidence deleted by later restarts; only log lines remain.
  - **RF-10 again:** F-51/F-52/F-53 rollouts (cap/run/smoke_*) and every log this batch cites are lab-only.

- **Batch F055-F081** (fmap_F055-F081.{csv,md}): 16 MAPPED, 11 PARTIAL. Copy status: 18 LAB-ONLY, 6 LAB=GIT, 3 GIT-ONLY (F-60, F-70, F-78). Preliminary values at the mapped keys match the ledger for F-62, F-63, F-65, F-68, F-72–F-75, F-79 (spot: F-80/81). Key flags (agent's):
  - F-58's 3B/8B matrices were replaced by the F-63 rerun; its numbers survive only in crossgen_scale_{3b,8b}.log.
  - F-67's 792-test and F-57's 21-layer 09-20 significance files are NOT-FOUND. In the regenerated cap_s23 file the q-values moved (0.227 vs 0.172; 0.0163 vs 0.013); verdicts unchanged.
  - F-61's "+0.006" is pc:unified − datapca (+0.0064), not − raw (+0.0093).
  - F-57: fig_data CI lower bound > 0 at L40/L44 vs the text "spans zero at every layer"; the fig3 described is in no committed script.
  - F-75 says 31/80 layers vs F-77/F-81 37/80.
  - The lab log prints cos(PC1, assistant) = +0.838; F-65 says −0.838.
  - The gate lines quoted by F-72/75/77/81 were wiped (RF-14).
- **ALL 109 MAPPED/merged → audit/phase1/fmap_all.csv** (same 20 columns). Totals:
  - mapping_status: 61 MAPPED, 47 PARTIAL, 1 NOT-FOUND;
  - copy_status: 74 LAB-ONLY, 17 LAB=GIT, 7 LAB≠GIT, 8 GIT-ONLY, 3 NOT-FOUND;
  - claim_type: 67 scientific, 24 engineering, 11 correction, 7 provenance;
  - script_writes_file: 66 y, 42 n (print-only).
- 22:45 Launched 2 inventory agents (instructions audit/phase1/INVENTORY_INSTRUCTIONS.md): Task A → audit/experiments.csv + phase1/experiments_notes.md; Task B → phase1/anomalies.{md,csv} (orphans, ghosts, duplicates, conflicting versions). papers.csv already done.

- **audit/experiments.csv DONE** (114 rows, 18 cols; notes audit/phase1/experiments_notes.md; builders audit/scripts/p1_exp_{defs,build,counts}.py; support audit/phase1/exp_work/, incl. 40 lab .status files dumped read-only):
  - Rows: 2 background, 86 owner, 1 unknown-owner lab folder, 5 print-only analysis families, 20 collaborator rows (19 folders + the INLP ghost).
  - Status: 106 complete, 1 crashed (O40 crossgen retrain 09-17, killed at L12), 2 partial (O05 own100 fp32 rebuild 1/100; O56 kd passAC C-root step failed), 1 superseded (O03 original paper16 vectors), 4 unknown.
  - 106/109 F-ids map to ≥1 experiment (F-70, F-78, F-88 are paper/doc provenance).
  - Candidate orphans (no F, no doc): 14, incl. O44 crossgen passA (92,184 files, 80 layers × 6 seeds × 8 banks), O45 passC (6,936), O46 passB_fine + passP (5,777), O26 length-control JSONs, O87 LAYER22 activation cache.
  - 46 rows have no F-number, only prose. This includes the whole 09-15..09-17 Q1–Q11 series, written up only in docs/thesis/persona-coordinates-70b-results.md + memory.
  - Early (09-15) analyses record `bank: axes_vectors_s3.pt` (old); from 09-16 `_fp32`. Per RF-15's measurement the axes are practically identical, so the impact is expected to be negligible except PC signs at a few layers.
  - The collaborator's exporter scripts exist as lab loose files: C:/Users/user4/build_share_L20.py, build_deceptive_L20.py (Phase 0 NOT-FOUND → now FOUND, lab-only).
  - Wave details: kd11 root + kd bank rewritten files are byte-identical to git except significance.json. passAC11 (O58) and premise-extras (O80) outputs have no older copy. The 09-22 passA crossgen re-run FAILED (split mismatch) → passA results are the 09-18/19 originals.
  - Single copies: run-paper16/scores_405b (25) + regroup_hf.json exist only in git/laptop. No Drive copy: famcentred axes, all 100 run-premise files, scale alpaca/mask activations (628/629).
  - Sarcasm-pilot stripped AUROC 0.947 (02:29 status) vs 0.921 (15:55) — different subsets.
  - The cap_ctrl layer gate reports failure because of RF-6, though all 42 matrices exist.

- **audit/phase1/anomalies.{md,csv} DONE** (65 rows; support audit/phase1/anomalies_detail/; scripts audit/scripts/p1_anom_*; 4 files uploaded to the lab scratch, logged):
  - **Orphans:** four complete crossgen grids with no reported result (only a coverage line in F-91 + memory): root passA (92,184 files), passC (6,936), passB_fine + passP (5,777; passP = the planned prompt-side construct-validity control).
    - HIGH doc gap: the fp32 paper16 bank (axes_vectors_s3_fp32.pt) behind every kd11/root matrix is named in no doc.
    - Length-control JSONs: lab-only driver.
  - **Ghosts:** docs cite 532 paths; 496 found; 16 real ghosts (his uncommitted activations/banks, make_numbers.py, smoke3b, sarcasm scratch files, hf_axis_check.py); 20 text artefacts. HIGH ghost: the 3,276-test kd_s23/significance.json (F-34, thesis fig2). F-04's 384-test and F-67's 792-test versions are also gone.
  - **Duplicates:** 144,971 multi-copy contents, nearly all expected (lab/Drive/git). Surprises:
    - kd11/passA and kd11/passC significance files are byte-identical to the root passA/passC ones → the 09-20 "kd11" grids ran on 10 datasets, not 11 (check against F-72/F-75/F-76).
    - 11 own100 roles scored 3 on all 1,200 rollouts (possible judge ceiling).
  - **Conflicts:** 50 by logical path. 27 git-vs-lab (the 26 RF-11 files + FINDINGS.md), 13 line-endings only, 2 real lab-vs-Drive (older own100 axes.pt). The join's driveDIFF column is unreliable (matches the same file name across different banks); don't use it.
  - CV-01 = RF-12/15; CV-02 → resolved by the lead as RF-16. MED: own100 banks were never moved to fp32 (affects F-90 and paper16-vs-own100); the root passA grid and passA11 rest on different HP-KR row sets.
  - Data safety: also lab-only are axes_vectors_s3_fp32_famcentred.pt, run/axes_strict.pt, data_syco, isarc_a.parquet, the LAYER22 cache. The 75 big passC files match Drive by path + size → the passC part of RF-10 is settled (backed up).

## Session mining (owner-requested) — part 1 of 2 back
- **audit/phase1/sessions/**: sessions_index.tsv, session_experiments.csv, session_events_all.csv, dataset_events.csv, run_events2_raw.tsv, SESSIONS_NOTES.md; scripts audit/scripts/p1_sess_*.py.
- Scanned 17 of 19 transcripts (2026-09-09→09-22): 40,814 records, 5,405 tool calls. 5 are `/clear` stubs, 1 is a subset fork → **11 substantive sessions**, no sidechains.
- **359 test/run events** (54 NEW vs audit/experiments.csv, 305 matched, 9 dup) and **254 dataset events** (66 builder runs, 6 file writes, 182 stated decisions: 115 edit / 69 build / 38 regenerate / 20 top-up / 6 filter / 6 discard). Lab 215, laptop 144.
- Leads worth Phase 2 (all DOC-ONLY until recomputed):
  - **RF-16's mechanism captured verbatim:** the 09-22 00:51 sweep was launched from a hand-written `task2.sh` retry loop with `--bank_name paper16_70b --bank …run-kdxes_vectors_s3.pt`, into the same out_root as the 09-20 L40 paper16 cell. The mislabel is in that loop, not in 27_crossgen.py. (Independently confirms the lead's fingerprint result; see audit/fix/RF16_NOTE.md.)
  - HP-KR lexical 0.870 → 0.913 is a ROW-SET change (all models → Llama-only), not a re-measurement → resolves the RF-9 doc conflict.
  - The random-direction null (which gates several zero-shot claims) exists only in a console log from `factcheck.py` (09-16): 95th pct |AUROC| 0.640 at one layer, 0.683 best-of-21; paper16 sycophancy axis L40 0.313 [0.283, 0.346].
  - Two caveats that qualify headlines: dropping the six position-holding personas kills the cap advantage (pc:cap_only@2 − raw = −0.008, p=0.56); `make_cap_page.py` itself printed pc:unified − raw +0.0757, CI [−0.006, +0.173], q=0.499, "clears? no".
  - Unreported negatives: a source-independence result measured then retracted in-session; persona-count peaks at n=4 (0.863), not n=7; premise cluster-permutation p=0.667; and **"L40 was picked because it maximised the persona advantage"** stated outright → a layer-selection issue that touches every L40 headline.
  - Dataset history: kd questions rebalanced hours before the bank ran; role renamed pc_candid_unknowing → pc_honest_novice (independently dates RF-1); paper16 regenerated on a shared 10+10 question set after PC1 turned out to be "which question set" (cos −0.98); the 11th dataset built by make_view.py as a free subset of passB (258 rows, 124/134) and deliberately NOT built for passA/C (98 rows, 89/9) — which explains the 10-dataset passA/C grids; cap questions v1→v2→v3 in one day; the s23 filter was chosen because score-3 filtering discards ~30% vs 1.1%.

## Session mining — part 2 of 2 (print-only numbers and unrecorded findings)
- **audit/phase1/sessions/**: findings_printed_numbers.csv (144 rows), findings_unrecorded.csv (35), findings_rerun_origin.md, FINDINGS_NOTES.md; scripts audit/scripts/p1_sessf_*.py (incl. p1_sessf_verify.py, a mechanical re-check of every row: 135 OK / 6 check-artefacts / 3 skip).
- **72 of 72 print-only F-numbers recovered** with their console output, most with the producing command verbatim. Only F-44 is unrecovered (it is also the only NOT-FOUND in fmap). matches_ledger: 113 y, 24 partial (each with a named discrepancy), 4 not-y.
- Only 35 unrecorded findings: 94.7% of numbers in assistant conclusions and 86.3% of all printed numbers already appear in some doc. The gap is controls and provenance, not headlines — axis norms not comparable across banks (sycophancy 1.535 vs deception 1.0037; kd 1.707–4.342), an 80-layer sweep where axes beat raw at only 28/80 and datapca at 31/80, a `random` control column in every scale table that no F-entry quotes, per-dataset zero-shot nulls (the deception axis beats the random p95 on only 5 of 9 datasets at L40), and unrecorded validation work.
- **New discrepancies for Phase 2 (DOC-ONLY leads, in the agent's priority order):**
  1. **F-77/F-81's "37/80 (46%)" is not an observed count.** The printed counts are 28/80 and 31/80, and 0.46 is F-76's bootstrap MEDIAN FRACTION — a bootstrap median written up as a layer count. This resolves the F-75 vs F-77/F-81 conflict in F-75's favour.
  2. F-24 says 40.22 GB; the final measurement was 56.64 GB and was retracted in-session.
  3. F-91 mixes seeds (L36 from seed 7; L32/L40 from seed 42; both known at write time).
  4. F-39 and F-58 used DIFFERENT "paper fives"; F-39's command names its five, closing the fmap CONFLICT (consistent with the lead's PDF check).
  5. F-58 → F-63 re-run 78 min apart: 3B gain +0.089 → +0.020, 8B +0.086 → +0.013; the matrices were overwritten, so F-58 survives only in the transcript.
  6. F-61's "50 per role" is one file (the print says 54; 26x50 != 1,400); its paper16 and kd sides differ in pair count and baseline.
  7. F-86's +0.0284 margin sits INSIDE its comparator's own 0.0824 subsampling spread, which the script printed and the ledger softened to a ±0.02 sd.
  8. Two cap "variants" are the same contrast (+0.9313, norm 4.411); identical-role PCA bases differ by 1.347.
  9. F-21 mixes 3B and 8B deployments; F-02's missing 125 rows = two dropped table cells (57+68); F-20's destroyed 89:9 state survives in the transcript; F-65's ±0.838 sign conflict dissolves (all-roles vs capitulation PCA, same run).
- Re-run origin, part (a): the 00:52–01:33 70B sweep WAS owner-requested work in session 1ac8bb19 ("all 3, step by step", 00:40 local); its launch command is the primary-source proof of RF-16. Part (b) is the reboot (see RF-13 above); why the lab rebooted is NOT-FOUND.
- Also found: Drive sync was dead 12:18–17:26 on 09-22 (expired OAuth) and then cleared a 68,666-file backlog; a previous session wrote rclone.conf to the lab PC 78 min before the audit baseline.
- Privacy note: the agent kept the flattened transcript corpus OUT of audit/ (it is rebuildable in ~1 min) so the repo holds no second copy of the owner's private session history. Good call; keep that rule.

## Owner items closed 2026-09-23 (after the fix round)
- **F-110 appended to kd/docs/FINDINGS.md** (owner said yes): the RF-16 correction, in the ledger's style, marked as written by the audit, with the like-for-like table, sources and caveats. `git diff --ignore-cr-at-eol` = 53 added lines and nothing else; the full-file churn in a plain `git diff` is a pre-existing CRLF-vs-LF checkout artefact (the HEAD blob is LF, the working file is CRLF) and disappears on commit.
- **The 8 script fixes stay uncommitted** — the owner will commit them.
- **Completed lab tasks disabled** (owner: "are they already completed? if yes then stop them"). Verified complete first, then disabled: PassAC-11 (status DONE 16:05:43), and PassC-PerToken / Night-Chain / Coords-Supervisor (run-coords passA/B/C `.done` present, so they exit early and only synced).
  - **Kept enabled on purpose:** KD-Sync-Periodic and Crossgen-Sync-Periodic — pure `rclone copy` to Drive; they cannot re-run an experiment, and Crossgen-Sync-Periodic is what backs up the new run-crossgen/scale_70b_paper16fp32 folder.
  - Every other project task is now Disabled or has no trigger (manual only).
  - **Flagged, not touched:** `Thesis_Train` is Ready with a boot trigger and runs run_train.ps1 under D:/user4fina/code (last run 2026-08-04). It belongs to the owner's older thesis work, outside this project — her call.
  - Unrelated: two python processes on the shared lab PC belong to another team (under C:/Users/user5, SYCON-Bench), not to this project.

## Mini independent check of Phase 1 (the gate before Phase 2)
Sample: audit/phase1/verify/sample.csv — 8 random scientific F-entries + 2 high-importance (F-100, F-65) + 4 experiments rows + 3 anomalies + 3 session recoveries, drawn with seed 42 (audit/scripts/p2_sample_for_verify.py). Agents re-derived each item from raw files with no access to the Phase-1 tables.
- **Batch A (10 ledger items): 9 CONFIRMED, 1 DIFFERS.** Outputs audit/phase1/verify/verify_A.{csv,md}.
  - **F-04 DIFFERS:** effect sizes reproduce (+0.0147, CI [−0.1701, +0.1872]; +0.0264) but the counts and both q-values do not, in ANY surviving copy — stated 384 tests vs git 4,032 / lab 4,788; stated "1 significant" vs **0** everywhere; q 0.931 → 0.963/0.973 and 0.855 → 0.896/0.906. Its numbers also sit in block '9' (10 datasets), not 11.
  - **A Phase-1 error found:** PROGRESS.md's F-17 flag ("+0.056 / −0.44 / −0.76 missing entirely") is WRONG — they recompute exactly (+0.0555, −0.4378, corr −0.7565) from run-crossgen/matrices/mean_L40_s42.json, which is in git. Flag withdrawn.
  - F-100 CONFIRMED (17/17 figures exact) with one mis-attributed parenthetical: "axes loses to raw at L44–68 (−0.004…−0.029)" is the best−raw row; axes−raw is −0.022…−0.106. It independently reproduced RF-16 (L40 has 2 zero-shot directions, L44 has 10 including knowledge_denial).
  - F-65 CONFIRMED (0.745 vs 0.374 and 26 sub-figures exact) except "every feature agrees to ≤0.004": pc:cap_only@2 differs by 0.0142 — the very feature its caveat (c) rests on.
  - Exact: F-62 (9/9 cells), F-109 (9 + 5 derived), F-83 (8/8), F-07 (3), F-55 (8/8 holds, 8/8 folds), F-09 (312/1,147; 119/1,340).
  - Source-line errors (the number is right, the cited file is not): F-17, F-55, F-09 (only the LAB copy reproduces; the committed one is the RF-5 truncation), F-07 (kd/results/phase2_gate.json is in no branch).
  - Unlocated: F-83's bootstrap CI (+0.143…+0.213) is in no premise JSON; F-100 never defines "best over two features" (+0.0835 vs +0.0791 if pc:unified is allowed).
  - **Phase-2 trap:** block '9' = 10 datasets and block '10' = 11 in kd11, but block '10' = 10 in the run-crossgen root.
  - Notes on the audit's own footprint: the ledger now contains the audit-written F-110 (so Phase 0's "no F-110+" is stale, by design), and the RF-16 recompute wrote into the owner's result tree (run-crossgen/scale_70b_paper16fp32) rather than audit_scratch — deliberate and owner-authorised, so her Drive sync backs it up; logged in audit/LABPC_WRITES.md.

## PHASE 2 — started 2026-09-23
- Plan: **audit/PHASE2_PLAN.md** (two levels, per-claim card, batches of 10 with rolling QA, W1/W2/W3/W4 workstreams). Batch instructions: audit/phase2/CLAIM_INSTRUCTIONS.md.
- **Claim inventory built: audit/claims_inventory.csv — 390 rows** (110 ledger + 237 draft + 8 abstract + 35 session-only). Still to add: his 17 result notes and the owner's other docs (thesis docs, briefing, roadmap, REPORT.md, memory Q1–Q11).
- **W1 layer sensitivity DONE → audit/phase2/layer_sensitivity.csv (78,516 rows)**: off-diagonal and diagonal means per store × pooling × layer × seed × feature × probe, for ALL datasets and for the paper's five (script p2_w1_layer_sensitivity.py, run read-only on the lab). 22 stores, 25 features, 80 layers.
  - First use: for kd11 (paper16 fp32), paper's five, mean pooling, probe=paper, `axes − raw` peaks at **L24 (+0.055)** and L20 (+0.051), is **+0.013 at L40**, and is negative below L16 and above L48. So for this store the session quote "L40 was picked because it maximised the persona advantage" does NOT hold — L40 is mid-range. Each layer-specific claim can now be checked the same way.
- **W2 provenance table → audit/phase2/provenance.csv**: store → bank field, bank_name, k. Six stores carry a bank BASENAME that exists in more than one folder (own100 `axes.pt`; paper16_shared and kd `axes_vectors_s3.pt`; cap_s23 and cap_ctrl `axes_s23.pt`; scale_70b `axes_vectors_s3.pt`) — the exact trap behind RF-16. scale_70b is already resolved (40 matrices kd, 2 paper16); the rest are flagged for fingerprinting before any claim relies on them.

- **Batch B (4 experiments + 3 anomalies + 3 session recoveries): 6 CONFIRMED, 1 CONFIRMED-DIFFERENT-SOURCE, 3 DIFFERS.** Outputs audit/phase1/verify/verify_B.{csv,md}; 33 more lab files cached (all md5 MATCH).
  - **F-58 (HIGH, ledger error).** The 70B row reproduces to 4 dp; 3B/8B do not (files 0.7619 / 0.7980 vs ledger 0.708 / 0.743; gains +0.033 / +0.013 vs +0.089 / +0.086). Cause, from logs/crossgen_scale_{3b,8b}.log: the run was done TWICE over different 5-dataset sets writing the SAME filenames — first with harm-pressure-choice, then 78 min 9 s later with mask. Drive holds the post-mask copies, so the version behind the ledger exists nowhere (NOT-RECOMPUTABLE). F-58's own caveat cites the first set while its 70B row matches the second → **the published three-rung table is not like-for-like**.
  - **F-61 (ledger self-contradiction).** paper16 is not uniform: 16 roles × 50 rows/10 questions + 10 roles × 60/12, 22 distinct questions, total 1,400 (26×50 = 1,300 ≠ its own total). F-18 states it correctly in the same ledger. Also F-61's "different questions, not a superset" fails: 20 of the 22 paper16 questions are among the kd 30.
  - **AN4 (Phase-1 error):** 44 loose root-level .py in the 09-19..09-21 window, **37 print-only**, not ~25.
  - **F-22 right, wrong line:** the cited line is the fix's comment; the bug was live at line 146 of 107879d/106d09f. NEW: the sibling handler at line 186 of the same file still uses str(e)[:200].
  - Exactly confirmed: CV-07, OR-04 (252 matrices = 21 layers × 6 seeds × 2 poolings, and no doc names a seed), OR-05, O72 (2,600 rollouts), O12 (5,372 files), C08 (bank 36/36 md5-identical to run-kd/vectors_s3, ext_axis1 0.9041 at L20).
  - Also: the thesis doc (line 492/1060) wrongly says the length-control results are "log only; no result JSON holds them" — the JSONs exist; and the length file says the positive class is longer in **8 of 10** datasets, not the 9 of 10 in ONBOARDING and memory.
- **GATE RESULT: 20 items, 15 CONFIRMED + 1 confirmed-different-source + 4 DIFFERS. Only 2 of the 4 are Phase-1 errors (the F-17 flag, the AN4 count); the other 2 are ledger errors the audit was meant to find (F-04, F-58, F-61 count as ledger).** → Phase 1's tables are good enough to build on; per-batch QA stays in force for Phase 2.
- **BASELINE MOVED (note for future sessions): START_STATE.md is no longer the comparison point.** The owner committed d280d23 "afteer audit" (2026-09-23 02:09) containing the 8 script fixes and F-110. The only uncommitted change now is one amended line in kd/docs/FINDINGS.md (~line 5254: F-110's definition of "best"). New baseline: HEAD = d280d23, working tree = that one line. Also created during the audit, by authorisation: lab folder run-crossgen/scale_70b_paper16fp32 (2026-09-23 00:16).

### Phase 2 wave 1 (launched 2026-09-23 ~03:00) — 50 claims in 5 batches
| batch | claims | why first |
|---|---|---|
| P1-hpkr | F-01, F-02, F-03, F-05, F-06, F-08, F-09, F-16, F-23, F-72 | the draft's headline is HP-KR at 0.919 |
| P2-pooled | F-79, F-80, F-81, F-83, F-85, F-86, F-94, F-95, F-96, F-109 | the owner's strongest challenge: the draft says a pooled baseline was "not compared" |
| P3-kdbank-cap | F-13, F-31, F-34, F-65, F-66, F-67, F-89, F-97, F-98, F-99 | register/capitulation + who the abstract's number belongs to |
| P4-scale | F-27, F-30, F-37, F-54, F-58, F-63, F-100, F-101, F-102, F-110 | draft topic 4; includes checking the audit's own F-110 |
| P5-draft-headlines | the draft's ~18 headline rows | Level 1 against his committed data (his activations are gone) |
Each batch writes audit/phase2/claims_batch_<name>.csv + audit/claims/<id>.md + scripts in audit/scripts/claims/. Rolling QA: the lead re-derives 2 of every 10 before accepting a batch.

### Wave 1 results (accepted after lead QA)
**Tally after P1 + P5 (+ partial P2/P3): 40 of 692 claims checked** (audit/claims.csv; merge script p2_merge_claims.py). Verdicts so far: 17 HOLDS, 15 MATCHES-HIS-DATA, 8 PARTLY; confidence HIGH on 38.

**P1-hpkr — 8 HOLDS, 2 PARTLY, 0 fail.** The HP-KR series is in good shape: F-01 (0.9125), F-03 (all 13 numbers), F-05/F-06 (recomputed end-to-end from lab activations, including two print-only values), F-08 (8/8), F-16 (7/7), F-23 (against the paper PDF) reproduce exactly.
- **F-02 PARTLY:** the four quoted counts are exact, but the 2×2 table omits two non-empty cells — (knows=F, denies=F, honest)=57 and (knows=T, denies=F, **deceptive**)=68 — so it sums to 6,943 of 7,068. Its "within knows=True a pure lexical probe scores 1.0" is false: 0.9758 all rows / 0.9460 Llama. The matched-slice half is exactly true.
- **F-72 provenance correction (HIGH, touches the draft):** its table splices two banks — the `axes` column is kd_s23 while F-03's headline 0.752 is paper16, and the entry never says so. Like-for-like paper16 column measured; the conclusion survives but middle layers differ by up to 0.198. Also "+0.033" is passC11's margin; F-03's own store gives +0.037, so the title's "ten times" and the body's "eleven times" disagree. NEW: the 80-layer passA11 sweep finished — span 0.365 → 0.409, and axes beats raw at only 28 of 80 layers (datapca 31/80). F-72 strengthens, F-03 weakens.
- **F-08/F-09 depend entirely on the lab copy:** against the truncated committed scores.jsonl (RF-5) nothing matches (0.817 not 0.786; 200 not 312). F-59 reads the same file → DOC_FIXES item.
- Doc errors: F-01's "68 deceptive rows it does not catch" is the superseded all-rows figure (Llama rows: 15); F-05 omits a fourth pre-committed gate cell (d_knowledge on knowledge-matched, 0.357/0.615/0.660 vs ~0.5 predicted); F-16's "items with a consistent belief: 1,962" conflates the total with 1,841 (**lead QA: confirmed from the raw rows — 1,962 total, 1,841 consistent**).

**P5-draft-headlines — all 18 headline rows, Level 1 exhaustive (every cell of his Tables 2, 3, 4, 6, 7 reproduces from his branch; 0 mismatches beyond two rounding cases).** 13 MATCHES HIS DATA, 4 PARTLY, 1 part-untested.
- **The abstract mixes two directions (lead QA: CONFIRMED from his zero_shot_operating.csv).** "0.919 AUROC and 0.910 balanced accuracy" belong to the **4v6** knowledge-denial axis; the **12v13** extended axis — the subject of the transfer headline — scores balanced accuracy **0.500 with recall 0.000 on HP-KR at L20, L40 and L44**, i.e. it flags nothing at the benchmark threshold. Only his Table 2 disambiguates.
- D-099 PARTLY: "matches or beats the published contrast" holds on the mean-over-training-sets basis (18/18) but not on the zero-shot basis (12v13 below 8v7 in 3 of 18 cells, all at L20); and Table 7 as printed has no published-contrast column, so the pointer cannot be checked where it points.
- **Two Phase-0 red flags RESOLVED in the draft's favour:** the draft's 0.103 is correct (his own summary.md:116 carries the wrong number — 0.109 is the base probe's value; **lead QA: confirmed, auroc_axis = 0.10286**); and the identical L40/L44 base-probe 0.626/0.281 is not a copied file (per-dataset values differ; they agree only at 3 dp because 4 of 6 datasets flag nothing at both layers).
- **The labels moved between versions:** his 2026-09-21 abstract attaches 0.882/0.912 to the 4v6 contrast, but those are the 12v13 values (4v6 = 0.890/0.919). The draft labels both correctly; nothing was re-run. One abstract sentence ("PC1 0.09 below the contrast") matches no file and is absent from the draft.
- **Margin decomposition (for Phase 3):** at L40 raw 0.732 → published 8v7 0.832 → extended 0.916, so **46% of the margin over raw comes from the 10 knowledge-denial roles** (99% at L20) — and those vectors are byte-identical to the owner's run-kd/vectors_s3, while the acknowledgements assign the bank to a placeholder name.
- Structural: for a 1-d feature the trained probe learns only a sign, so his "mean off-diagonal AUROC" equals the mean zero-shot AUROC (0.9158 both ways, verified) — the 0.916-vs-0.732 headline compares a fixed direction against probes fitted per training set, with no matched low-dimensional control (the very control the owner found removes the gain). D-028's 0.220 gap at L20 collapses to 0.0086 once HP-KR is removed.
- External numbers verified in the Liars' Bench PDF (Mean Probe 0.12/0.50, Follow-Up 0.14/0.26, Upper-Bound 0.97/0.95) and the persona-coordinates PDF (persona quotations verbatim; 8+7+5+6 = 26; 8+4 v 7+3+3 = 12v13).

**P3-kdbank-cap — 8 HOLDS, 2 PARTLY, 0 failures**, all HIGH confidence, everything Level-2 recomputed (column/row means from the 11x11 per-cell matrices; both bootstraps reimplemented independently).
- **F-97 / F-98 / F-99 HOLD exactly** — all 32+18+16 numbers re-derived, at his f386a72 and at his current head (files byte-identical). **Lead QA: confirmed** — his ext_axis1 off-diagonal mean is 0.9041 (L20) and 0.9158 (L40), i.e. exactly the abstract's 0.904/0.916, so the attribution to the kd-augmented 12v13 axis is right. Two citation slips only: ~~pool_activations.py:84 is really line 123~~ **[WITHDRAWN 2026-09-23 — the audit was wrong, see `audit/claims/F-97.md`: at `f386a72`, the commit F-97 cites, the file is 156 lines and line 84 IS the mean-pool line. The "slip" was read off the current 182-line file. F-97's citation is correct.]**; F-98's L20 premise share is 76.6%, not 76.5%.
- **F-65 PARTLY:** 26/27 figures exact, but "every feature agrees to ≤0.004" is false — pc:cap_only@2 is 0.5935 (s23) vs 0.6077 (all) = 0.0142, the very feature its caveat (c) rests on (the cap_only penalty is −0.120 under s23 vs −0.105 under all). F-62's "~0.01" is the honest bound.
- **F-67 PARTLY (new):** row 6 of its table (pc:capitulation@2 − random@2, minus-sycophancy block) is marked "clears? YES" but its BH q is 0.068 in the reconstructed 792-test family and 0.075 now, and the file's own significant_datasets is false. Row 2 is correctly written. Conclusion unchanged; one row overstated.
- **RF-13 version work paid off:** both vanished test families were rebuilt from current files and reproduce the entries exactly (F-34's 3,276-test family → q=0.130 at L44 and the exact 9-clearing/5-negative set; F-67's 792-test family → q=0.013 and 0.172). Raw statistics unchanged, only q moved. **But F-31's "Nothing beats raw" is now FALSE on the regenerated lab file** (6,300 tests): pc:kd_assertion − raw at L68 reads +0.1497, q=0.047, clears. Its "6 of 4,032, four negative" is true only of the git-copy family.
- **Two robustness results the entries never ran:** F-66's bootstrap clears at 14 of 21 mean-pooled layers (15/21 vs datapca@2) — not a single-layer artifact — but at only 3 layers under last pooling, so it is a mean-pooling result. F-65's trained margin peaks at L32 (+0.139), not L40. F-13 survives 21 layers x 2 poolings at 21/21.
- **F-89 provenance flag:** run-cap-ctrl/axes_s23.pt has mtime 2026-09-22 16:05 (inside the RF-13 wave) while the cap_ctrl matrices built from it are 09-21 20:07 — so F-89's cosines come from a file rebuilt AFTER the results that used it. Bank-name collision resolved by md5, not name (**lead QA: confirmed** — run-cap 96a9b42b… vs run-cap-ctrl 326310336d…).

**P2-pooled — 6 HOLDS (F-79, F-80, F-83, F-95, F-96, F-109), 4 PARTLY (F-81, F-85, F-86, F-94), 0 fail, 0 NO RAW DATA.** The owner's pooled-training challenge to the draft is solid: every headline number reproduces exactly (0.6763 → 0.8545, +0.178; row-matched 0.8500; 11/11 targets; 21/21 layers; 5 seeds +0.173–0.198), with all bootstrap intervals recomputed independently rather than read. run-premise (100 files) + 6 logs cached and md5-verified.
- **F-83's "unlocatable" CI resolved:** +0.143 to +0.213 is in no file — it is F-80's console-only bootstrap output; reimplementing it gives [+0.14295, +0.21291], and all four of F-80's comparisons reproduce digit-for-digit.
- **F-86 is the weak link (confirmed):** the +0.0284 "beats joint training" margin sits inside its comparator's own 0.0824 subsampling spread (sd 0.0200) — `30f_result2_audit.py` printed exactly that sentence and the entry omits it; the stored dataset bootstrap clears with a lower bound of only +0.0045. Also the ledger uses **0.8062 in F-86 and 0.8500 in F-79/F-80 for the SAME quantity** (pooled LODO row-budget matched, same layer/pooling/probe/seed) — against 0.8500 the margin is +0.033, not +0.077. And its source-count table omits k=4 = 0.7178, which is below k=3 and breaks the "rising and flattening" description.
- **F-94:** all measurements exact (including the retention formula), but three prose counts are wrong — content clears at 3 of 16 (its own table says 3, the prose says 4); presence at 10 of 16 (prose says 9); and "ordering holds at 15 of 16" reproduces under no reading (10/16 strict, 12/16 ignoring paraphrase, 16/16 as groups). Inversions are ≤0.011, so the conclusion survives.
- **F-81:** "11/11 targets won at 13 of 21 layers" is 12, and "the rest all early or very late" is wrong — L32, the peak layer, is one of the nine exceptions.
- **F-85 (new):** the dataset bootstrap it never ran shows that at **L20 — the layer the entry calls the claim's strongest — the pooled-vs-single margin does NOT clear** (+0.093, CI [−0.045, +0.210], P=0.089); L32 and L40 do.
- **F-96 holds fully**, but its caveat (1) is now stale: scale_70b holds 21 k=3 layers, not one → the decline is +0.120 → +0.103 → +0.055 like-for-like (RF-16 repair store) and L40 ranks 13th of 21.

**P4-scale — 2 HOLDS (F-30, F-110), 7 PARTLY, 1 REVISED→F-63 (F-58). No outright failure, but six entries carry a failing sub-claim.** Everything re-derived at Level 2 by a script independent of the audit's own.
- **F-110 (the audit's own entry) HOLDS** — all four rows reproduced independently (3B +0.0328/+0.1197, 8B +0.0089/+0.1027, 70B-kd +0.0079/+0.0751, 70B-paper16 **−0.0131 / +0.0541**); bank identity confirmed from the fields inside the files. Two precision faults found and **FIXED by the lead in the ledger** (verified first from layer_sensitivity.csv): its `last`-pooling pair was the 21-layer mean (+0.0069/+0.0681) quoted beside a 19-layer table (19L = +0.0061/+0.0688), and the corrected decline is **mean-pooling only** — under `last` the rungs read +0.0745 (3B) / +0.0090 (8B) / +0.0688 (70B), i.e. non-monotone.
- **F-58 CONFIRMED NOT-RECOMPUTABLE; F-63 does NOT inherit it.** The harm-pressure-choice run (09-20 22:18–22:21) was overwritten by the mask run (23:38–23:40) under the same filenames; no copy in git on either branch, none on Drive. F-58's 70B row verifies exactly; F-63's cells are the mask matrices and reproduce to 4 dp.
- **New defects:** (1) **F-27's "70B 1.07% (16/1,500)" is in no file** — run-paper16/scores gives 12/1,350 = 0.89%, so the title's "2.7x" is really 3.2x; the same literal is **hardcoded in scale/scripts/score_summary.py** and contaminates every rung report it prints. (2) **F-100** attaches "−0.004 to −0.029" to `axes − raw` at L44–68, but that is the `pc:deception − raw` range (axes − raw is −0.022 to −0.106), and "pc:deception holds up" is wrong — it loses to raw there too. (3) **F-101's "L40 is second-worst of 21 under last"** is a kd-bank artifact: like-for-like it is 14/21, mid-pack. (4) **F-102's "under last the 70B is the highest of the three"** becomes 3B > 70B > 8B like-for-like (its conclusion survives). (5) **F-37's 3B/8B columns are log-only** — both extractions wrote the same axes_{3b,8b}.pt, so the surviving files are the s23 ones (0.551/0.739, not 0.435/0.732); its "differs only in the unified basis" caveat is contradicted. (6) **F-54 repeats F-58's dataset-identity error** ("these five ARE the paper's Table 1 … HarmPressureChoice") — F-60 corrected it only against F-58, so it is still live at F-54.
- **F-63 robustness (confirms the memory note "no margin from one layer"):** its 3B cell is rank **1 of 7** stored layers (the maximum), 8B 4/8, 70B L40 13/21.
- F-30 is the one entry where every figure recomputes exactly with nothing left over.
- Also noted: run-scale/{3b,8b}/coords now hold 7 dataset dirs (alpaca + mask added), not the 5 F-54 reports.

**WAVE 1 COMPLETE: 58 claims checked** (audit/claims.csv). All five batches passed lead QA (2 items re-derived per batch).

### Wave 2 results
**P8-draft-rest — 40 rows: 26 MATCHES HIS DATA, 8 HOLDS (checked against the PDFs), 5 PARTLY, 1 DOES NOT MATCH.** ~90 individual values across his §5.1–5.5, §6 and appendices C–G reproduce from his committed CSVs; where his cell-level matrices allowed it the agent re-derived the aggregate rather than reading it (all 10 Table-10 cells, the 0.983/0.732/0.251 triple, the CG→HP-KR 0.06 minimum, the three PC-vs-contrast gaps).
- **D-237 DOES NOT MATCH:** `paper/make_numbers.py` exists on neither branch — a reproducibility promise the repo cannot keep (independently re-confirmed).
- **D-233 (Appendix G) half-verifiable:** the follow-up claim is exact (all 20 cells equal his persona_ood_followup_L20 store to 4 dp), but the three response-token runs committed **only plots, no results_table.csv**, so "matched for each layer" can be neither confirmed nor refuted. Worth asking him for those three files.
- Three wording overstatements: D-157 "sits AT its 0.5 floor in 18 of 30" is his script's tol=0.005 count (exactly 0.5 in 14); D-124 "inverts on one dataset at each depth" — at L20 the base probe is below 0.5 on two (HP-KR 0.109, HP-C 0.471); D-207 "the top-10% read is best" is contradicted by his own Table 9 (at L40 the single maximum wins on both BA 0.750 vs 0.737 and recall 0.548 vs 0.531).
- **Citations to other papers are accurate** (verified in the PDFs): Liars' Bench 72,863 / 7 datasets / 4 models; HP-KR 0.12/0.14/0.97 and 0.50/0.26/0.95; the published five-dataset means re-average exactly from LB Tables 6–7; the scratchpad ablation 0.93→0.97 is LB Table 17; Kumar's "0.61–0.80" is his own wording; the three persona quotes are verbatim from M&S Table 2 and all 8 deceptive roles are omission-type. **One exception, D-013/D-037:** "every method has recall near zero" on HP-KR is false for the Follow-Up Probe on Llama-3.3-70B (0.37, bought with an 84% FPR); it holds only of the cross-model averages.
- **NEW for Phase 3 (lead QA: CONFIRMED):** his branch carries `persona_ood_mod_L{20,40,44}` — the same 36 personas re-extracted to a uniform 142–150 rollouts, i.e. the clean rollout-count control — whose numbers are LOWER (ext_axis1 off-diagonal **0.9066 vs the reported 0.9158** at L40), and his own summary there is titled "More rollouts per persona did not help". The draft reports the `ext` bank only and never mentions this arm.
- **LEAD QA OVERTURNED one P8 finding.** The agent reported Table 3's random control as "a single draw, no seed, no spread". **False:** `persona_ood.py` averages `N_RANDOM_DRAWS = 5` seeded draws (rng = default_rng(SEED=42)), with the comment "averaged over draws so a single lucky basis cannot carry the comparison", and it has done so since the script's first version (2731cb0, 09-18). The narrower fair criticism stands: he reports the MEAN of 5 draws, not a null distribution with a percentile, so it bounds the centre and not the tail — the owner's own many-draw null does more. The claim row was corrected accordingly.
- Deferred by the agent (available on request): D-080/100/126, the qualitative §6 rows, App-A method rows, interpretive rows, and nine other-paper rows.

### Wave 3 results (P6, P7) — accepted after lead QA

**P6-kd-rest — 10 rows: 9 HOLDS, 1 PARTLY.**
- **F-04 is SOLVED — the vanished 384-test family is found.** It is `run-crossgen/kd11/significance.json` restricted to layers {40, 44}: 16 comparisons x 3 probes x 2 poolings x 2 layers = **192 per block, 2 blocks = 384**, BH applied *within each block* (the file's own `note` field says so). **Lead QA re-derived this from scratch and it reproduces to 4 dp:** block 9 gives `axes - raw` **+0.0147, q 0.9311** and `axes - datapca` **+0.0264, q 0.8546** — F-04's exact quoted numbers — with **exactly one significant test** in the family.
- **Two corrections owed on F-04** (both confirmed by lead QA):
  1. That single significant test is **`random@2 - raw`, last pooling, L44, mms, diff -0.2287, q 0.0192** — a *random projection doing worse than raw*, not a persona win. Any reading of "1 of 192 significant" as support for the persona method is backwards.
  2. F-04's entry calls it an **"11-dataset run"**, but the quoted values are block **'9'**, and `src/assistant_axis_ndif/27_crossgen.py:51,781` defines block 9 as `[d for d in DATASETS if d != "sycophancy_mt"]` — i.e. **10 of the 11 datasets**. On the full 11 (block '10') the signs flip: **-0.0215 / -0.0243**. The labels "10"/"9" are historical dataset counts from before the 11th dataset was added.
  - **Doc trap for every later reading:** in the kd11 era "block 9" means *sycophancy excluded*, never "9 datasets". Blocks '10' and '9' hold 11 and 10 datasets respectively.
- F-17's Phase-1 flag is **WITHDRAWN** (second withdrawal of one of my own flags): +0.0555 / -0.4378 / -0.7565 all reproduce, and `pc:unified@2 - datapca@2` is negative at 18 of 21 layers.
- RF-15 is **CLOSED for F-18**: old vs fp32 bank cosine +1.000, no effect on the number.
- F-20 **PARTLY** — the 98-row / 89:9 state is NO RAW DATA (the file no longer exists in that form) but is corroborated twice in independent sources.

**P7-register-sarcasm — 10 rows: 7 HOLDS, 3 PARTLY.** All five producing scripts are print-only, so every headline here is a Level-2 recompute with no Level-1 artifact to check against.
- **F-75 vs F-77/F-81 settled, and F-75 is the right one. Lead QA re-derived this independently on the lab PC with its own loader (`audit/scripts/qa_p7.py`): the kd matched slice beats raw at 28/80 layers and `datapca` at 31/80** — exactly what F-75 says. **F-77 (`kd/docs/FINDINGS.md:2934`) and F-81 (`:3233`) both print `37/80` in the row that pairs with the cap bank's genuine 21/21.** 37 is not the layer count under any reading; it is a **bootstrap median *fraction* of positive layers rendered as a count** (the batch measured 0.4625 x 80 = 37.0; my own bootstrap gives 0.4750 x 80 = 38.0 — implementation-sensitive, so treat the *provenance* as the likely origin and the *correction* as certain). **The error runs against F-77's own argument**, so fixing it strengthens the kd-vs-cap contrast rather than weakening it. Correction owed in two places.
- My QA bootstrap also lands F-76 within noise (L40 margin +0.0656, CI [-0.0738, +0.1904], P = 0.1646 vs F-76's [-0.0693, +0.1887], P = 0.161; argmax interval [L5, L79]; P(argmax = L40) = 0.0000) — F-76's failure verdict is robust to the implementation.
- **F-73 PARTLY.** Every AUROC is exact (length 0.5555 vs raw 0.6623, datapca 0.6651, random 0.5849 over 110 cells). Two internal errors: its "0.553 vs 0.665 confirmed to three decimals" is false (0.556/0.662 — its own closing line gets this right), and harm-pressure-choice's gap is +0.0035, not "exactly zero".
- **A separate documentation bug, in five documents plus a memory note: "the positive class is longer in 9 of 10 datasets".** Lead QA recomputed the gaps from `run-coords/passB/<dataset>/rows.jsonl` directly: it is **9 of 11** with `sycophancy_mt` and **8 of 10** without. **"9 of 10" matches neither slicing.** For DOC_FIXES.
- **Lead QA addition, not in the batch:** F-73's "median gap on the eight datasets a reviewer could see is 3.7 tokens" is a median of **|gap|** (`audit/scripts/claims/P7_lab.py:71`), which folds `roleplaying`'s -4.10 in as +4.10. The **signed** median over the same eight is **+2.63**. Since F-73's point is that the length effect is *small*, the absolute-value framing works against it. Also: `roleplaying` has **84 rows with `label: null`** that every reader of this store silently drops.
- **F-93's missing CI now exists** (the caveat is discharged, the verdict hardens): bootstrapping the 20 question ids on out-of-fold scores gives 0.9213, 95% **[0.872, 0.967]**, P(<= the 0.65 pre-registered ceiling) = **0.0000**, and an independent sklearn pipeline matches the repo's scorer to 4 dp on all four rows.
- **F-88's Phase-1 dating flag is RESOLVED, not a problem:** the three PDFs' mtimes are 17:20/17:23/17:24, 19-23 minutes *before* F-88's 17:43 commit; the 9-hour gap was a delayed `git add`. Every quotation checks out verbatim. **Not verified:** the venues "GEM @ ACL 2026" and "Findings of ACL 2026" appear nowhere in the PDFs.
- **F-105 PARTLY** — its numbers and polarity withdrawal are exact, but "none of them is persisted" is out of date (the persistence fix landed in the same commit, 6c9a5cf), and "all ten words appear in the top-25" is 11 of 12, with `definitely` being a MARKER_TOKEN — so F-93's feature list came from the **unstripped** fit.
- **RF-13 does not bite this batch:** no claim here quotes a BH q-value, every bootstrap was computed in-script from matrices, and the matrices read were lab originals (passA11 09-21 12:20-13:28, untouched by the 09-22 re-run).
- **Data-safety side effect:** `cap/run-sarc-pilot/` was lab-only with no Drive backup (RF-10). All 8 files (440 KB) copied into `audit/cache/labpc/cap/run-sarc-pilot/`, md5 MATCH, logged in the manifest.
- Flagged, not chased: F-76's owed cluster-permutation test may be `run-premise/cluster_perm_kd_matched.json` (09-22 16:07); none of F-88's three PDFs has a notes file under `docs/papers/`.

**Running tally after wave 3: 118 of 692 claims checked — 48 HOLDS, 41 MATCHES HIS DATA, 27 PARTLY, 1 REVISED, 1 DOES NOT MATCH.** Confidence HIGH on 116 of 118.

### Wave 4 — P10 and P11 in (P9, P12, P13 still running), accepted after lead QA

**P11-abstract — 11 rows: 1 HOLDS, 6 MATCHES HIS DATA, 4 PARTLY.** Every stated number in the
collaborator's abstract except one reproduces exactly from his committed CSVs. **The failures are
attribution and inference, not arithmetic** — which is precisely the owner's stated worry.
- **The abstract's "0.09 below the contrast" cannot be sourced [lead QA].** The comparison it
  describes is `ext_pc1` 0.7680 vs `ext_axis1` 0.9158 at L40 = **0.148**. I scanned every pc-vs-axis
  pairing in his own L40 summary: **no gap of 0.09 ± 0.005 exists.**
- **"The principal components do not contribute" is contradicted by his own files for the published
  bank [lead QA]:** `pc1 − axis1` = **+0.0158 / +0.0274 / +0.0246** at L20/L40/L44 — PC1 is *ahead*.
  The claim holds only for the kd-**extended** bank. My own refinement: the L20 win **reverses to
  −0.0033** when HP-KR is excluded, so its direction there rests on one dataset.
  **The owner's ledger reaches the same conclusion by a route that does not fail** — F-17, F-31,
  F-96/F-100/F-110 test the PCs against a *matched dataset-PCA control* rather than against the contrast.
- **The abstract's "lifts transfer by 0.18–0.21" is a sign artifact [lead QA].** In
  `matrix_axis1.csv` the row for the probe trained on HP-KR is **wholly sign-flipped** — row mean
  **0.2091** against 0.7906–0.7908 for all five other rows (a 1-d feature gives the same AUROC in
  every row *up to sign*, so the flip is unmistakable). That row drags `axis1`'s column means down
  and manufactures the lift. Replacing it with `1 − x` gives **+0.0293 / +0.0020 / −0.0048 /
  −0.0027**, matching his own zero-shot table (+0.0292 / +0.0020 / −0.0048 / −0.0028) — and
  zero-shot is the protocol the *same sentence's* 0.882/0.912 use. **On the four unrelated datasets
  the extended personas add essentially nothing; the real gain is HP-KR (+0.6159).**
- **Attribution confirmed and extended.** 0.904/0.916 are `ext_axis1`; the paper-faithful 8v7 reads
  0.7909/0.8319 zero-shot on the same rows, so **44.2% / 45.7% of the margin is the owner's bank**.
  A-11's "four feigned-ignorance vs six candour personas" is **exactly** her
  `kd/personas/kd_role_groups.json` (set-compared True on both poles), and his 36 `.pt` are
  byte-identical to her `run-kd/vectors_s3` (RF-1). **Neither sentence names the bank**, and his
  "six candour" pole **merges** her deliberately separate `sincere_ignorance` and `plain_assertion`
  groups — reintroducing the wording confound that split exists to remove.
- Two more: "six Liars' Bench datasets" is **five** (`repe_honesty` is not one; his own file says
  `n_datasets=5`), and L44 was run but omitted; "the **independently constructed** sycophancy axis"
  is false against his own code — `persona_ood.py:911` fits it on the **same bank object** with the
  paper's own 5v5 contrast, and his comment records |cos| = 0.54.

**P10-collab-notes — 10 rows: 4 MATCHES HIS DATA, 6 PARTLY, 0 fail.** **Every digit he quotes is
exact to ≤5e-4.** All six PARTLY verdicts are the same failure mode batch P8 found: the quantifier
around the number, never the number.
- **A genuine Level 2 on his work was possible here**: `loo_scores.npz` commits raw held-out scores
  *and* labels, so all 10 arm×weighting×6-dataset cells were recomputed at both depths with an
  independent rank-based Mann-Whitney — **every row of `loo_summary.csv` reproduces to 4 dp.**
- **"Within 0.03 of the privileged upper bound" is 0.0310 [lead QA]** (0.8990023 vs 0.93); 0.0432 at
  L20. And on balanced accuracy at 1% FPR the same pair is **0.694 vs 0.890** — a 0.196 gap the
  sentence omits.
- **"The pooled baseline inverts to 0.124 (L20) and 0.523 (L40)" [lead QA]:** the two numbers use
  **different weightings** of his own table — 0.1243 is `balanced`, 0.5230 is `naive`. **0.523 is
  above chance, so it is not an inversion**; holding weighting fixed at `balanced` gives 0.4887 at L40.
- His three published comparators are **correctly transcribed** (Tables 6/7/8 re-extracted from the
  Liars' Bench PDF: 0.682/0.636, 0.634/0.578, 0.930/0.890 all match).
- Provenance gap worth knowing: the `base probe (RepE-trained)` and `Average (Liars' Bench)` rows
  were committed only at the branch tip `45b16d9` (09-22), a day *after* the note (`fd17728`,
  09-21); the originating run's `results_table.csv` is **not committed** — only its plots.
- Found outside the batch: `loo_L20/summary.md:46-48` claims "no other arm gets within **0.05** at
  L40", but `ext_pc3|naive` is within **0.0458** — contradicted by his own `loo_summary.csv`.

**DOC_FIXES now carries 30 confirmed corrections (DF-01…DF-30), 14 of them lead-QA'd twice.**

### Wave 4, P12 — and the audit's widest-travelling error

**P12-briefing — 10 rows: 9 HOLDS, 1 PARTLY.** `docs/COLLABORATOR_BRIEFING.md` is the document the
owner sends her collaborator to correct his draft, so an error in it carries into his paper. Nine of
its ten checked claims reproduce exactly from raw files. The tenth does not, and it is big.

**O-001 — 0.374 is the *deception* axis's PC1, not the sycophancy axis. [lead QA, three-way]**
The briefing (`:74-78`) says "The paper's **sycophancy** axis … **It scores 0.374** on our
capitulation dataset." I verified this three independent ways and all three agree it is wrong:
1. **The code:** `27_crossgen.py:411-416` sets `_zs_axis = "deception"` whenever the bank has a
   `deception` axis — paper16 does — so the zero-shot rows named bare `contrast`/`PC1` **are the
   deception axis**; every other axis is written `contrast:<name>`.
2. **The file:** `run-crossgen/kd11/matrices/mean_L40_s42.json`, `sycophancy_mt` column —
   `contrast` **0.4089**, `PC1` **0.3739**. The quoted 0.374 is the deception axis's PC1.
3. **The owner's own earlier measurement:** memory note `q10-sycophancy-axis-inverted`, measured
   2026-09-17 with a different script (`24_q10_sycophancy_axis.py`), records **sycophancy axis
   0.313** [0.283, 0.346] and **deception axis 0.409** — matching the `contrast` row to 4 dp.
   Batch P12's independent re-measure on `passA11`/`passC11` gives 0.3061 / 0.2996.
**The correction strengthens the argument** (same shape as the F-77 fix): in the owner's own
1,000-random-direction null the **deception axis at L40 sits *within* random**, while the
**sycophancy axis is *beyond* it at p 0.001–0.008 — significantly inverted**. So "worse than a random
direction" is *not* supported for the number quoted, and *is* supported for the axis named.
**The error has travelled into nine documents** (briefing, `cap/docs/DESIGN.md` ×2, four REVIEW_PACKET
versions, the sarcasm write-up, the kd skill) plus six more that call it "the register axis", plus
`CLAUDE.md:59` and `docs/TODO.md:36` — and batch P12 reports it is **inherited from F-65**, so the
ledger entry needs it too. All enumerated in DOC_FIXES **DF-31**.
- Second protocol mix in the same sentence: the "(0.541)" it loses to is a **trained** random probe,
  not a zero-shot random direction.

**Good news on the neighbouring family:** the briefing carries **no** layers-positive count for the kd
matched slice, so the F-77/F-81 "37/80" error has **not** propagated into it. Its only such counts —
the cap bank's 21/21 and the collab sweep's 21/21 + 20/21 — were re-derived and are correct.

**Seven more briefing lines need correcting before it is sent** (DF-32…DF-38): "positive on 8–11 of
11" mixes denominators (measured 10/11, 11/11, 10/11 over eleven; 9/9, 9/9, 8/9 over the clean nine —
nothing is 8 of 11); "~666 rows" is the *cap*, the realised pool is 412–500 (mean 449); its p = 0.38
is a **one-sided** tail, two-sided gives +0.0045, CI [−0.024, +0.033], p 0.744; a wrong script path;
the drop-`repe_honesty` margins use a 5-dataset baseline; "both of your arms lose by 0.064/0.055"
describes only the extended arm (`axis1` loses by 0.146/0.122); "differ by 0.0004" holds for
`pc:capitulation@2` but `pc:cap_only@2` differs by 0.0142.

**Everything else in the briefing reproduced exactly**, including the LOCO gains, F-84's p = 0.38/B = 200,
`0.9041173922099752` / `0.9158352734669751` with |zero-shot − trained| = 2.2e-16 / 0.0, and the
21/21 and 20/21 counts.

**Version flag:** `run-premise/cluster_perm_kd_matched.json` (O-005) has mtime 2026-09-22 16:07, inside
the RF-13 boot-trigger window, while F-84 is dated 09-21 — the artifact read is a regenerated copy with
no pre-wave comparison available. **Ledger gap:** the last-token half of O-011/O-012 has no F-entry, so
the briefing is currently its only written record.

**Tally after P10, P11, P12: 149 of 692 claims checked — 58 HOLDS, 51 MATCHES HIS DATA, 38 PARTLY,
1 REVISED, 1 DOES NOT MATCH.** DOC_FIXES now carries **38 confirmed corrections**, 17 lead-QA'd twice.
P9 (ledger remainder) and P13 (session-only findings) are still running.

### Wave 4 complete — P9 and P13

**P13-session — 10 rows: 7 HOLDS, 3 PARTLY, 0 NO RAW DATA.** This is the batch that answers the
owner's central request. **Every one of the ten transcript-only leads had a surviving artefact or was
recomputable — nothing was unrecoverable.** All three PARTLY verdicts share one pattern: the numbers
reproduce exactly, but the session's "this is in no document" premise was wrong (S-03's anchor is in
F-11, S-13's inverted-cell table is in `docs/thesis/…:1403-1414`, S-14's per-dataset null grid is in
the same file's §8.2).

**Now backed by raw data AND absent from both the ledger and the draft — the owner's list:**
- **S-01/S-02** — the paper16 axes are bit-exact **unnormalised difference-of-means** vectors
  (`max|diff| = 0.000e+00`), with norms at L20 of 1.0120 / 1.0037 / **1.5350** / 0.9722: the
  sycophancy axis is **53% longer at every one of 80 layers**, so raw projections onto them are not
  comparable. **These are the vectors that were handed to the collaborator to dot-product.**
- **S-11** — dropping his in-distribution `repe_honesty` column, the paper's 8v7 contrast reads
  **0.5900 vs baseline 0.6085 (−0.0185) at L20**, reproduced at both his commits. F-97 records
  neither value nor this delta. **Lead QA: reproduced exactly — but see DF-46**, because at L20 that
  number is dominated by the inverted HP-KR training row; sign-corrected it is 0.7503.
- **S-12** — the −0.034 Figure-6 headline is **+0.0281** without `sycophancy_mt` and **+0.0457**
  inside the 7 instructed datasets.
- **S-04** — `run-kd/axes_vectors_all.pt` exists, is the **no-judge-filter twin** of the kd bank, is
  named in no document and **was used by no result store**. Substituting it moves dataset-averaged
  zero-shot AUROC by ≤0.026 overall, but drops `kd_assertion` on the matched slice 0.7173 → 0.5955
  (L40) and 0.6211 → **0.4416** (L20, below chance).
- **S-09** — the residualised-axes arm transfers at **0.4191 / 0.4083 at layer 40** (the inventory
  said layer 20; corrected in the note).
- **S-29** independently re-derived the F-77/F-81 **37/80 → 28/80 and 31/80** error — the same
  conclusion the lead reached from the other direction in wave 3. Two independent routes now agree.
- **Trap recorded for anyone reading `passA11`:** the bank store carries its own `datapca` column
  against which `axes` wins **0 of 80**; the matched control must come from the **root** store.
- **Reassuring:** `run-kd/axes_vectors_s3.pt` was rewritten in the RF-13 boot wave yet reproduces its
  09-20 printed numbers exactly — that rewrite was idempotent.

**P9-findings-rest — 10 rows: 3 HOLDS, 7 PARTLY, 0 fail.** The cap-bank confound entries were the
priority, and they did not survive as written.
- **F-32 certifies a bank generation that never produced a rollout [lead QA].** Three generations
  exist; F-32's acceptance table describes **v2**, while **v3** (the only one on disk, 09-20 22:49)
  produced F-65/F-66. Re-running F-32's own script on the live bank gives explain 3/5-vs-4/5,
  turn-1-commit **0/5 everywhere**, max 99 chars. "The same lexical counts were re-run" is **false** —
  all four regexes changed, and PRAISE changed target and lost four words.
- **But the cap result itself is not confounded — it is protected structurally, and I verified that
  independently on the lab PC.** My own reader over `run-cap/responses`: **1,300 rows, 65 personas,
  20 questions, exactly 20 distinct (question, turn-1) pairs at n = 65, and exactly ONE distinct
  turn-1 text per question.** Turn 1 is **byte-identical across all 65 persona arms** (teacher-forced),
  so persona wording cannot enter turn 1; only the turn-2 capitulation can differ. **The paper should
  rest the no-confound argument on that, not on F-32's lexical counts.**
- F-29's "fails as source" is **mean-pooling only** — under `last` pooling the inverted axis beats
  random as source at **18 of 21 layers** (mean: 7/21).
- F-33 says 22 roles failed; the log says **24 of 24**, and F-35 in the same ledger says 24 — an
  internal conflict the log settles.
- **`logs/cap_smoke.log` is UTF-16** — that alone is why Phase 1 marked F-26/F-28 PARTIAL. Decoded,
  every quotation is verbatim, and it is one of only two files preserving the v1 bank text.
- **"length-only 0.553 vs raw 0.665" matches no matrix**: 53 local matrices give 0.5555/0.6623 (11
  datasets) or 0.5494/0.6652 (10). It appears in F-26, F-33, F-73 and the memory note (DF-42).
- Out of batch: `run-crossgen/kd11/passC/significance.json` is **byte-identical** to
  `run-crossgen/passC/significance.json` — confirming the kd11 passC grid is **10-dataset, not 11**.

**Wave 4 is complete.** Tally: **169 of 692 claims checked — 68 HOLDS, 51 MATCHES HIS DATA, 48 PARTLY,
1 REVISED, 1 DOES NOT MATCH.** DOC_FIXES carries **46 confirmed corrections**, 21 lead-QA'd twice.

## COLLABORATOR PACKAGE DELIVERED (2026-09-23) — owner-authorised writes outside `audit/`

The owner chose, in this order: **(1) append rather than edit the ledger; (2) collaborator package
first.** This section records what was written outside `audit/`, since the brief otherwise forbids it.

**Authorisation:** owner's answers of 2026-09-23 — "Append, don't edit (edit the non-ledger docs in
place)" and "Collaborator package first". This covers the two files below and nothing else.
`kd/docs/FINDINGS.md` was **not** edited; its corrections are still owed as an appended entry.

**1. `docs/COLLABORATOR_BRIEFING.md` — corrected in place (8 changes, all verified after writing):**
- Claim 1: **0.374 → 0.313**, with a dated correction note explaining that 0.374 is the paper16
  **deception** axis's PC1 (per `27_crossgen.py:411-416` and the matrix's own `sycophancy_mt`
  column: `contrast` 0.4089 / `PC1` 0.3739), that the sycophancy axis's own value is 0.313, and
  that the correction **strengthens** the claim (deception is within the random null; sycophancy is
  significantly inverted at p 0.001–0.008). Also flags that a zero-shot axis must not be compared
  against a **trained** random probe (the old "0.541").
- Claim 2 table: the register row relabelled **paper16 deception axis, PC1**; a sycophancy row
  added at 0.313; the judge-filter row now gives both features (`pc:capitulation@2` 0.0004,
  `pc:cap_only@2` **0.0142**).
- "positive on 8–11 of 11" → **10/11, 11/11, 10/11** over eleven and **9/9, 9/9, 8/9** over the
  clean nine, in **both** places it appeared (the quote block and the §5 restatement).
- "~666 rows" → the realised pool **412–500 (mean 449)**, with 666 named as the cap
  (`30_premise.py:167`); the one-sided **p = 0.38** replaced by the two-sided **+0.0045,
  CI [−0.024, +0.033], p = 0.744**.
- Script path → `deception_detection/scripts/persona_ood.py`, axis located in `persona_pcs.py`;
  "to 1e-16" → 1e-15 with the exact figures; a note that his `KNOWLEDGE_CANDOUR_PERSONAS` merges
  `sincere_ignorance` + `plain_assertion`.
- "Both of your arms lose by 0.064 / 0.055" → named as the **extended** arm, with `axis1`'s own
  **0.146 / 0.122** added.
- The share range scoped to mean pooling, with last-token's **−156% to +145%** given beside it.
- **Verification:** every superseded string re-grepped to zero occurrences (the one surviving
  "8–11 of 11" is inside the new correction note, quoting it deliberately); every new string
  confirmed present.

**2. `docs/DRAFT_CORRECTIONS.md` — NEW file, the correction list for the collaborator.**
16 items to change, 2 attribution items, 3 additions, and a section of what was checked and found
correct — including the explicit statement that his random control is **not** a single draw and
that the HP-KR sign flip is **not** his error. Every item carries his number, the measured number
and the path on his branch. Nothing in it needs his activations.

**Still owed on the owner's side (the ledger, by her "append" decision):** one appended correction
entry superseding **F-04** (10-dataset block, and the one significant test being `random@2 − raw`),
**F-73** (0.556/0.662, the +0.0035 gap, median |gap| vs signed), **F-77/F-81** (37/80 → 28/80 and
31/80), **F-32** (the v2-vs-v3 acceptance table, replaced by the teacher-forcing argument), plus
F-29's pooling scope, F-33's 24-vs-22, F-105, F-26, F-36 and F-38. All are written up as
DF-01…DF-08 and DF-39…DF-46 in `audit/DOC_FIXES.md`, with file:line and evidence.
**Also still owed:** the length claim (DF-09/DF-10) across `docs/NOVELTY_CHECK_PROMPT.md`,
`docs/ONBOARDING.md`, two `docs/thesis/*`, `kd/docs/DESIGN.md` and the memory index; and the 0.374
relabel (DF-31) across `cap/docs/DESIGN.md`, four `REVIEW_PACKET*` files,
`docs/thesis/sarcasm-pilot-negative-result.md`, the kd skill, `CLAUDE.md` and `docs/TODO.md`.

## OWNER-SIDE FIXES APPLIED (2026-09-23) — the rest of DOC_FIXES, on her instruction "fix the rest from my side"

Method follows her two decisions: **append to the ledger, edit the non-ledger documents in place.**

**1. `kd/docs/FINDINGS.md` — NOT edited. One entry appended: `F-111`** (5,286 → 5,416 lines), which
restates eleven earlier entries and names what each supersedes, leaving the originals standing. It
covers F-77/F-81 (37/80 → 28/80 and 31/80), F-32 (the v2-vs-v3 acceptance table, replaced by the
teacher-forcing argument), F-04 (10-dataset block; the one significant test is `random@2 − raw`),
F-73 (0.556/0.662, the +0.0035 gap, median |gap| vs signed), F-29 (mean-pooling only), the
project-wide length correction, the 0.374 attribution, F-33, F-105, F-26, F-36, F-38 — and records
two items as **still open**: F-31's "nothing beats raw" being false on the regenerated lab file,
and F-20's missing raw data. It opens by stating the pattern once: **28 of 29 PARTLY verdicts found
the numbers reproducing and the sentence around them failing.**

**2. The 0.374 relabel — 13 edits across 9 files**, each naming the deception axis's PC1 explicitly
and giving the sycophancy axis's own 0.313 beside it: `cap/docs/DESIGN.md` (×3, including a dated
correction note), all four `cap/docs/REVIEW_PACKET*.md`, `docs/thesis/sarcasm-pilot-negative-result.md`,
`.claude/skills/knowledge-denial-context/SKILL.md`, `docs/NOVELTY_CHECK_PROMPT.md` (×2),
`docs/TODO.md` (the gate threshold relabelled; the gate still passes), and `CLAUDE.md`.

**3. The length claim — 6 edits** across `docs/NOVELTY_CHECK_PROMPT.md`, `docs/ONBOARDING.md`,
`docs/thesis/persona-coordinates-70b-results.md` (×2) and `kd/docs/DESIGN.md` (×2): "9 of 10" →
**9 of 11 (8 of 10 excluding `sycophancy_mt`)**, and the superseded token counts replaced.

**4. A new measurement made during the fix, worth keeping:** the length-only off-diagonal is
**0.5555 at every layer, pooling and probe** — it is a fixed 1-d feature, so training fits only a
threshold and the AUROC cannot move. Its raw comparator does move (0.6623 at L40/mean/paper, 0.6724
at L44/last/mms). This is why the quoted "0.553 vs 0.665" matches no matrix: **neither half is right**.
On 10 datasets the length value is 0.5494. `docs/thesis/persona-coordinates-70b-results.md:492`
corrected accordingly.

**5. Memory notes corrected** (minimal factual fixes only, no audit narrative imported, per the
brief's rule that audit conclusions live in `audit/`): `response-length-transfers-on-its-own.md`
(description + body table), `persona-axes-encode-register.md` (the 0.374 attribution),
`results-chapter-audit.md` (the length control), and both index hooks in `MEMORY.md`.

**Verification:** every superseded string re-grepped to zero outside the ledger. The remaining
"0.553" hits in `docs/thesis/persona-coordinates-70b-results.md:552,688,803` and
`memory/persona-set-decides-detection.md` are **different quantities** that coincidentally equal
0.553 (axes_resid, intent→intent, shuffled-label) and were correctly left alone.
`kd/docs/FINDINGS.md:2697` quotes the old wording deliberately in order to reject it — also left.

**Working tree now:** 15 modified files + `docs/DRAFT_CORRECTIONS.md` (new). Nothing committed —
the owner commits. **DOC_FIXES.md items now applied: DF-01…DF-11, DF-31…DF-46 on her side; DF-12…DF-30
are his and are delivered as `docs/DRAFT_CORRECTIONS.md` instead.**

## F-31 AND F-20 CLOSED (2026-09-23) — appended as F-112

**F-31 — there was never a conflict; the families are NESTED.** A keyed join of the git and lab
copies of `run-crossgen/kd11/banks/kd/significance.json`: all **4,032** git tests are present in
the lab's **6,300**, **none lost, and 0 changed `diff` or `dataset_p`**. The lab file adds 2,268
tests across nine comparisons that did not previously exist (`axes_resid`, `pc:kd_assertion`,
`pc:knowledge_denial` against datapca/random/raw). Only BH q moved, as it must when a family grows.
**The 6,300 family is canonical** — strict superset, identical statistics, more conservative BH.
- **"Nothing beats raw" is FALSE**, and substantively so: `pc:kd_assertion − raw` is positive at
  **18 of 21 layers**, rising from L24 to a **+0.10…+0.15 plateau from L32 to L76**, and it is
  **not local** — dropping both HP-KR columns *raises* it (to +0.12…+0.19), because raw degrades
  faster than the feature does. One test clears BH (L68, +0.1497, q 0.0467).
- **But the project's own bar is not cleared.** Against the matched `datapca` control the margin is
  positive at 20/21 layers (mean +0.0830) yet **0 of 126 tests clear**, and `mms` is much weaker
  (+0.0172, 13/21). So: a consistent, depth-robust, non-local advantage over raw that the dataset
  bootstrap cannot separate from its matched control — the F-76 category, which F-31 currently
  records as *absent*.
- **A reusable trap, now written into the ledger:** the bank store carries its own `datapca`
  column; `raw` and the matched control must be joined from the **root** store.

**F-20 — conclusion verified, numbers unrecoverable, and the constraint has lapsed.** passB reads
**258 {124/134}** and **2,139 {139/2,000}** exactly as stated. The passA/passC **98 / 89:9** state
cannot be re-derived (stdout only; stores topped up 09-21, views rebuilt 09-22 15:47 in the RF-13
wave) but is corroborated twice by pre-top-up `splits.json` (20 of 40 matched at 18:2 → ~100 at
~90:10). **Operationally the important part:** all three stores now read 258 {124/134}, so F-20's
"passA/passC must run with 10 datasets" was right when written and **no longer binds a fresh run**.

**Nothing is running.** Wave 4 finished; no agents, no lab jobs, no background tasks. Phase 3 can
begin on the owner's word.

## METHODS INVENTORY — EXPANDED (2026-09-23) — `audit/METHODS_INVENTORY.md`

**Rewritten on the owner's instruction** that the collaborator's draft is not an index of the work:
*"he even does not know what was built actually. My branch consists of the initial foundation and
most of the unique design, including fail and pass ones."* Built from the `asix` branch — ledger,
design docs, pipeline source, paper notes — not from what the draft cites. Three parts:

- **Part 1 — Lineage.** Every method traced *what we took → what we changed → why*: Mahadik &
  Skapars (the replicated method; we changed scale, breadth, k, specification and added the
  controls), Lu et al. (Assistant Axis, repurposed as a drift **diagnostic**), Chen et al. (the
  contrastive-prompt recipe, which became three banks of our own), Liars' Bench / MASK / Apollo /
  Pacchiardi (corpus and probe families), Ying et al. (corroboration **and** a predictor we tested
  that did not reproduce). Plus **§1.4: the 13 ambiguities in the replicated paper** that had to be
  resolved by decision — including the centering one, which `6_axes.py:19-21` settles **with a proof**.
- **Part 2 — Design → claim → outcome**, including the failures. Each row pairs a design with the
  claim it was built to decide and how it came out. Covers the six built datasets/banks, the
  fourteen controls, the three baselines, and the infrastructure.
- **Part 3 — The failures in full**, each with *what it limits* and *what would settle it*: the
  HP-KR growth gate (F-59, denial tracks confidence not correctness → slice capped at 258 rows, thin
  for training); **the sarcasm bank** (designed to isolate deceptive *intent*; v1 confounded by
  situation — the same failure Goldowsky-Dill et al. concede; Gate 2 failed at 0.92 bag-of-words,
  Gate 3 failed with the deception axis random on natural sarcasm; **v2 is specified and unbuilt**,
  and is the highest-value queued item); the cap result's locality (F-67); F-76 and the
  **11-dataset ceiling on significance** (F-04, and the 11→14 decision in OPEN_DECISIONS A3);
  F-112; and two negative results about other people's methods.

**The framing that matters for the thesis:** the controls are the largest contribution *because*
they are what kills this project's own headlines. A replication that only confirmed things would
not have needed any of them.

### (superseded first draft of this section)

The owner asked whether the report will cover methodology. **It is not in the brief** — Phase 5
specifies only `REPORT_PLAIN.md` and `REPORT_TECHNICAL.md`, both results-oriented. She was right
that this is a gap, and right that there is far more of it than the five items first offered.

**~30 methodological contributions found, in six groups**, each marked [AUDITED] / [LEDGER] /
[CODE] / [DOC-ONLY] so the unchecked ones cannot be mistaken for verified:
- **A. Datasets and banks built** (7): the matched slice, the kd bank, the cap bank, `sycophancy_mt`,
  the sarcasm bank, own WMDP knowledge scoring, the three-store activation design.
- **B. Controls and tests introduced** (14) — **the largest contribution, and most of them do not
  exist in the replicated paper**: the matched dataset-PCA control, the dataset-level bootstrap,
  the per-dataset random-direction null, the length control, the bag-of-words falsification gate,
  the format-matched control, leave-one-CLUSTER-out, the cluster-permutation test, layer
  pre-registration, factor probes, the PC1 permutation test, teacher forcing, seed stability, the
  Alpaca 1% FPR threshold. Several of these are what *kills* the project's own headlines, which is
  the point of them.
- **C. Baselines the paper did not run** (3): pooled training at equal data volume, single-source
  direction averaging, raw 8192-d probes.
- **D. Specifying an underspecified method**: the paper-notes list **13 ambiguities** in Mahadik &
  Skapars that an implementer must resolve by decision — k not fixed, activation sublocation
  unstated, test-time centering unstated, the anchor's role unstated, variance scaling unstated,
  Table 1 inconsistent with the figures, no code release. `6_axes.py:19-21` resolves the centering
  one **with a proof rather than a guess** (the centering cancels exactly in a difference of class
  means).
- **E. Infrastructure that is itself a method** (6): **the NDIF pipeline is reimplemented, not
  imported** — `git submodule status` is clean and nothing in `src/`, `kd/`, `cap/` or `scale/`
  imports `assistant_axis`; the attention-memory rule (rows × heads × T²); the resumable cache;
  scheduled-task orchestration; the bf16 reduction rule; matched-relative-depth scale rungs.

**Three things the inventory itself still owes**, stated in the file: several entries are [LEDGER]
and sit in the 523 unchecked claims; the extraction-vs-paper comparison is inferred from the
ambiguity list and `6_axes.py` rather than from a line-by-line reading of `2_activations_ndif.py`
against Eq (1)–(4); and **the attribution list** — on present evidence A1–A5, B1–B2, B5–B7, B9,
B12 and C1–C2 are the owner's design work and the collaborator's draft names none of them as such.
**Phase 3 should produce that attribution list explicitly.**

## PHASE M LAUNCHED (2026-09-23) — the methods track, `audit/PHASE_M_PLAN.md`

The owner asked for a proper phase for the methods work, run with agents, and for it to reach the
final paper. **The three gaps `METHODS_INVENTORY.md` recorded as owed turned out not to be blocked
on anything** — the earlier statement that the attribution list "belongs in Phase 3" was wrong:

| gap | thought to need | actually needs |
|---|---|---|
| inventory entries still [LEDGER] | the remaining 523 claims | **8 entries** — 18 of the 26 method-bearing ones are already checked |
| extraction vs the paper, line by line | — | one agent over `2_activations_ndif.py`, `6_axes.py`, `8_coords_ndif.py` |
| the attribution list | Phase 3 | git history on `asix` + his branch + the draft — all available now |

**M1 — `M1-method-claims`:** F-42, F-43, F-44 (the NDIF attention-memory rule: rows × heads × T²,
per-process), F-46, F-51 (teacher forcing — why it was needed, and that it works), F-106, F-107
(the sarcasm Gate 3: under-specified, then failed), F-108 (leave-one-CLUSTER-out and the row leak).

**M2 — `audit/methods/EXTRACTION_VS_PAPER.md`:** one row per equation/method step, classified
**RESOLVES AN AMBIGUITY / DELIBERATE DEVIATION / ACCIDENTAL DIVERGENCE / MATCHES**, with `file:line`
against the paper's own words. The third category is the one that matters. Includes verifying that
`6_axes.py:19-21`'s centering argument is actually true of the code, not merely asserted.

**M3 — `audit/methods/ATTRIBUTION.md`:** for every design, control and baseline in Part 2 — first
commit on `asix` (hash, date, author), presence on his branch, whether the draft names it, and an
attribution of **OWNER / COLLABORATOR / JOINT / UNCLEAR**. Instructed to be fair in both directions:
his `loo_scores.npz`, his follow-up-probe runs and his `persona_ood_mod_*` rollout-count control are
his work and are to be said so as clearly; genuine ambiguity is written UNCLEAR with the evidence
that would settle it, not resolved in either party's favour. Designs with **no committed artifact on
either branch** are flagged separately — the thesis should not claim those without one.

**M4 (lead, after M1–M3):** assemble `METHODS_CHAPTER.md` and `LIMITATIONS_AND_FUTURE_WORK.md`.

## M2 COMPLETE (2026-09-23) — `audit/methods/EXTRACTION_VS_PAPER.md`, accepted with one finding OVERTURNED

35 method steps compared against Mahadik & Skapars: **16 MATCHES · 11 RESOLVES AN AMBIGUITY ·
6 DELIBERATE DEVIATIONS · 3 accidental divergences claimed.** **The extraction mathematics itself
(Eq 1–4) is faithful** — everything that diverges is bank construction or k. That is a good result
for the thesis: the replication is sound where it matters most.

**The five open questions of §1.4 are now settled, not inferred:**
- **sublocation** = post-block `model.layers[L].output` = post-MLP residual stream, the same hook
  Lu et al.'s own code uses;
- **test-time centering** = none — and **provably cannot change any AUROC** (a constant per-dataset
  shift leaves the ranking identical);
- **the anchor** = excluded from the contrast, included in every PCA basis;
- **"standardize"** = mean-centering only, no variance scaling;
- **k** = 5 by default.
**And `6_axes.py:19-21`'s centering argument is TRUE of the code** — verified numerically in float64
at all 80 layers for all three contrasts, max |d_uncentered − d_centered| ≤ 2.9e-15.

**AD-1 CONFIRMED (lead QA).** `--k` default is **5** (`27_crossgen.py:882`), a value the paper never
uses. The result is k-dependent and the ledger already measures it: `pc:unified@2` beats its matched
control by **+0.138** while `pc:unified` at k=5 gives **+0.004** (`FINDINGS.md:1032-1040`,
re-verified). At k≈2, close to the paper's k=1, the axis reproduces the paper's order of effect.

**AD-2 CONFIRMED and it corroborates the owner's own note.** Unified PC1 correlates **−0.945 with
which question bank a persona answered** and only +0.230 with harmful/harmless. This matches her
memory note `unified-pc1-is-question-family` (cos −0.98) measured independently. Repaired in
`paper16_shared`, but `27_crossgen.py`'s default bank is still the confounded one.

**AD-3 OVERTURNED (lead QA).** The claim was that "70B ran k=5, the 3B/8B rungs ran 3, so the scale
comparison behind F-63/F-110 crosses a k boundary". **False.** Reading the `k` field of every matrix
in all four scale stores on the lab PC: **scale_3b k=3, scale_8b k=3, scale_70b k=3,
scale_70b_paper16fp32 k=3** — all four at k=3, all four internally consistent. Moreover **F-63's own
title says it corrected exactly this** ("F-58 compared different datasets AND different k"), and the
audit's RF-16 re-run passed `--k 3` explicitly (`fix_rf16_run.ps1:20`). The scale comparison does not
cross a k boundary. **The valid residual point** is narrower: the project's *default* is k=5, so the
scale numbers (k=3) are not directly comparable with the crossgen/kd11 headlines (k=5) — worth
stating, but it is not a defect in F-63/F-110.

**A correction to the audit's own document.** M2 was right that `METHODS_INVENTORY.md` §1.1/§1.4
claimed "we sweep k" and that **this is not supported** — three values across three separate arms,
no within-arm sweep, none of them the paper's. Corrected in place, with the k-dependence figures.

**Two cheap closes M2 identified**, both from rollouts already on disk: re-extract at 2 instruction
variants / k=1; and run the paper-faithful unfiltered bank `run-paper16/axes_vectors_all.pt` through
crossgen — **it was built for this and never used**. Also flagged: `persona-coordinates-notes.md`
contradicts itself on the anchor (§4.4 "excluded from PCA" vs §8.8 "not stated"), cited as
resolution-under-conflict rather than as a match.

## M3 COMPLETE (2026-09-23) — `audit/methods/ATTRIBUTION.md`, plus the M2 addendum

**26 Part-2 designs attributed: OWNER 24 · COLLABORATOR 1 · JOINT 1 · UNCLEAR 0**, plus 4 designs
with no committed artifact and **5 of his that the inventory omitted**.

**What the draft is missing, measured rather than asserted.** Searching the draft text: `capitulat`
**0**, `sarcas` **0**, `teacher` **0**, `datapca` **0**, `bag of words` **0**, `length` **0**,
`pre-regist` **0**, `factor probe` **0**, `leave-one` **0**, `sycophancy_mt` **0**. The only control
of either author's that reaches the draft is **his** `random3`. And the draft lists her **built and
run** `harm-pressure-kr-matched` + the kd feigned-vs-sincere split as **future work** — "would
separate the lie from its wording".

**Three corrections that run in the collaborator's favour — all verified by the lead from git:**
1. **The resumable activation cache + token-budget batcher are his.** `e391b3e`, **zami456**,
   2026-09-04, on his branch only. The owner's `ed8daca` is 2026-09-15 and its own commit message
   reads "**Port** the resumable activation cache and token-budget batcher". The port is hers, the
   design is his. The inventory listed it as infrastructure without attribution — corrected.
2. **The pooled leave-one-dataset-out baseline is his.** `6b16a46`, **zami456**, 2026-09-20,
   "Leave-one-dataset-out: the fair baseline for the generalization claim" — **one day before**
   `30_premise.py`. **This makes F-85's sentence false as written:** *"The comparison has never been
   made against a pooled probe"* (`FINDINGS.md:3413`). His `loo_L{20,40}/liars_bench_comparison_loo.csv`
   contains exactly that comparison — the lead read those rows directly during the P10 QA. **What
   remains the owner's is the equal-row-budget arm** and the "diversity, not volume" finding (F-80).
   **Correction owed to F-85.**
3. **His `persona_ood_mod_*` rollout-count control is his design** — but M3 verified on the lab PC
   that **both banks in it are hers**: his old/new n-counts match `run-kd/vectors_s3` and
   `run-kd/vectors_s23` exactly (6 of 6 values), i.e. her score-3 → score-2+3 judge filter.

**RED FLAG A — an overclaim in the audit's own document, now corrected.** `METHODS_INVENTORY` §2.2
said "most do not exist in the replicated paper". **False for two of the fourteen:** dataset-PCA and
random-subspace **are** in Mahadik & Skapars (notes §5.4–5.5, Appendix Figure 9), and
`27_crossgen.py:105` calls it *"The paper's dataset-PCA control"* in the owner's own words. The other
nine are genuinely absent. **The thesis must claim the narrower version: the matched-k discipline and
the dataset-level bootstrap applied to every headline.** Corrected in place.

**No committed artifact anywhere (the thesis should not claim these without one):** the sarcasm
**v2 bank** (designed, never built); `make_view.py` / `make_view_ac.py` — **which built the 11th
dataset behind every kd11 result** and exist only as loose files on the lab PC; `persona knowledge
denial modified/` (= her `vectors_s23`); `persona vectors/vectors_s3/` (26 × 80 layers).
`sycophancy_mt`'s 1,182 rows are lab/Drive-only, though its generator is committed.

### M2 addendum — AD-2's blast radius is narrower than first reported
`audit/claims/F-18.md` (already audited, HOLDS, HIGH) measures the same thing from the other
direction: cos(PC1_unified, unified contrast) = **+0.389 confounded vs +0.896 repaired**, higher at
80 of 80 layers. M2's independent rebuild from the laptop's raw vectors got **+0.389 at L40** and
F-18 read the lab PC's prebuilt axes and got **+0.389** — two paths, same value, which also checks
`6_axes.py`. **The refinement:** the *contrast directions barely move* between banks (cross-bank
cos = +0.984), so AD-2 damages **the PCA basis only** — it contaminates `pc:unified`,
`pc:unified@2` and the `PC1:unified` zero-shot scorer, and leaves `contrast:unified` and **all
deception and sycophancy numbers intact**. The earlier phrasing implied a wider blast radius than
the evidence supports.

**PRESERVATION FLAG (RF-10 class):** `run-paper16-shared/` — the faithful repair bank, the one any
"the paper's method at 70B" claim should rest on — **is not in git and not on the laptop.** It exists
only on the lab PC and Drive.

**Owed to the ledger:** one further appended entry (F-113) covering the M-track corrections — F-85's
"never been made" sentence, the k=5 default, and the AD-2 scoping. Deferred until M1 lands so they
go in together.

## M1 COMPLETE — PHASE M IS DONE (2026-09-23). Ledger corrections appended as F-113.

**M1-method-claims — 8/8: 6 HOLDS (F-44, F-46, F-51, F-106, F-107, F-108), 2 PARTLY (F-42, F-43),
none failed.** Every entry the methods chapter will cite is now checked.

**LEAD QA — F-107 re-derived independently, and it holds.** F-107 had **no stored artifact**, so it
was rebuilt twice: by the batch, and by the lead with its own loader, AUROC and null draws, reading
`run-coords/sarcasm/sarcasm_natural/chunk_*.pt` (600 rows, 300/300, aligned by `row_id`) against
both banks, both layers, both poolings, with a fresh 2,000-direction null per cell:

| axis | L40 mean | L40 last | L44 mean | L44 last |
|---|---|---|---|---|
| deception | 0.5427 | 0.5856 | 0.5239 | 0.5766 — **below p95 in all four** |
| sycophancy | **0.6930** | **0.7097** | 0.6763 | 0.7083 — above p95 in all four, beating the null **max** in three |

The two figures F-107 quotes, 0.6930 / 0.7097, **reproduce exactly.** **The conclusion is robust;
one of its counts is not** — F-107 says "below p95 in two cells and barely above in two", the batch
measured 1 above / 3 below, the lead's null gives **0 above / 4 below**. The count depends on the
null draw; the conclusion does not. State the conclusion, not the count.

**The other M1 findings, all recorded in F-113:** a **MiB-vs-MB unit error** in F-42/F-43 that is
also in shipped code (`scale/scripts/run_scale_coords.ps1:108-109`) — 512.00/384.00 MiB, not
537/402, though the decision is unaffected; F-43's 50-vs-**52** OOMs and span p99 **12** not 13;
**F-51's "elaborates" runs the wrong way** — `pc_caver` is longer on all four items (39.2 vs 26.5
words), which matters given F-73; and **F-108's "every one of its 258 rows"** overstates a real,
structural leak — the pool holds only HP-KR's 80% train split, so ~**81%** (42/52) of kr-matched's
test rows are contaminated, and **the same sentence is in `30g_clustered_holdout.py:12-14`**.
F-44 re-derived exactly from scratch with no artifact (2134→2134, 2139→2139, `truncated=92` = 4.30%).

**A strengthening M1 found for the thesis:** the quoted OOM line continues *"Process 3623751 has
18.14 GiB… Process 3605328 has 3.36 GiB… Process 3654827 has 7.72 GiB…"* — far better support for
the per-process reading than the two lines currently quoted. The per-process *mechanism* is still an
inference from the log, not a measurement.

**RF-13 positive data point:** `run-kd/axes_vectors_s23.pt` was rewritten 2026-09-22 16:00:24, inside
the boot-trigger wave, yet F-107 reproduces exactly against the post-wave bank — that rewrite was
numerically inert for these axes. The sarcasm activations (14:27–14:30) pre-date the wave.

**PRESERVED:** `make_view.py` (2,253 B) and `make_view_ac.py` (1,915 B) — which **build the 11th
dataset behind every kd11 result** and exist in no branch and nowhere but the lab PC (RF-8) — copied
byte-exact to `audit/cache/labpc/loose/` and logged in `audit/cache/MANIFEST.csv`.

**F-113 appended to the ledger** (5,514 → 5,641 lines), covering all of the above plus F-85's false
premise sentence, the k=5 default, and the scoped unified-axis confound. **Phase M's four
workstreams: M1 ✓ M2 ✓ M3 ✓ — M4 (assembling `METHODS_CHAPTER.md` and
`LIMITATIONS_AND_FUTURE_WORK.md`) is the lead's and is next.**

## M2 second addendum — AD-4 CONFIRMED, AD-3 OVERTURNED AGAIN (2026-09-23)

**AD-4 is real, and the lead settled it from the artifacts rather than from the scripts.** Each bank
records its own `vectors_dir`: the 3B and 8B rungs were built from **`vectors_s23`** (judge scores 2
and 3), the 70B from **`vectors_s3_fp32`** (score 3 only). So F-63/F-100/F-110's three-rung
comparison differs in the **judge filter** as well as in model size. **Impact bounded by F-62** —
which measured s3 vs s23 head-to-head and found the largest gap **0.005**, against margins of
+0.1197 / +0.1027 / +0.0541. **Further bound found here:** the group counts are identical across all
four banks (deception (8,7), sycophancy (5,5), unified (13,12), assistant (1,25)) — the filter
changed which *rollouts* fed each role's vector, not which roles exist. **Recommendation: name it as
a caveat, do not re-run** (DF-47).

**A real code bug behind it (DF-48).** `scale/scripts/run_scale_chain.ps1:197-203` loops
`foreach ($vt in @("vectors_all","vectors_s23"))` but writes every iteration to `axes_$tag.pt` — the
loop variable never reaches the output path. Confirmed by consequence on the lab PC: both vector
directories exist with **26 files each**, but there is one bank per rung and its `vectors_dir` says
`vectors_s23`. **The unfiltered arm was extracted and silently discarded.** Not fixed — outside the
authorised fix list, and the one-character change would produce two banks where
`run_scale_crossgen.ps1:53` expects one.

**AD-3 is overturned a second time.** The addendum restated it as "k=3 on the rungs vs k=5 at 70B",
which would make the comparison differ in two parameters. **It does not.** The lead read the `k`
field of every matrix in all four scale stores: `scale_3b` k=3, `scale_8b` k=3, **`scale_70b` k=3**,
`scale_70b_paper16fp32` k=3. The k=5 is the *crossgen/kd11 default*, a different arm entirely. The
three-rung comparison is internally consistent at k=3 and differs in **one** parameter (the filter),
not two.

## M4 COMPLETE — PHASE M IS FINISHED (2026-09-23)

Two thesis-ready deliverables assembled from M1–M3, the inventory and F-111/F-112/F-113:

**`audit/METHODS_CHAPTER.md`** — nine sections: what the work is; provenance (adopted → adapted, per
source); the 13 ambiguities and how each was resolved; the datasets and banks; the controls; the
baselines; the infrastructure; **authorship in both directions**; and an explicit section on what the
chapter does *not* certify. Every claim carries [AUDITED]/[LEDGER]/[CODE]/[DOC-ONLY].
Two things it states precisely that earlier drafts got wrong: **dataset-PCA and random-subspace are
the paper's, not ours** — what is original is the matched-k discipline and the dataset-level
bootstrap on every headline; and **the pooled LODO baseline is the collaborator's**, with the
equal-row-budget arm and the diversity finding the owner's.

**`audit/LIMITATIONS_AND_FUTURE_WORK.md`** — ten sections, each stating *what was found · what it
limits · what would settle it*: the eleven-dataset significance ceiling; the sarcasm bank's two
failed gates (with the v1 situation-confound story and the independently re-derived Gate 3 table);
the HP-KR growth gate; the two consistent-but-uncleared results; the cap result's locality; the three
parameter caveats (k=5, the filter asymmetry, the unified-axis confound — each with its bound); the
reproducibility gaps; the three known defects left unfixed; the five open decisions; and future work
ranked, with **build sarcasm v2** first.

**Phase M: M1 ✓ M2 ✓ M3 ✓ M4 ✓.** The three gaps `METHODS_INVENTORY.md` recorded as owed are all
closed — the [LEDGER] method entries are checked (8/8), the extraction-vs-paper comparison is read
line by line rather than inferred, and the attribution list exists with a commit hash per design.

**Phase 5's four deliverables now stand as:** `REPORT_PLAIN.md` and `REPORT_TECHNICAL.md` (still to
write, in Phase 5), plus **`METHODS_CHAPTER.md` ✓** and **`LIMITATIONS_AND_FUTURE_WORK.md` ✓**
(written). `DOC_FIXES.md` (48 entries) and `docs/DRAFT_CORRECTIONS.md` remain the two correction lists.

**Nothing is running. Phase 3 is next** — cross-checks, TIMELINE.md, and the owner's priority: what
the draft misses in both directions. The attribution list and the draft-credit search in
`METHODS_CHAPTER.md` §8 already supply a large part of that input.

## Phase 2 watch list (collected during Phase 1)
- BH q-values quoted in FINDINGS.md at lines ~80 (F-04), ~705–712 (F-31), ~1945, ~2296–2297 and ~2329 (F-66/F-67 area): check which test-family version they were computed on (RF-13 note).
- Draft-vs-ledger topics 1–8 (audit/phase0/draft_vs_ledger_preview.md): do these first.
- 23 print-only analysis scripts (phase0/local_results.md §3): their numbers exist only in console logs or prose; re-run read-only on the lab PC.

## PHASE 1 COMPLETE (2026-09-22 23:30)
- **Deliverables:**
  - audit/experiments.csv (114 rows), audit/papers.csv;
  - audit/phase1/fmap_all.csv (109 F-numbers → evidence);
  - audit/phase1/anomalies.{md,csv} (orphans / ghosts / duplicates / conflicts);
  - census files in audit/phase1/labpc/;
  - cache audit/cache/ (MANIFEST.csv, all rows MATCH).
- **End-of-phase checks (VERIFIED):**
  - git: branch asix, HEAD 43ee6d3, `git status --porcelain` EMPTY (= START_STATE baseline after Setup).
  - Lab PC: since 18:40 the only changed owner files are 9 logs of the owner's own sync tasks (audit/phase1/labpc_changed_since_audit_start_end_of_phase1.txt). The audit wrote only in audit_scratch (logged in audit/LABPC_WRITES.md).
- **Questions for the owner (Phase 1):**
  - (1) Who triggered the 09-22 re-runs? There were two: the 70B kd-bank sweep 00:52–01:33, and the wave 15:40–16:26 (RF-13, RF-14).
  - (2) RF-10: will the owner back up the lab-only folders to Drive? The audit may not.
  - (3) OK to start Phase 2?

## OWNER DECISIONS after Phase 1 (2026-09-22 ~23:40) — these EXTEND the brief's safety rules for these items only
The owner wrote: "1. fix that. 2. yes u do the backup, 3. remember to fix that in phase 2, 4. fix the bug, 5. do whatever best with f-39 … q1. i dunno who started, i have not run anything in lab pc after u started, q2. dont go for phase 2 yet, solve these issues first". She also asked that Claude session transcripts be mined for her tests, findings and dataset changes ("we should have more than him").

Mapping to my Phase 1 summary items and the authorised actions:
1. **RF-16 (70B scale-rung bank mix): FIX.** Compute a like-for-like paper16-bank 70B layer sweep from existing activations (CPU, no NDIF). Write to a NEW output folder, never overwriting existing results. Recompute F-100–F-102 both ways and record the correction.
2. **RF-10: BACK UP lab-only results to Drive.** `rclone copy` only, into a NEW Drive folder, so no existing Drive file is overwritten. Verify with `rclone check`. Never sync/move/delete.
3. **Print-only numbers:** recompute in Phase 2 (already in the plan).
4. **RF-14 guard bug: FIX** in the repo working tree and in the lab copies (the chains run there). Back up each lab file first and verify md5 before/after. I also fix RF-6 (the backspace byte in run_cap_ctrl_layers.ps1), same bug class. No git commit: the owner commits.
5. **F-39:** do what's best, i.e. a sensitivity analysis on existing matrices plus recommended wording for the paper.
- Q1: the 09-22 re-runs (00:52 sweep, 15:40 wave) were NOT started by the owner after the audit began. Both predate the audit (started 18:44). Origin unknown; likely an earlier Claude session (F-104 mentions "tonight's" re-triggers; DOC-ONLY). The transcripts may show it.
- Q2: do NOT start Phase 2 until these are solved.
- New task: mine the Claude session transcripts for tests, findings and dataset builds/changes that are not in files.

## FIX WORK (owner-authorised), 2026-09-22 23:45 →
- **Item 4, the bug: DONE and now PROVEN IN OPERATION (2026-09-23 03:41).** With a DONE line present, a normal run of run_falsify.ps1 **skipped in 0.3 s** and logged "SKIP already DONE; -Force to re-run" (audit/fix/guard_livetest.txt). The -Force half was proven by the 02:19–03:40 run, which recomputed all 63 cells despite that same DONE line. So: guard fires when it should, -Force still overrides.
- **Item 4, the bug: DONE (VERIFIED both copies).** audit/scripts/fix_guard_bug.py patched 8 files byte-exactly:
  - RF-14 guard in 7 chain scripts: `-Pattern '^DONE'` → `-Pattern '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+DONE\s*$'`;
  - RF-6: the 0x08 byte in run_cap_ctrl_layers.ps1 lines 31–32 → `\banks`.
  - Pattern tested read-only against 8 real lab status files first (audit/fix/guard_pattern_test.txt): old never matched; new matches every finished chain and does NOT match lines like "periodic DONE exit 0".
  - Repo working tree: 8 files modified, 9 lines (`git diff --stat`), NOT committed — the owner commits. Originals: audit/fix/originals_repo/.
  - Lab PC: the 8 copies were byte-identical to the repo before patching; patched copies uploaded and md5-verified (audit/fix/fix_lab_verify.txt). Originals kept in audit_scratch/backup_scripts_2026-09-22.tar and audit/fix/originals_lab/.
- **Item 2, the backup: DONE (VERIFIED).** 1,983 lab-only files (1.73 GB; not on Drive, not in git) → new Drive folder `tamara:expLB/_labpc_only_backup_2026-09-22/` via the lab's rclone, `copy --immutable` + `check --one-way`. Selection: audit/scripts/fix_backup_select.py → audit/fix/backup_list.tsv. Driver audit/scripts/fix_backup_run.ps1; log audit_scratch/out/backup_*.log.
  - Result 2026-09-23 00:15: copy exit 0 for all four roots; `rclone size` on the destination = 1,983 files / 1,731,072,282 B. `rclone check --one-way`: 1,977 files hash-identical; 6 flagged 'sizes differ' = the owner's live sync logs (coords_rclone.log, coords_supervisor.log, kd_rclone.log, kd_sync.status, night_chain.log, night_rclone.log), which her scheduled tasks appended to during the copy. Nothing existing on Drive was touched (new folder + --immutable).
- **Item 5, F-39: DONE → audit/fix/F39_NOTE.md.**
  - F-39's numbers reproduce exactly with the owner's own paper_comparison.py (raw 0.8090; pc:unified@2 +0.0262; datapca@2 −0.1120; k=5 −0.0173/−0.0214).
  - The paper's five datasets VERIFIED from the PDF = ConvGame, InstrDec, Mask, AILiar, Roleplay → F-39/F-60 right; F-45/F-48/F-54 (which name harm-pressure-choice) wrong.
  - Sensitivity (audit/fix/f39_sensitivity_L40.txt, _layers.txt; script fix_f39_sensitivity.py): axis@2 − raw swings −0.049…+0.128 with the dataset set (raw baseline 0.746…0.873), while axis@2 − datapca@2 stays +0.085…+0.19 at every variant and layer.
  - Recommendation: lead with the margin over the matched control (+0.138 on the paper's five at L40, vs the paper's +0.13…+0.23), report the raw comparison with its range, fix F-45/F-48/F-54, and always state k.
- **Item 1, RF-16: FIXED and MEASURED (2026-09-23 00:20) → audit/fix/RF16_NOTE.md.** (The first attempt was blocked by the permission classifier; the owner then allowed it.)
  - Re-ran the owner's 27_crossgen.py unchanged except the bank (run-paper16/axes_vectors_s3_fp32.pt) → new folder run-crossgen/scale_70b_paper16fp32 (42 matrices), CPU only.
  - Recompute (audit/scripts/fix_rf16_recompute.py, same off-diagonal recipe as paper_comparison.py; output audit/fix/rf16_recompute_mean.txt). Over F-100's 19 layers, mean pooling, best = max(axes, pc:deception):
    - 3B +0.1197, 8B +0.1027, 70B-kd +0.0751 vs dataset-PCA — **exactly F-100's published +0.120/+0.103/+0.075**, and best−raw 70B +0.0079 = F-100's +0.0079 → the method is validated and RF-16's diagnosis confirmed.
    - 70B like-for-like (paper16): **+0.0541 vs dataset-PCA, −0.0131 vs raw**.
  - Consequences: F-100's monotone decline SURVIVES and is steeper (+0.120 → +0.103 → +0.054); its sentence "against raw the 70B advantage is essentially nil" must become NEGATIVE (−0.013); its source line ("same bank as the L40 cell") is wrong; F-63 is unaffected; F-101/F-102 need the same treatment (last pooling: +0.0069 raw / +0.0681 dPCA like-for-like).
  - A suggested ledger entry (F-110) is drafted in the note but NOT written to kd/docs/FINDINGS.md — the owner decides.
- (superseded) Item 1 original status: BLOCKED. The like-for-like paper16-bank 70B sweep (audit/scripts/fix_rf16_run.ps1: the owner's 27_crossgen.py, passB activations, same 5 datasets, k=3, CPU only, new folder run-crossgen/scale_70b_paper16fp32) was **denied by the Claude Code auto-mode permission classifier**. Nothing ran; no folder was created; nothing was logged. The owner must allow it (or run the script themselves).
- **Transcript mining: RUNNING** (2 agents) over 19 Claude sessions, 140 MB, 2026-09-09..09-22 → audit/phase1/sessions/: sessions_index.tsv, session_experiments.csv, dataset_events.csv, findings_printed_numbers.csv, findings_unrecorded.csv, findings_rerun_origin.md. Transcripts are DOC-ONLY leads, never evidence.

## Next — READ THIS FIRST IF YOU ARE A NEW SESSION (updated 2026-09-23, after Phase 2 wave 4)

**State:** Phase 2 is in progress. **169 of 692 claims checked** (68 HOLDS, 51 MATCHES HIS DATA,
48 PARTLY, 1 REVISED, 1 DOES NOT MATCH; HIGH confidence on 165). Waves 1-4 are done and recorded
above ("Wave 1/2/3/4 results"). Nothing is mid-flight; no agent is running.

**To continue, run wave 5 the same way waves 1-4 ran:**
1. Read `audit/PHASE2_PLAN.md` (design + the two-level honesty rule) and
   `audit/phase2/CLAIM_INSTRUCTIONS.md` (the brief handed verbatim to each batch agent).
2. Pick ~10 unchecked claims per batch from `audit/claims_inventory.csv`. What is left:
   **his draft body ~169 (D-*), owner docs ~93 (O-*), memory notes ~97 (M-*), his notes ~92 (H-*),
   ledger remainder ~40 (F-*), session leads ~25 (S-*), abstract 0 (done).**
3. Launch the batches as parallel agents, each writing
   `audit/phase2/claims_batch_<name>.csv` + `audit/claims/<id>.md` + `audit/scripts/claims/<id>.py`.
4. **Rolling QA is not optional: the lead re-derives 2 of every 10 independently, from raw files,
   before accepting a batch.** Each batch nominates its two. This has already overturned agent
   findings (P8's random-control claim) and sharpened several others.
5. Record the batch in this file, add any confirmed documentation error to `audit/DOC_FIXES.md`,
   then run `python audit/scripts/p2_merge_claims.py` to refresh `audit/claims.csv` and the tally.

**Still owed inside Phase 2:** workstream **W3** (re-running the ~23 print-only analysis scripts, whose
numbers exist only in console output) and **W4** (the central controls / random-direction null,
collected once for every claim that needs it). `audit/phase2/layer_sensitivity.csv` (W1) and
`provenance.csv` (W2) already exist - reuse them rather than recomputing.

**Then:** Phase 3 (cross-checks, TIMELINE.md, and the owner's priority - **what the draft misses in
both directions**), Phase 4 (independent re-derivation of the HIGH claims plus a 20% sample),
Phase 5 (reports).

**PHASE 5 IS AMENDED, ON THE OWNER'S INSTRUCTION (2026-09-23) - DO NOT DROP THIS.** The brief lists
two report deliverables; there are now **four**:
1. `audit/REPORT_PLAIN.md` (brief) · 2. `audit/REPORT_TECHNICAL.md` (brief) ·
3. **`audit/METHODS_CHAPTER.md`** - lineage, designs, controls, baselines, every claim marked
[AUDITED]/[LEDGER] so nothing unverified reaches the thesis unlabelled ·
4. **`audit/LIMITATIONS_AND_FUTURE_WORK.md`** - each failure, what it limits, what would settle it,
and the decisions still open (the 11->14 grid; sarcasm v2).
Source material: `audit/METHODS_INVENTORY.md`, `audit/PHASE_M_PLAN.md`, `audit/methods/`.

**PHASE M (the methods track) runs alongside Phase 2** - see `audit/PHASE_M_PLAN.md`. M1 (8 method
claims), M2 (extraction vs the paper, line by line), M3 (the attribution list) launched 2026-09-23;
M4 (assembly) is the lead's, after them.

**Owner action still outstanding:** two amendments to F-110 in `kd/docs/FINDINGS.md` are written but
**uncommitted** (`git status` shows `M kd/docs/FINDINGS.md`). That is the only uncommitted change.

**Tooling you will need:** `audit/scripts/labpc_run.py` + `p1_labpc_runpy.ps1` (run a script read-only
on the lab PC), `labpc_upload.py` (upload into the approved scratch folder, auto-logged),
`p1_peek.py` (inspect one lab file). The Bash tool collapses doubled backslashes in heredocs - use
forward slashes. Every lab or Drive write goes in `audit/LABPC_WRITES.md`.

## Audit risks (not findings)
- Lab PC tasks still write to Drive every 15–30 min (copy only). Anything read from Drive or the lab PC should be stamped with read time + md5.
- F-ledger: kd/docs/FINDINGS.md has F-01..F-109 (contiguous, no duplicate headings, audit/phase0/fnumbers_asix.txt). A working-tree grep also hit "F-112" (1×) and "F-150" (4×) somewhere — location being checked by the docs subagent.

## Owner answers (2026-09-22, after Phase 0) — owner statements, not yet checked against files
- **The draft PDF IS the collaborator's draft** (`full paper draft for this experiments/main (1).pdf`). Settles the Phase 0 inference.
- **Division of work:** the collaborator used the owner's results for the personas and everything else. The only thing he made himself is the **activations he downloaded** (owner: "his activations for one layer is 62 GB"; owner is not sure exactly what he downloaded). Everything else was made by the owner + Claude.
- **His results:** do NOT ask him for files. Where recompute is impossible, read the real data from his branch (his committed CSV/npz/pt) and check the draft and his notes against those.
- **Second round (2026-09-22):**
  - **Joint work.** Owner and collaborator work on the same project. The owner set everything up; then each ran tests. He is **not aware of the owner's tests and the claims she tested**.
    → Importance = for the joint paper. Phase 3 must list what the draft misses from her side.
  - **Scope.** Only `asix` and `origin/feat/persona-pc-ood`. Mahdi's branch is skipped, and so are the July branches (Saving, claude/*, copilot/*).
  - **Where things are (owner):** her results on the lab PC, backed up on Drive; his results on his branch. No other locations.
  - **Lab PC use.** "Use the lab PC whenever you want; run everything there." The laptop lacks the RAM: a background scan was already killed for low memory in Phase 0.
    → Heavy scans and analysis run on the lab PC; the laptop runs only light scripts.
  - **Third round (2026-09-22), all questions now answered:**
    - (a) **No new experiments during the audit.** Neither the owner nor the collaborator will run any on the lab PC. The lab PC is multi-user; the team's account is `user4`, so stay inside `C:\Users\user4\`.
    - (b) **Scratch folder approved:** "create whatever you need, just keep it in the records". It goes at `C:\Users\user4\audit_scratch\`; every creation is logged in audit/LABPC_WRITES.md.
    - (c) **No earlier draft exists.** `main (1).pdf` is the only version, so the "One Direction Generalizes" text that F-85/F-88 critique stays NOT-FOUND.
  - **Fourth round (2026-09-22):**
    - **The earlier version was a title + abstract, not a draft.** The owner had Claude check it, which produced COLLABORATOR_BRIEFING, F-85/F-88/F-97. The verbatim text was RECOVERED from an earlier session transcript (b035ccb3…jsonl line 11117, 2026-09-21T10:38Z) → `audit/phase0/collab_abstract_2026-09-21.md`. It is now a claim source.
      - Numbers differ from the PDF draft, e.g. 4v6 kd contrast "0.882 and 0.912" (abstract) vs 0.919 (draft) → claim families.
    - **Unawareness runs both ways:** "we are not well aware of each other's findings and tests" → Phase 3 checks both directions.
    - **"All the builds are done here; he just did the dot product with the activation he downloaded"** (owner).
      - Factual note for later, not a dispute: his branch also contains trained 8192-d probes, leave-one-out probes and the benchmark's Mean/Follow-Up probe runs (VERIFIED, phase0/branch_compare.md §2).
  - **Owner's main worry (2026-09-22):** the draft (arXiv-style paper) may miss her findings or state things wrongly — that is why she wants the full audit.
    - The lead auditor has now read the draft first-hand (all 19 pp).
    - Preview of where the draft and her ledger touch → `audit/phase0/draft_vs_ledger_preview.md`: 8 topics, DOC-ONLY on both sides, no verdicts. Key topics: pooled baseline "not compared" vs F-79–F-85; "a layer between 20 and 40 may do better" vs F-91; wording test "not run" vs F-01–F-06, F-72, F-76, F-108; "one model" vs the 3B/8B rungs; 0.904/0.916 attribution vs F-97–F-99; sycophancy/register vs F-13, F-65–F-67, F-89.
    - This makes Phase 3 (a) the owner's top priority. Suggest Phase 2 start with the claims these 8 topics involve.
  - **Status:** Phase 0 is closed and all questions are answered. Waiting for the owner's "go" for Phase 1.
- Consequences for the plan are in audit/PLAN.md ("How his claims will be checked"). RF-1 now reads: *which version of the OWNER's kd vectors did he receive?* RF-2 is now accepted scope (Level 1 only for activation-derived numbers), not an open request.

## Open questions for the owner
Remaining list at the bottom of audit/PLAN.md (questions 2 and 3 are answered). FYI only: local `main` has 1 unpushed commit (608b2a7) — it is an ancestor of asix, so nothing is lost.

## Which file holds what
- audit/BRIEF.md — the brief (owner's)
- audit/START_STATE.md — git state before the audit; baseline for end-of-phase check (porcelain should be EMPTY now that audit/ is excluded)
- audit/.exclude.before — .git/info/exclude before Setup
- audit/PROJECT_MAP.md — one-page plain map for the owner; audit/PLAN.md — Phases 1–5 plan + owner questions
- **audit/SESSION_MINING.md — the owner's Claude session transcripts as an audit source** (19 sessions, ~140 MB, 2026-09-09→; where they are, the DOC-ONLY rules, and the extracted tables in audit/phase1/sessions/). Read this if the question is "what did the owner test / find" — much of it exists nowhere else.
- **audit/DOC_FIXES.md — the running list of documentation corrections the audit has CONFIRMED** (DF-01…), each with file:line, current wording, corrected value and the evidence path. Nothing in it has been applied; it is the owner's to act on. Entries marked [lead QA] were re-derived twice.
- audit/fix/ — owner-authorised fixes: F39_NOTE.md, guard-bug originals + verification, backup list, RF-16 run output
- audit/LABPC_WRITES.md — every write the audit made on the lab PC or Drive
- audit/phase0/ — all Phase 0 evidence: fnumbers_asix.txt; docs_inventory.md + fnumber_history.csv + memory_inventory.tsv; branch_compare.md + pcood_results_index.csv; draft_report.md + draft_claims_raw.csv + draft_text.txt + draft_pages/; local_results.md + local_results_index.csv + scripts_to_outputs.csv + local_figures_provenance.csv + local_files_flagged.csv; labpc_*.txt/csv (roots, sizes, tasks, task scripts, rclone verbs, docs, docs_vs_local); rclone_sizes.tsv
- audit/scripts/ — p0_* scripts that produced the above (re-runnable, read-only); lab-PC ones are .ps1 sent via `ssh -o BatchMode=yes user4@100.85.217.68 "powershell -NoProfile -NonInteractive -EncodedCommand <base64 UTF-16LE>"`
- audit/tools/pylib/ — pypdfium2 installed locally by the draft subagent (PDF page rendering only)
- Tooling gotcha: the Bash tool collapses doubled backslashes inside heredocs — write paths with forward slashes or chr(92).

## WAVE 5 — the owner's scope decision on Phase 2 (2026-09-23)

**The question she asked:** "do i need to complete phase 2?" — i.e. must all 692 claims be checked
before Phase 3 begins. **Her decision: no — run one targeted wave, then Phase 3.**

**The evidence that decision rests on.** 515 claims were unchecked. Each source type had already
been sampled by one batch in waves 1-4, and the sample tells you what the remainder holds:

| unchecked | n | what its own sample already found |
|---|---|---|
| draft body rest (D-) | 176 | P8 did 40: ~90 values reproduce, 1 fails, 5 wording-only. All 18 headline rows already done. |
| memory notes (M-) | 97 | **never sampled** |
| owner docs (O-) | 93 | P12 did 10: 9 held; the 1 failure was DF-31, which had spread to nine documents |
| his notes (H-) | 92 | P10 did 10: every digit exact to <=5e-4; all 6 PARTLY were the quantifier |
| ledger remainder (L-F-) | 32 | P9 did 10: **3 HOLDS / 7 PARTLY** — the worst rate measured anywhere |
| session leads (S-) | 25 | P13 did 10: all recoverable, 7 held |

The stable pattern is F-111's own opening line: **28 of 29 PARTLY verdicts found the numbers
reproducing and the sentence around them failing.** The remainder is unlikely to overturn a headline;
it will yield more overstated wording.

**So wave 5 takes the two slices where a real problem could still hide**, and leaves the rest to
Phase 4's random ~20% sample, which the brief already specifies for exactly this purpose:
- the **32 remaining ledger entries in full** — a *source*, not a restatement, and the highest
  observed defect rate;
- a **seeded sample of the never-sampled memory notes (12) and owner docs (10)** — memory notes are
  loaded into every future Claude session in this repo, so an error there propagates indefinitely.

Selection is re-runnable: `audit/scripts/p2_wave5_select.py` (seed 42) -> `audit/phase2/wave5_batches.csv`.

**54 claims in 5 parallel batches, launched 2026-09-23 ~11:10**, same machinery as waves 1-4
(`audit/phase2/CLAIM_INSTRUCTIONS.md` handed verbatim; each writes
`audit/phase2/claims_batch_<name>.csv` + `audit/claims/<id>.md` + `audit/scripts/claims/<id>.py`):

| batch | n | claims |
|---|---|---|
| W5A-ledger | 11 | F-103, F-21, F-39, F-45, F-49, F-53, F-57, F-62, F-69, F-78, F-90 |
| W5B-ledger | 11 | F-104, F-22, F-40, F-47, F-50, F-55, F-60, F-64, F-70, F-82, F-91 |
| W5C-ledger | 10 | F-14, F-24, F-41, F-48, F-52, F-56, F-61, F-68, F-71, F-84 |
| W5D-memory | 12 | M-004, M-012, M-014, M-015, M-018, M-029, M-032, M-036, M-070, M-082, M-087, M-095 |
| W5E-ownerdoc | 10 | O-014, O-015, O-022, O-038, O-040, O-065, O-075, O-082, O-086, O-088 |

**Rolling QA stays in force:** each batch nominates the 2 claims it is least sure of, and the lead
re-derives them independently from raw files before the batch is accepted. This has already
overturned agent findings (P8's random-control claim, M2's AD-3 twice).

**After wave 5:** Phase 2 closes at ~231/692 checked, with the residual bounded by Phase 4's random
sample rather than by exhaustion. W3 (re-running the ~23 print-only analysis scripts) and W4 (the
central random-direction null) remain open inside Phase 2 and are carried into Phase 3/4 as needed —
note that session mining already recovered 72/72 print-only F-numbers with their console output,
which lowers W3's urgency.

### WAVE 5 RESULTS — all 5 batches in, all lead-QA'd (2026-09-23)

**54 of 54 checked: 38 HOLDS · 15 PARTLY HOLDS · 1 DOES NOT HOLD.** No claim was left without raw
data. Running total: **231 of 692 — 112 HOLDS, 51 MATCHES HIS DATA, 53 PARTLY HOLDS, 12 PARTLY
MATCHES, 1 REVISED, 1 DOES NOT MATCH, 1 DOES NOT HOLD.** DOC_FIXES now carries **56 entries**
(DF-49…DF-56 are wave 5's). Full QA record: **`audit/phase2/qa_wave5.md`**.

| batch | n | verdicts |
|---|---|---|
| W5A-ledger | 11 | 5 HOLDS · 5 PARTLY · **1 DOES NOT HOLD** (F-49, already superseded by F-50) |
| W5B-ledger | 11 | 8 HOLDS · 3 PARTLY |
| W5C-ledger | 10 | 5 HOLDS · 5 PARTLY |
| W5D-memory | 12 | **12 HOLDS** — every quoted number reproduced; 9/9 evidence files md5-identical laptop=lab |
| W5E-ownerdoc | 10 | 8 HOLDS · 2 PARTLY |

**THE TWO RESULTS THAT MATTER MOST ARE CORRECTIONS TO THE AUDIT'S OWN WORK.** Both were raised by
batches and confirmed by the lead from raw files. This is the rolling-QA rule working in the
direction it was built for.

1. **DF-50 — `audit/fix/F39_NOTE.md` §2 was wrong, and the owner was right all along.** The note told
   her to "fix F-45, F-48 and F-54" for naming HarmPressureChoice as the paper's fifth dataset.
   Independent `pypdfium2` extraction: `HarmPressureChoice` appears **twice, both on page 4**; `Mask`
   appears **41 times and never on page 4**. **Table 1 declares one set and the figures evaluate
   another.** F-45/F-48/F-54 are correct about Table 1. W5A and W5C found this independently; with
   the lead's extraction that is three routes. The note is corrected in place and its recommendation 3
   withdrawn. **The owner had already documented the whole inconsistency three times** — F-60,
   `FINDINGS.md:417-419`, and the paper notes §8.1. The real owed fix is much smaller: F-60's
   correction was never back-annotated to F-48/F-54.
2. **DF-49 — the audit's own length correction is wrong and is already applied in six documents.**
   Our 2026-09-23 edit says "never '9 of 10'". Read from the producing file
   (`run-coords/confounds_L44_last.json`, key `length_only_auroc`): ten datasets, **nine above 0.5**,
   `roleplaying` the sole inversion at 0.4415. We changed **both the metric and the dataset set**
   without saying so — "9 of 11"/"8 of 10" are mean token gap over passB; "9 of 10" is length-only
   AUROC over the set the control was actually run on, which *includes* `sycophancy_mt` and *excludes*
   `harm-pressure-kr-matched`. **Not re-applied — needs the owner's word**, since the earlier edit
   round was authorised for a specific list.

**Other confirmed corrections** (all lead-QA'd, all in DOC_FIXES): F-61's "twelvefold" is **8.1×** —
the entry divides kd's margin over *raw* by paper16's over *datapca*, and 0.0757/0.0064 = 11.8
reproduces the stated figure exactly, proving the mis-copy (DF-51). F-56's stated mechanism is
backwards — `subsample()` is a balancer that took **all 134** deceptive rows that exist, so it
*raised* the deceptive share from 6.28% to 16.75% rather than preserving it (DF-52). F-91's "second,
independent method" mixes seeds **one way** — L32 and L36 quoted at their best seed, L40 at its worst,
when L40's best (0.8957) is the highest value in the table; no consistent reading supports "32-36,
not 40", and nothing there is separable anyway (DF-53). F-78 quotes Ying et al. Figure 4 **panel (b)**
(transfer *from* sycophancy) under a sentence describing panel (a); conclusion unaffected (DF-54).
**O-040 — the paper-facing one:** "probes report ~0.87 on HP-KR" has no source; a sweep of all 113
matrices puts the raw probe **diagonal at 0.9663 median, 0.96-0.98 at every headline layer**, so the
regex (0.9125) *loses* to the probe rather than beating it. The argument survives restated — the
regex reaches **94.4% of** the full probe — but `NOVELTY_ASSESSMENT.md:30` calls this quantification
the **only** original part of claim 9 (DF-55).

**Method note worth keeping:** W5D was the first memory-note batch ever run, and all 12 held. The
memory notes are in better shape than the ledger — the ledger remainder returned 15 PARTLY of 32,
consistent with P9's earlier 7-of-10. That asymmetry is itself a finding: **the errors live in the
ledger's prose, not in the notes that quote it.**

### WAVE 5 FIXES APPLIED (2026-09-23, on the owner's "fix it now")

**DF-49 fixed in place, metric now named at every site.** The wrong "never '9 of 10'" is gone from
all six documents plus memory: `docs/NOVELTY_CHECK_PROMPT.md:111`, `docs/ONBOARDING.md:78`,
`docs/thesis/persona-coordinates-70b-results.md:497` and `:1106`, `kd/docs/DESIGN.md:176` and `:203`,
`memory/response-length-transfers-on-its-own.md` (description + the correction block, which now
states both counts and why they differ), and the `MEMORY.md` hook. Each now says **which metric and
which dataset set**. Verified by grep: zero residual occurrences outside the ledger.
- Bonus fix at the same site: `persona-coordinates-70b-results.md:497` called the per-dataset
  length AUROCs "**(log only)**". They are not — they are in
  `run-coords/confounds_L44_last.json`, key `length_only_auroc`. Path now cited.
- Left alone deliberately: the "9 of 10" hits at `FINDINGS.md:2244`, `:3788` and in
  `memory/persona-axes-encode-register.md:50` are **different quantities** (cap/pc beating raw on
  N source datasets), not the length claim.

**`F-114` appended to `kd/docs/FINDINGS.md`** (5,640 → 5,791 lines, +151, append-only — verified:
the only deletions in the file's diff are the pre-existing uncommitted F-110 amendments at
`:5257` and `:5271-5273`). It covers all eight wave-5 ledger corrections **and the two corrections
to the audit's own work**, states what each supersedes, and leaves every original standing, per the
owner's "append, don't edit" rule. It closes with what wave 5 found reassuring, including the
memory-note/ledger asymmetry.

**Working tree after wave 5:** the same 15 modified files + `docs/DRAFT_CORRECTIONS.md`, i.e. no new
files touched beyond those already in the owner's uncommitted set. Nothing committed — she commits.

**Still owed to the ledger (not done, needs her word):** back-annotating **F-48 (`:1418`)** and
**F-54 (`:1659`)** with a pointer to F-60, which F-114 §1 identifies as the real defect behind DF-50.
F-114 records it; the two entries themselves are untouched.

## PHASE 3 LAUNCHED (2026-09-23 ~12:0x) — cross-checks and gaps

Shared brief handed verbatim to every workstream: **`audit/phase3/PHASE3_INSTRUCTIONS.md`**. It
states the one thing the owner cares about most (unawareness runs BOTH ways), lists what Phases 0-2
and M already established so nothing is redone, repeats the evidence and safety rules, and carries
the eight traps that have already cost this audit time - including the newest one from DF-49:
**never quote a count without its metric and its dataset set.**

Rule carried forward: anything with a verdict in `audit/claims.csv` is VERIFIED and may be cited;
**anything not in it is DOC-ONLY and must be tagged so.** 461 of 692 claims remain unchecked, so
this matters on every page.

| workstream | output | scope |
|---|---|---|
| **P3-A** | `audit/phase3/DRAFT_GAPS.md` | **the owner's top priority** - what the draft misses, overstates or gets wrong, in BOTH directions. Starts from the 8 preview topics, which can now finally be given verdicts, then sweeps beyond them. Equal care to the three corrections that run in HIS favour. |
| **P3-B** | `audit/phase3/DISAGREEMENTS.md` | where her results and his disagree on the same setting, and the cause. Traceable because his vectors ARE her files - so the cause is his activations, his probe code, or a layer/k/seed/dataset mismatch. |
| **P3-C** | `audit/phase3/WEAK_SPOTS.md` | stale foundations (does any surviving claim still rest on a superseded file, and does the number move?) + weak spots (the F-108 leak, the 11-dataset ceiling, single-seed/single-layer margins, unfixed code defects). |
| **P3-D** | `audit/phase3/UNFOLLOWED.md` | strong results nobody wrote up (four orphan grids incl. passP, the planned construct-validity control), the 25 remaining transcript-only measurements, and which negative results are publishable rather than caveats. |
| **P3-E** | `audit/TIMELINE.md` | the project's story in order, from git + the 242 lab logs' internal timestamps + the session transcripts. Warned that commit dates are BATCH dates, and that transcripts are UTC while both machines are UTC+6. |

Lead QA continues: each workstream nominates its 2 least-certain findings and the lead re-derives
them from raw files before acceptance. In wave 5 that rule caught two errors in the audit's own work.

## PHASE 3 COMPLETE (2026-09-23) — all five workstreams in, all lead-QA'd

**Deliverables:** `audit/phase3/DRAFT_GAPS.md` (13 items her way, 7 his way, 3 UNCLEAR),
`DISAGREEMENTS.md` (11 items), `WEAK_SPOTS.md` (676 lines), `UNFOLLOWED.md`,
`audit/TIMELINE.md` (716 lines, 16 dated sections), and the QA record
**`audit/phase3/QA_PHASE3.md`**. Re-runnable scripts in `audit/scripts/phase3/`.
End-of-phase git check: working tree = the same 15 modified files + `docs/DRAFT_CORRECTIONS.md`,
i.e. unchanged from the session baseline. Nothing written outside `audit/`.

### The finding that reframes the project

**The two pipelines agree. The method does not beat its controls.** P3-B established, over 55
bank-matched cells, that when the owner and the collaborator use the same bank on the same rows
their pipelines agree to **mean |gap| 0.0015** — so no disagreement anywhere is a code bug; every
one is a bank, an arm definition, a dataset set or a training protocol. With that settled, P3-C's
census says what the numbers actually support:

- **65,388 dataset-level tests in the crossgen family; 0.25% clear BH.** Median 95% interval
  **±0.125 AUROC**.
- Of **16,548 persona-vs-raw tests, 5 clear** — four are `pc:kd_assertion` (the owner's own bank,
  three under a non-paper probe) and **one clears in the wrong direction** (`pc:deception − raw`,
  own100, −0.1524, q 0.0474).
- **At the reported operating points (L20/40/44, paper probe, mean pooling): 276 tests, 0 clear**,
  median signed effect **−0.0263**, only **34.4% even positive**.
- All re-derived independently by the lead; every figure matched.

So "persona coordinates beat raw probes" is not merely unproven in this corpus — **it is
unresolvable by design at n = 11 datasets, and the central estimate at the reported settings is
negative.** The project's strongest, best-tested result is the *negative* one: pooled
leave-one-dataset-out training with **no personas** scores 0.8545 vs 0.6763 single-source
(+0.178, CI [+0.143, +0.213]), and the gain is **diversity, not volume** (0.8500 at matched row
budget). The draft reports none of this.

### Lead QA — three re-derived, and two corrections that made findings STRONGER

Full record in `audit/phase3/QA_PHASE3.md`.

1. **P3-D's passP result: exact, but scoped to 6 of 18 cells.** A prompt-only probe (no model
   response) recovers 0.60–1.09 of the response-side transfer signal — **under `last` pooling at
   L40/L44 only** (mean 0.856, reproducing the quoted range to 3 dp). Under `mean` pooling it
   recovers **0.18–0.51 (mean 0.407)** at every layer; at L20 under `last`, 0.33–0.54. The
   *diagonal* needs no caveat: passP reads 0.83–0.93 in-distribution at both poolings. The finding
   stands with the scope in the sentence — and it is a **pre-registered control
   (`8_coords_ndif.py:270-278`) that was run and never reported.**
2. **P3-C's "median effect 0.044" is a median ABSOLUTE effect.** Signed, it is **−0.0263**. Said
   plainly the result is stronger, not weaker: not "an advantage too small to resolve" but
   "typically no advantage or a negative one, unresolvable either way".
   **This is the third median-absolute-quoted-as-signed in this audit** (F-73's median |gap|,
   DF-49's metric confusion, now this). → standing rule: state the sign convention.
3. **`claims.csv` row D-107 corrected, in the collaborator's favour.** It called his random control
   "UNDERPOWERED (n=1 draw)". His `persona_ood.py:61` sets `N_RANDOM_DRAWS = 5` and `:1004-1009`
   averages them, commented "averaged over draws so a single lucky basis cannot carry the
   comparison". The P8 QA had already overturned this and `docs/DRAFT_CORRECTIONS.md` §4 says it
   correctly — only the ledger row was stale. Fixed by `audit/scripts/phase3/p3_fix_d107.py`
   (idempotent). The narrower criticism is preserved: a mean of 5 bounds the centre, not the tail.

The other seven nominations were accepted as written, each already carrying its own honest caveat;
the reasons are tabulated in `QA_PHASE3.md` so the gap is visible rather than implied.

### What the draft misses — both directions, now with verdicts

The eight Phase-0 touchpoints were DOC-ONLY when written; most of their claims are now checked, so
`DRAFT_GAPS.md` gives each a verdict. Headlines:

- **A-1 (new, strongest).** The draft's "the principal components add nothing" fails at the
  replicated paper's **own** setting: the paper's deception figures are **k = 1**, and his own
  `persona_ood_ext_L*/summary.csv` has published **PC1 beating the published contrast at all three
  layers** (+0.016 / +0.027 / +0.025). His Table 3 carries "Extended PC1" but no "Published PC1".
- **A-2 / A-3.** The draft denies making the pooled comparison **his own branch made**
  (`loo_L20/40`, 2026-09-20), where the +0.184 margin falls to +0.048 / +0.023; and it lists the
  wording-controlled test as future work when it **was run** and came back inconclusive
  (+0.066, CI [−0.069, +0.189], P = 0.161). He must not promise that test as the fix.
- **B-1 / D-03.** His 4-vs-6 knowledge-denial axis, decomposed on one bank: HP-KR reads **0.873**
  merged, **0.592** vs sincere-ignorance, **0.912** vs plain-assertion — so ~0.28 of his headline
  HP-KR repair is the **plain-assertion pole, i.e. register, not sincerity.** In neither document.
- **In his favour:** his cross-behaviour orthogonalisation arms (`dec3_orth` / `syc3_orth`) quantify
  the register effect and appear in **neither** document; he reached the pooled self-criticism first;
  his 8192-d baseline is the *stronger* one, so his margin is conservative.

### Other results worth the owner's attention

- **passA (80 layers, 92,184 files) is a clean, unreported negative:** persona features beat raw at
  **2–5 of 80 layers** and the matched control at ≤17/80, while a 1-d **length** feature beats raw at
  more layers in all six cells. Raw peaks at **L31–33, not L40** — L40's advantage is partly raw
  being in a trough (`axes−raw` −0.034 at L40 vs −0.181 at raw's peak L31).
- **L40 flatters persona features specifically** (P3-C W2): L40 is in the top third of its own
  21-layer distribution for 62% of persona readings vs 22% of controls, worth ~0.05 of free margin.
  Arithmetic VERIFIED; the causal reading is MEDIUM with no p-value.
- **Stale foundations are mostly clean.** All 20 findings citing a boot-wave artifact were
  re-checked: no effect size, CI or p moved — only BH q, flipping exactly **one** verdict (F-31).
  paper16 old-vs-fp32 cannot reach any number. **F-108's leak priced at −0.013 to −0.056** of the
  +0.178 headline; the conclusion survives either way — **quote +0.125**.
- **Two new code defects:** `str(e)[:200]` survives at three sites, and `3_judge_ndif.py:168,:188`
  are **named nowhere in the ledger** — in the judge that decides which rollouts feed every persona
  vector. Also `27_crossgen.py:612` records only the bank **basename**, which is what made RF-16
  invisible and can do so again.

### Open, carried to Phase 4

- **CONFLICT inside the audit's own files:** `audit/SESSION_MINING.md:7` says **42** print-only
  ledger entries; `audit/phase1/sessions/findings_targets.csv` has **72** rows. Neither re-derived.
- **Seeds 0–4 carry only 3 features / 2 probes** in `passA`, `passC` and `<root>` (P3-D), so any
  6-seed mean over those stores may be invalid. Added to the traps list; not yet confirmed.
- Two prose sites still carry an uncorrected **"+0.079"** where RF-16 moved the number to +0.054.

**Phase 4 is next** — fresh agents, given only the brief, the ledger row and the raw files, re-derive
every HIGH-importance claim plus a random ~20% of the rest; mismatch rate reported and each fix
logged in `audit/CORRECTIONS.md`. That 20% sample is also what bounds the 461 claims Phase 2 did not
check, which is why the owner's wave-5 scope decision works.

## W4 + WAVE 6 LAUNCHED (2026-09-23) — the reframe changes what is worth checking

**Context: the owner has decided to reframe the paper.** After Phase 3 she asked for a blunt
assessment and accepted it: the method does not beat its controls (276 persona-vs-raw tests at the
reported operating points, **0 clear BH**, median signed effect **−0.026**, only 34% positive;
0 of 126 against the matched dataset-PCA control). The paper becomes a **methods-and-negative-results**
paper built on the control suite. That decision reorders the remaining audit work.

### W4 — the last owed piece of Phase 2 (2 agents)

`audit/PHASE2_PLAN.md:37` specified a central control re-derivation and it was never done. Several
zero-shot claims still rest on a random-direction null that exists **only in a console log**
(`factcheck.py`, 2026-09-16: "95th pct |AUROC| 0.640 at one layer, 0.683 best-of-21"). Individual
nulls have been recomputed piecemeal since (the lead's 2,000-draw null for F-107 in M1; P3-D's
per-dataset nulls), but nothing central. **Under the reframe this becomes load-bearing** — if the
thesis leads with "persona features do not beat their controls", the null cannot be a log line.

- **W4-A** → `audit/phase2/controls.csv` + `audit/phase3/CONTROLS.md`. One citable table: random
  null (with n_draws, seed, p50/p95/p99/max), matched dataset-PCA, and raw, per store × layer ×
  pooling × probe × dataset. Told to find and consolidate any stored `zeroshot_nulls.json` before
  redrawing, to keep **trained random probes and zero-shot random directions in separate rows**
  (DF-31's second protocol mix), and to state its sign convention in the header.
- **W4-B** → `audit/phase3/CONFLICTS_RESOLVED.md`. The two conflicts parked at the end of Phase 3:
  (1) `SESSION_MINING.md:7` says **42** print-only ledger entries while `findings_targets.csv` has
  **72** — two audit deliverables contradicting each other in print; (2) P3-D's report that **seeds
  0–4 carry only 3 features / 2 probes** in `passA`, `passC` and `<root>`, which if true makes any
  6-seed mean over those stores invalid. Both answerable from files.

### Wave 6 — the owner's own unchecked claims (5 agents now, 2 held)

**Her instruction: check the unchecked claims from her side.** Her ledger is already 110/110 after
wave 5, so "her side" is **193 rows**: 83 owner-doc, 85 memory-note, 25 session-only.

**Why hers and not his** (the recalibration the reframe forces, recorded so it is not re-litigated):
his draft body is 176 of the 268 remaining rows, but it supports the positive framing being dropped
and **all 18 of his headline rows are already checked**. Her docs and memory notes are what the
reframed thesis will be **written from**. Reassuring signal: all 12 memory notes sampled in W5D held.
Worrying one: the worst-travelling error the audit found (0.374, into nine documents) was in an
owner doc.

Selection is re-runnable: `audit/scripts/p2_wave6_select.py` → `audit/phase2/wave6_batches.csv`,
grouped **by document** so each agent loads one context and one store.

| batch | n | document(s) |
|---|---:|---|
| **W6A-thesis1** | 17 | `docs/thesis/persona-coordinates-70b-results.md` (earlier lines) |
| **W6B-thesis2** | 18 | same file, later lines |
| **W6C-briefing** | 23 | `docs/COLLABORATOR_BRIEFING.md` |
| **W6D-ownerdocs** | 25 | `cap/docs/DESIGN.md`, sarcasm-pilot, REVIEW_REQUEST_POOLING, ROADMAP_TO_DEFENCE, `kd/docs/NEXT.md`, NOVELTY_ASSESSMENT |
| **W6E-memory-q** | 32 | the Q-series memory notes (q2,q3,q4,q5,q6,q8,q10,q11) |
| *(held)* W6F-memory-rest | 53 | the remaining memory notes — **to be split in two before launch** |
| *(held)* W6G-session | 25 | session-only findings (b035ccb3, 1ac8bb19, 6d4b2181, dc605d01) |

Each agent was given the standing trap list plus the two newest: **never quote a count or effect
without its metric, sign convention and dataset set** (three median-absolute-as-signed errors so far
— F-73, DF-49, P3-C's "median effect 0.044" whose signed value is −0.0263), and **verify any audit
edit that touched its own lines**, since the audit has now had to correct two of its own corrections
(DF-49, DF-50). W6A/W6B were additionally told that their file is the one the thesis gets written
from, so an error there goes straight into it.

Rolling QA unchanged: each batch nominates its 2 least-certain claims and the lead re-derives them
from raw files before acceptance.

**Still outside `audit/` and awaiting the owner's word** (twenty minutes' work, not started):
three prose sites still quote the superseded three-rung decline **+0.202 → +0.115 → +0.079**
(`docs/NOVELTY_CHECK_PROMPT.md:67` and `:84`, `docs/OPEN_DECISIONS.md:53`) where F-110's
like-for-like layer-averaged figure is **+0.120 → +0.103 → +0.054**; Phase 0's `find -newer`
stray-write scan, killed for low memory and never re-run; and **DF-48**, the
`run_scale_chain.ps1:197-203` bank-overwrite bug, unfixed because the one-character fix would
produce two banks where the downstream reader expects one.

## INCIDENT 2026-09-23 ~13:0x — session limit killed 7 agents, and the LAB PC IS NOW UNREACHABLE

**1. All 7 agents died on a rate limit (HTTP 429, session limit), not on an error in their work.**
W4-A, W4-B and W6A-W6E were launched together; the concurrency blew the session budget. **No
deliverable was written** - verified: no `controls.csv`, no `CONTROLS.md`, no
`CONFLICTS_RESOLVED.md`, no `claims_batch_W6*.csv`. So there are no partial or misleading results in
the tree. What survives is scripts (`audit/scripts/phase3/w4_*.py`, `w4b_*.py`,
`audit/scripts/claims/W6B_local.py`) and one usable intermediate,
`audit/phase3/out/w4_zeroshot_axes.csv` (2,441 rows of zero-shot AUROCs harvested from local
matrices). **Lesson: cap concurrency at ~3-4 agents, not 7.**

**Two agents had got somewhere before dying, per their last lines:**
- W4-B had already verified conflict 2's premise - *"The asymmetry is real and wider"* - i.e. the
  seeds-0-4 feature-coverage gap is confirmed AND affects more stores than P3-D reported. It died on
  the consequential half (which findings take a 6-seed mean). **DOC-ONLY until re-derived.**
- W6B had reached "now let me write the main verification script".

**2. The lab PC (`user4@100.85.217.68`) stopped responding at ~13:0x.** `ssh -o BatchMode=yes` times
out on port 22 on two attempts, and **ping fails with 100% loss**, so this is the host or the tailnet,
not sshd. It was reachable at 11:01 and the agents used it successfully at 12:52. **Reported to the
owner, per the brief's rule.** Not diagnosed further - the audit does not touch that machine's state.

**Open item created by the outage, to be closed when the host returns:** W4-A uploaded
`w4_null_lab.py` (9,848 B) to `audit_scratch/scripts/` at 12:52 and **may have launched a
null-draw run before it was killed**. The standing rule after an interrupted remote call is to check
for orphan processes before re-issuing - **that check could not be performed**, because the host went
away. First action when it is back: `Get-Process python` on the lab PC, then re-read
`audit_scratch/out/` before re-running anything. Nothing the audit ran writes outside
`C:/Users/user4/audit_scratch/`, so the owner's results are not at risk either way.

**What is still doable with the lab down:** a large part of wave 6. `audit/phase2/layer_sensitivity.csv`
(78,516 rows), `provenance.csv`, the `run-crossgen` matrices, `audit/claims.csv` and
`audit/cache/labpc/` (run-premise, run-paper16, run-kd, kd, cap, logs, loose) are all local. What is
NOT doable: W4-A's random-direction null draws (they need the activation tensors, which are lab-only),
and any claim whose only evidence is a lab-only store.

**Decision deferred to the owner** (asked at ~13:0x): wait for the lab, or run now the subset that is
answerable from local files and the cache, marking lab-only items DEFERRED rather than NO RAW DATA -
because "no raw data" would be a false verdict caused by an outage, not by the evidence.

### Lab outage DIAGNOSED (2026-09-23) — it is this laptop, not the lab PC

The owner confirmed the lab PC is on and in use. `tailscale status` on the laptop:

```
100.113.151.108  sat                  suraiyashanaz1@   windows  offline
100.85.217.68    desktop-h3erd3u...   thesis.t2520874@  windows  -
health: Tailscale cannot connect because the network is down.
BackendState: Running   Self online: False
```

**The laptop's own tailnet node is offline.** General internet is fine - HTTPS to github.com returns
200 in 1.7 s - so this is Tailscale specifically, not connectivity. ICMP to 8.8.8.8 also fails, which
is normal on a filtered network and not diagnostic. The lab PC's status shows `-` only because we
cannot reach the coordination server to ask. **Fix is on the laptop: `tailscale up` (may need re-auth
if the node key expired), or toggle it from the tray.** Told to the owner.

**Wave 6 relaunched LOCAL-ONLY, 3 agents (not 7):** W6C-briefing (23), W6E-memory-q (32),
W6A-thesis1 (17). Each was told the lab is unreachable, told not to call any lab tool, pointed at
`layer_sensitivity.csv` / `provenance.csv` / the working-tree matrices / `audit/cache/labpc/` / his
committed CSVs via `git show`, and instructed to mark lab-only items **`DEFERRED (lab unreachable)`
rather than `NO RAW DATA`** - because "no raw data" would be a false verdict caused by an outage
rather than by the evidence. W6B, W6D, W6F and W6G stay queued.

**W4-A cannot run at all until the lab is back** - the random-direction null needs the activation
tensors, which are lab-only. Its local harvest survived: `audit/phase3/out/w4_zeroshot_axes.csv`,
2,441 zero-shot AUROCs from local matrices.

### Wave 6 results — W6C and W6E in, both lead-QA'd (W6A still running)

**Running total: 286 of 692 claims checked** — 157 HOLDS, 62 PARTLY HOLDS, 51 MATCHES HIS DATA,
12 PARTLY MATCHES, 2 DOES NOT HOLD, 1 REVISED, 1 DOES NOT MATCH. Owner-side remaining: owner doc 60,
memory note 53, session 25. QA record: `audit/phase2/qa_wave5.md` (wave 6 section at the end).

**W6C-briefing — 23: 19 HOLDS, 3 PARTLY, 1 DOES NOT HOLD, 0 outright DEFERRED.** Most of this file's
subjects are HIS numbers, which are committed, so the lab outage barely bit.
- **O-036 DOES NOT HOLD, and the audit had repeated the same error.** The briefing's closing sentence
  sets *her* 110-pair median/sign-oriented mean (0.722 / 0.735) against *his* 0.732, which comes from
  his own 6x6 matrix. Lead re-derivation from his branch: **mean 0.7320** (= the draft's 0.732),
  **median over sources 0.7500**, median of all 30 cells 0.7887, **sign-oriented mean 0.7850**.
  **The argument gets stronger on his own data** — on hers the median sits *below* the mean, which
  does not support "you picked the lowest"; on his, every alternative is above 0.732.
  **`audit/phase3/DRAFT_GAPS.md` A-13 repeated the mixing and has been corrected in place.** That is
  the **fourth** correction to the audit's own work (DF-49, DF-50, P3-C's median-absolute, now this).
  One correction to the batch: its "his median 0.7608" reproduces under neither reading; use 0.7500
  / 0.7887.
- **O-033:** "+0.03" is exact (+0.0356) but "clears no significance test at any layer" is **false** —
  content clears at L0, L12, L36 mean-pooled. And "two to four times that" is the very ratio F-94
  says must never be quoted.
- **No audit correction to this file is wrong** — all four 09-23 changes touching these claims verify.

**W6E-memory-q — 32: 26 HOLDS, 6 PARTLY, 0 DEFERRED.** The Q-series numbers all live in local
git-tracked `run-coords/*.json` (mtimes 09-16/17, before the RF-13 wave), so the outage cost nothing.
- **M-057 is the strongest single result for the reframe, and the lead re-derived it independently**
  from `layer_sensitivity.csv` across **all six pooling x probe cells** of `passA` (seed 42, 10
  datasets, 80 layers): **persona axes beat raw at only 2-5 of 80 layers in every cell**, and the
  wins sit at the extreme ends of the stack (L77-79, plus a few at L0-1 / L7-9). **At both reported
  layers, L40 and L44, axes lose in all six cells**, by 0.013 to 0.066. This **independently
  reproduces P3-D's "2-5 of 80"** by a different route.
  Correction to the batch: its "raw beats axes at every layer 0-77" holds for the one cell it read
  (`mean/mms`), not for all six. Stated as "2-5 of 80 in every cell" it is both accurate and stronger.
- **M-072/M-073:** "top10pct beats mean, best +0.053 at L28" **understates the effect ~2x at the
  literature layers** — the largest zero-shot pooling gain is **+0.1027 at L20**, collapsing to
  **+0.005 at L40**. → DOC_FIXES.
- **Q4 and Q10 are solid** — all six Q4 rows reproduce to 4 dp including the bootstrap CIs and the BH
  pattern; Q10's note is *conservative* (L40 last also clears, p = 0.0130).
- Presentation defects worth fixing, numbers correct: M-054 compares a 5-set `paper` probe against a
  full-corpus `lr_reg10` probe (the reversal survives holding the probe fixed); M-067's "17 methods"
  is 23 rows (17 is the count carrying both L20 and L22).
- P3-D's seed-coverage warning **did not bite here** — no Q claim rests on a multi-seed mean.

**W6A-thesis1 — 17: 13 HOLDS, 4 PARTLY, 0 whole-claim DEFERRED.** Needed no lab access: `run-coords/`
sits in the working tree with 37 result JSONs and was **never listed in the audit's file map as a
laptop store** — check it before deferring anything for the rest of wave 6.
- **O-066: the audit's own DF-49 repair was itself defective, confirmed and now fixed.** The path the
  lead added to `...70b-results.md:497` (`run-coords/confounds_L44_last.json`) **is not in the repo**
  (lab-only; cached copy exists); the key is nested at
  **`confounds.length_only_auroc[].auroc_from_length`**, not top-level; and the edit contradicted a
  sentence two lines below which was itself false ("only the sycophancy value is in a result file" —
  all ten are). **Fifth correction to the audit's own work, and the second inside the DF-49 repair.**
  Pattern named for the record: *a number carried into a new document without re-opening the file it
  came from.*
- **O-073 / A-1 settled from the PDF by the lead, sharper than either agent had it.** Counts over all
  14 pages: `k=1` **12×** (pp. 12-14), `k=3` **9×** (pp. 6, 12-13), **`k=5` zero times**. The paper
  uses **two different k, one per behaviour** — *deception* figures are **k=1**, *sycophancy* k=3 —
  so "the paper's k" is ambiguous unless the behaviour is named. A-1's premise holds; M2's AD-1 is
  independently confirmed (our default k=5 is a value the paper never uses). **And a fourth
  independent confirmation of DF-50:** both deception figure titles read **"[HPC excluded]"**, i.e.
  the paper drops the dataset its own Table 1 declares.
- **O-070** sits under "What holds" but both its gaps are non-significant in their own file (q 0.86,
  0.67), and across 126 crossgen configurations the same best-vs-best gap has **median −0.031**,
  positive in only 28/126. O-071 and O-072 are the document's own counter-evidence and belong beside it.
- **O-057/O-067:** the hf275 columns are measured against the **published** Assistant Axis, not "each
  bank's own assistant direction" as §2.8/§7.1 say. Conclusions unchanged (≤0.04); wording wrong.
- **O-064 newly verified from raw vectors:** the ≤0.001 count-bug bound is real — max |Δcos| 0.00065
  over 80 layers, old-vs-fp32 cos ≥ 0.99996.

## THE FIVE PRE-PHASE-4 ISSUES — ALL RESOLVED (2026-09-23)

The owner: *"solve all the issues you mentioned... we have clear everything in phase 3 properly then
we can move into phase 4."*

### 1. W4, the central control table — DONE, and the premise was wrong

`audit/PHASE2_PLAN.md:37` said the random-direction null existed **"only in a console log"**. **It
does not.** `run-coords/zeroshot_nulls.json` sits in the git working tree with **1,000 draws per
cell**, per dataset, with p-values and a `beats_random_p95` flag, across four blocks. Nothing had to
be redrawn and the lab was never needed.

- **`audit/phase2/controls.csv`** — 29,414 rows (42 zero-shot null + 29,372 trained-control),
  built by `audit/scripts/phase3/w4_build_controls.py`.
- **`audit/phase3/CONTROLS.md`** — the prose version, with the sign convention stated in a header
  because five audit self-corrections came from a number quoted without one.

**What the table says, and it is the most citable thing the audit has produced:**
- The **deception axis beats its own 1,000-draw null on 5 of 9 datasets at L40/mean** (4 of 9 at
  L44/last). It is **inside the null on HP-KR** — the draft's central dataset — 0.6436 against a p95
  of 0.9377. On `mask` it is **inverted** (0.4669) at both layers.
- **Sleeper datasets: 0 of 4 beat the null.** AUROCs of 0.87-0.91 look strong and sit inside a null
  whose p95 is 0.92-0.93 — those datasets are easy for *any* direction. This is what a null is for.
- **sycophancy_mt: the sycophancy axis is significantly INVERTED** (0.3131; oriented 0.6869 > p95
  0.6304), the **assistant axis** is the one that reads it correctly (0.7102), and the **deception
  axis is inside the null in all four cells**. DF-31 with the numbers attached.

### 2. The seed-coverage conflict — SETTLED, and it is wider but harmless to existing numbers

`audit/scripts/phase3/p3_seed_coverage.py` over `layer_sensitivity.csv`. W4-B's dying note
("the asymmetry is real and wider") was right: **7 stores, not 3.**

| store | seeds 0-4 are missing |
|---|---|
| `<root>`, `passA`, `passC` | **8 features**: `axes_resid`, **`datapca`, `datapca@2`, `length`, `random`, `random@2`**, `pc:unified`, `pc:unified@2` |
| `banks/{hf100,own100,paper16_famcentred,paper16_shared}` | 2: `axes_resid`, `pc:unified@2` |

**Consequential half, which nobody had reached: NO claim anywhere takes a multi-seed mean.** A grep
over the ledger, `claims.csv`, all 303 claim notes and the thesis doc returns nothing. **So no
existing number is invalid.**

**But the finding inverts into a real limitation:** in `passA`/`passC`/`<root>` the missing features
are exactly the controls, so **every control comparison there — axes vs `datapca`, vs `random`, vs
`length` — exists at seed 42 only**, while raw-vs-axes replicates across six seeds. The negative
result stands (a single-seed negative is still negative), but the *size* of the control margin has no
seed-level error bar in those stores. → `LIMITATIONS_AND_FUTURE_WORK.md`.

### 3. The owner's remaining 121 claims — 3 agents launched (W6B 18, W6D 25, W6G 25); W6F (53) follows

### 4. The three stale +0.079 prose sites — FIXED

`docs/NOVELTY_CHECK_PROMPT.md:67` and `:84`, `docs/OPEN_DECISIONS.md:53`. Each now gives **both**
readings and says which is which: the single **L40 cell** +0.202 → +0.115 → **+0.079**, and the
**layer-averaged like-for-like** +0.120 → +0.103 → **+0.054** (F-110), with the 70B-vs-raw **−0.013**.
The decline survives either way and is steeper on the like-for-like reading.

### 5. DF-48 and the stray-write scan — BOTH DONE

**DF-48 fixed** in `scale/scripts/run_scale_chain.ps1:197-203`, original backed up to
`audit/fix/originals_repo/run_scale_chain.ps1.before-DF48` (md5 verified). The loop built both
`vectors_all` and `vectors_s23` banks and wrote **both to `axes_$tag.pt`**, so the unfiltered arm was
computed and silently overwritten — while the log claimed both were built. Each arm now writes
`axes_${tag}_$vt.pt`, and `axes_$tag.pt` is still produced from `vectors_s23` because
`run_scale_crossgen.ps1:53` reads exactly that path — **downstream behaviour is unchanged**, the
silent discard is gone. **The lab copy is NOT patched** (host unreachable); do it when Tailscale is back.

**The Phase-0 stray-write scan finally ran.** First two attempts returned empty and I nearly reported
"no stray writes" — the full-tree `find` was being killed by the timeout and `$?` after a pipe reports
`sort`, not `find`. Re-run per directory with honest timeout reporting. Result: **every modification
outside `audit/` is accounted for** — the authorised doc edits, the 8 guard-bug script fixes, and
today's DF-48 fix. **Two unaccounted writes found, both harmless:**
`cap/scripts/__pycache__/score_sarcasm_markers.cpython-310.pyc` and
`src/assistant_axis_ndif/__pycache__/pc_personas.cpython-310.pyc` — Python bytecode written when audit
scripts imported those modules locally. Both are gitignored (`.gitignore:2 __pycache__/`). The lab
runner sets `PYTHONDONTWRITEBYTECODE=1`; local runs did not. Logged for completeness, not a data risk.
`run-crossgen` has no new tracked, untracked or ignored files.

**Working tree now: 17 modified + `docs/DRAFT_CORRECTIONS.md` untracked.** Nothing committed.

## WAVE 6 — the owner's side, nearly complete (2026-09-23)

**371 of 692 checked** — 212 HOLDS, 89 PARTLY, 51 MATCHES HIS DATA, 12 PARTLY MATCHES, 4 DOES NOT
HOLD, 2 REVISED, 1 DOES NOT MATCH. **Her side is done except the memory notes:** ledger 110/110,
owner docs 103/103, session-only 35/35, abstract 8/8. W6F1 (26) + W6F2 (27) are running; after them
only his 176 draft-body and 92 note claims remain, which Phase 4's random sample bounds.

**DOC_FIXES is now 70 entries.** The three most serious found in wave 6, all lead-verified:

1. **DF-66 — the sarcasm gate was NOT pre-registered.** `docs/thesis/sarcasm-pilot-negative-result.md`
   says two gates "were registered in the design document, with thresholds, before generation", G2 =
   bag-of-words AUROC <= 0.65. **`0.65` appears in NO version of `cap/docs/SARCASM_DESIGN.md`** (count
   0 at all four of its commits). It first enters the repo in `cap/scripts/run_sarc_pilot.ps1` at
   **`b34ff2e`, 2026-09-21 21:48:54**, whose commit message is *"F-93: the sarcasm pilot fails its
   decisive check"* — **threshold and failure verdict committed together.** What v1 *did*
   pre-register (~0.75) **v2 deliberately removed**. The finding survives unchanged (0.921 and 0.947
   fail 0.65 and 0.75 alike), so restating costs nothing — but a pre-registration claim the repo's own
   history contradicts is the first thing a referee auditing "our controls are the contribution" finds.
2. **DF-63 — a pre-registered equivalence margin widened post hoc.** The thesis doc declares
   **+-0.02** at `:513` and concludes "equivalent within about **+-0.04**" at `:740`. All four stored
   tests read `equivalent_within_margin = False` (`run-coords/significance.json`, comparisons 7-10;
   the file stores no `margin` field). The conclusion is probably right — median gap 0.012 — but not
   by that route.
3. **DF-67 — `docs/REVIEW_REQUEST_POOLING.md` describes a split protocol the code does not use.**
   It claims "80/20 splits by item id"; `30_premise.py:106` uses `rng.permutation(n)` over **rows**
   and `group_split` appears **0 times**. This inflates the in-domain ceiling 0.9506 in its own table
   — **in the document being sent out for adversarial review.**

**S-32 — RF-13 finally bites a verdict.** The kd bank's largest effect (`pc:unified@2 - random@2`,
+0.2861, 79/90 pairs) reads **q = 0.067 not significant** in the git copy (4,032 tests) and
**q = 0.050 significant** in the lab copy regenerated by the boot wave (6,300 tests). Raw statistics
byte-identical; only the family grew. First case where the rewrite changes an answer.

**Unrecorded and worth the thesis** (W6G): the per-token probe was validated against sklearn to
**3.66e-06**, 0 mismatches on 12,000 rows — the only evidence that pipeline was ever validated, and
it is in no document. `cap_vs_default` and `cap_only` are **bit-identical at every layer**, so the cap
bank has **two** contrasts, not three. And **7 of 25 session findings turned out to be documented
already** — less was lost than the transcripts implied.

### Two further corrections to the audit's own work (now six and seven)

6. **The F-61 QA overclaimed.** It said the mis-copy "can be proved arithmetically". Two candidates
   fit and the audit cannot separate them: (A) kd's margin over *raw* divided by paper16's over
   *datapca* (0.0757/0.0064 = 11.8, no missing file); (B) F-61's console reading of **raw 0.6800 over
   109 pairs** (0.6861-0.6800 = 0.0061, 12.4), attested in the transcript but surviving in **no** copy
   — W6G checked git HEAD, the `3dfdb67` blob, the lab file and the cache, all 110 pairs. Corrected in
   `qa_wave5.md` to report both. Undisputed either way: **+0.0093 vs +0.0757 = 8.1x**.
7. **`CONTROLS.md`'s assistant-axis row needs a caveat.** It clears the *per-cell* p95 (0.710 vs
   0.634), but W6G reports a **length-only** scorer at **0.664** on the same dataset and a
   **best-of-layers** p95 of **0.694**, after either of which it falls below the null. Flagged in
   `CONTROLS.md`, not adopted (not re-derived). **The sycophancy inversion is unaffected** — 0.313 is
   far outside every null.

### OWNER ACTION — a credential, not a number

The lab PC's `AppData/Roaming/rclone/rclone.conf.bak-2026-09-22` (690 B, the same size as the live
config, written by an earlier Claude session at 17:19 on 09-22) sits on a **shared multi-user machine**
and holds a live Google Drive OAuth refresh token. **The audit verified its existence via metadata
only and did not read it**, and will not delete it — that is the owner's call.

**Lab PC is REACHABLE again** (W6G used it read-only all batch). Still owed there: patching the
**lab copy** of `run_scale_chain.ps1` with the DF-48 fix; the repo copy is done.

## PHASE 4 IS PREPARED AND READY (2026-09-23) — see `audit/PHASE4_PLAN.md`

**Phase 2 and 3 are closed. 424 of 692 claims checked; the owner's side is 100% complete**
(ledger 110/110, owner docs 103/103, memory notes 97/97, session-only 35/35, abstract 8/8).
Everything remaining is the collaborator's: 176 draft-body + 92 note claims.

**The owner chose the targeted Phase 4** over the brief's full scope (279 re-derivations), because
243 of 424 HIGH ratings were self-assigned by the batch agents that made them. The targeted sample is
**161 claims in 5 strata**, seeded and re-runnable:
`audit/scripts/p4_select_sample.py` -> `audit/phase4/sample.csv` (lead) and
`audit/phase4/sample_blinded.csv` (**agents get this one — no prior verdict, no tier**).

| tier | n |
|---|---:|
| T1 outright failures | 8 |
| T2 every claim the lead QA'd | 29 |
| T3 random HIGH-importance HOLDS (false-negative risk) | 40 |
| T3b random HIGH-importance PARTLY | 30 |
| T4 random 20% of the 268 unchecked | 54 |

**Purpose, stated plainly in the plan:** Phase 4 measures **how often the audit itself is wrong**. The
audit has already corrected its own work **nine times**, all caught by the rolling QA rule, and three
of those ran in the owner's favour. Every one had the same cause — *a number checked at the project
default instead of at the configuration the claim states.* Phase 4 puts a number on what is left.

**Clean state verified 2026-09-23 16:48:** no local python/ssh, **0 python processes on the lab PC**,
`git status` outside `audit/` = 18 entries (all authorised, all uncommitted).

### DF-48 lab copy patched (2026-09-23) — repo and lab now match

The last owed item outside `audit/`. Procedure as for the 09-22 guard-bug fix: verify the target is
the expected original, back it up, install, verify.

- lab target **before**: `155e6fc12c66f719f5b699a0684c6ff8` — byte-identical to the repo's pre-fix
  backup `audit/fix/originals_repo/run_scale_chain.ps1.before-DF48`, confirming the lab copy was
  unpatched and the same edit applied cleanly;
- backup written to `audit_scratch/backup_DF48_run_scale_chain.ps1`, hash re-checked as identical;
- lab target **after**: `37c12be11e726108e1220959ed70d5ce` = the patched repo copy, byte for byte;
- patched block confirmed present at lab lines 198-212 (`axes_${tag}_$vt.pt` per arm, then the
  explicit copy to `axes_$tag.pt` that `run_scale_crossgen.ps1:53` reads).

Logged in `audit/LABPC_WRITES.md` as an owner-authorised write outside `audit_scratch`.

**Tooling note for future sessions:** `audit/scripts/labpc_run.py` failed on this script, exiting 1
with **no stdout and no stderr** — the install was done by direct `ssh … powershell -Command`
instead, which worked first time. Nothing was written by the failed attempt (verified: target
unchanged, no backup created). If `labpc_run.py` goes silent, check the lab state before retrying and
fall back to direct ssh.

### Two session-hygiene lessons, written into `audit/PHASE4_PLAN.md` (2026-09-23)

Both came from the owner catching the audit, not the other way round.

**1. An agent finished its work and then burned 35 minutes re-verifying it.** W6F2 handed back a
complete report, then continued in a loop titled *"Verifying all 27 claim notes exist"* —
**35m54s and 355.1k tokens**. The lead confirmed all 27 notes, all 27 CSV rows and all 27 merges in
about one second and stopped it with `TaskStop`. Nothing was lost; a batch's budget was.
→ **Every Phase 4 agent prompt now carries a verbatim line forbidding self-verification of outputs**,
because the lead verifies every batch on merge anyway.

**2. "Is anything running?" was answered from the process table, and the answer was wrong — twice.**
The lead checked local `python`/`ssh` and `Get-Process python` on the lab PC and reported "nothing is
running" while W6F2 was live. **An agent is API calls: it appears in the task list, in neither process
table.** The owner spotted it.
→ This is the **same failure mode as all nine of the audit's self-corrections**: checking the right
thing in the wrong place. It is now stated at the top of the Phase 4 plan's "Session hygiene" section,
alongside the rule that a hand-back does not always end an agent.

---

# PHASE 4 EXECUTION LOG (live section — 2026-09-23)

Plan: `audit/PHASE4_PLAN.md`. Agent-facing rules: `audit/phase4/AGENT_INSTRUCTIONS.md`.
Adjudications: `audit/CORRECTIONS.md`.

## Pre-flight (all VERIFIED 2026-09-23 ~17:50)

- `git status --porcelain -uall` outside `audit/` = **18 entries** (17 modified + untracked
  `docs/DRAFT_CORRECTIONS.md`), matching the plan's expected state. Nothing committed.
- `audit/scripts/p4_select_sample.py` re-run: **byte-identical output** (sample.csv
  md5 `644957c88a409a718498cb050b780419`, sample_blinded.csv `4c63836b3aa176f0507c95cc66ae59a8`).
  161 rows, 5 tiers, seed 42. Reproducible.
- Lab PC `user4@100.85.217.68` (DESKTOP-H3ERD3U) **reachable**, **0 python processes**.
- Task list empty at launch (checked the TASK LIST, not the process table — the 09-23 lesson).
- Shared computed inputs present: `audit/phase2/layer_sensitivity.csv` (7.9 MB),
  `controls.csv` (3.5 MB), `provenance.csv`, `audit/cache/` (236 MB).

## Batching

`audit/scripts/p4_make_batches.py` -> `audit/phase4/batches/*.csv`, **by source**, 14 batches,
161 claims, no tier and no prior verdict in any of them (the independence rule):

| batch | n | source |
|---|---:|---|
| D1 D2 D3 | 11 each | the collaborator's draft PDF |
| D4 | 10 | the collaborator's draft PDF |
| F1 F2 F3 | 13 each | `kd/docs/FINDINGS.md` ledger |
| H1 H2 | 12 each | the collaborator's branch results |
| M1 | 10 | owner's auto-memory notes |
| M2 | 9 | owner's auto-memory notes |
| O1 | 13 | owner's docs |
| O2 | 12 | owner's docs |
| SA1 | 11 | session transcripts (9) + abstract (2) |

## Tooling built before launch

- `audit/scripts/p4_make_batches.py` — deterministic split, asserts 161.
- `audit/scripts/p4_merge.py` — joins `rederive_*.csv` to `sample.csv`, collapses both verdict
  vocabularies onto four families (HOLDS / PARTLY / FAILS / UNRESOLVED), classifies every row
  **AGREES / DIFFERS / UNRESOLVED / NEW**, and prints the **mismatch rate by tier**. T4 rows carry
  prior verdict `(unchecked)` so they classify as **NEW** — they bound the unchecked corpus rather
  than measuring audit error. Also reports duplicate ids across batches and ids not in the sample.
- `audit/CORRECTIONS.md` — created, with the three adjudication outcomes defined up front
  (AUDIT WAS WRONG / PHASE 4 WAS WRONG / BOTH PARTLY RIGHT) so the log cannot quietly become a
  list of the audit's wins.

## Waves

- **Wave 1 LAUNCHED ~17:55: F1, F2, F3, O1 (52 claims), 4 agents.** Four, not seven — seven blew the
  session rate limit on 09-23. Every prompt carries the verbatim no-self-verification line.
- **Wave 1 COMPLETE (F1, F2, F3, O1 — 52 claims).** Every batch delivered 13/13 rows, ids exact, no
  duplicates, no ids outside the sample.
- **Wave 2 LAUNCHED: O2, M1, M2, SA1 (42 claims).** Remaining after this: D1-D4, H1, H2 (67).

## THE HEADLINE PHASE 4 RESULT SO FAR (51 of 161 compared) — read `audit/CORRECTIONS.md`

**Raw DIFFERS rate 37.3% (19/51). Adjudicated audit-error rate 5.9% (3/51). Disagreements about a
NUMBER: zero.** Every one of the 19 disputes is about how strictly to grade a sentence whose number
both readers reproduce, usually to 4+ decimals.

**14 of the 19 are the same shape** — audit `PARTLY HOLDS` vs agent `HOLDS` — and **in 12 of those 14
the agent's own `notes` cell states the very defect the audit's PARTLY rests on.** The agent measured
it, wrote it down, and then graded the claim HOLDS.

### The meta-finding, and it is not what the plan predicted

The plan said every prior audit self-correction came from *"a number checked at the project default
instead of at the configuration the claim states."* **None of the three confirmed audit errors is
that.** All three are the same fault, and it runs in both directions:

> **the verdict cell does not match the checker's own notes.**

- **F-01** (`HOLDS` -> `PARTLY HOLDS`) — *the clearest result of Phase 4.* The audit's note says, in as
  many words, *"The entry attaches a superseded-row-set caveat to the current number"* — and graded it
  HOLDS. The Phase 4 agent found the identical defect independently and graded it PARTLY. **F-26 was
  graded PARTLY for a structurally identical fault**, so this is a demonstrably inconsistent grading
  boundary inside the audit's own ledger, not a missed check.
- **F-70** (`HOLDS` -> `PARTLY HOLDS`) — the audit wrote *"checked elsewhere"* about "1 of 384", a
  number in the claim's own `stated_numbers`, and graded HOLDS. The agent checked it: **"1 of 384"
  matches no significance family anywhere** (git 192/96; lab 4,032-6,300), consistent with the audit's
  own Phase 1 finding that F-04's 384-test run was overwritten the same evening.
- **O-022** (`HOLDS` -> `PARTLY HOLDS`) — the audit recomputed four numbers to 12 significant figures
  and never read the sentence they sat in. `COLLABORATOR_BRIEFING.md:354` labels **0.676**
  *"(what the abstract compares against)"*; lead-verified from raw, the abstract's comparator is the
  collaborator's **0.7320325942** (`persona_ood_ext_L40/summary.csv`, `baseline` row) over **his 30
  pairs**, against our 0.676 over **110 pairs / 11 datasets**. Two different corpora, conflated.

**So the dominant error mode in this project's checking — on both sides, independently — is not
mis-measurement. It is a correct observation that never reaches the verdict.** This belongs at the
front of `REPORT_TECHNICAL.md`; it is worth more than the mismatch rate itself.

### Honesty note: the agent instructions changed mid-stream, and it biases the RAW rate

Batches F1/F2/F3/O1 ran WITHOUT a verdict-discipline paragraph; every later batch has one (HOLDS
requires the sentence to be right too; your notes must not contradict your verdict). **This makes
later agents agree with the audit more often, i.e. it flatters the audit.** Declared at the top of
`CORRECTIONS.md`; the raw rate is not comparable across the two groups, the adjudicated rate is
unaffected (the lead re-checks raw files either way), and the split gets reported.

### Project red flags surfaced by Phase 4 (not audit problems)

- **`run-scale/8b/coords` is INCOMPLETE:** 5 of 12 dataset dirs have **no `rows.jsonl`**
  (harm-pressure-knowledge-report, harm-pressure-kr-matched, insider-trading, sandbagging_v2,
  sycophancy_mt); only 3,093 non-alpaca rows across 7 datasets, all capped at 800. Anything assuming a
  complete 8B corpus needs re-checking. (F1 agent, independently.)
- **F-58's 3B/8B numbers DO survive** — the crossgen logs are append-mode, so the pre-mask block
  persists beneath the re-run. The audit's "no surviving evidence" was too strong.
- **BH families remain unstable** between git and lab copies of the same `significance.json` (kd bank
  6/4,032 git vs 65/6,300 lab; kd_s23 14/468 vs 84/5,544) — any "n of N cleared" sentence must name
  the copy and the date. Independently re-hit by two Phase 4 agents.

### M2 lands: 9 of 9 AGREE — and it is the first batch with the verdict-discipline paragraph

| instruction group | AGREES | DIFFERS | raw mismatch |
|---|---:|---:|---:|
| **A** pre-discipline (F1, F2, F3, O1) | 32 | 19 | **37.3%** |
| **B** post-discipline (M2 so far) | 9 | 0 | **0.0%** |

n=9 is far too small to call, and M-claims may simply be cleaner than F-claims. But the direction is
exactly what the declared bias predicts, and `p4_merge.py` now prints this split on every run so the
effect stays visible instead of being absorbed into one headline number.

**M2 added no adjudications (no DIFFERS) but three project findings:**
- **M-056's "0.006-0.014" erasure cost is probe-cherry-picked.** The full table in
  `run-coords/q3_q4_bootstrap.json` (`q4_table[*].cost_assistant`) spans **0.0053-0.0177**; the note's
  upper bound is the `lr_reg10` value and omits `mms` at L40 (0.0177). The erased vector is also this
  project's own `bank['axes']['assistant']` contrast direction, **not the published Assistant Axis** —
  the note's capitalisation invites the wrong reading (consistent with the earlier
  `paper16-assistant-direction-not-lu-axis` finding).
- **Layer-dependence is repeatedly dropped from the Q8 headlines** — both zero-shot (+0.002 to +0.103)
  and per-dataset token-vs-pooled results flip or vanish between L20 and L40, yet single-layer values
  are quoted as general. Same disease as the four headline margins that already failed a layer sweep.
- **`run-coords/confounds_L44_last.json` is cited as a repo path but exists only on the lab PC / in
  `audit/cache/`.** This is an **independent confirmation of the audit's own DF-49 repair defect**
  (logged in wave 6 as O-066) — found here from the raw side, by an agent that never saw that note.
- Useful provenance: **all eight `run-coords/` JSONs used are md5-identical between the git working
  tree and the lab PC**, so this batch has no git-vs-lab divergence.

### O2 lands: 12 of 12 AGREE — and the agreement is SUBSTANTIVE, not label-matching (QA'd)

Group B (post-discipline) is now **21 of 21 agreeing, 0 mismatches**, across two different sources.
The obvious worry is that the discipline paragraph simply nudges agents onto the audit's favourite
label. **Tested and refuted:** on all 9 rows where both sides said PARTLY, the lead compared the
*reason* rather than the label — **9 of 9 cite the same defect, several to the decimal** (M-072/073
both give +0.053@L28 vs max +0.1027@L20; M-056 both give 0.0056/0.0086/0.0139 and mms 0.0177@L40;
O-079 both measure 0.80–2.64% against a stated 1–3%). Table in `audit/CORRECTIONS.md`.

**This is the strongest evidence the audit has produced about itself** — an independent blinded reader
working only from raw files reached the same numbers AND the same reasons as Phase 2/3. It tests the
reasoning, not the label, which the mismatch rate cannot.

### O2's project findings — one of them is serious

1. **The paper's headline is k=1, and this project never ran k=1 at all.** §7.2's
   *"pc:deception is never better than raw at any layer >= 20"* is **corpus-specific**: joined per
   cell, pc:deception wins in **326 of 5,848** comparable cells at layer >= 20 — including **94 of 192
   in `scale_70b`** (k=3, the five paper datasets, max **+0.0822**), which is the configuration
   closest to the paper. Combined with the already-established fact that the paper uses **k=1 for
   deception** (k=5 is our default, k=5 appears nowhere in the paper), the negative result is stated
   more broadly than the evidence supports. **This bears directly on the reframe** and needs the
   owner's attention.
2. **§7.1 mixed-reference mislabelling, independently confirmed.** The column headed
   *"hf275 (our grouping)"* (-0.686/-0.662/-0.610/-0.516) is **not** a 276-role hf275 bank against its
   own assistant direction — it is the **100-role** `axes_hf100.pt` deception contrast against the
   **published** Assistant Axis (exact 4/4 match). Same swap produces the "0.75–0.80 on hf275" figure.
   Reached independently by the Phase 4 agent and by Phase 2/3 (O-057, O-067).
3. **Sarcasm gate G2 was not pre-registered — third independent confirmation of DF-66.** `<=0.65`
   first appears in `b34ff2e` at **2026-09-21 21:48:54**, *after* all three rollout files
   (21:26:59–21:40:50) and all three judge files (21:42–21:44). Neither 0.20 nor 0.65 is in
   `cap/docs/SARCASM_DESIGN.md` at any commit. **Mitigating:** the measured 0.921/0.947 are far from
   0.65, so the outcome cannot have been tuned.
4. The **+0.032 label lift** in §7.12 exists only in the lab log (`logs/validate.log`);
   `17_validate.py` writes accuracies and drops the baselines.

## PHASE 4 MIDPOINT (2026-09-23): 93 of 161 compared, 11 of 14 batches done

Done: F1 F2 F3 O1 O2 M1 M2 SA1. Running: D1 D2 D3 D4. Left: H1 H2.
**Adjudicated audit-error rate 6.5% (6/93); raw DIFFERS 28.0%.** Full detail in `audit/CORRECTIONS.md`.

### Three things Phase 4 has produced that are worth more than the rate itself

**1. S-33 — an audit "cannot be determined" is now DETERMINED.** The audit's self-correction #6 said
two candidate explanations for F-61's "+0.006" fit and *"the audit cannot separate them"*, because the
"raw 0.6800 over 109 pairs" reading survives *"in no copy"* (git HEAD, the 3dfdb67 blob, the lab file
and the cache were all checked, all 110 pairs). **The audit was looking for the wrong kind of thing.**
No file ever held 0.6800 — it was manufactured at read time by a script bug.
`kd/scripts/paper_comparison.py` took its axis labels from `d["rows"]` / `d["columns"]`, which are
**human captions, not label lists** (`'trained on (80% split)'`, 22 chars). `zip` iterated
**characters**, so the diagonal test compared single letters and most in-distribution cells survived
into the "off-diagonal" mean. **Lead reproduced both code paths on the same file:**

| code path | pairs | raw off-diag | pc:unified − raw |
|---|---:|---:|---:|
| fixed (`rows = cols = d["datasets"]`) | **110** | **0.6623** | **+0.0093** |
| buggy (`rows = d["rows"]`) | **109** | **0.6800** | **+0.0061** |

All three stated numbers fall out exactly. **F-61's headline "+0.006" is a bug artifact; like-for-like
is +0.0093 and the ratio is 8.1x, not twelvefold.** The bug is already fixed in the committed script.
Self-correction #6 should be updated from "cannot be separated" to "resolved".

**2. A CENSUS replaces the sample for the one error mode Phase 4 found.** All confirmed audit errors
have the same machine-detectable shape — *the checker wrote the defect into `notes` and graded the
claim clean* — so `audit/scripts/p4_verdict_note_conflict.py` sweeps all 424 checked claims for it.
**69 of 304 clean-verdict rows carry defect language** (positive control: 43% of PARTLY rows trip the
same patterns, as they should). It catches 4 of the 5 then-confirmed errors. **It misses O-022, and
that miss defines the two sub-modes:**

| sub-mode | screenable? | bound |
|---|---|---|
| **observed, not graded** (F-01, M-007, M-054) | **yes** | **69 candidates** |
| **never observed** (O-022) | **no** | only re-derivation finds these |

**Deliberately NOT re-graded**: only the 6 errors confirmed by re-derivation are queued as ledger
fixes. Re-grading 63 more on the strength of a regex would repeat the exact fault being measured. The
classification is owed, then it is the owner's call.

**3. An independence BREACH, self-reported.** The SA1 agent's `grep` for a claim's wording crossed into
`audit/DOC_FIXES.md`, `audit/phase3/DRAFT_GAPS.md` and `audit/PROGRESS.md` and surfaced a prior
verdict before it had verified anything. **It disclosed this unprompted.** A-10 is marked
independence-compromised and excluded from the rate. **The audit's fault, not the agent's** — the
instructions named forbidden files but never said to scope searches away from `audit/`, and a
repo-wide grep for a claim's own wording is the first search any agent runs. Fixed from D4 onward;
**eleven batches ran without the guard**, so a silent breach is possible in any of them. It biases
only one way — contamination inflates agreement and **deflates** the measured error rate.

### Two more project findings from this wave

- **M-080 is thesis-level, not a grading question:** on **HP-KR, the draft's central dataset, a
  length-only baseline scores 0.776 — HIGHER than the deception axis's own zero-shot reading there**
  (0.6436 at L40/mean, 0.5830 at L44/last, both inside their random null).
- **S-31: `run-scale/8b` axes were overwritten.** Both the `all` and `s23` build blocks end
  `wrote .../run-scale/8b/axes_8b.pt` — the same path — so the s23 build overwrote the all build and
  only the s23 file survives. Same class as DF-48, in a different script.

### D2 lands — 4/4 AGREE, plus the first 7 T4 (previously unchecked) rows, all HOLDS

**Five problems with the collaborator's draft, none of them audit problems. Two are serious.**

1. **NOTHING in the draft is recomputable from raw.** `.../activations` on his branch is **empty** —
   no pooled activations are committed anywhere. Every check of his side is therefore (a) internal
   consistency between his tables, prose and figures, and (b) recomputation of *derived aggregates*
   (off-diagonal means, column means, per-dataset averages) from his committed matrices. **This must
   be said plainly in the report**: his headline numbers are attested, not reproduced. (Consistent
   with RF-2 and the owner's decision not to request his files.)
2. **SELECTIVE REPORTING — three persona banks exist, the draft reports two.** For the 8v7 axis on
   HP-KR at L20: paper16 original **0.103**, `persona including knowlede denial` **0.148**,
   `persona knowledge denial modified` **0.165**. The third is **never mentioned in the paper**, and
   **his own summary.md says it makes every leading arm worse at every depth**. A referee who finds
   this will treat it as bank selection.
3. **Bank-extraction noise is unquantified and exceeds the reported intervals.** Every bootstrap CI in
   the paper covers **example sampling only**, yet the "mod" bank re-extraction moved HP-KR by
   **0.036-0.077** — larger than those intervals. The error bars understate the real uncertainty.
4. **Table 3's "Random directions (k=3)" is a TRAINED probe averaged over 5 draws at seed 42**, with
   no spread stored; the table and prose both read as a single random basis. (This is the audit's own
   named trap — *a trained random probe is not a zero-shot random direction* — now found in his draft.)
5. **Balanced accuracy 0.500 in Table 2 means "flags nothing"** (recall 0.0, specificity 1.0), not
   chance. The caption does not say so; the code's docstring does.

Also: **D-099's second clause cites the wrong table.** "Matches or beats the published 8v7 contrast"
is true only under train-set averaging with a +0.009 margin at L20 HP-C, and the cited **Table 7 has
no published-contrast column** — while **Table 6** (zero-shot, same appendix) shows the published
contrast **ahead by 0.078**. A reader checking the citation would conclude it is false.

### D1 lands — and it is the most damaging batch for the draft so far

**D-028 is the sixth "defect in the notes, clean verdict" case, and it is the draft's headline
contribution number.** The audit's own note opens *"**AUDIT FLAG: the headline 0.220 is HP-KR-driven**"*
and the verdict cell reads a clean MATCHES. The blinded agent measured it: **drop HP-KR as train and
test set and the layer-20 gap collapses from +0.220 to +0.009** (0.9086 vs 0.9000), because the PC3
probe inverts on HP-KR. The draft's p.2 Contributions bullet quotes +0.220; **his own Table 3
excl-HP-KR column already shows the collapse.** HP-KR is the dataset the extended bank was
purpose-built to repair.

**Four more draft-level problems, all verified by the agent from primary sources:**

1. **HP-KR is ~91-96% solvable by surface wording alone — at or above the paper's own headline.** A
   **label-free regex on the response text** scores **AUROC 0.9125** ("negated know") and **0.9597**
   ("does not affirm knowledge"), against the paper's headline **0.919** for the knowledge-denial
   contrast. **The wording-controlled test the draft itself says would settle this appears never to
   have been run.** This makes the section-6 caveat far sharper than the draft states, and it
   compounds M-080 (length alone scores 0.776 there, above the deception axis's own 0.6436).
2. **The "why it inverts" mechanism does not survive its own controls.** The draft attributes the
   HP-KR inversion to "none of its deceptive personas denies knowledge". But the **persona-free RepE
   base probe also inverts** on HP-KR at L20 (0.109, CI [0.088, 0.132]), and **the same published
   contrast does not invert at L40/L44** (0.593/0.589) **despite identical persona composition**. A
   persona-composition cause cannot explain a layer-20-specific effect that also occurs without
   personas.
3. **Citation error, verified.** The bibliography titles arXiv:2602.01425 *"One probe won't catch them
   all: Towards targeted deception detection"*. Two PDFs in `papers/` (Mahadik & Skapars; Kirch et al.)
   both give that id as *"Building better deception probes using targeted instruction pairs"*. The
   paper is in neither the repo nor the lab PC, so the attributed finding is unverifiable.
4. **Provenance slip in Methods.** Ten of the twenty borrowed questions are **this project's own
   renderings** of M&S's ten deception *topics* — stated outright in
   `src/assistant_axis_ndif/pc_personas.py:260-262`; only the sycophancy ten are verbatim. The draft
   discloses reconstruction for persona prompts but **not** for these questions.

Free cross-checks the agent banked along the way: draft Table 1 reproduces exactly for all six
datasets; Table 3 to +-0.001 in both blocks; Table 6 zero-shot values reproduce; unified PC1 vs
question-family cosine independently re-derived at **-0.9800** (PC1 = 46.70% var) from
`run-paper16/vectors_s3/*.pt`; lab `run-kd/responses` = 36 files x exactly 150 lines = 5,400 with an
identical question-set hash across personas.

# PHASE 4 COMPLETE (2026-09-23) — see `audit/phase4/RESULTS.md`

**161 claims re-derived by 14 blinded agents; all 33 disagreements adjudicated by the lead from raw
files.** `p4_merge.py --strict` passes: 161/161, no missing ids, no extras, no duplicates.

| | value |
|---|---:|
| **Raw disagreement** | **30.8%** (33/107) |
| **ADJUDICATED AUDIT-ERROR RATE** | **6.5%** (7/107) |
| Disagreements about a *number* | **1** (S-33 — Phase 4 right, resolved a script bug) |

## The headline: the audit's error is concentrated in exactly one place

| tier | n | raw | **audit error** |
|---|---:|---:|---:|
| T1 outright failures | 8 | 50.0% | **0.0%** |
| T2 lead-QA'd | 29 | 27.6% | **3.4%** |
| **T3 HIGH-importance HOLDS** | 40 | 17.5% | **15.0%** |
| T3b HIGH-importance PARTLY | 30 | 46.7% | **0.0%** |

**All seven audit errors are `HOLDS` that should have been `PARTLY`. Not one claim was graded too
harshly.** This is precisely the risk T3 was built to measure — *"a wrong HOLDS puts a bad number into
the thesis unchallenged"* — and it materialised at 15%. Three consequences: **no true result was
wrongly retracted** (T1 is 8/8 upheld, 0 reversed); **the audit was lenient, never harsh**; and **not
re-deriving all 119 PARTLY rows was the right call** (T3b audit error 0%).

## The mechanism, and the plan predicted the wrong one

All seven errors — and 19 of the 21 Phase 4 errors — are **the verdict cell not matching the
checker's own notes**. The plan expected *"a number checked at the project default instead of at the
configuration the claim states"*; **none of the seven is that**. D-028's note literally opens
*"AUDIT FLAG:"* and the verdict is clean. **The dominant failure in this project's checking, on both
sides independently, is a correct observation that never reaches the verdict.**

A census for this one mode now exists (`p4_verdict_note_conflict.py`): **69 of 304 clean-verdict rows
carry defect language**. **Deliberately not acted on** — re-grading on a regex would repeat the fault
being measured. Owner's call.

## New coverage of the 268 unchecked claims (random 20%, n=54)

**77.8% HOLDS · 18.5% PARTLY · 1.9% FAILS** — against 71.7 / 26.4 / 1.9 for the 424 checked. **The
unchecked corpus is no worse than the checked one; there is no hidden pile of failures.** But the
single FAILS is **D-158**, the most consequential finding of the whole audit — so a low failure rate
does not make the remaining 214 safe to ignore.

## Limitations that cut AGAINST the rate (full list in RESULTS.md)

- **An independence breach occurred and was self-reported** (A-10, excluded). **11 of 14 batches ran
  before searches were scoped away from `audit/`**, so a silent breach is possible in any of them.
  **Contamination inflates agreement, so it DEFLATES the measured error rate.**
- **Instructions changed mid-stream**: raw 37.3% before the verdict-discipline paragraph, 23.6% after.
  The change **flatters the audit**; the adjudicated rate is unaffected.
- **`REVISED -> X` is structurally invisible to a blinded reader** (most of T1's 50% raw).
- **The lead adjudicated alone** — the same reader whose work is being measured. Itemised in
  `adjudications.csv` so each call can be checked.
- **The lead overstated D-158 and corrected it in-session** (weighting-shopping; see CORRECTIONS.md).

## Ledger fixes owed — QUEUED, NOT APPLIED

7 verdict changes (F-01, F-70, O-022, M-007, M-054, S-25 -> PARTLY HOLDS; D-028 -> PARTLY MATCHES)
and 5 note amendments (F-58, F-45, S-33 + resolve self-correction #6, D-237, O-041).
The ledger is not edited while it is the object of measurement; apply as one pass at the start of
Phase 5.

**NEXT: Phase 5** — `REPORT_PLAIN.md` and `REPORT_TECHNICAL.md` (`METHODS_CHAPTER.md` and
`LIMITATIONS_AND_FUTURE_WORK.md` already written). Both reports must carry the 6.5% measured
error rate and its limitations.

## End-of-phase state check (2026-09-23, VERIFIED)

- `p4_merge.py --strict` → **161/161**, no missing ids, no extras, no duplicates across batches.
- `git status --porcelain -uall` outside `audit/` = **18 entries** — unchanged from the Phase 4
  pre-flight and from `START_STATE.md` + the authorised edits. **Nothing committed.**
- Every batch file verified row-for-row against its blinded batch: 14/14 exact.
- **Lab PC process check — 6 python processes found, ALL belong to `user5`, none to `user4`.**
  Checked with `Win32_Process` + `GetOwner` (read-only). They are
  `SYCON-Bench` runs (`run_benchmark*.py gemma3-12b-it:bf16`) started 2026-09-23 18:58:15 from
  `C:\Users\user5\Downloads\`. **No audit orphan; nothing of the owner's is running.** The audit
  touched nothing of theirs.
  - **FOR THE OWNER:** another user is running GPU benchmark work on the shared lab PC as of this
    evening. Relevant if you plan to schedule anything there — not a data-safety issue.
- The batch-F2 agent's lab-pulled working files were **kept, not deleted**, as
  `audit/phase4/evidence_F2/` with a README stating provenance and that they are **not**
  md5-verified and therefore **not** in `audit/cache/MANIFEST.csv`. They include the append-mode
  `crossgen_scale_{3b,8b,70b}.log`, the only surviving source for F-58's 3B/8B rows.

## LEDGER FIXES APPLIED (2026-09-23) — Phase 4 is now fully closed

The brief's Phase 4 ends *"...report the mismatch rate, **fix the ledger**, and log each fix in
audit/CORRECTIONS.md."* The fixes were deferred until the phase closed so the ledger was never edited
while it was the object of measurement. All 12 are now in.

- Script: `audit/scripts/p4_apply_ledger_fixes.py` — **idempotent**, verified by re-run
  ("0 row(s) to change, 12 already applied").
- Backup: **`audit/claims.csv.before-phase4-fixes`**.
- **7 verdicts changed:** F-01, F-70, O-022, M-007, M-054, S-25 `HOLDS -> PARTLY HOLDS`;
  D-028 `MATCHES HIS DATA -> PARTLY MATCHES HIS DATA`.
- **5 notes amended, verdicts unchanged:** F-58, F-45, S-33 (**resolves audit self-correction #6**),
  D-237, O-041.
- All 12 claim notes in `audit/claims/` carry a dated `[PHASE 4 2026-09-23] amendment` section citing
  `CORRECTIONS.md` and `adjudications.csv`.
- **Ledger now reads HOLDS 297 / PARTLY 119 / FAILS 8** (was 304 / 112 / 8). 424 rows intact.
- `git status` outside `audit/` still **18 entries**. Nothing committed.

**Phase 5 can now be built from `claims.csv` directly** — it carries the corrected verdicts.

### The one Phase 4 item left open is a DECISION, not work

`audit/phase4/verdict_note_conflicts.csv` lists **69 clean-verdict claims whose notes contain defect
language**. 6 were confirmed as real audit errors by re-derivation and are fixed above. **The other
63 were found by a text screen, not by re-deriving the claim**, and have NOT been re-graded —
re-grading on the strength of a regex would repeat the exact fault Phase 4 measured. Options for the
owner are carried into `REPORT_PLAIN.md` as a decision with a recommendation.

### OWNER DECISION 2026-09-23: the 63 screen-flagged claims are REPORTED, NOT RE-GRADED

Options put to the owner: (a) report only; (b) hand-triage all 63; (c) re-derive all 63 blinded.
**She chose (a), the recommendation.** Rationale on the record: re-grading a claim on the strength of
a regex would repeat the exact fault Phase 4 measured, and the census is already informative as a
*bound* without any verdict changing.

**Consequences, to be honoured in Phase 5:**
- `claims.csv` keeps all 63 verdicts as they stand. **Only the 6 confirmed by re-derivation were
  changed** (plus D-028 = 7 verdicts total).
- `REPORT_TECHNICAL.md` reports the census as a measured property of the audit: **69 of 304
  clean-verdict rows carry defect language; 6 of the 7 sampled from that set proved to be real audit
  errors**; the other 63 are **flagged, unresolved, and bounded**.
- `REPORT_PLAIN.md` states the same in one short passage, as a known weakness with a number on it,
  **not** as 63 suspect results.
- `LIMITATIONS_AND_FUTURE_WORK.md` carries re-deriving them as the single highest-value follow-up,
  since the one comparable stratum (T4, random unchecked) yielded **D-158**, the audit's most
  consequential finding, from 54 claims.

# PHASE 5 STARTED (2026-09-23)

## PHASE 5 COMPLETE (2026-09-23)

**Deliverables written:**
- `audit/REPORT_PLAIN.md` — for the owner. Big picture in 10 sentences; strongest results ranked;
  negatives worth publishing; what failed or changed; red flags ranked by referee damage; what the
  draft gets wrong (with what it gets RIGHT first); promising leads; how much to trust the audit;
  **6 decisions with a recommendation each**; glossary.
- `audit/REPORT_TECHNICAL.md` — full detail, tables, paths, commits, scripts, plus a closing
  "what a referee attacks first" section with the state of each defence.
- `audit/DOC_FIXES.md` — extended 70 -> **75** entries with a new Phase 4 section (DF-71..DF-75).
- `audit/phase5/digest_{results,draft,story}.md` — the three agent digests the reports were built from.
- `audit/scripts/p5_stats.py` — recomputes every structural count in both reports.

**Lead verification done during Phase 5, not taken from the digests:**
- **The project's headline was corrected.** The pooled-training result is **+0.1254** under
  leave-one-CLUSTER-out, not the +0.1782 under plain LODO. Read directly from
  `audit/cache/labpc/run-premise/clustered/clustered_L40_mean_s42.json`:
  `summary.all.gain_lodo = 0.18145`, `summary.all.gain_loco = 0.12544`, `loco_positive = 10` of 11;
  `summary.clean` (dropping the 2 leaked targets) = 0.15039 LOCO / 0.16112 LODO. The file's own
  `leaked` field names **harm-pressure-knowledge-report** and **harm-pressure-kr-matched** — so the
  same dataset that carries the draft's advantage also inflates the owner's best result.
  **Report +0.125.**
- **`digest_story.md` carried two stale counts** ("231 of 692 checked") because it was written from
  `TIMELINE.md`, which predates Phase 3's close and all of Phase 4. Corrected in an appended section
  with the current 424 / 297 / 119 / 8 figures and the two Phase 4 findings that bear on its framing.

**The one coherent story the audit found:** HP-KR is implicated in all three of the project's biggest
numbers. It is ~91-96% solvable from surface wording (regex 0.9125/0.9597 vs the draft's 0.919);
it carries essentially the whole persona advantage in the draft (+0.220 -> +0.009 without it); and its
contamination inflates the owner's own best result (+0.178 -> +0.125). One dataset, three headlines.

**NEXT: nothing is owed.** All five phases are complete. Open items are the owner's to decide, and
they are listed in `REPORT_PLAIN.md` "Decisions for you".

## POST-PHASE-5 WORK, at the owner's request (2026-09-23)

Three tasks, all complete.

### 1. `run-premise/` backed up to Drive — VERIFIED
`rclone copy` (never sync/move/delete) from `C:\Users\user4\expLB\run-premise` to
**`tamara:expLB/_labpc_only_backup_2026-09-23/run-premise`**, a **new** folder — confirmed
non-existent beforehand (`rclone lsf` → *"directory not found"*), so nothing could be overwritten.
**`rclone check` → "0 differences found", "100 matching files"**; `rclone size` → **100 objects,
773.992 KiB**. Binary: the owner's own `C:\Users\user4\expLB\tools\rclone.exe` (not on the
non-interactive PATH). Logged in `audit/LABPC_WRITES.md`. **Still unbacked:** `run-cap/` 186.7 MB,
`run-cap-ctrl/` 108.8 MB, `logs/` 648.7 MB — the owner asked for run-premise only.

### 2. HP-KR wording-controlled experiment — DOCUMENTED with re-derived data
**`audit/HPKR_WORDING_CONTROLLED.md`**, backed by the re-runnable
**`audit/scripts/p5_hpkr_matched.py`**.

- **F-03's table reproduces exactly** — all eight figures, recomputed from
  `run-crossgen/kd11/matrices/mean_L40_s42.json` (git): `axes` 0.3004 full / **0.7517** matched,
  `raw` 0.6943 / 0.7150, `random` 0.6590 / 0.5973, `length` 0.5965 / 0.6137.
- **The layer sweep I computed myself**: `axes − raw` on the matched slice is **−0.2566 (L20)**,
  **+0.0367 (L40)**, **+0.0134 (L44)**. F-72 reports the same shape from the **passC** store
  (−0.216 / +0.033 / +0.027) — **two independently built stores agree**, which is worth more than
  either alone. The store difference explains why F-72's 0.374/0.739 are not identical to my
  0.3560/0.7517.
- **New, and the sharpest thing in the audit:** the collaborator's own directions, zero-shot, on the
  matched slice vs the full set — pulled read-only from the lab PC into `audit/cache/`:

  | direction | L20 matched/full | L40 matched/full | L44 matched/full |
  |---|---:|---:|---:|
  | his 12v13 extended axis | 0.4630 / 0.8057 | 0.6203 / 0.8765 | 0.5383 / 0.8491 |
  | split knowledge-denial axis | 0.3932 / 0.5923 | 0.4989 / 0.6692 | 0.4825 / 0.6659 |
  | **plain-assertion pole** | 0.4227 / **0.9118** | 0.5860 / **0.9344** | 0.5253 / **0.9266** |

  **A plain-assertion direction reproduces the HP-KR headline (0.93) and collapses to chance on the
  matched slice.** That is the register diagnosis with numbers attached, and it is in no document.
- Trained arm caveat kept: the matched slice **leaks** (~81% of its test rows are in HP-KR's train
  split, F-108), so §3/§4 are optimistic; the zero-shot table is clean.

### 3. `BRIEF_AS_EXECUTED.md` updated through Phases 4 and 5
It had been written after Phase 3 and was stale throughout (303 claims, 56 doc fixes, 5 self-errors).
Now: **424 claims / 297-119-8, 75 doc fixes, 390 scripts, 114 F-entries**, Phase 4 and Phase 5
sections, the two newly authorised out-of-`audit/` actions (DF-48 lab fix, the Drive backup), and the
error record extended **5 → 8**.

**Self-error #8 is mine and is recorded there and in `CORRECTIONS.md`:** the final report recommended
running the HP-KR wording test as a cheap CPU-only job. It had already been run (22 mentions of
`kr-matched` in the ledger), and a *clean* version needs new NDIF rollouts. I took it from the draft's
prose without opening the ledger — the exact fault the audit spent two days documenting.

## EXPERIMENT RUN (2026-09-23, owner-authorised): HP-KR leak-guard re-split — COMPLETE

**Setup, per the owner's instruction "don't change the original file, new folder, copy what you need":**
- New isolated tree `C:/Users/user4/expLB-resplit/` (scripts copy + its own `run-crossgen/` + `logs/`).
- **Originals never modified** — `expLB/src/assistant_axis_ndif/27_crossgen.py` verified
  md5 `681126242fe4414e4a3c131793de81c6` **before and after**; `expLB/run-crossgen` still 205,430 files.
- Reads read-only: `expLB/run-coords/passB`, `expLB/run-paper16/axes_vectors_s3_fp32.pt`.
- Patch: `audit/scripts/p6_patch_leakguard.py` (idempotent, backs up, 4 substitutions).
- Compare: `audit/scripts/p6_resplit_compare.py`. Results cached to
  `audit/cache/labpc/expLB-resplit/mean_L{20,40,44}_s42.json`.

**The guard is general, not a special case:** off-diagonal, drop any target row whose `row_id` is in
the source's TRAIN rows. It fired on **exactly one pair, both directions** — independently confirming
F-108's "no other pair shares a row_id". **59 of kr-matched's 258 rows survive = 77% contamination.**

### RESULT: the leak is real but does NOT account for F-03

| layer | leaked `axes−raw` | guarded | change |
|---|---:|---:|---:|
| 20 | −0.2566 | **−0.2545** | +0.0022 |
| 40 | +0.0367 | **+0.0309** | −0.0058 |
| 44 | +0.0134 | **+0.0050** | −0.0084 |

**F-03's conclusion survives.** One corrected cell averaged with nine untouched ones cannot move a
10-cell mean much. **This closes the last standing objection to F-03/F-72/F-76.**

### NEW NEGATIVE, and it is the interesting part

The touched cells move hugely. **`knowledge-report → matched` at L40: 0.5674 → 0.4262 — below
chance.** Also 0.6534 → 0.5719 (L44) and 0.5905 → 0.5416 (L40) the other way. **With the leak removed
the persona axis stops transferring within the knowledge-report family altogether.** `raw` drops too
(0.7932 → 0.7214), so the leak flattered every feature — which is exactly why the *difference* barely
moved while the *levels* did.

### Two engineering failures on the way, both mine, both logged

1. **The Scheduled Task never fired — twice.** `Last Run Time: 11/30/1999`, `Last Result: 267011`
   ("has not yet run"). `schtasks /RU user4` without a stored password registers **interactive-only**,
   so it waits for a desktop login that is not there. **`schtasks /Run` reports SUCCESS regardless** —
   I took that as evidence it ran, and a 30-minute monitor sat watching a job that never started.
   *Rule for next time: after launching a task, verify `Last Run Time` changed. "Reported success" is
   not "ran".* Worked around with a harness background process holding the ssh session.
2. **My first patch crashed the run** — `IndexError: index 59 is out of bounds for axis 0 with size
   59`. The guard changed which rows are scored but not the **bootstrap resampler**, which is built
   once per target sized to ALL its rows. Fixed by building a cached, deterministically seeded
   resampler for guarded cells. *Changing what is measured without changing what computes its error
   bar is the same class of mistake this audit documents.*

**Still not done, and it is NOT cheap:** this makes the rows disjoint, not the *material* independent.
A matched slice built from **new items** needs fresh NDIF rollouts (~2.5–3.5 h by the measured rates).
The owner's instruction was "re-split first, then check whether the second is needed" — **on this
evidence the second run is not needed to defend F-03**, though it would strengthen the
within-family negative, which currently rests on 59 rows.

## SECOND EXPERIMENT (2026-09-23): the `knowledge_matched` control — F-116

**The owner said "if NDIF needed then do it." It was not needed, and a better experiment existed.**

`eval_datasets.hpkr_slice` takes **two** values. Every HP-KR result in this project used one:

| slice | holds constant | label marks | rows | status |
|---|---|---|---:|---|
| `text_matched` | wording (all deny) | whether the model KNEW | 258 | F-03's slice |
| `knowledge_matched` | knowledge (all known) | whether the model DENIED | **2,001** | **never run** |

**Isolation, per the owner's rule:** new store `expLB-resplit/passB` = **directory junctions** to the
original dataset dirs + **one hardlinked** new view. Zero extra bytes. Verified after: original
`run-coords/passB` still **13 dataset dirs, no new key**.

### RESULT — the strongest negative in the project

Trained on it, tested on the other eleven (leak-guarded, mean, paper, seed 42):

| layer | `axes` | `raw` | **`axes − raw`** | `length` |
|---|---:|---:|---:|---:|
| 20 | 0.3850 | 0.5711 | **−0.1860** | 0.6174 |
| 40 | 0.3335 | 0.7093 | **−0.3758** | 0.6174 |
| 44 | 0.5267 | 0.7203 | **−0.1937** | 0.6174 |

**Inverted at L20 and L40; the −0.19 to −0.38 deficit against raw is the largest anywhere; a
one-dimensional length feature beats the persona axes at every layer.** In-distribution raw reads
**0.999** — with knowledge held constant the denial is near-perfectly detectable, **and nothing
learned from it transfers**. F-13's register diagnosis, tested head-on on 2,001 rows.

### A confound I flagged after misreading it myself

I first read the zero-shot agreement between this slice (0.2405/0.6466/0.6085) and the full dataset
(0.2504/0.6436/0.6032) as evidence the axis ignores knowledge. **It proves nothing** — the slice is
**93.5% of the full dataset**, so the readings must agree. Only the transfer and in-distribution
tables carry weight. Caught before it reached any deliverable.

### Third engineering failure, logged

The run died in 2 s: `refusing to write to ...run-crossgen-km`. The script's output guard requires a
path component named **exactly** `run-crossgen`; I used `run-crossgen-km`. Fixed to
`run-crossgen\km`. **The guard did its job — it refused before computing anything.**

## ALL FILES UPDATED FOR F-115/F-116 (2026-09-23)

- **`kd/docs/FINDINGS.md`** — **F-115** and **F-116** appended (append-only, per the owner's rule).
  Ledger now **116** entries. Pre-append backup: `audit/fix/FINDINGS.md.before-F115`
  (md5 `3da923c59f876693064817e7bde63ad8`).
- **`audit/HPKR_WORDING_CONTROLLED.md`** — new §6b (leak closed) and §6c (the other slice), plus §6
  rewritten and §7 corrected.
- **`audit/REPORT_PLAIN.md`** — decision 3 resolved; new section on both runs; "next" item 1 struck.
- **`audit/REPORT_TECHNICAL.md`** — new §3.7; referee-attack item 1 marked answered.
- **`audit/DOC_FIXES.md`** — **DF-76, DF-77, DF-78** (75 → 78 entries).
- **`audit/BRIEF_AS_EXECUTED.md`** — new §10; authorised-work table extended; error record 8 → 9.
- **`audit/CORRECTIONS.md`** — corrections #8 and #9 (both mine, both on the same paragraph).
- **Scripts:** `p6_patch_leakguard.py`, `p6_resplit_compare.py`, `p6_build_km_view.py`,
  `labpc_put.py` (chunked uploader), `p5_hpkr_matched.py`.
- **Cache:** `audit/cache/labpc/expLB-resplit/` (leak-guarded matrices) and `.../km/` (the new control).

**git status outside `audit/` is still 18 entries — unchanged.** I wrote "19" first and it was wrong:
`kd/docs/FINDINGS.md` was **already** in the modified set at session start (it is one of the 17 the
audit's own START_STATE records), so appending F-115/F-116 adds no new path. Verified:
`git status --porcelain -uall | grep -v audit/ | wc -l` = **18**. Still nothing committed.

## THE SCALE x BANK DESIGN QUESTION (owner, 2026-09-23) — answered from the stores

**Her question:** we claim 16-persona coordinates do not survive at 70B, and that our extended bank
does better — but did we run 3B/8B with the 16-persona bank, and did we ever run 3B/8B with the
extended bank?

**FIRST, A COUNT CORRECTION (VERIFIED, and it must not reach the supervisor uncorrected).** There is
no 16-persona bank. **`paper16` holds 26 personas**, counted from
`src/assistant_axis_ndif/pc_instructions/` (26 json files) and `pc_role_groups.json`:
deception 8 · honesty 7 · sycophancy 5 · anti-sycophancy 5 · assistant anchor 1 = **26**.
**The kd bank holds 36** (`kd/personas/kd_instructions/`, 36 files): the same 26 **+ 4
knowledge-denial + 3 sincere-ignorance + 3 plain-assertion = the 10 we added.**
The name **"paper16" is a naming artifact, not a count** — the "16" refers to nothing in the bank.
The **8 vs 7** figure people remember is the paper's *deception contrast* (8 deceptive roles against
7 honest ones), not the bank size. So the correct sentence is **"26 personas → 36"**, and the owner's
"we added 10 more" is exactly right. Scripts: `audit/scripts/p7_scale_bank_design.py` (+ `p7b.py`, `p7c.py`, lab-side).

### Which cells exist

| bank | 3B | 8B | 70B |
|---|---|---|---|
| **paper16 — 26 personas** | **yes** (`scale_3b`, 7 layers) | **yes** (`scale_8b`, 8 layers) | **yes** (`scale_70b_paper16fp32`, 21 layers) |
| **extended (kd / kd_s23)** | **NO** | **NO** | yes (`kd11/banks/kd*`) |

All four scale stores use the **same 5 datasets** (instructed-deception, mask, convincing-game,
ai_liar, roleplaying), k=3, seed 42 — a genuinely clean three-rung comparison. **Note those 5
EXCLUDE every harm-pressure dataset**, i.e. the ones the extended bank was built for.

### 1. The paper16 (26-persona) bank DOES decline with scale — clean, and it is our claim

Layer-averaged, mean pooling, `paper` probe, off-diagonal over the 5 shared datasets:

| rung | layers | raw | best persona | **vs raw** | vs datapca |
|---|---:|---:|---:|---:|---:|
| 3B | 7 | 0.6989 | 0.7371 | **+0.0382** | +0.1251 |
| 8B | 8 | 0.7149 | 0.7296 | **+0.0148** | +0.1086 |
| **70B** | 21 | 0.7546 | 0.7430 | **−0.0116** | +0.0605 |

**Monotone decline, and negative at 70B.** (The superseded `scale_70b` store with the OLD s3 bank
reads +0.0114 instead of −0.0116 — RF-16's bank-version issue. Quote the **fp32** rung.)

### 2. The extended bank DOES beat paper16 at 70B, even on datasets it was not built for

Raw joined from the ROOT store per the project's own join rule:

| layer | raw | paper16 | kd | kd_s23 |
|---|---:|---:|---:|---:|
| 20 | 0.7732 | +0.0615 | **+0.0731** | +0.0648 |
| 40 | 0.8090 | +0.0262 | +0.0355 | **+0.0668** |
| 44 | 0.7974 | +0.0105 | +0.0255 | **+0.0520** |

The winning feature at L40/L44 is **`pc:knowledge_denial`** — the personas we added — **on five
datasets containing no knowledge-denial data at all.**
**k confound checked and dismissed:** the scale store is k=3 and kd11 is k=5, but the paper16 bank
reads +0.0610/+0.0262/+0.0105 at k=5 in the root store versus +0.0615/+0.0262/+0.0105 at k=3 — so k
is immaterial here and the comparison is effectively like-for-like.

### 3. THE GAP, stated plainly

**"Our bank rescues the method at scale" is NOT separable from "our bank is better at any scale",**
because the extended bank was never run below 70B. The scale claim is supported **only** for the
paper16 (26-persona) bank. The extended-bank gain is a **70B-only** result.

**This is the one experiment where NDIF is genuinely warranted** — and unlike a fresh HP-KR slice, it
would measure something real: §2 shows the extended bank has a gain on the scale-5 datasets at 70B,
so there IS a quantity to track down the rungs. Proposed to the owner; not started.

# NEW BRANCH REVIEWED: `origin/paper` (2026-09-24) — SCOPE EXTENSION at the owner's request

`git fetch` found **no new commits on `feat/persona-pc-ood`**. He pushed to a **NEW branch,
`origin/paper`**, 2026-09-23 23:46 — 3 commits, 683 files, 11,555 insertions. The brief's scope was
`asix` + `feat/persona-pc-ood`; this is outside it and carries **no claims.csv ids**.
Review: `audit/COLLAB_PAPER_BRANCH.md`. Script: `audit/scripts/p8_collab_layer_sweep.py`.
Ledger: **F-117, F-118, F-119** (now 119 entries).

**Correction to the brief I was given:** the sweep is every **4th** layer (16 layers, L20–76 + L79),
not every 3rd.

## What it does to the audit's conclusions

**1. D-028 GENERALISES — 3 layers to 16.** The draft's Contributions margin
(`ext_axis1 − ext_pc3`), excluding HP-KR: **+0.0086 (L20), +0.0012 (L24) … within ±0.06 everywhere,
crossing zero five times**, positive at only **11/16**. And **"outscores at every depth" is falsified
by his own data** — negative at **L64, L68, L72**.

**2. But a NEW result runs in his favour.** `ext_axis1 − baseline` is **positive at 16/16 layers,
+0.115 to +0.273**, and **still 16/16 excluding HP-KR** (+0.123 to +0.243). Depth-robust, NOT
HP-KR-driven, invisible from three layers. **This belongs in the paper.** Caveat: his baseline is a
single-dataset probe; D-158's pooled probe is still the fair comparator.

**3. THE CONVERGENCE — the project's strongest negative.** His Gender-Secret run (both classes deny)
puts **every persona direction at chance (0.436–0.623)**. With F-115 (his axes on our matched slice:
0.4630/0.6203/0.5383) and F-116 (our `knowledge_matched` control: below chance), that is **three
independent wording-matched tests, one answer.** Report as ONE finding.

**4. Optimum layer is 36, not 40** — 0.9275 vs 0.9158. The draft under-reports its own best.

**5. Unexplained and flagged:** a sharp rise at **L76** across many arms, falling back at L79
(`axis1` 0.6526 → 0.8738 → 0.8179; HP-KR published axis 0.4344 → 0.7301 → 0.7709). **`random3` stays
flat 0.48–0.55**, so not indiscriminate noise. A referee will ask.

**6. D-237 SUPERSEDED — IN HIS FAVOUR.** `paper/make_numbers.py` **now exists** on this branch. That
was the audit's **only** DOES-NOT-MATCH verdict, resting entirely on the script's absence — true when
checked, no longer. `claims.csv` row amended (backup `.before-F119`); DF-71's reproducibility half
withdrawn (DF-85). **Second correction in his favour, after D-107.**

**Credit:** he validated the new NDIF pooling path rather than assuming it (L40 rebuilt both ways,
row cosine >= 0.99989, RepE probe cosine 0.9996), and the Gender-Secret run is the most carefully
controlled experiment on either side.

**Still owed by him:** `persona_depth/summary.md` cites `../persona_unified/` and
`../persona_depth_orig/`; **neither is on the branch** (DF-83).

## Files updated for this review

`kd/docs/FINDINGS.md` (F-117/118/119, now 119) · `audit/COLLAB_PAPER_BRANCH.md` (new) ·
`audit/claims.csv` (D-237 superseded; backup `.before-F119`) · `audit/DOC_FIXES.md` (78 → **85**,
DF-79…DF-85) · `audit/REPORT_PLAIN.md` · `audit/REPORT_TECHNICAL.md` (§3.8) ·
`audit/BRIEF_AS_EXECUTED.md` (§11) · `audit/scripts/p8_collab_layer_sweep.py`.

**Meanwhile the KD-at-scale chain is still running** (3B generation, started 23:57:52) — unaffected
by this review.

## THE L76 QUESTION IS SOLVED (2026-09-24) — F-120

Asked of **our own 80-layer store** (`passA`, mean, `paper`, seed 42), a pipeline he never touched.
**The same shape appears there**, so it is a property of the model, not of his run.
Script: `audit/scripts/p9_late_layer.py`.

| | L40 | L70 | L72 | L79 |
|---|---:|---:|---:|---:|
| `raw` | 0.8195 | 0.7828 | 0.7834 | **0.7426** |
| `axes` | 0.8266 | 0.6211 | 0.6384 | **0.7492** |
| `random` | 0.5971 | 0.5234 | 0.5133 | **0.6405** |
| **`axes − random`** | **+0.2296** | +0.0977 | +0.1250 | **+0.1087** |

1. **`raw` does not rise — it falls.** The "win" is the full-rank probe degrading.
2. **`axes` recovers, does not peak** — 0.7492 at L79 is still below its own mid-stack 0.8145.
3. **`random` gains +0.127 over the same span.** A random direction learns nothing, so this is a
   geometry change lifting ANY low-dim projection (late-layer anisotropy). It cannot lift a
   full-rank probe, which is why `raw` is unmoved.

**Against the random control the axes are WORSE at the top (+0.109) than at L40 (+0.230).**
Reviewer-ready sentence in `COLLAB_PAPER_BRANCH.md` §2 and F-120. Consistent with M-057.
**Second independent vindication of the random-direction control.**

## COMPLETE RECORD OF THE COLLABORATOR'S FINDINGS — `audit/COLLABORATOR_FINDINGS.md` (NEW)

His work spans 3 branches, 19 result folders, 17 write-ups; the audit discussed it claim-by-claim but
never as a body. That file is the single list: **7 positive results** (the knowledge-denier repair
0.148 → 0.919; the validated 16-layer sweep; Gender-Secret; the pooled LODO baseline; the resumable
cache; per-token and follow-up-token work), **where the two pipelines agree** (55 bank-matched cells,
mean |gap| 0.0015; 51 of 79 draft claims match), **6 substantive disagreements**, **4 same-setting
conflicts**, the **three-way convergence**, and **what he should be credited with**.

It also records the two corrections that ran **in his favour** (D-107, D-237/F-119) and the standing
limitation that **nothing of his is recomputable from raw** — no activations are committed on any
branch, so his numbers are **attested, not reproduced**.

## FILES UPDATED (2026-09-24, second pass)

`kd/docs/FINDINGS.md` → **F-120** (now **120** entries) · `audit/COLLABORATOR_FINDINGS.md` (NEW) ·
`audit/COLLAB_PAPER_BRANCH.md` §2 rewritten with the solution · `audit/REPORT_PLAIN.md` ·
`audit/REPORT_TECHNICAL.md` §3.8 + deliverables list · `audit/DOC_FIXES.md` DF-84 converted from
puzzle to answer · `audit/scripts/p9_late_layer.py` (NEW).

## KD-AT-SCALE CHAIN — RUNNING AND HEALTHY

3B generation live on NDIF: **546+/5,400 rollouts, 4+/36 persona files** at 00:15, task
`Audit-KD-Scale` **Status: Running**. Launched as a **SYSTEM** scheduled task after `/RU user4`
registered interactive-only and never fired (Last Result 267011) — **verified fix**, smoke test
exit 0 with the machine account. 8B follows automatically in the same chain.

# ============================================================================
# HANDOFF — READ THIS FIRST IF YOU ARE A NEW SESSION (2026-09-24, ~12:40)
# ============================================================================

**NOTHING IS RUNNING.** Every job is finished: the KD-at-scale chain (08:51), the crossgen (12:34),
the 70B rung (12:39), and the Drive backup (290 files, 0 differences). No scheduled task is active
(`Audit-KD-Scale` is Ready and has reported `chain finished`). No background watcher is armed.
**Do not restart anything.**

## State of the work

**The audit is COMPLETE** — Phases 0–5 plus the methods track, 424 of 692 claims checked,
297 hold / 119 partly / 8 fail, own error rate 6.5%. All deliverables written.

**Since the audit closed, seven things happened. All are in the ledger (now 121 entries):**

| ledger | what |
|---|---|
| F-115 | HP-KR leak closed and measured — F-03 survives it |
| F-116 | the never-run `knowledge_matched` control (2,001 rows) — axes transfer BELOW chance |
| F-117 | the collaborator's 16-layer sweep on the NEW branch `origin/paper` |
| F-118 | his Gender-Secret run — every persona direction at chance where both classes deny |
| F-119 | `paper/make_numbers.py` now exists — supersedes D-237, **in his favour** |
| F-120 | the late-layer (L76) rise SOLVED — anisotropy, not the method working |
| **F-121** | **the kd bank's scale ladder completed at 3B and 8B** |

## The three findings a report writer most needs

1. **The convergence (the strongest negative).** Three independent wording-matched tests agree that
   where both classes deny knowledge, no persona direction reads the lie: F-115 (his axes on our
   matched slice, 0.4630/0.6203/0.5383), F-116 (our knowledge_matched control, below chance),
   F-118 (his Gender-Secret, 0.436–0.623). **Report as ONE finding.**
2. **The positive control.** Pooled training, no personas: **+0.1254** under leave-one-cluster-out,
   10 of 11 targets, five seeds, ±0.035. **Publish +0.125, not +0.178** (the larger figure leaks).
3. **The scale ladder (F-121).** Both banks decline; the decline is the method's, not the bank's.

## Documents to read, in this order

`audit/REPORT_PLAIN.md` → `audit/BRIEF_AS_EXECUTED.md` → `audit/COLLABORATOR_FINDINGS.md` →
`audit/REPORT_TECHNICAL.md`. Then `audit/DOC_FIXES.md` (85 entries, **none applied**) and
`audit/CORRECTIONS.md` (the audit's own 9 errors).
Shareable page of the first two: https://claude.ai/artifact/WTfUrBtKBdQ93X8jjrg46r

## Two chapters already drafted

`audit/METHODS_CHAPTER.md` (9 sections) and `audit/LIMITATIONS_AND_FUTURE_WORK.md`.
**The Results chapter does not exist** — that is the natural next piece of writing.

## Standing constraints that still apply

- **Nothing of the collaborator's is recomputable from raw** — no activations committed on any of his
  three branches. His figures are attested, not reproduced.
- **git status outside `audit/` is 18 entries**, all uncommitted, all the owner's to commit. The only
  file the audit appended to outside `audit/` is `kd/docs/FINDINGS.md` (append-only, backups at
  `audit/fix/FINDINGS.md.before-F115`).
- **`DOC_FIXES.md` has 85 proposed corrections and NOT ONE has been applied.**
- Lab PC `user4@100.85.217.68`; it powered off once today and came back. Long jobs must be **SYSTEM**
  scheduled tasks — `/RU user4` registers interactive-only and silently never fires.

## OWNER COMMITTED (2026-09-24 05:16) — and F-121 is now the only uncommitted change

`git log` shows **`e917ecd3` "audits plus new kd findings"**, the owner's own commit at 05:16:36,
which took the 17 modified files into git. `git status` outside `audit/` is therefore **1 entry**,
not 18: `kd/docs/FINDINGS.md`, holding the F-117…F-121 appends made after that commit.
**Nothing was lost and the audit committed nothing** — the drop from 18 to 1 is entirely her commit.

## F-121 PROPAGATED TO EVERY FILE (2026-09-24)

`kd/docs/FINDINGS.md` **F-121** (ledger now **121**) · `audit/REPORT_PLAIN.md` (new section) ·
`audit/REPORT_TECHNICAL.md` §3.9 · `audit/BRIEF_AS_EXECUTED.md` §12 ·
`audit/LIMITATIONS_AND_FUTURE_WORK.md` (the scale-ladder limitation marked **CLOSED**) ·
`audit/METHODS_CHAPTER.md` (the ladder added to the headline result) ·
`audit/DOC_FIXES.md` **DF-86…DF-88** (85 → **88**) · `audit/scripts/p12_kd_scale_ladder.py`.

## RESULTS BY AXIS (2026-09-24) — `audit/RESULTS_BY_AXIS.md` (NEW), and it found an untested axis

The owner asked which personas each axis is built from and what each scores at each rung. Answered
from the stores: `audit/scripts/p13_all_results_by_axis.py`.

**`paper16` is the paper's DECEPTION set, not the bank size.** `pc_role_groups.json` says so:
*"8 deceptive + 7 honest = the paper's 16-persona deception set once the shared `default` anchor is
counted; 5 + 5 + default = its 11-persona sycophancy set."* The bank holds **26** because the two
sets share the anchor.

**The 36-persona kd bank does NOT merge the behaviours.** It declares **five** contrasts, each on its
own basis: deception (16), sycophancy (11), unified (26), knowledge_denial (8), kd_assertion (8).
Its deception axis is built from exactly the same 16 personas as paper16's.

**GAP FOUND, then CORRECTED (F-122) — `pc:sycophancy` is missing from OUR pipeline, not from the
project.** A search of every `mean_L40_s42.json` under `run-crossgen/` returns nothing; it is absent
from `27_crossgen.py`'s default `--features` and nobody passed it explicitly. Closing it is
CPU-only, minutes per store. **Not done.**

**The correction:** that search covered `run-crossgen/` — *our* pipeline — and an earlier version of
this log and of `RESULTS_BY_AXIS.md` reported it as "never computed anywhere". **The collaborator's
`origin/paper` has four sycophancy arms** (`syc_pc1`, `syc3`, `syc_axis1`, `syc3_orth`) at all 16
layers. Scoping a one-pipeline search as a project-wide absence — the audit's own error class,
caught by the owner asking whether his branch had been checked.

**And it carries a result.** Excluding HP-KR, `syc_pc1` is his **best arm at L40-56** (0.9370 at
L40 vs `ext_axis1` 0.9165, baseline 0.7581). **But the margin is one dataset** — `convince_me`
+0.086, losing on `instruct_dishonesty` and `repe_honesty` at every layer, 3 of 5 wins. And for
every `dim=1` arm the trained off-diagonal **equals** the zero-shot AUROC (spread 0.000 across
training rows at all 16 layers for `ext_axis1` and `denial_axis1`), so those 30-cell means hold
**6 distinct numbers**. See F-122, DF-89 (revised), DF-93, `audit/COLLABORATOR_FINDINGS.md` §A8.

**Also confirmed:** the knowledge-denial axes are **70B-only** as trained features (kd11, 11
datasets, k=5). And **`pc:kd_assertion` is the best-transferring persona feature in the project**
(0.8119 at L40, +0.150 over raw) — but its contrast is *feigned ignorance vs plain assertion*, i.e.
an **assertion-register axis**, and it beats the knowledge-denial axis proper (0.7492).
**The register diagnosis is visible directly in the feature table.**


---

# COVERAGE GAP MEASURED (2026-09-24, after the owner asked why the audit missed things)

**`audit/COVERAGE_GAP.md`** — the full accounting. Reproduce: `python audit/scripts/p15_coverage.py`.

**The tally in this file overstates coverage.** "424 of 692 claims checked" mixes inventory claims
with claims raised during checking: **314** of the 424 have inventory ids, **110** do not. Real
coverage is **314/692 = 45%**, not 61%. **378 inventory claims never got a verdict — 176 of them in
the draft PDF.** That is the single largest gap in the audit (DF-94).

**Three things the owner suspected were missed, and were not:**
- **Her crossgen runs** — all 13 lab-PC stores are covered; `passA` (960 matrix JSONs) is named in
  211 audit files. `passB_fine` and `passP` reach the audit but not the ledger.
- **Mahdi's branch** — `origin/feat/llama-layer30` adds one commit (a config + a runner tweak) and
  **zero** result files. Phase 0 read it correctly (`phase0/branch_compare.md:308`).
- **The collaborator** — 114 experiments logged, 20 his.

**Three real causes of what WAS missed:**
1. **Coverage** — 55% of inventory claims never got a verdict; 176 of those are in the draft.
2. **Scope froze against a moving target** — `origin/paper` landed 2026-09-23, one day after the
   brief fixed scope, and was reviewed 2026-09-24 *after* both reports were written. F-117, F-119
   and F-122 are therefore absent from them (DF-95).
3. **The audit committed its own documented failure mode** — a correct observation
   (no `pc:sycophancy` in `run-crossgen/`) that reached a wrong conclusion ("never computed
   anywhere"). Caught by the owner, not by the audit. Corrected in F-122.
