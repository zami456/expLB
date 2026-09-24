# Phase 0 — `asix` vs `origin/feat/persona-pc-ood`

Facts and locations only. No verdict on whether any number is right.

- Made 2026-09-22 from git objects only (`git show`, `ls-tree`, `log`, `diff`, `rev-list`, `cat-file`). No checkout, no worktree.
- Script: `audit/scripts/p0_branch_compare.py`. Re-run with
  `PYTHONIOENCODING=utf-8 python audit/scripts/p0_branch_compare.py > <scratch file>`.
  It prints sections A to J and writes `audit/phase0/pcood_results_index.csv` (one row per folder).
- Tags: VERIFIED = checked in git objects during this audit. DOC-ONLY = only in prose (commit messages, summary.md, docstrings). INFERRED = my reading, reason given. NOT-FOUND = looked, not there.
- Short names: **asix** = owner's branch (Suraiya). **pc-ood** = `origin/feat/persona-pc-ood` (collaborator zami456). **MB** = merge-base `2e2a785`. **DD** = `src/probes/deception-detection/`.

---

## 0. Summary

1. VERIFIED. All 490 files pc-ood changed are under DD: 483 added, 7 modified. The result files are in `DD/results/` (19 folders, 421 files, 13.7 MB: 316 CSV, 87 PNG, 15 MD, 2 NPZ, 1 YAML). He also added 3 input-data folders (43 files, 97.2 MB) and 20 Python files.
2. VERIFIED. No blob is shared between the branches in either direction: 0 of his 490 blobs occur anywhere in asix (HEAD or history), and 0 objects are new on both sides. The two branches exchanged work as **separately made files**, not as copies.
3. His persona inputs come from the owner's runs. The evidence is metadata and names, not blob hashes (details in §3):
   - VERIFIED: his `paper16_L20.pt` and `deceptive_L20.pt` both record `source_run: run-paper16-shared`. That run is made by asix's `run_paper16_shared_chain.ps1` (added 2026-09-18 11:26, six hours before he committed those files).
   - VERIFIED: his 36-persona bank uses exactly the role names of asix's `kd/personas/` (first added 2026-09-19 02:58).
4. VERIFIED. The owner then used his results: asix `4d0d046` (2026-09-22 02:46) reads pc-ood at `f386a72` in `kd/docs/FINDINGS.md`, `docs/COLLABORATOR_BRIEFING.md`, `docs/ROADMAP_TO_DEFENCE.md` and `kd/personas/kd_role_groups_collab.json`. Earlier, asix `ed8daca` (2026-09-15) ported his activation cache and batcher into `src/assistant_axis_ndif/`, with different content.
5. VERIFIED. asix has **not touched DD since MB**: its 363 files there are byte-identical to MB. So the only shared-path differences are his 7 modified files.
6. VERIFIED. There is no Git LFS: no `filter=lfs` line in any `.gitattributes`, and 0 LFS pointer files among the small blobs of asix, pc-ood, MB and origin/main.

---

## 1. Topology (VERIFIED)

| | tip | commits since MB | authors | dates (+0600) |
|---|---|---|---|---|
| asix | `43ee6d3` "collaborator breifing" | 137 | Suraiya Shanaz `<suraiya.shanaz@g.bracu.ac.bd>` | 2026-09-07 … 09-22 17:37 |
| pc-ood | `45b16d9` | 34 | zami456 `<abdullah.al.zami@g.bracu.ac.bd>` | 2026-09-04 … 09-22 12:45 |
| MB | `2e2a785` "adjust shard size for effective batching" | — | zami456 | 2026-07-12 01:47 |

- pc-ood's first 8 commits (`e391b3e` … `7c28299`, 09-04/05) are exactly `origin/feat/resumable-acts-cache`. They hold infrastructure only: a resumable per-example activation cache, token-budget batching, run_llama_all.sh and tests. They add no results.
- The persona work is the remaining 26 commits, `7a9c8ba` (09-18 17:31) … `45b16d9` (09-22 12:45).

---

## 2. The collaborator's result and data folders

The full table, with every column the task asked for, is in **`audit/phase0/pcood_results_index.csv`** (22 rows). The CSV's `script`, `config_or_input`, `how_known` and `what_it_tests` columns were written by hand from the scripts, configs, summary.md files and commit messages I read. Everything else in it is computed.

### 2a. Condensed table

All folders are under `DD/`. Bytes are summed from `git ls-tree -l`. The layer is VERIFIED three ways, and they agree for every folder: the folder name, the `layer` column inside its CSVs, and `detect_layers` in the activation config.

| folder | files | bytes | first commit (date) | later commits | layer | what it tests (1 line) |
|---|---|---|---|---|---|---|
| `results/persona_ood` | 30 (24 csv, 5 png, 1 md) | 652,468 | `a093c54` (09-18) | 7 | 20 | 6×6 train/test transfer matrix: 8192-d probe vs persona PC/axis projections (15 deception+honesty personas), random-3d and sycophancy-basis controls, zero-shot, Liars' Bench head-to-head |
| `results/persona_ood_L40` | 30 | 627,275 | `06da2de` (09-20) | 1 | 40 | same at layer 40 |
| `results/persona_ood_ext_L20` | 43 (35 csv, 7 png, 1 md) | 970,950 | `9774075` (09-20) | 7 | 20 | extended 36-persona bank: ext_axis1 (12v13), denial_axis1 (4v6), ext/denial PCs; zero-shot + operating points |
| `results/persona_ood_ext_L40` | 43 | 941,754 | `9774075` (09-20) | 5 | 40 | same at 40 |
| `results/persona_ood_ext_L44` | 43 | 949,893 | `29cc17c` (09-21) | 2 | 44 | same at 44 (third depth point) |
| `results/persona_ood_followup_L20` | 43 | 919,169 | `d2f4cce` (09-21) | 2 | 20 | extended bank on the follow-up "No" token instead of the response-mean |
| `results/persona_ood_mod_L20` / `_L40` / `_L44` | 43 each | 913,812 / 895,199 / 898,235 | `94db651` (09-21) | 0 | 20/40/44 | same 36 personas re-extracted with ~uniform 142–150 rollouts ("modified" bank) |
| `results/loo_L20` / `loo_L40` | 8 each (5 csv, 1 npz, 1 png, 1 md) | 1,074,051 / 1,064,853 | `6b16a46` (09-20) | 0 | 20/40 | leave-one-dataset-out: 8192-d probe trained on the pooled other 5 datasets (naive and 1/n weighting) vs persona arms; bootstrap CIs |
| `results/persona_positions_L20` / `_L40` / `_L44` | 7 each (4 csv, 2 png, 1 md) | 300,705 / 297,772 / 297,343 | `a8eb902` (09-22) | 0 | 20/40/44 | persona axis read per token: mean vs max / top-10% / first / last / pre-reply; position profile |
| `results/XXXX-9-llama-all` | 4 png | 591,035 | `148ddeb` (09-21) | 0 | 20 | Liars' Bench Mean Probe (LR on RepE), 5 datasets. **Plots only** |
| `results/XXXX-9-llama-all-L40` / `-L44` | 4 png each | 569,369 / 571,430 | `148ddeb` | 0 | 40/44 | Mean Probe at 40 / 44. **Plots only** |
| `results/XXXX-9-llama-all-followup` | 7 (1 csv, 1 yaml, 1 md, 4 png) | 633,733 | `fd17728` (09-21) | 1 (`148ddeb` +4 png) | 20 | Liars' Bench Follow-Up Probe (Table 7) measured |
| `results/XXXX-9_lr__repe_honesty__plain__llama-70b-3.3__20260712_053340` | 4 png | 489,786 | `148ddeb` | 0 | 20 | earlier two-dataset Mean Probe run (`configs/XXXX-9.yaml`, which exists at MB). **Plots only** |
| **input** `deceptive/` | 4 (pt, npz, json, md) | 576,110 | `7a9c8ba` (09-18) | 0 | 20 | 8 deceptive persona vectors + deception axis, from `run-paper16-shared` |
| **input** `paper16_L20_sharedquestions/` | 3 (pt, npz, json) | 2,217,084 | `7a9c8ba` (09-18) | 0 | 20 | 26-persona bank (vectors, 4 axes, 5 PCs), from `run-paper16-shared` |
| **input** `persona including knowlede denial/` (sic) | 36 pt | 94,425,189 | `148ddeb` (09-21 22:41) | 0 | all 80 | 36 personas × `vector[80, 8192]` + `n`; used by every `ext`, `followup`, `loo` and `positions` result |

### 2b. How I know which script and config made each folder

- `persona_ood*`, `loo_*` and `persona_positions_*` come from `DD/deception_detection/scripts/persona_ood.py` (subcommands `run`, `zeroshot`, `loo`, `verify`) and `persona_positions.py`. Figures come from `persona_ood_figures.py`. Pooled activations come from `pool_activations.py`.
  - VERIFIED from the scripts' default `OUT_DIR` and the `fire.Fire` entry points.
  - The exact command lines are DOC-ONLY. They are the "Reproduce" blocks in `loo_L20/summary.md` (lines 120–123), `persona_ood_ext_L44/summary.md` (92–100), `persona_ood_followup_L20/summary.md` (141–149), `persona_ood_mod_L40/summary.md` (100–108) and `persona_positions_L40/summary.md` (103–110).
  - No command line is recorded for `persona_ood`, `persona_ood_L40`, `ext_L20` or `ext_L40`.
- `XXXX-9-*` folders come from `scripts/experiment.py run`, via `run_llama_all.sh`.
  - VERIFIED: each config's `name:` field equals its folder name.
  - The four configs are `deception_detection/scripts/configs/XXXX-9-llama-all{,-L40,-L44,-followup}.yaml`. The L40 and L44 configs were committed only at `148ddeb` (09-21), after the L40 results from 09-20. The commit message says so (DOC-ONLY).
- `XXXX-9-llama-all-followup/cfg.yaml` is the dumped run config: `detect_layers: [20]`, `acts_positions: tail`, `use_followup_question: true` (VERIFIED).

### 2c. What is **not** in git (NOT-FOUND in any ref)

Results on pc-ood depend on these, but they are not committed. The claims that they exist, and where, are DOC-ONLY.

- **`persona vectors/vectors_s3/`**: 26 personas × 80 layers. Used by `persona_ood_L40`, and by `persona_ood` after its `24a237b` regeneration.
  - Commit `668b280` says: "The 66 MB vectors folder is left uncommitted".
  - Commit `148ddeb` says: "The original all-layer bank and the modified-rollout bank stay out".
  - `git log --all` over that path returns nothing (VERIFIED).
  - How the loader changed (VERIFIED from the script at each commit): at `a093c54`, `persona_ood.py` line 177 calls `persona_pcs.load_bank()`, which reads the committed `paper16_L20.pt`. At `06da2de`, line 503 calls `load_bank_at_layer(layer)`, whose default is this uncommitted folder.
- **`persona knowledge denial modified/`**: used by all three `persona_ood_mod_L*`. Not in any ref (VERIFIED).
- **Pooled activations** (`activations/pooled`, `pooled_L40`, `pooled_L44`, `pooled_followup_L20`; about 300 MB each per `persona_ood/summary.md`) and the token projections (`activations/token_proj_L*`). Every persona CSV is computed from these. `activations/` is gitignored (VERIFIED: `.gitignore` of DD, line 191).
- **Mean-Probe run outputs** (`detector.pt`, `scores*.json`, `results_table.csv`) for `XXXX-9-llama-all`, `-L40`, `-L44` and the 07-12 run. Only their 4 plots are committed.
  - Code refers to these files: `base_probe_arm` / `verify` in persona_ood.py read `detector.pt` and `results_table.csv`.
  - DD's `.gitignore` ignores `results/` (line 177) and the root `.gitignore` ignores `*.pt` (line 185). So every committed result file was force-added (VERIFIED).
- **The "INLP experiment"**, mentioned in `148ddeb` ("remains uncommitted"). DOC-ONLY. No INLP file or code is on pc-ood (NOT-FOUND by path/grep of the changed-file list).
- Activation caches on "D:" and "E:/liars-bench-acts/cache". DOC-ONLY, from config comments and commit `29cc17c`.

What the committed files do allow (VERIFIED by opening them in the script's §J):
- The **only per-example raw data in git** is `loo_L20/loo_scores.npz` and `loo_L40/loo_scores.npz`. Each has 66 arrays: scores for 5 arms (`baseline`, `axis1`, `pc3`, `ext_axis1`, `ext_pc3`) × 2 weightings × 6 datasets, plus `labels|<dataset>` for all 6.
- So the leave-one-out AUROCs can be recomputed from git.
- Every other persona CSV needs his pooled activations, which are not in git.

### 2d. Git LFS (VERIFIED)

- `.gitattributes` exists only on asix, at `src/assistant_axis_ndif/.gitattributes`. It sets `* text=auto eol=lf` and `*.pt binary`. There is no `lfs` line in it.
- LFS pointer scan over every blob under 1 KiB:
  - asix: 328 blobs, 0 pointers.
  - pc-ood: 561 blobs, 0 pointers.
  - MB: 280 blobs, 0 pointers.
  - origin/main: 280 blobs, 0 pointers.
- Per folder, the CSV column `lfs_pointer_files` is 0 everywhere. His `.pt` files are real binaries: the largest of his added files is 2,623,020 bytes, and all 38 load with `torch.load(weights_only=True)`.

---

## 3. His 17 .md docs

Line and date counts are VERIFIED. "Numeric tokens" means decimals (e.g. `0.904`, `.12`, `−0.03`) plus percentages, counted by regex. It is a rough proxy for numeric claims: tables inflate it, and integers such as "21 of 30" are not counted. "Cites asix?" checks for `paper16`, `kd`, `F-nn`, `cap`/capitulation, `asix`/owner names, `run-kd|run-cap|run-coords|run-paper16`, `vectors_s3`, `FINDINGS`.

| # | path (under DD) | lines | commits (dates) | what it reports (DOC-ONLY) | numeric tokens / lines with a number | cites asix? |
|---|---|---|---|---|---|---|
| 1 | `README.md` (modified) | 230 total, **+50 added** | `45c733d` (09-04) | how to run the resumable llama sweep; cache and token-budget batching | 0 / 0 (added part) | no |
| 2 | `deceptive/README.md` | 67 | `7a9c8ba` (09-18) | what the 8 deceptive vectors are; n per persona (26–82); projections onto the axis | 10 / 10 | **yes**: `run-paper16-shared`, "our persona-coordinates replication", path `../paper16_L20/shared_questions/` (not on his branch) |
| 3 | `results/XXXX-9-llama-all-followup/summary.md` | 68 | `fd17728` (09-21) | Follow-Up Probe measured vs Liars' Bench Table 7 (mean AUROC 0.636 vs 0.634) | 67 / 22 | no |
| 4 | `results/loo_L20/summary.md` | 124 | `6b16a46` (09-20) | leave-one-out baseline at L20 and L40; the persona axis keeps the best mean and worst case | 86 / 32 | no |
| 5 | `results/loo_L40/summary.md` | 2 | `6b16a46` | pointer to #4 | 0 / 0 | no |
| 6 | `results/persona_ood/summary.md` | 364 | 7 commits, `a093c54` … `06da2de` (09-18 → 09-20) | main L20 study: persona PCs vs 8192-d probe, controls, sycophancy basis, balanced accuracy, L40 survival section | **253 / 120** | no ("findings" hit is the ordinary word; cites arXiv 2605.09391) |
| 7 | `results/persona_ood_L40/summary.md` | 110 | `06da2de` (09-20) | L40 repeat; which L20 findings do not survive | 93 / 38 | names uncommitted `persona vectors/vectors_s3/` (same naming as the owner's `run-*/vectors_s3`) |
| 8 | `results/persona_ood_ext_L20/summary.md` | 127 | `9774075` (09-20), `f386a72` (09-21 correction), `45b16d9` (09-22) | knowledge-denial personas "fix HP-KR" at L20/L40; n per new persona | 109 / 37 | no explicit citation; the personas are the owner's kd roles (§4) |
| 9 | `results/persona_ood_ext_L40/summary.md` | 4 | `9774075` | pointer to #8 | 0 / 0 | no |
| 10 | `results/persona_ood_ext_L44/summary.md` | 104 | `29cc17c`, `a8eb902`, `45b16d9` (09-21/22) | L44: "layer 40 is a peak" | 75 / 35 | no |
| 11 | `results/persona_ood_followup_L20/summary.md` | 155 | `d2f4cce`, `c2a0c1f`, `45b16d9` | persona experiment at the follow-up token; IT flips, the HP-KR repair does not carry over | 109 / 36 | no |
| 12 | `results/persona_ood_mod_L20/summary.md` | 2 | `94db651` | pointer to #13 | 0 / 0 | no |
| 13 | `results/persona_ood_mod_L40/summary.md` | 111 | `94db651` (09-21) | "More rollouts per persona did not help" | 113 / 26 | names uncommitted `persona knowledge denial modified/` |
| 14 | `results/persona_ood_mod_L44/summary.md` | 2 | `94db651` | pointer to #13 | 0 / 0 | no |
| 15 | `results/persona_positions_L20/summary.md` | 2 | `a8eb902` | pointer to #16 | 0 / 0 | no |
| 16 | `results/persona_positions_L40/summary.md` | 113 | `a8eb902` (09-22) | per-token reads; mean ranks best, top-10% detects best | 88 / 28 | no |
| 17 | `results/persona_positions_L44/summary.md` | 2 | `a8eb902` | pointer to #16 | 0 / 0 | no |

- **Totals:** 1,003 numeric tokens on 384 lines across the 17 docs. Eight of the docs are 2–4-line pointers.
- **Commit messages count too.** His 34 commit messages hold another **383 numeric tokens**, and several claims appear only there. Example: "21 of 30 cross-dataset cells improve" in `a093c54`.
- **Numbers hard-coded in code.** `persona_ood.py` lines 65–92 contain Liars' Bench Tables 6–8 values, "READ FROM THE PAPER, not recomputed", used as baselines in the comparison tables (VERIFIED, read the file).
- **No F-numbers anywhere on his side.** None of his 17 docs mention F-numbers, FINDINGS.md, cap, kd/, asix or the owner (VERIFIED, grep count 0 for each doc). F-numbers exist only on asix.
- **Some claims were corrected in place.** The in-doc and commit-message history shows:
  - `e3f2258`: HP-KR "0.083" corrected to 0.593 at L40.
  - `f386a72`: the corrected L40 HP-KR row of `ext_L20/summary.md`, "the wrong version had already been pushed".
  - `45b16d9`: the L44 summary's "~0.98 on IT at L20 and L40" corrected to 0.973 / 0.338.
  - Phase 2 must track claim history, not only the latest text.

---

## 4. Dependencies between the branches

### 4a. Owner → collaborator (his inputs came from the owner's runs)

| his file(s) | evidence of origin | tag |
|---|---|---|
| `paper16_L20_sharedquestions/paper16_L20.{pt,npz,json}` (26 personas, L20) | JSON/pt field `source_run: "run-paper16-shared"`, `judge_model: Llama-3.1-70B`. asix `src/assistant_axis_ndif/run_paper16_shared_chain.ps1` writes `run-paper16-shared/`, first committed `b9a497e` 2026-09-18 11:26. His commit is `7a9c8ba` 2026-09-18 17:31. | VERIFIED (metadata + dates) |
| `deceptive/deceptive_L20.*` + `README.md` | `source_run: "run-paper16-shared"`. The README says "our persona-coordinates replication", offers "If you want the honest end too, ask", and points to `../paper16_L20/shared_questions/`, a path that exists on neither branch. | VERIFIED (text); **INFERRED** that the README was written on the owner's side as a hand-off note, from its wording and the foreign path |
| `persona including knowlede denial/*.pt` (36 roles × 80 layers) | Exactly the 36 role names of asix `kd/personas/kd_instructions/*.json` (default + 35 `pc_*`). His `KNOWLEDGE_CANDOUR_PERSONAS` (6) = asix's `sincere_ignorance` (3) + `plain_assertion` (3). His deception/honesty/sycophancy/anti-sycophancy lists equal asix's groups. **File schema** `{vector[80,8192], layers, type:"pos_3"/"mean", role, n}` is exactly what asix `src/assistant_axis_ndif/2_activations_ndif.py` writes (line 478 `torch.save({"vector": vec, "layers": ..., "type": vtype, ...`, line 457 `vtype = "pos_3"`). `n` for `default` is 150, which equals 30 questions (`kd_role_groups.json` `question_set`) × 5 instruction variants (`kd_instructions/pc_honest_novice.json` has 5). | VERIFIED (names, groups, schema compared); **INFERRED** that these are the owner's `run-kd/vectors_s3` files, written by asix's `2_activations_ndif.py`. Not checkable in git: `run-kd/` has 0 files in any ref. |
| `persona vectors/vectors_s3/` (uncommitted) | His docstring: "the same run as the single-layer bank" (whose `source_run` is run-paper16-shared). `vectors_s3` is the owner's naming (`kd/scripts/run_kd_chain.ps1`: tags `vectors_all` / `vectors_s3`). | DOC-ONLY + INFERRED |
| `persona knowledge denial modified/` (uncommitted) | His summary: the same 36 personas re-extracted to 142–150 rollouts. Origin not stated. | DOC-ONLY; origin NOT-FOUND |
| exporter that produced `paper16_L20.pt` / `deceptive_L20.pt` | No script on either branch writes the keys `persona_vector_norms_median` / `deception_axis_norm`. `git grep` over asix and pc-ood finds them only inside his four data files. | NOT-FOUND |

**Timeline of the kd bank.** A version mismatch is possible here, so it goes to Phase 1. All VERIFIED by commit dates (+0600):

1. 2026-09-19 02:58, asix `5b43a7a`: kd bank added, with role **`pc_candid_unknowing`**.
2. 2026-09-20 17:53, pc-ood `9774075`: first results using the 36-persona bank. The summary lists **`pc_honest_novice`** (n = 110). The bank itself is not committed yet.
3. 2026-09-21 11:07, asix `3dfdb67`: renames `pc_candid_unknowing` → **`pc_honest_novice`** and edits **all 36** instruction files ("Broaden the kd persona bank to genuine ignorance"; 12–26 changed lines each).
4. 2026-09-21 22:41, pc-ood `148ddeb`: commits the 36 `.pt` files. They include `pc_honest_novice.pt` and no `pc_candid_unknowing.pt`.

So his vectors carry a role name that asix committed only 17 hours after his first results. Which instruction text produced them cannot be told from git (INFERRED: the owner ran the renamed bank before committing it).

**His bank vs the paper16 bank** (VERIFIED, script §J). At layer 20, the 36-persona bank and `paper16_L20.pt` share 26 names, and **none is bit-identical**:
- max ‖diff‖ = 0.971.
- n per persona is higher in the extended bank for all 26, for example `default` 100 → 150 and `pc_handler` 64 → 106.
- His docstring says "vectors differ by up to 0.78", without naming a layer. My number differs, but the layer and reference may differ too, so this is a **point to check, not a CONFLICT yet**.

### 4b. Collaborator → owner

| asix item | what it took from pc-ood | date | tag |
|---|---|---|---|
| `src/assistant_axis_ndif/acts_cache.py`, `batching.py`, `tests/test_acts_cache.py`, `tests/test_batching.py` | Port of his resumable cache and batcher (same basenames and docstring; content differs, so there is no blob match) | `ed8daca` 2026-09-15 02:12 | VERIFIED (commit msg "Port the resumable activation cache and token-budget batcher" + text) |
| `kd/personas/kd_role_groups_collab.json` | Contrast `collab_extended` = "The collaborator's ext_axis1 (12v13) from feat/persona-pc-ood, reproduced exactly" | `4d0d046` 2026-09-22 02:46 | VERIFIED |
| `kd/docs/FINDINGS.md` (~lines 4389–4845), `docs/COLLABORATOR_BRIEFING.md`, `docs/ROADMAP_TO_DEFENCE.md` | Numbers from his `persona_ood_ext_L{20,40}/{summary.csv,zero_shot.csv,matrix_*.csv}`, read "via git show" at **`f386a72`**. The briefing says the abstract's 0.904/0.916 is his `ext_axis1` (ledger F-97). | `4d0d046` 09-22 02:46, briefing revised `43ee6d3` 09-22 17:37 | VERIFIED (text present); whether the numbers match is for Phase 2 |

- **Are the cited files still current?** VERIFIED: between `f386a72` and pc-ood's tip, `summary.csv`, `zero_shot.csv` and every `matrix_*.csv` in `ext_L20`/`ext_L40` are unchanged. The files that changed are `ext_L20/summary.md`, `zero_shot_operating.csv` (L20, L40; rows added per commit msg) and `zero_shot_fig2.png` (L20, L40).
- **Blob overlap.** Section D of the script, VERIFIED: 0 of his 490 blobs in asix HEAD, 0 in asix history, 0 objects new on both sides since MB, and 0 asix-new objects in pc-ood's tip.

---

## 5. Shared paths under DD with different content (VERIFIED)

- asix has **363** files under DD, all identical to MB. There are no asix commits touching DD since MB.
- pc-ood has 846 files under DD: 483 are new, and 7 differ from asix. All 7 differences are his changes:

| file (under DD) | MB→pc-ood +/− lines | his commits | what he changed (from diff/commit msgs) |
|---|---|---|---|
| `README.md` | +51 / −0 | `45c733d` | "Resumable runs and the activation cache" section |
| `deception_detection/activations.py` | +211 / −45 | `e391b3e`, `45c733d`, `fb039bb`, `d2f4cce` | per-example cache, token-budget batching, tail-only positions, shared `cache_entries` |
| `deception_detection/experiment.py` | +133 / −49 | `e391b3e`, `45c733d`, `fb039bb` | per-dataset score checkpointing and resume gate; follow-up/tail options |
| `deception_detection/postprocess.py` | +12 / −2 | `fd17728` | skip per-token HTML for tail-mode runs (diff read) |
| `deception_detection/scripts/experiment.py` | +5 / −5 | `e391b3e` | keep the run folder on failure instead of deleting it (diff read) |
| `deception_detection/scripts/export_html.py` | +12 / −1 | `7c28299` | control dataset read from config, not hard-coded "XXXX-22" |
| `tests/test_metrics.py` | +278 / −99 | `15fe502`, `d8c9427`, `fb039bb`, `fd17728`, `45b16d9` | tests for balanced accuracy, operating points, follow-up row |

Same basename, different path (no content match):
- asix `src/assistant_axis_ndif/{acts_cache,batching}.py` vs his `DD/deception_detection/{acts_cache,batching}.py` (see §4b).
- asix `DD/example_results/*/cfg.yaml` (upstream examples) vs his `results/XXXX-9-llama-all-followup/cfg.yaml`.

---

## 6. What each branch has that the other lacks (VERIFIED; entry counts from `ls-tree -r`)

**Only on asix** (MB and pc-ood have 0 unless shown):

| path | files | | path | files |
|---|---|---|---|---|
| `run-crossgen/` | 7,150 | | `docs/` (top) | 11 |
| `figures/` | 341 (MB/pc-ood: 54) | | `docs/concepts, developing, errors, gotchas, models, patterns, reference, remote, usage` (nnsight docs) | 105 |
| `src/assistant_axis_ndif/` | 134 | | `docs/papers/` | 8 |
| `cap/` | 94 | | `docs/thesis/` | 5 |
| `kd/` | 93 | | `papers/` (PDFs) | 11 |
| `run-paper16/` | 76 | | `.claude/` | 10 |
| `run-coords/` | 41 | | `scripts/` | 9 (MB/pc-ood: 8) |
| `scale/` | 8 | | `full paper draft for this experiments/main (1).pdf` | 1 (commit `cf11749`, 09-22) |
| `run-cap/` | 1 | | root files `CLAUDE.md`, `NNsight.md`, `remote_sizes.txt`, `test.ipynb` | 4 |
| gitlink `assistant_axis` | 1 | | | |

**Only on pc-ood:**

| path | files |
|---|---|
| `DD/results/` | 421 |
| `DD/persona including knowlede denial/` | 36 |
| `DD/deceptive/` | 4 |
| `DD/paper16_L20_sharedquestions/` | 3 |
| `DD/deception_detection/` | +12 (105 → 117): `acts_cache.py`, `batching.py`, `persona_pcs.py`, 4 configs, `scripts/{persona_ood,persona_ood_figures,persona_positions,pool_activations}.py`, `run_llama_all.sh` |
| `DD/tests/` | +7 (10 → 17): `test_acts_cache`, `test_batching`, `test_loo`, `test_persona_pcs`, `test_persona_positions`, `test_postprocess`, `test_score_resume` |

**Totals** (tree entries): MB 1,491 · asix 9,532 · pc-ood 1,974.

- `run-kd/` has **0 files on every ref** (VERIFIED). The owner's kd vectors are therefore not in git anywhere.
- INFERRED location of the collaborator's draft report: `full paper draft for this experiments/main (1).pdf`. It is on asix only and was committed by the owner; I did not open it. Confirm with the owner.

---

## 7. Submodule pins (VERIFIED)

| ref | assistant_axis | behavioral-self-awareness | mask | pacchiardi | soft-trigger | `.gitmodules` entries |
|---|---|---|---|---|---|---|
| asix | `a989619` | `9095b38` | `8c13097` | `0bd0f4d` | `8ddaf3c` | mask, soft-trigger, pacchiardi, assistant_axis |
| pc-ood | — | `9095b38` | `8c13097` | `0bd0f4d` | `8ddaf3c` | mask, soft-trigger, pacchiardi |
| origin/main | `a989619` | `9095b38` | `8c13097` | `0bd0f4d` | `8ddaf3c` | same as asix |
| main (local) | `a989619` | `9095b38` | `8c13097` | `0bd0f4d` | `8ddaf3c` | same as asix |
| MB `2e2a785` | — | `9095b38` | `8c13097` | `0bd0f4d` | `8ddaf3c` | same as pc-ood |

- The only difference: asix, main and origin/main add `assistant_axis` @ `a989619` (url `safety-research/assistant-axis.git`). The shared four pins are identical on all refs.
- `behavioral-self-awareness` is a gitlink with **no `.gitmodules` entry on any ref**.
- `.gitmodules` URLs are identical where present: `cadenza-labs/mask.git`, `maiush/cadenza-sleeperer-agents`, `KieronKretschmar/LLM-LieDetector`.
- `origin/claude/assistant-axis-submodule-p285ym` pins the same commit `a989619`, but at path **`assistant-axis`** (hyphen). asix uses **`assistant_axis`** (underscore).

---

## 8. Other branches (VERIFIED)

"vs asix" means commits on the branch not in asix / commits in asix not on the branch.

| ref | tip · author `<email>` · date (+0600 unless noted) | vs MB (ahead/behind) | vs asix (ahead/behind) | touches |
|---|---|---|---|---|
| `origin/feat/llama-layer30` | `df758c4` · **Mahdi `<mahdi.ndc21@gmail.com>`** · 2026-09-19 19:32 | 9 / 0 | 9 / 137 | = resumable-acts-cache (8 zami456 commits) **+ 1 Mahdi commit**: `configs/XXXX-9-llama-L30.yaml` (new) and `run_llama_all.sh` (+12/−4). No results. Third contributor name. Not an ancestor of pc-ood (merge-base with pc-ood = `7c28299`). |
| `origin/feat/resumable-acts-cache` | `7c28299` · zami456 · 2026-09-05 09:04 | 8 / 0 | 8 / 137 | 12 files under DD (cache, batching, config, runner, tests). Fully contained in pc-ood. |
| `origin/Saving` | `638bb9a` · Suraiya `<suraiya.shanaz@g.bracu.ac.bd>` · 2026-07-12 18:43 | 2 / 0 | 2 / 137 | `DD/deception_detection/activations.py` (+54), `DD/pyproject.toml` (+1): "features saving and resuming" |
| `origin/claude/assistant-axis-submodule-p285ym` | `2b22883` · Claude `<noreply@anthropic.com>` · 2026-09-07 15:36 **UTC** | 1 / 0 | 1 / 137 | `.gitmodules` + gitlink `assistant-axis` |
| `origin/claude/colab-activation-download-drive` | `97aee6a` · Suraiya (GitHub noreply) · 2026-07-16 21:51 | 18 / 0 | 18 / 137 | contains Saving; 15 Claude + 3 Suraiya commits; `notebooks/colab_extract_activations.ipynb`, `README.md`, `.env.example`, DD `activations.py`/`experiment.py`/`pyproject.toml` |
| `origin/claude/nnsight-colab-activation-download-f95pop` | `a005e10` · Claude · 2026-07-13 22:12 UTC | 4 / 0 | 4 / 137 | contains Saving; earlier version of the Colab notebook work |
| `origin/copilot/zami456explb` | `2e2a785` · zami456 · 2026-07-12 01:47 | 0 / 0 (= MB) | 0 / 137 | — |
| `origin/main` | `59cdaba` · Suraiya (GitHub noreply) · 2026-09-08 05:55 UTC "Add assistant_axis submodule pointer" | 2 / 0 | 0 / 135 | ancestor of asix |
| `main` (local) | `608b2a7` · Suraiya · 2026-09-09 20:45 "papers info claude skills" | 3 / 0 | 0 / 134 | 1 commit ahead of origin/main (unpushed); ancestor of asix, so nothing is lost |
| `origin/asix` | `43ee6d3` | 137 / 0 | 0 / 0 | identical to local asix |

---

## 9. Rough counts for pc-ood

- **Experiment families: 8 distinct** (INFERRED grouping, from scripts and configs):
  1. persona-PC transfer with the original bank (L20, L40)
  2. extended 36-persona bank (L20, L40, L44)
  3. extended bank at the follow-up token (L20)
  4. modified-rollout bank (L20, L40, L44)
  5. leave-one-dataset-out (L20, L40)
  6. per-token positions (L20, L40, L44)
  7. Mean-Probe runs (L20, L40, L44, plus the 07-12 run)
  8. Follow-Up Probe run (L20)
  - Plus the uncommitted INLP experiment (DOC-ONLY).
- **Result folders: 19.** That is 15 analysis folders plus 5 probe-run folders, of which 4 are plots-only. Seen as layers, 10 of the 15 analysis folders are layer variants of 4 designs.
- **Result files: 421** (316 CSV, 87 PNG, 15 MD, 2 NPZ, 1 YAML), 13,658,832 bytes.
- **Input-data files: 43** (38 pt, 2 npz, 2 json, 1 md), 97,218,383 bytes.
- **Code: 20 .py** (12 package/scripts + 8 tests), 5 yaml (4 configs + 1 dumped cfg), 1 sh.
- **Numeric claims:** about 1,000 numeric tokens in his 17 docs (384 lines), plus 383 in his commit messages. Many repeat across docs, messages and CSVs, so distinct claims are fewer. I would estimate 150–300 distinct claims (INFERRED, not counted).

---

## 10. Points to check in later phases (not verdicts)

1. **Which kd instruction version produced his 36-persona bank?** (§4a timeline) His vectors use `pc_honest_novice`, a name asix committed 17 h after his first results; the same asix commit rewrote all 36 instruction files. The bank itself was committed 29 h after the first results that used it. Phase 1 check on the lab PC: md5 of his 36 `.pt` vs the owner's `run-kd/vectors_s3/`, and that run's timestamps.
2. **Most of his CSVs cannot be recomputed from git.** They need pooled activations, token projections and Mean-Probe run files, all on his local disks (§2c). Two banks behind committed results are uncommitted: `persona vectors/vectors_s3` for `persona_ood_L40` and the regenerated `persona_ood`, and `persona knowledge denial modified` for the three `mod` folders. Only the LOO numbers (`loo_scores.npz`, with labels) are recomputable from git alone. The owner must decide whether to ask him for these files.
3. **Base-probe rows point at files that are not committed.** Their `results_table.csv` / `detector.pt` exist only locally (§2c).
4. **"Up to 0.78" vs 0.971.** His docstring says the re-extracted vectors move "up to 0.78". At L20, ext-bank vs `paper16_L20.pt` I measure max ‖diff‖ = 0.971. The layer and reference may differ; check in Phase 2.
5. **Claims changed after pushing** (e3f2258, f386a72, 45b16d9 corrections). asix read pc-ood at `f386a72`. The files it cites are unchanged at the tip (VERIFIED), but `summary.md` texts are not.
6. **Broken `role_groups` field.** `paper16_L20.json`'s `role_groups` holds key names (`["contrast_with","note","roles"]`) instead of members (VERIFIED). He recovered the groups by search and hard-coded them. His lists match asix `kd_role_groups.json` (VERIFIED by reading both). The exporter that wrote the broken field is NOT-FOUND in git.
7. **A third contributor.** "Mahdi `<mahdi.ndc21@gmail.com>`" (`origin/feat/llama-layer30`) added a layer-30 config and no results. Ask the owner whether this is in scope.
