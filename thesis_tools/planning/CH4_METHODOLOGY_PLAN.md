# CH4_METHODOLOGY_PLAN — Chapter 4 (Proposed Methodology)

Approved 2026-09-24. Writing starts only when the owner says so.

> **UPDATE 2026-09-24 (supervisor framing, THESIS_BRIEF Batch 4).** Methodology is now split into two chapters:
> - **Ch 4 "Proposed Methodology: The Pipeline"** (`chapter_5.tex`) keeps §4.1–4.3 and §4.5–4.9 below, framed as one holistic design, and adds a design-rationale section.
> - **§4.4 Persona Banks, plus the built datasets** (`sycophancy_mt`, the HP-KR slices) and the tools (controls, nulls, LOCO, factor probes), move to the new **Ch 5 "What We Built"** (new `chapter_4.tex`). The bank-version table below feeds that chapter.
> - Ch 4 keeps only a short pointer to the banks it uses.

**Target file:** `thesis/chapters/chapter_5.tex`. `main.tex:135-136` has `\chapter{Proposed Methodology}` and inputs `chapters/chapter_5.tex`.

**Primary source:** `thesis_guide/METHODS_SOURCES.md` §1–§14, which the owner audited against raw data. `audit/METHODS_CHAPTER.md` is a draft only.

**Order of work (owner, 2026-09-24):**
1. Outline from `thesis_guide/`.
2. Settle the open questions with the owner.
3. Write.
4. **Last step: verify every number against raw data in all four places.**

### Section outline (template headings kept where they fit)
- **4.1 Design Process / Methodology Overview.** A pipeline figure: persona banks → role-play rollouts → judge → persona vectors → PCA / contrast → projection features → probes → crossgen / LOCO / zero-shot.
- **4.2 Models and Compute** (METHODS_SOURCES §1):
  - Llama-3.3-70B-Instruct (80 layers, hidden 8192)
  - NDIF via nnsight; no local GPU
  - hook `model.layers[L].output`, the post-MLP residual
  - fp32 reductions (bf16 bug: 1,816 / 11,156 = 16.3%)
  - NDIF memory model (512 / 384 MiB)
  - `ai_audit` excluded
  - scale rungs: Llama-3.2-3B / 3.1-8B
  - judge: Llama-3.1-70B, with a 405B cross-check
  - orchestration: scheduled tasks, resumable cache (credited to the collaborator)
- **4.3 Data Collection** (§2–§5):
  - Table of 12 datasets: origin, rows (deceptive / honest), label semantics; total 15,278.
  - The collaborator's 6-dataset set.
  - Our built datasets:
    - `sycophancy_mt`: 1,182 rows, 434 / 748
    - HP-KR text-matched slice: 258 rows, 124 / 134
    - HP-KR knowledge-matched slice: 2,001 rows, 139 / 1,862
    - natural sarcasm
    - gender-secret and soft-trigger, off-policy, used in Q11 only
  - 4.3.1 **Data Cleaning:** MASK relabel (1,000 → 585), roleplaying's 84 null labels, Llama-only filter, the judge filter on persona rollouts.
  - 4.3.2 **Data Transformation:** activation stores passA / B / C / Q11 / B_fine / P, response span, max_length 8192 (1024 for the scale rungs), pooling (mean, last), the collaborator's detection-token mask.
  - 4.3.3 **Data Integration:** HP-KR label `knows AND denies`, the slices, the leak (~81%) and its guard (59 / 258 survive).
  - 4.3.4 **Leave-one-cluster-out clusters:** `lb_harm`, `lb_scenario`, `apollo`, `mask`, `ours` (`30g_clustered_holdout.py:57-64`), with the dataset membership of each.
- **4.4 Persona Banks** (§6):
  - Banks: hf275 / hf100 / own100, paper16 (26 personas; 8v7, 5v5, 13v12), paper16_shared, kd (36; s3 / s23 / all; 10 kd roles), the collaborator's arms (8v7, 4v6, 12v13, 5v5, 13v12, the mod control), cap (13 roles, A / B / C, teacher forcing, v3 questions), sarcasm (v1 pilot, natural; v2 never built), scale-rung banks.
  - **For each bank, the definition-file version each run used,** from the git history plus lab-PC hashes (see Bank versions below).
  - How the prompts were built: names from Mahadik & Skapars; system prompts written by us; 5 variants.
- **4.5 Persona Vector and Axis Extraction** (§7):
  - Generation: T = 0.7, top-p 0.9, ≤512 tokens.
  - Judge filter: s3 (score 3 only) or s23 (scores 2–3); min_count 20.
  - Masked mean.
  - Layers stored: 80 for paper16 **and for the kd bank** (settled: 36/36 `vectors_s3` are (80, 8192); see Bank versions below).
  - Centering cancels.
  - Contrast = mean(pos) − mean(neg); the anchor sits in the PCA basis only.
  - 5 PCs per layer, oriented by `orient_pcs`.
  - k: default 5, `@2` features, k = 3 for the scale rungs; the paper's k = 1 / 3 / 2 (k = 1 never run at 70B).
  - Layers: pre-registered L20 / L22 / L40, L44 post hoc; the collaborator's 16 layers; scale mid-layers 14 / 16.
  - Fidelity: 36 steps (16 match, 11 resolved, 6 deliberate, 4 accidental: AD-1–AD-4).
- **4.6 Features, Probes and Controls** (§9):
  - Features: raw, pc@k, axes, axes_resid, erase, zero-shot contrast / PC1.
  - Controls: datapca@k (fitted on the source's training rows), random@k (QR, seeded), `random3` (collaborator), length.
  - Probes: `paper` (LR, C = 1), `lr_reg10`, `mms`; the LB base probe (lr, reg 10, RepE, per token); raw 8192-d probe; the collaborator's probe (5-fold CV); **pooled probe** (the owner's `30_premise.py` LODO with the equal-budget `pooled_lodo_match` arm; the collaborator's `leave_one_out` naive / balanced, which came first); Follow-Up and Upper-Bound probes.
- **4.7 Evaluation Protocols and Statistics** (§10):
  - Crossgen: 80 / 20 split by item_id; 11×11 (owner), 6×6 (collaborator).
  - Zero-shot scoring, and the dim = 1 identity.
  - Pooled LODO / LOCO; the leak-guarded crossgen.
  - Metrics: AUROC; BA and TPR at 1% FPR (Alpaca p99), with realised FPR.
  - Uncertainty: row bootstrap (1,000); dataset bootstrap (10,000); BH at q < 0.05; random-direction null (1,000); cluster permutation; ±0.02 equivalence; seed 42.
- **4.8 Implementation of Selected Design:**
  - Code layout on both branches: `src/assistant_axis_ndif/` and the collaborator's `persona_ood.py` / `persona_pcs.py`.
  - The branches agree to 0.0015 on matched cells.
  - Reproducibility: `numbers.tex`, the audit scripts.
- **4.9 Summary of the method.** A table of which bank, features, probe, layer and protocol each result chapter uses.

### Bank versions: what the runs actually used (verification agent, local git plus audit caches)
**Method:** cached rollouts were compared text-for-text with each committed version of the definition files.

| Bank | Version the run used | Counts | Evidence |
|---|---|---|---|
| paper16 | 9511ba8d (2026-09-15), same as today | 26 personas; 1,400 rollouts; 1,350 judged; 1,089 kept at s3 (1,338 at s23); lowest n = 21 | Vectors (80, 8192) fp32. `logs/paper16_deception.log:1` runs 11 min after the commit. `run_paper16_chain.ps1:61-103` |
| paper16_shared | definitions b9a497ee / 9511ba8d, unchanged since | 2,600 rollouts | The collaborator's `paper16_L20.json` @16def2de: 1,755 of 2,500 kept, min 26. Stores are LAB-ONLY |
| kd (36) | **3dfdb673 (v2)** | 5,400 rollouts; 5,250 judged; s3 keeps 3,599 (68.6%); s23 keeps 5,153 | v2 was committed 2026-09-21, but the run was 2026-09-19/20, so the run preceded the commit. The cached rollouts match v2, not v1 (5b43a7af), which never ran. **Vectors are 80 layers**: 36 of 36 `vectors_s3` are md5-identical between the collaborator's branch and the lab cache |
| KD roles attribution | **owner** | — | Every commit to `kd/personas`, `kd/questions` and `kd_personas.py` is the owner's. The collaborator's branch holds only the 36 `.pt` files, which are identical to the owner's `vectors_s3` |
| Collaborator's arms | built on the owner's `vectors_s3` | 8v7 / 4v6 / 12v13 / 5v5 | `method.tex:75-85`. `persona_pcs.py` was last changed at 97740753; its read was denied, so still to verify |
| persona_ood_mod | the owner's `vectors_s23` | — | matched by n-counts |
| cap (13) | dc24e960, same as today; questions **v3** | 1,300 rollouts | Run 2026-09-20/21 preceded the commit. Turn-1 texts are teacher-forced; max 220 new tokens. Filters s23 and all; no s3 arm |
| cap format controls | — | 19 roles | Not verifiable: no cached rollouts |
| sarcasm pilot | 080a9794 | 3 arms × 20 questions | "v1–v5" are deadpan instruction variants of `pc_ironic_critic`, not five separate banks. Max 320 tokens |
| natural sarcasm | — | 600 rows (300 / 300), drawn from 4,400 FigLang rows | eval only |
| scale rungs, paper16 | definitions unchanged | 2,600 rollouts per rung | 3B / 8B, all layers (28 / 32) |
| scale rungs, kd | — | 5,400 rollouts per rung | Run 2026-09-23/24 in `expLB-resplit`. LAB-ONLY |
| own100 / hf100 / hf275 | `role_groups.json` 5d7b8560; hf275 groups b9a497ee | hf275 loose 120 + anchor 8 + breadth 39; strict 46 + 2 anchors | hf100 groups are verified equal to own100 |

**Corrections to the drafts:**
- `PERSONA_CONSTRUCTION.md:177-179` gives (21, 8192). That is the passB evaluation store, not the persona vectors, which are 80 layers.
- "unified arms added at 2ad758fb" is inaccurate: that commit adds only result CSVs.
- `kd/README.md:62` still says `candid_unknowing`.
- `cap_role_groups.json:6` still says "judgement questions".

### Code-level verification of the methods (two agents, local repo + 16def2de + lab caches)
**Verified as the draft states:**
- **Datasets:** all 13 row counts, including roleplaying at 203/84 deceptive/honest (287 load; 84 null).
- **HP-KR:** its 2×2 table, and the regex AUROC of 0.9125.
- **Splits:** the 80/20 item_id split; realised test fractions 0.185–0.202.
- **LOCO:** the clusters (`30g_clustered_holdout.py:57-64`):
  - `lb_harm` = HP-C, HP-KR, kr-matched
  - `lb_scenario` = ID, CG, IT
  - `apollo` = ai_liar, roleplaying, sandbagging_v2
  - `mask` = mask
  - `ours` = sycophancy_mt
- **LOCO result:** +0.125442 (10/11) at `run-premise/clustered/clustered_L40_mean_s42.json`.
- **Pooled LODO, both authors:** budget 666; realised 412–500. The owner's equal-budget arm is named `pooled_lodo_match`.
- **Statistics:**
  - AUROC
  - row bootstrap: 1,000
  - dataset bootstrap: 10,000
  - BH correction
  - random-direction null: 1,000
  - cluster permutation: B = 200, p = 0.38
  - equivalence test: ±0.02, 90% CI
  - seed: 42
  - Mahalanobis R²: 0.023
- **Models and settings:**
  - Model IDs: 70B, 3B, 8B, with a 70B judge and a 405B cross-check.
  - Hook, fp32 reductions, `max_length` 8192/1024, memory formula.
  - Generation: T 0.7, top-p 0.9, 512 tokens.
  - `min_count`: 20.
  - Vectors: 80 layers.
  - PCA: 5 PCs, orient rule; the contrast and zero-shot projection.
- **Features:** all features, with k = 5 and k = 3.
- **Probes:** `paper`, `lr_reg10`, `mms`, and the LB base probe.
- **Collaborator's code:** the collaborator's probe, pooled naive/balanced arms, Follow-Up probe and 16 layers.

**Must be written differently from the draft:**
1. **Pre-registration of L40.** Commit f5d6904 (2026-09-21 13:00) said L40 was *not* pre-registered. Commit 63aabb4b, 19 minutes later, reclassified it as a literature choice: Lu et al.'s midpoint. Disclose the amendment.
2. **Detection metric differs by author.** The owner's pipeline reports **TPR@1%FPR** (`27_crossgen.py:275-279`). The collaborator reports **BA@1%FPR** (`persona_ood.py:203-239`).
3. **Why `ai_audit` is excluded.** The code says "8 deceptive / 8 honest items -- too few" (`eval_datasets.py:286`). The docs say compute: 4.23–13.57 GiB per row, measured from `items.pt`.
4. **The 15,278 total** is 9 labelled datasets plus Alpaca. It excludes `sycophancy_mt`, and kr-matched is a view over HP-KR, not extra rows. Adding the 1,182 `sycophancy_mt` rows gives 16,460 (passB).
5. **LOCO and pooled LODO use a row-level split, not item_id.** The collaborator's LODO means include RepE (6 datasets).
6. **The two authors' PCA bases differ.** The owner puts the anchor (`default`) in every basis; the collaborator's basis is D∪H only, using sklearn PCA with k = 3.
7. **Two different knowledge-denial contrasts.**
   - Owner: `knowledge_denial` = 4 v 3 (sincere); `kd_assertion` = 4 v 3 (plain).
   - Collaborator: `denial` = 4 v 6; extended = 12 v 13.
8. **k = 1:** the owner never ran it at 70B; the collaborator did (trained `pc1`).
9. **Upper-Bound probe:** not run by either author. It is Liars' Bench Table 8, cited as-is.
10. **nnsight:** 0.7.0 in the pipeline venv (`setup_venv_aa.ps1:33`). The root `uv.lock` pins 0.3.6, which is unused by the pipeline.
11. **`random@k` is one seeded draw.** The collaborator's `random3` averages 5 draws.
12. **`datapca`** is fitted on the unbalanced 80% training split, with centering.
13. **Pooled activations are stored as fp16.**
14. **The collaborator's code for the 13 remote-pooled layers is not on the collaborator's branch.** `layer_sweep.py:200` passes `span_mask=`, a parameter the callee doesn't accept. **This covers L36, the post-hoc headline layer.**

**Not found locally (need SSH or the collaborator):**
- the `knowledge_matched` builder script;
- the original passA/passC commands and passQ11;
- the kd scale-rung crossgen config;
- `sycophancy_mt` response text;
- the collaborator's remote-pooling code and extended-vectors folder;
- the bf16 1,816/11,156 count (prose only).

### Owner decisions for Chapter 4 (2026-09-24)
- **Pre-registration:** disclose the amendment. L20 and L22 come from the literature. L40 is Lu et al.'s midpoint, fixed at 63aabb4b, 19 minutes after f5d6904, and before any 80-layer axes number. L44 and L36 are post hoc.
- **Metrics:** AUROC is the shared primary metric. Operating point, labelled per pipeline: TPR@1%FPR in our 11×11 grid; BA@1%FPR in the collaborator's LB comparison. Both carry the realised FPR.
- **ai_audit:** excluded for two reasons: too few items (8/8), and memory. **Do not mention NDIF in this sentence.** The memory figure is 4.23–13.57 GiB of attention per row.
- **L36 remote-pooling code:** the owner believes the collaborator committed everything, possibly on top of `asix` or before `asix` existed. I searched every fetched ref (`git log --all -S span_mask`):
  - The only commit touching `span_mask` is ea1ce52b (zami456, 2026-09-23 23:46). It *adds the caller*, `layer_sweep.py:200`.
  - `trace_layer_outputs` has never accepted `span_mask` on any branch. Its history is 638bb9a0 (owner, 2026-07-12) → e391b3e6 → 45c733d9 → fb039bbd → d2f4cce6 → 572095cb. It returns un-pooled `[batch, seq, layer, emb]`, but the commit message describes server-side means.
  - The owner's vendored copy on `asix` has no `span_mask`.
  - **So the working pooling function was never pushed.**
  - The results are committed: `persona_depth/depth.csv`. So is the L40 cross-check recorded in the commit message: row cosine 0.99989, same AUROC to 4 dp.
  - **Plan:**
    - In Chapter 4, describe the remote pooling as ea1ce52b documents it.
    - Mark `\todo{MISSING: committed pooling code for the 13 remote layers}`.
    - The owner asks the collaborator to push their local `activations.py`.
    - After that, `git fetch` (allowed) and re-check.
- **Carried to Chapter 5 (not Chapter 4):** ea1ce52b's own message says to quote the base probe at its best layer. That is L32: LB-average AUROC 0.8498, against 0.697 at L20. This narrows the zero-shot margin to about 0.055, which matters for the headline.

**Blocked:** the agent's SSH attempts were denied by the permission system. The lab-PC reads move to the final verification step, which I run myself (the owner approved read-only SSH).

### Final step: verify against raw data in all four places
Every `% src:` number in `chapter_5.tex` is checked at its raw location, and each gets a verification log row: value, place, path, read value, match.
- **GitHub: the collaborator's two branches.**
  - `origin/paper` pinned at `16def2de`; the tip is `d5b92c5a`, three abstract-only commits past the pin.
  - `origin/feat/persona-pc-ood` at `45b16d90`.
  - Read with `git show <sha>:<path>` only.
- **GitHub: the owner's branch.** `asix` @ `e917ecd3` plus the working tree.
- **Drive: the owner's store** (`tamara:`). Read with `rclone lsf / lsjson / cat / size` only.
- **Lab PC: the owner's store.** Read-only SSH with `-o BatchMode=yes`, inside `C:\Users\user4\expLB\` and `expLB-resplit\`.

A mismatch is reported to the owner. It never gets silently fixed.
