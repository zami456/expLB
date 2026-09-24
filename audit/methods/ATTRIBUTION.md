# Authorship attribution — who built each design, control and baseline

_Phase M, workstream M3 (`audit/PHASE_M_PLAN.md`). Written 2026-09-23. Read-only: every statement
below comes from a git object, a file on disk, or a read-only command on the lab PC, run during
this audit._

**Who this is for.** Both authors of the joint thesis. It is written to be readable by either of
them without either being flattered. Where the collaborator built something, it says so as plainly
as where the owner did.

**Refs used.** `asix` at **`d280d23`** (2026-09-23 02:09; the tip has moved past Phase 0's `43ee6d3`
— `d280d23` "afteer audit" only normalises line endings in 8 `.ps1` files and re-encodes
`kd/docs/FINDINGS.md`, and adds no design). `origin/feat/persona-pc-ood` at `45b16d9`
(2026-09-22 12:45). Merge-base `2e2a785` (2026-07-12).

**Short names.** **OWNER** = Suraiya Shanaz `<suraiya.shanaz@g.bracu.ac.bd>`, branch `asix`.
**COLLABORATOR** = zami456 `<abdullah.al.zami@g.bracu.ac.bd>`, branch `origin/feat/persona-pc-ood`
("pc-ood"). **MB** = merge-base `2e2a785` (2026-07-12). **DD** = `src/probes/deception-detection/`.
**The draft** = `audit/phase0/draft_text.txt`, the extracted text of
`full paper draft for this experiments/main (1).pdf` (committed on `asix` at `cf11749`; Phase 0
INFERRED it is the collaborator's — it has no author names, and its result sections track his
commits). **His notes** = the 17 `.md` files he committed under `DD/results/**` and `DD/deceptive/`.

---

## 0. How to read this, and what it cannot prove

Five rules were applied, and they matter more than the table:

1. **An earlier commit date is evidence of authorship, not proof of it.** Both authors worked
   through Claude sessions and talked to each other; a design can be discussed before anyone
   commits it. Every verdict below rests on *who committed a working artifact first*, which is the
   only thing the files can show. Where a design plausibly predates its commit, the row says so.
2. **Three dating channels exist, and only one is independent.** (a) Git commit dates — independent
   and trustworthy. (b) The ledger's own `### F-nn · <date>` headers — **self-reported**: Phase 0
   established (RF-9) that F-01…F-66 all entered git in a *single* commit, `6432257`
   (2026-09-21 03:26), although dated 09-19…09-21. So a finding's stated date is a claim, not a
   git fact. (c) Lab-PC file timestamps — real, but a copy resets them, so they are used here only
   where Phase 1 already verified the chain.
3. **Attribution is of *designs*, not of effort or of ideas.** Nothing below speculates about who
   thought of something. It reports which working file appeared first, and where.
4. **Much of the owner's pipeline predates its first commit.** Commit `8078702`
   (2026-09-10 16:19) says in its own message that the NDIF pipeline "lived untracked inside the
   `assistant_axis` submodule … unversioned in both repos". So for §2.4 the first-commit date is an
   **upper bound** on when the code was written, not the date it was written.
5. **Absence from his branch is not absence from his knowledge.** His branch holds only files under
   `DD/`. He may have read the owner's designs without copying them. What the table records is
   whether a *version of the artifact* exists on his branch.

### Verdict counts

| Verdict | §2.1 banks | §2.2 controls | §2.3 baselines | §2.4 infrastructure | **Total** |
|---|---:|---:|---:|---:|---:|
| **OWNER** | 6 | 11 | 2 | 5 | **24** |
| **COLLABORATOR** | 0 | 0 | 0 | 1 | **1** |
| **JOINT** | 0 | 0 | 1 | 0 | **1** |
| **UNCLEAR** | 0 | 0 | 0 | 0 | **0** |
| rows counted | 6 | 11 | 3 | 6 | **26** |

Counting notes: §2.2's last inventory row bundles three tests, which are split into 2.11–2.13
below and counted as one. §2.3's first inventory row bundles two designs with different
authorship, split into 3.1 (JOINT) and 3.2 (OWNER); row 3.4 (raw-probe comparator) is the
replicated paper's protocol and is not counted as a contribution on either side. §2.4 row 4.4
is counted OWNER and carries a separate collaborator finding inside it.

Separately, **§5 lists 5 designs that are the collaborator's own** and that `METHODS_INVENTORY.md`
Part 2 does not list at all.

Plus **4 designs listed in Part 2 that have no committed artifact on either branch** (§5), and
**5 designs of the collaborator's own** that Part 2 does not list at all (§4).

Two rows carry an important qualifier that is *not* a different verdict: **two of the eleven
"controls introduced" are controls the replicated paper already runs** (§2.2 rows 1 and 3). See
§6, red flag A.

---

## 1. §2.1 — Datasets and banks built

| # | Design | First commit on `asix` (hash · date · author) | On his branch? | Named in his draft / notes? | **Attribution** |
|---|---|---|---|---|---|
| 1.1 | **`harm-pressure-kr-matched`** — the matched slice (the HP-KR confound `deceptive = knows AND denies`, fourth cell empty; text-matched and knowledge-matched slices) | Design specified in `kd/docs/DESIGN.md` lines 54–68 at **`5b43a7a` · 2026-09-19 02:58 · Suraiya Shanaz** ("some fix and new persona"). Dataset name `harm-pressure-kr-matched` first in code at `6432257` · 2026-09-21 03:26; repaired at `6a0e016` · 2026-09-21 11:33 ("9 honest deniers become 134") | **No.** `git grep -iE "kr.matched\|kr_matched"` over the whole pc-ood tree returns nothing | **Draft: as FUTURE WORK only.** Draft p.12–13: *"Wording-controlled contrasts. The contrast between feigned and sincere ignorance, tested on HP-KR's honest denials, **would** separate the lie from its wording."* The design was built and run 2–3 days before the draft was compiled | **OWNER** |
| 1.2 | **The kd persona bank** — 36 roles, feigned-ignorance vs sincere-ignorance vs plain-assertion | `5b43a7a` · **2026-09-19 02:58** · Suraiya Shanaz — `kd/docs/DESIGN.md`, `kd/personas/kd_role_groups.json`, 36 × `kd/personas/kd_instructions/*.json`, `kd/scripts/kd_personas.py`, `kd/questions/kd_questions_shared.jsonl`, `kd/README.md`, `kd/docs/REPORT.md` all added in this one commit. Bank re-cut at `3dfdb67` · 2026-09-21 11:07 | **Yes — as 36 binary `.pt` files only** (`DD/persona including knowlede denial/`, `148ddeb`, 2026-09-21 22:41). **RF-1 (Phase 1): all 36 are byte-identical (md5) to the owner's lab `C:/Users/user4/expLB/run-kd/vectors_s3/*.pt`.** No persona text, no group file, no generator on his branch | **Notes: partly. Draft: no.** His `persona_ood_ext_L20/summary.md` names the roles and the two poles ("knowledge-denial" / "knowledge-candour") and says *"Groups were assigned by name; the files carry no group metadata"* — i.e. he re-derived the grouping. The draft describes "ten roles that feign ignorance, are sincerely ignorant, or plainly assert what they know" with no source | **OWNER** (bank, personas, questions, rollouts, vectors). His re-grouping of her poles is his own and differs from hers — see §6 red flag B |
| 1.3 | **The capitulation bank** — 13 roles, teacher-forced turn 1, three variants | Scripts `cap/scripts/cap_personas.py`, `cap/scripts/gen_multiturn_ndif.py`, `cap/scripts/cap_significance.py` at **`6432257` · 2026-09-21 03:26**; personas, `cap_role_groups.json`, `cap_questions.jsonl`, `cap/docs/DESIGN.md` at `dc24e96` · 2026-09-21 03:37 | **No.** No `cap/` path, persona, question file or capitulation artifact anywhere on pc-ood; the one `capitulat` string hit is text inside a pre-existing Alpaca rollout JSON | **No.** `grep -ci capitulat` on the draft = **0** | **OWNER** |
| 1.4 | **`sycophancy_mt`** — multi-turn capitulation (commit-then-challenge, judge-labelled) | `src/assistant_axis_ndif/13_sycophancy_multiturn.py` at **`152a174` · 2026-09-16 12:58** ("capitulation under challenge, judged rather than string-matched"). Supersedes the single-nudge `10_sycophancy_ndif.py`, `330642f` · 2026-09-15 18:22 | **No** | **No.** No `sycophancy_mt`, `multi-turn` or `multiturn` in the draft; his own files record `n_datasets=5` / six-dataset grids that exclude it | **OWNER** — with lineage stated in her own docstring: *"This is Anthropic's `are_you_sure` design instead"*. The **construct** is adopted; the dataset, the judge protocol and the correct-in-turn-1 filter are hers |
| 1.5 | **The deadpan/sarcasm bank** (v1 pilot, v2 design) | Design `cap/docs/SARCASM_DESIGN.md` at **`3d80071` · 2026-09-21 17:48** ("for review before anything is generated"); v2 rewrite `5ced550` · 2026-09-21 19:03; pilot personas + questions + marker scorer at `080a979` · 2026-09-21 20:34; natural-sarcasm Gate 3 (`cap/scripts/build_sarcasm_natural.py`, `cap/run-sarcasm-natural/`) at `cf11749` · 2026-09-22 16:51 | **No** | **No.** `grep -ci sarcas` on the draft = **0** | **OWNER** — see §5: the **v2 bank itself was never built** |
| 1.6 | **Own WMDP knowledge scoring** (re-scoring the model rather than trusting the benchmark's knowledge label) | `kd/scripts/build_knowledge_eval.py`, `kd/scripts/score_knowledge_ndif.py`, `kd/scripts/gen_hpkr_denial.py`, `kd/scripts/phase2_gate.py` at **`e9969af` · 2026-09-21 10:55**. Result file `kd/run/knowledge/scores.jsonl` committed at `3dfdb67` — **truncated at 204,800 B (RF-5)** | **No** | **No** | **OWNER** |

---

## 2. §2.2 — Controls and tests

| # | Control | First commit on `asix` (hash · date · author) | On his branch? | Named in draft / notes? | **Attribution** |
|---|---|---|---|---|---|
| 2.1 | **Matched dataset-PCA control at matched k** | `src/assistant_axis_ndif/27_crossgen.py` — `datapca_projections()` + `BASELINE_FEATURES` at **`b9a497e` · 2026-09-18 11:26** ("cross gen"). Stated as a rule in `kd/docs/DESIGN.md:147` at `5b43a7a` · 2026-09-19 | **No** (`datapca` / "dataset PCA" absent from pc-ood) | **No** (`grep -ciE "dataset.PCA\|datapca"` on the draft = 0) | **OWNER — but the control is the replicated paper's, not new.** Her own docstring says so: *"The paper's dataset-PCA control"* (`27_crossgen.py:105`), and `docs/papers/persona-coordinates-notes.md` §5.5 tabulates it from the paper's Figure 10. **What is hers** is running it *at matched k against every persona feature* and putting it under a dataset-level bootstrap |
| 2.2 | **Dataset-level bootstrap** (resampling unit = dataset, not row) | `src/assistant_axis_ndif/23_significance.py` at **`34c6b70` · 2026-09-17 01:09**; docstring defines "row bootstrap" vs "dataset bootstrap" and makes the dataset one decisive, with BH at q = 0.05. Phrase "dataset-level bootstrap" first at `6432257` · 2026-09-21 | **He has a bootstrap, but a different one.** `persona_ood.py:409-421 bootstrap_delta()` resamples **test rows, stratified by label** — his own docstring: *"an interval on the evaluation set, not on the fit"*. `loo_bootstrap.csv` is row-level. The draft agrees: *"Our bootstrap intervals resample examples, not models or prompts"* | Draft mentions "bootstrap" 4× — always the row-level one. The dataset-level unit is not in the draft | **OWNER** for the dataset-level unit (the thing being claimed). His row-level bootstrap is **his**, is correctly described in his own files, and answers a different question |
| 2.3 | **Per-dataset random-direction null** | `src/assistant_axis_ndif/25_zeroshot_nulls.py` at **`34c6b70` · 2026-09-17 01:09** — n random unit directions on the same rows/layer, empirical p on \|AUROC − 0.5\| | **Yes, independently and later.** `persona_ood.py:475 random_directions()` / `random3`, **`2731cb0` · 2026-09-18 17:36 · zami456** — k random *orthonormal* directions via QR, averaged over 5 draws, as a matched-rank control in the transfer matrix | Draft p.~5: *"As a control, three random directions replace the …"* — so **his** version is in the draft; hers is not | **OWNER** for the per-dataset zero-shot null (earlier, 09-17 vs 09-18, and a different object). **The random-subspace control is also in the replicated paper** (notes §5.5). His `random3` is his own implementation — genuinely independent work, and credited to him here |
| 2.4 | **Length-only control** | `src/assistant_axis_ndif/9_persona_probes.py` at **`19d11a6` · 2026-09-16 19:38** — commit message: *"a response-length control, because length transfers on its own"*; promoted to a first-class feature in `27_crossgen.py` at `b9a497e` | **No.** The only `length` hits on pc-ood are `max_llama_token_length`, which is **vendored Apollo code present at MB** | **No** (`grep -ci length` on the draft = 0) | **OWNER** |
| 2.5 | **Bag-of-words falsification gate** | Specified in `cap/docs/SARCASM_DESIGN.md` at **`3d80071` · 2026-09-21 17:48**; implemented in `cap/scripts/score_sarcasm_markers.py` at `080a979` · 2026-09-21 20:34; re-used as a floor in `cap/scripts/build_sarcasm_natural.py:79` at `cf11749` | **No** | **No** | **OWNER** |
| 2.6 | **Format-matched control** (persona *presence* vs persona *content*) | `cap/scripts/make_format_controls.py` + `cap_instructions_neutral/` + `cap_instructions_paraphrase/` at **`f4097cd` · 2026-09-21 13:22**; `cap/scripts/make_control_bank.py` + `cap_instructions_controls/` at `b548299` · 2026-09-21 17:47 | **No** | **No** | **OWNER** |
| 2.7 | **Leave-one-CLUSTER-out** (row-level leak crossing dataset boundaries) | `src/assistant_axis_ndif/30g_clustered_holdout.py` at **`43ee6d3` · 2026-09-22 17:37** (branch tip) | **No.** He has leave-one-**dataset**-out (`loo`, `6b16a46` · 2026-09-20 20:05) — the weaker split this control was built to replace | **No.** `grep -ciE "leave.one"` on the draft = 0 — his own `loo` is not in the draft either | **OWNER** for the cluster version. See §4.1: leave-one-**dataset**-out is **his** and precedes it |
| 2.8 | **Layer pre-registration** | `docs/LAYER_PREREGISTRATION.md` at **`f5d6904` · 2026-09-21 13:00** — commit message: *"before any axes number exists"* | **No** | **No** (`grep -ciE "pre-regist\|preregist"` on the draft = 0) | **OWNER** |
| 2.9 | **Factor probes (knows / denies)** | Specified in `kd/docs/DESIGN.md` lines 88–95 at **`5b43a7a` · 2026-09-19 02:58** (with the pass/fail gate written before the run); built by `kd/scripts/build_factor_probes.py` at `e9969af` · 2026-09-21 10:55; rows committed at `kd/run/factor_probes/kd_{knowledge,utterance}/rows.jsonl` | **No** | **No** (`grep -ci "factor probe"` on the draft = 0) | **OWNER** |
| 2.10 | **Teacher-forced turn 1** | `cap/scripts/gen_multiturn_ndif.py` at **`6432257` · 2026-09-21 03:26** — the docstring gives the reason (F-46), the three things prefilling fixes, and the wrong-turn-1 `--control` | **No.** `git grep -i teacher` over the whole pc-ood tree hits only `data/raw/geometry_of_truth/*.csv`, which are vendored data files present at MB | **No** (`grep -ci teacher` on the draft = 0) | **OWNER** |
| 2.11 | **Cluster-permutation test** | `src/assistant_axis_ndif/30c_cluster_perm.py` at **`88caf47` · 2026-09-21 16:00** — cites Maris & Oostenveld (2007) | **No** | **No** | **OWNER** (method adopted from Maris & Oostenveld; application is hers) |
| 2.12 | **PC1 permutation test / label-permutation machinery** | Permutation machinery first in `src/assistant_axis_ndif/9_persona_probes.py` at **`5070965` · 2026-09-15 17:08**; label-permutation test in `18_q5_semantics.py` at `ea046d6` · 2026-09-16 19:20 | **No** | **No** | **OWNER** |
| 2.13 | **Seed stability** | `cap/scripts/run_cap_seeds.ps1` at **`4d0d046` · 2026-09-22 02:46**; seed work also in `88caf47` · 2026-09-21 16:00 ("the three CPU items owed: seeds, within-class Mahalanobis, cluster permutation") | **No.** His scripts use one fixed `SEED` | **No** | **OWNER** |

_(2.11–2.13 are one row in `METHODS_INVENTORY.md`; split here because they have different commits. The
count in §0 treats them as **11 control rows**: 2.1–2.10 plus 2.11–2.13 collapsed to one.)_

---

## 3. §2.3 — Baselines the replicated paper did not run

| # | Baseline | First commit on `asix` | On his branch? | Named in draft / notes? | **Attribution** |
|---|---|---|---|---|---|
| 3.1 | **Pooled training, leave-one-dataset-out** | `src/assistant_axis_ndif/30_premise.py` at **`ae3d840` · 2026-09-21 14:27**; bootstrap at `1d94838` · 2026-09-21 14:31 | **YES, AND EARLIER.** `DD/deception_detection/scripts/persona_ood.py loo()` + `results/loo_L20/`, `loo_L40/` at **`6b16a46` · 2026-09-20 20:05 · zami456** — *"Leave-one-dataset-out: the fair baseline for the generalization claim"*. His docstring states the same argument hers does: *"anyone holding labels for five datasets would train on all five"* | **Draft: denied.** Draft p.12: *"A probe trained on several datasets pooled is a stronger supervised baseline, **and we do not compare against it**."* This contradicts his own committed `loo_L20`/`loo_L40` (already logged as RF-3 in Phase 0) | **JOINT — his first (2026-09-20), hers independently and one day later (2026-09-21), with a different justification.** Her `30_premise.py` docstring credits **Ying et al. (arXiv:2602.20273)** for the premise and does not mention his run; her ledger F-85 (2026-09-21) states *"The comparison has never been made against a pooled probe"*, which was **already false** by his `6b16a46`. Both authors should see this row |
| 3.2 | **Pooled training at EQUAL DATA VOLUME** (`pooled_match`: same total row budget, drawn across datasets, so diversity is varied and volume is not) | `src/assistant_axis_ndif/30_premise.py` at **`ae3d840` · 2026-09-21 14:27**. Docstring: *"pooled_match is the point … Ying et al. do not report this control; without it their result and ours are not comparable"* | **No.** His `loo` trains on all rows of the remaining datasets; no row-budget-matched arm anywhere on pc-ood | **No** | **OWNER.** This, not the pooled baseline itself, is the part that is new — and it is what turns F-79/F-80/F-81 into "the gain is diversity, not volume" |
| 3.3 | **Single-source direction averaging** (why pooling works: single-source directions are near-orthogonal, so averaging cancels noise) | `src/assistant_axis_ndif/30d_why_pooling.py` at **`5bd8961` · 2026-09-21 17:25** | **A reference table exists**: `results/loo_L20/single_source_reference.csv` (`single_source_mean`, `single_source_best`) at `6b16a46` · 2026-09-20. That is a *reporting* baseline, not the orthogonality analysis | **No** (`grep -ciE "single.source"` on the draft = 0) | **OWNER** for the orthogonality explanation. His `single_source_reference.csv` is a distinct, earlier artifact and is **his** |
| 3.4 | **Raw 8192-d probes on the same rows** | `src/assistant_axis_ndif/9_persona_probes.py` at **`5070965` · 2026-09-15 17:08** | **Yes** — `persona_ood.py` `baseline` arm, `2731cb0` · 2026-09-18 17:36 | **Yes**, extensively (the draft's 0.732 comparator) | **JOINT and adopted** — the raw-probe comparator is the replicated paper's own protocol (`docs/papers/persona-coordinates-notes.md` §9). Both implemented it. Not a contribution on either side |

_(§0 counts §2.3 as 3 rows — 3.1 JOINT, 3.2 OWNER, 3.3 OWNER — treating 3.4 as adopted-from-paper
rather than a contribution to attribute. 3.4 is shown for completeness.)_

---

## 4. §2.4 — Infrastructure that is itself a method

| # | Item | First commit on `asix` | On his branch? | **Attribution** |
|---|---|---|---|---|
| 4.1 | **The NDIF pipeline, reimplemented rather than imported** (~45 numbered scripts under `src/assistant_axis_ndif/`) | `8078702` · **2026-09-10 16:19** · Suraiya Shanaz — `1_generate_{hf,ndif}.py`, `2_activations_ndif.py`, `3_judge_{hf,ndif}.py` moved in from untracked. `6_axes.py` `8f68cf5` · 09-10 16:39. `8_coords_ndif.py` `c0349ff` · 09-15 12:37. `9_persona_probes.py` `5070965` · 09-15 17:08. Scripts 10–32 and 30a–30g follow, 09-15 → 09-22, all authored by Suraiya Shanaz. 134 tracked files in that directory | **No.** His branch adds nothing under `src/assistant_axis_ndif/` | **OWNER**, with the caveat of §0 rule 4: the commit message says the code existed untracked before 09-10, so this is an upper bound on its date |
| 4.2 | **Attention-memory budget — rows × heads × T², a per-process cap** | `98d9c76` · **2026-09-15 16:31** · Suraiya Shanaz — *"coords: batch by attention memory, which is what NDIF actually runs out of"*; sizing helper at `8_coords_ndif.py:190-210`; per-token variant bounded by returned bytes at `b1ae3ec` · 2026-09-16 12:50 | **No** | **OWNER** |
| 4.3 | **Resumable per-example activation cache + token-budget batcher** | `src/assistant_axis_ndif/acts_cache.py`, `batching.py`, `tests/test_acts_cache.py`, `tests/test_batching.py` at **`ed8daca` · 2026-09-15 02:12 · Suraiya Shanaz — commit message: "Port the resumable activation cache and token-budget batcher"** | **YES, AND IT ORIGINATES THERE.** `DD/deception_detection/acts_cache.py` + `batching.py` + `tests/` at **`e391b3e` · 2026-09-04 13:57 · zami456** — *"Resumable per-example activation cache and token-budget batching"*, 11 days earlier. Contents differ (223 vs 216 lines; blobs `91cd3c0…` vs `2f7fede…`), so the owner's is a **modified port**, not a copy | **COLLABORATOR — original design and implementation. OWNER — the port and the modifications.** This is the clearest case in the whole record of the collaborator's work being downstream-of-nothing and upstream-of-something |
| 4.4 | **The bf16 reduction rule** ("never sum a mask or index in the model's bfloat16") | `ae31943` · **2026-09-15 17:41** · Suraiya Shanaz — *"coords: do every masked reduction in fp32, not the model's bfloat16"*; same bug fixed in the eval pass at `0547fc1` · 09-15 21:47; the silent wrong-last-token variant documented at `8_coords_ndif.py:333-344` | The **underlying bf16-ULP observation** is in his test file: `DD/tests/test_acts_cache.py` (`e391b3e`, 09-04) records *"Measured 2026-09-04 on Llama-70B-3.3 via NDIF: deviation 2.0 at scale 490, which is one bfloat16 ULP"* | **OWNER** for the rule and the silent-corruption failure mode. **COLLABORATOR** for the earlier bf16-precision measurement the rule is consistent with. These are two different findings and both should be stated |
| 4.5 | **Scheduled-task orchestration** (Windows Scheduled Tasks, because OpenSSH kills the process tree) | `run_coords_chain.ps1` at **`12f517f` · 2026-09-15 12:57**; `register_coords_task.ps1` at `b7a3ed5` · 2026-09-15 13:05; hang-detection and wedged-instance recovery at `ffbced5` · 09-15 13:27 and `ca1cdbe` · 09-15 15:28. **36 `register_*.ps1` and 30 `run_*.ps1`** are tracked on `asix` in total | **No** | **OWNER** |
| 4.6 | **The three-store design** (80 layers on a subsample / 21 layers full corpus / per-token subsample = passA / passB / passC) | Pass naming first at **`12f517f` · 2026-09-15 12:57** ("stratified subsampling, plus a chain and supervisor"); per-token Pass C at `b1ae3ec` · 2026-09-16 12:50 | **No** | **OWNER** |

---

## 5. The collaborator's own designs — his work, stated plainly

`METHODS_INVENTORY.md` Part 2 does not list any of these. They belong in the thesis's methods
chapter under his name. All were verified from git objects on his branch during this audit.

| # | Design | Commit | What it is | Status |
|---|---|---|---|---|
| 5.1 | **Leave-one-dataset-out pooled baseline, with committed per-example scores** | `6b16a46` · **2026-09-20 20:05** · zami456 | Each held-out dataset scored by an 8192-d probe fitted on the pooled remainder; naive and 1/n weightings; row-level bootstrap CIs | **His, first, and reproducible.** `loo_L20/loo_scores.npz` and `loo_L40/loo_scores.npz` are **the only per-example raw data committed anywhere on either branch** (66 arrays each: 5 arms × 2 weightings × 6 datasets, plus labels). Audit batch **P10** recomputed all 10 arm×weighting cells at both depths with an independent rank-based Mann-Whitney: **every row of `loo_summary.csv` reproduces to 4 dp** |
| 5.2 | **The rollout-count control (`persona_ood_mod_L20/_L40/_L44`)** | `94db651` · **2026-09-21 13:40** · zami456 — *"More rollouts per persona did not help"* | Re-runs the whole 36-persona grid against a bank with near-uniform rollout counts, holding roles, pipeline and datasets fixed. Conclusion: every leading arm gets *worse* at every depth | **The control is his. The two banks are the owner's, and he may not have known.** Verified in this audit on the lab PC: his "old" counts (`pc_plain_dealer` 76, `pc_enabler` 51, `pc_coverup_operator` 47, `pc_ombudsman` 148, `default` 150, `pc_flatterer` 148) match `C:/Users/user4/expLB/run-kd/vectors_s3/*.pt` **exactly**, and his "new" counts (149, 146, 120, 149, 150, 148) match `run-kd/vectors_s23/*.pt` **exactly** — 36 personas read, 6 of 6 quoted values matching. So "more rollouts" is the owner's **score-2+3 judge filter** replacing her score-3 filter, not a new generation run |
| 5.3 | **Follow-up-probe runs** (`XXXX-9-llama-all-followup`, `persona_ood_followup_L20`) | `fd17728` · **2026-09-21 22:36** and `d2f4cce` · **2026-09-21 23:03** · zami456 | Liars' Bench Follow-Up Probe measured against the published Table 7, and the persona direction read at the follow-up "No" token instead of the response mean | **His.** P10 verified his three published comparators (Tables 6/7/8) are correctly transcribed from the Liars' Bench PDF |
| 5.4 | **Per-token persona-position profile** (`persona_positions_L20/_L40/_L44`) | `a8eb902` · **2026-09-22 01:21** · zami456 | The persona direction read at every token: mean vs max vs top-10% vs first vs last vs pre-reply | **His implementation, on his branch.** The owner reached the same conclusion earlier and separately: `src/assistant_axis_ndif/14_pertoken.py`, **`78232ea` · 2026-09-16 14:40** ("Q8 answered — top-decile token scoring beats mean pooling by +0.053"). Two independent arrivals, ~5 weeks apart in commit order; **neither is derivative of the other on the evidence available** |
| 5.5 | **`random3` matched-rank random-subspace control in the transfer matrix** | `2731cb0` · **2026-09-18 17:36** · zami456 | k random orthonormal directions via QR, averaged over 5 draws | **His implementation.** The control itself is the replicated paper's (notes §5.5). It is the only control of his that reaches the draft |

---

## 6. Designs in Part 2 with **no committed artifact on either branch**

These cannot be attributed from files, and the thesis should not claim them as built without one.

| Design | Status | What exists | What is missing |
|---|---|---|---|
| **The sarcasm bank v2** | **Designed, never built.** Already marked `[DOC-ONLY]` in `METHODS_INVENTORY.md` §3.2 — confirmed here | `cap/docs/SARCASM_DESIGN.md` (`3d80071` → `5ced550`, 2026-09-21) specifies v2; the v1 pilot and the natural-sarcasm Gate 3 corpus **are** committed and **did** run and fail | Any v2 persona file, question file, rollout or vector. Attribution of the *design* is OWNER; there is no result to attribute |
| **`make_view.py` / `make_view_ac.py`** — the builders of the 11th dataset (`harm-pressure-kr-matched`, 258 rows) | **In no branch and not in the local tree** (Phase 0, RF-8). They exist only as loose files on the lab PC (`C:\Users\user4\expLB\make_view.py` 2,253 B; `make_view_ac.py` 1,915 B) | The dataset's *consumers* are committed (`run_passAC_11.ps1`, `matched_profile.py`, `check_matched_slice.py`); the design is in `kd/docs/DESIGN.md` | The code that produced every `kd11` result. **Attribution is safe (OWNER — they are on her machine, in her run tree) but reproducibility is not** |
| **`persona knowledge denial modified/`** — the uniform-rollout bank behind `persona_ood_mod_L*` | **In no branch** (Phase 0 §2c). This audit identified it as the owner's `run-kd/vectors_s23` (§5.2), so the data exists — on her lab PC, not in git | 36 `.pt` at `C:/Users/user4/expLB/run-kd/vectors_s23/` | The committed bank. An md5 against his uncommitted copy would close it completely; the n-count match (6/6) is already decisive |
| **`persona vectors/vectors_s3/`** (26 personas × 80 layers) behind `persona_ood_L40` and the regenerated `persona_ood` | **In no branch** (Phase 0 §2c; his commit `668b280` says "The 66 MB vectors folder is left uncommitted") | The results that used it | The bank. Provenance is the owner's `run-paper16-shared` on metadata grounds (`source_run` field), but this is not blob-verified |

**Also not committed, though the design is:** the `sycophancy_mt` dataset itself (1,182 conversations).
The generator (`13_sycophancy_multiturn.py`) is committed; the rows live on the lab PC and Drive
(`tamara:expLB/sycophancy_mt`). The design attributes cleanly; the data does not travel with git.

---

## 7. Red flags and things both authors should see

**A. Two of the eleven "controls introduced" are the replicated paper's own controls.**
`METHODS_INVENTORY.md` §2.2 opens *"Most do not exist in the replicated paper."* For the
**matched dataset-PCA control** and the **random-subspace control** that is not right:
`docs/papers/persona-coordinates-notes.md` §5.4–5.5 and §9 record both from the paper's Figure 10
and Appendix Figure 9, and `27_crossgen.py:105` itself calls dataset PCA *"The paper's dataset-PCA
control"*. **What is new here is running them at matched k under a dataset-level bootstrap, and
against banks the paper does not have.** The thesis should make that narrower claim — it survives
review; the broad one does not. The other nine controls in §2.2 were checked against the same notes
file and are absent from it: `bootstrap` 0 hits, `length` 0, `bag of words` 0, `format-matched` 0,
`leave-one` 0, `pre-regist` 0, `permutation` 0, `teacher` 0, `factor probe` 0. (`cluster` hits 9
times and `seed` 3, but those are the paper's sycophancy *clusters* and its fixed seed 42, not a
leave-one-cluster-out control or a seed-stability analysis.)

**B. The collaborator re-derived the kd role groups, and his grouping is not hers.**
His `persona_ood_ext_L20/summary.md` states *"Groups were assigned by name; the files carry no group
metadata."* Batch P11 found the consequence: his "six candour" pole **merges** the owner's
deliberately separate `sincere_ignorance` and `plain_assertion` groups — the split that exists
specifically to remove the wording confound. This is not a credit dispute; it is a methods
discrepancy that changes what the axis measures, and the joint thesis has to state which grouping
each number used.

**C. The draft lists the owner's *built and run* design as future work.**
Draft p.12–13: *"Wording-controlled contrasts. The contrast between feigned and sincere ignorance,
tested on HP-KR's honest denials, **would** separate the lie from its wording."* That is
`harm-pressure-kr-matched` plus `kd_role_groups.json`, specified on 2026-09-19 (`5b43a7a`), run,
and reported in F-03/F-20/F-72/F-76/F-112. **This is the single clearest place where the draft is
missing credit.**

**D. The draft says the pooled comparison was not made; his own branch made it.**
Draft p.12: *"A probe trained on several datasets pooled is a stronger supervised baseline, and we
do not compare against it."* His `loo_L20`/`loo_L40` (`6b16a46`, 2026-09-20) are exactly that
comparison. **This one costs him credit, not her** — he under-claims his own strongest
methodological contribution. (Phase 0 RF-3.)

**E. And the mirror of D, in the owner's ledger.** F-85 (2026-09-21) states *"The comparison has
never been made against a pooled probe."* It had been, by him, the day before. Her F-79 credits
Ying et al. for the premise and does not mention his run. Both of these should be corrected
together, in both directions.

**F. The draft names none of §2.1–§2.4 except the random-direction control.** Counted over
`audit/phase0/draft_text.txt`: `capitulat` 0, `sarcas` 0, `kr-matched`/`matched slice` 0,
`teacher` 0, `dataset PCA`/`datapca` 0, `bag of words` 0, `length` 0, `pre-regist` 0,
`factor probe` 0, `leave-one` 0, `single-source` 0, `format` 0, `sycophancy_mt`/`multi-turn` 0,
`pooled` 11 (all describing mean-pooled activations or denying the pooled baseline — none is the
pooled-training baseline). The one design of either author's that the draft *does* carry is
**his** random-direction control.

---

## 8. What would settle what is still open

Nothing in the table is marked UNCLEAR, but three things rest on weaker evidence than the rest, and
each has a cheap test:

1. **Whether any design was agreed in conversation before either party committed it.** Git cannot
   see this. The owner's Claude transcripts
   (`~/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB/*.jsonl`, 19 sessions) and the
   plan file `thinking/plan.md` (lab PC + Drive, 2026-09-17, byte-identical to
   `~/.claude/plans/harmonic-cuddling-kay.md`) would date the owner's side earlier than git does.
   **Only the collaborator can supply the equivalent for his side** — his own session logs or notes.
   Until he does, every "OWNER, first commit" verdict is an upper bound on her date and says nothing
   about his.
2. **Row 3.1 (pooled training).** The evidence supports independent arrival one day apart. What
   would settle it: whether the owner had seen `6b16a46` before writing `30_premise.py`. Her
   docstring credits Ying et al. and not him, which is consistent with independence but does not
   prove it. A single answer from either author closes this.
3. **Row 5.2 (the modified bank).** The n-count match is 6 of 6 exact and the bank is the owner's
   `vectors_s23`. What would make it airtight: md5 of his uncommitted
   `persona knowledge denial modified/*.pt` against `C:/Users/user4/expLB/run-kd/vectors_s23/*.pt`.
   That needs a file only he holds.

---

## 9. Evidence index

Everything above traces to one of these. All git reads were `log` / `show` / `ls-tree` / `grep`
on existing refs; no checkout, no worktree, no fetch.

| Kind | Where |
|---|---|
| First-add commits on `asix` | `git log asix --diff-filter=A --follow --reverse -- <path>` for 100 paths; content-level additions via `git log asix -S'<string>' -- <dirs>` |
| **False-positive guard** | An *unrestricted* `git log asix -S` returns 2025 commits by **Kieron Kretschmar / Walter Laurito** (upstream Liars' Bench) for `bag-of-words`, `TfidfVectorizer`, `CountVectorizer`, `teacher` and `bootstrap`. These are **not** prior art for rows 2.2, 2.5 or 2.10: each hit was traced to its file and is **text inside a rollout or dataset CSV** — `results/external_evaluation_intermediate/alpaca/*.csv` (a model response using the phrase "bag-of-words"), `results/datasets/harm_refusal/cais/wmdp-cyber/llama4-maverick.csv` ("bootstrap"), `data/raw/common_claim.csv` ("teacher"). No code, no design. `grep -rn "CountVectorizer\|TfidfVectorizer\|bag-of-words" --include=*.py --include=*.ipynb src/` outside `src/assistant_axis_ndif/` returns nothing today |
| First-add commits on pc-ood | `git log origin/feat/persona-pc-ood --diff-filter=A --reverse -- <path>` |
| Presence on pc-ood | `git grep -i -E '<term>' origin/feat/persona-pc-ood -- <paths>`; `git diff --name-only 2e2a785 origin/feat/persona-pc-ood` |
| Blob-level comparison | `git rev-parse <ref>:<path>` (§4.3) |
| Draft | `audit/phase0/draft_text.txt` (844 lines, 19 pp), grepped by design name and by characteristic number |
| His 17 notes | `DD/results/**/summary.md`, `DD/deceptive/README.md` on pc-ood |
| Paper controls | `docs/papers/persona-coordinates-notes.md` §5.4, §5.5, §9 |
| Lab PC (read-only, `ssh -o BatchMode=yes`) | `run-kd/` directory listing; `torch.load(..., weights_only=True)['n']` over `run-kd/vectors_s3/*.pt` and `run-kd/vectors_s23/*.pt` (36 files) — **read-only, nothing written** |
| Carried over, not redone | RF-1 (his 36 `.pt` byte-identical to `run-kd/vectors_s3`) — `audit/phase1/census_summary_rf1.txt`; batch P10 (his `loo_summary.csv` reproduces to 4 dp) and P11 (abstract attribution) — `audit/PROGRESS.md` wave 4; RF-3, RF-5, RF-8, RF-9 — `audit/PROGRESS.md` red flags; branch topology — `audit/phase0/branch_compare.md` |
