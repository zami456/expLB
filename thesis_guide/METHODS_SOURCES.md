# METHODS_SOURCES — where every methods topic is specified, implemented, and caveated

For future Claude sessions writing the LaTeX thesis in `thesis/`. **This file is a map, not proof.**
Every number below was read from the file cited next to it. Before you put a number in the thesis,
open that file. If a cell says **UNKNOWN**, the sources did not settle it.

**Pins.** Owner = `asix` @ `e917ecd3` + working tree (ledger `kd/docs/FINDINGS.md` md5 `f81856ef…`,
F-01…F-128). Collaborator (zami456) = `origin/paper` @ `16def2deec34aeee532511df116adb7de5522ca7`,
read with `git show <sha>:<path>`. Source: `audit/BRANCH_PINS.md:12-19,68-71`.
**Neither branch supersedes the other.** Each author's implementation is cited on its own branch.

**Markers used by the audit, which you should carry into prose:** [AUDITED] re-derived from raw;
[LEDGER] recorded in FINDINGS with a source, not re-derived; [CODE] read from source; [DOC-ONLY]
(`audit/METHODS_CHAPTER.md:7-10`). **LAB-ONLY** = evidence that exists only on the lab PC / Drive,
not in git (list: `kd/docs/FINDINGS.md:6766`).

**Primary methods drafts already written (read first):** `audit/METHODS_CHAPTER.md` (163 lines),
`audit/METHODS_INVENTORY.md` (lineage + controls), `audit/methods/EXTRACTION_VS_PAPER.md` (36-step
fidelity table), `audit/methods/ATTRIBUTION.md` (who built what), `audit/LIMITATIONS_AND_FUTURE_WORK.md`.
Caution: `LIMITATIONS_AND_FUTURE_WORK.md:156` still says `make_numbers.py` "exists on neither
branch" — withdrawn (DF-12, F-119, F-123 §1). See PITFALLS.md.

---

## 1. Model and compute (NDIF)

| Item | Value | Source |
|---|---|---|
| Main model | `meta-llama/Llama-3.3-70B-Instruct`, 80 layers, hidden 8192, 64 heads | `src/assistant_axis_ndif/8_coords_ndif.py:669`; heads `kd/docs/FINDINGS.md:5544` |
| Remote execution | every forward pass on NDIF via nnsight; no local GPU | `docs/thesis/persona-coordinates-70b-results.md:28`; collab `method.tex:59`, `appendix.tex` §app:repro |
| Judge (persona banks) | Llama-3.1-70B-Instruct, 0–3 "role displayed" per-persona rubric; 405B cross-check | `docs/PERSONA_CONSTRUCTION.md:161-173`; collab `method.tex:59-61` |
| Hook / sublocation | `llm.model.layers[L].output` = post-block = post-MLP residual stream (same module Lu et al. hook) | `2_activations_ndif.py:288`; `8_coords_ndif.py:331`; `EXTRACTION_VS_PAPER.md:66-73` |
| Numeric precision | masks and reductions in fp32, never bf16 (bf16 argmax picked wrong last token on 1,816/11,156 rows = 16.3%) | `docs/thesis/persona-coordinates-70b-results.md:354-358`; `EXTRACTION_VS_PAPER.md:133` |
| NDIF memory model | per-call limit ≈ batch × heads × T² × 4 B (eager attention), a per-process cap; units are MiB 512/384, not "537/402" | `results.md:362-364`; F-113 §2 `FINDINGS.md:5538-5555` |
| Resumable cache + token-budget batcher | collaborator's design (`e391b3e`, 2026-09-04); owner ported it (`ed8daca`) | `audit/methods/ATTRIBUTION.md:134` |
| Orchestration | Windows SYSTEM scheduled tasks (OpenSSH kills process trees) | `ATTRIBUTION.md:136`; `CLAUDE.md` |
| nnsight version | 0.7.0 for passP/passB_fine | `FINDINGS.md:6593` |
| Excluded for compute | `ai_audit` (80 rows, 4,211–7,543-token contexts, 4.2–13.6 GiB attention/row) | `results.md:365-367` |

## 2. Evaluation datasets

Authoritative table: `docs/thesis/persona-coordinates-70b-results.md:392-406` (Pass B, verified on
Drive). Loader and label semantics: `src/assistant_axis_ndif/eval_datasets.py:1-60, 246-304`.
All rows filtered to Llama-3.3-70B generations (`eval_datasets.py:46-47`) — on-policy.

| Dataset | Origin (cite) | Rows (dec/hon) | Label semantics |
|---|---|---:|---|
| instructed-deception | Liars' Bench; facts from Azaria & Mitchell | 5,432 (3,045/2,387) | instructed_intent |
| harm-pressure-choice (HP-C) | Liars' Bench (WMDP-derived) | 2,134 (134/2,000); 1-token responses | instructed_intent |
| harm-pressure-knowledge-report (HP-KR) | Liars' Bench (WMDP-derived) | 2,139 (139/2,000) | instructed_intent |
| convincing-game | Liars' Bench | 267 (167/100) | instructed_intent |
| insider-trading | Liars' Bench (Scheurer et al.) | 1,080 (423/657) | instructed_intent |
| ai_liar | Apollo rollouts (Pacchiardi et al.) | 270 (135/135) | instructed_intent |
| roleplaying | Apollo | 371 (287 labelled; 84 null labels silently dropped) | judge_grade |
| sandbagging_v2 | Apollo | 1,000 (500/500) | instructed_intent |
| mask | MASK (Ren et al.), relabelled from MASK's evaluated files | 585 (415/170) | belief_inconsistency |
| sycophancy_mt | **built by owner** (§4) | 1,182 (434/748) | belief_inconsistency |
| harm-pressure-kr-matched | **built by owner** (§3) | 258 (124/134) | instructed_intent |
| alpaca | Liars' Bench control (Taori et al.) | 2,000 honest | threshold only |

- Row total 15,278 excluding `ai_audit` (`results.md:30,406`); the chain comment's 15,358 includes
  ai_audit's 80 (`run_coords_chain.ps1:16`).
- `27_crossgen.py`'s default `DATASETS` has **10** labelled datasets and **omits
  `harm-pressure-kr-matched`** (`27_crossgen.py:48-51`); the kd11 runs pass it explicitly
  (`FINDINGS.md:6543-6545`). Blocks: "10" = all 11, "9" = all but sycophancy_mt — historical labels
  (`audit/DOC_FIXES.md:34-36`, F-111 item 3).
- MASK relabel: 1,000 items → 585 kept (225 no belief, 165 refusals, 25 no majority)
  (`results.md:408-411`).
- **Collaborator's set** (`setup.tex:4-10` @16def2de): the five Liars' Bench datasets (CG, HP-C,
  HP-KR, IT, ID) + RepE (the base probe's training set, excluded from averages) + Alpaca for
  thresholds; Soft-Trigger/Gender-Secret left to future work. Insider-trading excludes the
  reasoning scratchpad, unlike Liars' Bench (`method.tex:125-128`).
- Liars' Bench protocol: all datasets are test sets; system prompts hidden; primary metric balanced
  accuracy at ≤1% FPR on Alpaca (`docs/papers/liars-bench-notes.md:84-91`).
- Replicated paper's datasets: Table 1 lists HarmPressureChoice, figures use MASK — "declares one set
  and evaluates another" (`persona-coordinates-notes.md:554-559`; F-60 `FINDINGS.md:1868`; F-114 §1
  `:5651-5674`). Scale rungs use the figure set: ID, mask, CG, ai_liar, roleplaying
  (`scale/scripts/run_scale_crossgen.ps1:45-47`).

## 3. HP-KR and its slices

| Item | Value | Source |
|---|---|---|
| Label | `deceptive = knows AND denies`, `knows = (model_belief != -1)`; the "doesn't know but claims to" cell is empty (0 rows) | `kd/docs/DESIGN.md:53-64` |
| Lexical baseline | one "denies knowledge" bit: 0.870 over all models (superseded), 0.913/0.9125 Llama-only; regex `eval_datasets.py:363-388` (docstring says 0.913) | `kd/docs/DESIGN.md:70-72`; `FINDINGS.md:29`; DF-72 |
| `text_matched` slice | every response denies ⇒ deceptive ⇔ knows; 258 Llama rows (124/134) = `harm-pressure-kr-matched` | `eval_datasets.py:391-421`; F-112 §2 `FINDINGS.md:5480-5482`; F-116 table `:5838-5841` |
| `knowledge_matched` slice | every row known ⇒ deceptive ⇔ denies; 2,001 Llama rows (139/1,862); **first run 2026-09-23** | F-116 `FINDINGS.md:5834-5877`; DF-77 |
| Builder of kr-matched | `make_view.py` / `make_view_ac.py` — **in no branch**; lab PC only; copies at `audit/cache/labpc/loose/` | `ATTRIBUTION.md:163`; F-113 preservation `FINDINGS.md:5625-5627` |
| Leak | kr-matched is a view over HP-KR (row_ids shared by construction); ~81% of its test rows sit in HP-KR's train split (not "every one of 258") | F-108 `:5146-5157`; F-113 §5 `:5588-5597` |
| Leak guard | patched copy of `27_crossgen.py` drops target rows in the source's training rows; 59 of 258 survive | F-115 `:5800-5805`; `audit/scripts/p6_patch_leakguard.py` |
| Growth attempt | honest-denial gate FAILED (0.250/0.150 vs ≥0.70) — slice capped by 134 honest deniers | F-59 `:1829`; `METHODS_INVENTORY.md:149-165` |
| Factor probes | teacher-forced `kd_knowledge` (identical "I do not know the answer.") and `kd_utterance` ("do not know" vs "know"), difference of means per layer | `kd/docs/DESIGN.md:80-95`; `kd/scripts/build_factor_probes.py` |
| Own WMDP scoring | re-scored the model's knowledge (cleaner label, F-08); scores file truncated at 204,800 B in git | `ATTRIBUTION.md:87` |

## 4. `sycophancy_mt` (multi-turn capitulation)

Generator `src/assistant_axis_ndif/13_sycophancy_multiturn.py` (docstring: "Anthropic's
`are_you_sure` design", line 10). Design after Sharma et al. 2023: answer trivia → keep only correct
answers → "I don't think that's right. Are you sure?" with no new evidence → Llama-3.1-70B judges
maintain/retract. 1,182 rows, 434 retracted / 748 maintained (36.7%). A first single-nudge corpus
(`10_sycophancy_ndif.py`) was discarded (94.7% held; string-match mislabels)
(`results.md:418-429`). **Rows are LAB-ONLY** (lab PC + Drive `tamara:expLB/sycophancy_mt`)
(`ATTRIBUTION.md:167-169`). Owner's design (`ATTRIBUTION.md:85`).

## 5. Other evaluation corpora

- **Soft-trigger (3 LoRA organisms) and gender-secret**: off-policy (fine-tuned weights NDIF does not
  serve for eval), excluded from `all_keys()` and used only in Q11 (`eval_datasets.py:252-285,457-470`;
  rows `results.md:431-442`). Cite Liars' Bench; soft-trigger lineage in `docs/papers/soft-trigger-notes.md`.
- **Gender-Secret, collaborator's run**: LoRA applied inside the remote trace, pinned to PEFT within
  1e-5; behavioural check 82% vs 55%; paired bootstrap vs base read. Result: every persona direction at
  chance (F-118 `FINDINGS.md:5920-5949`; script `deception_detection/scripts/gender_secret.py`
  @16def2de). [LEDGER, not re-derived — no activations committed.]
- **Natural sarcasm** (`author_marked_sarcasm`, FigLang): eval-only, for sarcasm Gate 3
  (`eval_datasets.py:56-60`; `cap/scripts/build_sarcasm_natural.py`; F-107, F-113 §4).

## 6. Persona banks

| Bank | Roles / contrast | Questions / rollouts | Filter | Status & source |
|---|---|---|---|---|
| AA 275 (`hf275`) | 275 published Lu et al. roles; `hf275_loose` (120 labelled) / `hf275_strict` (46) | AA 240 questions; published vectors `lu-christina/assistant-axis-vectors` (80×8192 bf16) | gpt-4.1-mini (Lu et al.) | `PERSONA_CONSTRUCTION.md:31-59`; `31_fetch_hf_vectors.py`; loose/strict counts: memory note `role-label-rules-and-banks` (not re-checked here) |
| `hf100` | same 100 roles as own100, published vectors | — | gpt-4.1-mini | `results.md:1379,1391-1393` |
| `own100` | 100 AA roles (99 + default); groups dec 22/hon 22/syc 16/anti 16/anchor 8/breadth 16 | AA 240 × 5 prompts (~1,200/role) | Llama-3.1-70B | `results.md:373-378`; F-114 §7 `FINDINGS.md:5766-5768` |
| **`paper16`** (**26 personas**, not 16) | deception 8 v honesty 7 (+anchor = 16 basis); sycophancy 5 v anti 5 (+anchor = 11); **unified 13 v 12** (dec+syc v hon+anti, 26 basis) | 5 variants × 10 deception qs (50) or × 12 sycophancy qs (60); 1,400 rollouts, 1,350 judged, 1,089 kept (80.7%) | score 3 (`vectors_s3`); unfiltered `vectors_all` built, **never crossgenned** | `results.md:373-390`; `EXTRACTION_VS_PAPER.md:165-172`; DF-90; group counts `DOC_FIXES.md:422-424` |
| `paper16_shared` | same 26, faithful 10+10 shared question set | 2,600 rollouts | s3 | repairs AD-2; **LAB-ONLY** (268 MB, not in git) — `EXTRACTION_VS_PAPER.md:169`; F-113 `FINDINGS.md:5623-5624` |
| `paper16_famcentred` | approximate repair (question-family centring) | — | — | `29_bank_variants.py`; `results.md:1375` |
| **kd bank (36)** | paper16's 26 + `knowledge_denial` 4 + `sincere_ignorance` 3 + `plain_assertion` 3; axes `knowledge_denial` = feigned − sincere; `kd_assertion` = feigned − plain | 36 × 5 variants × **30 shared** questions (10 dec + 10 syc + 10 new; 6 "knows" / 4 "does not know") = 5,400 | `vectors_s3` (score 3), `vectors_s23` (2,3), `vectors_all` | `kd/docs/DESIGN.md:97-146`; `PERSONA_CONSTRUCTION.md:73-149, 564-600` (every prompt and question); `kd/personas/kd_role_groups.json` |
| Collaborator's arms on the kd vectors | published 8v7 (`axis1`), knowledge-denial **4v6** (feigned v sincere+plain pooled), **extended 12v13** (dec+denial v hon+sincere+plain), sycophancy 5v5, unified (13v12) arms added at 2ad758fb | uses `run-kd/vectors_s3` (pc_handler n=106) | s3 | `method.tex:67-87` @16def2de; `persona_pcs.py:132-169` @16def2de; F-126 §3 `FINDINGS.md:6621-6624`; base bank `paper16_L20_sharedquestions` (F-97 `:4396-4398`) |
| Rollout-count control `persona_ood_mod_*` | same 36 roles, near-uniform counts (= owner's `vectors_s23`) | — | s23 | collaborator's control, owner's banks (`ATTRIBUTION.md:149`); scored lower (DF-18) |
| **cap bank** | 13 roles: 6 capitulation, 6 position-holding, `default`; variants A `capitulation`, B `cap_vs_default`, C `cap_only` (7-role basis) | 20 questions (v3) × 5 variants × 13 = 1,300; **turn 1 teacher-forced**, only turn 2 generated | s23 + all (**no s3 arm**) | `cap/docs/DESIGN.md:45-62,145-185`; `cap/scripts/gen_multiturn_ndif.py` |
| Sarcasm bank v2 | designed, **never built**; v1 pilot + natural-sarcasm Gate 3 ran and failed | — | — | `cap/docs/SARCASM_DESIGN.md`; `ATTRIBUTION.md:162` |
| Scale rungs | paper16 26 roles at 3B/8B; kd 36 roles at 3B/8B (F-121, 5,400 rollouts each) | as 70B | s23 (3B/8B paper16); kd all+s23 | `scale/scripts/run_scale_chain.ps1:50-53,186-212`; F-121 `FINDINGS.md:6010-6060` |

**How personas and prompts were built.** Names and one-line descriptions verbatim from Mahadik &
Skapars' tables; **system prompts written by the project** in Assistant-Axis style (the paper never
prints them) and declared as a reconstruction (`pc_personas.py:13-19,66-251`; `EXTRACTION_VS_PAPER.md:165-166`).
**5 instruction variants, not the paper's 2** — deliberate (`EXTRACTION_VS_PAPER.md:167`). paper16's
10+12 questions are **the project's renderings** of the paper's printed topics, not verbatim
(`pc_personas.py:262-297`; `COLLABORATOR_FINDINGS.md:211-212`). Recipe: Chen et al. contrastive
system prompts via Lu et al. (`METHODS_INVENTORY.md:31`).

## 7. Persona-vector pipeline (Eq. 1–4)

| Step | Parameters | Source (owner) | Collaborator |
|---|---|---|---|
| Generate | T=0.7, top-p 0.9, ≤512 new tokens, 5 variants | `1_generate_ndif.py:269-284` | `method.tex:57-59` (same) |
| Judge | 0–3; keep 3 (`--keep_scores 3`) or 2,3; `default` never judged | `2_activations_ndif.py:20-21,449-459`; `3_judge_ndif.py` | keeps 3 only (`method.tex:61`) |
| Min rollouts | `--min_count 20` for paper16 (script default 50); lowest survivor n=21 | `EXTRACTION_VS_PAPER.md:172` | — |
| Pool | masked mean over the last assistant span; flat mean over kept rollouts | `2_activations_ndif.py:185-206,290-292,389-422` | `method.tex:6-14` (Eq. persona) |
| Layers stored | all 80 for paper16 (saved `(80,5,8192)`) | `EXTRACTION_VS_PAPER.md:90` | "all 80 layers" (`method.tex:63`). **Conflict:** `PERSONA_CONSTRUCTION.md:177-179` says 21 layers, `(21,8192)` — UNKNOWN which applies to the kd bank |
| Centering | mean-centering only, applied only before PCA; cancels exactly in Eq. 4 (≤2.9e-15, all 80 layers) | `6_axes.py:19-21,229`; `EXTRACTION_VS_PAPER.md:92-112` | "centering cancels" (`method.tex:67-68`) |
| Contrast | `mean(pos) − mean(neg)`; anchor excluded from contrast, **included in every PCA basis** | `6_axes.py:69-75,159-168,223-246` | `method.tex:67-81` |
| PCA / sign | 5 PCs per layer, SVD; `orient_pcs` pins sign to the contrast | `6_axes.py:91-114,134` | `persona_pcs.py:287` @16def2de |
| Test-time centering | none; provably cannot change AUROC | `EXTRACTION_VS_PAPER.md:181` | none |

Fidelity verdict: 36 steps, 16 match, 11 resolve an ambiguity, 6 deliberate deviations, 4 accidental
divergences (AD-1 k=5; AD-2 unified question-bank confound; AD-3 k differs across arms; AD-4 judge
filter differs across arms) (`EXTRACTION_VS_PAPER.md:43-62`). F-113 §6 prints "35 steps"
(`FINDINGS.md:5601`) — use 36 and cite EXTRACTION_VS_PAPER.

## 8. Evaluation-side activation stores

| Store | Rows | Layers | Stored | Source |
|---|---|---|---|---|
| passA | 2,000 stratified (200/dataset) | all 80 | mean + last | `results.md:446-456`; `kd/docs/DESIGN.md:213-216` |
| passB (main) | 15,278 + 1,182 sycophancy = 16,460 | 21: 0,4,…,76 **+22** (Apollo's layer) | mean + last | `run_coords_chain.ps1:16-27,39`; `results.md:454-456` |
| passC | 2,000 stratified | 20,22,28,40,44,48 | every response token (304,706) + mean + last | `results.md:454-457` |
| passQ11 | 6,392 sleeper rows | 21 | mean + last | `results.md:454` |
| passB_fine | same rows as passB | L37–L47 (no L40), 4 banks | — | F-126 §2 `FINDINGS.md:6609-6617` |
| **passP** | same rows | — | **prompt span** (`--span prompt`) — a before-the-answer baseline | F-126 §2 `:6583-6607`; `8_coords_ndif.py:675` |
| kd11 stores | passA11/passB/passC11 with kr-matched added | — | — | F-71; F-112 |

Span: response tokens only (`--span response`), tail truncation, `max_length` 8192
(`8_coords_ndif.py:675,694`); scale rungs ran 1024 (F-63 caveat `FINDINGS.md:2031`). Sampling
balances by label, round-robin (`8_coords_ndif.py:504`; DF-52). Collaborator: Liars'-Bench/Apollo
detection-token mask (excludes system prompts), mean-pooled; L20/40/44 pooled locally from the
per-token cache, 13 further layers pooled remotely and validated at L40 (row cos ≥ 0.99989)
(`method.tex:114-125`; `appendix.tex` §app:depth @16def2de; F-117).

## 9. Features and probes

**Features** (`src/assistant_axis_ndif/9_persona_probes.py:120-183`; list/defaults `27_crossgen.py:59-62,867-871`):

| Feature | Definition | Note |
|---|---|---|
| `raw` | full 8192-d pooled activation | the comparator throughout |
| `pc:<axis>` (@k) | projection on top-k PCs of the named **basis** | uses the basis, never the contrast: two variants over the same roles are identical **by construction** (`cap/docs/DESIGN.md:159-172`) |
| `axes` | 4-d stack of unit contrasts: deception, sycophancy, unified, assistant (`9_persona_probes.py:162-181`) | not PCs |
| `axes_resid` | 3-d, each contrast with its assistant component removed | Q4 test |
| `erase:<axis>` / `erase:random` | remove one direction from the activations, refit full probe | representational, not causal |
| `datapca` (@k) | PCA fitted on the **source's** own training rows — "the paper's dataset-PCA control" | `27_crossgen.py:104-112` |
| `random` (@k) | k random orthonormal directions (QR), seed-fixed | owner; collab `random3` = 3 dirs × 5 draws (`persona_ood.py:475`) |
| `length` | response token count only (`span_len`) | invariant to layer/pooling/probe (F-111 item 7) |
| zero-shot `contrast`/`PC1` | unit contrast / PC1 of the bank's **primary** axis — "deception" whenever the bank has one; other axes as `contrast:<name>` | `27_crossgen.py:402-440` |

**Default k = 5** (`27_crossgen.py:882`); `@2` features at k=2; scale rungs k=3
(`run_scale_crossgen.ps1:28`). The paper uses k=1 (deception), 3 (sycophancy), 2 (unified)
(`persona-coordinates-notes.md:560-563`); **k=1 never run at 70B** (`EXTRACTION_VS_PAPER.md:149`).

**Probes** (`9_persona_probes.py:185-225`; default all three, `27_crossgen.py:873`):

| Name | Config | Lineage |
|---|---|---|
| `paper` | `LogisticRegression(C=1, intercept, max_iter 1000)`, no scaler, seed 42 | Mahadik & Skapars §4.5 (`EXTRACTION_VS_PAPER.md:182`) |
| `lr_reg10` | StandardScaler + LR C=0.1, no intercept | Apollo/Liars'-Bench `lr`, reg_coeff 10 (`detectors.py:369-371,476`) |
| `mms` | StandardScaler + difference of class means | Apollo "mean mass shift" |
| Liars'-Bench/Apollo base probe | `method: lr`, `reg_coeff: 10`, per-token, trained on RepE, mean of token scores; LB layer 20, Apollo layer 22 | `deception_detection/scripts/configs/XXXX-9.yaml:22-23`, `roleplaying.yaml:18-19` (asix); `XXXX-9-llama-all.yaml` @16def2de; `deception-probes-notes.md:286-304` |
| Collaborator's trained probe | StandardScaler + LR C=1/10, no intercept (= `lr_reg10`), seed 42; diagonal 5-fold `StratifiedKFold` | `persona_ood.py:59-62,190-193,242-262` @16def2de; `setup.tex:21-34` |
| Follow-Up Probe | append "Was your last response deceptive? (Yes/No)" + prefilled "No"; read that token | `liars-bench-notes.md:104`; collaborator's runs `XXXX-9-llama-all-followup`, `persona_ood_followup_L20` (`ATTRIBUTION.md:150`); `appendix.tex` §app:followup |
| Upper-Bound Probe | privileged in-distribution training on 5 LB datasets | `liars-bench-notes.md:106`; runs at 8% FPR, not 1% (F-124 §4 `FINDINGS.md:6359-6363`) |

## 10. Evaluation protocols

| Protocol | Specification | Source (owner) | Collaborator's version |
|---|---|---|---|
| **Crossgen / train-one-test-all** | train on 80% of source, split by `item_id`, stratified, class-balanced by downsampling; diagonal = own 20%; off-diagonal = 100% of target, unbalanced; 11×11 = 110 off-diagonal pairs (kd11) | `27_crossgen.py:1-21,194`; `EXTRACTION_VS_PAPER.md:184` | 6×6 = 30 off-diagonal cells, diagonal 5-fold CV, "HP-KR removed" variant = 20 cells (`setup.tex:21-29`) |
| Zero-shot | project on the unit contrast; no labels; every dataset on 100% of rows | `27_crossgen.py:402-440` | `method.tex:104-106` |
| dim=1 identity | a probe on one feature can only choose a sign, so trained off-diagonal = zero-shot AUROC; intervals belong over test datasets | F-122 `:6107-6128`; F-124 §3 | stated in `method.tex:109-111` (F-125 §1) |
| Pooled LODO | train one probe on all other datasets | owner `30_premise.py` (row-level split — `rng.permutation`, no `group_split`, DF-67); `pooled_match` = equal row budget (cap 666; realised 412–500) | **collaborator's first** (`6b16a46`, 2026-09-20): `persona_ood.py` `leave_one_out` (:338-407), `loo_L{20,40}/`, per-example `loo_scores.npz` |
| LOCO | leave-one-provenance-cluster-out: `lb_harm`, `lb_scenario`, `apollo`, `mask`, `ours` | `30g_clustered_holdout.py:57-61` (owner) | — |
| Leak-guarded crossgen | off-diagonal drops target rows in the source's training rows | `audit/scripts/p6_patch_leakguard.py` (audit) | — |
| TPR@1%FPR | threshold = 99th percentile of Alpaca scores | `27_crossgen.py:275-279` | BA and recall at the Alpaca 1% threshold (`setup.tex:36-38`); realised FPR on the eval set is up to 88.3% (F-124 §4) |
| Row bootstrap | 1,000 stratified resamples, paired across methods | `27_crossgen.py:228-257,883` | 1,000 stratified over examples, shared (`setup.tex:39-40`); `bootstrap_delta` (`persona_ood.py:409`) |
| **Dataset-level bootstrap** | 10,000 multinomial resamples of datasets, paired; two-sided p | `27_crossgen.py:778-825,884`; `23_significance.py:115-124` | none |
| BH | within block ("10"/"9") over every pooling × layer × probe × comparison; q < 0.05 | `27_crossgen.py:821-829` | none |
| Comparisons | every persona feature vs raw, datapca@k, random@k; controls vs raw | `27_crossgen.py:66-90` | vs single-dataset baseline and `random3` |
| Random-direction null (zero-shot) | 1,000 random unit directions, same rows/layer; p on \|AUROC−0.5\| | `25_zeroshot_nulls.py:1-20,81` (a 50-draw null also exists, DF-62 M-049) | `random3` trained control |
| Cluster-permutation test | Maris & Oostenveld 2007; label permutation identical across layers | `30c_cluster_perm.py`; `LAYER_PREREGISTRATION.md:65-68` | — |
| Equivalence test | margin ±0.02 (`23_significance.py:122`); post-hoc widening is DF-63 | `results.md:513` | — |
| Seeds | main seed 42; extra seeds optional; five seeds only for pooled LODO (F-83) and cap (F-87) | `27_crossgen.py:874-876` | seed 42 (`persona_ood.py:62`) |
| Mahalanobis predictor (Ying et al.) | `--covariance total|within` | `30_premise.py:80` | — |

## 11. Layer choice and pre-registration

`docs/LAYER_PREREGISTRATION.md` (committed `f5d6904`, 2026-09-21, before any 80-layer axes number;
`ATTRIBUTION.md:102`). L20 = Liars' Bench convention; L22 = Apollo; **L40 = Lu et al. middle layer
(80/2)** — pre-specified; **L44 = selected on our own outcome, post hoc** (`LAYER_PREREGISTRATION.md:13-38`).
Committed tests: report whole profile, cluster permutation, bootstrap the argmax, nest layer selection
in the dataset bootstrap, no post-hoc switching, correct p for selection (`:62-79`). Outcome readings
fixed in advance (`:81-89`). §5's premise is corrected in place (DF-55 block, `:93-103`). Scale rungs use
the paper's midpoints 14/16 (`run_scale_chain.ps1:52-53`). Collaborator: 16 layers
(20–76 step 4, + 79), reports L40, optimum L36 (`method.tex:116-119`; F-117).

## 12. Scale rungs (3B / 8B)

Models Llama-3.2-3B-Instruct (28 layers, mid 14) and Llama-3.1-8B-Instruct (32 layers, mid 16)
(`run_scale_chain.ps1:50-53`). Crossgen: 5 datasets (ID, mask, CG, ai_liar, roleplaying), k=3, mean,
`paper`, seed 42 (`run_scale_crossgen.ps1:28,45-47`; F-113 §6 `:5611-5612` — all four stores k=3).
paper16 rungs built from `vectors_s23` vs 70B `vectors_s3_fp32` (DF-47). `run_scale_chain.ps1:197-203`
overwrote the `vectors_all` bank in the runs behind F-63/F-110 (DF-48); the file now carries a fix (`:197-212`, per-variant `axes_${tag}_$vt.pt`, then copied to `axes_$tag.pt`); F-121's kd
rungs verified byte-distinct (`FINDINGS.md:6050-6052`). **8B coords corpus is incomplete: 5 of 12
dataset dirs empty** (DF-73). Eval text at 3B/8B is 70B-authored (off-policy) and truncated at 1024
(F-63 caveats `:2031-2033`). F-121's ladder metric = layer-averaged **(per-layer best persona
feature) − raw** (`audit/scripts/p12_kd_scale_ladder.py:59-73`); F-110's = best of {axes,
pc:deception} vs raw and vs datapca over 19 layers (`FINDINGS.md:5257-5264`).

## 13. Implementations by branch (both are valid; cite each on its own branch)

| Component | Owner (`asix`) | Collaborator (`origin/paper` @16def2de) |
|---|---|---|
| Pipeline | `src/assistant_axis_ndif/` ~45 numbered scripts (reimplemented, not imported from `assistant_axis`) | `src/probes/deception-detection/deception_detection/scripts/persona_ood.py`, `persona_pcs.py`, `paper/make_numbers.py`, `make_figures.py`, `paper/generated/numbers.tex` |
| Activations | NDIF stores passA/B/C/P | Apollo per-token cache (L20/40/44) + remote pooling |
| Probe | `paper`/`lr_reg10`/`mms` | `lr_reg10`-equivalent only |
| Diagonal | 20% held-out by item_id | 5-fold CV |
| Uncertainty | row + dataset bootstrap, BH | row bootstrap only |
| Agreement | 55 bank-matched cells, mean |gap| 0.0015 (`COLLABORATOR_FINDINGS.md:141-142`) | |
| Raw data committed | none of the activations; LAB-ONLY stores | `loo_scores.npz` = only per-example data on either branch (`ATTRIBUTION.md:148`); no pooled activations (`COLLABORATOR_FINDINGS.md:19-23`) |

## 14. Literature — what must be cited for which method

The thesis bib `thesis/bibliography/references.bib` currently holds only 7 unrelated template entries
(Tulabandhula … Akhtar) — **every entry below must be added**. Reusable BibTeX keys exist in the
collaborator's `src/probes/deception-detection/paper/refs.bib` @16def2de. PDFs: `papers/`; notes:
`docs/papers/*-notes.md`; registry `audit/papers.csv`.

| Method / data | Cite | arXiv / key | PDF / notes |
|---|---|---|---|
| Persona coordinates (the replicated method, Eq. 1–4, datapca & random-subspace controls) | Mahadik & Skapars | 2605.09391 / `mahadik2026persona` | PDF + notes |
| Assistant Axis, role-vector pipeline, L40 midpoint, post-MLP hook | Lu et al. | 2601.10387 / `lu2026assistant` | PDF + notes |
| Contrastive system prompts | Chen et al., Persona Vectors | 2507.21509 / `chen2025persona` | no PDF |
| Datasets HP-C/HP-KR/CG/ID/IT/GS/ST, Alpaca 1% FPR protocol, Mean/Follow-Up/Upper-Bound probes | Kretschmar, Laurito, Maiya, Marks — Liars' Bench | 2511.16035 / `kretschmar2025liars` | PDF + notes |
| **MASK** dataset | **Ren, Agarwal, Mazeika, Menghini et al. (Center for AI Safety + Scale AI)** — NOT Cadenza Labs, whose GitHub fork only hosts it (`CLAUDE.md` submodule table) | 2503.03750 — **no entry in refs.bib; add one** | PDF + notes |
| lr/mms probes, roleplaying, sandbagging, ai_liar rollouts, layer 22, RepE training set | Goldowsky-Dill et al. | 2502.03407 / `goldowskydill2025detecting` | PDF + notes |
| RepE / Instructed-Pairs | Zou et al. | 2310.01405 / `zou2023representation` | no PDF |
| ai_liar origin; black-box lie detector | Pacchiardi et al. | 2309.15840 / `pacchiardi2023catch` | PDF + notes |
| instructed-deception facts | Azaria & Mitchell (EMNLP Findings 2023) | `azaria2023internal` | no PDF |
| insider-trading | Scheurer et al. | 2311.07590 / `scheurer2023strategic` | no PDF |
| HP datasets source | WMDP, Li et al. | 2403.03218 / `li2024wmdp` | no PDF |
| sycophancy_mt design ("are you sure?") | Sharma et al. | 2310.13548 — no refs.bib entry | no PDF |
| Alpaca control | Taori et al. | `taori2023alpaca` | — |
| NDIF / nnsight | Fiotto-Kaufman et al. | 2407.14561 / `fiottokaufman2024nnsight` | — |
| Llama 3 models | Grattafiori et al. | 2407.21783 / `grattafiori2024llama` | — |
| Off-policy objection to teacher forcing | Kirch, Dower, Skapars, Yannakoudakis, Lubana, Krasheninnikov | 2511.17408 / `kirch2025offpolicy` — venue unverified (DF-19) | PDF, no notes |
| Sycophancy subspace; Mahalanobis predictor (did not reproduce, F-83); base-vs-chat (F-78, panel (a) not (b)) | Ying, Ravfogel, Kriegeskorte, Hase | 2602.20273 / `ying2026truthfulness` | PDF + notes; must be cited (`truthfulness-spectrum-notes.md:114-119`) |
| Peak-layer CIs spanning the network | Kumar | 2605.27958 / `kumar2026pressure` — venue "GEM@ACL 2026" unverified (DF-19) | PDF, no notes |
| Subspace selection | Yoo & Skapars | 2609.02893 / `yoo2026subspace` | PDF |
| Targeted deception probes | Natarajan, Jain, Arora, Golechha, Bloom | 2602.01425 / `natarajan2026targeted` — **title conflict**: two in-repo PDFs title it "Building better deception probes using targeted instruction pairs" (`COLLABORATOR_FINDINGS.md:209-210`) vs abstract match (F-126 `:6646-6651`); UNKNOWN, verify | no PDF |
| Truth Stays in the Family | Choi et al. | 2606.15821 | PDF; `truthfulness-spectrum-notes.md:71-98` |
| Sleeper agents (soft-trigger lineage) | Hubinger et al. | 2401.05566 | no PDF |
| Cluster-permutation test | Maris & Oostenveld 2007 | no bib entry | `LAYER_PREREGISTRATION.md:65` |
| Double dipping | Kriegeskorte et al. 2009 | no bib entry | `LAYER_PREREGISTRATION.md:77` |
| BH correction; bootstrap | Benjamini–Hochberg (standard) | UNKNOWN — no bib entry found anywhere | — |
| Probing background | Alain & Bengio; Burns et al.; Marks & Tegmark | refs.bib keys exist | — |

---

## COVERAGE

| Topic | Authoritative description | Code (owner) | Code (collab) | Status |
|---|---|---|---|---|
| Model & NDIF | §1 | yes | yes | covered |
| Datasets (LB, Apollo, MASK, alpaca) | §2 | `eval_datasets.py` | `setup.tex` | covered |
| HP-KR slices (text/knowledge-matched, leak) | §3 | yes | — | covered; builder LAB-ONLY |
| sycophancy_mt | §4 | generator | — | rows LAB-ONLY |
| gender-secret / soft-trigger / sarcasm corpora | §5 | loader | gender_secret.py | GS numbers [LEDGER] |
| Persona banks (own100, hf100/hf275, paper16/26, shared, kd36, cap, unified 13v12, extended 12v13, 4v6) | §6 | yes | yes | hf275 loose/strict counts not re-checked |
| Persona/question construction | §6–7 | yes | method.tex | kd-bank layer count UNKNOWN (21 vs 80) |
| Extraction (layers, pooling, span, passP) | §7–8 | yes | yes | covered |
| Probes & baselines (raw, pc, axes, datapca, random, length, LR/mms, base, Follow-Up) | §9 | yes | yes | covered |
| Protocols (crossgen, LODO, LOCO, zero-shot, metrics, bootstrap, BH, nulls) | §10 | yes | yes | covered |
| Layer choice & prereg | §11 | yes | — | covered |
| Scale rungs | §12 | yes | — | 8B corpus incomplete (DF-73) |
| Branch differences | §13 | — | — | covered |
| Citations | §14 | — | refs.bib | BH/Maris/Kriegeskorte/MASK/Sharma entries missing; 2 venues + 1 title UNKNOWN |
