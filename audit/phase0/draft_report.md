# Phase 0 — the collaborator's draft report: which file, and what it says

> **Update 2026-09-22 (owner):** the owner confirmed that `full paper draft for this experiments/main (1).pdf` is the collaborator's draft. The "INFERRED" identification below is now owner-confirmed. The owner also says every persona bank and other input came from her runs. Only the activations are his own: he downloaded them, about 62 GB per layer.

Written 2026-09-22 by a Phase-0 subagent. **Orientation only: nothing here says whether a claim is true.**
Tags: VERIFIED (checked in a file/git during this audit) · DOC-ONLY · CONFLICT · NOT-FOUND · INFERRED.
"VERIFIED" below means *the fact about the file/git* is verified (e.g. who committed it), never that the
draft's numbers are right.

Companion files:
- `audit/phase0/draft_claims_raw.csv` — 237 rows, one per claim / table row / figure (D-001 … D-237).
- `audit/phase0/draft_text.txt` — full extracted text, page-marked (from `audit/scripts/p0_draft_extract.py`).
- `audit/phase0/draft_pages/page{02,06,09,10,11}.png` — rendered figure pages (`audit/scripts/p0_draft_render.py`;
  renderer pypdfium2 installed into `audit/tools/pylib`, no repo venv touched).
- `audit/scripts/p0_draft_claims.py` — builds the CSV and prints the internal-arithmetic checks in §9.

---

## 1. Which file is "the collaborator's draft report"

**Answer: `full paper draft for this experiments/main (1).pdf` (on `asix`).** Confidence: high, but INFERRED —
the PDF carries no author names, and the owner, not the collaborator, committed it.

### Candidates examined

| # | Path · branch | Git author · dates | Size | Title / byline inside | What it is | Is it his draft? |
|---|---|---|---|---|---|---|
| A | `full paper draft for this experiments/main (1).pdf` · on `asix` and `origin/asix`; **not** on `origin/feat/persona-pc-ood` | only commit: `cf11749` Suraiya Shanaz `<suraiya.shanaz@g.bracu.ac.bd>` 2026-09-22 16:51:52 +0600, msg "finfdings" (VERIFIED) | 193,849 B, md5 `cd0bfede186d631a97f280ecae1f1dff`, 19 pages | **"Lying in Knowledge-Denier Persona Coordinates: Cross-Dataset Lie Detection at 70B"**; byline `[Author names] [Affiliation] [email]` (placeholders); no date printed. PDF metadata: Creator "LaTeX with hyperref", Producer "xdvipdfmx (0.1)", CreationDate `2026-09-22 07:57:30 UTC` (= 13:57 +06); no Title/Author fields (VERIFIED) | A full workshop-style paper: abstract, 8 sections, 7 appendices, 5 figures, 11 tables | **YES (INFERRED, high)** |
| B | `docs/COLLABORATOR_BRIEFING.md` · `asix` | 5 commits, all Suraiya Shanaz: `01e27b0` 2026-09-21 16:52 → `1a482b7` → `aea9f53` → `4d0d046` → `43ee6d3` 2026-09-22 17:37 (VERIFIED) | 38,546 B | "Where the project stands — a briefing"; line 3: "Written 2026-09-21 **for a collaborator** who has not seen the last day of work" | Owner-side briefing addressed to the collaborator ("your branch", "your abstract"); critiques "the draft abstract" | **No** — written *for* him, not *by* him (VERIFIED from its own text) |
| C | `docs/thesis/persona-coordinates-70b-results.md` · `asix` | `b9a497e` Suraiya Shanaz 2026-09-18 (VERIFIED) | 93,073 B | "Persona Coordinates at 70B — Results, Pipeline and Open Issues"; "**For:** collaborator review · Status as of 2026-09-17" | Owner-side results document sent for his review | No |
| C2 | `docs/thesis/hf275-role-labels-draft.md`, `docs/thesis/sarcasm-pilot-negative-result.md` · `asix` | Suraiya (`b9a497e` 2026-09-18; `6c9a5cf` 2026-09-22 03:25) | 13,863 B; 16,930 B | "275-role labels — DRAFT for your review"; "The Sarcasm Bank — A Pre-Registered Design That Failed Its Own Gate" | Owner-side thesis sections | No |
| D | His branch `origin/feat/persona-pc-ood`: 20 `.md` files under `src/probes/deception-detection/` | all new ones by zami456 `<abdullah.al.zami@g.bracu.ac.bd>`, 2026-09-18 → 2026-09-22 12:45 (VERIFIED) | 14–21,000 B | e.g. `results/persona_ood/summary.md` "Persona PC features and OOD probe generalization" (20,985 B); `results/persona_ood_ext_L20/summary.md` "Extended persona set: fixing HP-KR at the basis level" (7,038 B); `results/loo_L20/summary.md`; `results/persona_ood_followup_L20/summary.md`; `results/persona_ood_mod_L40/summary.md` "More rollouts per persona did not help"; `results/persona_positions_L40/summary.md`; `results/XXXX-9-llama-all-followup/summary.md` | Per-experiment lab-notebook write-ups sitting next to the result CSVs. The source material the PDF is built from (see §8) | No — working notes, not the report. The other 6 .md files there are infra (README, PR template, ACCESS, data READMEs, `deceptive/README.md` = notes on a persona file handed to him) |
| E | Any `.tex/.pdf/.docx/.bib` elsewhere | — | — | Working tree outside `papers/`: only file A (`find`, VERIFIED). All 13 refs: only `figures/figure_1/table.tex` + Liars'-Bench upstream figure PDFs, all older than the branch split (VERIFIED) | — | No. **The LaTeX source of A is NOT-FOUND anywhere.** |

### Evidence that A is the collaborator's (zami456's) draft — all INFERRED from these facts

1. **The writer credits someone else with the persona bank.** Acknowledgements (p13): *"[Collaborator name] built
   the persona bank and the rollout and judging pipeline."* Owner-side docs say the bank came from the owner's side:
   COLLABORATOR_BRIEFING §5a *"The ten extra roles are the knowledge-denial bank we sent you"*; FINDINGS F-97 *"Those
   ten extra roles are **ours**, matching `kd/personas/kd_role_groups.json`"* (DOC-ONLY). So the writer is not the
   bank builder → the writer is the collaborator.
2. **Every result section maps onto a zami456 commit whose subject says the same thing** (VERIFIED commit subjects;
   mapping INFERRED): `9774075` "Knowledge-denial personas fix HP-KR" (§5.1); `0f7b620` "Zero-shot results in the
   layout of Liars' Bench Figure 2" (Fig 5); `45b16d9` "The base probe in the zero-shot Fig 2: ahead at layer 20 at
   1% FPR, behind on ranking" (§5.4); `29cc17c` "Layer 44: the depth curve turns over"; `d2f4cce` "Persona experiment
   at the follow-up token: the direction's sign stops being universal" (App C); `a8eb902` "the mean ranks best, the
   top 10% detects best" (App D); `c2a0c1f` "k = 3 at both token positions: 1-d still wins on responses, denial PCs
   lead at the follow-up" (App E); `fd17728` "Follow-Up Probe measured: Table 7 reproduces to 0.002 AUROC" (App G).
3. **Timing.** His last commit `45b16d9` is 2026-09-22 12:45 +06; the PDF was compiled 13:57 +06 (72 min later); the
   owner committed it at 16:51 +06 (file mtime in the working tree 16:48).
4. The code footnote points to `https://github.com/zami456/expLB`.
5. The owner's documents already treat "the abstract" as his: briefing §7 "Your zero-shot contrast is untouched…";
   F-85/F-97 quote the abstract's 0.904 / 0.916 vs 0.648 / 0.732, which are this PDF's Table 3 numbers.

Counter-points: no author names or date in the PDF; the file name `main (1).pdf` looks like a duplicate download
(INFERRED); the owner, not he, committed it; no LaTeX source or `paper/` folder exists in the repo.

### The owner's briefing describes an EARLIER version of this draft — CONFLICT (doc vs doc)

| Owner-side statement (DOC-ONLY) | This PDF | Tag |
|---|---|---|
| Briefing §5c / F-88: "The draft title is *'One Direction Generalizes'*" (still in `43ee6d3`, 17:37 +06, *after* the PDF was committed) | Title is "Lying in Knowledge-Denier Persona Coordinates…"; "One direction" survives only as a contribution heading and the §5.3 title | CONFLICT |
| F-85: "The draft abstract reports … **0.904 (L20)** and **0.916 (L40)** against … **0.648** and **0.732**" | Abstract gives only L40 (0.916 vs 0.732); 0.904/0.648 appear in §5.2 and Table 3 | CONFLICT (minor) |
| Briefing §5c / F-97: "the abstract saying PC1 sits **0.09 below** the contrast at L40" (unsourceable) | No "0.09" anywhere in the PDF text (grep count 0). It gives Extended PC1 0.768 vs 0.916 at L40 | CONFLICT — the figure is gone |
| F-97: the abstract's headline is the kd-augmented 12v13 axis, not the paper's 8v7; ~45% of the margin is the kd bank | The PDF now names the kd bank as its main contribution (title, contribution 1), reports 8v7 separately (Table 3: 0.651/0.832/0.826) | INFERRED: the PDF was revised in response to F-97 / the briefing |
| Briefing §5a "Their knowledge-denial split … merges both controls into one pole" (F-97) | PDF §3.2 and §6 state this openly: "The contrasts we evaluate pool both honest groups" | consistent |

So the "draft abstract" the owner critiqued (title "One Direction Generalizes") is a **separate, earlier text that is
NOT-FOUND** in the repo. Only the later PDF exists.

---

## 2. Bibliographic facts (VERIFIED from the PDF)

- **Title:** Lying in Knowledge-Denier Persona Coordinates: Cross-Dataset Lie Detection at 70B
- **Authors / affiliation / email:** placeholders `[Author names] [Affiliation] [email]`
- **Date:** none printed; compiled 2026-09-22 07:57:30 UTC
- **Length:** 19 pages = 13 pages main text + references (pp. 14–15) + appendices A–G (pp. 16–19)
- **LLM use statement (p13):** Claude helped "write analysis code, produce the figures and tables, and draft text.
  Every number in the paper is generated from the released result files."
- **Reproduction script named (p19):** `paper/make_numbers.py` — **NOT-FOUND** in any of the 13 local/remote refs
  (`git ls-tree -r`) or in the working tree.

## 3. Section outline

| § | Title | Page |
|---|---|---|
| — | Abstract | 1 |
| 1 | Introduction (incl. Contributions list and two caveats) | 1–3 |
| 2 | Related work (probing for lies; personas as directions; why probes fail to transfer) | 3–4 |
| 3 | Method: 3.1 Persona vectors · 3.2 The persona bank · 3.3 Persona coordinates · 3.4 Scoring · 3.5 Activations | 4–6 |
| 4 | Experimental setup (datasets, transfer matrix, base probe, metrics) | 6–7 |
| 5 | Results: 5.1 The hardest dataset: lies told as denials · 5.2 Cross-dataset transfer · 5.3 One direction or the persona subspace · 5.4 Zero-shot against the benchmark's probe · 5.5 Depth | 7–12 |
| 6 | Limitations (denial wording; designed repair; calibration; supervised baseline; scope) | 12–13 |
| 7 | Future work | 13 |
| 8 | Conclusion; Acknowledgements | 13 |
| — | References (19 entries) | 14–15 |
| A | The persona bank (Table 5, two personas' prompts, the 10 new questions) | 16–17 |
| B | Further tables (Tables 6, 7) | 17 |
| C | The direction at the follow-up token (Table 8) | 17–18 |
| D | Reading the direction at individual tokens (Table 9) | 18 |
| E | Three persona components trained on RepE (Table 10) | 18–19 |
| F | A contrast from a different behaviour — sycophancy (Table 11) | 19 |
| G | Reproduction and compute | 19 |

## 4. The main thesis, in plain words (DOC-ONLY — what the draft says, not whether it holds)

1. Mahadik & Skapars (2026) built lie detectors from "persona coordinates": directions in the model's activations
   taken while it role-plays deceptive vs honest characters. The draft repeats this on a much bigger model,
   Llama-3.3-70B-Instruct, on five Liars' Bench datasets plus RepE.
2. On Liars' Bench's hardest dataset, HP-KR (the model falsely says it does not know a hazardous answer), the
   original persona direction points the wrong way (AUROC 0.148 at layer 20), because none of the original
   "deceptive" characters lies by claiming ignorance.
3. Adding 10 new characters built around denying knowledge (4 who fake ignorance, 3 truly ignorant, 3 who plainly
   assert) gives a direction that — with no Liars' Bench labels — scores 0.919 AUROC and 0.910 balanced accuracy on
   HP-KR at layer 40, near the benchmark's privileged upper bound (0.95) and far above its published probes (0.50, 0.26).
4. Across a 6×6 train-on-one/test-on-another matrix, the single "extended" direction (12 vs 13 personas) averages
   0.916 AUROC at layer 40 against 0.732 for ordinary probes on the full 8,192-dimensional activations, and beats
   using the top 3 persona principal components.
5. What does not work yet: the decision threshold (set to flag ≤1% of harmless Alpaca chats) does not carry over to
   every dataset (HP-C: almost nothing is flagged; HP-KR at layer 20: most honest answers are flagged).

## 5. What it describes: experiments, data, models, layers

- **Subject model:** `meta-llama/Llama-3.3-70B-Instruct` (80 layers, hidden 8,192), all via NDIF.
- **Judge model:** Llama-3.1-70B-Instruct scores persona rollouts 0–3; only score-3 rollouts kept.
- **Layers:** 20, 40, 44 (main: 40). Follow-up-token experiments at layer 20 only. Persona vectors "at all 80 layers".
- **Persona bank:** 36 personas = 26 from Mahadik & Skapars (8 deceptive, 7 honest, 5 sycophantic, 5 anti-sycophantic,
  1 default) + 10 new (4 knowledge-denial, 3 sincere-ignorance, 3 plain-assertion); 5 prompt variants × 30 shared
  questions (20 from M&S + 10 new) = 5,400 rollouts → 3,749 kept (Table 5 gives per-persona counts).
- **Directions:** published contrast 8v7; knowledge-denial 4v6; extended 12v13; sycophancy 5v5 (control); PCs
  (k=1, k=3) of the published and extended groupings; knowledge-denial PCs (App E); 3 random directions × 5 draws.
- **Evaluation data:** CG (267 / 167 lies), HP-C (2,134 / 134), HP-KR (2,139 / 139), IT (1,080 / 423),
  ID (5,432 / 3,045), RepE (550 / 275; train+val merged), Alpaca control (2,000 / 0; threshold only).
  Soft-Trigger and Gender-Secret excluded (fine-tuned models).
- **Probes:** logistic regression, L2, C = 0.1, standardised, no intercept on pooled features; "base probe" = the
  pipeline's per-token RepE-trained probe (λ = 10) averaged over response tokens (= Liars' Bench Mean Probe recipe).
- **Metrics:** AUROC; balanced accuracy and recall at the threshold flagging ≤1% of Alpaca; 95% stratified bootstrap
  over examples (1,000 resamples), shared across detectors. Averages over the 5 Liars' Bench datasets (RepE excluded).
- **External numbers used as baselines:** Liars' Bench published Mean / Follow-Up / Upper-Bound probe values (its
  Tables 6–8, 17); Mahadik & Skapars' persona descriptions (their Table 2); Kumar 2026's 0.61–0.80.

**Distinct experiments the draft describes: 11** (numbering is ours):

| E | Experiment | Where in draft |
|---|---|---|
| E1 | Build the 36-persona bank: rollouts, judge filter, persona vectors | §3.1–3.2, Table 5, Fig 1, App A |
| E2 | Extract + mean-pool response-token activations at L20/40/44; reproduce the pipeline's own RepE probe ("base probe") to 4 d.p. | §3.5, §4, App G |
| E3 | Zero-shot AUROC of the 8v7, 4v6, 12v13 contrasts per dataset | Tables 2, 6; Fig 3 |
| E4 | Operating-point detection at ≤1% Alpaca FPR, with bootstrap CIs, vs the base probe | §5.1, §5.4, Tables 2, 4; Figs 3, 5 |
| E5 | 6×6 train–test transfer matrix for raw / contrast / PC / random feature sets (+ excl. HP-KR) | §5.2–5.3, Tables 3, 7; Fig 4 |
| E6 | Original bank vs re-extracted bank for the 8v7 contrast on HP-KR (0.103 vs 0.148) | §5.1, §6 |
| E7 | Persona geometry: 36 vectors on the 8v7 × 4v6 plane, cosine 0.47 | §3.3, Fig 2 |
| E8 | Follow-up "No" token: Follow-Up Probe re-measured + persona contrasts at that token | App C, App G, Table 8 |
| E9 | Six per-token reads of the extended contrast (mean, top-10%, max, first, last, pre-reply) | App D, Table 9 |
| E10 | RepE-trained probes on 1-d contrast / k=3 PCs at both token positions | App E, Table 10 |
| E11 | Sycophancy 5v5 contrast as a cross-behaviour control | App F, Table 11 |

## 6. Figures and tables

**5 figures, 11 tables** (VERIFIED by reading every page; figure pages rendered and viewed).

| # | Page | Caption (short) | What it shows / data |
|---|---|---|---|
| Fig 1 | 2 | Projecting lies onto persona coordinates | Schematic (drawn in LaTeX): 36 personas × 5 variants × 30 questions → 5,400 rollouts → judge keeps 3,749 → persona vectors → contrast *d* and PCs; LB response → activations at L20/40/44 → mean over response tokens → s(z) = ⟨z, d̂⟩ |
| Fig 2 | 6 | The 36 persona vectors at layer 40 | Scatter of centred persona vectors: x = projection on 8v7, y = on 4v6 (cosine between axes 0.47). Caption numbers: sincere-ignorance mean −0.08 on y vs 2.90 for feigned deniers; published deceptive mean 0.05. Viewed: knowledge-denial personas at y ≈ 2.2–3.9; sycophantic personas sit at y ≈ 0.2–1.3, above the published deceptive ones (IMAGE-ONLY observation). **Producing script/data NOT-FOUND** on either branch |
| Fig 3 | 9 | HP-KR across depth | Left: AUROC vs layer (L20/40/44) for base probe, 8v7, 4v6, 12v13, with LB reference lines (upper bound, Mean/Follow-Up). Right: BA at 1% Alpaca FPR. Same numbers as Table 2 (INFERRED). **Script NOT-FOUND** |
| Fig 4 | 10 | Train–test AUROC at layer 40 | Two 6×6 heatmaps with cell labels: raw 8192-d probe (range 0.06–1.00) vs extended contrast (constant rows 0.91/0.71/0.91/0.98/0.98/1.00). All 72 cell values transcribed in the CSV |
| Fig 5 | 11 | Zero-shot persona contrasts vs RepE-trained base probe at L40, in the layout of LB Figure 2 | Bars for BA, AUROC, recall per dataset + Average (excl. RepE), 95% bootstrap error bars, RepE hatched. **Per-dataset BA/recall values exist only in this figure (IMAGE-ONLY in the draft)** |
| Table 1 | 7 | Datasets | Name, object of belief, reason for lying, #examples, #lies (7 rows) |
| Table 2 | 8 | Detection on HP-KR | AUROC and BA@1%FPR at L20/40/44 for 8v7, 4v6, 12v13, base probe; LB Mean / Follow-Up / Upper-Bound (reported) |
| Table 3 | 9 | Mean off-diagonal AUROC of the transfer matrix | 8 feature sets × (all six, excl. HP-KR) × L20/40/44 |
| Table 4 | 12 | Detection under the Liars' Bench protocol, 5-dataset means | BA, recall, AUROC for 4 detectors × 3 layers |
| Table 5 | 16 | The 36 personas | Group, n, kept rollouts per persona (of 150) |
| Table 6 | 17 | Zero-shot AUROC of the three contrasts per dataset | 6 datasets + mean × 3 contrasts × 3 layers |
| Table 7 | 17 | Transfer per test set: raw vs 12v13 | Mean AUROC over the 5 other training sets |
| Table 8 | 18 | Response-mean vs "No"-token AUROC at L20 | 3 contrasts × 6 datasets |
| Table 9 | 18 | Six token reads of 12v13 | AUROC, BA, recall × 3 layers |
| Table 10 | 19 | Probes trained on RepE at L20 | 5 feature sets × (response tokens, "No" token) |
| Table 11 | 19 | Sycophancy contrast (5v5) | Zero-shot AUROC per dataset + off-diagonal mean × 3 layers |

## 7. What the draft itself calls limitations and future work (DOC-ONLY)

**Limitations (§6):** (1) *Denial wording* — every HP-KR lie is a denial and most honest answers affirm, so a
denial-wording detector would score well; the 4v6 contrast pools truthful deniers with plain asserters and cannot rule
this out; the cleaner feigned-vs-sincere contrast "we have not run". (2) *Designed repair* — the kd personas were
written after seeing HP-KR fail, so HP-KR is not a blind test; at L20 excluding HP-KR the extended contrast's advantage
over the published one disappears. (3) *Calibration* — the Alpaca threshold fails on HP-C and on HP-KR at L20; raw-probe
BA sits at the 0.5 floor in 18 of 30 off-diagonal cells at L40. (4) *Supervised baseline* — "A probe trained on several
datasets pooled is a stronger supervised baseline, and we do not compare against it." (5) *Scope* — one model, three
layers, one bank; prompts written from role descriptions; judge filter may bias vectors; re-extracted bank had more
rollouts; bootstrap resamples examples, not models/prompts; IT excludes the scratchpad so LB comparisons on IT are not
like-for-like.

**Future work (§7):** other model families (Mistral-Small-24B, Qwen-2.5-72B, Gemma-3-27B); fine-tuned liars
(Soft-Trigger, Gender-Secret); wording-controlled contrasts on HP-KR's honest denials; combining with black-box
signals (Aletheia's Quest), noting the top-10% token read detects better; label-free calibration. §5.5 adds: "a layer
between 20 and 40 may do better."

## 8. Mapping the draft onto the collaborator's branch (all INFERRED; full cross-check is Phase 2)

Paths are under `origin/feat/persona-pc-ood:src/probes/deception-detection/`.

| Draft element | Likely source | Basis for the mapping |
|---|---|---|
| Tables 2 (persona rows), 6, 11; Fig 3 left; HP-KR AUROCs | `results/persona_ood_ext_L{20,40,44}/zero_shot.csv` (cols `auroc_axis`, `auroc_denial_axis`, `auroc_ext_axis`, `auroc_syc_axis`) | first rows round to the draft's values (e.g. CG 8v7 0.9399→0.940; HP-KR 8v7 0.1483→0.148; CG syc 0.9989→0.999) |
| Table 2 BA, Table 4, Fig 5, §5.4 operating-point numbers; base-probe rows | `results/persona_ood_ext_L*/zero_shot_operating.csv` (arms incl. "base probe (RepE-trained)", thresholds "control" and "in-dataset (oracle)", bootstrap lo/hi) | arm names match; Fig 5 ≈ `zero_shot_fig2.png` from `scripts/persona_ood_figures.py::fig2` (but legend text differs from the PDF) |
| Table 3, Table 7, Fig 4, App F off-diagonal means | `results/persona_ood_ext_L*/summary.csv` + `matrix_{baseline,axis1,denial_axis1,ext_axis1,ext_pc1,ext_pc3,pc3,random3,syc_axis1}.csv` | `summary.csv` L20: `ext_axis1` 0.9041, `baseline` 0.6478, `axis1` 0.6511, `random3` 0.5269 |
| §6 "18 of 30 cells at the 0.5 floor" | `results/persona_ood_ext_L40/matrix_balacc_baseline.csv`, `summary_balacc.csv` | file names |
| §5.1 "original bank 0.103" | `results/persona_ood/zero_shot.csv` (original paper16 bank, L20; HP-KR `auroc_axis` 0.1029) | value |
| Table 8, App C, Table 10 "No" column | `results/persona_ood_followup_L20/` | summary text |
| App G Follow-Up Probe 0.636 vs 0.634 | `results/XXXX-9-llama-all-followup/results_table.csv`, `summary.md` | commit `fd17728` |
| Base probe / reproduction to 4 d.p. | `results/XXXX-9-llama-all`, `-L40`, `-L44` | **only `plots/*.png` are committed for these three runs — no `results_table.csv`, `scores.json` or `detector.pt`** (VERIFIED by `git ls-tree`) |
| Table 9, App D | `results/persona_positions_L{20,40,44}/reads_summary.csv`, `reads.csv` | commit `a8eb902` |
| Table 10 (App E) | `results/persona_ood_ext_L20/` and `persona_ood_followup_L20/` `liars_bench_comparison.csv` | commit `c2a0c1f` |
| Persona vectors (Table 5, Figs 1–2) | `persona including knowlede denial/*.pt` (36 files incl. `default.pt`); original bank `paper16_L20_sharedquestions/paper16_L20.pt` | folder contents; the 10 new roles' n in his `persona_ood_ext_L20/summary.md` equal Table 5's |
| Rollouts, judge scores, prompts, questions | owner side (`kd/personas/`, `docs/PERSONA_CONSTRUCTION.md`, run folders) | NOT located in Phase 0 |
| Fig 2 (cosine 0.47, means −0.08 / 2.90 / 0.05), Fig 3 | — | **NOT-FOUND**: the strings "cosine between axes", "LB upper bound", "flag nothing: 0.50" occur in no file on either branch |
| `paper/make_numbers.py` | — | **NOT-FOUND** in any ref |

**On his branch but not in the draft** (existence only, results not checked):
- `results/loo_L20/`, `results/loo_L40/` — leave-one-dataset-out pooled-probe baseline (commit `6b16a46`, 2026-09-20,
  "Leave-one-dataset-out: the fair baseline for the generalization claim"). The draft (§6) says "we do not compare
  against it" → **CONFLICT between the draft's text and the branch's contents.**
- `results/persona_ood_mod_L{20,40,44}/` — same 36 personas re-extracted to 142–150 rollouts each (commit `94db651`,
  "More rollouts per persona did not help"); its summary reports every leading arm lower (e.g. `ext_axis1` L40
  0.9158→0.9066; HP-KR 12v13 L20 0.882→0.805). The draft never mentions this bank (DOC-ONLY in his summary).
- `results/persona_ood/`, `results/persona_ood_L40/` — the first runs on the original 15/26-persona bank (L20, L40);
  only the 0.103 figure is used in the draft.
- Transfer on the sycophancy basis (`transfer_sycophancy*.png`, `matrix_syc3*.csv`) — only the 5v5 axis appears (App F).

## 9. Internal-consistency observations (PDF numbers only — NOT a check against raw data)

Computed by `audit/scripts/p0_draft_claims.py`:
- Table 6 "Mean" rows equal the mean of the six rows (all 9 cells, to rounding); Table 4 AUROCs equal the Table 6 mean
  over the five non-RepE rows (all 12). Fig 4 left: diagonal mean 0.9833 (text 0.983), off-diagonal mean 0.7323 (text
  0.732); its per-row off-diagonal means match Table 7's L40 raw column to ±0.001. Table 3 gaps
  ext_axis1 − ext_pc3 = 0.220 / 0.050 / 0.023 as stated.
- Table 5 kept counts sum to **3,599 judged + 150 default = 3,749**, so "3,749 rollouts (69%)" includes the unjudged
  default; judged-only retention is 3,599 / 5,250 = 68.6%.

Items to check in Phase 2 (flagged, not judged):
1. **Base probe L40 and L44 give identical BA and recall (0.626 / 0.281)** in Table 4 (AUROC differs 0.778 vs 0.756)
   (D-127, D-144, D-148). Also identical in his `persona_ood_ext_L20/summary.md`. Could be real or a copy.
2. **D-099** "matches or beats the published contrast (Table 7)": Table 7 has no published-contrast column; in Table 6
   at L20 the 12v13 contrast is below 8v7 on HP-C (0.639 vs 0.717) and ID (0.954 vs 0.959). Similar tension in §6
   (D-156: "matches or beats … at every layer" vs "at layer 20 … disappears").
3. **App D (D-207)** says the top-10% read is "best" for detection, but Table 9 at L40 shows the maximum read higher on
   both BA (0.750 vs 0.737) and recall (0.548 vs 0.531); top-10% leads only at L20 and L44 (by averages).
4. **0.103 vs 0.109 (CONFLICT, doc vs doc; D-085):** the draft says the original bank gives 0.103 on HP-KR at L20; his
   `persona_ood_ext_L20/summary.md` line 116 says re-extraction "moves HP-KR zero-shot from 0.109 to 0.148" (0.109 is
   the base probe's value elsewhere), while his `persona_ood/zero_shot.csv` has 0.1029.
5. **Two RepE-trained raw probes at L20 (D-216):** 0.734 mean AUROC (Table 10, pooled features, C = 0.1) vs 0.697
   (Table 4 / App G "base probe", per-token λ = 10). Different recipes — confirm which is which.
6. **Trained vs zero-shot 8v7 at L20:** Table 3 uses the trained 1-d value 0.651 (sign flipped by the HP-KR training
   row), while the zero-shot mean is 0.791 (Table 6). The abstract's comparison is zero-shot; owner F-97 argues 0.791
   is the honest comparator. Table 3 arithmetic: 8v7 − raw = +0.003 / +0.100 / +0.110; kd increment (12v13 − 8v7) =
   +0.253 / +0.084 / +0.077 on the trained numbers.
7. Several numbers in the text appear in **no table or figure**: CG recall 0.677 vs 0.174 and IT/ID false-positive
   rates 13% / 11% / 0% / 2% (L20); base-probe AUROCs 0.973 (IT) and 0.950 (ID) at L20 and 0.289 (IT) at L44; the
   Follow-Up Probe's 0.088 and 88% on HP-KR; HP-C recall 0.015; RepE BA 0.998 → 0.638 (App D, no layer given); IT base
   probe BA 0.909 (App G); CG 0.993 / 0.991 max-read AUROCs.
8. Sycophancy contrast "inverts on HP-KR at every layer", but L44 is 0.480.

## 10. Rough counts

| | Count |
|---|---|
| Rows in `draft_claims_raw.csv` | **237** |
| of which table rows (every row of Tables 1–11) | 79 |
| figure rows (Fig 1, 2, 3, 5, Fig 4 cell rows, captions) | 11 |
| restatements of an earlier claim | 5 |
| claims about other papers / published numbers | 17 |
| draft's own quantitative claims in prose | 49 |
| draft's own qualitative claims in prose (incl. interpretations) | 30 |
| method / design statements | 27 |
| limitations / future work | 15 |
| provenance statements (code release, LLM use, make_numbers.py, acknowledgement) | 4 |
| rows tagged HEADLINE (D-003, 004, 006, 008, 009, 010, 027, 028, 083, 093, 098, 099, 104, 117, 166, 167, 168, 195) | 18 |
| Figures / tables | 5 / 11 |
| Distinct experiments described | 11 (§5) |
| F-numbers cited in the draft | 0 (none; it cites no internal ledger and no result file except `paper/make_numbers.py`) |

## 11. Headline claims in one sentence each (DOC-ONLY)

1. The published 8v7 persona contrast inverts on HP-KR (AUROC 0.148 at L20; 0.593/0.589 at L40/L44).
2. The cause is the persona set: no published deceptive persona lies by denying knowledge.
3. With 10 knowledge-denial-themed personas, the 4v6 contrast reaches 0.919 AUROC and 0.910 BA@1%FPR on HP-KR at L40
   with no benchmark labels, vs LB's 0.50 / 0.26 and upper bound 0.95.
4. The zero-shot 12v13 extended contrast averages 0.916 off-diagonal AUROC at L40 (0.904 L20, 0.903 L44) vs 0.732
   (0.648, 0.716) for single-dataset 8192-d probes, and beats them on every held-out dataset at every layer.
5. One contrast direction beats probes on the top-3 persona PCs at every depth (gap 0.220 → 0.023).
6. Under the LB protocol at L40/L44 the persona contrasts beat the RepE-trained base probe on BA, recall and AUROC;
   at L20 the base probe wins on BA and recall.
7. The pipeline is reproduced to 4 d.p., and the Follow-Up Probe re-measures at 0.636 vs published 0.634.
8. Open problem: an Alpaca-calibrated threshold does not transfer to every dataset.
