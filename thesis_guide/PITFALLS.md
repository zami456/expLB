# PITFALLS — what the thesis must not say, or must qualify

Companion to `METHODS_SOURCES.md`. Format: **wrong / risky → correct → source**. All sources opened in
this session; pins as in METHODS_SOURCES (owner `asix` e917ecd3 + tree, ledger md5 `f81856ef`, F-01…F-128;
collaborator `origin/paper` @16def2de). The ledger is append-only: a later F-entry corrects an earlier one
and the earlier text stays. **Always quote the latest correcting entry, not the original.**

## 0. Rules that apply to every sentence

1. **Name the metric, the dataset set, the layer, the pooling, the probe and the protocol (trained vs
   zero-shot) with every number.** Most audit failures were a correct number with the wrong label:
   "in 28 of 29 'partly' verdicts the numbers reproduced and the sentence failed" (`kd/docs/FINDINGS.md:5296-5300`).
2. **No margin from one layer.** Report the layer mean and range, and say where the quoted layer ranks.
   F-63's 3B +0.202 was the best of 3B's 7 layers; the layer mean is +0.120 (memory
   `no-margin-from-one-layer.md:19-24`; F-96 `FINDINGS.md:4317`). L44 is post hoc; L20/22/40 are
   pre-specified (`docs/LAYER_PREREGISTRATION.md:20-38`).
3. **Persona feature − matched control at the same k, never only − raw.** A low-dimensional feature
   moves toward 0.5 and "wins" where raw is weak (`27_crossgen.py:53-58`; memory `matched-control-kills-the-gain.md:11-21`).
   **Do not cite corr(raw, Δ) = −0.76 as evidence of a ceiling artifact.** It is mathematical coupling,
   inside its own null (P = 0.2135). It is withdrawn (F-74 `FINDINGS.md:2706-2726`; DF-68). It still sits in
   `27_crossgen.py:56` and `kd/docs/DESIGN.md:169-170`.
4. **Zero-shot numbers go against the random-direction null, not 0.5** (`25_zeroshot_nulls.py:4-15`).
   Several zero-shot AUROCs sit inside it: convincing-game and HP-C (memory `significance-and-random-nulls.md:35-40`).
   The deception axis on HP-KR, 0.6436, is also inside its null (DF-72, `audit/DOC_FIXES.md:799-800`).
5. **Dataset-level bootstrap is the bar.** Row-level intervals are tight by construction. The
   project has about 11 datasets, so almost nothing clears (`METHODS_CHAPTER.md:48-51`; F-04).
   Median effect 0.044 against a median interval of ±0.125 (`audit/REPORT_TECHNICAL.md:529-531`).
6. **Report the length control beside transfer numbers.** Always name the metric:
   - **Mean token gap:** the positive class is longer in 9 of 11 datasets, or 8 of 10 without
     sycophancy_mt.
   - **Length-only AUROC:** above 0.5 on 9 of the 10 datasets the control ran on.
   The two counts come from different dataset sets (F-114 §2 `FINDINGS.md:5676-5695`).
   The length-only off-diagonal is **0.5555** (L40/mean/paper, 110 pairs) or **0.5533** vs raw
   **0.6646** (L44/last/mms, 90 pairs). This session verified 0.5533/0.6646 in
   `audit/cache/labpc/run-coords/p1_L44_last_lengthctl.json` `p1[3]`/`p1[0]`. So **DF-42 and F-123 §2's
   "matches no matrix" is wrong for that configuration**, as memory `response-length-transfers-on-its-own.md:28-37`
   says. Never write 0.553/0.665 without "L44/last/mms".
7. **Dataset identity dominates the activations.** It is 99.3% decodable, against +3.2% lift for the label.
   Raw probes exploit it, and that inflates diagonals (memory `dataset-identity-dominates-variance.md:21-39`). [memory, not re-derived]
8. **Mark LAB-ONLY evidence.** Git cannot re-check it (§G).

## A. The findings that must be stated exactly

| Risky | Correct | Source |
|---|---|---|
| Three separate negative results (F-115, F-116, F-118) | **One finding, "the convergence":** where both classes deny knowledge, no persona direction reads the lie. It rests on three wording-matched tests. (1) The extended/denial axes on the text-matched slice: 0.4630/0.6203/0.5383 (these come from `audit/HPKR_WORDING_CONTROLLED.md:114-124` §5, **not "F-115 §5"**). (2) The knowledge_matched slice, 2,001 rows: the axes transfer below chance (−0.19 to −0.38 vs raw) while raw is 0.999 in-distribution. (3) Gender-Secret: every direction at chance (0.436–0.623) | `CLAUDE.local.md:83-86`; F-118 `FINDINGS.md:5936-5942`; F-128 `:6751`; F-116 `:5849-5866` |
| The convergence rows are "his exact arms" | Row (1) re-creates his arms on the **collab s23 bank** (`axes_collab_s23.pt`, k=5). His own bank is `vectors_s3`, and he found s3 → s23 costs him. Say so | `HPKR_WORDING_CONTROLLED.md:110-113`; `COLLABORATOR_FINDINGS.md:231` |
| Pooled training beats single-source by +0.178 | **+0.125** (+0.1254, leave-one-cluster-out, L40/mean/paper/seed 42, positive on 10 of 11 targets). +0.178 is LODO, inflated by the HP-KR ↔ kr-matched leak. Drop F-79's "HP-KR 0.972" entirely | F-108 `FINDINGS.md:5164-5189`; `REPORT_PLAIN.md:104-114` |
| "+0.125, five seeds, ±0.035" | The five seeds (+0.173…+0.198) and the ±0.035 dataset interval were measured on the **LODO** gain. The LOCO +0.125 is **one seed, three layers, with no dataset bootstrap** (F-108 caveats 2 and 4). Clean-9 LOCO is +0.150 | `REPORT_TECHNICAL.md:205`; F-108 `:5198-5202` |
| The pooled LODO baseline is the owner's | The **pooled LODO baseline is the collaborator's** (`6b16a46`, 2026-09-20). The **equal-row-budget arm and "diversity, not volume"** are the owner's | `ATTRIBUTION.md:118-119`; F-113 §1 `:5523-5531` |
| The decline with scale shown on one bank / "our personas rescue the method" | **Both banks decline.** Published 26-persona bank: +0.038 → +0.015 → −0.012. kd 36-persona bank (s23): +0.036 → +0.019 → +0.005. So the decline belongs to the **method**, not the bank. The metric is layer-averaged **(per-layer best persona feature) − raw**. That max over features is post hoc. The run is 5 datasets without HP data, k=3, one seed, no test | F-121 `FINDINGS.md:6026-6059`; `audit/scripts/p12_kd_scale_ladder.py:59-73`; DF-86/87 |
| 70B kd "stays positive" | +0.0054 is a whisker above zero. Give the direction (bank gap −0.005 → +0.005 → +0.017), not the number | DF-87 `DOC_FIXES.md:922-926` |
| Persona coordinates replicate at 3B/8B and die at 70B (+0.120/+0.103/+0.075) | Like-for-like 70B is **+0.054 vs datapca and −0.013 vs raw** (F-110; the +0.075 sweep used the kd bank). The monotone decline is a **mean-pooling** result; under `last` it is not monotone. The rungs differ in judge filter (s23 vs s3) and ran at 1024 tokens. The 8B corpus lacks 5 of 12 datasets | F-110 `:5259-5272`; DF-47, DF-73; F-63 caveats `:2031` |
| Cap bank: a capitulation axis detects sycophancy | **0.745 zero-shot vs 0.374** on sycophancy_mt, and it clears the bootstrap (+0.112 [+0.050, +0.185]). But it is **local**: over the full 11×11 it is +0.098 [−0.036, +0.208], and it clears at 14/21 layers under mean pooling and only 3/21 under `last`. The zero-shot 0.745 has no dataset bootstrap | `cap/docs/DESIGN.md:226-262`; DF-70 `DOC_FIXES.md:748-757` |
| pc:kd_assertion is a new positive result / "nothing beats raw" | It beats raw at 18/21 layers (+0.10–0.15 from L32), but **0 of 126** tests clear against datapca. It is 70B-only, 11 datasets, k=5. It is still worth reporting: 0.8119 vs pc:knowledge_denial 0.7492 is the register diagnosis made visible | F-112 `:5442-5473`; DF-91/92 |

## B. Superseded or corrected numbers — do not repeat

| Do not write | Write | Source |
|---|---|---|
| "The paper16 **sycophancy** axis scores 0.374 on sycophancy_mt" | 0.374 is the **deception** axis PC1 (contrast 0.409, inside the random null). The sycophancy axis reads **0.313** [0.283, 0.346], significantly inverted (p 0.001–0.008). Do not compare either with the trained random probe's 0.541 | DF-31; F-111 item 8 `:5370-5384`; `27_crossgen.py:411-416` |
| "re-measured 0.3061/0.2996 on passA11/passC11" | Both values are passA11's (contrast/PC1). passC11 reads 0.3066/0.2995 | DF-103; F-128 `:6748` |
| F-04's "11-dataset run" +0.0147 / +0.0264; "1 significant test" | That was block 9 = 10 datasets. On all 11 both flip, to −0.0215 / −0.0243. The one BH-significant test is `random@2 − raw` **losing** | DF-01, DF-02 |
| kd slice positive at "37/80 (46%)" | 28/80 vs raw, **31/80 vs datapca** | DF-03; F-128 `:6748` |
| F-61 "+0.006, twelvefold" | **+0.0093, 8.1×** (8.2× in F-123). The old figure came from a label-parsing bug | DF-51/71; F-123 §2 |
| F-03 matched-slice margins as first printed | Leak-guarded `axes − raw`: **−0.2545 (L20) / +0.0309 (L40) / +0.0050 (L44)**. At L40 over datapca, +0.066 [−0.069, +0.189], P = 0.161: **not significant**. The best-layer interval is L5–L79 | DF-78; F-76; `HPKR_WORDING_CONTROLLED.md:100-110` |
| "The matched slice reverses the HP-KR result" (unqualified) | The reversal is layer-localised: 0.739 at L40, 0.374 at L20 | F-72 `:2567` |
| "Every one of kr-matched's 258 rows is in the training pool" | About 81% (42/52 measured). The same wording is in `30g_clustered_holdout.py:12-14` | F-113 §5 |
| F-75's plateau "strengthens" | Withdrawn by F-76 | F-76 `:2904` |
| F-91 "L32–36, not 40" | "A plateau over roughly L32–L44 that this measurement cannot order." The quoted values mixed seeds | DF-53 |
| "Equivalent within ±0.04" (layers 20 vs 22) | The pre-registered margin was ±0.02, and all four tests fail it. Say "close; formal equivalence at ±0.02 not established" | DF-63 |
| "All 1,400 rollouts re-judged; judges differ by 0.8 pp" | **1,350** judged (default never judged). 0.74 pp; kept **sets** differ by 8.15%. Also in `PERSONA_CONSTRUCTION.md:167` | DF-64 |
| Regex (0.91) beats the probe (0.87) on HP-KR | The raw probe diagonal is **0.9663**. The regex is **0.9125**, which is 94.4% of the probe, not above it. The 0.9597 regex is **unrecorded** (a reconstruction gives 0.9413) | DF-55, DF-105; `LAYER_PREREGISTRATION.md:93-103` |
| sarcasm "gates registered before generation" (G2 ≤ 0.65) | 0.65 entered git in the same commit as the failure. v1 pre-registered ~0.75; v2 removed it. Measured 0.921/0.947 fail both | DF-66 |
| Sarcasm Gate 3 count "below p95 in two cells" | State the conclusion only: the deception axis is inside the random null in every cell (0.543–0.586). Sycophancy fires at 0.693/0.710 | F-113 §4 `:5566-5586` |
| Unified PC1 cos 0.557 | **0.389** (confounded bank); **0.896** on paper16_shared | F-18 `:304-306` |
| Unified PC1 "is the harm axis" | On the default bank, PC1 is the question bank (r −0.945). This damages `pc:unified*` and `PC1:unified` only, not the contrasts or the deception/sycophancy numbers | `EXTRACTION_VS_PAPER.md:169`; F-113 §6 |
| "We sweep k" | k=5 by default at 70B, k=3 at the rungs, k=2 for `@2`. k=1 was never run at 70B. The result is k-dependent (+0.138 at k=2 vs +0.004 at k=5) | `METHODS_INVENTORY.md:57-65`; AD-1/AD-3 |
| Dataset-PCA and random-subspace controls are ours | Both come from the replicated paper. **What is ours is running them at matched k under a dataset-level bootstrap.** 9 of the controls are absent from the paper | `METHODS_CHAPTER.md:90-94`; F-39 correction 1 `:1016` |
| F-56 "`--sample` preserves the class ratio" | It balances: 134/666 is the class ceiling; the source share is 6.28% | DF-52 |
| F-78 quotes Ying Fig 4 panel (b) as "transfer to sycophancy" | That direction is panel (a). The conclusion is unchanged | DF-54 |
| DF-46 "inversion only at L20 in axis1" | The inversion appears in **7 (arm, layer) cells**, always the HP-KR training row | F-124 §1 `:6288-6306` |
| "F-32's lexical counts show the cap bank unconfounded" | Rest it on **teacher forcing**: turn 1 is byte-identical across all 65 arms (1,300 rows, 20 questions) | DF-39; F-111 item 2 |
| F-29 "inverted axis fails as source" | Mean pooling only; under `last` it beats random at 18/21 | DF-40 |
| F-89 "78% / three-to-one" (format control) | Presence beats content about 2:1 on average. Never quote the ratio | F-94; DF-62 |
| q8 "top10pct +0.053" as the per-token effect | +0.1027 at L20, collapsing to +0.005 at L40. The trained case reverses | DF-61; `kd/docs/DESIGN.md:224-226` |
| Erasure costs "1–3%" | 0.80–2.64%; 3 of 6 settings are below 1% | DF-62, DF-65 |
| "Assistant Axis" when our direction is meant | Our direction has cos 0.30–0.37 with Lu et al.'s published axis. Say which one | `PERSONA_CONSTRUCTION.md:50-59`; DF-60 |
| "16-persona bank" | "the 26-persona bank" or "the paper's 16-persona deception set" | DF-90 |
| F-120 raw 0.8195 at L40 | That is the 5-dataset reading; over all 10 it is 0.6760 | F-128 `:6752`; DF-102 |
| F-117 "within ±0.06, crosses zero five times" | Excl. HP-KR it runs −0.033…+0.063 (L56 +0.0633) and changes sign **6** times, positive at 11/16 | DF-101; F-128 `:6750` |
| F-112 "8 clear BH over 6,300" | Scoped to block 10/mean/paper; 34 family-wide | F-128 `:6749` |
| "pc:sycophancy never trained in our pipeline" | Trained (6D-1). It loses to raw at all 6 (layer, pooling) cells on 11 datasets | F-126 §1; DF-109 |
| "Our pipeline has no sycophancy axis anywhere" | His branch has four syc arms. `syc_pc1` wins excl. HP-KR, carried by convince_me (n=1) | F-122 |

## C. The collaborator's draft at 16def2de — residual overstatements (do not import)

| Sentence (file:line @16def2de) | Correct | Source |
|---|---|---|
| "from layer 36 it ranks HP-KR better than any contrast" (`introduction.tex:68-70`) | Leads at 8 of 12 layers; trails at L44/48/52/76 | DF-96 |
| "≥0.026 on the other datasets at all 16 layers" (`limitations.tex:17-21`) | The 0.026 includes HP-KR; excl. HP-KR it is −0.011 at L20 and <0.026 at 8/16 | DF-97; DF-81 still open |
| "ranks HP-KR's lies better than we do"; "leads on all six by ≥0.026"; "higher than the contrast ever reaches" (`results.tex:254-255, 261-263`; `abstract.tex:18-19`) | Add "at 8 of 12 deep layers" / "on average" | DF-98 |
| "we have not run it" (wording-controlled test) (`limitations.tex:10-11`, future work `:67-68`) | Run three times: F-03/F-72/F-76, F-115, F-116 | DF-76; F-125 §4 |
| "A probe trained on several datasets pooled … we do not compare against it" (`limitations.tex:42-44`) | His own `loo_L20/L40` (2026-09-20) makes that comparison, and it erases most of the advantage | `ATTRIBUTION.md:203-207`; `COLLABORATOR_FINDINGS.md:168-172` |
| "the top-10% read is best" / "top tenth is a better read" (`appendix.tex` §app:reads; `limitations.tex:73-74`) | At L40 the single maximum wins (BA 0.750 vs 0.737; recall 0.548 vs 0.531) | DF-15; F-125 §4 |
| "recalls near zero for every method" on HP-KR | False for the Follow-Up Probe (0.37 at 84% FPR) and the Upper-Bound (0.97) | DF-13; F-125 §4 |
| BA@1%FPR as if every cell ran at 1% | 23 of 90 cells exceed 1% realised FPR, the worst at 88.3% (4v6, L20). The Upper Bound runs at 8% | F-124 §4 `:6343-6363` |
| "within 0.03 of the privileged upper bound" | 0.0310 (L40) and 0.0432 (L20). On BA@1%FPR the gap is 0.196 | DF-26 |
| abstract "lifts transfer by 0.18–0.21"; "PCs do not contribute"; "0.09 below the contrast"; "inverts to 0.124/0.523" | Sign artifact (corrected ≈ +0.03/0.00/−0.005/−0.003). PC1 beats the contrast on the published bank. The gap is 0.148. The two weightings are mixed | DF-20/21/22/27 |
| "six Liars' Bench datasets"; "independently constructed sycophancy axis" | Five (RepE is not Liars' Bench). The sycophancy axis is built on the same kd bank object, |cos| 0.54 | DF-23, DF-24 |
| Release: "every number is generated from the released files" | `make_numbers.py` and `numbers.tex` exist (F-119, F-127). The residual gaps are: the persona vectors (Figure 2 cannot be regenerated), `results_table.csv`, and hard-coded app:depth constants | DF-99; F-127 `:6698-6703` |
| "The benchmark reports 0.26" as general | The Llama-70B row. The cross-model average is 0.42 | F-124 §5 |

## D. Attribution — who built what

Neither branch supersedes the other. Pair each claim with the branch that holds its evidence, and
credit each author on the evidence in `audit/methods/ATTRIBUTION.md` (git first-commit dates). A first
commit is an upper bound on the date, not proof of idea priority (`ATTRIBUTION.md:28-48`).

- **Owner (24 of 26 designs):**
  - HP-KR matched slice and the factor probes.
  - The kd 36-persona bank: personas, questions, rollouts and vectors. His 36 `.pt` files are
    byte-identical to hers (RF-1).
  - The cap bank and teacher forcing.
  - sycophancy_mt.
  - The sarcasm design and the bag-of-words gate.
  - Controls: matched datapca at k, the dataset bootstrap, per-dataset random nulls, length control,
    format control, LOCO, the layer pre-registration, the permutation tests and the seed checks.
  - The equal-budget pooled arm and single-source orthogonality.
  - Infrastructure: the NDIF pipeline, the attention budget, scheduled tasks, the passA/B/C design,
    and the bf16 rule (`ATTRIBUTION.md:52-58, 80-137`).
- **Collaborator:**
  - The pooled LODO baseline, with `loo_scores.npz`, the only per-example data committed on either branch.
  - The resumable cache and batcher (the owner ported them).
  - The rollout-count control, built on her banks.
  - The Follow-Up-probe runs.
  - The per-token profile, reached independently of the owner's `14_pertoken.py` (hers came earlier).
  - `random3`.
  - The earlier bf16-ULP measurement.
  - The 16-layer sweep, validated.
  - Gender-Secret, the best-controlled experiment (`ATTRIBUTION.md:141-152`; `COLLABORATOR_FINDINGS.md:260-269`).
- **Conflict to resolve before writing credit for the ten knowledge-denial roles.**
  - `COLLABORATOR_FINDINGS.md:39-41,262-263` credits him with "building the ten roles".
  - `ATTRIBUTION.md:83` (git, RF-1), F-97 `FINDINGS.md:4389-4394` ("Those ten extra roles are ours"),
    DF-17, `PERSONA_CONSTRUCTION.md:83` and `kd/docs/DESIGN.md:38-49` (the assert/deny hypothesis,
    `5b43a7a`, 2026-09-19) all make the **roles and the diagnosis the owner's**.
  - His own contribution is the **4v6 / 12v13 regrouping and its evaluation**. That regrouping merges
    sincere_ignorance with plain_assertion, which is the split built to remove the wording confound
    (DF-25; `ATTRIBUTION.md:188-194`).
  - His paper's "the three groups marked new are ours" (`appendix.tex` tab:roster caption) is joint
    wording and needs the same care.
- **Abstract 0.904/0.916 is `ext_axis1`**, which runs on the owner's kd personas. The published 8v7
  reads 0.791/0.832; about 45% of the margin is her bank (DF-17; F-97; F-99 gives the median at 63%).
  "The dim=1 arms are zero-shot" is **disclosed** in his method (§3.4). Do not fault him for it (F-125 §1).
- **Owner's F-85 "never compared against a pooled probe"** was false when written (F-113 §1). Correct it
  in both directions together (`ATTRIBUTION.md:209-212`).
- **The audit's errors against him:** DF-12 (make_numbers "missing") is **WITHDRAWN**. The dim=1
  framing, the L20 sign flip and the release claim were all answered in his files (F-127 `:6705-6710`).

## E. Lab-only numbers — cannot be checked from git

The ledger's own list is F-110, F-113 (part), F-114, F-115, F-116, F-120, F-121 and F-126 (`FINDINGS.md:6766`).
Also LAB-ONLY:
- `run-paper16-shared/` (268 MB)
- sycophancy_mt rows
- `make_view.py` / `make_view_ac.py`, which build kr-matched; copies are in `audit/cache/labpc/loose/`
- all passA/B/C/P stores
- the scale-rung `run-scale/`
- `expLB-resplit` (F-115, F-116, F-121)
- F-107's numbers, which were console output only and later rebuilt (F-113 §4)

The collaborator's pooled activations are committed nowhere either. His numbers are "attested, not
reproduced" (`COLLABORATOR_FINDINGS.md:19-23`). Gender-Secret is [LEDGER], not re-derived (F-118
`:5949`). Mark these in the thesis as measured on the lab store, with the path.

## F. Stale meta-documents — do not quote as current

- `CLAUDE.local.md` gives "122 F-entries", "Phase 6 NOT started", "45%" and "DOC_FIXES not one applied". The
  current state is 128 entries, Phase 6 done, 533/735 = 72.5%, and 11 correction blocks applied
  (DF-108; F-128 `:6762`; `DOC_FIXES.md:3-6`).
- `LIMITATIONS_AND_FUTURE_WORK.md:156` says `make_numbers.py` "exists on neither branch". **False.** It is
  on origin/paper at `8fd071b2` (DF-12 withdrawn; F-123 §1).
- `LIMITATIONS…:163-167` and DF-48 say "not fixed". `run_scale_chain.ps1:197-212` now writes one bank per
  variant, but runs before the fix used the overwritten bank.
- `9_persona_probes.py:120-129` comment "9 of 10, 505 vs 310 / 245 vs 96" is superseded: sandbagging
  529.4 vs 340.4, mask 283.2 vs 176.4 (DF-10).
- F-113 §6 (`FINDINGS.md:5601`) says "35 steps". Use `EXTRACTION_VS_PAPER.md:43-53`:
  36 steps, with 4 accidental divergences in 3 rows. `METHODS_CHAPTER.md:72` agrees.
- `METHODS_CHAPTER.md:154-157` says "177 of 692 checked". Superseded (F-128).
- `docs/thesis/persona-coordinates-70b-results.md:1397-1401` still presents −0.76 as mechanism (withdrawn, §0.3).
- `PERSONA_CONSTRUCTION.md:177-179` says vectors are "(21, 8192)". The paper16 bank is 80 layers, and
  the collaborator's method says the kd bank is 80 layers. UNKNOWN until checked.

## G. Citation pitfalls

- **MASK** is Ren, Agarwal, Mazeika, Menghini et al. (CAIS + Scale AI), arXiv:2503.03750. It is **not**
  Cadenza Labs, which hosts the fork. Cadenza Labs is Liars' Bench (`CLAUDE.md` submodule table). His
  refs.bib has **no MASK entry** at all.
- **Unverified venues:** "GEM @ ACL 2026" (Kumar) and "Findings/Proceedings of ACL 2026" (Kirch) (DF-19).
  `cap/docs/DESIGN.md:91` says "ACL Findings 2026", while his refs.bib says "Proceedings of ACL 2026".
- **arXiv:2602.01425's title conflicts:**
  - "One Probe Won't Catch Them All" (refs.bib; F-126 §3)
  - "Building better deception probes using targeted instruction pairs" (in-repo PDFs per
    `COLLABORATOR_FINDINGS.md:209-210`)
  Verify before citing.
- `thesis/bibliography/references.bib` holds only template entries. Add every citation in METHODS_SOURCES §14.
- Liars' Bench names its judge `claude-3-sonnet-20250219`, a malformed model ID. Do not copy it
  (`liars-bench-notes.md:98`).
- "20th-percentile layer" in Liars' Bench is actually **layer 20**, the 25th percentile. Cite layer 20
  (`liars-bench-notes.md:103`).

---

## COVERAGE

| Requested source / topic | Covered in | Status |
|---|---|---|
| FINDINGS superseded/corrects scan (F-01…F-128, incl. F-111, F-113, F-114, F-123, F-124, F-125, F-127, F-128) | §B, §A | covered; F-01…F-107 scanned by header and supersede lines, not read in full |
| DOC_FIXES DF-01…DF-109 | §B, §C, §A | all P1 covered; P2/P3 selectively; DF-12 withdrawn noted |
| CLAUDE.local "three findings" | §A | covered, with the seeds/±0.035 scope correction |
| Memory rules (one layer, metric, matched control, nulls, dataset identity, length) | §0 | covered; the 0.553/0.665 conflict resolved against DF-42 by opening the cached file |
| Collaborator residuals DF-96…DF-99 + limitations lines | §C | verified against `git show 16def2de` |
| Attribution + branch pairing | §D | one internal conflict flagged (ten roles) |
| Lab-only numbers | §E | covered |
| Citations | §G | 1 title and 2 venues UNKNOWN |

---

## Addendum 2026-09-24 (lead): the collaborator's abstract changed after the pin

`origin/paper` moved from 16def2de to `d5b92c5a` (commits 54dc08c4, f24ca67c, d5b92c5a). The changes
are to `abstract.tex` and `main.pdf` only, and they are **not audited**. See `audit/BRANCH_PINS.md`, Addendum 2.
- **"We extend the persona bank with ten roles that feign ignorance…"** is an attribution risk. `audit/methods/ATTRIBUTION.md:83` says his 36 `.pt` files are byte-identical to the owner's, and F-97 and `kd/docs/DESIGN.md` say the owner built the ten knowledge-denial roles. His own contribution is the 4v6/12v13 regrouping. In the thesis, credit the roles to the owner and the regrouping to him.
- **"the baseline probe … flags nothing, as does every other detector we tried"** is unverified: no `claims.csv` row exists. Do not repeat it without re-deriving it from `persona_ood_ext_L40/zero_shot_operating.csv` @ 16def2de.
- Earlier abstract text (D2-002, the "8 of 12" wording) no longer appears in his abstract. The underlying numbers are unchanged.


---

## Addendum 2026-09-25 (Chapter 5 writing pass): corrections to this guide, kept as an addendum

Found by the Chapter 5 writers and verifiers (`thesis_tools/planning/CH5_VERIFY_TABLE.md`; proposals DF-110…DF-120 in `audit/DOC_FIXES.md`). The lines above are unchanged; read them with these corrections.

- **§C "23 of 90 cells exceed 1% realised FPR"** → **24 of 90** under the stated rule (6 arms × 5 LB datasets × 3 layers); 26 of 108 with RepE. DF-110.
- **§C "The Upper Bound runs at 8%"** → the 8% is the Llama-70B **HP-KR** cell of LB Table 8; over the five datasets the Llama-70B upper bound's FPR averages 0.18 (HP-C 0.50). DF-113.
- **§B "inversion appears in 7 (arm, layer) cells"** → **9 of 24** at the pin 16def2de (uni_axis1 was added in 2ad758fb and inverts at L20 and L40). DF-111.
- **§A row "Cap bank … clears at 14/21 layers under mean pooling and only 3/21 under `last`"** → state the control: vs `datapca@2` 15/21 (mean) and 6/21 (last); vs raw 13/21 and 3/21. DF-116.
- **"The pooled probe wins CG, IT, ID and RepE"** (THESIS_BRIEF :168, plan row 3.6) is false: it wins CG only. DF-112.
- **"D8 is the only arm above 0.5 BA on HP-KR at all 16 layers"** → 15 of 16 (L28 0.4801); the 8v7 arm reaches 0.5026 at L56. DF-115.
- **F-120's late-layer artifact reading** holds on the 5-dataset view only; on all 10 datasets random directions fall (0.5718 → 0.5498) while `axes` rises. DF-114.
- **Zero-shot values against the random null:** the null in `run-coords/zeroshot_nulls.json` is stored only for `corpus_L40_mean` and `corpus_L44_*`; neither pipeline stored a layer-20 null, so L20 zero-shot inversions are judged against 0.5 alone (say so).
- **Gender-Secret (F-118) "every direction at chance"**: two adapted readings exclude 0.5 upward (5v5 L20 0.5927 [0.5303, 0.6555]; 8v7 L44 0.6231 [0.5647, 0.6757]) and D8 excludes it downward at L40 (0.4359 [0.3728, 0.4962]); the convergence finding stands, worded with the intervals.
- **§E lab-only list:** `run-sycophancy-mt/rows.jsonl`, the 42 collab-bank matrices, the 48 cap_s23 and 42 cap_ctrl matrices, `kd/results/phase2_gate.json` and seven logs are now byte-exact in `audit/cache/labpc/` (md5-checked, `audit/cache/MANIFEST.csv`, 2026-09-25). `run-paper16-shared/axes_vectors_s3.pt` (63 MB) was read on the lab PC, not cached: cos(unified PC1, unified contrast) at L40 = 0.8956713.
