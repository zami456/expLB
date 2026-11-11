# Trajectory study, Step 0: preregistration

Written 2026-11-06, before any new number was computed. The gates below only reproduce numbers already
stored by earlier studies. Every constant is in `spec.py`; the freeze (`E:/trajectory/logs/freeze.txt`)
hashes this file, `spec.py`, the analysis code and the thesis and NARCBench modules it imports, and every
scoring command refuses to run if any of them changed.

## Question

The user asked (2026-11-06) whether, instead of a probe on one layer's activation, the trajectory (how
the representation changes from one layer to the next) carries a deception signal. A probe on the
difference h[l+1] - h[l] is a two-layer probe with tied weights (-w, w). It holds no information a
two-layer probe lacks, but it is a different bet: it reads what a layer writes rather than what it
carries. On the cached band (Mahadik & Skapars deception rows, span means, no labels), adjacent blocks'
centred states have cosine 0.94-0.98, and the change between them has only 4-13% of the state's
row-to-row variance, so a single-layer probe mostly reads carried content.

The hypothesis worth testing is that the instruction is written early and carried, while the lie is
recomputed. If so, differences should read the instruction less (a) and transfer better (b).

## Data (development only; no NDIF call)

- **(a) NARCBench Core**, development runs (`E:/narcbench/rows/runs_core_dev.json`: 31 scenarios, a
  collusion and a control run each, 62 coalition agents). Llama-3.3-70B reads of the released Core
  transcripts in the audit's conditions (orig, STRIP, PLANT; `E:/narcbench/acts/llama33/released__*__core`):
  content span means at blocks 20, 32-37, 40 and 44, and the end-of-turn token at 32-37 and 40.
- **(b) Liars' Bench**, the five sets (development data for Llama, as in the question-bank study), with
  RepE as a training source in the thesis's leave-one-out. Llama-3.3-70B span means at every block from
  35 to 45: blocks 36, 40 and 44 from the thesis's pooled files, the others from the emotion-coords band
  read (`E:/emotion_coords/pooled_band`, one NDIF pass, parity with the stored reads median cos .99996).
  Also the thesis's 16-layer sweep (20, 24, ..., 76, 79). Alpaca supplies the z-scores. HP-KR's
  identical-text and denials slices come from `recall_denial.Inputs("llama").hpkr()`.
  **Amendment 1 (before the freeze, found by gate G0d's first run):** the band read covers only the
  emotion-coords full run's capped rows, and the other rows are NaN in the band files. The capped rows
  are up to 1,000 per class per set (500 for IT), drawn with a fixed seed, plus every CG, HP-KR and RepE
  row. That is 6,463 Liars' Bench rows:
  - CG 267
  - HP-C 1,134 (134 lies, 1,000 honest)
  - HP-KR 2,139
  - IT 923 (423 lies, 500 honest)
  - ID 2,000 (1,000 lies, 1,000 honest)

  Every arm of (b) uses these rows, for training and scoring, at band and sweep blocks alike; Alpaca keeps
  all 2,000 rows. G0e still checks the protocol on the thesis's full rows. No (b) number had been computed
  when this was changed.
- **Never opened:** AISI held-out, Gender-Secret, the Aletheia test sets, NARCBench Transfer, Stego and
  Verdict. Every command runs under `audit.py`, which logs each file's first open and flags these. The
  report states the flagged count (expected 0).

## Gates (all must pass before any scoring command runs)

- **G0a** D9 (unit extended contrast at block 40, rebuilt here from the bank) on the stored layer-40
  reads gives the thesis's zero-shot AUROC on CG and HP-KR to 4 dp (0.9123, 0.9124).
- **G0b** That D9 equals NARCBench's frozen layer-40 D9 (max abs diff < 1e-9) and reproduces
  `armB_core.json`'s D9 numbers (Peak AUROC, STRIP and PLANT AUROCs, decomposition E_total, E_ctx, E_txt,
  share and share CI), each within 1e-6.
- **G0c** The study's cached Core training rows equal `narcbench.score.rose_rows`. Rose et al.'s probe,
  rebuilt as the mean over blocks 32-37 of per-layer leave-domain-out fits, reproduces `armB_core.json`'s
  rose_exact numbers within 1e-6.
- **G0d** In the Liars' Bench band:
  - every band-read block misses exactly the same rows, and the thesis blocks miss none (Amendment 1);
  - labels on the kept rows are identical across blocks 35-45;
  - at least 98% of each dataset's kept rows sit closer to their own vector in the next block than to a
    random row's (layer_average's rule).
- **G0e** The raw probe's leave-one-out at block 40, using this study's float32 loader and 6 BLAS threads,
  reproduces `results/loo_L40/loo.csv` (baseline, naive) within 0.003 on each of the five sets.
- **G0f** The loader equals `persona_ood.load_pooled` (labels identical, max abs diff < 1e-5). The
  contrasts equal the bank's (`bank_at(l).axes["extended"]`, blocks 20 and 40, relative error < 1e-12).
  The bank holds every block the study uses.

## (a) NARCBench Core: do differences read the instruction less than states?

**Decomposition (the audit's).** For each coalition agent, take its honest control score as the base.
- E_total = orig colluder - base
- E_ctx = PLANT - base (colluder's instruction planted, honest reply kept)
- E_txt = STRIP - base (instruction removed, colluding reply kept)

The context share is mean E_ctx / mean E_total. Its 95% interval comes from the audit's scenario
bootstrap (10,000 draws, seed 20261020). The same draws are used for every detector, so contrasts
between detectors are paired. Each detector also gets run-level AUROCs as in the audit: Peak Suspicion
collusion vs control, STRIP vs control (T) and PLANT vs control (C).

**Families.**
- **P-EOT (primary):** Rose et al.'s recipe (their hyperparameters: C = 25, liblinear, 8 iterations,
  balanced subsample), fitted per block on end-of-turn vectors, leave-domain-out, each agent scored by the
  fold that held its domain out. On a difference the same recipe is fitted on h[l'] - h[l].
- **D9 (primary):** zero-shot, content span. States h[l] @ unit(c[l]); differences
  (h[l'] - h[l]) @ unit(c[l'] - c[l]), where c is the raw extended contrast.
- **P-content (secondary):** the recipe on the content span.
- **D9-zdiff (secondary):** z[l'] - z[l]. Each block's D9 score is standardised by the mean and sd over
  all Core dev agents in the orig condition. It is compared with D9's states.

**Pairs.** Primary: the five consecutive pairs 32→33, 33→34, 34→35, 35→36, 36→37. Secondary, reported
only: content 20→32, 37→40, 40→44; end-of-turn 37→40.

**Rule A_F, per family.** Let ΔShare = the mean over the five primary pairs of
share(difference) - ½(share(state at l) + share(state at l')). It is computed per bootstrap draw, giving
a 95% CI. Let ΔT = the same average for the STRIP-vs-control AUROC (point values). A_F holds if all of
the following are true:
1. The family is informative: some state in blocks 32-37 has a context share ≥ 0.30.
2. ΔShare ≤ -0.20.
3. The upper end of ΔShare's CI is below 0.
4. ΔT ≥ -0.05: differences keep the reply signal.
5. At most 2 of the 5 primary differences have the "no effect" verdict (E_total CI includes 0).

**Decision.** Go to Step 1 (NDIF reads at all 80 layers) if A holds for P-EOT or for D9. The secondary
families are reported but do not decide.

**My predictions:** A fails for P-EOT, because a probe trained on Core labels finds the instruction in
what any layer writes; the difference stays context-dominated. A fails for D9 too: D9 is
context-dominated at block 40 (share 0.68), and I expect its differences to inherit that.

## (b) Liars' Bench: do differences transfer better than states?

**Trained arms.** The thesis's raw probe (`persona_ood.make_probe`: StandardScaler, then logistic
regression, C = 0.1, no intercept, max_iter 1000), with the thesis's leave-one-dataset-out (naive pooling
over the six datasets; each Liars' Bench set held out in turn, RepE always a training source). AUROC is
taken on each held-out set; "mean5" is the five-set mean.
- S[l]: the state h[l], l = 35..45.
- D[l]: h[l+1] - h[l], for the ten consecutive pairs.
- SD[l]: [h[l], h[l+1] - h[l]].

**Zero-shot arms.**
- D9S[l] = h[l] @ unit(c[l]).
- D9D[l, l'] = (h[l'] - h[l]) @ unit(c[l'] - c[l]), on the ten band pairs and the 15 sweep pairs.
- Trap arms Z[L] - Z[20] for L = 36, 40, 44, with each block's D9 score standardised on Alpaca.
- ZD[l] = Z[l+1] - Z[l] on the band pairs (descriptive).

**Bootstrap.** 1,000 paired draws per dataset (seed 20261106), stratified by class, with the same draws
for every arm. HP-KR's identical-text and denials slices are drawn within the slice.

**Rules.** Each comparison takes, per pair (l, l+1), the candidate's five-set mean minus the better of
the two endpoint states. The gain is computed per draw and averaged over the ten pairs.
- **B1:** D[l] against max(S[l], S[l+1]).
- **B2:** SD[l] against max(S[l], S[l+1]).
- **B3:** D9D[l] against max(D9S[l], D9S[l+1]).

A comparison shows superiority if the mean gain is ≥ +0.02 and its CI lies above 0.

**Wording gate W.** For each comparison, take the identical-text AUROC of the candidate minus that of the
endpoint state with the higher point five-set mean, averaged over the pairs. The gate passes unless that
difference is significantly negative (CI entirely below 0). B1, B2 and B3 hold only if superiority holds
and W passes.

**Trap prediction (T1).** For L = 36, 40, 44, Z[L] - Z[20] will beat D9S[L] on HP-KR overall (CI above
0) but not on the identical-text slice (CI not above 0). If confirmed, a "late minus early" feature would
look strong on HP-KR while reading the wording.

**Descriptive only.** The same rules computed on within-pass pairs (37→38, 38→39, 41→42, 42→43) and on
cross-pass pairs. The D9 depth profile over the 16-layer sweep (D9S and D9D).

**My predictions:** B1, B2 and B3 fail; T1 is confirmed at 36, 40 and 44.

## What Step 0 does not test

- The belief-to-assertion flip with an honest-trained truth probe across blocks 15-30 (needs NDIF reads).
- Trajectory geometry with anomaly detection (judged a context-change detector by design).
- Other models.

## Changes after the freeze

Mechanics only (bugs that do not change a rule or an arm). Each is logged in freeze.txt as POST-FREEZE
EDIT n, with its reason, followed by a new FREEZE block.
