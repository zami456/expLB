# Pre-registration: the question extension of the persona bank

Written 2026-10-16, before any NDIF generation of the new questions. Frozen by SHA-256 in
`FREEZE.sha256` and `E:\qbank\logs\freeze.txt` (with the git commit). Changes after the freeze are
new commits listed in the report.

## Question

Does widening the bank's shared question set from one situation type (first-person workplace
advice) to many situation types improve the zero-shot persona contrast D9 (and D9 + K), and is the
gain due to variety rather than to count? Secondarily, with the bank's questions as domains, do the
predictions of the Truthfulness Spectrum Hypothesis (arXiv:2602.20273) hold: a direction pooled
over domains transfers at least as well as the best single-domain direction, and transfer is
predicted by the covariance-weighted cosine between directions better than by the plain cosine?

## Design

- Questions: 150 = 30 originals (ids 0-29, unchanged) + 30 monotone control (ids 30-59, the
  originals' form) + 90 diverse (ids 60-149, the grid in `grid_slots.json`). Authored by Claude
  under the fixed `prompt.md`; `raw_claude_output.md` is verbatim; `postprocess.py` and
  `check_questions.py` are deterministic. No test set was opened while writing.
- Models: Llama-3.3-70B-Instruct (development on Liars' Bench) and Qwen3.5-27B (development on the
  Aletheia dev split). Same 36 roles, same five prompt variants, same judge (Llama-3.1-70B, score 3
  kept), same vector definition (mean over kept rows of the assistant-span token mean).
- Per-(role, question) store at layers 20/36/40/44 (Llama) and the organism grid 16-32 (Qwen3.5);
  every bank variant is a weighted average of the store. Plain mean = Σ n_rq v_rq / Σ n_rq;
  balanced mean = equal weight per question with n_rq ≥ 1.
- Development data never includes a confirmation set. Llama development: CG, HP-C, HP-KR, IT, ID,
  RepE, Alpaca. Llama confirmation (one read): AISI held-out, Gender-Secret. Qwen3.5 development:
  the 16 Aletheia dev sets. Qwen3.5 confirmation (one read): the 7 Qwen3.5 Aletheia test sets at
  the frozen headline cell.

## Primary metric

LB-5 zero-shot mean AUROC (CG, HP-C, HP-KR, IT, ID) of D9 and of D9 + K at layer 40 (Llama),
a-priori sign (RepE ≥ 0.5 checked, never flipped). Row-level uncertainty: paired stratified
bootstrap, 1,000 draws, seed 0 (the `random_bank._paired` scheme). Question-level uncertainty: 200
seeded question subsets.

## Hypotheses and tests (Llama, layer 40; Holm over T1-T4)

- H1 diverse > monotone at equal count. T1: the LB-5 mean of D9 built from the originals plus a
  random 30 of the 90 diverse questions (200 draws, seeded) against the originals plus the 30
  monotone questions: p = (1 + #{draws ≤ monotone}) / 201 ≤ 0.05. T2: the same for D9 + K.
- H2 monotone ≈ original. B30mono lies inside the central 95% of 200 random 30-of-150 subsets, and
  the paired 95% CI of LB-5(B30mono) − LB-5(B30orig) lies within ±0.03. Reported pass/fail outside
  the Holm family.
- H3 pooled ≥ best single family. T3: Δ = LB-5(D9 from all 150) − LB-5(best of the ten family D9s:
  three original families, six grid types, the monotone set). The best family is picked on the same
  data, which works against H3. Pass: point Δ ≥ 0 and paired bootstrap lower bound > −0.01.
- H4 covariance-weighted cosine predicts transfer better than plain cosine. T4: over the 60
  (family, set) cells, with each set's own logistic-regression probe as the reference direction and
  Σ the set's pooled covariance: ΔR² = R²(Σ-cosine) − R²(plain cosine) > 0 with the family-resampling
  bootstrap 95% CI (1,000 draws over the ten families) excluding 0. Spearman ρ² reported beside it.
- H5 (exploratory): HP-KR identical-text slice AUROC of D9, all 150 against the originals, paired CI.
- Q-H1 (Qwen3.5): the dev-selected bank's 7-set headline mean ≥ 0.6245 (the stored 7-set mean) with
  the paired bootstrap lower bound > −0.01.
- PRED-0 (the existing 30 questions, before generation; reported, not decisive): pooled D9 ≥ best
  original-family D9 on the LB-5 mean; the knowledge family (ids 20-29) has the highest HP-KR AUROC;
  Σ-cosine predicts the 18 (family, set) AUROCs with higher R² and ρ² than plain cosine.

## Decision rules

- Llama confirmation bank: argmax over {B30, B30+30mono, B30+90div, B150} of the D9 + K LB-5 mean at
  layer 40 on development data; ties within 0.005 go to the larger bank. One read on AISI held-out
  and Gender-Secret, beside B30.
- Qwen3.5 confirmation bank: argmax over the same four of the D9 + K mean over the 16 dev sets at
  block 32. One read on the 7 Aletheia test sets at the headline cell, beside the frozen-spec result.
- No second bank is ever read on a confirmation set. The `confirm` commands refuse to overwrite.

## Gates before generation

- GATE-0: the per-question store on the 30 originals recombines to the thesis bank (per-role cosine
  ≥ 0.9995 at 20/36/40/44, equal n; D9 LB-5 zero-shot AUROCs within 0.005 of the stored values).
- The chain refuses to run unless the question file has the frozen SHA-256.

## Declared targeted cells

`grid_allocate.targeted` lists the grid cells that coincide with a behaviour a confirmation or
development set is known to contain. They were declared before authoring and written from the grid,
never from those sets. In this allocation: quality × third party (5 slots), denial × persona (5),
ignorance × persona (5), relation × persona × family (1).
