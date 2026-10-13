# T1-4 — the OR-rule test for tail recall (review weakness W6)

**Asked by the owner 2026-10-12:** "run the OR-rule test on the per-row scores".

**The weakness (review, 2026-10-07):** "At strict operating points (1% FPR), adding K worsens tail recall compared
to the persona contrast alone, which matters for practical detection." The reviewer's suggestion: "Future variants
might explore score calibration or ensembling to restore tail performance."

## 1. Pre-registration (written before any OR-rule number exists)

The test is fixed in `thesis_tools/planning/REVIEW_RESPONSE_PLAN.md` §T1-4, committed by the owner in `0b9a096a`
(2026-10-10 13:43 +0600), quoted unchanged:

> **T1-4 Tail recall (W6).** Paired bootstrap of recall at 1% FPR (Alpaca cutoff) for D9, K and D9+K. Fixed now,
> before looking: an **OR-rule** — flag if z_D9 or z_K exceeds the Alpaca 99.5th percentile (so each half keeps
> roughly 0.5% FPR). Also partial AUROC up to 5% FPR. Inputs: `k_robustness.operating_points` or T1-0's store.
> *Success:* OR-rule recall ≥ D9's on ≥ 4 of 5 datasets at realised Alpaca FPR ≤ 1%. *Miss:* keep the "ranker
> first" framing, now with CIs (§5.4.4).

Primary layer L40 (pre-registered, plan §2 rule 4); L36 and L44 are reported as post hoc.

**Added 2026-10-12, before running, because zami's per-row scores do not exist on any branch (plan §12, A1):**

1. **Scores are rebuilt here** (`score_rows.py`) from our own stores with zami's recipe, on the laptop CPU, no NDIF:
   the Pass B store (`tamara:expLB/coords/passB/<dataset>/`, mean over the reply), our Llama K
   (`random_persona_bank/results/Llama-3.3-70B-Instruct/K/llama-70b-3.3/first/K.npz`), D9 from the kd bank
   (`%USERPROFILE%\xmodel-runs\Llama-3.3-70B-Instruct\d9\`), Alpaca projections stored in `d9k/D9K_<bank>.pt`.
2. **Bank choice, decided by the sanity step only** (`sanity.py`, run before `or_rule.py`): D9 exists for three
   banks (s3, s23, all). The bank whose D9 AUROCs at L40 are closest to zami's (`results/recall_denial/d5_plus_k.csv`,
   row "D9 alone", mean absolute difference over the five datasets) is used. The OR-rule is computed for that bank
   only.
3. **Reproduction tolerance:** if, for the chosen bank, any of D9 / K / D9 + K differs from zami's AUROC by more
   than 0.01 on any dataset at L40, the result is reported as "on our rebuild; does not reproduce zami's scores to
   0.01" rather than as a test of his numbers.
4. **Bootstrap:** 1,000 draws, seed 0. Each draw resamples the Alpaca rows (thresholds re-estimated) and, per
   dataset, the lies and the honest rows separately. All scorers use the same draws (paired).
5. **Not pre-registered, reported beside the verdict:** each rule's share of the dataset's *honest* rows flagged
   (the Alpaca cutoff does not control it; F-124), and balanced accuracy at the operating point,
   BA = (recall + 1 − honest flagged) / 2. These do not change the verdict; they say what the recall costs.
6. **Gender-Secret, added 2026-10-12 at the owner's request, before running; not part of the verdict.** Scored the
   same way and reported as an extra set, `GS-base`. Only the **base read** exists on our side: Pass Q11
   (`tamara:expLB/coords/passQ11/gender-secret/`, 392 rows, 116 lies) is the base model teacher-forced on the LoRA's
   text, with no adapter (`src/assistant_axis_ndif/20_q11_sleeper.py:274`). zami's headline K 0.8645 is the
   **adapted** read (`results/recall_denial/gender_secret.csv`, L40); on the base read his K is 0.6050, inside
   the random band (97.5th percentile 0.7926). An adapted read needs either zami's `adapted_L{layer}.npz` files
   (plan §12, A2) or a new NDIF read with the adapter applied in the forward pass (his `gender_secret.py extract`).
   Thresholds for an adapted read would also need the fine-tune's own Alpaca read, which no one has: zami's
   adapted rows carry no 1%-FPR balanced accuracy for that reason. The base read is consistent as it stands
   (base-model Alpaca thresholds, base-model reads).

### 1a. Sanity outcome (2026-10-13, after `sanity.py`, before `or_rule.py`)

- **Bank s3** (mean |ΔAUROC| of D9 vs zami at L40: s3 0.0003, all 0.0180, s23 0.0188).
- **CG, HP-C, HP-KR, ID reproduce zami exactly:** every AUROC within 0.0015, Alpaca-cutoff recall equal to
  4 decimals in 11 of 12 cells at L40. **GS-base** reproduces his base read within 0.0008.
- **IT does not:** K AUROC 0.8623 vs his 0.8813 at L40 (0.8756 vs 0.8410 at L36), and although D9's AUROC
  matches (0.9811 vs 0.9803), its Alpaca-cutoff recall is 0.3002 vs 0.7494 and D9 + K's honest-flag rate 0.1370
  vs 0.2298. The ranking within IT agrees, but IT's scores sit lower against the Alpaca threshold in our store,
  which points to different reply tokens being pooled for IT. Not resolved here.
- **Rule 3 therefore applies:** max |ΔAUROC| at L40 is 0.019 (IT, K), so the result is reported as "on our
  rebuild; does not reproduce zami's scores to 0.01", the failure being IT alone. Fixed now, before the OR-rule is
  computed: the pre-registered verdict is reported on all five datasets as planned, and the per-dataset table
  marks IT as not reproduced.

## 2. Files

| File | What |
|---|---|
| `score_rows.py` | per-row scores: D9 (three banks), K, at layers 20/36/40/44, for CG, HP-C, HP-KR, IT, ID and Alpaca |
| `sanity.py` | our AUROCs and Alpaca-cutoff recall against zami's stored values; picks the bank (rule 2) |
| `or_rule.py` | the pre-registered test and the bootstrap |
| `per_row_scores.csv`, `alpaca_scores.csv`, `inputs.json` | outputs of `score_rows.py` (inputs with md5) |

## 3. Result (2026-10-13; `or_rule_verdict.json`, `or_rule_points.csv`, `or_rule_bootstrap.csv`, `or_rule_pauc.csv`)

**Pre-registered verdict at L40: MISS.** OR recall ≥ D9's on **3 of 5** datasets (needed 4), at realised Alpaca
FPR 0.0090. Two of the three are ties at the floor, and the third is the dataset that does not reproduce:

| L40, Alpaca cutoff | D9 recall | OR recall | D9 + K recall | D9 honest flagged | OR honest flagged | D9 + K honest flagged |
|---|---|---|---|---|---|---|
| CG | **0.3593** | 0.2455 | 0.4251 | 0.0000 | 0.0200 | 0.0100 |
| HP-C | 0.0000 | 0.0075 | 0.0075 | 0 | 0 | 0 |
| HP-KR | 0.0000 | 0.0000 | 0.0072 | 0 | 0 | 0 |
| IT (not reproduced, §1a) | 0.3002 | 0.9693 | 0.9480 | 0.0000 | **0.7367** | 0.1370 |
| ID | **0.9560** | 0.9238 | 0.9264 | 0.1131 | 0.0721 | 0.1056 |
| mean of five | 0.3231 | 0.4292 | 0.4628 | 0.0226 | 0.1657 | 0.0505 |
| BA, mean of five | 0.6502 | 0.6317 | **0.7062** | | | |

- **Where OR loses, the loss is outside the noise.** OR − D9 recall: CG [−0.162, −0.012], ID [−0.114, −0.007]
  (95% paired bootstrap).
- **Where OR "wins", it is not a detection gain.** On HP-C and HP-KR every rule catches 0–1 lies at the Alpaca
  threshold (HP-C: 1 of 134). On IT, OR's extra recall comes with 74% of honest rows flagged (D9: 0%).
- **The mean-of-five recall gain is IT.** OR − D9 +0.069 to +0.147 on the mean, with honest flagged +0.049 to +0.219.
- **Partial AUROC to 5% FPR** (pre-registered as "also"; threshold-free, within each dataset): D9 0.7691,
  D9 + K 0.7323, K 0.6258 (mean of five). D9 + K is below D9 on CG, HP-KR, IT and ID, above on HP-C. **This
  measure confirms the review's W6** without any threshold choice: adding K lowers the low-FPR part of the ROC
  curve on 4 of 5 datasets.
- **Post hoc layers:** L36 also misses (3 of 5). L44 passes by the letter (4 of 5: CG, IT and the two floor ties),
  but it is post hoc (W3), and its CG and IT wins come with 12% and 47% of honest rows flagged.

**Gender-Secret, base read (extra, not in the verdict).** Every rule is at chance. At L40 the recalls are D9 0.0000,
K 0.1983, D9 + K 0.0690 and OR 0.1034, with honest flagged 0.0036, 0.1268, 0.0797 and 0.0471. Balanced
accuracy is 0.498, 0.536, 0.495 and 0.528, and pAUC5 is 0.49–0.51. The OR-rule changes nothing here, because the
base read carries no signal to keep (AUROC: D9 0.4508, K 0.6046). The test that matters for Gender-Secret is the
adapted read, which needs zami's files or a new NDIF read plus an adapted Alpaca read (§1, rule 6).

**What this means for W6 (interpretation).** The plan's *Miss* branch applies: keep the "ranker first" framing,
now with intervals. D9 + K is the better ranker over the whole curve (AUROC 0.8992 → 0.9062 here; BA at the
benchmark threshold 0.6502 → 0.7062). It is not the better strict-threshold alarm (pAUC5 0.7691 → 0.7323), and
the pre-registered OR-rule does not recover D9's tail. At the benchmark's Alpaca threshold, three of the five
datasets cannot rank detectors at all: HP-C and HP-KR are at the floor for every rule, and IT swings between 0%
and 100% honest-flagged by scorer. A calibration failure (F-124), not a property of K, dominates tail recall under
that protocol.
