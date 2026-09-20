# Leave-one-dataset-out: the fair supervised baseline

Covers `results/loo_L20/` and `results/loo_L40/`. Llama-3.3-70B, pooled response activations,
persona bank from `persona including knowlede denial/` (36 personas, 12 vs 13 on the extended
axis).

## Why this run exists

Every off-diagonal cell of the transfer matrix in `persona_ood_ext_L*/` trains the 8192-d probe
on **one** dataset and tests it on another. That is not how anyone would use labels they already
hold, so it understates supervision and inflates the persona axis's apparent margin. Here each
held-out dataset is scored by a probe fitted on the **pooled other five**.

Two poolings, because the datasets differ in size by a factor of twenty (ID is 5432 rows, CG is
267), and there is no principled answer to which is right:

- **naive** — plain concatenation; more than half the rows are ID.
- **balanced** — `sample_weight = 1/n_d`, so every dataset contributes equally.

## The straw-man charge was partly right

Pooled supervision really is stronger than single-source, at both depths:

| baseline, mean AUROC over held-out sets | L20 | L40 |
|---|---|---|
| single-source (mean over the 5 training sets) | 0.648 | 0.732 |
| pooled, naive | 0.694 | **0.850** |
| pooled, balanced | **0.794** | 0.804 |

So the headline comparison had to be redone, and the margin does shrink.

## It still does not overturn the result

| L20 | mean | **worst dataset** | excl. HP-KR |
|---|---|---|---|
| **ext_axis1** (1-d, zero-shot) | **0.904** | **0.639** | 0.909 |
| baseline, balanced | 0.794 | 0.124 | **0.928** |
| baseline, naive | 0.694 | 0.325 | 0.768 |

| L40 | mean | **worst dataset** | excl. HP-KR |
|---|---|---|---|
| **ext_axis1** (1-d, zero-shot) | **0.916** | **0.712** | 0.917 |
| baseline, naive | 0.850 | 0.523 | 0.915 |
| baseline, balanced | 0.804 | 0.489 | 0.853 |

**The persona axis has the best mean and the best floor at both depths, out of all ten
arm/weighting combinations.** No other arm gets within 0.31 of its worst-case at L20 or within
0.05 at L40.

## Three honest qualifications

**1. Excluding HP-KR, pooled supervision matches or beats it.** L20: 0.928 against 0.909, a loss.
L40: 0.915 against 0.917, a dead heat. **The persona axis's average advantage is carried almost
entirely by HP-KR** — where the pooled baseline inverts to 0.124 (L20) and 0.523 (L40).

**2. HP-KR is the dataset the extended personas were authored to fix.** This is circular and has
to be stated as such: the personas were written after seeing HP-KR fail, and HP-KR is now the
sole source of the remaining margin. The bootstrap intervals below are real, but they measure
the size of an effect that was selected for.

**3. The pooled baseline is unstable to a choice with no right answer.** At L20 balanced beats
naive by 0.10; at L40 naive beats balanced by 0.05. Same code, same data, opposite verdict.

## Per-dataset, against the stronger (balanced) pool, with 95% bootstrap CIs

Δ = pooled baseline − persona axis; negative means the persona axis wins. 1000 stratified draws.

| held-out (n deceptive) | L20 Δ [CI] | L40 Δ [CI] |
|---|---|---|
| CG (167) | **+0.030** [+0.015, +0.047] | **+0.085** [+0.056, +0.115] |
| HP-C (134) | **+0.116** [+0.063, +0.167] | −0.223 [−0.274, −0.173] |
| HP-KR (139) | −0.757 [−0.816, −0.691] | −0.351 [−0.400, −0.298] |
| IT (423) | −0.012 [−0.019, −0.007] | −0.181 [−0.208, −0.157] |
| ID (3045) | −0.005 [−0.008, −0.002] | −0.001 [−0.003, +0.000] *(ns)* |
| RepE (275) | −0.030 [−0.045, −0.017] | −0.000 *(ns)* |

Eleven of twelve intervals exclude zero, so the differences are real at this sample size; the
two marked *(ns)* are ties. **Pooled supervision beats the persona axis on CG at both depths and
on HP-C at layer 20** — it is not a clean sweep in either direction.

## Against the Liars' Bench published probes

Five shared datasets, `liars_bench_comparison_loo.csv`. The extended axis had never been entered
into this table before — `run` only ever passed the original geometry.

| method | balacc @1% FPR | AUROC |
|---|---|---|
| Mean Probe (Table 6) | 0.636 | 0.682 |
| Follow-Up Probe (Table 7) | 0.578 | 0.634 |
| Upper-Bound Probe (Table 8, **privileged**) | 0.890 | 0.930 |
| **persona axis, extended, L40 (zero-shot)** | **0.694** | **0.899** |
| pooled leave-one-out, L40 naive | 0.632 | 0.820 |
| pooled leave-one-out, L40 balanced | 0.678 | 0.765 |

The zero-shot extended axis beats both published non-privileged probes on both metrics, and comes
within 0.03 AUROC of the *privileged* upper bound, which trains on in-dataset examples.

## What this supports

The defensible claim is about **robustness, not peak accuracy**:

> A fixed, label-free persona direction never fails catastrophically on an unseen dataset, while
> probes trained on pooled labelled data do. Across six held-out Liars' Bench datasets and two
> depths, the persona axis's floor is 0.64 and 0.71; pooled supervision falls to 0.12 and 0.49.
> Its generalization gap is zero by construction, because it has no training set.

What it does **not** support is "persona directions are simply more accurate". Excluding the one
dataset they were authored to handle, pooled supervision is as good or better.

## Files

`loo.csv` (tidy, per arm × weighting × dataset), `loo_summary.csv` (means, excl-HP-KR, worst
dataset), `loo_bootstrap.csv` (both weightings), `single_source_reference.csv`,
`liars_bench_comparison_loo.csv`, `loo_scores.npz` (raw held-out scores, so further comparisons
need no refit), `loo.png`.

Reproduce:

```
python deception_detection/scripts/persona_ood.py loo --layer 20 \
  --pooled_dir activations/pooled --out_dir results/loo_L20 \
  --vectors "persona including knowlede denial"
python deception_detection/scripts/persona_ood_figures.py loo --out_dir results/loo_L20
```
