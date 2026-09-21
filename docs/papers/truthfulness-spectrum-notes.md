# Two papers added 2026-09-21, and what they change

Notes on `papers/The Truthfulness Spectrum Hypothesis.pdf` and
`papers/The Truth Stays in the Family.pdf`, read in full. The first is a close neighbour and
changes several of our claims. The second is methodological and offers a candidate explanation
for a failure we could not previously account for.

---

## 1. The Truthfulness Spectrum Hypothesis

**Ying, Ravfogel, Kriegeskorte & Hase** — Columbia / NYU / Stanford / Schmidt Sciences.
**arXiv:2602.20273v1**, 23 Feb 2026. Code at `github.com/zfying/truth_spec`.

**This is the "Ying et al." a reviewer named as the source of Yoo & Skapars' layer-15 choice.**
It is the closest published work to our project that we had not read.

### Their claim

Truth directions lie on a **spectrum** from domain-general to domain-specific, and both ends
exist simultaneously. They test five truth types (definitional, empirical, logical, fictional,
ethical) plus **sycophantic** and **expectation-inverted** lying, plus roleplaying, insider
trading and sandbagging.

### The four results that matter to us

**(a) Probes generalise well across most truth domains but FAIL on sycophantic lying.**
Independent corroboration of our F-13, from a different lab with a different method — they use
plain trained probes, not persona axes, and get the same failure on the same construct. Our
finding is no longer a curiosity of the persona method.

**(b) They give the mechanism we lacked: post-training moves sycophancy away.** Base models
transfer to sycophancy substantially better than their chat counterparts:

| model | base | chat |
|---|---:|---:|
| Llama 70B | 0.67 | 0.56 |
| Llama 8B | 0.69 | 0.56 |
| Llama 3B | 0.74 | 0.52 |

So sycophancy detection gets *worse* with instruction tuning, and the Mahalanobis similarity
between the sycophancy direction and other truth directions drops with it. **We have been
explaining F-13 as "the axis encodes register rather than capitulation". Their explanation is
complementary and deeper: post-training pushes sycophantic lying into a subspace of its own.**
Both can be true; theirs says *why* the construct is separable in the first place.

**(c) THE THREAT: training on all domains jointly recovers strong performance.** Their abstract
is explicit — *"training on all domains jointly recovers strong performance, confirming that
domain-general directions exist despite poor pairwise transfer."* A probe trained on the pooled
corpus generalises to everything, on par with the best individual probe.

This is a **direct challenge to the persona-coordinate premise**. The paper we replicate argues
persona axes are needed because probes trained on one dataset transfer badly. Ying et al. say the
fix is simply *train on more than one dataset*. **We have never tested this.** Our entire
transfer matrix is train-on-one / test-on-others; the pooled-training cell does not exist in any
of our runs.

**(d) Mahalanobis cosine similarity predicts transfer at R² = 0.98** (plain cosine: R² = 0.56).
Each point is a train/test dataset pair. This is a diagnostic we could apply directly to our own
11×11 matrix.

### Their setup, for comparison

Llama 3B / 8B / 70B (and 7B, 14B), residual stream, **layer 33**, logistic regression, **average
token** pooling, 5-fold CV. For the 70B they extract **every 5 layers** to save storage. Layer
selected by cross-domain performance on their FLEED datasets — i.e. selected on the outcome,
which is worth noting given our own pre-registration discussion.

---

## 2. The Truth Stays in the Family

**Choi, Choi, Kwon, Joung, Kim & Lee** — Korea University / Kakao Mobility. **ICML 2026**,
PMLR 306. Code at `github.com/miso-choi/TruthProbe`.

Different construct: **contextual truthfulness / hallucination**, not intentional deception.
Benchmarks are HaluEval, POPE, CHAIR. Not a direct competitor.

### What is useful to us

**(a) Truthfulness is carried by a sparse subset of attention HEADS, not spread across a layer.**
They compute head-level "Truth Scores" and build TruthProbe, a soft-gating scheme that amplifies
context-truthful heads while preserving the rest. They cite prior work showing transformers
"contain only a sparse subset of attention heads that are critical for specific tasks".

**This is a candidate explanation for F-76.** Our pipeline takes the **mean residual-stream
activation over response tokens** at a layer — averaging across all 64 heads. If the signal lives
in a handful of heads, layer-mean pooling dilutes it by roughly the ratio of signal heads to
total, and what survives is exactly the weak, unstable, dataset-sensitive effect F-76 measured.
It would also explain why the effect is *present* as a point estimate but *fails* a bootstrap:
diluted signal plus correlated noise across adjacent layers.

**(b) Truth Scores are preserved within model lineages** across instruction tuning and even
multimodal adaptation, and this tracks attention-head weight preservation — weight drift is
concentrated in early layers. Relevant if we ever compare across fine-tunes; not actionable now.

---

## 3. What we should do about it

| # | action | cost | why |
|---|---|---|---|
| **1** | **Add the pooled-training cell**: train one probe on all 11 datasets jointly, test on each. | CPU, moderate | Ying et al. predict this recovers performance *without* persona coordinates. If it does, the method's motivation weakens sharply. **This is the single most decisive experiment left.** |
| **2** | **Compute Mahalanobis cosine similarity** across our 11×11 and check whether R² ≈ 0.98 replicates | CPU, cheap | Either it reproduces — a strong, citable cross-check of a brand-new result — or it doesn't, which is itself interesting given they used plain probes and we use persona axes. |
| 3 | Cite Ying et al. as the mechanism for F-13 | none | Turns our diagnosis from an assertion into one with a published representational account. |
| 4 | Record head-sparsity as a candidate explanation for F-76 | none | Honest alternative to "the effect isn't there": it may be there and diluted by mean pooling. |
| 5 | Consider a head-level or per-head extraction | expensive | Only if (1) and (2) leave the method alive. |

**On (1) specifically.** This is uncomfortable and should be run anyway. If a pooled probe
matches the persona axes, then the persona-coordinates paper's central motivation — probes don't
transfer, so use persona structure — has a simpler answer that needs no personas at all. That
would be the most important negative result in the thesis, and it costs CPU we already have.

## 4. Citation-hygiene note

Neither paper was in our related work. Ying et al. in particular must be cited: they test the
same construct on the same model family at the same three scales, and they reach the sycophancy
finding independently. A reviewer who knows it and does not see it cited will assume we missed
the closest neighbour — the same exposure a reviewer already flagged for Kumar and Kirch et al.
