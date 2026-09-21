# Where the project stands — a briefing

Written 2026-09-21 for a collaborator who has not seen the last day of work. Self-contained:
you should not need any other file to follow it.

Short version: **we have three claims that survive their own significance tests, three that did
not and were withdrawn, and one new result that changes what the paper should lead with.** That
last one arrived this afternoon from a paper we had not read, and it bears directly on the draft
abstract. It is the reason for this document.

---

## 1. What a persona axis is

Give a model a system prompt that makes it play a role — *"you are a spin doctor"* — ask it a
question, and record the activations while it answers. Do that for many roles. Roles that share a
trait land in a similar region of activation space. Subtract the average of one group from
another — deceptive roles minus honest roles — and you get a **direction**.

That direction is the persona axis, and there are two ways to use it:

- **Zero-shot.** Project a new response's activations onto the axis and threshold. No training,
  no labels.
- **Persona coordinates.** Take the top few principal components of the persona cloud and use
  those *k* numbers as features for a trained probe.

The method is from **Mahadik & Skapars, *Do Linear Probes Generalize Better in Persona
Coordinates?*** (arXiv:2605.09391), which builds on **Lu et al.'s Assistant Axis**. Their
motivating claim: probes trained on one deception dataset transfer badly to another, so you
should use persona structure instead. They test at 3B and 8B and report it works.

We are testing that at **70B**, on eleven datasets, with controls the original does not run.

---

## 2. What we built

| | |
|---|---|
| **Persona banks** | 36 personas in 8 groups, 180 prompt variants, 30 shared questions, **5,400 rollouts** (verified on disk). 26 roles are our wordings of the paper's names; 10 are ours |
| **The capitulation bank** | 13 further roles, built to test one specific hypothesis (below) |
| **Model scales** | Llama 3.2-3B, Llama 3-8B, Llama-3.3-70B-Instruct |
| **Datasets** | 11 — Liars' Bench, Apollo's deception set, MASK, Pacchiardi's AI Liar, and one of ours |
| **Activation stores** | three: all 80 layers on a subsample, 21 layers on the full corpus, per-token on a subsample |

Full persona list, every prompt and every question: `docs/PERSONA_CONSTRUCTION.md`.

---

## 3. The three claims that survive

We hold every headline to a **dataset-level bootstrap** — resample *which datasets you happened
to pick*, not which rows. With ~15,000 rows, row-level intervals are tight by construction and
mean almost nothing; the thing that actually varies between replications is the choice of
datasets, and there are only eleven. It is a demanding bar: of 384 comparisons in one bank, **one**
cleared it; of 4,032 in another, **six**, and four of those were negative controls passing.

### Claim 1 — Persona axes encode *register*, not the construct the dataset labels

The paper's sycophancy axis has the cleanest geometry anything here has produced — cosine 0.993
with its own contrast, permutation p = 0.008. By every internal diagnostic it is a well-formed
axis.

It scores **0.374** on our capitulation dataset. Below chance. **Worse than a random
direction** (0.541). The same axis family scores **0.300** on the knowledge-report dataset.

Our reading: these axes separate *how the model talks* — agreeable vs challenging, hedged vs
blunt — while those two datasets label *what it does*: abandoning a correct answer under
pressure, and denying knowledge it holds. Register and behaviour correlate in ordinary text and
come apart exactly where those datasets live.

**This was independently confirmed today.** Ying, Ravfogel, Kriegeskorte & Hase
(arXiv:2602.20273, Feb 2026) train *ordinary* linear probes — no personas — across five truth
types plus sycophantic lying, on Llama 3B/8B/70B. Probes generalise well across most domains and
**fail on sycophantic lying**. Same construct, same model family, same three scales, a different
method, the same failure. They also give a mechanism we did not have: **post-training pushes
sycophancy into its own subspace**. Base models transfer to it far better than chat models
(70B 0.67 → 0.56, 8B 0.69 → 0.56, 3B 0.74 → 0.52).

So this is no longer a quirk of the persona method. It is a property of the construct.

### Claim 2 — A bank built for the *right* construct works, on its own construct

If the diagnosis is right, building personas that vary **capitulation** instead of register
should fix it. We did: 13 roles, teacher-forced first turn so context is byte-identical across
personas, matched model, judge, grid, seed and layers.

| | |
|---|---:|
| register axis, zero-shot | 0.374 |
| **capitulation axis, zero-shot** | **0.745** |
| trained, over raw activations | **+0.112**, 95% CI [+0.050, +0.185] |
| positive at | **21 of 21 layers**, all clearing the bootstrap |
| filtered vs unfiltered judge | differ by **0.0004** |

It works, it is depth-robust, and the judge filter does not drive it.

**But it is local.** Averaged over all eleven datasets the margin over raw is +0.098 with an
interval containing zero. It wins on the construct it was built for and ties elsewhere. A
construct-matched bank is a targeted instrument, not a general detector.

### Claim 3 — The problem persona axes exist to solve has a simpler answer

This is the new one, and it is the strongest result in the project.

The paper's motivation is *probes trained on one dataset transfer badly, so use persona
structure*. We had accepted the first half and never tested the obvious alternative: **train on
more than one dataset.** Every matrix we had built was train-on-one / test-on-others, because
that is the design the paper uses — and it is the design that makes the paper's argument work.

| condition | trained on | AUROC |
|---|---|---:|
| single-source transfer | one other dataset | **0.676** |
| **pooled, leave-one-dataset-out** | **ten others, never the target** | **0.854** |
| **pooled, same row budget** | **ten others, ~666 rows total** | **0.850** |
| in-domain ceiling | the target's own data | 0.951 |

**+0.178, 95% CI [+0.143, +0.213], winning on 11 of 11 targets.** Positive at **21 of 21
layers**, every one clearing the bootstrap. Stable across five seeds (0.173–0.198).

The third row is the control that matters. Pooled training with ~6,700 rows and pooled training
held to a *single-source* budget of ~666 rows are statistically indistinguishable (p = 0.38).
**The gain is diversity, not data volume.**

Two numbers worth pausing on. On `harm-pressure-knowledge-report` — the dataset every detector
inverts on, the reason half this project exists — pooled training reaches **0.972**. The
inversion is not a property of the dataset; it is a property of training on one source. And on
our capitulation dataset, pooled reaches **0.802** against the capitulation bank's 0.714: the
bank we built to fix that dataset is beaten by pooling datasets we already had.

---

## 4. What did *not* survive

Worth stating plainly, because a paper arguing "apply proper controls and effects vanish" has to
be willing to apply them to itself.

**The knowledge-report matched-slice reversal.** Our strongest positive result for most of the
project: on a slice where every response denies knowledge — wording held constant — the persona
axis read 0.752 against raw's 0.715. Then we tested it three ways:

| test | result |
|---|---|
| margin at the pre-specified layer, dataset bootstrap | CI [−0.069, +0.189] — **does not clear** |
| is the peak layer identified? | 95% interval **[L5, L79]** — 93% of the network |
| does the band survive label permutation? | **p = 0.38** |

Three tests, two different nulls, one answer. It is a point estimate that survives nothing. The
decision rule was written down *before* the numbers existed (`docs/LAYER_PREREGISTRATION.md`),
which is the only reason this reads as a finding rather than a retreat.

**Three other corrections today:** a correlation we cited as a mechanism turned out to be
mathematical coupling (measured −0.768 against a null of −0.746, p = 0.21); our response-length
claim held in aggregate but its per-dataset framing was wrong (median gap 3.7 tokens, and two
quoted figures were simply incorrect); and Ying et al.'s geometry→transfer predictor does not
reproduce here (R² 0.023 against their 0.98) — we checked both covariance conventions, so it is
not an implementation difference on our side.

---

## 5. What this means for the draft abstract

The abstract reports a zero-shot 1-D persona contrast at **0.904 / 0.916** against trained probes
at **0.648 / 0.732**, on six datasets at layers 20 and 40. Three things to resolve before it goes
out — none of which means the headline is wrong, all of which a reviewer will reach for.

### (a) The six-dataset subset is worth about +0.08

On our grid at layer 40, the zero-shot contrast reads:

| | |
|---|---:|
| on the same six datasets | **0.811** |
| on all eleven | **0.733** |
| cost of adding the other five | **−0.079** |

Our absolute numbers differ from the abstract's — different protocol, possibly a different bank —
but the *direction and size of the subset effect* should carry. Applied to 0.916 that lands near
**0.84**. Which is, almost exactly, where pooled training sits.

This is not a coincidence to wave away. We have an older measurement that adding `sycophancy_mt`
alone erases the persona-axes advantage at every layer. The six exclude it.

### (b) The trained baseline is the weakest one available

0.732 is a **single-source** probe — trained on one dataset, tested on another. That is the
baseline the original paper uses, so matching it is fair. But we now know a probe trained on
*ten* datasets and tested on a held-out eleventh reaches **0.854**, with no personas, no PCA, no
contrast design, and **no additional rows**.

So the abstract's comparison is against the weakest trained baseline that exists. The honest
question is: **does the contrast still beat a pooled probe?** At 0.916 vs 0.854 it might — those
numbers are not directly comparable across different dataset sets, and the contrast has a real
structural advantage in having no training set at all.

**That experiment has never been run on your six.** It is cheap — CPU only, on activations we
already hold — and it decides whether the headline stands. I would run it before submitting.

### (c) One direct contradiction to check

The abstract says the first principal component scores **0.09 below** the contrast at layer 40.
On our grid PC1 is slightly **above** it: 0.829 against 0.811.

Same claim, opposite sign. Probably a difference in which bank or which PCA basis — but it is the
kind of thing that gets found in review, and it is better resolved now. Our wider finding about
the PCs is *consistent* with the abstract's point (the components are not doing the work), so
this is a detail inside an agreement, not a disagreement.

---

## 6. Where sarcasm fits now

We have been planning a sarcasm persona bank. It is still worth doing, and its role has changed.

**Why sarcasm specifically.** Sarcasm is *saying something you don't mean, without intending to
mislead* — deception minus the intent. So a sarcasm axis tests directly whether our "deception"
detectors are detecting intent or just surface/meaning mismatch. If a sarcasm direction has high
cosine with the deception axis, that axis is substantially a mismatch detector. Apollo's own
paper concedes their probe partly fires on *"is this scenario deception-related"* rather than on
deceptive intent; this would put a number on it.

**It would be the first *positive* test of Claim 1.** Every test so far is negative — the axis
*fails* where register and construct come apart. A sarcasm bank builds the confound itself and
shows it explains the failure. That is the mirror of what the capitulation bank did.

**The design catch.** A naive sarcasm bank produces obviously-marked text — *"oh, brilliant"* —
and then the axis is a lexical detector, worthless for the same reason the knowledge-denial bank
had to be built as three groups rather than two. To be non-lexical you need sarcastic and sincere
personas saying **near-identical words**, separated by context alone. That is harder to author
than anything we have built so far, and it is the thing that decides whether the result means
anything.

**The honest demotion.** Before today, sarcasm would have been evidence in a paper about whether
persona coordinates work. After Claim 3, it is evidence in a paper about **why they fail** —
supporting a diagnosis, not carrying a headline. Worth doing, but not first.

---

## 7. What we do next

| | status |
|---|---|
| Pooled training on **your six datasets** | not run — **the experiment that settles the abstract** |
| Format-matched persona control | banks built and committed; needs GPU time |
| Sarcasm bank | design not started; authoring is the bottleneck, not compute |
| Resolve the PC1 sign discrepancy | needs your setup details |

**The framing I would suggest.** Not *"persona coordinates don't work"* — Claim 2 shows a
construct-matched bank does, robustly. And not *"one direction generalizes"* alone, because the
comparison behind it is against a baseline we can now beat without personas.

Something closer to: **persona axes fail when the personas vary register rather than the labelled
construct, work when they vary the construct, and the transfer problem that motivates the whole
method is better solved by training on more than one dataset.** All three parts are measured, all
three survive the dataset-level bootstrap, and the third is the one nobody in this project — or
in two rounds of external review — thought to test.

---

*Every number above is in `kd/docs/FINDINGS.md` with its source and caveats. Persona prompts and
questions: `docs/PERSONA_CONSTRUCTION.md`. The layer decision rule, written before the numbers
existed: `docs/LAYER_PREREGISTRATION.md`. What is novel versus already published:
`docs/NOVELTY_ASSESSMENT.md`.*
