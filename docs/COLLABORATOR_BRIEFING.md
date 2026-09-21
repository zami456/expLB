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
question is: **does the contrast still beat a pooled probe?**

**That experiment has now been run, on your six.** Same six datasets, same layers:

| comparison at L40 | margin |
|---|---:|
| 0.916 vs single-source 0.732 *(the abstract)* | **+0.184** |
| 0.916 vs pooled LODO 0.868 | +0.048 |
| **0.916 vs pooled, row-budget matched 0.893** | **+0.023** |

The headline survives, but **the margin goes from +0.184 to +0.023** — and that is against *your*
0.916. On our own grid the same contrast on the same six at L40 reads **0.811**, against pooled
row-matched at **0.893**, so like-for-like on our data the persona contrast **loses by 0.082**.

**How weak "0.732" is turns out to be measurable.** Across our 110 source→target pairs, **21 fall
below 0.5** and 14 below 0.4 (minimum 0.055). Averaging those in is what makes the single-source
baseline so low. Three alternative baselines on identical pairs:

| single-source baseline | AUROC |
|---|---:|
| mean over sources *(what the abstract compares against)* | 0.676 |
| median over sources | 0.722 |
| sign-oriented mean, max(a, 1−a) | 0.735 |
| best source per target *(oracle — needs test labels)* | **0.897** |

A reviewer will ask why the mean and not the median. Worth pre-empting.

### (c) One direct contradiction to check

The abstract says the first principal component scores **0.09 below** the contrast at layer 40.
On our grid PC1 is slightly **above** it: 0.829 against 0.811.

Same claim, opposite sign. Probably a difference in which bank or which PCA basis — but it is the
kind of thing that gets found in review, and it is better resolved now. Our wider finding about
the PCs is *consistent* with the abstract's point (the components are not doing the work), so
this is a detail inside an agreement, not a disagreement.

---

## 5b. Why our numbers differ from yours — and one comparison it puts at risk

You used our persona vectors but scored them yourself: downloading the full per-layer activations
(~50 GB per layer) and taking dot products, where our pipeline projects **mean-pooled**
activations through the crossgen harness. That is not a discrepancy to resolve — it is two
different protocols, and it probably explains the whole gap.

**We measured the size of that difference directly.** On Pass C (2,000 label-stratified rows,
304,706 response tokens, six layers), projecting onto the same axis and changing **only the token
aggregation**:

| layer | mean | last | max | **top-10%** |
|---|---:|---:|---:|---:|
| 20 | 0.695 | 0.741 | 0.775 | **0.798** |
| 22 | 0.699 | 0.743 | 0.740 | 0.763 |

**Token aggregation alone is worth up to +0.103.** Add the six-dataset subset effect (+0.079,
section 5a) and 0.811 → 0.916 is fully accounted for without anyone being wrong.

**But there is a comparison this puts at risk, and it is the headline one.** Our Q8 measurement
has a second half that points the *other* way:

> zero-shot, **top-decile token scoring beats mean pooling (+0.053)**; trained, **pooled-mean
> probes beat token-trained probes at every layer (−0.03 to −0.06)**.

So the two aggregations favour opposite methods. **If the zero-shot contrast is scored per-token
(or top-decile, or max) while the trained baseline is a probe on mean-pooled activations, part of
the 0.916-vs-0.732 gap is the aggregation choice rather than the feature.** On our numbers that
part could be 0.05–0.10 of it.

This is straightforward to close and worth doing before submission: **score both arms the same
way.** Either mean-pool both, or token-score both. If the contrast still wins under matched
aggregation, the result is clean and the objection disappears. If the gap shrinks substantially,
that belongs in the paper rather than in a reviewer's report.

Practical note: you should not need the 50 GB per-layer downloads again. `run-cap/` and
`run-coords/` hold the pooled activations, and the crossgen harness scores an arbitrary axis
against all eleven datasets in minutes. Happy to run your exact contrast through it so both
protocols sit side by side on the same rows.

## 5c. Three papers that landed today, and what each one does to the draft

All three are now read in full, not from summaries.

**Kumar, *Pressure-Testing Deception Probes in LLMs* (arXiv:2605.27958, GEM @ ACL 2026) — this
one needs a response in the paper.** Gemma 3, 1B–27B. From his abstract:

> "the single-direction hypothesis is **decisively rejected** (k=1 captures only 0.61–0.80 AUROC
> of the signal), with cross-domain transfer failure confirmed as geometric rather than
> layer-mismatch-driven"

The draft title is *"One Direction Generalizes"*. A concurrent paper asserts the opposite in its
abstract, and a reviewer who knows it will open with that.

**There is a good answer, and it is ours.** His k=1 is the top principal component of **one**
source. Our F-86 result is an average of **ten** single-source directions — also one direction —
and it reaches **0.883**, beating joint training. The reconciliation is that a single source's
leading direction is dominated by that source's surface features (we measure mean pairwise cosine
between single-source directions at **+0.051**, i.e. near-orthogonal), while the average across
sources is not. **Both papers are right; they are about different single directions.** Said that
way it strengthens the title rather than undermining it — but it has to be said.

Two more from Kumar worth knowing: his **style-augmented** probes recover 0.979–0.983 on unseen
styles, which is the same "diversify the training distribution" move as pooled training but over
styles rather than datasets, published May 2026. And his peak-layer confidence intervals "span
nearly the full network, indicating weak localization" — independently reproducing our own layer
finding on a different model family.

**Kirch, Dower, Skapars et al., *The Impact of Off-Policy Training Data* (arXiv:2511.17408,
Findings of ACL 2026).** Note the author list — **Skapars co-authors this and the paper we are
replicating.** Two things:

- **Text-evident vs text-ambiguous.** Off-policy training data is fine for behaviours
  identifiable from surface properties and harmful for behaviours where the same text can come
  from different underlying processes, with the **largest failures for intent-defined behaviours
  like strategic deception**. If any part of our pipeline uses prefilled or synthetic turns, this
  is the objection, and it should be pre-empted in the text.
- Their Appendix N fits **mixed-effects models with per-behaviour random effects**, because
  "ignoring this clustering inflates Type I error rates". Any claim we make about clustered
  statistics being absent from this subfield is wrong, and we should cite them instead.

**Yoo & Skapars, *Probe Generalization as Subspace Selection* (arXiv:2609.02893, ICML 2026
workshop).** Also Skapars. Convergent rather than threatening:

> "The directions a source probe weights heavily appear to encode **source-specific surface
> features**, while the directions that actually transfer appear to encode the same contrast more
> abstractly"

That is our near-orthogonality result in their vocabulary. They recover the transferable subspace
by *selecting* principal components with an LLM judge; we recover it by *averaging* across
sources — no judge, no target labels, no interpretation step. Worth citing as independent support
for the same underlying picture, with our method as the simpler route.

## 5d. An external review of the pooled result, and the one number that reframes everything

We sent the pooled-training result out for adversarial review. Three of the five challenges
landed. All of this is measured, in one process on one set of splits, reproducing the stored
baseline bit-for-bit.

**The finding that matters most to your abstract.** The reviewer asked whether we had ever run
the *persona method itself* through the same leave-one-dataset-out setup we were using for the
pooled probe. We had not — every persona number in this project, and in the paper we are
replicating, is train-on-one / test-on-others. Run like-for-like:

| feature (trained, LODO) | single-source | pooled LODO | vs raw pooled | 95% CI |
|---|---:|---:|---:|---|
| **raw 8192-d activations** | 0.676 | **0.854** | — | |
| `axes` | 0.679 | 0.745 | −0.109 | [−0.187, −0.039] |
| `pc:deception` | 0.621 | 0.731 | −0.124 | [−0.203, −0.055] |
| `pc:unified@2` | 0.574 | 0.692 | −0.162 | [−0.288, −0.059] |

**All three persona feature sets are significantly worse than ordinary logistic regression on raw
activations, once both are given the same ten training datasets.** Pooling helps the persona
features too (`pc:unified@2` gains +0.118), so diversity is not something personas provide.

This does *not* touch your zero-shot contrast, which has no training set and is a different
object. But it does mean the trained half of the persona-coordinates method loses to a baseline
the original paper never ran.

**Two corrections to our own numbers, which we found by being asked.**

- We had quoted the budget-matched pooled probe as 0.850 in one place and 0.806 in another. They
  are **the same estimator on different random row draws** — one script consumes its RNG
  differently. Over 20 draws: **0.834, range [0.791, 0.874], sd 0.020**. Both published figures
  were single draws reported as if they were the quantity.
- Our secondary "mean-of-directions" result is **+0.028, CI [+0.005, +0.052]**. It clears, but
  the lower bound is +0.005. It is not a headline.

**What we are dropping.** The claim that our pooled result is a *new mechanism*. The reviewer's
framing is better and we are adopting it: the contribution is **"the replicated paper omitted a
baseline that beats its trained method"**. That is a smaller claim than a mechanism and a much
harder one to argue with.

---

## 5e. Layer 40 is not the peak — which affects every number above

We had used layer 40 throughout, following the literature. Re-reading the 21-layer matrices we
already hold:

| bank / feature | peak layer | peak | at L40 | difference |
|---|---:|---:|---:|---:|
| `cap_s23` / raw | **32** | 0.682 | 0.662 | +0.020 |
| `cap_s23` / `pc:capitulation@2` | **32** | 0.780 | 0.761 | +0.019 |
| `kd_s23` / `axes` | 44 | 0.644 | 0.630 | +0.013 |

An independent sweep of the pooled probe agrees, peaking at layers 32–36 rather than 40. The
differences are small (~0.02) and do not reverse any ordering we have reported, but **every L40
figure in this document is therefore a slight understatement**, for your method and ours alike.
If the abstract reports L20 and L40, it is worth checking L32 before submitting — the activations
are already extracted, so it costs nothing but CPU.

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
| Pooled training on **your six datasets** | **done** — margin falls from +0.184 to +0.023 (§5b) |
| Persona method through the same LODO harness | **done** — loses to raw pooled LR by 0.11–0.16 (§5d) |
| Format-matched persona control | **done, and mostly negative** — see below |
| Layer-32 check on the key numbers | running; activations already held, CPU only |
| Sarcasm bank | design v2 written and externally reviewed; **elicitation pilot generating now** |
| Clustered hold-out for shared provenance | **not run** — the open methodological risk (§5d) |
| Resolve the PC1 sign discrepancy | still needs your setup details |

**The format-matched control, because it qualifies Claim 2's story.** We had found that the six
"position-holding" personas on the negative side of the capitulation contrast were load-bearing
even though they produce text almost identical to no system prompt — and read that as *the model
represents the disposition even when it does not act on it*. We then swapped those six for six
personas about **alphabetising lists, citing sources, using metric units, spelling out and
keeping time** — nothing to do with holding a position. Swapping the content costs only
**~+0.03** and clears no significance test at any layer; removing the six entirely costs **two to
four times that**. So the persona side is contributing mostly *an instruction is in force*, not
*this particular disposition*. Claim 2's measurement stands; that interpretation of it does not.

**The framing I would suggest.** Not *"persona coordinates don't work"* — Claim 2 shows a
construct-matched bank does, robustly. And not *"one direction generalizes"* alone, because the
comparison behind it is against a baseline we can now beat without personas.

Something closer to: **persona axes fail when the personas vary register rather than the labelled
construct, work when they vary the construct, and the transfer problem that motivates the whole
method is better solved by training on more than one dataset.** All three parts are measured and
all three survive the dataset-level bootstrap.

**One revision to that, after the review in §5d.** The third part should be stated as a claim
about the *literature*, not about a mechanism of ours: in a like-for-like leave-one-dataset-out
comparison, the trained persona-coordinates method loses to ordinary logistic regression on raw
activations by 0.11–0.16 AUROC. Related results already exist — Ying et al. on truth types, and
work on style-augmented probes — so the diversity effect itself is not new. **What is ours is
having run the comparison the method's own paper omits, on 70B, with the controls.** That is a
narrower claim than we were making this morning, and a considerably more defensible one.

Your zero-shot contrast is untouched by all of this: it has no training set, so none of the
pooled comparisons apply to it directly. Its exposure is the baseline question in §5b — whether
0.732 is the number a reviewer will accept as "the trained probe".

---

*Every number above is in `kd/docs/FINDINGS.md` with its source and caveats. Persona prompts and
questions: `docs/PERSONA_CONSTRUCTION.md`. The layer decision rule, written before the numbers
existed: `docs/LAYER_PREREGISTRATION.md`. What is novel versus already published:
`docs/NOVELTY_ASSESSMENT.md`.*
