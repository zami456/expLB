# Where the project stands — a briefing

Written 2026-09-21 for a collaborator who has not seen the last day of work. Self-contained:
you should not need any other file to follow it.

> **Revised 2026-09-22.** §5 has been rewritten. We had been comparing against the wrong axis:
> the abstract's 0.904 / 0.916 is `ext_axis1`, the contrast built on the knowledge-denial
> personas, not the persona-coordinates 8v7 contrast. Reading your branch
> `feat/persona-pc-ood` (head `f386a72`) settled it, and it invalidated three things we told you
> — the aggregation theory in §5b, the "same six" claim, and the PC1 contradiction in §5c. All
> three are corrected below, with the file and row for every number so you can check them
> yourself. Ledger entry: **F-97** in `kd/docs/FINDINGS.md`.
>
> **Further, later the same day:** §3's Claim 3 is corrected (a row-level leak; quote +0.12–0.16,
> not +0.178), §5d's open methodological risk is **closed**, §5e now has L32 measured on your
> six, and §6 records that the sarcasm bank's second gate also failed. Ledger: **F-107 → F-109**.

Short version: **we have three claims that survive their own significance tests, four that did
not and were withdrawn — three of them ours about your work — and one finding that changes what
the paper should lead with.** That last one is §5a: the abstract's headline number comes from an
axis built on the knowledge-denial personas, not from persona coordinates, and roughly half the
margin is attributable to that bank. It is now measured twice, on your six datasets and on our
eleven, and it is the reason for this revision.

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

**It scores 0.313** on our capitulation dataset (95% CI [0.283, 0.346], zero-shot, L40 mean).
Below chance — and **significantly** so: against 1,000 random directions it is beyond the null at
p 0.001–0.008. The same axis family scores **0.300** on the knowledge-report dataset.

> **Correction, 2026-09-23.** Earlier versions of this briefing put **0.374** here and called it
> the sycophancy axis. That number is the **deception** axis's **PC1**, not the sycophancy axis.
> `27_crossgen.py:411-416` sets `_zs_axis = "deception"` whenever a bank has a deception axis, so a
> paper16 matrix's bare `contrast`/`PC1` rows are the *deception* axis; every other axis is written
> `contrast:<name>`. In `run-crossgen/kd11/matrices/mean_L40_s42.json` the `sycophancy_mt` column
> reads `contrast` 0.4089 and `PC1` 0.3739. The sycophancy axis's own value, 0.313, comes from
> `24_q10_sycophancy_axis.py` (2026-09-17) and re-measures at 0.2996–0.3061 on `passA11`/`passC11`.
> The correction **strengthens this claim**: the deception axis at L40 sits *within* the random-direction
> null, whereas the sycophancy axis is significantly inverted. Note also that a **zero-shot** axis must
> not be compared against a **trained** random probe (the 0.541 previously quoted here); the matched
> trained pair is `axes` 0.3551 vs `random` 0.5413.

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
| register axis (paper16 **deception** axis, PC1), zero-shot | 0.374 |
| sycophancy axis, zero-shot | 0.313 |
| **capitulation axis, zero-shot** | **0.745** |
| trained, over raw activations | **+0.112**, 95% CI [+0.050, +0.185] |
| positive at | **21 of 21 layers**, all clearing the bootstrap |
| filtered vs unfiltered judge | `pc:capitulation@2` differs by **0.0004**; `pc:cap_only@2` by **0.0142** |

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

> **Corrected 2026-09-22 — quote +0.12 to +0.16, not +0.178.** We have since run the clustered
> hold-out that §5d listed as the open risk, and found a **row-level leak** on the way:
> `harm-pressure-kr-matched` is a *view* over `harm-pressure-knowledge-report` — 258 rows, and
> **100% of its row_ids are HP-KR row_ids** — so two of the eleven leave-one-dataset-out cells
> are train-on-test. Under **leave-one-CLUSTER-out**, where every dataset sharing the target's
> provenance is removed from training, the gain is **+0.129 (L20), +0.131 (L32), +0.125 (L40)**
> over all eleven, and **+0.157 / +0.117 / +0.150** over the nine uncontaminated targets —
> positive on **10/11, 11/11 and 10/11** over all eleven, and **9/9, 9/9 and 8/9** over the clean
> nine (the earlier "8–11 of 11" mixed the two denominators; nothing is 8 of 11). **The claim survives the strictest hold-out we can
> construct; the magnitude was inflated.** Ledger: F-108.

The third row is the control that matters. Pooled training with ~6,700 rows (mean `n_lodo`
6,669) and pooled training held to a *single-source* row budget are statistically
indistinguishable. **The gain is diversity, not data volume.**

> **Correction, 2026-09-23.** "~666 rows" is the **cap** in `30_premise.py:167`, not the realised
> pool: the budget-matched LODO pool is **412–500 rows (mean 449)**. And the quoted **p = 0.38** is a
> **one-sided** tail (`30b_premise_boot.py:62`); the two-sided bootstrap gives **+0.0045, 95% CI
> [−0.024, +0.033], p = 0.744**. Quote the interval rather than the p-value — it makes the
> equivalence claim properly, which a one-sided p does not.

~~On `harm-pressure-knowledge-report` pooled training reaches **0.972**.~~ **Withdrawn
2026-09-22.** That number is substantially the leak described above: HP-KR reads **0.9674 under
leave-one-dataset-out and 0.6492 under leave-one-cluster-out** at L40 — a 0.318 collapse the
moment its WMDP siblings leave the training set. Please do not quote it.

The other number stands. On our capitulation dataset pooled reaches **0.802** against the
capitulation bank's 0.714 — that pair shares no provenance, so the leak does not touch it: the
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
at **0.648 / 0.732**, on six datasets at layers 20 and 40.

> **Correction, 2026-09-22.** An earlier version of this section made three claims that were
> wrong, and they were wrong because we had not read your branch. We have now read
> `feat/persona-pc-ood` (head `f386a72`) file by file. What follows replaces §5(a), §5(b) and
> §5(c) of that version. The headline is not wrong; it is *about a different axis than we
> assumed*, and that turns out to matter more than any of our original three points.

### (a) 0.904 / 0.916 is the **extended** axis, not the persona-coordinates contrast

Those two numbers are the `ext_axis1` row of
`results/persona_ood_ext_L20/summary.csv` and `results/persona_ood_ext_L40/summary.csv`
(`off_diagonal_mean` = 0.9041173922 and 0.9158352735; identical to the mean of `auroc_ext_axis`
in the matching `zero_shot.csv`, to 1e-15 — the exact figure is 2.2e-16 at L20 and 0.0 at L40).
`ext_axis1` is the **12-vs-13** contrast defined in
`deception_detection/scripts/persona_ood.py::extended_arms()`, with the axis itself in
`deception_detection/persona_pcs.py` (`axes["extended"]`), as

```
positive = DECEPTION_PERSONAS + KNOWLEDGE_DENIAL_PERSONAS      # 8 + 4
negative = HONESTY_PERSONAS   + KNOWLEDGE_CANDOUR_PERSONAS     # 7 + 6
```

The ten extra roles are the knowledge-denial bank we sent you — they match
`kd/personas/kd_role_groups.json` name for name. **One caveat on the grouping:** your
`KNOWLEDGE_CANDOUR_PERSONAS` merges our deliberately separate `sincere_ignorance` (3) and
`plain_assertion` (3) groups into one arm. We split them precisely to remove a wording confound;
merging them reintroduces it. The persona-coordinates contrast as the paper
defines it is the 8-vs-7 `axis1`, which sits in the same files, on the same rows, over the same
six datasets. Your base bank is `paper16_L20_sharedquestions/paper16_L20.pt`
(`deception_detection/persona_pcs.py:31`), so `axis1` is exactly the axis the replication is of.

**Zero-shot, your six, your files:**

| | L20 | L40 |
|---|---:|---:|
| trained single-source baseline *(`summary.csv`, row `baseline`)* | 0.6478 | 0.7320 |
| **paper's 8v7 contrast** *(`zero_shot.csv`, `auroc_axis`)* | **0.7909** | **0.8319** |
| **extended 12v13 contrast** *(`auroc_ext_axis`)* — the abstract's number | **0.9041** | **0.9158** |
| margin, paper's contrast vs baseline | **+0.143** | **+0.100** |
| margin, extended contrast vs baseline | +0.256 | +0.184 |
| **what the knowledge-denial personas are worth** | **+0.113** | **+0.084** |
| **share of the headline margin from the kd bank** | **44%** | **46%** |

Drop `repe_honesty` — which your own `persona_ood/summary.md` flags as in-distribution for the
deployed probe, "its diagonal and its column say nothing about transfer" — and it holds: the
paper's contrast leads by +0.142 (L20) / +0.092 (L40), and the kd personas are worth
+0.136 / +0.101.

**So the result is real, and roughly half of it is the knowledge-denial bank.** That is worth
saying plainly in the paper, because a reviewer who diffs the two axes will find it, and because
it is a *better* story than the one the abstract currently tells: a construct-matched bank fixes
the dataset that breaks every other detector. It is the same shape as our Claim 2.

**We then rebuilt both of your arms from our own persona vectors and ran them on our eleven
datasets**, to check this is not an artifact of your corpus. It replicates:

| zero-shot, our eleven | L20 | L40 |
|---|---:|---:|
| our raw single-source baseline | 0.6289 | 0.6623 |
| `axis1` (8v7) | 0.6572 | 0.7325 |
| `ext_axis1` (12v13) | 0.7386 | 0.7996 |
| pooled leave-one-dataset-out | **0.8027** | **0.8545** |
| **share of the margin from the kd bank** | **74%** | **49%** |

Two things follow. **Both of your arms lose to a pooled probe** — the **extended** arm by 0.064
at L20 and 0.055 at L40, and `axis1` by **0.146 and 0.122** — which is §5d's point, now tested against the abstract's actual axis rather than a proxy.
And the kd contribution is **not** a general improvement: at L20 it is entirely
`harm-pressure-knowledge-report` (0.165 → 0.806, a +0.641 swing) plus `sandbagging_v2`; remove
those two and the kd personas are worth **+0.003**. At L40 it is more distributed (+0.039 with
both removed). A construct-matched bank is a targeted instrument — the same conclusion Claim 2
reached from the other direction.

**Swept across all 21 layers**, because two layers is not enough to quote a margin from:
`ext_axis1` beats `axis1` at **21 of 21 depths**, mean **+0.055**, range [+0.002, +0.081], peaking
at L20 — the layer your abstract leads with. So the effect is not a layer artifact.

**And swept under both poolings**, because we found elsewhere in this project that mean-vs-last
pooling can reverse a conclusion outright. This one is robust to it: the kd bank is worth
**+0.055 mean-pooled and +0.059 last-token**, positive at **21/21** and **20/21** layers
respectively (the single exception is L0, the embedding layer, where nothing is claimed to work).
The mechanism matters — both quantities being differenced are persona contrasts over the same
activations, so a shift in the representation affects them almost equally and cancels. Comparisons
between a persona feature and a *baseline of a different kind* are not so lucky, which is why the
share below needs its pooling named.

One honest caveat on how to phrase it: the *share* of the margin is unstable, because its
denominator shrinks wherever the persona arms sit near the raw baseline. **Mean-pooled** it reads
2%–114% across depth with a median of **63%**; **last-token** it ranges from **−156% to +145%**
with a median of **50%** — which makes the instability point more strongly, while the underlying
quantity barely moves (+0.055 vs +0.059). Above 100% is real rather than a glitch: at L60–L68
your `axis1` falls *below* raw activations, so the kd bank supplies more than the whole margin
there. **Quote the raw difference (+0.055–0.059, or +0.081 at L20 mean-pooled), and give a share
only with both the layer and the pooling named.**

The reason it is so robust is that the HP-KR repair holds at every depth: `axis1` is inverted or
near-chance on that dataset at essentially every layer (0.087–0.86, mostly 0.15–0.57) while
`ext_axis1` reads 0.57–0.91 throughout. We had previously believed that inversion was specific to
layer 20; it is not.

Sanity check on provenance: our `axis1` built from a different extraction seed reads 0.666 /
0.729 against 0.657 / 0.733, so the arm does not depend on which rollout set it came from.

### (b) The trained baseline is the weakest one available

0.732 is a **single-source** probe — trained on one dataset, tested on another. That is the
baseline the original paper uses, so matching it is fair. But we now know a probe trained on
*ten* datasets and tested on a held-out eleventh reaches **0.854**, with no personas, no PCA, no
contrast design, and **no additional rows**.

So the abstract's comparison is against the weakest trained baseline that exists. The honest
question is: **does the contrast still beat a pooled probe?**

**That experiment has been run — but not on your six, and we said otherwise.** Our corpus has no
`repe_honesty`, so at most five of your six overlap ours; the sixth is a substitution. The
numbers below are a pooled probe on **our** six, and the cross-comparison against your 0.916 is
therefore indicative, not like-for-like:

| comparison at L40 | margin | |
|---|---:|---|
| 0.916 vs single-source 0.732 *(the abstract)* | +0.184 | both yours, like-for-like |
| 0.916 vs our pooled LODO 0.868 | +0.048 | **mixed corpora** |
| 0.916 vs our pooled, row-budget matched 0.893 | +0.023 | **mixed corpora** |

We previously reported the +0.023 as the headline and said our own contrast "loses by 0.082" on
the same six. Both statements mixed two dataset sets and two banks — the 0.811 we quoted is our
*knowledge-denial* bank, not a register axis, so it was never the comparator we called it. Those
claims are withdrawn.

**What replaces them is stronger, and lives entirely inside your files.** The paper's own 8v7
contrast beats your own single-source baseline by **+0.143 (L20)** and **+0.100 (L40)** — same
matrices, same rows, same protocol, nothing of ours involved. That is the number the replication
is entitled to claim. The gap between it and 0.904 / 0.916 is the knowledge-denial bank (§5a).

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

### (c) The PC1 "contradiction" was ours, and it dissolves

We previously flagged a direct contradiction: the abstract saying PC1 sits **0.09 below** the
contrast at layer 40, against our grid putting PC1 slightly above. There is no contradiction,
because there is no single PC1-vs-contrast number on your side — it depends entirely on which
bank, which is the same confusion as §5a:

| at L40 (`persona_ood_ext_L40/summary.csv`) | PC1 | contrast | PC1 − contrast |
|---|---:|---:|---:|
| paper bank (8v7) | `pc1` 0.8593 | `axis1` 0.8319 | **+0.027 (above)** |
| extended bank (12v13) | `ext_pc1` 0.7680 | `ext_axis1` 0.9158 | **−0.148 (below)** |

Neither is −0.09, and we could not reproduce that figure from any arm at any of your three
layers. If it is in the draft, it needs a source. The underlying agreement stands and is
strengthened: **the components are not doing the work** — at L20 `ext_pc1` trails `ext_axis1` by
0.243, and the 1-d contrast beats the 3-d subspace at every depth you ran.

---

## 5b. Why our numbers differ from yours — the aggregation theory was wrong

**We got this section wrong and it is worth stating plainly.** The previous version asserted that
you scored our persona vectors per-token from ~50 GB per-layer activation downloads, while our
pipeline mean-pools — and concluded that "token aggregation alone is worth up to +0.103" and that
"0.811 → 0.916 is fully accounted for without anyone being wrong."

**You mean-pool.** `scripts/pool_activations.py:84` is

```python
pooled.append(acts[row_mask].float().mean(0).squeeze(0).numpy())
```

over the detection mask — its module docstring says so outright ("the mean over the model's
**response** tokens"), and every one of your summaries records `Δ = 0.0000` reproducing that
run's own `detector.pt` from the pooled matrix. The 50 GB was your Liars' Bench activation cache
(`ACTS_CACHE_DIR=E:/liars-bench-acts/cache`), not per-token scoring.

So both arms of your comparison — the zero-shot contrast and the trained baseline — read the
same pooled vectors. **The aggregation asymmetry we called "a comparison at risk" does not
exist**, and the pre-submission task we recommended ("score both arms the same way") is already
satisfied. Our Q8 measurement about top-decile vs mean pooling is a fact about *our* pipeline and
says nothing about yours.

The actual reason our numbers differ from yours is §5a: we were comparing a register axis against
your **kd-augmented** axis, and separately our corpus is not yours — no `repe_honesty`, plus five
datasets you do not have. Those two account for the gap; token aggregation accounts for none of it.

Practical note stands, minus the premise: `run-cap/` and `run-coords/` hold pooled activations,
and the crossgen harness scores an arbitrary axis against all eleven of our datasets in minutes.
Happy to run your exact `axis1` and `ext_axis1` through it so both banks sit side by side on the
same rows — that is the one comparison neither of us has.

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

**THE OPEN RISK IS NOW CLOSED (2026-09-22).** This section has listed "clustered hold-out for
shared provenance" as not run. It is run. Two findings:

1. **A row-level leak.** `harm-pressure-kr-matched` is a *view* over
   `harm-pressure-knowledge-report` — 258 rows, **100% of its row_ids are HP-KR row_ids** — so
   two of the eleven LODO cells are train-on-test. No other pair shares a single row_id.
2. **The claim survives anyway.** Under leave-one-cluster-out (provenance clusters: the WMDP
   harm-pressure trio, the Liars' Bench scenarios, Apollo's three, MASK, ours), the gain is
   **+0.129 / +0.131 / +0.125** at L20 / L32 / L40 over all eleven, and **+0.157 / +0.117 /
   +0.150** over the nine uncontaminated targets, positive on 10/11, 11/11 and 10/11 over
   all eleven and 9/9, 9/9 and 8/9 over the clean nine.

So the honest number is **+0.12 to +0.16**, not +0.178, and the HP-KR 0.972 in §3 is withdrawn.
Ledger: F-108.

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

**Now measured on your six (2026-09-22).** The pooled probe at L32 against L40:

| | single-source | pooled LODO | pooled LODO, row-matched |
|---|---:|---:|---:|
| L32 | **0.7203** | **0.8965** | **0.9010** |
| L40 | 0.6960 | 0.8683 | 0.8930 |

L32 wins on all three arms. So §5b's comparison understates the pooled baseline: against 0.916
the row-matched pooled probe reads **0.9010 at L32**, cutting that margin from +0.023 to
**+0.015**. If the abstract reports L20 and L40, it is worth reporting L32 — the activations are
already extracted, so it costs nothing but CPU. Ledger: F-109.

---

## 6. Where sarcasm fits now — the pilot ran, and it failed its own gating check

We had been planning a sarcasm persona bank. **It is now measured, and the on-policy design is
not viable.** Reporting it here because the previous version of this document recommended it.

**Why sarcasm was the right idea.** Sarcasm is *saying something you don't mean, without
intending to mislead* — deception minus the intent. A sarcasm axis therefore tests directly
whether our "deception" detectors track intent or merely surface/meaning mismatch. It would have
been the first **positive** test of Claim 1: every test so far shows the axis *failing* where
register and construct come apart; a sarcasm bank would build the confound deliberately and show
it explains the failure.

**The design required one thing:** with markers stripped, ironic and sycophantic text had to be
near-indistinguishable — both literally praise a bad plan, and the difference is supposed to be
purely pragmatic. The pre-registered reading was *"near 0.5 with markers removed = the arms are
textually alike; near 1.0 = a lexical detector."*

**It came out at 0.92.** Bag-of-words logistic regression, grouped by question id so no item
spans train and test:

| variant subset | ironic vs sycophant | raw | markers stripped |
|---|---|---:|---:|
| all v1–v5 | | 0.953 | **0.947** |
| **v4+v5 (marker rate 0.00)** | | **0.919** | **0.921** |

Deadpan *is* producible — the marker rate genuinely falls to 0.00 at variants v4 and v5, which
was the encouraging half. But stripping the markers does not make the arms textually alike: they
remain separable at 0.92 by word counts alone. **A bank built on this would be a register axis,
which is precisely the thing it was designed to diagnose** — the same failure mode that forced
the knowledge-denial bank into three groups rather than two.

**And the second gate has now failed too (2026-09-22).** The design's other pre-test asks whether
our *existing* deception axes fire on marked natural sarcasm at all — if they do not, the
"deception axis is really a mismatch detector" hypothesis is weak before any bank is built. Run
on 600 length-matched FigLang-2020 Reddit rows (author-labelled, so *intended* sarcasm), 70B
activations, axes projected zero-shot:

| | mean L40 | last L40 | random p95 | random max |
|---|---:|---:|---:|---:|
| **deception axis** | **0.543** | **0.586** | ~0.58–0.59 | ~0.63–0.64 |
| **sycophancy axis** | **0.693** | **0.710** | | |
| assistant axis | 0.428 | 0.411 | | |

**The deception axis sits inside the random-direction distribution** and never reaches the
bag-of-words floor of 0.632. It does not fire. The pipeline is not null — the assistant axis is
strongly *inverted*, so these activations carry real structure.

**What fires is sycophancy, and that is the point.** Ironic criticism's literal surface is
*praise*, so a direction tracking agreeable, flattering register responds to it — zero-shot,
beating a cross-validated bag-of-words model — while the direction meant to track insincere
*intent* does not. **The axes respond to what the text sounds like, not to what the speaker
means.** That is Claim 1 reached from a third independent route. Ledger: F-107.

**So: not first, and not in this form.** Both gates have failed — the arms cannot be produced
without a register confound, *and* the hypothesis that motivated the bank does not survive its own
cheap pre-test. The concept is interesting; we are not building it.

---

## 7. What we do next

| | status |
|---|---|
| Identify which axis the abstract reports | **done — it is the kd-augmented 12v13 axis, not the paper's 8v7 (§5a)** |
| Both your arms through our crossgen, our eleven | **done** — kd bank is 49–74% of the margin; both arms lose to pooled (§5a) |
| Pooled training vs the abstract | **done, then withdrawn as a headline** — our six ≠ your six (§5b) |
| Persona method through the same LODO harness | **done** — loses to raw pooled LR by 0.11–0.16 (§5d) |
| Format-matched persona control | **done, and mostly negative** — see below |
| Layer-32 check on the key numbers | **done, and now run on your six** — L32 beats L40 on all three arms (§5e) |
| Sarcasm bank | **both gates failed** — 0.92 lexically separable, and the deception axis does not fire on natural sarcasm (§6) |
| Clustered hold-out for shared provenance | **not run** — the open methodological risk (§5d) |
| Resolve the PC1 sign discrepancy | **resolved — there was no discrepancy (§5c)**; we still need a source for the "0.09 below" figure |

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
pooled comparisons apply to it directly. Its two real exposures are **§5a** — that roughly half
its margin over the baseline comes from the knowledge-denial personas rather than from persona
coordinates — and the baseline question: whether 0.732, a single-source mean, is the number a
reviewer will accept as "the trained probe" when the median over the same pairs is 0.722 and the
sign-oriented mean is 0.735.

---

*Every number above is in `kd/docs/FINDINGS.md` with its source and caveats. Persona prompts and
questions: `docs/PERSONA_CONSTRUCTION.md`. The layer decision rule, written before the numbers
existed: `docs/LAYER_PREREGISTRATION.md`. What is novel versus already published:
`docs/NOVELTY_ASSESSMENT.md`.*
