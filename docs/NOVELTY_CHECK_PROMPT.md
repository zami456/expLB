# Novelty-check prompt for Claude web chat

Paste everything below the line into a **new Claude.ai conversation with web search enabled**
(and extended thinking on, if available). It is self-contained — it assumes no knowledge of this
project.

If the answer gets cut off, follow up with *"continue from claim N"*.

---

You have web search. I need a rigorous **novelty assessment** of ten empirical claims from an
interpretability thesis, so I know which are genuinely new, which are already published, and
which are re-discoveries of something adjacent under a different name. Please search the
literature properly rather than answering from memory — I need citations I can check.

## Background

The work replicates and extends **Mahadik & Skapars, "Do Linear Probes Generalize Better in
Persona Coordinates?"** The method: write contrastive persona system prompts for a behaviour
(e.g. deceptive vs honest personas), collect model activations while each persona answers a
fixed question set, take activation differences and run PCA to get low-dimensional "persona
axes", then use those axes either **zero-shot** (project activations onto the axis, threshold)
or as **k-dimensional features for a supervised probe**. The claim under test is that probes
trained in these persona coordinates generalize across datasets better than probes on raw
residual-stream activations.

Also relevant: **Lu et al., "The Assistant Axis"** (a persona-space direction capturing how
"Assistant-like" a model is), and the broader literature on persona/steering vectors
(contrastive activation addition, representation engineering), linear probes for deception and
sycophancy, and probe generalization/shortcut-learning.

Setting for all numbers below: Llama-3.3-70B-Instruct, residual stream, ~11 deception/honesty
datasets, probes trained on one dataset and tested on the others (AUROC, off-diagonal mean).

## The ten claims

**1. Persona axes encode conversational REGISTER rather than the construct the dataset labels.**
The persona axes have excellent internal geometry (cos(PC1, own contrast) = 0.993, permutation
p = 0.008) yet score *below chance* on exactly the two datasets whose labels are the construct
they claim to detect: 0.300 on a knowledge-denial dataset and 0.374 on a capitulation/sycophancy
dataset — both **beaten by a random direction** (0.54–0.66). Diagnosis: contrastive persona
prompts mostly vary *how the model talks* (agreeable vs challenging, hedged vs blunt), which
correlates with the target behaviour in ordinary text but comes apart precisely where these
datasets live.

**2. The constructive fix: vary the target construct in the personas, and the axis works — but
only locally.** Rebuilding the persona bank so the personas vary *capitulation* (abandoning a
correct answer under social pressure) instead of register raises the zero-shot AUROC on the
capitulation dataset from **0.374 to 0.745**, and the trained version beats raw activations by
+0.112 (95% CI [+0.050, +0.185]) and beats matched controls. But averaged over all eleven
datasets the margin over raw is +0.098, CI [−0.036, +0.208] — **it does not generalize past the
construct it was built for**.

**3. Behavioural inertness does not imply representational inertness.** In the capitulation bank,
the six "position-holding" personas produce text nearly identical to a bare model with *no system
prompt at all* — behaviourally they do nothing. Yet removing them from the PCA basis destroys the
entire advantage (the axis becomes statistically indistinguishable from raw, p = 0.56), and
contrasting against them rather than against the default assistant is worth +0.045 zero-shot. So
a negative-side persona that changes nothing you can read in the outputs can still carry most of
the geometric signal — **you cannot screen personas by reading their generations**.

**4. Matched low-dimensional controls eliminate most apparent persona-coordinate gains.** The
honest comparison for a k-dimensional persona feature is not raw activations but (a) PCA on the
dataset's own activations truncated to the same k, and (b) a random k-dimensional projection.
A k-dimensional feature sits closer to 0.5 than an 8192-d probe, so it "wins" on exactly the
cells where the high-dimensional baseline transfers badly, even when the directions are
meaningless: a random 2-d projection gains +0.056 where raw < 0.6 and loses 0.44 where raw >
0.85, with corr(raw AUROC, delta) = **−0.76**. Against dataset-PCA at matched k, the reported
persona gain in this replication goes to roughly **−0.003**.

**5. Dataset-level, not row-level, bootstrap is the correct significance unit for cross-dataset
transfer claims — and almost nothing survives it.** With ~15,000 rows, row-level CIs are tight by
construction; the quantity that actually varies between replications is *which datasets you
happened to pick*, and there are only ~11. Resampling datasets instead of rows: **1 of 384**
comparisons cleared in one persona bank and **6 of 4,032** in another, four of those six being
negative controls passing.

**6. The persona-probe advantage is inversely proportional to how strong the baseline already
is.** Across model scale (3B → 8B → 70B) the raw off-diagonal AUROC rises 0.708 → 0.743 → 0.809
while the persona gain over the matched control falls +0.202 → +0.115 → +0.079. The same pattern
holds per-dataset: the axis wins where raw collapses (below chance from two source datasets) and
buys nothing where raw already reaches 0.73–0.81. So the method reproduces at small scale and
dies at large scale, not because it stops working but because the baseline catches up.

**7. PC1 of a persona bank is essentially the Assistant Axis.** Across many differently-themed
persona banks, the first principal component keeps coming out aligned with the
assistant-vs-not direction (cos ≈ 0.84, 57% of role variance). One direct cause measured here:
half of a typical persona contrast (the "negative" side) behaves the same as the model with no
system prompt, so the contrast is largely *persona vs. being the default assistant*.

**8. Teacher-forcing the first assistant turn to isolate a multi-turn behavioural construct.**
To measure capitulation you need two turns (state a position, then get challenged). Letting each
persona generate turn 1 failed twice: personas hedged before any challenge arrived, and then gave
outright wrong answers and "folded" into correctness. Prefilling turn 1 identically for every
persona makes the context byte-identical across personas, so the only varying text is the system
prompt and the only generated text is the thing being measured. A wrong-answer control then shows
the construct only exists on *true* answers — with a false turn 1 both sides correct it.

**9. A structural confound in "knowledge report" deception datasets.** The deceptive label is
defined as *knows the answer AND denies knowing it*, but the fourth cell of that 2×2 (doesn't
know AND asserts) is structurally empty in the data, so the label is nearly collinear with a
single lexical feature. A regex on denial phrasing alone reaches **0.91 AUROC** on a dataset where
probes report ~0.87, meaning the probe result is largely one lexical bit rather than a
representation of deceptive intent.

**10. Response length transfers across deception datasets as a free-riding confound.** The
positive class is the longer one in 9 of 10 datasets (505 vs 310 tokens on one, 245 vs 96 on
another), so a one-feature "longer is deceptive" rule achieves an off-diagonal AUROC of 0.553
against raw activations' 0.665. Any cross-dataset transfer claim needs to beat this.

## What I need from you

For **each** of the ten claims, search and then report:

| field | what I need |
|---|---|
| **Verdict** | `ESTABLISHED` (this exact claim is published) · `ADJACENT` (closely related result exists, but not this claim) · `APPARENTLY NOVEL` (found nothing close) · `UNCERTAIN` (couldn't search adequately) |
| **Closest prior work** | 1–3 papers with title, authors, year, venue, link. Say specifically *what* they show and *how it differs* from my claim. |
| **Framing** | Has this been described under different terminology? (e.g. shortcut learning, spurious correlation, construct validity, surface features, probe confounds, stylistic vs semantic) |
| **Risk to me** | If a reviewer knows this literature, what is the strongest "this is already known" objection? |

Then, at the end:

1. **Rank the ten by novelty** — which 2–3 are the strongest contribution, and which should be
   presented as replication/confirmation of known effects rather than as new findings?
2. **Name the single biggest gap** — is there an obvious paper I appear not to have read?
3. **Methodology check** — are claims 4 and 5 (matched controls, dataset-level bootstrap)
   standard practice in this subfield already, or is enforcing them itself a contribution?
   This matters a lot: if they are standard, my critique is that this specific paper omitted
   them; if they are not, the critique is about the subfield.
4. **Venue suggestion** — given what is and isn't novel, where would this realistically land
   (workshop, conference, which)?

Search terms worth trying: *persona vectors*, *persona coordinates*, *contrastive activation
addition*, *steering vectors*, *representation engineering*, *linear probe generalization*,
*probe confounds / shortcut features*, *assistant axis*, *sycophancy probe*, *deception
detection linear probes*, *construct validity interpretability*, *spurious features activation
probing*, *out-of-distribution probe transfer*.

Please be blunt. I would much rather learn now that six of these are already published than find
out from a reviewer. If a claim is not novel, say so plainly and give me the citation that kills
it.
