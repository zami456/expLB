# Review request: the pooled-training result

**What I want:** adversarial scrutiny of a result that is now the strongest claim in the project
and that **no external reader has seen**. Two rounds of expert review happened *before* it
existed. It was produced in a single afternoon. That combination is exactly when an error
survives.

Please try to break it. I have listed the places I think it is weakest; I would rather hear that
one of them is fatal than publish it.

---

## The claim

**Setting.** Llama-3.3-70B-Instruct, residual stream layer 40, mean-pooled over response tokens,
eleven deception/honesty datasets (Liars' Bench, Apollo's set, MASK, Pacchiardi's AI Liar, one of
ours). Probes are logistic regression on 8192-d activations, class-balanced training, 80/20
splits by item id.

**The motivating context.** The method we are replicating (Mahadik & Skapars, arXiv:2605.09391)
argues that probes trained on one dataset transfer badly, so you should use low-dimensional
persona structure instead. Every matrix in our project was train-on-one / test-on-others, because
that is the design the paper uses. We never tested the obvious alternative.

**Result 1 — pooled training.** A probe trained on ten datasets and tested on a held-out
eleventh:

| condition | trained on | AUROC |
|---|---|---:|
| single-source transfer | one other dataset | 0.676 |
| **pooled, leave-one-dataset-out** | **ten others, never the target** | **0.854** |
| **pooled, same row budget (~666)** | **ten others** | **0.850** |
| in-domain ceiling | the target's own data | 0.951 |

+0.178, 95% CI [+0.143, +0.213] under a **dataset-level** bootstrap (resampling which of the
eleven targets you happened to pick, paired, B = 10,000). Wins on 11/11 targets. Positive at
**21/21 layers**, every one clearing. Stable across five seeds (0.173–0.198).

The row-matched condition is the control: full pooling (~6,700 rows) and budget-matched pooling
(~666 rows) are statistically indistinguishable (p = 0.38). **The gain is diversity, not volume.**

**Result 2 — the mechanism, and this is the part I trust least.** Take the ten single-source
probe directions, unit-normalise each, average them, project. That probe — **never trained
jointly** — reaches **0.883**, beating joint training.

The geometry offered as explanation:

| | cosine |
|---|---:|
| single-source ↔ single-source | **+0.051** |
| pooled ↔ single-source | +0.227 |

Ten probes trained on ten deception datasets point in nearly **orthogonal** directions, yet their
average transfers at 0.883. Reading: each direction is a small shared component under large
mutually-orthogonal dataset-specific noise; averaging cancels the noise.

---

## Where I think it is weakest

1. **The mean-of-directions comparison may be unfair.** Those ten probes were each fit on their
   **full** 80% split, so the average has effectively seen all the data, while the budget-matched
   pooled probe saw ~666 rows. I compare it against full pooling (0.854) rather than the matched
   row (0.806) for that reason — but is that the right comparator, or is there a residual
   advantage I have not controlled?
2. **Unit-normalising before averaging is a choice I did not test.** An unnormalised mean would
   let high-norm directions dominate. I do not know how much of the 0.883 depends on that choice.
3. **The +0.051 cosine may be an artifact of the probe family.** It is measured on one probe type
   (L2 logistic regression). A different regulariser could give very different geometry — and
   activation space is anisotropic, so I have not established what "near-orthogonal" means against
   a proper null here.
4. **Result 2 is one layer, one seed.** Result 1 has 21 layers and five seeds; the mechanism does
   not, and it is the more surprising claim.
5. **Eleven is a small resampling unit.** The CI is tight because the effect appears on every
   target, not because n is comfortable.
6. **I have not established *why* single-source directions are near-orthogonal.** Dataset-specific
   surface features is the story, but it is a story.

## Prior work I know about, and where I may be reinventing

- **Ying et al., arXiv:2602.20273** — "training on all domains jointly recovers strong
  performance, confirming that domain-general directions exist despite poor pairwise transfer."
  This result is a replication of theirs on a more heterogeneous corpus, plus the row-budget
  control they do not report.
- **Kumar, arXiv:2605.27958** — style-augmented probes recover 0.979–0.983 on unseen styles. Same
  "diversify the training distribution" move, over styles rather than datasets, published first.
  **Is our dataset-level version a contribution or a corollary?**
- **Yoo & Skapars, arXiv:2609.02893** — "the directions a source probe weights heavily appear to
  encode source-specific surface features, while the directions that actually transfer encode the
  same contrast more abstractly." That is our near-orthogonality finding in their words. They
  recover the transferable subspace by *selecting* PCs with an LLM judge; we recover it by
  *averaging* across sources. **Is averaging a genuinely simpler route, or a special case of
  their selection?**
- Kumar also **rejects the single-direction hypothesis** (k=1 captures 0.61–0.80). Our
  mean-of-directions *is* a single direction at 0.883. My reconciliation: his k=1 is the top PC
  of **one** source, ours is an average across **ten**, and the near-orthogonality is exactly why
  they differ. **Does that hold, or am I rescuing it?**

## The questions I actually want answered

1. **Is the mean-of-directions result real, or an artifact of the comparison?** This is the one I
   would most like attacked.
2. **Is there a confound in leave-one-dataset-out I have not seen?** The probe never sees the
   target, but the ten sources share preprocessing, prompt templates and in some cases source
   corpora (several are WMDP-derived). Could shared provenance rather than domain-generality
   explain it?
3. **Given Ying and Kumar, what is left that is ours?** Honest answer wanted, including "not
   much."
4. **What would falsify the noise-cancellation story?** I can think of: if the average of ten
   *random* directions with matched norms transferred comparably, the story dies. I have not run
   that. What else?
5. **Is the dataset-level bootstrap the right unit here**, or should it be clustered differently —
   by source corpus rather than by dataset, given the WMDP overlap?

## Where to find things

Numbers: `kd/docs/FINDINGS.md`, entries F-79, F-80, F-81, F-83, F-86.
Code: `src/assistant_axis_ndif/30_premise.py` (pooled training),
`30b_premise_boot.py` (bootstrap), `30d_why_pooling.py` (mechanism).
Outputs: `run-premise/`.
