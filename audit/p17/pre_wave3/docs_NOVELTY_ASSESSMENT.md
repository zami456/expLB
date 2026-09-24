# Novelty assessment — two independent web-search reviews, 2026-09-21

Two separate Claude.ai sessions with web search were given `NOVELTY_CHECK_PROMPT.md` and asked to
rank the project's ten claims against the literature. They ran independently and reached
**different conclusions on the two claims that matter most**, which is more useful than agreement
would have been.

Neither reviewer verified any of our numbers. Both flagged that several cited sources are 2026
preprints read via abstract/HTML, so exact figures from them are preprint-grade.

---

## 1. What both reviewers agree on

**Safe to present as contributions (both said novel):**

| claim | what it is |
|---|---|
| **5 (empirical half)** | under a dataset-level bootstrap almost nothing survives — 1/384 and 6/4,032, four of the six negative controls passing |
| **8** | byte-identical teacher-forced turn 1 + the wrong-answer control as a construct-isolation design |

**Must be demoted to confirmation (both said established/adjacent):**

> **CORRECTION 2026-09-24 (DF-55, Phase 6).** The row for claim **9** below rests this project's
> **only** original contribution to that claim on *"the regex-0.91-vs-probe-0.87 quantification"*.
> **That comparison is inverted and 0.87 has no source.** Across all 113 stored matrices containing
> HP-KR, the raw probe diagonal is **0.9663 median** (0.96–0.98 at every headline layer), so the
> regex at **0.9125 loses by ~0.054** — it does not beat the probe. 0.870 appears to be this
> project's own superseded regex, which would make the comparison regex-against-itself.
> **Source:** 113-matrix sweep, `audit/scripts/qa_w5e_O040.py`.
> **How the claim is updated:** the contribution survives but must be restated as *"a label-free
> denial-phrasing regex reaches 94.4% of a full 8,192-d probe's in-distribution score on the same
> rows (0.9125 vs 0.9663)"* — a strong claim about how little the probe adds, not a claim that the
> regex wins. Same correction is owed at `docs/NOVELTY_CHECK_PROMPT.md:113` (annotated) and
> `docs/LAYER_PREREGISTRATION.md:93`.
> _The row below is superseded on this point and is kept for the record._

| claim | killed by |
|---|---|
| **7** — PC1 ≈ Assistant Axis | **Lu et al., arXiv:2601.10387.** This *is* their result, on the same Llama-3.3-70B, and our source paper cites them and reuses their residualisation. Never present as new. |
| **6** — baseline catches up with scale | Marks & Tegmark (2310.06824), Bürger et al., Kantamneni et al. — and it restates claim 4 |
| **10** — length confound | Singhal et al. 2023; Gururangan/Poliak hypothesis-only baselines |
| **1** — persona axes encode register | Goldowsky-Dill et al. §5.1 concede the probe fires on "is the scenario deception-related"; Vennemeyer et al. (2509.21305) separate agreement-register from the agreement act |
| **9** — the empty 2×2 cell | **Liars' Bench documents it themselves** (Appendix D: cases where the model claims knowledge without a consistent belief "are discarded"). Only the regex-0.91-vs-probe-0.87 quantification is ours |

**Both also said:** merge claims 4 and 6 — a reviewer reads them as one finding stated twice.
**Both venue verdicts:** BlackboxNLP or TMLR; main track only with a methods-first reframing.

**Claim 5's principle is textbook** — Demšar (JMLR 2006), Miller (arXiv:2411.00640, clustered SEs
"over 3X larger"), Dror et al. (TACL 2017). Frame as *importing* standard practice, not inventing
it. The contribution is that this subfield does not do it and that the effect vanishes when you do.

---

## 2. Where they disagree — and it is exactly claims 3 and 4

This is the most informative part of the exercise. Each reviewer called one of these novel and
the other adjacent, on **different prior art**:

| | Reviewer A | Reviewer B |
|---|---|---|
| **Claim 3** (inert personas carry the signal) | **ADJACENT** — Chen et al. *Persona Vectors* (2507.21509) show representation precedes behaviour (r = 0.75–0.83) and that persona-vector filtering survives LLM-judge behavioural screening | **APPARENTLY NOVEL** — checked Lu et al. directly and found they claim the *near-opposite* (drift tracks and predicts behaviour); "nowhere claims behaviorally inert personas carry most of the geometric signal" |
| **Claim 4** (ceiling artifact) | **APPARENTLY NOVEL** — "your best card"; found nothing doing the quantified decomposition | **ADJACENT** — Kantamneni et al. (ICML 2025) on SAE probes vs strong baselines; Kumar (2605.27958) calls the scaling pattern "a training-distribution artifact" |

**Reading:** neither claim is safe alone, and the disagreement is evidence that the prior art is
scattered rather than decisive. The honest position is that **3 and 4 are each defensible only
with their respective neighbour cited and distinguished in the same paragraph.**

---

## 3. Four papers we have not engaged, ranked by threat

1. **Kirch, Dower, Skapars, Yannakoudakis, Lubana & Krasheninnikov — "The Impact of Off-Policy
   Training Data on Probe Generalisation," ACL Findings 2026 (arXiv:2511.17408).**
   **The sharpest threat, and it is personal: Skapars co-authors this AND our source paper.**
   It shows off-policy data causes the largest generalisation failures precisely for intent-based
   behaviours like deception — which is the direct objection to claim 8, since prefilling turn 1
   makes that turn off-policy. It also splits "text-evident" from "intent-based" behaviours,
   pre-empting part of claims 1 and 9.
   **Our answer, which must appear in the paper:** the prefilled turn 1 is *context*, never
   scored. Only turn 2 is generated and only turn 2 enters the activations. Their result concerns
   off-policy **training data** for the probe; ours is an off-policy **prompt prefix** held
   byte-identical across conditions. Related but not the same object. Say so explicitly rather
   than hoping nobody asks.

2. **Kumar — "Pressure-Testing Deception Probes in LLMs," GEM @ ACL 2026 (arXiv:2605.27958).**
   *Concurrent* with our source paper (both May 2026). Independently reports the register/style
   confound, k=1 insufficiency, permutation nulls, and "the scaling pattern is a
   training-distribution artifact" — overlapping claims 1, 4 and 6.

3. **"Probe Generalization as Subspace Selection for OOD Deception Detection" (arXiv:2609.02893).**
   Reviewer A's biggest gap. Selects PCs to avoid "surface-feature" directions on Llama deception
   datasets, using an LLM judge to recover a transferable subspace without target data.
   Conceptually the closest neighbour to the whole thesis, and it *post-dates* the source paper.
   **Differentiation:** they select PCs to *improve* transfer and treat it as a positive method;
   we show the persona-coordinate instantiation is a scale-dependent artifact and enforce
   dataset-level significance.

4. **Vennemeyer, Duong, Zhan & Jiang — "Sycophancy Is Not One Thing," EMNLP 2026
   (arXiv:2509.21305).** Separates sycophantic agreement, sycophantic praise and genuine
   agreement into distinct linear directions. The construct-separation precedent for claims 1–2.

Also worth pulling: **Hewitt & Liang (EMNLP 2019)** control tasks/selectivity, and
**Kantamneni et al. (ICML 2025)** on strong baselines — both are the standard the matched-control
argument should be phrased against.

---

## 4. Decisions taken

- **Lead with claims 5 (empirical), 8, and whichever of 3/4 survives its named neighbour.** Both
  reviewers converged on 5 and 8; 3 and 4 are contested and need their neighbours addressed in
  text.
- **Demote 1, 6, 7, 9, 10 to "we confirm, and for 4/7 explain the mechanism of, known effects."**
  This is the framing both reviewers independently recommended, and it converts the weakness
  (limited novelty) into the paper's actual identity: a methodological audit.
> **CORRECTION 2026-09-24 (DF-68, Phase 6).** The bullet below states Claim 4's mechanism as
> *"ceiling artifact, corr −0.76"*, and §4 of this file nominates Claim 4 to lead with.
> **That correlation was withdrawn by its own evidence.** F-74 — whose verdict is HOLDS —
> measured **−0.7684 *inside* its own coupling null** (analytic −0.7462, **P = 0.2135**) and
> explicitly withdrew it; the clean uncoupled test (slope **−0.100**, r **−0.067**) does not
> support it either.
> **Source:** `kd/docs/FINDINGS.md` F-74, including its coupling-null section.
> **How the claim is updated:** state the ceiling mechanism **without** the correlation — the
> −0.76 is not evidence for it. If Claim 4 is to lead, it must lead on something that survives its
> own null.
> _The bullet below is superseded on this point and is kept for the record._

- **Merge 4 and 6.** 4 = mechanism (ceiling artifact, corr −0.76). 6 = the scaling trend.
- **Do not claim PC1 ≈ Assistant Axis.** Cite Lu et al.; present our numbers as corroboration
  plus the mechanism (half of each contrast behaves like the default assistant).
- **Sharpen claim 9 as an inversion**, not a discovery: Liars' Bench says HP-KR is *hard* to
  detect from the transcript; we say it is detectable *for the wrong reason* — one lexical bit.
  Frame the regex explicitly as a hypothesis-only baseline (Gururangan/Poliak).
- **Target BlackboxNLP or TMLR.** Both reviewers, independently.

## 5. The one experiment that would change the verdict

Reviewer B's, and it is cheap and runnable with the existing `cap/` bank:

> Ablate the inert position-holding personas, then **reintroduce them as a controlled non-persona
> prompt of identical format**. If the advantage returns with a format-matched non-persona, claim
> 3 is a prompt-format artifact. If it does not, the geometric-signal result is causal.

F-55 already has a *no-system-prompt* control, which is not the same thing — it differs from a
persona prompt in both content and format at once. A format-matched neutral prompt (same length,
same structure, no persona content) separates those. Both reviewers named the prompt-format
objection as the thing to pre-empt, and Reviewer B says surviving it could carry a main-track
short paper on its own.

**Status: proposed, not approved.** It needs a new 6-role generation run and a re-extraction.

## 6. Correction this exercise forced

We misattributed **MASK** to Cadenza Labs. It is **Ren, Agarwal, Mazeika, Menghini et al., Center
for AI Safety + Scale AI, arXiv:2503.03750** — verified from `papers/The MASK Benchmark
Disentangling.pdf`. Cadenza Labs is **Liars' Bench** (Kretschmar, Laurito, Maiya, Marks,
arXiv:2511.16035), verified from `papers/2511.16035v2 (1).pdf`. The error came from reading the
`mask/` submodule URL (`github.com/cadenza-labs/mask`, an evaluation *fork*) as the benchmark's
authorship. `CLAUDE.md` now says so at the point where the mistake was available to be made.
