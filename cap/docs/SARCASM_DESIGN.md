# The sarcasm bank — design v2, after review

**Status: revised, not built.** v1 was reviewed and the verdict was *"run a revised version, not
this one"* — the headline contrast could not answer the question it posed. The reviewer was
right on three structural points I had missed. This version is the revision; v1 is preserved in
git history at `3d80071`.

---

## What the review changed, in order of severity

### 1. My headline contrast was confounded — and by the fix I was proud of

v1's axis was `sarcastic_praise(bad question) − sincere_praise(good question)`. Those two arms
differ in **two** ways: whether the approval is meant (wanted) and whether the plan is good or
bad (not wanted). Response tokens attend to the question, so the direction could encode *"the
plan under discussion is risky"* rather than *"the speaker doesn't mean this."*

That is exactly the failure Goldowsky-Dill et al. concede for their own probe — it fires on *"is
the scenario deception-related"* — and it is worse here than it looks, because the deception
datasets are themselves built from bad-situation material (WMDP-derived harm pressure, insider
trading, sandbagging). **A high `cos(d_sarcasm, d_deception)` could come entirely from a shared
"this scenario is bad" component.**

I had cited the knowledge-denial bank as the precedent. The review showed the analogy breaks
exactly where it matters: there, both sides share the question and the difference lives in the
persona's knowledge state. In v1 the difference lived in the **input string**. Fixing the wording
confound had quietly introduced a situation confound.

### 2. There was no arm that isolates intent — which is the entire point

Once markers are stripped, approving text about a bad plan is textually **identical** whether the
speaker is being sarcastic or sycophantic. The difference is purely pragmatic:

> **sarcasm** — the insincerity is meant to be **recognised** (overt)
> **deception / sycophancy** — the insincerity is meant to be **believed** (covert)

**That distinction *is* the intent variable**, and v1 had no covert-insincere arm, so it could not
make the distinction it was built to test.

This also corrects my framing. I wrote *"sarcasm is deception minus the intent."* That is wrong:
sarcasm does not lack communicative intent, it has a **different** one. Grice (overt flouting of
Quality), Sperber & Wilson (echoic mention with a dissociative attitude), Clark & Gerrig (pretense
*"intended to be recognized as such"*) all locate irony in the recognition, not its absence.

### 3. A logical tension that makes v1 unfalsifiable

Our failures are **below** chance — 0.300 and 0.374. But a genuine statement-versus-belief
mismatch detector should score those datasets **above** chance, because denying knowledge you
hold *is* a mismatch. So *"the deception axis is really a mismatch detector"* (predicts above
chance) and *"a mismatch axis reproduces the below-chance failure"* cannot both hold unless the
operative variable is something else — plausibly **intended disagreement with the user**, which
ironic criticism and honest non-capitulation share.

Without pre-registering which component reproduces the failure, any outcome is explicable after
the fact. v1 was not falsifiable.

---

## The revised design

**Hold the question literally constant. Move the manipulation to the persona. Fix polarity at
"bad".** Four arms, ~20 plainly **imprudent** (not dangerous — harm content would correlate with
the harm-pressure datasets for the wrong reason) plan questions, answered by all four:

| arm | literal | intended | insincerity |
|---|---|---|---|
| `ironic_criticism` | approves | criticises | **overt** — meant to be recognised |
| `sycophantic_praise` | approves | approves falsely | **covert** — meant to be believed |
| `sincere_criticism` | criticises | criticises | none |
| `plain_assistant` | — | — | none (anchor) |

Renamed from `sarcastic_praise`: literal praise with intended criticism is **ironic criticism** in
the pragmatics literature; "ironic praise" conventionally means the reverse.

**The contrasts:**

```
ironic_criticism − sincere_criticism    mismatch, situation and intent held constant
sycophantic_praise − ironic_criticism   INTENT: both literally approve a bad plan,
                                         differing only in believed vs recognised
each arm − plain_assistant               DEMOTED by F-89 -- mostly measures "an instruction
                                         is in force", not this arm's stance. Descriptive only.
```

**The second contrast is the one the review says is non-negotiable**, and it yields a
discriminating pre-registration:

- If deception axes are **intent-blind mismatch detectors**, they align about equally with
  `(ironic − sincere)` and `(sycophantic − sincere)`.
- If they **track covert intent**, they align with `(sycophantic − ironic)` — sycophancy scores
  high, sarcasm does not.

That converts an ambiguous cosine into a test with two distinguishable outcomes.

**Personas as minimal pairs** off one template, differing only in the intent clause, with paired
differences — this attacks between-persona idiosyncrasy directly rather than by adding personas.
6–8 per arm, with the final number justified by leave-one-persona-out stability rather than
chosen up front.

**Optional extensions, clearly separate:** a `good`-polarity set to recover the full 2×2 and
project out the situation main effect; a fiction/role-play arm for family scope; a
factual-agreement arm mirroring capitulation, which addresses the domain mismatch between our
evaluative bank and the factual datasets.

---

## The headline is now causal, not a cosine

The review's strongest methodological point: a bare cosine between two mean-difference directions
is weak — activation space is anisotropic so raw cosines are inflated, there is no calibrated
null, nuisance components are shared, and truth is **not** one direction anyway (Bürger et al.
find a two-dimensional subspace, with the second component ≈ XOR(is_true, is_negated)).

Headline, in descending order of persuasiveness:

1. **Erasure.** LEACE-erase the mismatch direction, then measure the change in deception-probe
   AUROC across our eleven datasets. If erasing it moves knowledge-report and capitulation toward
   chance in the predicted direction, that is causal evidence for the register diagnosis.
2. **Firing rate.** Score existing deception probes on each arm at a **1%-FPR-calibrated**
   threshold (the Apollo convention, calibrated on Alpaca). *"Does the deception probe fire on
   overt sarcasm?"* is the most direct form of the question.
3. Mismatch-axis AUROC on the eleven datasets, with **per-dataset sign predictions registered in
   advance**.

Cosine is reported only against a null battery — random persona splits, unrelated persona axes,
shuffled labels — per layer, and whitened.

---

## The abandon-checks, corrected

**My lexical check was mis-specified and would have killed a sound bank.** In v1's two-polarity
design a bag-of-words classifier separates the arms near-perfectly from **question echo alone**,
with zero sarcasm markers. Worse, my sketched items encoded "bad" through *negation and absence*
("no rollback", "on holiday") against presence on the good side — a systematic cross-item lexical
confound, and negation has its own representational geometry.

Corrected: grouped-by-item cross-validation so no item spans train and test; mask question-echoed
tokens or restrict features to markers and function words; counterbalance negation and length;
and report **selectivity** — judged-sarcasm signal *minus* bag-of-words separability — rather than
a bare number against an arbitrary 0.75.

**The judge check needs to be blind and split into five ratings**: perceived sarcasm (question +
response, should be high), literal valence (response only, should **not** differ between ironic
and sincere arms — that is the whole point), intended valence (question + response, should
differ), an explicit marker checklist (response only, low and equal across arms), and a
plan-quality manipulation check.

**The polarity check needs a threshold and a symmetric arm.** And a failure here is informative
rather than fatal: if the *sincere* persona praises a bad plan, that is the model's own sycophancy
base rate, which bears directly on the sycophantic arm.

**Two new gating checks:**
- **Construct validity on held-out natural sarcasm** (iSarcasmEval, which labels *intended*
  sarcasm by the authors themselves). If `d_sarcasm` only fires on *marked* sarcasm, it is a
  marker detector.
- **A one-directional pre-test that costs almost nothing:** apply our *existing* deception axes to
  marked natural sarcasm. If they do not fire even on marked sarcasm, the "deception axis =
  mismatch detector" hypothesis is already weak and the bank drops in priority before we build it.

---

## Sequence

1. ~~**Format-matched control first**~~ — **done, and it came back mostly negative (F-89).**
   Swapping the six position-holding personas for six topically unrelated instructed personas
   (alphabetise, cite, metric units, name the speaker, spell out, keep time) costs only **+0.03
   AUROC**, which does not clear the dataset bootstrap on its own at any of layers 20, 40 or 44.
   Removing them entirely costs **two to four times that**. So the persona-side signal is
   *mostly about an instruction being present*, not about what it says.

   **What that changes here, concretely:** the third contrast below,
   `each arm − plain_assistant`, was justified as "per-side load, the F-66 question". It will now
   largely measure *an instruction is in force* rather than *this arm's stance*, so it cannot
   carry interpretive weight and is demoted to a descriptive check. **The two contrasts that
   matter are unaffected**, because both hold all four arms as instructed personas and difference
   them against each other — including `sycophantic_praise − ironic_criticism`, the one the
   review called non-negotiable. The design survives; its weakest contrast is now known to be
   weak *before* anything is generated, which is what this gate was for.
2. **Deadpan elicitation pilot**, 10–20 items. Models over-mark sarcasm (Sarc7: they "slip into
   blunt reproach"; SarcasmBench: LLMs underperform supervised baselines). If the model cannot
   produce unmarked sarcasm, the on-policy design is dead and we move to teacher-forced identical
   responses — accepting that Kirch et al. place text-ambiguous behaviours in exactly the regime
   where off-policy data is least reliable.
3. **One-directional pre-test** on natural sarcasm.
4. **Build the factorial** only if 2 and 3 pass.

---

## What I still think is right in v1

The motivation, the lexical trap, and the insistence that both arms produce approving text. The
review endorsed all three. What it removed is my belief that varying the *situation* was a clean
way to get there — and it was not.

**One thing the review surfaced that helps us:** the literature already documents deception probes
firing on fiction and role-play. That is our *positive* case, and it was sitting in the related
work unused.
