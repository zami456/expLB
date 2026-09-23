# The Sarcasm Bank — A Pre-Registered Design That Failed Its Own Gate

**Status as of:** 2026-09-22 · **Subject model:** `meta-llama/Llama-3.3-70B-Instruct`, run
remotely on NDIF · **Ledger:** F-92, F-93 (+ amendment), F-104 · **Design:**
`cap/docs/SARCASM_DESIGN.md` (v2, revised after external review) · **Data:**
`cap/run-sarc-pilot/`

**What this section is.** A negative result, reported at the same length as a positive one. We
designed a persona bank to test the central diagnosis of this thesis, pre-registered the
conditions under which it would be abandoned, ran the elicitation pilot, and hit one of those
conditions. The bank was not built. This section argues that the *manner* of the failure is
itself evidence for the diagnosis it was meant to test.

---

## 1. What the bank was for

The thesis's central negative claim is that **persona axes encode register rather than the
construct their datasets label** — how the model talks (agreeable vs challenging, hedged vs
blunt) rather than what it does. That claim is supported by two failures: the paper's sycophancy
axis scores **0.313** on our capitulation dataset and ~0.300 on knowledge-report, both below
chance (0.313 significantly so, beyond a 1,000-direction null at p 0.001-0.008). The **0.374**
this sentence used to give is the paper16 *deception* axis's PC1, not the sycophancy axis. Every test of it so far is negative in form: the axis *fails* where
register and construct come apart.

A sarcasm bank would have been the first **positive** test. Sarcasm is a case where literal
content and intended content are guaranteed to diverge, by construction. If a "deception"
direction fires on overt sarcasm — where nobody is being misled — then that direction is
substantially a mismatch detector, not an intent detector. Apollo's own probe paper concedes
their detector partly fires on *"is this scenario deception-related"* rather than on deceptive
intent; a sarcasm axis would put a number on that concession.

### 1.1 A framing error, corrected before it reached the design

We initially wrote *"sarcasm is deception minus the intent."* **That is wrong**, and external
review caught it. Sarcasm does not lack communicative intent; it has a *different* one. The
distinction is standard in pragmatics:

> **sarcasm** — the insincerity is meant to be **recognised** (overt)
> **deception / sycophancy** — the insincerity is meant to be **believed** (covert)

Grice locates irony in the overt flouting of the maxim of Quality; Sperber & Wilson in echoic
mention with a dissociative attitude; Clark & Gerrig in pretense *"intended to be recognized as
such."* All three put the phenomenon in the recognition, not in its absence. The corrected
framing is what makes the design testable: **overt versus covert insincerity is the intent
variable**, and a bank must contain both to isolate it.

---

## 2. The design, and the gates written before any data existed

Four arms answer the same ~20 plainly **imprudent** plan questions. The question is held
literally constant; the manipulation lives entirely in the persona.

| arm | literal content | intended content | insincerity |
|---|---|---|---|
| `ironic_criticism` | approves | criticises | **overt** — meant to be recognised |
| `sycophantic_praise` | approves | approves falsely | **covert** — meant to be believed |
| `sincere_criticism` | criticises | criticises | none |
| `plain_assistant` | — | — | none (anchor) |

The contrast the review called non-negotiable is `sycophantic_praise − ironic_criticism`: **both
arms literally approve a bad plan, and differ only in whether the insincerity is meant to be
believed or recognised.** That yields a discriminating prediction — if deception axes are
intent-blind mismatch detectors they align about equally with `(ironic − sincere)` and
`(sycophantic − sincere)`; if they track covert intent they align with `(sycophantic − ironic)`.

### 2.1 The abandon-conditions

Two gates were registered in the design document, with thresholds, before generation:

| gate | question | threshold | rationale |
|---|---|---|---|
| **G1 markers** | can the model produce *unmarked* sarcasm? | lowest ironic marker rate **≤ 0.20** | the literature says probably not — Sarc7 reports models "slip into blunt reproach"; SarcasmBench finds LLMs underperform supervised baselines |
| **G2 separability** | with markers stripped, are the ironic and sycophantic arms **textually alike**? | bag-of-words AUROC **≤ 0.65**, grouped by item | the design's own words: *"near 0.5 with markers removed = the arms are textually alike, which is the premise the factorial needs. Near 1.0 = a lexical detector."* |

G2 is the load-bearing one. If the two arms are lexically separable, any direction extracted from
their difference is a **vocabulary classifier**, and the bank would be a register axis — the exact
failure it exists to diagnose.

Grouped-by-item cross-validation was required so that no question spans train and test. This
corrected a mis-specification in v1, whose lexical check would have separated the arms on
*question echo alone* and killed a sound design for the wrong reason.

The ironic arm was generated at **five escalating instruction strengths** (v1 = no mention of
markers, v5 = exhaustive prohibition) so the pilot could report *which* instruction strength
works, not merely whether one does.

---

## 3. What was measured

300 single-turn rollouts: 3 arms × 5 variants × 20 questions, temperature 0.7, judged by
Llama-3.1-70B — the same judge as every other bank in this project.

### 3.1 G1 passes: deadpan *is* producible

| variant | ironic marker rate |
|---|---:|
| v1 (no mention) | **0.85** |
| v4, v5 (explicit prohibition) | **0.00** |

The model will suppress the surface furniture of sarcasm — scare quotes, "oh, brilliant",
exclamation marks, all-caps, the reveal-aside — when told to, and completely. Against the
literature's expectation, **this gate passed** (F-92).

*Caveat:* marker detection is a regex over eight patterns, so 0.00 means "none of these eight",
not "none".

### 3.2 G2 fails, and not narrowly

Bag-of-words logistic regression, grouped by question id:

| subset | pair | raw | **markers stripped** | n |
|---|---|---:|---:|---:|
| all v1–v5 | ironic vs sycophantic | 0.953 | **0.947** | 200 |
| v1 only (85% marked) | ironic vs sycophantic | 1.000 | 1.000 | 40 |
| **v4+v5 (marker rate 0.00)** | **ironic vs sycophantic** | **0.919** | **0.921** | 80 |
| v4+v5 | ironic vs sincere | 0.997 | 0.996 | 80 |

Against a threshold of 0.65, the measured value is **0.921**. Stripping every marker token moves
the number by **−0.002**: the separation does not live in the markers at all.

**Where it does live** is the finding. The discriminating features are ordinary content and
function words:

- **ironic arm** — *ability, fact, decision, undoubtedly, I'm* — abstract, first person
- **sycophantic arm** — *plan, you're, you've, team, straightforward* — concrete, second person

**That is a register difference.** Asked to praise a bad plan insincerely-but-overtly, the model
writes abstractly and about itself; asked to praise it insincerely-but-covertly, it writes
concretely and about the addressee. The pragmatic variable we intended to manipulate arrived
**bundled with a stylistic one**, and the stylistic one is what a linear probe would find first.

### 3.3 The polarity check passes; its trend does *not* support what we first read into it

The third check (does the *sincere* arm hold its polarity, or does the model's own sycophancy base
rate contaminate it?) passed. Judged role score, 0–3, by variant:

| arm | v1 | v2 | v3 | v4 | v5 | mean |
|---|---:|---:|---:|---:|---:|---:|
| `pc_ironic_critic` | 2.85 | 2.85 | 2.60 | 2.55 | **2.10** | 2.59 |
| `pc_sincere_critic` | 2.85 | 2.25 | 2.40 | 2.25 | **2.10** | 2.37 |
| `pc_sycophant` | 2.45 | 2.25 | 2.55 | 2.15 | **2.80** | 2.44 |

**An earlier reading of this table — that the ironic arm becomes *less legibly ironic* as the
deadpan instruction strengthens (2.85 → 2.10) — does not survive the control.** The *sincere*
arm declines over the same range and lands on exactly the same value (2.85 → 2.10). Since the
sincere arm's instruction says nothing about markers or irony, a decline shared by both critic
arms is more parsimoniously read as a general effect of longer, more constrained system prompts on
judged role adherence than as irony-specific degradation. The sycophantic arm's apparent rise is
also non-monotone (2.45, 2.25, 2.55, 2.15, 2.80) and rests on its final cell alone.

What the table *does* support is narrower and still relevant: at v5 — the wording the factorial
would have used — the ironic arm sits at its lowest judged-role value while the arms remain **0.92
separable on vocabulary**. Role legibility is weakest where the lexical confound is not. That is
worth stating; the causal story about markers suppressing irony is not, on this evidence.

*Caveat, and it matters:* this used the project's standard role-adherence judge, **not** the blind
five-rating protocol the design specifies (perceived sarcasm, literal valence, intended valence,
marker checklist, plan quality). In particular it cannot separate *literal* from *intended*
valence, which is the distinction that carries the argument. The polarity evidence is therefore
weak on its own; §3.2 is what decides the gate.

---

## 4. Why this is a result and not just a null

A null result would be "we built the bank and the axis did nothing." That is not what happened.
What happened is more specific and more useful:

**An intent-defined construct could not be elicited on-policy without a register confound.** The
model appears to *encode* the overt/covert distinction stylistically — different person, different
abstraction level — rather than holding style fixed and varying only what is meant. For persona
vectors this is close to fatal at the construction stage, because a persona axis is a difference
of group means over generated text: whatever systematically separates the groups' vocabulary
enters the direction, and here that is register.

This is the register diagnosis appearing **one level earlier than we had tested it**. Our other
evidence is about axes failing on datasets. This is about the *generation step* — the model's own
production of two personas that are supposed to differ only in intent — already carrying the
confound. The bank did not fail because the probe was weak; it failed because the material the
bank would have been built from is stylistically stratified by the very variable it was meant to
hold constant.

It also **generalises beyond sarcasm**. Any construct defined by intent rather than by surface
form is exposed to this: if the model marks intent through style, then on-policy persona
construction for that construct yields a style axis. That is a constraint on the persona-vector
method itself, not on our implementation of it, and it sits alongside the finding (F-89/F-94) that
the persona side of these contrasts is mostly measuring *that an instruction is in force* rather
than what the instruction says.

---

## 5. What the verdict was, and a process note

The pilot's own chain reports:

```
3a markers       measured=lowest ironic marker rate=0.00      predicted=<=0.20   PASS
3b separability  measured=bag-of-words AUROC stripped=0.947   predicted=<=0.65   GATE FAILED
VERDICT TEACHER-FORCED -- markers are suppressed but the arms remain lexically
        separable (0.947); an extracted direction would be mostly lexical
```

**This verdict was nearly not produced.** As first written, the chain gated on the marker rate
alone and merely *printed* separability, so on 2026-09-21 it logged `VERDICT PRODUCIBLE — build
the factorial` one line above `stripped=0.947`. The number was correct and the rule applied to it
was absent. The fix was written the same day but — as a later audit established (F-104) — was
never copied to the machine that runs the chain, so the box continued to emit the wrong verdict
until 2026-09-22. The conclusion in this section was reached by reading the number directly; the
chain has since been corrected and now reaches it independently.

We record this because a pre-registered gate is only worth what its implementation asserts, and
ours asserted the wrong thing for a day.

---

## 6. Limitations

1. **One model, one construct, one instruction family.** That deadpan sarcasm is stylistically
   separable from covert sycophancy *in Llama-3.3-70B under these five wordings* does not
   establish it in general. A different instruction design, or a model with different stylistic
   habits, could plausibly clear G2.
2. **80 responses carry the decisive number.** The v4+v5 subset is 40 per arm over 20 questions.
   Grouped CV protects against item leakage but not against small-sample optimism in either
   direction.
3. **The judge protocol was not the one specified** (§3.3), so the convergence trend is weaker
   evidence than designed.
4. **Marker regexes misfire on sincere text** — `allcaps` fires 23 times and `reveal_aside` 13
   times on the sincere arm, largely from phrases like "to be clear" — so marker rates outside the
   ironic arm should not be read as irony marking.
5. **G2 is a bag-of-words test.** It establishes that a *lexical* classifier separates the arms.
   It does not directly establish that a *representation-space* direction would be lexical, though
   that is the natural inference and the premise the design was written against.
6. **The decisive numbers are reproducible but not persisted.** `score_sarcasm_markers.py` writes
   exactly one separability figure to `marker_report.json` — the all-variant pair, 0.953 / 0.947
   over n=200 — and has no subset flag and no coefficient output. The per-subset rows in §3.2 and
   the feature lists were produced by ad-hoc code at the time and were never saved. They were
   re-derived for this document by importing the scorer's own `bow`, `strip_question` and
   `grouped_auc` and re-running them over `cap/run-sarc-pilot/responses/`; **all four rows
   reproduced to three decimals and all ten quoted words re-appeared in the top-25 coefficients of
   their respective sides** (six of the ironic five and five of the sycophantic five in the top
   12). The result is sound; the artifact is incomplete, and the scorer should persist the
   subsets and top features before this is cited further.

---

## 7. Why we did not fall back to teacher forcing

The design's stated fallback is to give all four arms **identical teacher-forced responses**,
varying only the system prompt — the construction the capitulation bank uses, for the same reason.
That would eliminate the lexical confound by fiat.

We did not take it, for a reason internal to the literature we cite. **Kirch, Dower, Skapars et
al.** (arXiv:2511.17408) find that off-policy training data is acceptable for behaviours
identifiable from surface properties and harmful for behaviours where the same text can come from
different underlying processes — with the **largest failures for intent-defined behaviours**.
Sarcasm is the most intent-defined construct in this project; identical text is the entire point.
The fallback therefore lands in precisely the regime its own authors identify as least reliable,
and the design document concedes that the objection becomes "load-bearing rather than
hypothetical."

A further gate — a one-directional pre-test applying our *existing* deception axes to natural
marked sarcasm (iSarcasmEval, which labels *intended* sarcasm by the authors themselves) — was
registered and **never run**. The design's sequence is explicit that the factorial should be built
only if gates 2 **and** 3 pass. Gate 2 failed; gate 3 has no result. Building the teacher-forced
version would mean proceeding on a design whose cheap pre-test was skipped and whose data regime
is contraindicated.

**The honest position is that the bank is not built, the on-policy route is closed on measurement,
and the teacher-forced route is open but unattractive.** Reviving it should begin with gate 3,
which costs almost nothing: if our existing deception axes do not fire even on *marked* natural
sarcasm, the mismatch-detector hypothesis is already weak and the bank should stay unbuilt.

---

## 8. Sources

| claim | file |
|---|---|
| design, arms, gates, thresholds | `cap/docs/SARCASM_DESIGN.md` (v2; v1 at git `3d80071`) |
| marker rates per variant | `cap/run-sarc-pilot/marker_report.json`; F-92 |
| separability, per subset | `cap/run-sarc-pilot/marker_report.json` (`lexical`); F-93 |
| judged-role trend | `cap/run-sarc-pilot/scores/*.json`; F-93 amendment |
| rollouts (300) | `cap/run-sarc-pilot/responses/*.jsonl` |
| chain, gates, verdict | `cap/scripts/run_sarc_pilot.ps1`; `logs/sarc_pilot.status` |
| pre-fix verdict, preserved | `logs/sarc_pilot.status.2026-09-21-buggy-verdict` |
| scorer | `cap/scripts/score_sarcasm_markers.py` |
| the undeployed-fix audit | F-104 |
