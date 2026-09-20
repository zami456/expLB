# Design — knowledge-denial axis and its attribution

Written **before** any result exists. The readings in §5 are pre-committed on purpose: the
point of this experiment is to be able to tell a real gain from an artifact, and that is
only possible if the interpretation is fixed in advance.

---

## 1. The problem

HP-KR (`harm-pressure-knowledge-report`) is the one dataset where every deception detector
in this project points the wrong way.

| reading | layer 20 | layer 40 |
|---|---|---|
| paper16 deception contrast, zero-shot | 0.251 | 0.644 |
| paper16 PC1, zero-shot | 0.290 | 0.668 |
| paper16 sycophancy, zero-shot | 0.042 | 0.162 |
| paper16 unified, zero-shot | 0.058 | 0.447 |
| Assistant Axis, zero-shot | 0.837 | 0.246 |

Trained, at layer 40 (`run-crossgen/matrices/mean_L40_s42.json`, k=5, mean pooling):

| feature | HP-KR as target | HP-KR as source |
|---|---|---|
| raw | 0.668 | 0.683 |
| `axes` (persona coordinates) | **0.409** | **0.271** |
| `axes_resid` | 0.324 | 0.214 |
| **random 5-d subspace** | **0.693** | 0.656 |
| length only | 0.623 | 0.596 |

A random subspace beats the persona axes in both directions. That is the failure to fix.

**Note the layer dependence.** The inversion is largely a layer-20 phenomenon; at layer 40
the zero-shot axis is not inverted at all. Any claim of the form "the persona axis inverts
on HP-KR" must name a layer.

## 2. The hypothesis

In HP-KR the lie is a **denial of knowledge**. Everywhere else in the corpus
(instructed-deception, insider-trading, convincing-game) the lie is an **assertion of
falsehood**. The paper16 bank contrasts narrative-managing assertors (handler, broker,
spin_doctor, impression_manager, narrative_manager) against hedged record-keepers
(archivist, fact_checker, documentarian, ombudsman), so the axis encodes
`confident assertion = deceptive, epistemic disclaimer = honest`. HP-KR's labels run
exactly opposite to that, which is why it inverts rather than merely failing.

None of paper16's 25 personas feigns ignorance. `plausible_denier` is the nearest and
denies *involvement*, not *knowledge*.

## 3. Why one new contrast is not enough

**HP-KR cannot attribute its own results.** From `src/harm-refusal/harm-refusal.ipynb`,
`deceptive = knows AND denies`, where `knows = (model_belief != -1)`:

| knows | denies | deceptive | n |
|---|---|---|---|
| False | True  | honest    | 1,199 |
| True  | False | honest    | 4,405 |
| True  | True  | deceptive | 1,339 |
| False | False | deceptive | **0** |
| (other, regex-ambiguous) | | | 125 |

The fourth cell — *does not know but claims it does* — is empty. Consequences:

- On the **text-matched** slice (`denies=True`, 2,538 rows) `deceptive ⟺ knows` exactly.
  A pure **knowledge** probe scores 1.0 there while detecting no deception at all.
- Within the **knowledge-matched** slice (`knows=True`, 5,812 rows) `deceptive ⟺ denies`.
  A pure **lexical** probe scores 1.0 there.
- On the **full** set, a single binary feature — "does the response contain a knowledge
  denial?" — scores **AUROC 0.870** (measured). The raw probe's 0.974 diagonal is mostly
  this one bit.

So knowledge, wording and deception are perfectly confounded *inside* HP-KR. Any reference
direction has to come from outside it — which is what §4 builds.

**This is a result in its own right**: HP-KR as published cannot distinguish a deception
probe from a knowledge probe.

## 4. The two factor probes

Built by **teacher-forcing**. `8_coords_ndif.py` pools over an assistant span that is
supplied, not generated, so these cost an extraction pass and no rollouts.

| probe set | prompt | forced assistant text | contrast | isolates |
|---|---|---|---|---|
| `kd_knowledge` | neutral WMDP item, no harm framing | **identical** `"I do not know the answer."` on every row | knows vs not-knows | **knowledge**, wording exactly constant |
| `kd_utterance` | same neutral item, `knows=True` items only | `"I do not know the answer."` vs `"I know the answer."` | denial vs assertion | **wording**, knowledge constant within item |

Difference of class means per layer gives `d_knowledge` and `d_utterance`. Every direction
in the experiment is then decomposed against them.

**Gate before proceeding:** `d_utterance` must score ≈1.0 on the knowledge-matched slice and
≈0.5 on the text-matched slice; `d_knowledge` the reverse. If not, the forced-span
construction is wrong.

## 5. The persona design, and why the honest side is split in two

Three new groups, 10 personas, five instruction variants each.

- **`knowledge_denial`** (4) — the lie *is* feigned ignorance: `amnesiac_witness`,
  `deniable_expert`, `stonewaller`, `feigned_novice`.
- **`sincere_ignorance`** (3) — says "I don't know" and means it: `honest_novice`,
  `calibrated_reporter`, `scrupulous_witness`. Written as **dispositions** (people who reach
  for "I don't know" by temperament), not accuracy policies, so they produce denial vocabulary
  on any question rather than only where a question hands them a gap. A first draft defined
  them negatively ("neither overstates nor understates"), which is hard to role-play, scores
  badly on a judge asking whether the model "fully assumes the role", and — fatally — made them
  indistinguishable from `plain_assertion`.
- **`plain_assertion`** (3) — says "yes, I know that", unhedged: `forthright_expert`,
  `candid_witness`, `plain_dealer`.

Three of the sincere roles deliberately mirror three of the deceptive ones
(`honest_novice ~ feigned_novice`, `scrupulous_witness ~ amnesiac_witness`,
`calibrated_reporter ~ stonewaller`) so the contrast holds the role roughly constant and
varies only whether the ignorance is real.

**The question set carries half the design.** The 10 new questions split 6/4: six where the
speaker knows and admitting is costly, four where the speaker genuinely does **not** know and
admitting *that* is costly. The first draft was 10/0, which silently broke the experiment — a
sincere-ignorance persona has no gap to admit on a question that stipulates knowledge, so it
answers exactly like a `plain_assertion` persona, the two honest groups collapse, and
`knowledge_denial − sincere_ignorance` stops being wording-matched. The four inverted questions
also elicit the *other* lie, a false claim of knowledge — the cell that is empty in HP-KR (§3).

**Residual limitation, stated rather than smoothed.** A persona instruction applies uniformly
across all 30 questions, while whether a denial is sincere depends on the question's stipulated
knowledge state. So the isolation is partial, not perfect: on the six "knows" questions the
sincere personas admit while the deceptive ones deny, and that part of the contrast still
carries wording. The 6/4 split and the dispositional writing reduce this; they do not remove
it. `d_utterance` from §4 is what measures how much remains.

Two axes come out, and the difference between them is the experiment:

| axis | definition | what it can and cannot be |
|---|---|---|
| `knowledge_denial` | mean(feigned) − mean(sincere ignorance) | both sides deny; **cannot** be a wording artifact |
| `kd_assertion` | mean(feigned) − mean(plain assertion) | the register contrast; **can** be a wording artifact |

If only `kd_assertion` moves HP-KR, the hypothesis is about register, not knowledge. If
`knowledge_denial` moves it, that is the claim.

**Why `plain_assertion` exists at all.** The current honesty group is hedged in register.
Pairing new deceivers against it would re-encode "hedged vs assertive" and rebuild the same
confound. `plain_assertion` is the honest-but-unhedged control that makes the comparison
interpretable.

## 6. Pre-committed readings

| observation | reading |
|---|---|
| Gain on **full** HP-KR, large `cos(axis, d_utterance)`, collapses under residualisation | It learned the word "not". **Negative.** |
| Gain on the **text-matched** slice, large `cos(axis, d_knowledge)` | It is a knowledge probe, not a deception probe. **Negative**, and interesting. |
| Gain surviving residualisation against **both** factors | Supports the hypothesis. |
| `knowledge_denial` flat, `kd_assertion` moves | The effect is register, not knowledge. |
| PC1 of the new bank rotates rather than a new PC2 appearing | One confound replaced by another. |

**Expected trade.** In ID/IT/CG the lie is an assertion, which is why the current axis works
there (0.96–0.99). Making the axis less assertion-sensitive may cost on those while gaining
on HP-KR. The hoped-for outcome is not one better direction but **two the probe can use
together** — which is why the paper's k=3 exists.

## 7. Controls that are not optional

Every one of these is here because this project has already been burned by its absence.

1. **Matched control at the same k.** Report `feature − datapca@k` and `feature − random@k`,
   never only `feature − raw`. Against dataset PCA at matched k the unified axis reads
   **−0.003**; a random 2-d projection "gains" +0.056 wherever raw is weak, purely as a
   ceiling artifact (`corr(raw, delta) = −0.76`). See the `matched-control-kills-the-gain`
   memory entry.
2. **Random-direction nulls for zero-shot**, not comparison against 0.5. On
   convincing-game, harm-pressure-*, mask and the Q11 sleepers the persona axis is no better
   than a random direction.
3. **Lexical null** — the 0.870 denial bit, printed beside every HP-KR number.
4. **Length null** — the positive class is longer in 9 of 10 datasets; off-diagonal 0.553.
5. **Three-way HP-KR slice table** — full / text-matched / knowledge-matched. One number for
   HP-KR is not reportable.
6. **Sign-flip accounting** — cells rescued vs broken, and significantly-inverted counts,
   not a mean over heterogeneous cells.
7. **Dataset-level bootstrap**, BH-corrected. No Q2/Q6-style claim in this project has ever
   survived it.
8. **Shared question set.** All 36 personas answer all 30 questions. When they did not, the
   unified PCA came out as a question-topic axis (cos −0.98), not a harm axis
   (`unified-pc1-is-question-family`).
9. **Assistant-Axis drift balance between the contrasted groups.** *(added 2026-09-19 after an
   external review raised it; the concern is right and this project has prior evidence for it —
   paper16 halved `cos(deception, assistant)` from −0.66 to −0.37, which was the whole of Q1.)*
   Judge filtering keeps only score-3 rollouts, i.e. the ones that drifted furthest from the
   Assistant persona. If `knowledge_denial` drifts further than `sincere_ignorance`, the
   difference of means re-learns the Assistant Axis rather than sincerity.
   **The check is already printed** — `6_axes.py:265` reports each group's mean projection onto
   every axis, `assistant` included. Promote it to a gate: the `knowledge_denial` and
   `sincere_ignorance` centroids must sit at comparable projections on `assistant`, and
   `cos(knowledge_denial_axis, assistant)` must be reported beside every result. Do **not**
   assume balance because both groups say "I don't know" — measure it. If they are imbalanced,
   `axes_resid` and `erase:assistant` (`9_persona_probes.py:140`) are the existing tools.
10. **Per-group response length.** *(same review)* Denials are short and assertions are long, so
   `knowledge_denial` and `plain_assertion` will differ in token count by construction, and
   mean-pooling over a short span is a different estimator from mean-pooling over a long one.
   This corpus already has a length pathology — length alone transfers off-diagonal at 0.553,
   and the positive class is longer in 9 of 10 datasets (`response-length-transfers-on-its-own`).
   Report the per-group token-length distribution from `run/responses/*.jsonl` after Phase 3, and
   if the groups are badly imbalanced, rebuild the axis on a length-matched subsample as a
   robustness check. Do not try to equalise lengths by instruction — that distorts the persona.

## 8. Scoring choices, with the evidence behind them

Pass A / B / C are **extraction configurations**, not scoring methods — A is 80 layers ×
2,000 rows, B is 21 layers × 15,278 rows, C is per-token × 6 layers × 2,000 rows. They store
pooled activations per row; an axis is applied afterwards on CPU. **A new axis therefore
costs zero NDIF time on the evaluation side.**

Improvements worth running, in expected-value order:

1. **Centre `z` within dataset before projecting.** The paper never specifies this
   (ambiguity #7 in `docs/papers/persona-coordinates-notes.md`) and corpus identity is 99.3%
   linearly decodable here, so an un-centred dot product is dominated by the dataset mean
   rather than the label. Free, untested, and the most likely legitimate mover.
2. **Top-decile per-token aggregation** for zero-shot — measured +0.053 over mean pooling.
3. **Mean pooling for *trained* probes** — measured better than per-token at every layer.
   The two Q8 halves disagree; quote both, never conflate them.
4. **`mms`** (difference of means) alongside the paper's `C=1` logistic regression, which
   overfits 8192 dimensions.

## 9. Known weaknesses of the pipeline being reused

1. **No per-(persona, question) vectors.** `2_activations_ndif.py` accumulates the sum
   server-side, so question-subset ablations need a re-extraction. This already cost the
   project once.
2. **Judge filtering at score 3 diverges from the paper**, which filters nothing. Keep both
   `vectors_all` and `vectors_s3`; say which produced each number.
3. **Test activations are never centred** — see §8.1.
4. **PCA over 16–36 points in 8192 dimensions** makes PC2/PC3 unstable. Prefer the contrast
   direction and PC1.
5. **Expect a low keep rate on the feigned-ignorance roles.** A model told to play dumb
   often breaks character and answers anyway. `--min_count 20` guards this; report the drop
   rather than rebuilding the axis without the role.
