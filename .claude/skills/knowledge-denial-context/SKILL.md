---
name: knowledge-denial-context
description: >-
  The kd/ experiment: a persona axis for deception by DENYING KNOWLEDGE one has,
  built because every detector in this project inverts on
  harm-pressure-knowledge-report. Covers the HP-KR confound (deceptive = knows AND
  denies, with one cell empty), the two teacher-forced factor probes that make
  attribution possible, and the pre-committed readings. Use when touching kd/ or
  interpreting any HP-KR number.
user-invocable: true
---

This skill covers `kd/` — a self-contained experiment extending the persona-coordinates
replication with a persona family whose deception **is feigned ignorance**.

**Read `kd/docs/DESIGN.md` before writing code, and `kd/docs/REPORT.md` for current
results.** `kd/README.md` has the layout and status table.

## The three facts that shape everything here

1. **HP-KR is largely lexical.** One binary feature — does the response deny knowledge —
   scores AUROC **0.870** on the full 7,068 rows. A probe's 0.974 diagonal there is mostly
   that bit.

2. **HP-KR cannot attribute its own results.** `deceptive = knows AND denies`
   (`src/harm-refusal/harm-refusal.ipynb`), and the cell *doesn't know but claims it does*
   has **zero** rows. So on the text-matched slice (`denies=True`, 2,538 rows)
   `deceptive ⟺ knows` — a pure knowledge probe scores 1.0 without detecting deception; and
   within `knows=True` (5,812 rows) `deceptive ⟺ denies` — a pure lexical probe scores 1.0.
   **Never report one number for HP-KR**; report full / text-matched / knowledge-matched.

3. **The inversion is layer-dependent.** The paper16 deception axis reads 0.251 zero-shot at
   layer 20 and **0.644 at layer 40** — not inverted there. Always name the layer.

## What the experiment adds

Three persona groups on top of paper16's 26, all answering the same 30 questions:
`knowledge_denial` (4, the lie is feigned ignorance), `sincere_ignorance` (3, denies
truthfully), `plain_assertion` (3, asserts unhedged). Two axes result:

- `knowledge_denial` = feigned − sincere ignorance. **Both sides deny**, so this axis cannot
  be a wording artifact. The primary object of the experiment.
- `kd_assertion` = feigned − plain assertion. The register contrast, kept separable so the
  two explanations can be told apart.

Attribution uses two **teacher-forced** probe sets built outside HP-KR — `kd_knowledge`
(identical forced text, knowledge varies) and `kd_utterance` (same items, wording varies) —
giving `d_knowledge` and `d_utterance` to decompose any direction against.

## Rules that are not optional

- **Report `feature − matched control` (datapca@k, random@k), never only `feature − raw`.**
  The unified axis reads −0.003 against dataset PCA at matched k, and a random 2-d
  projection "gains" wherever raw is weak purely as a ceiling artifact.
- Compare zero-shot AUROCs against **random directions**, not 0.5.
- Print the 0.870 lexical null beside every HP-KR number.
- A new axis costs **zero NDIF time on the evaluation side** — Pass A/B/C store pooled
  activations per row and the axis is applied on CPU afterwards.
- All 36 personas must answer the same 30 questions; when they did not, the unified PCA came
  out as a question-topic axis (cos −0.98) rather than a harm axis.

## The register hypothesis, and the one result that confirms it

**The thread that ties the project together.** The paper16 persona axes fail on exactly the two
datasets whose *labels* are the construct they claim to detect — `harm-pressure-knowledge-report`
(0.300) and `sycophancy_mt` (0.374) — and on both a **random direction beats them**. The
diagnosis (F-13): the axes separate *register* — agreeable-vs-challenging tone, hedged-vs-blunt
wording — while those datasets label *behaviour*: denying knowledge one has, and abandoning a
correct answer under pressure. Register and behaviour correlate in ordinary text and come apart
precisely where these datasets live.

**It was tested constructively and it holds.** The `cap/` bank varies *capitulation* instead of
register, at matched model, judge, grid, seed and layers. The capitulation axis reads **0.745
zero-shot** on `sycophancy_mt` where the register axis reads **0.374**, and the trained version
beats raw by **+0.112, 95% CI [+0.050, +0.185]** on a dataset-level bootstrap, winning on 9 of 10
source datasets (F-65, F-66).

**But the gain is LOCAL (F-67).** Over the full 11×11 off-diagonal mean the margin over raw is
+0.098, CI [−0.036, +0.208] — it does not clear. The axis wins on the construct it was built for
and ties elsewhere. **Never quote F-66 without F-67.**

This is the same shape as the scale ladder (F-63): raw off-diagonal 0.708 → 0.743 → 0.809 at
3B/8B/70B while the persona gain over the matched control falls +0.202 → +0.115 → +0.079. Stated
once: **the method works, and its usefulness is inversely proportional to how good your baseline
already is.**

## Two traps that have each cost a run

**`pc:<name>` projects onto the PCA *basis*, never the contrast.** `make_features` reads
`bank["pca"][name]["components"]`. Two bases built over the same role set are the same subspace —
their per-PC signs differ, and a linear probe is invariant to that — so they score **identically
by construction**. The cap bank's A and B variants are exactly this case, and their matching
AUROCs read like a finding. `6_axes.py` now prints a `NOTE` when two bases share a role set.
A contrast direction only ever reaches a number through the **zero-shot** rows.

**Paper16's names are hardcoded in seven places across this file family**, each reachable only
after the previous one was fixed. The worst were silent: `6_axes.py`'s report loop printed an
*empty* diagnostics table for a non-paper16 bank, and `27_crossgen.py` built its bootstrap
comparison list from a constant holding paper16's four feature names — so a new bank's persona
features were **in no significance test at all** (144 tests for the cap bank before the fix, 792
after). When adding a bank, check that the number of tests went **up**.

**A gate must assert the number it names.** Two have now passed while measuring nothing: one
printed `predicted=13` and asserted `-ge 12`; the cap chain's axes gate tested only that the
output file existed while `cos(PC1,contrast) ≥ 0.90` was never computed. Both logged `PASS`.

## Behavioural inertness ≠ representational contribution

F-55 found the cap bank's six position-holding personas behave identically to **no system prompt
at all** — a bare model holds the same way, in nearly the same words. That is the clearest
evidence for why PC1 keeps coming out as the Assistant Axis: half of a typical persona contrast
is just the assistant being itself.

It does **not** follow that they can be dropped. Remove them from the PCA basis and the entire
advantage disappears — `pc:cap_only@2` is indistinguishable from raw at every layer (−0.008,
p = 0.56), against +0.112 with them (F-66). Whether a persona *changes the text* and whether it
*shapes the subspace* are different questions, and only the second one predicts probe transfer.
