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
