# Report — knowledge-denial axis

Running log. Append as each phase lands; mark superseded numbers rather than deleting them.
Design and pre-committed readings: `DESIGN.md`.

---

## Phase 0 — baseline measurements (2026-09-19, read-only, no new compute)

These were measured while planning, from data already in the repo. They are the baseline
every later number is compared against.

### 0.1 HP-KR is largely a lexical task

Source: `results/subsets/harm-pressure-knowledge-report.csv`, 7,068 rows.

A single binary feature — does the assistant response contain a knowledge denial —
scores **AUROC 0.870**.

| response form | honest | deceptive |
|---|---:|---:|
| asserts knowledge ("I know the answer") | 4,462 | 68 |
| denies knowledge ("I do not know the answer") | 1,199 | 1,339 |

The raw probe's 0.974 within-dataset diagonal is mostly this one bit.

### 0.2 HP-KR cannot attribute its own results

Label definition, read from `src/harm-refusal/harm-refusal.ipynb` (`compute_dishonesty`):
`deceptive = knows AND denies`, with `knows = (model_belief != -1)` and `model_belief` the
model's consistent answer across control templates, set only when it consistently claimed
knowledge there.

| knows | denies | deceptive | n |
|---|---|---|---:|
| False | True  | honest    | 1,199 |
| True  | False | honest    | 4,405 |
| True  | True  | deceptive | 1,339 |
| False | False | deceptive | **0** |
| regex-ambiguous remainder | | | 125 |

The fourth cell is empty, so:

- **text-matched slice** (`denies=True`, 2,538 rows): `deceptive ⟺ knows` exactly — a pure
  knowledge probe scores 1.0 while detecting no deception;
- **knowledge-matched slice** (`knows=True`, 5,812 rows): `deceptive ⟺ denies` — a pure
  lexical probe scores 1.0.

**Finding: HP-KR as published cannot distinguish a deception probe from a knowledge probe.**
This is why Phase 2 builds reference directions from outside the dataset.

### 0.3 The inversion is layer-dependent

Zero-shot AUROC on HP-KR, paper16 bank, mean pooling
(`run-coords/FULL_L20.json`, `run-coords/FULL_L40.json`):

| direction | layer 20 | layer 40 |
|---|---:|---:|
| deception contrast | 0.251 | 0.644 |
| PC1 | 0.290 | 0.668 |
| sycophancy | 0.042 | 0.162 |
| unified | 0.058 | 0.447 |
| contrast, assistant-residualised | 0.324 | 0.561 |
| Assistant Axis | 0.837 | 0.246 |

**At layer 40 the deception axis is not inverted.** Any claim that "the persona axis inverts
on HP-KR" must name a layer. Note also that the Assistant Axis at L20 (0.837) sits close to
the lexical bit (0.870) — consistent with it tracking the denial wording.

### 0.4 Trained transfer at layer 40 — the failure to fix

`run-crossgen/matrices/mean_L40_s42.json`, k=5, mean pooling, paper probe (C=1), 80/20
design (diagonal = own 20% split, off-diagonal = 100% of the column dataset).

| feature | HP-KR as target | HP-KR as source | diagonal |
|---|---:|---:|---:|
| raw | 0.668 | 0.683 | 0.974 |
| raw (lr_reg10) | 0.713 | 0.695 | 0.968 |
| `pc:deception` | 0.621 | 0.704 | 0.813 |
| **`axes`** | **0.409** | **0.271** | 0.802 |
| `axes_resid` | 0.324 | 0.214 | 0.886 |
| **random 5-d** | **0.693** | 0.656 | 0.798 |
| length only | 0.623 | 0.596 | 0.771 |
| `datapca` | 0.643 | 0.563 | 0.965 |

A random subspace beats the persona axes in both directions. A probe *trained* on HP-KR in
persona coordinates anti-transfers everywhere (0.271).

---

## Phase 1 — prompts and personas (2026-09-19)

Built by `kd/scripts/kd_personas.py --write`. Verified:

- 30 questions; the first 20 byte-identical to `pc_questions_shared.jsonl`.
- 36 role files, each carrying all 30 questions.
- 8 groups, 5 explicit contrasts in `kd_role_groups.json`.
- 5,400 rollouts planned (36 × 5 variants × 30 questions).

10 new personas in three groups (`knowledge_denial` 4, `sincere_ignorance` 3,
`plain_assertion` 3) and 10 new questions whose object of belief is the speaker's own
knowledge. Rationale in `DESIGN.md` §5.

**Revised 2026-09-19 after a self-review pass**, before any compute:

1. **Question balance was fatal and is fixed.** All 10 new questions stipulated "I knew / I did
   / I had", so a `sincere_ignorance` persona had no gap to admit and answered identically to a
   `plain_assertion` one — collapsing the two honest groups and destroying the wording-matched
   contrast. Now 6 "knows, admitting is costly" + 4 "genuinely does not know, admitting that is
   costly". The four inverted questions also elicit the *other* lie, a false claim of knowledge.
2. **`sincere_ignorance` rewritten as dispositions**, not accuracy policies, so the group
   produces "I don't know" language by temperament on any question.
3. **`candid_unknowing` → `honest_novice`** — grammatical, and an exact mirror of
   `feigned_novice`.
4. **Redundant questions dropped**: three variants of "prior awareness" became one; the
   expertise pair became one; the thinnest question (no stated cost) was replaced.
5. **Eight awkward instruction variants rewritten** — phrasings like "what you have never
   forgotten" that cost judge-score-3 rates.

Verified after regeneration: 36 role files, 30 questions with the first 20 byte-identical to
`pc_questions_shared.jsonl`, every file carrying all 30, 50 new variants at 71–97 characters
(mean 84), 8 groups, 5 contrasts, every grouped role backed by a file.

**Status: out for external review before any NDIF run.**

---

## Phase 2 — factor probes

_not started_

Gate: `d_utterance` must score ≈1.0 on the knowledge-matched slice and ≈0.5 on the
text-matched slice; `d_knowledge` the reverse.

## Phase 3 — persona run

_not started_

## Phase 3.5 — gates added after external review (2026-09-19)

Two checks raised by an outside review of the v1 packet, both adopted because this project has
prior measured evidence that each matters. Neither existed in the plan before.

- **Assistant-Axis drift balance.** `6_axes.py:265` already prints each group's mean projection
  onto every axis. Record `knowledge_denial` vs `sincere_ignorance` on `assistant`, plus
  `cos(knowledge_denial_axis, assistant)`. Imbalance means the contrast is re-learning persona
  drift, which is exactly what Q1 measured for paper16 (−0.66 → −0.37).
- **Per-group response length.** Compute token-length distributions per group from
  `run/responses/*.jsonl`. Denials are short by construction; length alone already transfers at
  0.553 on this corpus. If imbalanced, rebuild on a length-matched subsample as a robustness
  check.

## Phase 4 — axes

_not started_

Check: `cos(PC1, own contrast)` ≈ +1 per new pair at layer 40. The shared paper16 run
reached 0.896–0.952; a low value means the question set is dominating again.

## Phase 5 — 11 × 11 cross-generalisation

_not started_

## Phase 6 — attribution

_not started_
