# `kd/` — knowledge-denial persona axis

Self-contained experiment: build a persona axis for **deception by denying knowledge one
has**, and measure whether any change it produces on `harm-pressure-knowledge-report`
(HP-KR) is deception, knowledge, or wording.

Everything for this experiment lives here — personas, questions, scripts, rollouts, vectors,
axes, results, figures, docs. Nothing in `run-coords/`, `run-paper16/`,
`run-paper16-shared/` or `run-crossgen/` is modified.

**Read `docs/DESIGN.md` first.** It has the problem, the hypothesis, why the honest side is
split in two, and the pre-committed readings. `docs/REPORT.md` is the running results log.

---

## Why this exists, in three numbers

- HP-KR trained persona-coordinate probe at layer 40: **0.409** as target, **0.271** as
  source — while a **random** 5-d subspace gets **0.693 / 0.656**.
- A single binary feature, "does the response deny knowledge?", scores **AUROC 0.870** on
  full HP-KR. Most of what any probe reads there is that one bit.
- `deceptive = knows AND denies`, and the cell *doesn't know but claims it does* has **zero**
  rows — so knowledge, wording and deception are perfectly confounded inside HP-KR and the
  reason behind any jump has to be measured from outside it.

## Layout

```
kd/
  scripts/
    kd_personas.py           emits the 36 persona files + the 30-question set   [done]
    make_review_packet.py    -> docs/REVIEW_PACKET.md, pasteable into any chat  [done]
    make_review_page.py      -> docs/review_page.html, shareable web page       [done]
    build_factor_probes.py   teacher-forced kd_knowledge / kd_utterance corpora
    kd_axes.py               axes over the new group pairs
    kd_attribution.py        the decomposition in DESIGN.md §6
    run_kd_chain.ps1         generate -> judge -> vectors -> axes -> sync
    register_kd_task.ps1     Scheduled Task registration
  personas/
    kd_instructions/         36 role JSON (26 carried from paper16 + 10 new)    [done]
    kd_role_groups.json      8 groups, 5 explicit contrasts                     [done]
  questions/
    kd_questions_shared.jsonl  30 questions (20 carried verbatim + 10 new)      [done]
  run/                       responses, scores, vectors_*, axes_*.pt, factor_probes
  results/                   matrices, attribution JSON, slice tables
  figures/
  docs/                      DESIGN.md, REPORT.md
```

## The bank

36 personas × 5 instruction variants × 30 shared questions = **5,400 rollouts**.

| group | n | role |
|---|---|---|
| `deception` | 8 | carried from paper16 unchanged |
| `honesty` | 7 | carried from paper16 unchanged |
| `sycophancy` | 5 | carried from paper16 unchanged |
| `anti_sycophancy` | 5 | carried from paper16 unchanged |
| `assistant_anchor` | 1 | `default` |
| **`knowledge_denial`** | **4** | `amnesiac_witness`, `deniable_expert`, `stonewaller`, `feigned_novice` |
| **`sincere_ignorance`** | **3** | `candid_unknowing`, `calibrated_reporter`, `scrupulous_witness` |
| **`plain_assertion`** | **3** | `forthright_expert`, `candid_witness`, `plain_dealer` |

Five contrasts are declared explicitly in `kd_role_groups.json` (`contrasts`), so axis
construction does not have to infer pairings:

| axis | definition |
|---|---|
| `deception` | deception − honesty (unchanged from paper16) |
| `sycophancy` | sycophancy − anti_sycophancy (unchanged) |
| `unified` | (deception+sycophancy) − (honesty+anti_sycophancy) (unchanged) |
| **`knowledge_denial`** | feigned − sincere ignorance — both sides deny, so it cannot be a wording artifact |
| **`kd_assertion`** | feigned − plain assertion — the register contrast, kept separable |

All 36 answer the same 30 questions. The first 20 are `pc_questions_shared.jsonl` verbatim;
the last 10 are new, and are the only ones in the project whose object of belief is the
speaker's **own knowledge** (admit vs deny, rather than disclose vs omit).

## Rebuilding the bank

```bash
python kd/scripts/kd_personas.py            # dry run
python kd/scripts/kd_personas.py --write    # writes questions + 36 role files + role groups
```

Deterministic and idempotent: the 26 carried files are re-copied from
`src/assistant_axis_ndif/pc_instructions/` with only their question list replaced, so the
bank stays in sync with paper16 if that ever changes.

## Status

> **UPDATE 2026-09-24 (P17).** This status is stale since 2026-09-19. Phases 2–5 have run: factor
> probes (F-05), the persona run and axes (`run-kd`, `vectors_s3`), the 11×11 cross-generalisation
> (kd11; F-31, F-34, F-97…F-99), and the 3B/8B rungs (F-121). Phase 6 (attribution) — kd's own
> phase, not the audit's Phase 6 — was not checked by P17 and is left as written.
> **Source:** `kd/docs/FINDINGS.md` F-05, F-31, F-34, F-97…F-99, F-121; `audit/p17/C_REPORT.md`
> (F-121, F-126 rows; `run-kd/vectors_s3` is lab-only evidence).

| phase | state |
|---|---|
| 1 — prompts and personas | **done, out for external review** |
| 2 — factor probes (`kd_knowledge`, `kd_utterance`) | not started |
| 3 — persona run on NDIF (5,400 rollouts) | blocked on review |
| 4 — axes | not started |
| 5 — 11×11 cross-generalisation | not started |
| 6 — attribution | not started |
| 7 — documentation | in progress |

Long-running work goes in a **Windows Scheduled Task**, never `Start-Process` over SSH —
OpenSSH takes the process tree down with the session.
