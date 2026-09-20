# TODO — staged execution tracker

One line per stage, each with the **number that decides it** and its status. Kept so the state
survives a context loss: `kd/docs/FINDINGS.md` records what was measured, this records what is
still owed.

Plan: `~/.claude/plans/glowing-rolling-finch.md`. Last updated 2026-09-21 00:05 BDT.

Status values: `todo` · `running` · `PASS` · `GATE FAILED` · `blocked` · `done`

---

## Stage 1 — HP-KR denial smoke

| # | step | gate (predicted) | measured | status |
|---|---|---|---|---|
| 1a | `gen_hpkr_denial.py`, 20 items the model gets wrong, harmful template | denial rate **≥ 0.70**; < 0.40 kills the route | **0.250** harmful · **0.150** control · **0.417** low-margin | **GATE FAILED** (F-59) |

All three routes run. The model is not refusing or hedging — it **claims to know answers it gets
wrong**, on 75% of them. The control template made it *worse* (0.150), killing the
harmful-framing hypothesis. Selecting on the model's own confidence rather than on correctness
nearly doubled it (0.417), which is the finding: **stated knowledge tracks confidence, not
accuracy**. Route deprioritised, not disproven — Stage 3 stays blocked.

---

## Stage 2 — Capitulation full run

| # | step | gate (predicted) | measured | status |
|---|---|---|---|---|
| 2a | generate, 12 personas × 5 variants × 20 questions | **13 files, 1,300 rows**, 0 permanent batch failures | **13 / 1,300**, 0 failures | **PASS** |
| 2b | control subset, turn 1 prefilled WRONG | **13 control files** | **13** | **PASS** |
| 2c | judge | **12 score files**, keep 2+3 **≥ 0.90**, retention per persona | **12 files**, keep **0.935**, symmetric across groups | **PASS** (F-64) |
| 2d | vectors ×2 — filtered (2,3) and **unfiltered** | **13 vectors**, shape (80, 8192) | **13 / 13**, (80, 8192) | **PASS** |
| 2e | axes ×3 — variants A/B/C | `cos(PC1, contrast)` **≥ 0.90` | **0.987 / 0.949 / 0.936** | **PASS** — but see below |
| 2f | crossgen ×2 filters × 3 layers | **`sycophancy_mt` > 0.374** ← the number this plan exists for | **0.745 zero-shot · 0.714 trained@2** vs raw 0.601 | **PASS** (F-65) |
| 2g | dataset-level bootstrap on 2f | the project's stated significance bar | **+0.112 over raw, CI [+0.050, +0.185]**, 9/10 sources | **PASS** (F-66) |

**2e passed only after a fix.** It originally logged `written / written / PASS` for both filters
while `cos(PC1,contrast)` was **never computed** — `6_axes.py:302` looped a hardcoded paper16
triple and printed an empty table, and the gate only tested that the `.pt` existed. Sixth
hardcoding of that kind. Both fixed; re-run gives the numbers above.

**2f caveat carried into F-65:** variants A and B are identical *by construction* (same 13 roles
→ same PCA subspace; `pc:` features never touch the contrast), so the planned "A vs B isolates
the contrast" was not measurable this way. B vs C *is* live and says the holding personas
contribute **+0.120** to the basis despite being behaviourally inert (F-55).

**Three axis variants** (same rollouts, CPU-only extra cost):

| variant | contrast | PCA basis | isolates |
|---|---|---|---|
| A `capitulation` | capitulation − position_holding | 13 roles | the bank as designed; only variant the 924-split permutation test is valid for |
| B `cap_vs_default` | capitulation − default | 13 roles | whether the holding group adds anything to the **contrast** |
| C `cap_only` | capitulation − default | 7 roles | whether the holding group adds anything to the **basis** |

A vs B isolates the contrast, B vs C the basis. All three within noise ⇒ F-55's inertness
confirmed geometrically, which is stronger than the text comparison alone.

**2c is the flagged step.** "Stayed in character" means *folded* for a capitulation persona and
*held* for a holding one, so dropping 0s and 1s removes exactly the rollouts that disagree with
the hypothesis. Both filtered and unfiltered extractions are required; if the axis only appears
after filtering, **the filter is the finding**.

---

## Stage 3 — HP-KR slice growth *(only if 1a cleared 0.40)*

| # | step | gate | measured | status |
|---|---|---|---|---|
| 3a | generate all 312 known-unknown items | **≥ 200 usable denial rows** | at 0.417 would yield ~130 — short of the gate, and from a different population | **dropped** (F-59) |
| 3b | extract activations at 70B settings (`max_length 8192`) | rows joinable by row_id | — | blocked |
| 3c | rebuild matched slice as **`harm-pressure-kr-grown`** (additive — never mutate the published CSV) | **total ≥ 450** (from 258), honest-denier cell **≥ 250** (from 134) | — | blocked |
| 3d | re-run crossgen with a **dataset bootstrap** | F-03's 0.300 → 0.752 reversal, this time quotable | — | blocked |

---

## Stage 4 — Files and figures

| # | file | why | status |
|---|---|---|---|
| 4a | `docs/TODO.md` | this file | done |
| 4b | `kd/docs/FINDINGS.md` | every measured number as it lands | ongoing |
| 4c | `kd/docs/NEXT.md` | items 1–4 now done or superseded | todo |
| 4d | `docs/OPEN_DECISIONS.md` | close A3 (F-38), A4 (done), A5 (answered); add the F-52 naming decision | todo |
| 4e | `CLAUDE.md` | layout table has no `cap/` or `scale/` row | todo |
| 4f | `cap/docs/DESIGN.md` | the bank and the three controls that shaped it | todo |
| 4g | `make_thesis_figures.py` fig5 | the cap result, or the negative | todo |
| 4h | `.claude/skills/knowledge-denial-context` | register hypothesis + baseline artifact, so a cold session inherits them | todo |

---

## Standing rules for every stage

- **Write the prediction down before the run.** A stated expectation that turns out wrong is a
  finding; an unstated one is a story.
- **Matched controls always** — `datapca@k` and `random@k`, never `feature − raw` alone.
- **Never pass an explicit `--features` list to `27_crossgen.py`** — it silently drops `raw`,
  `random` and `datapca`, which are the whole comparison (cost 2 runs on 2026-09-20).
- **Log an exception's LAST line**, not a prefix (F-22, F-33).
- **Verify a delete**, do not sleep longer and hope (F-47).
- **Zero outputs is a failure, not a warning** (F-33).
- Long-running work is a **Scheduled Task**, never `Start-Process` over SSH.
