# Phase M — the methods track

_Written 2026-09-23 on the owner's instruction: design a proper phase for the methods work, run it
with agents, and **make sure these things end up in the final paper**._

## Why this is its own phase

Phase 2 checks *claims* — "is this number right?". The methods work asks a different question:
**what did this project build, where did it come from, did it work, and who designed it?** The
answers live in different evidence (git history, source files, design docs, the replicated paper)
and feed different sections of the thesis (methods, limitations, future work, contributions).

Running it as a track alongside Phase 2 rather than after it, because **none of it is blocked**:
the three gaps `audit/METHODS_INVENTORY.md` records can all be closed now.

## The three gaps, and why none of them needs the remaining 523 claims

| Gap | Thought to need | Actually needs | Cost |
|---|---|---|---|
| Several inventory entries are **[LEDGER]**, not claim-checked | the rest of Phase 2 | **8 specific entries** — 18 of the 26 method-bearing ones are already done | **one batch** |
| The **extraction-vs-paper** comparison is inferred, not read line by line | — | one agent reading `2_activations_ndif.py`, `6_axes.py`, `8_coords_ndif.py` against Eq (1)–(4) | **one agent** |
| The **attribution list** | Phase 3 | git history on `asix` + his branch + the draft — all available now | **one agent** |

## Workstreams

### M1 — close the method-claim gap
The 8 unchecked method-bearing entries: **F-42, F-43, F-44** (the NDIF attention-memory rule —
rows × heads × T², a per-process cap, and what truncation cost), **F-46, F-51** (teacher forcing:
why it was needed and that it works), **F-106, F-107** (the sarcasm Gate 3 — under-specified, then
failed), **F-108** (leave-one-CLUSTER-out and the row-level leak).
Standard Phase 2 procedure (`audit/phase2/CLAIM_INSTRUCTIONS.md`), lead QA on 2 of 8.

### M2 — extraction vs the paper, line by line
Read the pipeline against Mahadik & Skapars Eq (1)–(4) and the 13 ambiguities in
`docs/papers/persona-coordinates-notes.md` §8. Produce a divergence table: **for each equation,
what the paper says · what our code does · whether that is a resolution of an ambiguity, a
deliberate deviation, or an accidental one.** This is what turns "our extraction may differ" into a
statement a reviewer can check.

### M3 — the attribution list
For every design, control and baseline in `METHODS_INVENTORY.md` Part 2: **first appearance in git
history on `asix`** (commit, date, author), whether it exists on `origin/feat/persona-pc-ood`, and
whether the draft or his notes name it. Output: a table with three columns — *ours / his / joint* —
each cell carrying a commit hash. This is what a joint thesis needs and what the draft currently
lacks.

### M4 — assemble (lead, after M1–M3)
Turn the inventory plus M1–M3 into the two sections the thesis needs:
`audit/METHODS_CHAPTER.md` (methods, with lineage and every number sourced) and
`audit/LIMITATIONS_AND_FUTURE_WORK.md` (each failure → what it limits → what would settle it).

## Making sure it reaches the paper

The brief's Phase 5 lists only `REPORT_PLAIN.md` and `REPORT_TECHNICAL.md`, both results-oriented.
**On the owner's instruction, Phase 5 is amended to four deliverables:**

1. `audit/REPORT_PLAIN.md` — as the brief specifies
2. `audit/REPORT_TECHNICAL.md` — as the brief specifies
3. **`audit/METHODS_CHAPTER.md`** — lineage, designs, controls, baselines; every claim marked
   [AUDITED]/[LEDGER] so nothing unverified reaches the thesis unlabelled
4. **`audit/LIMITATIONS_AND_FUTURE_WORK.md`** — the failures, what each limits, what would settle
   it, and the decisions still open (the 11→14 grid; sarcasm v2)

`audit/DOC_FIXES.md` and `docs/DRAFT_CORRECTIONS.md` remain the two correction lists.

**This amendment is recorded so a future session cannot drop it:** it is in this file, in
`audit/PROGRESS.md`, and in the resume block under `## Next`.

## The standing framing for the thesis

> The controls are the largest methodological contribution **precisely because they are what kills
> this project's own headlines.** The matched dataset-PCA control, the dataset-level bootstrap, the
> bag-of-words falsification gate and the format-matched control each removed a result this project
> wanted to keep. A replication that only confirmed things would not have needed any of them.

And the second framing, for the failures:

> A design that was built, run and failed its pre-registered gate is **evidence**, not a gap. The
> sarcasm bank failed two gates and produced the project's sharpest statement about what a
> deception axis actually reads; the HP-KR growth gate failed and established that stated knowledge
> tracks confidence rather than correctness.
