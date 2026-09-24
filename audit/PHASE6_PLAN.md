# Phase 6 — close the audit's coverage gap

_Written 2026-09-24, at the end of the session that measured the gap. Read
`audit/COVERAGE_GAP.md` first; this plan exists because of what it found._

## Why this phase exists

The audit reported "424 of 692 claims checked". **314** of those 424 are inventory claims; the
other **110** were raised while checking. Real coverage of the inventory is **314/692 = 45%**, and
**378 inventory claims never got a verdict — 176 of them in the draft PDF being submitted**
(`audit/COVERAGE_GAP.md`, DF-94).

Phase 6 checks those 378 and closes four smaller gaps the same measurement exposed.

---

## Order of work — REVISED 2026-09-24 after measuring cost

The original order (6A → 6B → 6C → 6D) was wrong in two ways, both fixed below.

**1. The DOC_FIXES triage goes FIRST, not last.** 95 proposed corrections are unapplied, so the
documents still contain errors the audit already found. Writing prose from them is the one way this
goes wrong. The owner triages; a session does not apply fixes on its own.

**2. 6A interleaves with writing the Results chapter, it does not precede it.** The chapter does
not exist, and 6A's output is its raw material. Re-deriving Table 4 and writing the Results
paragraph about Table 4 is nearly the same work — clear a table, write its text, move on.

**Measured cost basis:** 61 draft claims were already checked using **7 scripts** (8.7 claims per
script). **49 of the 61 read from `origin/paper` with `git show`; 12 from the local repo; none
needed the lab PC.** The draft claims are the cheapest category in the inventory, not the dearest.

**Level 1 vs Level 2 — decide this before starting, it drives the time more than the claim count.**
24 of those 61 verdicts read "MATCHES HIS DATA (Level 1 only)": the number was *located* in a
result file but never re-derived. For a paper being submitted that is not enough on the headline
numbers.

- **Level 2 (re-derive from matrices/rollouts) on 8 batches, 54 claims:**
  **B23** Abstract · **B41** Table 2 (HP-KR) · **B42** Table 3 (mean off-diagonal) ·
  **B43** Table 4 (LB protocol) · **B45** Table 6 (zero-shot 8v7/4v6/12v13) ·
  **B46** Table 7 (raw vs 12v13) · **B40** Table 11 + **B30** App F (sycophancy — F-122 just
  changed what is true here).
- **Level 1 (locate the number, flag mismatches) on the remaining 122.**

**6B and 6C are OPTIONAL.** They check his summary notes and the ledger — neither gates the paper.
If they are never done, say so in the limitations rather than implying coverage was complete.

---

## The work, in priority order

### 6A — the 176 draft claims · **the submission; starts after the 6D-4 triage** · batches B01–B48

The paper's own numbers. 121 of the 176 carry explicit figures. Batched one table / figure /
section per batch so a whole table is re-derived together rather than cell by cell.

The largest: **B43** Table 4 (12), **B42** Table 3 (8), **B44** Table 5 (8), **B38** Table 1 (7),
**B40** Table 11 (7), **B41** Table 2 (7), **B45** Table 6 (7), **B35** Fig 4 left (6),
**B46** Table 7 (6), **B47** Table 8 (6), **B48** Table 9 (6), **B23** Abstract (5),
**B39** Table 10 (5), **B21** Future work (5).

**Table 11 and App F (B40, B30) carry the sycophancy contrast** — check them against F-122, which
found four sycophancy arms on `origin/paper` and `syc_pc1` topping his excl-HP-KR table at L40–56.

### 6B — the 92 collaborator-note claims · batches B49–B55

From his `persona_ood*/summary.md` files. **These were written before `origin/paper`**, so expect
some to be superseded by the 16-layer sweep rather than wrong.

### 6C — the 110 ledger claims · batches B56–B63

`kd/docs/FINDINGS.md` F-entries never re-derived. Lowest priority of the three: the ledger is
append-only and already carries its own sources and caveats.

### 6D — four measurement gaps, all cheap

| # | gap | cost |
|---|---|---|
| 1 | **`pc:sycophancy` has never been trained in our pipeline.** Absent from `27_crossgen.py`'s default `--features`. F-122 showed his branch has four sycophancy arms and one of them is his best non-HP-KR arm — ours has none. | CPU-only, minutes per store |
| 2 | **`passB_fine` (20 matrices) and `passP` (6) reach the audit but never the ledger.** | read + one entry each |
| 3 | **Both reports close at 2026-09-23** and omit F-117, F-119, F-122 (DF-95). | editing |
| 4 | **95 DOC_FIXES, none applied.** Needs the owner's triage, not more checking. | owner's call |

---

## How to check a claim (unchanged from Phases 2–4)

1. Open the claim's `source_ref` — for draft claims, `audit/phase0/draft_text.txt` or the
   per-page files in `audit/phase0/draft_pages/`. The PDF itself is
   `full paper draft for this experiments/main (1).pdf`.
2. Find the file that should contain the number. Open it. **Record path + key/row/line.**
3. Level 1 = does the stated number appear in a result file? Level 2 = re-derive it from the
   matrices/rollouts with a script in `audit/scripts/`.
4. Append a row to `audit/claims.csv` with its existing columns. A batch is done when every row
   has `verdict`, `evidence_file` and `evidence_key`.
5. Verdicts: `HOLDS` · `PARTLY HOLDS` · `DOES NOT HOLD` · `MATCHES HIS DATA` · `UNRESOLVED`.

**The rule Phase 4 measured at a 6.5% error rate, all of it leniency:** if you write a defect into
the notes, it belongs in the verdict too. A `HOLDS` whose note says "but the layer is cherry-picked"
is a `PARTLY HOLDS`.

---

## TODO — in this order

```
[ ] 6D-4  FIRST. Hand the owner the 95 DOC_FIXES as a triage list, grouped by
          priority and by which document they touch. DO NOT apply any without her
          word. Writing starts only after she has ruled on them.

[ ] 6A    The 176 draft claims, interleaved with drafting the Results chapter.
          Clear a batch, write its Results text, move on.
      [ ] LEVEL 2 (re-derive): B23 Abstract · B41 T2 · B42 T3 · B43 T4 · B45 T6
                               B46 T7 · B40 T11 · B30 App F          [54 claims]
      [ ] LEVEL 1 (locate):    B38 T1 · B44 T5 · B47 T8 · B48 T9 · B39 T10
                               B35 Fig 4 left · remaining figures · prose batches
                                                                      [122 claims]

[ ] 6D-1  Train pc:sycophancy in our pipeline — we have none, his branch has four
          arms and one is his best non-HP-KR arm (F-122). Pass --features
          explicitly; NEVER drop the defaults raw/random/datapca/length.
[ ] 6D-2  Read passB_fine + passP; give each a ledger entry or record why not.
[ ] 6D-3  Bring REPORT_PLAIN.md and REPORT_TECHNICAL.md up to F-117/F-119/F-122,
          or head each with its close date (2026-09-23).
[ ] 6E    Rerun p15_coverage.py; update COVERAGE_GAP.md with the new numbers.

OPTIONAL, does not gate the paper:
[ ] 6B    collaborator notes — 92 in B49–B55
[ ] 6C    ledger claims — 110 in B56–B63
```

**Estimate, from the measured 8.7 claims/script:** 6A is ~20 scripts — one focused session at
Level 1 throughout, two to three at Level 2 throughout, about one and a half on the split above.
6D is half a session. **Every table that fails to reproduce becomes an investigation** — F-122
began as a one-line check and ended as a ledger entry. Two or three of those add a session.

Progress goes in `audit/phase6/RESULTS.md`, in the shape of `audit/phase4/RESULTS.md`.

**Where the Results chapter gets drafted.** The write-only-inside-`audit/` rule still binds, so
draft it at **`audit/RESULTS_CHAPTER.md`** — the same place `audit/METHODS_CHAPTER.md` and
`audit/LIMITATIONS_AND_FUTURE_WORK.md` already live. The owner moves it into the paper; a session
does not write into `docs/` or the LaTeX source.

---

## Rules that bind this phase

Unchanged from `CLAUDE.local.md`, repeated because they are easy to lose:

- **Write only inside `audit/`**, plus appending to `kd/docs/FINDINGS.md`. `DOC_FIXES.md` is a
  proposal list — the owner applies, not you.
- **Git: `fetch` only.** Never commit, push, checkout, pull, merge, rebase, reset, restore, clean,
  stash or `git submodule update`. Read branches with `git show` / `git ls-tree` / `git diff`.
- **Lab PC `user4@100.85.217.68`** — read-only on her files, always `ssh -o BatchMode=yes`.
  Output only under `C:\Users\user4\expLB-resplit\`. Log every write in `audit/LABPC_WRITES.md`.
  If SSH or rclone asks for a password, hangs or fails: **stop and tell the owner.**
- **Every number comes from a file opened or a script run**, with its path and key. Prose is a map
  of what to check, never proof.
- **Scope now includes `origin/paper`** — the brief's original two-branch scope is superseded.

## The traps this project has already paid for

- A long lab-PC job must be a **SYSTEM** scheduled task; `/RU user4` without a stored password
  registers interactive-only and **silently never fires** while `schtasks /Run` reports SUCCESS.
  Always verify `Last Run Time` changed.
- `27_crossgen.py` refuses any output root without a path component named exactly `run-crossgen`,
  and its default `DATASETS` omits `harm-pressure-kr-matched`.
- `scp` truncates every file at exactly 204,800 bytes and exits 0. Use `audit/scripts/labpc_put.py`.
- **For a `dim=1` feature the trained off-diagonal equals the zero-shot AUROC** (F-122) — a
  regression on one scalar can only pick a sign. Do not read those matrices as transfer results.
