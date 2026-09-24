# PHASE 4 — independent re-derivation. Instructions for every agent.

You are re-deriving a batch of claims about a research project **from raw data only**.

**Read `audit/BRIEF.md` first.** Then this file. Then do your batch.

---

## What you are doing, and what you are NOT doing

An audit has already checked most of these claims. **You are not reading its verdicts.** Your job is
to open the raw files yourself and say what the number actually is. Afterwards the lead compares your
answer to the audit's and counts the disagreements. That comparison is the entire point, and it only
works if you never see the prior answer.

**Therefore you MUST NOT open any of these — they contain the prior verdicts:**

- `audit/claims.csv`
- `audit/claims/` (any file)
- `audit/DOC_FIXES.md`
- `audit/phase4/sample.csv` (the unblinded sample — yours is `audit/phase4/batches/<BATCH>.csv`)
- `audit/PROGRESS.md`, `audit/phase3/`, `audit/CORRECTIONS.md`, `audit/qa_*`, `audit/REPORT_*`
- any other `audit/` file that states a verdict about a claim

**You MAY use these — they are computed data, not conclusions:**

- `audit/phase2/layer_sensitivity.csv` — per-(store, layer, seed, pooling, probe, feature, dataset) AUROCs
- `audit/phase2/controls.csv` — zero-shot nulls and trained-control rows
- `audit/phase2/provenance.csv`
- `audit/cache/` — result files copied off the lab PC, with `MANIFEST.csv`
- `audit/phase1/fmap_all.csv` and `audit/experiments.csv` — where things live (a map, not proof)
- the repo working tree, git history (`git show`, `git log`, `git diff` — **read-only**)
- the lab PC, read-only

If you accidentally see a prior verdict, **say so in your `notes` column** for that row. It is not a
disaster; an unreported one is.

---

## Evidence rules (from the brief — these are absolute)

1. **Raw data is the only source of truth.** Prose — findings docs, the draft, memory notes,
   CLAUDE.md, paper notes — is a map of what to check, never proof.
2. Anything you think you already know about this project is a hypothesis until you open a file.
3. **Every number you report must come from a file you opened or a script you ran**, with its path
   and the key/row/line inside it. Computed beats read-by-eye: prefer a small Python script.
4. **"NOT-FOUND" and "I don't know" are good answers.** A flagged gap helps. A gap filled with a
   guess does real damage.
5. Judge a file by its content, not its name. `final` in a filename means nothing.

---

## Safety (absolute)

- **Write only inside `audit/phase4/`.** No edits to any existing doc, script or result. No fixes.
- **Git: read-only.** `git show`, `git log`, `git diff`, `git ls-tree` are fine. Never commit, push,
  checkout, switch, pull, merge, rebase, reset, restore, clean, stash, or `git submodule update`.
- **Lab PC** (`ssh -o BatchMode=yes user4@100.85.217.68`, Windows): **read-only commands only**.
  The only writable place is `C:\Users\user4\audit_scratch\`, and every write there must be logged in
  `audit/LABPC_WRITES.md`. Never touch the owner's files. If ssh asks for a password or hangs, stop.
- **rclone**: only `lsf`, `lsjson`, `size`, `cat`, `check`, and `copy` FROM the remote INTO
  `audit/cache/`. Never sync/move/delete/purge/dedupe.

---

## The traps

Every one of these is verified and has already cost this audit time. Read them before you start.

- **Check the number at the configuration the claim states, not at the project default.** This is the
  single cause of every one of the audit's nine self-corrections so far. If a claim says layer 20,
  check layer 20 — do not report layer 40 because layer 40 is the default. `run-coords/` in the
  working tree holds 37 result JSONs at settings the crossgen stores do not use.
- **Never quote a count or an effect without its metric, its sign convention and its dataset set.**
  Nine self-corrections came from this one habit.
- **A block label is never a dataset count.** In the kd11-era stores `block "9"` = **10 datasets** and
  `block "10"` = 11; in the `run-crossgen` ROOT store `block "10"` = 10.
- **A bank filename never settles which bank was used.** `s3` != `s23`. Two different persona banks
  have been stored under the same basename (`axes_vectors_s3.pt`). If a bank matters, fingerprint it:
  same bank + same activations => identical zero-shot AUROCs.
- **BH q depends on the test family, and the family grew in a re-run wave on 2026-09-22 — this has
  already flipped a real verdict.** If a claim says something "clears" or "does not clear" a
  BH-corrected threshold, state the family size and which copy of the file you read (git vs lab).
  Identical raw statistics can carry different q.
- **`raw` and the matched `datapca` control must be joined from the ROOT store.**
- **"0.374" is the deception axis's PC1, not the sycophancy axis (which reads 0.313).**
- **A trained random probe is not a zero-shot random direction.** They are different controls.
- **The published paper uses k=1 for deception and k=3 for sycophancy. k=5 appears nowhere in it.**
  This project's default is k=5.
- **In `passA` / `passC` / `<root>`, seeds 0-4 carry a reduced feature set** (no `datapca`, `random`,
  `length`), so any control comparison in those stores is **seed-42-only**.
- **Git and the lab PC hold different versions of the same result files.** Record which copy you read.

---

## Where the data is

| Store | Where |
|---|---|
| `run-coords/` (37 result JSONs, Q-series) | **repo working tree** — check here before deferring anything |
| `run-crossgen/` (matrices, summary, significance, kd11) | repo working tree (partial) + lab PC (full) |
| `run-paper16/`, `run-hf275/`, `run-cap/`, `kd/run/`, `cap/run/` | repo working tree (partial) + lab PC |
| `run-kd/`, `run-scale/`, `run-premise/`, `run-cap-ctrl/`, `logs/` | **lab PC only** (some cached) |
| lab main store | `C:\Users\user4\expLB\` on `user4@100.85.217.68` |
| cached lab files | `audit/cache/labpc/` (see `audit/cache/MANIFEST.csv`) |
| collaborator's results | `git show origin/feat/persona-pc-ood:src/probes/deception-detection/results/...` |
| the draft PDF | `full paper draft for this experiments/main (1).pdf` on branch `asix`; extracted text at `audit/phase0/draft_text.txt`, page images at `audit/phase0/draft_pages/` |
| papers | `papers/*.pdf` |

Lab helper: `python audit/scripts/labpc_run.py audit/scripts/p1_labpc_runpy.ps1 FILE=p1_peek.py "ARGS=<path> <mode> [arg]"`
(modes: info/head/cat/keys/get/grep/ls/pt). If `labpc_run.py` exits 1 with no output, fall back to a
direct `ssh ... powershell -NoProfile -Command "..."`.

---

## Your output

Write **exactly one file**: `audit/phase4/rederive_<BATCH>.csv`, with this header:

```
id,rederived_value,evidence_path,evidence_key,my_verdict,my_confidence,notes
```

One row per claim in your batch, same ids, in any order.

- **`rederived_value`** — the number(s) YOU computed, with the metric and configuration named.
  e.g. `AUROC 0.6436 (deception axis, zero-shot, L40, mean pooling, hp-kr, seed 42)`.
  If the claim is qualitative, state what you found in one short sentence.
- **`evidence_path`** — the file you opened, repo-relative or an absolute lab path. Several separated
  by `;`. Say **git** or **lab** where both copies exist.
- **`evidence_key`** — the key, row or line inside that file. Be specific: `records[314].auroc`,
  `line 3873`, `row seed=42,layer=40,feature=pc:deception`.
- **`my_verdict`** — one of:
  - `HOLDS` — the claim's number and its sentence both reproduce
  - `PARTLY` — the number reproduces but the sentence around it is wrong, overstated or misscoped

  **VERDICT DISCIPLINE — added 2026-09-23 after batches F1, F2, F3 and O1 had already run
  (see `audit/CORRECTIONS.md`; those four batches did NOT receive this paragraph):**
  `HOLDS` means the number reproduces **and** the sentence built around it is accurate in scope,
  sign, metric, comparator and dataset set. If the number reproduces but the surrounding sentence
  overstates it, names the wrong comparator, omits the configuration it depends on, or would
  mislead a reader who trusted it, that is `PARTLY` — not `HOLDS`.
  **Your `notes` and your `my_verdict` must not contradict each other.** If you write in the notes
  that a claim's wording is wrong or misleading, the verdict is `PARTLY`.
  - `DOES NOT HOLD` — the number does not reproduce, or the claim is contradicted by its own data
  - `NO RAW DATA` — the file the claim needs does not exist anywhere you can reach
  - `CANNOT RESOLVE` — you found data but could not decide; say why in `notes`
- **`my_confidence`** — `HIGH` / `MEDIUM` / `LOW`, **followed by the reason** in the same cell.
- **`notes`** — anything the lead needs: which copy you read, a competing reading, a trap you hit, a
  configuration the claim left unstated, or a prior verdict you saw by accident.

Quote fields containing commas. Use `python`'s `csv` module to write it if that is easier.

**Do not write any other file.** No `.md` notes, no per-claim files. One CSV.

---

## When you finish

**When your last claim is written, hand back immediately. Do NOT re-read, re-verify, count or audit
your own output files — the lead verifies every batch on merge, and duplicating that wastes the
session budget.**

In your hand-back message give, in at most 15 lines: the batch name, the count by verdict, the 2-3
rows you are least sure about, and anything that looked like a real problem with the project (not with
the audit). Nothing else.
