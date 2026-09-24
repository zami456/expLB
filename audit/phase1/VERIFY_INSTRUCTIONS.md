# Mini independent check of Phase 1 (instructions for verification agents)

Read `audit/BRIEF.md` (fully), then the RED FLAGS of `audit/PROGRESS.md`, then this file. **Do not read `audit/phase1/fmap_all.csv`, `audit/experiments.csv`, `audit/phase1/anomalies.csv` or `audit/phase1/sessions/*` for your assigned items** — those hold the answers you are checking. If you open one by accident, say so in your row.

## Why
Phase 1's tables were produced by subagents. Before Phase 2 builds on them, we measure how often they are right. You re-derive a sample **from raw files only** and I compare your answers with what Phase 1 recorded.

## Your inputs
- `audit/phase1/verify/sample.csv` — the items (a seeded random sample plus two high-importance ones). Take only the rows your prompt assigns.
- For an F-number: the ledger entry in `kd/docs/FINDINGS.md` (find it by heading; the line range is in `audit/phase0/fnumber_history.csv`).
- Raw data: the lab PC (read-only), git (`git show`), and the census files under `audit/phase1/labpc/` (these are facts about files, not conclusions — using them is fine).
- Lab access: `python audit/scripts/labpc_run.py audit/scripts/p1_labpc_runpy.ps1 FILE=p1_peek.py "ARGS=<path> <mode> [arg]"`; for UTF-16 logs use `audit/scripts/p1_fmap_F082-F109_sgrep.ps1`. Lab python runs from the scratch folder with bytecode writes off. **Never write on the lab PC**; never modify anything outside your own output file.

## What to do per item
1. **Locate the evidence yourself**: which file holds the number, and where inside it (key path, row, or the log line). Do not assume the ledger's cited path is right — check it.
2. **Recompute the number** from that file with your own small script (put it in `audit/scripts/p2v_<item>.py`). For a print-only claim, re-run the producing script read-only on the lab PC, or reimplement its computation.
3. **Record the verdict** against what the ledger/table asserts:
   - `CONFIRMED` — same number (say the tolerance), same file, same key;
   - `CONFIRMED-DIFFERENT-SOURCE` — the number is right but lives somewhere else than stated;
   - `DIFFERS` — the number does not reproduce (give both values);
   - `NOT-RECOMPUTABLE` — the inputs no longer exist (say which);
   - `UNCLEAR` — the claim is too vague to test (say why).
4. Note the **version** you used (lab / git / Drive), since they differ for some files (RF-11, RF-13, RF-15, RF-16).

## Output
`audit/phase1/verify/verify_<your batch name>.csv`, one row per item:
`item, source_table, ledger_or_table_claim (short), my_located_file, my_key_path, my_value, stated_value, verdict, tolerance, version_used, script, minutes_spent, notes`

Plus `audit/phase1/verify/verify_<batch>.md`: what you had to guess, anything ambiguous in the claim itself, and any *new* problem you noticed while working (do not chase it — just note it).

Return a ≤8-line summary: items done, verdict counts, and the items where Phase 1 looks wrong.
