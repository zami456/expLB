# The owner's Claude session transcripts are an audit source

_Started 2026-09-22, on the owner's instruction: "claude and i have built and made changes in the dataset, claude and i made all the dataset runs, we should have more than him, so please try to find those."_

## Why this exists
Much of the owner's work never reached a file:
- 42 of the 109 F-entries quote numbers that were **only printed to the console** (`script_writes_file = n` in `audit/phase1/fmap_all.csv`);
- 46 rows of `audit/experiments.csv` have no F-number and appear only in prose;
- several source files were overwritten (RF-13), and status-file gate lines were erased (RF-14);
- the Q1–Q11 question series is written up only in `docs/thesis/persona-coordinates-70b-results.md` and in Claude's memory notes.

The transcripts of the sessions in which that work was done still hold the commands and their console output. They are therefore the best record of **what the owner tested and found**, which is what she needs in order to show her side of the project against her collaborator's draft.

## Where they are
`C:/Users/SURAIYA/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB/*.jsonl`
- 19 sessions, ~140 MB, 2026-09-09 → 2026-09-22.
- Largest: `b035ccb3…` (33 MB, to 09-21), `6d4b2181…` (30 MB, to 09-17), `dc605d01…` (11.7 MB, to 09-18), `40b4a5dd…` (8.4 MB, 09-12), `1ac8bb19…` (6 MB, 09-22).
- `4f0b290f…` is the audit session itself; `1a88b065…` its sidecar. Skip both as sources.
- A second project folder, `c--Users-SURAIYA-Documents-thesis-liar-bench-llama-liars-bench/`, holds only a `memory/` directory (no transcripts).
- Format: JSONL, one record per line — `type` (user/assistant/system), `timestamp`, `message` whose content is a string or a list of blocks (`text`, `tool_use`, `tool_result`). **Printed numbers live in `tool_result` blocks.**

## Rules for using them
1. **DOC-ONLY, always.** A transcript is model and user text, not raw data. Nothing from it is verified. Every number found there is a **lead**: Phase 2 must recompute it from result files.
2. **Stream** the files line by line and prefilter with cheap substring tests. Never load a transcript into memory, never paste large chunks into a report. The laptop is short on RAM.
3. Treat them as the owner's private working record: they stay inside `audit/`, and nothing from them is published or sent anywhere.

## Outputs (written 2026-09-22/23 by two mining agents)
In `audit/phase1/sessions/`:
- `sessions_index.tsv` — one row per session: span, record counts, topics, whether it drove the lab PC.
- `session_experiments.csv` — test events, each marked against `audit/experiments.csv` or **NEW**.
- `dataset_events.csv` — dataset builds and changes (the 11th dataset via `make_view.py`, kr-matched views, the HP-KR top-up, sycophancy_mt, cap and sarcasm banks, question sets, persona instruction edits).
- `findings_printed_numbers.csv` — for each print-only F-number, the console output that produced it, if it survives.
- `findings_unrecorded.csv` — conclusions stated in sessions that are in no doc.
- `findings_rerun_origin.md` — evidence on what triggered the two 2026-09-22 re-runs (RF-13).
- `SESSIONS_NOTES.md`, `FINDINGS_NOTES.md` — method, commands, caveats.

## How to re-run
The helper scripts are `audit/scripts/p1_sess_*.py` and `p1_sessf_*.py`; each notes file lists the exact commands.
