# Phase 3 — instructions for a workstream agent

Read `audit/BRIEF.md` (fully), the `## RED FLAGS` section of `audit/PROGRESS.md`, and this file.
Phase 3 is **cross-checks and gaps**. Phases 0–2 and M established the facts; your job is to put
them together and find what nobody has looked at.

## The one thing the owner cares about most

She and her collaborator **are not aware of each other's tests and findings** — her words, and it
runs **both ways**. Phase 3 must produce that list explicitly, in both directions:
- **(a)** her tested claims, positive *and negative*, that his draft misses or contradicts;
- **(b)** his tests and numbers that her docs miss or restate wrongly.

## What is already established — use it, do not redo it

| you need | it is here |
|---|---|
| 231 claims checked with verdicts and evidence paths | `audit/claims.csv` (+ `audit/claims/<id>.md` per claim) |
| every claim ever inventoried, checked or not | `audit/claims_inventory.csv` (692 rows) |
| 56 confirmed documentation errors | `audit/DOC_FIXES.md` |
| corrections already delivered to him | `docs/DRAFT_CORRECTIONS.md` |
| the draft's full text | `audit/phase0/draft_text.txt` (19 pp) |
| 8 draft-vs-ledger touchpoints, DOC-ONLY | `audit/phase0/draft_vs_ledger_preview.md` |
| who designed what, with commit hashes | `audit/methods/ATTRIBUTION.md` |
| method lineage, controls, baselines | `audit/METHODS_CHAPTER.md`, `audit/METHODS_INVENTORY.md` |
| failures and what would settle them | `audit/LIMITATIONS_AND_FUTURE_WORK.md` |
| 114 experiments with status and dependencies | `audit/experiments.csv` |
| orphans / ghosts / duplicates / conflicts | `audit/phase1/anomalies.{md,csv}` |
| F-number → evidence map | `audit/phase1/fmap_all.csv` |
| what the owner tested that lives only in transcripts | `audit/phase1/sessions/` + `audit/SESSION_MINING.md` |
| off-diagonal means per store/layer/pooling/seed/feature/probe | `audit/phase2/layer_sensitivity.csv` (78,516 rows) |
| store → bank → k provenance | `audit/phase2/provenance.csv` |
| lead QA records | `audit/phase2/qa_wave5.md`, `audit/phase1/verify/` |

**Anything in `audit/claims.csv` with a verdict is VERIFIED — cite it, do not recheck it.**
Anything not in it is **DOC-ONLY** and must be tagged as such. 461 of 692 claims are unchecked;
the unchecked ones are concentrated in the draft body, his notes and the memory notes.

## Evidence rules (from the brief, unchanged)

1. Raw data is the only source of truth. Prose is a map, never proof.
2. Every number you report carries its path and key/row/line.
3. Tag every statement: **VERIFIED · DOC-ONLY · IMAGE-ONLY · CONFLICT · NOT-FOUND · INFERRED**.
4. "Not found" and "I don't know" are good answers. A flagged gap helps her; a guess hurts her.
5. If two sources disagree, show both. Never pick one silently.

## Safety (absolute)

- Write **only** your own output file under `audit/phase3/` and scripts under `audit/scripts/phase3/`.
- Git is READ-ONLY (`git log`, `git show`, `git diff` fine). Never commit, checkout, switch, stash,
  reset, pull, merge, rebase.
- Lab PC (`ssh -o BatchMode=yes user4@100.85.217.68`): read-only analysis only, via
  `audit/scripts/labpc_run.py` + `p1_labpc_runpy.ps1`, or `p1_peek.py`. Never write outside
  `C:/Users/user4/audit_scratch/`; log any write in `audit/LABPC_WRITES.md`. Heavy work runs THERE.
- No new model calls, no NDIF, no judging, no training. CPU recomputation only.
- After any interrupted remote call, check for orphan processes before re-issuing it.
- If ssh or rclone hangs or asks for a password: stop and report.

## Traps that have already cost this audit time (all verified)

- `block "9"` in the kd11 era = **sycophancy_mt excluded = 10 datasets**; `block "10"` = 11. In the
  run-crossgen ROOT store `block "10"` = 10. A block label is never a dataset count.
- A bank **filename never settles which bank was used** (`axes_vectors_s3.pt` exists in several
  folders). Use the recorded fields or a zero-shot AUROC fingerprint (RF-16).
- BH q-values depend on the test family, which grew in the 2026-09-22 boot-trigger wave (RF-13).
- `kd/run/knowledge/scores.jsonl` in git is **truncated** at 204,800 B (RF-5). Use the lab copy.
- `raw` and the matched `datapca` control must come from the **ROOT** store — a bank store carries
  its own misleading `datapca` column.
- Some lab logs are **UTF-16** (PowerShell `*>` redirect).
- Mean vs last pooling flips several conclusions. Always state pooling, layer, k and seed.
- **Never quote a count without its metric and its dataset set** — that is what DF-49 got wrong.

## Output

One markdown file, the path given in your task. Structure it for a reader who knows the project's
goal but not its jargon: short sentences, main point first, define terms on first use, keep the
numbers. Each item: **what was tested (1 sentence) → what happened (key number, what it means) →
how sure, and why → why it matters → evidence (ids and paths, last)**.

Return a summary of **at most 12 lines**: what you found, the 3 most important items, anything the
lead must verify, and **nominate the 2 findings you are least sure of for the lead's independent QA**.
