# Phase 2 — instructions for a claim batch

Read `audit/BRIEF.md` (fully), the RED FLAGS of `audit/PROGRESS.md`, `audit/PHASE2_PLAN.md`, then this file.
You are checking a batch of ~10 claims. **Facts only: a verdict must come from a file you opened or a script you ran.**

## Safety (unchanged)
- Write only your batch outputs (`audit/claims/<id>.md`, your rows, scripts in `audit/scripts/claims/`).
- Git is read-only. Never commit, checkout, stash, reset.
- Lab PC: read-only analysis only, through `audit/scripts/labpc_run.py` + `p1_labpc_runpy.ps1` (scratch cwd, no bytecode) or `p1_peek.py`. **Never write outside `C:/Users/user4/audit_scratch/`.** After any interrupted remote call, check for orphan processes before re-issuing it.
- No new model calls (no NDIF, no judging). CPU recomputation only, from stored activations, vectors, rollouts and scores.
- Laptop RAM is limited: stream big files, never load a census whole.

## Inputs per claim
`audit/claims_inventory.csv` gives id, source, where it is stated, the numbers and any layer/dataset hints.
`audit/phase1/fmap_all.csv` is a **map, not proof**: use it to find candidate files, then re-check the path yourself.
Shared results to reuse rather than recompute: `audit/phase2/layer_sensitivity.csv` (W1), `provenance.csv` (W2), `controls.csv` (W4) once they exist.

## Steps
1. **Restate** the claim in one plain sentence, naming metric, dataset(s), layer, bank, pooling, k and seed. Missing specifics → note `UNDERSPECIFIED` and test the most favourable reading, saying which.
2. **Locate** the file that should hold the number; record path + key/row.
3. **Level 1** — does the document's number equal the file's number? Give both and the tolerance.
4. **Level 2** — recompute from lower-level data with your own script (`audit/scripts/claims/<id>.py`), run where the data lives. If the inputs are gone, say `NO RAW DATA` and name what is missing.
5. **Version** — which copy did you use (lab / git / Drive)? Was the file regenerated on 2026-09-22 (RF-13)? Does the claim quote a BH q-value (its test family changed)? Which persona bank and vector version (RF-12, RF-15, RF-16)? A file name alone never settles the bank — use the recorded fields, the launch command from the transcripts, or the zero-shot fingerprint.
6. **Robustness, where cheap** — the other pooling, other seeds, neighbouring layers, and the matched control. Record even when the claim does not mention them.
7. **Verdict** — HOLDS · PARTLY HOLDS · DOES NOT HOLD · REVISED → id · NO RAW DATA · CONFLICTING DATA · NOT TESTED. Then confidence (HIGH/MED/LOW + why) and importance for the paper (HIGH/MED/LOW).

## Outputs
1. Append rows to your batch file `audit/phase2/claims_batch_<name>.csv` with columns:
`id, claim_plain, stated_numbers, source_ref, evidence_file, evidence_key, level1_value, level1_match, level2_value, level2_method, version_used, robustness, verdict, confidence, importance, script, notes`
2. One note per claim: `audit/claims/<id>.md` — the claim, what you did, the script's actual output (pasted), and the verdict with its reason.
3. Return a ≤8-line summary: verdict counts, the claims that failed or changed, and anything that needs the lead's attention.

## Rules of thumb
- Quote numbers to the precision the source uses; state the tolerance you accepted.
- If a number reproduces only under one reading (one layer, one seed, one dataset set), the verdict is PARTLY HOLDS and the note says which.
- If the document and the file disagree, report both and do not pick a winner.
- If you find something outside your batch, note it in your summary; do not chase it.
