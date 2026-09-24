# Phase 2 — the claim ledger: design and workflow

_Written 2026-09-23, after Phase 1 and the fix round. Supersedes the shorter Phase 2 section of `audit/PLAN.md`._

## What Phase 2 must produce
`audit/claims.csv` (one row per claim) + `audit/claims/<id>.md` (one note per claim, with the script output), covering:
1. the 110 F-entries in `kd/docs/FINDINGS.md`;
2. the collaborator's draft (237 extracted rows) and his earlier abstract;
3. his 17 result notes;
4. the owner's other documents — `docs/thesis/*`, the collaborator briefing, the roadmap, `kd/docs/REPORT.md`, memory notes Q1–Q11;
5. the 35 findings recovered from sessions that are in no document.

After merging duplicates into claim families I expect **300–400 distinct claims**, of which perhaps 150 are numeric headline claims and the rest supporting or process statements.

## The two levels, and the honesty rule
- **Level 1 — the document matches the file.** The number in the text equals the number in the result file it points at.
- **Level 2 — the file is right.** The result file is recomputed from lower-level raw data (activations, rollouts, scores).

Every row records both. A claim that reaches Level 1 only is written as "matches its file; not recomputed", never as "holds". By the owner's decision, everything resting on the collaborator's activations is Level 1 by definition.

## Per-claim procedure (the claim card)
1. **State it in one plain sentence**, with metric, dataset(s), layer, bank, pooling, k and seed. If those are missing from the text, that is itself a finding (`UNDERSPECIFIED`).
2. **Locate the evidence** — start from `audit/phase1/fmap_all.csv`, but re-check the path; Phase 1 was mapping, not proof.
3. **Level 1 check** against the file.
4. **Level 2 recompute** with a script in `audit/scripts/claims/<id>.py`, run read-only on the lab PC where the data lives. Never trust a number read by eye.
5. **Version check** (this is where Phase 1's red flags bite): which copy (lab / git / Drive), was it regenerated on 09-22 (RF-13), does the claim quote a BH q-value whose test family changed (RF-13), which persona bank and which vector version (RF-12, RF-15, RF-16).
6. **Cheap robustness probes**, recorded even when the claim does not mention them: other layers, other seeds, the other pooling, and the matched control. A claim that only survives at one layer or one seed is reported that way.
7. **Verdict**: HOLDS · PARTLY HOLDS · DOES NOT HOLD · REVISED → id · NO RAW DATA · CONFLICTING DATA · NOT TESTED; plus confidence (with the reason) and importance for the paper.

## The three issues that get their own workstream
**W1 — layer selection (issue 5).** A session says "L40 was picked because it maximised the persona advantage". Rather than treat that as one claim, I compute the headline metrics at **every stored layer** for the main banks and dataset sets, once, into `audit/phase2/layer_sensitivity.csv`. Every layer-specific claim then carries a sensitivity line automatically: its value at L20/L32/L40/L44, the best layer, and how much L40 flatters it. The paper gets one honest paragraph instead of an accusation.

**W2 — provenance (issue 8).** Claims like "F-18 used the old bank" currently rest on file names. Rule for Phase 2: **a name is never sufficient**. Resolve each claim's inputs by (a) fields recorded inside the result file, (b) the launch command recovered from the session transcripts, or (c) the zero-shot fingerprint method that settled RF-16 (same bank + same activations ⇒ identical zero-shot AUROCs). Output: `audit/phase2/provenance.csv`, one row per result store → bank file, vector version, dataset version, split set.

**W3 — print-only numbers (issue 2).** 42 F-entries quote numbers that were never written to a file; the session mining recovered 72 with their commands. Those recoveries are **pointers, not evidence**: each one gets its computation re-run read-only on the lab PC, or reimplemented. Where the inputs are gone, the verdict is `NO RAW DATA` and the transcript is cited as provenance only.

**W4 — the controls.** Several zero-shot claims rest on a random-direction null that exists only in a console log, and on matched controls. These get recomputed once, centrally, so every claim that leans on them cites the same re-derived numbers.

## Batching and quality control
- Claims are worked in **batches of 10**, by subagents, 2–3 in parallel; each batch writes its rows plus notes and returns a short summary.
- **Rolling QA:** for every batch I re-derive 2 of the 10 myself, from raw files, before accepting it. Mismatches send the batch back.
- `audit/PROGRESS.md` is updated after every batch (the brief's rule), with counts like "claims checked 37/350".
- The brief's Phase 4 stays: at the end, fresh agents re-derive every HIGH-importance claim plus a random 20%, with no access to our notes.

## Order of work
1. **The draft's eight topics** and everything they depend on — this is the owner's stated priority: F-97–F-99, F-85, F-79–F-81, F-63 and F-100–F-102 with the new F-110, F-65–F-67, F-13, the HP-KR series F-01–F-09, F-39, plus the draft rows that touch them.
2. W1, W2 and W4 (they are shared inputs, so they run early and once).
3. The remaining scientific F-entries.
4. The draft's other rows and his notes, at Level 1 against his branch data.
5. The engineering entries, only where they affect validity (e.g. F-47's stale file, F-104's undeployed fixes, RF-14's guard).

## Effort
About 30–40 batches of 10 claims. With 3 agents in parallel and my QA between waves, that is roughly **5–8 hours of wall-clock work**, spread over 2–3 sessions, plus the final Phase 4 pass. W1/W2/W4 add about an hour of compute on the lab PC.

## What Phase 2 will not do
- It will not verify the collaborator's pipeline (his activations are gone — the owner accepted this).
- It will not re-run NDIF generation or judging: no new model calls, only CPU recomputation from stored activations, vectors and rollouts.
- It will not edit the owner's documents. Corrections are proposed in `audit/DOC_FIXES.md`; the only exception so far is F-110, which the owner asked for explicitly.
