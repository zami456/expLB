# Audit plan — Phases 1–5

_Written at the end of Phase 0 (2026-09-22). The brief's phases stand. Where I suggest a change, I say why. At the end there is ONE list of questions only you can answer._

## What Phase 0 showed that shapes the plan
- **Four sources, not two.** They are this laptop, his branch, the lab PC and Google Drive. The lab PC is the only complete result store. Drive is a copy of it, with no deletions. This laptop has only part of the results.
- **Your branch and his share no file in git.** Every dependency between you went through files outside git, so matching versions has to be done by content hash (md5), not by git.
- **Many numbers exist only in prose.**
  - 23 of your analysis scripts only print to the screen.
  - ~130 analysis scripts sit loose on the lab PC, outside git.
  - Most of his CSVs depend on activations that exist only on his disk.
- **The claim load is large.** There are 109 F-entries, 237 draft rows, ~1,000 numbers in his notes, ~40 memory notes and other docs. Many of these repeat the same number. After merging repeats, I expect **~250–350 distinct claims** (INFERRED).

## Suggested changes to your method (and why)
1. **Census first (Phase 1).** Build one file-level index of all four sources: path, size, time and md5. On Drive the md5 comes free with `rclone lsjson --hash`. For activation (`.pt`) files, record only name and size. Then orphans, duplicates, ghosts and "same name, different content" fall out mechanically instead of by reading. *Why:* your collaboration crossed machines, and hashes are the only reliable link.
2. **Two levels of checking in Phase 2.**
   - **Level 1:** "the doc's number equals the number in the result file".
   - **Level 2:** "the result file can be recomputed from lower-level raw data".
   - Record both. *Why:* for much of his work only Level 1 is possible without his files. Mixing the two levels would overstate how sure we are.
3. **Importance-first order, and triage the F-list.**
   - Do the claims the paper/draft relies on first, not F-01 → F-109.
   - Many F-entries are engineering notes (e.g. F-14 scp truncation, F-21 NDIF memory caps, F-35 throughput). I'd check those only where they affect data validity, e.g. F-47 (a stale file un-sampled a dataset) and F-104 (fixes never reached the lab PC).
   - *Why:* this gets you the answers that matter soonest if the audit has to stop early.
4. **Never copy with `scp`.** One committed file is already cut at exactly 200 KiB (RF-5). Copy from Drive with `rclone copy`, then compare every copied file's md5 with the lab PC original, computed there read-only.

## Phase 1 — Inventory (facts only). Size: 2–3 sessions
1. **Census** → `audit/census/{laptop,pcood,labpc,drive}.csv` + `census_joined.csv`.
   - Lab PC: one read-only PowerShell pass that prints path/size/mtime, plus md5 for every json/csv/jsonl/log/txt/yaml. That is ~210k files, mostly small.
   - Also md5 the lab PC's loose scripts and compare them with the repo. F-104 says some fixes never reached the lab PC (DOC-ONLY), so the code that actually ran may differ from git.
2. **Cache** result-type files into `audit/cache/drive/…` with `audit/cache/MANIFEST.csv` (source path, size, md5, lab-PC md5, match?). Activations stay where they are.
   - Measured on Drive (VERIFIED, `rclone size`, 2026-09-22):
     - result-type files: **272,325 files, 3.29 GB**;
     - `.pt` activations: **4,752 files, 83.6 GB**.
   - Everything result-type would fit your 5 GB limit. **But Drive is slow per file** (a memory note says ~97 files/min for uploads; DOC-ONLY). At that rate, 272k files would take days.
   - So I will copy only what the claims need: summary, significance and matrix JSONs, logs and small CSVs. The ~191k per-pair CSVs of the crossgen grids get hashed in the census but stay where they are, unless a claim needs them.
   - Where many files are needed, I'll test streaming a tar archive over SSH (`ssh … tar -cf - <folder>`). That writes nothing on the lab PC, and I'd verify every file by md5 after unpacking. I'll ask before using it for anything large.
3. **`audit/experiments.csv`.**
   - Sources: the 31 scheduled-task names and their driver scripts, the 184 lab logs (their first and last timestamps are the best "when did it run" evidence), `phase0/scripts_to_outputs.csv`, and his `summary.md` "Reproduce" blocks.
   - Status comes from logs and `.done` markers.
   - Dependencies come from md5 lineage, e.g. which persona-vector file a run read.
4. **Orphans, ghosts, duplicates and version conflicts**, derived from the census.
5. **`audit/papers.csv`**: the 11 PDFs, the 8 notes, and papers cited in docs or code but missing (4 PDFs already have no notes).

## Phase 2 — Claim ledger. Size: the biggest phase, ~8–15 sessions in batches of ~10 claims
- **Build the claim list:**
  - every F-entry;
  - the draft's 237 rows;
  - his notes and commit messages;
  - your other docs;
  - memory Q1–Q11;
  - `thinking/plan.md`.
  Merge repeats of one number into a single claim, and keep its history. Claim families come from "CORRECTION to F-xx" links plus the git history of each doc; he also corrected 3 claims after pushing.
- **The ledger's history is partial.** Despite "append-only", 7 F-entries were edited in place (F-65–F-68, F-89, F-93, F-94); `phase0/fnumber_history.csv` lists them. F-01–F-66 entered git in a single commit, so their earlier drafts are gone. Older numbers survive only in other docs and memory: for example, REPORT.md and memory still say 0.870 where F-01 says 0.913.
- **Checking each claim:**
  1. Find the script and the result file behind the claim.
  2. Recompute the number with a small script in `audit/scripts/claims/`.
  3. Record Level 1 and Level 2, the sample size, the seeds and the spread.
- **Print-only scripts:** re-run them read-only. Before running any script I read it to confirm it writes nothing. Where it needs activations, run it on the lab PC with its own Python, printing only, if you allow it (question 6).
- **Order:**
  1. The draft's headline claims and the F-entries the draft or the collaborator briefing lean on (F-85, F-97–F-99, F-65/66, F-79–F-81, F-63/F-100–F-103).
  2. The rest of the scientific F-entries.
  3. His notes.
  4. The engineering notes that affect validity.
- **Output:** `audit/claims.csv` and one note per claim in `audit/claims/<id>.md`.

### How his claims will be checked (your decision, 2026-09-22)
You said the only thing he made himself is the activations he downloaded (about 62 GB per layer). Everything else, including every persona bank, came from your runs with Claude. We will not ask him for files. So each of his claims gets one of three levels:

1. **Always: his text against his committed data.**
   - Every number in the draft and in his notes is compared with the CSV/npz files committed on his branch. I read them with `git show`, so there is no checkout and no copy outside `audit/`.
   - Result: the number matches his data, or it doesn't. This is the "read the real data" check you asked for.
2. **Recomputed where git allows.**
   - The leave-one-out AUROCs can be recomputed from `loo_scores.npz`, which contains both scores and labels.
   - Tables that combine other tables can be recomputed from their inputs: summaries built from matrices, averages, "wins on every test set".
   - His hard-coded Liars' Bench baseline numbers (`persona_ood.py` lines 65–92) are checked against the Liars' Bench PDF.
3. **His inputs checked against yours.**
   - His persona vectors and axes are your files (the 36-persona bank, `paper16_L20`, `deceptive_L20`). I compare them with your originals on the lab PC and Drive, by md5 and by numbers.
   - This answers RF-1: which version of your kd bank he used.
   - It also checks the 0.78 vs 0.971 point.

Anything that needs his activations can't go beyond level 1. The ledger will say "matches his data; not recomputable (activations not available)". It will not say "holds". Where the draft states a number about **your** results, it is checked against your raw data like any other claim.

### Phase 2 rules learned in Phase 1 (added 2026-09-22 after Phase 1)
- **Start from `audit/phase1/fmap_all.csv`.** It gives every F-number's source file and key, the producing script, the copy status and the version notes. Recompute on the lab PC. Staged helpers:
  - `labpc_run.py` + `p1_labpc_runpy.ps1`: runs with the working directory in scratch and no bytecode writes;
  - `p1_peek.py`;
  - the agents' UTF-8/UTF-16 readers.
- **Use the right version.**
  - Where the lab file was regenerated after the F-entry (RF-13), compare on keys (never by list position).
  - Summaries: the old records are unchanged (VERIFIED on 3 files).
  - significance.json: raw statistics are unchanged, but BH q-values move with the size of the test family. So a quoted q must be checked against the version it was computed on, usually the git copy.
- **Print-only numbers** (42 F-entries) are recomputed by re-running the script read-only on the lab PC, or by re-implementing it in `audit/scripts/claims/` when the script writes files or is lab-only.
- **Priority order:**
  1. The draft topics and the new HIGH flags: RF-16 (F-100–F-102 bank mix), F-39's dataset set (now matches the paper; report sensitivity), F-34/fig2 (3,276-test file is a ghost), F-04's q, F-95's 55 vs 51, F-61's label, F-65's sign.
  2. The remaining scientific F-entries.
  3. His notes and the draft's 237 rows, against his branch's data.
  4. The engineering notes that affect validity.

## Phase 3 — Cross-checks and gaps. Size: 2–3 sessions
- **Settings you both ran.**
  - The clearest case: your `4d0d046` says it reproduced his `ext_axis1` contrast "exactly". Check that on raw numbers.
  - HP-KR and the L20/L40 zero-shot tables are the next ones to compare.
  - His personas are your files. So when you and he disagree on the same bank, dataset and layer, the likely causes are narrow: his downloaded activations (which examples, which token positions, which pooling) or his probe code. That makes disagreements traceable.
- **Stale foundations**, via md5 lineage. RF-1 (which instruction version made his 36-persona bank) is the first case.
- **The draft**: what it misses, overstates or gets wrong. It cites no F-number and omits your ~100 findings.
- **Weak spots**: single seeds, missing baselines, small samples, crashed runs, leaks.
- **Results nobody followed up.**
- **`audit/TIMELINE.md`**: built from git plus the lab logs. Note that commit dates here are batch dates, not run dates (09-20 has 51 findings but no commit).

## Phase 4 — Independent check. Size: 1–2 sessions
- Fresh agents get only the brief, the ledger row and the raw files, and write their own scripts.
- They check every HIGH-importance claim plus a random ~20% of the rest.
- I report the mismatch rate and log each correction in `CORRECTIONS.md`.

## Phase 5 — Reports. Size: 1–2 sessions
- `REPORT_PLAIN.md`, `REPORT_TECHNICAL.md` and `DOC_FIXES.md`, as the brief specifies.

## Risks
- **The lab PC is not frozen.** Five scheduled tasks still fire every 15–30 min (Drive copies only; all compute chains report "done"). Someone could still launch a run. *Mitigation:* stamp every reading with a time and md5, and re-check key files before the final report.
- **Drive keeps old versions.** A copy never deletes, so a file renamed or deleted on the lab PC can survive on Drive as a "ghost". The census will flag it.
- **His activations are not available** (your decision: don't ask). Numbers derived from them get level 1 only: "matches his data". That is a weaker check than "recomputed", and the ledger will say so.
- **Hidden code.** Results may come from loose lab-PC scripts that differ from git, or from scripts not found anywhere (`make_view.py` is on the lab PC only).
- **My own tooling.** The shell tool collapses doubled backslashes in scripts; it broke two of my scripts in Phase 0 (caught and fixed). The same kind of bug put a stray control byte into one of your `.ps1` files (RF-6). I write file paths with forward slashes, or build the backslash with `chr(92)`.
- **Volume.** Many sessions. Everything goes into files, and `PROGRESS.md` is updated after each batch.

## Your answers (2026-09-22) and what they change
- **Target = the joint work.** You set everything up; then you each ran tests, and he doesn't know about yours.
  - "Importance for the paper" is judged for the joint paper.
  - Phase 3 gets a dedicated section: **what the draft misses from your side**. That means your tested claims, positive and negative, that bear on his text.
- **Scope = `asix` + `origin/feat/persona-pc-ood` only.** Mahdi's branch and the July branches are out. The original Liars' Bench results (upstream, in `results/`) are background: I check only the numbers the draft or your findings quote from them.
- **Where the data is:** your results on the lab PC (Drive = backup); his results on his branch; nothing elsewhere.
- **Compute on the lab PC.** The laptop lacks the RAM (one Phase 0 scan was killed for low memory). So:
  - the Phase 1 census runs on the lab PC; Drive is used only for hashes;
  - recomputations that need your persona vectors or activations run there too, with its own Python;
  - I read each script first to confirm it only reads;
  - the laptop runs only light scripts on small copied files.

## Questions only you can answer (one list)
1. **Project in one line.** Is the map's description right? Which is the target: his paper draft, your thesis, or both? This decides the "importance for the paper" column.
2. **The draft (partly answered).** You confirmed `main (1).pdf` is his draft. Still open: is it his latest version? F-85 and F-88 criticise an earlier version titled "One Direction Generalizes". If you have that earlier PDF, it would let me track which claims changed. If not, that's fine.
3. ~~His files~~ **Answered:** don't ask him; check his claims against his branch's data (see "How his claims will be checked").
4. ~~Scope~~ **Answered:** asix + persona-pc-ood only.
5. ~~New experiments during the audit~~ **Answered:** none by you or him. The lab PC is shared with other teams, so I stay inside `C:\Users\user4\`.
6. ~~Running code on the lab PC~~ **Answered:** yes, run everything there.
7. ~~Other places~~ **Answered:** lab PC + Drive backup + his branch, nothing else.
8. ~~Scratch folder on the lab PC~~ **Answered:** yes. It will be `C:\Users\user4\audit_scratch\`, with every creation recorded in `audit/LABPC_WRITES.md`.

All questions are answered.

Question 2: the earlier version was a title and abstract, not a draft. I recovered its exact text from the Claude session where you pasted it (2026-09-21) and saved it as `audit/phase0/collab_abstract_2026-09-21.md`. Its numbers become claims too; some differ from the PDF, e.g. 0.882/0.912 vs 0.919.

**Unawareness runs both ways,** so Phase 3 covers both directions:
- (a) your tested claims that his draft misses or contradicts;
- (b) his tests and numbers that your docs miss or restate wrongly, e.g. the briefing and F-97 reading his files.
