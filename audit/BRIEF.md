# AUDIT BRIEF — read all of this before doing anything

## Details (fill in what you know; "?" means: find it, then confirm with me)
- Project in one line: ?
- My branch: asix · Collaborator's branch: feat/persona-pc-ood
- Lab PC: ssh alias ? · results path(s) ?
- rclone: remote ? · path(s) ?
- Collaborator's draft report: ?
- Papers folder: ?
- Ask me before downloading more than: 5 GB in total
For any "?", look in ~/.ssh/config, `rclone listremotes`, and the repo's scripts/configs/docs, then confirm with me.

## The situation
My collaborator and I ran 100+ experiments over a long time, many through earlier Claude sessions. It was messy: when a claim didn't hold we changed it and re-ran, sometimes several times; he built experiments on my results and I built on his. Neither of us knows anymore exactly what was run, what came out, and which claims still stand. He pushed the result CSVs from his experiments to his branch.

Earlier Claude sessions wrote docs: a findings file, a report, memory notes (find them all; I don't remember exact names). Findings are numbered like F-112. My collaborator drafted a report from his side; I'm sure it's incomplete. The repo has the papers we followed (PDFs) and Claude-written notes on them; some papers may be missing. The repo has several large submodules.

I know very little about the details of this repo, so I'm relying on you to be thorough and honest. Take as many sessions and tokens as you need. Being right matters much more than being fast.

## Goal
A complete, verified picture of what we did and found, so no strong result and no real problem gets overlooked: where every experiment and result is → every claim checked against raw data → what the draft report misses or gets wrong → a report I can understand.

## Evidence rules (most important)
1. Raw data is the only source of truth: result files (json, csv, npy/npz, logs) and the code, configs and git history that produced them. Everything in prose (findings, report, memory notes, CLAUDE.md, your auto memory, paper notes, the draft) may be stale or wrong. Use it as a map of what to check, never as proof.
2. Anything you believe you already know about this project is a hypothesis until you check it in a file during this audit.
3. Every number you report must come from a file you opened or a script you ran during this audit, with its path (and key/row/line). Prefer small Python scripts in audit/scripts/ that read raw files and print the numbers; computed values beat values read by eye, and I can re-run them.
4. Tag every statement: VERIFIED (checked in raw data) · DOC-ONLY (only in prose) · IMAGE-ONLY (only a plot exists) · CONFLICT (sources disagree: show both, never pick one silently) · NOT-FOUND · INFERRED (your interpretation: say why).
5. "Not found" and "I don't know" are good answers. A flagged gap helps me; a gap filled with a guess hurts me.
6. Judge an experiment by its script, config and outputs, not its name. "final" in a filename doesn't mean final: work out which version is current and show why.
7. For heatmaps and plots, find the data and script behind each important figure and check the key values.
8. Don't summarize a paper you haven't opened during this audit.

## Safety (some of this data exists in only one place)
- Write only inside audit/ (plus the Setup files below). Don't edit, move or delete anything else: no code fixes, no edits to existing docs. No new experiments or training; light scripts that read existing results are fine.
- Git: `git fetch` is fine. Never commit, push, checkout/switch, pull, merge, rebase, reset, restore, clean, stash, or `git submodule update`. Read other branches with `git show` / `git ls-tree` / `git diff`, or a detached worktree under audit/worktrees/. Watch for Git LFS pointer files.
- Lab PC: read-only commands only (ls, find, du, stat, head, cat, md5sum, read-only python). Use `ssh -o BatchMode=yes` so it fails instead of hanging at a password prompt.
- rclone: only lsf, lsjson, size, cat, check, and copy FROM the remote INTO audit/cache/. Never sync, move, delete, purge or dedupe.
- Copy only result-type files (json, csv, yaml, logs, arrays under ~200 MB) into audit/cache/<source>/ keeping the folder structure; log source path, size and md5 in audit/cache/MANIFEST.csv. Never copy checkpoints, weights or datasets. For bigger arrays, compute what you need where they live, read-only. Check sizes first.
- If SSH or rclone asks for a password, hangs, or fails: stop and tell me.

## Staying on track
Your context will be compacted many times; anything not in a file will be lost.
- Keep audit/PROGRESS.md current after every meaningful step: done (with counts, e.g. "claims checked 37/112"), next, open questions for me, red flags, which file holds what. At the start of every session and after compaction, re-read this brief and PROGRESS.md before anything else.
- Build tables as files, in batches, never from memory. Don't save audit conclusions to auto memory; they belong in audit/.
- Use subagents for big reading jobs. Each reads this brief first, writes results to its own file under audit/, and returns a short summary.
- In submodules, read only the code our experiments actually call.
- Put red flags at the top of PROGRESS.md the moment you find them: a headline number its files don't support, a bug in an evaluation script, train/test leakage, results built on a file that was later regenerated.
- Do only the phase I ask for, then stop. At the end of each phase, confirm `git status` still matches audit/START_STATE.md.

## Setup (once; skip if audit/START_STATE.md exists)
1. Save date, `git status`, branch, HEAD, `git remote -v`, `git submodule status` to audit/START_STATE.md.
2. Add `audit/` and `CLAUDE.local.md` to .git/info/exclude so they are never committed by accident.
3. Create CLAUDE.local.md in the repo root (if one exists, add at the top): a short "AUDIT IN PROGRESS" block saying to read audit/BRIEF.md and audit/PROGRESS.md first, restating evidence rules 1–3 and the safety rules in a few lines.
4. Add these deny rules to .claude/settings.local.json, merging with anything already there: Bash(git push *), Bash(git commit *), Bash(git checkout *), Bash(git switch *), Bash(git reset *), Bash(git restore *), Bash(git clean *), Bash(git stash *), Bash(git pull *), Bash(git merge *), Bash(git rebase *), Bash(git submodule update *), Bash(rclone sync *), Bash(rclone move *), Bash(rclone moveto *), Bash(rclone delete *), Bash(rclone deletefile *), Bash(rclone purge *), Bash(rclone rmdir *), Bash(rclone rmdirs *), Bash(rclone dedupe *), Bash(rclone cleanup *), Bash(rsync *--delete*)
5. Show me what you changed.

## Phases
### Phase 0: Orientation and plan (no copying, no conclusions)
- Map the repo: where code, configs, results, docs and papers live; the submodules, and which commit each branch pins them to.
- Find every document on both branches that states findings, plus CLAUDE.md files and your auto memory. Explain the F-numbers: range, gaps, duplicates, which doc is newest.
- `git fetch`; compare asix with origin/feat/persona-pc-ood: where they split, what each has that the other lacks, where his result CSVs are.
- Check access to the lab PC and rclone; list result folders and sizes.
- Rough counts per source: experiments, result files, claims.
- audit/PROJECT_MAP.md: a one-page plain-language map of the project, for me.
- audit/PLAN.md: how you'll do Phases 1–5 (if you see a better way than mine, say so and why), how big each phase is, the risks, and ONE list of questions only I can answer.

### Phase 1: Inventory (facts only)
- audit/experiments.csv, one row per experiment/run across all four sources: id · what it tests (from script/config) · who and when (prefer git history and timestamps inside logs; copying resets file times) · branch + commit · script · config · where its results are · status (complete / partial / crashed / unknown) · what it depends on (whose results it read, which file version).
- Orphans (results no doc mentions), ghosts (results a doc mentions that you can't find), duplicates (same md5 in several places), conflicting versions.
- audit/papers.csv: PDFs present, notes present, papers cited in code/docs/notes but missing (title, where cited; you may web-search to identify them, but don't summarize from memory).

### Phase 2: Claim ledger
- audit/claims.csv, one row per claim: every F-number, plus claims in the report and the draft that have none. Columns: id · the claim in one plain sentence · where stated · in the draft? (y/n) · experiments that test it · key number(s) recomputed from raw data vs the number the doc states · runs/seeds/sample size/spread · status (HOLDS / PARTLY HOLDS / DOES NOT HOLD / REVISED → id / NO RAW DATA / CONFLICTING DATA / NOT TESTED) · confidence (HIGH / MEDIUM / LOW + reason) · importance for the paper (HIGH / MEDIUM / LOW) · evidence paths.
- One notes file per claim: audit/claims/<id>.md, including the script output.
- Claim families: how each claim changed through re-runs (the git history of the docs helps).
- Batches of ~10 claims; update PROGRESS.md after each batch.

### Phase 3: Cross-checks and gaps
- Where my results and my collaborator's disagree on similar settings, and the likely cause (commit, submodule version, config, seed, data).
- Stale foundations: experiments built on a result that was later revised or regenerated.
- The draft report: what it misses, overstates, or gets wrong.
- Bottlenecks and weak spots: single seeds, missing baselines, small samples, crashed runs, leaks, suspicious code.
- Strong or surprising results nobody followed up; negative results worth reporting.
- audit/TIMELINE.md: the project's story in order: what we tried, what failed, what we changed.

### Phase 4: Independent check
Fresh agents that get only this brief, the ledger row and the raw files (not your notes) re-derive every HIGH-importance claim plus a random ~20% of the rest. Report the mismatch rate, fix the ledger, and log each fix in audit/CORRECTIONS.md.

### Phase 5: Reports
- audit/REPORT_PLAIN.md, for me: the big picture in ~10 sentences; strongest results, ranked; claims that failed or changed and what we learned; red flags and bottlenecks; what the draft misses or gets wrong; promising leads; decisions I need to make, with your recommendation; glossary.
- audit/REPORT_TECHNICAL.md: full detail with tables, paths, commits, scripts.
- audit/DOC_FIXES.md: proposed corrections to existing docs, CLAUDE.md and auto memory. List only; don't apply.

## How to write for me
I know the project's general goal but not the details or all the jargon. Write as you would for a smart colleague from a neighboring field.
- Short sentences. Define each technical term the first time. Main point first.
- Plain doesn't mean vague: keep the numbers, drop the jargon. "Accuracy fell from 81% to 62% on the new test set (3 seeds, 2,000 examples)", not "performance degraded".
- Each finding: what we tested (1 sentence) → what happened (key number and what it means) → how sure we are, and why → why it matters for the paper → evidence (IDs and paths, last).
- Don't sound more certain than the evidence is.
- In chat keep it short; detail goes in files. End every phase with a plain summary of at most 10 lines.
- If I say "simpler", go simpler. If I say "details", go technical.