# Project map (Phase 0) — one page, plain language

_Written 2026-09-22. It shows where things are, not whether results are right. Tags: VERIFIED = I checked it in a file or with git. DOC-ONLY = only prose says so. INFERRED = my reading, with the reason. Detail files are in `audit/phase0/`._

## What the project is (DOC-ONLY + INFERRED — please confirm)
The repo started as **Liars' Bench**, a public benchmark of lie detectors for language models. On top of it, you and your collaborator tested one idea on **Llama-3.3-70B**, run remotely on **NDIF**, a service that lets you read a model's internal activations:

- **The idea:** a lie detector generalises better to new kinds of lies if it reads the model's activations in **"persona coordinates"**. Persona coordinates are a few directions obtained by prompting the model to play honest and deceptive characters ("personas"), instead of the raw 8,192-number activation.
- **Your side:** the replication, then new persona banks aimed at the two datasets where the method failed:
  - *kd* = knowledge-denial personas, for the dataset **HP-KR**;
  - *cap* = capitulation personas, for the dataset **sycophancy**.
- **Your side, also:** smaller-model runs (**3B and 8B**) and many robustness checks.
- **His side:** the same idea tested with the Apollo probe code on 6 datasets, including leave-one-out baselines and a comparison against Liars' Bench's own probes.
  - **You told me (2026-09-22):** every persona bank and everything else he used came from your runs with Claude. The only thing he made himself is the activations he downloaded, about 62 GB per layer; exactly what he downloaded is unknown. His draft (`main (1).pdf`) is confirmed as his.

## Where everything is — four places (VERIFIED)

| Place | What is there | Size |
|---|---|---|
| **This laptop** (`expLB`, branch `asix`, HEAD `43ee6d3`) | All of your code (`src/assistant_axis_ndif/`, `kd/`, `cap/`, `scale/`). Your docs. The papers (`papers/`, 11 PDFs + 8 notes in `docs/papers/`). The draft PDF. **Only partial copies of results.** Missing here: run-kd, run-scale, run-premise, run-cap-ctrl, run-coords passes A/B/C, and most logs. | 9,897 files / 5.9 GB, of which 4.3 GB is the original benchmark |
| **GitHub, his branch** `origin/feat/persona-pc-ood` (34 commits, zami456) | His 19 result folders under `src/probes/deception-detection/results/` (316 CSV). His 17 summary notes. His input vectors, copied from your runs. **His activations and base-probe run files are not in git**; they are on his own disk. | 490 files / ~111 MB |
| **Lab PC** `user4@100.85.217.68`, folder `C:\Users\user4\expLB` | **The main result store**: run-coords (77.6 GB of activations), run-crossgen (202,613 files), and every run-* folder. It also holds ~130 loose analysis scripts not in git, including `make_view.py`, which built a dataset used everywhere. It is not a git repo. | ~85 GB |
| **Google Drive** `tamara:expLB/` (rclone) | A copy of the lab PC results; scheduled jobs still add to it every 15–30 min, copy-only. `share/` holds what was handed to your collaborator. `thinking/` holds a 109 KB `plan.md` and a 09-17 snapshot of Claude's memory. `thesis-docs/` holds a results write-up. | ~87 GB |

Your branch and his **share no file byte-for-byte**. Every hand-over between you two went through files outside git (VERIFIED, `phase0/branch_compare.md`).

## Where claims are written
- **`kd/docs/FINDINGS.md`**, the findings ledger (VERIFIED from git):
  - It runs F-01 … F-109, with no gaps, duplicates, renumbering or deletions. **There is no F-110 or higher anywhere**: not in any branch, any doc or memory. "F-112" appears only in the brief.
  - It was created on 09-21 in one commit that already held F-01 … F-66, so how those first 66 were drafted is not in git.
  - It calls itself "append-only", but 7 entries were edited in place later (F-65–F-68, F-89, F-93, F-94). For example, a number in F-68 went from +0.051 to +0.054.
  - The lab PC holds an older copy, from 09-21.
- **Other docs on your branch:** 25 findings-bearing documents (VERIFIED list in `phase0/docs_inventory.md`). The main ones are:
  - the collaborator briefing, the roadmap, TODO, open decisions and onboarding (`docs/`);
  - the design, report and next-steps files in `kd/docs/` and `cap/docs/`;
  - the thesis sections in `docs/thesis/`.
  - The docs disagree on which file to trust: CLAUDE.md points to FINDINGS.md, ONBOARDING to a plan file plus memory, and `kd/README.md` to `kd/docs/REPORT.md` (CONFLICT).
- **Claude's plan files** in `~/.claude/plans/` (6 files). One of them, `harmonic-cuddling-kay.md` (109 KB), defines the questions Q1–Q11. The lab PC's `thinking/plan.md` is the same file (VERIFIED md5).
- **The draft paper** `full paper draft for this experiments/main (1).pdf`: your collaborator's draft (you confirmed it). It has 19 pages and cites no F-numbers; 237 claim rows are extracted in `phase0/draft_claims_raw.csv`.
- **His earlier title + abstract** ("One Direction Generalizes…", 2026-09-21): recovered word for word from the Claude session where you pasted it → `phase0/collab_abstract_2026-09-21.md`. Your briefing and F-85/F-88/F-97 critique this version.
- **His 17 `summary.md` notes** (about 1,000 numbers) and his commit messages (about 380 more).
- **Claude's auto memory** (45 files, answers to Q1–Q11 and more). The lab PC and Drive keep a 09-17 snapshot of it.
- **Numbering pitfall:** "Claim 3", "Q", "Stage" and "Gate" numbers are each reused with different meanings in different docs (`phase0/docs_inventory.md` §7).

## Rough size of the job (INFERRED counts)

| Source | Experiments | Result files | Claims |
|---|---|---|---|
| Your branch + laptop | ~26–40 result units with a known script | ~8,000 persona-work files (mostly CSV) | 109 F-entries + 24 other docs + 45 memory notes + plan files |
| Lab PC / Drive | the above + ~10 run folders not on the laptop | ~208,000 (191k CSV, 12k JSON, ~5k .pt) | — |
| His branch | 8 experiment families, 19 folders | 421 | ~150–300 distinct |
| Draft PDF | ~11 experiments | 0 (no data attached) | 237 rows |

## First things that already need checking (details: `audit/PROGRESS.md`, red flags)
1. **Which version of your knowledge-denial persona vectors did he use?** His copy may predate the instruction set it is named after, and the headline 0.919 / 0.916 numbers rest on it. I will compare his copy with your originals on the lab PC.
2. **Most of his CSVs cannot be recomputed**, because they come from his own activations. As you decided, I will check his text against the data on his branch instead, and label those numbers "matches his data", not "recomputed".
3. **One of your knowledge-label files was committed cut off** at 200 KiB.
4. **Some thesis figures use hand-typed numbers**, or data that doesn't match the local file.
