# S1 — Branches and where things live

Written 2026-09-24 under `thesis_tools/survey/RULES.md`. Read-only: no fetch, checkout, or any
write outside `thesis_tools/survey/`.

**Snapshot:** remote refs are as of the last `git fetch` (`.git/FETCH_HEAD` mtime
2026-09-24 17:28:54 +0600). I did not fetch again, so the remote may have moved since then.

**Re-run everything with:**

```
python thesis_tools/survey/scripts/s1_branches.py        # -> s1_data/branches.json, divergence.json   (parts 1-2)
python thesis_tools/survey/scripts/s1_untracked_sizes.py # -> s1_data/untracked_sizes.tsv             (part 3)
python thesis_tools/survey/scripts/s1_folders.py         # -> s1_data/folders.json                    (part 4)
git status --ignored --short > thesis_tools/survey/s1_data/status_ignored_short.txt
```

The part-2 drill-downs and part-5 checks were one-off inline commands. Each one is quoted where
its result is used.

---

## 1. Branches

HEAD: **`asix` @ `e917ecd3f95f`**, which tracks `origin/asix` (same SHA). There is one worktree
and one remote, `origin = https://github.com/zami456/expLB.git`. `origin/HEAD` → `origin/main`.

Commands: `git for-each-ref refs/heads refs/remotes`, `git rev-list --count <b>`,
`git log --reverse --format=%cI <b>`, `git shortlog -sne <b> --`.

| Branch | Tip | Last commit (committer date) | Commits | Authors (`git shortlog -sne`) |
|---|---|---|---|---|
| **asix** (HEAD) | e917ecd3 | 2026-09-24 05:16 +06 | 751 | Kretschmar 401+42+24 (3 identities), **Suraiya Shanaz 137 (bracu) + 2 (noreply)**, Laurito 129+14, zami456 2 |
| origin/asix | e917ecd3 | 2026-09-24 05:16 +06 | 751 | identical to asix |
| main (local) | 608b2a78 | 2026-09-09 20:45 +06 | 615 | upstream as above; Suraiya 2 (noreply) + 1 (bracu); zami456 2 |
| origin/main | 59cdaba4 | 2026-09-08 05:55 +00 | 614 | upstream; Suraiya 2 (noreply); zami456 2 |
| origin/paper | 16def2de | **2026-09-24 16:47 +06** | 652 | upstream; **zami456 42** |
| origin/feat/persona-pc-ood | 45b16d90 | 2026-09-22 12:45 +06 | 646 | upstream; zami456 36 |
| origin/feat/llama-layer30 | df758c4b | 2026-09-19 19:32 +06 | 621 | upstream; zami456 10; **Mahdi `<mahdi.ndc21@gmail.com>` 1** |
| origin/feat/resumable-acts-cache | 7c282990 | 2026-09-05 09:04 +06 | 620 | upstream; zami456 10 |
| origin/claude/colab-activation-download-drive | 97aee6a4 | 2026-07-16 21:51 +06 | 630 | upstream; Claude 15; Suraiya 2 (bracu) + 1 (noreply); zami456 2 |
| origin/claude/nnsight-colab-activation-download-f95pop | a005e100 | 2026-07-13 22:12 +00 | 616 | upstream; Claude 2; Suraiya 2 (bracu); zami456 2 |
| origin/claude/assistant-axis-submodule-p285ym | 2b228832 | 2026-09-07 15:36 +00 | 613 | upstream; zami456 2; Claude 1 |
| origin/Saving | 638bb9a0 | 2026-07-12 18:43 +06 | 614 | upstream; Suraiya 2 (bracu); zami456 2 |
| origin/copilot/zami456explb | 2e2a7856 | 2026-07-12 01:47 +06 | 612 | upstream; zami456 2 |

Every branch's first commit is dated 2025-02-18. "Upstream" means the Liars' Bench authors:
Kieron Kretschmar (401 commits as kieron.kretschmar@outlook.com, 42 as k.kretschmar@hotmail.de,
24 as KieronKretschmar) and Walter Laurito (129 + 14). They appear with identical counts on every
branch, so they come from the shared base.

Local `main` is **1 commit ahead** of `origin/main`: `608b2a78` 2026-09-09, Suraiya Shanaz,
"papers info claude skills". It is 0 commits behind (`git rev-list --left-right --count main...origin/main` → `1 0`).

## 2. What each non-HEAD branch holds that HEAD does not

Commands: `git merge-base HEAD <b>`, `git rev-list --count HEAD..<b>` / `<b>..HEAD`,
`git log --format='%an <%ae>' HEAD..<b>`, `git diff --name-status --no-renames <mb> <b>` (grouped by
folder with Python), and `ls-tree -r` set differences.

"Changed on both" means a path changed in both `mb..HEAD` and `mb..branch`.

| Branch | Merge-base with HEAD | Commits only on branch (authors) | Commits only on HEAD | What differs vs merge-base (A=added, M=modified) | Changed on both |
|---|---|---|---|---|---|
| main | 608b2a78 (its own tip) | 0 | 136 | nothing. `main` is an ancestor of HEAD. HEAD's tree lacks `.devcontainer/` (2 files) that main has, so HEAD deleted it | – |
| origin/main | 59cdaba4 (tip) | 0 | 137 | nothing; ancestor | – |
| origin/asix | e917ecd3 | 0 | 0 | identical | – |
| **origin/paper** | 2e2a7856 (2026-07-12) | **40 (zami456 40)**, 2026-09-04 → 09-24 | 139 | `src/probes/deception-detection/`: **A 1176, M 7**. Types: .csv 837, .png 179, .pt 51, .md 33, .py 27, .tex 23, .yaml 18, .pdf 6 | **0** |
| origin/feat/persona-pc-ood | 2e2a7856 | 34 (zami456 34) | 139 | `src/probes/deception-detection/`: A 483, M 7. **It is an ancestor of origin/paper** (`paper` is 6 commits ahead) | 0 |
| origin/feat/llama-layer30 | 2e2a7856 | 9 (zami456 8, Mahdi 1) | 139 | `src/probes/deception-detection/`: A 8, M 5 (acts_cache, batching, tests, layer-30 config, run_llama_all.sh) | 0 |
| origin/feat/resumable-acts-cache | 2e2a7856 | 8 (zami456 8) | 139 | the same folder: A 7, M 5. The same files as llama-layer30 minus one config; llama-layer30 is 1 commit ahead of it | 0 |
| origin/claude/colab-activation-download-drive | 2e2a7856 | 18 (Claude 15, Suraiya 3) | 139 | `notebooks/colab_extract_activations.ipynb` (A); M `.env.example`, `README.md`, 3 files in `src/probes/deception-detection` | 0 |
| origin/claude/nnsight-colab-activation-download-f95pop | 2e2a7856 | 4 (Claude 2, Suraiya 2) | 139 | the same notebook (A); M `.env.example`, `README.md`, 2 probe files | 0 |
| origin/claude/assistant-axis-submodule-p285ym | 2e2a7856 | 1 (Claude 1) | 139 | M `.gitmodules`; adds gitlink `assistant-axis` (hyphen; HEAD has `assistant_axis`) | 1 (`.gitmodules`) |
| origin/Saving | 2e2a7856 | 2 (Suraiya 2) | 139 | M 2 files in `src/probes/deception-detection` | 0 |
| origin/copilot/zami456explb | 2e2a7856 (its tip) | 0 | 139 | nothing; ancestor of HEAD | – |

**What `origin/paper` adds**, from
`git diff --name-status --no-renames HEAD...origin/paper` grouped by two sub-levels under
`src/probes/deception-detection/` (118 groups; the largest are shown):

- **Results**, all under `results/`:
  - `results/persona_ood_ext_L{20,24,…,76,79}`: 16 layers × 47 files.
  - `persona_ood_followup_L20`, `persona_ood_mod_L{20,40,44}`: 43 files each.
  - `persona_ood`, `persona_ood_L40`: 30 each.
  - `loo_L{20,40}`: 8 each.
  - `persona_positions_L{20,40,44}`: 7 each.
  - `persona_depth`: 3.
  - `XXXX-9-llama-all*`: layer-sweep plot folders.
  - `gender_secret`.
- **Paper:** `paper/` holds `sections/` (10 files), `generated/` (17 files) and `README.md`. The
  README (`git show origin/paper:src/probes/deception-detection/paper/README.md`, line 1) says:
  "Lying in Knowledge-Denier Persona Coordinates — LaTeX source of the paper. Every number in the
  text is generated from the committed results in `../results/`".
- **Code:** `deception_detection/`, including:
  - new: `persona_pcs.py`, `acts_cache.py`, `batching.py`, `lora_intervention.py`, `scripts/gender_secret.py`, `scripts/layer_sweep.py`;
  - modified: `activations.py`, `experiment.py`.
- **Vectors:** `deceptive/`. Its README, lines 1–5, says it holds "The 8 deceptive personas from our
  persona-coordinates replication, sliced at layer 20 … Source: `run-paper16-shared`". That is a
  claim in a doc, not verified here.

**Newer than the audit.** Commits after 2026-09-22 22:20 +06, the mtime of the audit maps:

- `asix`: 2 commits (d280d230, e917ecd3; Suraiya).
- `origin/paper`: **4 commits** (ea1ce52b 09-23 23:46, 2ad758fb, d1b88154, 16def2de 09-24 16:47; all zami456). Their subjects include "The persona study at every 4th layer from 20 to 79" and "Correct an overstated HP-KR ranking claim".
- `origin/feat/persona-pc-ood`: 0.

## 3. Ignored and untracked data in the working tree

Command: `git status --ignored --short` gives 39 lines: 11 ` M`, 1 `??`, 27 `!!`. Sizes come from
`s1_untracked_sizes.py`, which uses `os.walk` without following links. Two warnings came up:
`.venv/lib64/` and `.venv-aa/lib64/` could not be opened (symlinks).

**Modified tracked files (not data, listed for completeness).** The 11 files match the gitStatus
snapshot:

- `.claude/skills/knowledge-denial-context/SKILL.md`
- `cap/docs/{SARCASM_DESIGN.md, review_page.html}`
- `docs/{LAYER_PREREGISTRATION, NOVELTY_ASSESSMENT, NOVELTY_CHECK_PROMPT, REVIEW_REQUEST_POOLING}.md`
- `docs/thesis/{persona-coordinates-70b-results, sarcasm-pilot-negative-result}.md`
- `kd/docs/{FINDINGS.md, NEXT.md}`

| Status | Path | Files | Size | Newest mtime | Main types |
|---|---|---|---|---|---|
| !! | `audit/` | 1,913 | 357.5 MiB | 2026-09-24 | .md 493, .py 463, .txt 265, .json 218, .csv 164 |
| !! | `run-hf275/` | 378 | 531.8 MiB | 2026-09-17 | .pt 378 |
| !! | `run-paper16/vectors_all/` | 26 | 65.0 MiB | 2026-09-15 | .pt |
| !! | `run-paper16/vectors_s3/` | 26 | 65.0 MiB | 2026-09-15 | .pt |
| !! | `run-paper16/axes_vectors_all.pt`, `axes_vectors_s3.pt` | 2 | 120.0 MiB | 2026-09-15 | .pt |
| !! | `run-coords/{ai_audit,ai_liar,roleplaying}/` | 3 | 1.8 MiB | 2026-09-15 | .pt |
| !! | `aa-smoke/`, `aa-smoke-win/`, `aa-smoke.log` | 39 | 61.5 MiB | 2026-09-11 | .jsonl, .pt, .json, .log |
| !! | `logs/` | 10 | 16.3 MiB | 2026-09-15 | .log |
| !! | `.venv/`, `.venv-aa/`, `.venv-aa-win/` | 58,994 | 1,682.8 MiB | 2026-09-17 | Python envs |
| !! | `.env`, `src/assistant_axis_ndif/.env`, `CLAUDE.local.md`, `.claude/settings.local.json`, `.claude/scheduled_tasks.lock` | 5 | <0.1 MiB | – | secrets/config (not opened) |
| !! | `__pycache__` ×4, `.pytest_cache/`, `dishonesty_bench.egg-info/` | 35 | 0.4 MiB | – | build caches |
| ?? | `thesis_tools/` | this survey | – | 2026-09-24 | – |
| | **Total ignored** | | **2.83 GiB** | | |

Things to note:

- **`audit/` is entirely gitignored.** Its 1,913 files, including every audit map in part 5, are on
  this laptop only. No branch has them.
- **`RESULTAXIS/` is an empty directory**: `find` counts 0 files, and `git log --all -- RESULTAXIS`
  returns 0 commits. It is neither tracked nor ignored, only empty.
- Folders on disk compared with tracked files on HEAD (`find <d> -type f | wc -l` vs
  `git ls-files <d> | wc -l`):

  | Folder | On disk | Tracked | Why they differ |
  |---|---|---|---|
  | run-paper16 | 130 | 76 | the ignored .pt files |
  | run-coords | 44 | 41 | 3 .pt |
  | src | 585 | 560 | pycache, .env |
  | cap | 95 | 94 | pycache |

- Submodule folders are filled on disk: `assistant_axis` 577 files, `pacchiardi` 255, `mask` 78,
  `soft-trigger` 21. `behavioral-self-awareness` has 0 files.

## 4. Top-level folders on each branch

Command: `s1_folders.py` (`git ls-tree -r -l <b>`). File counts exclude submodule gitlinks.

`main`, `origin/main`, `Saving`, `copilot`, the `claude/*` branches and the `feat/*` branches hold
the same counts in every folder except the ones shown below. The full matrix is in
`s1_data/folders.json`.

| Folder | asix / origin/asix | main | origin/main | persona-pc-ood | paper | llama-layer30 | resumable-acts-cache | other branches |
|---|---|---|---|---|---|---|---|---|
| (root files) | 14 | 10 | 9 | 9 | 9 | 9 | 9 | 9 |
| .claude | 10 | 8 | – | – | – | – | – | – |
| .devcontainer | – | 2 | – | – | – | – | – | – |
| appendix | 2 | – | – | – | – | – | – | – |
| bibliography | 1 | – | – | – | – | – | – | – |
| cap | 94 | – | – | – | – | – | – | – |
| chapters | 7 | – | – | – | – | – | – | – |
| core | 7 | – | – | – | – | – | – | – |
| data | 281 | 281 | 281 | 281 | 281 | 281 | 281 | 281 |
| docs | 130 | 7 | – | – | – | – | – | – |
| figures | 341 | 54 | 54 | 54 | 54 | 54 | 54 | 54 |
| full paper draft for this experiments | 1 | – | – | – | – | – | – | – |
| images | 3 | – | – | – | – | – | – | – |
| kd | 93 | – | – | – | – | – | – | – |
| notebooks | – | – | – | – | – | – | – | 1 on each `claude/*colab*` branch |
| papers | 11 | 6 | – | – | – | – | – | – |
| results | 709 | 709 | 709 | 709 | 709 | 709 | 709 | 709 |
| run-cap | 1 | – | – | – | – | – | – | – |
| run-coords | 41 | – | – | – | – | – | – | – |
| run-crossgen | 7,150 | – | – | – | – | – | – | – |
| run-paper16 | 76 | – | – | – | – | – | – | – |
| scale | 8 | – | – | – | – | – | – | – |
| scripts | 9 | 8 | 8 | 8 | 8 | 8 | 8 | 8 |
| src | 560 | 426 | 426 | **909** | **1,602** | 434 | 433 | 426 |

Gitlinks: `assistant_axis`, `behavioral-self-awareness`, `mask`, `pacchiardi` and `soft-trigger`
are on asix, main and origin/main. All other branches lack `assistant_axis`, and `claude/assistant-axis-submodule` has
`assistant-axis` instead.

**What each folder holds (HEAD unless noted).** The Source column gives where the one-line
description comes from. GUESS means it rests on file names only.

| Folder | Main types (HEAD) | Size (HEAD, blob bytes) | What it holds | Source |
|---|---|---|---|---|
| (root files) | README, pyproject, uv.lock, main.tex, CLAUDE.md | 0.8 MiB | Liars' Bench package root | CLAUDE.md:23–28 |
| .claude | .md 9, .json 1 | – | the `<slug>-context` skills | CLAUDE.md:156–157 |
| .devcontainer (main only) | – | – | dev container | CLAUDE.md:63 |
| appendix, chapters, core, bibliography, images, main.tex | .tex, .bib, .jpg | <0.2 MiB | LaTeX thesis source | **GUESS** |
| cap | .json 50, .py 18, .ps1 12, .md 6, .jsonl 6 | 2.2 MiB | capitulation-bank experiment | CLAUDE.md:59; `cap/docs/DESIGN.md` |
| data | .png 171, .csv 107 | 83.7 MiB | probe inputs (raw/interim/processed/probes) | CLAUDE.md:55 |
| docs | .md 128, .csv 1, .xlsx 1 | 1.7 MiB | project docs (TODO, OPEN_DECISIONS, thesis/, papers/ notes) plus the nnsight docs tree (concepts/, usage/, …) | CLAUDE.md:62, 156; CLAUDE.md nnsight guide |
| figures | .pdf 188, .png 138, .csv 8 | 82.0 MiB | rendered figures; 287 more on asix than on main (e.g. `figures/crossgen/INDEX.md`) | CLAUDE.md:57 |
| full paper draft for this experiments | .pdf 1 | 0.2 MiB | the collaborator's draft `main (1).pdf` | audit/PROJECT_MAP.md:39 (owner-confirmed there) |
| kd | .json 39, .py 22, .ps1 20, .md 6, .jsonl 5 | 6.5 MiB | knowledge-denial persona experiment | CLAUDE.md:58; `kd/README.md` |
| notebooks (colab branches) | .ipynb 1 | – | `colab_extract_activations.ipynb` | **GUESS** |
| papers | .pdf 11 | 27.3 MiB | reference papers | audit/PROJECT_MAP.md:20 |
| results | .csv 636, .json 69, .pkl 4 | 3,997.8 MiB | Liars' Bench rollouts, subsets, blackbox and probe outputs; identical count on every branch | CLAUDE.md:56 |
| run-cap / run-coords / run-crossgen | .json / .json / .csv 7,008 + .json 142 | <0.1 / 0.9 / 64.5 MiB | run outputs; "the lab PC is authoritative" | CLAUDE.md:61 |
| run-paper16 | .json 50, .jsonl 26 (+ ignored .pt) | 3.0 MiB | persona-vector run outputs; **the name is not evidence of which personas were used** | **GUESS** (not in CLAUDE.md's table) |
| run-hf275 (ignored) | .pt 378 | 531.8 MiB on disk | activation or vector tensors | **GUESS** |
| scale | .ps1 6, .py 2 | <0.1 MiB | 3B/8B rung drivers | CLAUDE.md:60 |
| scripts | .sh 7, .py 2 | <0.1 MiB | probe-pipeline entry points | CLAUDE.md:54 |
| src | .py 187, .json 96, .txt 84, .yaml 52, .csv 41 | 446.2 MiB | first-party code; `src/assistant_axis_ndif/` has `README_COLLABORATOR.md` | CLAUDE.md:39–53 |
| src on paper / persona-pc-ood | paper: .csv 878, .png 179, .py 154; pc-ood: .csv 357, .py 147, .png 87 | – | the collaborator's persona-OOD results and the paper source under `src/probes/deception-detection/` | `paper/README.md` (branch), `results/persona_ood_ext_L40/summary.md` lines 1–4 |
| audit (ignored) | .md, .py, .txt, .json, .csv | 357.5 MiB | claim audit, Phases 0–5 | CLAUDE.local.md |
| logs, aa-smoke*, (ignored) | .log / .jsonl, .pt | – | run logs; Assistant-Axis smoke test | **GUESS**; `experiments.csv` row `O01-aa-smoke` names aa-smoke |
| RESULTAXIS | – | 0 files | empty directory, never in git | `find`, `git log --all` |
| mask, pacchiardi, soft-trigger, assistant_axis | submodules | – | MASK, Pacchiardi detector, sleeper agents, Assistant Axis | CLAUDE.md:77–80 |
| behavioral-self-awareness | gitlink, empty | – | stale gitlink | CLAUDE.md:85 |

**CONFLICT, `run-kd/`.** CLAUDE.md:61 lists `run-kd/` as a run-output folder. No branch contains it
(`git log --all -- run-kd` → 0 commits) and it is not on disk. `audit/PROJECT_MAP.md:20` says
run-kd is lab-PC-only. So the doc and this laptop disagree.

## 5. The audit's maps: what they cover and what they miss

Profiled with inline `csv.DictReader` scripts, shown in the session log.

| Map | Rows × cols | File mtime | Content dates | Covers |
|---|---|---|---|---|
| `audit/experiments.csv` | 114 × 18 | 2026-09-22 22:20 | `when_first`/`when_last` 2026-07-11 → 2026-09-22 16:59 | one row per experiment (`exp_id`, who, branch, commit, script, args, results_location, status, f_ids). `who`: owner 90, collaborator 20, other 4. `status`: complete 106, unknown 4, partial 2, other 2 |
| `audit/phase1/fmap_all.csv` | 109 × 20 | 2026-09-22 21:52 | `heading_date` has 6 values | one row per ledger entry **F-01 … F-109**. It maps each to a primary file, `copy_status` (LAB-ONLY 73, LAB=GIT 16, GIT-ONLY 8, LAB≠GIT 7, NOT-FOUND 3, mixed 2) and `mapping_status` (MAPPED 61, PARTIAL 47, NOT-FOUND 1) |
| `audit/cache/MANIFEST.csv` | 305 × 10 | 2026-09-23 09:56 | `copied_at` 2026-09-22 20:55 → 2026-09-23 03:48 | files copied from the **lab PC only** into `audit/cache/labpc/`. By source folder: run-premise 100, run-kd 72, run-crossgen 60, run-paper16 52, cap 8, logs 6, kd 3, run-coords 2, loose 2. `check`: MATCH 297, NOT_HASHED 6 |

**What they miss.** Each item is measured:

1. **Other branches.** `experiments.csv`'s `branch` column names only `asix` (plus variants),
   lab-only, and `origin/feat/persona-pc-ood` (20 rows). The substring search in the command below
   finds that 7 branches never appear:
   - `origin/paper`, `feat/llama-layer30`, `feat/resumable-acts-cache`
   - `Saving`, `copilot/zami456explb`
   - the three `claude/*` branches

   `origin/paper` does occur in the free text, since "paper" matches. But no row has
   `branch=origin/paper`, and 0 rows mention `paper/`. The **1,176 files that `origin/paper` adds**
   (part 2) are therefore not mapped, except for the part it shares with persona-pc-ood (483 files).
2. **Anything newer than the audit.**
   - **CONFLICT on the ledger size.** The working-tree `kd/docs/FINDINGS.md` has **127** F-headings
     (F-01…F-127, no gaps or duplicates; `re.findall(r"^#+\s*F-(\d+)")`). `HEAD:kd/docs/FINDINGS.md`
     has **120**. `CLAUDE.local.md` says "holds 122 F-entries" and "Phase 6 is … NOT started". But
     F-123…F-127 (lines 6145–6672, dated 2026-09-24) describe Phase-6 work (e.g. "6D-4", "6D-1 and
     6D-2 closed").
   - `fmap_all.csv` stops at F-109, so **18 entries (F-110…F-127) are unmapped**.
   - After the maps' mtime there are 4 `origin/paper` commits and 2 `asix` commits (part 2). None of
     them is in any map.
3. **Ignored folders on this laptop.** `experiments.csv` mentions some of them:

   | Folder | Rows mentioning it |
   |---|---|
   | `run-paper16` | 7 |
   | `run-hf275` | 2 |
   | `aa-smoke` | 1 |
   | `logs/` | 5 |
   | `RESULTAXIS` | 0 |

   `MANIFEST.csv` covers only lab-PC sources. No row points at a laptop ignored folder, a GitHub
   branch or Drive.
4. **The maps themselves are ignored.** `audit/` is `!!`, so none of these maps is in any branch.
5. **Data quality.** `MANIFEST.csv` lines 305–306 (the `make_view.py`, `make_view_ac.py` rows) are
   written with shifted columns: `copied_at` holds a local path, `source_path` holds a byte count,
   `source_md5` holds a timestamp. Any script that reads them by header gets wrong values.

Commands used for 5:

```
python -: csv.DictReader over the 3 files; Counter per column; re.findall dates; substring search of
          each `git for-each-ref` branch name in experiments.csv; re.findall(r"^#+\s*F-(\d+)") on
          FINDINGS.md working tree and `git show HEAD:kd/docs/FINDINGS.md`
git log --since='2026-09-22T22:20:00+06:00' --format='%h %cI %an %s' <b>   for asix, origin/paper, origin/feat/persona-pc-ood
```

---

## COVERAGE

| Searched | Scope | Matches | Examined | Skipped / reason |
|---|---|---|---|---|
| `git for-each-ref refs/heads refs/remotes` | all refs | 14 refs (2 local, 12 remote; minus `origin/HEAD` alias → 13) | 13 | none. Remote state is as of the last fetch (09-24 17:28); I did not re-fetch |
| `git shortlog -sne`, `rev-list --count` | 13 branches | 13 | 13 | none |
| `git diff --name-status HEAD...<b>` + `ls-tree` set diff | 12 non-HEAD branches | 1,176 + 483 + 13 + 12 + 5 + 4 + 2 + 2 files | all counted; grouped, not read | file contents on other branches not opened, except the 3 READMEs/summary cited |
| `git status --ignored --short` | working tree | 39 lines (11 M, 1 ??, 27 !!) | 39 | `.venv*/lib64` symlinks not traversed (OS error). `.env` files counted, not opened |
| `find -type f` vs `git ls-files` | 32 top-level dirs on disk | 32 | 32 | none |
| `git ls-tree -r -l` per branch per top-level folder | 13 branches | 25 folder names | 25 | per-folder types given for HEAD; for other branches only where counts differ |
| README/DESIGN/INDEX (depth ≤2) + CLAUDE.md table + audit/PROJECT_MAP.md | HEAD | 9 docs found | CLAUDE.md lines 39–86, PROJECT_MAP lines 1–40, kd/cap/paper READMEs (headers only) | folders with no doc marked GUESS (7) |
| `git log --all -- RESULTAXIS` / `-- run-kd` | all history | 0 / 0 | – | – |
| audit maps | experiments.csv, fmap_all.csv, cache/MANIFEST.csv | 114 / 109 / 305 rows | all rows profiled per column; not every cell read | row-level correctness was not checked. That is a later task |
| FINDINGS.md F-headings | working tree + HEAD | 127 / 120 | heading numbers only | entry contents not read |

---

## Question for you

By commit author:

- **`asix`** is the only branch with Suraiya Shanaz commits beyond the shared base: 136 commits in
  `main..asix`, all `suraiya.shanaz@g.bracu.ac.bd`, dated 09-10 → 09-24. The same author also owns
  the 2 `Saving` commits and 3 on the colab branches.
- **`origin/paper`** and its ancestor **`origin/feat/persona-pc-ood`** have only `zami456
  <abdullah.al.zami@g.bracu.ac.bd>` beyond the base (40 and 34 commits). So do
  `feat/resumable-acts-cache` and `feat/llama-layer30`; the latter also has one commit by
  **Mahdi `<mahdi.ndc21@gmail.com>`**.

Is `asix` yours and `origin/paper` your teammate's (zami456)? And who is Mahdi: a third
contributor, or your teammate on another machine?

---

## Owner's answer (2026-09-24)

These are the owner's own words, summarised. They are a statement, not something checked here.

- **`asix` is the owner's branch.** `origin/paper` and `origin/feat/persona-pc-ood` belong to the
  teammate **zami456**.
- **Mahdi is a third co-author.**
- **History of the work:**
  - zami456 built the repo.
  - The owner then designed parts of the work, and zami456 designed on top of that.
  - Both ran some of the same tests and some different ones, exploring different methods toward the same goal.
  - They worked separately, without updating each other.
- **Consequence:** either branch may hold the better result for the same experiment, and nobody
  knows which. Later survey tasks have to pair the same experiment across the two branches before
  choosing a "best" number. They must not assume one branch supersedes the other.

---

## Follow-up: is the audit up to date with zami456's branches? (2026-09-24)

**Method.** For every commit in `asix..<branch>`, I checked whether its 7-character SHA appears in
any text file under `audit/`. The script indexed 1,630 .md/.csv/.txt/.json/.py/.tsv/.log files.
The command is quoted in the session. A cited SHA shows that the audit referred to that commit; it
does not show that every claim in the commit was checked.

- **`origin/feat/persona-pc-ood`: covered.** 33 of its 34 commits are cited, including its tip
  `45b16d90`, which 66 files cite. The one uncited commit is `784f1350`, a venv fix in
  `run_llama_all.sh`.
- **`origin/paper`: NOT fully covered.** The newest commit the audit cites is **`ea1ce52b`**
  (2026-09-23 23:46; the layer 20–79 sweep). It is cited in `audit/claims.csv` (176 times),
  `COLLAB_PAPER_BRANCH.md`, `phase6/RESULTS.md`, `RESULTS_CHAPTER.md` and three scripts. **The 3
  newer commits are cited nowhere in `audit/`:**
  - `2ad758fb` 09-24 15:34 "The depth sweep in the paper: a curve replaces three layers, and corrects four claims"
  - `d1b88154` 09-24 16:43 "Make the case against the trained probe from the whole depth curve, not one layer"
  - `16def2de` 09-24 16:47 "Correct an overstated HP-KR ranking claim: 8 of 12 deep layers, not all of them"
- **What those 3 commits change** (`git diff --name-status ea1ce52b 16def2de`): 66 files, 53 modified
  and 13 added.
  - `paper/`: 7 sections, 7 generated files, `make_numbers.py`, `make_figures.py`, `main.pdf`.
  - **22 existing result CSVs** in `results/persona_ood_ext_L{20,40,44}/`, including
    `zero_shot.csv`, `matrix_baseline.csv`, `summary.csv` and `liars_bench_comparison.csv`.
  - So verdicts the audit recorded against `ea1ce52b` may now point at numbers or text that have changed.
- **The final reports are older still.**
  - `REPORT_PLAIN.md` names no `origin/paper` commit.
  - `REPORT_TECHNICAL.md` names only `8fd071b2` (09-22, "Paper draft").
  - `COLLABORATOR_FINDINGS.md` names none.
  - `COLLAB_PAPER_BRANCH.md` (mtime 09-24 00:18) goes up to `ea1ce52b`.
- **The other two zami456 branches are covered**, though they don't matter for results. Their commits
  are also on `origin/paper`, and `df758c4b` (llama-layer30) is cited in `START_STATE.md`.
