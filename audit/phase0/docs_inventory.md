# Phase 0 — Inventory of findings documents, CLAUDE.md/skills, auto-memory, and the F-number ledger's history

Written 2026-09-22 by a Phase-0 subagent. Repo state: branch `asix`, HEAD `43ee6d3`, working tree clean
(audit/ and CLAUDE.local.md excluded). **Facts and locations only — nothing here says whether any
claim is true.** What a document says about itself (its status, what it supersedes, which doc is
authoritative) is **DOC-ONLY**. Git facts (dates, commits, diffs) are **VERIFIED** from `git log` /
`git show`. Interpretations are marked **INFERRED** with the reason.

Companion files (all written by this task):

| file | what it holds |
|---|---|
| `audit/phase0/fnumber_history.csv` | one row per F-id F-01..F-109: title, dates, first commit, later edits, relations, line range (part B) |
| `audit/phase0/docs_inventory_raw.tsv` | machine output behind the doc table in §2 (lines, bytes, git dates, numbering-scheme counts) |
| `audit/phase0/memory_inventory.tsv` | machine output behind §3a (every auto-memory file) |
| `audit/scripts/p0_docs_inventory.py` | per-doc stats; `--authority` prints every "authoritative / supersedes / stale / ledger" line |
| `audit/scripts/p0_docs_fhistory.py` | ledger history → `fnumber_history.csv`; `--commits` prints the per-commit add/edit table |
| `audit/scripts/p0_docs_memory.py` | auto-memory inventory (read-only) |
| `audit/scripts/p0_docs_external.py` | `~/.claude/plans/*.md`, the older project memory dir, transcript counts (read-only) |
| `audit/scripts/p0_docs_pdfscan.py` | page counts / metadata of tracked PDFs; text scan of the draft paper PDF only |
| `audit/scripts/p0_docs_fhigh.py` | finds F-110..F-999 tokens in the working tree + memory dir, with context |

## 0. How to re-run (from the repo root, Git Bash)

```bash
export PYTHONIOENCODING=utf-8
python audit/scripts/p0_docs_inventory.py > audit/phase0/docs_inventory_raw.tsv
python audit/scripts/p0_docs_inventory.py --authority | grep -v '^kd/docs/FINDINGS.md'
python audit/scripts/p0_docs_memory.py > audit/phase0/memory_inventory.tsv
python audit/scripts/p0_docs_external.py
python audit/scripts/p0_docs_pdfscan.py
python audit/scripts/p0_docs_fhistory.py --commits          # writes fnumber_history.csv
python audit/scripts/p0_docs_fhigh.py                       # F-110+ in working tree + memory (slow; data CSVs hit "Ford F-150")
# git-only checks used below
git log --follow --name-status --format='%h %ad %s' --date=iso -- kd/docs/FINDINGS.md   # 1 A + 38 M, no rename/copy
git log --all --diff-filter=A --name-only --format='%h %ad' -- '*FINDINGS*' '*findings*'  # only kd/docs/FINDINGS.md @ 6432257
git log --all -S'### F-01' --format='%h %ad %s'                                            # only 6432257
git log --all -G'F-[0-9][0-9]' --format='%h %ad %s' -- '*.md' '*.html' '*.txt'            # oldest hit 6432257
git grep -lE '\bF-1[1-2][0-9]\b' $(git for-each-ref --format='%(refname)' refs/)           # no hits, any ref
git log --all -G'F-1[1-2][0-9]' -- '*.md' '*.html' '*.txt' '*.py' '*.ps1' '*.tex'          # no hits, any commit
```

## 1. Headline facts

- **31 findings-bearing tracked files on asix = 25 distinct documents** (the 4 cap review-packet versions,
  the 2 HTML renderings of review packets and the role-label csv/xlsx pair grouped; skills, paper notes,
  runbooks and library docs not counted; §2a–2f), plus **9 skills**, **8 paper-notes files**,
  **1 root CLAUDE.md** (no nested CLAUDE.md anywhere; one
  untracked `CLAUDE.local.md` created by this audit's Setup), **45 auto-memory files**, and **6 plan files
  outside the repo** (`~/.claude/plans/`) that project docs point to. VERIFIED (file listings).
- **The ledger** `kd/docs/FINDINGS.md`: 5,234 lines, 109 sections F-01..F-109, contiguous, no duplicates,
  no suffixed ids, in numeric order. 39 commits, all by Suraiya Shanaz, 2026-09-21 03:26 → 2026-09-22 17:37.
  **It was created in one commit (6432257) already holding F-01..F-66**, whose headings are dated
  2026-09-19..21; no earlier findings file, rename or copy exists on any ref. VERIFIED (git).
- **No F-id was ever deleted, renumbered, re-titled or reused.** 8 sections were edited after first commit;
  7 of those edits replaced or removed existing lines, which the ledger's own "append-only, nothing is
  edited" rule forbids (CONFLICT, §8). VERIFIED (git diff).
- **F-110 and above do not exist anywhere** — not in the ledger, any doc, script, memory file, any git ref,
  or any commit's diff. "F-112" occurs only in `audit/BRIEF.md` line 16 (the owner's example); "F-150"
  occurs 4× in `src/probes/deception-detection/data/rollouts/alpaca__plain__llama-70b-3.3.json` as the
  **Ford F-150 truck**. **NOT-FOUND** as findings. VERIFIED.
- Newest findings docs by commit: `kd/docs/FINDINGS.md` and `docs/COLLABORATOR_BRIEFING.md` (both last
  touched 43ee6d3, 2026-09-22 17:37). The docs disagree about which file is authoritative (§7).
- Collaborator branch `origin/feat/persona-pc-ood` has 16 `results/*/summary.md` files under
  `src/probes/deception-detection/`, **no F-ids** in any .md (git grep). VERIFIED. (Handled by the other agent.)

## 2. Findings-bearing documents on asix (tracked at HEAD 43ee6d3)

All are tracked and were committed only by **Suraiya Shanaz** (git log --follow, every file). Dates are
commit author dates in +06:00. "#c" = commits touching the file (`git log --follow`). "Numbering" lists
schemes present (counts in `docs_inventory_raw.tsv`). The last column is what the doc says about itself
or about which doc to trust — **DOC-ONLY**.

### 2a. Ledger and results logs

| path | lines | bytes | first commit | last commit | #c | type | numbering | covers | authority claim (DOC-ONLY) |
|---|---:|---:|---|---|---:|---|---|---|---|
| `kd/docs/FINDINGS.md` | 5234 | 330601 | 6432257 2026-09-21 03:26 | 43ee6d3 2026-09-22 17:37 | 39 | ledger | F-01..F-109 (109 headings); also cites Q-items, Claim N, Result 1/2, Tier 0.x, Gate 3 | every measured fact of the persona-coordinates / kd / cap / scale / sarcasm work, one entry per fact with source, caveats, supersedes, status | preamble: "**Append-only.** … Nothing is edited or deleted — a finding that turns out wrong gets a new entry that supersedes it"; status = verified / provisional / inherited. CLAUDE.md calls it "the project-wide append-only ledger (F-01…) — every measured number in the repo" |
| `kd/docs/REPORT.md` | 166 | 6980 | 5b43a7a 2026-09-19 02:58 | 6dd9f9b 2026-09-21 03:40 | 2 | report (running log, kd only) | Phase 0–6; §0.1–0.4 | kd experiment log; Phase 0 baselines (HP-KR lexical, attribution, layer inversion, L40 transfer); Phases 2–6 still read "_not started_" | "Running log. Append as each phase lands; mark superseded numbers rather than deleting them." `kd/README.md` calls it "the running results log" |
| `docs/thesis/persona-coordinates-70b-results.md` | 1475 | 93073 | b9a497e 2026-09-18 11:26 | same | 1 | report (results chapter for collaborator) | Q1–Q11 (161 mentions); §1–§14; passes A/B/C; §10 table of 29 record errors | answers to research questions Q1–Q11 at 70B, pipeline, statistics, sycophancy limitation, where the written record was wrong, open issues, 3B/8B ladder. "Status as of 2026-09-17" | "every number here was re-read from the result file that produced it … Where the plan (`harmonic-cuddling-kay.md`) or our memory notes disagreed with the file, the file wins" |
| `docs/thesis/sarcasm-pilot-negative-result.md` | 289 | 16930 | 6c9a5cf 2026-09-22 03:25 | same | 1 | report (thesis section, negative result) | §1–§8; **defines G1 markers / G2 separability** (§2.1) and also uses "gates 2 and 3" (design Sequence numbering) | the pre-registered sarcasm bank failing its own gate; cites F-89, F-92, F-93 (+amendment), F-94, F-104; says gate 3 "was registered and **never run**" (written before F-106/F-107) | header "Ledger: F-92, F-93 (+ amendment), F-104"; data path `cap/run-sarc-pilot/` |
| `docs/thesis/hf275-role-labels-draft.md` | 135 | 13863 | b9a497e 2026-09-18 11:26 | same | 1 | report (draft, labelling) | Q1 once | labels over the 275 Assistant-Axis roles; "48 behaviour roles" finding | "Status: not yet used for any result" |
| `docs/thesis/role-labels/role_labels_review.xlsx` / `.csv` | – / 1381 | 79193 / 196715 | b9a497e 2026-09-18 | same | 1 | data for the draft above | – | per-role labels for review | – |

### 2b. Briefings, reviews and requests written for other people

| path | lines | bytes | first commit | last commit | #c | type | numbering | covers | authority claim (DOC-ONLY) |
|---|---:|---:|---|---|---:|---|---|---|---|
| `docs/COLLABORATOR_BRIEFING.md` | 634 | 38546 | 01e27b0 2026-09-21 16:52 | 43ee6d3 2026-09-22 17:37 | 5 | briefing | **Claim 1–3** (defines them, §3); §1–§6 incl. §5a–§5e; cites F-86, F-97, F-107, F-108, F-109; Q8 once | where the project stands for the collaborator: 3 claims that survive, 4 withdrawn, the abstract's 0.904/0.916 being the kd-bank axis | "Self-contained: you should not need any other file"; "Revised 2026-09-22 … Ledger entry: F-97"; says it read `feat/persona-pc-ood` at head `f386a72` (pc-ood tip is now 45b16d9, 2 commits later — VERIFIED) |
| `docs/REVIEW_REQUEST_POOLING.md` | 120 | 6768 | 80fab2b 2026-09-21 19:11 | same | 1 | review request | **Result 1, Result 2**; weaknesses 1–6; reviewer questions 1–5; cites F-79, F-80, F-81, F-83, F-86 | asks for adversarial review of the pooled-training result | ledger F-90 (status) says this file "must be revised before it goes to anyone else" — it has never been edited since (1 commit) |
| `docs/NOVELTY_CHECK_PROMPT.md` | 189 | 13177 | e9969af 2026-09-21 10:55 | 8d50caf 2026-09-21 11:41 | 3 | review request (prompt) | **defines Claim 1–10** ("The ten claims") | prompt given to two web-search sessions to rank the ten claims by novelty | – |
| `docs/NOVELTY_ASSESSMENT.md` | 134 | 8850 | 8d50caf 2026-09-21 11:41 | same | 1 | review (external, summarised) | Claim 3–10 (novelty numbering); F-55 | what two independent reviews said about novelty of the ten claims | "Neither reviewer verified any of our numbers" |
| `kd/docs/REVIEW_PACKET.md` + `kd/docs/review_page.html` | 310 / 476 | 18947 / 28830 | 6dd9f9b 2026-09-21 03:40 | same | 1 | review packet | question items Q20–Q29 | 10 new kd personas + 10 questions for outside review | – |
| `cap/docs/REVIEW_PACKET.md`, `_V2`, `_V3`, `_V4` + `cap/docs/review_page.html` | 292 / 335 / 337 / 373 / 403 | 16078 / 18333 / 18926 / 20607 / 25714 | dc24e96 2026-09-21 03:37 | same | 1 each | review packets (4 versions) | question items Q0–Q19 | capitulation personas and questions for outside review | – (note: `git log --follow` wrongly chains `_V4` to `cap/scripts/make_cap_review.py` @6432257 via a 51% copy match; its real first commit is dc24e96) |

### 2c. Plans, trackers, decisions, roadmaps

| path | lines | bytes | first commit | last commit | #c | type | numbering | covers | authority claim (DOC-ONLY) |
|---|---:|---:|---|---|---:|---|---|---|---|
| `docs/ROADMAP_TO_DEFENCE.md` | 169 | 9848 | ebbe48d 2026-09-21 17:08 | 4d0d046 2026-09-22 02:46 | 2 | plan (roadmap) | **defines Tier 0 (0.1–0.4), Tier 1 (1.1–1.4), Tier 2**; Claim 1–3 (briefing numbering); 21 F-ids F-61..F-97 | every open caveat, the run that closes it, its cost | "Three claims survive the dataset-level bootstrap" |
| `docs/TODO.md` | 110 | 7054 | 6432257 2026-09-21 03:26 | dc24e96 2026-09-21 03:37 | 2 | todo (stage tracker) | **defines Stage 1–4** (+ rows 1a…4c); 13 F-ids F-03..F-67 | per-stage gate, measured value, status | "`kd/docs/FINDINGS.md` records what was measured, this records what is still owed"; "Plan: `~/.claude/plans/glowing-rolling-finch.md`. Last updated 2026-09-21 00:05" |
| `docs/OPEN_DECISIONS.md` | 184 | 11339 | 6dd9f9b 2026-09-21 03:40 | bc701d9 2026-09-21 04:01 | 2 | decisions | **defines A1–A5, B1–B10, C**; cap items Q0–Q16; 16 F-ids F-04..F-67 | owner decisions pending; standing caveats "read before quoting a number"; self-corrections | "Distinct from `kd/docs/NEXT.md` (queued work) and `kd/docs/FINDINGS.md` (measured facts)". "Last updated 2026-09-21 03:50" |
| `kd/docs/NEXT.md` | 107 | 5931 | 6dd9f9b 2026-09-21 03:40 | 20550e3 2026-09-21 03:59 | 2 | plan (queued work) | items 1–3 with (a)(b)(c); 11 F-ids F-03..F-68 | queued kd/cap work | "When one starts, its outcome goes into `FINDINGS.md`"; "Live execution state … is in `docs/TODO.md`" |

### 2d. Designs and pre-registrations

| path | lines | bytes | first commit | last commit | #c | type | numbering | covers | authority claim (DOC-ONLY) |
|---|---:|---:|---|---|---:|---|---|---|---|
| `kd/docs/DESIGN.md` | 238 | 14165 | 5b43a7a 2026-09-19 02:58 | 6dd9f9b 2026-09-21 03:40 | 2 | design + pre-committed readings | §1–§5; Q1, Q2, Q6, Q8, Q11 | kd (knowledge-denial) axis design and its attribution | "Written **before** any result exists. The readings in §5 are pre-committed" |
| `cap/docs/DESIGN.md` | 249 | 14404 | dc24e96 2026-09-21 03:37 | 2be35eb 2026-09-21 12:41 | 3 | design + results summary | 14 F-ids F-03..F-68; Q10 | capitulation bank; states "0.745 zero-shot … 0.374", clears bootstrap (F-65, F-66) | "This is the only place in the project where a persona axis beats everything it is matched against" |
| `cap/docs/SARCASM_DESIGN.md` | 225 | 12929 | 3d80071 2026-09-21 17:48 | b34ff2e 2026-09-21 21:48 | 4 | design (v2 after review) + pilot result | **§Sequence items 1–4 = the "gates"** (implicit); F-46, F-66, F-89, F-92, F-93 | sarcasm bank design; "PILOT RESULT (2026-09-21) — the on-policy design is dead" | – |
| `docs/LAYER_PREREGISTRATION.md` | 114 | 7640 | f5d6904 2026-09-21 13:00 | 63aabb4 2026-09-21 13:19 | 2 | preregistration (amended) | §1–§5; Claim 5, Claim 9 (novelty numbering); Q2, Q6; F-72 | how the 80-layer profile will be read; §1 AMENDED ~13:25 ("L40 is a literature choice") | "Written … while … the `axes` numbers this document governs … had not started … `git log` timestamps it" |
| `docs/PERSONA_CONSTRUCTION.md` | 603 | 34200 | ed179bd 2026-09-21 13:51 | 50fa69d 2026-09-21 14:34 | 2 | design (method description) | §1–…; F-01 | how every persona bank was built, kd bank worked example | lines 172–173: pre-ledger facts "are recorded in project memory rather than as numbered findings. Everything from F-01 onward is in the ledger." |

### 2e. Draft paper

| path | pages | bytes | commit | type | numbering | covers | notes |
|---|---:|---:|---|---|---|---|---|
| `full paper draft for this experiments/main (1).pdf` | 19 | 193849 | cf11749 2026-09-22 16:51 (1 commit, Suraiya) | draft paper (LaTeX; PDF CreationDate 2026-09-22 07:57 UTC) | none — no F-ids, Q-ids, "Claim N" or "Result N" in its text (pypdf scan) | title **"Lying in Knowledge-Denier Persona Coordinates: Cross-Dataset Lie Detection at 70B"**, authors "[Author names]"; sections 1 Intro … 5.1–5.5 Results … 8 Conclusion; abstract quotes 0.148 (L20 inversion), 0.919 / 0.910 bal-acc @1% FPR on HP-KR at L40, benchmark probes 0.50 / 0.26, upper bound 0.95, **0.916 mean off-diagonal vs 0.732** on six datasets at L40 | INFERRED to be the collaborator's draft: plan `glowing-rolling-finch.md` says "The collaborator has drafted a title and abstract (\"One Direction Generalizes: …\") … 0.904 / 0.916 … 0.648 / 0.732"; the committed PDF has a different title, so it is a later version (INFERRED). Only a text scan was done; the draft has not been read for content. |

### 2f. Instructions and context (CLAUDE.md, onboarding, READMEs, skills)

| path | lines | bytes | first commit | last commit | #c | type | numbering | covers | authority claim (DOC-ONLY) |
|---|---:|---:|---|---|---:|---|---|---|---|
| `CLAUDE.md` (root; the only CLAUDE.md in the repo — `find` over the working tree excl. .venv*/.git) | 331 | 26939 | 608b2a7 2026-09-09 20:45 | 8d50caf 2026-09-21 11:41 | 6 | readme / agent instructions | cites F-01 (range start), F-13, F-63, F-65, F-66 | repo layout, submodules, setup, paper-context skills; lines 172–331 are the nnsight agent guide (library text) | FINDINGS.md = "the project-wide append-only ledger … every measured number in the repo"; run outputs: "the lab PC is authoritative and syncs to Drive" |
| `CLAUDE.local.md` (untracked, git-excluded) | 26 | 1746 | – | – | 0 | audit instructions | – | "AUDIT IN PROGRESS" block written by this audit's Setup (per PROGRESS.md) | not a project doc |
| `docs/ONBOARDING.md` | 197 | 12212 | f4e304e 2026-09-16 02:09 | 5b43a7a 2026-09-19 02:58 | 5 | briefing (onboarding) | Q1–Q11 (one line each); §1–… | what the research is, "State as of 2026-09-17: All eleven questions are answered", how the owner works, mistakes paid for | names `~/.claude/plans/harmonic-cuddling-kay.md` "**The single most important file**" and memory `MEMORY.md` "Index of measured findings"; does **not** mention FINDINGS.md (grep) |
| `kd/README.md` | 104 | 4880 | 5b43a7a 2026-09-19 02:58 | 3dfdb67 2026-09-21 11:07 | 2 | readme | – | what `kd/` is | "Read `docs/DESIGN.md` first … `docs/REPORT.md` is the running results log" |
| `src/assistant_axis_ndif/README_COLLABORATOR.md`, `RUNBOOK.md` | 56 / 172 | 2576 / 8850 | 107879d 2026-09-11 11:05 | 5694378 2026-09-11 11:27 | 2 | readme / runbook | – | how to run the Assistant-Axis NDIF pipeline | procedural, no findings |
| `.claude/skills/knowledge-denial-context/SKILL.md` | 117 | 7153 | 5b43a7a 2026-09-19 02:58 | 6dd9f9b 2026-09-21 03:40 | 2 | skill | cites F-13, F-55, F-63, F-65, F-66, F-67 | kd experiment context incl. HP-KR confound; cap result | "**Never quote F-66 without F-67.**" |
| `.claude/skills/nnsight-ndif-context/SKILL.md` | 71 | 4167 | 404bc27 2026-09-15 21:24 | same | 1 | skill | – | this deployment's NDIF failure modes | – |
| `.claude/skills/{assistant-axis,deception-probes,liars-bench,mask,pacchiardi,persona-coordinates,soft-trigger}-context/SKILL.md` | 18–22 each | 804–1149 | 608b2a7 2026-09-09 20:45 | same | 1 each | skill (pointer) | – | each points to its `docs/papers/<slug>-notes.md` | – |

### 2g. Paper notes (type = paper-notes; they matter for later phases)

| path | lines | bytes | commit(s) | covers (from its first lines) | numbering |
|---|---:|---:|---|---|---|
| `docs/papers/assistant-axis-notes.md` | 226 | 33557 | 608b2a7 2026-09-09 | Lu et al., *The Assistant Axis* | Fig N |
| `docs/papers/deception-probes-notes.md` | 518 | 32809 | 608b2a7 2026-09-09 | *Detecting Strategic Deception Using Linear Probes* | Fig N |
| `docs/papers/liars-bench-notes.md` | 163 | 24738 | 608b2a7 2026-09-09 | Kretschmar et al., *Liars' Bench* arXiv:2511.16035v2 | – |
| `docs/papers/mask-notes.md` | 268 | 25557 | 608b2a7 2026-09-09 | Ren et al., *The MASK Benchmark* | Fig N |
| `docs/papers/pacchiardi-notes.md` | 205 | 23068 | 608b2a7 2026-09-09 | Pacchiardi et al., *How to Catch an AI Liar* | Fig N |
| `docs/papers/persona-coordinates-notes.md` | 650 | 38697 | 608b2a7 2026-09-09 | Mahadik & Skapars, *Do Linear Probes Generalize Better in Persona Coordinates?* | Fig N |
| `docs/papers/soft-trigger-notes.md` | 297 | 16234 | 608b2a7 2026-09-09 | the `soft-trigger/` submodule (no paper) | its own Q1–Q4 |
| `docs/papers/truthfulness-spectrum-notes.md` | 119 | 7025 | fcef82a 2026-09-21 14:11 | *The Truthfulness Spectrum Hypothesis* + *The Truth Stays in the Family* ("read in full") | cites F-13, F-76 |

Papers folder `papers/` (11 PDFs, not read): Liars' Bench 2511.16035v2 (112 p), Detecting Strategic
Deception (35 p), Do Linear Probes Generalize Better in Persona Coordinates (14 p), How to Catch an AI Liar
(36 p), The Assistant Axis (54 p), The MASK Benchmark (22 p) — all committed 608b2a7 2026-09-09; Pressure-Testing
Deception Probes (18 p), Probe Generalization as Subspace Selection (11 p), The Impact of Off-Policy Training
Data (56 p), The Truth Stays in the Family (24 p), The Truthfulness Spectrum Hypothesis (28 p) — committed
4d0d046 2026-09-22 02:46. By file name, **three PDFs have no notes file**: Pressure-Testing Deception Probes,
Probe Generalization as Subspace Selection, The Impact of Off-Policy Training Data (ledger F-78/F-88 discuss
papers read on 09-21 — not checked). `truthfulness-spectrum-notes.md` (committed 2026-09-21 14:11) says its two
PDFs were "read in full", a day before those PDFs were committed (they may have existed untracked — INFERRED).
`soft-trigger-notes.md` has no PDF (it covers a submodule). **Coverage is a Phase-1 (papers.csv) item.**

## 3. Claude-written documents outside the repo

### 3a. Auto-memory — `C:\Users\SURAIYA\.claude\projects\C--Users-SURAIYA-Documents-STUDY-thesis-expLB\memory\` (read-only)

45 files (44 notes + `MEMORY.md` index), 2026-09-15 .. 2026-09-22. F-ids cited anywhere in memory:
F-39, F-63, F-64, F-85, F-89, F-91, F-93, F-94, F-95, F-96, F-97, F-104, F-107 (max F-107; all exist in the
ledger). Q-ids Q1–Q11 = the research questions of plan `harmonic-cuddling-kay.md`. 37 of the 44 notes
were last modified before the ledger's first commit (2026-09-21 03:26). `fm modified` in the TSV is the note's own
frontmatter timestamp (UTC); one (`persona-axes-encode-register.md`) has a frontmatter time
2026-09-21T03:55Z that is ~6 h **after** its file mtime (2026-09-20 21:40Z) — cosmetic, noted only.

| file | bytes | fs mtime (+06) | type | F-ids cited | Q-ids cited | description (frontmatter, truncated) |
|---|---:|---|---|---|---|---|
| MEMORY.md | 8021 | 2026-09-22 14:33 | - | F-63 | Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q10,Q11 | (index, 45 lines, no frontmatter) |
| aa-run-live-labpc.md | 3562 | 2026-09-19 13:51 | project | - | - | The 100-role Assistant-Axis NDIF run COMPLETED 2026-09-14 on the lab PC; where the deliverable is and the lab-PC/SSH gotchas |
| abstract-number-is-our-kd-bank.md | 2435 | 2026-09-22 00:16 | project | F-85,F-97 | - | The collaborator abstract's 0.904/0.916 is ext_axis1 (kd-augmented 12v13), not the paper's 8v7 contrast, which reads 0.791/0.832 — ~45% of the headlin… |
| bf16-mask-reductions.md | 3153 | 2026-09-16 02:04 | feedback | - | - | Never cast a span mask or position index to the model's bfloat16 — integers above 256 are not representable and the last-token bug is silent |
| crossgen-80-20-retrain.md | 3982 | 2026-09-18 03:29 | project | - | Q2,Q7 | 10x10 cross-generalization retrain (train 80% of one dataset, test every dataset's 20% split) — separate run-crossgen/ folder; smoke passed; ID-traine… |
| crossgen-layer-coverage.md | 2119 | 2026-09-18 15:58 | project | - | - | What layers each activation store holds, which crossgen arms exist, and the two reasons results look like "layer 40 only" |
| dataset-identity-dominates-variance.md | 2495 | 2026-09-16 18:45 | project | - | Q2,Q3 | Eval activations are dominated by dataset identity (99.3% decodable vs +3.2% for the label) — and the deception axis is nearly orthogonal to it… |
| drive-sync-topology.md | 4013 | 2026-09-18 13:36 | project | - | - | Which jobs push to the tamara: Drive remote and which do not, plus the measured small-file upload rate (~97 files/min) |
| gates-must-assert-their-own-number.md | 3922 | 2026-09-21 23:34 | feedback | F-63,F-64,F-95 | - | Six chain gates have now logged PASS while measuring nothing; also, paper16's feature names are hardcoded in SEVEN places and two of them fail silentl… |
| github-push-to-zami456-expLB.md | 1224 | 2026-09-17 02:02 | feedback | - | - | Push to the EXISTING private repo zami456/expLB as collaborator SuraiyaShanaz — never create a new repo |
| hpkr-cannot-attribute-itself.md | 2457 | 2026-09-19 02:49 | project | - | - | HP-KR is 87% one lexical bit, and `deceptive = knows AND denies` with the fourth cell empty — so knowledge, wording and deception are perfectly confou… |
| judge-choice-barely-matters.md | 1731 | 2026-09-15 16:02 | project | - | Q7 | Q7 answered — Llama-3.1-70B vs 405B judging the SAME rollouts agree 91.2%, and differ only 0.8pp on the score-3 rate that builds the vectors |
| labpc-not-a-git-repo.md | 3047 | 2026-09-22 02:33 | project | F-104 | - | The lab PC's C:\Users\user4\expLB is a partial copy, not a clone — it cannot git pull, and script fixes silently fail to reach it |
| matched-control-kills-the-gain.md | 1877 | 2026-09-18 05:22 | project | - | - | Against a matched control at the same k, the unified persona axis is indistinguishable from PCA of the training dataset's own activations (-0.003)… |
| ndif-attention-memory-limit.md | 3219 | 2026-09-17 01:05 | project | - | - | NDIF's real per-call limit is the eager attention matrix (rows x heads x T^2 x 4), not padded tokens — and it makes ai_audit unrunnable |
| ndif-download-stalls.md | 1760 | 2026-09-15 13:56 | project | - | - | NDIF result downloads stall at ~35 KB/s on the lab PC; killing python and relaunching clears it |
| no-margin-from-one-layer.md | 2308 | 2026-09-21 23:34 | feedback | F-39,F-63,F-89,F-91,F-94,F-95,F-96 | - | Four separate headline margins have now failed a layer sweep; treat any single-layer number as provisional until swept |
| own-mistakes-correctness-over-speed.md | 2234 | 2026-09-20 12:09 | feedback | - | - | The owner wants mistakes owned, fixed and learned from out loud; token and time cost are explicitly not a constraint, correctness is |
| paper16-assistant-direction-not-lu-axis.md | 2031 | 2026-09-17 12:33 | project | - | Q1,Q4,Q10 | paper16's "assistant" axis is default minus mean(25 paper personas), only cos 0.30-0.37 with Lu et al.'s published axis… |
| paper16-used-paper-questions-only.md | 1590 | 2026-09-17 12:33 | project | - | Q1 | paper16 was generated as 5 variants x the paper-style questions (50/60 rollouts per persona, 1,400 total), NOT the plan's 10,000 on AA's 240-question… |
| pc-features-use-basis-not-contrast.md | 2326 | 2026-09-21 03:40 | project | - | - | pc:<name> projects onto the PCA BASIS, never the contrast — two bases over the same roles score identically by construction… |
| persona-axes-encode-register.md | 3955 | 2026-09-21 03:40 | project | - | - | HP-KR and sycophancy_mt break persona axes for the SAME reason — the axes encode register, not the labelled construct — and the cap bank CONFIRMED it… |
| persona-axis-diagnostics.md | 6938 | 2026-09-15 16:02 | project | - | - | Measured 2026-09-14 — PC1 in axes.pt is the Assistant Axis not a deception axis; the deception contrast is contaminated but sycophancy is clean |
| persona-set-decides-detection.md | 2051 | 2026-09-15 20:18 | project | - | Q1,Q4 | paper16 beats own100 at DETECTION, not just geometry — and residualising the Assistant Axis collapses own100 to near chance while paper16 survives |
| q10-sycophancy-axis-inverted.md | 2210 | 2026-09-17 01:07 | project | - | Q10 | Q10 behavioural half - on real capitulation data the paper16 sycophancy axis is significantly INVERTED (0.28-0.31)… |
| q11-sleeper-detection-is-falsity.md | 2982 | 2026-09-17 01:07 | project | - | Q11 | Q11 - base-model probes flag soft-trigger lies at 0.96, but it survives every shortcut check only as "answer is false for the passage"… |
| q2-answer-is-layer-dependent.md | 4083 | 2026-09-16 10:58 | project | - | Q2,Q4,Q6 | On the full corpus, persona coordinates beat the raw probe at layer 40 but lose badly at 20 and 22 — and the 5-dataset subset gave the opposite answer |
| q3-advantage-grows-with-n.md | 2094 | 2026-09-17 02:23 | project | - | Q3 | Q3 — the "raw degrades, axes improve with more data" trend is PROBE-DEPENDENT (lr_reg10 only) and not significant across datasets… (filename still says "grows") |
| q4-signal-survives-assistant-erasure.md | 1795 | 2026-09-17 02:24 | project | - | Q4 | Q4 re-run on the FULL corpus — erasing the Assistant Axis costs 0.005-0.018 AUROC (random erasures ~0.000)… |
| q5-semantics-does-not-predict-transfer.md | 3078 | 2026-09-16 19:35 | project | - | Q5 | Q5 answered 5b - sharing a label definition does NOT predict transfer; the apparent within-semantics advantage is provenance |
| q6-layer-choice-depends-on-method.md | 1704 | 2026-09-16 21:23 | project | - | Q6,Q8 | Q6 answered - 20 vs 22 is immaterial (median gap 0.012); moving to layer 40 is worth ~+0.025 for raw probes but ~+0.113 for persona methods |
| q8-per-token-beats-pooling.md | 3560 | 2026-09-16 21:22 | project | - | Q8 | Q8 answered in two halves - zero-shot, top-decile token scoring beats mean pooling (+0.053); TRAINED, pooled-mean probes beat token-trained… |
| regrouping-cannot-replace-personas.md | 2352 | 2026-09-16 18:08 | project | - | Q1,Q9 | Re-grouping the HF role library by disclosure stance buys only 11% of the contamination… |
| response-length-transfers-on-its-own.md | 2121 | 2026-09-16 19:39 | project | - | Q5 | Response length predicts the label in the same direction across 9 of 10 datasets, so it transfers by itself - measured at off-diagonal 0.553 |
| results-chapter-audit.md | 1480 | 2026-09-17 12:54 | project | - | Q3,Q5,Q8,Q11 | 2026-09-17 audit of plan/memory numbers against result JSONs — 29 discrepancies, written up in docs/thesis/persona-coordinates-70b-results.md… |
| resumable-acts-cache-branch.md | 4436 | 2026-09-17 12:03 | project | - | - | origin/feat/resumable-acts-cache holds a tested NDIF activation cache and token-budget batcher worth porting… |
| review-generated-prompts-before-ndif.md | 1805 | 2026-09-19 02:50 | feedback | - | - | The owner reviews generated persona prompts and questions before any NDIF run starts; build the bank, stop, show it, wait (cites docs/ONBOARDING.md, docs/REPORT.md) |
| role-label-rules-and-banks.md | 1571 | 2026-09-18 05:19 | project | - | - | Two labelling rules over the 275 AA roles (loose = own100's standard, 120 roles; strict = the paper's stance rule, 46)… |
| sarcasm-gate3-deception-axis-is-random.md | 2608 | 2026-09-22 14:33 | project | F-93,F-107 | - | Gate 3 ran: the deception axis on natural sarcasm is 0.54-0.58, inside the random-direction null; the SYCOPHANCY axis fires instead at 0.71 |
| save-insights-when-measured.md | 1706 | 2026-09-15 16:02 | feedback | - | Q1,Q7,Q11 | Write every significant finding to memory the moment it is measured, not at the end of the session |
| scp-truncates-at-200kib.md | 1935 | 2026-09-19 20:20 | project | - | - | scp from the lab PC silently truncates every file larger than 204,800 bytes to exactly 204,800 bytes and still exits 0… |
| significance-and-random-nulls.md | 2876 | 2026-09-17 01:07 | project | - | Q2,Q6,Q8,Q10,Q11 | Q2/Q6 differences are significant over test rows but NOT over datasets (9 datasets, BH); several zero-shot AUROCs, incl. Q11's 0.87-0.91, are no bette… |
| syc-dataset-erases-axes-advantage.md | 1774 | 2026-09-17 12:32 | project | - | Q2,Q4,Q6 | Adding sycophancy_mt as a 10th training/test dataset erases the persona-axes transfer advantage at EVERY layer… |
| unified-pc1-is-question-family.md | 1688 | 2026-09-17 22:10 | project | - | - | In our paper16 bank the unified-axis PC1 (47% variance, L40) is WHICH QUESTION SET a persona answered (cos -0.98), not harm; PC2 is harm (0.88)… |
| verify-after-interrupted-tool-call.md | 1250 | 2026-09-17 11:59 | feedback | - | - | When the owner rejects/interrupts a remote tool call, check the lab PC for its effects before saying "nothing ran"… |

### 3b. Plan files — `C:\Users\SURAIYA\.claude\plans\` (read-only; outside the repo but cited by repo docs)

| file | bytes | lines | mtime (+06) | title | IDs | cited by |
|---|---:|---:|---|---|---|---|
| `harmonic-cuddling-kay.md` | 109305 | 1690 | 2026-09-17 02:24 | "Persona Coordinates at 70B — full plan" | **defines Q1–Q11** (OUTCOMES section, each with pre-committed outcomes "1c", "5b", "7a"…; "✅ ANSWERED" dates 09-15..09-17); pipeline **Stage 1–5** | ONBOARDING.md ("the single most important file"), persona-coordinates-70b-results.md |
| `glowing-rolling-finch.md` | 6056 | 101 | 2026-09-21 16:44 | "A briefing document for the collaborator — plain language, with the numbers" | F-64..F-84 | docs/TODO.md names it as **the Stage plan** — but its current content has no "Stage" (grep, 0 hits) |
| `peppy-shimmying-pearl.md` | 12681 | 180 | 2026-09-18 04:20 | "Cross-generalization (10 × 10) — phase 2 …" | Q1, Q11 | – |
| `wiggly-sleeping-book.md` | 19075 | 349 | 2026-09-10 23:42 | "Assistant Axis pipeline on NDIF — 100 curated deception/sycophancy roles" | – | – |
| `src-assistant-axis-ndif-2-activations-nd-compiled-hejlsberg.md` | 9782 | 147 | 2026-09-14 18:01 | "Plan: single shareable Markdown code dump" | – | – |
| `distributed-roaming-torvalds.md` | 6314 | 105 | 2026-09-11 11:14 | "NDIF preflight status check — fail fast …" | – | – |

### 3c. Other Claude records (listed, not read)

- Older project memory for a previous clone path, `~/.claude/projects/c--Users-SURAIYA-Documents-thesis-liar-bench-llama-liars-bench/memory/`: `MEMORY.md` (300 B), `liars-bench-repo-map.md` (1962 B), `user_repo_familiarity.md` (1204 B), all 2026-07-09 14:31. Pre-dates the persona work.
- Session transcripts for this repo: 19 `*.jsonl` files, 110.0 MB, mtimes 2026-09-09 .. 2026-09-22, in the project dir above. They are the only record of what sessions actually ran; **not read** (a possible Phase-1 source for ghosts/orphans).

## 4. Documents that are NOT findings (listed once)

- **nnsight library docs — "library docs, not findings":** `docs/{concepts,usage,models,remote,patterns,errors,gotchas,reference,developing}/` = 105 `.md` files, every one added in a single commit `47e53c5` (2026-09-12, "assistant axis") and never changed; YAML front-matter like `title: Concepts Index … tags: [concept, index]`; grep finds no project term (`F-NN`, paper16, HP-KR, Liars, persona coord) in any of them. `NNsight.md` (root, 2375 lines, "NNsight: Design and Implementation (0.8)", same commit) and `docs/NNSIGHT_UPSTREAM.md` (provenance note: upstream commit `260c555…`, synced 2026-09-10 — DOC-ONLY) belong to the same drop. VERIFIED.
- **Upstream Liars' Bench / vendored files:** `README.md` (190 lines; 10 commits 2025-02-18..2026-05-07, last by Walter Laurito), `figures/figure_1/figure_1.html` + `table.tex`, `src/visualization/figure_1.html`, `src/false_cities/notebooks/tell_me_about_yourself.html`, and 5 `.md` under `src/probes/deception-detection/` (vendored Apollo code, commit 52281fa 2026-07-11, zami456).
- **Listings / data:** `remote_sizes.txt` (271 lines, "name|bytes" listing of remote result files, 3dfdb67 2026-09-21), `kd/results/fig_data.json`, `docs/thesis/role-labels/*` (see §2a).
- **Untracked / ignored in the working tree** (git ls-files --others): no untracked doc; ignored doc-like files are `CLAUDE.local.md` (audit), `.pytest_cache/README.md`, `dishonesty_bench.egg-info/*.txt`, and **logs** `logs/{judge_405b,judge_405b_raw,paper16_axes,paper16_chain,paper16_deception,paper16_judge,paper16_supervisor,paper16_sycophancy,paper16_vectors_all,paper16_vectors_s3}.log`, `aa-smoke.log`, `aa-smoke-win/*.log` — these are raw-run logs for Phase 1, not prose.
- **Submodules** `mask/`, `pacchiardi/`, `soft-trigger/`, `assistant_axis/`: clean (`git -C <sub> status` empty) and contain no project docs (grep for F-NN / expLB / paper16 / Liars' Bench in their .md: none). `behavioral-self-awareness/` is an empty stale gitlink.

## 5. Report-like file search (tracked, outside submodules)

| pattern | hits |
|---|---|
| `*REPORT*` / `*report*` | `kd/docs/REPORT.md`; the rest are data (`*knowledge-report*`, `*knowledge_report*`) or scripts (`register_crossgen_final_report_task.ps1`, `report_crossgen_done.ps1`) |
| `*RESULTS*` / `*results*.md` | `docs/thesis/persona-coordinates-70b-results.md`, `docs/thesis/sarcasm-pilot-negative-result.md` |
| `*SUMMARY*` | none on asix (`scale/scripts/score_summary.py` is a script); on pc-ood: 16 `results/*/summary.md` |
| `*thesis*` | `docs/thesis/` (3 .md + role-labels csv/xlsx), `figures/thesis/` (fig1–fig5 × pdf/png, 2026-09-21), `kd/scripts/make_thesis_figures.py` |
| `*draft*` | `full paper draft for this experiments/main (1).pdf`, `docs/thesis/hf275-role-labels-draft.md`, `src/assistant_axis_ndif/hf275_role_groups_DRAFT.json`, `cap/questions/cap_questions_v2_draft.jsonl` |
| `*.tex` | `figures/figure_1/table.tex` (upstream) only — no LaTeX source for the draft paper is in the repo |
| `*.docx` | none |
| `*.xlsx` | `docs/thesis/role-labels/role_labels_review.xlsx` |
| `*.html` | ours: `kd/docs/review_page.html`, `cap/docs/review_page.html`; upstream: 3 (see §4) |
| `*.pdf` outside `papers/` | the draft PDF; 188 figure PDFs under `figures/` (crossgen 251 files, persona-coords 26, thesis 10, older Liars' Bench figure dirs) |

## 6. Numbering schemes other than F-NN

| scheme | defined in | used in | maps to F-ids? |
|---|---|---|---|
| **Q1–Q11** (research questions) | `~/.claude/plans/harmonic-cuddling-kay.md` OUTCOMES (outside repo); one-line answers in `docs/ONBOARDING.md` §3; full write-up `docs/thesis/persona-coordinates-70b-results.md` §7 | memory `q2-…`..`q11-…` notes, `kd/docs/DESIGN.md`, `docs/LAYER_PREREGISTRATION.md`, a few ledger lines (e.g. F-38 → "Q11's finding", F-65 → "Q10's 0.71") | **No.** All answered 2026-09-15..17, before the ledger existed; `PERSONA_CONSTRUCTION.md` says pre-ledger facts live in memory, "Everything from F-01 onward is in the ledger" (DOC-ONLY). F-17 ("earlier sessions", status inherited) is the only ledger entry that explicitly imports earlier-session results. |
| **Q0–Q19** (cap question-bank items) | `cap/docs/REVIEW_PACKET*.md`, `cap/questions/` | `docs/OPEN_DECISIONS.md` A1, ledger F-26/F-28/F-46/F-52/F-53 etc. | no (item ids) |
| **Q20–Q29** (kd question items) | `kd/docs/REVIEW_PACKET.md` | – | no |
| **Q1–Q5** (external-review questions) | `docs/REVIEW_REQUEST_POOLING.md` reviewers' reply, as audited in F-90 | F-90, F-95 ("question 4") | answered in F-90/F-95 |
| **Q1–Q4** | `docs/papers/soft-trigger-notes.md` (its own open questions) | – | no |
| **Claim 1–10** (novelty numbering) | `docs/NOVELTY_CHECK_PROMPT.md` §"The ten claims" | `docs/NOVELTY_ASSESSMENT.md`, `docs/LAYER_PREREGISTRATION.md` (Claim 5, 9), ledger F-74 ("refutes Claim 4") | not 1:1; no table maps them |
| **Claim 1–3** (briefing numbering) | `docs/COLLABORATOR_BRIEFING.md` §3 ("### Claim 1 — … register"; "Claim 2 — A bank built for the right construct works"; "Claim 3 — The problem persona axes exist to solve has a simpler answer" = pooled training) | `docs/ROADMAP_TO_DEFENCE.md`, ledger F-98 ("Claim 2's targeted instrument") | the briefing cites ledger ids next to each claim (F-97, F-107, F-108, F-109 …) |
| **Result 1, Result 2** | `docs/REVIEW_REQUEST_POOLING.md` (Result 1 = pooled training, Result 2 = mean-of-directions mechanism) | ledger F-90, F-95 | Result 1 ≈ F-79/F-80/F-81/F-83; Result 2 ≈ F-86 (INFERRED from the cited ids in that file) |
| **Tier 0.1–0.4, 1.1–1.4, 2** | `docs/ROADMAP_TO_DEFENCE.md` | ledger titles | 0.1 → F-85 (then F-97), 0.2 → F-87, 0.3 → F-88, 0.4 → F-89 (then F-94) (from ledger titles "Tier 0.x —", DOC-ONLY) |
| **Gate 1–3** (sarcasm) | implicit: items 1–3 of §"Sequence" in `cap/docs/SARCASM_DESIGN.md` (1 format-matched control, 2 deadpan elicitation pilot, 3 one-directional pre-test) | `docs/thesis/sarcasm-pilot-negative-result.md` ("gates 2 and 3"), ledger F-106/F-107, memory `sarcasm-gate3-…`; `COLLABORATOR_BRIEFING.md` calls gate 3 "the second gate" | gate 1 → F-89; gate 2 → F-92/F-93 (+F-105); gate 3 → F-106/F-107 (INFERRED from the texts) |
| **G1, G2** (sarcasm pilot abandon-conditions) | `docs/thesis/sarcasm-pilot-negative-result.md` §2.1 (G1 markers ≤ 0.20; G2 bag-of-words AUROC ≤ 0.65) | same doc §3.1–3.2 | G1 → F-92, G2 → F-93 (INFERRED); both sit inside Sequence item 2 |
| **Stage 1–4** (execution tracker) | `docs/TODO.md` | ledger F-59 ("GATE FAILED") etc. | TODO rows cite F-ids (13) |
| **Stage 1–5** (pipeline) | `harmonic-cuddling-kay.md` §PIPELINE | older docs | no — **different meaning** from TODO.md's stages |
| **A1–A5, B1–B10, C** | `docs/OPEN_DECISIONS.md` | ledger F-38 ("OPEN_DECISIONS A3") | B-items cite F-ids |
| **NEXT items 1–3 (a–c)** | `kd/docs/NEXT.md` | ledger F-31 ("closes NEXT.md item 4"), F-40 ("NEXT.md item 3") | partly; note F-31 cites "NEXT.md item 4" and F-40 "item 3", but the committed NEXT.md has items 1–3 only and was first committed 2026-09-21 03:40 (6dd9f9b) — the version those entries referred to was never committed (INFERRED) |
| **Phase 0–6, §0.1–0.4** | `kd/docs/REPORT.md` | – | Phase 0 §0.1/§0.2 = same topics as F-01/F-02 (different numbers, see §7) |
| **§5a–§5e** | `docs/COLLABORATOR_BRIEFING.md` | ledger F-108 ("closes the §5d open risk"), F-109 ("§5e") | cross-referenced |
| **pass A / B / C** | activation stores (plan; `persona-coordinates-70b-results.md` §5) | everywhere | not findings |

## 7. Which doc is newest / most authoritative

**By last commit (VERIFIED):** `kd/docs/FINDINGS.md` and `docs/COLLABORATOR_BRIEFING.md` (43ee6d3,
2026-09-22 17:37) → draft PDF (cf11749, 16:51) → `docs/thesis/sarcasm-pilot-negative-result.md` (6c9a5cf, 03:25)
→ `docs/ROADMAP_TO_DEFENCE.md` (4d0d046, 02:46). Newest memory: `sarcasm-gate3-…md` and `MEMORY.md`
(2026-09-22 14:33). Everything else was last edited on or before 2026-09-21; `docs/ONBOARDING.md` last on
2026-09-19; `docs/thesis/persona-coordinates-70b-results.md` on 2026-09-18 (content "as of 2026-09-17").

**By what the docs claim (DOC-ONLY):** CLAUDE.md, TODO.md, NEXT.md, OPEN_DECISIONS.md and
PERSONA_CONSTRUCTION.md all point to `kd/docs/FINDINGS.md` as the record of measured facts. The
collaborator briefing presents itself as self-contained and cites ledger ids. Older docs point elsewhere
(see CONFLICT 2).

**CONFLICTs noticed in passing (not hunted systematically; none of the numbers were checked against raw data):**

1. **Ledger rule vs ledger history.** Preamble: "Nothing is edited or deleted". Git: 7 sections had
   existing lines replaced or removed after first commit — F-65, F-66 (dc24e96: caveats rewritten to say
   "F-67 tested …"), F-67 (fb651a7: source line), F-68 (bc701d9: "+0.051" → "+0.054" and a sentence
   replaced), F-89 (7b3f899: status rewritten to "superseded in its numbers by F-94"), F-93 (efe9d19:
   caveat "it is unfixed … still owed" replaced), F-94 (4345ee2, f0a54f4: caveat lines replaced). Most are
   labelled amendments in commit messages. Also F-100 and F-102 carry "AMENDED 01:05 / 01:52" blocks that
   already existed at their first commit (4d0d046, 02:46), so their pre-amendment text was never committed
   (INFERRED from the in-text times). VERIFIED (git diff) for the seven.
2. **Which doc is authoritative.** `docs/ONBOARDING.md` (last edit 2026-09-19): the plan file
   `harmonic-cuddling-kay.md` is "the single most important file" and memory is the "Index of measured
   findings"; it never mentions FINDINGS.md. `CLAUDE.md` (2026-09-21): FINDINGS.md is "the project-wide
   append-only ledger … every measured number in the repo". `kd/README.md`: `kd/docs/REPORT.md` "is the
   running results log". `docs/thesis/persona-coordinates-70b-results.md`: "the file wins" over plan and memory.
3. **HP-KR lexical AUROC.** `kd/docs/REPORT.md` §0.1: **0.870** (7,068 rows), and Phases 2–6 still
   "_not started_"; ledger **F-01: 0.913** on the 2,139 Llama rows and "supersedes: an earlier figure of
   0.870"; memory `hpkr-cannot-attribute-itself.md` and the MEMORY.md index still say "**87%** one lexical bit".
4. **"Claim 3" means two different things.** Novelty numbering: Claim 3 = "behavioural inertness does
   not imply representational inertness"; briefing numbering: Claim 3 = pooled training is the simpler
   answer. Claims 1 and 2 coincide in meaning; Claims 4–10 exist only in the novelty numbering. The ledger
   uses both (F-74 "Claim 4" = novelty; F-98 "Claim 2" = briefing).
5. **"Stage" and "Q" are overloaded** (§6). `docs/TODO.md` names `glowing-rolling-finch.md` as its stage
   plan, but that file's current content (mtime 2026-09-21 16:44, after TODO's "last updated 00:05") is the
   collaborator-briefing plan with no Stage content — the pointer is stale (VERIFIED content; INFERRED overwrite).
6. **CLAUDE.md summaries lag the ledger.** CLAUDE.md (last edit 09-21 11:41) states the cap result clears
   the bootstrap "(F-65, F-66)" without F-67 "the gain is LOCAL" (09-21 03:37), although the kd skill says
   "Never quote F-66 without F-67"; and it states the scale trend "(F-63)", which F-96/F-100/F-102
   (09-22) amend as single-layer and mean-pooling-specific. DOC-ONLY.
7. **`docs/NNSIGHT_UPSTREAM.md`** says everything under `docs/` except `docs/papers/` is nnsight's —
   no longer true (10 other project `.md` directly in `docs/`, plus 3 in `docs/thesis/`). VERIFIED (file list).
8. **`docs/REVIEW_REQUEST_POOLING.md`** is flagged by ledger F-90 as needing revision "before it goes to
   anyone else"; it has one commit (80fab2b) and was never revised; F-108 later cuts its +0.178. DOC-ONLY.
11. **Sarcasm gate numbering and staleness.** `docs/thesis/sarcasm-pilot-negative-result.md` (09-22 03:25)
    says gate 3 "was registered and never run"; ledger F-106/F-107 (09-22 16:51) and memory
    `sarcasm-gate3-…` record Gate 3 prepared, run, and failed. The same test is called "Gate 3" in the
    ledger/memory and "the sarcasm bank's **second** gate" in `docs/COLLABORATOR_BRIEFING.md` (lines 16,
    553); the thesis doc also has its own G1/G2 (pilot abandon-conditions). DOC-ONLY.
9. **Collaborator-draft versions.** Plan `glowing-rolling-finch.md` quotes a draft titled "One Direction
   Generalizes: …" (0.904 / 0.916 vs 0.648 / 0.732); the committed PDF is titled "Lying in Knowledge-Denier
   Persona Coordinates …" (0.916 vs 0.732 at L40). INFERRED: two versions of the same draft; which is
   current is a question for the owner.
10. **Data cited but absent locally.** `docs/thesis/sarcasm-pilot-negative-result.md` cites
    `cap/run-sarc-pilot/` (`marker_report.json`, `scores/*.json`); that directory is not in the working
    tree, not tracked and not ignored — NOT-FOUND locally (probably lab PC / Drive; Phase 1).

## 8. The F-number ledger: history (summary of `audit/phase0/fnumber_history.csv`)

Script: `audit/scripts/p0_docs_fhistory.py` (reads every committed version with `git show <c>:kd/docs/FINDINGS.md`,
splits at `### F-NN · date · title`). All statements in this section are VERIFIED from git unless tagged.

**Range, gaps, duplicates.** HEAD has 109 section headings, F-01..F-109, contiguous, no gaps, no
duplicates, no letter suffixes, in increasing order; the only other heading is `# Findings ledger`
(line 1). Heading dates: 2026-09-19 ×11, 2026-09-20 ×51, 2026-09-21 ×33, 2026-09-22 ×13, F-17 "earlier
sessions" ×1. Heading dates are not monotone: F-14 and F-16 are dated 09-19 but follow F-10..F-13 and
F-15 dated 09-20 (the preamble says entries are "in the order it was measured" — DOC-ONLY).

**Origin.** The file has one `A` and 38 `M` in `git log --follow --name-status` — no rename or copy. The
only file matching `*FINDINGS*`/`*findings*` ever added on any ref is this one. `### F-01` first appears
in 6432257 (`git log --all -S`), and the oldest commit on any ref whose .md/.html/.txt diff contains an
`F-NN` token is also 6432257. So **F-01..F-66 (66 sections, heading dates 09-19..09-21) entered git in one
commit, 6432257, 2026-09-21 03:26 ("Cap bank clears the dataset-level bootstrap …")**. The previous
asix commit is a796eaf (2026-09-19 14:01); there are no asix commits in between (on other branches:
9 commits by zami456 on 09-20 and 1 by "Mahdi" on `feat/llama-layer30`, 09-19 19:32). The drafting history of F-01..F-66 is therefore **not in git**;
63 sections have a heading date earlier than their first-commit day. Earlier findings live in
`kd/docs/REPORT.md` (09-19), memory notes (09-15..), and the plan file (outside the repo).

**Growth.** 39 commits (2026-09-21 03:26 → 2026-09-22 17:37): 6432257 adds F-01..F-66; then one new
section per commit through F-95, except bc701d9, fb651a7, 4345ee2, f0a54f4 and efe9d19, which add none
(they edit existing sections); 4d0d046 ("i dunno anymore
anything", 09-22 02:46) adds F-96..F-104 at once; 6c9a5cf adds F-105; cf11749 adds F-106, F-107; 43ee6d3
adds F-108, F-109. Per-commit table: `python audit/scripts/p0_docs_fhistory.py --commits`.

**Reuse / renumbering / deletion / reordering.** None. Every id seen in any version is still at HEAD; no
section was deleted, re-titled, moved or inserted before existing ones; the preamble never changed;
first-version vs HEAD text similarity is ≥ 0.5 for every section (no sign of an id reused for a
different finding).

**Edits after first appearance (8 sections).** Append-only additions: F-85 (4d0d046 adds a
`> SUPERSEDED BY F-97 (headline only)` block — the mechanism the preamble prescribes). Non-append edits
(existing lines replaced/removed): F-65, F-66 (dc24e96), F-67 (fb651a7), F-68 (bc701d9), F-89 (7b3f899),
F-93 (efe9d19 — the "F-93 amendment" commit, adds an "AMENDMENT 23:05" block and rewrites caveat (2)),
F-94 (4345ee2 "amended", f0a54f4 "second amendment"). Sections with in-text amendment markers at HEAD:
F-93 (1), F-94 (2), F-100 (1), F-102 (1).

**Corrections, as the ledger states them (DOC-ONLY; curated reading of each section's `supersedes:` /
`amends:` / `status:` fields, CORRECTION titles and SUPERSEDED BY lines — the script's automatic parse
over-collected in 4 sections, listed when it runs).** 34 sections say they correct / supersede / amend
another; 30 sections are targets. The families (arrow = "is corrected/superseded/amended by"):

- throughput projections F-35 → F-40 → F-45 → F-49 → F-50
- OOM budgets F-41 → F-42 → F-43
- the paper's baseline / scale ladder F-11 → F-39 → F-58 → F-60 → F-63 → F-96 → F-100 → F-101, F-102 (F-63 also amended by F-100, F-102)
- HP-KR matched slice F-20 → F-69 → F-71; F-03 → F-72 → F-75 → F-76
- cap bank F-65 → F-66 → F-67; F-65 → F-68
- pooled training F-79 → F-83, F-90, F-108; F-80 → F-81 → F-83; F-80 → F-108
- the abstract number F-85 → F-97 (SUPERSEDED BY line), F-85 → F-109 (margin)
- format-matched control F-89 → F-94 (status line in F-89)
- sarcasm pilot F-93 → F-105 (amends F-93's amendment, not its verdict)
- cap bank construction F-27 → F-30; F-28 → F-51 (body "CORRECTION TO F-28"); F-36 → F-46; F-98 → F-99
- non-correcting relations (column `other_relations_DOCONLY`): F-65 confirms F-13, F-62 and contradicts
  half of F-55; F-68 contradicts the other half of F-55; F-77 bounds F-66; F-80 confirms F-79;
  F-98/F-99/F-103 support F-97/F-98/F-99; F-103 bounds F-102; F-29 supplies the argument for F-13.

**Status field at HEAD (DOC-ONLY).** Every section has a status line. 93 start with `verified` (some
qualified, e.g. "verified (counts)", "verified, verdict provisional"); 5 start `provisional` (F-03, F-11,
F-13, F-40, F-42); F-27 "generation verified; scores provisional"; 2 `inherited` (F-15, F-17); 7
`measured …` (F-89..F-95); 1 `prepared and bounded` (F-106). "verified" means, per the preamble, "re-read
from the result file in this session" — a self-report, not checked here.

**F-ids that do not exist.** Every F-id cited in tracked files at HEAD (docs, scripts, JSON) is ≤ F-109;
memory cites ≤ F-107. **F-110..F-999: NOT-FOUND** in the ledger, any tracked or untracked file, the memory
dir, any of the 14 git refs, or any commit's diff (commands in §0). "F-112" appears only in
`audit/BRIEF.md:16` ("Findings are numbered like F-112") — the owner's recollection, not a finding.
"F-150" (4×) is the Ford F-150 in
`src/probes/deception-detection/data/rollouts/alpaca__plain__llama-70b-3.3.json` lines 157475, 157480,
328094, 328099; other F-1xx/F-3xx tokens in `data/raw/geometry_of_truth/likely*.csv` and soft-trigger
result CSVs are data text, not ids.

**Columns of `fnumber_history.csv`:** `f_id`, `heading_title_at_HEAD`, `date_in_heading`,
`first_commit`, `first_commit_date`, `n_later_commits_changing_section`, `n_later_commits_nonappend`,
`later_commits`, `title_changed` (y + old titles / n — all `n`), `similarity_first_vs_HEAD`,
`corrects_supersedes_amends_DOCONLY`, `later_said_to_correct_this_DOCONLY`, `other_relations_DOCONLY`,
`relation_field_text_DOCONLY` (raw `supersedes:`/`amends:`/`supports:` text, first 300 chars — includes
non-F supersessions such as F-01's "0.870"), `in_section_amendment_markers`,
`status_line_at_HEAD_DOCONLY`, `line_range_at_HEAD`.
