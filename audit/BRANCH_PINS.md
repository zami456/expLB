# Branch pins: the audited-up-to record (P17, 2026-09-24)

**Why this file exists.** Before P17 the audit had no record of which commit it had audited on each
branch. The Phase 6 scripts read the moving name `origin/paper`, but stamped their rows with a
fixed SHA. The reflog below shows that the name moved while those scripts ran (DF-95, P17). From
now on every audit read of another branch uses a SHA from this table, through
`audit/scripts/p17_common.py`.

`git fetch origin` ran once, 2026-09-24, before pinning. It moved no ref: the tips were already the
ones below.

| Branch | Owner | Previous audit mark | **P17 pin** | Tip date |
|---|---|---|---|---|
| `origin/paper` | zami456 | `ea1ce52b874e` (claims.csv `version_used`; COLLAB_PAPER_BRANCH.md:14-18) | **`16def2deec34aeee532511df116adb7de5522ca7`** | 2026-09-24 16:47 +06 |
| `origin/feat/persona-pc-ood` | zami456 | `45b16d90` (tip; 66 audit files cite it) | **`45b16d906da0e3da13b1abb34898e69311d688cf`** (ancestor of paper; nothing newer) | 2026-09-22 12:45 +06 |
| `origin/feat/llama-layer30` | zami456 (+1 Mahdi) | `df758c4b` (START_STATE.md) | `df758c4bb076f19304a87b801a3c7b9b139ea630` | 2026-09-19 |
| `origin/feat/resumable-acts-cache` | zami456 | `7c282990` | `7c282990f05b116d89571ddbab72d9eb198e28c7` | 2026-09-05 |
| `asix` / `origin/asix` (HEAD) | owner | `e917ecd3` (PROGRESS.md HANDOFF ~2731) | **`e917ecd3f95f8ec16c0ce3816e9878c0bacd4c7e`** + the uncommitted working tree below | 2026-09-24 05:16 +06 |
| `main` / `origin/main` | shared | – | `608b2a78` / `59cdaba4` | 09-09 / 09-08 |

The remaining `origin/*` branches (`Saving`, `copilot/*`, `claude/*`) hold no results. Their tips
are in `thesis_tools/survey/BRANCHES.md` §1.

## Uncommitted working-tree files at pin time (md5)

These are part of the owner's side of the pin. If an md5 changes, the file changed after P17 read it.

```
dc365834591b7295d2cebcbca2fef3d9  kd/docs/FINDINGS.md            (F-01..F-127)
e572fe0a4dc56a47c7a298053fd06ebe  kd/docs/NEXT.md
b002522a52fe160979ae6b7863df7e09  .claude/skills/knowledge-denial-context/SKILL.md
663a8d318be5b9c2c1468e55607b604d  cap/docs/SARCASM_DESIGN.md
3fe56ff2e642d3fe9bb884f1b7a9a198  cap/docs/review_page.html
3b0cfd2af469a1234bb978a4676859b8  docs/LAYER_PREREGISTRATION.md
114595a1c7fcaaa11bbc0f9e5becf56b  docs/NOVELTY_ASSESSMENT.md
b7195733116d893b5fa4b63de628c565  docs/NOVELTY_CHECK_PROMPT.md
b947bca61d2e292aa6038b8005b50d3e  docs/REVIEW_REQUEST_POOLING.md
57de188267571e93bbedad1e4c14d140  docs/thesis/persona-coordinates-70b-results.md
85aeb023c7e9295ea0860eb98a98ad22  docs/thesis/sarcasm-pilot-negative-result.md
```

## Provenance evidence: when the local `origin/paper` moved

`git reflog show --date=iso origin/paper`:

```
16def2de  2026-09-24 16:48:37 +0600  fetch: fast-forward
d1b88154  2026-09-24 16:45:34 +0600  fetch: fast-forward
2ad758fb  2026-09-24 15:44:27 +0600  fetch: fast-forward
ea1ce52b  2026-09-23 23:47:57 +0600  fetch: storing head
```

Timings of the scripts that read the ref:

- `p16_level1_verdicts.py` wrote `claims.csv` at 16:40. Its backup `claims.csv.before-phase6-level1` is also dated 16:40.
- `p16_level2_verdicts.py` has an mtime of 15:39. Its backup `claims.csv.before-phase6` is also dated 15:39.

Both stamp `origin/paper @ ea1ce52b`. The Level-1 pass most likely read `2ad758fb`, not `ea1ce52b`.
P17 Agent A settles this by re-deriving every stamped value at both SHAs.

## Backups taken before P17 writes

`audit/claims.csv.before-p17`, `audit/experiments.csv.before-p17`,
`audit/phase1/fmap_all.csv.before-p17`, `audit/claims_inventory.csv.before-p17`.

## Addendum (P17 lead, after the Wave 3 writes)

The lead appended **F-128** (the P17 summary) to `kd/docs/FINDINGS.md`, appending only
(`git diff --numstat`: 763 added, 0 deleted, counted against HEAD).
- The ledger md5 was **dc365834** at the pin and is **f81856ef** after the F-128 append.
- F-01…F-127 are unchanged. The old file content is a byte prefix of the new one; the append script asserted this.

Agents E and F then inserted P17 update blocks into other working-tree docs, so those md5s changed
too. Their logs are `audit/p17/E_LOG.md` and `F_LOG.md`. The snapshots taken before their writes
are in `audit/p17/pre_wave3/`.

## Addendum 2 (2026-09-24, after the thesis_guide build): origin/paper moved past the pin

The reflog shows three fetches after the pin. The new commits are zami456's and **are NOT audited**:

| Commit | Time (+06) | Subject |
|---|---|---|
| `54dc08c4` | 18:48 | Rewrite the abstract around how the persona bank was built, not around its numbers |
| `f24ca67c` | 19:27 | Reframe the abstract: the black-box/white-box trade-off as the opening |
| `d5b92c5a` | 19:34 | Frame the abstract's results as zero-shot against supervised baselines |

`git diff --stat 16def2de d5b92c5a`: only `paper/sections/abstract.tex` (+24/−29) and `paper/main.pdf`.
No result file, figure or numbers.tex key changed.

**What still needs checking before quoting the new abstract:**
- "We extend the persona bank with ten roles…". This is an attribution question. See `thesis_guide/PITFALLS.md` §attribution and `audit/methods/ATTRIBUTION.md:83`.
- "the baseline probe … flags nothing, as does every other detector we tried". This claim has no audit row.
- Its `\V{}` macros resolve through numbers.tex, which is unchanged at 16def2de, where they were verified.

The pin stays at 16def2de for results. Re-pin when this is audited.
