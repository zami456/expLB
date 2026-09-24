"""Writes audit/phase1/sessions/FINDINGS_NOTES.md (transcript-mining deliverable 4).

Re-run: python audit/scripts/p1_sessf_write_notes.py
"""
import io
import os

DOC = r'''# Transcript mining — method, caveats and the 15 most valuable recoveries

**Everything recovered here is DOC-ONLY.** Claude session transcripts are model and user
text, not verified data. Every number below is a **lead for Phase 2**, which must recompute
it from the result files. A transcript number agreeing with the ledger proves only that the
ledger copied the console correctly — not that the console was right.

## What was mined

`C:/Users/SURAIYA/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB/*.jsonl`
— 19 files, ~140 MB, 2026-09-09 to 2026-09-22. Two were excluded: `4f0b290f…` (this audit
session) and `1a88b065…` (its sidecar). Of the remaining 17, five are one-command stubs
(< 4 KB) and the substantive corpus is twelve sessions.

Transcript timestamps are **UTC**; the ledger's `F-… · 2026-09-xx` headings are **local
(UTC+06)**, so an entry dated 09-22 was usually written at `2026-09-21T18:xxZ`.

## Method

1. **Flatten** (`audit/scripts/p1_sessf_flatten.py`). Streams each JSONL line by line and
   writes one TSV row per 1,200-character text chunk:
   `session8 | lineno | timestamp | kind | chunk# | text`, with newlines and tabs escaped so
   every chunk is one physical line. `kind` ∈ `user`, `asst`, `think`, `sys`,
   `tool_use:<Tool>`, `tool_res`. Never loads a file into memory (the laptop has little RAM).

   ```
   FLAT=<scratchpad>/flat
   PYTHONIOENCODING=utf-8 python audit/scripts/p1_sessf_flatten.py \
       "C:/Users/SURAIYA/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB" \
       "$FLAT" 4f0b290f 1a88b065
   ```

   Result: 8,948 `tool_use` chunks, 7,967 `tool_res`, 4,237 `asst`, 2,113 `user`, 498 `sys`,
   149 `think`. **Printed numbers live in `tool_res`; the command that produced them is the
   `tool_use:Bash` or `tool_use:PowerShell` row at the same or a slightly lower `lineno`.**

2. **Search** (`audit/scripts/p1_sessf_find.py`): streaming regex over the flat corpus.

   ```
   PYTHONIOENCODING=utf-8 python audit/scripts/p1_sessf_find.py "$FLAT" "0\.9134" \
       --max 40 --win 300 [--kind tool_res] [--sess b035ccb3] [--ctx 2]
   ```

   Whole records are read back with awk on field 2 (`lineno`), printing fields 4, 5, 6.

3. **Printed-number recovery** (deliverable 1). 72 target F-numbers were taken from
   `audit/phase1/fmap_all.csv` — the 43 rows with `script_writes_file = n` plus the 29 mixed
   rows (`y (…) / n (…)`) whose headline statistic was print-only. They are listed in
   `findings_targets.csv`. Four subagents worked 18 ids each, searching 2–4 distinctive
   values per finding, then locating the producing command.

4. **Unrecorded-findings sweep** (deliverable 2). Mechanical filter first
   (`audit/scripts/p1_sessf_unrecorded.py`): build the set of every 3- or 4-decimal number in
   a "recorded" corpus, then report transcript records whose numbers are in none of it (in any
   of their 4-, 3-decimal spellings). The recorded corpus is 112 files / 1.44 MB:
   every tracked `*.md` outside the vendored nnsight docs and the submodules (so
   `kd/docs/FINDINGS.md`, `docs/`, `docs/thesis/`, `cap/`, `kd/`), all 45 auto-memory files,
   the collaborator's 17 committed `results/*/summary.md` on `origin/feat/persona-pc-ood`,
   and `audit/phase0/draft_text.txt` (the draft PDF's extracted text).

   ```
   PYTHONIOENCODING=utf-8 python audit/scripts/p1_sessf_unrecorded.py "$FLAT" \
       "<scratchpad>/recorded.txt" "<scratchpad>/unrec_asst.tsv" --kind asst --min 1
   ```

   Then keyword sweeps for conclusion language (`retract`, `withdraw`, `I was wrong`,
   `negative result`, `does not hold`, …) and a read of all 830 non-boilerplate owner messages.

5. **Verification of the subagents' work** (`audit/scripts/p1_sessf_verify.py`). For every one
   of the 144 recovered rows, the longest digit-bearing token of the quoted snippet must appear
   in that session's flat file **at the claimed lineno**.

   ```
   PYTHONIOENCODING=utf-8 python audit/scripts/p1_sessf_verify.py "$FLAT" \
       audit/phase1/sessions/findings_printed_numbers.csv
   ```

   **Result on the merged file: 135 OK, 6 MISS, 3 skipped** (the `found = n` row and two rows
   whose "snippet" is a command, not a measurement). **All 6 misses were checked by hand and are
   artefacts of the check, not of the data** — the snippet contains an elision (`...`) the agent
   inserted, or a Windows path whose backslashes the flattener stores doubled. Four were confirmed
   present by direct awk on the flat file. **No fabricated row was found.**

## Coverage caveats — read before trusting an absence

- **A `tool_res` in the transcript is the truncated text the model was shown**, not
  necessarily the process's full stdout. A number cut off by Claude Code's own truncation is
  unrecoverable here.
- **Background-task output files are not in the transcript.** Long runs wrote to
  `…/tasks/<id>.output` and the session read only parts of them. The F-99 / F-100 sweeps are
  the main example.
- **The flattener captures `user`, `assistant` and `system` records only.** It does not
  extract `attachment` records (auto-attached file contents, reminders, plan-mode snapshots),
  `file-history-snapshot` records (whole file versions), or the `toolUseResult` field — the
  last of these duplicates the `tool_result` block, so nothing is lost there, but a number
  that appeared *only* inside an attachment would be missed.
- **Sessions 2026-09-09 to 2026-09-13** (`f8aefa9d`, `47aabfcb`, `0d66bacf`, `2394bebd`,
  `40b4a5dd`, `638f732c`, most of `a748e742`) are almost entirely infrastructure: Colab, SSH
  key setup, Tailscale, Google Drive/rclone, NDIF shards for the 100-role Assistant-Axis run.
  They contain very few measurements, and the ones they do contain are in the
  `aa-run-live-labpc` memory note. Do not read "nothing found there" as "nothing happened".
- **The unrecorded sweep is number-driven.** A finding stated only as a percentage, a count,
  a ratio or plain prose can slip through. The keyword sweeps partly cover that, not fully.
- **`~/.claude/plans/*.md` and the lab-only `thinking/plan.md` were NOT counted as "docs"**
  (the brief's list is FINDINGS.md, docs/, docs/thesis/, memory, his branch). Two of them are
  byte-identical to each other and define Q1–Q11. Anything flagged unrecorded here was
  additionally checked against the main plan file where it looked plausible.
- **The comparison in `matches_ledger` is DOC-vs-DOC.** It says whether the console and the
  ledger agree, nothing about whether either is right.

## How complete is the written record? (a measured answer)

Over the whole corpus, counting distinct 3–4 decimal numbers and accepting a 3-decimal
rounding as a match:

| where the number was printed | distinct values | absent from every doc | recorded |
|---|---:|---:|---:|
| assistant messages (stated conclusions) | 833 | 44 | **94.7%** |
| tool output (everything printed) | 3,326 | 454 | 86.3% |

So the ledger, the thesis document and the memory notes capture nearly every number the
project ever *concluded*. The gap is in supporting detail: per-layer tables, controls,
diagnostics and self-checks. That shapes deliverable 2 — it lists 35 items, and they are
mostly controls and provenance, not lost headlines.

## Deliverable 1 — result

`findings_printed_numbers.csv`: **144 rows, all 72 target F-numbers covered, none missing.**
Merged from the four batch files by `audit/scripts/p1_sessf_merge.py`.

- `found`: 136 `y`, 7 `partial`, 1 `n` (**F-44 alone was not recovered** — it is also the one
  `NOT-FOUND` row in `fmap_all.csv`).
- `matches_ledger`: 113 `y`, 24 `partial` (a named discrepancy each), 4 not-`y`.
- Every row carries the producing command where it was visible, so Phase 2 can re-run the
  original one-liner rather than reinvent it. Many are self-contained python heredocs over a
  committed CSV.

## The 15 most valuable recoveries

1. **RF-16's cause is in the launch command, verbatim.** `1ac8bb19` L996,
   2026-09-21T18:51:15Z: `--out_root run-crossgen\scale_70b --bank_name paper16_70b --bank
   $r\run-kd\axes_vectors_s3.pt`. The 70B k=3 sweep behind F-100/F-101/F-102 ran the **kd**
   bank under a **paper16** label. The session justified it from L978, which printed only the
   basename `bank: axes_vectors_s3.pt` — so the match to the L40 cell was assumed, never checked.

2. **RF-13's cause: a reboot, not a person.** `1ac8bb19` L2344, 2026-09-22T09:46:31Z:
   `booted 09/22/2026 15:34:30` and `triggers: MSFT_TaskBootTrigger delay=PT20M / PT25M /
   PT30M`. ~24 chains fire over ~50 minutes after any boot. The session's own correction
   (L2351) is that `Next=none` never meant a task could not fire. Full detail in
   `findings_rerun_origin.md`.

3. **F-77 / F-81's "37/80 (46%)" is not an observed count.** The only printed counts
   (`b035ccb3` L10333) are `axes beats raw at 28/80 layers; beats datapca at 31/80`. F-76's
   bootstrap (L10370) printed *fraction of positive layers per resample: median 0.46*, and
   0.46 × 80 = 36.8. The ledger appears to have turned a bootstrap median into a layer count.
   This also resolves the F-75-vs-F-77 conflict in F-75's favour. **Phase 2 recompute.**

4. **F-24 understates by 40%.** The ledger's 40.22 GB was measured mid-download; once the
   download stopped the same session measured **56.64 GB** four times and explicitly retracted
   the 40 GB figure (`b035ccb3` L4293, L4319, L4383, L4393, L4410). F-24 was never updated.

5. **F-39's dataset choice is settled.** `b035ccb3` L5240: the producing command is
   `paper_comparison.py … --datasets convincing-game instructed-deception mask ai_liar
   roleplaying`, and the output reads `raw off-diagonal mean = 0.8090 over 20 pairs`.
   That closes the fmap CONFLICT flag: F-39 used the MASK set. F-58 (L6784/L6796) printed the
   HarmPressureChoice set — so **F-39 and F-58 used different "paper fives"**.

6. **F-58 → F-63 is a same-day re-run with a different answer**, 78 minutes apart, mask swapped
   for harm-pressure-choice: 3B raw 0.708 → 0.7619 and the gain +0.089 → **+0.020**; 8B raw
   0.743 → 0.7980 and +0.086 → **+0.013**. The 3B/8B matrices were overwritten, so F-58's
   numbers now survive only in the logs and this transcript.

7. **F-01 / F-02 / F-16 all come from two prints**, both recoverable as runnable heredocs over
   `results/subsets/harm-pressure-knowledge-report.csv`: `b035ccb3` L233 (all 7,068 rows:
   AUROC **0.870**, accuracy **0.821**) and L1214 (Llama-only: **0.913**, 2,139 rows,
   139 deceptive / 2,000 honest, text-matched 258, knowledge-matched 2,001).

8. **F-02's missing 125 rows are a table omission, not a parser drop.** The printed cross-tab
   has five cells; the ledger drops `(not-know, not-deny, honest) = 57` and
   `(know, not-deny, deceptive) = 68`. 57 + 68 = 125.

9. **F-20's now-destroyed state survives.** `b035ccb3` L3158:
   `passA/passC: HP-KR rows 200 -> matched 98, labels {1: 89, 0: 9}`. The stores were topped up
   later, so this print is the only record of the 89:9 state F-20 describes.

10. **F-65's sign "conflict" dissolves.** The same `6_axes.py` run printed `+0.838` for the
    *all-roles* PCA (`b035ccb3` L7840) and `−0.838` for the *capitulation-basis* PCA (L7881,
    L7935) over the same 13 roles — one quantity under two PC1 sign pinnings, not a discrepancy.
    PC1 variance 56.8% agrees everywhere.

11. **F-91 mixes seeds.** The falsify log (`b035ccb3` L12545) prints `L32 s42 real=0.8945`,
    `L36 s42 real=0.8919`, `L36 s7 real=0.8942`, `L40 s42 0.8828`. The ledger's
    0.8945 / 0.8942 / 0.8828 triple takes L36 from **seed 7** and the other two from seed 42 —
    and the session's own text at L12548 shows both values were known when the entry was written.

12. **F-95's contemporaneous coverage was 55/63, with 51/51 at layers ≥ 16.** `b035ccb3` L12889
    (2026-09-21T16:31:14Z): `GATE coverage measured 55/63 cells clear the best null by >0.02`,
    and L12901: `cells clearing >0.02: 55/63 | layers >=16 …: 51/51 | layers <16: 4/12`. The
    ledger (FINDINGS.md:4275) carries **both** numbers correctly. PROGRESS.md's flag
    "55/63 in the text vs 51/63 in the re-run falsify.status" should be re-checked — 51 is the
    printed `51/51` denominator, so the flag may be a misread rather than a re-run difference.

13. **F-21 mixes two deployments.** The `7.37 GiB allowed` cap, the 38 MiB and 14 MiB rows are
    the **3B** on GPU 3 (44.42 GiB total). The **140 MiB** row was measured on the **8B**
    deployment, GPU 0 (139.80 GiB total, process at 18.35 GiB), where no "allowed" line appears
    at all. The ledger presents all three as one 3B sweep.

14. **F-61's "50 rollouts per role" comes from one file.** The print says
    `paper16: 26 roles  1400 rollouts  54 per role`; the 50 was measured on
    `sorted(glob('run-paper16/responses/*.jsonl'))[0]` — a single role. 26 × 50 = 1,300 ≠ 1,400.
    Separately, F-61's paper16 side is 109 pairs against raw 0.6800 while the kd side is 110
    pairs against raw 0.6623, so its "twelvefold" ratio rests on an unstated asymmetry.

15. **Two cap axis "variants" are the same contrast, and PCA bases over identical roles diverge.**
    `b035ccb3` L7881: `cos(capitulation, cap_vs_default) = +0.9313 norm=4.411` and
    `cos(capitulation, cap_only) = +0.9313 norm=4.411`; and with `A roles == B roles : True`,
    `max|A-B| over all layers/PCs : 1.347` against `max|A-C| : 0.663`. Three declared variants
    are two distinct contrasts, and component-level differences are driven by the PCA basis, not
    the role set.

Runners-up worth reading in the CSVs: F-06's `0.784 → 0.509` is an orthogonalisation effect
*within* L40 and is re-quoted downstream as an L20-vs-L40 *layer* effect; F-34's ledger says
"best q 0.130 (L44)" while the same print gives q = 0.127 for L40; F-81's "13 of the 21 layers"
against a printed wins column of 12; F-86's margin +0.0284 against its comparator's own
subsampling spread of 0.0824 ("INSIDE the subsampling spread", printed three times); F-70's
"2 of 10 novel" came from a user-pasted review that names three defensible claims, not from any
tool; and F-69 / F-71's complete gate block survives in the transcript although RF-14's guard
bug wiped the `.status` files.

## Deliverable 2 — result

`findings_unrecorded.csv`: **35 rows.** Because the docs already hold 94.7% of concluded
numbers, these are dominated by **controls, diagnostics and provenance**, not lost headlines.
The heaviest ones:

- **Axis norms are not comparable** (paper16: sycophancy 1.535 vs deception 1.0037; kd: 1.707
  to 4.342 at L40). Ranked metrics are unaffected; any magnitude comparison is not. In no doc.
- **Across the full 80-layer sweep the persona axes lose more often than they win** — 28/80
  against raw, 31/80 against dataset PCA.
- **The kd bank's near-miss**: `pc:unified@2 − random@2 = +0.2861`, CI [+0.147, +0.427],
  q = 0.067 — larger than anything F-31 quotes, and unrecorded.
- **Every per-layer scale table has a `random` control column** that no F-entry quotes.
- **The per-dataset zero-shot random-direction nulls**: at L40 mean the deception axis beats the
  1,000-direction p95 on only 5 of 9 datasets.
- **Validation evidence nobody wrote down**: the Pass-C streaming probe checked against
  scikit-learn to 3.66e-06; `cos(mean,last) = 0.388–0.603` proving the two poolings are distinct
  vectors; the WMDP scorer's 5/8 smoke.
- **Four operational facts from 2026-09-22**, all inside the audit's own blind spot: the
  15:34:30 reboot, the 24 chains disabled at 17:31, the five-hour Drive-sync outage repaired at
  17:26, and a stale `.git/index.lock`.

## Files this job produced

| file | what |
|---|---|
| `audit/phase1/sessions/findings_printed_numbers.csv` | 144 rows, 72/72 F-numbers (merged) |
| `audit/phase1/sessions/findings_printed_numbers_p1..p4.csv` | the four batch files, kept for provenance |
| `audit/phase1/sessions/findings_unrecorded.csv` | 35 findings/numbers in no doc |
| `audit/phase1/sessions/findings_rerun_origin.md` | the two 2026-09-22 re-run events, with quotes |
| `audit/phase1/sessions/findings_targets.csv` | the 72 target F-numbers and their key numbers |
| `audit/phase1/sessions/FINDINGS_NOTES.md` | this file |
| `audit/scripts/p1_sessf_flatten.py` | JSONL → grep-able TSV (streaming) |
| `audit/scripts/p1_sessf_find.py` | streaming regex search over the flat corpus |
| `audit/scripts/p1_sessf_unrecorded.py` | numbers printed in the sessions but in no doc |
| `audit/scripts/p1_sessf_verify.py` | checks every recovered snippet against its claimed line |
| `audit/scripts/p1_sessf_merge.py` | merges the four batch CSVs |
| `audit/scripts/p1_sessf_write_rerun.py`, `p1_sessf_write_notes.py` | generate the two .md files |

The flattened corpus itself lives in this session's scratchpad (`<scratchpad>/flat`, ~17 MB,
18 TSV files) and is rebuilt in about a minute by `p1_sessf_flatten.py`. It was deliberately
kept out of `audit/` to avoid a second copy of the owner's private session history in the repo.

## What Phase 2 should take from this

1. Treat the four `matches_ledger = n` rows and the 24 `partial` rows as the first recompute
   queue: **F-24, F-58/F-63, F-77/F-81 (37/80), F-91 (seed mix), F-61 (54 vs 50 per role),
   F-34 (q 0.130 vs 0.127), F-21 (two deployments), F-39/F-58 (two "paper fives")**.
2. Where a row carries a `producing_command`, re-run that exact command against the current
   files — a one-line reproduction that also tests whether the input changed under RF-13.
3. `findings_unrecorded.csv` rows are candidates for `DOC_FIXES.md`, not for the paper: most are
   controls the project ran and then failed to report, which strengthens the write-up rather
   than weakening it.
'''

out = os.path.join("audit", "phase1", "sessions", "FINDINGS_NOTES.md")
with io.open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write(DOC)
print("wrote", out, len(DOC), "chars")
