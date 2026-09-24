# Phase 3, workstream P3-C — stale foundations and weak spots

Written 2026-09-23 by the P3-C agent. Read `audit/BRIEF.md`, the RED FLAGS section of
`audit/PROGRESS.md` and `audit/phase3/PHASE3_INSTRUCTIONS.md` first.

**Two questions.** Half 1: is any surviving claim still standing on a result file that was later
regenerated, overwritten or replaced — and if so, does the number move? Half 2: where is the
evidence thin — one seed, one layer, a missing control, a leak, a crashed run, a live code bug —
and which *reported* claims does that thinness actually threaten?

Every statement is tagged **VERIFIED** (checked in a file or by a script run during this audit) ·
**DOC-ONLY** · **CONFLICT** · **NOT-FOUND** · **INFERRED** · **NOT MEASURABLE**.

**Terms used throughout.** *Persona coordinates* = a small set of directions in the model's internal
activations, built by making the model role-play deceptive and honest characters; the method under
replication trains lie-detectors inside those coordinates instead of on the raw activations.
*raw* = the ordinary high-dimensional activations, the baseline the method must beat. *datapca* =
a matched control of the same low dimension built from the data itself, not from personas.
*random* = a random projection of the same dimension — a noise floor. *off-diagonal mean* = train a
probe on one dataset, test on all the others, and average: the project's transfer score.
*BH q* = a p-value corrected for how many tests were run at once (Benjamini–Hochberg).
*LODO* = leave-one-dataset-out. *Bank* = a saved set of persona vectors.

**Scripts written for this workstream** (all read-only, all re-runnable from the repo root):
`audit/scripts/phase3/p3c_bank_fingerprint.py`, `p3c_store_pairs.py`, `p3c_layer_rank.py`,
`p3c_layer_seed_summary.py`, `p3c_seed42_rank.py`, `p3c_f108_reach.py`, `p3c_ceiling.py`,
`p3c_who_clears.py`.

---

## Summary table — ranked by threat to a reported claim

| # | Item | Threatens | Does the number move? | Tag |
|---|---|---|---|---|
| **W1** | Dataset-level significance ceiling, measured project-wide | every "persona coordinates beat raw" headline | no number moves; **nothing clears** at the operating points | VERIFIED |
| **W2** | Layer 40 is a favourable layer *for persona features specifically* | draft contribution 4; F-65, F-91, F-96, F-100 | ~0.05 of free margin | VERIFIED |
| **S2** | RF-16 70B bank mix (repaired) | F-100/F-101/F-102, draft topic 4 | **+0.075 → +0.054**; **+0.008 → −0.013** | VERIFIED |
| **W3** | Single seed everywhere the `paper` probe is used | all 11-dataset and all scale-rung numbers | seed range 0.033 mean / 0.168 max | VERIFIED |
| **W4** | F-108 row-level leak between HP-KR and kr-matched | F-79/F-80/F-81/F-86/F-95 (+0.178) | −0.013 to −0.056 | VERIFIED |
| **S1** | 09-22 boot-trigger wave rewrote ~58,000 lab files | 20 F-entries | raw stats identical; **one verdict flips (F-31)** | VERIFIED |
| **S5** | Bank provenance is unrecoverable from a matrix file (live defect) | any future bank comparison | no current claim moves | VERIFIED |
| **S4** | F-58's inputs no longer exist | F-54 still repeats its error | NOT-RECOMPUTABLE | VERIFIED |
| **S7** | own100 bank never rebuilt in fp32 (run died at 1 of 100) | F-90; "persona set decides detection" | NOT MEASURED | VERIFIED file fact |
| **W5** | Live code defects: DF-48 + three `str(e)[:200]` sites (2 new) | reproducibility, not a number | — | VERIFIED |
| **S3** | paper16 old-vs-fp32 vectors | nothing | **no** — sign flips are at L23/L65, unused | VERIFIED CLEAN |
| **S6** | Baseline columns of the bank stores | nothing | **no** — 1,512/1,512 byte-identical | VERIFIED CLEAN |
| **W8** | The collaborator-bank trained arm duplicates the kd_s23 arm | nothing reported | orphan; tautological if ever quoted | VERIFIED |
| **W6/W7** | Crashed / partial runs; 5:1 class imbalance | unreported arms | — | VERIFIED |

---

# HALF 1 — STALE FOUNDATIONS

## S1 · The 09-22 boot-trigger wave: checked, and one verdict flips

**What was tested.** Three hours before the audit began, the lab PC rebooted and ~17 boot-triggered
scheduled tasks re-ran, rewriting about 58,000 result files — after many of the findings that cite
them had already been written (RF-13). I asked: which findings cite a rewritten file, were they all
re-checked, and did any number move?

**What happened.** Twenty findings cite an artifact that was regenerated after the finding was
written (`audit/phase1/fmap_all.csv`, rows tagged `REGENERATED-AFTER-F`): F-04, F-11, F-12, F-17,
F-19, F-20, F-31, F-34, F-57, F-58, F-65, F-66, F-67, F-82, F-83, F-84, F-87, F-89, F-93, F-105.
**All twenty have a Phase-2 claim row with a verdict** — none was left unchecked
(cross-join of `fmap_all.csv` against `audit/claims.csv`, 0 orphans).

The underlying statistics did not move. Keyed comparisons of the regenerated files (matching on
block, pooling, layer, probe and comparison rather than on row position) found every effect size,
confidence interval and raw p-value identical between versions; **only the multiple-testing
correction moved**, because the test family grew (e.g. the root file went from 192 to 4,788 tests;
`audit/phase1/significance_keyed_diff_sample.txt`).

**That is enough to flip exactly one verdict.** F-31 states "nothing beats raw activations" over a
4,032-test family. On the regenerated 6,300-test family that is **false**: `pc:kd_assertion − raw`
at layer 68, mean pooling, paper probe, 11 datasets reads **+0.1497 with q = 0.0467 — it clears**.
I reproduced this independently, from the cached lab file, without reading the Phase-2 note first
(`p3c_who_clears.py` output line
`/kd11/banks/kd/significance.json | pc:kd_assertion - raw | L68 mean paper blk10 diff=+0.1497 q=0.0467`).

**How sure, and why.** VERIFIED, HIGH. Two independent routes reach the same test. The direction of
the version effect is also understood: the larger family is a strict superset with identical raw
statistics, so it is the *more* conservative correction, and a test clearing there is a stronger
result, not a weaker one.

**Why it matters.** The wave is not a data-integrity problem; it is a *bookkeeping* problem. Any
sentence of the form "X clears / does not clear significance" is only true of a named test family,
and this project's families changed size at least three times. Every such sentence in the write-up
needs the family named beside it.

**Evidence.** `audit/PROGRESS.md` RF-13, RF-14; `audit/phase1/fmap_all.csv`;
`audit/phase1/significance_keyed_diff_sample.txt`; `audit/claims.csv` rows F-04, F-31, F-34, F-67;
`audit/scripts/phase3/p3c_who_clears.py`.

---

## S2 · The 70B scale rung mixed two persona banks — repaired, and the number does move

**What was tested.** The project's scale story is that the persona advantage shrinks as the model
grows: 3B → 8B → 70B. Phase 1 found the 70B rung was built from a *different* persona bank than the
3B and 8B rungs, because two folders each contain a file called `axes_vectors_s3.pt` (RF-16).

**What happened.** The audit re-ran the owner's own `27_crossgen.py` with everything identical
except the bank, into a new folder. Over the same 19 layers:

| rung | best persona feature − matched control | vs raw |
|---|---|---|
| 3B | +0.1197 | +0.0328 |
| 8B | +0.1027 | +0.0089 |
| 70B **as published** (kd bank) | +0.0751 | +0.0079 |
| 70B **like-for-like** (paper16 bank) | **+0.0541** | **−0.0131** |

The headline conclusion survives and gets *steeper*: the decline is +0.120 → +0.103 → **+0.054**,
not → +0.075. But one published sentence reverses: "against raw activations the 70B persona
advantage is essentially nil" (+0.008) becomes **negative** (−0.013). At 70B, averaged across depth,
the persona features are worse than the raw activations they are supposed to improve on.

**How sure, and why.** VERIFIED, HIGH. The recompute reproduces F-100's published +0.120 / +0.103 /
+0.075 exactly when given the kd bank, which validates the method before it is used to correct.

**What is still stale (CONFLICT / DOC-ONLY).** The ledger is append-only, so F-110 carries the
correction while F-100's own heading still reads "+0.120 / +0.103 / +0.075"
(`kd/docs/FINDINGS.md:4601`) — that is the ledger's convention and F-110 names what it supersedes.
Two other places carry a **third**, uncorrected figure and name nothing:
- `kd/docs/FINDINGS.md:4336` and `:4717` — F-96's "+0.120 / +0.103 / **+0.079**" (the L40-only 70B rung);
- the memory note `no-margin-from-one-layer.md:22` — the same "+0.079".

A new session reading memory gets +0.079; a reader of the ledger's F-100 gets +0.075; only F-110
has +0.054.

**Why it matters.** This is draft contribution 4. The draft does not yet quote any of these numbers
(`audit/phase0/draft_text.txt` has no "0.075", "0.120" or "0.054"), so the correction can still be
made before it is written in.

**Evidence.** `audit/fix/RF16_NOTE.md`; `audit/fix/rf16_recompute_mean.txt`;
`audit/phase1/bank_fingerprint_scale70b*.txt`; `kd/docs/FINDINGS.md:4601,4614,4661,5263-5283`.

---

## S3 · The two versions of the paper16 persona vectors — checked, and CLEAN

**What was tested.** Two versions of the paper's 26-persona vector bank exist: an original built in
16-bit arithmetic and an fp32 re-extraction made after a bug was found (RF-12/RF-15). Some findings
read one, some the other. Does it matter?

**What happened.** The four axes have old-vs-new cosine **1.0000 at every one of the 80 layers**,
with a maximum relative length difference of 0.5–0.9%; the PCA components differ by at most 0.009.
The only real hazard is that the sign of a principal component is pinned differently at a few
layers. Those layers are **L23** (deception) and **L65** (unified / all)
(`audit/phase1/axes_old_vs_fp32.txt:104,107,113`).

**Neither L23 nor L65 is a stored layer in any reported grid.** The crossgen grids store layers
0, 4, 8, 12, 16, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76 — 23 and 65 are
absent (`audit/phase2/layer_sensitivity.csv`). And no claim in `audit/claims.csv` mentions layer 23
or layer 65 (regex scan over claim text, stated numbers, evidence keys and notes: 0 hits).

**How sure, and why.** VERIFIED CLEAN, HIGH. The hazard is real but cannot reach any reported
number, by two independent routes (the layer is not stored; no claim names it).

**Why it matters.** This is the cheap good news: a version conflict flagged HIGH in Phase 1 costs
nothing. It also means the "which bank version" column in the provenance tables does not need to be
audited claim by claim for the paper16 family.

**Evidence.** `audit/phase1/axes_old_vs_fp32.txt`; `audit/phase2/layer_sensitivity.csv`;
`audit/claims.csv`.

---

## S4 · F-58's inputs are gone, and its error is still live one entry away

**What was tested.** F-58 reported the first three-rung scale comparison. Its 3B and 8B input
matrices were overwritten 78 minutes later by a second run over a *different* five datasets, under
the same filenames.

**What happened.** CONFIRMED NOT-RECOMPUTABLE: no copy in git on either branch, none on Drive. The
ledger's 3B/8B figures (0.708 / 0.743, gains +0.089 / +0.086) match no surviving file; the files
give 0.7619 / 0.7980 and gains +0.033 / +0.013. F-58's 70B row verifies exactly. The entry has been
revised to F-63, whose cells reproduce to four decimal places.

**The live residue.** F-58's dataset-identity error — calling its five "the paper's Table 1 …
HarmPressureChoice" — was corrected by F-60 **only against F-58**. The same wrong sentence is still
in **F-54** (`audit/claims.csv`, batch P4-scale). And the project now uses two different "fives":
the scale comparison's five are *instructed-deception, mask, convincing-game, ai_liar, roleplaying*
(`audit/scripts/fix_rf16_run.ps1:19`; `audit/fix/rf16_recompute_mean.txt` line 1), while the draft's
five are "the five Liars' Bench datasets that Llama-3.3-70B-Instruct generated without fine-tuning"
(`audit/phase0/draft_text.txt:293`), which is a different set. **Never write "the paper's five"
without listing them.**

**How sure, and why.** VERIFIED, HIGH for the file facts; the two-fives collision is VERIFIED by
reading both definitions.

**Evidence.** `audit/PROGRESS.md` (P4-scale section); `audit/claims.csv` F-58, F-54;
`scale/scripts/run_scale_coords.ps1:62-70,155-164`.

---

## S5 · Why RF-16 happened, why it can happen again, and a check that it has not

**What was tested.** RF-16 was possible because a result matrix records only the *basename* of the
bank it used. I checked whether that is a one-off or a structural defect, and whether any other
store is silently mislabelled.

**What happened — the defect is structural and still live (VERIFIED).**
- `src/assistant_axis_ndif/27_crossgen.py:612` writes `"bank": Path(args.bank).name` — the folder is
  discarded. `bank_name` beside it is free text passed on the command line.
- Three separate chain scripts each emit a file called `axes_vectors_s3.pt` into their own folder:
  `kd/scripts/run_kd_chain.ps1:95-101`, `src/assistant_axis_ndif/run_paper16_chain.ps1:109-119`,
  `src/assistant_axis_ndif/run_paper16_shared_chain.ps1:92-100`. The collision is produced by the
  pipeline, not by an accident of naming.
- `audit/phase2/provenance.csv` flags **six** stores whose bank basename exists in more than one
  folder. One (scale_70b) is RF-16.

**The check — the other five are CLEAN.** A bank determines a store's persona-feature numbers
exactly, so two stores built on the same activations with the same bank must agree to the last
digit. I compared every persona-feature cell of the ambiguous stores against the known paper16
stores, and every store against every other store
(`p3c_bank_fingerprint.py`, `p3c_store_pairs.py`, over `audit/phase2/layer_sensitivity.csv`):

| ambiguous store | cells vs kd11 (paper16) | identical | verdict |
|---|---|---|---|
| `kd11/banks/kd` | 1,008 | 0 | its own bank ✓ |
| `kd11/banks/cap_s23` | — (disjoint features) | — | distinct from cap_ctrl and cap_all ✓ |
| `kd11/banks/cap_ctrl` | — | — | distinct ✓ (also md5-separated in Phase 2) |
| `banks/paper16_shared` | 1,008 | 0 | its own bank ✓ |
| `banks/own100` | 1,008 | 0 | its own bank ✓ |

The pairwise sweep found only three pairs that are numerically identical, and all three are already
known and benign: `kd11/passA ≡ passA` and `kd11/passC ≡ passC` (the "kd11" folder name is
misleading — those grids ran on 10 datasets; DU-03), and `kd11/banks/collab ≡ kd11/banks/kd_s23`
(see W8).

**How sure, and why.** VERIFIED, HIGH for the defect (read in the source); VERIFIED, MEDIUM-HIGH for
the clean result — the fingerprint proves two stores are *not* the same bank, which is what was
asked; it cannot prove a store is the bank its label names, only that it is not one of the others
present.

**Why it matters.** RF-16 cost the project a wrong number in a headline table and took a
cross-store fingerprint to find. The one-line fix (`str(Path(args.bank).resolve())` instead of
`.name`) would make it impossible. Until then every new bank run is one basename collision away
from the same error.

**Evidence.** `src/assistant_axis_ndif/27_crossgen.py:612`; `audit/phase2/provenance.csv`;
`audit/scripts/phase3/p3c_bank_fingerprint.py`, `p3c_store_pairs.py`.

---

## S6 · Did the regenerated baselines drift under the bank stores? No.

**What was tested.** The cap and kd bank stores are scored against baselines copied from the `kd11`
store (`--baseline_root kd11`). The `kd11` store was rewritten in the 09-22 wave. If its baselines
changed, every bank comparison built on them would be comparing against a moved target.

**What happened.** All **1,512 of 1,512** baseline cells (`raw`, `datapca`, `datapca@2`, `random`,
`random@2`, `length`) in each of `kd11/banks/cap_all`, `cap_ctrl` and `cap_s23` are **byte-identical**
to `kd11`'s, max |difference| = 0.000000.

**How sure, and why.** VERIFIED, HIGH — an exact-equality check over every matched cell.

**Why it matters.** It removes a whole category of doubt from the cap results (F-65/F-66/F-67), the
project's only place where a persona axis beats what it is matched against.

**Evidence.** `audit/scripts/phase3/p3c_bank_fingerprint.py` (baseline section).

---

## S7 · The own100 bank was never rebuilt — so the bank comparison is not like-for-like

**What was tested.** The paper16 vectors were re-extracted in fp32 after a 16-bit bug. The same
re-extraction was launched for the 100-persona "own100" bank and **died after 1 of 100 vectors**
(`audit/experiments.csv` row `O05-own100-fp32`, status `partial`: "1 of 100 vectors on disk;
`rb_own100.log` last line is an NDIF progress line, no 'Done.'").

**What happened.** own100 therefore still rests on the buggy 16-bit-reduction vectors, while
paper16 rests on the fixed fp32 ones. Every paper16-vs-own100 comparison — F-90's persona arm, and
the memory note `persona-set-decides-detection` ("paper16 beats own100 on held-out AUROC") —
compares a repaired bank against an unrepaired one.

**How big is the effect?** **NOT MEASURABLE** without re-running the extraction, which needs NDIF
and is out of scope. The one bound available is indirect and weak: for *paper16*, old-vs-fp32 cost
nothing (S3). Whether the same holds for own100's 100 roles is unknown — the bug was in a masked
reduction, and the two banks differ in role count, rollout count and question set.

**How sure, and why.** VERIFIED for the file facts (the partial run, the vectors_dir each bank
records). INFERRED for the consequence, and explicitly NOT MEASURED for the size.

**Why it matters.** "Which persona set you choose decides whether detection works" is one of the
project's more interesting findings. It currently has a confound that nobody has priced.

**Evidence.** `audit/experiments.csv` rows `O05-own100-fp32`, `O35-hf-label-banks`;
`audit/phase1/anomalies.md` CV-11, OR-11; `audit/claims.csv` F-90.

---

# HALF 2 — WEAK SPOTS

## W1 · The binding constraint is the dataset count, and here is what it costs — **the biggest item in this report**

**What was tested.** Almost every claim in this project is tested by resampling the *datasets*
(there are 11) rather than the rows (there are thousands). F-04 noticed that this makes almost
nothing significant. I measured the ceiling across the whole crossgen corpus instead of one file.

**What happened.** Across **65,388** dataset-level tests in the ten cached crossgen
`significance.json` files:

- the median 95% interval is **±0.125 AUROC**; the 90th percentile is ±0.186;
- only **13.9%** of tests have an effect larger than their own interval half-width;
- **166 (0.25%) clear BH q < 0.05.**

And of those 166, almost none is the claim the project wants to make:

| what clears | count |
|---|---|
| a persona feature beats **`random`** (a noise floor of the same dimension) | 122 |
| a persona feature beats **`datapca`** (the matched low-dimensional control) | 6 |
| a persona feature beats **`raw` activations** | **4** |
| a *control* losing to raw (`random − raw`, `datapca@2 − raw`) | 33 |
| a persona feature *losing* | 1 |

There are **16,548** persona-vs-raw tests in that corpus. **17.6% of them even have a positive
sign**, and **4 clear correction** — all four are `pc:kd_assertion`, the owner's own
knowledge-denial axis, three of them under the `lr_reg10` probe rather than the paper's.

**Restricted to the operating points the project actually reports** (layers 20/40/44, `paper` probe,
mean pooling): 276 persona-minus-raw tests, 95 positive (34.4%), **0 clear BH**, 16 have an
uncorrected interval excluding zero. Median effect **0.044**; median interval half-width **0.133** —
the noise is three times the signal.

**Version robustness.** The same census on the git copies (10,152 tests, the smaller pre-wave
families) gives 28 clearing, 1 persona-beats-raw. The picture does not depend on which version you
read.

**Which headline claims are unresolvable because of this.** Every claim of the form "persona
coordinates transfer better than raw activations" — the replication's central proposition, draft
contributions 1 and 4, F-17, F-29, F-31, F-61, F-96, F-100. They are *consistent with* a real
effect; the design cannot separate a +0.04 effect from zero. The claims that do survive are the ones
whose effects are big enough to clear ±0.13: pooled-training-beats-single-source (+0.178, F-79/F-80,
where the pairing within target makes the interval unusually tight at ±0.035), and the cap axis on
its own column (+0.112 [+0.050, +0.185], F-66).

**How sure, and why.** VERIFIED, HIGH for the arithmetic — it is a census, not a sample, over files
opened during this audit. MEDIUM on scope: the cached corpus is the **crossgen** significance files
only. The cap, scale and premise significance files live on the lab PC and are not in
`audit/cache/`, so the cap arm (F-66/F-67) and the premise arm (F-80) are **not** in the 65,388 and
are counted separately above from their claim rows.

**Why it matters.** The honest framing for the paper is not "persona coordinates beat raw
activations" but "persona coordinates are not distinguishable from raw activations at 11 datasets,
and the one place they clearly win is a bank we wrote ourselves for a failure we had already seen".
The fix is the only one that attacks the constraint: **more datasets** (the 11 → 14 grid already
recorded as an open decision), not more rows, more seeds or more layers.

**Evidence.** `audit/scripts/phase3/p3c_ceiling.py`, `p3c_who_clears.py`;
`audit/cache/labpc/run-crossgen/**/significance.json`; `audit/claims.csv` F-04;
`audit/LIMITATIONS_AND_FUTURE_WORK.md` §1.

---

## W2 · Layer 40 flatters the persona features specifically — worth about 0.05

**What was tested.** The memory note `no-margin-from-one-layer` records four headline margins that
shrank or reversed when swept across depth. I tested whether the reported layers are favourable *in
general* or favourable *for persona features in particular* — the second is the one that indicates
selection rather than a good operating point.

**What happened.** For every (store, dataset set, feature) combination with at least 7 stored
layers, I ranked the reported layer's `feature − raw` gap among all that feature's layers
(1.0 = worst layer, 0.0 = best), at seed 42, mean pooling, paper probe:

| feature family | layer | cells | mean percentile | median | in the top third |
|---|---|---|---|---|---|
| **persona** (`axes`, `pc:*`) | **40** | 96 | **0.347** | **0.286** | **62%** |
| control (`random`, `datapca`, `length`) | 40 | 90 | 0.584 | 0.476 | 22% |
| persona | 44 | 106 | 0.442 | 0.381 | 25% |
| control | 44 | 100 | 0.731 | 0.794 | 1% |
| persona | 20 | 116 | 0.521 | 0.571 | 35% |
| control | 20 | 110 | 0.587 | 0.714 | 18% |

Layer 40 lands in the top third of its own distribution for **62% of persona readings and 22% of
control readings** — so it is not simply a layer where everything works; it is a layer where the
persona features in particular look good.

**What it costs in a concrete case.** On the 11-dataset `kd11` store, `axes − raw` reads **−0.0215
at L40 — rank 2 of 21** — against a layer-mean of **−0.0733**. Quoting L40 rather than the layer
average is worth **0.052** of margin, in the favourable direction, on the very quantity the
replication is about. `pc:deception` at L40 is rank 1 of 21. On the paper's five datasets the
picture moves to L20: `pc:deception` reads +0.0609 at L20 — **rank 1 of 21**, against a layer-mean
of −0.0450.

**Which claims this threatens.** The draft's contribution 4 is explicitly three layers: "evaluated
on five Liars' Bench datasets and RepE at **layers 20, 40 and 44**"
(`audit/phase0/draft_text.txt:97`), and the draft's own "every layer we tested" means those three
(claim D-027). Ledger entries F-65, F-91, F-96 and F-100 all
already carry a layer caveat from Phase 2 — F-65's own sweep found the cap margin peaks at **L32
(+0.1387)**, not at the reported L40 (+0.112). What is new here is the *comparative* measurement: the
selection is specific to persona features, so it cannot be explained as "layer 40 is where this
model is most linearly readable".

**How sure, and why.** VERIFIED for the arithmetic — 78,516 rows of
`audit/phase2/layer_sensitivity.csv`, all layers, one script. MEDIUM on interpretation: the cells are
heavily correlated (the same 21 layers reused across stores and probes), so I quote no p-value, and
the persona and control families differ in dimensionality (1–4 vs 2), which could contribute. This
is one of my two nominations for lead QA.

**What would fix it.** Report the layer-mean and the range beside any single-layer number, and say
where the quoted layer ranks — exactly what `docs/LAYER_PREREGISTRATION.md` already exists to
enforce. The data to do it is on disk and the computation is CPU-only.

**Evidence.** `audit/scripts/phase3/p3c_layer_rank.py`, `p3c_layer_seed_summary.py`;
`audit/phase2/layer_sensitivity.csv`; memory note `no-margin-from-one-layer`.

---

## W3 · Every headline number rests on one random split — and the split matters more than most margins

**What was tested.** How much of the project is single-seed, and how much does the seed move the
answer?

**What happened — three facts, all VERIFIED from `audit/phase2/layer_sensitivity.csv`:**

1. **The probe every headline uses is effectively single-seed.** Of 18,532 rows with `probe = paper`,
   **18,460 are seed 42**. The only 72 exceptions are `kd11/banks/cap_s23` at seeds 1, 7 and 13 —
   F-87's cap seed check. Every other paper-probe number in the project comes from one split.
2. **16 of the 22 result stores have exactly one seed** — including `kd11` (the 11-dataset store
   behind F-03/F-04/F-13/F-17/F-29/F-31/F-61/F-74), every kd and cap bank except cap_s23, the
   collaborator-bank store, `passB_fine`, `passP`, and **all four scale rungs** (3B, 8B, 70B,
   70B-repaired). The six multi-seed stores carry seeds only for the `mms` and `lr_reg10` probes.
3. **Where seeds exist, they move the answer by as much as the margins being claimed.** Over the
   1,712 multi-seed cells, the seed-to-seed **range** of `feature − raw` averages **0.033** for
   `axes` (p90 0.084, max **0.168**) and 0.021 for `pc:deception`. On the root store's five-dataset
   set — the scale comparison's quantity — the per-layer seed range averages **0.059** (mms) to
   **0.070** (lr_reg10), with individual layers reaching **0.16**.

For comparison, the margins these numbers carry: the 70B rung's `best − raw` is +0.008 as published
and −0.013 repaired; F-96's 3B→8B step is 0.017; the median persona-minus-raw effect at the
operating points is 0.044. **All of these are inside the measured seed range.**

**One thing I checked and it came out clean.** I suspected seed 42 had been picked because it looked
good. It has not: across all 1,712 multi-seed cells, seed 42's mean rank is **3.66 of 6** (chance
3.5) and it is the best seed **8.1%** of the time (chance 16.7%) — if anything slightly *unlucky*.
Structurally the seed-42 runs are ordinary (same `n_pairs = 90`, comparable `diag_mean`). So the
problem is **variance, not selection**. (In the `<root>` store alone seed 42 does rank better than
chance — 2.80 of 6, best 21% of the time, over 336 cells — but that is one store and the cells are
correlated; I do not treat it as an effect.)

**Which claims this threatens.** Any margin below ~0.05 quoted from one seed: F-63's "+0.020 /
+0.013, small and flat" for 3B/8B, F-96's 3B→8B step, F-100/F-110's 70B-vs-raw reading, and the bulk
of the persona-vs-raw readings at the operating points. It does **not** threaten the pooled-training
result (F-83 reports 5 seeds at L40, +0.173 to +0.198) or the cap result (F-87, 4 seeds, spread
0.0088 — an order of magnitude below F-66's interval).

**How sure, and why.** VERIFIED, HIGH. Counts and ranges are a census over the full table; the
seed-42 rank test is a direct rank computation with its own chance baseline stated.

**What would fix it.** The seeds are a re-run of an existing CPU-only script over activations
already on disk. Running the `paper` probe at seeds 0–4 on `kd11` and on the four scale rungs would
turn every single-seed margin into a margin with a spread, at no model cost.

**Evidence.** `audit/scripts/phase3/p3c_layer_seed_summary.py`, `p3c_seed42_rank.py`;
`audit/phase2/layer_sensitivity.csv`; `audit/claims.csv` F-83, F-87.

---

## W4 · The HP-KR / kr-matched leak: measured, bounded, and narrower than it reads

**What was tested.** `harm-pressure-kr-matched` is a filtered *view* of
`harm-pressure-knowledge-report`: all 258 of its rows are HP-KR rows. So in the leave-one-dataset-out
design, two of the eleven cells train on their own test data. F-108 established this. The question
here is which *reported* numbers it moves, and by how much.

**What happened — the leak's reach, measured from the cached premise files.**

The headline is "pooled training on ten datasets beats single-source transfer by **+0.178**"
(F-79/F-80; +0.1782 at layer 40, mean pooling, paper probe, seed 42). Two ways to price the leak:

| treatment | L20 | L32 | L40 |
|---|---|---|---|
| as reported (11 targets, LODO) | +0.1705 | +0.1312 | **+0.1782** |
| drop the two contaminated targets (9 clean targets) | +0.1664 | +0.1173 | **+0.1653** |
| **leave-one-CLUSTER-out** (F-108's fix: remove everything sharing the target's provenance) | +0.1289 | +0.1309 | **+0.1254** |

So the leak is worth **−0.013** at the reported layer if you simply drop the contaminated cells, and
**−0.056** under the stricter cluster holdout — which also removes three legitimate same-provenance
datasets, so −0.056 is an upper bound, not the leak alone. Swept over all 21 stored layers, dropping
the two cells changes the gain by between −0.018 and +0.035, and the change is *positive* (the leak
was suppressing the gain) at layers 4–28.

The two leaked cells are visible individually at L40: HP-KR gains **+0.2789** from pooling and
kr-matched **+0.1934**, against a median across the eleven of +0.1803 — high, but not the largest
(insider-trading +0.2485, convincing-game +0.2344 are clean).

**What the leak does NOT touch — and this is the important part.** The six-dataset comparison the
owner uses to answer her collaborator's abstract (F-85, F-109 — "the abstract's lead shrinks to
+0.023") runs on *instructed-deception, harm-pressure-choice, harm-pressure-knowledge-report,
convincing-game, insider-trading, ai_liar*
(`audit/cache/labpc/run-premise/six/premise_L40_mean_s42.json`, key `datasets`).
**`harm-pressure-kr-matched` is not among them**, so no cell in that comparison is contaminated.
VERIFIED.

**The overstatement is still in the code.** The claim row and the script docstring
(`src/assistant_axis_ndif/30g_clustered_holdout.py:12-14`) both say "every one of its 258 rows is in
the training pool". The pool holds only HP-KR's **80% train split**, so the measured contamination is
**42 of kr-matched's 52 test rows (80.8%)** — severe, but not total. Anyone re-reading that docstring
will over-state the leak; anyone reading the numbers will price it correctly.

**How sure, and why.** VERIFIED, HIGH. The three-way table is recomputed here from the cached lab
JSONs, and it agrees with the clustered store's own `summary.clean` to within 0.004 (0.1611 vs my
0.1653 at L40 — the two runs draw their 80/20 splits independently, which is itself a datapoint for
W3).

**Why it matters.** The conclusion (pooled training beats single-source transfer) survives every
treatment with a comfortable margin. The leak is a *presentation* problem, not a result-killing one:
the paper should quote the cluster-holdout number, **+0.125**, as the headline and the +0.178 as the
optimistic bound, rather than the other way round.

**What would fix it.** Nothing to re-run — F-108 already computed the fix. Quote it.

**Evidence.** `audit/scripts/phase3/p3c_f108_reach.py`;
`audit/cache/labpc/run-premise/premise_L*_mean_s42.json`;
`audit/cache/labpc/run-premise/clustered/clustered_L*_mean_s42.json`;
`audit/cache/labpc/run-premise/six/premise_L40_mean_s42.json`;
`audit/claims.csv` F-108, O-003; `src/assistant_axis_ndif/30g_clustered_holdout.py:12-14`.

---

## W5 · Live code defects — one known, one narrower than feared, two new

**DF-48 is real, and it is isolated (VERIFIED).** `scale/scripts/run_scale_chain.ps1:197-203` loops
over both vector variants (`vectors_all`, `vectors_s23`) but writes each iteration to
`"$out\axes_$tag.pt"` — the loop variable `$vt` never reaches the output path, and PowerShell's
case-insensitive variables make `$tag` the rung tag. Both banks are built; the second silently
overwrites the first. The surviving bank is the judge-filtered `vectors_s23` one, which is the
mechanical cause of the three-rung comparison's filter asymmetry (DF-47).

I checked whether the same typo exists in the sibling chains, and it does **not**:
`kd/scripts/run_kd_chain.ps1:95-101`, `src/assistant_axis_ndif/run_paper16_chain.ps1:109-119`,
`run_paper16_shared_chain.ps1:92-100` and `cap/scripts/run_cap_chain.ps1:111-125` all use the loop
variable in the output path correctly. **DF-48 is a one-off typo in the scale chain, not a pattern.**
Its cost is one unused bank per rung (`run-paper16/axes_vectors_all.pt` exists and has never been put
through crossgen — the paper-faithful unfiltered arm, paid for and never used).

**F-22's bug class — three sites still live, two of them never named (VERIFIED, NEW).** F-22 records
that a truncated exception handler, `str(e)[:200]`, hid the cause of a whole failed 8B run, and that
the fix reached some siblings but not others. Grepping every site at HEAD:

| file:line | context | state |
|---|---|---|
| `1_generate_ndif.py:145` | session handler | **fixed** (logs the exception's last line) |
| `2_activations_ndif.py:318` | session handler | **fixed** |
| `1_generate_ndif.py:186` | `generate_batch` retry | **still `str(e)[:200]`** — 40 lines below its own fix (known, in F-22's claim row) |
| `3_judge_ndif.py:168` | `judge_session` retry | **still `str(e)[:200]`** — **not named in F-22 or anywhere in the ledger** |
| `3_judge_ndif.py:188` | `judge_batch` retry | **still `str(e)[:200]`** — **not named anywhere** |

The two new ones are in the **judge**, the script that scores every rollout and decides which
rollouts feed each persona vector. A silent judge failure drops rollouts from a persona's vector
without a recoverable reason in the log — the same failure mode that cost the 3B extraction 22 roles
(F-33) and the 8B run entirely (F-22).

**How sure, and why.** VERIFIED, HIGH — read directly in the source at HEAD (`d280d23`), with the
fixed and unfixed forms quoted side by side.

**Why it matters.** F-22's own lesson is "grep the siblings". The grep was never completed. This is
a five-minute fix with no effect on any existing number, and it removes a known cause of silent data
loss from the two scripts that generate and score everything.

**A third, smaller one (VERIFIED).** `scale/scripts/run_scale_coords.ps1:155-164` carries a
"loud warning" comment saying `$DS` still holds the pre-F-63 dataset list. It no longer does — line
69 was aligned to the analysis set on 09-22, so the drift guard at line 163 can never fire and the
comment now describes a state that does not exist. Harmless, but it is a stale warning that reads as
a live one.

**Evidence.** `scale/scripts/run_scale_chain.ps1:186-203`; `src/assistant_axis_ndif/1_generate_ndif.py:140-152,180-190`;
`3_judge_ndif.py:160-195`; `2_activations_ndif.py:310-325`; `scale/scripts/run_scale_coords.ps1:60-80,150-170`;
`audit/DOC_FIXES.md` DF-47, DF-48; `audit/claims.csv` F-22.

---

## W6 · Crashed, partial and unknown runs — accounted for

`audit/experiments.csv` marks 114 runs: 106 complete, 1 superseded, 2 partial, 1 crashed, 4 unknown.
All eight non-complete rows checked:

| run | status | does it threaten a reported claim? |
|---|---|---|
| `O05-own100-fp32` | partial (1 of 100 vectors) | **yes — see S7.** The only one of the eight that does. |
| `O40-crossgen-retrain-attempt` | crashed (`PYTHON_EXIT -1`, killed during mean layer 12 of 21; cause NOT-FOUND) | no — no F-number; superseded by the completed grids. The only doc that mentions it is the memory note `crossgen-80-20-retrain`. |
| `O56-kd-passAC` | partial (C-root step exit 1) | no — superseded in practice by `O58-passAC11` after the HP-KR top-up; the failure itself is documented as F-25 (HOLDS), and the recovered file is byte-identical to the 09-18 original. |
| `O03-paper16-orig` | superseded | no — see S3; the replacement is numerically equivalent. |
| `AN5-inline-no-script` | unknown | **no, but only because Phase 2 rebuilt it.** This row carries ten findings — F-96, F-98–F-103, F-104, F-107, F-108 — that were computed inline in a chat session with no script and no output file saved. Every one now has a Phase-2 claim row with a verdict (all HOLDS or PARTLY HOLDS). Had the audit not rebuilt them, ten HIGH-importance numbers would have had no recomputable source. |
| `O86-acts-workspace` | unknown (env built, never run) | no |
| `O87-layer22-acts-cache` | unknown (97 cached tensors, no log, no results; provenance NOT-FOUND) | no |
| `C20-inlp-ghost` | unknown (named in a commit message only) | no |

**Why it matters.** The interesting one is `AN5`: 42 of 109 ledger entries write no output file at
all (`script_writes_file = n`, GH-05), and the ten in AN5 are the most recent and among the most
important. The practice of computing a headline in a chat session and pasting the number into the
ledger is the single largest reproducibility risk in the project, and it is not fixed by anything in
the audit — Phase 2 re-derived the numbers, but the *next* one will have the same problem.

**Evidence.** `audit/experiments.csv`; `audit/phase1/anomalies.md` GH-05; `audit/claims.csv`.

---

## W7 · Class imbalance in harm-pressure-choice — an unreported arm

`harm-pressure-choice` reaches the 3B/8B coordinate stores at **134 deceptive / 666 honest (5:1)**,
because the sampler took **all 134 deceptive rows that exist in the source** — it is a round-robin
over label buckets that ran out of one bucket, not a proportional draw (F-56, PARTLY HOLDS; the
ledger's stated mechanism is the reverse of the true one). The dataset is **no longer in the scale
analysis set**: the comparison uses *mask* in that slot (`audit/scripts/fix_rf16_run.ps1:19`). So the
imbalance sits in an arm that nothing reports — worth one line, not an entry. **VERIFIED.**

## W8 · The collaborator-bank trained arm duplicates the kd_s23 arm exactly

`kd11/banks/collab` (the collaborator's extended 12v13 contrast) and `kd11/banks/kd_s23` (the
owner's) produce **identical numbers in all 504 matched cells** — 21 layers × 2 poolings × 3 probes
× 2 features × 2 dataset sets, max |difference| **0.000000** (`p3c_store_pairs.py`). This is the
tautology the memory note `pc-features-use-basis-not-contrast` warns about: the trained `axes` and
`pc:deception` features are built from the PCA basis over the persona set, and the two banks share
that set, so the contrast definition cannot move them.

**No reported claim is affected**, because F-98/F-99/F-103 read the **zero-shot** rows of those
matrices (`kd/docs/FINDINGS.md:4525,4585,4857` all say "zero-shot rows"), which *do* depend on the
contrast. The trained arm is an orphan. **Flag it so nobody quotes it later as evidence that the
extended contrast helps — it cannot be.** VERIFIED.

## W9 · Missing baselines, listed

Three baselines that exist as data and were never run, each cheap and each answering a live
objection:
- **the paper's own k.** The project's `k = 5` (`27_crossgen.py:882`) is a value the paper never
  uses, and the result is k-dependent (+0.138 at k≈2 vs +0.004 at k=5 on the same comparison,
  F-39/F-113). A k=1 re-extraction from rollouts on disk would settle "is this the paper's method".
- **the unfiltered persona bank.** `run-paper16/axes_vectors_all.pt` was built for exactly this and
  has never been through crossgen (the DF-48 consequence). It would remove the judge-filter
  asymmetry from the three-rung comparison instead of caveating it.
- **the multi-seed `paper` probe** (W3) and **the layer sweep beside every quoted layer** (W2).

All four are CPU-only, over activations already on disk. **DOC-ONLY** for the first two (taken from
`audit/LIMITATIONS_AND_FUTURE_WORK.md` §6, which the audit established); **VERIFIED** for the last
two, which this workstream measured.

---

## What I checked and found clean (worth saying)

- The paper16 old-vs-fp32 vector conflict **cannot reach any reported number** (S3).
- All 20 findings citing a regenerated artifact **have been re-checked**, and no effect size,
  interval or p-value moved — only the multiple-testing correction (S1).
- The five remaining stores with an ambiguous bank basename are **not** repeats of RF-16 (S5).
- The baseline columns under the cap bank stores are **byte-identical** to the store they were
  scored against (S6).
- **Seed 42 was not cherry-picked** — it ranks 3.66 of 6, slightly worse than chance (W3). I
  formed the opposite hypothesis from six hand-read rows and it did not survive the census.
- The **F-108 leak does not touch** the six-dataset comparison against the collaborator's abstract
  (W4).
- The DF-48 typo is **not** repeated in the four sibling chain scripts (W5).

---

## Nominated for the lead's independent QA (my two least certain findings)

1. **W2, the layer-selection measurement.** The arithmetic is a census and is not in doubt, but the
   *interpretation* — that L40's top-third rate (62% persona vs 22% control) shows selection — rests
   on treating the control features as a fair comparison group. They differ from the persona features
   in dimensionality (2 vs 1–4) and in how they are constructed, and the cells are heavily correlated
   across stores and probes, so I quote no significance. An alternative explanation I cannot exclude:
   layer 40 is where the persona *subspace* is genuinely best defined, in which case quoting it is a
   defensible operating point rather than a selection effect. Re-run:
   `audit/scripts/phase3/p3c_layer_rank.py`, `p3c_layer_seed_summary.py`.

2. **W1, the project-wide significance census.** The counts (65,388 tests, 166 clearing, 4
   persona-beats-raw) are exact for the files read, but the corpus is the **ten cached crossgen
   `significance.json` files only**. The cap, scale and premise significance files are lab-only and
   not in `audit/cache/`, so the arm containing the project's strongest positive result (F-66) is
   outside the census and is quoted separately from its claim row. If the lead can read the lab
   copies, the census should be re-run over all of them before the number "4 of 16,548" is used in
   any write-up. Re-run: `audit/scripts/phase3/p3c_ceiling.py`, `p3c_who_clears.py`.
