"""Write audit/claims/O-0**.md for batch P12-briefing, from the batch CSV plus the pasted
script outputs.  Each note carries the claim, what was run, the verdict with its reason, the
ledger row and the relevant slice of the actual script output."""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CSV = os.path.join(ROOT, "phase2", "claims_batch_P12-briefing.csv")
OUT = os.path.join(ROOT, "claims")
OUTDIR = os.path.join(ROOT, "phase2", "out", "P12")
rows = {r["id"]: r for r in csv.DictReader(open(CSV, encoding="utf-8"))}


def slice_of(fname, start, end=None, maxlines=200):
    """Pull a labelled section out of a saved script output."""
    try:
        lines = open(os.path.join(OUTDIR, fname), encoding="utf-8").read().splitlines()
    except FileNotFoundError:
        return f"(missing {fname})"
    i = next((k for k, l in enumerate(lines) if start in l), None)
    if i is None:
        return f"(marker {start!r} not found in {fname})"
    j = len(lines)
    if end:
        j = next((k for k in range(i + 1, len(lines)) if end in lines[k]), len(lines))
    return "\n".join(lines[i:min(j, i + maxlines)])


BODY = {}

BODY["O-001"] = ("""# O-001 — "The paper's sycophancy axis scores 0.374 on our capitulation dataset"

**Batch:** P12-briefing · **Checked:** 2026-09-23 · **HIGH importance**
**Briefing line:** `docs/COLLABORATOR_BRIEFING.md:74-79`

## The claim, restated
The paper16 **sycophancy** persona axis — the one with the cleanest geometry — reads **0.374** on
`sycophancy_mt` (the capitulation dataset), which is below chance and worse than a **random
direction (0.541)**; and the same axis family reads **0.300** on `harm-pressure-knowledge-report`.

## What I did
`P12_local.py` recomputes every figure from the per-cell AUROC matrix
`run-crossgen/kd11/matrices/mean_L40_s42.json` (git working-tree copy) and `P12_lab.py` /
`P12_lab4.py` repeat the reads on the lab original (md5 `d7616aa50814267e38bd38931e25a6a0`,
2026-09-19 20:12) and print **every** zero-shot direction stored for the paper16 bank in the
neighbouring `passA11` / `passC11` stores.

## Verdict — PARTLY HOLDS
All three numbers exist and reproduce exactly. The sentence that carries them does not.

| briefing | measured | what it actually is |
|---|---|---|
| sycophancy axis 0.374 on sycophancy_mt | **0.3740** | the zero-shot **PC1 of the DECEPTION axis** |
| (the real sycophancy axis) | **0.3061** contrast / **0.2996** PC1 | paper16 sycophancy axis, same dataset, L40 |
| worse than a random direction 0.541 | **0.5413** | the **trained** `random` feature, sycophancy_mt as target |
| same family 0.300 on HP-KR | **0.3004** | the **trained** `axes` stack (k=5) with HP-KR as source |
| (the real sycophancy axis on HP-KR) | **0.1625** / **0.2768** | contrast / PC1 |

**Why 0.374 is the deception axis.** `src/assistant_axis_ndif/27_crossgen.py:411` picks the
bank's primary behaviour contrast for the bare `contrast` / `PC1` rows, and hardcodes
`"deception"` when present: `_cands = [n for n in ("deception",) if n in bank["axes"] and n in
bank["pca"]]`. The paper16 bank (`run-paper16/axes_vectors_s3_fp32.pt`) has
`axes = ['assistant','deception','sycophancy','unified']`, so `_zs_axis = 'deception'`. The
per-axis rows `contrast:sycophancy` / `PC1:sycophancy` were only added on 2026-09-21, after this
file was written, which is why it stores just two directions.

**The error does not damage the argument** — the sycophancy axis is *worse* (≈0.30 against 0.374)
— but the number attached to the named axis is wrong, and a reviewer who re-runs it will get 0.30.

**Second issue: three protocols in one sentence.** 0.374 is zero-shot, 0.541 is a trained probe on
a random direction, 0.300 is a trained probe on the persona-axis stack. No zero-shot random
direction exists in these stores, so the "worse than random" comparison cannot be made
like-for-like from this file. It *can* be made from the trained rows, and it survives there:
`axes` 0.3551 vs `random` 0.5413 as target, 0.3787 vs 0.6327 as source.

## Level 1 — the briefing vs the ledger
Matches. F-65 (`kd/docs/FINDINGS.md:2097-2099`) makes the identical attribution ("The paper16
**sycophancy** axis … reads **0.374** zero-shot"), and F-13 (`:220,223`) supplies 0.300 and
0.541. So the briefing restates the ledger faithfully; the ledger is what is wrong.
""", [("P12_local.txt", "O-001  -- Claim 1's three numbers", None),
      ("P12_lab4.txt", "passA11 (paper16 bank", None)])

BODY["O-002"] = ("""# O-002 — "filtered vs unfiltered judge: differ by 0.0004"

**Batch:** P12-briefing · **Checked:** 2026-09-23 · **MEDIUM importance**
**Briefing line:** `docs/COLLABORATOR_BRIEFING.md:104-110` (the Claim 2 table)

## The claim, restated
For the capitulation bank, the judge-filtered (`cap_s23`, score-3 rollouts only) and unfiltered
(`cap_all`) banks give results that **differ by 0.0004**, so the grader filter does not drive the
result. The same table also asserts **+0.112, 95% CI [+0.050, +0.185]** over raw activations and
**positive at 21 of 21 layers**.

## What I did
`P12_lab.py` reads all 42 cap matrices (21 layers × 2 banks) on the lab PC, rebuilds the
`sycophancy_mt` column as the 10 per-source transfer AUROCs, and runs its **own** paired
dataset-level bootstrap (`numpy.random.default_rng(42)`, B = 10,000, resampling the 10 sources)
rather than importing `cap/scripts/cap_layer_profile.py`.

## Verdict — HOLDS
Every number in the table reproduces.

| | `cap_s23` | `cap_all` | \\|diff\\| |
|---|---:|---:|---:|
| `pc:capitulation@2` at L40 | 0.7136 | 0.7132 | **0.0004** |
| margin − `datapca@2` | +0.1542 | +0.1538 | **0.0004** |
| margin − `raw` | +0.1123 | +0.1118 | **0.0004** |
| 95% CI on the margin over raw | [+0.0502, +0.1847] | [+0.0489, +0.1861] | — |
| layers positive vs raw / vs `datapca@2` | 21/21 · 21/21 | 21/21 · 21/21 | — |

`raw`, `datapca@2`, `random@2` and `length` are *identical to 4 dp* between the banks, as they
must be (the filter changes only the persona bank).

## Worth carrying
The briefing's "+0.112, 95% CI [+0.050, +0.185]" is exactly my independent bootstrap's
**+0.1123, [+0.0502, +0.1847]**. The argmax band is [L28, L60] with L40 landing in only 3.9%
(s23) / 3.1% (all) of resamples, so L40 should be quoted as a pre-specified layer, not as the best.

## The one caveat the briefing does not carry
Batch P3's check of F-65 measured `pc:cap_only@2` — the feature the "the basis needs the
position-holding personas" argument rests on — as **0.5935 (s23) vs 0.6077 (all), a difference of
0.0142**, not 0.0004. "The judge filter does not drive it" is established for the headline
feature; it is not true of every feature in the bank.
""", [("P12_lab.txt", "O-002  cap bank", "O-003  leave-one-CLUSTER-out")])

BODY["O-003"] = ("""# O-003 — "under leave-one-CLUSTER-out the gain is +0.129 / +0.131 / +0.125"

**Batch:** P12-briefing · **Checked:** 2026-09-23 · **HIGH importance**
**Briefing line:** `docs/COLLABORATOR_BRIEFING.md:135-143` (the 2026-09-22 correction to Claim 3)

## The claim, restated
With every dataset sharing the target's provenance removed from the training pool
(leave-one-cluster-out), pooled training still beats single-source transfer by **+0.129 (L20),
+0.131 (L32), +0.125 (L40)** over all eleven targets, and **+0.157 / +0.117 / +0.150** over the
nine uncontaminated ones, **positive on 8–11 of 11 at every layer**.

## What I did
`P12_lab2.py` reads the three `run-premise/clustered/clustered_L*_mean_s42.json` artifacts on the
lab PC, prints their stored `summary` blocks verbatim, and then **recomputes** each gain and each
positive count from the 11 per-target cells (`pooled_loco − single_offdiag`) rather than trusting
the summary.

## Verdict — HOLDS, with one count phrase that needs fixing
All six gains are exact:

| layer | gain LOCO, all 11 | gain LOCO, clean 9 | positive (all 11) | positive (clean 9) |
|---|---:|---:|---:|---:|
| 20 | **+0.1289** | **+0.1571** | 10/11 | 9/9 |
| 32 | **+0.1309** | **+0.1168** | 11/11 | 9/9 |
| 40 | **+0.1254** | **+0.1504** | 10/11 | 8/9 |

The clusters are as F-108 describes: `lb_harm` {HP-C, HP-KR, HP-KR-matched}, `lb_scenario`
{instructed-deception, convincing-game, insider-trading}, `apollo` {ai_liar, roleplaying,
sandbagging_v2}, `mask`, `ours` {sycophancy_mt}; `leaked` = {HP-KR, HP-KR-matched}. The leak
signature reproduces: HP-KR at L40 falls **0.9674 (LODO) → 0.6492 (LOCO)**, and at L20
**0.7090 → 0.4311**. The counter-intuitive "clean beats all-11" ordering reproduces at L20 and L40.

**"positive on 8–11 of 11 at every layer" mixes two denominators.** Over all eleven targets the
counts are 10, 11, 10 of 11; over the nine clean targets they are 9, 9, 8 of **9**. Nothing is
ever 8 of 11. F-108's own table states both correctly — the loss happens in the briefing's
one-line compression.

## Version
Lab only (`run-premise` is on no branch; RF-10). The three files are dated **2026-09-22
16:58:55 / 16:59:17 / 16:59:39** — *after* the 15:40–16:26 boot-trigger wave (RF-13), so they are
the F-108 run itself, not a regenerated older result. No BH q-value is quoted.
""", [("P12_lab2.txt", "L20   C:/Users/user4/expLB/run-premise/clustered",
       "the briefing's OTHER p = 0.38")])

BODY["O-004"] = ("""# O-004 — "margin at the pre-specified layer, dataset bootstrap: CI [−0.069, +0.189] — does not clear"

**Batch:** P12-briefing · **Checked:** 2026-09-23 · **HIGH importance**
**Briefing line:** `docs/COLLABORATOR_BRIEFING.md:171`

## The claim, restated
The knowledge-report matched-slice persona margin at the pre-specified layer (L40) has a 95%
dataset-level bootstrap interval of **[−0.069, +0.189]** and therefore does not clear.

## Status in this audit — already settled twice, not re-derived a third time
By the lead's instruction this row was checked at Level 1 and against the two existing
independent re-derivations rather than re-run:

| source | margin at L40 | 95% CI | P(≤0) |
|---|---:|---|---:|
| F-76 (`kd/docs/FINDINGS.md:2846`) | +0.0655 | [−0.0693, +0.1887] | 0.161 |
| batch P7 (`audit/scripts/claims/P7_lab.py f7576`) | +0.0655 | [−0.0693, +0.1887] | 0.1611 |
| lead QA, independent loader (`audit/scripts/qa_p7.py`) | +0.0656 | [−0.0738, +0.1904] | 0.1646 |

All three intervals contain zero, under two different implementations of the same design
(B = 10,000, seed 42, resampling the ten transfer datasets, `datapca` as the matched control),
over the same 160 lab-original `passA11` matrices (2026-09-21 12:20–13:28, untouched by the 09-22
wave).

## Verdict — HOLDS
The briefing rounds F-76's interval to [−0.069, +0.189] and draws the correct conclusion. The
adjacent row of the same table, *"95% interval [L5, L79] — 93% of the network"*, is also right:
F-76 measures 74 of 80 layers, i.e. 92.5%.

## Not owed
No further recomputation. The claim's weakness is the *point estimate*, not the interval, and the
briefing already says so.
""", [])

BODY["O-005"] = ("""# O-005 — "does the band survive label permutation? p = 0.38"

**Batch:** P12-briefing · **Checked:** 2026-09-23 · **MEDIUM importance**
**Briefing line:** `docs/COLLABORATOR_BRIEFING.md:173`

## The claim, restated
A cluster-based label-permutation test on the matched-slice persona band gives **p = 0.38**, i.e.
the band is not distinguishable from label noise.

## What I did
`P12_lab.py` reads `run-premise/cluster_perm_kd_matched.json` on the lab PC and prints it whole.

## Verdict — HOLDS
The artifact states, verbatim:

- `"p": 0.38`, `"B": 200`, 20 layers;
- `observed_max_mass = 0.9155195683453238`, `observed_clusters = [0.91552, 0.05881]`;
- `null_max_mass`: 200 draws, min 0.0000, max **3.9097** — so the observed cluster mass sits well
  inside its own null.

This is **Level 1 only for the p**: re-running 200 permutations with probe refits is a re-run, not
a recomputation, and the brief forbids new runs. The stored null and the observed statistic are
internally consistent with p = 0.38.

## Version warning
The file's mtime is **2026-09-22 16:07:02**, which falls inside the RF-13 boot-trigger re-run
window (15:40–16:26), while F-84 is dated 2026-09-21. The artifact read here is therefore a
**regenerated copy**. Its p equals F-84's stated 0.38 and the run is seeded, so no discrepancy is
visible — but there is no pre-wave copy to compare against.

## Two things found on the way (about the briefing's *other* p = 0.38)
`docs/COLLABORATOR_BRIEFING.md:146` also says *"Pooled training with ~6,700 rows and pooled
training held to a single-source budget of ~666 rows are statistically indistinguishable
(p = 0.38)."* That is a **different test** — the coincidence is genuine, not a copy error — and it
has two problems of its own:

1. **The p is one-sided.** `src/assistant_axis_ndif/30b_premise_boot.py:62` computes
   `p = float((bs <= 0).mean())`. My own bootstrap of `pooled_lodo − pooled_lodo_match` over the
   11 targets gives **+0.0045, 95% CI [−0.0235, +0.0331], two-sided p 0.744** (one-sided 0.372).
   Same conclusion; but for a "no difference" claim the interval, not a one-sided tail, is the
   statistic that supports it.
2. **"~666 rows" is not the realised pool.** From `run-premise/premise_L40_mean_s42.json`, the
   row-budget-matched LODO pool holds **412–500 rows (mean 449)** per target; 666 is
   `n_train.single_mean`, the budget the control is *capped at*
   (`30_premise.py:167`: `per_o = budget // 10`, then `min(per_o, len(tr))`). The control is
   therefore **stricter** than advertised — good for the claim, wrong in the text. The "~6,700"
   figure on the other side is right: mean `n_lodo` = 6,669.
""", [("P12_lab.txt", "O-005  label permutation", "O-009 / O-011 / O-012"),
      ("P12_lab2.txt", "the briefing's OTHER p = 0.38", "mtimes"),
      ("P12_lab3.txt", "n_train block", None)])

BODY["O-006"] = ("""# O-006 — "0.904 / 0.916 is the extended axis: off_diagonal_mean 0.9041173922 / 0.9158352735"

**Batch:** P12-briefing · **Checked:** 2026-09-23 · **HIGH importance**
**Briefing line:** `docs/COLLABORATOR_BRIEFING.md:199-216`

## The claim, restated
The abstract's headline is the `ext_axis1` row of the collaborator's own
`results/persona_ood_ext_L20/summary.csv` and `.../L40/summary.csv` —
`off_diagonal_mean` = **0.9041173922** and **0.9158352735** — identical to the mean of
`auroc_ext_axis` in the matching `zero_shot.csv` **to 1e-16**; `ext_axis1` is the **12-vs-13**
contrast `DECEPTION + KNOWLEDGE_DENIAL` vs `HONESTY + KNOWLEDGE_CANDOUR`, whose ten extra roles
are the owner's knowledge-denial bank, matching `kd/personas/kd_role_groups.json` **name for
name**, over the base bank `paper16_L20_sharedquestions/paper16_L20.pt`.

## What I did
`P12_local.py` reads his CSVs and his source out of git with `git show` (**no checkout**), at both
the commit the ledger cites (`f386a72`) and the current branch head (`45b16d9`). Every mean is
recomputed from the per-dataset rows; nothing is read off a summary line. The Alpaca control row
(`XXXX-20__plain`, `n_deceptive = 0`, empty AUROCs) is dropped, leaving the 6 scored datasets.

## Verdict — HOLDS
**The digits.** `summary.csv` holds `0.9041173922099752` (L20) and `0.9158352734669751` (L40);
the briefing truncates both to 10 dp and loses nothing. Identical at `f386a72` and `45b16d9`.

**The identity.** mean(`auroc_ext_axis`) = 0.904117392209975 / 0.9158352734669751;
|difference| = **2.220e-16** (L20) and **exactly 0.000** (L40). The same identity holds for
`denial_axis1` (1.11e-16 / 0.0) but **not** for `axis1` at L20, where it is **1.398e-01** —
because one training set disagrees on the sign. So "the trained off-diagonal equals the zero-shot
mean" is a property of sign-consistent 1-d features, not a general one.

**The arms.** `deception_detection/persona_pcs.py` builds
`axes["extended"] = mean_of(DECEPTION_PERSONAS + KNOWLEDGE_DENIAL_PERSONAS) −
mean_of(HONESTY_PERSONAS + KNOWLEDGE_CANDOUR_PERSONAS)` = 8 + 4 = **12** vs 7 + 6 = **13**,
exactly the block the briefing quotes.

**The roles.** His `KNOWLEDGE_DENIAL_PERSONAS` (4) and `KNOWLEDGE_CANDOUR_PERSONAS` (6) are
**set-identical** to the owner's `knowledge_denial` (4) + `sincere_ignorance` (3) +
`plain_assertion` (3); symmetric difference empty.

**The base bank.** `deception_detection/persona_pcs.py` **line 31** is literally
`BANK_PATH = REPO_ROOT / "paper16_L20_sharedquestions" / "paper16_L20.pt"` — the citation is exact.

## Three wording fixes for the sentence around the digits
1. The briefing cites `scripts/persona_ood.py::extended_arms()`. The file is at
   **`deception_detection/scripts/persona_ood.py`**, and `extended_arms()` *re-fits* the geometry —
   the axis itself is defined in `deception_detection/persona_pcs.py` (`axes["extended"]`).
2. *"they match `kd_role_groups.json` name for name"* is true of the **names** but not of the
   **grouping**: his `KNOWLEDGE_CANDOUR_PERSONAS` **merges** the owner's two distinct honest
   groups — `sincere_ignorance` and `plain_assertion` — into one six-role arm.
3. *"identical … to 1e-16"* is exact at L40 (0.0) and **2.2e-16** at L20, one ulp above the stated
   bound. "to within 1e-15" would be safe.
""", [("P12_local.txt", "O-006 / O-007  -- the collaborator's own files", "O-001  -- Claim 1")])

BODY["O-007"] = ("""# O-007 — "margin, paper's contrast vs baseline: +0.143 (L20) / +0.100 (L40)"

**Batch:** P12-briefing · **Checked:** 2026-09-23 · **HIGH importance**
**Briefing line:** `docs/COLLABORATOR_BRIEFING.md:220-233`

## The claim, restated
On the collaborator's six datasets, zero-shot, the paper-faithful **8-vs-7** contrast beats his
**trained single-source baseline** by **+0.143** at layer 20 and **+0.100** at layer 40; the
extended contrast beats it by +0.256 / +0.184, so the knowledge-denial personas are worth
+0.113 / +0.084 = 44% / 46% of the headline margin. Dropping `repe_honesty`, the paper's contrast
leads by +0.142 / +0.092 and the kd personas are worth +0.136 / +0.101.

## What I did
`P12_local.py`, reading his committed CSVs out of git at `f386a72` and at head `45b16d9`. The
baseline is recomputed **independently** as the off-diagonal mean of `matrix_baseline.csv`
(n = 30 over six datasets, n = 20 over five) and cross-checked against `summary.csv`'s own
`baseline` row; the zero-shot arms are recomputed from the per-dataset rows.

## Verdict — HOLDS

| | L20 | L40 |
|---|---:|---:|
| trained single-source baseline | 0.6478 | 0.7320 |
| `axis1` (8v7) zero-shot | 0.7909 | 0.8319 |
| `ext_axis1` (12v13) zero-shot | 0.9041 | 0.9158 |
| **margin, paper's contrast** | **+0.1431** | **+0.0999** |
| margin, extended contrast | +0.2563 | +0.1838 |
| kd personas worth | +0.1132 | +0.0839 |
| share of the headline margin | 44.2% | 45.7% |

The independently recomputed `matrix_baseline.csv` off-diagonal is **0.6478 / 0.7320**, equal to
`summary.csv`'s `baseline` row, so the margins do not depend on which of his files is trusted.

## One thing the briefing should say
The drop-`repe_honesty` line (+0.142 / +0.092, kd worth +0.136 / +0.101) is computed against a
baseline **recomputed over the five remaining datasets** (0.6085 / 0.7064), not against the
six-dataset baseline printed one row above. Against the six-dataset baseline the same margins
would be +0.1026 / +0.0662. The briefing's reading is the right one; naming the baseline would
stop a reviewer arriving at the other pair.
""", [("P12_local.txt", "O-006 / O-007  -- the collaborator's own files", "O-001  -- Claim 1")])

BODY["O-009"] = ("""# O-009 — "Both of your arms lose to a pooled probe — by 0.064 at L20 and 0.055 at L40"

**Batch:** P12-briefing · **Checked:** 2026-09-23 · **HIGH importance**
**Briefing line:** `docs/COLLABORATOR_BRIEFING.md:243-252`

## The claim, restated
With the collaborator's two zero-shot arms rebuilt from the owner's persona vectors and run on the
owner's eleven datasets, **both lose to a pooled leave-one-dataset-out probe** — by **0.064** at
layer 20 and **0.055** at layer 40.

## What I did
`P12_lab.py` recomputes each zero-shot figure as the mean over the 11 per-dataset entries of
`run-crossgen/kd11/banks/collab/matrices/mean_L{20,40}_s42.json`, the raw baseline as the mean of
the 110 off-diagonal cells of the matching paper16 matrix, and reads the pooled figure from
`run-premise/premise_L{20,40}_mean_s42.json` key `summary.pooled_lodo`.

## Verdict — HOLDS

| | L20 | L40 |
|---|---:|---:|
| raw single-source trained baseline | 0.6289 | 0.6623 |
| `axis1` (8v7) | 0.6572 | 0.7325 |
| `ext_axis1` (12v13) | 0.7386 | 0.7996 |
| **pooled LODO** | **0.8027** | **0.8545** |
| pooled − `ext_axis1` | **+0.0641** | **+0.0548** |
| pooled − `axis1` | +0.1455 | +0.1219 |

## Robustness the briefing does not claim
Pooled LODO beats `ext_axis1` at **all 21 stored layers** under mean pooling, by +0.0396 (L0) to
+0.1202 (L60). The gap is *smallest* at L22 (+0.0482) and L40 (+0.0548), so quoting L20 and L40 is
conservative rather than cherry-picked.

## Wording
0.064 and 0.055 are the gaps for the **extended** arm alone. The paper's 8v7 arm loses by 0.146
(L20) and 0.122 (L40). *"Both of your arms lose … by 0.064 at L20 and 0.055 at L40"* reads as if
the pair covers both arms; it is the smaller of the two gaps in each case. The statement is true;
the numbers belong to one arm.
""", [("P12_lab.txt", "O-009 exactly as the briefing states it", "FILES READ")])

BODY["O-011"] = ("""# O-011 — "the kd bank is worth +0.055 mean-pooled and +0.059 last-token, positive at 21/21 and 20/21"

**Batch:** P12-briefing · **Checked:** 2026-09-23 · **HIGH importance**
**Briefing line:** `docs/COLLABORATOR_BRIEFING.md:263-270`

## The claim, restated
`ext_axis1 − axis1` (zero-shot, the owner's eleven datasets) is worth **+0.055 mean-pooled** and
**+0.059 last-token**, positive at **21/21** and **20/21** layers respectively, the single
exception being **L0**, the embedding layer.

## What I did
`P12_lab.py` sweeps **all 42** `run-crossgen/kd11/banks/collab/matrices/{mean,last}_L*_s42.json`
on the lab PC, recomputing each zero-shot mean from the 11 per-dataset entries and each raw
baseline from the 110 off-diagonal cells of the matching paper16 matrix under the *same* pooling.

## Verdict — HOLDS

| pooling | mean `ext − axis1` | layers positive | range |
|---|---:|---:|---|
| mean | **+0.0551** | **21 / 21** | [+0.0019 (L0), +0.0814 (L20)] |
| last | **+0.0590** | **20 / 21** | [−0.0327 (**L0**), +0.0848 (L36)] |

The only non-positive layer under either pooling is **L0**, exactly as the briefing says.

## Level 1 — and a gap in the ledger
The mean-pooled half matches F-99 exactly (+0.0551, 21 of 21, range [+0.0019, +0.0814]).
**The last-token half has no ledger entry**: `audit/claims/F-99.md` records that "Last pooling was
not swept; the entry is explicitly a mean-pooling claim." So this briefing line is currently the
only place the 20/21 last-token result is written down, and this note is its first check.

## Worth noting
The peak moves with pooling — L20 under mean, L36 under last — but the *size* of the effect is
stable (+0.0551 vs +0.0590). That is the point the briefing makes, and it survives.
""", [("P12_lab.txt", "---- pooling = mean ----", "O-009 exactly as the briefing")])

BODY["O-012"] = ("""# O-012 — "the share reads 2%–114% across depth, median 63% mean-pooled and 50% last-token"

**Batch:** P12-briefing · **Checked:** 2026-09-23 · **MEDIUM-HIGH importance**
**Briefing line:** `docs/COLLABORATOR_BRIEFING.md:272-278`

## The claim, restated
The *share* of the margin over raw attributable to the knowledge-denial bank —
`(ext_axis1 − axis1) / (ext_axis1 − raw)` — is unstable: **2%–114%** across depth, median
**63%** mean-pooled and **50%** last-token, while the underlying difference barely moves
(+0.055 vs +0.059). Hence: quote the raw difference, and give a share only with layer and pooling
named.

## What I did
The same sweep as O-011 (`P12_lab.py`), computing the share at each of the 21 stored layers from
three independently recomputed quantities under each pooling.

## Verdict — HOLDS

| pooling | median share | mean share | range |
|---|---:|---:|---|
| mean | **62.9%** | 63.8% | [2.2% (L0), 113.9% (L68)] |
| last | **50.0%** | 48.0% | [−155.9% (L0), +144.9% (L8)] |

63% and 50% are exact. Above 100% under mean pooling is real: `axis1` falls **below** raw at
L60 (0.6379 vs 0.6386), L64 (0.6337 vs 0.6370) and L68 (0.6260 vs 0.6339), so the kd bank supplies
more than the whole margin there — which is what O-013 says one line later.

## Level 1
The mean-pooled figures match F-99 exactly (median 62.9%, mean 63.8%, range [2.2%, 113.9%]).
The 50% last-token median has **no ledger entry** — F-99 is a mean-pooling claim — so it is checked
here for the first time.

## One scoping point
The range "2%–114%" is the **mean-pooled** range. The sentence places it before "mean-pooled", so a
reader can take it as covering both poolings; under last-token the range is **−156% to +145%**.
Naming the pooling before the range, or giving both, would make the sentence airtight — and the
wider last-token range strengthens the very point the sentence is making.
""", [("P12_lab.txt", "---- pooling = mean ----", "O-009 exactly as the briefing")])

for cid, (head, outs) in BODY.items():
    row = rows[cid]
    parts = [head.rstrip(), "", "## The ledger row", ""]
    for k in ("source_ref", "evidence_file", "evidence_key", "level1_value", "level1_match",
              "level2_value", "level2_method", "version_used", "robustness", "verdict",
              "confidence", "importance", "script"):
        parts.append(f"- **{k}**: {row[k]}")
    if outs:
        parts += ["", "## Script output (verbatim)", ""]
        for fname, start, end in outs:
            parts += [f"### `{fname}` — from `{start}`", "", "```",
                      slice_of(fname, start, end), "```", ""]
    open(os.path.join(OUT, f"{cid}.md"), "w", encoding="utf-8").write("\n".join(parts) + "\n")
    print("wrote", os.path.join(OUT, f"{cid}.md"))
