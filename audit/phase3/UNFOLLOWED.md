# Unfollowed results — strong, surprising, and negative

_Phase 3, workstream **P3-D**, 2026-09-23. Companion files:
`audit/scripts/phase3/p3d_*.py` (re-runnable) and `audit/phase3/out/p3d_*.txt` (their output)._

**What this file is.** Three things nobody wrote up: (1) experiments that finished and produced no
reported number, (2) numbers measured in a working session and never put in a document, (3) results
that failed — which in this project are the strongest things it can say.

**Tags.** **VERIFIED** = recomputed from a raw file during this audit, path given · **DOC-ONLY** =
only in prose · **CONFLICT** = two sources disagree, both shown · **NOT-FOUND** · **INFERRED** =
my reading, reason given.

**One definition, used throughout.** *Off-diagonal mean* = train a probe on one dataset, test it on
each of the others, average the AUROCs. It is this project's measure of **cross-dataset transfer**.
0.5 = chance. *Raw* = a probe on the full 8,192-dimensional activation. *datapca* = the matched
control: the same number of dimensions as the persona feature, but from a PCA of the evaluation
data instead of the persona bank. *Persona features* (`axes`, `pc:unified`, `pc:deception`, …) =
the method under test.

---

# Part 1 — Orphans: experiments that finished and were never reported

`audit/phase1/anomalies.md` §1 lists four complete crossgen grids with no reported result. All four
were answerable from `audit/phase2/layer_sensitivity.csv` without touching the lab PC; one extra
per-dataset read was done read-only on the lab PC (no writes).

## 1.1 **passP — the prompt-side control. This is the biggest thing in this file.** PUBLISHABLE

**What was tested.** The same probes, trained on activations over the **prompt only** — everything
before the model's first generated token — instead of over the model's answer. The code states the
pre-registered reading itself (`src/assistant_axis_ndif/8_coords_ndif.py:270-278`):

> "The prompt mode exists for the construct-validity control: if a probe trained on PROMPT-side
> activations transfers as well as one trained on response tokens, the matrix is largely detecting
> the instructed condition rather than anything the model wrote."

**What happened.** It does transfer as well — at the project's own reporting settings. **VERIFIED**
(`audit/phase3/out/p3d_orphans.txt` §A/A2; `p3d_passp_perdataset.txt`).

| setting (seed 42, 10 datasets, raw probe) | prompt-side | response-side | ratio above chance |
|---|---|---|---|
| L40 last, paper probe | **0.6475** | 0.6359 | **1.085** |
| L40 last, mms | 0.6537 | 0.6590 | 0.967 |
| L44 last, paper | 0.6219 | 0.6268 | 0.962 |
| L44 last, mms | 0.6382 | 0.6708 | 0.809 |
| L40 mean, paper | 0.5687 | 0.6652 | 0.416 |
| all 18 (layer × pooling × probe) cells | mean 0.5812 | mean 0.6469 | mean **0.562** |

In-distribution the gap is even smaller: prompt-side reaches **90–96%** of the response-side AUROC
at every layer and pooling (e.g. L40 last: 0.9281 vs 0.9671).

Per dataset at L40, last pooling, raw, paper probe (prompt-side row mean vs response-side row mean):

| dataset | prompt-side in-dist. | prompt-side transfer | response-side transfer |
|---|---|---|---|
| convincing-game | **1.0000** | 0.6578 | 0.6386 |
| sandbagging_v2 | **1.0000** | 0.6374 | 0.6434 |
| instructed-deception | 0.9715 | 0.6288 | 0.6676 |
| harm-pressure-choice | 0.9484 | 0.5869 | 0.5619 |
| roleplaying | 0.9484 | 0.5564 | 0.6690 |
| harm-pressure-knowledge-report | 0.9234 | **0.6832** | 0.6300 |
| **sycophancy_mt** | 0.8449 | **0.7579** | 0.5158 |
| mask | 0.7675 | 0.5412 | 0.6500 |

**What it means, stated carefully.** For **convincing-game** and **sandbagging_v2** the label is
perfectly readable from the prompt — those two columns of every transfer matrix measure the
experimental condition, not the model's behaviour. Across the whole matrix at **last** pooling and
the project's two reporting layers, a probe that never sees the model's answer recovers
**0.60–1.09 of the response-side signal above chance** (six cells: 1.085, 0.967, 0.708, 0.962,
0.809, 0.602; mean **0.86**) — and at L40 with the paper probe it is **higher**. The one dataset
where the answer genuinely matters is **MASK** (0.54 prompt vs 0.65 response, and the lowest
prompt-side in-distribution score, 0.77) — which is what MASK was designed for. And the most
surprising single cell: on **sycophancy_mt**, the dataset where every persona axis fails, the
prompt-side probe transfers at **0.758** against the response-side probe's **0.516**.

**How sure.** VERIFIED. Both stores are the same rows, the same split seed, the same probe code;
only the token mask differs. Caveat: "last" pooling on the prompt means the final prompt token,
which already carries the whole instruction — so this is a lower bound on how much the prompt says,
not a claim that no information is in the response. At **mean** pooling the picture is different
(prompt-side is 0.10 lower), so the conclusion is pooling-dependent and must be stated that way.

**Why nobody followed it up.** The thesis document announced it as running
(`docs/thesis/persona-coordinates-70b-results.md:1428`, §15.5, "prompt-side activations for the
construct-validity control of §11.7") and the write-up stops there. The grid finished 2026-09-19
to 09-22 (`audit/experiments.csv` O11, O46). No doc, no F-number, no memory note reports a number.

**Publishable / next step.** **Publishable, and it should lead the limitations section** — it is a
control the project designed, ran, and has the data for. Next step costs nothing new: the per-dataset
matrices are on the lab PC; producing the table above took one read-only script. What would
strengthen it: a per-dataset significance test, and re-stating the headline transfer numbers with a
"prompt-side floor" column beside them.

**Evidence.** `audit/scripts/phase3/p3d_orphans.py` §A; `p3d_passp_snip.py` (run read-only on the
lab PC, piped over ssh, nothing written); outputs `audit/phase3/out/p3d_orphans.txt`,
`p3d_passp_perdataset.txt`. Store: lab `run-crossgen/passP/matrices/{mean,last}_L{20,40,44}_s42.json`.

## 1.2 **passA — the 80-layer grid, 92,184 files. The persona method loses almost everywhere.** PUBLISHABLE

**What was tested.** The full crossgen grid on the 80-layer activation store (200 label-stratified
rows per dataset), root + 7 banks, 6 seeds. The largest result set in the project.

**A coverage trap I hit and you must not.** In passA, passC and the root store, **seeds 0–4 carry
only three features (`raw`, `axes`, `pc:deception`) and only two probes (`mms`, `lr_reg10`)**; seed
42 carries all eleven features and all three probes. Averaging "over six seeds" silently compares a
6-seed mean with a 1-seed value. Every comparison below is **seed 42 only**. **VERIFIED**
(`p3d_orphans.txt` §0).

**What happened** (seed 42, 10 datasets, 80 layers). How many layers each feature beats raw, and
beats its matched dataset-PCA control:

| feature | mean/paper | mean/mms | last/paper | last/mms | mean margin vs raw (mean pooling, paper) |
|---|---|---|---|---|---|
| `axes` | 3/80 · 3/80 | 2/80 · 2/80 | 3/80 · 10/80 | 5/80 · 12/80 | **−0.1004** |
| `pc:unified` | 3/80 · 3/80 | 2/80 · 1/80 | 4/80 · 9/80 | 2/80 · 5/80 | −0.0430 |
| `pc:unified@2` | 0/80 · 13/80 | 0/80 · 8/80 | 3/80 · 13/80 | 5/80 · 17/80 | −0.1035 |
| `pc:deception` | 1/80 · 1/80 | 0/80 · 0/80 | 3/80 · 11/80 | 6/80 · 11/80 | −0.0828 |
| `length` (1-d control) | 7/80 · 14/80 | 8/80 · 15/80 | 13/80 · 17/80 | 14/80 · 17/80 | −0.0747 |
| `random` (control) | 0/80 · 0/80 | 0/80 · 0/80 | 6/80 · 7/80 | 6/80 · 4/80 | −0.1074 |

Read the last two rows first. **A one-dimensional response-length feature beats raw at more layers
than any persona feature does** — in all six (pooling × probe) cells, including the two lr_reg10
cells not shown here (`p3d_orphans.txt` §B). And on this grid a **random** direction's own shortfall
against raw (−0.1074) is within 0.007 of the persona axes' shortfall (−0.1004).

**The sharpest version of that** (`p3d_unrecorded2.txt` §X1): persona features minus random
directions of the same dimensionality, averaged over 80 layers —

| dataset set | pooling | `axes` − `random` | `pc:unified@2` − `random@2` |
|---|---|---|---|
| **10 datasets** | mean | **+0.0070** (44/80 layers) | +0.0137 (43/80) |
| 10 datasets | last | +0.0560 (72/80) | +0.0491 (63/80) |
| **the paper's 5** | mean | **+0.1922** (77/80) | +0.2078 (77/80) |
| the paper's 5 | last | +0.2281 (75/80) | +0.2249 (77/80) |

**What it means.** On the paper's own five datasets the persona basis is a genuinely good
low-dimensional basis — far better than a random projection (+0.19 to +0.23) and better than dataset
PCA at 55–76 of 80 layers. Broaden to ten datasets with mean pooling and that advantage is **+0.007
and a coin flip across layers**. The method's value is not in the personas *per se*; it is in the
personas matching the datasets.

**How sure.** VERIFIED at seed 42 from the matrices. MEDIUM on generality: only seed 42 has the full
feature set, so the 80-layer win-counts are single-split. The three features that do have six seeds
show the split matters — the `axes − raw` margin **changes sign across seeds** at L44
(s1 = −0.0234, s2 = +0.0156) and the root store at L44 on the paper's five spans
s3 = −0.1029 to s2 = +0.0284, sd 0.052 (`p3d_orphans.txt` §D).

**Publishable / next step.** **Publishable**, and it is the strongest single-number version of the
project's central negative result. Next step: re-run passA at seeds 0–4 with the full feature list
(~5 CPU-hours on the lab PC, no NDIF, inputs already on disk) so the 80-layer win-counts get error
bars. Cheap and it removes the main caveat.

**Evidence.** `p3d_orphans.py` §B, §H; `p3d_unrecorded2.py` §X1. Store `run-crossgen/passA`
(`audit/experiments.csv` O44; `anomalies.md` OR-01).

## 1.3 **passA + passB_fine — L40 is not the peak, and the peak is outside the window that was scanned.** PUBLISHABLE (one section)

**What was tested.** passB_fine extracted nine extra layers (37–39, 41–43, 45–47) for the whole
corpus so that, in the thesis document's words, "the peak can be located to ±1 layer instead of ±4"
(`docs/thesis/persona-coordinates-70b-results.md:1428`).

**What happened.** Raw transfer over 36–48 is **monotonically decreasing**: L36 0.6878 → L40 0.6652
→ L47 0.6537 (mean pooling, paper probe). L40 ranks **5th of 13**. The argmax of the scanned window
is its left edge, and the 80-layer passA grid puts the true peak at **L31–L33** in 5 of 6
(pooling × probe) cells — 0.7072 at L33 against 0.6760 at L40 (+0.031), and 0.7170 at L31 against
0.6593 at L40 (+0.058, mms). The root store agrees: argmax L32 in 5 of 6 cells. **VERIFIED**
(`p3d_orphans.txt` §C, §E).

**Why it matters.** Session mining recovered the owner stating outright that **"L40 was picked
because it maximised the persona advantage"** (`audit/PROGRESS.md`, session-mining leads, DOC-ONLY).
The passA grid shows why that happens mechanically: the persona margin is less negative at L40
because **raw is in a trough there**, not because the axis is better. At raw's own peak layer the
persona margin is far worse — e.g. passA mean/mms `axes − raw` is **−0.0336 at L40** but **−0.1811
at L31** (`p3d_orphans2.txt` §2); the same pattern holds in all 18 store × pooling × probe cells.

**How sure.** VERIFIED for the profiles (seed 42). The layer-selection intent is DOC-ONLY.

**Publishable / next step.** One section, not a chapter: state the peak, state that L40 was
pre-registered, and report the margin at both L40 and the raw-optimal layer. No new compute.

## 1.4 **passC — a duplicate of passA, and therefore an unclaimed reproducibility check.** WORTH ONE LINE

**What was tested.** The crossgen grid on the per-token store's pooled vectors, 6 layers × 6 seeds,
6,936 files.

**What happened.** passC and passA evaluate the **same 200-row sample** and agree to within
**0.0022 at worst over 1,512 matching cells** (mean |diff| **0.00016**; 22 cells above 0.001; none
above 0.01). **VERIFIED** (`p3d_orphans2.txt` §1).

**What it means.** The grid adds no new evidence about the method. What it *is*, for free, is a
**re-extraction reproducibility check**: two separate NDIF forward passes over the same rows, months
of pipeline between them, give downstream AUROCs that agree to 1e-4. That belongs in the methods
section in one sentence; the project has no other end-to-end reproducibility number.

**Dead end / one line.** One line. The per-token store is still valuable (it feeds Q8, §2.2); the
**pooled crossgen grid over it** is redundant.

## 1.5 The two smaller orphans

- **`expLBactivationLAYER22`** (97 `.pt`, 361 MB, lab-only, 2026-09-04…09-12). Owner unknown,
  no producing script, no doc, **not backed up to Drive**. **NOT-FOUND** provenance
  (`audit/experiments.csv` O87; `anomalies.md` OR-08). **Dead end** for results; **live item for
  data safety** — it is the only copy.
- **Length-control JSONs** (`run-coords/{confounds_L44_last, p1_L44_last_lengthctl,
  q5_semantics_lengthctl}.json`, lab + Drive, not in git; `anomalies.md` OR-05). Their content is
  already covered: the audit re-measured the length-only off-diagonal as **0.5555 at every layer,
  pooling and probe** (11 datasets) and **0.5494** on 10, and corrected six documents accordingly
  (`audit/PROGRESS.md`, "the length claim — 6 edits"). The `length` column in
  `audit/phase2/layer_sensitivity.csv` reproduces 0.5494 at every root layer. **Closed.**

---

# Part 2 — Measured in a session, never written down

`audit/phase1/sessions/findings_unrecorded.csv` has 35 rows (S-01…S-35). Batch P13 checked ten
(S-01–S-04, S-09, S-11–S-14, S-29). **The other 25 are below.** Where a file exists I recomputed it.

## 2.1 The three that are worth reporting

### S-15 · On real sycophancy, every persona direction is inside the random-direction null, and a response-length-only scorer beats all of them. PUBLISHABLE

**What was tested.** Zero-shot AUROC (no probe — just project onto a direction) of 5 directions ×
2 banks × 2 poolings on 1,182 sycophancy_mt rows, each with two length controls (score within
length quintiles; length regressed out), plus the project's own random-direction null and a
length-only scorer.

**What happened.** **VERIFIED** from `run-coords/q10_sycophancy_axis.json` (in git):

- **2 of the 20 (bank × pooling × direction) series beat the random-direction null at their best
  layer**, and both are the paper16 **assistant** direction (0.7263 mean / 0.7046 last, against a
  best-of-layers p95 null of 0.6935 / 0.6833). Every sycophancy, deception, unified and syc_PC1
  direction, in both banks, is inside the null.
- A **length-only scorer reads 0.6640** on the same rows. **18 of the 20 series never reach it.**
- The two surviving assistant cells lose 0.05–0.07 under the length controls: L24 0.7263 → 0.6719
  (quintiles) / 0.6545 (regressed); L40 0.7102 → 0.6601 / 0.6453; L44 0.7152 → 0.6510 / 0.6469.
  After control they are **at or below** the length-only scorer.
- Across all 60 controlled cells, **28 move by more than 0.05** under a length control.

**Why it matters.** Memory note `q10-sycophancy-axis-inverted` records "only the Assistant Axis
detects it (0.71)". That is true and it is also the whole story: after the length control the one
detector that works is no better than counting characters. This is the **strongest available
statement of the register diagnosis**, and it is an argument *for* the capitulation bank (which is
trained, and clears its bootstrap — F-66) rather than against the project.

**How sure.** VERIFIED; single file, single row set (n = 1,182; 434 retracted / 748 maintained).
The null is the file's own 1,000-direction null.

**Next step.** None needed to report it. To strengthen: apply the same two length controls to the
cap bank's 0.745, which currently has no length control reported.

**Evidence.** `audit/scripts/phase3/p3d_unrecorded.py` §S-15 → `audit/phase3/out/p3d_unrecorded.txt`.

### S-06 · AUROC and TPR-at-1%-FPR rank the methods differently at every layer, and the zero-shot axis wins the one that matters for deployment. PUBLISHABLE

**What was tested.** Leave-one-dataset-out over 9 held-out datasets, 2,000 rows, 16 tokens/row,
Alpaca as the false-positive control, 6 layers, comparing pooled probes, token-level probes and a
**zero-shot persona axis** (no training at all).

**What happened.** **VERIFIED** from `run-coords/q8_trained_pertoken.json` (in git). At **every one
of six layers** the best method by AUROC is not the best method at 1% FPR:

| layer | best by AUROC | best at 1% FPR |
|---|---|---|
| 20 | pool_mean_mms 0.857 (TPR 0.498) | tok_mms/mean **0.506** (AUROC 0.712) |
| 22 | pool_mean_mms 0.854 (TPR 0.535) | pool_last_mms **0.552** |
| 28 | pool_mean_mms 0.876 (TPR 0.605) | **zeroshot_axis/max 0.693** (AUROC 0.792) |
| 40 | pool_mean_mms 0.883 (TPR 0.473) | **zeroshot_axis/max 0.622** (AUROC 0.814) |
| 44 | pool_mean_mms 0.878 (TPR 0.425) | **zeroshot_axis/mean 0.571** |
| 48 | pool_mean_mms 0.874 (TPR 0.403) | **zeroshot_axis/mean 0.551** |

At L28 the untrained axis catches **69%** of lies at a 1% false-alarm rate against the best trained
probe's 60%; at L40 it is 62% against 47%.

**Why it matters.** Every headline in this project is an AUROC. AUROC is the average over all
thresholds; a deployed detector runs at one, low-FPR threshold. The project measured the
deployment-relevant number and reported only the AUROC. Memory note `q8-per-token-beats-pooling`
records the AUROC half only. This also gives the persona method its **best honest claim**: it is
weak when trained and competitive when used untrained at a strict threshold.

**How sure.** VERIFIED for the committed full run. **CONFLICT with the session's own quote**: the
transcript's "zeroshot_axis top10pct AUROC 0.856 / TPR 0.717" is from the L44 **smoke** run; the
committed full run gives L44 zeroshot best TPR 0.494 (AUROC 0.815). Quote the committed file.

**Evidence.** `p3d_unrecorded2.py` §S-06; `run-coords/q8_trained_pertoken.json`
(`protocol: leave-one-dataset-out`, `control: alpaca`, `n_rows: 2000`).

### S-23 · The project's central negative finding was stated, with its table, on day one — and the table is in git. WORTH ONE LINE (high narrative value)

**What was tested.** The first five-dataset probe comparison, 2026-09-15.

**What happened.** The session wrote: *"the signal is positive but about half the size the paper
claims, and it isn't robust yet."* **VERIFIED**: the table reproduces **exactly** from
`run-coords/probes_L20.json` (L20, mean, k=5, paper16 s3 bank, 12 off-diagonal pairs) —
`pc:deception` 0.8146 / 0.7871 / 0.6809, `axes` 0.7811 / 0.7853 / 0.8044, `raw` 0.7678 / 0.7634 /
0.7363 for probes paper / lr_reg10 / mms. The transcript's 0.815 / 0.787 / 0.681 etc. are these
values rounded.

**Why it matters.** Two uses. (1) It **dates** the project's main conclusion to five days before
F-39 said the same thing with more data — useful for the thesis narrative. (2) On *this* dataset set
the persona features **win** (0.815 vs 0.768), which is the same composition effect as §1.2: four
datasets, close to the paper's, and the method works; ten datasets and it does not.

**Evidence.** `p3d_unrecorded2.py` §S-23; `run-coords/probes_L20.json`, `probes_5sets.json`.

## 2.2 Verified, and they change or qualify something already written

### S-35 · The `random` control column: a positive control the project passed and never quoted. VERIFIED

Recomputed from the scale matrices (`p3d_unrecorded.txt` §S-35). At every rung and every layer the
persona features beat a random direction by a wide margin — 3B +0.13…+0.24, 8B +0.09…+0.14, 70B
+0.19…+0.23, with `axes` beating random at **21/21 layers** on both 70B banks. **This is reassuring
and should be said**: the failures against raw in §1.2 are not a wiring bug.

**But note the scope, which the scale tables do not state:** every scale store has **5 datasets**,
not 10 (`audit/phase2/layer_sensitivity.csv`, `n_datasets`). On five datasets the margin over random
is +0.19; on ten it is +0.007 (§1.2). The `random` column is a clean control **for the five-dataset
grid only**.

### S-32 · The kd bank's near-miss is a persona-vs-random test, not a persona-vs-raw test. VERIFIED, and it sharpens F-31

Recomputed from the git copy of `run-crossgen/kd11/banks/kd/significance.json` (4,032 tests, F-31's
version) with `p3d_unrecorded.py` §S-32:

- The largest effect in the whole bank is `pc:unified@2 − random@2 = +0.2861`, 79/110 pairs won,
  dataset 95% CI [+0.147, +0.427], **q_BH 0.067** — the session's number, exact.
- **But the comparator is a random direction.** The eight largest effects are all
  `pc:unified@2 − random@2` or `pc:unified − random`.
- The six tests that clear q < 0.05: **four are `random − raw` (negative controls confirming that
  random is worse than raw)** and two are `pc:unified − random`.
- **Not one of the 4,032 tests in which a persona feature is compared with `raw` or with `datapca`
  clears the bootstrap.** (Individual cells have positive point estimates; none survives.)

That last sentence is the quotable form of F-31 and of `audit/LIMITATIONS_AND_FUTURE_WORK.md` §1,
and it is stronger than either currently states.

### S-24 / S-25 · The three cap "variants" collapse in two different ways — and it is already in the ledger. CORRECTION to the session table

**VERIFIED, with a correction to the transcript row.** From the matrices
(`p3d_unrecorded.txt` §S-24): in `kd11/banks/cap_all` and `cap_s23`, **`pc:capitulation` and
`pc:cap_vs_default` are byte-identical at 252/252 and 270/270 cells (max |diff| = 0.000000)**, while
`pc:cap_only` differs (max 0.155). From the code (`cap/scripts/cap_personas.py:238,244`),
**`cap_vs_default` and `cap_only` are the same contrast** (identical `pos`/`neg`), differing only in
the PCA basis. So: as *contrasts*, B ≡ C; as `pc:` *features*, A ≡ B. **No pair of the three
variants is independent in both views**, and "all three variants agree" carries no information.

**The session row's meta-claim ("only the A-vs-B cosine is in the ledger") is wrong.**
`kd/docs/FINDINGS.md:2144-2154` (F-65a) and `:3649` (F-89) already state the tautology, and
`cap/scripts/check_basis.py` is committed and says it in its docstring. **Worth one line**, as a
confirmation.

### S-01 / S-02 extended · The axis-norm problem holds at all 80 layers, and grows. VERIFIED

The task brief's version of this is confirmed from the bank itself
(`run-paper16/axes_vectors_s3.pt`, local): **‖sycophancy‖ / ‖deception‖ = 1.316…1.588, mean 1.510,
above 1 at 80 of 80 layers** (L20 1.529, L40 1.550, L44 1.508); ‖sycophancy‖ / ‖assistant‖ above 1
at 80/80. The absolute norms also grow ~2.4× from L20 to L44 (assistant 1.012 → 2.931), so **a
threshold or a sum fixed at one layer does not carry to another** either. AUROC is rank-based and
unaffected; any magnitude comparison, average, sum or shared threshold across axes or across layers
is not. **Worth one line in methods.**

### S-10 · The 2026-09-14 review's cosines do **not** describe the paper16 bank. CONFLICT resolved as provenance

The session recorded `cos(syc, asst) = 0.04–0.14` and `cos²(deception, assistant) = 0.44 at L40
(56% orthogonal)`. On `run-paper16/axes_vectors_s3.pt` I measure
**cos(assistant, sycophancy) = −0.2701 at L40** (range −0.2526…−0.5791 over 80 layers) and
**cos(deception, assistant) = −0.3705, cos² = 0.137 — 86% orthogonal, not 56%**. The review is dated
**09-14**, before the paper16 bank was built (09-15, `audit/experiments.csv` O03), so those numbers
describe the **own100** bank (**INFERRED** from the dates). My paper16 values agree with the memory
note `persona-axis-diagnostics` (−0.66 → −0.37). **Do not carry the 0.04–0.14 figure into any
paper16 statement.**

### S-08 · The NDIF attention-memory conflict resolves in the transcript's favour. VERIFIED (arithmetic)

6 × 64 heads × 1,869² × 4 bytes = **4.997 GiB**, matching NDIF's own "Tried to allocate 5.00 GiB".
The memory note `ndif-attention-memory-limit` records **5.36 GiB** for the same configuration, which
would need 68.65 heads. The 64-head figure is right; the memory note should be corrected (it feeds
every batch-size calculation). **Worth one line / a DOC_FIX.**

### S-05 · The per-token probe's numerical self-check survives in a committed file. VERIFIED

`run-coords/q8_trained_pertoken.json` carries `self_checks: {offset_worst_rel_dev: 0.1786,
offset_rows_bad: 0, offset_rows: 12000}`. The session's stricter smoke figures (max |w·x+b − sklearn|
= 3.66e-06; 0 of 3,072 row-layer means outside tolerance) are smoke-run only and are **not** in this
file. So the Q8 implementation *does* have committed validation evidence — one sentence in methods,
citing the file, not the transcript.

## 2.3 The rest — classified, not chased

| id | what | status | verdict |
|---|---|---|---|
| S-07 | Pass-B activation smoke: cos(mean,last) 0.388–0.603, no all-zero rows | DOC-ONLY (lab log) | one line in methods; it is the check that catches the wrong-last-token bug the `bf16 mask reductions` note warns about |
| S-16 | RF-13's cause: lab PC reboot 15:34:30 + boot triggers | VERIFIED in `audit/PROGRESS.md` RF-13 | closed |
| S-17 | "Next=none" does not mean a task cannot fire (~20 wrong status reports) | VERIFIED, RF-13 | closed |
| S-18 | 24 boot-triggered chains disabled 17:31, 73 min before the audit baseline | VERIFIED, PROGRESS | closed |
| S-19 | Drive sync dead 12:18–17:26 on 09-22 (expired OAuth), backlog cleared before the census | VERIFIED, PROGRESS | closed; it means RF-10's "not on Drive" folders are **not** explained by the outage |
| S-20 | A previous session wrote `rclone.conf` on the lab PC 78 min before the baseline | VERIFIED, PROGRESS | closed |
| S-21 | The F-99/F-100 sweeps ran with 0.2–0.6 GB free RAM against another account's jobs, resumed 5×| DOC-ONLY | **one line as a reproducibility caveat** — neither F-entry records it |
| S-22 | Stale 0-byte `.git/index.lock` blocked a commit | VERIFIED, PROGRESS | closed |
| S-26 | kd bank geometry: PC1 25.6% of role variance, top-5 69.2%, cos(PC1, assistant) min +0.539 / median +0.598 — **below** the Assistant-Axis paper's >0.60-all-layers and >0.71-mid-layer thresholds | DOC-ONLY (lab `logs/kd_axes.log`; bank is lab-only) | **worth reporting**: the draft's headline bank fails the geometric criterion its own reference sets. Cost to verify: one read-only lab script over `run-kd/axes_vectors_s3.pt`, minutes |
| S-27 | WMDP scorer smoke: 5/8 correct on kn00000–kn00007, margins 0.097…1.000 | DOC-ONLY | one line (acceptance evidence for F-08's 0.786) |
| S-28 | The accidental 70B download was **56.64 GB**, not F-24's 40.22 GB; retracted in-session | DOC-ONLY | DOC_FIX only |
| S-30 | F-86's +0.0284 margin sits inside its comparator's own 0.0824 subsampling spread; the script printed "INSIDE the subsampling spread" three times, the ledger softened it to ±0.02 sd | DOC-ONLY (`run-premise`, lab-only) | **high priority for the ledger** — it turns a small positive into a null. Cost: one read-only lab re-run of the premise script, ~minutes |
| S-31 | The 8B axes build printed a second geometry block for the s23 bank (cos(PC1,assistant) +0.739 at L16, PC1 40.4%); F-37 quotes only the first, and the s23 run overwrote the axes file | DOC-ONLY | one line; the print may be the only surviving record |
| S-33 | F-61's paper16-vs-kd comparison is not like-for-like (109 pairs / raw 0.6800 vs 110 pairs / raw 0.6623) | already handled — `audit/claims.csv` F-61 PARTLY HOLDS | closed |
| S-34 | `response_lengths.py` printed p50=p90=p99=max=0 for every rung before a wrong-field fix minutes before F-33's table; and the units are **characters**, which no doc states | DOC-ONLY | DOC_FIX; re-check F-33's length attribution |

---

# Part 3 — Negative results worth reporting in their own right

The owner's framing — the controls are the largest contribution *because* they kill this project's
own headlines — is supported. Ranked by what each one can carry.

### N1. The matched dataset-PCA control removes the persona advantage. PUBLISHABLE — the paper's spine

**Claim.** `pc:kd_assertion` beats raw at 18 of 21 layers, on a +0.10…+0.15 plateau from L32 up, and
the advantage is **not local** — dropping both harm-pressure-knowledge-report columns *raises* it to
+0.12…+0.19. Against the matched dataset-PCA control the margin is positive at 20/21 layers
(mean +0.0830) and **0 of 126 tests clear** the dataset bootstrap.
**[AUDITED, F-112 = `audit/PROGRESS.md` "F-31 AND F-20 CLOSED"; `audit/METHODS_INVENTORY.md` §3.5]**

**Why it is a result and not a caveat.** It is a *specific, falsifiable* statement about what a
low-dimensional persona basis buys: it beats an 8,192-d probe's *transfer*, consistently and at
depth, but it does not beat **a PCA of the same rank taken from the evaluation data itself**. That
distinguishes "persona directions are special" from "low-dimensional projections regularise
transfer", and this project is the only place the distinction has been measured at 70B.

**Not a contradiction with §1.2.** F-112 is the **kd** bank on **eleven** datasets including the
matched slice; §1.2 is the **paper16** bank on **ten** datasets without it. F-112 says the kd
feature beats raw but not the matched control; §1.2 says the paper16 features beat neither. Both
are seed-42 statements about different banks, and they agree on the part that matters — the matched
control is never cleared.

**What my own grids add.** Three independent confirmations, all new:
1. On **ten** datasets the persona axes barely beat a **random** projection (+0.007, 44/80 layers) —
   §1.2. On the paper's five they beat it by +0.19.
2. **Not one of 4,032 kd-bank significance tests** shows a persona feature beating raw or datapca —
   §2.2/S-32.
3. The advantage at the pre-registered L40 is largely **raw being in a trough** there: the margin is
   −0.034 at L40 and −0.181 at raw's own peak L31 — §1.3.

**Honest counterweight, which the write-up needs:** on the paper's five datasets `pc:unified@2` beats
its matched control at **55–76 of 80 layers** (`p3d_orphans2.txt` §4). The control kills the claim on
the broad set, not on the paper's own set. Say both.

### N2. The prompt-side control (§1.1). PUBLISHABLE — and it is not yet written anywhere

Ranked second only because it is newly surfaced by this audit and has had no review. It is the only
control in the project that questions **the corpus** rather than the method.

### N3. The sarcasm bank: two pre-registered gates, both failed, informatively. PUBLISHABLE

- **Gate 2.** Unmarked ironic and sycophantic responses are separable at **0.9213** by bag-of-words
  alone (95% CI [0.872, 0.967]; P(≤ the 0.65 pre-registered ceiling) = 0.0000) — so any direction
  "detecting sarcasm" may be reading vocabulary **[AUDITED, F-93 HOLDS, F-105]**.
- **Gate 3.** On natural sarcasm the deception axis is **inside a random-direction null** in all four
  (layer × pooling) cells (0.5239–0.5856) while the **sycophancy** axis fires at 0.6763–0.7097
  **[AUDITED, F-107 HOLDS]**.
- **v1 was confounded and review caught it**, for the same reason Goldowsky-Dill et al. concede for
  their own probe: the contrast varied both the sincerity *and* the quality of the plan.

**Why publishable.** Sarcasm is the sharpest available test of whether a "deception" axis reads
*intent* or *form*, because sarcasm is literally false with no intent to mislead. Two failed gates
with pre-registered thresholds, plus a specified-but-unbuilt v2, is a complete and honest story —
and F-113's amendment (report the conclusion, not the count: the batch measured 1 above / 3 below,
the lead's independent null 0 above / 4 below) is exactly the right way to write it.

**Do not overstate.** The register diagnosis is supported on two constructs (HP-KR, sycophancy) and
**untested on the third**. The bank remains unbuilt.

### N4. Unified PC1 is "which question set the persona answered". PUBLISHABLE as a methods lesson

**[AUDITED]** `audit/methods/EXTRACTION_VS_PAPER.md` AD-2 / E-5: unified PC1 (47% of variance at
L40) correlates **−0.945** with the question bank and only +0.230 with harmful-vs-harmless; PC2 is
the harmfulness axis (+0.816); same at L20 (−0.929) and L60 (−0.943). Corroborated by F-18
(`cos(PC1_unified, unified contrast)` = +0.389 on the default bank vs +0.896 on the repaired one,
and the repaired bank is higher at **80 of 80 layers**).

**Scope, which must be stated with it:** it damages the **PCA basis only**. Contrast directions barely
move (cross-bank cosine +0.984), so `contrast:unified` and all deception/sycophancy numbers are
intact. It contaminates `pc:unified`, `pc:unified@2` and the `PC1:unified` zero-shot scorer — and
**the default crossgen bank is still the confounded one** (`27_crossgen.py:856`).

**CONFLICT to fix before quoting:** the memory note `unified-pc1-is-the-question-family` says
**cos −0.98**; the audit's own measurement is **−0.945**. Use the measured value with its script.

**Why it is a contribution.** It is a reproducible failure mode of contrastive-persona PCA that the
replicated paper's own design avoids by using a balanced shared question set — so it is direct
evidence that that design choice is load-bearing, not cosmetic. Anyone reimplementing the method
needs this.

### N5. Dataset identity is 99.3% decodable; the deception label adds +0.032. PUBLISHABLE as the frame for every transfer number

**[AUDITED, claim O-082, verdict HOLDS]** `dataset_acc = 0.9934` against `label_acc = 0.6462`
(+0.032 over the majority baseline) on L40 mean-pooled evaluation activations. Two precision points
from the audit to carry with it: the realised run was **3,624 rows over 10 datasets** (400 is the
`--cap`, not the per-dataset count), and the lift is **log-only** (`17_validate.py:126-131` never
writes it).

**Why it matters.** It is the explanation for everything above. Almost all the variance in these
activations is *which corpus a row came from*; the deception label is a thin slice on top. A method
that improves cross-dataset transfer is therefore fighting a 99.3%-decodable nuisance variable with a
3.2% signal — which is why matched controls, random-direction nulls and the prompt-side control all
bite, and why composition (5 vs 10 datasets) decides every sign in this project.

### N6. The capitulation bank: the register diagnosis confirmed constructively, and its limits. PUBLISHABLE, already well covered

**[AUDITED]** Zero-shot the cap contrast reads **0.745** at L40 on sycophancy_mt against the paper16
register axis's PC1 **0.374** (F-65 PARTLY HOLDS; note the 0.374 is the **deception** axis's PC1 —
the paper16 **sycophancy** axis reads 0.313). Trained, `pc:capitulation@2` beats raw by **+0.112
[+0.050, +0.185]**, clearing a dataset-level bootstrap (F-66 HOLDS). **Its limits are equally
measured:** over all eleven datasets the margin over raw is **+0.098 with CI [−0.036, +0.208]**, i.e.
it does not clear (F-67); and the **format-matched control** shows swapping the six position-holding
personas for six topically unrelated instructed personas costs almost nothing (F-89 HOLDS), so much
of the gain is format, not construct.

**How to write it:** a construct-matched bank is a **targeted instrument, not a general detector**.
That is a finding about the method. Add S-15 (§2.1) as the reason it was needed.

### N7. Two smaller negatives worth a paragraph each

- **Growing the HP-KR matched slice failed its pre-registered gate.** Denial rate **0.250** (harmful
  template) / **0.150** (harmless control) against a ≥0.70 gate and a 0.40 kill line. The finding
  behind it is independently interesting: **stated knowledge tracks confidence, not correctness** —
  the model asserts answers it does not have **[LEDGER, F-59]**. That caps the slice at 258 rows.
- **k = 5 is not the paper's k and the result is k-dependent.** On the paper's five datasets at L40,
  `pc:unified@2` beats its matched control by **+0.138**, while `pc:unified` at k=5 gives **+0.004**
  **[AUDITED, F-39, F-113]**. Two cheap re-runs from rollouts already on disk (k=1, and the
  unfiltered `run-paper16/axes_vectors_all.pt` bank, which was built for this and never used) would
  close it.

---

# Part 4 — Dead ends, said plainly

Flagging these is meant to save weeks.

1. **The passC crossgen grid** (6,936 files). A duplicate of passA to 0.0022 (§1.4). Do not analyse
   it as a separate result. Keep the per-token store; drop the pooled grid.
2. **`expLBactivationLAYER22`.** No script, no doc, no owner, no Drive copy. Unless the owner
   recognises it, it is unusable as evidence. Back it up, then ignore it.
3. **Re-running passA/passC "for more seeds on all features" as a way to rescue the persona result.**
   The margin is −0.10 on ten datasets with a seed sd of ~0.007–0.015 (§1.2 §D). More seeds will
   sharpen the negative, not reverse it. Run them for error bars, not for hope.
4. **Chasing the S-10 cosines (0.04–0.14) as a paper16 fact.** They are pre-paper16 (§2.2). The
   paper16 values are measured and different.
5. **Quoting the kd bank's q = 0.067 near-miss as evidence for the method.** Its comparator is a
   random direction (§2.2/S-32). Against raw and against datapca, 0 of 4,032.
6. **Reporting "all three cap variants agree".** Two of the three are the same object in whichever
   view you take (§2.2/S-24).
7. **The 09-22 passA crossgen re-run.** It exited 1 on a split mismatch after the HP-KR top-up; the
   passA results in use are the 09-18/19 originals (`audit/experiments.csv` O44). Do not try to
   "complete" it without rebuilding the splits — and note the root passA grid and passA11 rest on
   **different HP-KR row sets** (`anomalies.md` CV-12).

---

# Part 5 — What the next steps cost

| # | Step | Why | Cost |
|---|---|---|---|
| 1 | Write up passP as a construct-validity section, with the per-dataset table | §1.1 — the largest unreported result | writing only; data measured |
| 2 | Per-dataset significance for passP vs root | turns §1.1 from a table into a test | one read-only lab script, minutes |
| 3 | Re-run passA seeds 0–4 with the full feature list | error bars on the 80-layer win-counts | CPU only, inputs on disk, ~hours |
| 4 | Verify S-26 (kd bank cos(PC1, assistant) below the Assistant-Axis thresholds) | the draft's headline bank vs its own reference's criterion | one read-only lab read of `run-kd/axes_vectors_s3.pt`, minutes |
| 5 | Re-run the premise script for S-30 (F-86's 0.0824 spread) | turns a reported small positive into a null | read-only lab re-run, minutes |
| 6 | Length controls on the cap bank's 0.745 | S-15 shows length is the live objection on this construct | reuses `q10` tooling, ~hour |
| 7 | Re-extract at k = 1 and run `run-paper16/axes_vectors_all.pt` through crossgen | closes three §N7 caveats at once | CPU, rollouts on disk |
| 8 | Back up `expLBactivationLAYER22`, `run-premise/`, `run-cap/`, `logs/` to Drive | single copies (RF-10) | `rclone copy`, owner's call |

---

# Part 6 — Corrections this workstream owes to earlier audit files

1. **`audit/phase1/sessions/findings_unrecorded.csv` S-24/S-25**: the duplicated pair in the
   *matrices* is `pc:capitulation` ≡ `pc:cap_vs_default`, not `cap_only` ≡ `cap_vs_default`; and the
   tautology **is** already in `kd/docs/FINDINGS.md:2144-2154` and `:3649` and in
   `cap/scripts/check_basis.py`. The row's "in no doc" is wrong.
2. **S-06**: the 0.717 TPR is a smoke-run value; the committed full run gives 0.494 at L44 and
   0.693 at L28. Quote the committed file.
3. **S-10**: the cosines describe the pre-paper16 (own100) bank; they conflict with paper16's
   measured values and must not be attributed to it.
4. **`audit/phase2/layer_sensitivity.csv` users**: seeds 0–4 carry only 3 features and 2 probes in
   passA / passC / `<root>`. Any "mean over six seeds" that mixes features is invalid. Add this to
   the Phase-3 traps list.
5. **Memory note `ndif-attention-memory-limit`**: 5.36 GiB should be 4.997 GiB (64 heads) — S-08.
6. **Memory note `unified-pc1-is-the-question-family`**: cos −0.98 vs the audit's measured −0.945.

---

# Appendix — scripts and raw output

| script (re-runnable, read-only) | output | covers |
|---|---|---|
| `audit/scripts/phase3/p3d_orphans.py` | `audit/phase3/out/p3d_orphans.txt` | passP, passA, passB_fine, passC, seed spread, layer profiles |
| `audit/scripts/phase3/p3d_orphans2.py` | `audit/phase3/out/p3d_orphans2.txt` | passA≡passC, the L40 decomposition, the paper-5 grid |
| `audit/scripts/phase3/p3d_unrecorded.py` | `audit/phase3/out/p3d_unrecorded.txt` | S-15, S-35, S-24, S-32, S-08, S-14b |
| `audit/scripts/phase3/p3d_unrecorded2.py` | `audit/phase3/out/p3d_unrecorded2.txt` | X1 (axes vs random), S-06, S-23, S-14c |
| `audit/scripts/phase3/p3d_passp_snip.py` | `audit/phase3/out/p3d_passp_perdataset.txt` | per-dataset passP vs root (run read-only on the lab PC via stdin; **no lab writes**) |

All five read only `audit/phase2/layer_sensitivity.csv`, files in the repo working tree, or — for
the last one — lab files opened read-only. Nothing was written outside `audit/`.
