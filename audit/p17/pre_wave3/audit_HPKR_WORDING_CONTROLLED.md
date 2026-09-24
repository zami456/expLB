# The HP-KR wording-controlled experiment — complete, with verified data

_Audit deliverable, 2026-09-23. Every number below was recomputed from raw matrices during this
audit by `audit/scripts/p5_hpkr_matched.py`, which is re-runnable. Where a figure comes from the
ledger rather than from my own recomputation, it is marked **[ledger]**._

**Status of the experiment: it was designed on 2026-09-19, run, and recorded as F-03, F-72 and F-76.
The draft states twice (p.12, p.13) that it has not been run. That statement is false.**

---

## 1. What the experiment is, and why it is the decisive one

`harm-pressure-knowledge-report` (**HP-KR**) is the dataset where the model denies knowledge it
actually has. It confounds two things:

- the **act** — denying knowledge (what the label marks), and
- the **wording** — the distinctive phrasing a denial uses ("I don't know", "I have no information
  about…").

A detector can score well on HP-KR by reading either one. That matters because HP-KR carries almost
the whole persona-coordinate result: the draft's headline contribution collapses from **+0.220 to
+0.009** when HP-KR is removed as train and test set (D-028, adjudicated in `CORRECTIONS.md`).

**`harm-pressure-kr-matched` is the control.** 258 rows in which **both classes deny knowledge**. The
wording is matched across the label, so a detector that is only reading the words cannot separate
them. Anything that still separates must be reading something else.

**This is the test the draft itself names as the one that would settle the question.**

---

## 2. How much of HP-KR is just wording? Two lexical baselines

| baseline | AUROC on HP-KR | note |
|---|---:|---|
| regex, "negated know" | **0.9125** | label-free; reads the response text only |
| regex, "does not affirm knowledge" | **0.9597** | label-free |
| **response length alone** | **0.776** | no semantics at all |
| — for comparison — the draft's headline | 0.919 | the knowledge-denial contrast |
| — for comparison — the deception axis, zero-shot | 0.6436 | **inside its own random-direction null** (p95 0.9377) |

**A regular expression with no access to labels matches or beats the paper's headline.** Counting
characters gets 0.776, above the deception axis's own reading. On this dataset, "detecting the lie"
and "detecting the phrasing" are not currently distinguishable.
*(Regex and length figures: F-01 and the Phase 4 D1/M-080 re-derivations, 2,139 Llama-3.3-70B rows.)*

---

## 3. The trained result — F-03, recomputed

`run-crossgen/kd11/matrices/mean_L{20,40,44}_s42.json` (in the git tree), mean pooling, `paper` probe,
seed 42, k=5, bank `paper16`, 11 datasets. "As source" = train on that dataset, test on the other ten,
averaged.

### Layer 40 — the layer F-03 reports

| feature | full HP-KR | matched slice | difference |
|---|---:|---:|---:|
| `axes` (persona) | 0.3004 | **0.7517** | **+0.4512** |
| `raw` | 0.6943 | 0.7150 | +0.0207 |
| `random` | 0.6590 | 0.5973 | −0.0617 |
| `length` | 0.5965 | 0.6137 | +0.0172 |
| `datapca` (matched control) | 0.5738 | 0.6861 | +0.1123 |
| | | **`axes` − `raw` = +0.0367** | (full set: **−0.3939**) |

**All eight of F-03's published figures reproduce exactly** (it rounds to 3 dp: 0.300 / 0.752 / 0.694 /
0.715 / 0.659 / 0.597 / 0.596 / 0.614).

**The headline reading is real:** on the full set the persona axes are far *worse* than a raw probe
(−0.394); on the wording-matched slice they are slightly *better* (+0.037). Removing the wording
confound reverses the sign.

### But it does not hold at other layers

| layer | `axes` | `raw` | **`axes` − `raw`** |
|---|---:|---:|---:|
| **20** | 0.3560 | 0.6126 | **−0.2566** |
| **40** | 0.7517 | 0.7150 | **+0.0367** |
| **44** | 0.7253 | 0.7119 | **+0.0134** |

**At layer 20 the effect inverts and is seven times larger in the wrong direction than the reported
effect is in the right one.**

**[ledger]** F-72 ran the same cell over the `passC` store's six layers and found the same shape with
slightly different magnitudes — `axes` 0.374 (L20) → 0.739 (L40), margin −0.216 → +0.033, **span
0.365**. Two independently built stores agree on the qualitative result and roughly on its size. F-72's
own summary: *"choosing the layer moves the headline by eleven times the size of the effect it
reports."*

---

## 4. Does the layer-40 margin survive a proper test? No — F-76

**[ledger]** F-76 nested layer selection inside a dataset-level bootstrap (B = 10,000, seed 42,
control `datapca`), which is the correct unit here because the quantity is an average over datasets:

| test | result |
|---|---|
| margin at the pre-specified L40 over `datapca` | **+0.0655** |
| 95% CI over datasets | **[−0.0693, +0.1887]** |
| P(margin ≤ 0) | **0.161 — does not clear** |
| 95% interval for the *best* layer | **[L5, L79]** — 93% of the network |
| label permutation | p = 0.38 |
| over all 80 layers | persona axes beat raw at **28 of 80**; `datapca` at **31 of 80** |

**The margin does not clear the bootstrap, and the data cannot say which layer is best.** F-76 is
explicitly a correction to F-75, which had returned "strengthens" on a narrower reading.

---

## 5. The sharpest result: the collaborator's own directions on the matched slice

Zero-shot (no training, so the F-108 leak in §6 does not apply). Collaborator's bank
(`bank_name=collab`, `axes_collab_s23.pt`), k=5, mean pooling, seed 42. Recomputed by me from
`run-crossgen/kd11/banks/collab/matrices/mean_L{20,40,44}_s42.json` (pulled read-only from the lab PC
into `audit/cache/`).

| direction | L20 matched / full | L40 matched / full | L44 matched / full |
|---|---:|---:|---:|
| **his 12v13 extended axis** | **0.4630** / 0.8057 | **0.6203** / 0.8765 | **0.5383** / 0.8491 |
| **split knowledge-denial axis** | **0.3932** / 0.5923 | **0.4989** / 0.6692 | **0.4825** / 0.6659 |
| **plain-assertion pole** (`kd_assertion`) | 0.4227 / **0.9118** | 0.5860 / **0.9344** | 0.5253 / **0.9266** |
| paper's 8v7 deception contrast | 0.5037 / 0.1652 | 0.6707 / 0.5562 | 0.5983 / 0.5733 |

**Read the third row first.** A **plain-assertion** direction — confident assertion versus hedging,
nothing to do with knowledge — scores **0.9118 / 0.9344 / 0.9266** on full HP-KR, matching the
headline. On the matched slice it falls to **0.4227 / 0.5860 / 0.5253**, at or near chance.

**The signal on HP-KR is the assertion-denial polarity.** That is the register diagnosis (F-13) with
the numbers attached.

His extended axis drops by **0.34 / 0.26 / 0.31** between the full set and the matched slice, landing
**at or below chance at two of three layers**. The split knowledge-denial axis is **below 0.5 at all
three**.

**[ledger]** The same picture on his own contrast definitions: 4-denial vs 6-merged reads
0.8734 / 0.8826 / 0.8734, but 4-denial vs 3-**sincere-ignorance** (wording-matched) reads
**0.5923 / 0.6692 / 0.6659**. **0.21–0.28 AUROC of the HP-KR result is the plain-assertion half.**

**None of these numbers appears in either document.**

---

## 6. The leak — and it does NOT need new NDIF data to fix

**CORRECTED 2026-09-23** (this section said the opposite; see `audit/CORRECTIONS.md` #8/#9).

**What the leak is, exactly.** `harm-pressure-kr-matched` is **a view over HP-KR**, not a separate
dataset: **258 rows, and 100% of its row_ids are HP-KR row_ids** (F-108, measured). So:

| direction | contamination |
|---|---|
| test on `kr-matched`, trained on HP-KR | ~**80%** of its 258 rows sit in HP-KR's 80% train split |
| test on HP-KR, trained on `kr-matched` (F-03's design) | **258 of 2,139 = 12%** of the HP-KR test column |

F-108 also established the leak is **confined to this one pair** — no other dataset pair shares a
single row_id.

**The fix is a re-split, not new data.** Exclude the 258 shared row_ids from whichever side is the
training set and recompute. **This runs on activations that already exist: CPU-only, no NDIF, no
GPU.** It is not hypothetical — **F-108 already did exactly this** for the pooled-training claim, via
leave-one-**cluster**-out (HP-KR and `kr-matched` in the same `lb_harm` cluster, so a sibling is never
on both sides), and reported **+0.125 to +0.157 against the leaked +0.178**.

**So the honest cost of closing the gap on F-03 is hours of CPU, not days of GPU.** The same
`clustered_*` machinery that produced `clustered_L40_mean_s42.json` applies directly.

**New NDIF rollouts would only be needed for a different, stronger experiment** — a matched slice
built from **new material** (fresh items, independent of HP-KR's pool) rather than a disjoint re-split
of the existing pool. That would test generalisation to new wording-matched *content*, which is a
worthwhile thing to want but is **not** what removing this leak requires.

## 6b. THE LEAK IS NOW CLOSED, AND MEASURED (2026-09-23, run by the audit)

**Done.** A copy of `27_crossgen.py` was patched so that, off-diagonal, a target row is dropped
whenever its `row_id` appears in the **source's training rows**. Everything else is untouched. The
guard fired on **exactly one dataset pair, in both directions**, confirming F-108's measurement that
no other pair shares a row_id. Of `kr-matched`'s 258 rows, only **59** survive removing those in the
knowledge-report training split — **77% contamination**, as F-108 predicted.

Reproduce: `python audit/scripts/p6_resplit_compare.py`.
Provenance: run on the lab PC from `C:/Users/user4/expLB-resplit` (a copy), reading the same
activations read-only and writing nowhere else. The original script is byte-identical before and
after (md5 `681126242fe4414e4a3c131793de81c6`).

### F-03's headline barely moves — the leak did NOT manufacture it

| layer | leaked `axes − raw` | **guarded `axes − raw`** | change |
|---|---:|---:|---:|
| 20 | −0.2566 | **−0.2545** | +0.0022 |
| 40 | +0.0367 | **+0.0309** | −0.0058 |
| 44 | +0.0134 | **+0.0050** | −0.0084 |

**The conclusion of §3 and §4 stands.** The margin shrinks at every layer but the picture is
unchanged: still inverted at L20, still tiny at L40/L44, still not clearing F-76's bootstrap. The
change is small for an arithmetic reason — one corrected cell is averaged with nine untouched ones.

### The individual cells move a great deal — and this is the new finding

| cell | layer | leaked | **guarded** | change |
|---|---|---:|---:|---:|
| `matched → knowledge-report` | 40 | 0.5905 | **0.5416** | −0.049 |
| | 44 | 0.6534 | **0.5719** | −0.081 |
| `knowledge-report → matched` | 20 | 0.6004 | **0.7190** | +0.119 |
| | 40 | 0.5674 | **0.4262** | **−0.141 — below chance** |
| | 44 | 0.6675 | **0.5667** | −0.101 |

**With the leak removed, the persona axis stops transferring within the knowledge-report family.** At
layer 40 it falls to **0.4262 — inverted**. The apparent ability to carry from the full dataset to its
wording-matched slice was substantially the shared rows.

**`raw` moves too** (0.7932 → 0.7214 at L40 on that cell), so the leak flattered every feature, not
only the persona one. That is why the *difference* barely changes while the *levels* do.

### What this settles

- **For F-03:** the leak was real but immaterial to its headline. **Report the guarded numbers**
  (−0.2545 / +0.0309 / +0.0050) and state that the leak was tested and does not account for the result.
- **Against the method:** a new, cleanly measured negative — within-family transfer at L40 is
  **inverted** once contaminated rows are removed.
- **Still open:** 59 rows is a small test set, so the affected cells carry wide intervals. And this
  re-split makes the rows disjoint; it does not make the *material* independent. A matched slice built
  from new items remains a stronger, and unrun, experiment.

---

## 6c. THE OTHER CONTROL SLICE — never run before, and it is the bigger one (2026-09-23)

`eval_datasets.hpkr_slice` has **two** modes. The project only ever used one.

| slice | holds constant | so the label marks | rows | status |
|---|---|---|---:|---|
| `text_matched` | **wording** — every response denies | whether the model **knew** | 258 | used by F-03 |
| `knowledge_matched` | **knowledge** — every row is known | whether the model **denied** | **2,001** | **never run until now** |

Built as an isolated view (junctions to the original datasets + one hardlinked new folder;
`C:/Users/user4/expLB/run-coords/passB` verified unchanged, still 13 dataset dirs, no new key).
Leak guard on. 12 datasets, mean pooling, `paper` probe, seed 42.

### Trained on it, tested on the other eleven

| layer | `axes` | `raw` | **`axes − raw`** | `datapca` | `random` | `length` |
|---|---:|---:|---:|---:|---:|---:|
| 20 | 0.3850 | 0.5711 | **−0.1860** | 0.5208 | 0.5455 | 0.6174 |
| 40 | 0.3335 | 0.7093 | **−0.3758** | 0.6636 | 0.6718 | 0.6174 |
| 44 | 0.5267 | 0.7203 | **−0.1937** | 0.7037 | 0.6237 | 0.6174 |

- **The persona axes are inverted at L20 and L40** (0.385, 0.334 — both below 0.5) and barely above
  chance at L44.
- **They lose to a raw probe by 0.19 to 0.38** — the largest deficit anywhere in the project.
- **A one-dimensional length feature (0.6174) beats them at every layer.**

### In-distribution, on its own held-out 20%

| layer | `axes` | `raw` | `length` |
|---|---:|---:|---:|
| 20 | 0.9541 | **0.9995** | 0.8318 |
| 40 | 0.8347 | **0.9992** | 0.8318 |
| 44 | 0.7209 | **0.9993** | 0.8318 |

**A raw probe separates this slice essentially perfectly (0.999).** With knowledge held constant, the
only thing left to detect is the denial itself — and it is trivially detectable. Even *length alone*
gets 0.832. The persona axes get 0.95 → 0.72, **falling as depth increases**, and never reach raw.

### What this establishes, and what it does not

**Establishes.** On the slice where **only denial varies**, the signal is near-perfectly present
(raw 0.999) yet **nothing learned there transfers** — the persona axes go *below chance* carrying it
elsewhere. The dataset's separability is the denial, and the denial is local to it. This is the
register diagnosis tested head-on, on **2,001 rows** rather than inferred from the plain-assertion
direction, and it is the **strongest negative result in the project against persona coordinates.**

**Does NOT establish — a confound I have to flag.** I first read the zero-shot numbers as evidence
too: the contrast reads 0.2405 / 0.6466 / 0.6085 on the knowledge-matched slice against
0.2504 / 0.6436 / 0.6032 on the full dataset — nearly identical, which looks like "the axis is
indifferent to whether the model knows". **That comparison proves nothing.** The knowledge-matched
slice is **93.5% of the full dataset** (2,001 of 2,139), so the two readings must agree by
construction. Only the transfer and in-distribution tables above carry weight.

---

## 7. What can and cannot be concluded

**Can be stated as measured:**

1. **The wording-controlled test was run.** F-03, F-72, F-76, 2026-09-19 onward.
2. **A label-free regex scores 0.9125–0.9597 on HP-KR**, at or above the paper's headline of 0.919;
   length alone reads 0.776.
3. **Zero-shot, on the wording-matched slice, every direction tested is at or near chance** — the
   collaborator's extended axis 0.4630 / 0.6203 / 0.5383, his knowledge-denial axis below 0.5 at all
   three layers.
4. **A plain-assertion direction reproduces the HP-KR headline** (0.9118–0.9344) and collapses on the
   matched slice. The construct being detected is register, not knowledge denial.
5. **Trained, the matched-slice margin is +0.0655, CI [−0.0693, +0.1887], P = 0.161** — it does not
   clear a dataset-level bootstrap — and it **inverts to −0.2566 at layer 20**.

**Cannot be concluded, in either direction:**

- **Not** that persona coordinates definitely fail on knowledge denial — the trained arm is
  underpowered (59 rows after the guard, one seed). **The leak is now closed and measured (§6b): it
  does not account for F-03's result**, so that objection is answered rather than outstanding.
- **Not** that they work — nothing clears its control.
- The honest summary is **inconclusive-trending-negative**, and the zero-shot arm (§5), which is clean,
  is the negative part.

**The sentence to use, measured rather than promised:**

> *"We ran a wording-matched slice in which both classes deny knowledge. At the pre-registered layer
> the margin is +0.066 (95% CI [−0.069, +0.189], P = 0.161) and does not clear a dataset-level
> bootstrap; at layer 20 it inverts to −0.257. Zero-shot, every persona direction we tested reads at or
> near chance on the matched slice, while a plain-assertion direction reproduces the full-dataset
> result (0.93). We therefore cannot separate knowledge-denial detection from assertion-register
> detection on this dataset."*

---

## 8. What to change in the documents

| document | change |
|---|---|
| **the draft, p.12 and p.13** | **delete the claim that this test has not been run.** Replace with §7's measured sentence. This is the single most exposed statement in the paper. |
| **the draft, §6 caveat** | the wording confound is not "not ruled out" — it is **not currently distinguishable**. Quote the 0.9125 regex beside the 0.919 headline. |
| **`kd/docs/FINDINGS.md` F-03** | still marked `status: provisional` with "no dataset-level bootstrap" as a caveat. F-76 supplied that bootstrap and it does not clear. **Mark F-03 superseded by F-76.** |
| **anywhere quoting F-03's 0.752** | add the layer sweep (−0.2566 at L20) and the F-76 bootstrap. The number alone is misleading. |
| **new** | the §5 zero-shot table is in no document and is the strongest single piece of evidence for the register diagnosis. It should be a figure. |
| **next experiment** | re-run F-03's cell with the 258 shared row_ids excluded from training — **CPU-only, existing activations** (§6). It is the cheapest remaining experiment in the project and it closes the last real objection to §3/§4. |

---

## 9. Reproduce this

```
python audit/scripts/p5_hpkr_matched.py
```

Reads `run-crossgen/kd11/matrices/mean_L{20,40,44}_s42.json` (git) and
`audit/cache/labpc/run-crossgen/kd11/banks/collab/matrices/mean_L{20,40,44}_s42.json` (pulled
read-only from the lab PC). Prints §3 and §5 in full.

**[ledger]**-marked figures come from `kd/docs/FINDINGS.md` F-72, F-76, F-108 and were not
independently recomputed here; everything else in §3 and §5 was.
