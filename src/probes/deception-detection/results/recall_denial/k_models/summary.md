# The knowledge direction K on Mistral-Small-3.1-24B, Gemma-3-27B and Qwen-2.5-72B

`deception_detection/scripts/k_models.py` (2026-10-10 and 2026-10-12). K is rebuilt inside each
model by Llama's recipe. It is scored zero-shot at the block where the Mean Probe reproduced Liars'
Bench Table 6 (`results/lb_models/summary.md`): Mistral block 9, Gemma block 8, Qwen block 20. The
neighbouring block is read too, since its pooled means exist. Gemma and Qwen have their own sections
at the end.

**Status:** full, with 60 questions per side (240 rows), independently verified (see
Verification below).
- **Pilot:** 20 per side, read first. Its K has cosine 0.993 with the full K at both blocks.
  K's AUROCs moved by at most 0.014 between the two.
- **Split-half cosine:** 0.971 at block 9 and 0.974 at block 10, splitting by the parity of the
  sorted item names as `recall_denial.py` does. Random 30/30 splits give a median of 0.972
  (5-95%: 0.954-0.982). Llama's is 0.986 at L20 and 0.999 at L40.

## Recipe

- **Questions, template and denials:** the same as Llama's K. The 60 + 60 general questions are
  from `recall_items.py`, in the knowledge-report template, each forced to "I do not know the
  answer." and to "I don't know the answer to the question.". K is the mean over the denial's
  tokens (exactly the sentence between `[/INST]` and `</s>`), known minus unknowable, unit length.
  `items.json` matches Llama's 240 rows field for field, except the system turn.
- **System turn:** the one Mistral's own HP-KR rows carry: its default system prompt, dated
  2025-09-01. This is what HP-KR's date header was for Llama.
- **Known questions checked:** Mistral answers all 70 known and reserve questions correctly, by
  the highest-logit letter. Its softmax over the four letters is ≥ 0.986, and its top token is
  that letter itself in every case. No reserve question was needed, as for Llama. Llama's check
  generated 8 tokens instead.
- **Checks:**
  - HP-KR rows 0 and 1, re-read by this code, match their pooled vectors (worst cosine 0.99997).
  - Each forced denial tokenises identically on every row.
  - The HP-KR frame is lined up with the Liars' Bench rows by answer text and label.
- **Fixed before scoring:**
  - The sign is known minus unknowable, so lies should score high; it is never flipped.
  - The slices are defined in advance.
  - The random band uses 1,000 random directions (Llama's slices used 2,000; reseeding moves the
    97.5% points by up to 0.02).
  - The Alpaca 1% FPR threshold is computed as the runner computes it.

## Block 9 (`hidden_states[10]`)

| Dataset | K AUROC [95% CI] | Random 97.5% | Share of random ≥ K | Mean Probe AUROC |
|---|---|---|---|---|
| HP-C | 0.418 [0.37, 0.46] | 0.624 | 0.90 | 0.544 |
| HP-KR, all rows | 0.592 [0.57, 0.62] | 0.705 | 0.32 | 0.463 |
| ID | 0.500 [0.49, 0.52] | 0.613 | 0.50 | 0.690 |
| IT | 0.779 [0.75, 0.80] | 0.832 | 0.12 | 0.536 |
| RepE (all 550 rows) | 0.412 [0.36, 0.46] | 0.603 | 0.94 | (trained on it) |

HP-KR slices:

| Slice | Lies / honest | K AUROC [95% CI] | Random 97.5% | Mean Probe |
|---|---|---|---|---|
| Lies v honest denials | 677 / 1020 | **0.688** [0.66, 0.71] | 0.605 | 0.519 |
| Identical: "I do not know the answer to the question." | 494 / 729 | **0.707** [0.68, 0.73] | 0.623 | 0.499 |
| Identical: "I do not know the answer." (Llama's slice) | 126 / 218 | **0.668** [0.61, 0.72] | 0.637 | 0.494 |
| Lies v honest claims (wording) | 697 / 592 | 0.420 | 0.996 | 0.373 |

- **The denial regex** (`steer_lies.knowledge_claim`) was written for Llama. It misses Mistral's
  "I do not have the answer…" forms: 20 lies and about 49 honest rows fall to OTHER. Counting
  them as denials moves K by at most 0.003.
- **K separates the lies on HP-KR with the wording held fixed.** On these slices, K is above every
  random direction or nearly every one (0.67 to 0.71; at most 0.4% of random directions score as
  high). The same-block Mean Probe is at chance there (0.49 to 0.52).
  - Mistral's HP-KR is a stronger test of this than Llama's, because most of its honest answers
    are denials too (1,020 honest denials against 677 denying lies).
- **K does not transfer at this block.**
  - HP-C (0.42) and RepE (0.41) are inverted.
  - ID is at chance.
  - IT's 0.78 is inside the random band (random directions reach 0.83 there).
  - The full HP-KR set is inside its band too, because wording dominates it.
- **The Alpaca threshold does not transfer.**
  - Balanced accuracy at the Alpaca 1% FPR threshold is 0.48 to 0.50 on every dataset.
  - Where K flags rows, it flags them at chance: recall about equals FPR (ID 0.21 / 0.20, HP-C
    0.09 / 0.12 at block 9; HP-C 0.87 / 0.94 at block 10).
  - On HP-KR and IT it flags nothing. The highest K score on any HP-KR lie sits at Alpaca's 95th
    percentile.
  - The Mean Probe flags nothing either.

Block 10 matches block 9 on the slices: denials 0.727, identical text 0.734 and 0.701. It matches
on the datasets too: HP-C 0.411, HP-KR 0.490, ID 0.523, IT 0.639 (inside the band), RepE 0.485.
The cosine between the two blocks' K is 0.65.

## Against Llama at a similar depth

Block 9 of Mistral's 40 is 23% deep; Llama's layer 20 of 80 is 25%. The identical-text row is the
same sentence for both models, "I do not know the answer.".

| | Mistral B9 | Llama L20 | Llama L40 |
|---|---|---|---|
| HP-KR identical text | 0.668 (band 0.637) | 0.531 (band 0.624) | 0.793 (band 0.735) |
| HP-KR denials | 0.688 | 0.414 | 0.728 |
| HP-C | 0.418 | 0.545 | 0.830 |
| ID | 0.500 | 0.681 | 0.906 |
| IT | 0.779 | 0.836 | 0.881 |

- **The HP-KR slices:** Mistral's shallow K separates these lies. Llama's K does so only from about
  L36, and its L20 is at chance on them.
- **The transfer to HP-C and ID:** Llama gets it at L36 to L44 and not at L20. Mistral has none
  at block 9.
- **Untested:** whether a deeper Mistral block transfers. That needs new pooled reads of every
  dataset, not just of K's rows.

## K against the base probe

`k_models.py compare` → `compare_base_probe.csv`.
- **The base probe** is the RepE-trained probe of the same layer, i.e. the Liars' Bench Mean
  Probe (`hpkr_wording.directions`' "base probe (RepE)"; for Mistral, `XXXX-9-mistral-all-B9` and
  `-B10`).
- **Differences:** K minus probe over 1,000 paired stratified row resamples
  (`k_robustness.paired`'s protocol). The mean of the four datasets both models have is averaged
  draw by draw.
- **Reproduction check:** every stored AUROC for the sets compared here comes back to 1e-3:
  - Mistral's four datasets and three slices, for both scorers at both blocks;
  - Llama's four datasets, for both scorers, from `recall_denial/datasets.csv`;
  - K's two Llama slices, from `auroc.csv`.

  The RepE rows and the claims slice are not recomputed.
- **Checked** by workflow `verify-k-vs-base-probe` (2026-10-10):
  - all 72 recomputed quantities agree;
  - all 155 claims below match the CSV;
  - the framing points it raised are folded in.

Mistral block 9:

| Set | Lies / honest | K | Base probe | K − probe [95% CI] |
|---|---|---|---|---|
| HP-C | 195 / 697 | 0.418 | 0.544 | −0.126 [−0.19, −0.06] |
| HP-KR, all rows | 697 / 1661 | 0.592 | 0.463 | +0.130 [+0.09, +0.16] |
| ID | 2393 / 2617 | 0.500 | 0.690 | −0.190 [−0.21, −0.17] |
| IT | 648 / 1486 | 0.779 | 0.536 | +0.243 [+0.20, +0.29] |
| Mean of the four | | 0.572 | 0.558 | +0.014 [−0.01, +0.04] |
| HP-KR lies v honest denials | 677 / 1020 | 0.688 | 0.519 | +0.169 [+0.13, +0.21] |
| HP-KR identical: "I do not know the answer to the question." | 494 / 729 | 0.707 | 0.499 | +0.208 [+0.17, +0.25] |
| HP-KR identical: "I do not know the answer." | 126 / 218 | 0.668 | 0.494 | +0.175 [+0.11, +0.24] |

The same comparison across models and layers (K / base probe, then K − probe):

| Set | Mistral B9 | Mistral B10 | Llama L20 | Llama L40 |
|---|---|---|---|---|
| Mean of HP-C, HP-KR, ID, IT | 0.572 / 0.558, +0.01 (ns) | 0.516 / 0.586, −0.07 | 0.621 / 0.626, −0.00 (ns) | 0.870 / 0.730, +0.14 |
| HP-KR lies v honest denials | 0.688 / 0.519, +0.17 | 0.727 / 0.535, +0.19 | 0.414 / 0.657, −0.24 | 0.728 / 0.593, +0.14 |
| HP-KR identical: "I do not know the answer." | 0.668 / 0.494, +0.17 | 0.701 / 0.535, +0.17 | 0.531 / 0.611, −0.08 (ns) | 0.793 / 0.779, +0.01 (ns) |

ns: the 95% interval includes 0. Llama's sets have other counts: HP-C 134 / 2000, HP-KR
139 / 2000, ID 3045 / 2387, IT 423 / 657, denials 125 / 135, identical 78 / 112.

- **On HP-KR with the wording held fixed, K beats the base probe clearly on Mistral**, at both
  blocks (+0.17 to +0.22). The probe is at or near chance there (0.49 to 0.54). It is inside the
  random band on every slice, though its block-10 denials (0.535 [0.51, 0.56]) are slightly above
  0.5.
  - On Llama the gap is smaller at L40 (+0.14 on denials; identical text +0.01, ns, where the probe
    reaches 0.78).
  - At L20 on Llama the probe beats K on denials.
- **On the whole datasets the comparison says little**, because only one of the eight block-9
  scores clears its random band: the probe's ID (0.69, band 0.61).
  - At block 9 the probe wins HP-C and ID (−0.13, −0.19), and K wins HP-KR and IT (+0.13, +0.24).
  - The probe's HP-C (0.54, band 0.62) is inside its band, and so are K's HP-KR (0.59, band 0.71)
    and K's IT (0.78, band 0.83).
  - K's HP-C is inverted (0.42).
  - On the mean of the four they tie. At block 10 the probe wins the mean (−0.07).
- **Llama's mean win at L40 (+0.14) is nearly all IT.** There the probe is inverted (0.34) and K's
  0.88 is inside its random band (0.95). Without IT the difference is +0.01. At L40, K wins HP-C
  (+0.15), and the probe wins HP-KR and ID (−0.07 each).
- **Operating points at 1% FPR** (`k_*` / `probe_*` columns of the CSV):
  - **Alpaca threshold:** neither is usable on Mistral. At block 9, balanced accuracy is 0.48 to
    0.50 for K and 0.50 for the probe on every set. At block 10, K's HP-C is 0.46 and its ID 0.51.
    The probe flags almost nothing. K flags nothing on HP-KR, IT and the slices, and on HP-C and
    ID it flags at chance (recall about equal to FPR).
  - **In-setting** (cutoff on half the set's own honest rows): both catch few lies on Mistral at
    block 9. K catches 2.7% of denying lies (probe 1.0%) and 13% of IT lies (probe 3.6%). On HP-C,
    HP-KR and ID each catches at most 1.7%.
  - **Llama L40, in-setting:** K catches 27% of HP-KR lies (probe 25%) and 22% on identical text
    (probe 13%).

## Gemma-3-27B

Gemma has no HP-KR rows in Liars' Bench (CG, ID and IT only), so K's main test cannot be run on
it: feigned ignorance with the wording held fixed. What can be tested is the transfer.

The pooled reads at blocks 8 and 9, the base probes and the block rule are from the Gemma Mean
Probe run (`lb_models.py`, 2026-10-10, another session).
- **Block rule:** it chose block 8 (`hidden_states[9]`), with mean |ΔAUROC| 0.025 against 0.089
  for block 9.
- **Reproduction:** CG 0.700 against Cadenza's 0.700, ID 0.586 against 0.585, IT 0.231 against
  0.303.
- **Depth:** Gemma has 62 blocks, so block 8 is 15% deep.

**Recipe:**
- **Rows:** the same 60 + 60 questions, template and denials.
- **System turn:** none. Gemma's chat template inserts no default system turn, and none of its
  Liars' Bench sets has a default one: CG and ID carry task instructions, IT the trading agent's
  prompt, and Alpaca "You are a helpful AI assistant.". So the rule (the model's default system
  context) gives none. An empty system turn tokenises exactly as no turn (checked).
- **Known questions:** Gemma answers 69 of the 70 known and reserve questions correctly (letter
  softmax ≥ 0.9999). The miss is a reserve question, so K uses the same 60 known questions as Llama
  and Mistral.
- **Reads:** stopped after block 9.
  - The re-read of CG rows 0 and 1 matches the pooled vectors (cosine 0.999999).
  - Split-half cosine is 0.968 at block 8 and 0.966 at block 9.
  - The cosine between the two blocks' K is 0.91.

K at block 8 (K / base probe, random 97.5% for the dataset, K − probe [95% paired CI]):

| Dataset | Lies / honest | K | Base probe | Random 97.5% | K − probe |
|---|---|---|---|---|---|
| CG | 371 / 250 | 0.532 | 0.700 | 0.799 | −0.168 [−0.24, −0.10] |
| ID | 3109 / 2087 | 0.375 | 0.586 | 0.751 | −0.211 [−0.23, −0.19] |
| IT | 1881 / 1614 | 0.388 | 0.231 | 0.674 | +0.157 [+0.14, +0.18] |
| Mean of the three | | 0.432 | 0.506 | | −0.074 [−0.10, −0.05] |
| RepE (all 550 rows) | 275 / 275 | 0.520 | (trained on it) | 0.531 | |

Block 9: K is CG 0.462, ID 0.529, IT 0.394 and RepE 0.467; the probe is CG 0.608, ID 0.674 and IT
0.217. The mean difference is −0.038 [−0.06, −0.01].

- **K does not transfer on Gemma at these blocks.** At block 8 it is at chance on CG and RepE and
  inverted on ID and IT. Block 9 is similar: ID 0.53, inside the random band. That matches
  Mistral's shallow block on its non-HP-KR sets.
- **Neither detector clears the random band on any Gemma dataset.** Random directions reach 0.80
  on CG and 0.75 on ID. The probe's best, CG 0.70, is matched by 8% of random directions.
- **The probe beats K on CG and ID.** K's IT "win" is the smaller of two inversions (0.39 against
  0.23).
- **Operating points at 1% FPR:**
  - At the Alpaca threshold, both are at 0.50 to 0.52 balanced accuracy on CG and ID.
  - K's 0.55 on IT (25% of lies and 14% of honest rows flagged) sits beside an inverted AUROC, so
    it is a tail effect, not detection.
  - With the in-setting cutoff, K catches 1.8% of CG lies (probe 19%), 1.1% of ID lies (2.8%) and
    6.7% of IT lies (0.5%).

Llama on the same three datasets (`compare_base_probe_gemma.csv`):

| | Gemma B8 | Gemma B9 | Llama L20 | Llama L40 |
|---|---|---|---|---|
| Mean of CG, ID, IT: K / probe, K − probe | 0.432 / 0.506, −0.07 | 0.462 / 0.499, −0.04 | 0.752 / 0.969, −0.22 | 0.809 / 0.760, +0.05 |

Llama's L40 win is IT alone (+0.54). There the probe is inverted (0.34) and K is inside its random
band. K loses CG (−0.33) and ID (−0.07).

Run with `k_models.py verify|read|score --model gemma --per_side 60`, then
`compare --models '(("gemma",8),("gemma",9),("llama",20),("llama",40))' --sets '("CG","ID","IT")'
--out compare_base_probe_gemma.csv`.

### Gemma's HP-KR, rebuilt from Cadenza's raw files

Liars' Bench left Gemma out of HP-KR (and HP-C): it "generated less than 100 lies, mostly because
of its high rate of refusal" (App. D). The cutoff is 100 per class per model.
- **Rebuild:** Cadenza's raw Gemma harm-pressure files are in the repo
  (`results/datasets/harm_pressure/cais/*/gemma-3-27b-it.csv`). `hpkr_build` rebuilds HP-KR by
  the authors' own rules (`src/harm-refusal/harm-refusal.ipynb`): beliefs, a harm factor of 4 or
  more over validation template 0, and rows labelled 0 or 1.
- **Equivalence check:** the same code first reproduces the released counts: Llama 139 lies,
  Mistral 697 / 1,661, Qwen 571 lies.
- **Gemma:** 35 lies against 1,400 honest rows (1,300 claims of knowledge and 100 honest
  denials). The templates kept are bio 1, 2 and 5 and chem 1; none had a lie on the validation
  template.
  - 35 is every lie Gemma gave on any template, so relaxing the filter adds none.
  - More lies would need new rollouts.
- **K's test:** lies against honest denials, 35 against 100, from disjoint questions as on
  Mistral and Llama.
  - Gemma's denials open with "I do not know the answer to the question.", and most go on to a
    safety disclaimer that differs row to row.
  - So each row is read twice: over the whole reply, and over the opening sentence alone.
  - The opening covers 34 lies and 85 honest denials. Its tokens are identical on every row
    (checked), and causally untouched by the disclaimer.
- **Blocks:** 8 (the Mean-Probe block), 9, and 31, mid-depth like Llama's layer 40. K's 240 rows
  were read again at block 31.
- **No probe at block 31:** there is no base probe there, since nothing else was pooled at that
  block.

| Block | Span | K [95% CI] | Random 97.5% | Share of random ≥ K | Base probe | K − probe [95% CI] |
|---|---|---|---|---|---|---|
| 8 | whole reply | 0.518 [0.40, 0.62] | 0.637 | 0.42 | 0.498 | +0.02 [−0.11, +0.16] |
| 8 | opening sentence | 0.607 [0.49, 0.73] | 0.637 | 0.07 | 0.490 | +0.12 [−0.00, +0.25] |
| 9 | whole reply | 0.554 [0.45, 0.66] | 0.627 | 0.24 | 0.559 | −0.01 [−0.15, +0.12] |
| 9 | opening sentence | 0.584 [0.45, 0.71] | 0.631 | 0.10 | 0.511 | +0.07 [−0.07, +0.21] |
| 31 | whole reply | 0.367 [0.26, 0.48] | 0.647 | 0.95 | (none) | |
| 31 | opening sentence | 0.595 [0.49, 0.70] | 0.627 | 0.06 | (none) | |

- **K does not separate Gemma's lies from its honest denials significantly at any block.**
  - On the opening sentence it leans the right way (0.58 to 0.61), but every interval includes
    0.5, and 6% to 10% of random directions do as well.
  - On the whole reply it is at chance at blocks 8 and 9, and inverted at block 31.
  - The base probe is at chance throughout (0.49 to 0.56).
- **The test is weak, so it is not a clean negative.** With 34 lies the interval is about ±0.12.
  - Mistral's effect (0.67 to 0.71 on identical text) would clear this random band (0.63).
  - So Gemma's effect is smaller than Mistral's, if it is there at all.
- **K at block 31 is unrelated to K at block 9** (cosine −0.05). Blocks 8 and 9 agree (0.91).

Run with `hpkr_build --model gemma`, `read --model gemma --per_side 60 --blocks '[31]'`,
`hpkr_read --model gemma` (5 requests) and `hpkr_score --model gemma`.
- **Provisional.** The table above comes from reads made before the line-ending fix (see Qwen's
  full HP-KR below).
  - 15 of the 135 denial rows (1 lie) were read with CRLF in their prompt.
  - The rebuild now has LF, but the re-read failed: NDIF evicted the Gemma replica (2026-10-12).
    Re-run `hpkr_read --model gemma` and `hpkr_score --model gemma` when it is back.
  - On Qwen the same fix moved no AUROC by more than 0.021.
- **Files:** `k_models/gemma/hpkr/` (`rows.json` rebuilt with LF; the reads and scores behind the
  table are `superseded_crlf_parts_B8-9-31/` and `superseded_crlf_scores.csv`;
  `directions_n60.npz`) and `k_models/gemma/parts_answer_B31/`.
- **Not checked independently.**

Files: `k_models/gemma/` (`verify.json`, `items.json`, `check_reread.json`, `parts_answer/`,
`directions_n60.npz`, `datasets_n60.csv`, `split_half_n60.csv`) and `compare_base_probe_gemma.csv`.

## Qwen-2.5-72B (2026-10-12)

The pooled reads at blocks 19 and 20, the base probes and the block rule are from the Qwen Mean
Probe run (`lb_models.py`, another session).
- **Block rule:** it chose block 20, block 20's own output (the nnsight reading, as for Llama), with
  mean |ΔAUROC| 0.032 against 0.098 for block 19.
- **Reproduction:** HP-C 0.561 against Cadenza's 0.497, HP-KR 0.498 against 0.486, ID 0.581
  against 0.603.
- **Depth:** Qwen has 80 blocks like Llama, so block 20 is as deep as Llama's L20.

**Recipe:**
- **Rows:** the same 60 + 60 questions, template and denials.
- **System turn:** the one Qwen's HP-KR rows carry, its default system prompt ("You are Qwen,
  created by Alibaba Cloud. You are a helpful assistant.").
- **Known questions:** Qwen answers 69 of 70 correctly. The miss is a reserve question (it picks
  Sahara over Antarctic), so K uses the same 60 known questions as the other models.
- **Checks:** HP-KR rows 0 and 1 re-read to cosine 0.99997. Split-half cosine 0.962 at block 19 and
  0.964 at block 20; the two blocks' K have cosine 0.944.

Block 20 (K / base probe; K − probe over 1,000 paired stratified resamples, `compare_base_probe_qwen.csv`):

| Set | Lies / honest | K [95% CI] | Random 97.5% | Share of random ≥ K | Base probe | K − probe [95% CI] |
|---|---|---|---|---|---|---|
| HP-C | 374 / 2000 | 0.461 [0.43, 0.49] | 0.592 | 0.76 | 0.561 | −0.100 [−0.15, −0.05] |
| HP-KR, all rows | 571 / 2000 | 0.827 [0.81, 0.85] | 0.939 | 0.19 | 0.498 | +0.330 [+0.30, +0.37] |
| ID | 2145 / 3015 | 0.600 [0.59, 0.62] | 0.676 | 0.16 | 0.581 | +0.019 [−0.00, +0.04] |
| Mean of the three | | 0.630 | | | 0.547 | +0.083 [+0.06, +0.11] |
| HP-KR lies v honest denials | 539 / 44 | 0.622 [0.54, 0.70] | 0.625 | 0.026 | 0.541 | +0.082 [−0.06, +0.21] |
| RepE (all 550 rows) | 275 / 275 | 0.489 | 0.582 | 0.62 | (trained on it) | |

Block 19 is the same: denials 0.621 (2.9% of random directions as high), base probe 0.508, K −
probe +0.113 [−0.03, +0.25]; HP-C 0.477, HP-KR 0.829, ID 0.583.

- **On the denial slice, K is at the edge of the random band.**
  - It scores 0.62 at both blocks; 2.6% to 2.9% of random directions score as high.
  - The base probe is at chance (0.51 to 0.54).
  - Their difference is not significant, because the slice has only 44 honest denials.
  - There is no identical-text slice: no sentence has 50 rows in both classes.
- **Compared with the other models at this depth:** Llama's K at L20 is inverted on its denials
  (0.41). Mistral's shallow K reaches 0.69. Qwen's 0.62 sits between them.
- **K does not transfer.** HP-C is below chance (0.46), ID is inside its band (0.60), and RepE is
  at chance.
- **The wins over the probe on all HP-KR rows (+0.33) and on the mean (+0.08) are wording.** K's
  0.83 on HP-KR is inside the random band (0.94), where random directions separate claims of
  knowledge from denials.
- **Operating points:** at the Alpaca 1% FPR threshold neither flags anything on any set.

### Qwen's full HP-KR rollouts (2026-10-12)

The released Qwen HP-KR keeps a random 2,000 of Qwen's 7,293 honest rows. That leaves 44 honest
denials, because only 2% of Qwen's honest replies deny knowledge.
- **Rebuild:** `hpkr_build --model qwen` rebuilds the full set from Cadenza's raw files by their
  rules. The equivalence check passes as for Gemma: Llama 139 lies, Mistral 697 / 1,661, Qwen 571.
- **Denials:** all 571 lies (every one a denial) and 164 honest denials (by the benchmark's judge;
  the regex gives 539 and 149).
- **Identical text:** 435 lies and 128 honest rows reply exactly "I do not know the answer to the
  question.".
- **Reads:** all 735 denial rows were read on NDIF at blocks 19, 20 and 40 (`hpkr_read`, 22
  requests). Block 40 is the depth of Llama's layer 40. K's 240 rows were read at block 40 too.
- **Line endings:** the raw files carry CRLF where the released rows have LF, so 61 rebuilt lies
  first matched no released row.
  - The rebuild now turns them back to LF.
  - A rebuilt row the benchmark released takes the released text exactly; two lies keep a genuine
    CR there.
  - All 571 lies and 49 honest denials are released rows. They carry the released label, and they
    read as their pooled vectors do (cosine ≥ 0.99997 at blocks 19 and 20).
  - The first, CRLF reads are kept as `superseded_crlf_*`. They moved no AUROC by more than 0.021.

| Block | Slice | Lies / honest | K [95% CI] | Random 97.5% | Share of random ≥ K | Base probe | K − probe [95% CI] |
|---|---|---|---|---|---|---|---|
| 19 | denials, whole reply | 571 / 164 | 0.585 [0.53, 0.63] | 0.583 | 0.022 | 0.487 | +0.098 [+0.02, +0.18] |
| 19 | identical reply | 435 / 128 | 0.576 [0.52, 0.63] | 0.579 | 0.034 | 0.524 | +0.052 [−0.03, +0.14] |
| 20 | denials, whole reply | 571 / 164 | 0.582 [0.53, 0.63] | 0.576 | 0.016 | 0.502 | +0.079 [−0.00, +0.17] |
| 20 | identical reply | 435 / 128 | 0.577 [0.52, 0.63] | 0.580 | 0.035 | 0.533 | +0.044 [−0.04, +0.13] |
| 40 | denials, whole reply | 571 / 164 | 0.613 [0.56, 0.66] | 0.581 | 0.003 | 0.533 | +0.080 [+0.00, +0.16] |
| 40 | identical reply | 435 / 128 | 0.602 [0.55, 0.66] | 0.577 | 0.005 | 0.542 | +0.060 [−0.02, +0.14] |

Over the opening sentence alone (474 / 140), K is 0.566 at blocks 19 and 20 (8% to 9% of random
directions as high) and 0.598 at block 40 (1.5%). The base probe there is 0.524 to 0.551.

The block-40 base probe was trained for this comparison (2026-10-12), with the runner's own code
as at blocks 19 and 20: `lb_models.py probe --model qwen --block 40`, which wrote
`XXXX-9-qwen-all-B40`. It is fitted on the per-token activations of RepE's 245 honest and 245
deceptive training rows. The released datasets were then pooled at block 40 too; see the next
section.

### Qwen block 40, every dataset

All of Qwen's released datasets except Alpaca were pooled at block 40 with the code behind the
block-19 and 20 reads (2026-10-12, about 1.5 hours on NDIF):

    layer_sweep.py pool --layers '[40]' --config XXXX-9-qwen-all.yaml --datasets '[RepE, RepE val, HP-C, HP-KR, ID]'

The pooled means are in `activations/qwen-72b/pooled_B40`. Alpaca was left out at the user's
request, so block 40 has no 1% FPR threshold on Alpaca. `score` and `compare` give AUROCs, bands,
paired differences and in-setting operating points there.

| Set | Lies / honest | K | Base probe | Random 97.5% | K − probe [95% CI] |
|---|---|---|---|---|---|
| HP-C | 374 / 2000 | 0.412 | 0.476 | 0.603 | −0.064 [−0.11, −0.02] |
| HP-KR, all rows | 571 / 2000 | 0.621 | 0.193 | 0.932 | +0.428 [+0.39, +0.47] |
| ID | 2145 / 3015 | 0.604 | 0.826 | 0.652 | −0.222 [−0.24, −0.20] |
| Mean of the three | | 0.545 | 0.498 | | +0.047 [+0.03, +0.07] |
| HP-KR denials, released (539 / 44) | | 0.631 | 0.565 | 0.626 | +0.066 [−0.06, +0.19] |
| HP-KR denials, full rollouts (571 / 164) | | 0.613 | 0.533 | 0.581 | +0.080 [+0.00, +0.16] |
| HP-KR identical reply, full rollouts (435 / 128) | | 0.602 | 0.542 | 0.577 | +0.060 [−0.02, +0.14] |
| RepE (all 550 rows) | 275 / 275 | 0.632 | (trained on it) | 0.664 | |

- **Only two kinds of score clear their random band at block 40.**
  - The base probe on ID (0.83, against 0.65).
  - K on the HP-KR denial slices (0.60 to 0.63).
  - Everything else is inside its band or below chance.
- **The base probe wins the ordinary datasets.** It beats K on ID by 0.22 and on HP-C by 0.06,
  although on HP-C both are below chance.
- **K's wins over the probe come only from HP-KR.**
  - The +0.43 on all HP-KR rows is the probe inverting (0.19: it scores honest claims of
    knowledge as more deceptive than the denying lies). K's 0.62 there is inside the band.
  - On the denial slices, where wording is held fixed, K leads by 0.06 to 0.08. That is
    significant only on the full rollouts.
  - So the mean of the three (+0.05) is HP-KR's inverted probe. Without HP-KR, the probe is ahead.
- **Across depth, the probe changes and K hardly does.**
  - The probe gains on ID with depth (0.58 at block 20, 0.83 at 40) and inverts on HP-KR (0.50,
    then 0.19).
  - K is flat: ID 0.60 and 0.60, HP-KR denials 0.62 and 0.63 (released) or 0.58 and 0.61 (full).
- **Against Llama at L40** (same three sets): K 0.867 against the probe's 0.861 on the mean, +0.006,
  not significant. K wins HP-C by 0.155; the probe wins HP-KR and ID by about 0.07 each.
- **In-setting operating point (1% FPR on the set's own honest rows):** both catch at most 4.4% of
  the lies on HP-C, HP-KR and ID.

Files: `qwen/datasets_n60.csv` and `qwen/slices_n60.csv` (blocks 19, 20 and 40; the 19 and 20 rows
are unchanged), and `compare_base_probe_qwen.csv` (Qwen 19, 20 and 40; Llama 20 and 40).

- **K separates Qwen's lies from its honest denials, weakly.**
  - It scores about 0.58 at blocks 19 and 20, against random 97.5% points of 0.58. Only 2% to 4%
    of random directions are as high.
  - It is clearer at block 40: 0.60 to 0.61, with 0.3% to 1.2% of random directions as high.
  - The identical-reply slice agrees with the whole-reply slice, so it is not wording.
- **The base probe is at chance on these rows at every block** (0.49 to 0.55), always inside the
  random band.
  - K beats it on the whole-reply denials at blocks 19 (+0.10) and 40 (+0.08). At block 20 the
    interval touches 0.
  - On identical text K leads by +0.04 to +0.06, which is not significant at any block.
  - On Llama the same probe recipe at L40 reaches 0.59 on denials and 0.78 on identical text.
- **The effect is smaller than Mistral's and Llama's.**
  - On identical text Qwen gets about 0.58 to 0.60, against Mistral's 0.67 to 0.71 at its
    Mean-Probe block and Llama's 0.79 at L40.
  - Llama's K goes from inverted at L20 (0.41 on denials) to 0.73 at L40. Qwen's rises only from
    0.58 to 0.61.
- **K at block 40 is a different direction from K at block 20** (cosine 0.27). Blocks 19 and 20
  agree (0.94).
- **The released-set estimate fits.** It was 0.62 on 539 against 44 denials, with interval
  [0.54, 0.70].
- The classes are disjoint question sets, as on every model.

Run with `hpkr_build --model qwen`, `read --model qwen --per_side 60 --blocks '[40]'`,
`hpkr_read --model qwen` and `hpkr_score --model qwen`.
- **Files:** `k_models/qwen/hpkr/` (`rows.json`, `parts_B19-20-40/`, `directions_n60.npz`,
  `scores.csv`, `superseded_crlf_*`) and `k_models/qwen/parts_answer_B40/`.

Files: `k_models/qwen/` (`verify.json`, `items.json`, `check_reread.json`, `parts_answer/`,
`directions_n60.npz`, `datasets_n60.csv`, `slices_n60.csv`, `split_half_n60.csv`) and
`compare_base_probe_qwen.csv`. Not checked independently.

## Verification (Mistral)

Three independent checks (workflow `verify-k-mistral`, 2026-10-10). None edited the repo or called
NDIF. The Gemma numbers have not been checked independently.

**Recomputation.** Every reported number reproduces from the raw files, by code that does not
import `k_models.py`:
- K itself has cosine 1.0 with the saved direction.
- Every AUROC, band, count and operating point matches within 0.002.
- **Block labels:** the K rows read at L9 match `pooled_B9` and not `pooled_B10`. Their mean for
  "I do not know the answer." has cosine 0.959 with HP-KR's B9 rows of that text, against 0.778
  with B10.

**Code audit.** No bugs. The detection mask covers exactly the forced sentence on all 240 rows.
- **Deviations from Llama's recipe**, none affecting a result:
  - the knowledge check by letter logits rather than generation;
  - 1,000 rather than 2,000 random directions;
  - identical-text slices chosen by a count rule (every sentence with at least 50 rows per class)
    rather than Llama's single sentence.
- **A stricter null:** random contrasts of K's own items, with K projected out, reach at most
  0.636 on the block-9 denials, against K's 0.688.

**Confounds on the denial slice.**
- **Surface features alone score 0.49 to 0.54.** That covers prompt, question and option length
  (characters or tokens), the pressure prefix, and the denial sentence (0.50 out of fold).
- **Subject is untestable.** Every Mistral HP-KR row is wmdp-bio.
- **K holds or rises within strata.** In each prompt-length quartile it is 0.71 to 0.76 at block 9
  and 0.75 to 0.80 at block 10.
- **Residualised on every surface feature**, K is 0.704 at block 9 and 0.730 at block 10. The same
  residualisation leaves random directions at 0.61 (97.5%).
- **Against a surface-only model:** a logistic regression on the surface features reaches 0.61,
  cross-validated with folds grouped by question. Adding K gives χ²(1) = 232 at block 9.

**What stays open: question identity.**
- Lies and honest denials are disjoint sets of questions: 461 against 587 question stems, none
  shared. Belief is one label per question.
- A TF-IDF model on the question text, with folds grouped by question, reaches 0.70. The clearest
  content cue is that quantitative questions are the ones Mistral has no consistent answer to:
  fewer digits marks a lie (0.616).
- Residualising K on the surface features plus that text model's score leaves 0.642 at block 9
  and 0.668 at block 10. That is still above the random band.
- So K is not a length or template artefact. But HP-KR cannot separate what the model knows from
  which questions these are.
- **The same holds for Llama's slices.** Its identical-text slice (78 lies against 112 honest) and
  its denial slice share no question between the classes either.

## Files

`k_models/mistral/`:
- `verify.json` and `items.json` (240 rows, ranked);
- `check_reread.json`;
- `parts_answer/` (blocks 9 and 10);
- `directions_n{20,60}.npz`;
- `datasets_n{20,60}.csv`, `slices_n{20,60}.csv` and `split_half_n{20,60}.csv`.

The verification's scratch scripts are in the session scratchpad (`kverify/`, `kaudit/`,
`k_surface/`) and are not kept.
