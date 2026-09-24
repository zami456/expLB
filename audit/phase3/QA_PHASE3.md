# Phase 3 — lead QA record

Each workstream nominated the 2 findings it was least sure of. The lead re-derives them
**independently from raw files**, with its own script, not following the agent's route. Same rule as
waves 1–5, where it twice overturned the audit's own work.

Priority order here was by **how much the finding would change if wrong**, not by batch order.

---

## P3-D · passP — CONFIRMED to the digit, but the scope is narrower than stated

**The finding.** `passP` is the prompt-side construct-validity control: a probe trained on the
*prompt only*, with no model response. P3-D reports it recovers 0.60–1.09 of the response-side
transfer signal (mean 0.86), and at L40/paper it is *higher* than the response-side probe.
If that holds generally, it is a serious construct-validity problem: the transfer being reported
would be partly a property of the questions, not of deception in the answers.

**Re-derived** from `audit/phase2/layer_sensitivity.csv` (store `passP` vs store `<root>`, raw
feature, seed 42, `dataset_set=all`), as a ratio of above-chance signal, (passP − 0.5)/(root − 0.5):

```
pool     L    probe  passP off  root off   ratio     passP diag  root diag
last    20    paper     0.5593    0.6312   0.452       0.8995    0.9558
last    20 lr_reg10     0.5645    0.6204   0.536       0.8809    0.9505
last    20      mms     0.5597    0.6787   0.334       0.8612    0.8956
last    40    paper     0.6475    0.6359   1.085       0.9281    0.9671
last    40 lr_reg10     0.5887    0.6253   0.708       0.9213    0.9675
last    40      mms     0.6537    0.6590   0.967       0.9122    0.9198
last    44    paper     0.6219    0.6268   0.962       0.9285    0.9644
last    44 lr_reg10     0.5597    0.5992   0.602       0.9245    0.9634
last    44      mms     0.6382    0.6708   0.809       0.9146    0.9163
mean  (all 9 cells)                        0.179 – 0.509,  mean 0.407
```

**Every number P3-D quotes is exact**, including the L40/paper inversion (0.6475 vs 0.6359).

**The correction — its "0.60–1.09, mean 0.86" is 6 of 18 stored cells.** Those six are exactly
`last` pooling at L40 and L44 (min 0.602, max 1.085, mean 0.8555 — reproducing the quoted range to
3 dp). Outside them the claim does not hold:

| slice | ratio range | mean |
|---|---|---|
| `last`, L40+L44 (the quoted set) | 0.602 – 1.085 | **0.856** |
| `last`, L20 | 0.334 – 0.536 | 0.441 |
| `mean`, all layers | 0.179 – 0.509 | **0.407** |

**So the honest statement is: under `last` pooling at L40/L44 a prompt-only probe recovers most of
the transfer signal; under `mean` pooling it recovers about 40% at every layer.** P3-D flagged the
pooling dependence itself, which is why this was nominated — the fix is to put the scope in the
sentence, not to drop the finding.

**What survives unscoped:** the *diagonal*. passP's in-distribution AUROC is 0.83–0.93 against the
root store's 0.90–0.97, at **both** poolings. "The prompt alone is highly learnable in-distribution"
needs no caveat; only the transfer claim does.

**Verdict: HOLDS, scope corrected.** Importance **HIGH** — this is a pre-registered control
(`8_coords_ndif.py:270-278`) that was run and never reported, and even the scoped version is worth a
section. Seed 42 only, as P3-D said.

---

## P3-C · W1 significance census — CONFIRMED exactly, with two corrections that make it STRONGER

**Re-derived** independently over the same 10 cached `run-crossgen/**/significance.json` files,
65,388 tests, with my own reader:

| quantity | P3-C | lead | |
|---|---|---|---|
| total tests | 65,388 | **65,388** | ✓ |
| clear BH | 0.25% | **166 = 0.25%** | ✓ |
| median 95% half-width | ±0.125 | **±0.1252** | ✓ |
| persona-vs-raw tests | 16,548 | **16,548** | ✓ |
| … positive sign | 17.6% | **2,913 = 17.6%** | ✓ |
| at L20/40/44, paper, mean | 276 tests, 0 clear | **276 tests, 0 clear** | ✓ |
| … median half-width | 0.133 | **0.1325** | ✓ |

**Correction 1 — "median effect 0.044" is a median ABSOLUTE effect, and saying it plainly makes the
result stronger.** At the reported operating points:

```
median |diff|        0.0440      <- the quoted figure
median SIGNED diff  -0.0263
mean   SIGNED diff  -0.0169
positive sign       95 of 276 = 34.4%
```

Quoted as "median effect 0.044 vs half-width 0.133" it reads as *a real advantage swamped by noise*.
The data says something sharper: **the median persona-vs-raw effect at the reported operating points
is negative, and only a third of the tests are even positive.** It is not an effect too small to
resolve — it is typically no effect or a negative one, and unresolvable either way.

*This is the third time in this audit that a median-absolute has been quoted as if signed* (F-73's
median |gap| in wave 3, DF-49's metric confusion in wave 5, now this). Worth a standing rule.

**Correction 2 — 5 tests clear, not 4.** P3-C's "exactly 4, all `pc:kd_assertion`" counts only the
positive ones. The fifth is **`pc:deception − raw`, own100 bank, L20/last/mms, diff −0.1524,
q 0.0474** — a persona feature significantly **worse** than raw. Excluding it from the count hides a
result that runs the same way as the rest of the finding.

```
pc:deception    - raw   L20 last mms      -0.1524  q 0.0474  own100    <- negative, omitted by P3-C
pc:kd_assertion - raw   L68 mean paper    +0.1497  q 0.0467  kd
pc:kd_assertion - raw   L40 mean lr_reg10 +0.1624  q 0.0347  kd_s23
pc:kd_assertion - raw   L40 last lr_reg10 +0.1771  q 0.0411  kd_s23
pc:kd_assertion - raw   L40 mean lr_reg10 +0.1596  q 0.0470  kd_s23
```

All four positives are `pc:kd_assertion` — the owner's own bank — and three are under `lr_reg10`,
not the paper probe. P3-C is right about that.

**Scope caveat, which P3-C stated honestly and I confirm:** this census is the ten **cached crossgen**
files only. The cap, scale and premise significance files are lab-only and are NOT in it, so F-66's
arm sits outside the "4 of 16,548". Any quotation must say "in the crossgen family".

**Verdict: HOLDS, two corrections applied, both strengthening.** Importance **HIGH** — this is the
strongest single statement Phase 3 produced.

---

## P3-A · D-107 — the flag is right, and the fix runs in the collaborator's favour

P3-A reported that `audit/claims.csv` row D-107 calls his random control "UNDERPOWERED (n=1 draw)".
**Verified from his source on `origin/feat/persona-pc-ood`:**

```python
N_RANDOM_DRAWS = 5                                                    # persona_ood.py:61
# random control, averaged over draws so a single lucky basis cannot carry the comparison
for _ in range(N_RANDOM_DRAWS):                                       # :1006
    dirs = random_directions(hidden, 3, rng)
    random_mats.append(transfer_matrix(...))
matrices["random3"] = sum(random_mats) / len(random_mats)             # :1009
```

It is the **mean of 5 seeded draws**, with a comment stating exactly why. `PROGRESS.md` already
records this being overturned during the P8 lead QA, and `docs/DRAFT_CORRECTIONS.md` §4 states it
correctly — **only the claims.csv row was never updated.**

**Corrected** by `audit/scripts/phase3/p3_fix_d107.py` (idempotent, re-runnable). The narrower
criticism is preserved in the row: he reports a *mean*, not a percentile of a null distribution, so
it bounds the centre and not the tail — the owner's own 1,000-direction null does more.

---

## Accepted without independent re-derivation, with the reason

The remaining seven nominations were accepted as the agents wrote them, each **already carrying its
own honest caveat**, and none load-bearing enough to outrank the three above. Recorded so the gap is
visible rather than implied:

| nomination | why accepted as-is |
|---|---|
| P3-E, collaborator start ~1 h after the Drive export | the agent tagged the causal link INFERRED and the dates VERIFIED; the timeline does not depend on the causal reading |
| P3-E, bf16→fp32 "expensive in time, not results" | independently supported by `audit/phase1/axes_old_vs_fp32.txt` (cosine 1.0000 at every layer, ≤0.9% norm); P3-C reached the same conclusion by a different route |
| P3-A, A-1's force (paper's deception setting is k=1) | the numbers and the k=1 reading are solid; whether it is the right comparator is a judgement for the owner, not a measurement |
| P3-A, B-1's coupling (her arms on the weaker s23 bank) | tagged INFERRED by the agent; P3-B's D-01 measured the same coupling directly and found it small (\|gap\| 0.0015 vs his control) |
| P3-B, D-08 insider-trading | already tagged UNCLEAR/INFERRED; one dataset, one layer, ≤0.069 |
| P3-B, D-10 token position | her side is explicitly DOC-ONLY in the writeup; correctly labelled |
| P3-C, W2's interpretation | the arithmetic is VERIFIED; the causal reading is flagged MEDIUM with no p-value, which is the honest label |

**Two cross-file conflicts P3-E and P3-D raised, neither resolved, both logged:**
- `audit/SESSION_MINING.md:7` says **42** ledger entries are print-only; `audit/phase1/sessions/findings_targets.csv` has **72** rows. Both cited in audit files; neither re-derived. → Phase 4.
- P3-D reports that seeds 0–4 carry only 3 features / 2 probes in `passA`, `passC` and `<root>`, so
  **any 6-seed mean over those stores is invalid**. If true this touches earlier layer work. → Phase 4,
  and added to the traps list meanwhile.
