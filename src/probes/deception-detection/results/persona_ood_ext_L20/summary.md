# Extended persona set: fixing HP-KR at the basis level

Uses `persona including knowlede denial/` — 36 personas, the original 26 plus **10 built around
denial of knowledge rather than assertion of falsehood**. Run at layers 20 and 40
(`../persona_ood_ext_L40/`), against the same pooled activations as every earlier run.

## Why these personas

HP-KR inverted under every method tried: 0.109 for the RepE probe, 0.124 in the Liars' Bench
paper, 0.103 for the persona axis. Reading its transcripts showed why — **its deceptive response
is a denial** ("I do not know the answer", 87% of deceptive rows) while its **honest response is
an assertion** ("Yes." / "I know the answer", 87% of honest rows). Every original deception
persona asserts and manages narrative (handler, spin_doctor, impression_manager), so the axis
encodes *confident assertion = deceptive* and HP-KR runs exactly opposite to it.

The new roles supply the missing contrast:

| knowledge-denial (deceptive) | n | knowledge-candour (honest) | n |
|---|---|---|---|
| `pc_amnesiac_witness` | 129 | `pc_candid_witness` | 64 |
| `pc_deniable_expert` | 72 | `pc_forthright_expert` | 80 |
| `pc_feigned_novice` | 137 | `pc_honest_novice` | 110 |
| `pc_stonewaller` | 129 | `pc_plain_dealer` | 76 |
| | | `pc_calibrated_reporter` | 83 |
| | | `pc_scrupulous_witness` | 74 |

`feigned_novice` against `honest_novice` is the sharpest pair: pretending not to know versus
genuinely not knowing — precisely the distinction HP-KR turns on. Groups were assigned by name;
the files carry no group metadata.

## It works, and the mechanism was the right one

Zero-shot AUROC on HP-KR, no Liars' Bench labels involved:

| layer | deception axis (original 8v7) | **knowledge-denial axis (4v6)** |
|---|---|---|
| 20 | 0.148 *(inverted)* | **0.890** |
| 40 | 0.593 | **0.919** |

The inversion is a layer-20 phenomenon: at layer 40 the published grouping sits just above chance,
so there the new personas are a large gain rather than a sign correction. *(Corrected: an earlier
version of this table gave 0.083 and 0.914 at layer 40, which match no arm. The values above are
`auroc_axis` and `auroc_denial_axis` in `../persona_ood_ext_L40/zero_shot.csv`.)*

## Transfer: the extended axis is the best arm at both depths

Mean off-diagonal AUROC:

| arm | L20 all six | L20 excl. HP-KR | L40 all six | L40 excl. HP-KR |
|---|---|---|---|---|
| baseline (8192-d) | 0.6478 | 0.7282 | 0.7320 | 0.7581 |
| deception axis (original) | 0.6511 | 0.9194 | 0.8319 | 0.8798 |
| **extended axis (12v13)** | **0.9041** | 0.9086 | **0.9158** | 0.9165 |
| knowledge-denial axis (4v6) | 0.8660 | 0.8613 | 0.9115 | 0.9101 |
| extended PCs (3-d) | 0.6837 | 0.9000 | 0.8655 | 0.8670 |
| random directions | 0.5269 | 0.5598 | 0.5038 | 0.5437 |

**The gap between "all six" and "excluding HP-KR" closes.** For the original axis at layer 20 it
is 0.651 vs 0.919 — a 0.27 penalty paid entirely to one dataset. For the extended axis it is
0.904 vs 0.909. HP-KR stops being pathological, which is the whole point.

Per test set at layer 20, mean off-diagonal:

| test set | baseline | deception axis | **extended axis** |
|---|---|---|---|
| CG | 0.663 | 0.786 | **0.969** |
| HP-C | 0.535 | 0.622 | 0.639 |
| **HP-KR** | 0.394 | **0.103** | **0.882** |
| IT | 0.799 | 0.792 | **0.990** |
| ID | 0.778 | 0.776 | **0.954** |
| RepE | 0.718 | 0.795 | **0.991** |

The extended axis beats both the baseline and the original axis on **every** dataset — it is not
trading HP-KR against the rest. HP-C stays the weakest (0.639 at L20, 0.712 at L40), so the
harm-pressure-choice problem is separate and still open.

## The confound, and why the result survives it

This folder is **not a superset**: all 26 original personas were re-extracted with more rollouts
(`default` 100→150, `pc_handler` 64→106, `pc_plausible_denier` 26→64), so vectors moved by up to
0.78 against a typical norm of ~4.1. A naive comparison would confound "more rollouts" with "more
personas".

Running the *original* 8v7 groups from *this* folder separates them. The re-run alone moves the
original arms by ±0.05 at most (`d_rerun` column in the comparison), and moves HP-KR zero-shot
from 0.109 to 0.148 — still inverted. The jump to 0.890 comes from the new personas.

## Caveats

- **Group assignment is by name.** The files carry no group metadata, so the 4/6 split is a
  reading of the role names. A different split would give a different axis.
- **Asymmetric groups** (4 deceptive vs 6 honest) — deliberate on the author's part as far as the
  pairings suggest, but it makes the denial axis's midpoint unbalanced.
- **1-d beats 3-d again, more strongly.** `ext_axis1` 0.904 against `ext_pc3` 0.684 at layer 20.
  The pattern from every earlier run holds: extra components hurt.
- **HP-C is untouched.** Its honest and deceptive responses both score high on every axis; that
  failure has a different cause and these personas do not address it.
