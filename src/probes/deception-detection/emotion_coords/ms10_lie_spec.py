"""Lie coordinates: directions learned from labelled lies outside the ten sets, used as probe coordinates on Mahadik &
Skapars' ten datasets, against the persona (D4), emotion, training-set-PC and raw spaces. The user's request of
2026-10-31, after the cure results. Fixed before any lie coordinate is fitted or scored (ANALYSIS_PLAN.md, "Lie
coordinates from RepE"). `ms10_lie.score` refuses to run unless this file's SHA-256 is recorded in FREEZE and `ms10`'s
freeze holds.

The lie directions come from INLP (iterative nullspace projection; Ravfogel et al. 2020) on the RepE instructed-honesty
pairs, the data the thesis's base probe D2 was trained on, which are none of the ten sets:
- each round fits the thesis's probe (`persona_ood.make_probe`) on RepE's train split after projecting out the
  directions found so far, and keeps its direction (projected, normalised) if the probe still finds the lie on
  RepE's validation split (held-out AUROC >= STOP_AUROC); the first round that fails ends the search;
- the validation rows exclude the one statement both halves share, so the held-out test is statement-disjoint;
- k, the number of lie directions, is set this way at each block, on RepE alone.

Everything else is `ms10`'s: rows, splits, block reads, bootstrap draws and its probe for the coordinates (standardised
L2 logistic regression, C by inner stratified-group CV on the training rows, folded into one direction).
"""

from pathlib import Path

BLOCKS = tuple(range(35, 46))  # `ms10`'s band, so every stored `ms10` arm is re-scored at every block
STOP_AUROC = 0.55  # INLP stops at the first round whose held-out AUROC on RepE's validation split is below this
MAX_DIRECTIONS = 100
K_PERSONA = 3  # D4: the extended bank's top 3 PCs (the thesis's design); "lie3" takes the first 3 lie directions
N_SEEDS = 5  # random k-dimensional subspaces
RANDOM_SEED = 70_000

NEW_ARMS = {
    "lie": "the k lie directions as coordinates",
    "lie3": "the first 3 lie directions (matched to D4's k)",
    "persona_ext3": "D4's basis: the extended bank's top 3 PCs, under `ms10`'s probe",
    "trainpc_k": "the training set's own top k PCs (k matched to the lie directions)",
    "trainpc3": "the training set's own top 3 PCs",
    "random_k": "k random orthonormal directions (5 seeds)",
}
ZERO_SHOT = "lie_w1"  # the first lie direction scored directly: RepE's own probe, no training on the ten sets
STORED_ARMS = ("raw", "emotion_all", "emotion_pc3", "trainpca_all", "persona_dec3", "persona_syc3")  # `ms10`'s parts

REGIONS = ("deception", "sycophancy", "new deception")  # `ms10`'s and `ms10_thesis`'s definitions
MIN_BLOCKS = 6  # of the 11: a prediction holds when its paired 95% CI is above 0 at this many blocks

PREDICTIONS = {
    "L1": "lie minus raw activations, deception region: CI above 0 at >= 6 of the 11 blocks",
    "L2": "lie minus a random subspace of the same dimension (mean of 5 seeds), deception region: CI above 0 at >= 6",
    "L3": "lie minus the training set's own top-k PCs, deception region: CI above 0 at >= 6",
    "L4": "lie3 against D4 (k = 3 each), deception region, two-sided: lie coordinates win if the CI is above 0 at >= 6 "
          "blocks, persona coordinates win if it is below 0 at >= 6, otherwise neither",
}
REPORTED = ("the sycophancy and new-deception regions; cross-behaviour, diagonal and train-on-all",
            "k and the INLP trace at each block", "the zero-shot first lie direction",
            "every new arm against the emotion arms and the paper's PCs")
KNOWN = ("`ms10`'s arms (raw, all-170, emotion PCs, the training set's top 170 PCs, the paper's PCs) and the thesis's "
         "D2 and D4 (under the thesis's probe, at 36/40/44) are already scored on these splits; the cure study's "
         "post hoc lead (keeping InstrDec's top 5 PCs) has been seen; no lie coordinate has been fitted")

FREEZE = Path("E:/aletheia/logs/freeze_emotion_ms10_lie.txt")
