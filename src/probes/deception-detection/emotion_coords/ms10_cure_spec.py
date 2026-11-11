"""Can instructed deception be "cured" of its non-deception variance before a probe is trained on it? The user's request
of 2026-10-31: erase what a probe can learn about instructed deception besides the lie, train on the cured rows and
test on the other sets. Fixed before any erasure is scored (ANALYSIS_PLAN.md, "Curing instructed deception").
`ms10_cure.score` refuses to run unless this file's SHA-256 is recorded in FREEZE and `ms10`'s and `ms10_thesis`'s
freezes hold.

The user's own loop (random two-class labels on instructed deception, a probe, LEACE when the probe succeeds, repeat)
is run as a control without its stopping test. A probe cannot predict labels that are independent of the rows on
held-out rows (AUROC 0.5 in expectation), and with 600 rows in 8,192 dimensions it always separates them in-sample,
so the test never discriminates; `score` records both numbers to show it. Each round of LEACE on random labels
erases one random direction of the whitened space: k rounds are the orthogonal complement of k Gram-Schmidt-ordered
whitened mean differences of random splits.

The erasure the question is about removes the training rows' non-deception variance: the top k principal directions
of InstrDec's within-class covariance (each row centred on its own class's mean; nuisance attribute projection).
Everything else is fixed by earlier freezes: `ms10`'s rows, splits, block reads and bootstrap draws, and the thesis's
raw probe D1 (`persona_ood.make_probe`: standardised, L2 with reg 10, no intercept) as `ms10_thesis` fits it, so the
unerased probe is `ms10_thesis`'s D1 trained on InstrDec, and must reproduce its AUROCs exactly.
"""

from pathlib import Path

LAYERS = (36, 40, 44)
SOURCE = "InstrDec"  # its `ms10` train split (600 rows); its test split (200 rows) is the in-distribution check
RANKS = (5, 10, 20, 50, 100, 200, 400)  # directions erased (or, for "keep", kept)
KEEP_REFERENCE = 170  # `ms10`'s training-set-PC arm kept the top 170: the projection that already helped
N_SEEDS = 5  # random directions and random-label LEACE: seeds averaged per bootstrap draw
RANDOM_SEED = 50_000  # random subspaces: `linalg.random_subspace(hidden, k, RANDOM_SEED + seed)`
LABEL_SEED = 60_000  # random-label LEACE: round i of seed s splits the rows by default_rng([LABEL_SEED + s, i])
WHITEN_TOL = 1e-6  # singular values below this share of the largest are dropped when whitening (n < hidden)
VARIANTS = {
    "none": "no erasure (= ms10_thesis's D1 trained on InstrDec)",
    "nuisance": "erase the top k PCs of the within-class covariance (each row centred on its class mean)",
    "total": "erase the top k PCs of the total covariance",
    "random": "erase k random orthonormal directions (5 seeds)",
    "random-label LEACE": "the user's loop: k rounds of LEACE on balanced random two-class splits (5 seeds)",
    "keep": "keep only the top k PCs of the total covariance (their coordinates as features)",
}

# regions (the test splits' AUROC, each set weighted equally): deception transfer = the other four deception sets;
# new deception = AILiar and Roleplaying; sycophancy = the five sycophancy sets; in-distribution = InstrDec's test split
REGIONS = ("deception", "new deception", "sycophancy", "in-distribution")
MIN_LAYERS = 2  # of the 3: a prediction holds when its paired 95% CI is above 0 at this many layers

PREDICTIONS = {
    "C1": "nuisance erasure minus no erasure, deception transfer, averaged over the 7 ranks: CI above 0 at >= 2 of 3 "
          "layers",
    "C2": "nuisance erasure minus random erasure at the same rank, averaged over the ranks: CI above 0 at >= 2 of 3",
    "C3": "nuisance erasure minus the random-label LEACE loop at the same rank, averaged over the ranks: CI above 0 at "
          ">= 2 of 3",
    "C4": "nuisance erasure (averaged over the ranks) minus keeping the top 170 PCs: CI above 0 at >= 2 of 3",
}
REPORTED = ("each region at each rank for every variant", "the new-deception, sycophancy and in-distribution regions",
            "the random-label probe's held-out and in-sample AUROC")
KNOWN = ("D1 trained on InstrDec is already scored on these splits (`ms10_thesis`), as is `ms10`'s training-set-PC "
         "arm (C by inner CV); no erasure has been")

FREEZE = Path("E:/aletheia/logs/freeze_emotion_ms10_cure.txt")
