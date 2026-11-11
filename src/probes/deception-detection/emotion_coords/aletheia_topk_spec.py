"""Top-k emotion PCs on Aletheia's Quest: the study's preregistered emotion-PC arm (the first k PCs of the emotion
vectors, k by inner CV) in place of all 170. Asked by the user after the all-170 results were seen (2026-10-30), and
fixed before any top-k probe is scored (ANALYSIS_PLAN.md, "Top-k emotion PCs on Aletheia's Quest").
`aletheia_topk.score` refuses to run unless this file's SHA-256 is the one recorded in FREEZE.

Everything not named here is `aletheia170_spec`'s: the bases (the saved emotion and shuffled bases of each model's
middle block), the training sources and their inner CV, the reads (each test row by its organism), the 8 datasets,
the bootstrap draws and the random band.
"""

from pathlib import Path

KS = (2, 3, 5, 10, 20)  # `probes.KS`: k and C chosen together by the source's inner CV; every k is also reported
FEATURE_SETS = {
    "emotion_pc": "the first k PCs of the emotion vectors (the saved basis)",
    "topic_pc": "the first k PCs of the topic vectors (`vectors.topic_basis` on the rank-0 stories, same denoising)",
    "shuffled_pc": "the first k PCs of each shuffled basis (the saved 5)",
    "random": "the first k rows of 20 random 20-d orthonormal subspaces, seeds RANDOM_SEED + s (phase 3's)",
}
N_RANDOM = 20
RANDOM_SEED = 10_000
SOURCES = ("LB", "RepE", "dev")  # LB primary

# P1's rule (ANALYSIS_PLAN.md), on the 8 sets: k_e is the emotion basis's inner-CV choice for each model and source
PREDICTIONS = {
    "TK1": "emotion_pc at k_e (LB, organism read) minus random at k_e (mean of the 20 seeds) >= 0.03 in mean AUROC "
           "over the 8 sets, paired 95% CI excluding 0; and minus topic_pc at k_e likewise. Both must hold",
}
REPORTED = ("every fixed k of each arm", "emotion_pc (k_e) minus shuffled_pc (k_e), raw (LB), all-170 (LB) and the "
            "thesis's headline", "rank of 31", "the RepE and dev sources")
GATE = "emotion_pc (LB, k_e) on the base model's instructed-deception dev rows above the random band's 97.5th percentile"

FREEZE = Path("E:/aletheia/logs/freeze_emotion_topk.txt")
