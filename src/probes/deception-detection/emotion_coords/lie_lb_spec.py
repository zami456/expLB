"""Lie coordinates from RepE on Liars' Bench's five sets: train on one set, test on the other four. The user's request of
2026-10-31, after `ms10_lie`. Fixed before any probe is fitted here (ANALYSIS_PLAN.md, "Lie coordinates on Liars'
Bench"). `lie_lb.score` refuses to run unless this file's SHA-256 is recorded in FREEZE and `ms10_lie`'s freeze holds.

The arms are `ms10_lie`'s, with its lie directions (INLP on RepE alone, k = 100 at every block) and random subspaces,
under `ms10`'s probe (standardised L2 logistic regression folded into one direction). The data and protocol are phase
3's protocol C: the study's capped rows (`lbdata.full_subset`), each probe fitted on all rows of one set with C by a
stratified 5-fold inner CV, scored on every row of the other four. RepE, the lie directions' source, is neither a
training nor a test set. Liars' Bench is the thesis's development data, so this is a re-test on familiar data.
"""

from pathlib import Path

BLOCKS = tuple(range(35, 46))
SETS = ("convince_me__plain", "harm_refusal_choice__plain", "harm_refusal_knowledge_report__plain",
        "insider_trading_cadenza__plain", "instruct_dishonesty__plain")
SHORT = {"convince_me__plain": "CG", "harm_refusal_choice__plain": "HP-C", "harm_refusal_knowledge_report__plain": "HP-KR",
         "insider_trading_cadenza__plain": "IT", "instruct_dishonesty__plain": "ID"}
HPKR = "harm_refusal_knowledge_report__plain"

TRAINED_ARMS = ("raw", "lie", "lie3", "persona_ext3", "persona_dec3", "trainpc_k", "trainpc3", "random_k")
ZERO_SHOT = ("lie_w1", "D9")  # the first lie direction (RepE's own probe) and the thesis's extended contrast
N_SEEDS = 5  # random subspaces, `ms10_lie`'s seeds
K_SMALL = 3

# regions: "all" = the 20 train-one-test-other cells; "excl. HP-KR" = the 12 among the other four sets (the thesis's
# two transfer columns). A zero-shot detector's region is its mean over the region's test sets.
REGIONS = ("all", "excl. HP-KR")
N_BOOT = 1000
BOOT_SEED = 42  # each test set's stratified draws: default_rng([BOOT_SEED, 200 + its index in SETS])
MIN_BLOCKS = 6

PREDICTIONS = {
    "LB1": "lie (k = 100) minus raw activations, all 20 cells: CI above 0 at >= 6 of the 11 blocks",
    "LB2": "lie minus a random subspace of the same dimension (mean of 5 seeds): CI above 0 at >= 6",
    "LB3": "lie minus the training set's own top-k PCs: CI above 0 at >= 6",
    "LB4": "lie3 against D4 (k = 3 each), two-sided: lie coordinates win if the CI is above 0 at >= 6 blocks, persona "
           "coordinates win if it is below 0 at >= 6, otherwise neither",
}
REPORTED = ("the excl. HP-KR region", "the 5 x 5 matrices", "the zero-shot first lie direction and D9",
            "the paper's PCs")
KNOWN = ("`ms10_lie` on the ten datasets: L1 fails, L2 holds, L3 fails, L4 neither. Liars' Bench is the thesis's "
         "development data (D9 + K was selected on it), and phase 3 ran raw activations and the emotion arms under "
         "protocol C on these rows")

FREEZE = Path("E:/aletheia/logs/freeze_emotion_lie_lb.txt")
