"""What the all-170 emotion arm's test on Aletheia's Quest measures, fixed before any story is read through
Qwen3.5-27B or Gemma-3-27B-it (ANALYSIS_PLAN.md, "Aletheia's Quest transfer"). `aletheia170.score` refuses to run
unless this file's SHA-256 is the one recorded in FREEZE.

The arm (exploratory on Llama-3.3-70B, RESULTS.md): L2 logistic regression on the coordinates of a pooled read in
all principal directions of the 171 centred, denoised emotion vectors. Here the vectors are built inside each
organism's base model at its middle block, as the thesis builds D9 from the base model's persona bank, and the probe
is scored as the thesis scored its detectors on the competition's test sets (`scripts/aletheia/spec.py`): each row
read by the organism that wrote it, at the middle block, reasoning dropped, the final reply's content pooled; AUROC
per dataset (lie = positive, no sign flipped), the mean over the datasets scored, and its rank among the 30 published
entries over the same datasets.
"""

from pathlib import Path

# --- models, blocks, datasets -------------------------------------------------------------------------------------
MODELS = ("qwen35", "gemma")
BLOCK = {"qwen35": 32, "gemma": 31}  # round(40 / 80 * n_blocks): the thesis's middle block, `spec.MODELS[m].mid`
SCORED = ("Echo", "Harm", "GS-F", "GS-M", "BotC", "AB-AW", "AB-HC", "AB-SP")  # the thesis's 8; HG-G, HG-N not run

# --- the emotion basis (phase1.final's recipe, unchanged) ---------------------------------------------------------
STORY_ROWS = "stories"  # Llama's rows file: the same 8,280 Gemotions stories (171 emotions x 40 topics, rank 0,
#                         and ranks 1-2 for the emotions in data/emotion_coords/topups.json), re-tokenised per model
TOPUP_RANKS = (1, 2)
POOL_FROM = 50  # mean over positions 50 onward of the model's own tokenisation (add_special_tokens=True)
NEUTRAL_VARIANCE = 0.5  # denoise: the neutral dialogues' top PCs explaining this share of their variance
N_SHUFFLED = 5  # emotion labels permuted within each topic (`vectors.shuffled_bases`, seed `seeds.stories`)

# --- probes --------------------------------------------------------------------------------------------------------
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)  # `probes.C_GRID`; C by inner CV on the training rows only
FEATURE_SETS = (
    "emotion_all",  # coordinates in all r PCs of the emotion vectors (r = 170)
    "raw",  # every dimension of the read
    "shuffled_all",  # all r PCs of each shuffled basis (N_SHUFFLED)
    "random_all",  # r random orthonormal directions, seeds RANDOM_SEED + s (N_RANDOM)
    "alpaca_all",  # the top r PCs of the base model's Alpaca control
    "trainpca_all",  # the top r PCs of the training rows (min(r, rows - 1)), fitted on the training rows
)
N_RANDOM = 5
RANDOM_SEED = 30_000
SOURCES = {  # training rows: (role, inner CV)
    "LB": ("primary", "leave one Liars' Bench set out: Gemma's own CG, IT, ID at B31; Qwen2.5's HP-C, HP-KR, ID text "
                      "read through Qwen3.5 base at B32 (D4 (LB)'s rows, posthoc.read_lb)"),
    "RepE": ("secondary", "5-fold grouped by statement: R1's training rows (Gemma: the pooled train split; Qwen3.5: "
                          "repe.py's first 80%, each row's span tokens averaged)"),
    "dev": ("secondary", "leave one organism out (Qwen3.5); stratified 5-fold (Gemma's single dev set): R2's rows, "
                         "each dev set read by its organism"),
}
READS = {"organism": "primary", "base": "secondary: the LB-trained probes on the base model's reads of the test rows"}
THESIS_CELLS = ("headline", "R1-repe", "R2-dev-probe", "D4 (LB)")  # re-scored on the same draws for paired deltas

# --- statistics (as `aletheia/score.run`) -----------------------------------------------------------------------
SEED = 42  # `spec.SEED`: bootstrap draws default_rng([SEED, dataset index]); random band default_rng([SEED, 7, index])
N_BOOT = 1000
N_NULL = 1000

# --- declared predictions (exploratory arm, its first out-of-model test) -------------------------------------------
PREDICTIONS = {
    "H1": "emotion_all (LB, organism read): mean AUROC over SCORED above raw (LB), paired bootstrap 95% CI of the "
          "difference above 0",
    "H2": "emotion_all (LB, organism read) above each 170-d control (shuffled_all mean, random_all mean, alpaca_all, "
          "trainpca_all), each paired 95% CI above 0",
}
REPORTED = ("rank of 31 and its bootstrap interval", "deltas to the thesis cells", "the secondary sources and read")

# --- gates: a failure stops the run for the user's decision ---------------------------------------------------------
PARITY_MIN_COS = 0.9998  # reader parity: 20 stored rows per model re-read (Qwen3.5 Alpaca rows, Gemma dev rows)
VALENCE_MIN_ABS_R = 0.5  # |r(PC1, valence)| of each new basis, Llama's stop rule
DEV_GATE = "emotion_all (LB) on the base model's instructed-deception dev rows above the 97.5th percentile of " \
           "N_NULL random unit directions (`aletheia/score.dev_gate`)"

FREEZE = Path("E:/aletheia/logs/freeze_emotion170.txt")
