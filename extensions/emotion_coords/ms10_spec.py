"""Emotion coordinates on the ten datasets of Mahadik & Skapars (arXiv 2605.09391, "Do Linear Probes Generalize Better
in Persona Coordinates?"), on Llama-3.3-70B at blocks 35-45: the user's request of 2026-10-31, with their choices
(sycophancy added; the paper's splits; the band 35-45). Fixed before any read of the new sets (ANALYSIS_PLAN.md,
"The paper's ten datasets"). `ms10.score` refuses to run unless this file's SHA-256, and those of the `sources.json`
and `splits.csv` that `ms10.sets` writes, are the ones recorded in FREEZE.

The paper probes Llama-3.2-3B and Llama-3-8B in a 3-d persona space against raw activations, over a train-one-test-all
matrix of these ten sets. Here the space is the emotion basis (phase 1's recipe, Llama's band bases), with the
paper's own persona coordinates as reference arms. The sycophancy sets ship labelled replies, so they are read as
written (no generation, no judge); the deception sets are Llama-3.3-70B's own rollouts.
"""

from pathlib import Path

BLOCKS = tuple(range(35, 46))  # Llama blocks 35-45; bases `data/emotion_coords/bases/full_L<b>.npz`
DECEPTION = ("AILiar", "Roleplaying", "ConvGame", "InstrDec", "HarmPressureChoice")
SYCOPHANCY = ("Sycophancy", "OpenEndedSycophancy", "OEQValidation", "OEQIndirectness", "OEQFraming")
SETS = DECEPTION + SYCOPHANCY

# label 1 = deceptive / sycophantic. AILiar and Roleplaying by the repo's own rules: AILiar's graded honest or
# deceptive (refusal and unknown dropped), Roleplaying's grade >= 6 deceptive and <= 2 honest (3-5 dropped).
# The Liars' Bench sets are the study's rows at every band block (`lbdata.full_subset`) and their pooled reads.
SOURCES = {
    "AILiar": {"rollouts": "data/rollouts/ai_liar__original_without_answers__llama-70b-3.3.json",
               "dataset_class": "AILiarDataset"},
    "Roleplaying": {"rollouts": "data/rollouts/roleplaying__plain__llama-70b-3.3.json",
                    "dataset_class": "RolePlayingDataset"},
    "ConvGame": {"lb": "convince_me__plain"},
    "InstrDec": {"lb": "instruct_dishonesty__plain"},
    "HarmPressureChoice": {"lb": "harm_refusal_choice__plain"},
    "Sycophancy": {"hf": "arianaazarbal/sycophancy_dataset", "file": "data/train-00000-of-00001.parquet",
                   "replies": ("sycophantic", "not_sycophantic")},  # Azarbal (2024)
    "OpenEndedSycophancy": {"hf": "henrypapadatos/Open-ended_sycophancy", "file": "Open-ended_sycophancy_dataset.csv",
                            "replies": ("sycophantic", "not_sycophantic")},  # Papadatos (2025); Rimsky et al. (2024)
    "OEQValidation": {"hf": "Prasadmahadik/oeq-prepared-oeq-validation_human-balanced-200",
                      "file": "data/train-00000-of-00001.parquet", "label": "validation_human"},  # Cheng et al. (2025)
    "OEQIndirectness": {"hf": "Prasadmahadik/oeq-prepared-oeq-indirectness_human-balanced-200",
                        "file": "data/train-00000-of-00001.parquet", "label": "indirectness_human"},
    "OEQFraming": {"hf": "Prasadmahadik/oeq-prepared-oeq-framing_human-balanced-200",
                   "file": "data/train-00000-of-00001.parquet", "label": "framing_human"},
}
OEQ = ("OEQValidation", "OEQIndirectness", "OEQFraming")
CODES = {"AILiar": 21, "Roleplaying": 22, "Sycophancy": 23, "OpenEndedSycophancy": 24, "OEQ": 25}  # read id = code * 1e6 + n
SYSTEM_PROMPT = None  # the sycophancy dialogues are user -> assistant; the template adds its own date block
MASK_PADDING = {"left": 0, "right": 0}  # the reply's content tokens, as the Liars' Bench sets (`XXXX_9.py`)

# the paper's splits: at most CAP rows a set; a seeded stratified 75/25 train/test split. A group is the hash of
# everything before the scored reply, so rows sharing a prompt (AILiar's five completions, the paired sycophancy
# replies) stay on one side. A set over CAP: Liars' Bench at the full set's label ratio (from the rows read at
# every band block), Sycophancy as CAP / 2 seeded prompts with both replies. The OEQ sets share some records, so
# they are split once, jointly, over their distinct texts, stratified on the triple of human labels.
CAP = 800
N_SPLITS = 4  # StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED); fold 0 is the test split
SEED = 0

# arms: the all-170 arm and its 170-d controls exactly as the band study (`aletheia_band._feature_sets`), and the
# paper's k = 3 as reference arms (emotion PCs; its deception space, 8 v 7 personas; its sycophancy space, 5 v 5)
FEATURE_SETS = ("emotion_all", "raw", "shuffled_all", "random_all", "alpaca_all", "trainpca_all")
REFERENCE_SETS = ("emotion_pc3", "persona_dec3", "persona_syc3")
CONTROLS = ("shuffled_all", "random_all", "alpaca_all", "trainpca_all")  # seeds averaged per bootstrap draw
N_SHUFFLED = 5
N_RANDOM = 5
RANDOM_SEED = 30_000
K_REFERENCE = 3

# probe: standardised L2 logistic regression (`probes.fit`), C by inner stratified-group CV on the training rows only,
# folded into one direction and offset (`aletheia170._direction`). A training source is one set's train split, or
# all ten train splits pooled ("all").
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
INNER_FOLDS = 5
TRAIN_SOURCES = SETS + ("all",)

# scoring: AUROC of every source on every test split. Bootstrap: N_BOOT stratified draws per test split,
# default_rng([BOOT_SEED, set index]), shared by every arm, source and block (paired comparisons). A region's
# mean is taken per draw.
N_BOOT = 1000
BOOT_SEED = 42
REGIONS = {
    "deception": "train and test both deception sets, off the diagonal (20 cells)",
    "sycophancy": "train and test both sycophancy sets, off the diagonal (20 cells)",
    "cross": "train on one behaviour, test on the other (50 cells)",
    "diagonal": "train and test on the same set (10 cells)",
    "all": "trained on all ten train splits (10 cells)",
}
MIN_BLOCKS = 6  # of the 11: a prediction holds when its paired 95% CI is above 0 at this many blocks

PREDICTIONS = {
    "M1": "all-170 minus raw activations, deception region mean: CI above 0 at >= 6 of the 11 blocks",
    "M2": "all-170 minus each 170-d control (shuffled mean, random mean, Alpaca PCs, training-set PCs), deception "
          "region mean: CI above 0 at >= 6 of the 11 blocks, for every control",
    "M3": "as M1 on the sycophancy region mean",
    "M4": "as M2 on the sycophancy region mean",
}
REPORTED = ("the cross-behaviour, diagonal and train-on-all regions", "the k = 3 reference arms, among them the "
            "paper's claim (its persona coordinates beat raw activations off the diagonal) at 70B",
            "the 10 x 10 matrices at block 40 and averaged over the band", "every arm's region means per block")
KNOWN = ("ConvGame, InstrDec and HarmPressureChoice are Liars' Bench sets the earlier Llama studies used; AILiar, "
         "Roleplaying and the five sycophancy sets were never read before this freeze")

# checks before the bulk reads: every span decodes to its reply exactly; the reader's redeploy check on 20 stored
# Liars' Bench parity rows (`ec_read.py recheck --name lb_parity --n 20`, blocks 20/28/40/52, solo passes)
PARITY_MIN_COS = 0.9998

FREEZE = Path("E:/aletheia/logs/freeze_emotion_ms10.txt")
