"""The emotion-coordinate study on the other Liars' Bench models, as the thesis's cross-model replication tested its
own detectors (`lb_models.py`, `make_xmodel_assets.py`; thesis section 5.6 and Appendix D). The user's request
(2026-10-30) and choices: the thesis's 4 layers and the band; the core study (no MLP, no erasure P6, no HP-KR
diagnostics). Fixed before any read (ANALYSIS_PLAN.md, "Every Liars' Bench model"). `xmodel.summary` refuses to run
unless this file's SHA-256 is the one recorded in FREEZE.

Each model is read through itself: the stories and neutral dialogues for its emotion bases, and its own Liars' Bench
rollouts, Alpaca replies and RepE pairs for the probes. Llama's rows are the study's existing results.
"""

from pathlib import Path

MODELS = {  # keys are `lb_models` keys
    "qwen": {"hf": "Qwen/Qwen2.5-72B-Instruct", "reader": "qwen25", "folder": "qwen-72b", "n_blocks": 80,
             "named": (20, 36, 40, 44), "band": tuple(range(35, 46)), "parity_block": 40},
    "mistral": {"hf": "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "reader": "mistral",
                "folder": "mistral-small-3.1-24b", "n_blocks": 40, "named": (9, 18, 20, 22),
                "band": tuple(range(17, 24)), "parity_block": 20},
    "gemma": {"hf": "google/gemma-3-27b-it", "reader": "gemma", "folder": "gemma-3-27b", "n_blocks": 62,
              "named": (8, 28, 31, 34), "band": tuple(range(27, 36)), "parity_block": 31},
}
LLAMA = {"named": (20, 36, 40, 44), "band": tuple(range(35, 46))}  # the study's own runs, reused

# --- rows ---------------------------------------------------------------------------------------------------------
STORY_ROWS = "stories"  # Llama's 8,280 stories (with its top-ups) and every neutral dialogue, re-tokenised by each
POOL_FROM = 50  # model (add_special_tokens=True; Mistral's tokenizer with fix_mistral_regex=True), mean from 50 on
CAP, CAP_IT = 1000, 500  # Liars' Bench rows per label per set, as the Llama study (`rows.pick_lb`, seed seeds.stories)
# Liars' Bench rows at a block: the thesis's pooled read where it exists (`lb_models.spec(m).pooled_dirs`), else this
# study's read of the same tokenisation (`layer_sweep.pool`'s cached `tokenized.pt`); Gemma's 27-35 are the
# Aletheia band's reads (`aqb_gemma_lb/alpaca/repe`, all rows; capped here)

# --- analysis (phase 3's, unchanged) --------------------------------------------------------------------------------
FEATURE_SETS = "phase3.feature_sets + exploratory_sets + the 171 emotion vectors; persona bases from the model's own " \
               "primary bank (extended: 25 contrast personas; published: 8 deception + 7 honesty)"
PROBE = "lr"  # logistic regression only
PROTOCOLS = "A (RepE train + val, statement-grouped 5-fold), B (leave one set out), C (train one, test each other)"

# --- declared predictions (Llama's results known) -------------------------------------------------------------------
PREDICTIONS = {
    "P1-P5": "Llama's preregistered rules (ANALYSIS_PLAN.md) at each model's 4 named layers, where its sets allow "
             "(P2 and P5 need IT; P3 needs HP-C and HP-KR); a prediction holds for a model at >= 3 of its 4 layers",
    "XA1": "per model, all-170 minus raw activations, protocol B (mean over the model's sets), paired 95% CI above 0 "
           "at >= half of its band blocks",
    "XA2": "per model, all-170 minus the 170-d shuffled control (mean of 5), protocol B, CI above 0 at >= half of its "
           "band blocks",
}
REPORTED = ("protocols A and C", "train-one-test-all transfer at the 4 named layers beside the thesis's detectors "
            "(results/xmodel/tables/transfer.csv) and Llama's rows", "each model's band curve", "the basis checks")

# --- gates ------------------------------------------------------------------------------------------------------------
PARITY_MIN_COS = 0.9998  # each new Liars' Bench read: a 20-row subset, one row per pass, against the thesis's pooled read
VALENCE_MIN_ABS_R = 0.5  # |r(PC1, valence)| per block: reported and flagged, not dropped

N_BOOT = 1000
FREEZE = Path("E:/aletheia/logs/freeze_emotion_xmodel.txt")
