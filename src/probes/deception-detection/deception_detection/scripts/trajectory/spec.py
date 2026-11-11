"""Step 0 of the trajectory study: every constant the analysis uses, in one file the freeze hashes.

Question (the user's, 2026-11-06): instead of a probe on one layer's activation, does the change from
one layer to the next carry a deception signal? Step 0 answers it on reads already on disk, with no
NDIF call:

  (a) NARCBench Core (development tier), Llama-3.3-70B reads with the context swapped and the reply
      fixed (orig / PLANT / STRIP): is what the layers write (h[l+1] - h[l]) less of an instruction
      reader than the state (h[l])? This decides whether depth can separate the instruction from the
      lie at all.
  (b) Liars' Bench, the five sets (development data for Llama, as in the question-bank study):
      probes on states vs probes on differences vs both, leave-one-dataset-out over the thesis's
      datasets, plus D9 zero-shot on differences; HP-KR's identical-text slice is the wording gate.

PREREGISTRATION.md states the hypotheses and decision rules in words; the numbers they use are here.
"""

from pathlib import Path

from deception_detection.utils import REPO_ROOT

ROOT = Path("E:/trajectory")
LB_OUT = ROOT / "lb"
CORE_OUT = ROOT / "core"
LOGS = ROOT / "logs"
RESULTS = ROOT / "results"
REPO_RESULTS = REPO_ROOT / "results" / "trajectory"  # small summaries copied here for the commit
FREEZE = LOGS / "freeze.txt"
HERE = Path(__file__).resolve().parent

# --- compute -----------------------------------------------------------------------------------------
BLAS_THREADS = 6  # of 12 logical CPUs; fixed because LR solutions move with the thread count
FEATURE_DTYPE = "float32"  # 16 GB laptop, ~5 GB free: features are held in float32

# --- (b) Liars' Bench --------------------------------------------------------------------------------
BAND = tuple(range(35, 46))  # every block 35-45
BAND_PAIRS = tuple((l, l + 1) for l in range(35, 45))  # ten consecutive pairs
# Blocks 36, 40 and 44 come from the thesis's pooled files; the others from the emotion-coords band
# read (one NDIF pass, `E:/emotion_coords/pooled_band`). A pair is "within-pass" when both ends come
# from the band read.
THESIS_BAND = (36, 40, 44)
BAND_DIR = Path("E:/emotion_coords/pooled_band")
WITHIN_PASS_PAIRS = ((37, 38), (38, 39), (41, 42), (42, 43))
# the thesis's 16-layer sweep (layer_sweep.ALL_LAYERS), for the zero-shot depth profile
SWEEP = (20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 79)
SWEEP_PAIRS = tuple(zip(SWEEP[:-1], SWEEP[1:]))
TRAP_BASE = 20
TRAP_LAYERS = (36, 40, 44)  # "late minus early": z(L) - z(20)
N_BOOT_LB = 1000
BOOT_SEED_LB = 20261106
ALIGNED = 0.98  # layer_average's row-alignment rule

# --- (a) NARCBench Core --------------------------------------------------------------------------------
CORE_CONTENT = (20, 32, 33, 34, 35, 36, 37, 40, 44)  # narcbench spec.CONTENT_LAYERS
CORE_EOT = (32, 33, 34, 35, 36, 37, 40)  # narcbench spec.EOT_LAYERS
CORE_PRIMARY_PAIRS = ((32, 33), (33, 34), (34, 35), (35, 36), (36, 37))
CORE_SECONDARY_CONTENT = ((20, 32), (37, 40), (40, 44))
CORE_SECONDARY_EOT = ((37, 40),)
CORE_STATE_RANGE = (32, 33, 34, 35, 36, 37)  # where "informative" is judged

# --- decision thresholds -----------------------------------------------------------------------------
MIN_GAIN = 0.02  # (b): smallest AUROC gain that counts
SHARE_DROP = 0.20  # (a): the context share must fall by at least this much
INFORMATIVE_SHARE = 0.30  # (a): a family needs a context share this high somewhere in 32-37
T_NONINFERIOR = 0.05  # (a): STRIP-vs-control AUROC may fall by at most this much

# --- gates (known numbers to reproduce before anything new is scored) ------------------------------
G0A_D9_L40 = {"convince_me__plain": 0.9123, "harm_refusal_knowledge_report__plain": 0.9124}  # 4 dp
G0E_LOO_L40 = {  # results/loo_L40/loo.csv, arm "baseline", naive
    "convince_me__plain": 0.9866, "harm_refusal_choice__plain": 0.6796,
    "harm_refusal_knowledge_report__plain": 0.523, "insider_trading_cadenza__plain": 0.9373,
    "instruct_dishonesty__plain": 0.974}
G0E_TOL = 0.003  # float32 features and a different BLAS thread count may move the 3rd decimal
CORE_TOL = 1e-6

# --- confirmation data Step 0 must never open (file-audit flags) -------------------------------------
FORBIDDEN = ("/activations/aisi_llama/", "/activations/gender_secret/", "/aletheia/data/labels/test_",
             "/aletheia/data/test/", "/aletheia/acts/qwen35/test__", "/aletheia/acts/gemma/test__",
             "/emotion_coords/acts/aqb_test_", "/narcbench/verdict/", "__transfer", "__stego",
             "runs_transfer", "runs_stego")
