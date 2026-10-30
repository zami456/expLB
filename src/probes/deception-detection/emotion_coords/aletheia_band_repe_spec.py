"""The band with RepE training rows: the user asked to "train on RepE on the band" (2026-10-30), after the band's
Liars' Bench results were seen; labelled post hoc. Fixed before any RepE band read (ANALYSIS_PLAN.md, "Aletheia's
Quest band, RepE-trained"). `aletheia_band_repe.score` refuses to run unless this file's SHA-256 is the one recorded
in FREEZE.

Everything not named here is the band's (`aletheia_band_spec`): the blocks and offsets, the bases at each block, the
six feature sets, the organism reads of the test rows, the bootstrap draws. The test sets are the band's 6 readable
ones (its recorded deviation); the 8 once Echo and BotC can be read.
"""

from pathlib import Path

SOURCE = "RepE"
ROWS = {
    "gemma": "Gemma's pooled RepE train split (448 rows), from its cached tokenisation (`layer_sweep.pool`), as the "
             "middle-block test's RepE source",
    "qwen35": "R1's training rows: the first 80% of `aletheia/repe.py`'s rows (440), rendered as `repe.read` rendered "
              "them, each row's span pooled",
}
INNER_CV = "5-fold grouped by statement (Gemma: Llama's statement groups, the label order checked; Qwen3.5: the fact)"

PREDICTIONS = {
    "HR1": "all-170 (RepE) minus raw activations (RepE): paired 95% CI above 0 at >= 5 of the 9 offsets",
    "HR2": "all-170 (RepE) minus each 170-d control (RepE): CI above 0 at >= 5 of the 9 offsets, for every control",
}
REPORTED = ("each offset's mean, rank of 31 and among the 18 white-box entries",
            "all-170 (RepE) minus all-170 (LB) and minus raw (LB) by offset, paired",
            "the best offset (post hoc)", "offset 0 against the middle-block test's RepE cells")

PARITY_MIN_COS = 0.9998  # every RepE row, read one per pass, against the thesis's reads at the shared blocks
FREEZE = Path("E:/aletheia/logs/freeze_emotion_band_repe.txt")
