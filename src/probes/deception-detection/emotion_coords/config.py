"""Settings and paths for the emotion-coordinate study, read from configs/emotion_coords.yaml.

Cached arrays live in `data/emotion_coords/`, which carries its own `.gitignore` (`*`) so that no
activation array can be committed even though the shared `.gitignore` does not list the folder.
Small tables, figures and reports go to `results/emotion_coords/`.
"""

from functools import cache
from pathlib import Path

import yaml

from deception_detection.utils import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "configs" / "emotion_coords.yaml"
DATA = REPO_ROOT / "data" / "emotion_coords"
RESULTS = REPO_ROOT / "results" / "emotion_coords"
PACKAGE = Path(__file__).resolve().parent
LIARS_BENCH_ROOT = REPO_ROOT.parents[2]  # deception-detection -> probes -> src -> liars-bench
ENV_FILE = LIARS_BENCH_ROOT / ".env"


@cache
def load() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def data_dir(*parts: str) -> Path:
    """A folder under data/emotion_coords/, created on first use with the ignore-everything guard."""
    DATA.mkdir(parents=True, exist_ok=True)
    guard = DATA / ".gitignore"
    if not guard.exists():
        guard.write_text("# activation arrays and downloads: never committed\n*\n", encoding="utf-8")
    path = DATA.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(*parts: str) -> Path:
    path = RESULTS.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def pooled_dir(layer: int) -> Path:
    """The thesis's pooled Liars' Bench activations at `layer` (`layer_sweep.pooled_dir`); for a block the thesis
    never pooled (the band 35-45), the same files built from this study's own reads (`lbdata.write_band_pooled`)."""
    thesis = REPO_ROOT / "activations" / ("pooled" if layer == 20 else f"pooled_L{layer}")
    return thesis if thesis.exists() else DATA / "pooled_band" / f"pooled_L{layer}"


def load_env() -> None:
    """HF_TOKEN and NDIF_API_KEY from the repo's .env, without printing them."""
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE)


def int_list(x) -> list[int]:
    """A layer list from the command line: fire turns "32,36" into a tuple and "20" into an int."""
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(v) for v in str(x).split(",")]
