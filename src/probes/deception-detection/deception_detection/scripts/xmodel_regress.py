"""The Llama regression for the cross-model generalisation: re-run the thesis's CPU stages with the
generalised code (no `--model`, or `--model llama`) and compare every output with the committed one.

Each stage overwrites tracked files under `results/`; afterwards `git diff` decides. Cells that may
move are the 8192-d logistic-regression fits on HP-C and HP-KR (about 135 lies against 2,000 honest
rows, not converged within `max_iter`): their AUROCs vary in the fourth decimal from run to run on
this machine. Everything else must come back byte-identical. Restore with `git checkout -- <paths>`.

    python deception_detection/scripts/xmodel_regress.py run [--stages k_robustness,recall_denial]
    python deception_detection/scripts/xmodel_regress.py compare
"""

import io
import subprocess
import sys

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from deception_detection.utils import REPO_ROOT

GIT_ROOT = REPO_ROOT.parents[2]
PREFIX = REPO_ROOT.relative_to(GIT_ROOT).as_posix()
PY = sys.executable
SCRIPTS = "deception_detection/scripts"
TOL = 1e-3  # for the solver cells above; byte-identical is expected everywhere else

STAGES = {
    "recall_denial": [f"{SCRIPTS}/recall_denial.py datasets", f"{SCRIPTS}/recall_denial.py combine",
                      f"{SCRIPTS}/recall_denial.py d5_plus_k", f"{SCRIPTS}/recall_denial.py d5_plus_k_cosines"],
    "k_robustness": [f"{SCRIPTS}/k_robustness.py {c}" for c in
                     ("bootstrap", "depth", "calibration", "operating_points", "paired", "balacc_depth")],
    "k_transfer": [f"{SCRIPTS}/k_transfer.py run --layer=40", f"{SCRIPTS}/k_transfer.py sweep"],
    "persona": [f"{SCRIPTS}/layer_sweep.py analyse --layer {layer}" for layer in (20, 40, 44)]
               + [f"{SCRIPTS}/layer_sweep.py depth"],
}


def run(stages=None) -> None:
    """Run the named stages (all by default) from the repo root, in order."""
    names = list(STAGES) if stages is None else ([stages] if isinstance(stages, str) else list(stages))
    for name in names:
        for command in STAGES[name]:
            logger.info(f"[{name}] python {command}")
            subprocess.run([PY, *command.split()], cwd=REPO_ROOT, check=True)


def _committed(path: str) -> str:
    return subprocess.run(["git", "show", f"HEAD:{path}"], cwd=GIT_ROOT, capture_output=True, text=True,
                          check=True).stdout


def compare(tol: float = TOL) -> pd.DataFrame:
    """Every tracked file under results/ that differs from HEAD: byte-identical, or for a CSV the
    largest absolute difference over its numeric cells."""
    changed = subprocess.run(["git", "diff", "--name-only", "--", f"{PREFIX}/results"], cwd=GIT_ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    rows = []
    for path in changed:
        if not path.endswith(".csv"):
            rows.append({"file": path, "max_abs_diff": np.nan, "verdict": "binary differs (figure)"})
            continue
        old, new = pd.read_csv(io.StringIO(_committed(path))), pd.read_csv(GIT_ROOT / path)
        if list(old.columns) != list(new.columns) or old.shape != new.shape:
            rows.append({"file": path, "max_abs_diff": np.inf, "verdict": "shape or columns differ"})
            continue
        numeric = old.select_dtypes("number").columns
        text_same = old.drop(columns=numeric).equals(new.drop(columns=numeric))
        diff = float((old[numeric] - new[numeric]).abs().max().max()) if len(numeric) else 0.0
        verdict = "ok" if text_same and diff <= tol else ("text cells differ" if not text_same else "too large")
        rows.append({"file": path, "max_abs_diff": diff, "verdict": verdict})
    table = pd.DataFrame(rows, columns=["file", "max_abs_diff", "verdict"])
    pd.set_option("display.width", 250)
    logger.info(f"{len(changed)} tracked result files differ from HEAD:\n"
                f"{table.to_string(index=False) if len(table) else '(none)'}")
    return table


if __name__ == "__main__":
    fire.Fire({"run": run, "compare": lambda tol=TOL: None if compare(tol) is None else None})
