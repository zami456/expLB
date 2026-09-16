#!/usr/bin/env python
"""Shared figure style and naming for the persona-coordinates figures.

The short dataset names are copied from `src/visualization/visualization.py`
(`SUBSET_PRETTY_NAMES`) so the figures match the rest of the repo. They are COPIED, not
imported: that module runs `savefig` at import time, so importing it writes files as a
side effect of asking for a dictionary.

Runs on the laptop. The lab PC's Application Control blocks matplotlib's `_image` and
`ft2font` DLLs, so figures cannot be built there -- but every figure here reads a small
result JSON rather than the activation store, so nothing large has to move.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGDIR = REPO_ROOT / "figures" / "persona-coords"

# copied from src/visualization/visualization.py, extended with the datasets this
# pipeline added (the probe-repo rollouts and our own generated corpora)
SHORT = {
    "harm-pressure-choice": "HP-C",
    "harm-pressure-knowledge-report": "HP-KR",
    "convincing-game": "CG",
    "instructed-deception": "ID",
    "mask": "MASK",
    "insider-trading": "IT",
    "gender-secret": "GS",
    "alpaca": "AL",
    "ai_liar": "AIL",
    "ai_audit": "AUD",
    "roleplaying": "RP",
    "sandbagging_v2": "SB",
    "sycophancy_mt": "SYC",
    "soft-trigger-gender": "ST-gen",
    "soft-trigger-greeting": "ST-greet",
    "soft-trigger-time": "ST-time",
    "total": "Average",
}

SEMANTICS = {
    "instructed-deception": "instructed_intent",
    "harm-pressure-knowledge-report": "instructed_intent",
    "harm-pressure-choice": "instructed_intent",
    "insider-trading": "instructed_intent",
    "convincing-game": "instructed_intent",
    "ai_liar": "instructed_intent",
    "sandbagging_v2": "instructed_intent",
    "ai_audit": "instructed_intent",
    "roleplaying": "judge_grade",
    "mask": "belief_inconsistency",
    "sycophancy_mt": "belief_inconsistency",
    # alpaca carries no deception label at all -- it is honest-only and exists to set the
    # TPR@1%FPR threshold. It appears in the transfer matrix as a TARGET and never as a
    # source, so it needs a tag without being a construct.
    "alpaca": "control",
    "soft-trigger-gender": "trigger_conditioned",
    "soft-trigger-greeting": "trigger_conditioned",
    "soft-trigger-time": "trigger_conditioned",
    "gender-secret": "trigger_conditioned",
}
SEM_SHORT = {"instructed_intent": "intent", "judge_grade": "judge",
             "belief_inconsistency": "belief", "control": "control",
             "trigger_conditioned": "trigger"}
SEM_COLOR = {"instructed_intent": "#4C72B0", "judge_grade": "#DD8452",
             "belief_inconsistency": "#55A868", "control": "#B0B0B0",
             "trigger_conditioned": "#8172B3"}

# one colour per feature set, used consistently in every figure
FEATURE_COLOR = {
    "raw": "#4C72B0",
    "pc:deception": "#DD8452",
    "axes": "#55A868",
    "axes_resid": "#8172B3",
    "random": "#9A9A9A",
}
FEATURE_LABEL = {
    "raw": "raw activation (8192-d)",
    "pc:deception": "persona PCs (k=5)",
    "axes": "persona axes (4-d)",
    "axes_resid": "axes, assistant removed (3-d)",
    "random": "random subspace (k=5)",
}


def setup() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
    })


def chance(ax, y: float = 0.5, label: str = "chance") -> None:
    """Every AUROC panel gets this. Several results in this project sit BELOW it, which
    is only readable if the line is drawn."""
    ax.axhline(y, color="#B0B0B0", lw=0.9, ls="--", zorder=0)
    if label:
        ax.annotate(label, xy=(0.995, y), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7, color="#808080")


def save(fig, name: str) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path
