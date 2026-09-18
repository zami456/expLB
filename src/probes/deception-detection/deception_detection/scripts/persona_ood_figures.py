"""Heatmaps for the persona-PC transfer matrices.

AUROC is treated as *polarity* rather than magnitude: 0.5 is chance, and a cell below it
means the probe is anti-correlated with the label, which is qualitatively different from a
small positive result. HP-KR reaches 0.024 here, so a sequential ramp would bury the most
interesting cells at its light end. Both matrices therefore use a diverging map centred on
chance, with the delta map centred on zero. Blue-to-red with a neutral gray midpoint; no
rainbow, no status colours.
"""

from pathlib import Path

import fire
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from deception_detection.utils import REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "persona_ood"

SURFACE = "#fcfcfb"
INK, INK_MUTED = "#1a1a19", "#6b6b68"
NEUTRAL = "#f0efec"
# blue arm from the documented sequential ramp; red arm anchored on the categorical red
DIVERGING = LinearSegmentedColormap.from_list(
    "persona_div", ["#8f2020", "#e34948", NEUTRAL, "#3987e5", "#104281"]
)

SHORT = {
    "convince_me__plain": "CG",
    "harm_refusal_choice__plain": "HP-C",
    "harm_refusal_knowledge_report__plain": "HP-KR",
    "insider_trading_cadenza__plain": "IT",
    "instruct_dishonesty__plain": "ID",
    "repe_honesty__plain_val": "RepE",
}


def _load(out_dir: Path, name: str) -> pd.DataFrame:
    matrix = pd.read_csv(out_dir / f"matrix_{name}.csv", index_col=0)
    matrix.index = [SHORT.get(i, i) for i in matrix.index]
    matrix.columns = [SHORT.get(c, c) for c in matrix.columns]
    return matrix


def _heatmap(ax, matrix: pd.DataFrame, norm, title: str, fmt: str) -> mpl.image.AxesImage:
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=DIVERGING, norm=norm, aspect="equal")

    ax.set_xticks(range(len(matrix.columns)), matrix.columns, fontsize=9, color=INK)
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=9, color=INK)
    ax.set_xlabel("tested on", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("trained on", fontsize=9, color=INK_MUTED)
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc="left")

    # values on every cell: 36 cells is few enough to read, and the exact number is the point
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            rgba = DIVERGING(norm(values[i, j]))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                j, i, format(values[i, j], fmt),
                ha="center", va="center", fontsize=8,
                color=SURFACE if luminance < 0.5 else INK,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    return image


def run(out_dir: str | Path = OUT_DIR) -> None:
    out_dir = Path(out_dir)
    baseline, pc3 = _load(out_dir, "baseline"), _load(out_dir, "pc3")
    delta = _load(out_dir, "delta_pc3_vs_baseline")

    # --- the two AUROC matrices, on a shared scale centred at chance -------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    for ax, (matrix, title) in zip(
        axes,
        [
            (baseline, "Baseline — pooled activations (8192-d)"),
            (pc3, "Treatment — top-3 persona PCs (3-d)"),
        ],
        strict=True,
    ):
        ax.set_facecolor(SURFACE)
        image = _heatmap(ax, matrix, norm, title, ".2f")

    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    bar.set_label("AUROC  (0.5 = chance)", fontsize=9, color=INK_MUTED)
    bar.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    bar.outline.set_visible(False)
    fig.suptitle(
        "Cross-dataset probe transfer, Llama-3.3-70B layer 20",
        fontsize=12, color=INK, x=0.125, ha="left", y=0.98,
    )
    fig.savefig(out_dir / "transfer_matrices.png", dpi=200, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)

    # --- the delta, centred at zero ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    limit = float(np.nanmax(np.abs(delta.to_numpy(dtype=float))))
    image = _heatmap(
        ax, delta, TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        "Persona PCs minus baseline", "+.2f",
    )
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("Δ AUROC   (blue = PCs better)", fontsize=9, color=INK_MUTED)
    bar.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    bar.outline.set_visible(False)
    fig.savefig(out_dir / "transfer_delta.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    print(f"wrote transfer_matrices.png and transfer_delta.png to {out_dir}")


if __name__ == "__main__":
    fire.Fire(run)
