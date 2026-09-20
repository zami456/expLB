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
    "repe_honesty__plain": "RepE",
    "repe_honesty__plain_val": "RepE (val only)",
}


def _layer(out_dir: Path) -> str:
    """The layer this run used, read from summary.csv rather than hardcoded.

    With more than one layer in play a mislabelled figure is worse than no figure, so the title
    is derived from the same file the numbers come from.
    """
    try:
        layers = pd.read_csv(out_dir / "summary.csv")["layer"].unique()
    except (OSError, KeyError):
        return "layer unknown"
    return f"layer {layers[0]}" if len(layers) == 1 else f"layers {sorted(layers)}"


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
        f"Cross-dataset probe transfer, Llama-3.3-70B {_layer(out_dir)}",
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
        f"Persona PCs minus baseline  ({_layer(out_dir)})", "+.2f",
    )
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("Δ AUROC   (blue = PCs better)", fontsize=9, color=INK_MUTED)
    bar.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    bar.outline.set_visible(False)
    fig.savefig(out_dir / "transfer_delta.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)

    print(f"wrote transfer_matrices.png and transfer_delta.png to {out_dir}")





# categorical slots 1 and 2, validated against the light surface
SERIES = ("#2a78d6", "#eb6834")

# ordered so the comparison reads: reference, deception arms, sycophancy arms, control
VARIANT_LABELS = [
    ("baseline", "pooled activations (8192-d)"),
    ("pc1", "deception PC1 (1-d)"),
    ("pc3", "deception PCs (3-d)"),
    ("axis1", "deception axis (1-d)"),
    ("syc_pc1", "sycophancy PC1 (1-d)"),
    ("syc3", "sycophancy PCs (3-d)"),
    ("syc_axis1", "sycophancy axis (1-d)"),
    ("dec3_orth", "deception PCs, sycophancy removed *"),
    ("syc3_orth", "sycophancy PCs, deception removed *"),
    ("random3", "random directions (3-d)"),
]

# the two starred rows are not in the paper; it never orthogonalises one behaviour's axis out
# of another's basis, and never applies one behaviour's basis to the other's datasets
NOT_IN_PAPER = {"dec3_orth", "syc3_orth"}


def variants(out_dir: str | Path = OUT_DIR) -> None:
    """Off-diagonal transfer per feature set, with and without the inverted HP-KR."""
    out_dir = Path(out_dir)
    summary = pd.read_csv(out_dir / "summary.csv").set_index("features")
    rows = [(lab, key) for key, lab in VARIANT_LABELS if key in summary.index]
    labels = [lab for lab, _ in rows]
    incl = [summary.loc[k, "off_diagonal_mean"] for _, k in rows]
    excl = [summary.loc[k, "off_diagonal_excl_hpkr"] for _, k in rows]

    y = np.arange(len(rows))
    height = 0.38
    gap = 0.02  # 2px-equivalent surface gap between adjacent fills
    fig, ax = plt.subplots(figsize=(9.2, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for offset, values, colour, name in (
        (+height / 2 + gap / 2, excl, SERIES[1], "excluding HP-KR"),
        (-height / 2 - gap / 2, incl, SERIES[0], "all six datasets"),
    ):
        bars = ax.barh(y + offset, values, height=height, color=colour, label=name, zorder=3)
        for bar, value in zip(bars, values, strict=True):
            ax.text(value + 0.008, bar.get_y() + bar.get_height() / 2, f"{value:.3f}",
                    va="center", fontsize=8, color=INK)

    ax.axvline(0.5, color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(0.5, len(rows) - 0.35, " chance", fontsize=8, color=INK_MUTED, va="center")

    ax.set_yticks(y, labels, fontsize=9, color=INK)
    for tick, (_, key) in zip(ax.get_yticklabels(), rows, strict=True):
        if key in NOT_IN_PAPER:
            tick.set_color(INK_MUTED)  # visually demote what the paper does not do
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("mean off-diagonal AUROC  (cross-dataset transfer)", fontsize=9, color=INK_MUTED)
    ax.tick_params(axis="x", labelsize=8, colors=INK_MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=NEUTRAL, lw=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title(f"Which persona basis transfers across datasets?  ({_layer(out_dir)})",
                 fontsize=12, color=INK, loc="left", pad=12)
    ax.text(0.0, -0.13, "* not in the paper - added here because the deception and sycophancy "
            "axes overlap at |cos| = 0.54", transform=ax.transAxes, fontsize=8, color=INK_MUTED)
    fig.savefig(out_dir / "variants.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote variants.png to {out_dir}")




def syc(out_dir: str | Path = OUT_DIR) -> None:
    """The two persona bases side by side, and where they differ.

    Compares deception PCs against sycophancy PCs directly rather than each against the raw
    baseline -- the baseline already has its own figure, and the question here is what changes
    when the basis is fitted on a different behaviour.
    """
    out_dir = Path(out_dir)
    pc3, syc3 = _load(out_dir, "pc3"), _load(out_dir, "syc3")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    for ax, (matrix, title) in zip(
        axes,
        [
            (pc3, "Deception PCs (3-d)"),
            (syc3, "Sycophancy PCs (3-d)"),
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
        f"Matched vs mismatched persona basis, on deception datasets  ({_layer(out_dir)})",
        fontsize=12, color=INK, x=0.125, ha="left", y=0.98,
    )
    fig.savefig(out_dir / "transfer_sycophancy.png", dpi=200, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    delta = syc3 - pc3
    limit = float(np.nanmax(np.abs(delta.to_numpy(dtype=float))))
    image = _heatmap(
        ax, delta, TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        f"Sycophancy PCs minus deception PCs  ({_layer(out_dir)})", "+.2f",
    )
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    bar.set_label("Δ AUROC   (blue = sycophancy better)", fontsize=9, color=INK_MUTED)
    bar.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    bar.outline.set_visible(False)
    fig.savefig(out_dir / "transfer_sycophancy_delta.png", dpi=200, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)

    print(f"wrote transfer_sycophancy.png and transfer_sycophancy_delta.png to {out_dir}")


if __name__ == "__main__":
    fire.Fire({"run": run, "variants": variants, "syc": syc})
