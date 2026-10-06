"""Figures for the INLP-subspace transfer experiment (inlp_ood.py).

Reuses the persona figures' palette and heatmap so the two experiments read as one system: the
same diverging map centred on chance for AUROC matrices, and categorical slots 1-3 (validated
all-pairs on the light surface) for the line series. Reference marks -- the raw baseline, chance,
the persona arms -- are drawn in ink rather than a series colour, since they are landmarks rather
than series. Slot 3 (aqua) sits below 3:1 contrast on the surface, so every series also carries a
legend entry, and summary.md carries the values as tables.

    python deception_detection/scripts/inlp_ood_figures.py run
"""

from pathlib import Path

import fire
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from deception_detection.scripts.inlp_ood import CHANCE_ACCURACY, OUT_DIR, SOURCE
from deception_detection.scripts.persona_ood_figures import (
    INK,
    INK_MUTED,
    NEUTRAL,
    SHORT,
    SURFACE,
    _heatmap,
    _load,
)

# categorical slots 1-3, in fixed order
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
DASHED = (0, (4, 3))

ARMS = [
    ("inlp", "INLP directions (RepE)"),
    ("pca", "top PCs of RepE, no labels"),
    ("random", "random directions"),
]


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(color=NEUTRAL, lw=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def trace(out_dir: Path) -> None:
    """How fast RepE's deception signal runs out as INLP removes directions."""
    frame = pd.read_csv(out_dir / "inlp_trace.csv")
    t_star = int(np.load(out_dir / "inlp_basis.npz")["t_star"])

    fig, ax = plt.subplots(figsize=(9.2, 4.4), facecolor=SURFACE)
    _style(ax)
    for (column, label), colour in zip(
        [
            ("dev_accuracy", "dev accuracy (102 held-out RepE rows)"),
            ("dev_auroc", "dev AUROC"),
            ("train_accuracy", "train accuracy (448 rows)"),
        ],
        SERIES,
        strict=True,
    ):
        ax.plot(frame["iteration"], frame[column], color=colour, lw=2, label=label,
                solid_capstyle="round", solid_joinstyle="round", zorder=3)

    ax.axhline(0.5, color=INK_MUTED, lw=1, ls=DASHED, zorder=2)
    ax.text(1, 0.5, "chance", fontsize=8, color=INK_MUTED, va="bottom")
    ax.axvline(t_star + 0.5, color=INK, lw=1, ls=DASHED, zorder=2)
    ax.text(t_star + 1.2, 0.97, f"T* = {t_star}: dev accuracy first\nfalls to {CHANCE_ACCURACY}",
            fontsize=8, color=INK, va="top")
    last = frame.iloc[-1]
    if not bool(last["kept"]):
        ax.annotate("solver returns w = 0:\nnothing left to remove", xy=(last["iteration"], 0.5),
                    xytext=(last["iteration"] - 17, 0.62), fontsize=8, color=INK_MUTED,
                    arrowprops={"arrowstyle": "-", "color": INK_MUTED, "lw": 1})

    ax.set_ylim(0.4, 1.02)
    ax.set_xlim(0, frame["iteration"].max() + 1)
    ax.set_xlabel("INLP iteration (directions removed = iteration - 1)", fontsize=9,
                  color=INK_MUTED)
    ax.set_ylabel("score of that iteration's classifier", fontsize=9, color=INK_MUTED)
    ax.set_title("INLP on RepE: how many directions carry the deception signal?", fontsize=12,
                 color=INK, loc="left", pad=12)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=INK)
    fig.savefig(out_dir / "inlp_trace.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def k_sweep(out_dir: Path) -> None:
    """Transfer and train-set sensitivity against subspace rank, per basis."""
    summary = pd.read_csv(out_dir / "summary.csv").set_index("features")
    baseline = summary.loc["baseline"]
    panels = [
        ("off_diagonal", "cross-dataset AUROC", "all datasets"),
        ("off_diagonal_excl_hpkr", "cross-dataset AUROC", "excluding HP-KR"),
        ("train_set_sensitivity", "std across training sets", "train-set sensitivity"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    for ax, (column, ylabel, title) in zip(axes, panels, strict=True):
        _style(ax)
        for (arm, label), colour in zip(ARMS, SERIES, strict=True):
            rows = summary[summary["arm"] == arm].sort_values("dim")
            ax.plot(rows["dim"], rows[column], color=colour, lw=2, label=label, zorder=3,
                    marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5,
                    solid_capstyle="round", solid_joinstyle="round")

        ax.axhline(baseline[column], color=INK_MUTED, lw=1, ls=DASHED, zorder=2)
        ax.text(0.99, baseline[column], "raw 8192-d ", fontsize=8, color=INK_MUTED,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
        for name, k in (("persona_pc1", 1), ("persona_pc3", 3)):
            if name in summary.index:
                ax.plot(k, summary.loc[name, column], marker="D", markersize=7, color=INK, ls="",
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
                        label="persona PCs (reference)" if k == 1 else None)

        ax.set_xscale("log")
        dims = sorted(summary.loc[summary["arm"] == "inlp", "dim"])
        ax.set_xticks(dims, [str(d) for d in dims])
        ax.minorticks_off()
        ax.set_xlabel("k = subspace rank", fontsize=9, color=INK_MUTED)
        ax.set_ylabel(ylabel, fontsize=9, color=INK_MUTED)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=4,
               frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Does projecting onto a RepE-derived subspace help probes transfer?", fontsize=12,
                 color=INK, x=0.06, ha="left", y=1.07)
    fig.text(0.06, -0.04, f"Means exclude the {SHORT[SOURCE]} test column and diagonal: the "
             "INLP and PCA bases were fitted on RepE. Random is 5 draws, averaged. A 1-d basis can "
             "only flip sign between training sets, so its sensitivity is 0 unless they disagree "
             "on that sign (as they do for PCA's first component).",
             fontsize=8, color=INK_MUTED)
    fig.savefig(out_dir / "k_sweep.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def _outline_source(ax, matrix: pd.DataFrame) -> None:
    """Dashed box on the source's test column: in-sample for the basis, so out of the means."""
    j = list(matrix.columns).index(SHORT[SOURCE])
    ax.add_patch(Rectangle((j - 0.5, -0.5), 1, len(matrix.index), fill=False, lw=1.2,
                           ls=DASHED, edgecolor=INK, zorder=5))


def matrices(out_dir: Path, ks: tuple[int, ...]) -> None:
    """Baseline against INLP at each k, and the deltas."""
    baseline = _load(out_dir, "baseline")
    treated = {k: _load(out_dir, f"inlp_k{k}") for k in ks}

    panels = [(baseline, "Raw pooled activations (8192-d)")] + [
        (treated[k], f"INLP subspace, k = {k}") for k in ks
    ]
    ncols = 2 if len(panels) == 4 else len(panels)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.8 * nrows), facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    for ax, (matrix, title) in zip(axes, panels, strict=False):
        ax.set_facecolor(SURFACE)
        image = _heatmap(ax, matrix, norm, title, ".2f")
        _outline_source(ax, matrix)
    for ax in axes[len(panels):]:
        ax.set_visible(False)
    bar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.015, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    bar.set_label("AUROC  (0.5 = chance)", fontsize=9, color=INK_MUTED)
    bar.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    bar.outline.set_visible(False)
    fig.suptitle("Cross-dataset probe transfer, Llama-3.3-70B layer 20 — INLP basis from RepE",
                 fontsize=12, color=INK, x=0.1, ha="left", y=0.99)
    fig.text(0.1, 0.03, "Dashed column: RepE is in-sample for the basis and is left out of every "
             "headline mean.", fontsize=8, color=INK_MUTED)
    fig.savefig(out_dir / "transfer_matrices.png", dpi=200, bbox_inches="tight",
                facecolor=SURFACE)
    plt.close(fig)

    deltas = {k: treated[k] - baseline for k in ks}
    limit = max(float(np.nanmax(np.abs(d.to_numpy(dtype=float)))) for d in deltas.values())
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(1, len(ks), figsize=(6.0 * len(ks), 4.8), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, ks, strict=True):
        ax.set_facecolor(SURFACE)
        image = _heatmap(ax, deltas[k], norm, f"INLP k = {k} minus raw", "+.2f")
        _outline_source(ax, deltas[k])
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    bar.set_label("Δ AUROC   (blue = INLP better)", fontsize=9, color=INK_MUTED)
    bar.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    bar.outline.set_visible(False)
    fig.savefig(out_dir / "transfer_delta.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def directions(out_dir: Path) -> None:
    """What each INLP direction does on its own, before any training set can weight it."""
    table = pd.read_csv(out_dir / "direction_auroc.csv").set_index("direction")
    table.columns = [SHORT.get(c, c) for c in table.columns]
    table.index = [str(i) for i in table.index]

    fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(table) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    image = _heatmap(ax, table, TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0),
                     "Zero-shot AUROC of each INLP direction alone", ".2f")
    _outline_source(ax, table)
    ax.set_aspect("auto")
    ax.set_xlabel("dataset", fontsize=9, color=INK_MUTED)
    ax.set_ylabel("INLP direction (fitted on RepE)", fontsize=9, color=INK_MUTED)
    bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03, ticks=[0, 0.25, 0.5, 0.75, 1.0])
    bar.set_label("AUROC  (blue: RepE's polarity, red: inverted)", fontsize=9, color=INK_MUTED)
    bar.ax.tick_params(labelsize=8, colors=INK_MUTED, length=0)
    bar.outline.set_visible(False)
    fig.savefig(out_dir / "direction_auroc.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def run(out_dir: str | Path = OUT_DIR, ks: tuple[int, ...] | None = None) -> None:
    """All figures. `ks` picks the INLP matrices to show; default k = 1, 2 and T*."""
    out_dir = Path(out_dir)
    if ks is None:
        ks = (1, 2, int(np.load(out_dir / "inlp_basis.npz")["t_star"]))
    trace(out_dir)
    k_sweep(out_dir)
    matrices(out_dir, tuple(ks))
    directions(out_dir)
    print(f"wrote inlp_trace.png, k_sweep.png, transfer_matrices.png, transfer_delta.png, "
          f"direction_auroc.png to {out_dir}")


if __name__ == "__main__":
    fire.Fire({"run": run})
