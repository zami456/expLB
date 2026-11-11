"""Paper figures: vector PDFs sized for the page, drawn from the committed results.

The screen figures in results/ are up to 16 inches wide, and shrunk to a 6.5-inch text block
their type falls below 4 pt. These redraw the same data at the paper's width with 8-9 pt type.
Colours, labels and bar order come from persona_ood_figures, so the two sets of figures agree.

    python paper/draft/make_figures.py
"""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from deception_detection import persona_pcs as pcs
from experiments.persona import persona_ood as P
from experiments.persona import persona_ood_figures as F

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parents[1] / "results"
OUT = HERE / "generated"
TEXT_WIDTH = 6.5  # inches, the article's text block
LABEL_X = 80.8  # the reference-line labels, just past the last layer on the depth axes

mpl.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42,
    "axes.edgecolor": F.INK_MUTED, "axes.labelcolor": F.INK, "text.color": F.INK,
    "xtick.color": F.INK_MUTED, "ytick.color": F.INK_MUTED,
})

ARM_COLOURS = {  # one colour per detector, shared by every figure
    "base": F.BASE_BAR, "dec": "#eb6834", "den": "#8ab6e8", "ext": "#2a78d6"}
ARM_LABELS = {
    "base": "base probe (trained on RepE)", "dec": "published contrast (8 v 7)",
    "den": "knowledge-denial contrast (4 v 6)", "ext": "extended contrast (12 v 13)"}
OP_ARMS = {P.BASE_ARM: "base", "deception axis (8v7)": "dec",
           "knowledge-denial axis (4v6)": "den", "extended axis (12v13)": "ext"}
# the same four detectors as depth.csv names them, keyed our way
ARMS_BY_KEY = {key: arm for arm, key in OP_ARMS.items()}
ORDER = ["convince_me__plain", "harm_refusal_choice__plain", "harm_refusal_knowledge_report__plain",
         "insider_trading_cadenza__plain", "instruct_dishonesty__plain", "repe_honesty__plain"]


def clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2)


def save(fig, name: str) -> None:
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {name}")


def transfer_matrices(layer: int = 40) -> None:
    """The headline: the raw-activation probe against the extended contrast, train x test."""
    panels = (("baseline", "Probe on raw activations (8192-d)"),
              ("ext_axis1", "Extended persona contrast (1-d)"))
    folder = RESULTS / f"persona_ood_ext_L{layer}"
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.85), constrained_layout=True)
    for ax, (arm, title) in zip(axes, panels, strict=True):
        m = pd.read_csv(folder / f"matrix_{arm}.csv", index_col=0).loc[ORDER, ORDER]
        values = m.to_numpy(float).T  # rows = test set, columns = training set
        image = ax.imshow(values, cmap=F.DIVERGING, norm=norm)
        for i in range(len(ORDER)):
            for j in range(len(ORDER)):
                v = values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v - 0.5) > 0.33 else F.INK)
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, lw=1.1,
                                       edgecolor=F.INK))
        labels = [F.SHORT[n] for n in ORDER]
        ax.set_xticks(range(len(ORDER)), labels, rotation=0)
        ax.set_yticks(range(len(ORDER)), labels)
        ax.set_xlabel("training set")
        ax.set_ylabel("test set")
        ax.set_title(title, loc="left")
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    bar = fig.colorbar(image, ax=axes, shrink=0.85, pad=0.02)
    bar.set_label("AUROC (0.5 = chance)")
    bar.outline.set_visible(False)
    save(fig, "fig_transfer.pdf")


def hpkr() -> None:
    """HP-KR across depth: every detector we ran, against the benchmark's reported probes."""
    from experiments.persona import layer_sweep as S

    layers = list(S.ALL_LAYERS)
    zero = {L: pd.read_csv(RESULTS / f"persona_ood_ext_L{L}" / "zero_shot.csv").set_index("dataset")
            for L in layers}
    ops = {}
    for L in layers:
        t = pd.read_csv(RESULTS / f"persona_ood_ext_L{L}" / "zero_shot_operating.csv")
        ops[L] = t[(t["threshold_source"] == "control") & (t["dataset"] == P.HPKR)].set_index("arm")
    columns = {"dec": "auroc_axis", "den": "auroc_denial_axis", "ext": "auroc_ext_axis"}
    series = {
        "auroc": {k: [zero[L].loc[P.HPKR, c] for L in layers] for k, c in columns.items()},
        "balacc": {OP_ARMS[a]: [ops[L].loc[a, "balacc"] for L in layers] for a in OP_ARMS},
    }
    series["auroc"]["base"] = [ops[L].loc[P.BASE_ARM, "auroc"] for L in layers]
    reported = {m: {p: P.PAPER_PROBES[p][m][P.PAPER_COLUMNS.index("HP-KR")]
                    for p in ("Mean Probe", "Follow-Up Probe", "Upper-Bound Probe")}
                for m in ("auroc", "balacc")}

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.45), constrained_layout=True, sharey=True)
    reference = "#b7b5ae"
    for ax, (metric, title) in zip(axes, (("auroc", "AUROC"),
                                          ("balacc", "Balanced accuracy at 1% FPR on Alpaca")),
                                   strict=True):
        ax.axhline(0.5, color=F.INK_MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        upper = reported[metric]["Upper-Bound Probe"]
        ax.axhline(upper, color=reference, lw=0.9, ls="--", zorder=1)
        ax.text(LABEL_X, upper, "LB upper bound\n(privileged)", va="center", fontsize=6.3,
                color=F.INK_MUTED, linespacing=0.95)
        ax.set_xticks(layers, [str(L) for L in layers], fontsize=6)
        ax.set_xlim(layers[0] - 1.5, layers[-1] + 1.5)
        ax.set_xlabel("layer (block output)")
        ax.set_ylim(0, 1.02)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", color=F.NEUTRAL, lw=0.8, zorder=0)
        clean(ax)

    # AUROC: the two reported probes sit 0.02 apart, so they share one label
    low = [reported["auroc"][p] for p in ("Mean Probe", "Follow-Up Probe")]
    for value, style in zip(low, (":", "-."), strict=True):
        axes[0].axhline(value, color=reference, lw=0.9, ls=style, zorder=1)
    axes[0].text(LABEL_X, float(np.mean(low)), "LB Mean and\nFollow-Up probes", va="center",
                 fontsize=6.3, color=F.INK_MUTED, linespacing=0.95)
    for key in ("base", "dec", "den", "ext"):
        axes[0].plot(layers, series["auroc"][key], marker="o", ms=3.0, lw=1.4, zorder=3,
                     color=ARM_COLOURS[key], label=ARM_LABELS[key],
                     markeredgecolor="white" if key != "base" else F.INK_MUTED,
                     markeredgewidth=0.6)

    # Balanced accuracy: three detectors flag nothing and sit exactly on 0.5, on top of the
    # benchmark's Mean Probe. Drawn as spread markers so none hides another.
    follow_up = reported["balacc"]["Follow-Up Probe"]
    axes[1].axhline(follow_up, color=reference, lw=0.9, ls="-.", zorder=1)
    axes[1].text(LABEL_X, follow_up, "LB Follow-Up\nprobe", va="center", fontsize=6.3,
                 color=F.INK_MUTED, linespacing=0.95)
    axes[1].plot(layers, series["balacc"]["den"], marker="o", ms=3.0, lw=1.4, zorder=3,
                 color=ARM_COLOURS["den"], markeredgecolor="white", markeredgewidth=0.6)
    for key, shift in (("base", -1.1), ("dec", 0.0), ("ext", 1.1)):
        axes[1].plot([L + shift for L in layers], series["balacc"][key], ls="none", marker="o",
                     ms=3.0, zorder=3, color=ARM_COLOURS[key],
                     markeredgecolor="white" if key != "base" else F.INK_MUTED,
                     markeredgewidth=0.6)
    axes[1].annotate("flag nothing at any depth: 0.50,\nas does the LB Mean Probe", xy=(58, 0.5),
                     xytext=(48, 0.30), ha="center", fontsize=6.5, color=F.INK_MUTED,
                     arrowprops=dict(arrowstyle="-", color=F.INK_MUTED, lw=0.6))
    axes[0].set_ylabel("HP-KR score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.1))
    save(fig, "fig_hpkr.pdf")


def operating(layer: int = 40) -> None:
    """Zero-shot detection under the Liars' Bench protocol, in the layout of its Figure 2."""
    table = pd.read_csv(RESULTS / f"persona_ood_ext_L{layer}" / "zero_shot_operating.csv")
    table = table[table["threshold_source"] == "control"]
    columns = ORDER + [P.LB_AVERAGE]
    labels = [F.SHORT[n] for n in ORDER] + ["Average\n(excl. RepE)"]
    x = np.array([i if i < len(ORDER) else i + 0.6 for i in range(len(columns))])
    arms = [a for a in OP_ARMS if a in set(table["arm"])]
    width, gap = 0.78 / len(arms), 0.015

    metrics = (("balacc", "Balanced accuracy", 0.5), ("auroc", "AUROC", 0.5), ("recall", "Recall", None))
    fig, axes = plt.subplots(3, 1, figsize=(TEXT_WIDTH, 4.6), sharex=True, constrained_layout=True)
    for ax, (metric, title, chance) in zip(axes, metrics, strict=True):
        for index, arm in enumerate(arms):
            key = OP_ARMS[arm]
            rows = table[table["arm"] == arm].set_index("dataset").loc[columns]
            values = rows[metric].to_numpy(float)
            err = np.vstack([values - rows[f"{metric}_lo"].to_numpy(float),
                             rows[f"{metric}_hi"].to_numpy(float) - values])
            offset = (index - (len(arms) - 1) / 2) * (width + gap)
            bars = ax.bar(x + offset, values, width=width, color=ARM_COLOURS[key],
                          label=ARM_LABELS[key], zorder=3)
            if key == "base":  # RepE is its training set
                own = bars[columns.index("repe_honesty__plain")]
                own.set(hatch=F.IN_DISTRIBUTION_HATCH, edgecolor=F.SURFACE, linewidth=0)
            ax.errorbar(x + offset, values, yerr=err, fmt="none", ecolor=F.INK_MUTED,
                        elinewidth=0.7, capsize=1.2, zorder=4)
        if chance is not None:
            ax.axhline(chance, color=F.INK_MUTED, lw=0.8, ls=(0, (4, 3)), zorder=2)
        ax.axvline(x[-1] - 0.8, color=F.NEUTRAL, lw=1.2, zorder=1)
        ax.set_ylim(0, 1.04)
        ax.set_ylabel(title)
        ax.grid(axis="y", color=F.NEUTRAL, lw=0.8, zorder=0)
        clean(ax)
    axes[-1].set_xticks(x, labels)
    handles, names = axes[0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=F.BASE_BAR, edgecolor=F.SURFACE,
                                 hatch=F.IN_DISTRIBUTION_HATCH, linewidth=0))
    names.append("hatched: training set")
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.075))
    save(fig, "fig_operating.pdf")


def personas(layer: int = 40) -> None:
    """The 36 personas in the plane of the published and the knowledge-denial contrasts."""
    bank = pcs.load_bank_at_layer(layer, pcs.EXTENDED_VECTORS_DIR)
    centred = bank.personas - bank.personas.mean(0)
    x_axis, y_axis = bank.axis_unit("deception"), bank.axis_unit("denial")
    groups = (
        ("deceptive (published)", pcs.DECEPTION_PERSONAS, "#eb6834", "o", True),
        ("honest (published)", pcs.HONESTY_PERSONAS, "#2a78d6", "o", False),
        ("knowledge denial (new)", pcs.KNOWLEDGE_DENIAL_PERSONAS, "#8f2020", "^", True),
        ("sincere ignorance (new)", ("pc_honest_novice", "pc_calibrated_reporter",
                                     "pc_scrupulous_witness"), "#1baf7a", "^", False),
        ("plain assertion (new)", ("pc_forthright_expert", "pc_candid_witness", "pc_plain_dealer"),
         "#4a3aa7", "s", False),
        ("sycophantic", pcs.SYCOPHANCY_PERSONAS, "#eda100", "D", True),
        ("anti-sycophantic", pcs.ANTI_SYCOPHANCY_PERSONAS, "#e87ba4", "D", False),
        ("default assistant", ("default",), F.INK, "*", True),
    )
    fig, ax = plt.subplots(figsize=(4.6, 3.3), constrained_layout=True)
    ax.axhline(0, color=F.NEUTRAL, lw=0.8, zorder=0)
    ax.axvline(0, color=F.NEUTRAL, lw=0.8, zorder=0)
    for label, names, colour, marker, filled in groups:
        idx = [bank.names.index(n) for n in names]
        xs, ys = centred[idx] @ x_axis, centred[idx] @ y_axis
        ax.scatter(xs, ys, s=30 if marker != "*" else 70, marker=marker, label=label, zorder=3,
                   facecolors=colour if filled else "white", edgecolors=colour, linewidths=1.1)
    ax.set_xlabel("projection on the published contrast (8 v 7)")
    ax.set_ylabel("projection on the knowledge-denial\ncontrast (4 v 6)")
    ax.set_title(f"Persona vectors, layer {layer} (cosine between axes "
                 f"{x_axis @ y_axis:.2f})", loc="left")
    clean(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, handletextpad=0.3)
    save(fig, "fig_personas.pdf")


def depth() -> None:
    """Transfer and the zero-shot average against layer, over the stride-4 grid from 20 to 79.

    Two panels, not the six of `results/persona_depth/depth.png`: HP-KR gets its own figure across
    the same grid (`hpkr`), and the balanced-accuracy curves of these two metrics say nothing the
    text argues from. The two contrasts built on knowledge denial share a hue, so dash and marker
    carry the distinction between them.
    """
    from experiments.persona import layer_sweep as S

    table = pd.read_csv(RESULTS / "persona_depth" / "depth.csv")
    layers = sorted(table["layer"].unique())
    style = {  # key -> (dash, marker); colour comes from ARM_COLOURS
        "base": ((0, (6, 2, 1, 2)), "v"), "dec": ("-", "o"),
        "den": ((0, (5, 2)), "s"), "ext": ("-", "o")}
    transfer_arm = dict(ARMS_BY_KEY, base="single-source probe (8192-d)")
    panels = [
        (S.TRANSFER_SOURCE, "auroc", "Transfer: mean off-diagonal AUROC", transfer_arm),
        (S.LB_SOURCE, "auroc", "Zero-shot: Liars' Bench average AUROC", ARMS_BY_KEY),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.45), constrained_layout=True)
    for index, (ax, (source, metric, title, arms)) in enumerate(zip(axes.ravel(), panels)):
        for key, arm in arms.items():
            rows = table[(table.source == source) & (table.metric == metric) & (table.arm == arm)]
            if rows.empty:
                continue
            rows = rows.sort_values("layer")
            dash, marker = style[key]
            label = "single-source probe" if (key == "base" and source == S.TRANSFER_SOURCE) \
                else ARM_LABELS[key]
            ax.plot(rows["layer"], rows["value"], color=ARM_COLOURS[key], lw=1.2, ls=dash,
                    marker=marker, ms=2.6, label=label)
        ax.axhline(0.5, color=F.INK_MUTED, lw=0.7, ls=(0, (4, 3)))
        # where every arm sits on chance, autoscale magnifies float noise into structure
        low, high = ax.get_ylim()
        if high - low < 0.06:
            ax.set_ylim((high + low) / 2 - 0.03, (high + low) / 2 + 0.03)
        ax.set_title(title, loc="left")
        ax.set_xticks(layers)
        ax.set_xticklabels([str(layer) for layer in layers], fontsize=6)
        ax.grid(axis="y", color=F.NEUTRAL, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel("layer (block output)")
        clean(ax)
        # Each panel gets its own legend: the grey line is a different detector in each -- the
        # single-source transfer probe on the left, the RepE-trained base probe on the right --
        # so one shared key would name it wrongly in one of them.
        # an opaque background: the chance line runs behind the last entry otherwise
        ax.legend(loc="lower left", fontsize=6, frameon=True, facecolor="white",
                  edgecolor="none", framealpha=0.95, handlelength=2.0, labelspacing=0.25,
                  borderpad=0.2)
    save(fig, "fig_depth.pdf")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    transfer_matrices()
    hpkr()
    operating()
    personas()
    depth()


if __name__ == "__main__":
    main()
