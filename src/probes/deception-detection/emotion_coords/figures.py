"""Figures for the emotion-coordinate study: static PNGs in results/emotion_coords/figures/, each with a CSV of
the plotted numbers beside it (the table view).

    python -m emotion_coords.figures all [--out phase3]

Colours follow the dataviz reference palette: categorical slots in fixed order (validated: adjacent pairs pass
for lines; the first three pass all pairs for points), a diverging blue/red pair with a gray midpoint for AUROC
(midpoint 0.5) and valence (midpoint 0). Text stays in ink colours; series carry identity by colour and label.
"""

import json

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

from emotion_coords import config as C  # noqa: E402
from emotion_coords import lbdata as LB  # noqa: E402
from emotion_coords import vectors as V  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#104281", "#3987e5", "#cde2fb", "#f0efec", "#f7c4c3", "#e34948", "#9c2423"])
SHORT = {"convince_me__plain": "CG", "harm_refusal_choice__plain": "HP-C",
         "harm_refusal_knowledge_report__plain": "HP-KR", "insider_trading_cadenza__plain": "IT",
         "instruct_dishonesty__plain": "ID"}
FAMILIES = [("emotion_pc", "emotion PCs"), ("topic_pc", "topic PCs"), ("persona_pub_pc", "persona PCs (published)"),
            ("shuffled_pc", "shuffled-label emotion PCs"), ("alpaca_pc", "Alpaca PCs"), ("trainpca_pc", "training-set PCs"),
            ("persona_pc", "persona PCs (extended)")]


def _style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "text.color": INK, "legend.frameon": False, "lines.linewidth": 2,
    })


TAG = ""  # appended to every file name (`all --tag _band` for the post hoc band)


def _out(name: str):
    stem, ext = name.rsplit(".", 1)
    return C.results_dir("figures") / f"{stem}{TAG}.{ext}"


def _probes(out: str) -> pd.DataFrame:
    return pd.read_csv(C.results_dir(out) / "probes.csv")


def k_sweep(out: str = "phase3", probe: str = "lr") -> None:
    """Protocol B, pooling (a): mean held-out AUROC at each fixed k, per family, with the random band."""
    f = _probes(out)
    b = f[(f["protocol"] == "B") & (f["pooling"] == "a") & (f["probe"] == probe)]
    layers = sorted(b["layer"].unique())
    fig, axes = plt.subplots(1, len(layers), figsize=(3.2 * len(layers), 3.4), sharey=True, squeeze=False)
    rows = []
    for ax, layer in zip(axes[0], layers):
        bl = b[b["layer"] == layer]
        rnd = bl[(bl["feature_set"] == "random") & bl["k"].isin([2, 3, 5, 10, 20])]  # k = 6 only matches P+E
        rnd = rnd.groupby(["k", "seed"])["auroc"].mean().unstack()
        ax.fill_between(rnd.index, rnd.quantile(0.025, axis=1), rnd.quantile(0.975, axis=1), color=GRID, lw=0,
                        label="random subspaces, 95% band")
        ax.plot(rnd.index, rnd.mean(axis=1), color=MUTED, lw=1.5, label="random subspaces, mean")
        rows += [{"layer": layer, "arm": "random mean", "k": k, "auroc": v} for k, v in rnd.mean(axis=1).items()]
        for slot, (fs, label) in enumerate(FAMILIES):
            sub = bl[bl["feature_set"] == fs]
            if sub.empty:
                continue
            curve = sub.groupby(["k", "seed"])["auroc"].mean().groupby("k").mean()  # shuffled: mean over 5 bases
            ax.plot(curve.index, curve.values, color=SLOTS[slot], marker="o", ms=4, label=label)
            rows += [{"layer": layer, "arm": label, "k": k, "auroc": v} for k, v in curve.items()]
        raw = bl[bl["feature_set"] == "raw"]["auroc"].mean()
        ax.axhline(raw, color=INK2, lw=1, ls="--")
        ax.text(20, raw, " raw", color=INK2, va="center", fontsize=8)
        ax.axhline(0.5, color=AXIS, lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([2, 3, 5, 10, 20])
        ax.set_xticklabels(["2", "3", "5", "10", "20"])
        ax.set_title(f"layer {layer}")
        ax.set_xlabel("k (dimensions)")
        rows.append({"layer": layer, "arm": "raw", "k": np.nan, "auroc": raw})
    axes[0][0].set_ylabel("mean held-out AUROC (protocol B)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(_out(f"k_sweep_{probe}.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(_out(f"k_sweep_{probe}.csv"), index=False)


def transfer(out: str = "phase3", probe: str = "lr",
             sets=("raw", "emotion_pc", "persona_pub_pc", "persona_pc", "topic_pc")) -> None:
    """Protocol C (train on one set, test on each; diagonal by CV), at each arm's chosen k."""
    f = _probes(out)
    c = f[(f["protocol"] == "C") & (f["pooling"] == "a") & (f["probe"] == probe) & f["chosen"]]
    layers = sorted(c["layer"].unique())
    order = list(SHORT)
    fig, axes = plt.subplots(len(sets), len(layers), figsize=(2.6 * len(layers), 2.5 * len(sets)), squeeze=False)
    norm = TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)
    rows = []
    for i, fs in enumerate(sets):
        for j, layer in enumerate(layers):
            m = (c[(c["feature_set"] == fs) & (c["layer"] == layer)]
                 .pivot_table(index="train", columns="test", values="auroc").reindex(index=order, columns=order))
            ax = axes[i][j]
            ax.imshow(m.values, cmap=DIVERGING, norm=norm)
            for a in range(len(order)):
                for b_ in range(len(order)):
                    v = m.values[a, b_]
                    if np.isfinite(v):
                        ax.text(b_, a, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                                color="#ffffff" if abs(v - 0.5) > 0.32 else INK)
            ax.set_xticks(range(len(order)), [SHORT[o] for o in order], fontsize=7, rotation=45)
            ax.set_yticks(range(len(order)), [SHORT[o] for o in order], fontsize=7)
            ax.grid(False)
            ax.set_title(f"{fs}, layer {layer}", fontsize=8)
            rows += [{"feature_set": fs, "layer": layer, "train": SHORT[a], "test": SHORT[b_], "auroc": m.loc[a, b_]}
                     for a in order for b_ in order]
    fig.supxlabel("test set", color=INK2, fontsize=9)
    fig.supylabel("training set", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(_out(f"transfer_{probe}.png"), dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(_out(f"transfer_{probe}.csv"), index=False)


def hpkr_subgroups(layers=(20, 52)) -> None:
    """D1: each scorer's HP-KR scores by subgroup (z-scored per scorer)."""
    z = np.load(C.data_dir("phase4") / f"d1_scores{TAG}.npz")
    subgroup = z["subgroup"]
    groups = [("lie", "lie"), ("honest_admission", "honest admission"), ("honest_denial", "honest denial")]
    scorers = sorted({k.split("|", 1)[1] for k in z.files if "|" in k},
                     key=lambda n: (not n.startswith("A "), not n.startswith("B "), n))
    keep = [n for n in scorers if n.startswith(("A ", "B ")) or n in
            ("zero-shot afraid", "zero-shot guilty", "zero-shot compassionate", "zero-shot calm")]
    fig, axes = plt.subplots(len(layers), len(keep), figsize=(1.9 * len(keep), 2.8 * len(layers)), squeeze=False,
                             sharey=True)
    rows = []
    for i, layer in enumerate(layers):
        for j, name in enumerate(keep):
            ax = axes[i][j]
            s = z[f"L{layer}|{name}"].astype(float)
            s = (s - s.mean()) / s.std()
            data = [s[subgroup == g] for g, _ in groups]
            parts = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                               medianprops={"color": INK, "lw": 1.2}, whiskerprops={"color": MUTED},
                               capprops={"color": MUTED})
            for patch, color in zip(parts["boxes"], SLOTS[:3]):
                patch.set_facecolor(color)
                patch.set_edgecolor(SURFACE)
            ax.set_xticks([1, 2, 3], ["lie", "adm.", "den."], fontsize=7)
            ax.set_title(f"{name}\nlayer {layer}", fontsize=7.5)
            rows += [{"layer": layer, "scorer": name, "subgroup": g, "median_z": float(np.median(d)), "n": len(d)}
                     for (g, _), d in zip(groups, data)]
    axes[0][0].set_ylabel("score (z within scorer)")
    fig.tight_layout()
    fig.savefig(_out("hpkr_subgroups.png"), dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(_out("hpkr_subgroups.csv"), index=False)


def d2d3_effects(layers=None) -> None:
    """Interaction t (known x harmful) of the P8 emotions against the random-direction and topic nulls."""
    from emotion_coords.hpkr import P8_COMPASSION, P8_THREAT

    layers = C.load()["layers"] if layers is None else list(layers)
    fig, axes = plt.subplots(2, len(layers), figsize=(3.0 * len(layers), 6.0), squeeze=False, sharex="row")
    rows = []
    for i, name in enumerate(("d2", "d3")):
        for j, layer in enumerate(layers):
            path = C.results_dir("phase4") / f"{name}_L{layer}.csv"
            if not path.exists():
                continue
            table = pd.read_csv(path).set_index("emotion")
            nulls = np.load(C.data_dir("phase4") / f"{name}_L{layer}_nulls.npz")
            ax = axes[i][j]
            lo, hi, q95 = np.quantile(nulls["random"], [0.025, 0.975, 0.95])
            ax.axvspan(lo, hi, color=GRID, lw=0, label="random directions, 95% band")
            ax.axvline(q95, color=MUTED, lw=1, ls="--", label="random, 95th percentile")
            ax.axvline(np.quantile(nulls["topic"], 0.95), color=INK2, lw=1, ls=":", label="topic, 95th percentile")
            names = [e for e in P8_COMPASSION + P8_THREAT if e in table.index]
            for yi, e in enumerate(names):
                color = SLOTS[0] if e in P8_COMPASSION else SLOTS[1]
                ax.plot(table.loc[e, "t_interaction"], yi, "o", ms=6, color=color, mec=SURFACE, mew=1)
                rows.append({"kind": name, "layer": layer, "emotion": e, "t": table.loc[e, "t_interaction"],
                             "pct_random": table.loc[e, "pct_random"], "pct_topic": table.loc[e, "pct_topic"]})
            ax.set_yticks(range(len(names)), names, fontsize=7)
            ax.set_title(f"{name.upper()}, layer {layer}", fontsize=9)
            ax.set_xlabel("interaction t (known x harmful)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], marker="o", ls="", color=SLOTS[0]), plt.Line2D([], [], marker="o", ls="", color=SLOTS[1])]
    labels += ["compassion cluster", "threat cluster"]
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(_out("d2d3_effects.png"), dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(_out("d2d3_effects.csv"), index=False)


def valence_scatter(prefix: str = "full", label=("afraid", "terrified", "calm", "serene", "guilty", "ashamed",
                                                 "angry", "hostile", "compassionate", "loving", "happy", "desperate"),
                    layers=None) -> None:
    """Emotion vectors on PC1 x PC2 at each layer, coloured by NRC valence (unmatched in hollow gray)."""
    vad = V.vad_table()
    layers = [L for L in (C.load()["layers"] if layers is None else layers)
              if (C.data_dir("bases") / f"{prefix}_L{L}.npz").exists()]
    fig, axes = plt.subplots(1, len(layers), figsize=(3.4 * len(layers), 3.4), squeeze=False)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    cmap = DIVERGING.reversed()  # red = unpleasant (low), blue = pleasant (high)
    rows = []
    for ax, layer in zip(axes[0], layers):
        with np.load(C.data_dir("bases") / f"{prefix}_L{layer}.npz") as z:
            vecs, comps, labels = z["E_vectors"], z["E_components"], list(z["E_labels"])
        xy = vecs @ comps[:2].T
        val = np.array([vad.loc[e, "valence"] if e in vad.index else np.nan for e in labels])
        ok = np.isfinite(val)
        r = np.corrcoef(xy[ok, 0], val[ok])[0, 1]
        ax.scatter(xy[~ok, 0], xy[~ok, 1], s=14, facecolors="none", edgecolors=MUTED, lw=0.8)
        sc = ax.scatter(xy[ok, 0], xy[ok, 1], c=val[ok], cmap=cmap, norm=norm, s=16, edgecolors=SURFACE, lw=0.6)
        span = np.ptp(xy, axis=0)
        placed: list[np.ndarray] = []
        for e in label:  # selective labels; one that would sit on another goes below its point instead
            if e not in labels:
                continue
            k = labels.index(e)
            near = any(np.all(np.abs(xy[k] - p) < 0.06 * span) for p in placed)
            right = xy[k, 0] > xy[:, 0].min() + 0.75 * span[0]
            ax.annotate(e, xy[k], fontsize=7, color=INK2, textcoords="offset points",
                        xytext=(-3 if right else 3, -9 if near else 3), ha="right" if right else "left")
            placed.append(xy[k])
        ax.set_title(f"layer {layer}   r(PC1, valence) = {r:.2f}", fontsize=9)
        ax.set_xlabel("emotion PC1")
        rows += [{"layer": layer, "emotion": e, "pc1": xy[k, 0], "pc2": xy[k, 1], "valence": val[k]}
                 for k, e in enumerate(labels)]
    axes[0][0].set_ylabel("emotion PC2")
    cbar = fig.colorbar(sc, ax=axes[0].tolist(), shrink=0.8, pad=0.01)
    cbar.set_label("NRC valence (hollow gray: not in the lexicon)", color=INK2, fontsize=8)
    cbar.set_ticks([-1, 0, 1], labels=["-1 unpleasant", "0", "+1 pleasant"])
    fig.savefig(_out(f"valence_scatter_{prefix}.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(_out(f"valence_scatter_{prefix}.csv"), index=False)


def all(out: str = "phase3", layers=None, tag: str = "") -> None:  # noqa: A001 (the CLI verb)
    """`out`, `layers` and `tag` draw the post hoc band (`all --out phase3_band --layers 36,40,44 --tag _band`)."""
    global TAG
    TAG = tag
    todo = None if layers is None else C.int_list(layers)
    _style()
    valence_scatter(layers=todo)
    k_sweep(out)
    transfer(out)
    hpkr_subgroups(layers=(20, 52) if todo is None else tuple(todo))
    d2d3_effects(layers=todo)


if __name__ == "__main__":
    _style()
    fire.Fire({"all": all, "k_sweep": k_sweep, "transfer": transfer, "hpkr_subgroups": hpkr_subgroups,
               "d2d3_effects": d2d3_effects, "valence_scatter": valence_scatter})
