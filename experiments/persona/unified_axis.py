"""The unified axis as the control for the extended axis, across depth and both persona folders.

The persona paper (arXiv 2605.09391, App. C) builds a *unified* harmful-behaviour axis by pooling
its deception and sycophancy personas over a balanced question set. This study builds an
*extended* axis the same way but with knowledge-denial roles instead. Both widen the published
8v7 deception contrast by five roles a side, so putting them next to each other separates two
explanations of the extended axis's gain:

    a wider persona set helps           -> the unified axis should gain too
    these particular roles help         -> only the extended axis should gain

Both folders are swept because they answer different questions. `orig` (`persona vectors/
vectors_s3/`, the published 26) gives the paper's own unified axis; its layer-20 slice is
bit-identical to `paper16_L20.pt`. `ext` (`data/persona_banks/extended_kd/`, 36 personas)
rebuilds it on the same extraction as the extended axis, which is the only comparison free of the
"more rollouts" confound documented in `results/persona_ood_ext_L20/summary.md`.

    python experiments/persona/unified_axis.py report
"""

from pathlib import Path

import fire
import numpy as np

from deception_detection.log import logger
from experiments.persona.layer_sweep import ALL_LAYERS, BANKS, results_dir
from deception_detection.utils import REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "persona_unified"

# transfer feature -> zero-shot column -> label. The four one-dimensional contrast axes; the PCs
# and the 8192-d baseline are in each layer's own summary.csv and not repeated here.
ARMS = {
    "axis1": ("auroc_axis", "deception axis (8v7)", "#eb6834"),
    "uni_axis1": ("auroc_uni_axis", "unified axis (13v12)", "#b5399e"),
    "denial_axis1": ("auroc_denial_axis", "knowledge-denial axis (4v6)", "#8ab6e8"),
    "ext_axis1": ("auroc_ext_axis", "extended axis (12v13)", "#2a78d6"),
}
BANK_LABEL = {"ext": "36-persona extraction", "orig": "published 26 personas"}


def table(layers=ALL_LAYERS, banks=BANKS) -> "pd.DataFrame":
    """Long format: bank, layer, arm, metric, value. Missing arms come back absent, not zero."""
    import pandas as pd

    from experiments.persona.persona_ood import HPKR, LB_AVERAGE

    rows = []
    for bank in banks:
        for layer in layers:
            out = results_dir(layer, bank)
            if not (out / "summary.csv").exists():
                logger.warning(f"no results at {out}; skipped")
                continue
            transfer = pd.read_csv(out / "summary.csv").set_index("features")
            zero = pd.read_csv(out / "zero_shot.csv").set_index("dataset")
            operating = pd.read_csv(out / "zero_shot_operating.csv")
            operating = operating[operating["threshold_source"] == "control"]
            for feature, (column, label, _) in ARMS.items():
                if feature not in transfer.index:
                    continue  # this folder has no roles for that contrast
                values = {
                    "transfer_all_six": float(transfer.loc[feature, "off_diagonal_mean"]),
                    "transfer_excl_hpkr": float(transfer.loc[feature, "off_diagonal_excl_hpkr"]),
                    "zeroshot_hpkr_auroc": float(zero.loc[HPKR, column]),
                }
                row = operating[(operating["arm"] == label) & (operating["dataset"] == LB_AVERAGE)]
                if len(row):
                    values["zeroshot_lb_auroc"] = float(row["auroc"].iloc[0])
                    values["zeroshot_lb_balacc"] = float(row["balacc"].iloc[0])
                    hpkr_row = operating[(operating["arm"] == label) & (operating["dataset"] == HPKR)]
                    if len(hpkr_row):
                        values["zeroshot_hpkr_balacc"] = float(hpkr_row["balacc"].iloc[0])
                for metric, value in values.items():
                    rows.append({"bank": bank, "layer": layer, "arm": label,
                                 "metric": metric, "value": value})
    return pd.DataFrame(rows)


PANELS = [
    ("transfer_all_six", "Transfer: off-diagonal AUROC, all six datasets"),
    ("transfer_excl_hpkr", "Transfer: off-diagonal AUROC, excluding HP-KR"),
    ("zeroshot_hpkr_auroc", "Zero-shot: HP-KR AUROC"),
    ("zeroshot_lb_auroc", "Zero-shot: Liars' Bench average AUROC"),
]


def figure(data, out_dir: Path) -> None:
    """One row of panels per persona folder; the arms share colours with every other figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.persona.persona_ood_figures import INK, INK_MUTED, SURFACE

    banks = [b for b in BANKS if b in set(data["bank"])]
    fig, axes = plt.subplots(len(banks), len(PANELS), figsize=(4.0 * len(PANELS), 3.4 * len(banks)),
                             facecolor=SURFACE, squeeze=False)
    for row, bank in enumerate(banks):
        for col, (metric, title) in enumerate(PANELS):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            for _, (_, label, colour) in ARMS.items():
                rows = data[(data.bank == bank) & (data.arm == label) & (data.metric == metric)]
                if rows.empty:
                    continue
                rows = rows.sort_values("layer")
                ax.plot(rows["layer"], rows["value"], marker="o", ms=4, lw=2,
                        color=colour, label=label)
            ax.axhline(0.5, color=INK_MUTED, lw=1, ls=(0, (4, 3)))
            ax.set_title(f"{title}\n{BANK_LABEL[bank]}", fontsize=9, color=INK, loc="left")
            ax.set_xticks(list(ALL_LAYERS))
            ax.tick_params(colors=INK_MUTED, labelsize=7, length=0)
            ax.grid(color="#e6e5e1", lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if col == 0:
                ax.legend(fontsize=7, labelcolor=INK, frameon=True, facecolor=SURFACE,
                          edgecolor="none", framealpha=0.95)
            if row == len(banks) - 1:
                ax.set_xlabel("layer (block output)", fontsize=8, color=INK_MUTED)
    fig.suptitle(
        "Widening the deception contrast: with sycophancy (unified) or with knowledge denial "
        "(extended)", fontsize=12, color=INK, x=0.06, ha="left")
    fig.tight_layout()
    fig.savefig(out_dir / "unified_axis.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    logger.info(f"wrote {out_dir / 'unified_axis.png'}")


def _best(data, bank: str, label: str, metric: str) -> tuple[float, int]:
    rows = data[(data.bank == bank) & (data.arm == label) & (data.metric == metric)]
    if rows.empty:
        return float("nan"), -1
    best = rows.loc[rows["value"].idxmax()]
    return float(best["value"]), int(best["layer"])


def report(layers=ALL_LAYERS, banks=BANKS, out_dir: str | Path = OUT_DIR) -> None:
    """`unified.csv`, `unified_axis.png` and a summary, into `results/persona_unified/`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = table(layers, banks)
    data.to_csv(out_dir / "unified.csv", index=False)
    figure(data, out_dir)

    lines = []
    for bank in [b for b in BANKS if b in set(data["bank"])]:
        lines.append(f"\n### {BANK_LABEL[bank]} (`{bank}`)\n")
        lines.append("| arm | best transfer AUROC (layer) | best excl. HP-KR (layer) | "
                     "best HP-KR zero-shot AUROC (layer) |")
        lines.append("|---|---|---|---|")
        for _, (_, label, _) in ARMS.items():
            cells = []
            for metric in ("transfer_all_six", "transfer_excl_hpkr", "zeroshot_hpkr_auroc"):
                value, layer = _best(data, bank, label, metric)
                cells.append("n/a" if not np.isfinite(value) else f"{value:.3f} (L{layer})")
            if cells != ["n/a"] * 3:
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
    (out_dir / "summary_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"wrote {out_dir / 'unified.csv'} and summary_tables.md\n" + "\n".join(lines))


if __name__ == "__main__":
    fire.Fire({"table": table, "report": report})
