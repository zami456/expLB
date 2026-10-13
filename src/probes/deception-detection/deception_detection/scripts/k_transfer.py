"""Train-one-test-all transfer for the knowledge direction and the selected design, D9 + K.

The thesis reports this matrix for the persona directions (`persona_ood.py`, stored in
`results/persona_ood_ext_L{layer}/matrix_*.csv`). This scores K and D9 + K the same way. Each is a
fixed direction, so a probe on its one coordinate learns only a sign on the training set and is
then scored on every other set; the diagonal uses the same stratified 5-fold CV. D9 is recomputed
alongside and checked against the stored matrix, so the three are scored identically.

D9 + K is `recall_denial.standardised_sum`: each direction divided by its spread on Alpaca, then
added (`recall_denial.combined_scorers`, "D9 extended + general (standardised)").

    python deception_detection/scripts/k_transfer.py run --layer=40
    python deception_detection/scripts/k_transfer.py sweep            # all 16 sweep layers
    python deception_detection/scripts/k_transfer.py figure --layer=40 --out=<png>

Outputs go to `results/recall_denial/transfer/`.
"""

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from deception_detection.scripts import persona_ood as P
from deception_detection.scripts import recall_denial as rd
from deception_detection.scripts.layer_sweep import pooled_dir
from deception_detection.utils import REPO_ROOT

OUT = rd.OUT / "transfer"
RESULTS = REPO_ROOT / "results"


def stored_dir(layer: int):
    """Where `persona_ood.py` stored the thesis's transfer matrices for this layer."""
    return RESULTS / f"persona_ood_ext_L{layer}"


ORDER = ["convince_me__plain", "harm_refusal_choice__plain", P.HPKR,
         "insider_trading_cadenza__plain", "instruct_dishonesty__plain", "repe_honesty__plain"]
DIRECTIONS = {"ext_axis1": "D9 extended alone", "k": "knowledge general alone",
              "ninek": "D9 extended + general (standardised)"}


def _context(model: str | None, bank: str | None) -> tuple["rd.Inputs", object, list[str]]:
    """The model's inputs, its output folder and its datasets in the thesis's order (RepE last)."""
    inputs = rd.Inputs(model, bank)
    out = OUT if inputs.thesis else inputs.out / "transfer"
    order = ORDER if inputs.llama else [n for n in ORDER if n in (*inputs.datasets, "repe_honesty__plain")]
    return inputs, out, order


def run(layer: int | None = None, model: str | None = None, bank: str | None = None) -> None:
    """`model` and `bank` as `recall_denial.Inputs`; `layer` defaults to the model's mid-depth block."""
    inputs, out, order = _context(model, bank)
    layer = int(layer if layer is not None else inputs.spec.mid)
    pooled = inputs.pooled(layer)
    control = pooled[rd.CONTROL].z
    scorers = rd.combined_scorers(layer, control, inputs)
    without = [n for n in order if n != P.HPKR]
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, name in DIRECTIONS.items():
        w = scorers[name]
        m = P.transfer_matrix(lambda z, w=w: (z @ w)[:, None], pooled, order)
        m.round(4).to_csv(out / f"matrix_{key}_L{layer}.csv")
        rows.append({"direction": key, "layer": layer, "off_diagonal_mean": P.off_diagonal_mean(m),
                     "off_diagonal_excl_hpkr": P.off_diagonal_mean(m.loc[without, without]),
                     "diagonal_mean": P.diagonal_mean(m)})
    table = pd.DataFrame(rows).set_index("direction")
    stored_folder = stored_dir(layer) if inputs.thesis else inputs.spec.results("persona_ood", layer, inputs.bank)
    stored = pd.read_csv(stored_folder / "summary.csv").set_index("features")
    for column in ("off_diagonal_mean", "off_diagonal_excl_hpkr"):
        mine, theirs = table.loc["ext_axis1", column], stored.loc["ext_axis1", column]
        if abs(mine - theirs) > 1e-3:
            raise AssertionError(f"D9 {column}: recomputed {mine:.4f}, stored {theirs:.4f}")
    for key in ("baseline", "random"):
        if key in stored.index:
            table.loc[f"stored {key}"] = [layer, stored.loc[key, "off_diagonal_mean"],
                                          stored.loc[key, "off_diagonal_excl_hpkr"],
                                          stored.loc[key, "diagonal_mean"]]
    table.to_csv(out / f"summary_L{layer}.csv")
    logger.info(f"layer {layer} transfer (D9 checked against the stored matrix):\n{table.round(4).to_string()}")


def sweep(layers=None, model: str | None = None, bank: str | None = None) -> None:
    """D9 and D9 + K at every sweep layer: transfer, HP-KR's wording slices and the five-dataset
    mean. K is rebuilt at each layer from the stored reads (`k_robustness.item_means`), as the
    thesis's depth analysis does; each number is checked against a stored result where one exists.
    Another model's sweep is its `ModelSpec.layers`, checked against its own depth tables
    (`layer_sweep.py depth --model`, `k_robustness.py depth --model`)."""
    from deception_detection.scripts import k_robustness as kr
    from deception_detection.scripts.layer_sweep import ALL_LAYERS, TRANSFER_SOURCE

    inputs, out, order = _context(model, bank)
    layers = [int(x) for x in (layers or (ALL_LAYERS if inputs.llama else inputs.spec.layers))]
    depth_folder = RESULTS / "persona_depth" if inputs.thesis else inputs.spec.results("persona_depth", bank=inputs.bank)
    depth = pd.read_csv(depth_folder / "depth.csv")
    d9_stored = depth[(depth["source"] == TRANSFER_SOURCE) & (depth["metric"] == "auroc")
                      & (depth["bank"] == inputs.bank) & (depth["arm"] == "extended axis (12v13)")].set_index("layer")["value"]
    k_depth = pd.read_csv(kr.out_dir(inputs) / "depth.csv").set_index("layer")
    stored_k = kr._stored_k(inputs)
    hpkr = inputs.hpkr()
    frame, slices = hpkr if hpkr is not None else (None, {})
    without = [n for n in order if n != P.HPKR]
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer in layers:
        X, known, _ = inputs.item_means(layer)
        k = rd._unit(X.T @ kr.weights(known, np.flatnonzero(known), np.flatnonzero(~known)))
        if layer in stored_k and not np.allclose(k, stored_k[layer], atol=1e-6):
            raise AssertionError(f"layer {layer}: rebuilt K differs from the stored one")
        pooled = inputs.pooled(layer)
        control = pooled[rd.CONTROL].z
        d9 = inputs.bank_at(layer).axis_unit("extended")
        ninek = rd.standardised_sum(d9, k, control)
        row = {"layer": layer}
        for name, w in (("D9", d9), ("D9 + K", ninek)):
            m = P.transfer_matrix(lambda z, w=w: (z @ w)[:, None], pooled, order)
            row[f"{name}: transfer"] = P.off_diagonal_mean(m)
            row[f"{name}: transfer without HP-KR"] = P.off_diagonal_mean(m.loc[without, without])
        if abs(row["D9: transfer"] - d9_stored[layer]) > 1e-3:
            raise AssertionError(f"layer {layer}: D9 transfer {row['D9: transfer']:.4f}, stored {d9_stored[layer]:.4f}")
        if frame is not None:
            y = pooled[P.HPKR].y
            if not (frame["lie"].to_numpy() == y).all():
                raise AssertionError(f"layer {layer}: HP-KR rows do not line up with hpkr_frame")
            s = pooled[P.HPKR].z @ ninek
            for sname, label in (("identical text", "HP-KR identical"), ("denials", "HP-KR denials")):
                if sname in slices:
                    m = slices[sname]
                    row[f"D9 + K: {label}"] = P.auroc(y[m], s[m])
        row["D9 + K: mean of five"] = float(np.mean([P.auroc(pooled[n].y, pooled[n].z @ ninek) for n in inputs.datasets]))
        if abs(row["D9 + K: mean of five"] - k_depth.loc[layer, "D9 + k: mean of five"]) > 1e-3:
            raise AssertionError(f"layer {layer}: D9 + K mean of five {row['D9 + K: mean of five']:.4f}, "
                                 f"k_robustness {k_depth.loc[layer, 'D9 + k: mean of five']:.4f}")
        rows.append(row)
        logger.info(f"layer {layer}: " + ", ".join(f"{k2} {v:.3f}" for k2, v in row.items() if k2 != "layer"))
        del pooled
    table = pd.DataFrame(rows)
    table.round(4).to_csv(out / "depth_ninek.csv", index=False)
    if inputs.thesis and 40 in table["layer"].values:
        at40 = table.set_index("layer").loc[40]
        for label, thesis in (("HP-KR identical", 0.787), ("HP-KR denials", 0.724)):
            if abs(round(at40[f"D9 + K: {label}"], 3) - thesis) > 1e-9:
                raise AssertionError(f"L40 D9 + K {label} {at40[f'D9 + K: {label}']:.4f}, thesis {thesis}")
    logger.info(f"wrote {out / 'depth_ninek.csv'}")


def d5k(layers=None, model: str | None = None, bank: str | None = None) -> None:
    """Train-one-test-all transfer for D5 + K and for K alone, beside D9 + K, at the model's 20, 36,
    40 and 44 (`ModelSpec.named_layers`): `matrix_d5k_L{layer}.csv`, `matrix_kalone_L{layer}.csv` and
    their off-diagonal means in `summary_d5k.csv`. D5 + K is `recall_denial.d5_plus_k`'s: the
    deception axis and K, each divided by its spread on Alpaca, then added. D9 + K and K are scored
    the same way and must reproduce `run`'s stored matrices where they exist."""
    inputs, out, order = _context(model, bank)
    layers = [int(x) for x in (layers or inputs.spec.named_layers)]
    without = [n for n in order if n != P.HPKR]
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for layer in layers:
        pooled = inputs.pooled(layer)
        control = pooled[rd.CONTROL].z
        bank_l, k = inputs.bank_at(layer), inputs.k(layer)
        scorers = {"d5k": rd.standardised_sum(bank_l.axis_unit("deception"), k, control), "kalone": k,
                   "ninek": rd.standardised_sum(bank_l.axis_unit("extended"), k, control)}
        row = {"layer": layer}
        for key, w in scorers.items():
            m = P.transfer_matrix(lambda z, w=w: (z @ w)[:, None], pooled, order)
            stored = out / f"matrix_{'k' if key == 'kalone' else key}_L{layer}.csv"
            if key in ("ninek", "kalone") and stored.exists():
                s = pd.read_csv(stored, index_col=0)
                gap = float(np.abs(m.loc[s.index, s.columns].to_numpy(float) - s.to_numpy(float)).max())
                if gap > 1e-3:
                    raise AssertionError(f"layer {layer} {key}: recomputed matrix differs from {stored.name} by {gap:.4f}")
            if key != "ninek":
                m.round(4).to_csv(out / f"matrix_{key}_L{layer}.csv")
            row[f"{key}: transfer"] = P.off_diagonal_mean(m)
            row[f"{key}: transfer without HP-KR"] = P.off_diagonal_mean(m.loc[without, without])
        rows.append(row)
        logger.info(f"layer {layer}: " + ", ".join(f"{k2} {v:.3f}" for k2, v in row.items() if k2 != "layer"))
        del pooled
    pd.DataFrame(rows).round(4).to_csv(out / "summary_d5k.csv", index=False)
    logger.info(f"wrote {out / 'summary_d5k.csv'}")


def figure(layer: int = 40, out: str | None = None, dpi: int = 300) -> None:
    """The thesis's transfer figure (paper/make_figures.py, transfer_matrices) with D9 + K on the right."""
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    from deception_detection.scripts import persona_ood_figures as F

    layer = int(layer)
    mpl.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "pdf.fonttype": 42,
        "axes.edgecolor": F.INK_MUTED, "axes.labelcolor": F.INK, "text.color": F.INK,
        "xtick.color": F.INK_MUTED, "ytick.color": F.INK_MUTED,
    })
    panels = ((stored_dir(layer) / "matrix_baseline.csv", "Probe on raw activations (8192-d)"),
              (OUT / f"matrix_ninek_L{layer}.csv", "D9 + K, the selected design (1-d)"))
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.85), constrained_layout=True)
    for ax, (path, title) in zip(axes, panels, strict=True):
        m = pd.read_csv(path, index_col=0).loc[ORDER, ORDER]
        values = m.to_numpy(float).T  # rows = test set, columns = training set
        image = ax.imshow(values, cmap=F.DIVERGING, norm=norm)
        for i in range(len(ORDER)):
            for j in range(len(ORDER)):
                v = values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v - 0.5) > 0.33 else F.INK)
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, lw=1.1, edgecolor=F.INK))
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
    target = out or str(OUT / f"fig_transfer_ninek_L{layer}.png")
    fig.savefig(target, dpi=int(dpi), bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    logger.info(f"wrote {target}")


if __name__ == "__main__":
    fire.Fire({"run": run, "sweep": sweep, "d5k": d5k, "figure": figure})
