"""Average each response's zero-shot score over layers instead of reading one layer.

Every zero-shot result reads a single layer, fixed in advance at 40, and the depth sweep shows that
the choice matters: D9 ranks best at layer 36, and D8's HP-KR operating point is a spike at 40. This
asks whether an average over layers removes the choice without costing accuracy. The rule stays
zero-shot: each layer's score is standardised on Alpaca (the mean and sd of the control's scores at
that layer) and every layer counts equally -- `recall_denial.standardised_sum`'s rule, applied across
layers instead of across directions. The mean of the single-layer results is kept beside it as a
reference: what a layer picked at random scores on average.

    scores    each arm's raw scores on every dataset at the 16 sweep layers, one layer in memory at
              a time, after checking that every layer's rows are the same responses
    analyse   averages over layer sets (all 16, windows around layer 40, each single layer), scored
              zero-shot, at the benchmark threshold and as transfer; paired bootstraps of the
              all-layer average against layer 40 and the best single layer; the single layers
              checked against the stored results
    figure    the single layers beside the all-layer average, the window curve, and how correlated
              D9's scores are between layers

    python deception_detection/scripts/layer_average.py scores
    python deception_detection/scripts/layer_average.py analyse
    python deception_detection/scripts/layer_average.py figure

Outputs go to `results/layer_average/`.
"""

import time

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from deception_detection.scripts import k_robustness as kr
from deception_detection.scripts import persona_ood as P
from deception_detection.scripts import recall_denial as rd
from deception_detection.scripts.k_transfer import ORDER
from deception_detection.scripts.layer_sweep import (ALL_LAYERS, HPKR_SOURCE, LB_SOURCE, TRANSFER_SOURCE,
                                                     base_run, pooled_dir)
from deception_detection.utils import REPO_ROOT

OUT = REPO_ROOT / "results" / "layer_average"
RESULTS = REPO_ROOT / "results"
# D5, D8 and D9 are the 36-persona bank's published, knowledge-denial and extended contrasts, K the
# knowledge direction and D9 + K the selected design. The RepE-trained base probe is the control: does
# averaging help any detector, or only the zero-shot ones?
PERSONA_AXES = {"D5": "deception", "D8": "denial", "D9": "extended"}
ARMS = (*PERSONA_AXES, "K", "D9 + K", "probe")
CENTRE = 40  # the layer fixed in advance
HALF_WIDTHS = (0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 39)  # windows around it; 39 reaches all 16 layers
FPR = 0.01
N_DRAWS = 1000
SEED = 0
TOL = 1e-3
ALIGNED = 0.98  # share of rows closer to their own previous-layer vector than to a random row's


# --- scores ------------------------------------------------------------------------------------------

def _centred_rows(z: np.ndarray) -> np.ndarray:
    """Each row minus the dataset mean, at unit length. Float32: one copy per dataset is kept from the
    previous layer."""
    c = z - z.mean(0)
    return (c / np.linalg.norm(c, axis=1, keepdims=True)).astype(np.float32)


def scores(layers=ALL_LAYERS) -> None:
    """Every arm's raw score on every dataset at each layer, into `scores.npz`.

    Averaging over layers pairs row i of one layer's pooled file with row i of another's, and nothing
    in the files names a row. So each layer's rows are checked against the previous layer's before
    they are used: the labels must match, and at least ALIGNED of the rows must sit closer to their own
    previous-layer vector than to a random row's (cosines of rows centred on their dataset mean).
    Aligned files score 0.996 or more at every layer pair; a shuffled file scores about half. A cut on
    the random rows' 99th percentile instead fails aligned files: HP-KR's templated denials make random
    pairs of its rows alike, and at layers 76 to 79 that percentile passes the same-row median.
    """
    from deception_detection import persona_pcs as pcs

    layers = [int(x) for x in layers]
    OUT.mkdir(parents=True, exist_ok=True)
    stored_k = dict(np.load(rd.OUT / "directions.npz"))
    rng = np.random.default_rng(SEED)
    saved: dict[str, np.ndarray] = {}
    previous: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
    alignment = []
    for layer in layers:
        start = time.time()
        # K rebuilt from the stored reads, as `k_transfer.sweep` rebuilds it
        X, known, _ = kr.item_means(layer)
        k = rd._unit(X.T @ kr.weights(known, np.flatnonzero(known), np.flatnonzero(~known)))
        key = f"L{layer}|none|general"
        if key in stored_k and not np.allclose(k, stored_k[key], atol=1e-6):
            raise AssertionError(f"layer {layer}: rebuilt K differs from the stored one")
        del X
        pooled = P.load_pooled(pooled_dir(layer))
        control = pooled[rd.CONTROL].z
        bank = pcs.load_bank_at_layer(layer, pcs.EXTENDED_VECTORS_DIR)
        directions = {arm: bank.axis_unit(axis) for arm, axis in PERSONA_AXES.items()} | {"K": k}
        directions["D9 + K"] = rd.standardised_sum(directions["D9"], k, control)
        probe = P.run_probe(base_run(layer))
        for name, data in pooled.items():
            for arm, w in directions.items():
                saved[f"{arm}|L{layer}|{name}"] = data.z @ w
            saved[f"probe|L{layer}|{name}"] = probe(data.z)
            rows = _centred_rows(data.z)
            if name in previous:
                before, y, rows_before = previous[name]
                if not np.array_equal(data.y, y):
                    raise AssertionError(f"{name}: labels differ between layers {before} and {layer}")
                same = (rows * rows_before).sum(1)
                other = (rows * rows_before[rng.permutation(len(rows))]).sum(1)
                share = float(np.mean(same > other))
                alignment.append({"dataset": name, "layer_from": before, "layer_to": layer, "n": len(rows),
                                  "same_row_min": float(same.min()), "same_row_median": float(np.median(same)),
                                  "random_row_median": float(np.median(other)),
                                  "random_row_p99": float(np.quantile(other, 0.99)), "share_closer_to_own": share})
                if share < ALIGNED:
                    raise AssertionError(f"{name}: only {share:.3f} of rows at layer {layer} sit closer to their "
                                         f"own layer-{before} vector than to a random row's")
            else:
                saved[f"labels|{name}"] = data.y
            previous[name] = (layer, data.y, rows)
        del pooled, control
        logger.info(f"layer {layer}: {len(directions) + 1} arms scored ({time.time() - start:.0f} s)")
    np.savez_compressed(OUT / "scores.npz", **saved)
    pd.DataFrame(alignment).round(4).to_csv(OUT / "alignment.csv", index=False)
    logger.info(f"wrote {OUT / 'scores.npz'}; rows line up across all {len(layers)} layers")


# --- analysis ----------------------------------------------------------------------------------------

def _load() -> tuple[dict[str, dict[int, dict[str, np.ndarray]]], dict[str, np.ndarray]]:
    """Every arm's scores standardised on Alpaca, as [arm][layer][dataset], and the labels."""
    saved = dict(np.load(OUT / "scores.npz"))
    labels = {key.split("|", 1)[1]: v for key, v in saved.items() if key.startswith("labels|")}
    z: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for arm in ARMS:
        z[arm] = {}
        for layer in ALL_LAYERS:
            c = saved[f"{arm}|L{layer}|{rd.CONTROL}"]
            mu, sd = c.mean(), c.std()
            z[arm][layer] = {name: (saved[f"{arm}|L{layer}|{name}"] - mu) / sd for name in labels}
    return z, labels


def average(by_layer: dict[int, dict[str, np.ndarray]], layers: list[int]) -> dict[str, np.ndarray]:
    """The mean of one arm's standardised scores over `layers`, per dataset."""
    return {name: np.mean([by_layer[layer][name] for layer in layers], axis=0) for name in by_layer[layers[0]]}


def layer_sets() -> list[dict]:
    """Each single layer, the windows of half-width h around layer 40, and all 16 layers. The window
    curve runs from layer 40 alone (h = 0) to all 16 layers (h = 39), so its two ends are the L40 and
    all-16 rows rather than windows of their own."""
    if [layer for layer in ALL_LAYERS if abs(layer - CENTRE) <= HALF_WIDTHS[-1]] != list(ALL_LAYERS):
        raise AssertionError("the widest window does not reach every layer")
    sets = [{"set": f"L{layer}", "kind": "single", "layers": [layer],
             "half_width": 0 if layer == CENTRE else np.nan} for layer in ALL_LAYERS]
    for h in HALF_WIDTHS[1:-1]:
        layers = [layer for layer in ALL_LAYERS if abs(layer - CENTRE) <= h]
        # named by the layers they span: past h = 20 the windows are cut off at layer 20
        sets.append({"set": f"L{layers[0]}-{layers[-1]}", "kind": "window", "half_width": h, "layers": layers})
    sets.append({"set": "all 16", "kind": "all", "layers": list(ALL_LAYERS), "half_width": HALF_WIDTHS[-1]})
    return sets


def metrics(s: dict[str, np.ndarray], y: dict[str, np.ndarray], slices: dict[str, np.ndarray]) -> dict[str, float]:
    """One scorer's numbers, by the thesis's definitions: zero-shot AUROC on each dataset and over the
    five Liars' Bench sets, HP-KR's wording slices, the benchmark's operating point (1% FPR on Alpaca),
    the lies caught at 1% FPR with an in-setting cutoff, and train-one-test-all transfer with the score
    as a one-dimensional feature."""
    row = {rd.SHORT[n]: P.auroc(y[n], s[n]) for n in ORDER}
    row["mean of five"] = float(np.mean([row[rd.SHORT[n]] for n in kr.FIVE]))
    for sname, label in (("identical text", "HP-KR identical"), ("denials", "HP-KR denials")):
        m = slices[sname]
        row[label] = P.auroc(y[P.HPKR][m], s[P.HPKR][m])
    threshold = float(np.quantile(s[rd.CONTROL], 1 - FPR))
    balacc = {n: P.balanced_accuracy(y[n], s[n], threshold) for n in kr.FIVE}
    row["balacc mean of five"] = float(np.mean(list(balacc.values())))
    row["balacc HP-KR"] = balacc[P.HPKR]
    row["recall mean of five"] = float(np.mean([(s[n][y[n] == 1] > threshold).mean() for n in kr.FIVE]))
    # a fresh generator per dataset, so every scorer is cut on the same honest-answer splits
    row["caught in-setting mean of five"] = float(np.mean(
        [kr.caught_at(s[n], y[n], FPR, np.random.default_rng(SEED))[0] for n in kr.FIVE]))
    matrix = P.transfer_matrix(lambda x: x, {n: P.Pooled(n, s[n][:, None], y[n]) for n in ORDER}, ORDER)
    without = [n for n in ORDER if n != P.HPKR]
    row["transfer"] = P.off_diagonal_mean(matrix)
    row["transfer without HP-KR"] = P.off_diagonal_mean(matrix.loc[without, without])
    return row


def check(table: pd.DataFrame) -> None:
    """The single-layer rows must reproduce the stored single-layer results to TOL. Standardising is
    affine, so a one-layer "average" is the stored layer itself; a mismatch means the scores, or their
    pairing with the labels, have gone wrong."""
    single = table[table["kind"] == "single"].set_index(["arm", "set"])
    k_depth = pd.read_csv(kr.OUT / "depth.csv").set_index("layer")
    ninek = pd.read_csv(rd.OUT / "transfer" / "depth_ninek.csv").set_index("layer")
    depth = pd.read_csv(RESULTS / "persona_depth" / "depth.csv")
    labels = {"D5": "deception axis (8v7)", "D8": "knowledge-denial axis (4v6)", "D9": "extended axis (12v13)",
              "probe": "base probe (RepE-trained)"}
    from_depth = ((LB_SOURCE, "auroc", "mean of five"), (LB_SOURCE, "balacc", "balacc mean of five"),
                  (LB_SOURCE, "recall", "recall mean of five"), (HPKR_SOURCE, "auroc", "HP-KR"),
                  (HPKR_SOURCE, "balacc", "balacc HP-KR"), (TRANSFER_SOURCE, "auroc", "transfer"))
    expected = []  # arm, layer, column, stored value, where it is stored
    for layer in ALL_LAYERS:
        for arm, column, stored in (("D5", "mean of five", "D5 alone: mean of five"),
                                    ("D9", "mean of five", "D9 alone: mean of five"),
                                    ("K", "mean of five", "k: mean of five"),
                                    ("K", "HP-KR identical", "k: HP-KR identical"),
                                    ("K", "HP-KR denials", "k: HP-KR denials"),
                                    ("D9 + K", "mean of five", "D9 + k: mean of five")):
            expected.append((arm, layer, column, k_depth.loc[layer, stored], f"k_robustness/depth.csv {stored!r}"))
        for arm, column, stored in (("D9", "transfer", "D9: transfer"),
                                    ("D9", "transfer without HP-KR", "D9: transfer without HP-KR"),
                                    ("D9 + K", "transfer", "D9 + K: transfer"),
                                    ("D9 + K", "transfer without HP-KR", "D9 + K: transfer without HP-KR"),
                                    ("D9 + K", "HP-KR identical", "D9 + K: HP-KR identical"),
                                    ("D9 + K", "HP-KR denials", "D9 + K: HP-KR denials")):
            expected.append((arm, layer, column, ninek.loc[layer, stored], f"transfer/depth_ninek.csv {stored!r}"))
        at = depth[depth["layer"] == layer]
        for arm, label in labels.items():
            for source, metric, column in from_depth:
                value = at[(at["source"] == source) & (at["arm"] == label) & (at["metric"] == metric)]["value"]
                if len(value) and np.isfinite(value.iloc[0]):
                    expected.append((arm, layer, column, float(value.iloc[0]),
                                     f"persona_depth/depth.csv {source}, {metric}"))
    wrong = [f"{arm} L{layer} {column}: {single.loc[(arm, f'L{layer}'), column]:.4f}, stored {value:.4f} ({where})"
             for arm, layer, column, value, where in expected
             if abs(single.loc[(arm, f"L{layer}"), column] - value) > TOL]
    if wrong:
        raise AssertionError("single layers do not reproduce the stored results:\n" + "\n".join(wrong))
    logger.info(f"single layers reproduce all {len(expected)} stored numbers to within {TOL}")


NARROW = ("L36-44", "L32-48")  # the top of the window curve: half-widths 4 and 8 around layer 40
PAIRS = (("all 16", "L40"), ("all 16", "best single"), *((window, "L40") for window in NARROW))


def paired(z: dict, y: dict[str, np.ndarray], slices: dict[str, np.ndarray], best: dict[str, int],
           n_draws: int = N_DRAWS, seed: int = SEED) -> pd.DataFrame:
    """Is the all-layer average better than layer 40 and the best single layer, or only higher? Paired
    stratified row bootstraps, as `k_robustness.paired`: every scorer is scored on the same resamples
    of each dataset and HP-KR slice, and the five-dataset mean is averaged draw by draw. The best
    single layer is picked on these same rows, so that comparison is post hoc and favours the layer.
    So are the narrow windows around layer 40, which were noticed on the window curve of these rows."""
    rng = np.random.default_rng(seed)
    five = [rd.SHORT[n] for n in kr.FIVE]
    spans = {spec["set"]: spec["layers"] for spec in layer_sets()}
    rows = []
    for arm in ARMS:
        scorers = {"all 16": average(z[arm], spans["all 16"]), "L40": z[arm][CENTRE],
                   "best single": z[arm][best[arm]]} | {window: average(z[arm], spans[window]) for window in NARROW}
        sets = {rd.SHORT[n]: (y[n], {k: v[n] for k, v in scorers.items()}) for n in kr.FIVE}
        for sname, label in (("identical text", "HP-KR identical"), ("denials", "HP-KR denials")):
            m = slices[sname]
            sets[label] = (y[P.HPKR][m], {k: v[P.HPKR][m] for k, v in scorers.items()})
        point = {name: {k: P.auroc(labels, v) for k, v in by.items()} for name, (labels, by) in sets.items()}
        draws = {name: kr.paired_draws(labels, by, n_draws, rng) for name, (labels, by) in sets.items()}
        point["mean of five"] = {k: float(np.mean([point[n][k] for n in five])) for k in scorers}
        draws["mean of five"] = {k: np.mean([draws[n][k] for n in five], axis=0) for k in scorers}
        for set_name in point:
            for a, b in PAIRS:
                d = draws[set_name][a] - draws[set_name][b]
                lo, hi = np.quantile(d, [0.025, 0.975])
                rows.append({"arm": arm, "set": set_name, "a": a,
                             "b": f"best single, L{best[arm]} (post hoc)" if b == "best single" else b,
                             "a_auroc": point[set_name][a], "b_auroc": point[set_name][b],
                             "delta": point[set_name][a] - point[set_name][b], "lo": float(lo), "hi": float(hi),
                             "share_not_above": float(np.mean(d <= 0))})
        logger.info(f"{arm}: paired bootstrap done")
    return pd.DataFrame(rows)


def correlations(z: dict, y: dict[str, np.ndarray]) -> pd.DataFrame:
    """Spearman correlation between the layers' standardised scores, within each of the five datasets
    and then averaged over them, over every row and over honest rows only. The honest-row version says
    whether averaging can help: it is how far the layers' fluctuations around the honest level move
    together, and an average cancels only the part that does not."""
    from scipy.stats import spearmanr

    rows = []
    for arm in ARMS:
        per: dict[str, list[np.ndarray]] = {"rho_all": [], "rho_honest": []}
        for n in kr.FIVE:
            M = np.column_stack([z[arm][layer][n] for layer in ALL_LAYERS])
            per["rho_all"].append(spearmanr(M).statistic)
            per["rho_honest"].append(spearmanr(M[y[n] == 0]).statistic)
        mean = {key: np.mean(v, axis=0) for key, v in per.items()}
        for i, a in enumerate(ALL_LAYERS):
            for j, b in enumerate(ALL_LAYERS):
                rows.append({"arm": arm, "layer_a": a, "layer_b": b} | {key: float(m[i, j]) for key, m in mean.items()})
    return pd.DataFrame(rows)


def analyse(n_draws: int = N_DRAWS, seed: int = SEED) -> None:
    """Every arm over every layer set, the paired bootstraps and the layer correlations."""
    from deception_detection.scripts.hpkr_wording import hpkr_frame
    from deception_detection.scripts.knowledge_bank import _slices

    z, y = _load()
    frame = hpkr_frame()
    if not (frame["lie"].to_numpy() == y[P.HPKR]).all():
        raise AssertionError("HP-KR rows do not line up with hpkr_frame")
    slices = _slices(frame)
    for arm in ARMS:
        for layer in ALL_LAYERS:
            c = z[arm][layer][rd.CONTROL]
            if abs(c.mean()) > 1e-9 or abs(c.std() - 1) > 1e-9:
                raise AssertionError(f"{arm} L{layer}: Alpaca scores not standardised")

    rows = []
    for arm in ARMS:
        for spec in layer_sets():
            rows.append({"arm": arm, "set": spec["set"], "kind": spec["kind"], "half_width": spec["half_width"],
                         "n_layers": len(spec["layers"]), "layers": ",".join(map(str, spec["layers"]))}
                        | metrics(average(z[arm], spec["layers"]), y, slices))
        logger.info(f"{arm}: {len(layer_sets())} layer sets scored")
    table = pd.DataFrame(rows)
    check(table)
    measures = [c for c in table.columns if c not in ("arm", "set", "kind", "half_width", "n_layers", "layers")]
    single = table[table["kind"] == "single"]
    reference = single.groupby("arm", sort=False)[measures].mean().reset_index().assign(
        set="mean of single layers", kind="mean of layers", n_layers=len(ALL_LAYERS),
        layers=",".join(map(str, ALL_LAYERS)))
    table = pd.concat([table, reference], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(OUT / "summary.csv", index=False)

    best = {arm: int(single[single["arm"] == arm].sort_values("mean of five")["set"].iloc[-1][1:]) for arm in ARMS}
    pairs = paired(z, y, slices, best, n_draws, seed)
    pairs.round(4).to_csv(OUT / "paired.csv", index=False)
    correlations(z, y).round(4).to_csv(OUT / "correlation.csv", index=False)

    pd.set_option("display.width", 250)
    show = ["mean of five", "transfer", "HP-KR identical", "balacc mean of five", "balacc HP-KR",
            "caught in-setting mean of five"]
    picks = []
    for arm in ARMS:
        by = table[table["arm"] == arm].set_index("set")
        for label, key in (("layer 40", "L40"), (f"best single, L{best[arm]} (post hoc)", f"L{best[arm]}"),
                           ("mean of single layers", "mean of single layers"), ("all 16 averaged", "all 16"),
                           *((f"{window} averaged (post hoc)", window) for window in NARROW)):
            picks.append({"arm": arm, "reading": label} | by.loc[key, show].to_dict())
    logger.info(f"layer 40, the best single layer, the mean of layers and the all-layer average:\n"
                f"{pd.DataFrame(picks).round(3).to_string(index=False)}")
    logger.info(f"paired differences on the five-dataset mean ({n_draws} stratified row resamples):\n"
                f"{pairs[pairs['set'] == 'mean of five'].round(3).to_string(index=False)}")
    logger.info(f"wrote summary.csv, paired.csv and correlation.csv to {OUT}")


# --- figure ------------------------------------------------------------------------------------------

# the thesis's colours for these detectors (paper/make_figures.py, thesis_revised/make_thesis_assets.py)
SERIES = {"D9": ("#2a78d6", "o", "D9, extended contrast"),
          "D9 + K": ("#2E3A4F", "D", "D9 + K, selected design"),
          "probe": ("#898781", "v", "base probe (RepE-trained)")}


def figure(dpi: int = 200) -> None:
    """Three panels: each single layer beside the all-layer average, the average over windows widening
    around layer 40, and the correlation of D9's scores between layers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    from deception_detection.scripts import persona_ood_figures as F

    table = pd.read_csv(OUT / "summary.csv")
    corr = pd.read_csv(OUT / "correlation.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9), facecolor=F.SURFACE,
                             gridspec_kw={"width_ratios": [1.3, 1.0, 0.95]})

    def style(ax, title: str) -> None:
        ax.set_facecolor(F.SURFACE)
        ax.set_title(title, fontsize=10, color=F.INK, loc="left")
        ax.tick_params(colors=F.INK_MUTED, labelsize=8, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # one legend under the two line panels, a column per detector: its line (single layers on the left,
    # windows in the middle) above its all-layer average (the dashed line on the left)
    handles, labels = [], []
    ax = axes[0]
    for arm, (colour, marker, label) in SERIES.items():
        rows = table[(table["arm"] == arm) & (table["kind"] == "single")]
        whole = float(table[(table["arm"] == arm) & (table["set"] == "all 16")]["mean of five"].iloc[0])
        line, = ax.plot(rows["set"].str[1:].astype(int), rows["mean of five"], color=colour, marker=marker,
                        ms=4.5, lw=2)
        dashed = ax.axhline(whole, color=colour, lw=1.3, ls=(0, (5, 3)))
        handles += [line, dashed]
        labels += [label, f"all 16 layers averaged: {whole:.3f}"]
    ax.axvline(CENTRE, color=F.INK_MUTED, lw=1, ls=(0, (1, 2)))
    ax.text(CENTRE + 0.6, 0.98, "layer 40,\nfixed in advance", transform=ax.get_xaxis_transform(),
            fontsize=7.5, color=F.INK_MUTED, va="top")
    ax.set_xticks(list(ALL_LAYERS))
    ax.set_xlabel("layer (block output)", fontsize=9, color=F.INK_MUTED)
    ax.set_ylabel("AUROC, mean of five Liars' Bench sets", fontsize=9, color=F.INK_MUTED)
    ax.grid(color="#e6e5e1", lw=0.6)
    style(ax, "Each layer read alone, against the average over all 16")

    ax = axes[1]
    counts = {}
    for arm, (colour, marker, label) in SERIES.items():
        rows = table[(table["arm"] == arm) & table["half_width"].notna()].sort_values("half_width")
        ax.plot(rows["half_width"], rows["mean of five"], color=colour, marker=marker, ms=4.5, lw=2)
        counts = dict(zip(rows["half_width"].astype(int), rows["n_layers"].astype(int)))
    ax.set_xticks(list(counts), [f"±{h}\n{n}" for h, n in counts.items()])
    ax.set_xlabel("window around layer 40: half-width, then layers averaged", fontsize=9, color=F.INK_MUTED)
    ax.set_ylabel("AUROC, mean of five Liars' Bench sets", fontsize=9, color=F.INK_MUTED)
    ax.grid(color="#e6e5e1", lw=0.6)
    style(ax, "Widening the average around layer 40")

    ax = axes[2]
    d9 = corr[corr["arm"] == "D9"].pivot(index="layer_a", columns="layer_b", values="rho_honest")
    image = ax.imshow(d9.to_numpy(), cmap=F.DIVERGING, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
    ticks = range(len(ALL_LAYERS))
    ax.set_xticks(ticks, [str(layer) for layer in ALL_LAYERS], rotation=90)
    ax.set_yticks(ticks, [str(layer) for layer in ALL_LAYERS])
    ax.set_xlabel("layer", fontsize=9, color=F.INK_MUTED)
    ax.set_ylabel("layer", fontsize=9, color=F.INK_MUTED)
    bar = fig.colorbar(image, ax=ax, shrink=0.85, pad=0.02)
    bar.set_label("Spearman ρ, honest rows, mean of five sets", fontsize=8, color=F.INK_MUTED)
    bar.ax.tick_params(labelsize=7.5, colors=F.INK_MUTED, length=0)
    bar.outline.set_visible(False)
    style(ax, "How D9's scores move together between layers")
    for side in ax.spines.values():
        side.set_visible(False)

    fig.suptitle("Averaging the zero-shot score over layers: Llama-3.3-70B-Instruct, 16 layers, each standardised "
                 "on Alpaca", fontsize=12, color=F.INK, x=0.01, ha="left")
    fig.tight_layout()
    left, right = axes[0].get_position(), axes[1].get_position()
    fig.legend(handles, labels, ncol=len(SERIES), fontsize=8, labelcolor=F.INK, frameon=False, loc="upper center",
               bbox_to_anchor=((left.x0 + right.x1) / 2, -0.01))
    target = OUT / "layer_average.png"
    fig.savefig(target, dpi=int(dpi), bbox_inches="tight", facecolor=F.SURFACE)
    plt.close(fig)
    logger.info(f"wrote {target}")


if __name__ == "__main__":
    fire.Fire({"scores": scores, "analyse": analyse, "figure": figure})
