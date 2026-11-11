"""Confidence intervals for the cross-model transfer claims: paired, stratified bootstraps of the
train-one-test-all transfer (off-diagonal mean AUROC) and of the zero-shot mean over the model's
datasets.

Every detector is fitted as `persona_ood.transfer_matrix` fits it: the raw probe is the
standardised logistic regression on the block's activations, trained on all of one dataset; a fixed
direction (D5, D9, D5 + K, D9 + K) gets the same probe on its one coordinate, which learns a sign.
Each fitted probe's scores on every other dataset are kept. A draw resamples each test dataset's
lies and honest rows with replacement (`k_robustness.paired_draws`'s scheme) and recomputes every
off-diagonal cell, for every detector, from that same resample, so a difference between two
detectors is paired draw by draw. The fitted probes are held fixed: an interval covers which test
rows were sampled, not which training rows. The point estimates must reproduce the stored transfer
summaries (`persona_ood`, `k_transfer.py run/sweep/d5k`) within 1e-3.

Comparisons, at each model's four thesis layers (its primary persona bank, or `--bank`):
    transfer:      D9 - raw probe, D9 - D5, D9 + K - raw probe, D9 + K - D9, D5 + K - D5,
                   D9 + K - D5 + K
    dataset mean:  D9 - D5 (zero-shot, the model's Liars' Bench datasets)

    python deception_detection/scripts/transfer_ci.py run --model mistral
    python deception_detection/scripts/transfer_ci.py run --model llama

Outputs go to `results/xmodel/<model>/transfer_ci.csv`.
"""

import time

import fire
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from deception_detection.log import logger
from deception_detection.scripts import lb_models as LB
from deception_detection.scripts import recall_denial as rd

N_DRAWS = 1000
SEED = 0
TOL = 1e-3
TRANSFER = (("D9", "raw probe"), ("D9", "D5"), ("D9 + K", "raw probe"), ("D9 + K", "D9"), ("D5 + K", "D5"),
            ("D9 + K", "D5 + K"))
MEAN = (("D9", "D5"),)


def _auroc(y: np.ndarray, s: np.ndarray) -> float:
    """AUROC by ranks (ties averaged): roc_auc_score's value, faster on many resamples."""
    r = rankdata(s)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _stored(model: str, layer: int, inputs) -> dict[str, float]:
    """The stored off-diagonal means this must reproduce."""
    s = inputs.spec
    out = {}
    summary = pd.read_csv(s.results("persona_ood", layer, inputs.bank) / "summary.csv").set_index("features")
    for name, feature in (("raw probe", "baseline"), ("D9", "ext_axis1"), ("D5", "axis1")):
        out[name] = float(summary.loc[feature, "off_diagonal_mean"])
    from deception_detection.scripts import k_transfer as ktr

    folder = ktr._context(None if inputs.thesis else model, None if inputs.thesis else inputs.bank)[1]
    ninek = pd.read_csv(folder / "depth_ninek.csv").set_index("layer")
    if layer in ninek.index:
        out["D9 + K"] = float(ninek.loc[layer, "D9 + K: transfer"])
    d5k = folder / "summary_d5k.csv"
    if d5k.exists():
        t = pd.read_csv(d5k).set_index("layer")
        if layer in t.index:
            out["D5 + K"] = float(t.loc[layer, "d5k: transfer"])
    return out


def _scores(model: str, layer: int, inputs, order: list[str]):
    """Each detector's fitted scores: {detector: {(train, test): scores}}, and the fixed directions."""
    from deception_detection.scripts import persona_ood as P

    pooled = inputs.pooled(layer)
    control = pooled[rd.CONTROL].z
    bank, k = inputs.bank_at(layer), inputs.k(layer)
    d5, d9 = bank.axis_unit("deception"), bank.axis_unit("extended")
    dirs = {"D5": d5, "D9": d9, "D5 + K": rd.standardised_sum(d5, k, control),
            "D9 + K": rd.standardised_sum(d9, k, control)}
    data = {n: (pooled[n].y.astype(int), pooled[n].z) for n in order}
    scores: dict[str, dict] = {}
    started = time.time()
    raw = {}
    for train in order:
        probe = P.make_probe()
        probe.fit(data[train][1], data[train][0])
        for test in order:
            if test != train:
                raw[(train, test)] = probe.decision_function(data[test][1])
    scores["raw probe"] = raw
    logger.info(f"{model} block {layer}: raw probes fitted in {time.time() - started:.0f} s")
    for name, w in dirs.items():
        cells = {}
        for train in order:
            probe = P.make_probe()
            probe.fit((data[train][1] @ w)[:, None], data[train][0])
            for test in order:
                if test != train:
                    cells[(train, test)] = probe.decision_function((data[test][1] @ w)[:, None])
        scores[name] = cells
    labels = {n: data[n][0] for n in order}
    zero_shot = {name: {n: data[n][1] @ w for n in inputs.datasets} for name, w in dirs.items()}
    del pooled
    return scores, labels, zero_shot


def run(model: str, layers=None, n_draws: int = N_DRAWS, seed: int = SEED, bank: str | None = None) -> None:
    """`bank` as `recall_denial.Inputs` (the model's primary persona bank by default). Another bank's
    intervals go to `transfer_ci_<bank>.csv`, beside the primary bank's `transfer_ci.csv`."""
    from deception_detection.scripts import k_transfer as ktr

    key = LB.spec(model).key
    inputs, _, order = ktr._context(None if key == "llama" else model, bank)
    names = inputs.spec.named_layers
    layers = [int(x) for x in (layers or names)]
    rng = np.random.default_rng(seed)
    rows = []
    for layer in layers:
        scores, labels, zero_shot = _scores(model, layer, inputs, order)
        cells = [(a, b) for a in order for b in order if a != b]
        point = {d: float(np.mean([_auroc(labels[b], scores[d][(a, b)]) for a, b in cells])) for d in scores}
        for d, value in _stored(model, layer, inputs).items():
            if abs(point[d] - value) > TOL:
                raise AssertionError(f"{model} block {layer}: {d} transfer {point[d]:.4f}, stored {value:.4f}")
        means = {d: float(np.mean([_auroc(labels[n], s) for n, s in zero_shot[d].items()])) for d in zero_shot}
        draws = {d: np.empty(n_draws) for d in scores}
        mean_draws = {d: np.empty(n_draws) for d in zero_shot}
        strata = {n: (np.flatnonzero(labels[n] == 1), np.flatnonzero(labels[n] == 0)) for n in order}
        started = time.time()
        for b in range(n_draws):
            idx = {n: np.concatenate([rng.choice(p, len(p)), rng.choice(q, len(q))]) for n, (p, q) in strata.items()}
            for d in scores:
                draws[d][b] = np.mean([_auroc(labels[t][idx[t]], scores[d][(a, t)][idx[t]]) for a, t in cells])
            for d in zero_shot:
                mean_draws[d][b] = np.mean([_auroc(labels[n][idx[n]], s[idx[n]]) for n, s in zero_shot[d].items()])
        logger.info(f"{model} block {layer}: {n_draws} draws in {time.time() - started:.0f} s")
        name = {v: k for k, v in names.items()}.get(layer, f"B{layer}") if isinstance(names, dict) else \
            dict(zip(names, ("L20", "L36", "L40", "L44"))).get(layer, f"B{layer}")
        for metric, pairs, pt, dr in (("transfer", TRANSFER, point, draws), ("dataset mean", MEAN, means, mean_draws)):
            for a, b in pairs:
                d = dr[a] - dr[b]
                lo, hi = np.quantile(d, [0.025, 0.975])
                rows.append({"model": key, "layer": name, "block": layer, "metric": metric, "a": a, "b": b,
                             "a_value": pt[a], "b_value": pt[b], "delta": pt[a] - pt[b], "lo": float(lo),
                             "hi": float(hi), "share_not_above": float(np.mean(d <= 0)), "n_draws": n_draws})
        logger.info(f"{model} block {layer} done")
    table = pd.DataFrame(rows)
    primary = inputs.bank == inputs.spec.primary_bank
    out = LB.XMODEL_RESULTS / key / ("transfer_ci.csv" if primary else f"transfer_ci_{inputs.bank}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(out, index=False)
    pd.set_option("display.width", 250)
    logger.info(f"wrote {out}:\n{table.round(3).to_string(index=False)}")


if __name__ == "__main__":
    fire.Fire({"run": run})
