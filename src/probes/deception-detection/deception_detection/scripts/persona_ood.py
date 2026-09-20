"""Do persona PCs improve out-of-distribution probe generalization? (arXiv 2605.09391)

Three methods from the paper, all against the pooled layer-20 activations:

  1. contrastive direction, zero-shot   s(z) = <z, d_hat>
  2. PC1, zero-shot                     s(z) = <z, u_1>
  3. top-3 PC features -> probe         trained on [<z,u_1>, <z,u_2>, <z,u_3>]

and the claim itself: a train/test AUROC matrix over datasets, comparing probes on the full
8192-d pooled activations against probes on the 3-d persona projection. The paper reports gains
off the diagonal (cross-dataset) with some loss on it (within-dataset).

Two controls decide whether any gain means what it appears to:

  - random 3-d projections. If the persona PCs do not beat random directions of the same rank,
    the effect is dimensionality reduction, not persona structure.
  - axis-only, 1-d. PC1 is nearly the deception axis (cos 0.95), so this separates "the deception
    direction carries it" from "PC2 and PC3 add something".

    python deception_detection/scripts/persona_ood.py
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from deception_detection import persona_pcs
from deception_detection.log import logger
from deception_detection.utils import REPO_ROOT

POOLED_DIR = REPO_ROOT / "activations" / "pooled"
OUT_DIR = REPO_ROOT / "results" / "persona_ood"

# honest-only, so it cannot be a train/test cell; kept as a zero-shot reference
CONTROL_DATASET = "XXXX-20__plain"

# The experiment splits RepE 80/20 into train and a "_val" remainder, but both halves are the
# same 306 underlying facts, so for transfer purposes they are one dataset. Merging them gives
# the RepE row the training protocol the paper actually uses (and the deployed probe was fitted
# with) instead of the 102-row remainder. Keeping them apart would also create a train->val cell
# that looks like transfer while being purely in-distribution.
MERGE_INTO = {"repe_honesty__plain_val": "repe_honesty__plain"}

# inverted as a test set and poisonous as a training set; aggregates are reported without it too
HPKR = "harm_refusal_knowledge_report__plain"

# matches the deployed probe (detectors.py:311-376): standardise, then L2 logistic regression
# with reg_coeff 10 and no intercept
REG_COEFF = 10.0
MAX_ITER = 1000
N_RANDOM_DRAWS = 5
SEED = 42


# --- Liars' Bench probe baselines, transcribed ------------------------------------------------
# Llama-3.3-70B rows of Tables 6-8 (arXiv 2511.16035), columns in the paper's order. These are
# READ FROM THE PAPER, not recomputed: none can be reproduced from the activation cache. The
# Follow-Up probe rewrites every input (experiment.py:422-432 appends a follow-up question and a
# prefilled "No", and scores that one token), so its token ids -- and therefore every cache key --
# differ; it needs a fresh remote extraction. The Upper-Bound probe additionally trains on
# privileged in-dataset examples, so it is the paper's ceiling estimate rather than a baseline.
PAPER_COLUMNS = ("HP-C", "HP-KR", "CG", "ST", "ID", "GS", "IT")
PAPER_PROBES = {
    "Mean Probe": {
        "source": "Table 6 p.54",
        "privileged": False,
        "auroc": (0.41, 0.12, 1.00, 0.57, 0.95, 0.45, 0.93),
        "balacc": (0.50, 0.50, 0.83, 0.50, 0.85, 0.49, 0.50),
    },
    "Follow-Up Probe": {
        "source": "Table 7 p.55",
        "privileged": False,
        "auroc": (0.50, 0.14, 0.75, 0.61, 0.93, 0.24, 0.85),
        "balacc": (0.50, 0.26, 0.60, 0.57, 0.87, 0.35, 0.66),
    },
    "Upper-Bound Probe": {
        "source": "Table 8 p.56",
        "privileged": True,
        "auroc": (0.74, 0.97, 1.00, float("nan"), 0.95, float("nan"), 0.99),
        "balacc": (0.67, 0.95, 1.00, float("nan"), 0.90, float("nan"), 0.93),
    },
}
# the five Liars' Bench subsets this work covers, as named here and in the paper
SHARED_DATASETS = {
    "convince_me__plain": "CG",
    "harm_refusal_choice__plain": "HP-C",
    "harm_refusal_knowledge_report__plain": "HP-KR",
    "insider_trading_cadenza__plain": "IT",
    "instruct_dishonesty__plain": "ID",
}


@dataclass
class Pooled:
    name: str
    z: np.ndarray  # [n, hidden] pooled activations
    y: np.ndarray  # [n] 1 = deceptive

    @property
    def n_deceptive(self) -> int:
        return int(self.y.sum())

    @property
    def two_class(self) -> bool:
        return 0 < self.n_deceptive < len(self.y)


def load_pooled(pooled_dir: Path = POOLED_DIR, merge: bool = True) -> dict[str, Pooled]:
    """Load every pooled dataset, keeping only HONEST/DECEPTIVE rows.

    `merge=False` keeps the RepE train and "_val" splits apart, for callers that need the split
    itself (INLP fits on one half and scores on the other) rather than the dataset as a whole.
    """
    parts: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for path in sorted(pooled_dir.glob("*.npz")):
        raw = np.load(path, allow_pickle=False)
        labels = raw["labels"].astype(str)
        keep = np.isin(labels, ["HONEST", "DECEPTIVE"])
        name = str(raw["name"])
        dropped = int((~keep).sum())
        if dropped:
            logger.info(f"{name}: dropping {dropped} rows that are neither honest nor deceptive")
        target = MERGE_INTO.get(name, name) if merge else name
        parts.setdefault(target, []).append(
            (raw["pooled"][keep].astype(np.float64), (labels[keep] == "DECEPTIVE").astype(int))
        )

    out: dict[str, Pooled] = {}
    for name, pieces in parts.items():
        if len(pieces) > 1:
            logger.info(f"{name}: merged {len(pieces)} splits ({[len(y) for _, y in pieces]} rows)")
        out[name] = Pooled(
            name=name,
            z=np.concatenate([z for z, _ in pieces]),
            y=np.concatenate([y for _, y in pieces]),
        )
    return out


def make_probe() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1 / REG_COEFF, fit_intercept=False, max_iter=MAX_ITER),
    )


def auroc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def balanced_accuracy(y: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """(recall + specificity) / 2 at a fixed threshold -- Liars' Bench's headline metric.

    Note the floor. When every score falls below the threshold, recall is 0 and specificity is 1,
    giving exactly 0.5 -- and a probe that is strongly *anti*-correlated lands there too, since it
    also flags nothing. So this cannot distinguish "inverted" from "no signal", which is why it is
    reported beside AUROC rather than instead of it.
    """
    if len(set(y.tolist())) < 2:
        return float("nan")
    recall = float((scores[y == 1] > threshold).mean())
    specificity = float((scores[y == 0] <= threshold).mean())
    return (recall + specificity) / 2


@dataclass(frozen=True)
class Metric:
    """A cell scorer, plus whether it needs a decision threshold.

    Threshold-free metrics can pool out-of-fold scores across a CV split and score once.
    Thresholded ones cannot: each fold's probe has its own scale, so their scores are not
    comparable and the metric has to be computed per fold and averaged.
    """

    name: str
    fn: Callable[[np.ndarray, np.ndarray, float], float]
    needs_threshold: bool


AUROC = Metric("auroc", lambda y, scores, _threshold: auroc(y, scores), needs_threshold=False)
BALANCED_ACCURACY = Metric("balacc", balanced_accuracy, needs_threshold=True)


def control_threshold(probe: object, control_features: np.ndarray, fpr: float = 0.01) -> float:
    """The score above which `fpr` of the control dataset is flagged."""
    scores = probe.decision_function(control_features)  # type: ignore[attr-defined]
    return float(np.quantile(scores, 1 - fpr))


def diagonal_auroc(
    features: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    metric: "Metric" = AUROC,
    control_features: np.ndarray | None = None,
) -> float:
    """Within-dataset score by stratified CV, so a diagonal cell is never train-on-test.

    For a threshold-free metric the out-of-fold scores are pooled and scored once. For a
    thresholded one they cannot be: every fold has its own probe and therefore its own scale, so
    the metric is computed per fold -- against a threshold taken from that same fold's probe --
    and averaged. Reusing a whole-dataset threshold inside CV would leak the held-out rows.
    """
    n_splits = min(n_splits, int(y.sum()), int((1 - y).sum()))
    if n_splits < 2:
        return float("nan")
    if metric.needs_threshold and control_features is None:
        raise ValueError(f"metric {metric.name!r} needs a control dataset for its threshold")

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    pooled = np.zeros(len(y))
    per_fold: list[float] = []

    for train, test in splitter.split(features, y):
        probe = make_probe()
        probe.fit(features[train], y[train])  # type: ignore[attr-defined]
        scores = probe.decision_function(features[test])  # type: ignore[attr-defined]
        if metric.needs_threshold:
            assert control_features is not None
            per_fold.append(metric.fn(y[test], scores, control_threshold(probe, control_features)))
        else:
            pooled[test] = scores

    if metric.needs_threshold:
        return float(np.mean(per_fold))
    return metric.fn(y, pooled, float("nan"))


def transfer_matrix(
    featurize: Callable[[np.ndarray], np.ndarray],
    datasets: dict[str, Pooled],
    order: list[str],
    metric: "Metric" = AUROC,
    control: Pooled | None = None,
) -> pd.DataFrame:
    """Score every (train, test) pair. Diagonal uses CV; off-diagonal trains on all of A.

    For a thresholded metric each *row* carries its own threshold, taken from that row's probe
    scored on the control dataset. A single shared threshold would be meaningless: probes fitted
    on different datasets put their decision function on different scales.
    """
    if metric.needs_threshold and control is None:
        raise ValueError(f"metric {metric.name!r} needs a control dataset for its threshold")

    feats = {name: featurize(datasets[name].z) for name in order}
    control_features = featurize(control.z) if control is not None else None
    matrix = pd.DataFrame(index=order, columns=order, dtype=float)

    for train_name in order:
        probe = make_probe()
        probe.fit(feats[train_name], datasets[train_name].y)  # type: ignore[attr-defined]
        threshold = (
            control_threshold(probe, control_features)
            if metric.needs_threshold and control_features is not None
            else float("nan")
        )
        for test_name in order:
            if train_name == test_name:
                matrix.loc[train_name, test_name] = diagonal_auroc(
                    feats[train_name],
                    datasets[train_name].y,
                    metric=metric,
                    control_features=control_features,
                )
            else:
                scores = probe.decision_function(feats[test_name])  # type: ignore[attr-defined]
                matrix.loc[train_name, test_name] = metric.fn(
                    datasets[test_name].y, scores, threshold
                )
    return matrix


@dataclass(frozen=True)
class LOOScores:
    """One leave-one-out sweep: the held-out scores, and each fold's control threshold.

    Kept as scores rather than a metric so that AUROC and balanced accuracy can both be read off
    a single set of fits. The baseline arm fits an 8192-d probe on ~9500 rows six times over;
    refitting it once per metric would double that for nothing.
    """

    scores: dict[str, np.ndarray]
    thresholds: dict[str, float]


def leave_one_out_scores(
    featurize: Callable[[np.ndarray], np.ndarray],
    datasets: dict[str, "Pooled"],
    order: list[str],
    control: "Pooled | None" = None,
    weighting: str = "naive",
) -> LOOScores:
    """Train on the pooled other datasets, score the held-out one.

    The off-diagonal of `transfer_matrix` trains on a *single* dataset, which is not how anyone
    would use labels they already hold. This is the fair supervised baseline: for each held-out
    dataset, one probe fitted on every other dataset at once.

    `weighting` decides what "pooled" means, and the two are reported side by side because the
    datasets differ in size by a factor of twenty -- instruct-dishonesty alone is 5432 of the
    ~9500 rows, so a naive pool is mostly one dataset:

      naive     plain concatenation; what someone would actually do.
      balanced  every dataset gets the same total weight (1/n_d per row), separating "pooled
                supervision helps" from "the largest dataset happens to transfer well".
    """
    if weighting not in ("naive", "balanced"):
        raise ValueError(f"unknown weighting {weighting!r}")

    feats = {name: featurize(datasets[name].z) for name in order}
    control_features = featurize(control.z) if control is not None else None
    scores: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    for held_out in order:
        sources = [n for n in order if n != held_out]
        x = np.vstack([feats[n] for n in sources])
        y = np.concatenate([datasets[n].y for n in sources])
        probe = make_probe()
        if weighting == "balanced":
            weights = np.concatenate(
                [np.full(len(datasets[n].y), 1.0 / len(datasets[n].y)) for n in sources]
            )
            probe.fit(x, y, logisticregression__sample_weight=weights)  # type: ignore[attr-defined]
        else:
            probe.fit(x, y)  # type: ignore[attr-defined]
        scores[held_out] = probe.decision_function(feats[held_out])  # type: ignore[attr-defined]
        thresholds[held_out] = (
            control_threshold(probe, control_features)
            if control_features is not None
            else float("nan")
        )
    return LOOScores(scores=scores, thresholds=thresholds)


def leave_one_out(
    featurize: Callable[[np.ndarray], np.ndarray],
    datasets: dict[str, "Pooled"],
    order: list[str],
    metric: "Metric" = AUROC,
    control: "Pooled | None" = None,
    weighting: str = "naive",
) -> pd.Series:
    """`leave_one_out_scores` reduced to one metric per held-out dataset."""
    if metric.needs_threshold and control is None:
        raise ValueError(f"metric {metric.name!r} needs a control dataset for its threshold")
    fitted = leave_one_out_scores(featurize, datasets, order, control, weighting)
    return pd.Series(
        {
            name: metric.fn(datasets[name].y, fitted.scores[name], fitted.thresholds[name])
            for name in order
        },
        dtype=float,
    )


def bootstrap_delta(
    y: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Point estimate and 95% interval for AUROC(a) - AUROC(b) on one test set.

    Resampled stratified by label: HP-C and HP-KR are ~6% deceptive, and an unstratified draw
    would sometimes hold too few positives to score at all. Both probes are fitted once and held
    fixed, so this is an interval on the *evaluation set*, not on the fit -- the right object when
    comparing two given probes on the same rows, and cheap enough to run for every arm.
    """
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    point = auroc(y, scores_a) - auroc(y, scores_b)
    if len(positive) == 0 or len(negative) == 0:
        return point, float("nan"), float("nan")

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate(
            (
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            )
        )
        deltas[i] = auroc(y[idx], scores_a[idx]) - auroc(y[idx], scores_b[idx])
    low, high = np.quantile(deltas, [0.025, 0.975])
    return point, float(low), float(high)


def off_diagonal_mean(matrix: pd.DataFrame) -> float:
    values = matrix.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, np.nan)
    return float(np.nanmean(values))


def diagonal_mean(matrix: pd.DataFrame) -> float:
    return float(np.nanmean(np.diag(matrix.to_numpy(dtype=float))))


def train_set_sensitivity(
    matrix: pd.DataFrame,
    exclude_train: tuple[str, ...] = (HPKR,),
    exclude_test: tuple[str, ...] = (),
) -> float:
    """How much a test set's AUROC depends on which dataset the probe was trained on.

    Per test column, the population std of the off-diagonal cells across training rows, then the
    mean over columns. HP-KR's training row is excluded by default: it is inverted, and its spread
    would swamp every other row's. This is the "mechanism" metric of the persona summary (baseline
    0.167, persona PCs 0.027), which was computed ad hoc before it lived here.

    A 1-d feature set can only change sign between training sets -- a one-feature probe learns a
    sign and a scale -- so it scores 0 unless training sets disagree on that sign. A 0 there is an
    artifact of dimension, not evidence of invariance; a nonzero value means sign flips.
    """
    values = matrix.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, np.nan)
    cells = pd.DataFrame(values, index=matrix.index, columns=matrix.columns)
    cells = cells.drop(index=list(exclude_train), columns=list(exclude_test), errors="ignore")
    return float(cells.std(axis=0, ddof=0).mean())


def random_directions(hidden: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """k random orthonormal directions, via QR of a Gaussian matrix."""
    q, _ = np.linalg.qr(rng.standard_normal((hidden, k)))
    return q.T


def cells_at_floor(matrix: pd.DataFrame, floor: float = 0.5, tol: float = 0.005) -> int:
    """Off-diagonal cells sitting on the metric's floor, and therefore carrying no information.

    Balanced accuracy at a fixed threshold bottoms out at 0.5 whenever a probe flags nothing:
    recall 0, specificity 1. Reporting the mean of a matrix that is half-constant without this
    count would make a saturated measurement look like a merely weaker one.
    """
    values = matrix.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, np.nan)
    return int(np.nansum(np.abs(values - floor) < tol))


def off_diagonal_sd(matrix: pd.DataFrame) -> float:
    values = matrix.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, np.nan)
    return float(np.nanstd(values))


def balacc_summary(
    variants: dict[str, Callable[[np.ndarray], np.ndarray]],
    datasets: dict[str, Pooled],
    order: list[str],
    control: Pooled,
    without_hpkr: list[str],
    dims: dict[str, object],
    out_dir: Path,
) -> pd.DataFrame:
    """The same transfer matrices scored with balanced accuracy at 1% FPR on the control.

    Secondary to the AUROC tables, never a replacement: about half these cells land exactly on
    the 0.5 floor, and a probe that is strongly anti-correlated scores the same there as one with
    no signal at all. `cells_at_floor` is reported so that saturation is visible in the table
    rather than hidden inside the mean.
    """
    rows = []
    for name, featurize in variants.items():
        matrix = transfer_matrix(featurize, datasets, order, metric=BALANCED_ACCURACY,
                                 control=control)
        matrix.round(4).to_csv(out_dir / f"matrix_balacc_{name}.csv")
        rows.append(
            {
                "features": name,
                "dim": dims.get(name, ""),
                "off_diagonal_mean": off_diagonal_mean(matrix),
                "off_diagonal_excl_hpkr": off_diagonal_mean(matrix.loc[without_hpkr, without_hpkr]),
                "diagonal_mean": diagonal_mean(matrix),
                "off_diagonal_sd": off_diagonal_sd(matrix),
                "cells_at_floor": cells_at_floor(matrix),
                "off_diagonal_cells": len(order) * (len(order) - 1),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary_balacc.csv", index=False)
    logger.info(f"balanced accuracy @1% FPR:\n{summary.round(4).to_string(index=False)}")
    return summary


def operating_point(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Recall, specificity and flagged fraction at one threshold, beside balanced accuracy.

    `balanced_accuracy` collapses these into a single number, and on the harm-pressure datasets
    that number is exactly 0.5 -- which can mean "no signal" or "nothing crossed the threshold".
    The decomposition tells them apart: recall 0 with specificity 1 is a calibration failure and
    says nothing about whether the direction ranks that dataset correctly.
    """
    flagged = scores > threshold
    recall = float(flagged[y == 1].mean()) if (y == 1).any() else float("nan")
    specificity = float((~flagged)[y == 0].mean()) if (y == 0).any() else float("nan")
    return {
        "threshold": threshold,
        "recall": recall,
        "specificity": specificity,
        "balacc": (recall + specificity) / 2,
        "flagged_frac": float(flagged.mean()),
    }


def zero_shot_operating_table(
    datasets: dict[str, Pooled],
    arms: dict[str, Callable[[np.ndarray], np.ndarray]],
    control: Pooled,
    layer: int,
    fpr: float = 0.01,
) -> pd.DataFrame:
    """Zero-shot detection at a fixed false-positive rate, under two ways of setting it.

    A persona axis is a direction, not a calibrated classifier, so the threshold has to come from
    somewhere. Neither source uses deception labels on the rows being scored:

      control     the `fpr` point on a held-out honest corpus. Label-free and deployable, and
                  what the Liars' Bench tables use.
      in-dataset  the `fpr` point on that dataset's own honest rows. This needs labels, so it is
                  an oracle -- reported only to separate "the direction cannot detect here" from
                  "the threshold is in the wrong place for this prompt distribution".
    """
    rows = []
    for arm, score_fn in arms.items():
        from_control = float(np.quantile(score_fn(control.z), 1 - fpr))
        for name, ds in datasets.items():
            if not ds.two_class:
                continue
            scores = score_fn(ds.z)
            in_dataset = float(np.quantile(scores[ds.y == 0], 1 - fpr))
            for source, threshold in (
                ("control", from_control),
                ("in-dataset (oracle)", in_dataset),
            ):
                rows.append(
                    {
                        "layer": layer,
                        "arm": arm,
                        "dataset": name,
                        "n_deceptive": ds.n_deceptive,
                        "auroc": auroc(ds.y, scores),
                        "threshold_source": source,
                    }
                    | operating_point(ds.y, scores, threshold)
                )
    return pd.DataFrame(rows)


def extended_arms(
    bank: persona_pcs.PersonaBank,
) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    """Zero-shot score functions for the persona groups only the extended bank has.

    Shared by `run` and `zeroshot` so the two cannot write different versions of
    `zero_shot.csv`; an empty dict for a bank without the knowledge-denial personas.
    """
    if "denial" not in bank.axes:
        return {}
    ext_geom = persona_pcs.fit_geometry(
        bank,
        k=3,
        deception=persona_pcs.DECEPTION_PERSONAS + persona_pcs.KNOWLEDGE_DENIAL_PERSONAS,
        honesty=persona_pcs.HONESTY_PERSONAS + persona_pcs.KNOWLEDGE_CANDOUR_PERSONAS,
        axis_name="extended",
    )
    denial_geom = persona_pcs.fit_geometry(
        bank,
        k=3,
        deception=persona_pcs.KNOWLEDGE_DENIAL_PERSONAS,
        honesty=persona_pcs.KNOWLEDGE_CANDOUR_PERSONAS,
        axis_name="denial",
    )
    return {
        "auroc_ext_axis": lambda z, g=ext_geom: g.axis_score(z),
        "auroc_ext_pc1": lambda z, g=ext_geom: g.project(z, k=1).ravel(),
        "auroc_denial_axis": lambda z, g=denial_geom: g.axis_score(z),
    }


def zero_shot_table(
    datasets: dict[str, Pooled],
    geom: persona_pcs.PersonaGeometry,
    syc_geom: persona_pcs.PersonaGeometry,
    extra: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
) -> pd.DataFrame:
    """Methods 1 and 2: score by projection alone, no training.

    The sycophancy columns are the cross-behaviour comparison -- the same scoring rule driven by
    a basis fitted on a different behaviour entirely.
    """
    rows = []
    for name, ds in datasets.items():
        rows.append(
            {
                "dataset": name,
                "n": len(ds.y),
                "n_deceptive": ds.n_deceptive,
                "auroc_axis": auroc(ds.y, geom.axis_score(ds.z)),
                "auroc_pc1": auroc(ds.y, geom.project(ds.z, k=1).ravel()),
                "auroc_syc_axis": auroc(ds.y, syc_geom.axis_score(ds.z)),
                "auroc_syc_pc1": auroc(ds.y, syc_geom.project(ds.z, k=1).ravel()),
            }
            | {label: auroc(ds.y, fn(ds.z)) for label, fn in (extra or {}).items()}
        )
    return pd.DataFrame(rows)


def liars_bench_comparison(
    datasets: dict[str, Pooled],
    control: Pooled,
    geom: persona_pcs.PersonaGeometry,
    out_dir: Path,
    train_on: str = "repe_honesty__plain",
    extra: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
    extra_rows: list[dict[str, object]] | None = None,
    out_name: str = "liars_bench_comparison.csv",
) -> pd.DataFrame:
    """Head-to-head against the Liars' Bench probe baselines, over the five shared datasets.

    The paper's rows are transcribed constants (see PAPER_PROBES); only their mean over the
    shared columns is computed here. Everything on this side is measured from the pooled
    activations, trained on RepE where a probe is involved -- the same protocol the paper's Mean
    Probe uses.

    Cross-pipeline rows are not strictly comparable: this repo's reproduction of the Mean Probe
    method scores well above the paper's on insider-trading because `trim_reasoning` is on, a
    difference the paper's own Table 17 documents. The trustworthy comparison is between the rows
    measured here, which share a pipeline exactly.
    """
    shared = [n for n in SHARED_DATASETS if n in datasets]
    missing = sorted(set(SHARED_DATASETS) - set(shared))
    if missing:
        logger.warning(f"shared datasets absent from the pooled set: {missing}")

    def evaluate(score_fn: Callable[[np.ndarray], np.ndarray], label: str) -> dict[str, object]:
        threshold = float(np.quantile(score_fn(control.z), 0.99))  # 1% FPR on the control
        balaccs, aurocs = [], []
        for name in shared:
            ds = datasets[name]
            scores = score_fn(ds.z)
            balaccs.append(balanced_accuracy(ds.y, scores, threshold))
            aurocs.append(auroc(ds.y, scores))
        return {
            "method": label,
            "source": "measured here",
            "privileged": False,
            "balacc_1pct_fpr": float(np.mean(balaccs)),
            "auroc": float(np.mean(aurocs)),
            "n_datasets": len(shared),
        }

    train = datasets[train_on]
    pc3_probe = make_probe()
    pc3_probe.fit(geom.project(train.z, 3), train.y)  # type: ignore[attr-defined]
    raw_probe = make_probe()
    raw_probe.fit(train.z, train.y)  # type: ignore[attr-defined]

    rows = [
        {
            "method": name,
            "source": spec["source"],
            "privileged": spec["privileged"],
            "balacc_1pct_fpr": float(
                np.nanmean(
                    [spec["balacc"][PAPER_COLUMNS.index(SHARED_DATASETS[n])] for n in shared]
                )
            ),
            "auroc": float(
                np.nanmean(
                    [spec["auroc"][PAPER_COLUMNS.index(SHARED_DATASETS[n])] for n in shared]
                )
            ),
            "n_datasets": len(shared),
        }
        for name, spec in PAPER_PROBES.items()
    ]
    rows += [
        evaluate(lambda z: geom.axis_score(z), "persona axis (zero-shot)"),
        evaluate(lambda z: geom.project(z, 1).ravel(), "persona PC1 (zero-shot)"),
        evaluate(
            lambda z: pc3_probe.decision_function(geom.project(z, 3)),  # type: ignore[attr-defined]
            "persona PC3 (RepE-trained)",
        ),
        evaluate(
            lambda z: raw_probe.decision_function(z),  # type: ignore[attr-defined]
            "raw activations (RepE-trained)",
        ),
    ]

    # Arms that are a fixed scoring rule go through `evaluate`; ones whose probe changes per
    # dataset (leave-one-out) cannot, and arrive already scored as `extra_rows`.
    if extra:
        rows += [evaluate(fn, label) for label, fn in extra.items()]
    if extra_rows:
        rows += list(extra_rows)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / out_name, index=False)
    logger.info(f"vs Liars' Bench probe baselines:\n{table.round(4).to_string(index=False)}")
    return table


def run(
    pooled_dir: str | Path = POOLED_DIR,
    out_dir: str | Path = OUT_DIR,
    layer: int = 20,
    vectors: str | Path | None = None,
) -> None:
    pooled_dir, out_dir = Path(pooled_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # `layer` must match the layer the pooled activations were extracted at. Nothing enforces
    # it -- persona directions live in the residual stream at one layer, so pairing layer-40
    # activations with layer-20 directions yields a meaningless number rather than an error. The
    # layer is recorded in summary.csv so a mismatch is at least visible after the fact.
    bank = persona_pcs.load_bank_at_layer(
        layer, vectors if vectors is not None else persona_pcs.VECTORS_DIR
    )
    geom = persona_pcs.fit_geometry(bank, k=3)
    syc_geom = persona_pcs.fit_geometry(
        bank,
        k=3,
        deception=persona_pcs.SYCOPHANCY_PERSONAS,
        honesty=persona_pcs.ANTI_SYCOPHANCY_PERSONAS,
        axis_name="sycophancy",
    )
    for tag, g in (("deception", geom), ("sycophancy", syc_geom)):
        logger.info(
            f"{tag} geometry: {len(g.persona_names)} personas, "
            f"explained variance {np.round(g.explained_variance_ratio, 4).tolist()}, "
            f"|cos(PC_k, deception axis)| {np.round(g.alignment(), 4).tolist()}"
        )
    logger.info(f"persona bank at layer {layer}, pooled activations from {pooled_dir}")
    dec_axis, syc_axis = bank.axis_unit("deception"), bank.axis_unit("sycophancy")
    logger.info(f"|cos(deception axis, sycophancy axis)| = {abs(float(dec_axis @ syc_axis)):.4f}")

    datasets = load_pooled(pooled_dir)
    logger.info(f"loaded {len(datasets)} pooled datasets from {pooled_dir}")

    zero_shot = zero_shot_table(datasets, geom, syc_geom, extra=extended_arms(bank))
    zero_shot.insert(1, "layer", layer)
    zero_shot.to_csv(out_dir / "zero_shot.csv", index=False)
    logger.info(f"zero-shot:\n{zero_shot.to_string(index=False)}")

    order = [n for n, d in sorted(datasets.items()) if d.two_class and n != CONTROL_DATASET]
    control = datasets[CONTROL_DATASET]
    logger.info(f"transfer matrix over {len(order)} two-class datasets: {order}")

    hidden = datasets[order[0]].z.shape[1]
    rng = np.random.default_rng(SEED)

    # The deception and sycophancy axes sit at |cos| = 0.54, so a raw sycophancy arm would
    # score well largely through the shared component. The orthogonalised arms remove each
    # axis from the other's basis, which is what makes the comparison interpretable.
    syc_orth = persona_pcs.orthogonalize(syc_geom.components, dec_axis)
    dec_orth = persona_pcs.orthogonalize(geom.components, syc_axis)
    logger.info(
        f"orthogonalised bases: sycophancy {syc_orth.shape[0]} of 3 rows survive, "
        f"deception {dec_orth.shape[0]} of 3"
    )

    # The paper's own feature sets: Figure 4 reports PC1 for deception, Figure 5 k=3 for
    # sycophancy, and it never orthogonalises. Those arms come first; the *_orth arms below are
    # this repo's addition, not a replication of anything in the paper.
    # The extended folder adds roles whose deception is *denial of knowledge* rather than
    # assertion of falsehood. HP-KR inverts under the original axis precisely because its lie is
    # a denial, so these arms exist to test whether that is fixable at the basis level.
    extended = "denial" in bank.axes
    if extended:
        ext_geom = persona_pcs.fit_geometry(
            bank,
            k=3,
            deception=persona_pcs.DECEPTION_PERSONAS + persona_pcs.KNOWLEDGE_DENIAL_PERSONAS,
            honesty=persona_pcs.HONESTY_PERSONAS + persona_pcs.KNOWLEDGE_CANDOUR_PERSONAS,
            axis_name="extended",
        )
        denial_geom = persona_pcs.fit_geometry(
            bank,
            k=3,
            deception=persona_pcs.KNOWLEDGE_DENIAL_PERSONAS,
            honesty=persona_pcs.KNOWLEDGE_CANDOUR_PERSONAS,
            axis_name="denial",
        )
        logger.info(
            f"extended geometry: {len(ext_geom.persona_names)} personas, "
            f"var {np.round(ext_geom.explained_variance_ratio, 3).tolist()}; "
            f"denial-only: {len(denial_geom.persona_names)} personas, "
            f"var {np.round(denial_geom.explained_variance_ratio, 3).tolist()}"
        )

    variants: dict[str, object] = {
        "baseline": lambda z: z,
        "pc1": lambda z: geom.project(z, k=1),
        "pc3": lambda z: geom.project(z, k=3),
        "axis1": lambda z: geom.axis_score(z)[:, None],
        "syc_pc1": lambda z: syc_geom.project(z, k=1),
        "syc3": lambda z: syc_geom.project(z, k=3),
        "syc_axis1": lambda z: syc_geom.axis_score(z)[:, None],
        # --- not in the paper: added because the two axes overlap at |cos| = 0.54 ---
        "syc3_orth": lambda z: z @ syc_orth.T,
        "dec3_orth": lambda z: z @ dec_orth.T,
    }
    if extended:
        variants |= {
            "ext_pc1": lambda z: ext_geom.project(z, k=1),
            "ext_pc3": lambda z: ext_geom.project(z, k=3),
            "ext_axis1": lambda z: ext_geom.axis_score(z)[:, None],
            "denial_pc3": lambda z: denial_geom.project(z, k=3),
            "denial_axis1": lambda z: denial_geom.axis_score(z)[:, None],
        }
    matrices = {name: transfer_matrix(fn, datasets, order) for name, fn in variants.items()}

    # random control, averaged over draws so a single lucky basis cannot carry the comparison
    random_mats = []
    for _ in range(N_RANDOM_DRAWS):
        dirs = random_directions(hidden, 3, rng)
        random_mats.append(transfer_matrix(lambda z, d=dirs: z @ d.T, datasets, order))
    matrices["random3"] = sum(random_mats) / len(random_mats)

    matrices["delta_pc3_vs_baseline"] = matrices["pc3"] - matrices["baseline"]
    for name, matrix in matrices.items():
        matrix.round(4).to_csv(out_dir / f"matrix_{name}.csv")

    # HP-KR is inverted as a test set and poisonous as a training set, so the aggregate is
    # reported both ways -- it moves the off-diagonal mean by more than any variant does.
    without_hpkr = [n for n in order if n != HPKR]
    dims = {
        "baseline": hidden, "pc1": 1, "pc3": 3, "axis1": 1, "random3": 3,
        "syc_pc1": 1, "syc3": 3, "syc_axis1": 1,
        "syc3_orth": syc_orth.shape[0], "dec3_orth": dec_orth.shape[0],
        "ext_pc1": 1, "ext_pc3": 3, "ext_axis1": 1, "denial_pc3": 3, "denial_axis1": 1,
    }
    summary = pd.DataFrame(
        [
            {
                "features": name,
                "layer": layer,
                "dim": dims.get(name, ""),
                "off_diagonal_mean": off_diagonal_mean(matrices[name]),
                "off_diagonal_excl_hpkr": off_diagonal_mean(
                    matrices[name].loc[without_hpkr, without_hpkr]
                ),
                "diagonal_mean": diagonal_mean(matrices[name]),
            }
            for name in dims
            if name in matrices
        ]
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    logger.info(f"\n{summary.round(4).to_string(index=False)}")
    balacc_summary(variants, datasets, order, control, without_hpkr, dims, out_dir)
    liars_bench_comparison(datasets, control, geom, out_dir)
    logger.info(f"wrote matrices and summary to {out_dir}")



def zeroshot(
    pooled_dir: str | Path = POOLED_DIR,
    out_dir: str | Path = OUT_DIR,
    layer: int = 20,
    vectors: str | Path | None = None,
) -> None:
    """Rewrite just `zero_shot.csv`, with every axis the bank supports.

    Separate from `run` because the zero-shot numbers cost four dot products while `run` costs
    sixteen transfer matrices, and the zero-shot table is the one that gets extended as new
    persona groups arrive. The columns `run` already wrote are reproduced exactly; the extended
    and knowledge-denial arms are added beside them.
    """
    pooled_dir, out_dir = Path(pooled_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = persona_pcs.load_bank_at_layer(
        layer, vectors if vectors is not None else persona_pcs.VECTORS_DIR
    )
    geom = persona_pcs.fit_geometry(bank, k=3)
    syc_geom = persona_pcs.fit_geometry(
        bank,
        k=3,
        deception=persona_pcs.SYCOPHANCY_PERSONAS,
        honesty=persona_pcs.ANTI_SYCOPHANCY_PERSONAS,
        axis_name="sycophancy",
    )

    extra = extended_arms(bank)

    datasets = load_pooled(pooled_dir)
    table = zero_shot_table(datasets, geom, syc_geom, extra=extra)
    table.insert(1, "layer", layer)
    table.to_csv(out_dir / "zero_shot.csv", index=False)
    logger.info(f"zero-shot at layer {layer}:\n{table.round(4).to_string(index=False)}")
    arms: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "deception axis (8v7)": lambda z, g=geom: g.axis_score(z),
        "sycophancy axis (5v5)": lambda z, g=syc_geom: g.axis_score(z),
    }
    if "auroc_ext_axis" in extra:
        arms["extended axis (12v13)"] = extra["auroc_ext_axis"]
        arms["knowledge-denial axis (4v6)"] = extra["auroc_denial_axis"]
    operating = zero_shot_operating_table(datasets, arms, datasets[CONTROL_DATASET], layer)
    operating.round(4).to_csv(out_dir / "zero_shot_operating.csv", index=False)
    logger.info(f"wrote zero_shot.csv and zero_shot_operating.csv to {out_dir}")


def loo(
    pooled_dir: str | Path = POOLED_DIR,
    out_dir: str | Path = REPO_ROOT / "results" / "loo",
    layer: int = 20,
    vectors: str | Path | None = None,
    n_boot: int = 1000,
) -> None:
    """Leave-one-dataset-out: the fair supervised baseline for the generalization claim.

    `run` compares a fixed persona direction against probes trained on *one* dataset and tested
    on another. That understates supervision: anyone holding labels for five datasets would train
    on all five. Here each held-out dataset is scored by a probe fitted on the pooled remainder,
    which is the baseline the cross-dataset claim actually has to beat.

    `layer` must match the layer `pooled_dir` was extracted at -- nothing enforces it, and a
    mismatch yields meaningless numbers rather than an error, so it is recorded in every file.
    """
    pooled_dir, out_dir = Path(pooled_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = persona_pcs.load_bank_at_layer(
        layer, vectors if vectors is not None else persona_pcs.VECTORS_DIR
    )
    geom = persona_pcs.fit_geometry(bank, k=3)
    extended = "denial" in bank.axes
    ext_geom = (
        persona_pcs.fit_geometry(
            bank,
            k=3,
            deception=persona_pcs.DECEPTION_PERSONAS + persona_pcs.KNOWLEDGE_DENIAL_PERSONAS,
            honesty=persona_pcs.HONESTY_PERSONAS + persona_pcs.KNOWLEDGE_CANDOUR_PERSONAS,
            axis_name="extended",
        )
        if extended
        else None
    )

    datasets = load_pooled(pooled_dir)
    order = [n for n, d in sorted(datasets.items()) if d.two_class and n != CONTROL_DATASET]
    control = datasets[CONTROL_DATASET]
    logger.info(f"leave-one-out over {len(order)} datasets at layer {layer}: {order}")

    arms: dict[str, tuple[Callable[[np.ndarray], np.ndarray], int]] = {
        "baseline": (lambda z: z, datasets[order[0]].z.shape[1]),
        "axis1": (lambda z: geom.axis_score(z)[:, None], 1),
        "pc3": (lambda z: geom.project(z, k=3), 3),
    }
    if ext_geom is not None:
        arms |= {
            "ext_axis1": (lambda z, g=ext_geom: g.axis_score(z)[:, None], 1),
            "ext_pc3": (lambda z, g=ext_geom: g.project(z, k=3), 3),
        }

    fits: dict[tuple[str, str], LOOScores] = {}
    tidy: list[dict[str, object]] = []
    for weighting in ("naive", "balanced"):
        for arm, (featurize, dim) in arms.items():
            fitted = leave_one_out_scores(featurize, datasets, order, control, weighting)
            fits[(arm, weighting)] = fitted
            for name in order:
                y, scores = datasets[name].y, fitted.scores[name]
                tidy.append(
                    {
                        "layer": layer,
                        "arm": arm,
                        "dim": dim,
                        "weighting": weighting,
                        "dataset": name,
                        "auroc": auroc(y, scores),
                        "balacc_1pct_fpr": balanced_accuracy(y, scores, fitted.thresholds[name]),
                    }
                )
            logger.info(f"{arm:>10s} / {weighting:<8s} done")

    table = pd.DataFrame(tidy)
    table.round(4).to_csv(out_dir / "loo.csv", index=False)

    # Reported with and without HP-KR for the same reason `run` does it: HP-KR is where the
    # supervised probes inverted, so a single mean lets one dataset decide the comparison.
    summary = (
        table.groupby(["arm", "dim", "weighting"], as_index=False)[["auroc", "balacc_1pct_fpr"]]
        .mean()
        .sort_values("auroc", ascending=False)
    )
    excl = (
        table[table["dataset"] != HPKR]
        .groupby(["arm", "dim", "weighting"], as_index=False)[["auroc"]]
        .mean()
        .rename(columns={"auroc": "auroc_excl_hpkr"})
    )
    summary = summary.merge(excl, on=["arm", "dim", "weighting"], how="left")
    # The floor, not the mean, is what a detector is judged on: a single dataset scoring 0.12
    # means the probe confidently calls lies honest there, and averaging hides it.
    worst = (
        table.groupby(["arm", "dim", "weighting"], as_index=False)[["auroc"]]
        .min()
        .rename(columns={"auroc": "auroc_worst_dataset"})
    )
    summary = summary.merge(worst, on=["arm", "dim", "weighting"], how="left")
    summary.insert(0, "layer", layer)
    summary.round(4).to_csv(out_dir / "loo_summary.csv", index=False)
    logger.info(f"leave-one-out means:\n{summary.round(4).to_string(index=False)}")

    # The single-source reference: the same held-out column of the ordinary transfer matrix, as a
    # mean over the five training sets and as the best of them. `run` reports the mean; the max is
    # the most generous single-source baseline available, reported so the comparison cannot be
    # accused of having picked the weak one.
    single = transfer_matrix(lambda z: z, datasets, order)
    reference = pd.DataFrame(
        [
            {
                "layer": layer,
                "dataset": name,
                "single_source_mean": float(single[name].drop(index=name).mean()),
                "single_source_best": float(single[name].drop(index=name).max()),
                "pooled_loo_naive": auroc(
                    datasets[name].y, fits[("baseline", "naive")].scores[name]
                ),
                "pooled_loo_balanced": auroc(
                    datasets[name].y, fits[("baseline", "balanced")].scores[name]
                ),
            }
            for name in order
        ]
    )
    reference.round(4).to_csv(out_dir / "single_source_reference.csv", index=False)
    logger.info(f"baseline, single-source vs pooled:\n{reference.round(4).to_string(index=False)}")

    # The headline interval: pooled supervision minus the fixed persona direction, per held-out
    # dataset. HP-C and HP-KR carry ~135 positives each, so the point estimates there need one.
    # Against *both* pooled baselines. The balanced one is much the stronger of the two, so
    # bootstrapping only against the naive pool would be comparing with the weaker opponent.
    best_persona = "ext_axis1" if ext_geom is not None else "axis1"
    rows = []
    for weighting in ("naive", "balanced"):
        for name in order:
            y = datasets[name].y
            baseline_scores = fits[("baseline", weighting)].scores[name]
            persona_scores = fits[(best_persona, weighting)].scores[name]
            point, low, high = bootstrap_delta(y, baseline_scores, persona_scores, n_boot=n_boot)
            rows.append(
                {
                    "layer": layer,
                    "weighting": weighting,
                    "dataset": name,
                    "n": len(y),
                    "n_deceptive": int(y.sum()),
                    "pooled_baseline": auroc(y, baseline_scores),
                    best_persona: auroc(y, persona_scores),
                    "delta": point,
                    "ci_low": low,
                    "ci_high": high,
                    "excludes_zero": bool(low > 0 or high < 0),
                }
            )
    bootstrap = pd.DataFrame(rows)
    bootstrap.round(4).to_csv(out_dir / "loo_bootstrap.csv", index=False)
    # The fits are the expensive part (six 8192-d probes per arm per weighting); saving the
    # scores means any later comparison is arithmetic rather than another sweep.
    np.savez_compressed(
        out_dir / "loo_scores.npz",
        **{
            f"{arm}|{weighting}|{name}": fitted.scores[name]
            for (arm, weighting), fitted in fits.items()
            for name in order
        },
        **{f"labels|{name}": datasets[name].y for name in order},
    )
    logger.info(
        f"pooled baseline minus {best_persona}, 95% CI over {n_boot} draws:\n"
        f"{bootstrap.round(4).to_string(index=False)}"
    )

    # The head-to-head against the paper's probes, widened: `run` only ever entered the original
    # axis, never the extended one, and never a pooled-supervision row.
    shared = [n for n in SHARED_DATASETS if n in datasets]
    extra_rows: list[dict[str, object]] = []
    for weighting in ("naive", "balanced"):
        fitted = fits[("baseline", weighting)]
        extra_rows.append(
            {
                "method": f"raw activations (pooled leave-one-out, {weighting})",
                "source": "measured here",
                "privileged": False,
                "balacc_1pct_fpr": float(
                    np.nanmean(
                        [
                            balanced_accuracy(datasets[n].y, fitted.scores[n], fitted.thresholds[n])
                            for n in shared
                        ]
                    )
                ),
                "auroc": float(
                    np.nanmean([auroc(datasets[n].y, fitted.scores[n]) for n in shared])
                ),
                "n_datasets": len(shared),
            }
        )
    extra = (
        {
            "persona axis, extended (zero-shot)": lambda z, g=ext_geom: g.axis_score(z),
            "persona PC1, extended (zero-shot)": lambda z, g=ext_geom: g.project(z, 1).ravel(),
        }
        if ext_geom is not None
        else None
    )
    liars_bench_comparison(
        datasets,
        control,
        geom,
        out_dir,
        extra=extra,
        extra_rows=extra_rows,
        out_name="liars_bench_comparison_loo.csv",
    )
    logger.info(f"wrote leave-one-out tables to {out_dir}")


def verify(
    run_folder: str | Path = REPO_ROOT / "results" / "XXXX-9-llama-all",
    pooled_dir: str | Path = POOLED_DIR,
) -> None:
    """Check the pooled matrix against the completed sweep, using its own probe.

    Scoring the pooled vectors with the trained direction should reproduce the AUROCs in
    results_table.csv. It will not match exactly: the deployed probe scores every response token
    then averages the scores, whereas this averages the activations then scores once, and those
    differ whenever examples have unequal token counts. Close agreement confirms the cache lookup
    and mask alignment; a large gap means the pooling is wrong and everything downstream is void.
    """
    import pickle

    run_folder, pooled_dir = Path(run_folder), Path(pooled_dir)
    with open(run_folder / "detector.pt", "rb") as f:
        detector = pickle.load(f)
    direction = detector["directions"].double().numpy().ravel()
    mean = detector["scaler_mean"].double().numpy().ravel()
    scale = detector["scaler_scale"].double().numpy().ravel()

    published = pd.read_csv(run_folder / "results_table.csv", index_col=0)
    rows = []
    for name, ds in sorted(load_pooled(pooled_dir).items()):
        if not ds.two_class:
            continue
        scores = ((ds.z - mean) / scale) @ direction
        mine = auroc(ds.y, scores)
        theirs = float(published.loc[name, "auroc"]) if name in published.index else float("nan")
        rows.append(
            {"dataset": name, "pooled_auroc": mine, "run_auroc": theirs, "delta": mine - theirs}
        )

    table = pd.DataFrame(rows)
    logger.info(f"\n{table.round(4).to_string(index=False)}")
    worst = table["delta"].abs().max()
    logger.info(f"largest |delta| = {worst:.4f}")
    if worst > 0.05:
        logger.error("pooling disagrees with the run by more than 0.05 AUROC; do not trust results")

if __name__ == "__main__":
    fire.Fire({"run": run, "verify": verify, "loo": loo, "zeroshot": zeroshot})
