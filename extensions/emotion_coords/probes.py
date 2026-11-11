"""Probes for the emotion-coordinate study: feature sets, two probe families, protocols A, B and C.

A feature set maps pooled activations (n, hidden) to features: coordinates on the first k rows of a basis
(keep-k), or all dimensions (raw, and raw with a subspace erased). Probe families:

  lr   standardise on the training rows, L2 logistic regression; C (and k) chosen by cross-validation inside
       the training data: leave one training set out (protocol B), stratified 5-fold within the source set
       (protocol C), statement-grouped 5-fold within the instructed pairs (protocol A)
  dom  difference of class means on the unstandardised features; k chosen the same way

Protocols:
  A  train on the instructed pairs (RepE), test on each Liars' Bench set
  B  leave one Liars' Bench set out, train on the other four
  C  train on one set, test on every other; the diagonal is stratified 5-fold CV within the set, with the C
     chosen on the whole set (a slight optimism the off-diagonal cells do not have)

Each probe also scores Alpaca, which sets its 1%-FPR threshold for balanced accuracy and recall. Every k is
scored (for comparisons at matched k), and the inner-CV choice is flagged. Held-out scores are kept so
paired bootstraps compare feature sets on the same rows.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from extensions.emotion_coords import linalg as la
from extensions.emotion_coords.lbdata import Split

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
MLP_ALPHAS = (1e-2, 1.0)  # L2 weight decay of the exploratory MLP probe, chosen by inner CV
KS = (2, 3, 5, 10, 20)
SEED = 0


# --- feature sets --------------------------------------------------------------------------------


@dataclass
class FeatureSet:
    """`basis` rows give coordinates (keep-k for k in `ks`); with `basis` None the features are `transform(x)`
    at full dimension (k reported as the hidden size). `per_fold` builds the basis from the training rows."""

    name: str
    family: str
    basis: np.ndarray | None = None
    ks: tuple[int, ...] = ()
    transform: Callable[[np.ndarray], np.ndarray] | None = None
    per_fold: Callable[[np.ndarray], np.ndarray] | None = None
    seed: int = 0

    def usable_ks(self, rank: int | None = None) -> tuple[int, ...]:
        limit = rank if rank is not None else (len(self.basis) if self.basis is not None else None)
        return tuple(k for k in self.ks if limit is None or k <= limit)


def coordinates(fs: FeatureSet, x: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    b = basis if basis is not None else fs.basis
    if b is None:
        return fs.transform(x) if fs.transform is not None else x
    return x @ b.T


# --- probe families ------------------------------------------------------------------------------


@dataclass
class Fitted:
    family: str
    weights: np.ndarray
    offset: float
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.mean is not None:
            x = (x - self.mean) / self.scale
        return x @ self.weights + self.offset


@dataclass
class FittedMLP:
    """Exploratory nonlinear probe: standardise, one hidden layer of 64 ReLU units, L2 weight decay `alpha`,
    early stopping on 10% of the training rows. Scores are the logit of P(lie)."""

    model: object

    def score(self, x: np.ndarray) -> np.ndarray:
        p = np.clip(self.model.predict_proba(x)[:, 1], 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))


def fit(family: str, x: np.ndarray, y: np.ndarray, c: float | None = None) -> Fitted | FittedMLP:
    if family == "mlp":
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline

        model = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64,), alpha=c, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
            max_iter=300, batch_size=200, random_state=SEED))
        return FittedMLP(model.fit(x, y))
    if family == "dom":
        mu1, mu0 = x[y == 1].mean(0), x[y == 0].mean(0)
        w = mu1 - mu0
        return Fitted("dom", w, -float(w @ (mu1 + mu0) / 2))
    scaler = StandardScaler().fit(x)
    scale = np.where(scaler.scale_ > 0, scaler.scale_, 1.0)
    z = (x - scaler.mean_) / scale
    model = LogisticRegression(C=c, max_iter=3000, tol=1e-5).fit(z, y)
    return Fitted("lr", model.coef_.ravel(), float(model.intercept_[0]), scaler.mean_, scale)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def operating_point(y: np.ndarray, s: np.ndarray, control: np.ndarray, fpr: float = 0.01) -> dict[str, float]:
    """Balanced accuracy and recall at the threshold flagging `fpr` of the control (Alpaca) scores."""
    t = float(np.quantile(control, 1 - fpr))
    recall = float((s[y == 1] > t).mean()) if (y == 1).any() else float("nan")
    spec = float((s[y == 0] <= t).mean()) if (y == 0).any() else float("nan")
    return {"threshold": t, "recall": recall, "balacc": (recall + spec) / 2}


# --- model selection -----------------------------------------------------------------------------


def _folds(kind: str, y: np.ndarray, groups: np.ndarray | None, n: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(len(y))
    if kind == "sets":  # leave one training set out; groups = set ids
        return [(idx[groups != g], idx[groups == g]) for g in np.unique(groups)]
    if kind == "grouped":
        return list(GroupKFold(n_splits=n).split(idx, y, groups))
    n = max(2, min(n, int(y.sum()), int((1 - y).sum())))
    return list(StratifiedKFold(n_splits=n, shuffle=True, random_state=SEED).split(idx, y))


def select(family: str, feats: np.ndarray, y: np.ndarray, folds, ks: tuple[int, ...] | None,
           c_grid=C_GRID) -> pd.DataFrame:
    """Mean validation AUROC over `folds` for every (k, C); k None = all columns."""
    rows = []
    grid = {"lr": c_grid, "mlp": MLP_ALPHAS}.get(family, (None,))
    for k in ks or (None,):
        f = feats if k is None else feats[:, :k]
        for c in grid:
            aucs = []
            for tr, va in folds:
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                    continue
                aucs.append(auroc(y[va], fit(family, f[tr], y[tr], c).score(f[va])))
            rows.append({"k": k, "C": c, "inner_auroc": float(np.mean(aucs)) if aucs else float("nan")})
    return pd.DataFrame(rows)


def best_per_k(table: pd.DataFrame) -> pd.DataFrame:
    """For each k, the C with the highest inner AUROC (the smallest C on ties), and which k wins overall."""
    t = table.sort_values(["k", "inner_auroc", "C"], ascending=[True, False, True], na_position="last")
    per_k = t.groupby("k", dropna=False).head(1).copy()
    winner = per_k["inner_auroc"].idxmax()
    per_k["chosen"] = per_k.index == winner
    return per_k


# --- protocols -----------------------------------------------------------------------------------


@dataclass
class Store:
    """Result rows plus the held-out and Alpaca scores behind them, keyed by the row's identity."""

    rows: list[dict] = field(default_factory=list)
    scores: dict[tuple, np.ndarray] = field(default_factory=dict)

    def add(self, key: dict, y: np.ndarray, s: np.ndarray, control: np.ndarray | None, op: dict | None = None,
            **extra) -> None:
        row = key | extra | {"auroc": auroc(y, s), "n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())}
        row |= op if op is not None else operating_point(y, s, control)
        self.rows.append(row)
        self.scores[tuple(sorted((k, v) for k, v in key.items()))] = s.astype(np.float32)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _fit_and_score(family, fs, train_x, train_y, tests: dict[str, Split], control_x, sel: pd.DataFrame, key, store,
                   basis=None):
    feats_tr = coordinates(fs, train_x, basis)
    feats_ctrl = coordinates(fs, control_x, basis)
    feats_te = {n: coordinates(fs, s.x, basis) for n, s in tests.items()}
    for r in sel.itertuples():
        cols = slice(None) if pd.isna(r.k) else slice(None, int(r.k))
        model = fit(family, feats_tr[:, cols], train_y, None if pd.isna(r.C) else float(r.C))
        ctrl = model.score(feats_ctrl[:, cols])
        k = int(r.k) if not pd.isna(r.k) else (feats_tr.shape[1])
        for name, s in tests.items():
            store.add(key | {"k": k, "test": name}, s.y, model.score(feats_te[name][:, cols]), ctrl,
                      C=r.C, inner_auroc=r.inner_auroc, chosen=bool(r.chosen))


def protocol_b(family: str, fs: FeatureSet, data: dict[str, Split], lb5: list[str], control: str, layer: int,
               pooling: str, store: Store, fixed_c: dict[str, float] | None = None) -> dict[str, float]:
    """Leave one set out. Returns the C chosen per held-out set (for arms that reuse the raw arm's C)."""
    chosen_c = {}
    for held in lb5:
        train = [n for n in lb5 if n != held]
        x = np.concatenate([data[n].x for n in train])
        y = np.concatenate([data[n].y for n in train])
        sets = np.concatenate([np.full(len(data[n].y), i) for i, n in enumerate(train)])
        basis = fs.per_fold(x) if fs.per_fold is not None else None
        feats = coordinates(fs, x, basis)
        ks = fs.usable_ks(len(basis) if basis is not None else None) if (fs.basis is not None or basis is not None) else None
        if fixed_c is not None:
            sel = pd.DataFrame([{"k": np.nan, "C": fixed_c[held], "inner_auroc": np.nan, "chosen": True}])
        else:
            sel = best_per_k(select(family, feats, y, _folds("sets", y, sets), ks))
        chosen_c[held] = float(sel.loc[sel["chosen"], "C"].iloc[0]) if family in ("lr", "mlp") else float("nan")
        key = {"protocol": "B", "feature_set": fs.name, "fs_family": fs.family, "seed": fs.seed, "layer": layer,
               "pooling": pooling, "train": "others", "probe": family}
        _fit_and_score(family, fs, x, y, {held: data[held]}, data[control].x, sel, key, store, basis)
    return chosen_c


def protocol_c(family: str, fs: FeatureSet, data: dict[str, Split], lb5: list[str], control: str, layer: int,
               pooling: str, store: Store) -> None:
    """Single source: train on one set, test on every other; the diagonal by 5-fold CV."""
    for src in lb5:
        x, y = data[src].x, data[src].y
        basis = fs.per_fold(x) if fs.per_fold is not None else None
        feats = coordinates(fs, x, basis)
        ks = fs.usable_ks(len(basis) if basis is not None else None) if (fs.basis is not None or basis is not None) else None
        sel = best_per_k(select(family, feats, y, _folds("strat", y, None), ks))
        key = {"protocol": "C", "feature_set": fs.name, "fs_family": fs.family, "seed": fs.seed, "layer": layer,
               "pooling": pooling, "train": src, "probe": family}
        _fit_and_score(family, fs, x, y, {t: data[t] for t in lb5 if t != src}, data[control].x, sel, key, store,
                       basis)
        ctrl_x = data[control].x
        fold_feats = []  # each fold's features once (a per-fold basis is fitted once), sliced per k below
        for tr, te in _folds("strat", y, None):
            b = fs.per_fold(x[tr]) if fs.per_fold is not None else None
            fold_feats.append((tr, te, coordinates(fs, x[tr], b), coordinates(fs, x[te], b),
                               coordinates(fs, ctrl_x, b)))
        for r in sel.itertuples():  # diagonal: out-of-fold scores with each k's chosen C
            cols = slice(None) if pd.isna(r.k) else slice(None, int(r.k))
            oof = np.zeros(len(y))
            ops = []
            for tr, te, ftr, fte, fctrl in fold_feats:
                model = fit(family, ftr[:, cols], y[tr], None if pd.isna(r.C) else float(r.C))
                oof[te] = model.score(fte[:, cols])
                ops.append(operating_point(y[te], oof[te], model.score(fctrl[:, cols])))
            k = int(r.k) if not pd.isna(r.k) else feats.shape[1]
            # fold models have their own scales: AUROC from the pooled scores, the operating point per fold, averaged
            op = {m: float(np.nanmean([o[m] for o in ops])) for m in ("threshold", "recall", "balacc")}
            store.add(key | {"k": k, "test": src}, y, oof, None, op=op,
                      C=r.C, inner_auroc=r.inner_auroc, chosen=bool(r.chosen), diagonal=True)


def protocol_a(family: str, fs: FeatureSet, data: dict[str, Split], train: str, lb5: list[str], control: str,
               layer: int, pooling: str, store: Store) -> None:
    """Train on the instructed pairs (statement-grouped CV for k and C), test on each Liars' Bench set."""
    x, y, groups = data[train].x, data[train].y, data[train].groups
    basis = fs.per_fold(x) if fs.per_fold is not None else None
    feats = coordinates(fs, x, basis)
    ks = fs.usable_ks(len(basis) if basis is not None else None) if (fs.basis is not None or basis is not None) else None
    sel = best_per_k(select(family, feats, y, _folds("grouped", y, groups), ks))
    key = {"protocol": "A", "feature_set": fs.name, "fs_family": fs.family, "seed": fs.seed, "layer": layer,
           "pooling": pooling, "train": train, "probe": family}
    _fit_and_score(family, fs, x, y, {n: data[n] for n in lb5}, data[control].x, sel, key, store, basis)


def zero_shot(directions: dict[str, tuple[np.ndarray, int]], data: dict[str, Split], lb5: list[str], control: str,
              layer: int, pooling: str, store: Store) -> None:
    """Untrained single directions with a sign fixed in advance: score = sign * <x, unit direction>."""
    for name, (vec, sign) in directions.items():
        u = la.unit(vec)
        ctrl = sign * (data[control].x @ u)
        for t in lb5:
            store.add({"protocol": "zero-shot", "feature_set": f"zs_{name}", "fs_family": "zero_shot", "seed": 0,
                       "layer": layer, "pooling": pooling, "train": "none", "probe": "direction", "k": 1, "test": t},
                      data[t].y, sign * (data[t].x @ u), ctrl, C=np.nan, inner_auroc=np.nan, chosen=True)


# --- bootstrap ------------------------------------------------------------------------------------


@dataclass
class Resamples:
    """Stratified bootstrap weights for one test set, shared by every score vector compared on it."""

    pos: np.ndarray  # (n_boot, n_pos) counts
    neg: np.ndarray  # (n_boot, n_neg) counts

    @classmethod
    def draw(cls, y: np.ndarray, n_boot: int = 1000, seed: int = SEED) -> "Resamples":
        rng = np.random.default_rng(seed)
        n1, n0 = int(y.sum()), int(len(y) - y.sum())
        return cls(rng.multinomial(n1, np.full(n1, 1 / n1), size=n_boot).astype(np.float32),
                   rng.multinomial(n0, np.full(n0, 1 / n0), size=n_boot).astype(np.float32))

    def aurocs(self, y: np.ndarray, s: np.ndarray) -> np.ndarray:
        """AUROC of every resample: weighted Mann-Whitney over the (positive, negative) comparison matrix."""
        sp, sn = s[y == 1], s[y == 0]
        m = (sp[:, None] > sn[None, :]).astype(np.float32) + 0.5 * (sp[:, None] == sn[None, :])
        return ((self.pos @ m) * self.neg).sum(1) / (self.pos.shape[1] * self.neg.shape[1])


def ci(values: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(values, [0.025, 0.975])
    return float(lo), float(hi)
