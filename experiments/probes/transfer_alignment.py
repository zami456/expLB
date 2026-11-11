"""Does alignment with each test set's own probe explain the transfer matrix?

Ying et al., "The Truthfulness Spectrum Hypothesis" (arXiv 2602.20273), predict how well a probe trained on one
domain transfers to another from a covariance-weighted ("Mahalanobis") cosine between the two domains' probes,

    CosΣ(w_A, w_B) = w_Aᵀ Σ w_B / sqrt(w_Aᵀ Σ w_A · w_Bᵀ Σ w_B),

with Σ the test domain's pooled sample covariance (`get_covariance` and `compute_transfer_metrics` in their
`scripts/geometric_analysis.ipynb`, github.com/zfying/truth_spec). They report R² = 0.98 for it against 0.56 for the
plain cosine. With that Σ the cosine is exactly the Pearson correlation of the two probes' scores on the test set, so
no 8192 x 8192 matrix is ever formed.

Here the same question is put to Liars' Bench. The detectors are the probes trained on one set and tested on another,
as in the transfer matrices (D1 on raw activations, D4 on the extended PCs), and the zero-shot reads (D2, D3, D5-D9, K,
D5 + K, D9 + K). Each test set's own probe is `persona_ood.make_probe()` fitted in the folds `persona_ood.diagonal_auroc`
uses. The alignment is kept two ways: the paper's (the mean of the fold directions, scored on every row) and a held-out
one (each fold's probe on the rows it did not see). Under Gaussian classes with a shared covariance the held-out one
fixes the transfer AUROC with nothing fitted,

    AUROC(A on B) = Φ(ρ_within · d_own / √2),

with ρ_within the correlation of A's scores with the own probe's inside each class, and d_own the own probe's held-out
separation (its class-mean gap over the pooled within-class sd).

    scores    every detector's and the own probe's scores on every test set, one file per layer; resumable
    check     the scores reproduce the stored transfer matrices, zero-shot AUROCs and layer-average scores, and the
              covariance-weighted cosine equals the score correlation on real activations
    analyse   one row per cell (layer, detector, test set), the own probes, and the fits per layer
    figure    layer 40's three panels, and the fits over depth

    python experiments/probes/transfer_alignment.py scores --layers 40
    python experiments/probes/transfer_alignment.py check
    python experiments/probes/transfer_alignment.py analyse
    python experiments/probes/transfer_alignment.py figure

Outputs go to `results/transfer_alignment/`.
"""

import pickle
import time
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm, spearmanr
from sklearn.model_selection import StratifiedKFold

from deception_detection.log import logger
from experiments.knowledge import k_robustness as kr
from experiments.persona import persona_ood as P
from experiments.knowledge import recall_denial as rd
from experiments.persona.layer_sweep import ALL_LAYERS, base_run, pooled_dir, results_dir
from deception_detection.utils import REPO_ROOT

OUT = REPO_ROOT / "results" / "transfer_alignment"
RESULTS = REPO_ROOT / "results"
SEED = P.SEED  # the seed of P.diagonal_auroc's folds, so the own probe's held-out scores are the stored diagonal's
N_FOLDS = 5
N_RANDOM = 10
N_BOOT = 1000
TOL = 1e-3
# A refitted probe that needs hundreds of lbfgs iterations (HP-C's 277 and HP-KR's 550 at layer 40) stops at a
# slightly different point than the stored run's did, and moves a cell on the 267-row CG set by up to 0.002. Refits here
# are deterministic, so trained cells get this looser bound and anything past TOL is listed.
TRAINED_TOL = 5e-3
IDENTITY_TOL = 1e-10
CENTRE = 40  # the layer fixed in advance
EXTRA_LAYERS = (20, 40, 44)  # where Gender-Secret, the AISI split and the AISI-trained probe were read
GS = "gender_secret (fine-tuned read)"
AISI = "aisi_llama__validation"
EXTRA = (GS, AISI)
REPE = "repe_honesty__plain"
IT = "insider_trading_cadenza__plain"
SHORT = rd.SHORT | {GS: "GS", AISI: "AISI"}

# D5-D9 are the 36-persona bank's contrasts, as `zero_shot.csv` reads them
PERSONA_AXES = {"D5": "deception", "D6": "sycophancy", "D7": "unified", "D8": "denial", "D9": "extended"}
ZERO_SHOT_COLUMNS = {"D5": "auroc_axis", "D6": "auroc_syc_axis", "D7": "auroc_uni_axis",
                     "D8": "auroc_denial_axis", "D9": "auroc_ext_axis"}
LAYER_AVERAGE_ARMS = {"D5": "D5", "D8": "D8", "D9": "D9", "K": "K", "D9 + K": "D9 + K", "D2": "probe"}
FIXED = ("D2", "D3", "D5", "D6", "D7", "D8", "D9", "K", "D5 + K", "D9 + K")
TRAINED = {"D1": "matrix_baseline.csv", "D4": "matrix_ext_pc3.csv"}
NULL = tuple(f"R{i}" for i in range(N_RANDOM))
AISI_RUN = "aisi-llama-L{layer}"
AISI_TRAIN = "aisi_llama__pooled"
IN_DISTRIBUTION = {("D2", REPE), ("D3", AISI)}
# the inversions named before the analysis: (detector, layer, test set)
NAMED_INVERSIONS = (("D6", 20, P.HPKR), ("D2", 20, P.HPKR), ("D5", 20, P.HPKR), ("D2", 40, IT), ("D2", 44, IT))
NEAR_CHANCE = 0.05  # a cell this close to 0.5 has no direction for its alignment to agree with


# --- the measures ----------------------------------------------------------------------------------------------

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """The correlation of two score vectors. With Σ the pooled covariance of the rows they score, it is the
    covariance-weighted cosine of their directions (`mahalanobis_cosine`), whatever the scores' offsets."""
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def mahalanobis_cosine(w_a: np.ndarray, w_b: np.ndarray, z: np.ndarray) -> float:
    """The paper's Eq. 1 with Σ the pooled sample covariance of z (ddof 1), as `get_covariance` takes it. Σw is formed
    as X_cᵀ(X_c w) / (n - 1), never Σ itself."""
    c = z - z.mean(0)
    sigma_a, sigma_b = c.T @ (c @ w_a) / (len(z) - 1), c.T @ (c @ w_b) / (len(z) - 1)
    return float(w_a @ sigma_b / np.sqrt((w_a @ sigma_a) * (w_b @ sigma_b)))


def class_centred(s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Scores minus their class's mean: what is left of them once the label is known."""
    out = np.asarray(s, dtype=np.float64).copy()
    for label in (0, 1):
        out[y == label] -= out[y == label].mean()
    return out


def within_class_corr(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
    """The correlation of two scores inside each class, pooled over the classes: the covariance-weighted cosine with
    Σ the pooled within-class covariance (the notebook's `compute_within_class_covariance`)."""
    return pearson(class_centred(a, y), class_centred(b, y))


def separation(s: np.ndarray, y: np.ndarray) -> float:
    """d': the class-mean gap of a score over its pooled within-class sd (ddof 2)."""
    resid = class_centred(s, y)
    return float((s[y == 1].mean() - s[y == 0].mean()) / np.sqrt(resid @ resid / (len(s) - 2)))


def predicted_auroc(rho_within: float, d_own: float) -> float:
    """Φ(ρ_within · d_own / √2). For Gaussian classes with a shared covariance, a direction's separation is its
    within-class correlation with the best linear direction times the best one's separation, and AUROC = Φ(d / √2).
    The own probe stands in for the best direction."""
    return float(norm.cdf(rho_within * d_own / np.sqrt(2)))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def stratified_draws(y: np.ndarray, n_draws: int, rng: np.random.Generator) -> np.ndarray:
    """[n_draws, n] row indices, each class resampled within itself, as `P.zero_shot_operating_table` resamples."""
    parts = [rng.choice(np.flatnonzero(y == label), size=(n_draws, int((y == label).sum()))) for label in (0, 1)]
    return np.concatenate(parts, axis=1)


def rowwise_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


# --- the probes ------------------------------------------------------------------------------------------------

def own_folds(y: np.ndarray) -> np.ndarray:
    """Each row's fold under the splitter `P.diagonal_auroc` uses (it reads only the labels), so the own probe's
    held-out scores are the ones behind the stored diagonal."""
    n_splits = min(N_FOLDS, int(y.sum()), int((1 - y).sum()))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold = np.full(len(y), -1)
    for i, (_, test) in enumerate(splitter.split(np.zeros((len(y), 1)), y)):
        fold[test] = i
    return fold


def raw_direction(probe) -> np.ndarray:
    """A fitted `P.make_probe()` as a direction in its input space. It scores coef · (x - μ) / σ with no intercept,
    so the direction is coef / σ."""
    scaler, logistic = probe[0], probe[-1]
    return logistic.coef_.ravel() / scaler.scale_


def own_probe(z: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """A test set's own probe: each fold's probe scores the rows it did not see (`own_held`), and the mean of the fold
    directions, as the authors' `get_mean_direction` takes it, scores every row (`own_mean`)."""
    fold = own_folds(y)
    held = np.zeros(len(y))
    directions = []
    for f in range(fold.max() + 1):
        test = fold == f
        probe = P.make_probe().fit(z[~test], y[~test])
        held[test] = probe.decision_function(z[test])
        directions.append(raw_direction(probe))
    direction = np.mean(directions, axis=0)
    return {"fold": fold, "own_held": held, "own_dir": direction, "own_mean": z @ direction}


def trained(z: dict[str, np.ndarray], y: dict[str, np.ndarray], train_sets: list[str], test_sets: list[str],
            basis: np.ndarray | None = None) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, np.ndarray]]:
    """D1 (no basis) or D4 (basis = the extended PCs): a `P.make_probe()` fitted on all of each training set, as
    `P.transfer_matrix` fits its off-diagonal cells, its scores on every other set, and its direction in activation
    space."""
    def features(x: np.ndarray) -> np.ndarray:
        return x if basis is None else x @ basis.T

    scores, directions = {}, {}
    for a in train_sets:
        probe = P.make_probe().fit(features(z[a]), y[a])
        w = raw_direction(probe)
        directions[a] = w if basis is None else basis.T @ w
        for b in test_sets:
            if b != a:
                scores[(a, b)] = probe.decision_function(features(z[b]))
    return scores, directions


def knowledge_direction(layer: int) -> np.ndarray:
    """K rebuilt from the stored reads, as `layer_average.scores` rebuilds it, and checked against the stored
    direction where there is one."""
    X, known, _ = kr.item_means(layer)
    k = rd._unit(X.T @ kr.weights(known, np.flatnonzero(known), np.flatnonzero(~known)))
    stored = dict(np.load(rd.OUT / "directions.npz"))
    key = f"L{layer}|none|general"
    if key in stored and not np.allclose(k, stored[key], atol=1e-6):
        raise AssertionError(f"layer {layer}: rebuilt K differs from the stored one")
    return k


def detector_direction(run: Path) -> np.ndarray:
    """A runner probe (`detector.pt`) as a direction in activation space. It scores ((z - mean) / scale) @ direction
    (`P.run_probe`), which is z @ (direction / scale) less a constant."""
    with open(run / "detector.pt", "rb") as f:
        detector = pickle.load(f)
    return detector["directions"].double().numpy().ravel() / detector["scaler_scale"].double().numpy().ravel()


def extra_sets(layer: int) -> dict[str, P.Pooled]:
    """Gender-Secret's fine-tuned read and the AISI held-out split, loaded as `aisi_probe.compare` loads them."""
    from experiments.probes import aisi_probe

    gs = np.load(aisi_probe.GS_DIR / f"adapted_L{layer}.npz", allow_pickle=False)
    out = {GS: P.Pooled(name=GS, z=gs["pooled"].astype(np.float64),
                        y=(gs["labels"].astype(str) == "DECEPTIVE").astype(int))}
    with np.load(aisi_probe.VALIDATION_MEANS / f"validation_L{layer}.npz", allow_pickle=False) as saved:
        if not saved["done"].all():
            raise ValueError(f"the AISI split is not fully read at layer {layer}")
        out[AISI] = P.Pooled(name=AISI, z=saved["pooled"][:, 0].astype(np.float64),
                             y=saved["deceptive"].astype(int))
    return out


# --- scores ----------------------------------------------------------------------------------------------------

def _layers(layers) -> list[int]:
    if isinstance(layers, (int, str)):
        return [int(layers)]
    return [int(x) for x in layers]


def scores(layers=ALL_LAYERS, force: bool = False) -> None:
    """Every detector's and the own probe's scores on every test set, into `scores_L{layer}.npz`. A layer already
    written is skipped unless `force`, so an interrupted sweep resumes."""
    from deception_detection import persona_pcs as pcs

    OUT.mkdir(parents=True, exist_ok=True)
    for layer in _layers(layers):
        target = OUT / f"scores_L{layer}.npz"
        if target.exists() and not force:
            logger.info(f"layer {layer}: {target.name} exists; skipped")
            continue
        start = time.time()
        pooled = P.load_pooled(pooled_dir(layer))
        control = pooled[P.CONTROL_DATASET]
        order = [n for n, d in sorted(pooled.items()) if d.two_class and n != P.CONTROL_DATASET]
        sets = {n: pooled[n] for n in order} | (extra_sets(layer) if layer in EXTRA_LAYERS else {})
        saved: dict[str, np.ndarray] = {"layer": np.array(layer), "order": np.array(order),
                                        "sets": np.array(list(sets))}

        bank = pcs.load_bank_at_layer(layer, pcs.EXTENDED_VECTORS_DIR)
        directions = {name: bank.axis_unit(axis) for name, axis in PERSONA_AXES.items()}
        directions["K"] = knowledge_direction(layer)
        directions["D5 + K"] = rd.standardised_sum(directions["D5"], directions["K"], control.z)
        directions["D9 + K"] = rd.standardised_sum(directions["D9"], directions["K"], control.z)
        random = P.random_directions(control.z.shape[1], N_RANDOM, np.random.default_rng(SEED))
        directions |= dict(zip(NULL, random))
        scorers = {name: (lambda x, w=w: x @ w) for name, w in directions.items()}

        # the RepE-trained probe is checked against its own run where the run scored the benchmark (20, 40, 44)
        run = base_run(layer)
        five = {n: pooled[n] for n in order if n in P.SHARED_DATASETS}
        scorers["D2"] = P.base_probe_arm(run, layer, five, control, check=layer in EXTRA_LAYERS)
        directions["D2"] = detector_direction(run)
        if layer in EXTRA_LAYERS:
            aisi_run = RESULTS / AISI_RUN.format(layer=layer)
            cfg = yaml.safe_load((aisi_run / "cfg.yaml").read_text())
            if cfg.get("detect_layers") != [layer] or cfg.get("train_data") != AISI_TRAIN:
                raise ValueError(f"{aisi_run.name} is not the AISI-trained probe at layer {layer}")
            scorers["D3"] = P.run_probe(aisi_run)
            directions["D3"] = detector_direction(aisi_run)

        for name, ds in sets.items():
            saved[f"labels|{name}"] = ds.y
            for key, value in own_probe(ds.z, ds.y).items():
                saved[f"{key}|{name}"] = value
            for detector, fn in scorers.items():
                saved[f"{detector}|{name}"] = fn(ds.z)
        for detector, w in directions.items():
            saved[f"dir|{detector}"] = w

        ext_geom = pcs.fit_geometry(bank, k=3,
                                    deception=pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS,
                                    honesty=pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS,
                                    axis_name="extended")
        z = {n: ds.z for n, ds in sets.items()}
        y = {n: ds.y for n, ds in sets.items()}
        for family, basis in (("D1", None), ("D4", ext_geom.components[:3])):
            cell_scores, cell_directions = trained(z, y, order, list(sets), basis)
            for (a, b), s in cell_scores.items():
                saved[f"{family}|{a}|{b}"] = s
            for a, w in cell_directions.items():
                saved[f"dir|{family}|{a}"] = w

        np.savez_compressed(target, **saved)
        logger.info(f"layer {layer}: {len(sets)} test sets, {len(scorers)} fixed reads, "
                    f"{2 * len(order)} trained probes ({time.time() - start:.0f} s); wrote {target.name}")
        del pooled, sets, z


def _score_files(layers=None) -> list[Path]:
    files = sorted(OUT.glob("scores_L*.npz"), key=lambda p: int(p.stem.split("_L")[1]))
    if layers is not None:
        wanted = set(_layers(layers))
        files = [p for p in files if int(p.stem.split("_L")[1]) in wanted]
    if not files:
        raise FileNotFoundError(f"no scores in {OUT}; run `scores` first")
    return files


def _cells(saved: dict[str, np.ndarray], test_set: str) -> list[tuple[str, str, str, np.ndarray, np.ndarray]]:
    """Every detector scored on one test set: (family, detector, trained on, scores, direction)."""
    out = []
    for detector in FIXED + NULL:
        if f"{detector}|{test_set}" in saved:
            family = "random" if detector in NULL else "fixed"
            out.append((family, detector, "", saved[f"{detector}|{test_set}"], saved[f"dir|{detector}"]))
    for family in TRAINED:
        for a in saved["order"]:
            if f"{family}|{a}|{test_set}" in saved:
                out.append((family, family, str(a), saved[f"{family}|{a}|{test_set}"], saved[f"dir|{family}|{a}"]))
    return out


# --- check -----------------------------------------------------------------------------------------------------

def check(layers=None) -> None:
    """Raise unless the scores reproduce what is stored: D1's matrix (both its off-diagonal cells and the pooled
    held-out diagonal) and D4's, the zero-shot AUROCs of D5-D9, and layer_average's raw scores. At layer 40 the
    covariance-weighted cosine is also computed on the activations themselves and must equal the score
    correlation."""
    rows = []
    averaged = np.load(RESULTS / "layer_average" / "scores.npz")
    for path in _score_files(layers):
        saved = dict(np.load(path))
        layer, order = int(saved["layer"]), [str(n) for n in saved["order"]]
        stored = results_dir(layer)
        for family, csv in TRAINED.items():
            matrix = pd.read_csv(stored / csv, index_col=0)
            for a in order:
                for b in order:
                    y = saved[f"labels|{b}"]
                    if a == b and family != "D1":
                        continue  # D4's own-set cells are not refitted here
                    s = saved[f"own_held|{b}"] if a == b else saved[f"{family}|{a}|{b}"]
                    rows.append({"layer": layer, "check": f"{family} transfer matrix",
                                 "item": f"{SHORT[a]} -> {SHORT[b]}", "got": P.auroc(y, s),
                                 "stored": float(matrix.loc[a, b]), "tol": TRAINED_TOL})
        zero_shot = pd.read_csv(stored / "zero_shot.csv").set_index("dataset")
        for detector, column in ZERO_SHOT_COLUMNS.items():
            for b in order:
                rows.append({"layer": layer, "check": "zero_shot.csv", "item": f"{detector} on {SHORT[b]}",
                             "got": P.auroc(saved[f"labels|{b}"], saved[f"{detector}|{b}"]),
                             "stored": float(zero_shot.loc[b, column]), "tol": TOL})
        for detector, arm in LAYER_AVERAGE_ARMS.items():
            for b in order:
                ours, theirs = saved[f"{detector}|{b}"], averaged[f"{arm}|L{layer}|{b}"]
                rows.append({"layer": layer, "check": "layer_average scores (relative)", "item": f"{detector} on "
                             f"{SHORT[b]}", "got": float(np.abs(ours - theirs).max() / np.abs(theirs).max()),
                             "stored": 0.0, "tol": 1e-6})
        if layer == CENTRE:
            rows += _identity_rows(saved, layer)
    table = pd.DataFrame(rows)
    table["diff"] = (table["got"] - table["stored"]).abs()
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "checks.csv", index=False)
    bad = table[table["diff"] > table["tol"]]
    logger.info(f"{len(table)} checks, largest difference per check:\n"
                f"{table.groupby('check')['diff'].max().to_string()}")
    loose = table[(table["diff"] > TOL) & (table["diff"] <= table["tol"])]
    if len(loose):
        logger.warning(f"{len(loose)} trained cells within {TRAINED_TOL} but past {TOL}:\n"
                       f"{loose.to_string(index=False)}")
    if len(bad):
        raise AssertionError(f"{len(bad)} checks fail:\n{bad.to_string(index=False)}")
    logger.info(f"all checks pass; wrote {OUT / 'checks.csv'}")


def _identity_rows(saved: dict[str, np.ndarray], layer: int) -> list[dict]:
    """On real activations: Eq. 1 with Σ the test set's pooled covariance against the correlation of the scores, for
    a trained probe and for a fixed read."""
    pooled = P.load_pooled(pooled_dir(layer))
    rows = []
    for detector, a, b in (("D1", "convince_me__plain", IT), ("D9", "", P.HPKR), ("D2", "", P.HPKR)):
        key = f"{detector}|{a}|{b}" if a else f"{detector}|{b}"
        w = saved[f"dir|{detector}|{a}"] if a else saved[f"dir|{detector}"]
        explicit = mahalanobis_cosine(w, saved[f"own_dir|{b}"], pooled[b].z)
        rows.append({"layer": layer, "check": "CosΣ = score correlation", "item": f"{detector} on {SHORT[b]}",
                     "got": explicit, "stored": pearson(saved[key], saved[f"own_mean|{b}"]), "tol": IDENTITY_TOL})
    return rows


# --- analyse ---------------------------------------------------------------------------------------------------

def _r2_linear(x: np.ndarray, y: np.ndarray) -> float:
    slope, intercept = np.polyfit(x, y, 1)
    return _r2(slope * x + intercept, y)


def _r2(prediction: np.ndarray, y: np.ndarray) -> float:
    return float(1 - ((y - prediction) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def fit_row(cells: pd.DataFrame) -> dict:
    """How well each measure explains the cells' AUROC: R² of a straight-line fit (the paper's), Spearman ρ², the
    Gaussian prediction's R² and MAE with nothing fitted, and how often the alignment's sign is the AUROC's side of
    0.5."""
    y = cells["auroc"].to_numpy()
    row = {"cells": len(cells)}
    for measure in ("rho_paper", "cos_plain", "rho_heldout", "rho_within", "r_label"):
        x = cells[measure].to_numpy()
        row[f"r2_{measure}"] = _r2_linear(x, y)
        row[f"spearman2_{measure}"] = float(spearmanr(x, y)[0] ** 2)
    row["r2_pred"] = _r2(cells["auroc_pred"].to_numpy(), y)
    row["mae_pred"] = float(np.abs(cells["auroc_pred"] - cells["auroc"]).mean())
    signed = cells[(cells["auroc"] - 0.5).abs() >= NEAR_CHANCE]
    row["sign_cells"] = len(signed)
    row["sign_agreement"] = float((np.sign(signed["rho_paper"]) == np.sign(signed["auroc"] - 0.5)).mean())
    return row


def analyse() -> None:
    """One row per cell, the own probes, the fits per layer and family, the inverted cells and layer 40's
    decomposition."""
    rng = np.random.default_rng(SEED)
    cells, own = [], []
    for path in _score_files():
        saved = dict(np.load(path))
        layer = int(saved["layer"])
        for b in (str(n) for n in saved["sets"]):
            y, fold = saved[f"labels|{b}"], saved[f"fold|{b}"]
            held, mean_scores, own_dir = saved[f"own_held|{b}"], saved[f"own_mean|{b}"], saved[f"own_dir|{b}"]
            folds = [fold == f for f in range(fold.max() + 1)]
            d_own = float(np.mean([separation(held[m], y[m]) for m in folds]))
            auroc_own = float(np.mean([P.auroc(y[m], held[m]) for m in folds]))
            own.append({"layer": layer, "test_set": SHORT[b], "n": len(y), "n_deceptive": int(y.sum()),
                        "auroc_own": auroc_own, "auroc_own_pooled": P.auroc(y, held), "d_own": d_own,
                        "gauss_auroc_own": float(norm.cdf(d_own / np.sqrt(2)))})
            draws = stratified_draws(y, N_BOOT, rng)  # shared by every detector on this set
            own_draws = mean_scores[draws]
            for family, detector, trained_on, s, w in _cells(saved, b):
                rho_within = float(np.mean([within_class_corr(s[m], held[m], y[m]) for m in folds]))
                row = {"layer": layer, "family": family, "detector": detector,
                       "trained_on": SHORT.get(trained_on, trained_on), "test_set": SHORT[b],
                       "extra_set": b in EXTRA, "in_distribution": (detector, b) in IN_DISTRIBUTION,
                       "auroc": P.auroc(y, s), "rho_paper": pearson(s, mean_scores),
                       # post hoc: the label itself in the own probe's place (the point-biserial correlation)
                       "r_label": pearson(s, y.astype(np.float64)),
                       "rho_heldout": float(np.mean([pearson(s[m], held[m]) for m in folds])),
                       "rho_within": rho_within, "cos_plain": cosine(w, own_dir), "auroc_own": auroc_own,
                       "d_own": d_own, "auroc_pred": predicted_auroc(rho_within, d_own)}
                if family != "random":
                    lo, hi = np.quantile(rowwise_pearson(s[draws], own_draws), [0.025, 0.975])
                    row |= {"rho_paper_lo": float(lo), "rho_paper_hi": float(hi)}
                cells.append(row)
        logger.info(f"layer {layer} analysed")
    cells = pd.DataFrame(cells)
    own = pd.DataFrame(own)
    OUT.mkdir(parents=True, exist_ok=True)
    cells.round(4).to_csv(OUT / "cells.csv", index=False)
    own.round(4).to_csv(OUT / "own_probes.csv", index=False)

    fits = []
    usable = cells[~cells["in_distribution"]]
    for scope, extra in (("six matrix sets", False), ("GS and AISI", True)):
        for layer in [*sorted(usable["layer"].unique()), "all"]:
            here = usable[usable["extra_set"] == extra]
            here = here if layer == "all" else here[here["layer"] == layer]
            groups = {"D1": here[here["family"] == "D1"], "D4": here[here["family"] == "D4"],
                      "fixed": here[here["family"] == "fixed"], "trained and fixed": here[here["family"] != "random"],
                      "random": here[here["family"] == "random"]}
            for family, group in groups.items():
                if len(group) >= 3:
                    fits.append({"scope": scope, "layer": layer, "family": family} | fit_row(group))
    fits = pd.DataFrame(fits)
    fits.round(4).to_csv(OUT / "fits.csv", index=False)

    named = {(d, layer, SHORT[b]) for d, layer, b in NAMED_INVERSIONS}
    key = list(zip(usable["detector"], usable["layer"], usable["test_set"]))
    usable = usable.assign(named=[k in named and t == "" for k, t in zip(key, usable["trained_on"])])
    inverted = usable[(usable["family"] != "random") & ((usable["auroc"] < 0.5 - NEAR_CHANCE) | usable["named"])]
    columns = ["layer", "family", "detector", "trained_on", "test_set", "named", "auroc", "rho_paper",
               "rho_paper_lo", "rho_paper_hi", "rho_within", "auroc_pred"]
    inverted[columns].sort_values(["named", "layer", "auroc"], ascending=[False, True, True]).round(4).to_csv(
        OUT / "inversions.csv", index=False)

    decomposition_table(cells, own).round(4).to_csv(OUT / f"decomposition_L{CENTRE}.csv", index=False)
    complement = pd.DataFrame([row for path in _score_files() for row in complement_rows(dict(np.load(path)))])
    complement.round(4).to_csv(OUT / "complement.csv", index=False)
    headline = fits[(fits["scope"] == "six matrix sets") & fits["layer"].isin([CENTRE, "all"])]
    logger.info(f"fits at layer {CENTRE} and over every layer:\n{headline.round(3).to_string(index=False)}")
    logger.info(f"wrote cells.csv, own_probes.csv, fits.csv, inversions.csv, decomposition_L{CENTRE}.csv and "
                f"complement.csv to {OUT}")


COMPLEMENT_READS = ("D2", "D3", "D8", "D9", "K", "D9 + K")


def _standardised(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / s.std()


def complement_rows(saved: dict[str, np.ndarray]) -> list[dict]:
    """(post hoc) Does a fixed read carry anything its test set's own probe misses? The sum of the own probe's
    held-out scores (standardised within each fold) and the read's (standardised on the set), against each alone.
    No labels enter the sum."""
    layer, rows = int(saved["layer"]), []
    for b in (str(n) for n in saved["order"]):
        y, fold, held = saved[f"labels|{b}"], saved[f"fold|{b}"], saved[f"own_held|{b}"]
        own = held.copy()
        for f in range(fold.max() + 1):
            own[fold == f] = _standardised(held[fold == f])
        for read in COMPLEMENT_READS:
            if f"{read}|{b}" not in saved or (read, b) in IN_DISTRIBUTION:
                continue
            x = _standardised(saved[f"{read}|{b}"])
            rows.append({"layer": layer, "test_set": SHORT[b], "read": read, "auroc_own": P.auroc(y, own),
                         "auroc_read": P.auroc(y, x), "auroc_sum": P.auroc(y, own + x),
                         "rho_within": within_class_corr(x, own, y)})
    return rows


def decomposition_table(cells: pd.DataFrame, own: pd.DataFrame, layer: int = CENTRE) -> pd.DataFrame:
    """Per test set at one layer: how separable the set is for its own probe, the best-aligned fixed read, and D9's
    and D9 + K's observed against predicted AUROC."""
    here = cells[(cells["layer"] == layer) & (cells["family"] == "fixed") & ~cells["in_distribution"]]
    rows = []
    for _, o in own[own["layer"] == layer].iterrows():
        mine = here[here["test_set"] == o["test_set"]]
        if mine.empty:
            continue
        best = mine.loc[mine["rho_within"].idxmax()]
        row = o.to_dict() | {"best_fixed": best["detector"], "best_rho_within": best["rho_within"],
                             "best_auroc": best["auroc"]}
        for detector in ("D9", "D9 + K"):
            d = mine[mine["detector"] == detector].iloc[0]
            row |= {f"{detector} rho_within": d["rho_within"], f"{detector} auroc": d["auroc"],
                    f"{detector} auroc_pred": d["auroc_pred"]}
        rows.append(row)
    return pd.DataFrame(rows)


# --- figure ----------------------------------------------------------------------------------------------------

STYLE = {  # detector -> colour, marker, label
    "D1": ("#bcbab4", "o", "D1 raw-activation probe, trained on one set"),
    "D4": ("#9b8fc2", "s", "D4 extended PCs (k = 3), trained on one set"),
    "D2": ("#898781", "v", "D2 base probe (RepE)"),
    "D3": ("#4d4c48", "^", "D3 base probe (AISI)"),
    "D5": ("#eb6834", "o", "D5 published contrast"),
    "D6": ("#e34948", "o", "D6 sycophancy contrast"),
    "D7": ("#b5399e", "o", "D7 unified contrast"),
    "D8": ("#8ab6e8", "o", "D8 knowledge-denial contrast"),
    "D9": ("#2a78d6", "o", "D9 extended contrast"),
    "K": ("#c9a227", "D", "K knowledge direction"),
    "D5 + K": ("#a0522d", "D", "D5 + K"),
    "D9 + K": ("#2E3A4F", "D", "D9 + K, selected design"),
    "random": ("#d9d7d1", ".", "random directions"),
}


def figure(dpi: int = 200) -> None:
    """Layer 40's cells against each measure, the label control (post hoc), the Gaussian prediction, and whether a
    persona read adds to each set's own probe (post hoc); then the fits over depth."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.persona import persona_ood_figures as F

    cells = pd.read_csv(OUT / "cells.csv")
    fits = pd.read_csv(OUT / "fits.csv")
    complement = pd.read_csv(OUT / "complement.csv")
    here = cells[(cells["layer"] == CENTRE) & ~cells["extra_set"] & ~cells["in_distribution"]]
    fit = fits[(fits["scope"] == "six matrix sets") & (fits["layer"].astype(str) == str(CENTRE))].set_index("family")

    def style(ax, title: str) -> None:
        ax.set_facecolor(F.SURFACE)
        ax.set_title(title, fontsize=10, color=F.INK, loc="left")
        ax.tick_params(colors=F.INK_MUTED, labelsize=8, length=0)
        ax.grid(color="#e6e5e1", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    def scatter(ax, x: str) -> list:
        handles = []
        for key, (colour, marker, label) in STYLE.items():
            if key in ("D1", "D4", "random"):
                rows = here[here["family"] == key]
            else:
                rows = here[(here["family"] == "fixed") & (here["detector"] == key)]
            if rows.empty:
                continue
            handles.append(ax.scatter(rows[x], rows["auroc"], s=14 if key == "random" else 30, color=colour,
                                      marker=marker, label=label, edgecolors="none" if key == "random" else "white",
                                      linewidths=0.4, zorder=3))
        ax.axhline(0.5, color=F.INK_MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.set_ylabel("AUROC on the test set", fontsize=9, color=F.INK_MUTED)
        return handles

    def r2_note(ax, measure: str) -> None:
        ax.axvline(0, color=F.INK_MUTED, lw=0.8, ls=(0, (2, 2)))
        ax.text(0.98, 0.03, f"R² of a line fit: D1 {fit.loc['D1', f'r2_{measure}']:.2f}, "
                f"trained and fixed {fit.loc['trained and fixed', f'r2_{measure}']:.2f}, "
                f"random {fit.loc['random', f'r2_{measure}']:.2f}", transform=ax.transAxes, fontsize=8.2,
                color=F.INK, ha="right", va="bottom")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), facecolor=F.SURFACE)
    panels = (("rho_paper", "correlation with the test set's own probe (= the paper's covariance-weighted cosine)",
               "(a) The paper's alignment"),
              ("cos_plain", "plain cosine with the test set's own probe", "(b) The plain cosine"),
              ("r_label", "correlation with the label itself (point-biserial)", "(c) The label alone (post hoc)"),
              ("rho_within", "correlation with the own probe inside each class, held out",
               "(d) Within-class alignment"))
    for ax, (measure, xlabel, title) in zip(axes.flat, panels):
        handles = scatter(ax, measure)
        r2_note(ax, measure)
        ax.set_xlabel(xlabel, fontsize=9, color=F.INK_MUTED)
        style(ax, title)

    ax = axes[1, 1]
    scatter(ax, "auroc_pred")
    ax.plot([0, 1], [0, 1], color=F.INK_MUTED, lw=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("predicted AUROC, Φ(ρ_within · d_own / √2), nothing fitted", fontsize=9, color=F.INK_MUTED)
    tf = fit.loc["trained and fixed"]
    ax.text(0.98, 0.03, f"trained and fixed: R² {tf['r2_pred']:.2f}, MAE {tf['mae_pred']:.3f}",
            transform=ax.transAxes, fontsize=8.2, color=F.INK, ha="right", va="bottom")
    style(ax, "(e) The Gaussian prediction")

    ax = axes[1, 2]
    sets = [SHORT[n] for n in rd.SHORT]
    rows = complement[(complement["layer"] == CENTRE) & (complement["read"] == "D9")].set_index("test_set")
    x = np.arange(len(sets))
    for offset, column, colour, marker, label in ((-0.18, "auroc_own", "#898781", "s", "own probe, held out"),
                                                  (0.0, "auroc_read", "#2a78d6", "o", "D9 alone"),
                                                  (0.18, "auroc_sum", "#2E3A4F", "D", "own probe + D9")):
        ax.scatter(x + offset, rows.loc[sets, column], s=34, color=colour, marker=marker, label=label, zorder=3)
    ax.set_xticks(x, sets)
    ax.set_ylim(0.5, 1.005)
    ax.set_ylabel("AUROC on the test set", fontsize=9, color=F.INK_MUTED)
    ax.legend(fontsize=8, frameon=False, labelcolor=F.INK, loc="lower left")
    style(ax, "(f) Does D9 add to each set's own probe? (post hoc)")

    fig.suptitle(f"Transfer against alignment with each test set's own probe: Llama-3.3-70B-Instruct, layer "
                 f"{CENTRE}, the six transfer-matrix sets", fontsize=12, color=F.INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.legend(handles=handles, ncol=5, fontsize=8, labelcolor=F.INK, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, 0.0))
    target = OUT / "transfer_alignment.png"
    fig.savefig(target, dpi=int(dpi), bbox_inches="tight", facecolor=F.SURFACE)
    plt.close(fig)
    logger.info(f"wrote {target}")

    depth = fits[(fits["scope"] == "six matrix sets") & (fits["layer"].astype(str) != "all")].copy()
    depth["layer"] = depth["layer"].astype(int)
    series = (("r2_r_label", "#eb6834", "s", "the label alone (post hoc)"),
              ("r2_rho_paper", "#2a78d6", "o", "the paper's alignment"),
              ("r2_rho_within", "#2E3A4F", "D", "within-class alignment"),
              ("r2_cos_plain", "#bcbab4", "o", "plain cosine"))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), facecolor=F.SURFACE)
    for ax, family, title in ((axes[0], "D1", "D1 cells (the paper's setting): R² of a line fit"),
                              (axes[1], "trained and fixed", "Trained and fixed cells: R² of a line fit")):
        rows = depth[depth["family"] == family].sort_values("layer")
        for column, colour, marker, label in series:
            ax.plot(rows["layer"], rows[column], color=colour, marker=marker, ms=4.5, lw=2, label=label)
        ax.set_ylim(min(0.0, float(rows[[s[0] for s in series]].min().min())) - 0.02, 1.0)
        ax.legend(fontsize=8, frameon=False, labelcolor=F.INK)
        style(ax, title)
    ax = axes[2]
    rows = depth[depth["family"] == "trained and fixed"].sort_values("layer")
    ax.plot(rows["layer"], rows["r2_pred"], color="#2a78d6", marker="o", ms=4.5, lw=2, label="R², nothing fitted")
    ax.plot(rows["layer"], rows["mae_pred"], color="#eb6834", marker="s", ms=4.5, lw=2, label="mean absolute error")
    ax.axhline(0, color=F.INK_MUTED, lw=0.8)
    ax.legend(fontsize=8, frameon=False, labelcolor=F.INK)
    style(ax, "The Gaussian prediction, trained and fixed cells")
    for ax in axes:
        ax.axvline(CENTRE, color=F.INK_MUTED, lw=1, ls=(0, (1, 2)))
        ax.set_xticks(sorted(depth["layer"].unique()))
        ax.set_xlabel("layer (block output)", fontsize=9, color=F.INK_MUTED)
    fig.tight_layout()
    target = OUT / "transfer_alignment_depth.png"
    fig.savefig(target, dpi=int(dpi), bbox_inches="tight", facecolor=F.SURFACE)
    plt.close(fig)
    logger.info(f"wrote {target}")


if __name__ == "__main__":
    fire.Fire({"scores": scores, "check": check, "analyse": analyse, "figure": figure})
