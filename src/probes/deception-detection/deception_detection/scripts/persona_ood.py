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

# matches the deployed probe (detectors.py:311-376): standardise, then L2 logistic regression
# with reg_coeff 10 and no intercept
REG_COEFF = 10.0
MAX_ITER = 1000
N_RANDOM_DRAWS = 5
SEED = 42


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


def load_pooled(pooled_dir: Path = POOLED_DIR) -> dict[str, Pooled]:
    """Load every pooled dataset, keeping only HONEST/DECEPTIVE rows."""
    out: dict[str, Pooled] = {}
    for path in sorted(pooled_dir.glob("*.npz")):
        raw = np.load(path, allow_pickle=False)
        labels = raw["labels"].astype(str)
        keep = np.isin(labels, ["HONEST", "DECEPTIVE"])
        name = str(raw["name"])
        dropped = int((~keep).sum())
        if dropped:
            logger.info(f"{name}: dropping {dropped} rows that are neither honest nor deceptive")
        out[name] = Pooled(
            name=name,
            z=raw["pooled"][keep].astype(np.float64),
            y=(labels[keep] == "DECEPTIVE").astype(int),
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


def diagonal_auroc(features: np.ndarray, y: np.ndarray, n_splits: int = 5) -> float:
    """Within-dataset AUROC by stratified CV, so a diagonal cell is never train-on-test."""
    n_splits = min(n_splits, int(y.sum()), int((1 - y).sum()))
    if n_splits < 2:
        return float("nan")
    scores = np.zeros(len(y))
    for train, test in StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED).split(
        features, y
    ):
        probe = make_probe()
        probe.fit(features[train], y[train])  # type: ignore[attr-defined]
        scores[test] = probe.decision_function(features[test])  # type: ignore[attr-defined]
    return auroc(y, scores)


def transfer_matrix(
    featurize: Callable[[np.ndarray], np.ndarray],
    datasets: dict[str, Pooled],
    order: list[str],
) -> pd.DataFrame:
    """AUROC for every (train, test) pair. Diagonal uses CV; off-diagonal trains on all of A."""
    feats = {name: featurize(datasets[name].z) for name in order}
    matrix = pd.DataFrame(index=order, columns=order, dtype=float)

    for train_name in order:
        probe = make_probe()
        probe.fit(feats[train_name], datasets[train_name].y)  # type: ignore[attr-defined]
        for test_name in order:
            if train_name == test_name:
                matrix.loc[train_name, test_name] = diagonal_auroc(
                    feats[train_name], datasets[train_name].y
                )
            else:
                scores = probe.decision_function(feats[test_name])  # type: ignore[attr-defined]
                matrix.loc[train_name, test_name] = auroc(datasets[test_name].y, scores)
    return matrix


def off_diagonal_mean(matrix: pd.DataFrame) -> float:
    values = matrix.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, np.nan)
    return float(np.nanmean(values))


def diagonal_mean(matrix: pd.DataFrame) -> float:
    return float(np.nanmean(np.diag(matrix.to_numpy(dtype=float))))


def random_directions(hidden: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """k random orthonormal directions, via QR of a Gaussian matrix."""
    q, _ = np.linalg.qr(rng.standard_normal((hidden, k)))
    return q.T


def zero_shot_table(
    datasets: dict[str, Pooled], geom: persona_pcs.PersonaGeometry
) -> pd.DataFrame:
    """Methods 1 and 2: score by projection alone, no training."""
    rows = []
    for name, ds in datasets.items():
        rows.append(
            {
                "dataset": name,
                "n": len(ds.y),
                "n_deceptive": ds.n_deceptive,
                "auroc_axis": auroc(ds.y, geom.axis_score(ds.z)),
                "auroc_pc1": auroc(ds.y, geom.project(ds.z, k=1).ravel()),
            }
        )
    return pd.DataFrame(rows)


def run(pooled_dir: str | Path = POOLED_DIR, out_dir: str | Path = OUT_DIR) -> None:
    pooled_dir, out_dir = Path(pooled_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = persona_pcs.load_bank()
    geom = persona_pcs.fit_geometry(bank, k=3)
    logger.info(
        f"persona geometry: {len(geom.persona_names)} personas, "
        f"explained variance {np.round(geom.explained_variance_ratio, 4).tolist()}, "
        f"|cos(PC_k, axis)| {np.round(geom.alignment(), 4).tolist()}"
    )

    datasets = load_pooled(pooled_dir)
    logger.info(f"loaded {len(datasets)} pooled datasets from {pooled_dir}")

    zero_shot = zero_shot_table(datasets, geom)
    zero_shot.to_csv(out_dir / "zero_shot.csv", index=False)
    logger.info(f"zero-shot:\n{zero_shot.to_string(index=False)}")

    order = [n for n, d in sorted(datasets.items()) if d.two_class and n != CONTROL_DATASET]
    logger.info(f"transfer matrix over {len(order)} two-class datasets: {order}")

    hidden = datasets[order[0]].z.shape[1]
    rng = np.random.default_rng(SEED)

    variants: dict[str, object] = {
        "baseline": lambda z: z,
        "pc3": lambda z: geom.project(z, k=3),
        "axis1": lambda z: geom.axis_score(z)[:, None],
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

    summary = pd.DataFrame(
        [
            {
                "features": name,
                "dim": {"baseline": hidden, "pc3": 3, "axis1": 1, "random3": 3}.get(name, ""),
                "off_diagonal_mean": off_diagonal_mean(matrices[name]),
                "diagonal_mean": diagonal_mean(matrices[name]),
            }
            for name in ("baseline", "pc3", "random3", "axis1")
        ]
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    logger.info(f"\n{summary.round(4).to_string(index=False)}")
    logger.info(f"wrote matrices and summary to {out_dir}")



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
    fire.Fire({"run": run, "verify": verify})
