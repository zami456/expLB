"""Does an INLP-derived subspace improve OOD probe generalization?

persona_ood.py projects the pooled layer-20 activations onto 3 persona PCs before probing, and
cross-dataset transfer improves. That basis comes from persona role-play. This asks whether it can
be learned from the data instead, with INLP (deception_detection/inlp.py, a port of Ravfogel et
al.'s reference implementation) run on RepE honesty alone:

    W_1..W_T   the classifiers INLP finds on RepE, each fitted with all earlier ones projected out
    B_k        orthonormal basis of span(W_1..W_k), nested in k
    features   standardise(z) @ B_k^T   -- coordinates under the rowspace projection B_k^T B_k

then the same train/test AUROC matrix as persona_ood.py, against:

    baseline         raw 8192-d pooled activations
    random_k         k random orthonormal directions in the same standardised space (5 draws,
                     averaged cell by cell) -- is it just rank?
    pca_k            top-k PCs of the standardised RepE train split, no labels -- is it just
                     "directions the RepE data varies along"? Random directions miss the data span
                     entirely, which makes random alone a weak control.
    persona_pc1/pc3  the persona arms, recomputed; they must match results/persona_ood exactly

INLP runs in standardised space: a StandardScaler is fitted once on RepE's train split and INLP
then runs unmodified, so classifier 1 is exactly the repo's probe (StandardScaler -> LR, C=0.1, no
intercept) and the reference loop needs no change. The same scaler maps every dataset.

Leakage. The basis saw RepE's labels (train split to fit, "_val" split to score), so RepE is not a
clean test set here. Headline means drop the RepE test column and the RepE diagonal cell, for every
arm alike; RepE's training row stays, since training on RepE is the deployable pipeline. The
unrestricted 6x6 means are written too, for continuity with the persona summary.

Layer. Pooled npz files carry no layer tag. The persona arms only mean anything against layer 20;
pass with_persona=False for any other pool.

    python deception_detection/scripts/inlp_ood.py run
"""

from collections.abc import Callable
from pathlib import Path

import fire
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from deception_detection import inlp, persona_pcs
from deception_detection.log import logger
from deception_detection.scripts import persona_ood
from deception_detection.scripts.persona_ood import (
    CONTROL_DATASET,
    HPKR,
    MAX_ITER,
    MERGE_INTO,
    N_RANDOM_DRAWS,
    POOLED_DIR,
    REG_COEFF,
    SEED,
    Pooled,
)
from deception_detection.utils import REPO_ROOT

OUT_DIR = REPO_ROOT / "results" / "inlp_ood"

# INLP fits on the split the deployed probe was trained on and scores on its "_val" remainder
SOURCE = "repe_honesty__plain"
SOURCE_DEV = "repe_honesty__plain_val"

# the reference's own __main__ example keeps every classifier (min_accuracy 0) for a fixed number
# of iterations; doing the same keeps the k sweep nested
NUM_CLASSIFIERS = 100
MIN_ACCURACY = 0.0
KS = (1, 2, 3, 5, 10, 20, 50, 100)

# T* = the leading run of iterations whose dev accuracy beats this. 102 dev rows put one standard
# error at chance near 0.05, so 0.55 is about one SE above 0.5.
CHANCE_ACCURACY = 0.55

# the benchmark's headline metric: threshold at 1% FPR on the honest-only control
FPR = 0.01


def load_with_split(pooled_dir: Path) -> tuple[dict[str, Pooled], Pooled, Pooled]:
    """The merged datasets for the matrix, plus RepE's train and dev halves for INLP.

    Loads once, unmerged, then merges in load_pooled's own order (the split file sorts after the
    main one), so the merged RepE rows -- and hence the CV folds -- match the persona run's.
    """
    unmerged = persona_ood.load_pooled(pooled_dir, merge=False)
    datasets = {name: ds for name, ds in unmerged.items() if name not in MERGE_INTO}
    for part, whole in MERGE_INTO.items():
        datasets[whole] = Pooled(
            name=whole,
            z=np.concatenate([unmerged[whole].z, unmerged[part].z]),
            y=np.concatenate([unmerged[whole].y, unmerged[part].y]),
        )
    return datasets, unmerged[SOURCE], unmerged[SOURCE_DEV]


def affine_featurizer(
    scaler: StandardScaler, basis: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """z -> standardise(z) @ basis.T, without materialising the standardised activations.

    Folding the scaler into the basis keeps memory at one copy of the pooled data; the result is
    identical to transforming first.
    """
    readout = basis / scaler.scale_  # [k, hidden]
    offset = (scaler.mean_ / scaler.scale_) @ basis.T  # [k]
    return lambda z: z @ readout.T - offset


def raw_readout(scaler: StandardScaler, basis: np.ndarray) -> np.ndarray:
    """Orthonormal rows spanning the raw-activation directions the standardised basis reads.

    A feature b . (z - mu) / sigma is the raw functional (b / sigma) . z plus a constant, so this is
    the subspace to compare against raw-space bases such as the persona PCs.
    """
    q, _ = np.linalg.qr((basis / scaler.scale_).T)
    return q.T


def t_star(trace: pd.DataFrame, threshold: float = CHANCE_ACCURACY) -> int:
    """Number of leading iterations whose dev accuracy stays above chance."""
    below = np.flatnonzero(trace["dev_accuracy"].to_numpy() <= threshold)
    return int(below[0]) if len(below) else len(trace)


def cell_mean(
    matrix: pd.DataFrame, drop_train: tuple[str, ...] = (), drop_test: tuple[str, ...] = ()
) -> float:
    """Mean of the off-diagonal cells left after dropping train rows and test columns.

    The diagonal is blanked by name before dropping, so a non-square remainder still excludes it.
    """
    values = matrix.to_numpy(dtype=float).copy()
    np.fill_diagonal(values, np.nan)
    cells = pd.DataFrame(values, index=matrix.index, columns=matrix.columns)
    cells = cells.drop(index=list(drop_train), columns=list(drop_test), errors="ignore")
    return float(np.nanmean(cells.to_numpy()))


def summarise(matrix: pd.DataFrame, source: str) -> dict[str, float]:
    diagonal = pd.Series(np.diag(matrix.to_numpy(dtype=float)), index=matrix.index)
    return {
        # headline: the source never appears as a test set
        "off_diagonal": cell_mean(matrix, drop_test=(source,)),
        "off_diagonal_excl_hpkr": cell_mean(
            matrix, drop_train=(HPKR,), drop_test=(source, HPKR)
        ),
        "diagonal": float(diagonal.drop(source).mean()),
        "train_set_sensitivity": persona_ood.train_set_sensitivity(
            matrix, exclude_test=(source,)
        ),
        # the persona summary's definitions, unrestricted
        "off_diagonal_all": persona_ood.off_diagonal_mean(matrix),
        "off_diagonal_all_excl_hpkr": cell_mean(matrix, drop_train=(HPKR,), drop_test=(HPKR,)),
        "diagonal_all": persona_ood.diagonal_mean(matrix),
        "train_set_sensitivity_all": persona_ood.train_set_sensitivity(matrix),
    }


def benchmark(
    featurize: Callable[[np.ndarray], np.ndarray],
    datasets: dict[str, Pooled],
    source: str,
    order: list[str],
) -> list[dict[str, float | str]]:
    """Balanced accuracy and recall at 1% FPR on the control, for the source-trained probe.

    Mirrors prompt_scorer.prompt_score_threshold (quantile of the control scores) and
    metrics.get_balanced_accuracy (balanced_accuracy_score on scores > threshold). Scoring pooled
    activations equals averaging per-token scores, because every probe here is affine in z.
    """
    probe = persona_ood.make_probe()
    probe.fit(featurize(datasets[source].z), datasets[source].y)  # type: ignore[attr-defined]
    control = featurize(datasets[CONTROL_DATASET].z)
    threshold = float(np.nanquantile(probe.decision_function(control), 1 - FPR))  # type: ignore[attr-defined]

    rows = []
    for name in order:
        if name == source:
            continue
        ds = datasets[name]
        scores = probe.decision_function(featurize(ds.z))  # type: ignore[attr-defined]
        flagged = scores > threshold
        rows.append(
            {
                "dataset": name,
                "balanced_accuracy": float(balanced_accuracy_score(ds.y, flagged)),
                "recall": float(flagged[ds.y == 1].mean()),
                "auroc": persona_ood.auroc(ds.y, scores),
            }
        )
    return rows


def direction_table(
    datasets: dict[str, Pooled], order: list[str], scaler: StandardScaler, basis: np.ndarray
) -> pd.DataFrame:
    """Zero-shot AUROC of each INLP direction alone, on every dataset.

    Each row of the basis points along its classifier, so above 0.5 means the direction carries
    that dataset's deception signal with RepE's polarity, and below 0.5 means inverted. No probe
    is trained: this is what each direction contributes before any dataset gets to weight them.
    """
    rows = []
    for i, direction in enumerate(basis, start=1):
        featurize = affine_featurizer(scaler, direction[None])
        rows.append(
            {"direction": i}
            | {name: persona_ood.auroc(datasets[name].y, featurize(datasets[name].z).ravel())
               for name in order}
        )
    return pd.DataFrame(rows)


def directions(
    pooled_dir: str | Path = POOLED_DIR, out_dir: str | Path = OUT_DIR, n: int = 20
) -> None:
    """Rewrite direction_auroc.csv from a saved basis, without rerunning INLP or the matrices."""
    out_dir = Path(out_dir)
    saved = np.load(out_dir / "inlp_basis.npz")
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = saved["scaler_mean"], saved["scaler_scale"]
    datasets, _, _ = load_with_split(Path(pooled_dir))
    order = [n for n, d in sorted(datasets.items()) if d.two_class and n != CONTROL_DATASET]
    table = direction_table(datasets, order, scaler, saved["basis"][:n])
    table.round(4).to_csv(out_dir / "direction_auroc.csv", index=False)
    logger.info(f"zero-shot AUROC per INLP direction:\n{table.round(3).to_string(index=False)}")


def run(
    pooled_dir: str | Path = POOLED_DIR,
    out_dir: str | Path = OUT_DIR,
    num_classifiers: int = NUM_CLASSIFIERS,
    with_persona: bool = True,
) -> None:
    pooled_dir, out_dir = Path(pooled_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets, train, dev = load_with_split(pooled_dir)
    order = [n for n, d in sorted(datasets.items()) if d.two_class and n != CONTROL_DATASET]
    logger.info(f"transfer matrix over {len(order)} two-class datasets: {order}")
    logger.info(
        f"INLP on {SOURCE}: {len(train.y)} train rows ({train.n_deceptive} deceptive), "
        f"{len(dev.y)} dev rows ({dev.n_deceptive} deceptive)"
    )
    hidden = train.z.shape[1]

    # --- INLP, in the space the probe sees ------------------------------------------------
    scaler = StandardScaler().fit(train.z)
    result = inlp.get_debiasing_projection(
        LogisticRegression,
        {"C": 1 / REG_COEFF, "fit_intercept": False, "max_iter": MAX_ITER},
        num_classifiers,
        hidden,
        MIN_ACCURACY,
        scaler.transform(train.z),
        train.y,
        scaler.transform(dev.z),
        dev.y,
    )
    kept = len(result.Ws)
    basis = result.basis()
    t = t_star(result.trace)
    logger.info(f"INLP kept {kept} classifiers; dev accuracy stays above {CHANCE_ACCURACY} for {t}")

    trace = result.trace.copy()
    if with_persona:
        bank = persona_pcs.load_bank()
        geom = persona_pcs.fit_geometry(bank, k=3)
        # each direction's raw-space readout against the persona deception axis
        readouts = basis / scaler.scale_
        readouts /= np.linalg.norm(readouts, axis=1, keepdims=True)
        trace.loc[trace["kept"], "cos_persona_axis"] = np.abs(readouts @ geom.axis)
        if kept:
            overlap = inlp.principal_cosines(raw_readout(scaler, basis[:3]), geom.components)
            logger.info(f"principal cosines, INLP-3 vs persona PC3: {np.round(overlap, 4)}")
    trace.to_csv(out_dir / "inlp_trace.csv", index=False)
    np.savez(
        out_dir / "inlp_basis.npz",
        Ws=result.Ws,
        basis=basis,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        source=SOURCE,
        source_dev=SOURCE_DEV,
        num_classifiers=num_classifiers,
        reg_coeff=REG_COEFF,
        t_star=t,
    )

    table = direction_table(datasets, order, scaler, basis[:20])
    table.round(4).to_csv(out_dir / "direction_auroc.csv", index=False)
    logger.info(f"zero-shot AUROC per INLP direction:\n{table.round(3).to_string(index=False)}")

    ks = sorted({k for k in KS if k <= kept} | ({t} if t >= 1 else set()))
    logger.info(f"k sweep: {ks}")

    # --- arms ----------------------------------------------------------------------------
    variants: dict[str, Callable[[np.ndarray], np.ndarray]] = {"baseline": lambda z: z}
    dims = {"baseline": hidden}
    # full SVD: at this shape "auto" picks the randomised solver, which is not reproducible
    pca = PCA(n_components=max(ks), svd_solver="full").fit(scaler.transform(train.z))
    for k in ks:
        variants[f"inlp_k{k}"] = affine_featurizer(scaler, basis[:k])
        variants[f"pca_k{k}"] = affine_featurizer(scaler, pca.components_[:k])
        dims[f"inlp_k{k}"] = dims[f"pca_k{k}"] = dims[f"random_k{k}"] = k
    if with_persona:
        variants["persona_pc1"] = lambda z: geom.project(z, k=1)
        variants["persona_pc3"] = lambda z: geom.project(z, k=3)
        dims["persona_pc1"], dims["persona_pc3"] = 1, 3

    matrices = {}
    for name, featurize in variants.items():
        matrices[name] = persona_ood.transfer_matrix(featurize, datasets, order)
        logger.info(f"{name}: done")

    # random control, averaged over draws so a single lucky basis cannot carry the comparison
    rng = np.random.default_rng(SEED)
    for k in ks:
        draws = []
        for _ in range(N_RANDOM_DRAWS):
            directions = persona_ood.random_directions(hidden, k, rng)
            draws.append(
                persona_ood.transfer_matrix(affine_featurizer(scaler, directions), datasets, order)
            )
        matrices[f"random_k{k}"] = sum(draws) / len(draws)
        logger.info(f"random_k{k}: done")

    for name, matrix in matrices.items():
        matrix.round(4).to_csv(out_dir / f"matrix_{name}.csv")

    # --- summary -------------------------------------------------------------------------
    summary = pd.DataFrame(
        [
            {
                "features": name,
                "arm": name.rpartition("_k")[0] or name,
                "dim": dims[name],
                **summarise(matrix, SOURCE),
            }
            for name, matrix in matrices.items()
        ]
    )
    summary.to_csv(out_dir / "summary.csv", index=False)
    logger.info(f"\n{summary.round(4).to_string(index=False)}")

    # --- benchmark metric, source-trained row --------------------------------------------
    # random is left out: it is a rank control for transfer, and a threshold averaged over
    # draws describes no probe anyone would deploy
    bench = [
        {"features": name, **row}
        for name, featurize in variants.items()
        for row in benchmark(featurize, datasets, SOURCE, order)
    ]
    bench_df = pd.DataFrame(bench)
    bench_df.to_csv(out_dir / "benchmark.csv", index=False)
    metrics = ["balanced_accuracy", "recall", "auroc"]
    means = bench_df.groupby("features", sort=False)[metrics].mean()
    logger.info(f"benchmark, mean over the five Liars' Bench sets:\n{means.round(4).to_string()}")
    logger.info(f"wrote INLP basis, trace, matrices, summary and benchmark to {out_dir}")


if __name__ == "__main__":
    fire.Fire({"run": run, "directions": directions})
