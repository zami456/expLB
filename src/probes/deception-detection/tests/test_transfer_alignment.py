"""The transfer-alignment measures: the covariance-weighted cosine is the score correlation, its within-class version,
the Gaussian prediction, and the own probe's folds.

Eq. 1 below is written from the paper (arXiv 2602.20273, Sec. 5). The authors' notebook has no license, so nothing is
copied from it; the covariances follow its `get_covariance` (ddof 1) and `compute_within_class_covariance`
(ddof 2).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from deception_detection.scripts import persona_ood as P
from deception_detection.scripts import transfer_alignment as T


def _eq1(w_a: np.ndarray, w_b: np.ndarray, sigma: np.ndarray) -> float:
    """The paper's Eq. 1 with an explicit covariance matrix."""
    return float((w_a @ sigma @ w_b) / np.sqrt((w_a @ sigma @ w_a) * (w_b @ sigma @ w_b)))


class TestCovarianceWeightedCosine:
    """The measure the paper regresses transfer on. With Σ the test set's pooled covariance it must be the correlation
    of the two probes' scores; that is what lets the analysis skip the 8192 x 8192 matrix."""

    def setup_method(self):
        rng = np.random.default_rng(0)
        self.z = rng.normal(size=(300, 20)) @ rng.normal(size=(20, 20)) + 5.0  # correlated, off-centre features
        self.w_a, self.w_b = rng.normal(size=20), rng.normal(size=20)

    def test_equals_eq1_with_the_explicit_covariance(self):
        sigma = np.cov(self.z, rowvar=False)  # pooled over every row, ddof 1
        assert np.isclose(T.mahalanobis_cosine(self.w_a, self.w_b, self.z), _eq1(self.w_a, self.w_b, sigma))

    def test_equals_the_score_correlation_whatever_the_offset_and_scale(self):
        expected = T.mahalanobis_cosine(self.w_a, self.w_b, self.z)
        a, b = self.z @ self.w_a, self.z @ self.w_b
        assert np.isclose(T.pearson(a, b), expected)
        assert np.isclose(T.pearson(3 * a - 7, 0.5 * b + 2), expected)
        assert np.isclose(T.pearson(-a, b), -expected)

    def test_is_not_the_plain_cosine(self):
        assert not np.isclose(T.cosine(self.w_a, self.w_b), T.mahalanobis_cosine(self.w_a, self.w_b, self.z))


class TestWithinClass:
    """The Gaussian prediction needs the correlation inside each class, where the label no longer drives both
    scores."""

    def test_equals_eq1_with_the_pooled_within_class_covariance(self):
        rng = np.random.default_rng(1)
        y = (rng.random(400) < 0.3).astype(int)
        z = rng.normal(size=(400, 15)) + np.outer(y, 2 * rng.normal(size=15))
        w_a, w_b = rng.normal(size=15), rng.normal(size=15)
        centred = z.copy()
        for label in (0, 1):
            centred[y == label] -= z[y == label].mean(0)
        sigma_within = centred.T @ centred / (len(z) - 2)
        assert np.isclose(T.within_class_corr(z @ w_a, z @ w_b, y), _eq1(w_a, w_b, sigma_within))

    def test_ignores_agreement_that_comes_only_through_the_label(self):
        rng = np.random.default_rng(2)
        y = np.repeat([0, 1], 500)
        a, b = y + rng.normal(scale=0.3, size=1000), y + rng.normal(scale=0.3, size=1000)
        assert T.pearson(a, b) > 0.5
        assert abs(T.within_class_corr(a, b, y)) < 0.1


class TestGaussianPrediction:
    """For Gaussian classes with a shared covariance, a direction's separation is its within-class correlation with
    the best (LDA) direction times the best one's, so its AUROC is Φ(ρ_within · d* / √2) with nothing fitted."""

    def setup_method(self):
        rng = np.random.default_rng(3)
        d, n = 40, 20_000
        root = rng.normal(size=(d, d)) / np.sqrt(d)
        self.sigma = root @ root.T + 0.5 * np.eye(d)
        self.delta = 0.3 * rng.normal(size=d)
        chol = np.linalg.cholesky(self.sigma)
        self.z = np.vstack([rng.normal(size=(n, d)) @ chol.T, rng.normal(size=(n, d)) @ chol.T + self.delta])
        self.y = np.repeat([0, 1], n)
        self.best = np.linalg.solve(self.sigma, self.delta)
        unit = self.best / np.linalg.norm(self.best)
        noise = rng.normal(size=(6, d))
        noise /= np.linalg.norm(noise, axis=1, keepdims=True)
        # from nearly the best direction to nearly orthogonal to it, and one pointing against it
        self.directions = [unit + c * e for c, e in zip((0.2, 0.7, 1.5, 3.0, 8.0), noise)] + [-unit + 0.5 * noise[5]]

    def test_separation_is_the_correlation_times_the_best_separation(self):
        d_best = np.sqrt(self.delta @ self.best)
        for w in self.directions:
            d_w = (w @ self.delta) / np.sqrt(w @ self.sigma @ w)
            assert np.isclose(d_w, _eq1(w, self.best, self.sigma) * d_best)

    def test_predicted_auroc_matches_the_empirical_one(self):
        own = self.z @ self.best
        d_own = T.separation(own, self.y)
        aurocs = []
        for w in self.directions:
            s = self.z @ w
            predicted = T.predicted_auroc(T.within_class_corr(s, own, self.y), d_own)
            aurocs.append(roc_auc_score(self.y, s))
            assert abs(predicted - aurocs[-1]) < 0.01
        assert max(aurocs) > 0.85 and min(aurocs) < 0.2  # the check covers strong, weak and inverted cells


class TestOwnProbe:
    """The own probe's held-out scores must be the ones behind the stored diagonal, so its folds have to be
    `P.diagonal_auroc`'s."""

    def setup_method(self):
        rng = np.random.default_rng(4)
        self.y = (rng.random(240) < 0.4).astype(int)
        self.z = rng.normal(size=(240, 12)) + np.outer(self.y, rng.normal(size=12))

    def test_pooled_held_out_auroc_is_the_diagonal_cell(self):
        own = T.own_probe(self.z, self.y)
        assert np.isclose(P.auroc(self.y, own["own_held"]), P.diagonal_auroc(self.z, self.y), atol=1e-12)

    def test_every_row_is_held_out_once_in_stratified_folds(self):
        fold = T.own_folds(self.y)
        assert (fold >= 0).all() and set(fold.tolist()) == set(range(5))
        shares = [self.y[fold == f].mean() for f in range(5)]
        assert max(shares) - min(shares) < 0.05

    def test_the_mean_direction_scores_every_row(self):
        own = T.own_probe(self.z, self.y)
        assert np.allclose(own["own_mean"], self.z @ own["own_dir"])


class TestDirections:
    """Plain cosines and the identity check need each probe as a direction in activation space."""

    def test_a_probe_scores_along_its_raw_direction_up_to_a_constant(self):
        rng = np.random.default_rng(5)
        y = (rng.random(200) < 0.5).astype(int)
        z = rng.normal(loc=3.0, scale=2.0, size=(200, 8)) + np.outer(y, rng.normal(size=8))
        probe = P.make_probe().fit(z, y)
        gap = probe.decision_function(z) - z @ T.raw_direction(probe)
        assert np.allclose(gap, gap[0])

    def test_a_probe_on_projected_features_maps_back_through_the_basis(self):
        rng = np.random.default_rng(6)
        basis = np.linalg.qr(rng.normal(size=(10, 3)))[0].T  # 3 orthonormal rows, as the extended PCs
        y = {name: (rng.random(150) < 0.5).astype(int) for name in ("A", "B")}
        z = {name: rng.normal(size=(150, 10)) + np.outer(y[name], rng.normal(size=10)) for name in ("A", "B")}
        scores, directions = T.trained(z, y, ["A"], ["A", "B"], basis)
        assert list(scores) == [("A", "B")]  # no own-set cell
        gap = scores[("A", "B")] - z["B"] @ directions["A"]
        assert np.allclose(gap, gap[0])


def test_bootstrap_draws_keep_the_class_sizes_and_rowwise_correlation_is_pearson():
    rng = np.random.default_rng(7)
    y = np.array([0] * 30 + [1] * 10)
    draws = T.stratified_draws(y, 50, rng)
    assert draws.shape == (50, 40) and (y[draws].sum(1) == 10).all()
    a, b = rng.normal(size=40), rng.normal(size=40)
    assert np.allclose(T.rowwise_pearson(a[draws], b[draws]), [T.pearson(a[d], b[d]) for d in draws])


SCORES_L40 = T.OUT / "scores_L40.npz"
MATRIX_L40 = T.results_dir(40) / "matrix_baseline.csv"


@pytest.mark.skipif(not (SCORES_L40.exists() and MATRIX_L40.exists()), reason="layer-40 scores not computed")
def test_layer_40_scores_reproduce_the_stored_raw_activation_matrix():
    saved = dict(np.load(SCORES_L40))
    matrix = pd.read_csv(MATRIX_L40, index_col=0)
    order = [str(n) for n in saved["order"]]
    for a in order:
        for b in order:
            s = saved[f"own_held|{b}"] if a == b else saved[f"D1|{a}|{b}"]
            assert abs(P.auroc(saved[f"labels|{b}"], s) - matrix.loc[a, b]) < T.TRAINED_TOL
