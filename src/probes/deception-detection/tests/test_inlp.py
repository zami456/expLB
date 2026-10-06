"""Tests for the INLP port (deception_detection/inlp.py) against the verbatim reference.

The reference (tests/inlp_reference.py) forms d x d matrices, so the comparison runs at small d;
the port's only change is to avoid those matrices, and these tests are what vouch for it at 8192.
"""

import inlp_reference as reference
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from deception_detection import inlp
from deception_detection.scripts.persona_ood import train_set_sensitivity
from deception_detection.utils import REPO_ROOT

D = 40
# the probe's own LR (persona_ood.make_probe), with a tight tolerance so both implementations
# reach the same optimum rather than wherever lbfgs happened to stop
PARAMS = {"C": 0.1, "fit_intercept": False, "max_iter": 10_000, "tol": 1e-10}


def planted(
    strength: float, n: int = 1000, d: int = D, seed: int = 0
) -> tuple[np.ndarray, ...]:
    """Isotropic noise with the label planted along one unit direction; 80/20 train/dev.

    Returns X_train, Y_train, X_dev, Y_dev and the planted direction.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    direction = rng.standard_normal(d)
    direction /= np.linalg.norm(direction)
    x = rng.standard_normal((n, d)) + np.outer(2 * y - 1, strength * direction)
    cut = int(0.8 * n)
    return x[:cut], y[:cut], x[cut:], y[cut:], direction


def run_port(num_classifiers: int, min_accuracy: float, data: tuple) -> inlp.INLPResult:
    x_train, y_train, x_dev, y_dev = data[:4]
    return inlp.get_debiasing_projection(
        LogisticRegression, PARAMS, num_classifiers, D, min_accuracy,
        x_train, y_train, x_dev, y_dev,
    )


class TestMatchesReference:
    @pytest.fixture(scope="class")
    def both(self):
        data = planted(1.5)
        ours = run_port(8, 0.0, data)
        np.random.seed(0)  # the reference draws a (no-op) dropout mask from the global RNG
        P, _, Ws = reference.get_debiasing_projection(
            LogisticRegression, PARAMS, 8, D, True, 0.0, *data[:4]
        )
        return ours, P, np.vstack(Ws)

    def test_same_classifiers(self, both):
        ours, _, ref_ws = both
        # The reference trains on X * 1/(1 + 1e-6) even with dropout off -- its dropout mask is
        # all ones times that scale -- so the two optima differ at about 1e-6, not at round-off.
        assert ours.Ws.shape == ref_ws.shape
        cos = np.sum(ours.Ws * ref_ws, axis=1) / (
            np.linalg.norm(ours.Ws, axis=1) * np.linalg.norm(ref_ws, axis=1)
        )
        assert cos.min() > 1 - 1e-8
        # relative to each row's norm: elementwise rtol is meaningless on near-zero coefficients
        deviation = np.abs(ours.Ws - ref_ws).max(axis=1) / np.linalg.norm(ref_ws, axis=1)
        assert deviation.max() < 1e-5

    def test_same_projection(self, both):
        ours, P, _ = both
        basis = inlp.rowspace_basis(ours.Ws)
        assert np.allclose(np.eye(D) - basis @ basis.T, P, atol=1e-6)

    def test_rowspace_basis_is_the_factor_of_the_reference_projection(self):
        W = np.random.default_rng(1).standard_normal((3, D))
        basis = inlp.rowspace_basis(W)
        assert np.allclose(basis @ basis.T, reference.get_rowspace_projection(W), atol=1e-12)

    def test_ben_israel_intersection_matches(self):
        W = np.random.default_rng(2).standard_normal((4, D))
        rowspaces = [reference.get_rowspace_projection(w[None]) for w in W]
        P = reference.get_projection_to_intersection_of_nullspaces(rowspaces, D)
        x = np.random.default_rng(3).standard_normal((5, D))
        assert np.allclose(inlp.project_to_nullspace(x, inlp.rowspace_basis(W)), x @ P, atol=1e-12)


class TestReferenceInvariants:
    """The checks the reference runs in its own `__main__`, applied to the port."""

    @pytest.fixture(scope="class")
    def result(self) -> inlp.INLPResult:
        return run_port(8, 0.0, planted(1.5))

    def test_projection_kills_every_classifier(self, result):
        x = np.random.default_rng(4).standard_normal((10, D))
        assert np.abs(result.nullspace(x) @ result.Ws.T).max() < 1e-8

    def test_projection_is_idempotent(self, result):
        x = np.random.default_rng(5).standard_normal((10, D))
        once = result.nullspace(x)
        assert np.allclose(result.nullspace(once), once, atol=1e-12)

    def test_classifiers_are_mutually_orthogonal(self, result):
        unit = result.Ws / np.linalg.norm(result.Ws, axis=1, keepdims=True)
        gram = unit @ unit.T
        assert np.abs(gram - np.eye(len(unit))).max() < 1e-6


class TestBasis:
    @pytest.fixture(scope="class")
    def result(self) -> inlp.INLPResult:
        return run_port(6, 0.0, planted(1.5))

    def test_orthonormal_and_nested(self, result):
        full = result.basis()
        assert np.allclose(full @ full.T, np.eye(len(full)), atol=1e-10)
        assert np.allclose(result.basis(3), full[:3], atol=1e-10)

    def test_spans_the_first_k_classifiers(self, result):
        b = result.basis(3)
        residual = result.Ws[:3] - (result.Ws[:3] @ b.T) @ b
        assert np.abs(residual).max() < 1e-8

    def test_each_row_points_along_its_classifier(self, result):
        assert np.all(np.sum(result.basis() * result.Ws, axis=1) > 0)

    def test_ignores_weight_scale(self):
        """Regression: norms fall by orders of magnitude late in a run, which is not dependence."""
        ws = np.eye(3, D) * np.array([[1.0], [1e-6], [1e-12]])
        result = inlp.INLPResult(Ws=ws, trace=pd.DataFrame())
        assert np.allclose(result.basis(), np.eye(3, D))

    def test_project_gives_coordinates(self, result):
        x = np.random.default_rng(6).standard_normal((7, D))
        assert result.project(x, k=2).shape == (7, 2)


class TestFindsTheSignal:
    def test_recovers_a_planted_direction(self):
        data = planted(2.0)
        result = run_port(1, 0.0, data)
        assert float(result.basis(1)[0] @ data[4]) > 0.9

    def test_signal_is_exhausted_once_its_span_is_removed(self):
        """One planted direction takes INLP two classifiers, not one.

        The first classifier is an estimate (cos 0.993 here), so the part of the planted direction
        it misses is still decodable and the second classifier picks it up (dev 0.56). The count of
        useful iterations therefore overstates the number of directions carrying the label, by more
        the noisier each estimate is -- worth remembering at n = 448, d = 8192.
        """
        data = planted(1.5, n=10_000)
        result = run_port(6, 0.0, data)
        assert float(np.linalg.norm(result.basis(2) @ data[4])) > 0.99
        dev = result.trace["dev_accuracy"].to_numpy()
        assert dev[0] > 0.9
        assert dev[2:].max() < 0.53

    def test_stops_at_the_zero_classifier_fixed_point(self):
        """Regression: on RepE the solver returned w = 0 from iteration 83 and the basis broke.

        Twins that differ only along the first axis make the fixed point exact: once that axis
        is removed the two classes coincide, the gradient at w = 0 vanishes, and every later fit
        would return the same zero classifier.
        """

        def twins(n_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
            rest = np.random.default_rng(seed).standard_normal((n_pairs, D - 1))
            honest = np.column_stack([-np.ones(n_pairs), rest])
            deceptive = np.column_stack([np.ones(n_pairs), rest])
            return np.concatenate([honest, deceptive]), np.repeat([0, 1], n_pairs)

        result = run_port(5, 0.0, (*twins(160, 8), *twins(40, 9)))
        assert len(result.Ws) == 1
        assert len(result.trace) == 2
        assert result.trace["weight_norm"].iloc[-1] == 0
        assert not result.trace["kept"].iloc[-1]
        assert abs(result.basis(1)[0, 0]) > 0.99

    def test_noise_labels_give_an_empty_basis(self):
        x_train, _, x_dev, _, _ = planted(0.0)
        rng = np.random.default_rng(7)
        noise = (x_train, rng.integers(0, 2, len(x_train)), x_dev, rng.integers(0, 2, len(x_dev)))
        result = run_port(3, 0.6, noise)
        assert result.Ws.shape == (0, D)
        assert not result.trace["kept"].any()
        assert result.basis().shape == (0, D)


class TestTrainSetSensitivity:
    """Pins the persona summary's figures, which were computed ad hoc before the metric had code."""

    @pytest.mark.parametrize(("name", "expected"), [("baseline", 0.1674), ("pc3", 0.0274)])
    def test_reproduces_the_persona_summary(self, name: str, expected: float):
        path = REPO_ROOT / "results" / "persona_ood" / f"matrix_{name}.csv"
        if not path.exists():
            pytest.skip(f"{path} not present")
        matrix = pd.read_csv(path, index_col=0)
        assert round(train_set_sensitivity(matrix), 4) == expected

    def test_one_dimensional_features_score_zero(self):
        # every row identical: a probe that cannot depend on its training set
        matrix = pd.DataFrame(
            [[0.9, 0.8, 0.7], [0.9, 0.8, 0.7], [0.9, 0.8, 0.7]],
            index=list("abc"), columns=list("abc"),
        )
        assert train_set_sensitivity(matrix, exclude_train=()) == 0.0
