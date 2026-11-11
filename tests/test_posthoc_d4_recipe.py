import numpy as np

from extensions.narcbench import posthoc_d4_recipe as PR
from extensions.narcbench import score as SC


def _data(n: int = 120, h: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.4).astype(int)
    X = rng.standard_normal((n, h)) + 0.8 * y[:, None] * np.eye(h)[0]
    return X, y


def test_identity_basis_is_their_probe():
    X, y = _data()
    m, s, w = PR.fit_in(X, y, np.eye(X.shape[1]))
    m0, s0, w0 = SC.fit_rose(X, y)
    assert np.allclose(m, m0) and np.allclose(s, s0) and np.allclose(w, w0)
    x = X[0]
    assert np.isclose(PR.project(x, np.eye(len(x)), (m, s, w)), ((x - m0) / s0) @ w0)


def test_subspace_score_ignores_the_complement():
    X, y = _data()
    rng = np.random.default_rng(1)
    B = np.linalg.qr(rng.standard_normal((X.shape[1], 3)))[0].T  # three orthonormal rows
    p = PR.fit_in(X, y, B)
    x, orth = X[3], rng.standard_normal(X.shape[1])
    orth -= B.T @ (B @ orth)
    assert np.isclose(PR.project(x, B, p), PR.project(x + orth, B, p))


def test_subspace_directions_are_seeded_unit_vectors():
    U = PR._sub_dirs(PR.RECIPE, 32, 50)
    assert U.shape == (50, 3) and np.allclose(np.linalg.norm(U, axis=1), 1)
    assert U is PR._sub_dirs(PR.RECIPE, 32, 50)
    assert not np.allclose(U, PR._sub_dirs(PR.SPAN40, 32, 50))
