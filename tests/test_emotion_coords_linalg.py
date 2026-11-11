"""The emotion-coordinate study's linear-algebra helpers (emotion_coords/linalg.py)."""

import numpy as np
import pytest

from extensions.emotion_coords import linalg as la


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    basis = la.orthonormal(rng.standard_normal((6, 40)))
    return rng.standard_normal((25, 40)), basis


def test_orthonormal_rows(data):
    _, basis = data
    assert np.allclose(basis @ basis.T, np.eye(6), atol=1e-12)


def test_orthonormal_drops_dependent_rows():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((3, 10))
    basis = la.orthonormal(np.vstack([a, a[0] + 2 * a[1]]))
    assert basis.shape == (3, 10)


def test_keep_k_is_acts_times_basis(data):
    acts, basis = data
    assert np.allclose(la.keep_k(acts, basis, 4), acts @ basis[:4].T)
    assert la.keep_k(acts, basis).shape == (25, 6)


@pytest.mark.parametrize("k", [1, 3, 6])
def test_keep_plus_erase_reconstructs(data, k):
    acts, basis = data
    rebuilt = la.erase_k(acts, basis, k) + la.keep_k(acts, basis, k) @ basis[:k]
    assert np.allclose(rebuilt, acts, atol=1e-12)


def test_erased_has_no_component_left(data):
    acts, basis = data
    assert np.allclose(la.erase_k(acts, basis, 4) @ basis[:4].T, 0, atol=1e-12)


@pytest.mark.parametrize("k", [1, 6])
def test_projectors_idempotent_and_complementary(data, k):
    _, basis = data
    p = la.projector(basis, k)
    q = np.eye(p.shape[0]) - p
    assert np.allclose(p @ p, p, atol=1e-12)
    assert np.allclose(q @ q, q, atol=1e-12)
    assert np.allclose(p @ q, 0, atol=1e-12)
    assert np.allclose(p, p.T, atol=1e-12)


def test_random_subspace_seeded_and_orthonormal():
    a, b = la.random_subspace(50, 5, seed=3), la.random_subspace(50, 5, seed=3)
    assert np.array_equal(a, b)
    assert np.allclose(a @ a.T, np.eye(5), atol=1e-12)
    assert not np.allclose(a, la.random_subspace(50, 5, seed=4))


def test_variance_basis_reaches_share():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((200, 30)) * np.linspace(5, 0.1, 30)
    basis = la.variance_basis(x, 0.5)
    _, ratio = la.pca_basis(x)
    assert ratio[: len(basis)].sum() >= 0.5 > ratio[: len(basis) - 1].sum()


def test_randomized_top_pcs_match_exact_subspace():
    rng = np.random.default_rng(6)
    scale = np.r_[np.geomspace(20, 5, 10), np.geomspace(1, 0.1, 70)]  # a gap after the 10th direction
    x = rng.standard_normal((300, 80)) * scale
    exact, _ = la.pca_basis(x)
    approx = la.top_pcs(x, 10)
    # same subspace: the projector difference is tiny
    assert np.abs(la.projector(exact, 10) - la.projector(approx)).max() < 1e-6


def test_leace_removes_class_covariance():
    rng = np.random.default_rng(4)
    n, d, classes = 300, 60, 5
    labels = rng.integers(0, classes, n)
    means = rng.standard_normal((classes, d)) * 2
    x = means[labels] + rng.standard_normal((n, d))
    eraser = la.fit_leace(x, labels)
    erased = eraser(x)
    z = np.eye(classes)[labels]
    cross = (erased - erased.mean(0)).T @ (z - z.mean(0)) / n
    assert eraser.rank == classes - 1
    assert np.abs(cross).max() < 1e-8
    # the class means of the erased data coincide
    centroids = np.stack([erased[labels == c].mean(0) for c in range(classes)])
    assert np.abs(centroids - centroids.mean(0)).max() < 1e-8


def test_leace_with_fewer_samples_than_dimensions():
    rng = np.random.default_rng(5)
    labels = np.repeat(np.arange(4), 10)
    x = rng.standard_normal((40, 120)) + np.eye(4, 120)[labels] * 3
    eraser = la.fit_leace(x, labels)
    assert 0 < eraser.shrinkage <= 1
    erased = eraser(x)
    centroids = np.stack([erased[labels == c].mean(0) for c in range(4)])
    assert np.abs(centroids - centroids.mean(0)).max() < 1e-8
