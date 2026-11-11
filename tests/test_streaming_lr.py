"""The streaming fit must be sklearn's fit, as `LogisticRegressionDetector` runs it.

The reference is the detector's own code path: `StandardScaler().fit_transform`, then
`LogisticRegression(C=1/reg, fit_intercept=False, max_iter)` with the default lbfgs solver. The
comparison has power: an unstandardised fit, or a penalty off by a factor of n, misses by a wide
margin (checked below).
"""

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from deception_detection.streaming_lr import fit, standardiser


def data(n=3000, d=64, seed=0):
    rng = np.random.default_rng(seed)
    # uneven scales and offsets, as residual-stream features have
    X = (rng.normal(size=(n, d)) * rng.uniform(0.1, 20, d) + rng.normal(0, 5, d)).astype(np.float32)
    w = rng.normal(size=d)
    y = (rng.random(n) < 1 / (1 + np.exp(-((X - X.mean(0)) / X.std(0)) @ w / 4))).astype(int)
    return X, y


def reference(X, y, C, max_iter=1000):
    scaled = StandardScaler().fit_transform(X)
    model = LogisticRegression(C=C, fit_intercept=False, max_iter=max_iter).fit(scaled, y)
    return model.coef_.ravel()


@pytest.mark.parametrize("C", [0.1, 1.0])
def test_matches_sklearn_lbfgs(C):
    X, y = data()
    ours = fit(X, y, C=C, block=257)  # a block size that does not divide n
    theirs = reference(X, y, C)
    cosine = ours.coef @ theirs / np.linalg.norm(ours.coef) / np.linalg.norm(theirs)
    assert ours.converged
    assert cosine > 1 - 1e-6
    np.testing.assert_allclose(ours.coef, theirs, rtol=1e-3, atol=1e-4 * np.abs(theirs).max())


def test_standardiser_is_standard_scaler():
    X, _ = data()
    X[:, 3] = 7.0  # a constant feature gets scale 1, as in sklearn
    scaler = StandardScaler().fit(X)
    mean, scale = standardiser(X, block=500)
    np.testing.assert_allclose(mean, scaler.mean_, rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(scale, scaler.scale_, rtol=1e-8)


def test_the_comparison_has_power():
    X, y = data()
    theirs = reference(X, y, C=0.1)
    wrong_penalty = reference(X, y, C=0.1 * len(y))  # sklearn's scaling mistaken for a sum
    gap = np.abs(wrong_penalty - theirs).max() / np.abs(theirs).max()
    assert gap > 1e-2


def test_bfloat16_tensors_fit_as_their_float32_values():
    """Activations are cached in bfloat16; fitting the tensor must equal fitting a float32 copy."""
    X, y = data()
    cached = torch.from_numpy(X).to(torch.bfloat16)
    as_float = cached.float().numpy()
    a, b = fit(cached, y, C=0.1, block=300), fit(as_float, y, C=0.1, block=300)
    np.testing.assert_allclose(a.coef, b.coef, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(a.scale, b.scale, rtol=1e-12)
