"""The knowledge direction's robustness checks: item weights and the projection shortcut."""

import numpy as np

from deception_detection.scripts import k_robustness as kr
from deception_detection.scripts import recall_denial as rd


def test_weights_are_a_difference_of_means_even_with_repeats():
    known = np.array([True] * 4 + [False] * 6)
    c = kr.weights(known, np.array([0, 0, 1, 3]), np.array([4, 5, 5, 6, 7, 9]))
    assert np.isclose(c[known].sum(), 1.0) and np.isclose(c[~known].sum(), -1.0)
    assert np.isclose(c[0], 0.5) and c[2] == 0 and np.isclose(c[5], -2 / 6)


def test_bootstrap_keeps_each_side_and_shuffles_keep_the_group_sizes():
    rng = np.random.default_rng(0)
    known = np.array([True] * 60 + [False] * 60)
    b = kr.bootstrap_weights(known, rng)
    assert (b[known] >= 0).all() and (b[~known] <= 0).all()
    s = kr.shuffled_weights(known, rng)
    assert (s > 0).sum() == 60 and (s < 0).sum() == 60 and np.isclose(s.sum(), 0.0)
    assert not np.array_equal(s > 0, known)


def test_projecting_onto_item_means_equals_scoring_with_the_rebuilt_direction():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 30))
    Z = rng.normal(size=(50, 30))
    known = np.array([True] * 6 + [False] * 6)
    c = kr.bootstrap_weights(known, rng)
    direct = Z @ rd._unit(X.T @ c)
    shortcut = (Z @ X.T) @ c / np.sqrt(c @ (X @ X.T) @ c)
    assert np.allclose(direct, shortcut)


def test_balanced_splits_cancel_the_known_unknown_contrast():
    rng = np.random.default_rng(2)
    known = np.array([True] * 60 + [False] * 60)
    c = kr.balanced_weights(known, rng)
    assert np.isclose(c[known].sum(), 0.0) and np.isclose(c[~known].sum(), 0.0)  # no class-mean component
    assert (c > 0).sum() == 60 and (c < 0).sum() == 60
    assert (c[known] > 0).sum() == 30 and (c[~known] > 0).sum() == 30


def test_standardising_centres_on_the_median_with_unit_interquartile_range():
    s = np.random.default_rng(3).normal(loc=5.0, scale=3.0, size=2001)
    z = kr.standardise(s)
    q1, med, q3 = np.quantile(z, [0.25, 0.5, 0.75])
    assert abs(med) < 1e-9 and np.isclose(q3 - q1, 1.0)
    assert np.allclose(kr.standardise(2 * s + 7), z)  # blind to a shift and a rescaling


def test_caught_at_matches_the_extremes():
    rng = np.random.default_rng(4)
    y = np.array([0] * 400 + [1] * 200)
    separated = np.concatenate([rng.normal(0, 1, 400), rng.normal(10, 1, 200)])
    tpr, fp = kr.caught_at(separated, y, 0.05, rng)
    assert tpr == 1.0 and abs(fp - 0.05) < 0.03
    same = rng.normal(0, 1, 600)
    tpr, fp = kr.caught_at(same, y, 0.05, rng)
    assert abs(tpr - 0.05) < 0.04  # no signal: lies are caught only as often as honest answers


def test_paired_draws_score_every_scorer_on_the_same_rows():
    rng = np.random.default_rng(5)
    y = np.array([0] * 300 + [1] * 100)
    s = rng.normal(size=400) + y
    draws = kr.paired_draws(y, {"a": s, "b": s.copy(), "c": -s}, 200, rng)
    assert np.array_equal(draws["a"], draws["b"])  # the same rows in every draw
    assert np.allclose(draws["a"] + draws["c"], 1.0)  # a reversed score mirrors its AUROC
    assert draws["a"].std() > 0  # and the rows do change between draws
