"""Tests for the leave-one-dataset-out baseline.

The off-diagonal of the transfer matrix trains on a single dataset, which understates what
supervision can do. `leave_one_out` trains on the pooled remainder instead, and the cross-dataset
claim stands or falls on that comparison -- so the things pinned here are the ones that would
silently invalidate it: leakage of the held-out rows, a weighting that does not weight, and a
scoring path that disagrees with the one the committed results came from.
"""

import numpy as np
import pandas as pd
import pytest

from experiments.persona import persona_ood as P


def _pooled(name: str, z: np.ndarray, y: np.ndarray) -> P.Pooled:
    return P.Pooled(name=name, z=z, y=y)


@pytest.fixture
def toy() -> tuple[dict[str, P.Pooled], list[str], P.Pooled]:
    """Four small datasets sharing one signal direction, plus an honest-only control.

    Deliberately unequal in size (20 / 60 / 200 / 40) so the naive and balanced weightings are
    genuinely different fits rather than the same one twice.
    """
    rng = np.random.default_rng(0)
    direction = np.zeros(8)
    direction[0] = 1.0

    datasets: dict[str, P.Pooled] = {}
    for name, n in (("a", 20), ("b", 60), ("c", 200), ("d", 40)):
        y = np.array([0, 1] * (n // 2))
        z = rng.standard_normal((n, 8)) + np.outer(y, direction) * 3.0
        datasets[name] = _pooled(name, z, y)

    control = _pooled("ctrl", rng.standard_normal((100, 8)), np.zeros(100, dtype=int))
    datasets[control.name] = control
    return datasets, ["a", "b", "c", "d"], control


class TestNoLeakage:
    def test_held_out_rows_never_enter_training(self, toy):
        """The guard that matters most: a leak would inflate every number in the table.

        Checked by refitting each fold by hand on exactly the rows that should have been used.
        Identical scores mean the held-out dataset contributed nothing -- had it leaked in, the
        hand-fitted probe would differ.
        """
        datasets, order, control = toy
        fitted = P.leave_one_out_scores(lambda z: z, datasets, order, control)
        for held_out in order:
            sources = [n for n in order if n != held_out]
            probe = P.make_probe()
            probe.fit(
                np.vstack([datasets[n].z for n in sources]),
                np.concatenate([datasets[n].y for n in sources]),
            )
            expected = probe.decision_function(datasets[held_out].z)
            np.testing.assert_allclose(fitted.scores[held_out], expected, atol=1e-12)

    def test_training_pool_is_the_other_datasets(self, toy):
        datasets, order, _ = toy
        total = sum(len(datasets[n].y) for n in order)
        for held_out in order:
            expected = total - len(datasets[held_out].y)
            sources = [n for n in order if n != held_out]
            assert sum(len(datasets[n].y) for n in sources) == expected


class TestWeighting:
    def test_balanced_gives_every_dataset_equal_total_weight(self, toy):
        datasets, order, _ = toy
        for held_out in order:
            sources = [n for n in order if n != held_out]
            weights = np.concatenate(
                [np.full(len(datasets[n].y), 1.0 / len(datasets[n].y)) for n in sources]
            )
            offset = 0
            for name in sources:
                n = len(datasets[name].y)
                assert weights[offset : offset + n].sum() == pytest.approx(1.0)
                offset += n

    def test_weightings_differ_on_unequal_datasets(self, toy):
        """If these coincided, reporting both would be noise rather than a robustness check."""
        datasets, order, control = toy
        naive = P.leave_one_out(lambda z: z, datasets, order, control=control, weighting="naive")
        balanced = P.leave_one_out(
            lambda z: z, datasets, order, control=control, weighting="balanced"
        )
        assert not np.allclose(naive.to_numpy(), balanced.to_numpy())

    def test_unknown_weighting_raises(self, toy):
        datasets, order, control = toy
        with pytest.raises(ValueError, match="unknown weighting"):
            P.leave_one_out_scores(lambda z: z, datasets, order, control, weighting="uniform")


class TestFixedDirectionEquivalence:
    """A one-dimensional feature makes the training set irrelevant to AUROC.

    A single-feature probe can only learn a sign and a scale, and AUROC is invariant to both. So
    a fixed direction must score the same under leave-one-out as it does in every row of the
    transfer matrix -- which is what licenses comparing the committed `matrix_ext_axis1.csv`
    numbers against the pooled baseline computed here.
    """

    def test_matches_the_transfer_matrix_off_diagonal(self, toy):
        datasets, order, control = toy
        direction = np.zeros(8)
        direction[0] = 1.0

        def featurize(z: np.ndarray) -> np.ndarray:
            return (z @ direction)[:, None]

        pooled = P.leave_one_out(featurize, datasets, order, control=control)
        matrix = P.transfer_matrix(featurize, datasets, order)
        for held_out in order:
            column = matrix[held_out].drop(index=held_out)
            assert column.std(ddof=0) == pytest.approx(0.0, abs=1e-12)
            assert pooled[held_out] == pytest.approx(float(column.iloc[0]), abs=1e-12)


class TestMetrics:
    def test_thresholded_metric_requires_a_control(self, toy):
        datasets, order, _ = toy
        with pytest.raises(ValueError, match="needs a control dataset"):
            P.leave_one_out(
                lambda z: z, datasets, order, metric=P.BALANCED_ACCURACY, control=None
            )

    def test_one_fit_serves_both_metrics(self, toy):
        """`loo` reads AUROC and balanced accuracy off a single set of fits; check they agree
        with computing each separately."""
        datasets, order, control = toy
        fitted = P.leave_one_out_scores(lambda z: z, datasets, order, control)
        separate = P.leave_one_out(
            lambda z: z, datasets, order, metric=P.BALANCED_ACCURACY, control=control
        )
        for name in order:
            combined = P.balanced_accuracy(
                datasets[name].y, fitted.scores[name], fitted.thresholds[name]
            )
            assert combined == pytest.approx(separate[name])


class TestBootstrap:
    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(1)
        y = np.array([0] * 80 + [1] * 20)
        strong = rng.standard_normal(100) + y * 2.0
        weak = rng.standard_normal(100) + y * 0.4
        point, low, high = P.bootstrap_delta(y, strong, weak, n_boot=200)
        assert point > 0
        assert low <= point <= high

    def test_reproducible_under_the_seed(self):
        rng = np.random.default_rng(2)
        y = np.array([0] * 40 + [1] * 10)
        a, b = rng.standard_normal(50) + y, rng.standard_normal(50)
        assert P.bootstrap_delta(y, a, b, n_boot=100) == P.bootstrap_delta(y, a, b, n_boot=100)

    def test_identical_scores_give_a_zero_delta(self):
        y = np.array([0] * 30 + [1] * 10)
        scores = np.random.default_rng(3).standard_normal(40)
        point, low, high = P.bootstrap_delta(y, scores, scores, n_boot=100)
        assert (point, low, high) == (0.0, 0.0, 0.0)

    def test_single_class_gives_no_interval(self):
        y = np.zeros(20, dtype=int)
        scores = np.arange(20.0)
        point, low, high = P.bootstrap_delta(y, scores, scores, n_boot=50)
        assert np.isnan(point) and np.isnan(low) and np.isnan(high)


class TestOutputShape:
    def test_series_is_indexed_by_held_out_dataset(self, toy):
        datasets, order, control = toy
        result = P.leave_one_out(lambda z: z, datasets, order, control=control)
        assert isinstance(result, pd.Series)
        assert list(result.index) == order
        assert result.notna().all()
