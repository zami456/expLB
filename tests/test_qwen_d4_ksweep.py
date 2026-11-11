import numpy as np

from extensions.narcbench import qwen_d4_ksweep as QK
from extensions.narcbench import qwen_d4_pipeline as QP


def test_random_subspace_is_orthonormal_and_matches_the_3_dim_null():
    for k in (1, 3, 8):
        B = QK.random_subspace(5, 26, k, hidden=64)
        assert B.shape == (k, 64) and np.allclose(B @ B.T, np.eye(k), atol=1e-10)
    assert np.array_equal(QK.random_subspace(7, 28, 3, hidden=64), QP.random_subspace(7, 28, hidden=64))


def test_stat_is_mean_and_standard_error_over_seeds_and_layers():
    raws = [{"g": {m: [0.5, 0.7] for m in QP.METHODS}}, {"g": {m: [0.6, float("nan")] for m in QP.METHODS}}]
    got = QK.stat(raws, ["g"])["g"]["peak_suspicion"]
    arr = np.array([0.5, 0.7, 0.6])
    assert np.isclose(got[0], arr.mean()) and np.isclose(got[1], arr.std() / np.sqrt(3))


def test_band_p_counts_ties_as_at_or_above():
    b = QK._band(0.6, [0.5, 0.6, 0.7, 0.4])
    assert b["p"] == (1 + 2) / 5 and b["median"] == 0.55
