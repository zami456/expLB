"""Helpers of scripts/persona_attention_probe.py, on synthetic data."""

import numpy as np
import pytest

from deception_detection.scripts import persona_attention_probe as pap
from deception_detection.scripts.persona_ood import bootstrap_delta
from deception_detection.scripts.persona_positions import row_reads


def _coords(lengths, columns=76, seed=0):
    rng = np.random.default_rng(seed)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    labels = np.array(["HONEST", "DECEPTIVE"] * (len(lengths) // 2) + ["HONEST"] * (len(lengths) % 2))
    return pap.Coords("d", rng.normal(size=(offsets[-1], columns)).astype(np.float32), offsets, labels)


def test_fixed_reads_are_the_per_token_studys_reads():
    d = _coords([1, 7, 23, 10, 4])
    column = pap.READ_AXES["knowledge-denial"]
    mine = pap._fixed_reads(d, column)
    for i, (a, b) in enumerate(zip(d.offsets[:-1], d.offsets[1:], strict=True)):
        theirs = row_reads(d.coords[a:b, column], np.ones(b - a, bool))
        for read in pap.READS:
            assert mine[read][i] == pytest.approx(theirs[read], rel=1e-6)


def test_the_repe_validation_split_is_appended(tmp_path, monkeypatch):
    monkeypatch.setattr(pap, "_coords_dir", lambda layer: tmp_path)
    parts = {"repe_honesty__plain": _coords([2, 3], seed=1), "repe_honesty__plain_val": _coords([4], seed=2)}
    for name, d in parts.items():
        np.savez(tmp_path / f"{name}.npz", coords=d.coords, offsets=d.offsets, labels=d.labels,
                 columns=np.array(["c"] * 76), layer=40)
    data = pap.load_coords(40)
    assert list(data) == ["repe_honesty__plain"]
    merged = data["repe_honesty__plain"]
    assert merged.offsets.tolist() == [0, 2, 5, 9]
    assert merged.labels.tolist() == ["HONEST", "DECEPTIVE", "HONEST"]
    assert np.array_equal(merged.coords, np.concatenate([parts["repe_honesty__plain"].coords,
                                                         parts["repe_honesty__plain_val"].coords]))


def test_the_bootstrap_of_one_model_is_persona_oods():
    rng = np.random.default_rng(3)
    y = (rng.random(300) < 0.2).astype(int)
    a, b = y + rng.normal(0, 1, 300), y + rng.normal(0, 1.5, 300)
    mine = pap._bootstrap_mean_delta(y, [a], b, n_boot=200, seed=0)
    theirs = bootstrap_delta(y, a, b, n_boot=200, seed=0)
    assert mine == pytest.approx(theirs, abs=1e-9)


def test_the_attention_profile_is_a_distribution_over_tenths():
    lengths = np.array([3, 10, 25])
    attn = np.zeros((3, 25, 2))
    for i, n in enumerate(lengths):
        attn[i, :n] = 1.0 / n
    profile = pap._attention_profile(attn, lengths)
    assert profile[: pap.N_BINS].sum() == pytest.approx(1.0)
    assert profile[pap.N_BINS] == pytest.approx(1.0)  # uniform attention: entropy equals log(length)
