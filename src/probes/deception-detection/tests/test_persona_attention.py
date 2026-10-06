"""The persona-anchored attention probe's defining properties, on synthetic tokens."""

import json

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from deception_detection.scripts import persona_attention as pa


def _ragged(lengths):
    example = np.repeat(np.arange(len(lengths)), lengths)
    return torch.from_numpy(example), len(lengths)


def test_softmax_sums_to_one_within_each_example_and_ignores_shared_shifts():
    ex, n = _ragged([3, 1, 4])
    logits = torch.randn(8, dtype=torch.float64)
    a = pa.segment_softmax(logits, ex, n)
    sums = torch.zeros(n, dtype=torch.float64).index_add(0, ex, a)
    assert torch.allclose(sums, torch.ones(n, dtype=torch.float64))
    shifted = logits + torch.tensor([5.0, 5.0, 5.0, -2.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64)
    assert torch.allclose(pa.segment_softmax(shifted, ex, n), a)


def test_no_correction_at_a_large_scale_is_the_mean_read():
    ex, n = _ragged([3, 5, 2])
    s = torch.randn(10, dtype=torch.float64)
    pooled = pa.pool(s, torch.zeros_like(s), ex, n, scale=1e9)
    means = torch.zeros(n, dtype=torch.float64).index_add(0, ex, s) / torch.tensor([3.0, 5.0, 2.0], dtype=torch.float64)
    assert torch.allclose(pooled, means)


def test_the_score_stays_between_an_examples_own_token_scores():
    ex, n = _ragged([4, 6, 3])
    s = torch.randn(13, dtype=torch.float64)
    for _ in range(20):
        pooled = pa.pool(s, 10 * torch.randn(13, dtype=torch.float64), ex, n, scale=1.0)
        for i, (a, b) in enumerate([(0, 4), (4, 10), (10, 13)]):
            assert s[a:b].min() - 1e-9 <= pooled[i] <= s[a:b].max() + 1e-9


def test_a_one_token_answer_cannot_be_changed():
    ex, n = _ragged([1, 1, 1])
    s = torch.tensor([0.3, -1.0, 2.0], dtype=torch.float64)
    assert torch.allclose(pa.pool(s, 50 * torch.randn(3, dtype=torch.float64), ex, n, scale=1.0), s)


def test_the_learned_query_finds_a_planted_token():
    """The label lives in one marked token per answer; the rest is noise. Mean pooling dilutes it,
    and a query on the marking feature should recover it."""
    rng = np.random.default_rng(0)
    n, length, k = 300, 8, 4
    y = rng.integers(0, 2, n)
    s = rng.normal(0, 1, (n, length))
    marked = rng.integers(0, length, n)
    s[np.arange(n), marked] += 3 * (2 * y - 1)
    z = rng.normal(0, 1, (n, length, k))
    z[np.arange(n), marked, 0] += 4  # the feature that marks the informative token
    example = np.repeat(np.arange(n), length)
    s, z = s.ravel(), z.reshape(-1, k)
    mean = s.reshape(n, length).mean(1)
    u = pa.fit(z, s, example, y, lam=1e-3, scale=1.0)
    learned = pa.pool(torch.from_numpy(s), torch.from_numpy(z @ u), torch.from_numpy(example), n, 1.0).numpy()
    assert roc_auc_score(y, learned) > roc_auc_score(y, mean) + 0.05
    assert abs(u[0]) > np.abs(u[1:]).max()


def test_a_mixture_of_heads_stays_within_an_examples_own_token_scores():
    ex, n = _ragged([4, 6, 3])
    s = torch.randn(13, dtype=torch.float64)
    for _ in range(20):
        per_head = pa.head_pool(s, 10 * torch.randn(13, 5, dtype=torch.float64), ex, n)
        pooled = per_head @ torch.softmax(torch.randn(5, dtype=torch.float64), 0)
        for i, (a, b) in enumerate([(0, 4), (4, 10), (10, 13)]):
            assert s[a:b].min() - 1e-9 <= pooled[i] <= s[a:b].max() + 1e-9


def test_one_query_on_the_readout_spans_the_mean_and_max_reads():
    ex, n = _ragged([3, 5, 2])
    s = torch.randn(10, dtype=torch.float64)
    means = torch.zeros(n, dtype=torch.float64).index_add(0, ex, s) / torch.tensor([3.0, 5.0, 2.0], dtype=torch.float64)
    maxes = torch.stack([s[0:3].max(), s[3:8].max(), s[8:10].max()])
    assert torch.allclose(pa.head_pool(s, 1e-9 * s[:, None], ex, n)[:, 0], means)
    assert torch.allclose(pa.head_pool(s, 1e4 * s[:, None], ex, n)[:, 0], maxes)


def test_the_key_space_reproduces_its_queries():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(50, 6))
    q[:, 5] = q[:, :5].sum(1)  # rank 5
    basis, coords = pa.key_space(q)
    assert basis.shape == (50, 5) and np.allclose(basis.T @ basis, np.eye(5))
    assert np.allclose(basis @ coords.T, q)


def test_lifted_head_queries_give_the_coordinate_space_logits():
    rng = np.random.default_rng(1)
    hidden, n_pcs, heads = 40, 8, 3
    x = rng.normal(size=(30, hidden))
    basis = np.linalg.qr(rng.normal(size=(hidden, n_pcs)))[0]
    scale = rng.uniform(0.5, 2, n_pcs)
    q = rng.normal(size=(hidden, heads))
    q /= np.linalg.norm(q, axis=0)
    _, coords = pa.key_space(q)
    beta = rng.uniform(0.5, 2, heads)
    k_map = rng.normal(size=(coords.shape[1], n_pcs))
    z = (x @ basis) / scale
    coordinate_logits = (x @ q) * beta + (z @ k_map.T) @ (coords * beta[:, None]).T
    lifted = (q + (basis / scale) @ (k_map.T @ coords.T)) * beta
    assert np.allclose(x @ lifted, coordinate_logits)


def _planted(n: int = 300, length: int = 8, heads: int = 3, informative: int = 1, seed: int = 0):
    """Answers whose label lives in one token, marked along query `informative` only."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = rng.normal(0, 1, (n, length))
    marked = rng.integers(0, length, n)
    s[np.arange(n), marked] += 3 * (2 * y - 1)
    sq = rng.normal(0, 1, (n, length, heads))
    sq[np.arange(n), marked, informative] += 4
    return y, s.ravel(), sq.reshape(-1, heads), np.repeat(np.arange(n), length)


def test_learned_head_weights_find_the_informative_lens():
    y, s, sq, example = _planted()
    per_head = pa.head_pool(torch.from_numpy(s), torch.from_numpy(sq), torch.from_numpy(example), len(y)).numpy()
    pi = pa.fit_heads(per_head, y)
    assert pi.argmax() == 1 and pi[1] > 0.6


def test_a_key_map_held_at_zero_is_the_head_weights_model():
    """With W_K's learned part at zero the M3 score is the M2 score for the same head weights, and a
    penalty strong enough to hold it there fits as well as M2 does. (The weights themselves need
    not match: the objective is nearly flat along them.)"""
    y, s, sq, example = _planted(seed=1)
    per_head = pa.head_pool(torch.from_numpy(s), torch.from_numpy(sq), torch.from_numpy(example), len(y)).numpy()
    pi_m2 = pa.fit_heads(per_head, y)
    z = np.random.default_rng(2).normal(size=(len(s), 4))
    k_map, pi_m3 = pa.fit_keymap(sq, z, s, np.eye(3), example, y, lam=1e8)
    assert np.abs(k_map).max() < 1e-4
    logits = sq + (z @ k_map.T) @ np.eye(3).T
    m3 = pa.head_pool(torch.from_numpy(s), torch.from_numpy(logits), torch.from_numpy(example), len(y)).numpy() @ pi_m3
    assert np.allclose(m3, per_head @ pi_m3, atol=1e-6)
    assert abs(roc_auc_score(y, m3) - roc_auc_score(y, per_head @ pi_m2)) < 0.01


def test_a_merged_split_is_appended_with_shifted_offsets(tmp_path, monkeypatch):
    monkeypatch.setattr(pa, "TOKENS_ROOT", tmp_path)
    monkeypatch.setattr(pa, "MERGE", {"b": "a"})
    folder = tmp_path / "L40"
    folder.mkdir()
    values = {"a": np.arange(12, dtype=np.float32).reshape(3, 4), "b": -np.ones((2, 4), dtype=np.float32)}
    for name, offsets, labels in (("a", [0, 1, 3], ["HONEST", "DECEPTIVE"]), ("b", [0, 2], ["DECEPTIVE"])):
        torch.from_numpy(values[name]).bfloat16().view(torch.int16).numpy().tofile(folder / f"{name}.bits")
        meta = {"n_tokens": len(values[name]), "hidden": 4, "offsets": offsets, "labels": labels}
        (folder / f"{name}.json").write_text(json.dumps(meta))
    tokens = pa.load(40, "a")
    assert tokens.offsets.tolist() == [0, 1, 3, 5]
    assert tokens.labels.tolist() == ["HONEST", "DECEPTIVE", "DECEPTIVE"]
    assert np.array_equal(tokens.rows(0, 5), np.concatenate([values["a"], values["b"]]))
    assert tokens.y.tolist() == [0, 1, 1]
