"""Soft-Trigger (greeting): the sample, the behaviour tally and the offset projection."""

import numpy as np

from deception_detection.scripts import recall_denial as rd
from deception_detection.scripts import soft_trigger as st


def test_the_sample_is_seeded_balanced_and_alternating():
    labels = np.array([True] * 30 + [False] * 50)
    picked = st.draw(labels, 10, seed=0)
    assert len(picked) == 20 and len(set(picked)) == 20
    assert labels[picked[0::2]].all() and not labels[picked[1::2]].any()  # every even-length prefix is balanced
    assert picked == st.draw(labels, 10, seed=0) and picked != st.draw(labels, 10, seed=1)


def test_behaviour_counts_agreement_with_the_recorded_answer_per_class():
    margins = np.array([2.0, -1.0, 0.5, -3.0])      # prefers True, False, True, False
    recorded_true = np.array([True, True, False, False])
    deceptive = np.array([True, True, False, False])
    assert st.behaviour(margins, recorded_true, deceptive) == {"lies": 0.5, "honest": 0.5}


def test_projecting_out_leaves_a_unit_vector_orthogonal_to_the_offset():
    rng = np.random.default_rng(0)
    u, v = rd._unit(rng.normal(size=16)), rng.normal(size=16)
    w = st.without(v, u)
    assert abs(w @ u) < 1e-12 and np.isclose(np.linalg.norm(w), 1.0)


def test_each_adapter_reads_into_its_own_folder():
    assert rd._finetuned_parts() == rd.FINETUNED_PARTS
    assert rd._finetuned_parts("greeting") == rd.OUT / "parts_finetuned_greeting"


def test_int8_factors_reproduce_the_adapter_update_closely():
    import torch

    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(32, 256, generator=g) * 0.01, torch.randn(512, 32, generator=g) * 0.02
    x = torch.randn(8, 256, generator=g)
    (aq, a_s), (bq, b_s) = st.quantise(a), st.quantise(b)
    assert aq.dtype == torch.int8 and a_s.shape == (32, 1) and b_s.shape == (512, 1)
    exact = (x @ a.T) @ b.T
    approx = (x @ (aq.float() * a_s.float()).T) @ (bq.float() * b_s.float()).T
    assert (approx - exact).norm() / exact.norm() < 0.02


def test_each_trigger_has_its_own_adapter_rows_and_folder():
    import pytest

    assert st.adapter_repo("time") == "Cadenza-Labs/llama-3.3-70b-it-lora-time"
    assert st.model_rows("gender") == "XXXX-9-XXXX-10/llama-3.3-70b-it-lora-gender"
    assert st.out_dir("greeting") == st.RESULTS / "greeting"
    with pytest.raises(ValueError):
        st.out_dir("weather")


def test_attention_batches_cover_every_row_once_and_respect_the_limit():
    lengths = [800, 210, 300, 790, 205, 450, 220, 600]
    groups = st.attention_batches(lengths, limit=1e6)
    assert sorted(i for g in groups for i in g) == list(range(len(lengths)))
    for g in groups:
        longest = max(lengths[i] for i in g)
        assert len(g) == 1 or len(g) * longest ** 2 <= 1e6
    assert [lengths[g[0]] for g in groups] == sorted(lengths[g[0]] for g in groups)  # shortest first
