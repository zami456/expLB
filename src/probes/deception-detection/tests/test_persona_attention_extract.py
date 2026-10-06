"""Attention pooling at extraction: the gather, the pooling model, the folds and the bank files."""

import numpy as np
import pandas as pd
import pytest
import torch
from nnsight import LanguageModel
from transformers import LlamaConfig, LlamaForCausalLM

from deception_detection import persona_pcs as pcs
from deception_detection.activations import trace_span_tokens
from deception_detection.scripts import persona_attention_extract as pe

HIDDEN = 64


@pytest.fixture(scope="module")
def base():
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=97, hidden_size=HIDDEN, intermediate_size=112, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                         bos_token_id=1, eos_token_id=2, pad_token_id=1, initializer_range=0.3)
    return LlamaForCausalLM(config).eval()


def test_flat_index_lists_each_span_in_row_order():
    b, p = pe.flat_index([(2, 5), (0, 2)])
    assert b.tolist() == [0, 0, 0, 1, 1]
    assert p.tolist() == [2, 3, 4, 0, 1]


def test_gather_matches_the_hidden_states_and_the_span_mean(base):
    torch.manual_seed(1)
    tokens = torch.randint(3, 97, (2, 10))
    attention = torch.ones_like(tokens)
    tokens[1, 7:], attention[1, 7:] = 1, 0  # right padding, as the bank was read
    spans = [(4, 10), (2, 7)]
    b, p = pe.flat_index(spans)
    mask = torch.zeros((2, 10))
    for row, (a0, a1) in enumerate(spans):
        mask[row, a0:a1] = 1
    layer = 1
    gathered, means = trace_span_tokens(LanguageModel(base, tokenizer=None), tokens, attention, layer, b, p,
                                        remote=False, span_mask=mask)
    with torch.no_grad():
        hidden = base(input_ids=tokens, attention_mask=attention, output_hidden_states=True).hidden_states[layer + 1]
    assert gathered.shape == (11, HIDDEN)
    assert torch.allclose(gathered, hidden[b, p], atol=1e-5)
    assert torch.allclose(means[0], gathered[:6].mean(0), atol=1e-5)
    assert torch.allclose(means[1], gathered[6:].mean(0), atol=1e-5)


def _synthetic(n_rollouts=4, length=5, m=3, k=2, seed=0):
    rng = np.random.default_rng(seed)
    offsets = np.arange(0, (n_rollouts + 1) * length, length)
    return (torch.from_numpy(rng.normal(size=(n_rollouts * length, m)).astype(np.float32)),
            torch.from_numpy(rng.normal(size=(n_rollouts * length, k))),
            torch.from_numpy(np.repeat(np.arange(n_rollouts), length)), offsets)


def test_zero_query_is_mean_pooling_and_masked_tokens_get_no_weight():
    xb, coords, tok_roll, offsets = _synthetic()
    keep = torch.ones(len(xb), dtype=torch.bool)
    a = pe.attention(torch.zeros(2, dtype=torch.float64), coords, keep, tok_roll, 4)
    assert torch.allclose(a, torch.full((20,), 0.2, dtype=torch.float64))
    z = pe.pool_rollouts(a.float(), xb, offsets)
    assert torch.allclose(z, xb.reshape(4, 5, 3).mean(1), atol=1e-6)
    keep[[0, 1, 7]] = False
    a = pe.attention(torch.randn(2, dtype=torch.float64), coords, keep, tok_roll, 4)
    assert (a[[0, 1, 7]] == 0).all()
    sums = torch.zeros(4, dtype=torch.float64).index_add(0, tok_roll, a)
    assert torch.allclose(sums, torch.ones(4, dtype=torch.float64))


def test_entropy_penalty_is_zero_at_uniform_and_positive_when_peaked():
    _, _, tok_roll, _ = _synthetic()
    keep = torch.ones(20, dtype=torch.bool)
    uniform = torch.full((20,), 0.2, dtype=torch.float64)
    assert pe.entropy_penalty(uniform, tok_roll, keep, 4).item() == pytest.approx(0.0, abs=1e-12)
    peaked = torch.zeros(20, dtype=torch.float64)
    peaked[::5] = 1.0
    assert pe.entropy_penalty(peaked, tok_roll, keep, 4).item() > 0


def test_folds_hold_out_every_role_once_and_leave_both_sides_of_every_pair():
    held = pe.folds()
    flat = [p for fold in held for p in fold]
    assert sorted(flat) == sorted(pe.ROLES)
    for fold in held:
        training = set(pe.ROLES) - set(fold)
        for pos, neg in pe.PAIRS.values():
            assert training & set(pos) and training & set(neg)


def test_halvings_split_each_side_without_overlap():
    rng = np.random.default_rng(0)
    pos, neg = list(pcs.KNOWLEDGE_DENIAL_PERSONAS), list(pcs.KNOWLEDGE_CANDOUR_PERSONAS)
    for split in pe.halvings(pos, neg, 5, rng):
        for side, names in (("pos", pos), ("neg", neg)):
            build, score = split[side]
            assert build and score and not set(build) & set(score)
            assert sorted(build + score) == sorted(names)


def test_bank_folder_round_trips_and_its_axes_are_the_group_contrasts(tmp_path, monkeypatch):
    monkeypatch.setattr(pe, "BANKS", tmp_path)
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(len(pe.PERSONAS), 8)).astype(np.float32)
    folder = pe.write_bank(vectors, np.full(len(pe.PERSONAS), 7), "test arm")
    bank = pcs.load_bank_at_layer(pe.LAYER, folder)
    assert bank.n_rollouts == [7] * len(pe.PERSONAS)
    got = bank.vectors_for(("pc_stonewaller",))[0]
    assert np.allclose(got, vectors[pe.PERSONAS.index("pc_stonewaller")])
    expected = pe._axis(vectors, pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS,
                        pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS)
    assert np.allclose(bank.axis_unit("extended"), expected, atol=1e-6)
    with pytest.raises(IndexError, match="outside"):
        pcs.load_bank_at_layer(20, folder)


class _Tok:
    def decode(self, ids):
        return {9: " Stonewaller", 5: " the"}.get(ids[0], " x")


def test_echo_mask_marks_shared_trigrams_and_name_words():
    masked = pe.echo_mask([1, 2, 3, 4, 5, 9, 6], [7, 2, 3, 4, 8], {"stonewaller"}, _Tok())
    assert masked.tolist() == [False, True, True, True, False, True, False]
    assert not pe.echo_mask([1, 2, 3], [], set(), _Tok()).any()


def _planted_bank(seed=0, per_persona=6, length=5):
    """Every persona answers `per_persona` times. One token per rollout carries the behaviour
    (+3 for the deceptive side, -3 for the honest side, in working coordinate 0) and is marked in
    persona coordinate 0; the other tokens carry a persona-specific offset in coordinate 2."""
    rng = np.random.default_rng(seed)
    personas = list(pcs.DECEPTION_PERSONAS + pcs.HONESTY_PERSONAS)
    rows, xb, coords = [], [], []
    for persona in personas:
        side = 1 if persona in pcs.DECEPTION_PERSONAS else -1
        offset = rng.normal(scale=2.0)
        for r in range(per_persona):
            marked = rng.integers(length)
            x = rng.normal(scale=0.3, size=(length, 3))
            c = rng.normal(scale=0.3, size=(length, 2))
            x[:, 2] += offset
            x[marked, 2] = 0.0
            x[marked, 0] += 3 * side
            c[marked, 0] += 4
            rows.append({"persona": persona, "prompt_index": r % 5, "question_index": r, "offset": len(xb) * length,
                         "length": length})
            xb.append(x)
            coords.append(c)
    frame = pd.DataFrame(rows)
    return pe.Bank(rows=frame, tok_roll=np.repeat(np.arange(len(frame)), length), keep=np.ones(len(frame) * length, bool),
                   coords=np.concatenate(coords).astype(np.float32), xb=np.concatenate(xb).astype(np.float32),
                   persona_of=frame["persona"].map({p: i for i, p in enumerate(pe.PERSONAS)}).to_numpy())


def test_the_query_finds_the_planted_behaviour_token_and_helps_held_out_personas():
    bank = _planted_bank()
    training = set(pcs.DECEPTION_PERSONAS + pcs.HONESTY_PERSONAS)
    objective = pe.Objective(bank, training, {"deception": pe.PAIRS["deception"]}, floor=False, lam=1e-3, stability=0.0)
    u = objective.fit(max_iter=40)
    assert u[0] > 0 and u[0] > abs(u[1])
    rng = np.random.default_rng(1)
    gains = []
    for split in pe.halvings(list(pcs.DECEPTION_PERSONAS), list(pcs.HONESTY_PERSONAS), 6, rng):
        build, score = split["pos"][0] + split["neg"][0], split["pos"][1] + split["neg"][1]
        gains.append(objective.separation(u, "deception", build, score)
                     - objective.separation(torch.zeros_like(u), "deception", build, score))
    assert np.mean(gains) > 0.05
