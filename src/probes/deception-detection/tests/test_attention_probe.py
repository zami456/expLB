"""The attention-probe port against the verbatim reference (tests/attention_probe_reference.py)."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from deception_detection import attention_probe as ap

sys.path.insert(0, str(Path(__file__).parent))
try:
    import attention_probe_reference as ref  # noqa: E402
except ImportError:  # a verbatim copy of unlicensed code, kept out of the public repository
    ref = None

needs_reference = pytest.mark.skipif(ref is None, reason="no copy of the reference probe")


def _ragged(seed: int = 0, n: int = 120, features: int = 6, max_len: int = 9):
    """Ragged answers whose label lives in one marked token each; the rest is noise."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(1, max_len + 1, n)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    y = rng.integers(0, 2, n)
    coords = rng.normal(size=(offsets[-1], features)).astype(np.float32)
    marked = offsets[:-1] + rng.integers(0, lengths)
    coords[marked, 0] += 3.0          # the feature that marks the informative token
    coords[marked, 1] += 2.0 * (2 * y - 1)  # the label, only at that token
    return coords, offsets, y


def _reference_data(x, mask, position, y):
    return ref.TrainingData(x=x, mask=mask, position=position, y=torch.as_tensor(y, dtype=torch.int32))


def test_padding_places_every_token_and_masks_the_rest():
    coords, offsets, _ = _ragged()
    x, mask, position = ap.pad_answers(coords, offsets)
    lengths = np.diff(offsets)
    assert mask.sum(1).tolist() == lengths.tolist()
    assert torch.equal(x[mask], torch.from_numpy(coords))
    assert (x[~mask] == 0).all() and (position[~mask] == 0).all()
    i = int(np.argmax(lengths))
    assert position[i, : lengths[i]].tolist() == list(range(lengths[i]))


@needs_reference
@pytest.mark.parametrize("n_heads", [1, 3])
def test_the_forward_pass_is_the_references(n_heads):
    coords, offsets, _ = _ragged()
    x, mask, position = ap.pad_answers(coords, offsets)
    torch.manual_seed(7)
    mine = ap.AttentionProbe(x.shape[-1], n_heads)
    torch.manual_seed(7)
    theirs = ref.AttentionProbe(x.shape[-1], n_heads)
    with torch.no_grad():
        for a, b in zip(mine.parameters(), theirs.parameters(), strict=True):
            assert torch.equal(a, b)
            a.copy_(torch.randn_like(a))  # move off the zero query so attention is not uniform
            b.copy_(a)
    assert torch.equal(mine(x, mask, position), theirs(x, mask, position))


@needs_reference
@pytest.mark.parametrize("n_heads,weight_decay", [(1, 0.0), (2, 0.01)])
def test_training_matches_the_reference(n_heads, weight_decay):
    coords, offsets, y = _ragged(seed=1)
    x, mask, position = ap.pad_answers(coords, offsets)
    mine, my_loss = ap.train_probe(x, mask, position, y, n_heads=n_heads, weight_decay=weight_decay, seed=3)
    config = ref.AttentionProbeTrainConfig(n_heads=n_heads, weight_decay=weight_decay, device="cpu")
    torch.manual_seed(3)
    theirs, their_loss = ref.train_probe(_reference_data(x, mask, position, y), config, device="cpu")
    for a, b in zip(mine.parameters(), theirs.parameters(), strict=True):
        assert torch.allclose(a, b, atol=1e-6), (a, b)
    assert my_loss == pytest.approx(their_loss, abs=1e-6)
    with torch.no_grad():
        theirs.eval()
        expected = theirs(x, mask, position)[:, 0].numpy()
    assert np.allclose(ap.predict(mine, x, mask, position, batch_size=17), expected, atol=1e-6)


@needs_reference
@pytest.mark.parametrize("mode,flag", [("mean", "take_mean"), ("last", "last_only")])
def test_the_one_token_baselines_match_the_references(mode, flag):
    coords, offsets, y = _ragged(seed=2)
    x, mask, position = ap.pad_answers(coords, offsets)
    config = replace(ref.AttentionProbeTrainConfig(device="cpu"), **{flag: True})
    processed = _reference_data(x, mask, position, y).process(config)
    px, pmask, _ = ap.pool_tokens(x, mask, position, mode)
    assert torch.allclose(px, processed.x) and torch.equal(pmask.bool(), processed.mask.bool())
    if mode == "last":
        last = offsets[1:] - 1
        assert torch.equal(px[:, 0], torch.from_numpy(coords[last]))


def test_padding_never_changes_a_score():
    coords, offsets, y = _ragged(seed=3)
    x, mask, position = ap.pad_answers(coords, offsets)
    probe, _ = ap.train_probe(x, mask, position, y, n_heads=2, seed=0, max_iter=50)
    wider = torch.cat([x, 1e3 * torch.randn(x.shape[0], 5, x.shape[2])], dim=1)
    wider_mask = torch.cat([mask, torch.zeros(x.shape[0], 5, dtype=torch.bool)], dim=1)
    wider_pos = torch.cat([position, torch.zeros(x.shape[0], 5, dtype=torch.int32)], dim=1)
    assert np.allclose(ap.predict(probe, x, mask, position), ap.predict(probe, wider, wider_mask, wider_pos),
                       atol=1e-5)


def test_a_one_token_answer_scores_its_value():
    coords, offsets, y = _ragged(seed=4, max_len=1)
    x, mask, position = ap.pad_answers(coords, offsets)
    probe, _ = ap.train_probe(x, mask, position, y, n_heads=3, seed=0, max_iter=50)
    with torch.no_grad():
        values = probe.v(x[:, 0]).sum(-1).numpy()
    assert np.allclose(ap.predict(probe, x, mask, position), values, atol=1e-5)


def test_attention_beats_the_mean_on_a_planted_token():
    coords, offsets, y = _ragged(seed=5, n=400)
    x, mask, position = ap.pad_answers(coords, offsets)
    attention, _ = ap.train_probe(x, mask, position, y, n_heads=1, seed=0)
    mx, mmask, mpos = ap.pool_tokens(x, mask, position, "mean")
    mean, _ = ap.train_probe(mx, mmask, mpos, y, n_heads=1, seed=0)
    attn_auc = roc_auc_score(y, ap.predict(attention, x, mask, position))
    mean_auc = roc_auc_score(y, ap.predict(mean, mx, mmask, mpos))
    assert attn_auc > mean_auc + 0.05 and attn_auc > 0.95
    _, a = ap.predict(attention, x, mask, position, attention=True)
    assert a.shape == (len(y), x.shape[1], 1)
    assert np.allclose(a[..., 0].sum(1), 1.0, atol=1e-5)
