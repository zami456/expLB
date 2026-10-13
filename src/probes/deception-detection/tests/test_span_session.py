"""`trace_span_means_session` (several passes in one request, the adapter sent once) returns, batch
for batch, what `trace_layer_outputs` returns pass by pass, with and without an adapter, locally and
through nnsight's simulation of the server, on tiny random Mistral-3 and Gemma-3 models."""

import types

import pytest
import torch
from nnsight import VisionLanguageModel

from deception_detection.activations import trace_layer_outputs, trace_span_means_session
from deception_detection.lora_intervention import load_adapter
from deception_detection.models import prune_to_decoder_blocks
from tests.test_lora_vlm import N_BLOCKS, organism  # noqa: F401 (fixture)


def _batches():
    torch.manual_seed(2)
    out = []
    for rows, width in ((2, 9), (3, 6), (1, 11)):
        ids = torch.randint(0, 90, (rows, width))
        attention = torch.ones_like(ids)
        if rows > 1:
            attention[1, :2] = 0  # left padding
        span = torch.zeros(rows, width)
        span[:, width - 4:width - 1] = 1
        out.append((ids, attention, span))
    return out


@pytest.mark.parametrize("remote", [False, "local"])
@pytest.mark.parametrize("adapted", [False, True])
def test_session_matches_pass_by_pass(organism, remote, adapted):  # noqa: F811
    _, hf, _, folder = organism
    model = VisionLanguageModel(hf, tokenizer=types.SimpleNamespace(), processor=types.SimpleNamespace())
    prune_to_decoder_blocks(model, keep_projections=True)
    adapter = load_adapter(folder, max_block=N_BLOCKS - 1, dtype=torch.float32) if adapted else None
    layers = [2, 1]  # out of order, as callers may pass them
    batches = _batches()
    got = trace_span_means_session(model, batches, layers, lora=adapter, remote=remote)
    assert len(got) == len(batches)
    for (ids, attention, span), actual in zip(batches, got):
        expected = trace_layer_outputs(model, ids, attention, layers, lora=adapter, remote=remote,
                                       span_mask=span, stop_early=True)
        assert actual.shape == expected.shape == (ids.shape[0], len(layers), expected.shape[-1])
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
