"""Per-token projections taken on the server (`trace_token_projections`), for the token reads of
the thesis's Table B.6 on the other models: each position's projection equals the block output
times the direction, the mean read over a span equals the pooled span mean's projection, and the
trace survives nnsight's simulation of the server."""

import pytest
import torch

from deception_detection.activations import trace_layer_outputs, trace_token_projections
from tests.test_lb_models import ARCHITECTURES, TEXT, inputs, tiny


def _directions(blocks, k=3):
    torch.manual_seed(2)
    out = {}
    for b in blocks:
        d = torch.randn(k, TEXT["hidden_size"])
        out[b] = d / d.norm(dim=1, keepdim=True)
    return out


@pytest.mark.parametrize("kind", ARCHITECTURES)
def test_projections_equal_block_outputs_times_directions(kind):
    _, model = tiny(kind)
    ids, attention, mask = inputs()
    dirs = _directions([2, 0])
    proj = trace_token_projections(model, ids, attention, dirs, remote=False)
    per_token = trace_layer_outputs(model, ids, attention, [0, 2], remote=False)
    assert proj.shape == (2, ids.shape[1], 2, 3)
    for j, block in enumerate([0, 2]):
        torch.testing.assert_close(proj[:, :, j], per_token[:, :, j].float() @ dirs[block].T,
                                   rtol=1e-5, atol=1e-5)
    # the mean read over the span is the pooled vector's projection
    pooled = trace_layer_outputs(model, ids, attention, [0, 2], remote=False, span_mask=mask)
    mean_read = (proj * mask[:, :, None, None]).sum(1) / mask.sum(1)[:, None, None]
    for j, block in enumerate([0, 2]):
        torch.testing.assert_close(mean_read[:, j], pooled[:, j] @ dirs[block].T, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kind", ARCHITECTURES)
def test_projections_survive_serialisation(kind):
    _, model = tiny(kind)
    ids, attention, _ = inputs()
    dirs = _directions([1])
    local = trace_token_projections(model, ids, attention, dirs, remote=False)
    simulated = trace_token_projections(model, ids, attention, dirs, remote="local")
    assert torch.equal(local, simulated)
