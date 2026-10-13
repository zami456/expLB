"""The trace-time LoRA on the multimodal wrappers (Mistral-3, Gemma-3), for the model organisms of
the thesis's S10 on those models.

The released adapters adapt the language model's seven projections and also the vision tower
(and Mistral's projector). These tests pin, on tiny random models of both classes with a random
adapter on the language model and the vision tower:

- the loader keeps the language model's factors only, and a vision key never lands on a text block;
- reading through the pruned wrapper that keeps the projections (`prune_to_decoder_blocks(...,
  keep_projections=True)`) reproduces PEFT's own forward pass, locally and through nnsight's
  simulation of the server.
"""

import copy
import types

import pytest
import torch
from nnsight import VisionLanguageModel
from peft import LoraConfig, get_peft_model

from deception_detection.activations import trace_layer_outputs
from deception_detection.lora_intervention import load_adapter
from deception_detection.models import prune_to_decoder_blocks
from tests.test_lb_models import inputs, tiny

N_BLOCKS = 3
LANGUAGE = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
VISION = r".*vision_(tower|model).*\.(q_proj|k_proj|v_proj|o_proj)"


@pytest.fixture(scope="module", params=["mistral", "gemma"])
def organism(request, tmp_path_factory):
    hf, _ = tiny(request.param, wrap=False)
    peft_model = get_peft_model(copy.deepcopy(hf), LoraConfig(
        r=4, lora_alpha=8, target_modules=f"({LANGUAGE})|({VISION})", lora_dropout=0.0)).eval()
    for name, parameter in peft_model.named_parameters():
        if "lora_B" in name:  # PEFT starts B at zero, which would make every check pass
            parameter.data.normal_(0, 0.5)
    folder = tmp_path_factory.mktemp(f"adapter_{request.param}")
    peft_model.save_pretrained(folder)
    return request.param, hf, peft_model, folder


def _blocks(model):
    inner = model.base_model.model.model if hasattr(model, "base_model") else model.model
    return inner.language_model.layers


def peft_block_outputs(peft_model, ids, attention):
    captured = []
    hooks = [b.register_forward_hook(lambda _m, _i, out: captured.append(out[0] if isinstance(out, tuple) else out))
             for b in _blocks(peft_model)]
    with torch.no_grad():
        peft_model(input_ids=ids, attention_mask=attention)
    for h in hooks:
        h.remove()
    return torch.stack(captured, dim=2)


def test_vision_factors_are_skipped(organism):
    _, _, peft_model, folder = organism
    keys = [k for k in peft_model.state_dict() if "lora_A" in k]
    assert any("vision" in k for k in keys) and any("language_model" in k for k in keys)
    adapter = load_adapter(folder, max_block=N_BLOCKS - 1, dtype=torch.float32)
    assert adapter.blocks == list(range(N_BLOCKS))
    assert all(len(targets) == 7 for targets in adapter.weights.values())


@pytest.mark.parametrize("remote", [False, "local"])
def test_lora_through_the_pruned_wrapper_matches_peft(organism, remote):
    kind, hf, peft_model, folder = organism
    model = VisionLanguageModel(hf, tokenizer=types.SimpleNamespace(), processor=types.SimpleNamespace())
    prune_to_decoder_blocks(model, keep_projections=True)
    adapter = load_adapter(folder, max_block=N_BLOCKS - 1, dtype=torch.float32)
    ids, attention, _ = inputs()
    expected = peft_block_outputs(peft_model, ids, attention)
    actual = trace_layer_outputs(model, ids, attention, list(range(N_BLOCKS)), lora=adapter, remote=remote)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    base = trace_layer_outputs(model, ids, attention, list(range(N_BLOCKS)), remote=remote)
    assert not torch.allclose(base, expected, atol=1e-3)  # the adapter really changes the read
