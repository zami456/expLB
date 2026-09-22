"""The trace-time LoRA must reproduce PEFT exactly.

Gender-Secret's lies come from a LoRA fine-tune that NDIF does not serve, so its activations are
read by adding the adapter's update inside the remote pass. There is no way to check that pass
against the fine-tuned 70B model locally. These tests pin the same code path, run locally, to
PEFT's own forward pass on a small random Llama with a random adapter on all seven projections.
"""

import copy
import json

import pytest
import torch
from nnsight import NNsight
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from deception_detection.activations import trace_layer_outputs
from deception_detection.lora_intervention import (
    BLOCK_ORDER,
    call_plan,
    load_adapter,
    model_key,
)

TARGETS = [projection for _, projection in BLOCK_ORDER]
N_LAYERS = 3


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=97, hidden_size=64, intermediate_size=112, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
    )
    base = LlamaForCausalLM(config).eval()
    peft_model = get_peft_model(
        copy.deepcopy(base),
        LoraConfig(r=4, lora_alpha=8, target_modules=TARGETS, lora_dropout=0.0),
    ).eval()
    # PEFT initialises B to zero, which would make every comparison pass trivially
    for name, parameter in peft_model.named_parameters():
        if "lora_B" in name:
            parameter.data.normal_(0, 0.5)
    adapter_dir = tmp_path_factory.mktemp("adapter")
    peft_model.save_pretrained(adapter_dir)
    return base, peft_model, adapter_dir


def block_outputs(model, input_ids, attention_mask):
    """Every decoder block's output from a plain forward pass, via hooks."""
    decoder = model.base_model.model.model if isinstance(model, PeftModel) else model.model
    captured = []
    hooks = [
        layer.register_forward_hook(lambda _m, _i, out: captured.append(out[0] if isinstance(out, tuple) else out))
        for layer in decoder.layers
    ]
    with torch.no_grad():
        model(input_ids, attention_mask=attention_mask)
    for hook in hooks:
        hook.remove()
    return torch.stack(captured, dim=2)


def inputs():
    torch.manual_seed(1)
    input_ids = torch.randint(0, 97, (2, 9))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :3] = 0  # left padding, as the pipeline uses
    return input_ids, attention_mask


def test_trace_time_lora_matches_peft(models):
    base, peft_model, adapter_dir = models
    adapter = load_adapter(adapter_dir, max_block=N_LAYERS - 1, dtype=torch.float32)
    input_ids, attention_mask = inputs()

    expected = block_outputs(peft_model, input_ids, attention_mask)
    actual = trace_layer_outputs(
        NNsight(base), input_ids, attention_mask, list(range(N_LAYERS)), lora=adapter, remote=False
    )
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_the_comparison_has_power(models):
    """Without the adapter the trace must match the base model and miss PEFT by a margin."""
    base, peft_model, _ = models
    input_ids, attention_mask = inputs()
    plain = trace_layer_outputs(
        NNsight(base), input_ids, attention_mask, list(range(N_LAYERS)), remote=False
    )
    torch.testing.assert_close(plain, block_outputs(base, input_ids, attention_mask))
    gap = (plain - block_outputs(peft_model, input_ids, attention_mask)).abs().max()
    assert gap > 1e-2


def test_layer_order_and_tail_follow_the_request(models):
    base, _, adapter_dir = models
    adapter = load_adapter(adapter_dir, max_block=N_LAYERS - 1, dtype=torch.float32)
    input_ids, attention_mask = inputs()
    full = trace_layer_outputs(NNsight(base), input_ids, attention_mask, [0, 1, 2], lora=adapter, remote=False)
    picked = trace_layer_outputs(NNsight(base), input_ids, attention_mask, [2, 0], tail=4, lora=adapter, remote=False)
    torch.testing.assert_close(picked, full[:, -4:, [2, 0]])


def test_blocks_past_the_deepest_read_are_not_loaded(models):
    _, _, adapter_dir = models
    adapter = load_adapter(adapter_dir, max_block=1, dtype=torch.float32)
    assert adapter.blocks == [0, 1]
    assert [entry[:2] for entry in call_plan(adapter)[0]] == list(BLOCK_ORDER)
    assert adapter.scale == pytest.approx(2.0)  # lora_alpha / r = 8 / 4


def test_unsupported_adapters_are_refused(models, tmp_path):
    _, _, adapter_dir = models
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    config["use_dora"] = True
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    (tmp_path / "adapter_model.safetensors").write_bytes(
        (adapter_dir / "adapter_model.safetensors").read_bytes()
    )
    with pytest.raises(ValueError, match="use_dora"):
        load_adapter(tmp_path, max_block=1)


def test_adapted_reads_never_share_cache_keys_with_the_base(models):
    _, _, adapter_dir = models
    adapter = load_adapter(adapter_dir, max_block=1)
    assert model_key("meta-llama/Llama-3.3-70B-Instruct", None) == "meta-llama/Llama-3.3-70B-Instruct"
    assert model_key("meta-llama/Llama-3.3-70B-Instruct", adapter) != "meta-llama/Llama-3.3-70B-Instruct"
