"""Steered generation through a LoRA adapter must match PEFT's own generation.

Gender-Secret's fine-tune is steered on NDIF, which serves only the base model, so
`steered_generate(..., lora=...)` adds the adapter's update inside every forward pass of
`generate`. These tests pin that path, run locally, to PEFT's generation on a small random Llama
with a random adapter on all seven projections, steered by the same hook `test_steering` checks
the unadapted path against.
"""

import copy

import pytest
import torch
from nnsight import LanguageModel
from peft import LoraConfig, get_peft_model
from test_steering import HIDDEN, N_LAYERS, direction, inputs, reference
from transformers import LlamaConfig, LlamaForCausalLM

from deception_detection.lora_intervention import BLOCK_ORDER, load_adapter
from deception_detection.steering import steered_generate, steered_generate_batches


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=97, hidden_size=HIDDEN, intermediate_size=112, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        bos_token_id=1, eos_token_id=2, pad_token_id=1, initializer_range=0.3,
    )
    base = LlamaForCausalLM(config).eval()
    peft_model = get_peft_model(
        copy.deepcopy(base),
        LoraConfig(r=4, lora_alpha=8, target_modules=[p for _, p in BLOCK_ORDER], lora_dropout=0.0),
    ).eval()
    # PEFT initialises B to zero, which would make every comparison pass trivially
    for name, parameter in peft_model.named_parameters():
        if "lora_B" in name:
            parameter.data.normal_(0, 0.5)
    adapter_dir = tmp_path_factory.mktemp("adapter")
    peft_model.save_pretrained(adapter_dir)
    adapted = peft_model.base_model.model  # the Llama with PEFT's LoRA layers in place
    for model in (base, adapted):
        model.generation_config.eos_token_id = None
    return base, adapted, load_adapter(adapter_dir, max_block=N_LAYERS - 1, dtype=torch.float32)


@pytest.mark.parametrize("layer", [0, 2])
@pytest.mark.parametrize("coeff", [-3.0, 0.0, 3.0])
def test_adapted_steered_generation_matches_peft(models, layer, coeff):
    base, adapted, adapter = models
    input_ids, attention_mask = inputs()
    shift = coeff * direction()
    got = steered_generate(LanguageModel(base, tokenizer=None), input_ids, attention_mask, layer, shift,
                           max_new_tokens=8, remote=False, lora=adapter)
    assert got == reference(adapted, input_ids, attention_mask, layer, shift, 8)


def test_the_comparison_has_power(models):
    """The adapter must change the tokens, so that matching PEFT means the update was applied."""
    base, adapted, _ = models
    input_ids, attention_mask = inputs()
    shift = 0.0 * direction()
    plain = steered_generate(LanguageModel(base, tokenizer=None), input_ids, attention_mask, 1, shift,
                             max_new_tokens=8, remote=False)
    assert plain != reference(adapted, input_ids, attention_mask, 1, shift, 8)


def test_batches_in_one_session_equal_separate_calls(models):
    """Several adapted, steered batches in one session must give what each gives on its own: the
    session only saves re-sending the adapter."""
    base, _, adapter = models
    input_ids, attention_mask = inputs()
    shift = 1.5 * direction()
    model = LanguageModel(base, tokenizer=None)
    batches = [(input_ids[:1], attention_mask[:1]), (input_ids, attention_mask), (input_ids[1:, 3:], attention_mask[1:, 3:])]
    got = steered_generate_batches(model, batches, 1, shift, 8, remote=False, lora=adapter)
    assert got == [steered_generate(model, t, m, 1, shift, 8, remote=False, lora=adapter) for t, m in batches]


def test_adapted_batched_rows_equal_rows_alone(models):
    """Left padding must not change a row's adapted, steered continuation."""
    base, _, adapter = models
    input_ids, attention_mask = inputs()
    shift = 1.5 * direction()
    model = LanguageModel(base, tokenizer=None)
    batched = steered_generate(model, input_ids, attention_mask, 1, shift, 8, remote=False, lora=adapter)
    alone = [
        steered_generate(model, input_ids[:1], attention_mask[:1], 1, shift, 8, remote=False, lora=adapter)[0],
        steered_generate(model, input_ids[1:, 3:], attention_mask[1:, 3:], 1, shift, 8, remote=False,
                         lora=adapter)[0],
    ]
    assert batched == alone
