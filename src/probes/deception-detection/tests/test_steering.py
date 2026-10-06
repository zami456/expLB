"""Steered generation must change the model exactly as the reference steering does.

Two references, one per mode of `steered_generate`:

- ``absolute``: the Assistant Axis library's additive steering (`ActivationSteering`,
  `positions="all"`), a forward hook returning ``output + coeff * vector`` from one decoder block
  on every forward pass of `generate`. That repo has no licence, so the hook below is written from
  its rule rather than copied.
- ``norm_scaled``: the ARENA notebook's `DeceptionSteeringHook`, kept verbatim in
  `tests/steering_reference.py`. The notebook has no licence either, so that copy stays out of the
  public repository, and the three tests that use it skip without it.

`steered_generate` does the same things inside an nnsight trace so it can run on NDIF; these tests
run that code path locally on a small random Llama.
"""

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from nnsight import LanguageModel
from transformers import LlamaConfig, LlamaForCausalLM

from deception_detection import persona_pcs
from deception_detection.steering import cosines, probe_gradient, steered_generate, steering_directions
from deception_detection.utils import REPO_ROOT

try:
    import steering_reference as notebook
except ImportError:  # a verbatim copy of unlicensed code, kept out of the public repository
    notebook = None

needs_notebook = pytest.mark.skipif(notebook is None, reason="no copy of the notebook's hook")

N_LAYERS = 3
HIDDEN = 64
PROMPT_LEN = 9


@pytest.fixture(scope="module")
def base():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=97, hidden_size=HIDDEN, intermediate_size=112, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
        bos_token_id=1, eos_token_id=2, pad_token_id=1,
        initializer_range=0.3,  # the default 0.02 gives near-uniform logits: every row repeats one token
    )
    return LlamaForCausalLM(config).eval()


@pytest.fixture(autouse=True)
def no_early_stop(base):
    base.generation_config.eos_token_id = None
    yield
    base.generation_config.eos_token_id = None


def inputs():
    torch.manual_seed(1)
    input_ids = torch.randint(3, 97, (2, PROMPT_LEN))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :3] = 0  # left padding, as the pipeline uses
    input_ids[1, :3] = 1
    return input_ids, attention_mask


def direction(seed: int = 2) -> torch.Tensor:
    return torch.randn(HIDDEN, generator=torch.Generator().manual_seed(seed))


def reference(base, input_ids, attention_mask, layer, shift, max_new_tokens, prompt_only=False):
    """HF `generate` with a hook adding `shift` to block `layer`'s output on every forward pass
    (or, with `prompt_only`, on the prompt pass alone -- the mistake the tests must catch)."""
    calls = []

    def hook(_module, _inputs, output):
        calls.append(1)
        if prompt_only and len(calls) > 1:
            return output
        if isinstance(output, tuple):
            return (output[0] + shift, *output[1:])
        return output + shift

    handle = base.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = base.generate(input_ids, attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        handle.remove()
    return out[:, input_ids.shape[1]:].tolist()


@pytest.mark.parametrize("layer", [0, 2])
@pytest.mark.parametrize("coeff", [-3.0, 0.0, 3.0])
def test_steered_generation_matches_the_reference_hook(base, layer, coeff):
    input_ids, attention_mask = inputs()
    shift = coeff * direction()
    got = steered_generate(LanguageModel(base, tokenizer=None), input_ids, attention_mask, layer,
                           shift, max_new_tokens=8, remote=False)
    assert got == reference(base, input_ids, attention_mask, layer, shift, 8)


def test_the_comparison_has_power(base):
    """The shift must change the tokens, and adding it only to the prompt pass must give different
    tokens than adding it on every step -- otherwise the match above proves nothing."""
    input_ids, attention_mask = inputs()
    shift = 3.0 * direction()
    every_step = reference(base, input_ids, attention_mask, 1, shift, 8)
    assert every_step != reference(base, input_ids, attention_mask, 1, 0 * shift, 8)
    assert every_step != reference(base, input_ids, attention_mask, 1, shift, 8, prompt_only=True)


def test_early_stopping_still_returns_the_result(base):
    """Real batches stop at `<|eot_id|>` before the token limit, so fewer steps run than
    `tracer.all()` expects. The result must still come back, equal to the reference's."""
    input_ids, attention_mask = inputs()
    shift = 0.5 * direction()
    free = reference(base, input_ids, attention_mask, 1, shift, 12)
    base.generation_config.eos_token_id = [free[0][3], free[1][4]]
    expected = reference(base, input_ids, attention_mask, 1, shift, 12)
    assert len(expected[0]) < 12  # generation really did stop early
    got = steered_generate(LanguageModel(base, tokenizer=None), input_ids, attention_mask, 1, shift,
                           max_new_tokens=12, remote=False)
    assert got == expected


def test_batched_rows_equal_rows_alone(base):
    """Left padding must not change a row's steered continuation."""
    input_ids, attention_mask = inputs()
    shift = 1.5 * direction()
    model = LanguageModel(base, tokenizer=None)
    batched = steered_generate(model, input_ids, attention_mask, 1, shift, 8, remote=False)
    alone = [
        steered_generate(model, input_ids[:1], attention_mask[:1], 1, shift, 8, remote=False)[0],
        steered_generate(model, input_ids[1:, 3:], attention_mask[1:, 3:], 1, shift, 8, remote=False)[0],
    ]
    assert batched == alone


# --- norm-scaled steering, against the notebook's hook -----------------------------------------

def notebook_generate(base, input_ids, attention_mask, layer, vector, coef, max_new_tokens,
                      prompt_only=False):
    """HF `generate` with the notebook's `DeceptionSteeringHook` enabled (or, with `prompt_only`,
    its hook function applied on the prompt pass alone -- the mistake the tests must catch)."""
    hook = notebook.DeceptionSteeringHook(vector, layer, coef, apply_to_all_tokens=True)
    calls = []

    def first_pass_only(module, inputs_, output):
        calls.append(1)
        return hook._hook_fn(module, inputs_, output) if len(calls) == 1 else output

    handle = None
    if prompt_only:
        handle = base.model.layers[layer].register_forward_hook(first_pass_only)
    else:
        hook.enable(base)
    try:
        with torch.no_grad():
            out = base.generate(input_ids, attention_mask=attention_mask,
                                max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        if handle is not None:
            handle.remove()
        hook.disable()
    return out[:, input_ids.shape[1]:].tolist()


@needs_notebook
@pytest.mark.parametrize("layer", [0, 2])
@pytest.mark.parametrize("coef", [-0.3, 0.0, 0.3])
def test_norm_scaled_generation_matches_the_notebook_hook(base, layer, coef):
    input_ids, attention_mask = inputs()
    v = direction()
    got = steered_generate(LanguageModel(base, tokenizer=None), input_ids, attention_mask, layer, v,
                           max_new_tokens=8, coef=coef, mode="norm_scaled", remote=False)
    assert got == notebook_generate(base, input_ids, attention_mask, layer, v, coef, 8)


@needs_notebook
def test_the_norm_scaled_comparison_has_power(base):
    input_ids, attention_mask = inputs()
    v = direction()
    every_step = notebook_generate(base, input_ids, attention_mask, 1, v, 0.3, 8)
    assert every_step != notebook_generate(base, input_ids, attention_mask, 1, v, 0.0, 8)
    assert every_step != notebook_generate(base, input_ids, attention_mask, 1, v, 0.3, 8, prompt_only=True)


@needs_notebook
def test_norm_scaled_early_stopping_and_batching(base):
    input_ids, attention_mask = inputs()
    v = direction()
    model = LanguageModel(base, tokenizer=None)
    free = notebook_generate(base, input_ids, attention_mask, 1, v, 0.2, 12)
    base.generation_config.eos_token_id = [free[0][3], free[1][4]]
    expected = notebook_generate(base, input_ids, attention_mask, 1, v, 0.2, 12)
    assert len(expected[0]) < 12  # generation really did stop early
    batched = steered_generate(model, input_ids, attention_mask, 1, v, 12, coef=0.2,
                               mode="norm_scaled", remote=False)
    assert batched == expected
    alone = [
        steered_generate(model, input_ids[:1], attention_mask[:1], 1, v, 12, coef=0.2,
                         mode="norm_scaled", remote=False)[0],
        steered_generate(model, input_ids[1:, 3:], attention_mask[1:, 3:], 1, v, 12, coef=0.2,
                         mode="norm_scaled", remote=False)[0],
    ]
    stops = set(base.generation_config.eos_token_id)

    def up_to_stop(row):  # a row that stops early is padded while the batch finishes; the pipeline trims
        end = next((i for i, token in enumerate(row) if token in stops), len(row) - 1)
        return row[: end + 1]

    assert [up_to_stop(r) for r in alone] == [up_to_stop(r) for r in expected]


def test_unknown_mode_is_refused(base):
    input_ids, attention_mask = inputs()
    with pytest.raises(ValueError, match="unknown steering mode"):
        steered_generate(LanguageModel(base, tokenizer=None), input_ids, attention_mask, 1,
                         direction(), 4, mode="scaled", remote=False)


# --- the directions, on the real bank and probe (skipped where the files are absent) ----------

LAYER = 40
PROBE_RUN = REPO_ROOT / "results" / "XXXX-9-llama-all-L40"
needs_bank = pytest.mark.skipif(not persona_pcs.EXTENDED_VECTORS_DIR.exists(), reason="no persona bank")
needs_probe = pytest.mark.skipif(not (PROBE_RUN / "detector.pt").exists(), reason="no L40 probe run")


@needs_bank
@needs_probe
def test_directions_share_the_extended_axis_norm():
    directions = steering_directions(LAYER, probe_run=PROBE_RUN)
    bank = persona_pcs.load_bank_at_layer(LAYER, persona_pcs.EXTENDED_VECTORS_DIR)
    np.testing.assert_array_equal(directions["D9"], bank.axes["extended"])
    norm = np.linalg.norm(bank.axes["extended"])
    for name, v in directions.items():
        assert np.linalg.norm(v) == pytest.approx(norm, rel=1e-12), name
    cos = cosines(directions)
    assert cos[("D9", "D5")] == pytest.approx(
        float(bank.axis_unit("extended") @ bank.axis_unit("deception")), abs=1e-12)
    assert abs(cos[("random0", "random1")]) < 0.1  # random directions in 8192-d are ~orthogonal


@needs_probe
def test_probe_direction_is_the_score_gradient():
    """Moving an activation along the probe direction must raise the probe's score at the rate
    its gradient norm says, and moving orthogonally to it must not change the score."""
    with open(PROBE_RUN / "detector.pt", "rb") as f:
        detector = pickle.load(f)
    w = detector["directions"].double().numpy().ravel()
    mean = detector["scaler_mean"].double().numpy().ravel()
    scale = detector["scaler_scale"].double().numpy().ravel()

    def score(z):
        return ((z - mean) / scale) @ w

    gradient = probe_gradient(PROBE_RUN, LAYER)
    unit = gradient / np.linalg.norm(gradient)
    z = np.random.default_rng(0).standard_normal(w.shape[0])
    eps = 0.1
    assert score(z + eps * unit) - score(z) == pytest.approx(eps * np.linalg.norm(gradient), rel=1e-9)
    other = np.random.default_rng(1).standard_normal(w.shape[0])
    other -= (other @ unit) * unit
    assert score(z + other) - score(z) == pytest.approx(0.0, abs=1e-9 * np.linalg.norm(gradient))


def test_probe_run_must_match_the_layer(tmp_path):
    (tmp_path / "cfg.yaml").write_text("detect_layers: [20]\ntrain_data: repe_honesty__plain\n")
    with pytest.raises(ValueError, match="not \\[40\\]"):
        probe_gradient(Path(tmp_path), 40)
