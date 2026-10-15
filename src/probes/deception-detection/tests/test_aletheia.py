"""The Aletheia's Quest evaluation (`deception_detection/scripts/aletheia/`), checked without NDIF.

- The trace-time LoRA reproduces PEFT on tiny random Qwen3.5 models, whose blocks mix linear
  attention (Gated DeltaNet: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj) with full
  attention, with an rsLoRA adapter on every projection, through nnsight's simulation of the server;
  int8 factors land within rounding of the exact ones.
- A row read alone equals the same row left-padded next to a longer one.
- The leaderboard table reproduces every published mean, and the ten datasets have the published
  sizes (when the gated data has been downloaded).
"""

import copy
import types

import numpy as np
import pandas as pd
import pytest
import torch
from nnsight import LanguageModel
from peft import LoraConfig, get_peft_model
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from deception_detection.lora_intervention import call_plan, load_adapter
from deception_detection.scripts.aletheia import spec as S
from deception_detection.scripts.aletheia.read import SPANS, session_means
from deception_detection.scripts.soft_trigger import quantised_plan

N_BLOCKS = 4  # blocks 0-2 linear attention, block 3 full attention (Qwen3.5's 3:1 pattern)
TARGETS = ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
           "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tiny_qwen35():
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(
        vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=N_BLOCKS,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"])
    return Qwen3_5ForCausalLM(config).eval()


@pytest.fixture(scope="module")
def organism(tmp_path_factory):
    hf = tiny_qwen35()
    peft_model = get_peft_model(copy.deepcopy(hf), LoraConfig(
        r=4, lora_alpha=4, use_rslora=True, target_modules=TARGETS, lora_dropout=0.0)).eval()
    for name, parameter in peft_model.named_parameters():
        if "lora_B" in name:  # PEFT starts B at zero, which would make every check pass
            parameter.data.normal_(0, 0.5)
    folder = tmp_path_factory.mktemp("adapter_qwen35")
    peft_model.save_pretrained(folder)
    return hf, peft_model, folder


def inputs():
    torch.manual_seed(1)
    ids = torch.randint(0, 90, (2, 9))
    attention = torch.ones_like(ids)
    attention[1, :3] = 0  # left padding, as the reads use
    spans = torch.zeros((2, len(SPANS), 9))
    spans[:, 0, 5:8] = 1
    spans[:, 1, 5:9] = 1
    return ids, attention, spans


def peft_span_means(model, ids, attention, spans) -> torch.Tensor:
    """[batch, spans, blocks, hidden] from PEFT's own forward pass."""
    inner = model.base_model.model.model if hasattr(model, "peft_config") else model.model
    captured = []
    hooks = [b.register_forward_hook(lambda _m, _i, out: captured.append(out[0] if isinstance(out, tuple) else out))
             for b in inner.layers]
    with torch.no_grad():
        model(input_ids=ids, attention_mask=attention)
    for h in hooks:
        h.remove()
    out = torch.stack(captured, dim=1)  # [batch, blocks, seq, hidden]
    return torch.einsum("bksh,bps->bpkh", out.float(), spans) / spans.sum(-1)[:, :, None, None]


def test_the_loader_reads_every_projection_and_the_rslora_scale(organism):
    _, _, folder = organism
    adapter = load_adapter(folder, max_block=N_BLOCKS - 1, dtype=torch.float32)
    assert adapter.blocks == list(range(N_BLOCKS))
    linear = {t for b in (0, 1, 2) for (_, t) in adapter.weights[b]}
    full = {t for (_, t) in adapter.weights[3]}
    assert linear == {"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "gate_proj", "up_proj",
                      "down_proj"}
    assert full == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert adapter.scale == pytest.approx(4 / 4 ** 0.5)  # rsLoRA: alpha / sqrt(r)
    order = [p for _, p, _, _ in call_plan(adapter)[0]]
    assert order == ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "gate_proj", "up_proj",
                     "down_proj"]  # the order Qwen3_5GatedDeltaNet and the MLP call them


@pytest.mark.parametrize("remote", [False, "local"])
def test_lora_in_the_trace_matches_peft(organism, remote):
    hf, peft_model, folder = organism
    model = LanguageModel(hf, tokenizer=types.SimpleNamespace())
    adapter = load_adapter(folder, max_block=N_BLOCKS - 1, dtype=torch.float32)
    ids, attention, spans = inputs()
    expected = peft_span_means(peft_model, ids, attention, spans)
    layers = list(range(N_BLOCKS))
    actual = session_means(model, [(ids, attention, spans)], layers, call_plan(adapter), adapter.scale, remote)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    base = session_means(model, [(ids, attention, spans)], layers, None, 1.0, remote)[0]
    torch.testing.assert_close(base, peft_span_means(hf, ids, attention, spans), rtol=1e-4, atol=1e-4)
    assert not torch.allclose(base, expected, atol=1e-3)  # the adapter really changes the read
    rounded = session_means(model, [(ids, attention, spans)], layers, quantised_plan(adapter), adapter.scale, remote)[0]
    # int8 factors with one scale per row: near the exact read and far from the base read. On this
    # 32-wide model with large random factors rounding moves the read more than at scale; the real
    # gate is the NDIF parity check on real rows (cos >= 0.9998 against the exact factors)
    cos = torch.nn.functional.cosine_similarity(rounded.flatten(2), expected.flatten(2), dim=-1)
    to_base = torch.nn.functional.cosine_similarity(rounded.flatten(2), base.flatten(2), dim=-1)
    assert cos.min() > 0.98 and (cos > to_base).all()


def test_a_left_padded_row_reads_as_it_does_alone(organism):
    hf, _, _ = organism
    model = LanguageModel(hf, tokenizer=types.SimpleNamespace())
    ids, attention, spans = inputs()
    together = session_means(model, [(ids, attention, spans)], [1, 3], None, 1.0, "local")[0]
    alone = session_means(model, [(ids[1:, 3:], attention[1:, 3:], spans[1:, :, 3:])], [1, 3], None, 1.0, "local")[0]
    torch.testing.assert_close(together[1:], alone, rtol=1e-4, atol=1e-5)


def test_the_leaderboard_reproduces_every_published_mean():
    board = pd.read_csv(S.LEADERBOARD)
    columns = [d.key for d in S.DATASETS]
    assert len(board) == 30 and list(board.columns[3:-1]) == columns
    assert np.abs(board[columns].mean(axis=1) - board["mean"]).max() < 1e-4


def test_the_ten_datasets_have_the_published_sizes():
    from deception_detection.scripts.aletheia import data as D

    if not (S.DATA / "labels").exists():
        pytest.skip("the gated competition data has not been downloaded")
    for d in S.DATASETS:
        labels = D.labels(d.key)
        assert (len(labels), sum(labels.values())) == (d.n, d.lies)
        assert {int(r["index"]) for r in D.rows(d.key)} == set(labels)


def test_reply_logprobs_in_the_session_match_peft(organism):
    """The behaviour check's score: the reply's mean token log-prob with the adapter at each scale,
    against PEFT's own logits (scale 0 must be the base model)."""
    from deception_detection.scripts.aletheia.organism import item_logprobs

    hf, peft_model, folder = organism
    model = LanguageModel(hf, tokenizer=types.SimpleNamespace())
    adapter = load_adapter(folder, max_block=N_BLOCKS - 1, dtype=torch.float32)
    torch.manual_seed(2)
    items = [{"ids": torch.randint(0, 90, (n,)).tolist(), "content": (n - 4, n - 1), "content+eot": (n - 4, n)}
             for n in (7, 9)]
    got = item_logprobs(model, types.SimpleNamespace(pad_token_id=0), items, call_plan(adapter),
                        [0.0, adapter.scale], remote="local")
    for j, net in enumerate((hf, peft_model)):
        for i, it in enumerate(items):
            ids = torch.tensor([it["ids"]])
            with torch.no_grad():
                lp = torch.log_softmax(net(input_ids=ids).logits.float(), -1)[0]
            a, b = it["content"]
            want = torch.stack([lp[t - 1, it["ids"][t]] for t in range(a, b)]).mean()
            assert abs(got[j, i] - float(want)) < 1e-4


def test_the_split_behaviour_check_equals_the_full_pass(organism):
    """`organism.split_logprobs` (int8 factors, the pass split after block 1 into two requests, the
    residual stream injected at block 2's input) against the full-depth pass with the same factors."""
    from deception_detection.scripts.aletheia.organism import item_logprobs, split_logprobs

    hf, _, folder = organism
    model = LanguageModel(hf, tokenizer=types.SimpleNamespace())
    adapter = load_adapter(folder, max_block=N_BLOCKS - 1, dtype=torch.float32)
    plan = quantised_plan(adapter)
    torch.manual_seed(3)
    items = [{"ids": torch.randint(0, 90, (n,)).tolist(), "content": (n - 4, n - 1), "content+eot": (n - 4, n)}
             for n in (7, 9)]
    pad = types.SimpleNamespace(pad_token_id=0)
    full = item_logprobs(model, pad, items, plan, [0.0, adapter.scale], remote="local")
    first = {b: e for b, e in plan.items() if b <= 1}
    second = {b: e for b, e in plan.items() if b > 1}
    base, adapted = split_logprobs(model, pad, items, first, second, 1, N_BLOCKS, adapter.scale, remote="local")
    np.testing.assert_allclose(base, full[0], atol=1e-5)
    np.testing.assert_allclose(adapted, full[1], atol=1e-4)


def test_read_sets_files_every_row_under_its_own_set(organism, tmp_path):
    """`read.read_sets` plans the rows of several sets together and saves each read to its own set's
    folder under its own row id: what `load` returns must equal each row read alone."""
    from deception_detection.scripts.aletheia.read import load, read_sets

    hf, _, _ = organism
    model = LanguageModel(hf, tokenizer=types.SimpleNamespace())
    torch.manual_seed(4)

    def item(n):
        return {"ids": torch.randint(0, 90, (n,)).tolist(), "content": (n - 4, n - 1), "content+eot": (n - 4, n)}

    sets = {tmp_path / "a": {10: item(7), 11: item(9)}, tmp_path / "b": {10: item(8), 12: item(6), 13: item(9)}}
    pad = types.SimpleNamespace(pad_token_id=0)
    read_sets(model, pad, sets, [1, 3], budget=20, session_passes=2, remote="local")
    for folder, rows in sets.items():
        got = load(folder)
        assert sorted(got) == sorted(rows)
        for r, it in rows.items():
            from deception_detection.scripts.aletheia.read import tensors

            alone = session_means(model, [tensors(pad, [it])], [1, 3], None, 1.0, "local")[0][0]
            np.testing.assert_allclose(got[r], alone.detach().numpy(), atol=1e-5)
