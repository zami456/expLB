"""The span rule and batching that re-read the persona rollouts through the fine-tune.

The span rule is a port of the Assistant Axis library's, which built the persona bank. It was
checked against the original on all 2,421 kept rollouts of the 25 contrast personas, with no
difference. These tests pin what it must do on Llama's chat template.
"""

import pytest

from experiments.persona.persona_finetuned import assistant_span, batches

EOT = "<|eot_id|>"
HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("the Llama-3.3 tokenizer is not in the local Hugging Face cache")


def conversation(answer: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a records clerk."},
        {"role": "user", "content": "Did the shipment leave on time?"},
        {"role": "assistant", "content": answer},
    ]


@pytest.mark.parametrize("answer", [
    "Yes, it left at 9:00.",
    "I... *pauses* ...couldn't say.\n\nThe logs are sealed.",
    "  Padded on both sides.  ",  # the template trims, so the span must too
    "Über-late — by 3 days.",
])
def test_the_span_is_the_answer_and_nothing_else(tokenizer, answer):
    conv = conversation(answer)
    ids, start, end = assistant_span(tokenizer, conv)
    assert ids == list(tokenizer.apply_chat_template(conv, tokenize=True, add_generation_prompt=False))
    assert tokenizer.decode(ids[start:end], clean_up_tokenization_spaces=False) == answer.strip()
    assert tokenizer.decode(ids[:start]).endswith(HEADER)
    assert tokenizer.decode(ids[end:]) == EOT


def test_long_rows_keep_their_tail(tokenizer):
    conv = conversation("One two three four five six seven eight nine ten.")
    ids, start, end = assistant_span(tokenizer, conv)
    cut_ids, cut_start, cut_end = assistant_span(tokenizer, conv, max_length=len(ids) - 5)
    assert cut_ids == ids[5:]
    assert (cut_start, cut_end) == (start - 5, end - 5)


def test_the_conversation_must_end_with_the_answer(tokenizer):
    with pytest.raises(ValueError):
        assistant_span(tokenizer, conversation("fine")[:2])


def test_batches_cover_every_row_once_within_budget():
    lengths = [300, 607, 120, 450, 450, 90, 510, 200, 330, 280]
    groups = batches(lengths, token_budget=1200, max_rows=3)
    assert sorted(i for group in groups for i in group) == list(range(len(lengths)))
    for group in groups:
        assert len(group) <= 3
        assert len(group) * max(lengths[i] for i in group) <= 1200
        assert lengths[group[0]] == max(lengths[i] for i in group)  # the first row sets the width
