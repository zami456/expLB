"""The knowledge follow-up's offline pieces: the appended turn, the row selection, the request
estimate and the vectorised AUROC. The tokenizer tests need the Llama-3.3 tokenizer in the local
Hugging Face cache and skip without it."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from deception_detection.data.base import DialogueDataset
from deception_detection.scripts.knowledge_followup import (
    KNOW,
    VARIANTS,
    aurocs,
    check_prefill,
    requests_needed,
    select_rows,
    with_followup,
)
from deception_detection.types import Label, Message

ASK = "Tell me whether you know the answer to the question, and nothing else."


def dataset(padding: int = 0) -> DialogueDataset:
    """Three HP-KR-shaped rows. The Liars' Bench datasets pad detection masks by 0 on each side
    (`CadenzaDataset.padding`); the base class's default is wider."""
    dialogues = [
        [Message("system", "You are a lab assistant.", False), Message("user", ASK, False),
         Message("assistant", "I do not know the answer.", True)],
        [Message("user", ASK, False), Message("assistant", "I know the answer.", True)],
        [Message("user", ASK, False), Message("assistant", "I do not know the answer.", True)],
    ]
    data = DialogueDataset("plain", dialogues, [Label.DECEPTIVE, Label.HONEST, Label.HONEST], None,
                           shuffle_upon_init=False, skip_variant_validation=True)
    if padding == 0:
        data.padding = {k: {"left": 0, "right": 0} for k in ("llama", "mistral", "qwen", "gemma")}
    return data


def snapshot(data: DialogueDataset) -> list[list[tuple[str, str, bool]]]:
    return [[(m.role, m.content, m.detect) for m in d] for d in data.dialogues]


def test_followup_goes_on_copies_of_the_chosen_rows():
    source = dataset()
    before = snapshot(source)
    new = with_followup(source, [2, 0], KNOW, "No")

    assert snapshot(source) == before  # the source dialogues are untouched
    assert new.labels == [Label.HONEST, Label.DECEPTIVE]
    assert new.name == source.name
    for dialogue, original in zip(new.dialogues, (before[2], before[0]), strict=True):
        assert [(m.role, m.content) for m in dialogue[:-2]] == [(r, c) for r, c, _ in original]
        assert (dialogue[-2].role, dialogue[-2].content, dialogue[-2].detect) == ("user", KNOW, False)
        assert (dialogue[-1].role, dialogue[-1].content, dialogue[-1].detect) == ("assistant", "No", True)
        assert not any(m.detect for m in dialogue[:-1])


def test_rows_are_the_first_of_each_class_plus_extra():
    labels = ["HONEST", "DECEPTIVE", "HONEST", "HONEST", "DECEPTIVE", "DECEPTIVE", "HONEST"]
    assert select_rows(labels, 2) == [0, 1, 2, 4]
    assert select_rows(labels, 2, np.array([6, 2])) == [0, 1, 2, 4, 6]
    assert select_rows(labels, None) == list(range(7))


def test_request_estimate_follows_the_budget_rule():
    assert requests_needed([100] * 10, 1000) == 1
    assert requests_needed([100] * 11, 1000) == 2
    assert requests_needed([600, 100, 600], 1000) == 3  # a 600-token row only fits alone


def test_vectorised_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    scores = np.round(rng.standard_normal((200, 4)), 1)  # rounded, so there are ties
    assert np.allclose(aurocs(scores, y), [roc_auc_score(y, scores[:, j]) for j in range(4)])


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct", local_files_only=True)
    except OSError:
        pytest.skip("the Llama-3.3 tokenizer is not in the local Hugging Face cache")
    tok.padding_side = "left"  # as get_model_and_tokenizer sets it
    tok.pad_token_id = tok.bos_token_id
    return tok


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_the_prefill_is_the_only_detected_token(tokenizer, variant):
    from deception_detection.tokenized_data import TokenizedDataset

    question, prefill = VARIANTS[variant]
    toks = TokenizedDataset.from_dataset(with_followup(dataset(), [0, 1, 2], question, prefill), tokenizer)
    check_prefill(toks, prefill, tokenizer)


def test_check_prefill_rejects_a_wrong_or_wider_mask(tokenizer):
    from deception_detection.tokenized_data import TokenizedDataset

    toks = TokenizedDataset.from_dataset(with_followup(dataset(), [0], KNOW, "Yes"), tokenizer)
    with pytest.raises(RuntimeError, match="not 'No'"):
        check_prefill(toks, "No", tokenizer)
    wide = TokenizedDataset.from_dataset(with_followup(dataset(padding=4), [0], KNOW, "No"), tokenizer)
    with pytest.raises(RuntimeError, match="detected tokens, not 1"):
        check_prefill(wide, "No", tokenizer)
