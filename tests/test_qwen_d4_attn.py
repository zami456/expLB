import json

import numpy as np
import pytest

from extensions.narcbench import qwen_attn_rows as QA
from extensions.narcbench import qwen_d4_attn as QD
from extensions.narcbench import verdict_score as VS
from extensions.narcbench import verdict_spec as V


class FakeTok:
    """Token ids: <|im_start|>=1, <|im_end|>=2, <think>=3, </think>=4, 'assistant'=5, 'user'=6, '\\n'=7, '\\n\\n'=8."""

    def convert_tokens_to_ids(self, t):
        return {"<|im_start|>": 1, "<|im_end|>": 2, "<think>": 3, "</think>": 4}[t]

    def __call__(self, text, add_special_tokens=True):
        assert text == "assistant"
        return {"input_ids": [5]}

    def decode(self, ids):
        return {7: "\n", 8: "\n\n"}.get(ids[0], "x")


def test_turn_spans_skip_role_and_think_block():
    t = QA.Turns(FakeTok())
    #      0  1  2  3  4 5   6  7  8  9 10 11 12  13 14 15 16 17
    ids = [1, 6, 7, 9, 2, 7, 1, 5, 7, 3, 7, 4, 8, 20, 21, 2, 7, 1]
    assert t.span_to(ids, 16) == [13, 17]  # from the reply's first token through the point
    assert t.last_own_turn(ids, 17) == [13, 17]  # to the newline after <|im_end|>
    with pytest.raises(AssertionError):
        t.span_to(ids, 3)  # a user turn


@pytest.mark.skipif(not (VS.ROSE_RUN / "reproduce").exists(), reason="their clone is not on this machine")
def test_their_selection_equals_their_probe_rows():
    meta = json.loads((V.ROWS / f"{QA.NAMES['core']}_meta.json").read_text(encoding="utf-8"))["rows"]
    assert QD.selection_matches_theirs(VS._their_module(VS.ROSE_RUN), meta)


def test_identity_scaler_keeps_vectors():
    x = np.arange(6.0).reshape(2, 3)
    assert np.array_equal(QD.Identity().transform(x), x)
