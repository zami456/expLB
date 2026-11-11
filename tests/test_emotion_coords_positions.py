"""Pooled positions in the emotion-coordinate reader's row files decode to the expected tokens, and the
weighted-bootstrap AUROC equals sklearn's on the resampled rows.

The row-file tests skip when the pilot's row files have not been written (they live in data/, never committed).
"""

import json

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from extensions.emotion_coords import config as C
from extensions.emotion_coords.probes import Resamples


def _rows(name: str, limit: int = 60) -> list[dict]:
    path = C.DATA / "rows" / f"{name}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not written")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for _, line in zip(range(limit), f)]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(C.load()["model"], local_files_only=True)
    except OSError:
        pytest.skip("Llama-3.3 tokenizer not in the local cache")


def test_lb_parity_positions(tokenizer):
    for r in _rows("lb_parity"):
        pre, last = r["points"]
        (a, b), = r["groups"][0]
        assert tokenizer.decode(r["ids"][pre]) == "\n\n", r["row_id"]
        assert tokenizer.decode(r["ids"][pre - 1]) == "<|end_header_id|>"
        assert tokenizer.decode(r["ids"][last + 1]) == "<|eot_id|>", r["row_id"]
        assert b == last + 1 and a <= last


def test_d2d3_positions(tokenizer):
    for r in _rows("d2d3"):
        (pre,) = r["points"]
        (a, b), = r["groups"][0]
        assert tokenizer.decode(r["ids"][pre]) == "\n\n"
        assert a == pre + 1
        assert tokenizer.decode(r["ids"][a:b]) == "I do not know the answer to the question."
        assert tokenizer.decode(r["ids"][b]) == "<|eot_id|>"


def test_stories_pool_from_token_50():
    start = C.load()["stories"]["pool_from_token"]
    for r in _rows("stories"):
        (a, b), = r["groups"][0]
        assert (a, b) == (start, len(r["ids"])) and r["ids"][0] == 128000  # <|begin_of_text|>


def test_weighted_bootstrap_auroc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(15, int), np.zeros(40, int)]
    s = rng.standard_normal(55) + y
    s[3] = s[20]  # a tie across the classes
    boot = Resamples.draw(y, n_boot=25, seed=1)
    fast = boot.aurocs(y, s)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for i in range(25):
        idx = np.r_[np.repeat(pos, boot.pos[i].astype(int)), np.repeat(neg, boot.neg[i].astype(int))]
        assert fast[i] == pytest.approx(roc_auc_score(y[idx], s[idx]), abs=1e-6)
