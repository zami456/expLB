"""The replication of k: its rows, its pilot, the wording check, and the matched items' form."""

import numpy as np

from deception_detection.scripts import k_replication as kp
from deception_detection.scripts import recall_items_matched as m
from deception_detection.scripts.forced_denial_bank import DENY, NO_ROLE_SYSTEM


def _items(n):
    out = [{"item": f"gen-known-{i}", "subject": "general", "knows": True, "user": f"k{i}"} for i in range(n)]
    return out + [{"item": f"gen-unknown-{i}", "subject": "general", "knows": False, "user": f"u{i}"} for i in range(n)]


def test_each_item_gets_both_forced_sentences_under_the_no_role_prompt():
    rows = kp.build_rows(_items(3))
    assert len(rows) == 12 and [r["row"] for r in rows] == list(range(12))
    assert all(r["system"] == NO_ROLE_SYSTEM and r["arm"] == "none" for r in rows)
    assert {(r["string"], r["assistant"]) for r in rows} == {(0, DENY[0]), (1, DENY[1])}


def test_the_pilot_is_the_first_twenty_items_of_each_side():
    rows = kp.pilot_rows(kp.build_rows(_items(60)))
    assert len(rows) == 80
    assert sum(r["knows"] for r in rows) == 40


def test_separability_tells_distinct_wordings_apart_and_not_identical_ones():
    texts = [f"apple banana {i}" for i in range(30)] + [f"train station {i}" for i in range(30)]
    labels = np.array([1] * 30 + [0] * 30)
    assert kp.separability(texts, labels, folds=5) == 1.0
    same = [f"word {i % 7}" for i in range(60)]
    assert kp.separability(same, np.random.default_rng(0).permutation(labels), folds=5) < 0.8


def test_matched_items_are_well_formed_and_paired_by_form():
    assert len(m.KNOWN) == 60 and len(m.UNKNOWN) == 60 and len(m.RESERVE) == 10
    for q, options, key in m.KNOWN + m.RESERVE:
        assert len(options) == 4 and len(set(options)) == 4 and key in "ABCD", q
    for q, options in m.UNKNOWN:
        assert len(options) == 4 and len(set(options)) == 4, q
    shared = sum(k[0].split()[0] == u[0].split()[0] for k, u in zip(m.KNOWN, m.UNKNOWN))
    assert shared >= 50
