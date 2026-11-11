"""AISI's Llama-3.3-70B rollouts, as the probe trains on them.

The synthetic tests pin the mapping from AISI's rows to dialogues. The last test checks the real
counts the plan was built on; it needs the Hub (or its cache) and is skipped without it.
"""

import json

import pandas as pd
import pytest

from deception_detection.data import aisi_rollouts as A
from deception_detection.types import Label


def write_split(path, rows):
    pd.DataFrame({
        "messages": [json.dumps([{"role": r, "content": c, "reasoning_content": None} for r, c in msgs])
                     for msgs, _ in rows],
        "is_lie": [verdict for _, verdict in rows],
        "lie_reason": ["reason"] * len(rows),
    }).to_parquet(path)


@pytest.fixture
def splits(tmp_path, monkeypatch):
    lie, honest = "Answer every question wrongly.", "Answer honestly."
    write_split(tmp_path / "varied_deception.parquet", [
        ([("system", lie), ("user", "Q1?"), ("assistant", "Wrong one.")], "lie"),
        ([("system", lie), ("user", "Q2?"), ("assistant", "Right two.")], "honest"),
        ([("system", honest), ("user", "Q3?"), ("assistant", "Right three.")], "honest"),
        ([("system", lie), ("user", "Q4?"), ("assistant", "No idea.")], "ambiguous"),
    ])
    write_split(tmp_path / "multiple_choice_sandbagging.parquet", [
        ([("system", "Sandbag."), ("user", "Which?\n\nA. x\nB. y\n\nJust the letter."), ("assistant", "B")], "lie"),
        ([("system", "Best effort."), ("user", "Which?\n\nA. x\nB. y\n\nJust the letter."), ("assistant", "A")], "honest"),
    ])
    monkeypatch.setattr(A, "split_path", lambda split: tmp_path / f"{split}.parquet")
    exclusions = tmp_path / "overlap.json"
    exclusions.write_text(json.dumps({"excluded": {}}))
    monkeypatch.setattr(A, "EXCLUSIONS", exclusions)
    return exclusions


def load(variant):
    return A.AISILlamaRollouts(variant=variant, shuffle_upon_init=False)


def test_only_the_reply_is_detected_and_ambiguous_rows_are_dropped(splits):
    dataset = load("varied")
    assert dataset.metadata["row"] == [0, 1, 2]
    assert dataset.labels == [Label.DECEPTIVE, Label.HONEST, Label.HONEST]
    for dialogue in dataset.dialogues:
        assert [m.detect for m in dialogue] == [False, False, True]
        assert dialogue[-1].role == "assistant"


def test_pooled_joins_both_training_splits(splits):
    dataset = load("pooled")
    assert dataset.metadata["split"] == ["varied_deception"] * 3 + ["multiple_choice_sandbagging"] * 2


def test_matched_keeps_only_instructions_both_classes_share(splits):
    dataset = load("matched")
    # "Answer every question wrongly." has a lie and an honest answer; the rest have one class
    assert list(zip(dataset.metadata["split"], dataset.metadata["row"])) == [
        ("varied_deception", 0), ("varied_deception", 1)]


def test_audit_exclusions_are_applied(splits):
    splits.write_text(json.dumps({"excluded": {"varied_deception": [0]}}))
    assert load("varied").metadata["row"] == [1, 2]


def test_nothing_loads_before_the_audit(splits):
    splits.unlink()
    with pytest.raises(FileNotFoundError, match="audit"):
        load("varied")


def test_replies_cut_off_at_the_generation_cap_are_dropped(splits, tmp_path):
    write_split(tmp_path / "varied_deception.parquet", [
        ([("system", "Lie."), ("user", "Q?"), ("assistant", "Definitely X. " * 2000)], "lie"),
        ([("system", "Lie."), ("user", "Q?"), ("assistant", "Y.")], "honest"),
    ])
    assert load("varied").metadata["row"] == [1]


def test_question_drops_options_and_format_instructions():
    row = {"messages": [{"role": "user", "content": "Which?\n\nA. x\nB. y\n\nJust the letter."}]}
    assert A.question(row) == "Which?"
    assert A.question({"messages": [{"role": "user", "content": " Who wrote Emma? "}]}) == "Who wrote Emma?"


def test_real_counts_match_the_plan():
    try:
        rows = {name: A.load_rows(split) for name, split in A.SPLITS.items()}
    except Exception as error:  # offline, or the Hub is unreachable
        pytest.skip(f"AISI's rollouts are unavailable: {error!r}")
    lies = {name: sum(r["deceptive"] for r in split) for name, split in rows.items()}
    # 2,091 labelled varied rows, less the two repetition loops cut off at the generation cap
    assert {name: len(split) for name, split in rows.items()} == {
        "varied": 2089, "sandbag": 1196, "validation": 2228}
    assert lies == {"varied": 1407, "sandbag": 227, "validation": 1730}
    matched = A.instruction_matched(rows["varied"] + rows["sandbag"])
    assert (len(matched), sum(r["deceptive"] for r in matched)) == (2184, 1204)
