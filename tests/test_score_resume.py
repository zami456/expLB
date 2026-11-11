"""Tests for resuming a partially-completed scoring sweep.

Scores are written after each dataset rather than once at the end, so a crash part way
through no longer discards the datasets that finished. The subtlety that comes with it:
a scores file now exists while still being incomplete, so "the file is there" can no
longer mean "the work is done".
"""

import json
from dataclasses import dataclass

import pytest

from deception_detection.experiment import Experiment
from deception_detection.scores import Scores
from deception_detection.types import Label


@dataclass
class FakeDataset:
    name: str
    n: int = 3

    def __len__(self) -> int:
        return self.n


def make_scores(n: int = 3) -> Scores:
    return Scores(scores=[[0.1, 0.2]] * n, labels=[Label.HONEST] * n)


def write_scores(path, names: list[str]) -> None:
    Experiment._write_scores_file(path, {name: make_scores() for name in names}, "test")


@pytest.fixture
def experiment() -> Experiment:
    # the resume logic touches no model or tokenizer, so skip the expensive __init__
    return Experiment.__new__(Experiment)


class TestScoresFileRoundTrip:
    def test_write_then_load(self, tmp_path, experiment):
        path = tmp_path / "scores.json"
        write_scores(path, ["a", "b"])
        loaded = experiment._load_scores_file(path)
        assert set(loaded) == {"a", "b"}
        assert len(loaded["a"]) == 3

    def test_write_is_atomic(self, tmp_path):
        path = tmp_path / "scores.json"
        write_scores(path, ["a"])
        assert list(tmp_path.glob("*.tmp")) == []

    def test_missing_file_loads_as_empty(self, tmp_path, experiment):
        assert experiment._load_scores_file(tmp_path / "nope.json") == {}

    def test_corrupt_file_loads_as_empty_rather_than_raising(self, tmp_path, experiment):
        path = tmp_path / "scores.json"
        path.write_text("{ truncated")
        assert experiment._load_scores_file(path) == {}


class TestScoresAreComplete:
    def test_absent_file_is_not_complete(self, tmp_path, experiment):
        assert not experiment._scores_are_complete(tmp_path / "scores.json", [FakeDataset("a")])

    def test_partial_file_is_not_complete(self, tmp_path, experiment):
        # the regression this guards: a crash after dataset 1 of 5 leaves a real file, and
        # treating it as done would silently skip the other four
        path = tmp_path / "scores.json"
        write_scores(path, ["a"])
        datasets = [FakeDataset(n) for n in ("a", "b", "c", "d", "e")]
        assert not experiment._scores_are_complete(path, datasets)

    def test_file_covering_every_dataset_is_complete(self, tmp_path, experiment):
        path = tmp_path / "scores.json"
        write_scores(path, ["a", "b", "c"])
        datasets = [FakeDataset(n) for n in ("a", "b", "c")]
        assert experiment._scores_are_complete(path, datasets)

    def test_empty_datasets_do_not_block_completeness(self, tmp_path, experiment):
        # compute_scores skips empty datasets, so requiring them would never be satisfied
        path = tmp_path / "scores.json"
        write_scores(path, ["a"])
        datasets = [FakeDataset("a"), FakeDataset("empty", n=0)]
        assert experiment._scores_are_complete(path, datasets)

    def test_extra_datasets_in_the_file_are_fine(self, tmp_path, experiment):
        # e.g. a dataset scored by an earlier config and since removed from eval_data
        path = tmp_path / "scores.json"
        write_scores(path, ["a", "b", "leftover"])
        assert experiment._scores_are_complete(path, [FakeDataset("a"), FakeDataset("b")])

    def test_survives_a_real_json_round_trip(self, tmp_path, experiment):
        path = tmp_path / "scores.json"
        write_scores(path, ["a"])
        assert set(json.loads(path.read_text())) == {"a"}
