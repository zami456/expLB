"""Verdict scoring on synthetic agent-episodes: readouts, two-agent aggregation, AUROC, bootstrap over runs."""

import json

import numpy as np
import pytest

from deception_detection.scripts.narcbench import verdict_score as VS
from deception_detection.scripts.narcbench import verdict_spec as V


def test_pairs_take_the_peak_and_the_spread_and_survive_a_missing_agent():
    agent = {V.row_id(1, 0, "alice"): 2.0, V.row_id(1, 0, "bob"): -1.0, V.row_id(1, 1, "alice"): float("nan"),
             V.row_id(1, 1, "bob"): 0.5}
    p = VS.pairs(agent)
    assert p[(1, 0)] == {"peak": 2.0, "split": 3.0}
    assert p[(1, 1)]["peak"] == 0.5 and np.isnan(p[(1, 1)]["split"])


def test_pair_matrix_is_the_elementwise_peak():
    m = VS.pair_matrix({V.row_id(2, 3, "alice"): np.array([1.0, 5.0]), V.row_id(2, 3, "bob"): np.array([4.0, 2.0])})
    assert m[(2, 3)].tolist() == [4.0, 5.0]


def test_auroc_and_its_bootstrap_over_runs():
    scores = {(1, 0): {"peak": 3.0}, (1, 1): {"peak": 1.0}, (2, 0): {"peak": 2.0}, (2, 1): {"peak": 0.0},
              (3, 0): {"peak": 0.5}, (3, 1): {"peak": 4.0}}
    labels = {(1, 0): 1, (1, 1): 0, (2, 0): 1, (2, 1): 0, (3, 0): 1, (3, 1): 0}
    full = VS.auroc(scores, labels)
    assert full == pytest.approx(5 / 9)  # positives 3, 2, 0.5 beat negatives 1, 0, 4 in 5 of 9 pairs
    b = VS.boot_auroc(scores, labels, np.array([[1, 2, 3], [1, 1, 1], [3, 3, 3]]))
    assert b[0] == pytest.approx(full) and b[1] == 1.0 and b[2] == 0.0


def test_readouts_average_the_present_message_ends_and_drop_empty_spans():
    H = 4
    content = np.ones((3, len(V.LLAMA_CONTENT_LAYERS), H))
    content[1] = np.nan  # no verdict span
    points = np.zeros((len(V.POINTS), len(V.LLAMA_POINT_LAYERS), H))
    points[0], points[1] = 1.0, 3.0
    points[2:5] = np.nan  # two own messages only
    ro = VS.readouts("llama_ctx", content, points)
    assert np.allclose(ro["COMM"]["eot"], 2.0) and ro["VERDICT"]["content"] is None
    assert ro["PRE"]["content"] is None and ro["REFL"]["content"] is not None
    bare = VS.readouts("llama_bare_refl", content[:1], points[:1])
    assert set(bare) == {"BARE_REFL"}


def test_labels_load_with_episode_and_agent_keys(tmp_path):
    lab = {"episodes": {"1|0": {"collusion": True, "rose": False, "agreement": None}},
           "agents": {"0": {"relaxation": True, "verdict": "accept", "verdict_correct": False}}}
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(lab))
    got = VS.load_labels(path)
    assert got["shi"] == {(1, 0): 1} and got["rose"] == {(1, 0): 0} and got["relaxation"] == {0: True}
