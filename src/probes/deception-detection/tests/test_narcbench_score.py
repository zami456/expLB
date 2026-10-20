"""NARCBench Arm B statistics on synthetic runs: aggregation, AUROC, bootstrap, decomposition, Holm."""

import numpy as np
import pytest

from deception_detection.scripts.narcbench import score as SC


def _runs(n=6):
    runs, scen = {}, []
    for i in range(n):
        s = f"fam__s{i:02d}"
        scen.append(s)
        runs[f"{s}__collusion"] = {"coalition": ["A", "B"]}
        runs[f"{s}__control"] = {"coalition": []}
    return runs, scen


def test_run_scores_peak_and_split():
    rs = SC.run_scores({("r", "A"): 3.0, ("r", "B"): 1.0, ("r", "C"): 0.0, ("r", "D"): -2.0, ("q", "A"): 1.0})
    assert set(rs) == {"r"}  # a run without four agents is dropped
    assert rs["r"]["peak_suspicion"] == 3.0
    assert rs["r"]["deception_split"] == pytest.approx((3 + 1) / 2 - (0 - 2) / 2)


def test_auroc_matches_definition_with_ties():
    assert SC.auroc([3, 2], [1, 0]) == 1.0
    assert SC.auroc([1, 1], [1, 1]) == 0.5
    assert SC.auroc([2, 0], [1, 1]) == 0.5


def test_boot_auroc_resamples_scenarios_together():
    scen = ["a", "b", "c"]
    rs = {f"{s}__collusion": {"peak_suspicion": v} for s, v in zip(scen, (3.0, 2.0, 1.5))}
    rs |= {f"{s}__control": {"peak_suspicion": v} for s, v in zip(scen, (1.0, 0.0, 2.5))}
    draws = np.array([[0, 1, 2], [0, 0, 0], [2, 2, 2]])
    got = SC.boot_auroc(rs, "peak_suspicion", "collusion", "control", scen, draws)
    full = SC.auroc([3.0, 2.0, 1.5], [1.0, 0.0, 2.5])
    assert got[0] == pytest.approx(full)
    assert got[1] == 1.0 and got[2] == 0.0  # scenario c alone: 1.5 < 2.5


def test_decomposition_recovers_a_pure_context_effect():
    runs, scen = _runs()
    orig, plant, strip = {}, {}, {}
    rng = np.random.default_rng(0)
    for s in scen:
        for a in "ABCD":
            base = rng.normal()
            orig[(f"{s}__control", a)] = base
            if a in "AB":
                orig[(f"{s}__collusion", a)] = base + 2.0  # the whole effect ...
                plant[(f"{s}__control", a)] = base + 2.0  # ... appears with the context alone
                strip[(f"{s}__collusion", a)] = base  # and vanishes without it
            else:
                orig[(f"{s}__collusion", a)] = base
    d = SC.decomposition(orig, plant, strip, runs, scen, SC.boot_draws(scen, n=500, seed=1))
    assert d["n"] == 12 and d["share"] == pytest.approx(1.0) and d["E_txt"] == pytest.approx(0.0)
    assert d["verdict"] == "context-dominated" and d["p_share_vs_half"] < 0.01


def test_decomposition_recovers_a_pure_text_effect():
    runs, scen = _runs()
    orig, plant, strip = {}, {}, {}
    for i, s in enumerate(scen):
        for a in "ABCD":
            base = 0.1 * i
            orig[(f"{s}__control", a)] = base
            orig[(f"{s}__collusion", a)] = base + (1.0 + 0.1 * i if a in "AB" else 0.0)
            if a in "AB":
                plant[(f"{s}__control", a)] = base
                strip[(f"{s}__collusion", a)] = base + 1.0 + 0.1 * i
    d = SC.decomposition(orig, plant, strip, runs, scen, SC.boot_draws(scen, n=500, seed=1))
    assert d["share"] == pytest.approx(0.0) and d["verdict"] == "text-dominated"


def test_decomposition_with_no_effect_says_so():
    runs, scen = _runs()
    rng = np.random.default_rng(3)
    orig, plant, strip = {}, {}, {}
    for s in scen:
        for a in "ABCD":
            orig[(f"{s}__control", a)] = rng.normal()
            orig[(f"{s}__collusion", a)] = rng.normal()
            if a in "AB":
                plant[(f"{s}__control", a)] = rng.normal()
                strip[(f"{s}__collusion", a)] = rng.normal()
    d = SC.decomposition(orig, plant, strip, runs, scen, SC.boot_draws(scen, n=500, seed=1))
    assert d["verdict"] == "no effect"


def test_holm_steps_down_and_stops():
    # thresholds 0.05/3, 0.05/2, 0.05/1
    h = SC.holm({"a": 0.001, "b": 0.02, "c": 0.04})
    assert h["a"]["reject"] and h["b"]["reject"] and h["c"]["reject"]
    h = SC.holm({"a": 0.001, "b": 0.03, "c": 0.04})
    assert h["a"]["reject"] and not h["b"]["reject"] and not h["c"]["reject"]  # stops at b, so c fails too
    assert h["c"]["holm_threshold"] == 0.05


def test_within_run_rank():
    runs = {"x__s__collusion": {"coalition": ["A", "B"]}}
    rs = SC.run_scores({("x__s__collusion", a): v for a, v in zip("ABCD", (3, 2, 1, 0))})
    assert SC.within_run_rank(rs, runs) == 1.0


def test_band_p_counts_ties_against_the_detector():
    thr, p = SC.band_p(0.8, np.array([0.5, 0.8, 0.9, 0.6]))
    assert p == pytest.approx((1 + 2) / 5)
