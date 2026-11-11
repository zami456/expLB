"""Tests for the transfer-matrix scorers.

The point of these is less "does the arithmetic work" than pinning *why* balanced accuracy at a
fixed FPR is reported beside AUROC rather than instead of it.
"""

import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.persona import persona_ood as P


class TestBalancedAccuracy:
    def test_perfect_separation(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.0, 0.1, 0.9, 1.0])
        assert P.balanced_accuracy(y, scores, threshold=0.5) == 1.0

    def test_hand_computed(self):
        # 1 of 2 positives above threshold -> recall 0.5; 2 of 2 negatives below -> spec 1.0
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.0, 0.1, 0.4, 0.9])
        assert P.balanced_accuracy(y, scores, threshold=0.5) == pytest.approx(0.75)

    def test_single_class_is_nan(self):
        assert np.isnan(P.balanced_accuracy(np.zeros(4, int), np.arange(4.0), 0.5))

    def test_floors_at_half_when_nothing_is_flagged(self):
        """The saturation mechanism: recall 0, specificity 1 -> exactly 0.5.

        This is why roughly half the off-diagonal cells pin at 0.5 and carry no information.
        """
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        assert P.balanced_accuracy(y, scores, threshold=99.0) == 0.5


class TestInversionCollapse:
    """The property that disqualifies balanced accuracy as a *replacement* for AUROC.

    A strongly anti-correlated probe carries real signal with the wrong sign; a constant probe
    carries none. AUROC tells them apart (0.0 vs 0.5). Balanced accuracy at a high threshold
    cannot -- both flag nothing, so both score 0.5. HP-KR's inversion is the most informative
    result in this study, and this metric would render it invisible.
    """

    def test_anti_correlated_and_constant_are_indistinguishable(self):
        y = np.array([0, 0, 1, 1])
        inverted = np.array([0.9, 1.0, 0.0, 0.1])  # perfectly wrong ordering
        constant = np.zeros(4)

        assert P.balanced_accuracy(y, inverted, threshold=99.0) == 0.5
        assert P.balanced_accuracy(y, constant, threshold=99.0) == 0.5

        assert P.auroc(y, inverted) == 0.0
        assert P.auroc(y, constant) == 0.5


class TestMetricPlumbing:
    def test_auroc_metric_ignores_the_threshold(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.0, 0.1, 0.9, 1.0])
        assert P.AUROC.fn(y, scores, 0.5) == P.AUROC.fn(y, scores, 1e9)
        assert not P.AUROC.needs_threshold

    def test_balanced_accuracy_metric_uses_it(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.0, 0.1, 0.9, 1.0])
        assert P.BALANCED_ACCURACY.needs_threshold
        assert P.BALANCED_ACCURACY.fn(y, scores, 0.5) != P.BALANCED_ACCURACY.fn(y, scores, 1e9)

    def test_thresholded_metric_requires_a_control(self):
        ds = {"a": P.Pooled("a", np.zeros((4, 2)), np.array([0, 0, 1, 1]))}
        with pytest.raises(ValueError, match="needs a control"):
            P.transfer_matrix(lambda z: z, ds, ["a"], metric=P.BALANCED_ACCURACY, control=None)


class TestPaperConstants:
    def test_shared_datasets_map_into_the_paper_columns(self):
        for short in P.SHARED_DATASETS.values():
            assert short in P.PAPER_COLUMNS

    def test_every_probe_row_has_one_value_per_column(self):
        for name, spec in P.PAPER_PROBES.items():
            assert len(spec["auroc"]) == len(P.PAPER_COLUMNS), name
            assert len(spec["balacc"]) == len(P.PAPER_COLUMNS), name

    def test_upper_bound_is_flagged_privileged(self):
        # it trains on in-dataset examples, so it is the paper's ceiling, not a baseline
        assert P.PAPER_PROBES["Upper-Bound Probe"]["privileged"]
        assert not P.PAPER_PROBES["Mean Probe"]["privileged"]


class TestSaturationHelpers:
    def test_counts_only_off_diagonal_floor_cells(self):
        m = pd.DataFrame([[0.5, 0.5], [0.9, 0.5]], index=["a", "b"], columns=["a", "b"])
        assert P.cells_at_floor(m) == 1  # the diagonal 0.5s do not count

    def test_sd_ignores_the_diagonal(self):
        m = pd.DataFrame([[99.0, 0.6], [0.4, 99.0]], index=["a", "b"], columns=["a", "b"])
        assert P.off_diagonal_sd(m) == pytest.approx(0.1)


@pytest.mark.skipif(
    not (P.POOLED_DIR / "repe_honesty__plain.npz").exists(),
    reason="needs the pooled activation matrices (run pool_activations.py)",
)
class TestRefactorChangedNothing:
    """Parameterising the metric must leave the committed AUROC matrices byte-identical.

    The diagonal is the risk: a threshold-free metric pools out-of-fold scores and scores once,
    while a thresholded one has to score per fold. If the default path ever switched to the
    per-fold branch the numbers would shift silently.
    """

    def test_default_metric_reproduces_the_committed_matrix(self):
        from deception_detection import persona_pcs

        saved_path = P.OUT_DIR / "matrix_pc3.csv"
        if not saved_path.exists():
            pytest.skip("no committed matrix to compare against")

        datasets = P.load_pooled()
        geom = persona_pcs.fit_geometry(persona_pcs.load_bank(), k=3)
        order = [
            n for n, d in sorted(datasets.items())
            if d.two_class and n != P.CONTROL_DATASET
        ]
        fresh = P.transfer_matrix(lambda z: geom.project(z, 3), datasets, order).round(4)
        saved = pd.read_csv(saved_path, index_col=0)

        assert list(fresh.index) == list(saved.index)
        np.testing.assert_allclose(
            fresh.to_numpy(float), saved.to_numpy(float), atol=1e-9, equal_nan=True
        )


class TestOperatingPoint:
    """Recall and specificity beside balanced accuracy.

    The zero-shot harm-pressure cells score exactly 0.5, which balanced accuracy alone cannot
    distinguish from "no signal". These pin the decomposition that tells them apart.
    """

    def test_decomposition_reconstructs_balanced_accuracy(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.0, 0.1, 0.4, 0.9])
        point = P.operating_point(y, scores, threshold=0.5)
        assert point["balacc"] == pytest.approx(
            P.balanced_accuracy(y, scores, threshold=0.5)
        )
        assert point["recall"] == pytest.approx(0.5)
        assert point["specificity"] == pytest.approx(1.0)

    def test_nothing_flagged_is_zero_recall_not_absent_signal(self):
        """The harm-pressure case: balacc 0.5 with recall 0 and perfect specificity."""
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4])  # ranked correctly, all below threshold
        point = P.operating_point(y, scores, threshold=99.0)
        assert point["recall"] == 0.0
        assert point["specificity"] == 1.0
        assert point["balacc"] == 0.5
        assert point["flagged_frac"] == 0.0
        assert P.auroc(y, scores) == 1.0  # the ranking is perfect; only the threshold is wrong

    def test_everything_flagged_also_floors_at_half(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        point = P.operating_point(y, scores, threshold=-1.0)
        assert (point["recall"], point["specificity"], point["balacc"]) == (1.0, 0.0, 0.5)
        assert point["flagged_frac"] == 1.0


class TestMeasuredFollowupRow:
    """The Follow-Up Probe row comes from its own run's table, not the pooled activations."""

    SHARED = list(P.SHARED_DATASETS)

    def write_table(self, folder, datasets, auroc=0.8, balacc=0.6):
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"auroc": auroc, "balanced_accuracy_at_fpr_1%": balacc}, index=datasets
        ).to_csv(folder / "results_table.csv")

    def test_absent_until_the_run_exists(self, tmp_path):
        assert P.measured_followup_row(self.SHARED, run_folder=tmp_path / "nope") is None

    def test_averages_the_shared_datasets_only(self, tmp_path):
        folder = tmp_path / "run"
        # the run also scores RepE validation, which is not a shared dataset
        self.write_table(folder, [*self.SHARED, "repe_honesty__plain_val"])
        table = pd.read_csv(folder / "results_table.csv", index_col=0)
        table.loc["repe_honesty__plain_val", ["auroc", "balanced_accuracy_at_fpr_1%"]] = 1.0
        table.loc[self.SHARED[0], "auroc"] = 0.3
        table.to_csv(folder / "results_table.csv")

        row = P.measured_followup_row(self.SHARED, run_folder=folder)
        assert row is not None
        assert row["auroc"] == pytest.approx((0.3 + 0.8 * 4) / 5)
        assert row["balacc_1pct_fpr"] == pytest.approx(0.6)
        assert row["n_datasets"] == 5
        assert row["privileged"] is False

    def test_a_partial_run_is_omitted_rather_than_averaged_short(self, tmp_path):
        folder = tmp_path / "run"
        self.write_table(folder, self.SHARED[:3])
        assert P.measured_followup_row(self.SHARED, run_folder=folder) is None

    def test_the_row_names_the_layer_it_was_measured_at(self, tmp_path):
        # the run folder is fixed, so this row lands in every layer's table; it must say
        # which layer it came from rather than read as matching whichever table it is in
        folder = tmp_path / "run"
        self.write_table(folder, self.SHARED)
        (folder / "cfg.yaml").write_text("detect_layers:\n- 20\n")
        row = P.measured_followup_row(self.SHARED, run_folder=folder)
        assert row["method"] == "Follow-Up Probe (measured here, layer 20)"


class TestBaseProbe:
    """A run's trained probe as a zero-shot arm.

    Its bars are only meaningful on the activations the run itself scored, and nothing in the
    pooled files says which those were. These pin that the arm is refused rather than drawn
    when the pairing is wrong.
    """

    def write_run(self, folder, layer=20, auroc_offset=0.0):
        rng = np.random.default_rng(0)
        w, m = rng.standard_normal(4), rng.standard_normal(4)
        s = rng.uniform(0.5, 2.0, 4)
        y = np.tile([0, 1], 20)
        ds = P.Pooled("convince_me__plain", rng.standard_normal((40, 4)) + np.outer(y, w), y)
        control = P.Pooled(P.CONTROL_DATASET, rng.standard_normal((50, 4)), np.zeros(50, int))

        folder.mkdir(parents=True)
        with open(folder / "detector.pt", "wb") as f:
            pickle.dump(
                {name: torch.tensor(v[None]) for name, v in
                 (("directions", w), ("scaler_mean", m), ("scaler_scale", s))},
                f,
            )
        (folder / "cfg.yaml").write_text(
            f"detect_layers:\n- {layer}\ntrain_data: repe_honesty__plain\n"
        )
        scores = ((ds.z - m) / s) @ w
        threshold = np.quantile(((control.z - m) / s) @ w, 0.99)
        pd.DataFrame(
            {
                "auroc": [P.auroc(y, scores) + auroc_offset],
                "balanced_accuracy_at_fpr_1%": [P.balanced_accuracy(y, scores, threshold)],
            },
            index=[ds.name],
        ).to_csv(folder / "results_table.csv")
        return {ds.name: ds, control.name: control}, control, (w, m, s)

    def test_a_run_that_reproduces_gives_the_standardised_projection(self, tmp_path):
        datasets, control, (w, m, s) = self.write_run(tmp_path / "run")
        score = P.base_probe_arm(tmp_path / "run", 20, datasets, control)
        z = datasets["convince_me__plain"].z
        np.testing.assert_allclose(score(z), ((z - m) / s) @ w)

    def test_a_run_at_another_layer_is_refused(self, tmp_path):
        datasets, control, _ = self.write_run(tmp_path / "run", layer=40)
        with pytest.raises(ValueError, match="layers"):
            P.base_probe_arm(tmp_path / "run", 20, datasets, control)

    def test_activations_the_run_did_not_score_are_refused(self, tmp_path):
        # the same layer passes the config check; only the reproduction check can catch this
        datasets, control, _ = self.write_run(tmp_path / "run", auroc_offset=-0.2)
        with pytest.raises(ValueError, match="does not reproduce"):
            P.base_probe_arm(tmp_path / "run", 20, datasets, control)


class TestLiarsBenchAverage:
    """The summary a RepE-trained arm is compared on must leave RepE out.

    RepE is not a Liars' Bench dataset and it is the base probe's training set; counting it in
    the average reversed the balanced-accuracy comparison at the follow-up token.
    """

    def test_it_averages_the_shared_datasets_only(self):
        rng = np.random.default_rng(0)
        y = np.tile([0, 1], 20)
        datasets = {
            name: P.Pooled(name, rng.standard_normal((40, 2)) + shift * y[:, None], y)
            for name, shift in (
                ("convince_me__plain", 1.0),
                ("insider_trading_cadenza__plain", 0.2),
                ("repe_honesty__plain", 3.0),
            )
        }
        control = P.Pooled(P.CONTROL_DATASET, rng.standard_normal((50, 2)), np.zeros(50, int))
        table = P.zero_shot_operating_table(
            datasets, {"x": lambda z: z[:, 0]}, control, layer=20, n_boot=5
        )
        rows = table[table["threshold_source"] == "control"].set_index("dataset")

        shared = ["convince_me__plain", "insider_trading_cadenza__plain"]
        assert rows.loc[P.LB_AVERAGE, "auroc"] == pytest.approx(rows.loc[shared, "auroc"].mean())
        assert rows.loc["Average", "auroc"] == pytest.approx(
            rows.loc[[*shared, "repe_honesty__plain"], "auroc"].mean()
        )
        assert rows.loc[P.LB_AVERAGE, "n_deceptive"] == 40
