"""Tests for per-token persona reads.

The reads reduce one example's per-token projections to a single score. What matters is that
each picks the tokens it claims to -- a read that silently took the wrong position would produce
a plausible AUROC for the wrong question -- and that the mean read is exactly the pooled score
the rest of the study used, which is what lets `verify` check against committed results.
"""

import numpy as np
import pytest

from experiments.persona import persona_positions as pp

# A row as stored: prompt tokens, then a 5-token response. Left padding is stripped before
# storage, so the row starts at its first real token.
PROJ = np.array([9.0, 7.0, 3.0,  1.0, 5.0, 2.0, 8.0, 4.0])
DETECT = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=bool)


class TestRowReads:
    def test_each_read_takes_the_tokens_it_names(self):
        reads = pp.row_reads(PROJ, DETECT)
        assert reads["mean"] == pytest.approx((1 + 5 + 2 + 8 + 4) / 5)
        assert reads["max"] == 8.0
        assert reads["first"] == 1.0
        assert reads["last"] == 4.0
        # "pre" is the token just before the response, not the first prompt token (the BOS)
        assert reads["pre"] == 3.0

    def test_top10_keeps_at_least_one_token(self):
        # 10% of 5 tokens rounds up to one: the top-10% mean is the max
        assert pp.row_reads(PROJ, DETECT)["top10"] == 8.0

    def test_top10_on_a_longer_response(self):
        proj = np.arange(20, dtype=float)
        detect = np.ones(20, dtype=bool)
        detect[0] = False  # one prompt token, 19 response tokens: top 2 are 18 and 19
        assert pp.row_reads(proj, detect)["top10"] == pytest.approx(18.5)

    def test_mean_read_equals_projecting_the_mean(self):
        """The property `verify` rests on: the axis score is linear, so averaging per-token
        projections equals projecting the averaged activation."""
        rng = np.random.default_rng(0)
        acts = rng.standard_normal((8, 16))
        axis = rng.standard_normal(16)
        axis /= np.linalg.norm(axis)
        per_token = pp.row_reads(acts @ axis, DETECT)["mean"]
        pooled = acts[DETECT].mean(0) @ axis
        assert per_token == pytest.approx(pooled, abs=1e-12)

    def test_a_response_starting_the_row_has_no_pre_token(self):
        detect = np.ones(8, dtype=bool)
        assert np.isnan(pp.row_reads(PROJ, detect)["pre"])


class TestProfiles:
    def test_bins_cover_the_response_in_order(self):
        proj = np.arange(23, dtype=float)
        detect = np.zeros(23, dtype=bool)
        detect[3:] = True  # a 20-token response, two tokens per bin
        bins = pp.row_bins(proj, detect)
        expected = [np.mean([3 + 2 * b, 4 + 2 * b]) for b in range(10)]
        np.testing.assert_allclose(bins, expected)

    def test_short_responses_are_left_out_of_the_relative_profile(self):
        assert np.isnan(pp.row_bins(PROJ, DETECT)).all()

    def test_first_token_profile_pads_short_responses_with_nan(self):
        first = pp.row_first(PROJ, DETECT, n_first=7)
        np.testing.assert_allclose(first[:5], [1, 5, 2, 8, 4])
        assert np.isnan(first[5:]).all()


class TestStorage:
    def make(self, name, rows, labels):
        lengths = [len(p) for p, _ in rows]
        return pp.TokenProjections(
            name=name,
            layer=20,
            axis_names=["extended", "denial"],
            proj=np.concatenate([np.stack([p, -p], 1) for p, _ in rows]).astype(np.float32),
            detect=np.concatenate([d for _, d in rows]),
            offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
            labels=np.array(labels),
        )

    def test_round_trip_preserves_rows(self, tmp_path):
        tp = self.make("x", [(PROJ, DETECT), (PROJ[:4], DETECT[:4])], ["HONEST", "DECEPTIVE"])
        path = tmp_path / "x.npz"
        tp.save(path)
        back = pp.TokenProjections.load(path)
        proj, detect = back.row(1)
        np.testing.assert_allclose(proj[:, 0], PROJ[:4])
        np.testing.assert_array_equal(detect, DETECT[:4])
        assert list(back.labels) == ["HONEST", "DECEPTIVE"]
        assert back.axis_names == ["extended", "denial"]

    def test_merge_keeps_each_row_intact(self):
        a = self.make("repe_honesty__plain", [(PROJ, DETECT)], ["HONEST"])
        b = self.make("repe_honesty__plain_val", [(PROJ[:5], DETECT[:5])], ["DECEPTIVE"])
        merged = pp.merge([a, b])
        assert len(merged.labels) == 2
        proj, detect = merged.row(1)
        np.testing.assert_allclose(proj[:, 0], PROJ[:5])
        np.testing.assert_array_equal(detect, DETECT[:5])

    def test_load_projections_merges_repe_splits(self, tmp_path):
        self.make("repe_honesty__plain", [(PROJ, DETECT)], ["HONEST"]).save(
            tmp_path / "repe_honesty__plain.npz")
        self.make("repe_honesty__plain_val", [(PROJ, DETECT)], ["DECEPTIVE"]).save(
            tmp_path / "repe_honesty__plain_val.npz")
        loaded = pp.load_projections(tmp_path)
        assert list(loaded) == ["repe_honesty__plain"]
        assert len(loaded["repe_honesty__plain"].labels) == 2
