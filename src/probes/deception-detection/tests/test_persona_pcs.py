"""Tests for the persona vector geometry (arXiv 2605.09391, Eq. 1-3)."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from deception_detection import persona_pcs


@pytest.fixture(scope="module")
def bank() -> persona_pcs.PersonaBank:
    return persona_pcs.load_bank()


@pytest.fixture(scope="module")
def geom(bank: persona_pcs.PersonaBank) -> persona_pcs.PersonaGeometry:
    return persona_pcs.fit_geometry(bank, k=3)


class TestBank:
    def test_shape_and_provenance(self, bank: persona_pcs.PersonaBank):
        assert bank.personas.shape == (26, 8192)
        assert bank.layer == 20
        assert bank.model == "meta-llama/Llama-3.3-70B-Instruct"

    def test_groups_are_disjoint_and_present(self, bank: persona_pcs.PersonaBank):
        dec, hon = set(persona_pcs.DECEPTION_PERSONAS), set(persona_pcs.HONESTY_PERSONAS)
        assert len(dec) == 8 and len(hon) == 7
        assert not dec & hon
        assert (dec | hon) <= set(bank.names)

    def test_honesty_group_recovers_from_the_axis(self, bank: persona_pcs.PersonaBank):
        # the bank's role_groups field stores key names rather than members, so the hard-coded
        # honesty list is only trustworthy because it is re-derivable from the axis identity
        # mean(honesty) = mean(deception) - deception_axis
        assert persona_pcs.recover_honesty_group(bank) == tuple(
            sorted(persona_pcs.HONESTY_PERSONAS)
        )

    def test_recovered_group_reproduces_the_axis(self, bank: persona_pcs.PersonaBank):
        dec = bank.vectors_for(persona_pcs.DECEPTION_PERSONAS).mean(0)
        hon = bank.vectors_for(persona_pcs.HONESTY_PERSONAS).mean(0)
        assert np.allclose(dec - hon, bank.axes["deception"], atol=1e-5)

    def test_axis_unit_is_normalised(self, bank: persona_pcs.PersonaBank):
        # the bank stores raw mean-differences; norm is 0.890 for deception
        assert not np.isclose(np.linalg.norm(bank.axes["deception"]), 1.0)
        assert np.isclose(np.linalg.norm(bank.axis_unit("deception")), 1.0)

    def test_unknown_persona_is_an_error(self, bank: persona_pcs.PersonaBank):
        with pytest.raises(KeyError, match="not in bank"):
            bank.vectors_for(("pc_nonexistent",))


class TestGeometry:
    def test_p_is_the_fifteen(self, geom: persona_pcs.PersonaGeometry):
        assert len(geom.persona_names) == 15
        assert sum(g == "positive" for g in geom.group_of.values()) == 8
        assert sum(g == "negative" for g in geom.group_of.values()) == 7

    def test_centring_removes_the_mean(self, bank, geom):
        centred = bank.vectors_for(geom.persona_names) - geom.mean
        assert np.allclose(centred.mean(0), 0.0, atol=1e-10)

    def test_components_are_orthonormal(self, geom: persona_pcs.PersonaGeometry):
        assert np.allclose(geom.components @ geom.components.T, np.eye(3), atol=1e-9)

    def test_explained_variance_is_non_increasing(self, geom: persona_pcs.PersonaGeometry):
        ev = geom.explained_variance_ratio
        assert np.all(np.diff(ev) <= 0)
        assert 0 < ev.sum() < 1

    def test_pc1_is_essentially_the_deception_axis(self, geom: persona_pcs.PersonaGeometry):
        # the reason for restricting P to the 15: over all 26 this falls to ~0.72
        assert geom.alignment()[0] > 0.9

    def test_pcs_are_oriented_along_the_axis(self, geom: persona_pcs.PersonaGeometry):
        # sign of a PC is arbitrary; without orienting them "higher = more deceptive" would
        # hold or not at random between runs
        assert np.all(geom.components @ geom.axis >= 0)

    def test_pc1_separates_the_two_persona_groups(self, bank, geom):
        scores = geom.project(bank.vectors_for(geom.persona_names) - geom.mean, k=1).ravel()
        deceptive, honest = scores[:8], scores[8:]
        assert deceptive.min() > honest.max()

    def test_cannot_take_more_components_than_rank(self, bank: persona_pcs.PersonaBank):
        with pytest.raises(ValueError, match="rank"):
            persona_pcs.fit_geometry(bank, k=15)

    def test_projection_matches_explicit_dot_products(self, geom: persona_pcs.PersonaGeometry):
        z = np.arange(2 * 8192, dtype=np.float64).reshape(2, 8192) / 8192
        projected = geom.project(z, k=3)
        assert projected.shape == (2, 3)
        for i in range(2):
            for k in range(3):
                assert np.isclose(projected[i, k], np.dot(z[i], geom.components[k]))

    def test_axis_score_is_scale_invariant_for_auroc(self, geom: persona_pcs.PersonaGeometry):
        rng = np.random.default_rng(0)
        z = rng.standard_normal((200, 8192))
        y = (z @ geom.axis > 0).astype(int)
        unit = roc_auc_score(y, geom.axis_score(z))
        raw = roc_auc_score(y, z @ (geom.axis * 0.8903))
        assert np.isclose(unit, raw)


class TestRestrictingPHelps:
    def test_fifteen_beats_twentysix_on_axis_alignment(self, bank: persona_pcs.PersonaBank):
        """The documented reason for deviating from the paper's P = all personas."""
        fifteen = persona_pcs.fit_geometry(bank, k=1).alignment()[0]
        all_names = tuple(n for n in bank.names if n != "default")
        twentyfive = persona_pcs.fit_geometry(
            bank, k=1, deception=all_names, honesty=()
        ).alignment()[0]
        assert fifteen > twentyfive


class TestBankDecomposition:
    """The four contrast groups plus `default` should account for the whole bank."""

    def test_all_26_personas_are_accounted_for(self, bank: persona_pcs.PersonaBank):
        groups = (
            persona_pcs.DECEPTION_PERSONAS,
            persona_pcs.HONESTY_PERSONAS,
            persona_pcs.SYCOPHANCY_PERSONAS,
            persona_pcs.ANTI_SYCOPHANCY_PERSONAS,
        )
        assert [len(g) for g in groups] == [8, 7, 5, 5]
        union = set().union(*groups)
        assert len(union) == 25, "groups overlap"
        assert union | {"default"} == set(bank.names)

    def test_anti_sycophancy_recovers_from_its_axis(self, bank: persona_pcs.PersonaBank):
        assert persona_pcs.recover_anti_sycophancy_group(bank) == tuple(
            sorted(persona_pcs.ANTI_SYCOPHANCY_PERSONAS)
        )

    def test_recovered_group_reproduces_the_sycophancy_axis(self, bank: persona_pcs.PersonaBank):
        syc = bank.vectors_for(persona_pcs.SYCOPHANCY_PERSONAS).mean(0)
        anti = bank.vectors_for(persona_pcs.ANTI_SYCOPHANCY_PERSONAS).mean(0)
        assert np.allclose(syc - anti, bank.axes["sycophancy"], atol=1e-5)


class TestSycophancyGeometry:
    @pytest.fixture(scope="class")
    def syc(self, bank: persona_pcs.PersonaBank) -> persona_pcs.PersonaGeometry:
        return persona_pcs.fit_geometry(
            bank,
            k=3,
            deception=persona_pcs.SYCOPHANCY_PERSONAS,
            honesty=persona_pcs.ANTI_SYCOPHANCY_PERSONAS,
            axis_name="sycophancy",
        )

    def test_pc1_is_the_sycophancy_axis(self, bank, syc):
        assert abs(float(syc.components[0] @ bank.axis_unit("sycophancy"))) > 0.99

    def test_pc1_separates_its_own_groups(self, bank, syc):
        scores = syc.project(bank.vectors_for(syc.persona_names) - syc.mean, k=1).ravel()
        assert scores[:5].min() > scores[5:].max()

    def test_the_two_behaviours_overlap_substantially(self, bank: persona_pcs.PersonaBank):
        """The premise of the orthogonalised arms.

        If a future bank made these independent, the raw sycophancy arm would already be a clean
        control and the orthogonalised ones would be redundant -- so this failing is informative,
        not merely a broken test.
        """
        overlap = abs(float(bank.axis_unit("deception") @ bank.axis_unit("sycophancy")))
        assert 0.53 < overlap < 0.55, f"axis overlap moved to {overlap:.4f}; revisit the design"


class TestOrthogonalize:
    def test_removes_the_direction_and_stays_orthonormal(self, bank, geom):
        against = bank.axis_unit("sycophancy")
        out = persona_pcs.orthogonalize(geom.components, against)
        assert np.allclose(out @ out.T, np.eye(len(out)), atol=1e-9)
        assert np.abs(out @ against).max() < 1e-10

    def test_rank_drops_when_the_direction_lies_in_the_span(self, geom):
        # removing PC1 itself must leave a 2-d basis, not a degenerate 3-d one
        out = persona_pcs.orthogonalize(geom.components, geom.components[0])
        assert out.shape[0] == 2

    def test_rejects_removing_everything(self):
        single = np.eye(1, 8)
        with pytest.raises(ValueError, match="nothing left"):
            persona_pcs.orthogonalize(single, single[0])


class TestAxisNameIsRespected:
    """Regression: fit_geometry used to hardcode the deception axis.

    With it hardcoded, a sycophancy geometry silently carried the deception contrast, so its
    axis_score was byte-identical to the deception arm's and the "cross-behaviour" comparison
    measured nothing. The identical AUROC columns were the only symptom.
    """

    def test_sycophancy_geometry_carries_its_own_axis(self, bank: persona_pcs.PersonaBank):
        syc = persona_pcs.fit_geometry(
            bank,
            k=3,
            deception=persona_pcs.SYCOPHANCY_PERSONAS,
            honesty=persona_pcs.ANTI_SYCOPHANCY_PERSONAS,
            axis_name="sycophancy",
        )
        assert np.allclose(syc.axis, bank.axis_unit("sycophancy"))
        assert not np.allclose(syc.axis, bank.axis_unit("deception"))

    def test_the_two_geometries_score_differently(self, bank, geom):
        syc = persona_pcs.fit_geometry(
            bank,
            k=3,
            deception=persona_pcs.SYCOPHANCY_PERSONAS,
            honesty=persona_pcs.ANTI_SYCOPHANCY_PERSONAS,
            axis_name="sycophancy",
        )
        rng = np.random.default_rng(0)
        z = rng.standard_normal((50, 8192))
        assert not np.allclose(geom.axis_score(z), syc.axis_score(z))
