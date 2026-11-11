"""The unified-axis arm in `persona_ood`.

The unified axis is the paper's own widening of the 8v7 deception contrast (its Appendix C:
deception and sycophancy pooled over a balanced question set). Here it is the control for the
extended axis, which widens the same contrast with knowledge-denial roles instead. These tests
pin what makes the comparison meaningful: that the arm really is the bank's stored unified
direction, that it is a different direction from both of its parents, and that it is available
from a bank without the knowledge-denial personas as well as one with them.
"""

import numpy as np
import pytest

from deception_detection import persona_pcs
from experiments.persona import persona_ood


@pytest.fixture(scope="module")
def bank() -> persona_pcs.PersonaBank:
    return persona_pcs.load_bank()


class TestUnifiedGeometry:
    def test_p_is_the_twenty_five(self, bank: persona_pcs.PersonaBank):
        # 8 deception + 5 sycophancy against 7 honesty + 5 anti-sycophancy; `default` is excluded
        geom = persona_ood.unified_geometry(bank)
        assert len(geom.persona_names) == 25
        assert "default" not in geom.persona_names

    def test_axis_is_the_banks_stored_unified_axis(self, bank: persona_pcs.PersonaBank):
        geom = persona_ood.unified_geometry(bank)
        assert np.allclose(geom.axis, bank.axis_unit("unified"))

    def test_axis_is_neither_of_its_parents(self, bank: persona_pcs.PersonaBank):
        # if it were a rescaling of one parent the arm would only duplicate an existing column
        unified = bank.axis_unit("unified")
        for parent in ("deception", "sycophancy"):
            assert abs(float(unified @ bank.axis_unit(parent))) < 0.95, parent

    def test_pcs_are_oriented_along_the_axis(self, bank: persona_pcs.PersonaBank):
        geom = persona_ood.unified_geometry(bank)
        assert all(float(component @ geom.axis) >= 0 for component in geom.components)


class TestUnifiedArms:
    def test_keys_and_shapes(self, bank: persona_pcs.PersonaBank):
        arms = persona_ood.unified_arms(bank)
        assert set(arms) == {"auroc_uni_axis", "auroc_uni_pc1"}
        z = np.random.default_rng(0).normal(size=(7, bank.personas.shape[1]))
        for name, fn in arms.items():
            assert fn(z).shape == (7,), name

    def test_axis_score_is_the_projection_on_the_unit_axis(self, bank: persona_pcs.PersonaBank):
        arms = persona_ood.unified_arms(bank)
        z = np.random.default_rng(1).normal(size=(5, bank.personas.shape[1]))
        assert np.allclose(arms["auroc_uni_axis"](z), z @ bank.axis_unit("unified"))

    def test_available_without_the_knowledge_denial_personas(self, bank: persona_pcs.PersonaBank):
        # `extended_arms` returns {} for this bank; the unified arm must not follow it
        assert persona_ood.extended_arms(bank) == {}
        assert persona_ood.unified_arms(bank)


@pytest.mark.skipif(
    not persona_pcs.EXTENDED_VECTORS_DIR.exists(),
    reason=f"needs the extended persona vectors at {persona_pcs.EXTENDED_VECTORS_DIR}",
)
class TestUnifiedFromTheExtendedFolder:
    """The same axis rebuilt from the larger extraction, which is what makes the two folders
    comparable: the extended folder re-extracted all 26 original personas with more rollouts, so
    a unified axis from there is not the published one."""

    @pytest.fixture(scope="class")
    def extended_bank(self) -> persona_pcs.PersonaBank:
        return persona_pcs.load_bank_at_layer(20, persona_pcs.EXTENDED_VECTORS_DIR)

    def test_the_arm_exists_there_too(self, extended_bank: persona_pcs.PersonaBank):
        assert "unified" in extended_bank.axes
        assert persona_ood.extended_arms(extended_bank)  # this bank has both
        assert persona_ood.unified_arms(extended_bank)

    def test_it_is_the_re_extracted_axis_not_the_published_one(self, bank, extended_bank):
        # close, because it is the same 25 roles -- but not identical, which is the confound the
        # two-folder comparison exists to separate
        published = bank.axis_unit("unified")
        re_extracted = extended_bank.axis_unit("unified")
        cosine = abs(float(published @ re_extracted))
        assert 0.9 < cosine < 1.0
