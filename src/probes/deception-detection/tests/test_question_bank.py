"""Tests for the question extension: the bank arithmetic, the contrast definition, the Holm loop, the seeded
subsets, and the question-file gates."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from deception_detection import persona_pcs as pcs
from deception_detection.scripts import question_bank as QB

QDIR = QB.QDIR
PY = sys.executable


def _toy_store(rng, roles=4, questions=6, hidden=5):
    V = rng.normal(size=(roles, questions, hidden)).astype(np.float32)
    n = rng.integers(0, 4, size=(roles, questions))
    n[:, 0] = 2  # every role has at least one kept row somewhere
    return V, n


def test_bank_vectors_plain_is_the_n_weighted_mean():
    rng = np.random.default_rng(0)
    V, n = _toy_store(rng)
    out = QB.bank_vectors(V, n, [0, 1, 2], "plain")
    for r in range(V.shape[0]):
        want = (V[r, :3].astype(np.float64) * n[r, :3, None]).sum(0) / n[r, :3].sum()
        np.testing.assert_allclose(out[r], want, rtol=1e-12)


def test_bank_vectors_balanced_weights_questions_equally():
    rng = np.random.default_rng(1)
    V, n = _toy_store(rng)
    out = QB.bank_vectors(V, n, [0, 1, 2], "balanced")
    for r in range(V.shape[0]):
        has = n[r, :3] > 0
        want = V[r, :3][has].astype(np.float64).mean(0)
        np.testing.assert_allclose(out[r], want, rtol=1e-12)


def test_bank_vectors_refuses_a_role_with_no_kept_row():
    rng = np.random.default_rng(2)
    V, n = _toy_store(rng)
    n[1, 1:3] = 0
    with pytest.raises(ValueError):
        QB.bank_vectors(V, n, [1, 2], "plain")


def test_directions_match_the_direct_group_contrast():
    rng = np.random.default_rng(3)
    names = set(pcs.DECEPTION_PERSONAS + pcs.HONESTY_PERSONAS + pcs.SYCOPHANCY_PERSONAS + pcs.ANTI_SYCOPHANCY_PERSONAS
                + pcs.KNOWLEDGE_DENIAL_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS) | {"default"}
    vec = {n: rng.normal(size=16) for n in names}
    D = QB.directions(vec)
    m = lambda g: np.stack([vec[n] for n in g]).mean(0)  # noqa: E731
    raw9 = m(pcs.DECEPTION_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS) - m(pcs.HONESTY_PERSONAS + pcs.KNOWLEDGE_CANDOUR_PERSONAS)
    np.testing.assert_allclose(D["D9"], raw9 / np.linalg.norm(raw9), rtol=1e-12)
    raw5 = m(pcs.DECEPTION_PERSONAS) - m(pcs.HONESTY_PERSONAS)
    np.testing.assert_allclose(D["D5"], raw5 / np.linalg.norm(raw5), rtol=1e-12)
    assert abs(np.linalg.norm(D["D8"]) - 1) < 1e-12


def test_holm_matches_the_textbook_step_down():
    p = np.array([0.01, 0.04, 0.03, 0.20])
    adj = QB.holm(p)
    # sorted p: 0.01 (x4 = 0.04), 0.03 (x3 = 0.09), 0.04 (x2 = 0.09 after the running max), 0.20 (x1 = 0.20)
    np.testing.assert_allclose(adj, [0.04, 0.09, 0.09, 0.20])


def test_question_file_and_variants():
    rows = QB.questions()
    assert len(rows) == 150
    v = QB.variants(rows)
    assert len(v["B30"][0]) == 30 and len(v["B150"][0]) == 150 and len(v["B30+90div"][0]) == 120
    fam = QB.families(rows)
    assert len(fam) == 10 and sum(len(q) for q in fam.values()) == 150
    assert all(len(q) == 15 for k, q in fam.items() if k.startswith("imt_"))


def test_seeded_subsets_are_reproducible():
    a = np.random.default_rng(QB.SEED).choice(150, 30, replace=False)
    b = np.random.default_rng(QB.SEED).choice(150, 30, replace=False)
    assert (a == b).all()


def _run_check(path: Path) -> int:
    return subprocess.run([PY, "-I", str(QDIR / "check_questions.py"), str(path)], capture_output=True).returncode


def test_check_questions_passes_on_the_frozen_file():
    assert _run_check(QDIR / "kd_questions_shared_v2.jsonl") == 0


@pytest.mark.parametrize("mutation", ["duplicate", "blank", "reorder", "alter_original"])
def test_check_questions_fails_on_a_broken_file(tmp_path, mutation):
    lines = (QDIR / "kd_questions_shared_v2.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(l) for l in lines]
    if mutation == "duplicate":
        rows[40]["question"] = rows[41]["question"]
    elif mutation == "reorder":
        rows[40], rows[41] = rows[41], rows[40]
    elif mutation == "alter_original":
        rows[3]["question"] = rows[3]["question"] + " Really?"
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    if mutation == "blank":
        text = text.replace("\n", "\n\n", 1)
    p = tmp_path / "broken.jsonl"
    p.write_text(text, encoding="utf-8")
    assert _run_check(p) == 1
