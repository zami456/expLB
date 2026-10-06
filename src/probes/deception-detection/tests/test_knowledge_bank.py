"""The knowledge-grounded bank's own logic: its cells, the policy filter, and its bank files."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from deception_detection import persona_pcs as pcs
from deception_detection.scripts import forced_denial_bank as fdb
from deception_detection.scripts import knowledge_bank as kb

PAIR_A = {"assert": "I know the answer.", "deny": "I do not know the answer."}


def _fake(pairs: int = 20) -> tuple[pd.DataFrame, dict[str, dict[int, str]]]:
    """A matched sample shaped like `sample_chem`'s, and five prompt variants per role."""
    rows = [{"pair": p, "knows": knows, "prompt": f"question {p}, {'known' if knows else 'unknown'}"}
            for p in range(pairs) for knows in (True, False)]
    sample = pd.DataFrame(rows, index=range(1000, 1000 + 2 * pairs))  # chem row indices
    prompts = {role: {v: f"{role}, variant {v}" for v in range(5)}
               for role in fdb.FEIGNED + fdb.SINCERE + kb.PLAIN}
    return sample, prompts


def test_roles_are_the_thesis_groups():
    assert set(fdb.FEIGNED) == set(pcs.KNOWLEDGE_DENIAL_PERSONAS)
    assert set(fdb.SINCERE) | set(kb.PLAIN) == set(pcs.KNOWLEDGE_CANDOUR_PERSONAS)
    assert not set(fdb.SINCERE) & set(kb.PLAIN)


@pytest.mark.parametrize("reply, kind", [
    ("I do not know the answer.", "deny"),
    ("I know the answer.", "assert"),
    ("  I know the answer.\r\n", "assert"),
    ('"I know the answer."', "other"),  # quoted, so not the same text
    ("I don't know the answer to the question.", "other"),  # the other pair's sentence
    ("I do not know the answer. It is outside my remit.", "other"),
    ("I do not know the", "other"),
])
def test_classify_needs_one_of_the_offered_sentences(reply: str, kind: str):
    assert kb.classify(reply, PAIR_A) == kind


def test_cells_have_their_roles_questions_and_policies():
    sample, prompts = _fake()
    rows = pd.DataFrame(kb.build_rows(sample, 20, prompts))
    for cell, (roles, knows, policy) in kb.CELLS.items():
        sub = rows[rows["cell"] == cell]
        assert set(sub["role"]) == set(roles)
        assert (sub["knows"] == knows).all() and (sub["policy"] == policy).all()
        # every role in a cell answers the same questions
        assert len({frozenset(g["item"]) for _, g in sub.groupby("role")}) == 1
    items = {cell: set(rows.loc[rows["cell"] == cell, "item"]) for cell in kb.CELLS}
    assert items["feigned"] == items["plain"]  # both known
    assert items["sincere"] == items["stance"]  # both unknown
    assert not items["feigned"] & items["sincere"]


def test_matched_questions_share_their_rank():
    sample, prompts = _fake()
    rows = pd.DataFrame(kb.build_rows(sample, 20, prompts))
    for _, g in rows.groupby("pair"):
        assert g["rank"].nunique() == 1 and g["assert"].nunique() == 1 and g["assert_first"].nunique() == 1
        assert g["knows"].nunique() == 2


def test_pair_order_and_variant_balance_within_every_cell():
    sample, prompts = _fake()
    rows = pd.DataFrame(kb.build_rows(sample, 20, prompts))
    for (cell, role), g in rows.groupby(["cell", "role"]):
        combos = g.groupby(["assert", "assert_first"]).size()
        assert len(combos) == 4 and set(combos) == {5}, (cell, role)
        assert set(g["variant"].value_counts()) == {4} and g["variant"].nunique() == 5, (cell, role)


def test_user_turn_offers_both_sentences_in_its_order():
    sample, prompts = _fake(4)
    for r in kb.build_rows(sample, 4, prompts):
        first, second = (r["assert"], r["deny"]) if r["assert_first"] else (r["deny"], r["assert"])
        assert r["user"].endswith(f'"{first}" or "{second}"')
        assert r["user"].startswith(sample.loc[r["item"], "prompt"])
        assert r["system"] == prompts[r["role"]][r["variant"]]


def test_a_small_run_is_a_prefix_of_the_full_one():
    sample, prompts = _fake()
    full = {(r["cell"], r["role"], r["item"]): r for r in kb.build_rows(sample, 20, prompts)}
    small = kb.build_rows(sample, 8, prompts)
    assert len(small) == 8 * 14
    assert all(full[(r["cell"], r["role"], r["item"])] == r for r in small)


def test_kept_rows_follow_each_cells_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(kb, "OUT", tmp_path)
    sample, prompts = _fake(4)
    rows = [r | {"row": i} for i, r in enumerate(kb.build_rows(sample, 4, prompts))]
    (tmp_path / "items.json").write_text(json.dumps(rows), encoding="utf-8")
    replies = []
    for r in rows:
        text = [r["deny"], r["assert"], "Sorry, I can't help."][r["row"] % 3]
        replies.append({"row": r["row"], "reply": text, "finished": True, "kind": kb.classify(text, r)})
    (tmp_path / "rollouts.jsonl").write_text("".join(json.dumps(x) + "\n" for x in replies), encoding="utf-8")
    kept = kb.kept_rows()
    expected = {r["row"] for r, x in zip(rows, replies) if x["kind"] == r["policy"]}
    assert kept and {r["row"] for r in kept} == expected
    assert all(r["assistant"] == r[r["policy"]] for r in kept)


# --- bank files ----------------------------------------------------------------------------------

BANK_NAMES = (pcs.DECEPTION_PERSONAS + pcs.HONESTY_PERSONAS + pcs.SYCOPHANCY_PERSONAS
              + pcs.ANTI_SYCOPHANCY_PERSONAS + pcs.KNOWLEDGE_DENIAL_PERSONAS
              + pcs.KNOWLEDGE_CANDOUR_PERSONAS + ("default",))


def _write_bank(path, rows: int, layers: list[int] | None, hidden: int = 4) -> None:
    rng = np.random.default_rng(0)
    for name in BANK_NAMES:
        payload = {"vector": torch.from_numpy(rng.standard_normal((rows, hidden)).astype(np.float32)), "n": 3}
        if layers is not None:
            payload["layers"] = layers
        torch.save(payload, path / f"{name}.pt")


def test_a_partial_file_loads_by_its_layer_list(tmp_path):
    _write_bank(tmp_path, 2, [20, 40])
    bank = pcs.load_bank_at_layer(40, tmp_path)
    stored = torch.load(tmp_path / "pc_stonewaller.pt", weights_only=False)["vector"]
    assert np.array_equal(bank.vectors_for(("pc_stonewaller",))[0], stored[1].double().numpy())
    with pytest.raises(IndexError, match="outside"):
        pcs.load_bank_at_layer(36, tmp_path)


def test_a_full_file_indexes_by_layer_with_or_without_the_list(tmp_path):
    for layers in (None, list(range(80))):
        _write_bank(tmp_path, 80, layers)
        bank = pcs.load_bank_at_layer(40, tmp_path)
        stored = torch.load(tmp_path / "pc_handler.pt", weights_only=False)["vector"]
        assert np.array_equal(bank.vectors_for(("pc_handler",))[0], stored[40].double().numpy())


def test_a_layer_list_that_does_not_fit_the_rows_is_an_error(tmp_path):
    _write_bank(tmp_path, 2, [20, 36, 40])
    with pytest.raises(ValueError, match="layers listed"):
        pcs.load_bank_at_layer(40, tmp_path)
