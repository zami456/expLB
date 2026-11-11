"""The layer sweep's storage, and the base-probe arm for probes the runner never scored.

The sweep pools on NDIF one request at a time and assembles each layer's file from those requests.
These tests pin that assembly (order, resumption, completeness), the output format
`persona_ood.load_pooled` reads, and `base_probe_arm(check=False)`.
"""

import pickle

import numpy as np
import pytest
import torch
import yaml

from experiments.persona import layer_sweep as S
from experiments.persona import persona_ood as P


def test_requests_assemble_into_each_layer_in_row_order(tmp_path):
    rng = np.random.default_rng(0)
    kept = [0, 2, 3, 5, 8]  # rows with detected tokens, in dataset order
    means = {row: rng.normal(size=(2, 4)).astype(np.float32) for row in kept}  # [layers, hidden]
    for batch in ([5, 0], [8], [3, 2]):  # requests arrive in any order and size
        S.save_request(tmp_path, batch, [24, 28], np.stack([means[r] for r in batch]))
    assert S.done_rows(tmp_path) == set(kept)
    for j, layer in enumerate((24, 28)):
        np.testing.assert_array_equal(S.assemble(tmp_path, kept, layer),
                                      np.stack([means[r][j] for r in kept]))


def test_assembly_refuses_missing_rows(tmp_path):
    S.save_request(tmp_path, [0], [24], np.zeros((1, 1, 4), dtype=np.float32))
    with pytest.raises(RuntimeError, match="not pooled"):
        S.assemble(tmp_path, [0, 1], 24)


def test_written_files_load_as_pooled_datasets(tmp_path):
    z = np.arange(12, dtype=np.float32).reshape(3, 4)
    S.write_pooled(tmp_path / "harm_refusal_choice__plain.npz", z,
                   ["HONEST", "DECEPTIVE", "AMBIGUOUS"], "harm_refusal_choice__plain")
    loaded = P.load_pooled(tmp_path)
    ds = loaded["harm_refusal_choice__plain"]
    np.testing.assert_array_equal(ds.z, z[:2])  # AMBIGUOUS dropped, as for every pooled file
    np.testing.assert_array_equal(ds.y, [0, 1])


def fake_run(folder, layer, direction, results_table=None):
    folder.mkdir()
    (folder / "cfg.yaml").write_text(yaml.safe_dump({"detect_layers": [layer],
                                                     "train_data": "repe_honesty__plain"}))
    with open(folder / "detector.pt", "wb") as f:
        pickle.dump({"directions": torch.tensor([direction]), "scaler_mean": torch.zeros(1, 2),
                     "scaler_scale": torch.ones(1, 2)}, f)
    if results_table is not None:
        results_table.to_csv(folder / "results_table.csv")


def datasets():
    z = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    return {"convince_me__plain": P.Pooled("convince_me__plain", z, np.array([0, 0, 1, 1]))}


def test_a_probe_without_runner_scores_joins_only_when_asked(tmp_path):
    import pandas as pd

    fake_run(tmp_path / "run", 24, [1.0, 0.0])
    control = P.Pooled("XXXX-20__plain", np.zeros((5, 2)), np.zeros(5, dtype=int))
    with pytest.raises(FileNotFoundError):
        P.base_probe_arm(tmp_path / "run", 24, datasets(), control)
    score = P.base_probe_arm(tmp_path / "run", 24, datasets(), control, check=False)
    assert P.auroc(np.array([0, 0, 1, 1]), score(datasets()["convince_me__plain"].z)) == 1.0

    # with runner scores present, a mismatch is still refused
    wrong = pd.DataFrame({"auroc": [0.5], "balanced_accuracy_at_fpr_1%": [0.5]},
                         index=["convince_me__plain"])
    fake_run(tmp_path / "other", 24, [1.0, 0.0], results_table=wrong)
    with pytest.raises(ValueError, match="does not reproduce"):
        P.base_probe_arm(tmp_path / "other", 24, datasets(), control)


def test_the_layer_check_still_applies_without_runner_scores(tmp_path):
    fake_run(tmp_path / "run", 24, [1.0, 0.0])
    control = P.Pooled("XXXX-20__plain", np.zeros((5, 2)), np.zeros(5, dtype=int))
    with pytest.raises(ValueError, match="not \\[28\\]"):
        P.base_probe_arm(tmp_path / "run", 28, datasets(), control, check=False)


@pytest.mark.parametrize(
    "first_line, replaced",
    [
        ("# Layer 44 (stride-4 sweep)", True),              # this script's own output
        ("# Layer 44: the depth curve turns over", False),  # the hand-written layer-44 write-up
        ("# Extended persona set: fixing HP-KR at the basis level", False),
        ("# Layer 40 (stride-4 sweep)", False),             # another layer's template
    ],
)
def test_only_a_generated_summary_is_replaced(tmp_path, first_line, replaced):
    """Re-running `analyse` over layers 20, 40 and 44 once overwrote their hand-written
    write-ups with the template. The guard keeps anything this script did not write."""
    summary = tmp_path / "summary.md"
    summary.write_text(f"{first_line}\n\nbody\n", encoding="utf-8")
    assert S.replaceable(summary, layer=44) is replaced


def test_a_missing_summary_is_written(tmp_path):
    assert S.replaceable(tmp_path / "summary.md", layer=44) is True


def test_the_generated_summary_is_replaceable_by_its_own_writer():
    """The template `_layer_summary` produces must pass `replaceable`, or a sweep would stop
    refreshing its own output after the first run."""
    text = S._layer_summary(44, "ext")
    assert text.lstrip().startswith("# Layer 44 (stride-4 sweep)")
