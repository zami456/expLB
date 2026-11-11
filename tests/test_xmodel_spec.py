"""The per-model spec the revised thesis's studies run from on Qwen, Mistral and Gemma.

- Each model's sweep is Llama's 16 layers scaled to its depth, round(L / 80 * n_blocks), capped at
  the last block; the named blocks (Liars' Bench's, mid-depth, the post-hoc pair) are inside it or
  read beside it, and the depths match the models' configs.
- Llama's entry names the thesis's own folders, so `model="llama"` changes nothing.
- The published probe rows: Llama's are the transcribed constants; another model's come from
  Cadenza's runs and Table 8.
"""

import math

import numpy as np
import pytest

from deception_detection.models import ModelName, get_num_layers_for_model
from experiments.other_models import lb_models as LB
from experiments.persona import layer_sweep, persona_ood
from deception_detection.utils import REPO_ROOT


@pytest.mark.parametrize("key", ["qwen", "mistral", "gemma"])
def test_sweep_scales_llamas(key: str) -> None:
    s = LB.SPECS[key]
    expected = tuple(min(s.n_blocks - 1, round(L / 80 * s.n_blocks)) for L in LB.LLAMA_SWEEP)
    assert s.sweep == expected
    assert len(set(s.sweep)) == 16


@pytest.mark.parametrize("key", list(LB.SPECS))
def test_named_blocks(key: str) -> None:
    s = LB.SPECS[key]
    assert s.n_blocks == get_num_layers_for_model(ModelName(s.model))
    assert {s.mid, *s.posthoc} <= set(s.sweep)
    assert s.lb_block in s.layers
    # the post-hoc pair sits about 5% of depth either side of mid-depth, as 36 and 44 do of 40
    for block, sign in zip(s.posthoc, (-1, 1)):
        assert math.isclose((block - s.mid) / s.n_blocks, sign * 0.05, abs_tol=0.025)
    assert s.key_layers == (s.lb_block, s.mid, s.posthoc[1])


def test_lb_blocks_are_the_reproduced_ones() -> None:
    """The LB block of each model is the one `lb_models.py score` chose (`blocks.csv`)."""
    chosen = {"qwen": 20, "mistral": 9, "gemma": 8}
    for key, block in chosen.items():
        assert LB.SPECS[key].lb_block == block == LB.TARGETS[key].prior


def test_llama_keeps_the_thesis_paths() -> None:
    s = LB.spec("llama")
    assert LB.spec(None) is s
    for layer in (20, 40, 44, 79):
        assert s.results("persona_ood", layer, "ext") == layer_sweep.results_dir(layer, "ext")
        assert s.results("persona_ood", layer, "orig") == layer_sweep.results_dir(layer, "orig")
        assert s.pooled_dirs(layer) == [layer_sweep.pooled_dir(layer)]
        assert s.probe_run(layer) == layer_sweep.base_run(layer)
    assert s.results("persona_depth", bank="ext") == layer_sweep.DEPTH_DIR
    assert s.results("loo", 40, "ext") == REPO_ROOT / "results" / "loo_L40"
    assert s.bank_dir("ext") == layer_sweep.vectors_dir("ext")
    assert s.followup_run() == persona_ood.FOLLOWUP_RUN
    assert s.layers == layer_sweep.ALL_LAYERS
    # the lenient bank is a comparison of this work's own, beside the thesis's folders
    assert s.results("persona_ood", 40, "s23") == LB.XMODEL_RESULTS / "llama" / "persona_ood_s23_L40"


def test_other_models_paths() -> None:
    m = LB.spec("mistral")
    assert m.pooled_dirs(20) == [LB.XMODEL_ROOT / "activations" / "mistral-small-3.1-24b" / "pooled_B20",
                                 LB.XMODEL_ROOT / "activations" / "mistral-small-3.1-24b" / "pooled_B20_trim"]
    assert m.pooled_dirs(9, followup=True) == [
        LB.XMODEL_ROOT / "activations_followup" / "mistral-small-3.1-24b" / "pooled_B9"]
    assert m.results("persona_ood", 20, "s3") == LB.XMODEL_RESULTS / "mistral" / "persona_ood_s3_B20"
    assert m.probe_run(20).name == "XXXX-9-mistral-all-B20"
    # Qwen has no insider trading, so no trimmed folder; its Follow-Up reads "No" alone
    q = LB.spec("qwen")
    assert len(q.pooled_dirs(40)) == 1
    assert q.followup_run().name == "XXXX-9-qwen-all-followup-B20-last"
    # the user_stay strict bank leads; the system-format lenient bank stays as a comparison
    assert q.banks == ("s3u", "s23u", "s23")
    assert q.bank_dir() == LB.BANK_DIRS[("qwen", "s3u")]
    assert q.bank_dir("s23") == LB.BANK_DIRS[("qwen", "s23")]
    with pytest.raises(ValueError):
        q.bank_dir("s3")


def test_every_bank_resolves() -> None:
    """Each bank a spec names has a folder: Llama's thesis banks through persona_pcs, the rest in BANK_DIRS."""
    for key in LB.SPECS:
        s = LB.spec(key)
        assert s.primary_bank == s.banks[0]
        for bank in s.banks:
            assert s.bank_dir(bank).name, (key, bank)
            if not (key == "llama" and bank in ("ext", "orig")):
                assert (key, bank) in LB.BANK_DIRS


def test_load_pooled_later_folder_wins(tmp_path) -> None:
    """A dataset in the `_trim` folder replaces the full folder's file of the same name."""
    full, trim = tmp_path / "full", tmp_path / "trim"
    for folder, value in ((full, 0.0), (trim, 1.0)):
        folder.mkdir()
        np.savez(folder / "insider_trading_cadenza__plain.npz", pooled=np.full((2, 3), value, np.float32),
                 labels=np.array(["HONEST", "DECEPTIVE"]), name="insider_trading_cadenza__plain")
    np.savez(full / "XXXX-20__plain.npz", pooled=np.zeros((2, 3), np.float32),
             labels=np.array(["HONEST", "HONEST"]), name="XXXX-20__plain")
    loaded = persona_ood.load_pooled([full, trim])
    assert set(loaded) == {"insider_trading_cadenza__plain", "XXXX-20__plain"}
    assert (loaded["insider_trading_cadenza__plain"].z == 1.0).all()
    assert persona_ood.load_pooled(full)["insider_trading_cadenza__plain"].z.sum() == 0.0


def test_published_rows() -> None:
    assert LB.published("llama") is persona_ood.PAPER_PROBES
    rows = LB.published("mistral")
    columns = persona_ood.PAPER_COLUMNS
    mean = dict(zip(columns, rows["Mean Probe"]["auroc"]))
    # Mistral has no CG rows in Liars' Bench; it has the other four base sets
    assert math.isnan(mean["CG"])
    assert all(0.0 <= mean[c] <= 1.0 for c in ("HP-C", "HP-KR", "ID", "IT"))
    assert rows["Upper-Bound Probe"]["privileged"]
    assert dict(zip(columns, rows["Upper-Bound Probe"]["auroc"]))["HP-KR"] == 0.96
