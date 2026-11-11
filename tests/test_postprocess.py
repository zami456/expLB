"""The postprocessor must not render per-token pages for a tail-mode run.

A tail-mode run (acts_positions: "tail") downloads only the positions the probe reads, so
every other token is scored from a zero vector. Rendering those scores produces pages that
look like results and are not -- the first follow-up run wrote 13,673 of them before this
guard existed. Prompt-level plots read only detected tokens and are still made.
"""

from types import SimpleNamespace

import pytest

from deception_detection import postprocess


@pytest.fixture
def calls(monkeypatch):
    """Replace every rendering function with a recorder, so no experiment is needed."""
    made: list[str] = []

    def recorder(name):
        return lambda *args, **kwargs: made.append(name)

    for name in (
        "plot_roc_curve_with_auroc",
        "violin_plot",
        "plot_roc_comparison",
        "make_dataset_specific_plots",
        "export_dataset_html",
        "write_style_css",
    ):
        monkeypatch.setattr(postprocess, name, recorder(name))
    monkeypatch.setattr(
        postprocess.token_html, "plot_control_token_scores", recorder("control_token_scores")
    )
    return made


def postprocessor(tmp_path, acts_positions: str) -> postprocess.ExperimentPostprocessor:
    # skip __init__: it builds a scorer from a real experiment, which this does not need
    processor = object.__new__(postprocess.ExperimentPostprocessor)
    cfg = SimpleNamespace(folder=tmp_path, acts_positions=acts_positions)
    processor.experiment = SimpleNamespace(cfg=cfg)
    processor.cfg = cfg
    processor.scorer = None
    processor.include_black_box = False
    processor.control_dataset_name = "XXXX-20__plain"
    return processor


PER_TOKEN = {"control_token_scores", "export_dataset_html", "write_style_css"}
PROMPT_LEVEL = {
    "plot_roc_curve_with_auroc",
    "violin_plot",
    "plot_roc_comparison",
    "make_dataset_specific_plots",
}


def test_tail_mode_skips_every_per_token_page(tmp_path, calls):
    postprocessor(tmp_path, "tail").make_all_plots()
    assert not PER_TOKEN & set(calls)
    assert not (tmp_path / "html").exists()


def test_tail_mode_still_makes_the_prompt_level_plots(tmp_path, calls):
    postprocessor(tmp_path, "tail").make_all_plots()
    assert PROMPT_LEVEL <= set(calls)


def test_full_sequence_runs_are_unchanged(tmp_path, calls):
    postprocessor(tmp_path, "all").make_all_plots()
    assert (PER_TOKEN | PROMPT_LEVEL) <= set(calls)
