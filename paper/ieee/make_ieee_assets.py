"""The IEEE short version's assets, taken from the thesis's own (paper/thesis/make_thesis_assets.py).

The short version names every design in words, never by its thesis label (D1 to D9, K, D9 + K).
This script

1. copies the thesis's numbers, bibliography and figure style into `generated/`, so every number
   in the paper is the thesis's own `\\V{key}`;
2. copies the knowledge direction's diagram and the generated tables the paper uses, with each
   design label replaced by its name; only header and row-name cells change, and the run stops
   unless every number survives, in order;
3. redraws the four plots whose text carries a label into `images/`. They are drawn by the
   thesis's own functions, from the same result files, at the same size and in the same colours;
   their text is renamed just before each is saved, and the run stops if a label survives.

The thesis folder is only read. Run from `src/probes/deception-detection`:

    python paper/ieee/make_ieee_assets.py
"""

import re
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # repo root
THESIS = ROOT / "paper" / "thesis"
GENERATED = HERE / "generated"
IMAGES = HERE / "images"

sys.path.insert(0, str(THESIS))
import make_thesis_assets as T  # noqa: E402  the thesis's generator: its data, sizes and colours

# A design label left anywhere in a plot or table: D1 to D9, or K standing alone (HP-KR's K is
# not alone, and the tables' k is lower case).
LABEL = re.compile(r"\bD[1-9]\b|\bK\b")
NUMBER = re.compile(r"\d+\.\d+")

# The thesis's plots print these labels; the paper prints the names.
PLOT_NAMES = {
    "D5, published contrast": "published contrast",
    "D8, knowledge-denial contrast": "knowledge-denial contrast",
    "D9, extended contrast": "extended contrast",
    "K, knowledge direction": "knowledge direction",
    "K$'$, new questions": "knowledge direction, new questions",
    "D9 + K, selected design": "selected detector",
    "benchmark probe, same layer": "base probe, same layer",
    "extended contrast + K, selected": "selected detector",
    "Extended contrast + K, selected (1-d)": "Selected detector (1-d)",
    # Gender-Secret's bar groups, on two lines so that neighbouring names do not overlap
    "benchmark probe": "base\nprobe",
    "D9": "extended\ncontrast",
    "K": "knowledge\ndirection",
    "D9 + K": "selected\ndetector",
}

# tab_designs.tex: its first column is the label, which goes; its row names become the paper's.
DESIGN_ROWS = {
    "Raw-activation probe": "Probe on raw activations",
    "Base probe (AISI)": "Base probe, own lies (AISI)",
    "Extended PCs ($k=3$)": "Persona method, extended PCs ($k=3$)",
    "Published + knowledge": "Published contrast + knowledge direction",
    "Extended + knowledge": "Selected detector (extended + knowledge)",
}
DESIGN_LABEL = re.compile(r"^(D[1-9]|K|D5 \+ K|D9 \+ K) & ")


def _numbers(text: str) -> list[str]:
    return NUMBER.findall(text)


def _check(name: str, before: str, after: str) -> None:
    """The renamed copy must keep every number of the thesis's file, in order, and no label."""
    if _numbers(before) != _numbers(after):
        raise AssertionError(f"{name}: the numbers changed while renaming")
    body = "\n".join(line for line in after.splitlines() if not line.lstrip().startswith("%"))
    if LABEL.search(body):
        raise AssertionError(f"{name}: a design label is left: {LABEL.findall(body)}")


def copies() -> list[str]:
    """Files the paper takes from the thesis unchanged."""
    GENERATED.mkdir(exist_ok=True)
    pairs = [(THESIS / "generated" / "numbers.tex", GENERATED / "numbers.tex"),
             (THESIS / "generated" / "fig_style.tex", GENERATED / "fig_style.tex"),
             (THESIS / "generated" / "tab_steering.tex", GENERATED / "tab_steering.tex"),
             (THESIS / "bibliography" / "references.bib", HERE / "references.bib")]
    for source, target in pairs:
        shutil.copyfile(source, target)
    _check("tab_steering.tex", *(p.read_text(encoding="utf-8") for p in pairs[2]))
    return [target.name for _, target in pairs]


def knowledge_diagram() -> str:
    """The knowledge direction's diagram, with its equation chip naming the direction by symbol."""
    source = (THESIS / "generated" / "fig_knowledge.tex").read_text(encoding="utf-8")
    header = ("% The knowledge direction's construction (copied by make_ieee_assets.py from the thesis's\n"
              "% fig_knowledge.tex). The same denial is written into the model's reply after a question it\n"
              "% knows and after one nobody can know; the direction is the difference of the two means.\n")
    body = source[source.index("\\ifdefined"):]
    if body.count("$\\mathrm{K}=") != 1:
        raise AssertionError("fig_knowledge.tex no longer has the one equation chip this script renames")
    out = header + body.replace("$\\mathrm{K}=", "$\\hat v_{\\mathrm{know}}=")
    _check("fig_knowledge.tex", source, out)
    (GENERATED / "fig_knowledge.tex").write_text(out, encoding="utf-8")
    return "fig_knowledge.tex"


def knowledge_table() -> str:
    """AUROC of each detector on every set at layer 40: the header names the detectors."""
    source = (THESIS / "generated" / "tab_knowledge.tex").read_text(encoding="utf-8")
    old = "Evaluation set & Mean Probe & probe, layer 40 & D9 & K & D9 + K \\\\"
    new = ("Evaluation set & Mean Probe & \\shortstack{base probe,\\\\layer 40} & \\shortstack{extended\\\\contrast}"
           " & \\shortstack{knowledge\\\\direction} & \\shortstack{selected\\\\detector} \\\\")
    if source.count(old) != 1:
        raise AssertionError("tab_knowledge.tex no longer has the header this script renames")
    out = source.replace(old, new)
    _check("tab_knowledge.tex", source, out)
    (GENERATED / "tab_knowledge.tex").write_text(out, encoding="utf-8")
    return "tab_knowledge.tex"


def designs_table() -> str:
    """The designs against the criteria: the label column goes and the rows keep their names."""
    source = (THESIS / "generated" / "tab_designs.tex").read_text(encoding="utf-8")
    replacements = (("\\begin{tabular}{@{}l l l cc cc cc@{}}", "\\begin{tabular}{@{}l l cc cc cc@{}}"),
                    ("\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
                     "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}"))
    for old, new in replacements:
        if source.count(old) != 1:
            raise AssertionError(f"tab_designs.tex no longer has {old!r}")
        source = source.replace(old, new)
    lines = []
    for line in source.splitlines():
        if line.startswith(" & & & "):  # the first header row: one empty cell fewer
            line = line[len(" & "):]
        elif line.startswith(" & Design & "):  # the second header row
            line = line[len(" & "):]
        elif DESIGN_LABEL.match(line):
            line = DESIGN_LABEL.sub("", line)
            name = line.split(" & ", 1)[0]
            line = DESIGN_ROWS.get(name, name) + line[len(name):]
        lines.append(line)
    out = "\n".join(lines) + "\n"
    _check("tab_designs.tex", (THESIS / "generated" / "tab_designs.tex").read_text(encoding="utf-8"), out)
    (GENERATED / "tab_designs.tex").write_text(out, encoding="utf-8")
    return "tab_designs.tex"


def _rename(fig) -> None:
    """Rename every design label in `fig`. Legend entries and titles are renamed in place; a tick
    label is redrawn from its axis's formatter, so tick labels are renamed through the axis."""
    from matplotlib.text import Text

    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labels = [t.get_text() for t in axis.get_ticklabels()]
            if any(label in PLOT_NAMES for label in labels):
                axis.set_ticks(axis.get_ticklocs(), [PLOT_NAMES.get(label, label) for label in labels])
    for text in fig.findobj(Text):
        name = PLOT_NAMES.get(text.get_text())
        if name is not None:
            text.set_text(name)
    left = [t.get_text() for t in fig.findobj(Text) if LABEL.search(t.get_text())]
    for ax in fig.axes:
        left += [t.get_text() for axis in (ax.xaxis, ax.yaxis) for t in axis.get_ticklabels()
                 if LABEL.search(t.get_text())]
    if left:
        raise AssertionError(f"design labels left in a plot: {left}")


@contextmanager
def _renamed_plots():
    """Inside, the thesis's plotting functions save into this folder, with their labels renamed."""
    from matplotlib.figure import Figure

    save, folder = Figure.savefig, T.IMAGES

    def renamed_save(fig, path, *args, **kwargs):
        _rename(fig)
        return save(fig, path, *args, **kwargs)

    Figure.savefig, T.IMAGES = renamed_save, IMAGES
    try:
        yield
    finally:
        Figure.savefig, T.IMAGES = save, folder


def plots() -> list[str]:
    IMAGES.mkdir(exist_ok=True)
    with _renamed_plots():
        T.knowledge_hpkr_figure(T.hpkr_slice_curves())
        T.knowledge_gs_figure()
        T.transfer_selected_figure()
        T.selected_depth_figure()
    return ["fig_knowledge_hpkr.pdf", "fig_knowledge_gs.pdf", "fig_transfer_selected.pdf", "fig_selected_depth.pdf"]


def main() -> None:
    written = copies() + [knowledge_diagram(), knowledge_table(), designs_table()] + plots()
    print("wrote", ", ".join(written))


if __name__ == "__main__":
    main()
