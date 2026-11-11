"""The arXiv version's assets, taken from the thesis's own generators (paper/thesis/) and from the
result files of the studies the thesis does not cover.

The paper names every detector in words, never by its thesis label (D1 to D9, K, D9 + K). This script

1. copies the thesis's numbers (both files), its figure style and its bibliography into `generated/`
   and `refs.bib`, so every number the paper shares with the thesis is the thesis's own `\\V{key}`;
2. copies the generated tables the paper uses, with each design label replaced by its name; only
   header and row-name cells change, and the run stops unless every number survives, in order, and
   no label is left;
3. redraws the plots whose text carries a label, through the thesis's own plotting functions with
   their text renamed just before each is saved (the IEEE short version's hook);
4. adds the numbers, tables and one figure for the studies outside the thesis: Aletheia's Quest,
   the 150-question extension of the persona bank, the random persona-bank control and the AISI
   held-out comparison against the same-layer probe. Each claim the text makes about them is
   asserted here, so a changed result file stops the build instead of leaving a stale sentence.

The thesis folder is only read. Run from `src/probes/deception-detection`:

    python paper/arxiv/make_arxiv_assets.py
"""

import re
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # repo root
THESIS = ROOT / "paper" / "thesis"
IEEE = ROOT / "paper" / "ieee"
RESULTS = ROOT / "results"
GENERATED = HERE / "generated"
IMAGES = HERE / "images"

sys.path.insert(0, str(THESIS))
sys.path.insert(0, str(IEEE))
import make_ieee_assets as I  # noqa: E402  the short version's renaming hook and label guard
import make_thesis_assets as T  # noqa: E402  the thesis's generator: its data, sizes and colours
import make_xmodel_assets as X  # noqa: E402  the cross-model section's generator

I.IMAGES = IMAGES  # the renaming hook saves the thesis's plots here

# Plot text the thesis prints that the IEEE map does not cover (the cross-model depth figure).
I.PLOT_NAMES.update({
    "D9 transfer (off-diagonal AUROC)": "extended contrast: transfer (off-diagonal AUROC)",
    "D9 zero-shot, Liars' Bench mean AUROC": "extended contrast: zero-shot mean AUROC",
    "K on HP-KR's identical-text slice": "knowledge direction: HP-KR identical text",
})

# Design labels in table text, replaced in this order (longer phrases first). Lower-case names are
# capitalised afterwards where they open a cell.
TOKENS = (
    (r"Base probe D2", "Base probe"),
    (r"\bD3, trained on the model's own lies", "Base probe, trained on the model's own lies"),
    (r"Probe on raw activations \(D1\)", "Probe on raw activations"),
    (r"Base probe D2, same block", "Base probe, same block"),
    (r"Published contrast D5", "Published contrast"),
    (r"Knowledge-denial contrast D8", "Knowledge-denial contrast"),
    (r"Extended contrast D9", "Extended contrast"),
    (r"published contrast D5", "published contrast"),
    (r"\(D9 above D5\)", "(extended above published)"),
    (r"Extended contrast \+ K, selected \(1-d\)", "Selected detector (1-d)"),
    (r"\(D5\)", "(published contrast)"),
    (r"\(D8\)", "(knowledge-denial contrast)"),
    (r"D9 \+ K's", "selected detector's"),
    (r"D9 \+ K", "selected detector"),
    (r"D5 \+ K", "published contrast + knowledge direction"),
    (r"D9's", "extended contrast's"),
    (r"D5's", "published contrast's"),
    (r"D8's", "knowledge-denial contrast's"),
    (r"\bD1\b", "probe on raw activations"),
    (r"\bD2\b", "base probe"),
    (r"\bD3\b", "base probe, own lies"),
    (r"\bD4\b", "persona PCs"),
    (r"\bD5\b", "published contrast"),
    (r"\bD6\b", "sycophancy contrast"),
    (r"\bD7\b", "unified contrast"),
    (r"\bD8\b", "knowledge-denial contrast"),
    (r"\bD9\b", "extended contrast"),
    (r"K\$'\$", "knowledge direction, new questions"),
    (r"Template-free K\b", "Template-free knowledge direction"),
    (r"Template K's", "template knowledge direction's"),
    (r"(?<![A-Za-z\\\-])K's", "knowledge direction's"),
    (r"(?<![A-Za-z\\\-])K(?![A-Za-z'$])", "knowledge direction"),
)
CELL_NAMES = ("selected", "published", "knowledge", "knowledge-denial", "extended", "base", "probe",
              "sycophancy", "unified", "persona")


def _relabel(text: str, extra: tuple[tuple[str, str], ...] = ()) -> str:
    out = []
    for line in text.splitlines():
        if line.lstrip().startswith("%"):
            out.append(line)
            continue
        for pattern, name in extra + TOKENS:
            line = re.sub(pattern, name, line)
        cells = line.split(" & ")
        for i, cell in enumerate(cells):
            stripped = cell.lstrip()
            first = stripped.split(" ", 1)[0].rstrip("'s:,")
            if first.lower() in CELL_NAMES and stripped[:1].islower():
                lead = cell[: len(cell) - len(stripped)]
                cells[i] = lead + stripped[0].upper() + stripped[1:]
        out.append(" & ".join(cells))
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


SHORT_HEADERS = {"Published contrast": "\\shortstack{Published\\\\contrast}",
                 "Knowledge-denial contrast": "\\shortstack{Knowledge-\\\\denial contrast}",
                 "Extended contrast": "\\shortstack{Extended\\\\contrast}",
                 "Knowledge direction": "\\shortstack{Knowledge\\\\direction}",
                 "Selected detector": "\\shortstack{Selected\\\\detector}"}


def _shortstack_headers(text: str) -> str:
    """In a wide table's header rows (before the first \\midrule), stack the two-word names."""
    out, in_header = [], True
    for line in text.splitlines():
        if "\\midrule" in line:
            in_header = False
        if in_header and " & " in line:
            for name, stacked in SHORT_HEADERS.items():
                line = line.replace(f" & {name} & ", f" & {stacked} & ").replace(f" & {name} \\\\", f" & {stacked} \\\\")
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def relabelled_table(source: Path, target_name: str, extra: tuple[tuple[str, str], ...] = (),
                     stack_headers: bool = False, column_spec: tuple[str, str] | None = None) -> str:
    before = source.read_text(encoding="utf-8")
    after = _relabel(before, extra)
    if stack_headers:
        after = _shortstack_headers(after)
    I._check(target_name, before, after)
    if column_spec is not None:  # a layout change to the preamble, applied after the number guard
        old, new = column_spec
        if after.count(old) != 1:
            raise AssertionError(f"{target_name}: column spec {old!r} not found once")
        after = after.replace(old, new)
    (GENERATED / target_name).write_text(after, encoding="utf-8")
    return target_name


# --- 1. copies -------------------------------------------------------------------------------------

def copies() -> list[str]:
    GENERATED.mkdir(exist_ok=True)
    IMAGES.mkdir(exist_ok=True)
    pairs = [(THESIS / "generated" / "numbers.tex", GENERATED / "numbers.tex"),
             (THESIS / "generated_xmodel" / "numbers.tex", GENERATED / "numbers_xmodel.tex"),
             (THESIS / "generated" / "fig_style.tex", GENERATED / "fig_style.tex")]
    for source, target in pairs:
        shutil.copyfile(source, target)
    # The figure style declares TeX Gyre Heros through XeTeX's TU encoding; the paper is built for
    # pdflatex (arXiv) under T1, so its figures use the document's sans-serif face instead.
    style = (GENERATED / "fig_style.tex").read_text(encoding="utf-8")
    old = "\\newcommand{\\figsf}{\\fontfamily{pipeheros}}%"
    if style.count(old) != 1:
        raise AssertionError("fig_style.tex no longer defines \\figsf the way this script expects")
    style = re.sub(r"\\DeclareFont(Family|Shape)\{TU\}\{pipeheros\}[^\n]*\n", "", style)
    (GENERATED / "fig_style.tex").write_text(style.replace(old, "\\newcommand{\\figsf}{\\sffamily}%"),
                                             encoding="utf-8")
    bib = (THESIS / "bibliography" / "references.bib").read_text(encoding="utf-8")
    # natbib's author-year labels need a year; the dataset entry has only an access date
    old = "  howpublished = {Hugging Face dataset, \\url{https://huggingface.co/datasets/ai-safety-institute/lie-detection-rollouts}, revision 177a816},"
    if bib.count(old) != 1:
        raise AssertionError("references.bib: the AISI dataset entry changed")
    bib = bib.replace(old, old + "\n  year         = {2026},")
    extra = HERE / "extra_refs.bib"
    if extra.exists():
        bib = bib.rstrip() + "\n\n" + extra.read_text(encoding="utf-8")
    (HERE / "refs.bib").write_text(bib, encoding="utf-8")
    return [t.name for _, t in pairs] + ["refs.bib"]


def knowledge_diagram() -> str:
    """The knowledge direction's diagram, its equation chip naming the direction by the paper's symbol."""
    source = (THESIS / "generated" / "fig_knowledge.tex").read_text(encoding="utf-8")
    header = ("% The knowledge direction's construction (copied by make_arxiv_assets.py from the thesis's\n"
              "% fig_knowledge.tex). The same denial is written into the model's reply after a question it\n"
              "% knows and after one nobody can know; the direction is the difference of the two means.\n")
    body = source[source.index("\\ifdefined"):]
    if body.count("$\\mathrm{K}=") != 1:
        raise AssertionError("fig_knowledge.tex no longer has the one equation chip this script renames")
    out = header + body.replace("$\\mathrm{K}=", "$\\hat v_{\\mathrm{know}}=")
    I._check("fig_knowledge.tex", source, out)
    (GENERATED / "fig_knowledge.tex").write_text(out, encoding="utf-8")
    return "fig_knowledge.tex"


# --- 2. the thesis's tables, relabelled -----------------------------------------------------------------

def thesis_tables() -> list[str]:
    written = []
    # the IEEE script's two renamings, written into this folder
    I.GENERATED = GENERATED
    written += [I.knowledge_table(), I.designs_table()]
    gen = THESIS / "generated"
    for name in ("tab_transfer", "tab_zeroshot", "tab_hpkr", "tab_datasets", "tab_steering", "tab_k_paired",
                 "tab_k_depth", "tab_k_operating", "tab_knowledge_robust", "tab_oracle", "tab_operating",
                 "tab_reads", "tab_followup", "tab_kthree", "tab_syc", "tab_pertest", "tab_roster",
                 "tab_k_questions"):
        written.append(relabelled_table(gen / f"{name}.tex", f"{name}.tex"))
    xm = THESIS / "generated_xmodel"
    for name in ("tab_xm_models", "tab_xm_claims", "tab_xm_mid", "tab_xm_wording", "tab_xm_organisms",
                 "tab_xm_roleplay", "tab_xm_intervals", "tab_xm_zeroshot", "tab_xm_transfer", "tab_xm_operating",
                 "tab_xm_aisi", "tab_xm_banks", "tab_xm_checks"):
        extra = ((r"the thesis's layer", "Llama's layer"), (r"the thesis's", "Llama's"), (r"The thesis's", "Llama's"))
        # the claim column is made wide enough for one line per claim; the float is rescaled to the text width
        spec = ("p{5.1cm}", "p{7.2cm}") if name == "tab_xm_claims" else None
        wide = name in ("tab_xm_wording", "tab_xm_organisms", "tab_xm_transfer", "tab_xm_zeroshot", "tab_xm_operating")
        written.append(relabelled_table(xm / f"{name}.tex", f"{name}.tex", extra, stack_headers=wide, column_spec=spec))
    return written


# --- 3. the thesis's plots, redrawn with names ----------------------------------------------------------

@contextmanager
def _xmodel_images():
    folder = X.OUT_FIG
    X.OUT_FIG = IMAGES
    try:
        yield
    finally:
        X.OUT_FIG = folder


def plots() -> list[str]:
    with I._renamed_plots():
        T.knowledge_hpkr_figure(T.hpkr_slice_curves())
        T.knowledge_gs_figure()
        T.transfer_selected_figure()
        T.selected_depth_figure()
        with _xmodel_images():
            X.depth_figure(X.depth(), X.k_depth())
    for stale in ("fig_xmodel_depth.png",):
        (IMAGES / stale).unlink(missing_ok=True)
    return ["fig_knowledge_hpkr.pdf", "fig_knowledge_gs.pdf", "fig_transfer_selected.pdf", "fig_selected_depth.pdf",
            "fig_xmodel_depth.pdf"]


# --- 4. the studies outside the thesis -------------------------------------------------------------------

def _round3(x: float) -> float:
    return float(T.M.fmt(x, 3))


def aletheia(n) -> list[str]:
    """Aletheia's Quest: the per-dataset table, the per-dataset figure and the ax-* numbers."""
    from experiments.external.aletheia import report as AR
    from experiments.external.aletheia import spec as S

    table = pd.read_csv(RESULTS / "aletheia" / "auroc.csv")
    board = AR._board()
    keys = list(AR.KEYS)
    cells = {"headline": "headline", "d9": "S1-d9", "repe": "R1-repe", "reasoning-d9": "S2-reasoning-d9",
             "reasoning": "S2-reasoning", "k": "S6-k", "d5": "S6-d5", "d8": "S6-d8", "base": "S3-base"}
    rows = {k: AR._cell_rows(table, c) for k, c in cells.items()}
    read = [k for k in keys if k in rows["headline"].index]  # the datasets with an organism read
    white = board[board["track"] == "white-box"]
    board_means = board[read].mean(axis=1)

    def rank_of(value: float) -> int:
        return int(1 + (board_means > value).sum())

    for k, cell in cells.items():
        r = rows[k].loc[read, "auroc"]
        n.put(f"ax-{k}-mean", float(r.mean()))
        n.put(f"ax-{k}-rank", rank_of(float(r.mean())))
    head = rows["headline"]
    boots = dict(np.load(RESULTS / "aletheia" / "boots.npz"))
    means = AR._boot_mean(boots, "headline", read)
    n.put("ax-headline-lo", float(np.percentile(means, 2.5)))
    n.put("ax-headline-hi", float(np.percentile(means, 97.5)))
    ranks = np.array([rank_of(m) for m in means])
    r_lo, r_hi = np.quantile(ranks, [0.025, 0.975])  # the report's rank interval
    n.put("ax-rank-lo", int(r_lo))
    n.put("ax-rank-hi", int(r_hi))
    assert (int(r_lo), int(r_hi)) == (13, 20), "the headline's rank interval moved"
    n.put("ax-n-read", len(read))
    n.put("ax-n-entries", len(board))
    for d in read:
        key = d.lower().replace("-", "")
        n.put(f"ax-{key}-headline", float(head.loc[d, "auroc"]))
        n.put(f"ax-{key}-d9", float(rows["d9"].loc[d, "auroc"]))
        n.put(f"ax-{key}-k", float(rows["k"].loc[d, "auroc"]))
        n.put(f"ax-{key}-band", float(head.loc[d, "band97.5"]))
    n.put("ax-echo-base", float(rows["base"].loc["Echo", "auroc"]))
    # the claims the text makes
    assert round(float(head.loc[read, "auroc"].mean()), 3) == 0.657, "the headline mean moved"
    assert rank_of(float(head.loc[read, "auroc"].mean())) == 16
    assert rank_of(float(rows["repe"].loc[read, "auroc"].mean())) < rank_of(float(rows["d9"].loc[read, "auroc"].mean())) \
        < rank_of(float(head.loc[read, "auroc"].mean())), "the ordering probe < extended contrast < selected detector moved"
    assert float(head.loc["Echo", "auroc"]) < 0.25 and float(rows["d9"].loc["Echo", "auroc"]) > 0.5
    # the table: one row per dataset, the two unread ones included
    lines = ["% Generated by make_arxiv_assets.py from results/aletheia/auroc.csv and the published leaderboard",
             "\\begin{tabular}{@{}l r ccc c c@{}}", "\\toprule",
             "Dataset & $n$ (lies) & \\shortstack{Selected\\\\detector} & \\shortstack{Extended\\\\contrast} & "
             "\\shortstack{RepE probe,\\\\same block} & \\shortstack{Random\\\\97.5\\%} & \\shortstack{Best\\\\entry} \\\\",
             "\\midrule"]
    counts = {d.key: (d.n, d.lies) for d in S.DATASETS} if hasattr(S, "DATASETS") else {}
    for d in keys:
        if d in read:
            h = head.loc[d]
            nrow, lies = int(h["n"]), int(h["lies"])
            lines.append(f"{d} & {nrow} ({lies}) & {T.M.fmt(h['auroc'], 3)} [{T.M.fmt(h['lo'], 2)}, {T.M.fmt(h['hi'], 2)}]"
                         f" & {T.M.fmt(rows['d9'].loc[d, 'auroc'], 3)} & {T.M.fmt(rows['repe'].loc[d, 'auroc'], 3)}"
                         f" & {T.M.fmt(h['band97.5'], 3)} & {T.M.fmt(white[d].max(), 3)} \\\\")
        else:
            nrow, lies = counts.get(d, ("", ""))
            lines.append(f"{d} & {nrow} ({lies}) & \\multicolumn{{4}}{{c}}{{not read (gated adapter)}} & "
                         f"{T.M.fmt(white[d].max(), 3)} \\\\")
    lines += ["\\midrule",
              f"Mean over the {len(read)} read & & {T.M.fmt(head.loc[read, 'auroc'].mean(), 3)} "
              f"[{T.M.fmt(np.percentile(means, 2.5), 2)}, {T.M.fmt(np.percentile(means, 97.5), 2)}]"
              f" & {T.M.fmt(rows['d9'].loc[read, 'auroc'].mean(), 3)} & {T.M.fmt(rows['repe'].loc[read, 'auroc'].mean(), 3)}"
              f" & & {T.M.fmt(board_means.max(), 3)} \\\\",
              f"Rank among the {len(board)} entries & & {rank_of(float(head.loc[read, 'auroc'].mean()))}"
              f" & {rank_of(float(rows['d9'].loc[read, 'auroc'].mean()))} & {rank_of(float(rows['repe'].loc[read, 'auroc'].mean()))}"
              f" & & 1 \\\\",
              "\\bottomrule", "\\end{tabular}"]
    (GENERATED / "tab_aletheia.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    aletheia_figure(rows, read, white, board)
    return ["tab_aletheia.tex", "fig_aletheia.pdf"]


def aletheia_figure(rows: dict, read: list[str], white: pd.DataFrame, board: pd.DataFrame) -> None:
    import make_figures as MF
    from matplotlib import pyplot as plt

    head, d9, repe = rows["headline"], rows["d9"], rows["repe"]
    x = np.arange(len(read))
    width = 0.26
    fig, ax = plt.subplots(figsize=(MF.TEXT_WIDTH * 0.85, 2.3), constrained_layout=True)
    for d_i, d in enumerate(read):  # the random band of the organism read, per dataset
        band = float(head.loc[d, "band97.5"])
        ax.add_patch(plt.Rectangle((d_i - 0.5, 1 - band), 1, 2 * band - 1, color=MF.F.NEUTRAL, lw=0, zorder=0))
    series = [("selected detector", head, T.COMBO_COLOUR), ("extended contrast", d9, MF.ARM_COLOURS["ext"]),
              ("RepE probe, same block", repe, MF.ARM_COLOURS["base"])]
    for j, (label, r, colour) in enumerate(series):
        vals = [float(r.loc[d, "auroc"]) for d in read]
        err = None
        if label == "selected detector":
            err = [[float(r.loc[d, "auroc"] - r.loc[d, "lo"]) for d in read],
                   [float(r.loc[d, "hi"] - r.loc[d, "auroc"]) for d in read]]
        ax.bar(x + (j - 1) * width, vals, width * 0.92, color=colour, label=label, zorder=2, yerr=err,
               error_kw={"lw": 0.7, "capsize": 1.5, "ecolor": MF.F.INK_MUTED})
    ax.scatter(x, [float(white[d].max()) for d in read], marker="_", s=180, color=MF.F.INK, lw=1.4, zorder=3,
               label="best white-box entry")
    ax.axhline(0.5, color=MF.F.INK_MUTED, lw=0.7, ls=(0, (4, 3)))
    ax.set_xticks(x, read, fontsize=7)
    ax.set_xlim(-0.6, len(read) - 0.4)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUROC")
    MF.clean(ax)
    fig.legend(loc="outside lower center", ncol=4, fontsize=6.3, frameon=False)
    fig.savefig(IMAGES / "fig_aletheia.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def question_bank(n) -> list[str]:
    """The 150-question extension: the pre-registered tests, the family means, the confirmation read."""
    folder = RESULTS / "question_bank" / "llama"
    verdict = pd.read_csv(folder / "verdict.csv")
    values = pd.read_csv(folder / "values.csv")
    paired = pd.read_csv(folder / "paired_ci_L40.csv")
    confirm = pd.read_csv(folder / "confirm.csv")
    families = pd.read_csv(folder / "spectrum_pooled_vs_family_L40.csv")
    fits = pd.read_csv(folder / "spectrum_fits_L40.csv").set_index("probe")
    lodo = pd.read_csv(folder / "spectrum_lodo_L40.csv")
    keep = pd.read_csv(folder / "keep_rates_s3.csv")

    v40 = values[values["layer"] == 40].set_index(["bank", "direction"])
    for bank, key in (("B30", "b30"), ("B30+30mono", "b30mono"), ("B30+90div", "b30div"), ("B150", "b150")):
        for direction, dkey in (("D9", "d9"), ("D9 + K", "ninek")):
            r = v40.loc[(bank, direction)]
            n.put(f"qb-{key}-{dkey}-lb", float(r["LB-5 mean"]))
            n.put(f"qb-{key}-{dkey}-hpkr", float(r["auroc HP-KR"]))
            n.put(f"qb-{key}-{dkey}-cg", float(r["auroc CG"]))
    tests = verdict.set_index("test")
    for t in ("T1", "T2", "T3", "T4"):
        n.put(f"qb-{t.lower()}-pass", "passes" if bool(tests.loc[t, "pass"]) else "fails")
        n.put(f"qb-{t.lower()}-p", float(tests.loc[t, "p"]), 3)
    assert not tests.loc["T1", "pass"] and not tests.loc["T2", "pass"] and not tests.loc["T3", "pass"]
    assert tests.loc["T4", "pass"] and tests.loc["H2", "pass"] and tests.loc["H5", "pass"]
    t3 = re.search(r"= (-?[\d.]+), CI \[(-?[\d.]+), (-?[\d.]+)\]", tests.loc["T3", "statistic"])
    n.put("qb-t3-delta", float(t3.group(1)))
    n.put("qb-t3-lo", float(t3.group(2)))
    n.put("qb-t3-hi", float(t3.group(3)))
    lr = fits.loc["LR"]
    n.put("qb-t4-r2-sigma", float(lr["r2_cos_sigma_target"]), 2)
    n.put("qb-t4-r2-plain", float(lr["r2_cos_plain"]), 2)
    n.put("qb-t4-delta", float(lr["delta_r2_target_minus_plain"]), 2)
    n.put("qb-t4-lo", float(lr["delta_r2_lo"]), 2)
    n.put("qb-t4-hi", float(lr["delta_r2_hi"]), 2)
    dom = fits.loc["DoM"]
    n.put("qb-t4-r2-sigma-dom", float(dom["r2_cos_sigma_target"]), 2)
    n.put("qb-t4-r2-plain-dom", float(dom["r2_cos_plain"]), 2)
    t1 = re.search(r"LB-5 ([\d.]+) vs 200 draws .* \(median ([\d.]+)\)", tests.loc["T1", "statistic"])
    n.put("qb-t1-mono", float(t1.group(1)))
    n.put("qb-t1-median", float(t1.group(2)))
    # B150 against B30 at layer 40, paired
    def delta(direction: str, metric: str) -> pd.Series:
        r = paired[(paired["direction"] == direction) & (paired["metric"] == metric) & (paired["a"] == "B150")
                   & (paired["b"] == "B30")]
        if r.empty:
            raise KeyError(f"no paired row B150 - B30 for {direction} {metric}")
        return r.iloc[0]
    for direction, dkey in (("D9", "d9"), ("D9 + K", "ninek")):
        for metric, mkey in (("auroc CG", "cg"), ("auroc HP-KR", "hpkr"), ("LB-5 mean", "lb")):
            r = delta(direction, metric)
            n.put(f"qb-b150-b30-{dkey}-{mkey}", float(r["delta"]))
            n.put(f"qb-b150-b30-{dkey}-{mkey}-lo", float(r["lo"]))
            n.put(f"qb-b150-b30-{dkey}-{mkey}-hi", float(r["hi"]))
    assert n["qb-b150-b30-d9-hpkr-hi"].startswith("\\ensuremath{-}"), "the 150-question bank no longer lowers HP-KR"
    # families
    fam = families.set_index("family")
    best = fam.drop(index="pooled (all)")["LB-5 mean"].idxmax()
    n.put("qb-best-family-lb", float(fam.loc[best, "LB-5 mean"]))
    n.put("qb-pooled-lb", float(fam.loc["pooled (all)", "LB-5 mean"]))
    n.put("qb-worst-family-lb", float(fam.drop(index="pooled (all)")["LB-5 mean"].min()))
    assert best == "orig_knowledge"
    # confirmation read at layer 40
    c40 = confirm[confirm["layer"] == 40].set_index(["set", "direction"])
    for s, skey in (("AISI held out", "aisi"), ("Gender-Secret", "gs")):
        for direction, dkey in (("D9", "d9"), ("D9 + K", "ninek")):
            r = c40.loc[(s, direction)]
            n.put(f"qb-confirm-{skey}-{dkey}", float(r["auroc"]))
            n.put(f"qb-confirm-{skey}-{dkey}-b30", float(r["B30 auroc"]))
            n.put(f"qb-confirm-{skey}-{dkey}-delta", float(r["delta"]))
            n.put(f"qb-confirm-{skey}-{dkey}-lo", float(r["lo"]))
            n.put(f"qb-confirm-{skey}-{dkey}-hi", float(r["hi"]))
    assert float(c40.loc[("Gender-Secret", "D9 + K"), "hi"]) < 0 and float(c40.loc[("AISI held out", "D9 + K"), "lo"]) > 0
    # the run
    n.put("qb-rollouts", 27000)
    n.put("qb-new-rollouts", 21600)
    n.put("qb-questions", 150)
    n.put("qb-new-questions", 120)
    judged = keep[keep["judged"] > 0]  # the unjudged `default` role keeps every row
    n.put("qb-judged-rows", int(judged["judged"].sum()))
    n.put("qb-kept-share", 100 * float(judged["kept"].sum() / judged["judged"].sum()), 1)
    # tables
    names = {"orig_disclosure": "Original: disclosure", "orig_sycophancy": "Original: sycophancy",
             "orig_knowledge": "Original: knowledge", "mono": "Monotone (new, same form)",
             "imt_quality": "Quality (false assertion)", "imt_quantity": "Quantity (omission)",
             "imt_relation": "Relation (evasion)", "imt_manner": "Manner (equivocation)",
             "imt_denial": "Denial of knowledge", "imt_ignorance": "Sincere ignorance", "pooled (all)": "Pooled, all 150"}
    lodo_map = dict(zip(lodo.iloc[:, 0], lodo.iloc[:, -1])) if len(lodo.columns) >= 2 else {}
    lines = ["% Generated by make_arxiv_assets.py from results/question_bank/llama/spectrum_pooled_vs_family_L40.csv",
             "\\begin{tabular}{@{}l r c c@{}}", "\\toprule",
             "Question family & Questions & \\shortstack{Five-dataset\\\\mean AUROC} & \\shortstack{Leaving it out:\\\\change vs pooled} \\\\",
             "\\midrule"]
    order = fam.drop(index="pooled (all)").sort_values("LB-5 mean", ascending=False).index.tolist() + ["pooled (all)"]
    for f in order:
        change = lodo_map.get(f)
        ch = "" if change is None or f == "pooled (all)" else _signed(float(change))
        lines.append(f"{names.get(f, f)} & {int(fam.loc[f, 'n_questions'])} & {T.M.fmt(fam.loc[f, 'LB-5 mean'], 3)} & {ch} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GENERATED / "tab_qbank_families.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # the tests table
    words = {"T1": ("Variety beats count (extended contrast)", "30 monotone vs 200 random 30-of-90 diverse draws"),
             "T2": ("Variety beats count (selected detector)", "the same, for the selected detector"),
             "H2": ("Monotone additions behave like the originals", "paired interval within $\\pm$0.03; inside the random-subset band"),
             "T3": ("Pooling all 150 beats the best single family", "150-question contrast minus the best family"),
             "T4": ("Covariance-weighted cosine predicts transfer better than plain cosine",
                    "$R^2$ over 60 family $\\times$ set cells, family bootstrap"),
             "H5": ("More questions help on HP-KR's identical text (exploratory)", "150 vs 30 questions, paired")}
    lines = ["% Generated by make_arxiv_assets.py from results/question_bank/llama/verdict.csv",
             "\\begin{tabular}{@{}l >{\\raggedright\\arraybackslash}p{5.6cm} l@{}}", "\\toprule",
             "Pre-registered test & Statistic & Result \\\\", "\\midrule"]
    for t in ("T1", "T2", "H2", "T3", "T4", "H5"):
        r = tests.loc[t]
        stat = str(r["statistic"]).replace("LB-5", "five-dataset mean").replace("B30+30mono", "30 originals + 30 monotone") \
            .replace("B30+rand30div", "30 originals + 30 random diverse").replace("B30mono", "the monotone 30") \
            .replace("B30orig", "the original 30").replace("rand30of150", "random 30 of 150") \
            .replace("fam_orig_knowledge", "original knowledge family").replace("B150", "all 150").replace("B30", "the 30 originals") \
            .replace("R2 sigma", "$R^2$ with the target's covariance").replace("vs plain", "vs plain cosine") \
            .replace("+-0.03", "$\\pm$0.03").replace(">=", "$\\geq$").replace("~", "$\\approx$") \
            .replace(" - ", " minus ").replace("delta", "difference")
        stat = re.sub(r"(?<![\d.])-(?=\d)", "$-$", stat)  # a typeset minus before every negative number
        if pd.isna(r["p"]):
            p = ""
        else:
            hp = float(r["holm_p"])
            p = ", $p < 0.001$ (Holm)" if hp < 0.0005 else f", $p$ = {T.M.fmt(hp, 2)} (Holm)"
        result = "passes" if bool(r["pass"]) else "fails"
        lines.append(f"{words[t][0]} & {stat}{p} & {result} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GENERATED / "tab_qbank_tests.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # the confirmation table
    lines = ["% Generated by make_arxiv_assets.py from results/question_bank/llama/confirm.csv",
             "\\begin{tabular}{@{}l l l c c c@{}}", "\\toprule",
             "Layer & Set & Direction & 150 questions & 30 questions & Difference [95\\%] \\\\", "\\midrule"]
    for _, r in confirm.sort_values(["layer", "set", "direction"]).iterrows():
        name = {"D9": "extended contrast", "D9 + K": "selected detector"}[r["direction"]]
        lines.append(f"{int(r['layer'])} & {r['set']} & {name} & {T.M.fmt(r['auroc'], 3)} & {T.M.fmt(r['B30 auroc'], 3)}"
                     f" & {_signed(r['delta'])} [{_signed(r['lo'])}, {_signed(r['hi'])}] \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GENERATED / "tab_qbank_confirm.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ["tab_qbank_families.tex", "tab_qbank_tests.tex", "tab_qbank_confirm.tex"]


def _signed(value: float) -> str:
    s = T.M.fmt(float(value), 3)
    if s.startswith("-"):
        return "$-$" + s[1:]
    return "$+$" + s


def random_bank(n) -> list[str]:
    """The random persona-bank control: six pre-fixed tests of the extended contrast against the null
    of contrasts drawn from a bank of random roles built by the same pipeline."""
    folder = RESULTS / "xmodel" / "llama" / "random_bank" / "s3"
    v = pd.read_csv(folder / "verdict.csv").set_index("test")
    what = {"T1 transfer": ("Transfer (off-diagonal AUROC)", "extended contrast vs random 12 v 13 contrasts"),
            "T2 zero-shot": ("Zero-shot five-dataset mean", "extended contrast vs random 12 v 13 contrasts"),
            "T3 with K": ("Transfer with the knowledge direction", "selected detector vs random contrast + knowledge direction"),
            "T4 HP-KR": ("HP-KR, all rows, AUROC", "extended contrast vs random 12 v 13 contrasts"),
            "T5 roles": ("Gain from widening, five-dataset mean", "extended minus published contrast vs random 12 v 13 minus random 8 v 7 of the same split"),
            "T6 held out": ("AISI held-out lies, AUROC", "extended contrast vs random 12 v 13 contrasts")}
    keys = {"T1 transfer": "t1", "T2 zero-shot": "t2", "T3 with K": "t3", "T4 HP-KR": "t4", "T5 roles": "t5",
            "T6 held out": "t6"}
    lines = ["% Generated by make_arxiv_assets.py from results/xmodel/llama/random_bank/s3/verdict.csv",
             "\\begin{tabular}{@{}l l c c c c l@{}}", "\\toprule",
             "Test & Comparison & Designed bank & \\shortstack{Null\\\\median} & \\shortstack{Null\\\\97.5\\%} & "
             "\\shortstack{Holm\\\\$p$} & Verdict \\\\", "\\midrule"]
    for test, (title, comparison) in what.items():
        r = v.loc[test]
        k = keys[test]
        n.put(f"rb-{k}-d", float(r["D"]))
        n.put(f"rb-{k}-null-median", float(r["null median"]))
        n.put(f"rb-{k}-null", float(r["null 97.5%"]))
        n.put(f"rb-{k}-p", float(r["Holm p (9,999 splits)"]), 3)
        n.put(f"rb-{k}-verdict", str(r["verdict"]))
        shown = f"{_signed(r['D'])}" if k == "t5" else T.M.fmt(r["D"], 3)
        median = f"{_signed(r['null median'])}" if k == "t5" else T.M.fmt(r["null median"], 3)
        null = f"{_signed(r['null 97.5%'])}" if k == "t5" else T.M.fmt(r["null 97.5%"], 3)
        lines.append(f"{title} & {comparison} & {shown} & {median} & {null} & {T.M.fmt(r['Holm p (9,999 splits)'], 3)}"
                     f" & {r['verdict']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (GENERATED / "tab_random_bank.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert all(v.loc[t, "verdict"] == "pass" for t in ("T1 transfer", "T2 zero-shot", "T3 with K", "T6 held out"))
    assert all(v.loc[t, "verdict"] == "fail" for t in ("T4 HP-KR", "T5 roles"))
    n.put("rb-splits", 100)
    n.put("rb-roles", 35)
    return ["tab_random_bank.tex"]


def aisi_vs_probe(n) -> None:
    """Llama at layer 40: the directions against the RepE base probe trained at the same layer, on AISI's
    held-out lies (ai-*) and on HP-KR's identical-text slice (li-*)."""
    table = pd.read_csv(RESULTS / "xmodel" / "llama" / "k_final" / "vs_probe.csv")
    table = table[table["block"] == 40]
    scorers = (("D9", "d9"), ("D9 + original K", "ninek"), ("original K", "k"), ("rule K", "rulek"),
               ("D9 + rule K", "ninerulek"))
    for set_name, prefix in (("held out (AISI)", "ai"), ("HP-KR identical", "li")):
        t = table[table["set"] == set_name].set_index("scorer")
        if t.empty:
            raise KeyError(f"vs_probe.csv has no block-40 rows for {set_name!r}")
        n.put(f"{prefix}-probe", float(t["base probe auroc"].iloc[0]))
        for scorer, key in scorers:
            if scorer in t.index:
                r = t.loc[scorer]
                n.put(f"{prefix}-{key}", float(r["auroc"]))
                n.put(f"{prefix}-{key}-delta", float(r["delta"]))
                n.put(f"{prefix}-{key}-lo", float(r["lo"]))
                n.put(f"{prefix}-{key}-hi", float(r["hi"]))
    ai = table[table["set"] == "held out (AISI)"].set_index("scorer")
    assert float(ai.loc["D9", "lo"]) >= 0 and float(ai.loc["D9 + original K", "hi"]) < 0, \
        "AISI: the probe is no longer level with the extended contrast and above the selected detector"
    li = table[table["set"] == "HP-KR identical"].set_index("scorer")
    assert float(li.loc["original K", "lo"]) < 0 < float(li.loc["original K", "hi"]), \
        "identical text: K and the same-layer probe are no longer level on Llama"
    # the two directions in words, for the text
    n.put("li-k-vs-probe-ident", float(li.loc["original K", "delta"]))
    n.put("li-k-vs-probe-ident-lo", float(li.loc["original K", "lo"]))
    n.put("li-k-vs-probe-ident-hi", float(li.loc["original K", "hi"]))


def _put_interval(n, key: str, value: float, lo: float, hi: float, flip: bool = False) -> None:
    """A difference and its interval; flip=True reports it the other way round (b minus a), so that a
    sentence saying 'trails it by' can quote positive numbers."""
    if flip:
        value, lo, hi = -value, -hi, -lo
    n.put(key, float(value))
    n.put(f"{key}-lo", float(lo))
    n.put(f"{key}-hi", float(hi))


def extra_keys(n) -> None:
    """Numbers the text quotes that neither thesis generator writes."""
    # the raw probe's transfer matrix at layer 40: off-diagonal cells below chance
    m = pd.read_csv(RESULTS / "persona_ood_ext_L40" / "matrix_baseline.csv", index_col=0)
    values = m.to_numpy(float)
    off = values[~np.eye(len(values), dtype=bool)]
    n.put("tr-base-cells", int(off.size))
    n.put("tr-base-below", int((off < 0.5).sum()))
    assert abs(off.mean() - 0.7320) < 5e-4, "raw-probe transfer moved"
    # the extended contrast's lead over the published contrast in transfer, with HP-KR left out
    s = pd.read_csv(RESULTS / "persona_ood_ext_L40" / "summary.csv").set_index("features")
    n.put("tr-ext-over-dec-offx", float(s.loc["ext_axis1", "off_diagonal_excl_hpkr"] - s.loc["axis1", "off_diagonal_excl_hpkr"]))

    # the question extension: how close the four candidate banks are, and T1's corrected p
    folder = RESULTS / "question_bank" / "llama"
    v = pd.read_csv(folder / "values.csv")
    v = v[(v["layer"] == 40) & (v["direction"] == "D9 + K") & v["bank"].isin(["B30", "B30+30mono", "B30+90div", "B150"])]
    assert len(v) == 4, "the four candidate banks are not all in values.csv"
    spread = float(v["LB-5 mean"].max() - v["LB-5 mean"].min())
    n.put("qb-decision-spread", spread)
    assert spread < 0.005, "the candidate banks no longer tie under the pre-registered rule"
    tests = pd.read_csv(folder / "verdict.csv").set_index("test")
    n.put("qb-t1-holm", float(tests.loc["T1", "holm_p"]), 2)
    confirm = pd.read_csv(folder / "confirm.csv")
    r = confirm[(confirm["layer"] == 40) & (confirm["set"] == "Gender-Secret") & (confirm["direction"] == "D9 + K")].iloc[0]
    _put_interval(n, "qb-confirm-gs-ninek-gap", r["delta"], r["lo"], r["hi"], flip=True)

    # the random bank: the null's size and T5's uncorrected p
    rb = pd.read_csv(RESULTS / "xmodel" / "llama" / "random_bank" / "s3" / "verdict.csv").set_index("test")
    n.put("rb-null-splits", 99)
    n.put("rb-holm-splits", 9999)
    n.put("rb-t5-p-raw", float(rb.loc["T5 roles", "p (99 splits)"]), 2)

    # Aletheia: the post hoc cells' ranks
    ph = pd.read_csv(RESULTS / "aletheia" / "posthoc" / "summary.csv")
    n.put("ax-posthoc-rank-min", int(ph["rank of 31"].min()))
    n.put("ax-posthoc-rank-max", int(ph["rank of 31"].max()))

    # held-out lies on Llama: the selected detector trails the same-layer probe (reported as a gap)
    t = pd.read_csv(RESULTS / "xmodel" / "llama" / "k_final" / "vs_probe.csv")
    r = t[(t["block"] == 40) & (t["set"] == "held out (AISI)") & (t["scorer"] == "D9 + original K")].iloc[0]
    _put_interval(n, "ai-ninek-gap", r["delta"], r["lo"], r["hi"], flip=True)

    # the extended contrast against the base probe trained at the same block, Mistral and Gemma
    for model, block in (("mistral", 20), ("gemma", 31)):
        t = pd.read_csv(RESULTS / "xmodel" / model / "k_final" / "vs_probe.csv")
        r = t[(t["block"] == block) & (t["set"] == "dataset mean") & (t["scorer"] == "D9")].iloc[0]
        _put_interval(n, f"xs-{model}-dnine-probe", r["delta"], r["lo"], r["hi"])
    # Mistral: level with the probe; Gemma: above it
    assert n["xs-mistral-dnine-probe-lo"].startswith("\\ensuremath{-}") and not n["xs-mistral-dnine-probe-hi"].startswith("\\ensuremath")
    assert not n["xs-gemma-dnine-probe-lo"].startswith("\\ensuremath")

    # Qwen's second bank (role line in the user turn, built after the first Qwen results): transfer at
    # block 40. The paper's main Qwen numbers stay those of the first bank (the thesis's); this is the
    # robustness check on the role-play explanation.
    qwen = RESULTS / "xmodel" / "qwen"
    s = pd.read_csv(qwen / "persona_ood_s3u_B40" / "summary.csv").set_index("features")
    ext, raw, den = (float(s.loc[f, "off_diagonal_mean"]) for f in ("ext_axis1", "baseline", "denial_axis1"))
    n.put("xq-s3u-off-dnine", ext)
    n.put("xq-s3u-off-raw", raw)
    n.put("xq-s3u-off-deight", den)
    ci = None
    for name in ("transfer_ci.csv", "transfer_ci_s3u.csv"):
        path = qwen / name
        if path.exists():
            c = pd.read_csv(path)
            c = c[(c["layer"] == "L40") & (c["metric"] == "transfer") & (c["a"] == "D9") & (c["b"] == "raw probe")]
            if len(c) and abs(float(c["a_value"].iloc[0]) - ext) < 1e-3:
                ci = c.iloc[0]
                break
    if ci is None:
        raise AssertionError("no transfer interval for Qwen's user-turn bank (s3u) was found")
    _put_interval(n, "xq-s3u-ci-dnine-raw", ci["delta"], ci["lo"], ci["hi"])
    assert float(ci["hi"]) < 0 and den > raw, "Qwen's user-turn bank: the extended contrast no longer fails"
    # and the paper's main Qwen numbers are still the first (system-message, lenient) bank's
    s23 = pd.read_csv(qwen / "persona_ood_s23_B40" / "summary.csv").set_index("features")
    assert abs(float(s23.loc["ext_axis1", "off_diagonal_mean"]) - 0.4533) < 1e-3, "Qwen's first bank moved"
    # The Qwen paragraph of results_models.tex is written for the thesis's Qwen keys of 2026-10-15 (the
    # first bank). A regenerated cross-model file (the user-turn re-run made primary) must stop the build
    # until that paragraph is rewritten.
    xm = dict(re.findall(r"\\setV\{([^}]*)\}\{([^}]*)\}", (GENERATED / "numbers_xmodel.tex").read_text(encoding="utf-8")))
    expected = {"xm-qwen-roleplay": "28", "xm-qwen-off-dnine": "0.453", "xm-qwen-off-raw": "0.559",
                "xm-qwen-off-deight": "0.679"}
    changed = {k: (xm.get(k), v) for k, v in expected.items() if xm.get(k) != v}
    if changed:
        raise AssertionError(f"the thesis's Qwen numbers changed {changed}: rewrite the Qwen paragraph of "
                             "sections/results_models.tex (and the abstract's Qwen sentence) before rebuilding")

    # the persona bank's judged and kept rollouts (Llama): the anchor role is never judged
    rp = pd.read_csv(RESULTS / "xmodel" / "tables" / "roleplay.csv").set_index("model")
    judged, kept3 = int(rp.loc["Llama-3.3-70B", "judged"]), int(rp.loc["Llama-3.3-70B", "score 3"])
    n.put("bank-judged", judged)
    n.put("bank-judged-kept", kept3)
    th = dict(re.findall(r"\\setV\{([^}]*)\}\{([^}]*)\}", (GENERATED / "numbers.tex").read_text(encoding="utf-8")))
    generated, kept = (int(th[k].replace(",", "")) for k in ("bank-generated", "bank-kept"))
    n.put("bank-anchor-rows", generated - judged)
    assert kept == kept3 + (generated - judged), "the bank's kept count no longer equals judged-kept plus the anchor"

    # Qwen's role-play under both prompt formats (data/qwen_roleplay.csv, written by make_qwen_roleplay.py
    # from the judge's score files)
    rq = pd.read_csv(HERE / "data" / "qwen_roleplay.csv").set_index("format")
    sys_share = round(100 * float(rq.loc["system message", "share fully in role"]))
    n.put("xq-sys-roleplay", sys_share)
    n.put("xq-s3u-roleplay", round(100 * float(rq.loc["user turn", "share fully in role"])))
    n.put("xq-sys-roles-under", int(rq.loc["system message", "roles under 20"]))
    n.put("xq-strict-min", 20)
    assert str(sys_share) == xm["xm-qwen-roleplay"], "the system-message share no longer matches the thesis's"
    assert int(rq.loc["user turn", "roles under 20"]) == 0, "a role misses the strict filter in the user-turn bank"
    # Qwen's transfer with HP-KR left out, on both banks: the failure is HP-KR's
    for bank in ("s23", "s3u"):
        s = pd.read_csv(qwen / f"persona_ood_{bank}_B40" / "summary.csv").set_index("features")
        ext_x, raw_x = float(s.loc["ext_axis1", "off_diagonal_excl_hpkr"]), float(s.loc["baseline", "off_diagonal_excl_hpkr"])
        n.put(f"xq-{bank}-offx-dnine", ext_x)
        n.put(f"xq-{bank}-offx-raw", raw_x)
        assert ext_x > raw_x, f"Qwen {bank}: without HP-KR the extended contrast no longer leads the raw probe"

    # Llama's template-free knowledge direction on Gender-Secret (the rule-K table)
    rk = pd.read_csv(RESULTS / "xmodel" / "tables" / "rule_k.csv")
    r = rk[(rk["model"] == "Llama-3.3-70B") & (rk["metric"] == "Gender-Secret") & (rk["with"] == "K")].iloc[0]
    n.put("ru-llama-gs-rule", float(r["rule K (template-free)"]))
    assert float(r["rule K (template-free)"]) < float(th["th-k-gs-ft-band-L40"]), "the template-free K left the band"

    # the lenient banks against the strict ones (Llama, Mistral, Gemma), unrounded
    bc = pd.read_csv(RESULTS / "xmodel" / "tables" / "bank_comparison.csv")
    cols = ["D5 LB mean", "D8 LB mean", "D9 LB mean", "D9 transfer"]
    gaps = []
    for model in ("Llama-3.3-70B", "Mistral-Small-3.1-24B", "Gemma-3-27B"):
        part = bc[bc["model"] == model].set_index("bank")
        strict = next(b for b in ("thesis (score 3)", "ext", "s3") if b in part.index)
        gaps.append((part.loc[strict, cols] - part.loc["s23", cols]).abs().max())
    n.put("xs-bank-gap", float(max(gaps)), 2)


def main() -> None:
    written = copies() + [knowledge_diagram()]
    written += thesis_tables()
    n = T.M.Numbers()
    written += aletheia(n)
    written += question_bank(n)
    written += random_bank(n)
    aisi_vs_probe(n)
    extra_keys(n)
    n.write(GENERATED / "numbers_arxiv.tex")
    written += ["numbers_arxiv.tex"]
    if "--no-plots" not in sys.argv:  # the plots take minutes; skip them when only numbers or tables changed
        written += plots()
    print(f"wrote {len(n.values)} arXiv-only numbers and {len(written)} assets into {HERE.name}")


if __name__ == "__main__":
    main()
