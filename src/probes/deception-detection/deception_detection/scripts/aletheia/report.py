"""results/aletheia/summary.md and the figure, from `score.py`'s tables alone.

The headline cell (D9 + K, each row read by its organism, middle block, reasoning dropped) is placed
among the 30 published entries over the datasets it covers; wherever a dataset is missing or read by
the base model only, every entry's mean is recomputed over the same datasets. The rank interval is
the 2.5th and 97.5th percentile of the rank of our bootstrap means among the fixed published means.
Every secondary cell is listed with its mean and the rank it would take, marked secondary.

    python deception_detection/scripts/aletheia/report.py
"""

import json
from pathlib import Path

import fire
import numpy as np
import pandas as pd

from deception_detection.log import logger
from deception_detection.scripts.aletheia import spec as S

KEYS = [d.key for d in S.DATASETS]
TOP = ("GMU-Veritaserum", "FlaiChi", "yy")


def _fmt(x: float, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _board() -> pd.DataFrame:
    board = pd.read_csv(S.LEADERBOARD)
    assert np.abs(board[KEYS].mean(axis=1) - board["mean"]).max() < 1e-4, "leaderboard means do not reproduce"
    return board


def _deviation(key: str) -> str | None:
    """The recorded deviation of a dataset's organism check, if any (`organism.accept_on_test_rows`)."""
    path = S.RESULTS / "checks" / f"organism_{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("deviation")


def _cut_lines() -> list[str]:
    """Every row the reads cut (frozen rule) or left unread, from checks/cut_rows.jsonl and unread_rows.jsonl."""
    out = []
    for name, what in (("cut_rows.jsonl", "cut by the frozen rule (kept its last 4,096 tokens)"),
                       ("unread_rows.jsonl", "left unread (too long for the server; its cells are not scored)")):
        path = S.RESULTS / "checks" / name
        recs = [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()] if path.exists() else []
        if recs:
            by = {}
            for r in recs:
                by.setdefault(Path(r["folder"]).parent.name + "/" + Path(r["folder"]).name, set()).add(r["row"])
            out.append(f"- Rows {what}: " + "; ".join(f"{k}: {len(v)}" for k, v in sorted(by.items())) + ".")
            untried = {}
            for r in recs:
                if str(r.get("reason", "")).startswith("longer than"):
                    untried.setdefault(Path(r["folder"]).parent.name + "/" + Path(r["folder"]).name, set()).add(r["row"])
            if untried:
                out.append("  - Of them cut untried: " + "; ".join(f"{k}: {len(v)}" for k, v in sorted(untried.items()))
                           + ". Each is longer than a row of the same read that ran out of memory alone on all four "
                           "tries with no other read on the cluster, and memory grows with length (the user's decision, "
                           "a deviation from the frozen rule's wording, which asks each row to fail itself). Every row "
                           "shorter than that row was read whole.")
            if name == "cut_rows.jsonl" and any("BotC" in k for k in by):
                out.append("  - BotC: a first pass through the adapter cut 151 rows after a row failed while other reads "
                           "shared the cluster; those reads were set aside unscored (`E:\\aletheia\\acts\\superseded\\"
                           "botc_cut_under_contention`) and the rows reread with the cluster idle. The organism and base "
                           "reads cut at their own thresholds, so their BotC cut sets differ (148 and 163 rows).")
    return out or ["- No row was cut or left unread."]


def _posthoc_lines(board: pd.DataFrame) -> list[str]:
    """`posthoc.py`'s cells, asked for after the pre-registered results were seen."""
    path = S.RESULTS / "posthoc" / "auroc.csv"
    if not path.exists():
        return []
    table = pd.read_csv(path)
    names = list(dict.fromkeys(table["detector"]))
    piv = table.pivot(index="dataset", columns="detector", values="auroc")
    lines = ["## Post hoc: D5 + K, rule K, D5 + rule K, D9 + rule K", "",
             "Not in the protocol: asked for after the results above were seen. Read through each organism at the "
             "middle block, reasoning dropped, as the headline; rule K (`posthoc.py`, the thesis's `k_final` recipe) "
             "is built inside each organism from its own reads.", "",
             "| Dataset | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for k in KEYS:
        if k in piv.index:
            lines.append(f"| {k} | " + " | ".join(_fmt(piv.loc[k, n]) for n in names) + " |")
    means = []
    for n in names:
        have = list(table[table["detector"] == n]["dataset"])
        mean = float(piv.loc[have, n].mean())
        means.append(f"{n} {_fmt(mean)} (rank {_rank(mean, board[have].mean(axis=1).to_numpy())} of 31, "
                     f"{len(have)} datasets)")
    return lines + ["", "Means: " + "; ".join(means) + ".", ""]


def _cell_rows(table: pd.DataFrame, cell: str) -> pd.DataFrame:
    return table[table["cell"] == cell].set_index("dataset")


def _boot_mean(boots: dict, cell: str, datasets: list[str]) -> np.ndarray:
    return np.mean([boots[f"{cell}|{k}"] for k in datasets], axis=0)


def _rank(value: float, means: np.ndarray) -> int:
    return int(1 + (means > value).sum())


def build() -> str:
    table = pd.read_csv(S.RESULTS / "auroc.csv")
    boots = dict(np.load(S.RESULTS / "boots.npz"))
    board = _board()
    head = _cell_rows(table, "headline")
    covered = [k for k in KEYS if k in head.index]
    others = [k for k in KEYS if k not in covered]
    lines = ["# The thesis's persona detector on the Aletheia's Quest test sets", ""]
    lines += [f"Leaderboard: {S.LEADERBOARD_URL} (30 entries; each entry's 10-dataset mean reproduced to 1e-4).",
              "Protocol: `deception_detection/scripts/aletheia/spec.py`, fixed before any test row was read.", ""]

    # the headline in the leaderboard
    ours = float(head.loc[covered, "auroc"].mean())
    boot = _boot_mean(boots, "headline", covered)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    # every entry's mean over the same datasets (the published mean when all ten are covered); the
    # rank and its interval use the same 30 entries, and the baselines are compared on their own
    sub = board.assign(sub_mean=board["mean"] if len(covered) == len(KEYS) else board[covered].mean(axis=1))
    entries = sub["sub_mean"].to_numpy()
    ranks = np.array([_rank(b, entries) for b in boot])
    r_lo, r_hi = np.quantile(ranks, [0.025, 0.975])
    lines += ["## Headline", "",
              f"D9 + K (the thesis's selected design), read through each organism at the middle block, reasoning "
              f"dropped: mean AUROC **{_fmt(ours)}** [95% {_fmt(lo)}, {_fmt(hi)}] over {len(covered)} datasets "
              f"({', '.join(covered)}).",
              f"Rank among the 30 entries over the same datasets: **{_rank(ours, sub['sub_mean'].to_numpy())}** "
              f"(rank interval {int(r_lo)}-{int(r_hi)})."]
    if others:
        lines.append(f"Not in the headline (no organism read yet): {', '.join(others)}.")
    if len(covered) == len(KEYS):
        lines.append("Baselines (means only, so compared at ten datasets): " + "; ".join(
            f"{k} {v:.4f}, our bootstrap mean above it in {np.mean(boot > v):.0%} of draws" for k, v in S.BASELINES.items())
            + ".")
    lines.append("")

    # the leaderboard over the covered datasets, ours inserted
    show = sub[["team", "track", "sub_mean"]].copy()
    show.loc[len(show)] = ["**ours: D9 + K (headline)**", "white-box (zero-shot)", ours]
    show = show.sort_values("sub_mean", ascending=False).reset_index(drop=True)
    lines += [f"## Leaderboard over {len(covered)} datasets", "", "| # | Entry | Track | Mean AUROC |", "|---|---|---|---|"]
    for i, r in show.iterrows():
        lines.append(f"| {i + 1} | {r['team']} | {r['track']} | {_fmt(r['sub_mean'], 4)} |")
    lines.append("")

    # per dataset
    best = board[board["track"] == "white-box"][KEYS].max()
    d9 = _cell_rows(table, "S1-d9")
    lines += ["## Per dataset", "",
              "| Dataset | n (lies) | D9 + K [95%] | D9 | random 97.5% | " + " | ".join(TOP) + " | best white-box |",
              "|---|---|---|---|---|" + "---|" * len(TOP) + "---|"]
    for d in S.DATASETS:
        if d.key in head.index:
            h = head.loc[d.key]
            cell = f"{_fmt(h['auroc'])} [{_fmt(h['lo'])}, {_fmt(h['hi'])}]"
            n = f"{int(h['n'])} ({int(h['lies'])})"
            band = _fmt(h["band97.5"])
        else:
            cell, n, band = "not read", f"{d.n} ({d.lies})", "n/a"
        d9v = _fmt(d9.loc[d.key, "auroc"]) if d.key in d9.index else "n/a"
        tops = " | ".join(_fmt(float(board.loc[board["team"].eq(t) & board["track"].eq("white-box"), d.key].iloc[0]), 4)
                          for t in TOP)
        mark = "†" if _deviation(d.key) else ""
        lines.append(f"| {d.key}{mark} | {n} | {cell} | {d9v} | {band} | {tops} | {_fmt(float(best[d.key]), 4)} |")
    flagged = [d.key for d in S.DATASETS if _deviation(d.key)]
    if flagged:
        lines += ["", "† " + "; ".join(f"{k}: {_deviation(k)}" for k in flagged) + "."]
    lines.append("")

    # the variants
    lines += ["## Every pre-registered cell", "",
              f"Means over the headline's {len(covered)} datasets where the cell covers all of them; otherwise "
              "over the datasets it covers (listed).", "",
              "| Cell | Role | Detector | Read | Render | Span | Block | Mean | Rank it would take | Datasets |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for cell in S.CELLS:
        rows = _cell_rows(table, cell.id)
        have = [k for k in covered if k in rows.index] or [k for k in KEYS if k in rows.index]
        if not have:
            lines.append(f"| {cell.id} | {cell.role} | {cell.detector} | {cell.read} | {cell.render} | {cell.span} | "
                         f"{cell.block} | not yet | | |")
            continue
        mean = float(rows.loc[have, "auroc"].mean())
        rank = _rank(mean, board[have].mean(axis=1).to_numpy())
        lines.append(f"| {cell.id} | {cell.role} | {cell.detector} | {cell.read} | {cell.render} | {cell.span} | "
                     f"{cell.block} | {_fmt(mean)} | {rank} of 31 | {len(have)} |")
    lines.append("")

    # the base model's read beside the organism's
    base = _cell_rows(table, "S3-base")
    lines += ["## The organism's read and the base model's", "",
              "| Dataset | D9 + K, organism read | D9 + K, base read |", "|---|---|---|"]
    for d in S.DATASETS:
        a = _fmt(head.loc[d.key, "auroc"]) if d.key in head.index else "not read"
        b = _fmt(base.loc[d.key, "auroc"]) if d.key in base.index else "not read"
        lines.append(f"| {d.key} | {a} | {b} |")
    lines.append("")

    lines += _posthoc_lines(board)
    lines += ["## Caveats", "",
              "- A base-model read of an organism's transcript is the thesis's control, not a result for that "
              "dataset: the base model has no fine-tuned belief or goal to conceal, so it reads only the text. A "
              "dataset whose organism could not be read (the Hidden-Goal adapters are gated) is reported as not run; "
              "its base read appears only beside the organism read, as the control. Harmful-Looking is different: "
              "plain Gemma wrote it, so the base model is its organism.",
              "- The top entries are active methods (the organism judging its own transcript, follow-up questions, "
              "probes trained on the competition's dev data); this detector is a passive zero-shot read of one "
              "direction.",
              "- The starter kit dropped the `reasoning` field and cut rows to their last 256 tokens by default; the "
              "headline drops the reasoning as the kit did; rows are cut only by the frozen rule, listed below.",
              "- A July 22 starter-kit commit fixed a silent no-op in which Qwen3.5 adapter keys matched 0 of 512; "
              "whether the official white-box reads of the Qwen organisms had their adapters applied is not known, "
              "so the base-model read is shown beside the organism read.",
              "- AB-AW has 26 lies: per-dataset intervals there are wide.", ""]
    lines += _cut_lines()
    missing = pd.read_csv(S.RESULTS / "missing.csv") if (S.RESULTS / "missing.csv").stat().st_size > 1 else pd.DataFrame()
    if len(missing):
        lines += [f"Cells not yet scored: {len(missing)} (results/aletheia/missing.csv).", ""]
    return "\n".join(lines) + "\n"


def figure() -> None:
    """Per-dataset AUROC: the headline (95% interval), D9 alone, and the top three white-box entries."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = pd.read_csv(S.RESULTS / "auroc.csv")
    board = _board()
    head, d9 = _cell_rows(table, "headline"), _cell_rows(table, "S1-d9")
    series = [("D9 + K (ours, headline)", head, "#1f4e79"), ("D9 (ours)", d9, "#6fa8dc")]
    tops = [(t, board[(board["team"] == t) & (board["track"] == "white-box")].iloc[0], c)
            for t, c in zip(TOP, ("#b45f06", "#e69138", "#f6b26b"))]
    x = np.arange(len(KEYS))
    width = 0.16
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for j, (name, rows, color) in enumerate(series):
        vals = [rows.loc[k, "auroc"] if k in rows.index else np.nan for k in KEYS]
        err = None
        if name.startswith("D9 + K"):
            lo = [rows.loc[k, "auroc"] - rows.loc[k, "lo"] if k in rows.index else 0 for k in KEYS]
            hi = [rows.loc[k, "hi"] - rows.loc[k, "auroc"] if k in rows.index else 0 for k in KEYS]
            err = [lo, hi]
        ax.bar(x + (j - 2) * width, vals, width, color=color, label=name, yerr=err, capsize=2, error_kw={"lw": 0.8})
    for j, (name, row, color) in enumerate(tops):
        ax.bar(x + (j + 0) * width, [row[k] for k in KEYS], width, color=color, label=f"{name} (white-box)")
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_xticks(x, KEYS)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("AUROC")
    ax.legend(ncol=3, fontsize=8, loc="lower left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(S.RESULTS / "fig_aletheia.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(S.RESULTS / "fig_aletheia.png", dpi=160)
    plt.close(fig)


def depth_figure() -> None:
    """Post hoc: D9 and D9 + K by relative depth on each dataset, base read (solid) and organism read
    (dashed, up to the middle block); the protocol's block is marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = S.RESULTS / "depth.csv"
    if not path.exists():
        return
    table = pd.read_csv(path)
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.2), sharey=True)
    for ax, d in zip(axes.flat, S.DATASETS):
        part = table[table["dataset"] == d.key]
        for (read, det), style in {("base", "D9"): ("#6fa8dc", "-"), ("base", "D9+K"): ("#1f4e79", "-"),
                                    ("organism", "D9"): ("#6fa8dc", "--"), ("organism", "D9+K"): ("#1f4e79", "--")}.items():
            rows = part[part["read"] == read].sort_values("depth")
            if det in rows and rows[det].notna().any():
                ax.plot(rows["depth"], rows[det], color=style[0], ls=style[1], lw=1.2, label=f"{det}, {read} read")
        ax.axvline(d.base.mid / d.base.n_blocks, color="grey", lw=0.6, ls=":")
        ax.axhline(0.5, color="grey", lw=0.6)
        ax.set_title(d.key, fontsize=9)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes.flat[0].legend(fontsize=6, frameon=False, loc="lower left")
    fig.supxlabel("relative depth (block / blocks); dotted: the protocol's middle block", fontsize=9)
    fig.supylabel("AUROC (post hoc)", fontsize=9)
    fig.tight_layout()
    fig.savefig(S.RESULTS / "fig_aletheia_depth.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(S.RESULTS / "fig_aletheia_depth.png", dpi=150)
    plt.close(fig)


def run() -> None:
    text = build()
    (S.RESULTS / "summary.md").write_text(text, encoding="utf-8")
    figure()
    depth_figure()
    logger.info(f"-> {S.RESULTS / 'summary.md'}, fig_aletheia.pdf/png")


if __name__ == "__main__":
    fire.Fire({"run": run})
