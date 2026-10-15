"""results/aletheia/summary.md and the figure, from `score.py`'s tables alone.

The headline cell (D9 + K, each row read by its organism, middle block, reasoning dropped) is placed
among the 30 published entries over the datasets it covers; wherever a dataset is missing or read by
the base model only, every entry's mean is recomputed over the same datasets. The rank interval is
the 2.5th and 97.5th percentile of the rank of our bootstrap means among the fixed published means.
Every secondary cell is listed with its mean and the rank it would take, marked secondary.

    python deception_detection/scripts/aletheia/report.py
"""

import json

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
    sub = board.assign(sub_mean=board[covered].mean(axis=1))
    entries = np.concatenate([sub["sub_mean"].to_numpy(), list(S.BASELINES.values())]) if len(covered) == len(KEYS) \
        else sub["sub_mean"].to_numpy()
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
        lines.append(f"Baselines: " + "; ".join(f"{k} {v:.4f}" for k, v in S.BASELINES.items()) + ".")
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
        lines.append(f"| {d.key} | {n} | {cell} | {d9v} | {band} | {tops} | {_fmt(float(best[d.key]), 4)} |")
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

    lines += ["## Caveats", "",
              "- The top entries are active methods (the organism judging its own transcript, follow-up questions, "
              "probes trained on the competition's dev data); this detector is a passive zero-shot read of one "
              "direction.",
              "- The starter kit dropped the `reasoning` field and cut rows to their last 256 tokens by default; the "
              "headline drops the reasoning as the kit did, and no row is cut.",
              "- A July 22 starter-kit commit fixed a silent no-op in which Qwen3.5 adapter keys matched 0 of 512; "
              "whether the official white-box reads of the Qwen organisms had their adapters applied is not known, "
              "so the base-model read is shown beside the organism read.",
              "- AB-AW has 26 lies: per-dataset intervals there are wide.", ""]
    missing = pd.read_csv(S.RESULTS / "missing.csv") if (S.RESULTS / "missing.csv").stat().st_size > 1 else pd.DataFrame()
    if len(missing):
        lines += [f"Cells not yet scored: {len(missing)} (results/aletheia/missing.csv).", ""]
    return "\n".join(lines) + "\n"


def run() -> None:
    text = build()
    (S.RESULTS / "summary.md").write_text(text, encoding="utf-8")
    logger.info(f"-> {S.RESULTS / 'summary.md'}")


if __name__ == "__main__":
    fire.Fire({"run": run})
