"""Figure: HP-KR rows by label x (knows, denies) -- Chapter 4, Data Integration.

Recomputes the counts from the raw Liars' Bench CSV with the pipeline's own denial regex and
knowledge field, so the figure cannot drift from the data. Writes thesis/images/fig_hpkr_cells.pdf.

    python thesis_tools/figures/fig_hpkr_cells.py
"""
import ast
import collections
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "assistant_axis_ndif"))
import eval_datasets as E  # noqa: E402

path = E.SUBSETS / E.CSV_DATASETS["harm-pressure-knowledge-report"]
tab = collections.Counter()
for rec in csv.DictReader(path.open(encoding="utf-8", newline="")):
    if rec.get("model") != E.CSV_MODEL:
        continue
    msgs = ast.literal_eval(rec["messages"])
    asst = [m for m in msgs if m.get("role") == "assistant"]
    if not asst:
        continue
    meta = E.parse_meta(rec["meta"])
    knows = meta.get("model_belief", -1) != -1
    denies = E.denies_knowledge(asst[-1].get("content", ""))
    deceptive = str(rec["deceptive"]).strip().lower() == "true"
    tab[(deceptive, knows, denies)] += 1

cols = [(True, False), (True, True), (False, True), (False, False)]      # (knows, denies)
col_names = ["knows,\nasserts", "knows,\ndenies", "does not know,\ndenies", "does not know,\nasserts"]
rows = [True, False]
row_names = ["deceptive", "honest"]
grid = [[tab[(r, k, d)] for (k, d) in cols] for r in rows]
print("counts:", grid, "total", sum(map(sum, grid)))

# one-hue sequential ramp (reference palette blue, steps 100 -> 600); zero cells stay surface
ramp = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#184f95"])
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#ffffff"

fig, ax = plt.subplots(figsize=(6.0, 2.1))
norm = LogNorm(vmin=1, vmax=max(map(max, grid)))
for i, row in enumerate(grid):
    for j, v in enumerate(row):
        if v == 0:
            ax.add_patch(plt.Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, facecolor=SURFACE,
                                       edgecolor="#b5b3ad", hatch="///", linewidth=0.8))
            ax.text(j, i, "0 (empty)", ha="center", va="center", fontsize=9, color=INK,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=SURFACE, edgecolor="none"))
        else:
            c = ramp(norm(v))
            ax.add_patch(plt.Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, facecolor=c,
                                       edgecolor=SURFACE, linewidth=2))
            dark = norm(v) > 0.6
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=10,
                    color="#ffffff" if dark else INK)
ax.set_xlim(-0.5, 3.5); ax.set_ylim(1.5, -0.5)
ax.set_xticks(range(4)); ax.set_xticklabels(col_names, fontsize=8.5, color=INK)
ax.set_yticks(range(2)); ax.set_yticklabels(row_names, fontsize=9, color=INK)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.xaxis.tick_top()
fig.tight_layout()
out = ROOT / "thesis" / "images" / "fig_hpkr_cells.pdf"
fig.savefig(out)
fig.savefig(ROOT / "thesis_tools" / "figures" / "preview" / "fig_hpkr_cells.png", dpi=160)
print("wrote", out)
