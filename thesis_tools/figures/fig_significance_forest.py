"""Figure: row-bootstrap vs dataset-bootstrap intervals for one family of tests -- Chapter 5, Statistical Analysis.

Reads run-coords/significance.json (written by src/assistant_axis_ndif/23_significance.py:
Pipeline A, passB store, nine labelled datasets, mms probe, paper16 bank) and draws every
comparison in it: the point difference, its row-bootstrap 95% interval (thick) and its
dataset-bootstrap 95% interval (thin). A test is coloured as significant only if the file marks
it significant after Benjamini-Hochberg over the dataset-bootstrap p-values. Nothing is typed in
by hand; every plotted number is printed so it can be checked against the file.

    python thesis_tools/figures/fig_significance_forest.py

Writes thesis/images/fig_significance_forest.pdf and a PNG preview in thesis_tools/figures/preview/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "run-coords" / "significance.json"
doc = json.loads(SRC.read_text(encoding="utf-8"))
comps = doc["comparisons"]
margin = doc["margin"]

GROUP = {"Q2": "persona v raw", "Q6": "layer", "Q8": "pooled v per-token"}


def label(c):
    if "A" in c and "B" in c:
        (pa, fa, la), (pb, fb, lb) = c["A"], c["B"]
        return f"{GROUP[c['question']]}: {fa} L{la} − {fb} L{lb} ({pa})"
    # Q8 names read "pooled-mean mms vs tok_lr/<read>, layer <L>"
    name = c["name"]
    read = name.split("tok_lr/")[1].split(",")[0]
    layer = name.rsplit("layer ", 1)[1]
    return f"{GROUP[c['question']]}: pooled mms − token lr/{read} L{layer}"


def significant(c):
    return bool(c.get("significant", False))


# reference palette (dataviz skill, light mode): slot 1 blue, slot 2 orange; ink for text
BLUE, ORANGE, INK, MUTED, BAND = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e7e6e2"

print(f"source: {SRC}")
print(f"margin: +/-{margin}; datasets: {len(doc['datasets'])} ({', '.join(doc['datasets'])})")
print(f"{'#':>2}  {'label':<62} {'diff':>8}  {'row CI95':>20}  {'dataset CI95':>20}  {'q_bh':>7}  sig")

fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(comps) + 0.9))
n = len(comps)
for i, c in enumerate(comps):
    y = n - 1 - i
    col = ORANGE if significant(c) else BLUE
    lo, hi = c["dataset_ci95"]
    ax.plot([lo, hi], [y, y], color=col, lw=1.1, solid_capstyle="round")
    if c.get("row_ci95"):
        rlo, rhi = c["row_ci95"]
        ax.plot([rlo, rhi], [y, y], color=col, lw=5.0, solid_capstyle="butt")
    ax.plot([c["diff"]], [y], marker="o", ms=4.2, color=INK, zorder=5,
            markeredgecolor="#ffffff", markeredgewidth=0.8)
    rc = c.get("row_ci95")
    q = c.get("dataset_q_bh")
    print(f"{i+1:>2}  {label(c):<62} {c['diff']:>+8.4f}  "
          f"{('[%+.4f, %+.4f]' % tuple(rc)) if rc else '--':>20}  "
          f"{'[%+.4f, %+.4f]' % (lo, hi):>20}  {('%.4f' % q) if q is not None else '--':>7}  "
          f"{significant(c)}")

ax.axvspan(-margin, margin, color=BAND, lw=0, zorder=0)
ax.axvline(0, color=MUTED, lw=0.8, ls="--", zorder=1)
ax.set_yticks(range(n))
ax.set_yticklabels([label(c) for c in comps][::-1], fontsize=6.6, color=INK)
ax.set_ylim(-0.7, n - 0.3)
ax.set_xlabel("difference in mean AUROC (A − B)", fontsize=8, color=INK)
ax.tick_params(axis="x", labelsize=7, colors=INK)
ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
handles = [plt.Line2D([], [], color=BLUE, lw=5.0, label="row bootstrap 95% interval"),
           plt.Line2D([], [], color=BLUE, lw=1.1, label="dataset bootstrap 95% interval"),
           plt.Line2D([], [], color=ORANGE, lw=5.0, label="significant after BH (dataset level)"),
           plt.Rectangle((0, 0), 1, 1, color=BAND, label=f"equivalence margin ±{margin:.2f}")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.35, -0.07), ncol=2, fontsize=6.5,
          frameon=False)
fig.tight_layout()
out = ROOT / "thesis" / "images" / "fig_significance_forest.pdf"
fig.savefig(out)
fig.savefig(ROOT / "thesis_tools" / "figures" / "preview" / "fig_significance_forest.png", dpi=160)
print("wrote", out)
