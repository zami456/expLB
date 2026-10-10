"""Figure: the widening control over 16 layers -- Chapter 5, Section "The roles, not the widening".

Reads Pipeline B's 16-layer sweep at the pinned commit (no copy, no hard-coded numbers):

    git show 16def2de:src/probes/deception-detection/results/persona_depth/depth.csv

and plots, for four zero-shot contrasts on the kd bank (bank field "ext"),
  left : source "transfer, off-diagonal mean", metric auroc
  right: source "zero-shot, HP-KR",            metric auroc
against layer. It prints every plotted value, the 16-layer average of each series, and the
layer counts quoted in the text, so a verifier can check them against the CSV.

Writes thesis/images/fig_widening.pdf and thesis_tools/figures/preview/fig_widening.png.

    python thesis_tools/figures/fig_widening.py
"""
import csv
import io
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PIN = "16def2de"
PATH = "src/probes/deception-detection/results/persona_depth/depth.csv"

raw = subprocess.run(["git", "show", f"{PIN}:{PATH}"], cwd=ROOT, check=True,
                     capture_output=True).stdout.decode("utf-8")
rows = list(csv.DictReader(io.StringIO(raw)))

# (source, arm) -> {layer: value}, AUROC only, kd bank only
series = defaultdict(dict)
for r in rows:
    if r["metric"] != "auroc" or r["bank"] != "ext":
        continue
    series[(r["source"], r["arm"])][int(r["layer"])] = float(r["value"])

SOURCES = [("transfer, off-diagonal mean", "Transfer: mean off-diagonal AUROC"),
           ("zero-shot, HP-KR", "HP-KR: zero-shot AUROC")]
# (arm name in the CSV, label in the figure, colour, marker, line style)
# Colours follow the entities in Pipeline B's fig_depth.pdf (published = orange, knowledge-denial
# = light blue dashed, extended = blue), with the unified control added in aqua. The set passes
# the dataviz validator on white, all pairs (aqua and light blue sit below 3:1 contrast, so every
# series also has its own marker and line style, and a legend).
ARMS = [("deception axis (8v7)", "published (8 v 7)", "#eb6834", "o", "-"),
        ("unified axis (13v12)", "unified (13 v 12), the control", "#1baf7a", "^", "-."),
        ("knowledge-denial axis (4v6)", "knowledge-denial (4 v 6)", "#6da7ec", "s", "--"),
        ("extended axis (12v13)", "extended (12 v 13)", "#1c5cab", "o", "-")]

layers = sorted(series[(SOURCES[0][0], ARMS[0][0])])
assert len(layers) == 16, layers
for src, _ in SOURCES:
    for arm, *_ in ARMS:
        assert sorted(series[(src, arm)]) == layers, (src, arm)

# ---- print what is plotted -------------------------------------------------------------
print(f"source file: git show {PIN}:{PATH}  ({len(rows)} rows)")
print("layers:", layers)
for src, _ in SOURCES:
    print(f"\n[{src}] auroc")
    for arm, label, *_ in ARMS:
        v = series[(src, arm)]
        avg = sum(v.values()) / len(v)
        print(f"  {arm:30s} 16-layer average = {avg:.4f}   "
              + " ".join(f"L{L}:{v[L]:.4f}" for L in layers))

uni_t = series[(SOURCES[0][0], "unified axis (13v12)")]
dec_t = series[(SOURCES[0][0], "deception axis (8v7)")]
uni_h = series[(SOURCES[1][0], "unified axis (13v12)")]
dec_h = series[(SOURCES[1][0], "deception axis (8v7)")]
below_h = [L for L in layers if uni_h[L] < dec_h[L]]
above_chance = [L for L in layers if uni_h[L] > 0.5]
below_t = [L for L in layers if uni_t[L] < dec_t[L]]
print("\ncounts quoted in the text")
print(f"  HP-KR: unified below published at {len(below_h)}/16 layers; not below at "
      f"{[L for L in layers if L not in below_h]}")
print(f"  HP-KR: unified above 0.5 at {len(above_chance)}/16 layers: {above_chance}")
print(f"  transfer: unified below published at {len(below_t)}/16 layers; not below at "
      f"{[L for L in layers if L not in below_t]}")

# ---- draw ------------------------------------------------------------------------------
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.75), sharex=True)
for ax, (src, title) in zip(axes, SOURCES):
    for arm, label, colour, marker, ls in ARMS:
        v = series[(src, arm)]
        ax.plot(layers, [v[L] for L in layers], color=colour, lw=1.6, ls=ls,
                marker=marker, ms=3.6, mec="white", mew=0.6, label=label, zorder=3)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ax.set_xlabel("layer", color=MUTED)
    ax.set_xticks([20, 28, 36, 44, 52, 60, 68, 76])
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axvline(40, color="#898781", lw=0.8, ls=":", zorder=1)
axes[0].set_ylim(0.5, 0.95)
axes[1].set_ylim(0.0, 1.0)
axes[1].axhline(0.5, color="#898781", lw=0.8, ls="--", zorder=1)
axes[1].text(21, 0.515, "chance", ha="left", va="bottom", fontsize=7, color=MUTED)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7.5,
           bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.2)
fig.tight_layout(rect=(0, 0, 1, 0.9))

out = ROOT / "thesis" / "images" / "fig_widening.pdf"
fig.savefig(out)
prev = ROOT / "thesis_tools" / "figures" / "preview" / "fig_widening.png"
fig.savefig(prev, dpi=160)
print("\nwrote", out)
print("wrote", prev)
