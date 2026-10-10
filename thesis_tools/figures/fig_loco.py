"""Figure: pooled leave-one-cluster-out training per target -- Chapter 5, Section "With pooled
supervised training" (fig:r-loco).

Reads Pipeline A's clustered-holdout result from the local copy of the lab-PC file (no
hard-coded numbers):

    audit/cache/labpc/run-premise/clustered/clustered_L40_mean_s42.json
    (lab PC original: C:/Users/user4/expLB/run-premise/clustered/clustered_L40_mean_s42.json)

For each of the 11 targets it plots
  * single_offdiag : the mean off-diagonal AUROC of the raw single-source probes transferred to
                     that target (every other dataset as the source);
  * pooled_loco    : the AUROC of one raw probe trained on the pooled rows of every dataset
                     OUTSIDE the target's provenance cluster.
Layer 40, mean pooling, probe "paper", seed 42 -- one seed, one layer, no dataset bootstrap.

It prints both values, the per-target gain, the summary block and the leaked targets, so a
verifier can check them against the JSON.

Writes thesis/images/fig_loco.pdf and thesis_tools/figures/preview/fig_loco.png.

    python thesis_tools/figures/fig_loco.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "audit" / "cache" / "labpc" / "run-premise" / "clustered" / "clustered_L40_mean_s42.json"
d = json.loads(SRC.read_text(encoding="utf-8"))
assert (d["layer"], d["pooling"], d["probe"], d["seed"]) == (40, "mean", "paper", 42)
per = d["per_dataset"]

# display names: the thesis abbreviations (GLOSSARY section 6)
SHORT = {"instructed-deception": "ID", "harm-pressure-choice": "HP-C",
         "harm-pressure-knowledge-report": "HP-KR", "harm-pressure-kr-matched": "HP-KR text-matched",
         "convincing-game": "CG", "insider-trading": "IT", "ai_liar": "ai_liar",
         "roleplaying": "roleplaying", "sandbagging_v2": "sandbagging_v2", "mask": "mask",
         "sycophancy_mt": "sycophancy_mt"}
CLUSTER_NAME = {"lb_harm": "LB harm pressure", "lb_scenario": "LB scenarios", "apollo": "Apollo",
                "mask": "MASK", "ours": "ours"}

# order: by cluster as listed in the file, then by name inside a cluster
order = [t for c in d["clusters"] for t in d["clusters"][c] if t in per]
assert sorted(order) == sorted(per), (order, sorted(per))

print(f"source: {SRC.relative_to(ROOT)}  layer={d['layer']} pooling={d['pooling']} "
      f"probe={d['probe']} seed={d['seed']}")
print("leaked targets:", d["leaked"])
print(f"{'target':34s} {'cluster':12s} {'single':>8s} {'LOCO':>8s} {'gain':>8s} {'n_loco':>6s}")
for t in order:
    r = per[t]
    print(f"{t:34s} {r['cluster']:12s} {r['single_offdiag']:8.4f} {r['pooled_loco']:8.4f} "
          f"{r['pooled_loco'] - r['single_offdiag']:+8.4f} {r['n_loco_datasets']:6d}")
print("summary:", json.dumps(d["summary"]))
gains = [per[t]["pooled_loco"] - per[t]["single_offdiag"] for t in order]
print(f"mean gain over {len(gains)} targets = {sum(gains) / len(gains):+.4f}; "
      f"positive at {sum(g > 0 for g in gains)} of {len(gains)}")

# ---- draw ------------------------------------------------------------------------------
# Two series: the single-source reference in neutral grey (a baseline, not an entity) and the
# pooled probe in the thesis blue. A connecting segment shows the gain; the two leaked targets
# are marked in the tick label.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
GREY, BLUE = "#898781", "#1c5cab"
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(6.0, 3.8))
ys = list(range(len(order)))[::-1]
for y, t in zip(ys, order):
    r = per[t]
    ax.plot([r["single_offdiag"], r["pooled_loco"]], [y, y], color="#c3c2b7", lw=2, zorder=2)
ax.scatter([per[t]["single_offdiag"] for t in order], ys, s=34, color=GREY, zorder=3,
           edgecolors="white", linewidths=0.8, label="single-source probes (mean over sources)")
ax.scatter([per[t]["pooled_loco"] for t in order], ys, s=40, color=BLUE, marker="D", zorder=4,
           edgecolors="white", linewidths=0.8, label="pooled, leave-one-cluster-out")
labels = []
for t in order:
    lab = SHORT[t] + (" *" if t in d["leaked"] else "")
    labels.append(f"{lab}  [{CLUSTER_NAME[per[t]['cluster']]}]")
ax.set_yticks(ys, labels)
ax.set_xlabel("AUROC on the held-out target", color=MUTED)
ax.set_xlim(0.5, 1.0)
ax.grid(axis="x", color=GRID, lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=7)
fig.text(0.01, 0.01, "* target shares rows with another dataset (leaked under leave-one-dataset-out)",
         fontsize=6.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.94))

out = ROOT / "thesis" / "images" / "fig_loco.pdf"
fig.savefig(out)
prev = ROOT / "thesis_tools" / "figures" / "preview" / "fig_loco.png"
fig.savefig(prev, dpi=160)
print("\nwrote", out)
print("wrote", prev)
