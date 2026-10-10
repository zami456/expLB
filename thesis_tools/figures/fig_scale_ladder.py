"""Figure: the scale ladder -- Chapter 5, Section "Across model scale" (fig:r-scale).

Reads Pipeline A's crossgen matrices from the local copies of the lab-PC stores (no hard-coded
numbers):

    audit/p17/labcache/expLB/run-crossgen/scale_{3b,8b}/matrices/mean_L*_s42.json
    audit/p17/labcache/expLB/run-crossgen/scale_70b_paper16fp32/matrices/mean_L*_s42.json
    audit/p17/labcache/expLB-resplit/run-crossgen/kdscale_{3b,8b,70b}_{s23,all}/matrices/mean_L*_s42.json

(lab PC originals: C:/Users/user4/expLB/run-crossgen/... and C:/Users/user4/expLB-resplit/run-crossgen/...)

The recipe is audit/scripts/p12_kd_scale_ladder.py, re-implemented read-only:
  * off-diagonal mean AUROC over the 20 train != test pairs of the five datasets
    (instructed-deception, mask, convincing-game, ai_liar, roleplaying), probe "paper";
  * per layer, best persona feature = max over {axes, pc:*} (a post-hoc max);
  * layer-averaged over each rung's own stored layers (3B: 7, 8B: 8, 70B: 21);
  * the plotted quantity is (best persona feature - raw), and, per feature, (pc:deception - raw).
k = 3, mean pooling, seed 42, one run per rung.

It prints every plotted value plus the dataset-PCA margins and the kd "all" variant, so a
verifier can check them against the matrices.

Writes thesis/images/fig_scale_ladder.pdf and thesis_tools/figures/preview/fig_scale_ladder.png.

    python thesis_tools/figures/fig_scale_ladder.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "audit" / "p17" / "labcache" / "expLB" / "run-crossgen"
NEW = ROOT / "audit" / "p17" / "labcache" / "expLB-resplit" / "run-crossgen"
POOL, PROBE = "mean", "paper"
DS5 = {"instructed-deception", "mask", "convincing-game", "ai_liar", "roleplaying"}

RUNGS = [("3B", 3), ("8B", 8), ("70B", 70)]
STORES = {
    ("paper16", "3B"): OLD / "scale_3b",
    ("paper16", "8B"): OLD / "scale_8b",
    ("paper16", "70B"): OLD / "scale_70b_paper16fp32",
    ("kd s23", "3B"): NEW / "kdscale_3b_s23",
    ("kd s23", "8B"): NEW / "kdscale_8b_s23",
    ("kd s23", "70B"): NEW / "kdscale_70b_s23",
    ("kd all", "3B"): NEW / "kdscale_3b_all",
    ("kd all", "8B"): NEW / "kdscale_8b_all",
    ("kd all", "70B"): NEW / "kdscale_70b_all",
}


def offdiag(d, feature):
    blk = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}.get((feature, PROBE))
    if blk is None:
        return None
    ds = d["datasets"]
    assert set(ds) == DS5, ds
    v = [blk[i][j] for i in range(len(ds)) for j in range(len(ds))
         if i != j and blk[i][j] is not None]
    return sum(v) / len(v)


def ladder_cell(store):
    per_layer = {}
    meta = set()
    for fn in sorted((store / "matrices").glob(f"{POOL}_L*_s42.json")):
        d = json.loads(fn.read_text(encoding="utf-8"))
        meta.add((d.get("bank_name"), d.get("k"), d.get("pooling")))
        feats = sorted({r["feature"] for r in d["results"]})
        pers = [f for f in feats if f.startswith("pc:") or f == "axes"]
        vals = {f: offdiag(d, f) for f in feats}
        best_f = max(pers, key=lambda f: vals[f])
        per_layer[d["layer"]] = dict(raw=vals["raw"], best=vals[best_f], best_f=best_f,
                                     dp=vals["datapca"], pcdec=vals["pc:deception"])
    L = sorted(per_layer)
    avg = {k: sum(per_layer[l][k] for l in L) / len(L) for k in ("raw", "best", "dp", "pcdec")}
    return L, per_layer, avg, meta


res = {}
print(f"pooling={POOL} probe={PROBE} k=3 seed=42 datasets={sorted(DS5)}")
for (bank, rung), store in STORES.items():
    L, per_layer, avg, meta = ladder_cell(store)
    res[(bank, rung)] = avg
    print(f"\n[{bank} {rung}] {store.relative_to(ROOT)}  meta={meta}  layers={L}")
    print("  per layer (best feature, best - raw): "
          + " ".join(f"L{l}:{per_layer[l]['best_f']}:{per_layer[l]['best'] - per_layer[l]['raw']:+.4f}"
                     for l in L))
    print(f"  raw={avg['raw']:.4f} best={avg['best']:.4f} datapca={avg['dp']:.4f}")
    print(f"  best - raw     = {avg['best'] - avg['raw']:+.4f}")
    print(f"  best - datapca = {avg['best'] - avg['dp']:+.4f}")
    print(f"  pc:deception - raw = {avg['pcdec'] - avg['raw']:+.4f}")

print("\npc:deception - raw > 0, per layer (paper16):")
for rung, _ in RUNGS:
    _, per_layer, _, _ = ladder_cell(STORES[("paper16", rung)])
    wins = [l for l in sorted(per_layer) if per_layer[l]["pcdec"] > per_layer[l]["raw"]]
    print(f"  {rung:4s} {len(wins)} of {len(per_layer)} layers: {wins}")

# last-token pooling exists only for the paper16 rungs (the kdscale stores hold mean_L* only);
# printed, not plotted, for the pooling caveat in the text
print("\nlast-token pooling, paper16 (not plotted):")
POOL = "last"
for rung, _ in RUNGS:
    L, _, avg, meta = ladder_cell(STORES[("paper16", rung)])
    print(f"  {rung:4s} layers={len(L)} meta={meta} best - raw = {avg['best'] - avg['raw']:+.4f}  "
          f"best - datapca = {avg['best'] - avg['dp']:+.4f}")
for bank in ("kd s23", "kd all"):
    for rung, _ in RUNGS:
        n = len(list((STORES[(bank, rung)] / "matrices").glob("last_L*_s42.json")))
        print(f"  {bank} {rung}: {n} last_L* matrices")
POOL = "mean"

print("\nbank gap (kd - paper16, best - raw):")
for rung, _ in RUNGS:
    p = res[("paper16", rung)]
    for v in ("kd s23", "kd all"):
        k = res[(v, rung)]
        print(f"  {rung:4s} {v}: {(k['best'] - k['raw']) - (p['best'] - p['raw']):+.4f}")

# ---- draw ------------------------------------------------------------------------------
# Colour follows the bank: the published 26-persona bank orange, the kd 36-persona bank blue
# (the same entities and hues as fig_widening.py; the pair passes the dataviz validator on white).
# Solid = best persona feature (post-hoc max), dashed = the fixed feature pc:deception.
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BANKS = [("paper16", "paper16 bank (26 personas)", "#eb6834", "o"),
         ("kd s23", "kd bank (36 personas, scores 2-3)", "#1c5cab", "s")]
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(figsize=(5.4, 2.9))
x = list(range(len(RUNGS)))
for bank, label, colour, marker in BANKS:
    best = [res[(bank, r)]["best"] - res[(bank, r)]["raw"] for r, _ in RUNGS]
    pcd = [res[(bank, r)]["pcdec"] - res[(bank, r)]["raw"] for r, _ in RUNGS]
    ax.plot(x, best, color=colour, lw=2, marker=marker, ms=6, mec="white", mew=0.8,
            label=f"{label}: best feature", zorder=3)
    ax.plot(x, pcd, color=colour, lw=1.4, ls="--", marker=marker, ms=5, mfc="white",
            mec=colour, mew=1.2, label=f"{label}: pc:deception", zorder=3)
ax.axhline(0, color="#898781", lw=0.9, zorder=1)
ax.text(-0.27, 0.0008, "raw baseline", ha="left", va="bottom", fontsize=7, color=MUTED)
ax.set_xticks(x, [f"Llama {r}" for r, _ in RUNGS])
ax.set_xlim(-0.3, len(RUNGS) - 1 + 0.35)
ax.set_ylabel("off-diagonal AUROC minus raw\n(layer average)", color=MUTED)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="lower left", frameon=False, fontsize=7, handlelength=2.4)
fig.tight_layout()

out = ROOT / "thesis" / "images" / "fig_scale_ladder.pdf"
fig.savefig(out)
prev = ROOT / "thesis_tools" / "figures" / "preview" / "fig_scale_ladder.png"
fig.savefig(prev, dpi=160)
print("\nwrote", out)
print("wrote", prev)
