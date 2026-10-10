"""Figure: the capitulation bank on sycophancy_mt -- Chapter 5, Section "The capitulation bank and
the sycophancy axis" (label fig:r-cap).

Why a new script. The owner's figure (figures/thesis/fig5_capitulation.pdf, drawn by
kd/scripts/make_thesis_figures.py::fig5_capitulation) hard-codes numbers: its footer prints
"+0.098, 95% CI [-0.036, +0.208]" and its title "unfiltered agrees to 0.004", and its "register
axis" line comes from a constant (register_baseline_zeroshot = 0.374) written into
cap/scripts/export_cap_fig.py. This script reads only result files and hard-codes no number.

Inputs (all read, none copied):
  run-cap/export/cap_fig_data.json          exported by cap/scripts/export_cap_fig.py from the lab
                                            matrices run-crossgen/kd11/banks/cap_s23/matrices/
                                            mean_L{20,40,44}_s42.json (bank axes_s23.pt, judge
                                            scores 2-3); read: banks.cap_s23_L40.{zeroshot,
                                            per_source}
  run-crossgen/kd11/matrices/mean_L40_s42.json
                                            the paper16 bank (axes_vectors_s3_fp32.pt); read:
                                            zeroshot[PC1], sycophancy_mt column, with its row CI
  run-coords/q10_sycophancy_axis.json       read: results["paper16|mean"].{sycophancy,deception,
                                            assistant}.key_layers["40"] (row-bootstrap CI)
  run-coords/zeroshot_nulls.json            the 1,000-direction null on sycophancy_mt, L40 mean:
                                            sycophancy_mt.L40_mean.random_p95_auroc_dev
Optional cross-check (lab-only cache): audit/p17/labcache/expLB/run-crossgen/kd11/banks/cap_s23/
matrices/mean_L40_s42.json -- the per-source columns must equal the export's.

It also recomputes, for layers 20 / 40 / 44, the paired dataset bootstrap of
cap/scripts/cap_significance.py (B = 10,000, numpy default_rng(42), one resampling of the 10
source datasets shared by every pair) from the exported per-source columns, and prints it, so
the numbers in the text can be checked against this output.

Writes thesis/images/fig_capitulation.pdf and thesis_tools/figures/preview/fig_capitulation.png.

    python thesis_tools/figures/fig_capitulation.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DS = "sycophancy_mt"
CAP = ROOT / "run-cap" / "export" / "cap_fig_data.json"
P16 = ROOT / "run-crossgen" / "kd11" / "matrices" / "mean_L40_s42.json"
Q10 = ROOT / "run-coords" / "q10_sycophancy_axis.json"
LABCACHE = (ROOT / "audit" / "p17" / "labcache" / "expLB" / "run-crossgen" / "kd11" / "banks"
            / "cap_s23" / "matrices" / "mean_L40_s42.json")

cap = json.loads(CAP.read_text(encoding="utf-8"))
p16 = json.loads(P16.read_text(encoding="utf-8"))
q10 = json.loads(Q10.read_text(encoding="utf-8"))
assert cap["dataset"] == DS
L40 = cap["banks"]["cap_s23_L40"]

# ---- panel (a): zero-shot directions on sycophancy_mt, layer 40, mean pooling ---------------
j = p16["datasets"].index(DS)
assert p16["layer"] == 40 and p16["pooling"] == "mean" and p16["bank_name"] == "paper16"
pc1 = next(z for z in p16["zeroshot"] if z["direction"] == "PC1")
q = q10["results"]["paper16|mean"]
zs = L40["zeroshot"]
# (label, auroc, lo, hi, group) ; group "cap" = the capitulation bank, "p16" = paper16
ZERO = [
    ("capitulation v position holding", zs["contrast"]["auroc"], zs["contrast"]["lo"],
     zs["contrast"]["hi"], "cap"),
    ("capitulation v anchor", zs["contrast:cap_vs_default"]["auroc"],
     zs["contrast:cap_vs_default"]["lo"], zs["contrast:cap_vs_default"]["hi"], "cap"),
    ("paper16 assistant direction", q["assistant"]["key_layers"]["40"]["auroc"],
     *q["assistant"]["key_layers"]["40"]["ci95"], "p16"),
    ("paper16 deception contrast", q["deception"]["key_layers"]["40"]["auroc"],
     *q["deception"]["key_layers"]["40"]["ci95"], "p16"),
    ("paper16 deception PC1", pc1["auroc"][j], pc1["ci95_lo"][j], pc1["ci95_hi"][j], "p16"),
    ("paper16 sycophancy contrast", q["sycophancy"]["key_layers"]["40"]["auroc"],
     *q["sycophancy"]["key_layers"]["40"]["ci95"], "p16"),
]
NULLS = ROOT / "run-coords" / "zeroshot_nulls.json"
nul = json.loads(NULLS.read_text(encoding="utf-8"))["sycophancy_mt"]["L40_mean"]
assert nul["n_random"] == 1000
assert abs(nul["sycophancy_axis"]["auroc"] - q["sycophancy"]["key_layers"]["40"]["auroc"]) < 1e-12
null_hi = nul["random_p95_auroc_dev"]          # 95th pct of |AUROC-0.5| over 1,000 directions
null_lo = 1.0 - null_hi

# ---- panel (b): trained transfer into sycophancy_mt, per source, layer 40 -------------------
srcs = L40["sources"]
ps = L40["per_source"]
cap2, raw, dp2 = ps["pc:capitulation@2"], ps["raw"], ps["datapca@2"]

# optional cross-check against the cached lab matrix
if LABCACHE.exists():
    m = json.loads(LABCACHE.read_text(encoding="utf-8"))
    dsm = m["datasets"]; jj = dsm.index(DS)
    for feat in ("pc:capitulation@2", "raw", "datapca@2"):
        r = next(r for r in m["results"] if r["feature"] == feat and r["probe"] == "paper")
        col = [r["auroc"][i][jj] for i in range(len(dsm)) if i != jj]
        assert np.allclose(col, ps[feat]), feat
    zc = next(z for z in m["zeroshot"] if z["direction"] == "contrast")
    assert abs(zc["auroc"][jj] - zs["contrast"]["auroc"]) < 1e-12
    print("cross-check vs lab cache matrix: per-source columns and zero-shot contrast identical")

# ---- print everything plotted, and the bootstrap of cap_significance.py ---------------------
print(f"source: {CAP.relative_to(ROOT)}; {P16.relative_to(ROOT)}; {Q10.relative_to(ROOT)}; "
      f"{NULLS.relative_to(ROOT)}")
print("\n(a) zero-shot AUROC on sycophancy_mt, layer 40, mean pooling (row-bootstrap 95% CI)")
for lab, a, lo, hi, g in ZERO:
    print(f"  {lab:34s} {a:.4f}  [{lo:.4f}, {hi:.4f}]  ({g})")
print(f"  random-direction null (1,000 directions), 95th percentile of |AUROC-0.5|: "
      f"[{null_lo:.4f}, {null_hi:.4f}]")
for k in ("sycophancy_axis", "assistant_axis", "deception_axis"):
    print(f"  null p for {k}: {nul[k]['p_vs_random']:.4f} (auroc {nul[k]['auroc']:.4f})")
print(f"  A - B (capitulation v holding minus v anchor), L20/L40/L44: " + ", ".join(
    f"L{L} {cap['banks'][f'cap_s23_L{L}']['zeroshot']['contrast']['auroc'] - cap['banks'][f'cap_s23_L{L}']['zeroshot']['contrast:cap_vs_default']['auroc']:+.4f}"
    for L in (20, 40, 44)))
print("  capitulation contrast L20/L40/L44: " + ", ".join(
    f"{cap['banks'][f'cap_s23_L{L}']['zeroshot']['contrast']['auroc']:.4f}" for L in (20, 40, 44)))

print("\n(b) trained transfer into sycophancy_mt, layer 40, paper probe, per source")
print(f"  {'source':34s} {'cap@2':>7s} {'raw':>7s} {'datapca@2':>9s}")
for s, a, b, c in zip(srcs, cap2, raw, dp2):
    print(f"  {s:34s} {a:7.4f} {b:7.4f} {c:9.4f}")
print(f"  means: cap@2 {np.mean(cap2):.4f}  raw {np.mean(raw):.4f}  datapca@2 {np.mean(dp2):.4f}; "
      f"cap@2 beats raw on {int(np.sum(np.array(cap2) > np.array(raw)))} of {len(srcs)} sources")

PAIRS = [("pc:capitulation@2", "raw"), ("pc:capitulation@2", "datapca@2"),
         ("pc:capitulation@2", "random@2"), ("pc:capitulation@2", "length"),
         ("pc:capitulation@2", "pc:cap_only@2"), ("pc:cap_only@2", "raw")]
print("\npaired dataset bootstrap over the 10 sources (cap_significance.py method)")
for L in (20, 40, 44):
    rec = cap["banks"][f"cap_s23_L{L}"]
    cols = {f: np.array(v, dtype=float) for f, v in rec["per_source"].items()}
    n = len(rec["sources"])
    idx = np.random.default_rng(42).integers(0, n, size=(10000, n))
    for a, b in PAIRS:
        dif = cols[a] - cols[b]
        bs = dif[idx].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  L{L}  {a+' - '+b:36s} {dif.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"p(diff<=0) {float((bs <= 0).mean()):.4f}")

# ---- draw ------------------------------------------------------------------------------
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE, GREY = "#1c5cab", "#eb6834", "#898781"
COL = {"cap": BLUE, "p16": ORANGE}
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.8, 2.9),
                               gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.95})

# (a)
axA.axvspan(null_lo, null_hi, color="#f0efea", zorder=0)
axA.text(0.5, -0.62, "random-direction 95% band", ha="center", va="center",
         fontsize=6.5, color=MUTED)
axA.axvline(0.5, color=GREY, lw=0.8, ls="--", zorder=1)
for i, (lab, a, lo, hi, g) in enumerate(ZERO):
    axA.plot([lo, hi], [i, i], color=COL[g], lw=1.6, zorder=3, solid_capstyle="round")
    axA.plot([a], [i], "o", color=COL[g], ms=5.5, mec="white", mew=0.8, zorder=4)
    axA.text(hi + 0.012, i, f"{a:.3f}", va="center", fontsize=7, color=INK)
axA.set_yticks(range(len(ZERO)))
axA.set_yticklabels([z[0] for z in ZERO], fontsize=7.5)
axA.set_ylim(len(ZERO) - 0.5, -0.95)
axA.set_xlim(0.2, 0.85)
axA.set_xlabel("zero-shot AUROC", color=MUTED)
axA.set_title("(a) zero-shot, layer 40", fontsize=8.5, loc="left", color=INK)

# (b)
order = sorted(range(len(srcs)), key=lambda i: raw[i])
for r, i in enumerate(order):
    axB.plot([raw[i], cap2[i]], [r, r], color=GRID, lw=1.4, zorder=1)
    axB.plot([raw[i]], [r], "o", color=GREY, ms=5, mec="white", mew=0.8, zorder=3)
    axB.plot([dp2[i]], [r], "D", color=ORANGE, ms=4, mec="white", mew=0.6, zorder=3)
    axB.plot([cap2[i]], [r], "o", color=BLUE, ms=5.5, mec="white", mew=0.8, zorder=4)
axB.axvline(0.5, color=GREY, lw=0.8, ls="--", zorder=1)
axB.set_yticks(range(len(order)))
axB.set_yticklabels([srcs[i].replace("harm-pressure-knowledge-report", "HP-KR")
                     .replace("harm-pressure-kr-matched", "HP-KR text-matched")
                     .replace("harm-pressure-choice", "HP-C")
                     .replace("instructed-deception", "ID").replace("convincing-game", "CG")
                     .replace("insider-trading", "IT") for i in order], fontsize=7.5)
axB.invert_yaxis()
axB.set_xlim(0.3, 0.8)
axB.set_xlabel("trained AUROC into sycophancy_mt", color=MUTED)
axB.set_title("(b) trained, per source, layer 40", fontsize=8.5, loc="left", color=INK)
axB.plot([], [], "o", color=BLUE, ms=5, label="capitulation basis, k = 2")
axB.plot([], [], "D", color=ORANGE, ms=4, label="dataset PCA, k = 2")
axB.plot([], [], "o", color=GREY, ms=5, label="raw")
axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False,
           fontsize=6.5, handletextpad=0.3, columnspacing=0.8)
axA.plot([], [], "o", color=BLUE, ms=5, label="capitulation bank")
axA.plot([], [], "o", color=ORANGE, ms=5, label="paper16 bank")
axA.legend(loc="upper center", bbox_to_anchor=(0.3, -0.2), ncol=2, frameon=False,
           fontsize=6.5, handletextpad=0.3, columnspacing=0.8)
for ax in (axA, axB):
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

out = ROOT / "thesis" / "images" / "fig_capitulation.pdf"
fig.savefig(out, bbox_inches="tight")
prev = ROOT / "thesis_tools" / "figures" / "preview" / "fig_capitulation.png"
fig.savefig(prev, dpi=160, bbox_inches="tight")
print("\nwrote", out)
print("wrote", prev)
