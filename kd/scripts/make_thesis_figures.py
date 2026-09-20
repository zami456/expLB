#!/usr/bin/env python
"""Figures for the four findings that carry the thesis claim.

The claim: persona-coordinate probes measure REGISTER rather than the target behaviour, and
their generalisation gains are an artifact of weak baselines. Each figure is one leg of it.

  fig1_register      the axis scores BELOW chance on the two datasets whose labels are the
                     construct it is supposed to detect, and below a random direction (F-13)
  fig2_layer_profile the gain is smooth across depth, negative early, peaks at L44 -- a real
                     depth-dependent effect that still clears no significance bar (F-34, F-31)
  fig3_baseline      against the MATCHED control the paper replicates (+0.202 at 3B vs their
                     +0.22) and the margin shrinks with scale; against raw it buys almost
                     nothing at any size (F-63, which superseded F-58)
  fig4_geometry      persona geometry is size-dependent and NOT monotone -- cos(PC1, assistant)
                     peaks at 8B, and the 3B fails the source paper's own thresholds (F-37)
  fig5_capitulation  the constructive half: vary CAPITULATION instead of register and the
                     axis reads 0.745 zero-shot where the register axis reads 0.374, clears
                     a dataset-level bootstrap against raw and both matched controls, and
                     wins by being STABLE across sources, not high on any one (F-65, F-66)

Runs on the LAPTOP. matplotlib cannot load on the lab PC (Application Control blocks its
DLLs), so the numbers are exported there by export_fig_data.py and plotted here.

    python kd/scripts/make_thesis_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures" / "thesis"

INK = "#1b1b1f"
MUTED = "#6b6b76"
GRID = "#dcdce4"
AXIS = "#2f6f9f"      # the persona axis
CTRL = "#c4623a"      # controls: random, dataset PCA
RAW = "#7a7a86"       # raw activations
GOOD = "#3f7d58"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/thesis/{name}.png / .pdf")


# --------------------------------------------------------------------------------------
def fig1_register() -> None:
    """F-13: on the two datasets that test the construct, the axis is below chance."""
    datasets = ["harm-pressure-\nknowledge-report", "sycophancy_mt"]
    axis = [0.300, 0.379]
    random_ = [0.659, 0.633]
    raw = [0.694, 0.665]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = range(len(datasets))
    w = 0.26
    ax.bar([i - w for i in x], raw, w, label="raw activations", color=RAW)
    ax.bar(list(x), random_, w, label="random direction", color=CTRL)
    ax.bar([i + w for i in x], axis, w, label="persona axis", color=AXIS)

    ax.axhline(0.5, color=INK, lw=1, ls="--", zorder=0)
    ax.text(1.42, 0.505, "chance", fontsize=8, color=INK, va="bottom", ha="right")
    for i, v in enumerate(axis):
        ax.text(i + w, v + 0.012, f"{v:.3f}", ha="center", fontsize=8.5,
                color=AXIS, fontweight="bold")

    ax.set_xticks(list(x)); ax.set_xticklabels(datasets)
    ax.set_ylabel("AUROC (trained, as source)")
    ax.set_ylim(0, 0.85)
    ax.set_title("The persona axis is beaten by a random direction\n"
                 "on both datasets whose labels test its own construct", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=1)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    save(fig, "fig1_register")


# --------------------------------------------------------------------------------------
def fig2_layer_profile(doc: dict) -> None:
    """F-34 + F-31: a clean depth-dependent shape that clears no significance bar."""
    prof = doc["profile"]
    vs_raw = prof["pc:unified - raw"]
    vs_pca = prof["pc:unified - datapca"]
    layers = [r["layer"] for r in vs_raw]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhline(0, color=INK, lw=1, zorder=1)

    lo = [r["lo"] for r in vs_raw]; hi = [r["hi"] for r in vs_raw]
    if all(v is not None for v in lo):
        ax.fill_between(layers, lo, hi, color=AXIS, alpha=0.13, lw=0,
                        label="95% CI over datasets (vs raw)")
    ax.plot(layers, [r["diff"] for r in vs_raw], "-o", ms=3.4, lw=1.7, color=AXIS,
            label="persona axis − raw")
    ax.plot([r["layer"] for r in vs_pca], [r["diff"] for r in vs_pca], "--s", ms=3.0,
            lw=1.5, color=CTRL, label="persona axis − dataset PCA (matched k)")

    peak = max(vs_raw, key=lambda r: r["diff"])
    ax.annotate(f"peak L{peak['layer']}\n{peak['diff']:+.3f}",
                xy=(peak["layer"], peak["diff"]), xytext=(peak["layer"] + 7, peak["diff"] + 0.055),
                fontsize=8, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("negative in the first quarter\nof the network", xy=(10, -0.030),
                xytext=(2, -0.145), fontsize=8, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

    c = doc["census"]
    ax.text(0.99, 0.04,
            f"{c['n_clearing']} of {c['n_tests']:,} tests clear a dataset-level bootstrap\n"
            f"({c['n_clearing_positive']} of those are positive; best q = {min(r['q'] for r in vs_raw):.3f})",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)

    ax.set_xlabel("layer (of 80)"); ax.set_ylabel("Δ mean off-diagonal AUROC")
    ax.set_title("The gain is depth-dependent and well behaved — and still not significant",
                 loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    save(fig, "fig2_layer_profile")


# --------------------------------------------------------------------------------------
def fig3_baseline() -> None:
    """F-63: two series that tell different stories, and both are needed.

    Against RAW the gain is small and flat -- we do not reproduce the paper's +0.10/+0.12 at any
    size, because our raw off-diagonal is 0.76-0.81 where theirs is ~0.60.
    Against the MATCHED control the paper replicates (+0.202 at 3B vs their +0.22) and the
    margin declines monotonically with scale.

    Superseded F-58's version, which compared different dataset sets AND different k.
    """
    rungs = ["3B\nL16", "8B\nL16", "70B\nL40"]
    raw = [0.7619, 0.7980, 0.8090]
    vs_raw = [0.020, 0.013, 0.013]
    vs_pca = [0.202, 0.115, 0.079]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    x = range(len(rungs))
    ax.plot(x, vs_pca, "-o", ms=6, lw=2, color=AXIS,
            label="persona axis − dataset PCA (matched k)")
    ax.plot(x, vs_raw, "--s", ms=5, lw=1.6, color=RAW, label="persona axis − raw")
    ax.axhline(0, color=INK, lw=1)

    # The 3B point sits directly under the paper's star and its raw label lands on the zero
    # line, so both of its labels are offset sideways rather than stacked vertically.
    for i, (a, b) in enumerate(zip(vs_pca, vs_raw)):
        dx, ha = (0.10, "left") if i == 0 else (0.0, "center")
        ax.text(i + dx, a - 0.018, f"{a:+.3f}", ha=ha, fontsize=8.5,
                color=AXIS, fontweight="bold")
        ax.text(i + dx, b + 0.012, f"{b:+.3f}", ha=ha, fontsize=8, color=MUTED)

    # The paper's own 3B deception figure: axis +0.10 against dataset PCA -0.12.
    ax.scatter([0], [0.22], marker="*", s=180, color=CTRL, zorder=4)
    ax.annotate("paper, 3B: +0.22", xy=(0, 0.22), xytext=(0.30, 0.243), fontsize=8,
                color=CTRL, arrowprops=dict(arrowstyle="-", color=CTRL, lw=0.8))

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{r}\nraw {v:.3f}" for r, v in zip(rungs, raw)])
    ax.set_ylabel("Δ mean off-diagonal AUROC")
    ax.set_ylim(-0.04, 0.27)
    ax.set_title("Against the matched control the paper replicates, and the margin\n"
                 "shrinks with scale. Against raw it buys almost nothing at any size.",
                 loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.text(0.99, 0.55, "same 5 datasets, same k=3\nonly the model changes",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=MUTED, style="italic")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    save(fig, "fig3_baseline_artifact")


# --------------------------------------------------------------------------------------
def fig4_geometry() -> None:
    """F-37: size-dependent, non-monotone; the 3B fails the source paper's thresholds."""
    rungs = ["3B\n(L14/28)", "8B\n(L16/32)", "70B\n(L40/80)"]
    assistant = [0.435, 0.732, 0.623]
    deception = [0.924, 0.921, 0.955]
    sycophancy = [0.948, 0.983, 0.993]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.3),
                                 gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.3})

    a1.bar(rungs, assistant, 0.55, color=[CTRL, GOOD, AXIS])
    a1.axhline(0.71, color=INK, lw=1, ls="--")
    # Placed low-left: at upper-right it collided with the 0.732 bar label, which by
    # construction sits just above the threshold line.
    a1.text(-0.42, 0.66, "Assistant Axis paper threshold", fontsize=7.5,
            color=INK, ha="left", va="top")
    for i, v in enumerate(assistant):
        a1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    a1.set_ylim(0, 0.92); a1.set_ylabel("cos(PC1, assistant axis)")
    a1.set_title("Not monotone — it peaks at 8B", loc="left")
    a1.yaxis.grid(True); a1.set_axisbelow(True)

    x = range(len(rungs)); w = 0.35
    a2.bar([i - w / 2 for i in x], deception, w, label="deception", color=AXIS)
    a2.bar([i + w / 2 for i in x], sycophancy, w, label="sycophancy", color=GOOD)
    a2.set_xticks(list(x)); a2.set_xticklabels(rungs)
    a2.set_ylim(0.85, 1.0); a2.set_ylabel("cos(PC1, own contrast)")
    a2.set_title("Contrast alignment does rise with size —\nand is a construction check, not a result",
                 loc="left")
    a2.legend(frameon=False, fontsize=8, loc="lower right")
    a2.yaxis.grid(True); a2.set_axisbelow(True)
    save(fig, "fig4_geometry_by_scale")


# --------------------------------------------------------------------------------------
def fig5_capitulation(cap: dict) -> None:
    """F-65/F-66: vary the right construct and the axis works -- significantly, and because
    it is STABLE rather than because it is high.

    Three panels because the result needs all three to be honest. (a) it beats everything it
    is matched against; (b) the margin clears a dataset-level bootstrap, which nothing
    positive in this project had cleared before; (c) the mechanism is that the axis barely
    notices which dataset trained it, while raw swings from 0.35 to 0.75.
    """
    L = cap["banks"]["cap_s23_L40"]
    m, zs, p16 = L["means"], L["zeroshot"], cap["paper16_L40"]

    def zsv(name):
        """A zero-shot entry is {auroc, lo, hi} since 2026-09-21 and was a bare float before.
        Accept both so an older exported payload still renders."""
        v = zs.get(name)
        return v.get("auroc") if isinstance(v, dict) else v

    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(15.0, 4.8),
        gridspec_kw={"width_ratios": [1.2, 1.0, 0.85], "wspace": 0.62})

    # ---- (a) the column -----------------------------------------------------------
    rows = [
        ("pc:capitulation@2", m.get("pc:capitulation@2"), AXIS, "capitulation axis, k=2"),
        ("pc:capitulation", m.get("pc:capitulation"), AXIS, "capitulation axis, k=5"),
        ("pc:cap_only@2", m.get("pc:cap_only@2"), "#9ab8cf",
         "cap axis, holding personas dropped"),
        ("raw", m.get("raw"), RAW, "raw activations (8192-d)"),
        ("length", m.get("length"), CTRL, "response length (1-d control)"),
        ("datapca@2", m.get("datapca@2"), CTRL, "dataset PCA, k=2 (matched control)"),
        ("random@2", m.get("random@2"), CTRL, "random directions, k=2 (matched)"),
        ("pc:unified@2", p16.get("pc:unified@2"), "#b04a4a", "paper16 unified axis, k=2"),
        ("pc:deception", p16.get("pc:deception"), "#b04a4a", "paper16 deception axis"),
        ("axes", p16.get("axes"), "#b04a4a", "paper16 stacked axes"),
    ]
    rows = [r for r in rows if r[1] is not None]
    y = list(range(len(rows)))
    axA.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.68, zorder=3)
    axA.set_yticks(y)
    axA.set_yticklabels([r[3] for r in rows], fontsize=8)
    axA.invert_yaxis()
    for i, r in enumerate(rows):
        axA.text(r[1] + 0.006, i, f"{r[1]:.3f}", va="center", fontsize=7.5, color=INK)
    axA.axvline(0.5, color=GRID, lw=1.0, zorder=1)
    # The two zero-shot numbers are the cleanest form of the result, but they are a DIFFERENT
    # quantity from the trained bars (no probe, 100% of rows), so they are reference lines
    # labelled as such rather than bars in the same series.
    # Stacked on two rows of headroom ABOVE the bars: at 0.374 the register line runs straight
    # through the paper16 bars, so a label beside it is unreadable either way up.
    axA.set_ylim(len(rows) - 0.4, -1.45)
    for xv, col, lab, yv in ((zsv("contrast"), AXIS, "capitulation contrast", -0.62),
                             (cap["register_baseline_zeroshot"], "#b04a4a",
                              "register axis", -1.08)):
        axA.axvline(xv, color=col, ls="--", lw=1.2, zorder=2)
        axA.plot([xv], [yv], "v", color=col, ms=4, zorder=4)
        axA.text(xv + 0.008, yv, f"{lab}, zero-shot {xv:.3f}",
                 va="center", ha="left", fontsize=7.5, color=col)
    axA.set_xlim(0.30, 0.82)
    axA.set_xlabel("AUROC on sycophancy_mt  (off-diagonal mean)")
    axA.set_title("(a)  vary capitulation instead of register", loc="left", fontweight="bold")
    axA.xaxis.grid(True, zorder=0)

    # ---- (b) the bootstrap --------------------------------------------------------
    LAB = {
        "pc:capitulation@2 - raw": "vs raw",
        "pc:capitulation@2 - datapca@2": "vs dataset PCA, k=2",
        "pc:capitulation@2 - random@2": "vs random, k=2",
        "pc:capitulation@2 - length": "vs length",
        "pc:capitulation@2 - pc:cap_only@2": "vs itself minus\nholding personas",
        "pc:cap_only@2 - raw": "holding personas dropped:\nvs raw",
    }
    keys = [k for k in LAB if k in L["boot"]]
    for i, k in enumerate(keys):
        b = L["boot"][k]
        c = GOOD if b["lo"] > 0 else MUTED
        axB.plot([b["lo"], b["hi"]], [i, i], color=c, lw=2.0, solid_capstyle="round", zorder=3)
        axB.plot([b["diff"]], [i], "o", color=c, ms=6, zorder=4)
        axB.text(b["hi"] + 0.008, i,
                 f"p={b['p']:.4f}" if b["p"] >= 1e-4 else "p<0.0001",
                 va="center", fontsize=7.5, color=c)
    axB.axvline(0, color=INK, lw=1.0, zorder=2)
    axB.set_yticks(range(len(keys)))
    axB.set_yticklabels([LAB[k] for k in keys], fontsize=8)
    axB.invert_yaxis()
    axB.set_xlim(-0.16, 0.34)
    axB.set_xlabel("AUROC difference  (95% CI, 10 sources)")
    axB.set_title("(b)  clears the dataset-level bar\n      on the column it was aimed at",
                  loc="left", fontweight="bold")
    axB.xaxis.grid(True, zorder=0)

    # ---- (c) why: stability, not height -------------------------------------------
    srcs = L["sources"]
    ca, cb = L["per_source"]["pc:capitulation@2"], L["per_source"]["raw"]
    order = sorted(range(len(srcs)), key=lambda i: cb[i])
    for r, i in enumerate(order):
        axC.plot([cb[i], ca[i]], [r, r], color=GRID, lw=1.2, zorder=1)
        axC.plot([cb[i]], [r], "o", color=RAW, ms=5, zorder=3)
        axC.plot([ca[i]], [r], "o", color=AXIS, ms=5, zorder=3)
    axC.axvline(0.5, color="#b04a4a", ls=":", lw=1.0, zorder=2)
    axC.set_yticks(range(len(order)))
    axC.set_yticklabels([srcs[i] for i in order], fontsize=7.5)
    axC.yaxis.tick_right()          # else these labels land on top of panel (b)
    axC.invert_yaxis()
    axC.set_xlim(0.30, 0.82)
    axC.set_xlabel("AUROC into sycophancy_mt")
    axC.set_title("(c)  the axis barely notices its source", loc="left", fontweight="bold")
    axC.xaxis.grid(True, zorder=0)
    axC.plot([], [], "o", color=RAW, ms=5, label="raw")
    axC.plot([], [], "o", color=AXIS, ms=5, label="capitulation axis, k=2")
    axC.legend(loc="lower left", fontsize=7.5, frameon=False)

    fig.suptitle("The register diagnosis, demonstrated constructively  "
                 "(Llama-3.3-70B, layer 40, judge-filtered; unfiltered agrees to 0.004)",
                 fontsize=10, y=1.03)
    fig.text(0.5, -0.06,
             "The gain is LOCAL (F-67): averaged over all eleven columns the margin over raw is "
             "+0.098, 95% CI [-0.036, +0.208] -- it does not clear. The axis wins where it was "
             "aimed and ties elsewhere,\nthe same inverse relation between persona gain and "
             "baseline strength that F-63 found across model scales.",
             ha="center", va="top", fontsize=7.5, color=MUTED)
    save(fig, "fig5_capitulation")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(REPO / "kd" / "results" / "fig_data.json"))
    ap.add_argument("--cap", default=str(REPO / "run-cap" / "export" / "cap_fig_data.json"))
    args = ap.parse_args()
    doc = json.load(open(args.data, encoding="utf-8"))
    print("building thesis figures:")
    fig1_register()
    fig2_layer_profile(doc)
    fig3_baseline()
    fig4_geometry()
    if Path(args.cap).exists():
        fig5_capitulation(json.load(open(args.cap, encoding="utf-8")))
    else:
        print(f"  SKIP fig5: {args.cap} not found -- export it on the lab PC "
              "with cap/scripts/export_cap_fig.py")
    print(f"\nall in {OUT}")


if __name__ == "__main__":
    main()
