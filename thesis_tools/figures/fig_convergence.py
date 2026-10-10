"""Figure: the convergence -- three wording-controlled tests of HP-KR-style knowledge denial.

Chapter 5, Section "HP-KR, the factor probes and the convergence" (fig:r-convergence).
Reads the three raw sources and hard-codes no number; every plotted value is printed.

  (a) text-matched HP-KR slice (258 rows; every response denies knowledge), Pipeline A,
      zero-shot, kd bank at judge scores 2-3 (the "collab" bank, which rebuilds Pipeline B's
      12 v 13 contrast), mean pooling. Source (lab-only, local cache):
      audit/cache/labpc/run-crossgen/kd11/banks/collab/matrices/mean_L{20,40,44}_s42.json
      :: zeroshot[direction].auroc / ci95_lo / ci95_hi, columns harm-pressure-kr-matched and
      harm-pressure-knowledge-report.
  (b) knowledge-matched HP-KR slice (2,001 rows; the model knows every answer), Pipeline A,
      trained `paper` probe on the slice's 80% split, leak guard on, paper16 bank, mean pooling.
      Off-diagonal = mean over the other 11 datasets; in-distribution = own 20% split.
      Source (lab-only, local cache):
      audit/cache/labpc/expLB-resplit/km/mean_L{20,40,44}_s42.json :: results[feature,paper].auroc
  (c) Gender-Secret (392 rows, 116 lies), Pipeline B, zero-shot, adapted read.
      Source: git show 16def2de:src/probes/deception-detection/results/gender_secret/zero_shot.csv
      :: rows read == "adapted", columns auroc / auroc_lo / auroc_hi.

    python thesis_tools/figures/fig_convergence.py

Writes thesis/images/fig_convergence.pdf and thesis_tools/figures/preview/fig_convergence.png.
"""
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LAYERS = [20, 40, 44]
XPOS = [0, 1, 2]  # equally spaced x positions for layers 20, 40, 44
COLLAB = ROOT / "audit/cache/labpc/run-crossgen/kd11/banks/collab/matrices"
KM = ROOT / "audit/cache/labpc/expLB-resplit/km"
GS_BLOB = "16def2de:src/probes/deception-detection/results/gender_secret/zero_shot.csv"
OUT_PDF = ROOT / "thesis/images/fig_convergence.pdf"
OUT_PNG = ROOT / "thesis_tools/figures/preview/fig_convergence.png"

TM, FULL, KMD = "harm-pressure-kr-matched", "harm-pressure-knowledge-report", \
    "harm-pressure-kr-knowledge-matched"

# reference palette (dataviz skill, light mode); colour follows the entity across panels
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, VIOLET, RED = "#e87ba4", "#4a3aa7", "#e34948"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e2dc"


def zeroshot(layer: int) -> dict:
    d = json.loads((COLLAB / f"mean_L{layer}_s42.json").read_text(encoding="utf-8"))
    assert d["bank_name"] == "collab" and d["pooling"] == "mean", (d["bank_name"], d["pooling"])
    ds = d["datasets"]
    out = {}
    for z in d["zeroshot"]:
        out[z["direction"]] = {col: (z["auroc"][ds.index(col)], z["ci95_lo"][ds.index(col)],
                                     z["ci95_hi"][ds.index(col)]) for col in (TM, FULL)}
    return out


def km(layer: int) -> dict:
    d = json.loads((KM / f"mean_L{layer}_s42.json").read_text(encoding="utf-8"))
    assert d["bank_name"] == "paper16" and "LEAK GUARD" in d["design"]
    ds = d["datasets"]
    i, n = ds.index(KMD), len(ds)
    out = {}
    for r in d["results"]:
        if r["probe"] != "paper":
            continue
        a = np.array(r["auroc"], dtype=float)
        out[r["feature"]] = {"off": float(np.mean([a[i, j] for j in range(n) if j != i])),
                             "diag": float(a[i, i]), "n_targets": n - 1}
    return out


def gender_secret() -> pd.DataFrame:
    txt = subprocess.run(["git", "show", GS_BLOB], cwd=ROOT, capture_output=True, text=True,
                         check=True).stdout
    g = pd.read_csv(io.StringIO(txt))
    return g[g["read"] == "adapted"].set_index(["detector", "layer"])


def style(ax, title):
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.set_xticks(XPOS, ["20", "40", "44\n(post hoc)"])
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(0.2, 1.02)
    ax.set_xlabel("layer")
    ax.set_title(title, loc="left", fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2, labelsize=7)


def main() -> None:
    plt.rcParams.update({"font.size": 7.5, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.75), sharey=True, constrained_layout=True)
    x = np.array(XPOS, dtype=float)

    # (a) text-matched slice, zero-shot, collab bank (kd, judge scores 2-3)
    zs = {L: zeroshot(L) for L in LAYERS}
    print("(a) text-matched slice, zero-shot AUROC [row-bootstrap 95% CI], collab bank, mean")
    ax = axes[0]
    series_a = [("contrast:collab_extended", TM, "extended (12 v 13), text-matched", BLUE, "o", "-", -0.08),
                ("contrast:knowledge_denial", TM, "knowledge denial (4 v 3), text-matched", MAGENTA, "s", "-", 0.0),
                ("contrast:kd_assertion", TM, "denial v assertion (4 v 3), text-matched", VIOLET, "^", "-", 0.08),
                ("contrast:collab_extended", FULL, "extended (12 v 13), full HP-KR", BLUE, "o", ":", -0.08),
                ("contrast:kd_assertion", FULL, "denial v assertion (4 v 3), full HP-KR", VIOLET, "^", ":", 0.08)]
    for key, col, lab, colr, mk, ls, dx in series_a:
        v = np.array([zs[L][key][col] for L in LAYERS])
        print(f"    {key:28s} {col:32s} " + "  ".join(
            f"L{L} {a:.4f} [{lo:.4f}, {hi:.4f}]" for L, (a, lo, hi) in zip(LAYERS, v)))
        hollow = ls == ":"
        ax.errorbar(x + dx, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]],
                    color=colr, lw=1.4 if not hollow else 1.0, ls=ls, marker=mk, ms=4.2,
                    mfc="white" if hollow else colr, mec=colr, mew=1.0,
                    elinewidth=0.8, capsize=0 if hollow else 1.5, label=lab, zorder=3)
    style(ax, "(a) text-matched slice\nzero-shot, Pipeline A")
    ax.set_ylabel("AUROC")

    # (b) knowledge-matched slice as the training source, leak guard on
    kms = {L: km(L) for L in LAYERS}
    print("(b) knowledge-matched slice, trained paper probe, paper16 bank, mean, leak guard")
    ax = axes[1]
    series_b = [("axes", "off", "persona axes, transfer", RED, "o", "-"),
                ("raw", "off", "raw, transfer", MUTED, "s", "-"),
                ("raw", "diag", "raw, own 20% split", MUTED, "s", ":"),
                ("length", "off", "length, transfer", "#9a9890", "D", "--")]
    for feat, which, lab, colr, mk, ls in series_b:
        v = [kms[L][feat][which] for L in LAYERS]
        print(f"    {feat:8s} {which:4s} " + "  ".join(f"L{L} {a:.4f}" for L, a in zip(LAYERS, v))
              + f"   (off-diagonal over {kms[LAYERS[0]][feat]['n_targets']} targets)")
        hollow = ls == ":"
        ax.plot(x, v, color=colr, lw=1.4 if not hollow else 1.0, ls=ls, marker=mk, ms=4.2,
                mfc="white" if hollow else colr, mec=colr, mew=1.0, label=lab, zorder=3)
    for L in LAYERS:
        print(f"    L{L} axes - raw (transfer) {kms[L]['axes']['off'] - kms[L]['raw']['off']:+.4f}")
    style(ax, "(b) knowledge-matched slice\ntrained on it, Pipeline A")

    # (c) Gender-Secret, zero-shot, adapted read (Pipeline B)
    g = gender_secret()
    print("(c) Gender-Secret, zero-shot AUROC [95% CI], adapted read (Pipeline B)")
    ax = axes[2]
    series_c = [("extended (12v13)", "extended (12 v 13)", BLUE, "o", -0.12),
                ("published (8v7)", "published (8 v 7)", ORANGE, "v", -0.04),
                ("knowledge-denial (4v6)", "knowledge denial (4 v 6)", AQUA, "s", 0.04),
                ("sycophancy (5v5)", "sycophancy (5 v 5)", YELLOW, "D", 0.12)]
    for det, lab, colr, mk, dx in series_c:
        v = np.array([[g.loc[(det, L), "auroc"], g.loc[(det, L), "auroc_lo"],
                       g.loc[(det, L), "auroc_hi"]] for L in LAYERS], dtype=float)
        n = int(g.loc[(det, LAYERS[0]), "n"])
        print(f"    {det:24s} n={n} " + "  ".join(
            f"L{L} {a:.4f} [{lo:.4f}, {hi:.4f}]" for L, (a, lo, hi) in zip(LAYERS, v)))
        ax.errorbar(x + dx, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=colr,
                    lw=1.4, marker=mk, ms=4.2, mec="white", mew=0.6, elinewidth=0.8,
                    capsize=1.5, label=lab, zorder=3)
    style(ax, "(c) Gender-Secret\nzero-shot, Pipeline B")

    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), frameon=False, fontsize=6.2,
                  handlelength=2.2, borderaxespad=0.0)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT_PNG, bbox_inches="tight", pad_inches=0.02, dpi=200)
    print(f"wrote {OUT_PDF}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
