#!/usr/bin/env python
"""Figures for the 10 x 10 cross-generalization retrain (27_crossgen.py).

Reads run-crossgen/ and writes ONLY to figures/crossgen/ -- never figures/persona-coords/.
Runs on the laptop (the lab PC blocks matplotlib's DLLs).

    python 28_crossgen_figures.py --only fig6 --layers 40

Colour: one diverging pair everywhere, validated for colour-vision deficiency -- blue = above
chance / better than raw, red = below chance / worse than raw, neutral grey at the midpoint.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]
RUN = REPO_ROOT / "run-crossgen"
FIGDIR = REPO_ROOT / "figures" / "crossgen"
# --run / --figdir rebind these (see main). The 11-dataset grid lives in
# run-crossgen/kd11 because 27_crossgen.py's guard() demands "run-crossgen" as an exact
# path component, so it cannot sit under kd/.
N_DATASETS = 10   # rebound from the JSON actually loaded; titles must not assert 10

SHORT = {"instructed-deception": "ID", "harm-pressure-choice": "HP-C",
         "harm-pressure-knowledge-report": "HP-KR",
         # the 258 HP-KR rows whose responses ALL deny knowledge: wording held constant,
         # so only the underlying knowledge state differs
         "harm-pressure-kr-matched": "HP-KRm", "convincing-game": "CG",
         "insider-trading": "IT", "ai_liar": "AIL", "sandbagging_v2": "SB",
         "roleplaying": "RP", "mask": "MASK", "sycophancy_mt": "SYC"}
PROBE_LABEL = {"paper": "logistic regression, C=1 (paper)",
               "lr_reg10": "logistic regression, λ=10 (Apollo)",
               "mms": "mean-difference (mms)"}
FEATURE_LABEL = {"raw": "raw activation (8192-d)", "random": "random subspace",
                 "datapca": "dataset PCA", "pc:unified": "unified-axis PC projection",
                 "pc:deception": "deception-axis PC projection", "axes": "persona axes (4-d)",
                 "axes_resid": "axes, assistant removed", "length": "response length"}

# validated diverging pair (blue/red poles, neutral grey midpoint)
INK, INK_2, RULE = "#1b1b1a", "#52514e", "#8f8d87"
DIVERGING = LinearSegmentedColormap.from_list("crossgen_div", ["#b8322f", "#e34948", "#f0efec",
                                                               "#2a78d6", "#184f95"])


def guard_out(path: Path) -> Path:
    r = path.resolve()
    if FIGDIR.resolve() not in r.parents and r != FIGDIR.resolve():
        raise SystemExit(f"refusing to write outside {FIGDIR}: {r}")
    return r


def setup():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": RULE,
        "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "font.family": "DejaVu Sans"})


def save(fig, rel: str) -> Path:
    path = guard_out(FIGDIR / rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    return path.with_suffix(".png")


BASELINE_FEATURES = ("raw", "random", "random@2", "datapca", "datapca@2", "length")
BANK_LABEL = {"paper16": "paper16 (25 personas + default; family-specific questions)",
              "paper16_famcentred": "paper16, question-family centred",
              "paper16_shared": "paper16 on shared 10+10 questions",
              "own100": "own100 (100 library roles; our rollouts, 70B judge)",
              "hf100": "hf100 (same 100 roles; published vectors)",
              "hf275": "hf275 (275 library roles; published vectors)"}


def load_matrix_doc(pooling: str, layer: int, seed: int = 42, bank: str | None = None):
    """Root run = paper16 + baselines. A bank run holds persona features only; the baseline
    features are merged in from the root file with the same name (identical splits)."""
    base = RUN / "matrices" / f"{pooling}_L{layer}_s{seed}.json"
    p = base if bank in (None, "paper16") else RUN / "banks" / bank / "matrices" / base.name
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    if p != base and base.exists():
        bd = json.loads(base.read_text(encoding="utf-8"))
        have = {(r["feature"], r["probe"]) for r in doc["results"]}
        doc["results"] += [r for r in bd["results"] if r["feature"] in BASELINE_FEATURES
                           and (r["feature"], r["probe"]) not in have]
    return doc


def mat(doc, feature: str, probe: str):
    for r in doc["results"]:
        if r["feature"] == feature and r["probe"] == probe:
            return np.array([[np.nan if v is None else v for v in row] for row in r["auroc"]],
                            dtype=float)
    return None


def text_color(rgba) -> str:
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.45 else INK


def draw_matrix(ax, M, norm, labels, fmt, title, subtitle, show_y):
    n = len(labels)
    im = ax.imshow(M, cmap=DIVERGING, norm=norm, interpolation="nearest")
    for i in range(n):
        for j in range(n):
            v = M[i, j]
            if np.isnan(v):
                continue
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=6.2,
                    color=text_color(DIVERGING(norm(v))))
    for i in range(n):   # outline the diagonal: own 20% test split
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.1))
    for b in (6.5, 7.5):  # construct boundaries: intent | judge | belief
        ax.axhline(b, color="white", lw=2.2)
        ax.axvline(b, color="white", lw=2.2)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([SHORT[k] for k in labels], rotation=90)
    ax.set_yticklabels([SHORT[k] for k in labels] if show_y else [])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("tested on")
    if show_y:
        ax.set_ylabel("trained on (80% split)")
    ax.set_title(title, loc="left", fontweight="bold", color=INK, pad=18)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8, color=INK_2, va="bottom")
    return im


def fig6(pooling: str, layer: int, probe: str, k: int, seed: int = 42, bank: str | None = None):
    """Paper Figure 6 layout: raw AUROC, then random / dataset-PCA / unified-axis PCs minus raw."""
    doc = load_matrix_doc(pooling, layer, seed, bank)
    if doc is None:
        return None
    labels = doc["datasets"]
    n = len(labels)
    off = ~np.eye(n, dtype=bool)
    suffix = "" if k == 5 else f"@{k}"
    raw = mat(doc, "raw", probe)
    methods = [("random" + suffix, f"random subspace (k={k}) − raw"),
               ("datapca" + suffix, f"dataset PCA (k={k}) − raw"),
               ("pc:unified" + suffix, f"unified-axis PCs (k={k}) − raw")]
    diffs = []
    for f, _ in methods:
        M = mat(doc, f, probe)
        if M is None or raw is None:
            logger.warning("  %s/%s missing in %s_L%d", f, probe, pooling, layer)
            return None
        diffs.append(M - raw)
    m = max(0.2, np.ceil(max(np.nanmax(np.abs(d)) for d in diffs) * 10) / 10)

    fig, axes = plt.subplots(1, 4, figsize=(21.5, 6.4), gridspec_kw={"wspace": 0.06})
    norm_auc = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    norm_d = TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)
    im0 = draw_matrix(axes[0], raw, norm_auc, labels, lambda v: f"{v:.2f}", "Raw-probe AUROC",
                      f"diagonal mean {np.nanmean(np.diag(raw)):.3f} · off-diagonal mean "
                      f"{np.nanmean(raw[off]):.3f}", True)
    im1 = None
    for ax, (f, title), D in zip(axes[1:], methods, diffs):
        wins = int(np.nansum(D[off] > 0))
        im1 = draw_matrix(ax, D, norm_d, labels, lambda v: f"{v:+.2f}", title,
                          f"mean off-diagonal Δ {np.nanmean(D[off]):+.3f} · beats raw on "
                          f"{wins}/{int(off.sum())} pairs", False)
    c0 = fig.colorbar(im0, ax=axes[0], orientation="horizontal", fraction=0.045, pad=0.16)
    c0.set_label("AUROC (0.5 = chance)")
    c0.outline.set_visible(False)
    c1 = fig.colorbar(im1, ax=axes[1:].tolist(), orientation="horizontal", fraction=0.045,
                      pad=0.16, aspect=60)
    c1.set_label("change in AUROC vs raw probe (blue = better than raw)")
    c1.outline.set_visible(False)
    bank_txt = "" if bank in (None, "paper16") else f" · persona bank: {BANK_LABEL.get(bank, bank)}"
    fig.suptitle(f"Layer {layer} · {pooling} pooling · probe: {PROBE_LABEL[probe]} · "
                 f"Llama-3.3-70B-Instruct, {N_DATASETS} datasets{bank_txt}",
                 x=0.125, ha="left", y=1.03, fontsize=12, fontweight="bold", color=INK)
    fig.text(0.125, 0.985, "Outlined diagonal: tested on the training dataset's own 20% split. "
             "Off-diagonal: tested on 100% of the other dataset. White lines separate label "
             "constructs (intent | judge | belief).", ha="left", fontsize=8.5, color=INK_2)
    sub = "" if bank in (None, "paper16") else f"banks/{bank}/"
    return save(fig, f"figure6_style/{sub}L{layer}_{pooling}_{probe}_k{k}")


def dataset_boot_mean_offdiag(D, B=10000, seed=42):
    """Dataset bootstrap of the mean off-diagonal value of a 10x10 matrix (pairs weighted by
    how often each dataset is drawn; a dataset paired with itself is excluded)."""
    n = D.shape[0]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=B)
    W = (counts[:, :, None] * counts[:, None, :]).astype(float)
    W[:, np.arange(n), np.arange(n)] = 0
    mask = ~np.isnan(D) & ~np.eye(n, dtype=bool)
    Dz = np.where(mask, D, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (W * Dz[None]).sum(axis=(1, 2)) / (W * mask[None]).sum(axis=(1, 2))
    b = b[~np.isnan(b)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def bank_comparison(pooling: str, layer: int, probe: str, banks, seed: int = 42):
    """One row per persona bank; each panel = one persona feature's mean off-diagonal change vs
    raw, with a dataset-bootstrap 95% interval. Also writes the numbers as CSV."""
    feats = [("pc:unified@2", "unified-axis PCs (k=2)"), ("pc:unified", "unified-axis PCs (k=5)"),
             ("pc:deception", "deception-axis PCs (k=5)"), ("axes", "persona axes (4-d)")]
    rows = []
    for b in banks:
        doc = load_matrix_doc(pooling, layer, seed, b)
        if doc is None:
            continue
        raw = mat(doc, "raw", probe)
        n = raw.shape[0]
        off = ~np.eye(n, dtype=bool)
        rec = {"bank": b}
        for f, _ in feats:
            M = mat(doc, f, probe)
            if M is None:
                continue
            D = M - raw
            lo, hi = dataset_boot_mean_offdiag(D)
            rec[f] = (float(np.nanmean(D[off])), lo, hi, int(np.nansum(D[off] > 0)))
        z = next((zz for zz in doc["zeroshot"] if zz["direction"] == "contrast"), None)
        rec["zeroshot"] = float(np.nanmean([np.nan if v is None else v for v in z["auroc"]])) if z else np.nan
        rows.append(rec)
    if not rows:
        return None
    ypos = np.arange(len(rows))[::-1]
    fig, axes = plt.subplots(1, len(feats), figsize=(4.1 * len(feats), 0.62 * len(rows) + 1.9),
                             sharey=True, gridspec_kw={"wspace": 0.08})
    all_vals = [v for r in rows for f, _ in feats if f in r for v in r[f][1:3]]
    lim = max(0.15, np.ceil(max(abs(v) for v in all_vals) * 20) / 20)
    for ax, (f, title) in zip(axes, feats):
        ax.axvline(0, color=RULE, lw=1)
        ax.axvline(0.10, color=INK_2, lw=0.8, ls=(0, (3, 3)))
        for y, r in zip(ypos, rows):
            if f not in r:
                continue
            m, lo, hi, wins = r[f]
            ax.plot([lo, hi], [y, y], color="#2a78d6", lw=2, solid_capstyle="round")
            ax.plot(m, y, "o", color="#2a78d6", ms=7, mec="white", mew=1.5)
            ax.text(lim * 0.98, y + 0.22, f"{m:+.3f} · {wins}/90", ha="right", va="bottom",
                    fontsize=7.5, color=INK_2)
        ax.set_xlim(-lim, lim)
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", color=INK)
        ax.set_xlabel("mean change in AUROC vs raw\n(dataset-bootstrap 95% interval)")
        ax.grid(axis="x", color="#e6e5e1", lw=0.6)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels([BANK_LABEL.get(r["bank"], r["bank"]) for r in rows])
    axes[-1].text(0.10, ypos.max() + 0.75, "paper, 3B: ≈ +0.10", fontsize=7.5, color=INK_2, ha="center")
    fig.suptitle(f"Persona banks compared · layer {layer} · {pooling} pooling · probe: "
                 f"{PROBE_LABEL[probe]} · {N_DATASETS} datasets, {N_DATASETS * (N_DATASETS - 1)} train→test pairs",
                 x=0.02, ha="left", y=1.02, fontsize=11.5, fontweight="bold", color=INK)
    path = save(fig, f"bank_comparison/L{layer}_{pooling}_{probe}")
    lines = ["bank,feature,mean_delta_vs_raw,dataset_ci_lo,dataset_ci_hi,pairs_better,zeroshot_contrast_mean_auroc"]
    for r in rows:
        for f, _ in feats:
            if f in r:
                m, lo, hi, w = r[f]
                lines.append(f"{r['bank']},{f},{m:.4f},{lo:.4f},{hi:.4f},{w},{r['zeroshot']:.4f}")
    guard_out(path.with_suffix(".csv")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


FEATURE_COLOR = {"raw": "#2a78d6", "axes": "#eb6834", "pc:deception": "#1baf7a",
                 "pc:unified": "#4a3aa7", "pc:unified@2": "#e34948",
                 "datapca@2": "#8f8d87", "datapca": "#8f8d87",
                 "random@2": "#b9b7b0", "random": "#b9b7b0", "length": "#eda100"}
FEATURE_LINE = {"datapca@2": (0, (4, 2)), "datapca": (0, (4, 2)),
                "random@2": (0, (1, 2)), "random": (0, (1, 2)), "length": (0, (5, 1, 1, 1))}
FEATURE_NAME = {"raw": "raw (8192-d)", "axes": "persona axes (4-d)",
                "pc:deception": "deception PCs", "pc:unified": "unified PCs (k=5)",
                "pc:unified@2": "unified PCs (k=2)", "datapca@2": "dataset PCA (k=2)",
                "datapca": "dataset PCA (k=5)", "random@2": "random (k=2)",
                "random": "random (k=5)", "length": "response length"}


def bank_dir(bank):
    return RUN if bank in (None, "paper16") else RUN / "banks" / bank


def load_summary(bank=None):
    p = bank_dir(bank) / "summary.json"
    root = RUN / "summary.json"
    if not p.exists():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    if p != root and root.exists():           # baselines live in the root summary
        rd = json.loads(root.read_text(encoding="utf-8"))
        have = {(r["pooling"], r["layer"], r["seed"], r["feature"], r["probe"]) for r in doc["records"]}
        doc["records"] += [r for r in rd["records"] if r["feature"] in BASELINE_FEATURES
                           and (r["pooling"], r["layer"], r["seed"], r["feature"], r["probe"]) not in have]
    return doc


def load_significance(bank=None):
    p = bank_dir(bank) / "significance.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _series(recs, pooling, probe, feature, key, seed=42):
    pts = sorted((r["layer"], r[key]) for r in recs
                 if r["pooling"] == pooling and r["probe"] == probe
                 and r["feature"] == feature and r["seed"] == seed and r.get(key) is not None)
    return [p[0] for p in pts], [p[1] for p in pts]


def fig_curves(pooling: str, probe: str, bank=None, block: str = "10"):
    """Off-diagonal and diagonal transfer against layer, with a band over the 5 extra splits."""
    doc = load_summary(bank)
    if doc is None:
        return None
    recs = doc["records"]
    key = f"offdiag_mean_{block}"
    feats = [f for f in ("raw", "axes", "pc:deception", "pc:unified@2", "datapca@2", "random@2", "length")
             if any(r["feature"] == f for r in recs)]
    if not feats:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharex=True)
    for f in feats:
        c = FEATURE_COLOR.get(f, "#666")
        ls = FEATURE_LINE.get(f, "-")
        for ax, k in zip(axes, [key, "diag_mean"]):
            xs, ys = _series(recs, pooling, probe, f, k)
            if not xs:
                continue
            ax.plot(xs, ys, color=c, ls=ls, lw=1.8, label=FEATURE_NAME.get(f, f))
            by_layer = defaultdict(list)
            for r in recs:
                if (r["pooling"], r["probe"], r["feature"]) == (pooling, probe, f) and r["seed"] != 42:
                    by_layer[r["layer"]].append(r[k])
            band = sorted((L, min(v), max(v)) for L, v in by_layer.items() if len(v) > 1)
            if band:
                ax.fill_between([b[0] for b in band], [b[1] for b in band], [b[2] for b in band],
                                color=c, alpha=0.14, lw=0)
    for ax, title in zip(axes, [f"cross-dataset transfer ({N_DATASETS if block == '10' else 9} datasets)",
                                "own dataset (20% test split)"]):
        ax.axhline(0.5, color=RULE, lw=0.9)
        ax.set_xlabel("layer")
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
        ax.grid(color="#e9e8e4", lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("mean AUROC")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(f"{BANK_LABEL.get(bank or 'paper16', bank)} · {pooling} pooling · {PROBE_LABEL[probe]}"
                 " · shaded band = range over 5 further 80/20 splits",
                 x=0.02, ha="left", y=1.02, fontsize=11, fontweight="bold", color=INK)
    sub = "" if bank in (None, "paper16") else f"{bank}_"
    return save(fig, f"curves/{sub}{pooling}_{probe}_block{block}")


def fig_matched(layer: int, pooling: str, probe: str, bank=None):
    """Each persona feature against raw AND against a matched control at the same k."""
    sig = load_significance(bank)
    if sig is None:
        return None
    rows = [t for t in sig["tests"] if t["block"] == "10" and t["layer"] == layer
            and t["pooling"] == pooling and t["probe"] == probe
            and t["comparison"].split(" - ")[0] in ("axes", "pc:deception", "pc:unified", "pc:unified@2")]
    if not rows:
        return None
    order = ["raw", "datapca", "datapca@2", "random", "random@2"]
    feats = sorted({t["comparison"].split(" - ")[0] for t in rows})
    fig, axes = plt.subplots(1, len(feats), figsize=(3.9 * len(feats), 3.2), sharex=True)
    axes = np.atleast_1d(axes)
    lim = max(0.12, max(abs(v) for t in rows for v in t["dataset_ci95"]))
    for ax, f in zip(axes, feats):
        mine = [t for t in rows if t["comparison"].startswith(f + " - ")]
        mine.sort(key=lambda t: order.index(t["comparison"].split(" - ")[1])
                  if t["comparison"].split(" - ")[1] in order else 9)
        ys = np.arange(len(mine))[::-1]
        for y, t in zip(ys, mine):
            ctrl = t["comparison"].split(" - ")[1]
            col = "#2a78d6" if ctrl == "raw" else "#8f8d87"
            ax.plot(t["dataset_ci95"], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(t["diff"], y, "o", ms=7, color=col, mec="white", mew=1.4)
            ax.text(lim * 0.97, y + 0.18, f"{t['diff']:+.3f} · {t['pairs_won']}/{t['pairs']}",
                    ha="right", va="bottom", fontsize=7.5, color=INK_2)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"vs {FEATURE_NAME.get(t['comparison'].split(' - ')[1], '')}" for t in mine])
        ax.axvline(0, color=RULE, lw=1)
        ax.set_xlim(-lim, lim)
        ax.set_title(FEATURE_NAME.get(f, f), loc="left", fontweight="bold", color=INK)
        ax.set_xlabel("change in mean AUROC")
        ax.grid(axis="x", color="#e9e8e4", lw=0.6)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"Matched controls · layer {layer} · {pooling} pooling · {PROBE_LABEL[probe]} · "
                 f"{BANK_LABEL.get(bank or 'paper16', bank)} · bars = dataset-bootstrap 95%",
                 x=0.02, ha="left", y=1.04, fontsize=11, fontweight="bold", color=INK)
    sub = "" if bank in (None, "paper16") else f"{bank}_"
    return save(fig, f"matched/{sub}L{layer}_{pooling}_{probe}")


def fig_blocks(layer: int, pooling: str, probe: str, bank=None):
    """Block structure: transfer by construct pair, and the sign-flip counts."""
    doc = load_summary(bank)
    if doc is None:
        return None
    recs = [r for r in doc["records"] if r["layer"] == layer and r["pooling"] == pooling
            and r["probe"] == probe and r["seed"] == 42 and "blocks" in r]
    feats = [f for f in ("raw", "axes", "pc:deception", "pc:unified@2", "datapca@2", "random@2")
             if any(r["feature"] == f for r in recs)]
    if not feats:
        return None
    blocks = [b for b in recs[0]["blocks"] if "->" in b] + ["sycophancy as source", "sycophancy as target"]
    M = np.full((len(blocks), len(feats)), np.nan)
    for j, f in enumerate(feats):
        r = next((r for r in recs if r["feature"] == f), None)
        for i, b in enumerate(blocks):
            if r and b in r["blocks"]:
                M[i, j] = r["blocks"][b]["mean_auroc"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.42 * len(blocks) + 3),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    im = axes[0].imshow(M, cmap=DIVERGING, norm=TwoSlopeNorm(vmin=0.2, vcenter=0.5, vmax=0.9))
    for i in range(len(blocks)):
        for j in range(len(feats)):
            if not np.isnan(M[i, j]):
                axes[0].text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                             color=text_color(DIVERGING(TwoSlopeNorm(vmin=0.2, vcenter=0.5, vmax=0.9)(M[i, j]))))
    n_cells = [recs[0]["blocks"][b]["n_cells"] if b in recs[0]["blocks"] else 0 for b in blocks]
    axes[0].set_yticks(range(len(blocks)))
    axes[0].set_yticklabels([f"{b}  ({n})" for b, n in zip(blocks, n_cells)])
    axes[0].set_xticks(range(len(feats)))
    axes[0].set_xticklabels([FEATURE_NAME.get(f, f) for f in feats], rotation=30, ha="right")
    axes[0].set_title("mean AUROC by construct block", loc="left", fontweight="bold", color=INK)
    axes[0].grid(False)
    for s in axes[0].spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02, label="AUROC")

    keys = ["below_chance", "significantly_inverted", "rescued_vs_raw", "broken_vs_raw"]
    w = 0.2
    for i, k in enumerate(keys):
        vals = [next((r["sign_flips"].get(k, 0) for r in recs if r["feature"] == f), 0) for f in feats]
        axes[1].bar(np.arange(len(feats)) + (i - 1.5) * w, vals, width=w,
                    color=["#b8322f", "#e34948", "#2a78d6", "#8f8d87"][i], label=k.replace("_", " "))
    axes[1].set_xticks(range(len(feats)))
    axes[1].set_xticklabels([FEATURE_NAME.get(f, f) for f in feats], rotation=30, ha="right")
    axes[1].set_ylabel("cells (of 90)")
    axes[1].set_title("inverted cells: a reversed signal is not a missing one", loc="left",
                      fontweight="bold", color=INK)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", color="#e9e8e4", lw=0.6)
    for s in ("top", "right"):
        axes[1].spines[s].set_visible(False)
    fig.suptitle(f"Layer {layer} · {pooling} pooling · {PROBE_LABEL[probe]} · "
                 f"{BANK_LABEL.get(bank or 'paper16', bank)}",
                 x=0.02, ha="left", y=1.03, fontsize=11, fontweight="bold", color=INK)
    sub = "" if bank in (None, "paper16") else f"{bank}_"
    return save(fig, f"blocks/{sub}L{layer}_{pooling}_{probe}")


def fig_per_dataset(layer_key: str, pooling: str, probe: str, bank=None):
    """Small multiples: how each dataset does as a training source / as a test target, by layer."""
    doc = load_summary(bank)
    if doc is None:
        return None
    recs = [r for r in doc["records"] if r["pooling"] == pooling and r["probe"] == probe
            and r["seed"] == 42 and layer_key in r]
    if not recs:
        return None
    names = sorted(recs[0][layer_key])
    feats = [f for f in ("raw", "axes", "pc:unified@2", "random@2") if any(r["feature"] == f for r in recs)]
    ncol = 5
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.3 * nrow), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for ax, ds in zip(axes.ravel(), names):
        for f in feats:
            pts = sorted((r["layer"], r[layer_key][ds]) for r in recs if r["feature"] == f
                         and r[layer_key].get(ds) is not None)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], lw=1.6,
                        color=FEATURE_COLOR.get(f, "#666"), ls=FEATURE_LINE.get(f, "-"),
                        label=FEATURE_NAME.get(f, f))
        ax.axhline(0.5, color=RULE, lw=0.8)
        ax.set_title(SHORT.get(ds, ds), loc="left", fontsize=9.5, fontweight="bold", color=INK)
        ax.grid(color="#ecebe7", lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    what = "trained on this dataset" if layer_key == "per_source_offdiag" else "tested on this dataset"
    fig.suptitle(f"Mean transfer AUROC, {what} · {pooling} pooling · {PROBE_LABEL[probe]} · "
                 f"{BANK_LABEL.get(bank or 'paper16', bank)}",
                 x=0.02, ha="left", y=1.0, fontsize=11, fontweight="bold", color=INK)
    tag = "per_source" if layer_key == "per_source_offdiag" else "per_target"
    sub = "" if bank in (None, "paper16") else f"{bank}_"
    return save(fig, f"{tag}/{sub}{pooling}_{probe}")


FIGURES = {"fig6": None, "banks": None, "curves": None, "matched": None, "blocks": None,
           "per_dataset": None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=["fig6", "banks"], choices=sorted(FIGURES))
    ap.add_argument("--layers", nargs="+", type=int, default=[40])
    ap.add_argument("--poolings", nargs="+", default=["mean", "last"])
    ap.add_argument("--probes", nargs="+", default=["paper", "lr_reg10", "mms"])
    ap.add_argument("--ks", nargs="+", type=int, default=[2, 5])
    ap.add_argument("--banks", nargs="+",
                    default=["paper16", "paper16_famcentred", "paper16_shared", "own100", "hf100", "hf275"])
    ap.add_argument("--run", default=None,
                    help="results root (default run-crossgen). Use run-crossgen/kd11 for the "
                         "11-dataset grid; its figures must not overwrite the 10-dataset ones.")
    ap.add_argument("--figdir", default=None,
                    help="figure root (default figures/crossgen)")
    args = ap.parse_args()

    # Rebind the module globals rather than threading a root through twenty call sites. The
    # dataset count is read from the JSON that was actually loaded, because every title used
    # to assert "10 datasets" regardless -- which on an 11-dataset grid would be a caption
    # that silently contradicts its own figure.
    global RUN, FIGDIR, N_DATASETS
    if args.run:
        RUN = (REPO_ROOT / args.run).resolve()
        if not RUN.is_dir():
            raise SystemExit(f"--run {RUN} does not exist")
    if args.figdir:
        FIGDIR = (REPO_ROOT / args.figdir).resolve()
    elif args.run:
        FIGDIR = REPO_ROOT / "figures" / "crossgen" / RUN.name
    probe_json = next(iter(sorted((RUN / "matrices").glob("*.json"))), None)
    if probe_json is not None:
        N_DATASETS = len(json.loads(probe_json.read_text(encoding="utf-8"))["datasets"])
    logger.info("run=%s  figdir=%s  datasets=%d", RUN, FIGDIR, N_DATASETS)
    setup()
    made = 0
    if "fig6" in args.only:
        for bank in args.banks:
            for L in args.layers:
                for pool in args.poolings:
                    for probe in args.probes:
                        for k in args.ks:
                            p = fig6(pool, L, probe, k, bank=bank)
                            if p:
                                made += 1
                                logger.info("  %s", p.relative_to(REPO_ROOT))
    if "banks" in args.only:
        for L in args.layers:
            for pool in args.poolings:
                for probe in args.probes:
                    p = bank_comparison(pool, L, probe, args.banks)
                    if p:
                        made += 1
                        logger.info("  %s", p.relative_to(REPO_ROOT))
    for name, fn in (("curves", lambda b, L, pool, probe: fig_curves(pool, probe, b)),
                     ("matched", lambda b, L, pool, probe: fig_matched(L, pool, probe, b)),
                     ("blocks", lambda b, L, pool, probe: fig_blocks(L, pool, probe, b))):
        if name not in args.only:
            continue
        for bank in args.banks:
            for pool in args.poolings:
                for probe in args.probes:
                    seen = set()
                    for L in (args.layers if name != "curves" else args.layers[:1]):
                        if (name, bank, pool, probe, L if name != "curves" else 0) in seen:
                            continue
                        seen.add((name, bank, pool, probe, L if name != "curves" else 0))
                        p = fn(bank, L, pool, probe)
                        if p:
                            made += 1
                            logger.info("  %s", p.relative_to(REPO_ROOT))
    if "per_dataset" in args.only:
        for bank in args.banks:
            for pool in args.poolings:
                for probe in args.probes:
                    for key in ("per_source_offdiag", "per_target_offdiag"):
                        p = fig_per_dataset(key, pool, probe, bank)
                        if p:
                            made += 1
                            logger.info("  %s", p.relative_to(REPO_ROOT))
    index = FIGDIR / "INDEX.md"
    if made:
        lines = ["# figures/crossgen — index\n",
                 "Every figure is drawn from the JSONs in `run-crossgen/`; none reads the activation store.\n",
                 "| folder | what it shows |", "|---|---|",
                 "| `figure6_style/` | the paper's Figure 6 layout: raw-probe AUROC, then random subspace, dataset PCA and unified-axis PCs as changes from raw. `banks/<name>/` for each persona bank. |",
                 "| `bank_comparison/` | every persona bank's mean change vs raw, per feature, with dataset-bootstrap intervals (CSV beside each) |",
                 "| `curves/` | transfer and own-dataset AUROC against layer, shaded band = range over the 5 extra 80/20 splits |",
                 "| `matched/` | each persona feature against raw AND against a matched control at the same k (dataset PCA, random subspace) |",
                 "| `blocks/` | mean AUROC per construct block (intent/judge/belief, sycophancy row and column) and the inverted-cell counts |",
                 "| `per_source/`, `per_target/` | small multiples: each dataset as a training source, and as a test target, against layer |",
                 "", f"Files: {len(list(FIGDIR.rglob('*.png')))} PNG (each with a PDF twin).", ""]
        guard_out(index).write_text("\n".join(lines), encoding="utf-8")
    logger.info("%d figures written under %s", made, FIGDIR.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
