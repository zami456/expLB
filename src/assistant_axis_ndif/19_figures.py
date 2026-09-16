#!/usr/bin/env python
"""Every figure for the persona-coordinates replication, from the result JSONs.

Each panel reads a small JSON written by an earlier stage, so this runs on the laptop
and never touches the 43 GB activation store. Figures are written to
`figures/persona-coords/` as both PDF and PNG.

    python 19_figures.py              # all of them
    python 19_figures.py --only q5    # one

The figures deliberately draw the chance line on every AUROC panel. Several results in
this project sit below 0.5 -- transfer across a label construct is systematically
inverted -- and that is invisible on an axis that starts at 0.5.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

import plot_style as ps

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]
RUN = REPO_ROOT / "run-coords"


def load(name: str):
    p = RUN / name
    if not p.exists():
        logger.warning("  skipping: %s not found", name)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def entries(doc):
    return doc["p1"] if isinstance(doc, dict) and "p1" in doc else doc


# --------------------------------------------------------------------------- #
def fig_layers(plt):
    """Q2: where the persona-coordinate advantage lives, on the layer axis."""
    docs = [(load("layer_sweep_passB_mms.json"), "mean pooling"),
            (load("layer_sweep_passB_last.json"), "last-token pooling")]
    if not any(d for d, _ in docs):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), sharey=True)
    for ax, (doc, title) in zip(axes, docs):
        if doc is None:
            continue
        sweep = doc["sweep"]
        L = [r["layer"] for r in sweep]
        for feat in ("raw", "pc:deception", "axes"):
            if feat not in sweep[0]:
                continue
            y = [r[feat] for r in sweep]
            ax.plot(L, y, marker="o", color=ps.FEATURE_COLOR[feat],
                    label=ps.FEATURE_LABEL[feat])
            best = int(np.argmax(y))
            ax.plot([L[best]], [y[best]], marker="*", ms=11,
                    color=ps.FEATURE_COLOR[feat], zorder=5)
        ps.chance(ax)
        # the three layers the literature disagrees on
        for x, name in ((20, "LB"), (22, "Apollo"), (40, "AA")):
            ax.axvline(x, color="#C0C0C0", lw=0.7, ls=":", zorder=0)
            ax.annotate(name, xy=(x, 0.505), fontsize=6.5, color="#909090",
                        rotation=90, va="bottom", ha="right")
        ax.set_title(title)
        ax.set_xlabel("layer (of 80)")
    axes[0].set_ylabel("mean off-diagonal AUROC")
    axes[0].legend(loc="lower right")
    fig.suptitle("Q2 — the persona-coordinate advantage is a layer window, not a fact",
                 y=1.02, fontsize=10.5)
    return ps.save(fig, "q2_layer_sweep.pdf")


def fig_transfer(plt):
    """Q2/Q5: the transfer matrix itself, raw against persona axes."""
    doc = load("p1_L44_last_withsyc.json")
    if doc is None:
        return None
    want = [("raw", "mms"), ("axes", "mms")]
    picked = [e for e in entries(doc)
              if (e["features"], e["probe"]) in want]
    if not picked:
        return None
    keys = sorted({k.split("->")[0] for e in picked for k in e["matrix"]} |
                  {k.split("->")[1] for e in picked for k in e["matrix"]},
                  key=lambda k: (ps.SEMANTICS.get(k, "z"), k))
    fig, axes = plt.subplots(1, len(picked), figsize=(5.1 * len(picked), 4.6))
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, picked):
        M = np.full((len(keys), len(keys)), np.nan)
        for kk, v in e["matrix"].items():
            s, t = kk.split("->")
            if v is not None and v == v:
                M[keys.index(s), keys.index(t)] = v
        im = ax.imshow(M, cmap="RdBu_r", vmin=0.2, vmax=0.8)
        ax.set_xticks(range(len(keys)))
        ax.set_yticks(range(len(keys)))
        ax.set_xticklabels([ps.SHORT.get(k, k) for k in keys], rotation=90)
        ax.set_yticklabels([ps.SHORT.get(k, k) for k in keys])
        for i in range(len(keys)):
            for j in range(len(keys)):
                if M[i, j] == M[i, j]:
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(M[i, j] - 0.5) > 0.22 else "#303030")
        # bracket the label constructs down the source axis
        for i, k in enumerate(keys):
            ax.add_patch(plt.Rectangle((-1.05, i - 0.5), 0.35, 1.0, clip_on=False,
                                       color=ps.SEM_COLOR[ps.SEMANTICS[k]], lw=0))
        ax.set_title(f"{ps.FEATURE_LABEL[e['features']]}")
        ax.set_xlabel("tested on")
        ax.grid(False)
    axes[0].set_ylabel("trained on")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.75, label="AUROC")
    handles = [plt.Line2D([], [], marker="s", ls="", color=c, label=ps.SEM_SHORT[s])
               for s, c in ps.SEM_COLOR.items()]
    axes[-1].legend(handles=handles, title="label construct", loc="upper left",
                    bbox_to_anchor=(1.22, 1.0))
    fig.suptitle("Transfer matrix, layer 44, last-token pooling", y=0.99, fontsize=10.5)
    return ps.save(fig, "transfer_L44_last.pdf")


def fig_q5(plt):
    """Q5: the decisive comparison, across layers."""
    files = sorted(RUN.glob("q5_layers/p1_L*.json"),
                   key=lambda p: int(p.stem[4:]))
    if not files:
        return None
    rows = {}
    for f in files:
        L = int(f.stem[4:])
        for e in entries(json.loads(f.read_text(encoding="utf-8"))):
            m = {k: v for k, v in e["matrix"].items() if v is not None and v == v}
            ii = [v for k, v in m.items() for s, t in [k.split("->")]
                  if s != t and ps.SEMANTICS[s] == ps.SEMANTICS[t] == "instructed_intent"]
            bb = [m[k] for k in ("mask->sycophancy_mt", "sycophancy_mt->mask") if k in m]
            rows.setdefault(e["features"], []).append(
                (L, float(np.mean(ii)) if ii else np.nan,
                 float(np.mean(bb)) if bb else np.nan))

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
    for ax, feat in zip(axes, ("raw", "axes")):
        if feat not in rows:
            continue
        d = sorted(rows[feat])
        L = [r[0] for r in d]
        ax.plot(L, [r[1] for r in d], marker="o", color="#4C72B0",
                label="within instructed_intent\n7 datasets, 42 pairs")
        ax.plot(L, [r[2] for r in d], marker="s", color="#55A868",
                label="within belief_inconsistency\nmask and sycophancy, 2 pairs")
        ps.chance(ax)
        ax.set_title(ps.FEATURE_LABEL[feat])
        ax.set_xlabel("layer")
    axes[0].set_ylabel("mean off-diagonal AUROC")
    axes[0].set_ylim(0.15, 0.9)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.19))
    fig.suptitle("Q5 — sharing a label definition does not predict transfer",
                 y=1.03, fontsize=10.5)
    return ps.save(fig, "q5_semantics_layers.pdf")


def fig_q8(plt):
    """Q8: per-token aggregation, and where in a response the signal sits."""
    doc = load("pertoken_q8.json")
    if doc is None:
        return None
    aggs = ["mean", "last8", "last", "first", "max", "top10pct"]
    rows = doc["aggregations"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    ax = axes[0]
    width = 0.13
    L = [r["layer"] for r in rows]
    for i, a in enumerate(aggs):
        ax.bar(np.arange(len(L)) + (i - 2.5) * width, [r[a] for r in rows],
               width, label=a)
    ax.set_xticks(range(len(L)))
    ax.set_xticklabels(L)
    ax.set_xlabel("layer")
    ax.set_ylabel("AUROC (zero-shot, no training)")
    ax.set_ylim(0.45, 0.85)
    ps.chance(ax)
    ax.legend(ncol=3, loc="upper left", fontsize=7)
    ax.set_title("aggregation of per-token scores")

    best = max(rows, key=lambda r: max(r[a] for a in aggs))
    loc = doc["localisation"][str(best["layer"])]
    ax = axes[1]
    for lab, name, c in ((1, "deceptive", "#C44E52"), (0, "honest", "#4C72B0")):
        d = np.array(loc[str(lab)])
        y = np.where(d[:, 1] > 0, d[:, 0] / np.maximum(d[:, 1], 1), np.nan)
        ax.plot(np.arange(10) * 10 + 5, y, marker="o", color=c, label=name)
    ax.set_xlabel("position within the response (decile, %)")
    ax.set_ylabel("mean projection on the deception axis")
    ax.set_title(f"where the signal sits (layer {best['layer']})")
    ax.legend()
    fig.suptitle("Q8, zero-shot — scoring a fixed axis per token beats pooling first (trained: see q8_trained)",
                 y=1.03, fontsize=10.5)
    return ps.save(fig, "q8_pertoken.pdf")


def fig_q9(plt):
    """Q9: what more personas actually buy."""
    doc = load("persona_count_L40.json")
    if doc is None:
        return None
    c = doc["curve"]
    n = [r["n"] for r in c]
    m = np.array([r["auroc_mean"] for r in c])
    sd = np.array([r["auroc_sd"] for r in c])
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
    axes[0].errorbar(n, m, yerr=sd, marker="o", color="#55A868", capsize=3)
    axes[0].set_xlabel("personas per side")
    axes[0].set_ylabel("AUROC (zero-shot)")
    axes[0].set_title("accuracy: flat after n=4")
    ps.chance(axes[0])
    axes[1].plot(n, sd, marker="o", color="#DD8452")
    axes[1].set_xlabel("personas per side")
    axes[1].set_ylabel("sd across persona draws")
    axes[1].set_title("reliability: this is what improves")
    fig.suptitle("Q9 — more personas buy reliability, not accuracy",
                 y=1.03, fontsize=10.5)
    return ps.save(fig, "q9_persona_count.pdf")


def fig_validate(plt):
    """The pipeline check: what actually dominates these activations."""
    doc = load("validate_L40.json")
    if doc is None:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    var = np.array(doc["variance"]) * 100
    axes[0].bar(np.arange(1, len(var) + 1), var, color="#4C72B0")
    axes[0].set_xlabel("principal component")
    axes[0].set_ylabel("variance explained (%)")
    axes[0].set_title(f"PCA, layer {doc['layer']}")

    ax = axes[1]
    pcs = doc["pcs"]
    x = np.arange(len(pcs))
    ax.bar(x - 0.2, [max(p["auroc_label"], 1 - p["auroc_label"]) for p in pcs], 0.4,
           label="the label", color="#DD8452")
    ax.bar(x + 0.2, [p["auroc_dataset"] for p in pcs], 0.4,
           label="which dataset", color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{p['pc']}" for p in pcs])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.02)
    ps.chance(ax)
    ax.legend(loc="lower left")
    ax.set_title("what the leading PCs track")

    ax = axes[2]
    cos = doc["cos_deception_pcs"]
    ax.bar(np.arange(1, len(cos) + 1), cos, color="#55A868")
    ax.set_xlabel("principal component")
    ax.set_ylabel("|cos| with the deception axis")
    ax.set_title(f"axis sits off the main PCs\n({100*doc['deception_in_top10']:.1f}% "
                 "of it inside the top-10 subspace)")
    fig.suptitle("Pipeline check — dataset identity, not the label, dominates the variance",
                 y=1.04, fontsize=10.5)
    return ps.save(fig, "validate_pca.pdf")


def fig_p2(plt):
    """Q3: sample efficiency."""
    doc = load("p2_L44_last.json")
    if doc is None or "p2" not in doc:
        return None
    by = {}
    for r in doc["p2"]:
        by.setdefault((r["features"], r["probe"]), []).append(r)
    probes = sorted({p for _, p in by})
    fig, axes = plt.subplots(1, len(probes), figsize=(4.4 * len(probes), 3.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, probe in zip(axes, probes):
        for (feat, pr), rows in sorted(by.items()):
            if pr != probe:
                continue
            rows.sort(key=lambda r: r["n"])
            n = [r["n"] for r in rows]
            m = np.array([r["auroc_mean"] for r in rows])
            sd = np.array([r["auroc_sd"] for r in rows]) / np.sqrt(
                [max(r.get("n_obs", 1), 1) for r in rows])
            ax.plot(n, m, marker="o", color=ps.FEATURE_COLOR.get(feat, "#666"),
                    label=ps.FEATURE_LABEL.get(feat, feat))
            ax.fill_between(n, m - sd, m + sd, alpha=0.15,
                            color=ps.FEATURE_COLOR.get(feat, "#666"))
        ax.set_xscale("log", base=2)
        ax.set_xlabel("training rows")
        ax.set_title(probe)
        ps.chance(ax)
    axes[0].set_ylabel("AUROC on the full pool of every other dataset")
    axes[0].legend(loc="lower right")
    fig.suptitle(f"Q3 — sample efficiency, layer {doc['layer']} {doc['pooling']} pooling",
                 y=1.03, fontsize=10.5)
    return ps.save(fig, "q3_sample_efficiency.pdf")


def fig_contamination(plt):
    """Q1: how much of each deception axis is the Assistant Axis."""
    # measured in the plan's Q1 table; kept here so the figure is reproducible from one file
    banks = {
        "own100": {20: -0.729, 22: -0.710, 40: -0.660, 60: -0.573},
        "hf275": {20: -0.686, 22: -0.662, 40: -0.610, 60: -0.516},
        "paper16 (all)": {20: -0.415, 22: -0.387, 40: -0.401, 60: -0.414},
        "paper16 (score-3)": {20: -0.364, 22: -0.338, 40: -0.370, 60: -0.374},
    }
    doc = load("regroup_hf.json")
    if doc:
        banks["hf275 regrouped\nby disclosure"] = {
            r["layer"]: r["disclosure"] for r in doc["results"]}
    layers = [20, 22, 40, 60]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    width = 0.8 / len(banks)
    for i, (name, d) in enumerate(banks.items()):
        ax.bar(np.arange(len(layers)) + (i - (len(banks) - 1) / 2) * width,
               [abs(d.get(L, np.nan)) for L in layers], width, label=name)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"layer {L}" for L in layers])
    ax.set_ylabel("|cos(deception axis, assistant axis)|")
    ax.legend(ncol=2, fontsize=7.5)
    ax.set_title("Q1 — the persona set halves the Assistant-Axis contamination;\n"
                 "regrouping the role library does not", fontsize=10)
    return ps.save(fig, "q1_contamination.pdf")


def fig_q11(plt):
    """Q11: each shortcut test, side by side, at the two layers analysed."""
    docs = [(L, load(f"q11_sleeper_L{L}_last.json")) for L in (20, 44)]
    docs = [(L, d) for L, d in docs if d]
    if not docs:
        return None
    trig = ["soft-trigger-gender", "soft-trigger-greeting", "soft-trigger-time"]
    fig, axes = plt.subplots(1, len(docs), figsize=(4.6 * len(docs), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (L, doc) in zip(axes, docs):
        t = doc["tests"]
        labels, raw, axs = [], [], []

        def mean_of(d, feat, cond=lambda k: True):
            v = [x for k, x in d.items() if k.startswith(feat + "|") and cond(k)]
            return float(np.mean(v)) if v else np.nan

        labels.append("within\norganism")
        raw.append(mean_of(t["within_organism"], "raw", lambda k: "gender-secret" not in k))
        axs.append(mean_of(t["within_organism"], "axes", lambda k: "gender-secret" not in k))
        labels.append("cross-\ntrigger")
        raw.append(mean_of(t["cross_trigger"], "raw"))
        axs.append(mean_of(t["cross_trigger"], "axes"))
        labels.append("same answer\ntoken")
        ts = t.get("token_stratified", {})
        raw.append(float(np.mean([v["probe_from_instructed-deception"] for k, v in ts.items()
                                  if k.startswith("raw|")])) if ts else np.nan)
        axs.append(float(np.mean([v["probe_from_instructed-deception"] for k, v in ts.items()
                                  if k.startswith("axes|")])) if ts else np.nan)
        labels.append("from\ncorpus")
        for feat, dest in (("raw", raw), ("axes", axs)):
            v = [x for k, row in t["corpus_to_sleeper"].items() if k.startswith(feat + "|")
                 for tt, x in row.items() if tt in trig]
            dest.append(float(np.mean(v)) if v else np.nan)
        labels.append("gender-\nsecret")
        for feat, dest in (("raw", raw), ("axes", axs)):
            v = [row.get("gender-secret") for k, row in t["corpus_to_sleeper"].items()
                 if k.startswith(feat + "|")]
            dest.append(float(np.nanmean([x for x in v if x is not None])) if v else np.nan)

        x = np.arange(len(labels))
        ax.bar(x - 0.2, raw, 0.4, color=ps.FEATURE_COLOR["raw"], label=ps.FEATURE_LABEL["raw"])
        ax.bar(x + 0.2, axs, 0.4, color=ps.FEATURE_COLOR["axes"], label=ps.FEATURE_LABEL["axes"])
        if "token_only|soft-trigger-gender" in ts:
            tok = np.mean([v for k, v in ts.items() if k.startswith("token_only|")])
            ax.axhline(tok, color="#C44E52", lw=0.9, ls=":")
            ax.annotate("answer token alone", xy=(0.01, tok), xycoords=("axes fraction", "data"),
                        fontsize=6.5, color="#C44E52", va="bottom")
        ps.chance(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0.3, 1.0)
        ax.set_title(f"layer {L}")
    axes[0].set_ylabel("AUROC")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Q11 — base-model probes flag sleeper answers as false, across triggers;\n"
                 "not where falsity is uncheckable (gender-secret)", y=1.05, fontsize=10)
    return ps.save(fig, "q11_sleeper.pdf")


def fig_q6(plt):
    """Q6: the three literature layers side by side, plus each method's own best."""
    doc = load("q6_layers.json")
    if doc is None:
        return None
    want = [
        ("trained transfer, full corpus", "raw / mms / last", "raw probe\n(last)"),
        ("trained transfer, full corpus", "raw / mms / mean", "raw probe\n(mean)"),
        ("leave-one-dataset-out, Pass C", "tok_lr / top10pct", "per-token\ntrained probe"),
        ("trained transfer, full corpus", "axes / mms / last", "persona axes\n(last)"),
        ("trained transfer, full corpus", "axes / mms / mean", "persona axes\n(mean)"),
        ("zero-shot, full corpus", "deception contrast axis / mean", "deception axis\nzero-shot"),
    ]
    rows = {(r["source"], r["method"]): r for r in doc["rows"]}
    picked = [(rows[(s, m)], lab) for s, m, lab in want if (s, m) in rows]
    if not picked:
        return None
    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    x = np.arange(len(picked))
    cols = [("20", "layer 20 (Liars' Bench)", "#8FB3D9"),
            ("22", "layer 22 (Apollo)", "#4C72B0"),
            ("40", "layer 40 (Assistant Axis)", "#DD8452")]
    w = 0.2
    for i, (key, label, color) in enumerate(cols):
        vals = [r["at"].get(key) for r, _ in picked]
        ax.bar(x + (i - 1.5) * w, [v if v is not None else np.nan for v in vals], w,
               color=color, label=label)
    ax.bar(x + 1.5 * w, [r["best"] for r, _ in picked], w, color="#55A868",
           label="method's own best layer")
    for xi, (r, _) in zip(x, picked):
        ax.annotate(f"L{r['best_layer']}", xy=(xi + 1.5 * w, r["best"] + 0.005),
                    ha="center", va="bottom", fontsize=6.5, color="#2F6B3A")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in picked], fontsize=7.5)
    ax.set_ylim(0.45, 0.95)
    ax.set_ylabel("AUROC")
    ps.chance(ax)
    ax.legend(ncol=4, fontsize=7, loc="upper left")
    fig.suptitle("Q6 — 20 vs 22 is immaterial; moving to 40 matters far more for persona "
                 "methods than for raw probes", y=1.02, fontsize=10.5)
    return ps.save(fig, "q6_layers.pdf")


def fig_q8_trained(plt):
    """Q8 finished: trained per-token probes vs trained pooled probes, same rows and folds."""
    doc = load("q8_trained_pertoken.json")
    if doc is None:
        return None
    by = {}
    for r in doc["results"]:
        by.setdefault((r["probe"], r["aggregation"]), {})[int(r["layer"])] = r
    layers = doc["layers"]
    lines = [
        ("tok_lr", "top10pct", "token LR, top-10% tokens", "#C44E52", "-", "o"),
        ("tok_lr", "mean", "token LR, mean of tokens", "#C44E52", "--", "o"),
        ("pool_mean_mms", "pooled", "pooled mean, mms", "#4C72B0", "-", "s"),
        ("pool_last_mms", "pooled", "pooled last, mms", "#4C72B0", "--", "s"),
        ("zeroshot_axis", "top10pct", "zero-shot axis, top-10%", "#55A868", "-", "^"),
        ("length_only", "n_tok", "response length only", "#9A9A9A", ":", None),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    for ax, metric, ylab in ((axes[0], "auroc_mean", "AUROC, mean over 9 held-out datasets"),
                             (axes[1], "tpr_mean", "TPR at 1% FPR (alpaca threshold)")):
        for probe, agg, label, color, ls, mk in lines:
            s = by.get((probe, agg))
            if not s:
                continue
            ax.plot(layers, [s[L][metric] for L in layers], color=color, ls=ls,
                    marker=mk, label=label)
        if metric == "auroc_mean":
            ps.chance(ax)
        ax.set_xlabel("layer")
        ax.set_ylabel(ylab)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.16))
    axes[0].set_title("detection on an unseen dataset")
    axes[1].set_title("at a deployable false-positive rate")
    fig.suptitle("Q8 — trained per-token probe vs trained pooled probe "
                 "(leave-one-dataset-out, Pass C)", y=1.03, fontsize=10.5)
    return ps.save(fig, "q8_trained_pertoken.pdf")


FIGURES = {
    "q8trained": fig_q8_trained,
    "q6": fig_q6,
    "q11": fig_q11,
    "q1": fig_contamination,
    "q2": fig_layers,
    "transfer": fig_transfer,
    "q5": fig_q5,
    "q8": fig_q8,
    "q9": fig_q9,
    "q3": fig_p2,
    "validate": fig_validate,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=sorted(FIGURES),
                    choices=sorted(FIGURES))
    args = ap.parse_args()

    ps.setup()
    import matplotlib.pyplot as plt

    made = []
    for name in args.only:
        try:
            p = FIGURES[name](plt)
        except Exception as exc:  # noqa: BLE001
            logger.error("  %s FAILED: %s: %s", name, type(exc).__name__, exc)
            continue
        if p is None:
            logger.warning("  %s: inputs missing, skipped", name)
        else:
            logger.info("  %-10s -> %s", name, p.relative_to(REPO_ROOT))
            made.append(p)
    logger.info("%d figures in %s", len(made), ps.FIGDIR.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
