"""Figure: the model-organism (Q11) shortcut tests -- Chapter 5, Section "Model organisms" (fig:r-sleeper).

A redraw of src/assistant_axis_ndif/19_figures.py :: fig_q11 without the embedded Q11 title, so the
thesis caption is the only title. It reads the same result files and hard-codes no numbers:

    run-coords/q11_sleeper_L20_last.json
    run-coords/q11_sleeper_L44_last.json

For each layer it plots the mean AUROC of the raw-activation probe and of the persona `axes`
feature under five tests: within organism (Soft-Trigger organisms only), cross-trigger, the
same-answer-token stratification, from the labelled corpus to the sleeper organisms, and from the
corpus to Gender-Secret. The dotted line is the answer token alone. Layer 44 is post hoc.

It prints every plotted value so a verifier can check them against the JSON.
Writes thesis/images/fig_sleeper.pdf and thesis_tools/figures/preview/fig_sleeper.png.

    python thesis_tools/figures/fig_sleeper.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "assistant_axis_ndif"))
import plot_style as ps  # noqa: E402  (the pipeline's colours and chance line)

RUN = ROOT / "run-coords"
OUT_PDF = ROOT / "thesis" / "images" / "fig_sleeper.pdf"
OUT_PNG = ROOT / "thesis_tools" / "figures" / "preview" / "fig_sleeper.png"
TRIG = ["soft-trigger-gender", "soft-trigger-greeting", "soft-trigger-time"]


def mean_of(d, feat, cond=lambda k: True):
    v = [x for k, x in d.items() if k.startswith(feat + "|") and cond(k)]
    return float(np.mean(v)) if v else np.nan


def main() -> None:
    docs = []
    for L in (20, 44):
        p = RUN / f"q11_sleeper_L{L}_last.json"
        if p.exists():
            docs.append((L, json.loads(p.read_text(encoding="utf-8")), p))
    if not docs:
        raise SystemExit("no q11_sleeper_L*_last.json under run-coords/")
    fig, axes = plt.subplots(1, len(docs), figsize=(4.6 * len(docs), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (L, doc, p) in zip(axes, docs):
        t = doc["tests"]
        labels, raw, axs = [], [], []
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
                 for tt, x in row.items() if tt in TRIG]
            dest.append(float(np.mean(v)) if v else np.nan)
        labels.append("gender-\nsecret")
        for feat, dest in (("raw", raw), ("axes", axs)):
            v = [row.get("gender-secret") for k, row in t["corpus_to_sleeper"].items()
                 if k.startswith(feat + "|")]
            dest.append(float(np.nanmean([x for x in v if x is not None])) if v else np.nan)
        x = np.arange(len(labels))
        ax.bar(x - 0.2, raw, 0.4, color=ps.FEATURE_COLOR["raw"], label=ps.FEATURE_LABEL["raw"])
        ax.bar(x + 0.2, axs, 0.4, color=ps.FEATURE_COLOR["axes"], label=ps.FEATURE_LABEL["axes"])
        tok = None
        if "token_only|soft-trigger-gender" in ts:
            tok = float(np.mean([v for k, v in ts.items() if k.startswith("token_only|")]))
            ax.axhline(tok, color="#C44E52", lw=0.9, ls=":")
            ax.annotate("answer token alone", xy=(0.01, tok), xycoords=("axes fraction", "data"),
                        fontsize=6.5, color="#C44E52", va="bottom")
        ps.chance(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(0.3, 1.0)
        ax.set_title(f"layer {L}" + (" (post hoc)" if L == 44 else ""))
        print(f"{p.relative_to(ROOT)}  layer {L}")
        for lab, r, a in zip([l.replace(chr(10), ' ') for l in labels], raw, axs):
            print(f"  {lab:<18} raw {r:.4f}  axes {a:.4f}")
        if tok is not None:
            print(f"  answer token alone {tok:.4f}")
    axes[0].set_ylabel("AUROC")
    axes[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print("wrote", OUT_PDF.relative_to(ROOT), "and", OUT_PNG.relative_to(ROOT))


if __name__ == "__main__":
    main()
