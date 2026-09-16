#!/usr/bin/env python
"""Q6: which layer -- 20 (Liars' Bench), 22 (Apollo), or 40 (Assistant Axis / midpoint)?

Three literatures probe the same model at three different layers, and nobody has put them on
one corpus under one protocol. Everything needed is already measured; this script collects it
from the result JSONs so the Q6 table is regenerated from data, never retyped.

For every method it reports the score at 20, 22 and 40, the method's own best stored layer,
and two gaps that are the actual answer:

  20 vs 22   the Liars' Bench / Apollo disagreement
  {20,22} vs 40   the probe literature against the Assistant-Axis midpoint

and classifies each method:  AGREE  if all three are within `--tol`, otherwise DISAGREE.

Sources (each optional; a missing file is reported, not faked):
  layer_sweep_passB_{mms,last}.json   full corpus, 21 layers, trained transfer (P1 off-diag)
  layer_sweep_{mean,mms}.json         Pass A, 2,000 stratified rows, every 4th layer (no 22)
  dotproducts_mean.json               full corpus, zero-shot <activation, axis> AUROC
  pertoken_q8.json                    Pass C, zero-shot per-token axis, best aggregation
  q8_trained_pertoken.json            Pass C, trained per-token vs pooled probes (LODO)

    python 22_q6_layers.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parents[1]
TARGETS = (20, 22, 40)


def load(run: Path, name: str):
    p = run / name
    if not p.exists():
        logger.warning("  missing %s -- its rows are omitted", name)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def series_from_sweep(doc, feat):
    return {int(r["layer"]): float(r[feat]) for r in doc["sweep"] if feat in r}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=str(REPO_ROOT / "run-coords"))
    ap.add_argument("--tol", type=float, default=0.02,
                    help="AUROC gap under which layers count as agreeing")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    run = Path(args.run)

    methods = []   # (group, name, {layer: auroc}, note)

    for fname, pool in (("layer_sweep_passB_mms.json", "mean"),
                        ("layer_sweep_passB_last.json", "last")):
        d = load(run, fname)
        if d:
            for feat in ("raw", "pc:deception", "axes"):
                methods.append(("trained transfer, full corpus",
                                f"{feat} / {d['probe']} / {pool}",
                                series_from_sweep(d, feat), fname))
    for fname in ("layer_sweep_mean.json", "layer_sweep_mms.json"):
        d = load(run, fname)
        if d:
            for feat in ("raw", "pc:deception", "axes"):
                methods.append(("trained transfer, Pass A (no layer 22)",
                                f"{feat} / {d['probe']} / {d['pooling']}",
                                series_from_sweep(d, feat), fname))
    d = load(run, "dotproducts_mean.json")
    if d:
        for key, label in (("contrast", "deception contrast axis"), ("pc1", "PC1")):
            methods.append(("zero-shot, full corpus", f"{label} / mean",
                            {int(r["layer"]): float(r[key]) for r in d["matched"]},
                            "dotproducts_mean.json"))
    d = load(run, "pertoken_q8.json")
    if d:
        aggs = [a for a in d["aggregations"][0] if a != "layer"]
        for a in ("mean", "top10pct"):
            if a in aggs:
                methods.append(("zero-shot per-token, Pass C", f"axis / {a}",
                                {int(r["layer"]): float(r[a]) for r in d["aggregations"]},
                                "pertoken_q8.json"))
    d = load(run, "q8_trained_pertoken.json")
    if d:
        by = {}
        for r in d["results"]:
            by.setdefault((r["probe"], r["aggregation"]), {})[int(r["layer"])] = r["auroc_mean"]
        for (probe, agg) in (("tok_lr", "top10pct"), ("tok_lr", "mean"), ("tok_mms", "mean"),
                             ("pool_mean_mms", "pooled"), ("pool_mean_lr", "pooled"),
                             ("pool_last_mms", "pooled"), ("zeroshot_axis", "top10pct")):
            if (probe, agg) in by:
                methods.append(("leave-one-dataset-out, Pass C", f"{probe} / {agg}",
                                by[(probe, agg)], "q8_trained_pertoken.json"))

    rows = []
    logger.info("")
    logger.info("%-40s %-28s %7s %7s %7s   %-11s %8s %8s  %s",
                "source", "method", "L20", "L22", "L40", "best", "20-vs-22", "probe-v40",
                "verdict")
    last_group = None
    for group, name, s, src in methods:
        if not s:
            continue
        best_L = max(s, key=s.get)
        v = {L: s.get(L) for L in TARGETS}
        have = [x for x in v.values() if x is not None]
        g2022 = abs(v[20] - v[22]) if v[20] is not None and v[22] is not None else None
        lit = [x for x in (v[20], v[22]) if x is not None]
        g40 = (v[40] - max(lit)) if lit and v[40] is not None else None
        spread = max(have) - min(have) if len(have) >= 2 else None
        verdict = ("AGREE" if spread is not None and spread <= args.tol else
                   "DISAGREE" if spread is not None else "n/a")
        rows.append({"source": group, "method": name, "file": src,
                     "at": {str(L): v[L] for L in TARGETS},
                     "best_layer": best_L, "best": s[best_L],
                     "gap_20_22": g2022, "gap_40_minus_best_of_20_22": g40,
                     "spread_20_22_40": spread, "verdict": verdict})
        fmt = lambda x: f"{x:7.3f}" if x is not None else f"{'--':>7}"
        logger.info("%-40s %-28s %s %s %s   L%-2d %.3f %8s %8s  %s",
                    group if group != last_group else "", name, fmt(v[20]), fmt(v[22]),
                    fmt(v[40]), best_L, s[best_L],
                    f"{g2022:.3f}" if g2022 is not None else "--",
                    f"{g40:+.3f}" if g40 is not None else "--", verdict)
        last_group = group

    # ---- the answer ---------------------------------------------------------------
    logger.info("")
    logger.info("=" * 100)
    g = [r["gap_20_22"] for r in rows if r["gap_20_22"] is not None]
    if g:
        logger.info("20 vs 22 (Liars' Bench vs Apollo): max gap %.3f, median %.3f over %d methods",
                    max(g), float(np.median(g)), len(g))
    for fam, test in (("raw-activation probes", lambda r: r["method"].startswith(("raw", "tok_", "pool_"))),
                      ("persona-coordinate methods", lambda r: r["method"].startswith(("axes", "pc:", "deception contrast", "PC1", "axis", "zeroshot")))):
        sel = [r for r in rows if test(r) and r["gap_40_minus_best_of_20_22"] is not None]
        if sel:
            gaps = [r["gap_40_minus_best_of_20_22"] for r in sel]
            bests = sorted({r["best_layer"] for r in sel})
            logger.info("%-28s layer 40 minus best of {20,22}: median %+.3f (range %+.3f..%+.3f); "
                        "own best layers %s", fam, float(np.median(gaps)), min(gaps), max(gaps),
                        bests)

    path = Path(args.out) if args.out else run / "q6_layers.json"
    path.write_text(json.dumps({"targets": list(TARGETS), "tol": args.tol, "rows": rows},
                               indent=2), encoding="utf-8")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
