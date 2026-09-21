#!/usr/bin/env python3
"""Like-for-like comparison against Mahadik & Skapars, Figure 9 panel A.

The paper reports MEAN AUROC DELTA VS RAW ON OFF-DIAGONAL PAIRS, at k=1, for three features:

    feature            3B Dec   8B Dec   8B Syc   3B Syc
    Random (k=1)        -0.14    -0.14    -0.11    -0.01
    Dataset PCA (k=1)   -0.12    -0.11    +0.01    -0.02
    Axis PC Proj (k=1)  +0.10    +0.12    +0.14    +0.17

So the paper DOES run the matched control, and its axis beats dataset PCA by +0.13 to +0.23.
This script computes the same quantity from our matrices so the two can be set side by side.

Off-diagonal means train != test. The diagonal is in-distribution and is not what either paper
is claiming.

    python kd/scripts/paper_comparison.py run-crossgen/kd11/matrices/mean_L40_s42.json
"""

from __future__ import annotations

import argparse
import json
import statistics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--probe", default="paper")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="restrict to these (e.g. the paper's 5 deception sets)")
    ap.add_argument("--baseline", default=None,
                    help="matrix holding raw/random/datapca. A BANK run stores only its own "
                         "persona features and points at --baseline_root for the controls, so "
                         "its matrices have no raw block and deltas cannot be computed without "
                         "this.")
    args = ap.parse_args()

    d = json.load(open(args.path, encoding="utf-8"))
    res = d["results"]
    # "rows"/"columns" are human captions ("trained on (80% split)"), NOT label lists.
    # Both axes are indexed by d["datasets"], in that order.
    rows = cols = d["datasets"]
    print(f"{args.path}\n  layer={d['layer']} pooling={d['pooling']} k={d.get('k')} "
          f"seed={d['seed']}  probe={args.probe}")
    print(f"  datasets ({len(d['datasets'])}): {', '.join(d['datasets'])}")

    keep = set(args.datasets) if args.datasets else None

    # `results` is a LIST of {feature, probe, auroc: [[...]]} records, not a nested dict.
    index = {(r["feature"], r["probe"]): r["auroc"] for r in res}

    def offdiag(feature: str):
        vals = []
        block = index.get((feature, args.probe))
        if block is None:
            return None
        for tr, row in zip(rows, block):
            for te, v in zip(cols, row):
                if tr == te or v is None:
                    continue
                if keep and (tr not in keep or te not in keep):
                    continue
                vals.append(v)
        return vals

    raw = offdiag("raw")
    if not raw and args.baseline:
        base_doc = json.load(open(args.baseline, encoding="utf-8"))
        base_index = {(r["feature"], r["probe"]): r["auroc"] for r in base_doc["results"]}
        index.update({k: v for k, v in base_index.items() if k not in index})
        raw = offdiag("raw")
        print(f"  controls from baseline: {args.baseline}")
    if not raw:
        print("  no raw block -- pass --baseline <baseline matrix>")
        return
    base = statistics.fmean(raw)
    print(f"\n  raw off-diagonal mean = {base:.4f}  over {len(raw)} pairs\n")
    print(f"  {'feature':<22} {'mean':>8} {'d vs raw':>10} {'pairs':>7}")
    feats = sorted({f for f, pr in index if f != "raw" and pr == args.probe})
    out = []
    for f in sorted(feats):
        v = offdiag(f)
        if not v:
            continue
        m = statistics.fmean(v)
        out.append((m - base, f, m, len(v)))
    for delta, f, m, n in sorted(out, reverse=True):
        print(f"  {f:<22} {m:>8.4f} {delta:>+10.4f} {n:>7}")


if __name__ == "__main__":
    main()
