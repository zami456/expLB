#!/usr/bin/env python
"""Dataset-level bootstrap on F-79's +0.178, the project's standing significance bar.

F-79 measured pooled leave-one-dataset-out transfer at 0.8545 against single-source transfer's
0.6763. The resampling unit is the TARGET dataset: each of the 11 contributes one paired
observation (its LODO score, its off-diagonal single-source mean), so resampling targets asks
"would this hold if I had happened to pick a different collection of evaluation datasets".

Paired throughout -- both conditions are scored on the same resampled target list, so the
difference carries no between-sample variance of its own. This is the test that 1 of 384 paper16
comparisons and 6 of 4,032 kd ones cleared, and that F-76 used to overturn F-75.

n = 11 is a small resampling unit and the interval will be wide. That is the honest width, not a
defect: it is F-04's standing point about this corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PAIRS = [("pooled_lodo", "single_offdiag"),
         ("pooled_lodo_match", "single_offdiag"),
         ("pooled_lodo", "pooled_lodo_match"),
         ("single_own", "pooled_lodo")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?",
                    default="run-premise/premise_L40_mean_s42.json")
    ap.add_argument("-B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = json.load(open(args.path, encoding="utf-8"))
    per = d["per_dataset"]
    names = sorted(per)
    n = len(names)
    print(f"{args.path}\n  layer={d['layer']} pooling={d['pooling']} probe={d['probe']}")
    print(f"  resampling unit = TARGET dataset, n={n}, B={args.B}, paired\n")

    print(f"  {'target':<34}{'single_off':>11}{'lodo':>9}{'lodo_m':>9}{'lodo-single':>13}")
    for k in names:
        r = per[k]
        print(f"  {k:<34}{r['single_offdiag']:>11.3f}{r['pooled_lodo']:>9.3f}"
              f"{r['pooled_lodo_match']:>9.3f}{r['pooled_lodo'] - r['single_offdiag']:>+13.3f}")

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.B, n))
    print(f"\n  {'comparison':<40}{'diff':>9}{'95% CI':>22}{'P(<=0)':>10}")
    for a, b in PAIRS:
        va = np.array([per[k][a] for k in names])
        vb = np.array([per[k][b] for k in names])
        dif = va - vb
        bs = dif[idx].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = float((bs <= 0).mean())
        mark = "  CLEARS" if lo > 0 else ("  (sig. negative)" if hi < 0 else "")
        print(f"  {a + ' - ' + b:<40}{dif.mean():>+9.4f}   [{lo:+.4f}, {hi:+.4f}]"
              f"{p:>10.4f}{mark}")

    wins = sum(1 for k in names if per[k]["pooled_lodo"] > per[k]["single_offdiag"])
    print(f"\n  pooled LODO beats single-source transfer on {wins} of {n} targets")
    worst = min(names, key=lambda k: per[k]["pooled_lodo"] - per[k]["single_offdiag"])
    print(f"  worst target for pooling: {worst} "
          f"({per[worst]['pooled_lodo'] - per[worst]['single_offdiag']:+.3f})")


if __name__ == "__main__":
    main()
