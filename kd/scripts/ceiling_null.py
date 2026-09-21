#!/usr/bin/env python3
"""Is our corr(raw, X - raw) = -0.76 evidence, or arithmetic?

CLAIM 4 says a low-dimensional feature "wins" where the high-dimensional baseline is weak, and
cites corr(raw AUROC, delta) = -0.76 for a random 2-d projection as the mechanism.

A REVIEWER'S OBJECTION, which is correct and has to be answered: correlating a change with its
own baseline is mathematically coupled. corr(raw, X - raw) is negative BY CONSTRUCTION. If X is
independent of raw with spreads s_X and s_raw, the null value is

    rho_0 = -s_raw / sqrt(s_raw^2 + s_X^2)

and if X were pinned at 0.5 (a feature carrying nothing), it is exactly -1. This is the
regression-to-the-mean / mathematical-coupling warning from the clinical-statistics literature.

So -0.76 is only evidence if it is meaningfully MORE negative than rho_0 computed from our own
spreads. This script computes rho_0 three ways -- analytically, by permuting the pairing to break
any real association while preserving both marginals, and by simulating X as pure noise with our
observed spread -- and reports the measured value beside them.

It also runs the test the reviewer suggested INSTEAD, which does not contain raw in the
difference: regress the persona-minus-matched-control margin on raw. If the persona feature is
doing something beyond occupying a low-dimensional ceiling, that slope should be flatter than
the raw-minus-control one.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np


def cells(doc, feature, probe="paper"):
    """Every off-diagonal cell for one feature, in a fixed order."""
    ds = doc["datasets"]
    n = len(ds)
    for r in doc["results"]:
        if r["feature"] == feature and r["probe"] == probe:
            A = r["auroc"]
            return np.array([A[i][j] for i in range(n) for j in range(n)
                             if i != j and A[i][j] is not None], dtype=float)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matrix", default="run-crossgen/kd11/matrices/mean_L40_s42.json", nargs="?")
    ap.add_argument("--low", default="random@2", help="the low-dimensional feature under test")
    ap.add_argument("--persona", default="pc:unified@2")
    ap.add_argument("--control", default="datapca@2")
    ap.add_argument("-B", type=int, default=20000)
    args = ap.parse_args()

    doc = json.load(open(args.matrix, encoding="utf-8"))
    raw = cells(doc, "raw")
    low = cells(doc, args.low)
    if raw is None or low is None:
        print("missing raw or low feature"); return
    ok = ~(np.isnan(raw) | np.isnan(low))
    raw, low = raw[ok], low[ok]
    d = low - raw
    measured = float(np.corrcoef(raw, d)[0, 1])

    s_raw, s_low = raw.std(ddof=1), low.std(ddof=1)
    rho_analytic = -s_raw / math.sqrt(s_raw ** 2 + s_low ** 2)

    rng = np.random.default_rng(42)
    # (a) permute the pairing: destroys any real raw<->low association, keeps both marginals
    perm = []
    for _ in range(args.B):
        p = rng.permutation(low)
        perm.append(np.corrcoef(raw, p - raw)[0, 1])
    perm = np.array(perm)
    # (b) simulate the low feature as pure noise with our observed mean and spread
    sim = []
    for _ in range(args.B // 10):
        x = rng.normal(low.mean(), s_low, size=raw.size)
        sim.append(np.corrcoef(raw, x - raw)[0, 1])
    sim = np.array(sim)

    print(f"{args.matrix}\n  n cells = {raw.size}   low-dim feature = {args.low}")
    print(f"  sd(raw) = {s_raw:.4f}   sd({args.low}) = {s_low:.4f}\n")
    print(f"  MEASURED   corr(raw, {args.low} - raw) = {measured:+.4f}")
    print(f"  null (analytic, X independent of raw) = {rho_analytic:+.4f}")
    print(f"  null (permuted pairing)  mean {perm.mean():+.4f}  "
          f"95% [{np.percentile(perm, 2.5):+.4f}, {np.percentile(perm, 97.5):+.4f}]")
    print(f"  null (simulated noise)   mean {sim.mean():+.4f}  "
          f"95% [{np.percentile(sim, 2.5):+.4f}, {np.percentile(sim, 97.5):+.4f}]")
    p = float((perm <= measured).mean())
    print(f"\n  P(null <= measured) = {p:.4f}")
    if p > 0.05:
        print("  => NOT distinguishable from mathematical coupling. The -0.76 is arithmetic,")
        print("     not evidence, and Claim 4 must rest on a statistic without raw in it.")
    else:
        print("  => more negative than coupling alone predicts; the correlation carries signal,")
        print("     but report it beside this null, never bare.")

    # ---- the test that does NOT contain raw in the difference ---------------------------
    per = cells(doc, args.persona)
    ctl = cells(doc, args.control)
    if per is not None and ctl is not None:
        per, ctl = per[ok], ctl[ok]
        print(f"\n  --- margin over a MATCHED control, which has no raw in the difference ---")
        for name, y in ((f"{args.persona} - {args.control}", per - ctl),
                        (f"raw - {args.control}", raw - ctl)):
            good = ~np.isnan(y)
            slope = np.polyfit(raw[good], y[good], 1)[0]
            r = float(np.corrcoef(raw[good], y[good])[0, 1])
            print(f"    {name:<32} slope vs raw {slope:+.3f}   r {r:+.3f}   "
                  f"mean {statistics.fmean(y[good]):+.4f}")
        print("    A persona feature doing more than occupying a low-dimensional ceiling should")
        print("    show a FLATTER slope against raw than the control does.")


if __name__ == "__main__":
    main()
