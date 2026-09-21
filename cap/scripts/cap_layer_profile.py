#!/usr/bin/env python3
"""The cap result across every stored layer, with the same dataset-level bootstrap F-76 used.

F-66 reported +0.112 over raw on `sycophancy_mt` at L40, measured at 3 of the 21 layers passB
stores. F-76 then showed that the analogous kd claim -- a 16-layer plateau containing a
pre-specified layer -- did NOT survive resampling the source datasets: the margin's 95% interval
included zero and the argmax landed on the pre-specified layer in 0.0% of resamples.

This asks the same question of the cap bank, which is the one the reviewers called contested-novel.

PREDICTION, recorded in run_cap_layers.ps1 before the run: the margin is positive across roughly
L22-L48 and falls off at both ends. If it is positive at only 3-5 layers, F-66 needs F-72's
treatment.

`sycophancy_mt` is read AS TARGET -- trained on each other dataset, tested on it -- which is the
column, matching how F-65/F-66 reported it.
"""
from __future__ import annotations

import argparse
import glob
import json
import re

import numpy as np

DS = "sycophancy_mt"


def load(root: str, feats: set[str], pooling: str = "mean") -> dict:
    """layer -> feature -> vector of per-SOURCE transfer AUROCs into DS (the column)."""
    out: dict = {}
    for f in glob.glob(f"{root}/matrices/{pooling}_L*_s42.json"):
        lay = int(re.search(rf"{pooling}_L(\d+)_s42", f).group(1))
        d = json.load(open(f, encoding="utf-8"))
        if DS not in d["datasets"]:
            continue
        j = d["datasets"].index(DS)
        n = len(d["datasets"])
        for r in d["results"]:
            if r["probe"] == "paper" and r["feature"] in feats:
                A = r["auroc"]
                col = [A[i][j] for i in range(n) if i != j]
                out.setdefault(lay, {})[r["feature"]] = np.array(
                    [np.nan if v is None else v for v in col], dtype=float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", default="run-crossgen/kd11/banks/cap_s23")
    ap.add_argument("--feature", default="pc:capitulation@2")
    ap.add_argument("--control", default="datapca@2")
    ap.add_argument("--prespec", type=int, default=40)
    ap.add_argument("-B", type=int, default=10000)
    args = ap.parse_args()

    # the persona feature and BOTH comparators live in the same bank matrices here
    cells = load(args.bank, {args.feature, args.control, "raw"})
    layers = sorted(cells)
    have = [L for L in layers
            if all(k in cells[L] for k in (args.feature, args.control, "raw"))]
    if not have:
        print("feature/control/raw not all present; check --bank and --feature")
        print("available at first layer:", sorted(cells[layers[0]]) if layers else None)
        return

    print(f"{len(have)} layers   feature={args.feature}   control={args.control}\n")
    print(f"{'layer':>6}{'feat':>8}{'raw':>8}{'ctrl':>8}{'f-raw':>8}{'f-ctrl':>8}")
    for L in have:
        f, r, c = (cells[L][k].mean() for k in (args.feature, "raw", args.control))
        star = "  <- pre-specified" if L == args.prespec else ""
        print(f"{L:>6}{f:>8.3f}{r:>8.3f}{c:>8.3f}{f - r:>+8.3f}{f - c:>+8.3f}{star}")

    M = np.vstack([cells[L][args.feature] - cells[L][args.control] for L in have])
    ok = ~np.isnan(M).any(axis=0)
    M = M[:, ok]
    n_src = M.shape[1]
    pos = int((M.mean(axis=1) > 0).sum())
    run = best = 0
    for v in M.mean(axis=1):
        run = run + 1 if v > 0 else 0
        best = max(best, run)
    print(f"\n  margin over {args.control} positive at {pos}/{len(have)} layers; "
          f"longest contiguous run {best}")

    rng = np.random.default_rng(42)
    idx = rng.integers(0, n_src, size=(args.B, n_src))
    li = have.index(args.prespec)
    pre = M[li][idx].mean(axis=1)
    lo, hi = np.percentile(pre, [2.5, 97.5])
    print(f"\n  L{args.prespec} margin  point {M[li].mean():+.4f}  "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  P(<=0) {float((pre <= 0).mean()):.4f}"
          f"{'   CLEARS' if lo > 0 else '   does not clear'}")

    allb = M[:, idx].mean(axis=2)
    arg = np.array(have)[allb.argmax(axis=0)]
    alo, ahi = np.percentile(arg, [2.5, 97.5])
    print(f"  argmax layer  median L{int(np.median(arg))}  "
          f"95% [L{int(alo)}, L{int(ahi)}]  "
          f"lands on L{args.prespec} in {float((arg == args.prespec).mean()):.1%} of resamples")


if __name__ == "__main__":
    main()
