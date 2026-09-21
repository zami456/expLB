#!/usr/bin/env python3
"""Two of the three tests `docs/LAYER_PREREGISTRATION.md` owes, run from existing matrices.

The pre-registration committed to four things before the 80-layer profile existed. F-75 reported
the profile and called its verdict PROVISIONAL because these had not run.

  (2) cluster-based permutation test   -- needs label permutation and probe refits; NOT here
  (3) bootstrap the argmax layer       -- run here
  (4) nest layer selection inside the dataset-level bootstrap -- run here

(3) and (4) need no refitting: every layer's full matrix already exists, so resampling the
DATASETS and recomputing the profile from the stored cells is exact, not an approximation.

WHY (4) IS THE ONE THAT MATTERS. F-75's headline is that the margin over the matched control is
positive across a 16-layer plateau containing the pre-specified L40. The honest version of that
claim has to survive re-selecting the layer inside each bootstrap resample -- otherwise "there
exists a good layer" is being reported as "this layer is good", which is the selection effect the
whole document exists to guard against.

Reports three things:
  * the margin at the PRE-SPECIFIED layer (L40), bootstrapped over datasets -- no selection
  * the margin at the BEST layer chosen afresh inside each resample -- selection priced in
  * the bootstrap distribution of the argmax layer itself -- is the peak location stable?
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path

import numpy as np

DS = "harm-pressure-kr-matched"


def load(root: str, feats: set[str]) -> dict:
    """layer -> feature -> per-source-dataset vector of transfer AUROCs (row of the matrix)."""
    out: dict = {}
    for f in glob.glob(f"{root}/matrices/mean_L*_s42.json"):
        lay = int(re.search(r"mean_L(\d+)_s42", f).group(1))
        d = json.load(open(f, encoding="utf-8"))
        if DS not in d["datasets"]:
            continue
        i = d["datasets"].index(DS)
        n = len(d["datasets"])
        for r in d["results"]:
            if r["probe"] == "paper" and r["feature"] in feats:
                A = r["auroc"]
                # the ROW: trained on the matched slice, tested on each other dataset
                vec = [A[i][j] for j in range(n) if j != i]
                out.setdefault(lay, {})[r["feature"]] = np.array(
                    [np.nan if v is None else v for v in vec], dtype=float)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", default="run-crossgen/kd11/passA11/banks/kd_s23")
    ap.add_argument("--root", default="run-crossgen/kd11/passA11")
    ap.add_argument("--control", default="datapca")
    ap.add_argument("--prespec", type=int, default=40)
    ap.add_argument("-B", type=int, default=10000)
    args = ap.parse_args()

    bank = load(args.bank, {"axes"})
    root = load(args.root, {args.control, "raw"})
    layers = sorted(set(bank) & set(root))
    M = np.vstack([bank[L]["axes"] - root[L][args.control] for L in layers])  # (n_layers, n_src)
    ok = ~np.isnan(M).any(axis=0)
    M = M[:, ok]
    n_src = M.shape[1]
    print(f"{len(layers)} layers x {n_src} source datasets   control={args.control}")
    print(f"pre-specified layer = L{args.prespec}\n")

    li = layers.index(args.prespec)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n_src, size=(args.B, n_src))

    # (a) the PRE-SPECIFIED layer, no selection
    pre = M[li][idx].mean(axis=1)
    lo, hi = np.percentile(pre, [2.5, 97.5])
    print(f"  margin at L{args.prespec} (no selection)")
    print(f"    point {M[li].mean():+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
          f"P(<=0) {float((pre <= 0).mean()):.4f}"
          f"{'   CLEARS' if lo > 0 else '   does not clear'}")

    # (b) re-select the best layer INSIDE each resample -- selection priced in
    allb = M[:, idx].mean(axis=2)          # (n_layers, B)
    best = allb.max(axis=0)
    blo, bhi = np.percentile(best, [2.5, 97.5])
    print(f"\n  margin at the BEST layer, re-selected within each resample")
    print(f"    point {M.mean(axis=1).max():+.4f}   95% CI [{blo:+.4f}, {bhi:+.4f}]")
    print("    (this interval is inflated by selection BY CONSTRUCTION; it is the honest")
    print("     yardstick for a 'there exists a good layer' claim, not for L40)")

    # (c) where does the argmax land?
    arg = np.array(layers)[allb.argmax(axis=0)]
    alo, ahi = np.percentile(arg, [2.5, 97.5])
    print(f"\n  argmax layer across resamples")
    print(f"    median L{int(np.median(arg))}   95% interval [L{int(alo)}, L{int(ahi)}]   "
          f"width {int(ahi - alo)} layers of {len(layers)}")
    share = float((arg == args.prespec).mean())
    inband = float(((arg >= 27) & (arg <= 48)).mean())
    print(f"    argmax lands exactly on L{args.prespec} in {share:.1%} of resamples; "
          f"inside the L27-L48 plateau in {inband:.1%}")
    if ahi - alo > len(layers) * 0.5:
        print("    => the peak LOCATION is not identified; report the band, never the argmax")
    else:
        print("    => peak location is reasonably stable")

    # (d) how much of the network is positive at all, per resample
    frac = (allb > 0).mean(axis=0)
    print(f"\n  fraction of layers with a positive margin, per resample: "
          f"median {np.median(frac):.2f}  95% [{np.percentile(frac, 2.5):.2f}, "
          f"{np.percentile(frac, 97.5):.2f}]")


if __name__ == "__main__":
    main()
