#!/usr/bin/env python3
"""F-03's cell, read across every layer and pooling the store holds.

F-03 is `axes` (k=5) trained ON the text-matched HP-KR slice and tested everywhere else --
the ROW of the matrix, not the column -- at L40, mean pooling, `paper` probe: 0.752 against
raw 0.715, random 0.597, length 0.614. Its stated caveats are 258 rows, ONE LAYER, one seed,
and a per-row bootstrap only.

This removes the one-layer caveat. passA stores all 80 layers and passC stores per-token, so
the same cell can now be read down the depth axis and across pooling. F-34 measured a factor
probe moving 0.784 -> 0.509 between L20 and L40, so a single-layer headline carries an
unquantified researcher degree of freedom: this says how big it is.

    python kd/scripts/matched_profile.py run-crossgen/kd11/passA11/banks/kd_s23
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

DS = "harm-pressure-kr-matched"


def row_mean(entry, datasets, i):
    """Trained on DS, tested on every other dataset: row i, diagonal skipped."""
    A = entry["auroc"]
    v = [A[i][j] for j in range(len(datasets)) if j != i and A[i][j] is not None]
    return statistics.fmean(v) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="+", help="a bank or root dir holding matrices/")
    ap.add_argument("--probe", default="paper")
    ap.add_argument("--features", nargs="+",
                    default=["axes", "raw", "random", "length", "datapca"])
    args = ap.parse_args()

    for root in args.root:
        mdir = Path(root) / "matrices"
        files = sorted(mdir.glob("*.json"))
        if not files:
            print(f"{root}: no matrices\n"); continue
        print(f"=== {root}  ({len(files)} matrices, probe={args.probe}) ===")
        rows = []
        for f in files:
            d = json.load(open(f, encoding="utf-8"))
            if DS not in d["datasets"]:
                continue
            i = d["datasets"].index(DS)
            vals = {}
            for r in d["results"]:
                if r["probe"] == args.probe and r["feature"] in args.features:
                    vals[r["feature"]] = row_mean(r, d["datasets"], i)
            if vals:
                rows.append((d["pooling"], d["layer"], vals))
        if not rows:
            print(f"  {DS} not present in any matrix here\n"); continue

        for pooling in sorted({p for p, _, _ in rows}):
            sel = sorted([(l, v) for p, l, v in rows if p == pooling])
            print(f"\n  pooling={pooling}   ({len(sel)} layers)")
            head = "  " + f"{'layer':>6}" + "".join(f"{f:>10}" for f in args.features)
            print(head)
            for lay, v in sel:
                line = f"  {lay:>6}" + "".join(
                    (f"{v[f]:>10.3f}" if f in v else f"{'--':>10}") for f in args.features)
                print(line)
            if "axes" in args.features:
                prof = [(l, v["axes"]) for l, v in sel if "axes" in v]
                if prof:
                    best = max(prof, key=lambda t: t[1])
                    worst = min(prof, key=lambda t: t[1])
                    span = best[1] - worst[1]
                    print(f"    axes: peak L{best[0]} {best[1]:.3f} · "
                          f"floor L{worst[0]} {worst[1]:.3f} · "
                          f"SPAN {span:.3f}"
                          + ("   <- the single-layer degree of freedom" if span > 0.05 else ""))
                    # Only meaningful when raw is in THIS matrix. The baselines live in the
                    # ROOT matrices and the persona features in banks/<name>, so a bank-only
                    # read has no raw to compare against and must say so rather than printing
                    # "beats raw at 0 of N", which reads like a result and is an absence.
                    have_raw = [l for l, v in sel if "raw" in v]
                    if have_raw:
                        beat = [l for l, a in prof if a > dict(sel)[l]["raw"]]
                        print(f"    beats raw at {len(beat)} of {len(prof)} layers")
                    else:
                        print("    (raw not in this matrix -- read the ROOT dir for baselines)")
        print()


if __name__ == "__main__":
    main()
