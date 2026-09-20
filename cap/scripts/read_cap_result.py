#!/usr/bin/env python3
"""The number this bank exists to produce: does a capitulation axis beat 0.374 on sycophancy_mt?

The paper16 sycophancy axis has the cleanest geometry in the project -- cos(PC1, own contrast)
0.993, permutation p = 0.0079 over all 252 splits -- and reads **0.374 zero-shot / 0.355
trained-as-target** on `sycophancy_mt`. Below chance, worse than a random direction (0.541).
F-13's diagnosis: it encodes agreeable-vs-challenging REGISTER while the dataset labels
CAPITULATION.

This bank varies capitulation instead. Beating 0.374 demonstrates the diagnosis constructively;
failing to leaves it as a critique. Both are reportable.

Prints the sycophancy_mt COLUMN (as target: trained elsewhere, tested there) and ROW (as source)
for every feature, so the persona variants can be read against raw, random and datapca at
matched k rather than in isolation.

    python cap/scripts/read_cap_result.py run-crossgen/kd11/banks/cap_s23/matrices/mean_L40_s42.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

REGISTER_BASELINE = 0.374   # F-13, the number to beat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--probe", default="paper")
    ap.add_argument("--dataset", default="sycophancy_mt")
    ap.add_argument("--baseline", default=None,
                    help="matrix carrying raw/random/datapca if this one does not")
    args = ap.parse_args()

    d = json.load(open(args.path, encoding="utf-8"))
    ds = d["datasets"]
    if args.dataset not in ds:
        print(f"{args.dataset} not in this matrix: {ds}")
        return
    j = ds.index(args.dataset)

    index = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}
    if args.baseline:
        base = json.load(open(args.baseline, encoding="utf-8"))
        for r in base["results"]:
            index.setdefault((r["feature"], r["probe"]), r["auroc"])

    print(f"{args.path}")
    print(f"  layer={d['layer']} pooling={d['pooling']} k={d.get('k')} probe={args.probe}")
    print(f"  dataset={args.dataset}   register-axis baseline to beat: {REGISTER_BASELINE}\n")
    print(f"  {'feature':<24} {'as TARGET':>10} {'as SOURCE':>10}   vs 0.374")

    rows = []
    for (feat, pr), A in index.items():
        if pr != args.probe:
            continue
        # as TARGET: trained on every other dataset, tested on this one -> column j, skip diagonal
        tgt = [A[i][j] for i in range(len(ds)) if i != j and A[i][j] is not None]
        # as SOURCE: trained on this one, tested on every other -> row j, skip diagonal
        src = [A[j][i] for i in range(len(ds)) if i != j and A[j][i] is not None]
        if not tgt:
            continue
        rows.append((statistics.fmean(tgt), feat, statistics.fmean(src) if src else float("nan")))

    for t, feat, s in sorted(rows, reverse=True):
        mark = "  BEATS" if t > REGISTER_BASELINE else ""
        print(f"  {feat:<24} {t:>10.4f} {s:>10.4f}   {t - REGISTER_BASELINE:+.4f}{mark}")


if __name__ == "__main__":
    main()
