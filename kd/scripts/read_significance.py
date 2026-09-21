#!/usr/bin/env python3
"""Read out the dataset-level bootstrap for a crossgen bank (NEXT.md item 4).

The bar this answers: in the paper16 11-dataset run, 1 of 384 significance tests cleared a
dataset-level bootstrap (F-04). Any number quoted as a RESULT rather than an observation has to
be placed against that. The kd bank's headline -- unified axis +0.076 over raw at L40, 62/110
pairs -- has never been placed against it.

Row-level CIs are not the test. With ~15,000 rows they are tight by construction and say only
that the mean is well estimated; the quantity that actually varies between replications of this
experiment is WHICH DATASETS you happened to pick, and there are only 11 of those.

    python kd/scripts/read_significance.py run-crossgen/kd11/banks/kd/significance.json
    python kd/scripts/read_significance.py <path> --layer 40 --pooling mean
"""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--layer", type=int, default=40)
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--probe", default=None, help="restrict to one probe")
    args = ap.parse_args()

    d = json.load(open(args.path, encoding="utf-8"))
    tests = d["tests"]
    print(f"{args.path}: {len(tests)} tests, boot_datasets={d.get('boot_datasets')}")

    sig = [t for t in tests if t.get("significant_datasets")]
    print(f"cleared the DATASET-level bootstrap: {len(sig)} / {len(tests)}")
    for t in sig[:12]:
        print(f"   L{t['layer']:<3} {t['pooling']:<5} {t['probe']:<9} {t['comparison']:<22}"
              f" diff {t['diff']:+.4f}  q={t['dataset_q_bh']:.4f}")

    sel = [t for t in tests
           if t["layer"] == args.layer and t["pooling"] == args.pooling
           and (args.probe is None or t["probe"] == args.probe)]
    if not sel:
        print(f"\nno tests at layer {args.layer} pooling {args.pooling}")
        sys.exit(0)

    print(f"\nlayer {args.layer}, pooling {args.pooling} -- {len(sel)} tests, "
          f"largest positive diffs:")
    print(f"  {'probe':<9} {'comparison':<24} {'diff':>8} {'pairs':>8} "
          f"{'dataset 95% CI':>22} {'q_BH':>7}  sig")
    for t in sorted(sel, key=lambda x: -x["diff"])[:14]:
        ci = t.get("dataset_ci95")
        cis = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "        n/a       "
        q = t.get("dataset_q_bh")
        qs = f"{q:.3f}" if q is not None else "  n/a"
        print(f"  {t['probe']:<9} {t['comparison']:<24} {t['diff']:>+8.4f} "
              f"{t['pairs_won']:>4}/{t['pairs']:<3} {cis:>22} {qs:>7}  "
              f"{'YES' if t.get('significant_datasets') else 'no'}")


if __name__ == "__main__":
    main()
