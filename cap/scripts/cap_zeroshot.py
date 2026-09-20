#!/usr/bin/env python3
"""Every zero-shot direction in a bank, with the row-level CI the matrix already stores.

WHY THIS EXISTS. The trained `pc:<name>` features project onto the named PCA *basis* and never
touch the contrast, so two variants built over the same role set are identical by construction
(cap's A and B are exactly that). The ZERO-SHOT rows are the only place a contrast direction
reaches a number, which makes them the only place the A-vs-B question -- does the position-holding
group contribute to the CONTRAST? -- can be answered at all.

Until 2026-09-21 `27_crossgen.py` scored one direction per bank, so B and C had no number. It now
scores every behaviour axis; the primary one keeps the bare names `contrast`/`PC1` so older
results stay bit-identical, the rest arrive as `contrast:<name>` / `PC1:<name>`.

The interval here is a ROW bootstrap, not the dataset-level one. Zero-shot trains on nothing, so
there are no source datasets to resample and the project's usual bar is undefined for it. Say
which interval it is wherever the number appears.

    python cap/scripts/cap_zeroshot.py run-crossgen/kd11/banks/cap_s23/matrices/mean_L40_s42.json
"""

from __future__ import annotations

import argparse
import json
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

REGISTER_BASELINE = 0.374     # F-13, the number to beat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--dataset", default="sycophancy_mt")
    args = ap.parse_args()

    for path in args.paths:
        try:
            d = json.load(open(path, encoding="utf-8"))
        except FileNotFoundError:
            print(f"{path}  MISSING\n")
            continue
        ds = d["datasets"]
        if args.dataset not in ds:
            print(f"{path}: {args.dataset} not in this matrix\n")
            continue
        j = ds.index(args.dataset)
        zs = d.get("zeroshot") or []
        print(f"{path}")
        print(f"  layer={d['layer']} pooling={d['pooling']}  dataset={args.dataset}"
              f"   register axis = {REGISTER_BASELINE}")
        if not zs:
            print("  no zero-shot rows\n")
            continue
        print(f"  {'direction':<26}{'AUROC':>8}{'95% CI (rows)':>22}{'vs 0.374':>11}")
        rows = []
        for z in zs:
            a = z["auroc"][j]
            lo = (z.get("ci95_lo") or [None] * len(ds))[j]
            hi = (z.get("ci95_hi") or [None] * len(ds))[j]
            rows.append((a, z["direction"], lo, hi))
        for a, name, lo, hi in sorted(rows, reverse=True):
            ci = f"[{lo:.3f}, {hi:.3f}]" if lo is not None else "--"
            mark = "  BEATS" if a > REGISTER_BASELINE else ""
            print(f"  {name:<26}{a:>8.4f}{ci:>22}{a - REGISTER_BASELINE:>+11.4f}{mark}")
        print()


if __name__ == "__main__":
    main()
