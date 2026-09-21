#!/usr/bin/env python3
"""Per-layer profile of one comparison, across every layer a bank was scored at.

The point of extending kd_s23 from 3 layers to 21 (NEXT.md item 3) was to tell two different
claims apart: "the effect peaks at L40" and "the effect exists only at L40". Three points cannot
distinguish those, and every kd headline so far is read at L40 with L20/L44 as the only
neighbours. F-06 already showed layer choice is not cosmetic here -- knowledge and wording are
entangled at L40 (cos 0.410) and nearly orthogonal at L20 (0.063).

A profile that rises and falls smoothly means a real depth-dependent effect. One that spikes at
a single layer means a layer-specific artifact, and the headline would be a coincidence of
where we happened to look.

    python kd/scripts/layer_profile.py run-crossgen/kd11/banks/kd_s23/significance.json \
        --comparison "pc:unified - raw" --probe paper --pooling mean
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--comparison", default="pc:unified - raw")
    ap.add_argument("--probe", default="paper")
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--pairs", type=int, default=None, help="restrict to this pair count")
    args = ap.parse_args()

    tests = json.load(open(args.path, encoding="utf-8"))["tests"]
    sel = [t for t in tests
           if t["comparison"] == args.comparison and t["probe"] == args.probe
           and t["pooling"] == args.pooling
           and (args.pairs is None or t["pairs"] == args.pairs)]
    if not sel:
        print("no matching tests")
        return

    by_pairs = {}
    for t in sel:
        by_pairs.setdefault(t["pairs"], []).append(t)

    for pairs, rows in sorted(by_pairs.items()):
        rows.sort(key=lambda r: r["layer"])
        print(f"\n{args.comparison}   probe={args.probe}  pooling={args.pooling}  "
              f"pairs={pairs}   ({len(rows)} layers)")
        print(f"  {'layer':>5} {'diff':>8} {'won':>8}  {'dataset 95% CI':>20} {'q':>6}   profile")
        lo = min(r["diff"] for r in rows)
        hi = max(r["diff"] for r in rows)
        span = (hi - lo) or 1.0
        for r in rows:
            ci = r.get("dataset_ci95")
            cis = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "        n/a       "
            q = r.get("dataset_q_bh")
            qs = f"{q:.3f}" if q is not None else " n/a"
            bar = "#" * int(round(28 * (r["diff"] - lo) / span))
            star = " *" if r.get("significant_datasets") else ""
            print(f"  L{r['layer']:<4} {r['diff']:>+8.4f} {r['pairs_won']:>4}/{pairs:<3} "
                  f"{cis:>20} {qs:>6}   {bar}{star}")
        best = max(rows, key=lambda r: r["diff"])
        pos = sum(1 for r in rows if r["diff"] > 0)
        print(f"  peak L{best['layer']} at {best['diff']:+.4f}; "
              f"{pos}/{len(rows)} layers positive; range {lo:+.4f} .. {hi:+.4f}")


if __name__ == "__main__":
    main()
