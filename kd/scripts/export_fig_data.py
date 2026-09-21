#!/usr/bin/env python3
"""Export a compact JSON for the thesis figures, small enough to move by scp.

Two constraints make this necessary rather than optional:
  * matplotlib cannot run on the lab PC -- Application Control blocks its `_image` and
    `ft2font` DLLs (see plot_style.py), so figures are built on the laptop;
  * `scp` from the lab PC silently truncates anything over 204,800 bytes (F-14), and the
    21-layer significance.json is 1.38 MB.

So the numbers are reduced HERE and plotted THERE. Everything exported is a scalar per
(layer, comparison) -- a few KB.

    python kd/scripts/export_fig_data.py run-crossgen/kd11/banks/kd_s23/significance.json \
        --out kd/results/fig_data.json
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

WANT = ["pc:unified - raw", "pc:unified - datapca", "axes - raw", "axes - datapca",
        "random - raw", "datapca - raw"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--out", default="kd/results/fig_data.json")
    ap.add_argument("--probe", default="paper")
    ap.add_argument("--pooling", default="mean")
    ap.add_argument("--pairs", type=int, default=90)
    args = ap.parse_args()

    tests = json.load(open(args.path, encoding="utf-8"))["tests"]

    profile = collections.defaultdict(list)
    for t in tests:
        if (t["probe"] != args.probe or t["pooling"] != args.pooling
                or t["pairs"] != args.pairs or t["comparison"] not in WANT):
            continue
        ci = t.get("dataset_ci95") or [None, None]
        profile[t["comparison"]].append({
            "layer": t["layer"], "diff": t["diff"],
            "lo": ci[0], "hi": ci[1], "q": t.get("dataset_q_bh"),
            "won": t["pairs_won"], "pairs": t["pairs"],
            "sig": bool(t.get("significant_datasets")),
        })
    for k in profile:
        profile[k].sort(key=lambda r: r["layer"])

    # Significance census over EVERY test in the file, not just the plotted subset -- the
    # headline "N of M clear a dataset-level bootstrap" has to be over all of them.
    sig = [t for t in tests if t.get("significant_datasets")]
    census = {
        "n_tests": len(tests),
        "n_clearing": len(sig),
        "clearing": [{"layer": t["layer"], "pooling": t["pooling"], "probe": t["probe"],
                      "comparison": t["comparison"], "diff": t["diff"],
                      "q": t.get("dataset_q_bh")} for t in sig],
        "n_clearing_positive": sum(1 for t in sig if t["diff"] > 0),
        "q_min": min((t.get("dataset_q_bh") for t in tests
                      if t.get("dataset_q_bh") is not None), default=None),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"source": args.path, "probe": args.probe, "pooling": args.pooling,
           "pairs": args.pairs, "profile": profile, "census": census}
    out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    n = out.stat().st_size
    print(f"wrote {out}  ({n:,} bytes, {'OK for scp' if n < 204800 else 'TOO BIG for scp'})")
    print(f"  comparisons: {', '.join(sorted(profile))}")
    print(f"  layers per comparison: {len(next(iter(profile.values())))}")
    print(f"  census: {census['n_clearing']} of {census['n_tests']} clear, "
          f"{census['n_clearing_positive']} positive")


if __name__ == "__main__":
    main()
