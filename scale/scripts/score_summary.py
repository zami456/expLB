#!/usr/bin/env python3
"""Judge-score distribution for one scale rung, as a keep-rate against the 70B's.

The rungs exist to answer whether the persona-coordinate result is model-size dependent, and
the first thing that can invalidate that comparison is a rung whose rollouts are not in
character. The 70B scored 0-1 on 16 of 1,500 rollouts (1.1%). A small model that breaks
character far more often is not a weaker version of the same experiment -- it is a different
one, and the axis it produces is built from different material.

Reports 2+3 (the filter this project uses; score-3-only imposes a 1.31x sample imbalance
correlated with Assistant-Axis drift) and the per-score breakdown behind it.

    python scale/scripts/score_summary.py run-scale/8b/scores
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "run-scale/8b/scores"
    files = sorted(glob.glob(os.path.join(root, "*.json")))
    if not files:
        print(f"no score files under {root}")
        return

    counts = collections.Counter()
    per_role = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        role = os.path.basename(f)[:-5]
        rc = collections.Counter(d.values())
        per_role[role] = rc
        counts.update(rc)

    total = sum(counts.values())
    print(f"{root}: {len(files)} roles, {total} scored rollouts")
    for s in sorted(counts):
        print(f"  score {s}: {counts[s]:5d}  {100 * counts[s] / total:5.1f}%")
    keep = counts[2] + counts[3]
    print(f"  keep 2+3: {keep}/{total} = {100 * keep / total:.2f}%")
    print(f"  drop 0+1: {total - keep}/{total} = {100 * (total - keep) / total:.2f}%")
    print("  (70B reference: 16/1500 = 1.07% dropped)")

    worst = sorted(per_role.items(),
                   key=lambda kv: (kv[1][0] + kv[1][1]) / max(1, sum(kv[1].values())),
                   reverse=True)[:5]
    print("\n  worst roles by drop rate:")
    for role, rc in worst:
        n = sum(rc.values())
        print(f"    {role:28s} {rc[0] + rc[1]:3d}/{n:3d} dropped  ({100 * (rc[0] + rc[1]) / n:.1f}%)")

    # Per-GROUP keep rate, because an asymmetric one is a live confound rather than a curiosity.
    # At 70B the contrasted groups kept 116.8 vs 89.0 rows on average, and that imbalance
    # correlates with an Assistant-Axis drift gap of 1.82 between them -- so a difference means
    # the axis is partly built from "which side survived the filter", not from the construct.
    groups_path = sys.argv[2] if len(sys.argv) > 2 else "src/assistant_axis_ndif/pc_role_groups.json"
    if os.path.exists(groups_path):
        groups = json.load(open(groups_path, encoding="utf-8"))["groups"]
        print(f"\n  per-group keep rate ({os.path.basename(groups_path)}):")
        for gname, g in groups.items():
            rows = [per_role[r] for r in g.get("roles", []) if r in per_role]
            if not rows:
                continue
            n = sum(sum(rc.values()) for rc in rows)
            kept = sum(rc[2] + rc[3] for rc in rows)
            print(f"    {gname:20s} {len(rows):2d} roles  kept {kept:5d}/{n:5d} = {100 * kept / n:6.2f}%"
                  f"   mean kept/role {kept / len(rows):6.1f}")


if __name__ == "__main__":
    main()
