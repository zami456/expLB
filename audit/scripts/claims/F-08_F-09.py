#!/usr/bin/env python3
"""Recompute F-08 and F-09 from kd/run/knowledge/scores.jsonl.

The committed copy is truncated at 204,800 B (RF-5), so the default input is the
complete lab copy cached under audit/cache/labpc/. Pass a path to override.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

DEFAULT = Path("audit/cache/labpc/kd/run/knowledge/scores.jsonl")


def load(p):
    recs, bad = [], 0
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return recs, bad


def deciles(vals):
    """The 10 decile *group means*? No -- the ledger prints 10 numbers, so try both:
    (a) the 9 inner quantile cut points plus min/max, and (b) the mean of each decile bin."""
    s = sorted(vals)
    n = len(s)
    bins = []
    for d in range(10):
        lo, hi = int(round(d * n / 10)), int(round((d + 1) * n / 10))
        bins.append(s[lo:hi])
    return [statistics.fmean(b) for b in bins], [b[0] for b in bins], [b[-1] for b in bins]


def main():
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    recs, bad = load(p)
    print(f"file: {p}  size={p.stat().st_size} B")
    print(f"records parsed: {len(recs)}   unparseable lines: {bad}")
    print(f"unique item_ids: {len(set(r['item_id'] for r in recs))}")
    print()

    # ---- F-08 ----
    print("=== F-08 ===")
    correct = [r["correct"] for r in recs]
    acc = sum(correct) / len(correct)
    print(f"overall accuracy = {acc:.4f}  ({sum(correct)}/{len(correct)})")
    for sub in sorted(set(r["subset"] for r in recs)):
        s = [r["correct"] for r in recs if r["subset"] == sub]
        print(f"  {sub:<18} n={len(s):>5}  accuracy={sum(s)/len(s):.4f}")
    print("prediction spread (pred -> n):",
          dict(sorted(Counter(r["pred"] for r in recs).items())))
    print("gold spread       (gold -> n):",
          dict(sorted(Counter(r["gold"] for r in recs).items())))
    nb = sum(1 for r in recs if r.get("hpkr_belief") in (-1, -1.0))
    wrong = sum(1 for r in recs if r["correct"] == 0)
    print(f"HP-KR model_belief == -1 : {nb}")
    print(f"got it wrong             : {wrong}")
    print(f"ratio wrong/belief-1     : {wrong/nb:.3f}x" if nb else "")
    # disagreement between the two binary measures
    dis = sum(1 for r in recs
              if (r.get("hpkr_belief") in (-1, -1.0)) != (r["correct"] == 0))
    print(f"items where the two measures disagree: {dis}  ({dis/len(recs)*100:.1f}%)")
    both = sum(1 for r in recs if r.get("hpkr_belief") in (-1, -1.0) and r["correct"] == 0)
    print(f"  both say 'does not know': {both}")
    print()

    # ---- F-09 ----
    print("=== F-09 ===")
    lg = [r["logprob_gold"] for r in recs]
    means, lows, highs = deciles(lg)
    print("logprob_gold decile MEANS :", [round(x, 1) for x in means])
    print("logprob_gold decile MINs  :", [round(x, 1) for x in lows])
    print("logprob_gold decile MAXs  :", [round(x, 1) for x in highs])
    zero = sum(1 for v in lg if v == 0.0)
    print(f"records with logprob_gold == 0.0 exactly: {zero}  ({zero/len(lg)*100:.1f}%)")
    nz = sum(1 for v in lg if v > -1e-6)
    print(f"records with logprob_gold > -1e-6       : {nz}  ({nz/len(lg)*100:.1f}%)")
    m = [r["margin"] for r in recs]
    near1 = sum(1 for v in m if v >= 0.999)
    print(f"margin >= 0.999: {near1}  ({near1/len(m)*100:.1f}%)   median margin={statistics.median(m):.4f}")
    print(f"binary split: wrong={wrong}  right={len(recs)-wrong}")
    print(f"HP-KR split : belief==-1 {nb}  vs {len(recs)-nb}")


if __name__ == "__main__":
    main()
