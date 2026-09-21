#!/usr/bin/env python3
"""Re-measure the length confound against our own store, because a reviewer got a different answer.

OUR CLAIM (F-?? / the `length` control): the positive class is the longer one in 9 of 10 datasets,
by 505 vs 310 tokens on sandbagging_v2 and 245 vs 96 on mask, and a one-feature "longer is
deceptive" rule reaches an off-diagonal mean AUROC of 0.553 against raw activations' 0.665.

A REVIEWER CLONED THE RELEASED BENCHMARK AND GOT: positive class longer in 4 of 8, length-only
off-diagonal 0.493 (chance), and per-class gaps an order of magnitude smaller than ours (11 vs 10
words on knowledge-report, 123 vs 101 on insider-trading). They flagged three differences that
could explain it, and asked the right question: is the gap a property of the DATASETS or of OUR
GENERATION CONFIG?

The three differences, which this script controls for where it can:
  1. They counted WORDS in the last assistant message; we store `span_len`, the TOKEN count of
     the scored span. Tokens run ~1.3x words, which cannot explain an order of magnitude.
  2. The released rollouts pool FOUR generator models; ours are Llama-3.3-70B only. If our gap is
     much larger, that points at our generation config, not at the benchmark.
  3. Five of our eleven datasets are not in their repo at all.

If our gap survives on the datasets they could see, the claim holds and the difference is
generator/config. If it does not, the claim inverts and must be rewritten.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

# The eight they could measure, so the comparison is like-for-like where possible.
IN_THEIR_REPO = {"instructed-deception", "harm-pressure-choice",
                 "harm-pressure-knowledge-report", "convincing-game", "insider-trading",
                 "ai_liar", "roleplaying", "sandbagging_v2"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="run-coords/passB")
    args = ap.parse_args()

    root = Path(args.store)
    print(f"store: {root}   (span_len = TOKENS of the scored span, Llama-3.3-70B only)\n")
    print(f"  {'dataset':<34}{'n':>6}{'pos len':>9}{'neg len':>9}{'gap':>8}{'ratio':>7}  in-repo")
    longer, total = 0, 0
    rows_by = {}
    for d in sorted(root.iterdir()):
        rj = d / "rows.jsonl"
        if not rj.is_dir() and rj.exists():
            rows = [json.loads(l) for l in rj.open(encoding="utf-8") if l.strip()]
            by = defaultdict(list)
            for r in rows:
                lab = r.get("label")
                sl = r.get("span_len")
                if lab in (0, 1) and isinstance(sl, (int, float)):
                    by[lab].append(float(sl))
            if len(by) != 2 or not all(by.values()):
                continue
            pos = statistics.fmean(by[1])
            neg = statistics.fmean(by[0])
            rows_by[d.name] = (pos, neg, len(rows))
            total += 1
            if pos > neg:
                longer += 1
            mark = "yes" if d.name in IN_THEIR_REPO else "-"
            print(f"  {d.name:<34}{len(rows):>6}{pos:>9.1f}{neg:>9.1f}"
                  f"{pos - neg:>+8.1f}{pos / neg if neg else float('nan'):>7.2f}  {mark}")

    print(f"\n  positive class longer in {longer} of {total} datasets")
    sub = {k: v for k, v in rows_by.items() if k in IN_THEIR_REPO}
    sl = sum(1 for p, n, _ in sub.values() if p > n)
    print(f"  restricted to the 8 they could see: {sl} of {len(sub)}")
    if sub:
        gaps = [abs(p - n) for p, n, _ in sub.values()]
        print(f"  median |gap| on those 8: {statistics.median(gaps):.1f} tokens")
        print(f"  their reported gaps for comparison: ~1 word (knowledge-report), "
              f"~22 words (insider-trading)")


if __name__ == "__main__":
    main()
