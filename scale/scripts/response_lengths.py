#!/usr/bin/env python3
"""Response-length distribution for a scale rung.

Why this exists: on 2026-09-20 the 3B vector extraction failed on every role with an instant
NDIF dispatch error, batches reporting "widest 4856 padded tokens". NDIF cannot run a sequence
past roughly 4,000 tokens at ANY batch size -- batch 1 is the floor and the allocation happens
inside layer 0's attention -- so a single over-long rollout poisons its whole batch.

The question this answers is whether that is a few pathological rows (drop them) or the 3B
routinely running to max_new_tokens (a property of the rung, and a length confound, since
response length transfers on its own in this project at 0.553 off-diagonal).

    python scale/scripts/response_lengths.py run-scale/3b/responses run-scale/8b/responses
"""

from __future__ import annotations

import glob
import json
import os
import sys


def stats(root: str) -> None:
    lens, over = [], []
    for f in sorted(glob.glob(os.path.join(root, "*.jsonl"))):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            conv = d.get("conversation") or []
            txt = next((m.get("content", "") for m in reversed(conv)
                        if m.get("role") == "assistant"), "")
            n = len(txt)
            lens.append(n)
            if n > 12000:                      # ~3k tokens of English
                over.append((os.path.basename(f), n))
    if not lens:
        print(f"{root}: no rows")
        return
    lens.sort()
    def pct(p): return lens[min(len(lens) - 1, int(len(lens) * p))]
    print(f"{root}: {len(lens)} rows  chars "
          f"p50={pct(.50):,}  p90={pct(.90):,}  p99={pct(.99):,}  max={lens[-1]:,}")
    print(f"   rows over 12,000 chars (~3k tokens): {len(over)}  "
          f"({100 * len(over) / len(lens):.1f}%)")
    for name, n in sorted(over, key=lambda x: -x[1])[:5]:
        print(f"     {name:32s} {n:,} chars")


if __name__ == "__main__":
    for root in (sys.argv[1:] or ["run-scale/3b/responses"]):
        stats(root)
