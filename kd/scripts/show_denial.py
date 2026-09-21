#!/usr/bin/env python3
"""Print denial rollouts for a human read, with the denies/knows labels beside each response.

The gate on this route is a rate, but the rate alone does not say WHICH fallback to try. A
model that refuses the harmful framing needs the control template; one that confidently names
an answer needs different items; one that hedges needs a different regex. Those look identical
in a rate and obvious in the text.

    python kd/scripts/show_denial.py kd/run/denial/rollouts.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "kd/run/denial/rollouts.jsonl"
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 190
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    c = Counter(r["denies"] for r in rows)
    print(f"{path}: {len(rows)} rows  denies={c[True]}  asserts={c[False]}  "
          f"rate={c[True] / max(len(rows), 1):.3f}\n")
    for r in rows:
        tag = "DENIES " if r["denies"] else "asserts"
        print(f"[{tag}] {r['item_id']} {r.get('subset','')}")
        print(f"   {r['response'][:width].strip()}")
    print()


if __name__ == "__main__":
    main()
