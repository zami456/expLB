#!/usr/bin/env python3
"""Print two-turn rollouts for a human read: forced turn 1, then the generated turn 2.

The go/no-go on this bank is a human read, not a number (F-26, F-46). What has to be visible
is whether turn 2 abandons the forced answer, and whether the two persona groups differ in
that rather than in tone or length.

    python cap/scripts/show_rollouts.py cap/run/smoke_tf
"""

from __future__ import annotations

import glob
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else "cap/run/smoke_tf"
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 420
    for f in sorted(glob.glob(os.path.join(root, "*.jsonl"))):
        print("=" * 78)
        print(os.path.basename(f))
        print("=" * 78)
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            t2 = d["conversation"][-1]["content"].replace("\n", " ")
            flag = "" if d.get("turn1_is_correct", True) else "  [CONTROL: turn 1 is WRONG]"
            print(f"\n--- Q{d['question_index']} [{d.get('framing')}/{d.get('kind')}] "
                  f"{d['question']}{flag}")
            print(f"    forced turn 1: {d['turn1']!r}   (correct: {d['correct_answer']!r})")
            print(f"    turn 2: {t2[:width]}")
        print()


if __name__ == "__main__":
    main()
