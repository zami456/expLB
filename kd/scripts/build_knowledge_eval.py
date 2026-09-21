#!/usr/bin/env python
"""Extract the WMDP items behind HP-KR, so we can measure what the model actually knows.

WHY NOT USE HP-KR's OWN `model_belief`. Two problems, both measured 2026-09-19:

  1. SIZE. On the Llama-3.3-70B rows only 121 items come out as "does not know", against
     1,841 that do. A difference-of-means direction with 121 on the thin side is the noisiest
     part of the whole attribution. Pooling harm-pressure-choice does not help: it contains
     ZERO not-know items by construction (its deceptive label needs a belief to contradict)
     and shares no items with HP-KR.

  2. MEANING. `model_belief` is -1 unless the model BOTH consistently claimed knowledge AND
     gave consistent answers across control templates. So -1 conflates "does not know" with
     "was unstable", and the negative class is a mixture of two different things.

Scoring the items ourselves fixes both: every item gets a graded, continuous measure of
whether the model knows it, taken under the same neutral prompt the probe uses.

This script only prepares the items (local, no network). score_knowledge_ndif.py does the
forward passes.

    python kd/scripts/build_knowledge_eval.py            # counts
    python kd/scripts/build_knowledge_eval.py --write    # -> kd/run/knowledge/items.jsonl
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

KD_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = KD_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "assistant_axis_ndif"))
import eval_datasets as E  # noqa: E402

OUT = KD_ROOT / "run" / "knowledge"
MARK = "Consider the following question:"
# HP-KR appends a "do you know?" instruction after option D, and it has FIVE template
# variants. In some the instruction sits in its own paragraph; in others it is glued onto
# option D's own sentence (". Do not tell me which option..."), so splitting on "\n\n" keeps
# 558 of 1,962 items and silently drops the rest. Cut instead at the earliest instruction
# marker occurring AFTER option D.
KNOW_Q = re.compile(
    r"(?:\.\s*|\n\s*)(?:IMPORTANT:\s*)?(?:"
    r"Do you know the answer\?"
    r"|Do not tell me which option"
    r"|I need to know if you are competent"
    r"|(?:Just|Only) (?:tell me|answer) whether you"
    r")", re.I)
LETTERS = "ABCD"


def extract() -> tuple[dict, Counter]:
    """question block (no trailing instruction) -> {'gold', 'subset', 'hpkr_belief'}."""
    path = E.SUBSETS / E.CSV_DATASETS["harm-pressure-knowledge-report"]
    items: dict[str, dict] = {}
    drops: Counter = Counter()
    with path.open(encoding="utf-8", newline="") as fh:
        for rec in csv.DictReader(fh):
            if rec.get("model") != E.CSV_MODEL:
                drops["other_model"] += 1
                continue
            try:
                msgs = ast.literal_eval(rec["messages"])
                meta = E.parse_meta(rec["meta"])
            except (ValueError, SyntaxError, NameError):
                drops["unparseable"] += 1
                continue
            user = msgs[-2]["content"]
            i = user.find(MARK)
            if i < 0:
                drops["no_marker"] += 1
                continue
            block = user[i + len(MARK):].strip()
            m = KNOW_Q.search(block)
            if not m:
                drops["no_know_instruction"] += 1
                continue
            question = block[:m.start()].strip()
            gold = meta.get("correct_option")
            if gold is None or not (0 <= int(gold) < 4):
                drops["no_gold"] += 1
                continue
            # every option must be present, or the letter probabilities are not comparable
            if not all(f"\n{L}." in question for L in LETTERS):
                drops["missing_options"] += 1
                continue
            if question in items:
                continue
            items[question] = {
                "gold": int(gold),
                "subset": rec.get("dataset", ""),
                # kept only so the new measure can be compared against the old one
                "hpkr_belief": meta.get("model_belief", -1),
            }
    return items, drops


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    items, drops = extract()
    gold = Counter(v["gold"] for v in items.values())
    subset = Counter(v["subset"] for v in items.values())
    old_knows = sum(1 for v in items.values() if v["hpkr_belief"] != -1)

    print(f"items                 {len(items):,}   (drops: {dict(drops)})")
    print(f"  by subset           {dict(subset)}")
    print(f"  gold option spread  {dict(sorted(gold.items()))}")
    print(f"  HP-KR says knows    {old_knows:,}")
    print(f"  HP-KR says NOT      {len(items) - old_knows:,}   <- the 121 problem")
    print()
    print("after scoring, 'knows' becomes 'answered correctly and confidently', which on a")
    print("70B at typical WMDP accuracy should put roughly a third of these in the negative")
    print("class -- several hundred rather than 121, and meaning one thing rather than two.")

    if not args.write:
        print("\ndry run; pass --write")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "items.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for i, (q, v) in enumerate(sorted(items.items())):
            fh.write(json.dumps({"item_id": f"kn{i:05d}", "question": q, **v},
                                ensure_ascii=False) + "\n")
    print(f"\nwrote {p}  ({len(items):,} items)")


if __name__ == "__main__":
    main()
