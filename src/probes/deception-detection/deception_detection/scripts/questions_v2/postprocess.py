"""raw_claude_output.md -> kd_questions_shared_v2.jsonl. Deterministic; no manual edits.

Rows 0-29 are the originals, copied byte for byte from the pipeline's question file. Rows 30-59
are Set B (the monotone control), rows 60-149 Set A (the diverse grid), each with its grid cell.
`id` equals the line position, which is what the pipeline's question_index means.

    python postprocess.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw_claude_output.md"
OUT = HERE / "kd_questions_shared_v2.jsonl"
ORIGINALS = Path(r"E:\aletheia\banks\Qwen3.5-27B\code\kd\questions\kd_questions_shared.jsonl")
ORIG_DOMAIN = {**{i: "orig_disclosure" for i in range(0, 10)}, **{i: "orig_sycophancy" for i in range(10, 20)},
               **{i: "orig_knowledge" for i in range(20, 30)}}
IMT_ORDER = ("quality", "quantity", "relation", "manner", "denial", "ignorance")


def blocks(text: str) -> list[list[dict]]:
    found = re.findall(r"```json\s*\n(.*?)\n```", text, flags=re.S)
    if len(found) != 2:
        raise SystemExit(f"expected two fenced json blocks, found {len(found)}")
    return [json.loads(b) for b in found]


def main() -> None:
    set_a, set_b = blocks(RAW.read_text(encoding="utf-8"))
    slots = {s["slot"]: s for s in json.loads((HERE / "grid_slots.json").read_text(encoding="utf-8"))}
    orig = [json.loads(l) for l in ORIGINALS.read_text(encoding="utf-8").splitlines() if l.strip()]
    if [r["id"] for r in orig] != list(range(30)):
        raise SystemExit("originals are not ids 0-29 in order")
    if sorted(a["slot"] for a in set_a) != list(range(90)):
        raise SystemExit("Set A must hold slots 0-89 exactly once")
    if sorted(b["mono"] for b in set_b) != list(range(30)):
        raise SystemExit("Set B must hold mono 0-29 exactly once")

    rows = []
    for r in orig:
        rows.append({"id": r["id"], "question": r["question"], "set": "orig", "domain": ORIG_DOMAIN[r["id"]],
                     "grid": None, "targeted": None, "slot": None})
    for b in sorted(set_b, key=lambda x: x["mono"]):
        imt = IMT_ORDER[b["mono"] % 6]
        if b["imt"] != imt:
            raise SystemExit(f"mono {b['mono']}: type {b['imt']} but the order requires {imt}")
        rows.append({"id": 30 + b["mono"], "question": b["question"].strip(), "set": "mono", "domain": "workplace",
                     "grid": {"imt": imt, "deceiver": "user", "domain": "workplace", "format": "bare"},
                     "targeted": None, "slot": None})
    for a in sorted(set_a, key=lambda x: x["slot"]):
        s = slots[a["slot"]]
        rows.append({"id": s["id"], "question": a["question"].strip(), "set": "div", "domain": s["domain"],
                     "grid": {"imt": s["imt"], "deceiver": s["deceiver"], "domain": s["domain"], "format": s["format"]},
                     "targeted": s["targeted"], "slot": s["slot"]})
    if [r["id"] for r in rows] != list(range(150)):
        raise SystemExit("ids are not 0-149 in order")
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
