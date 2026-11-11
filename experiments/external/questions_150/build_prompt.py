"""Build the frozen authoring prompt: prompt_template.md + grid_slots.json + the 30 originals -> prompt.md.

Deterministic. The originals are read from the pipeline's question file (the bank's own questions
are not test data). Nothing else is read.

    python build_prompt.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINALS = Path(r"E:\aletheia\banks\Qwen3.5-27B\code\kd\questions\kd_questions_shared.jsonl")


def originals() -> list[dict]:
    rows = [json.loads(l) for l in ORIGINALS.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert [r["id"] for r in rows] == list(range(30)), "the originals must be ids 0-29 in order"
    return rows


def main() -> None:
    slots = json.loads((HERE / "grid_slots.json").read_text(encoding="utf-8"))
    lines = ["| slot | type | who would deceive | domain | format |", "|---|---|---|---|---|"]
    for s in slots:
        lines.append(f"| {s['slot']} | {s['imt']} | {s['deceiver']} | {s['domain']} | {s['format']} |")
    orig = originals()
    olines = [f"{r['id'] + 1}. {r['question']}" for r in orig]
    template = (HERE / "prompt_template.md").read_text(encoding="utf-8")
    prompt = template.replace("{{SLOTS}}", "\n".join(lines)).replace("{{ORIGINALS}}", "\n".join(olines))
    (HERE / "prompt.md").write_text(prompt, encoding="utf-8")
    sha = hashlib.sha256((HERE / "prompt.md").read_bytes()).hexdigest()
    osha = hashlib.sha256(ORIGINALS.read_bytes()).hexdigest()
    print(f"prompt.md written: {len(prompt.split())} words, sha256 {sha}")
    print(f"originals sha256 {osha}")


if __name__ == "__main__":
    main()
