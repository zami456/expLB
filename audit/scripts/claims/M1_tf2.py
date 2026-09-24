#!/usr/bin/env python
"""M1 F-46 pass 2 -- pull the six v2-smoke rollouts out of the UTF-16 PowerShell log,
plus the framing/kind fields of the F-51 teacher-forced records."""
import json
import re
from pathlib import Path

R = Path("C:/Users/user4/expLB")
L = R / "logs" / "cap_smoke_v2.log"
lines = L.read_bytes().decode("utf-16", "replace").splitlines()

NOISE = re.compile(r"\[K|RECEIVED|QUEUED|DISPATCHED|RUNNING|COMPLETED|Downloading|"
                   r"HTTP Request|NDIF|nnsight|Bytes|bytes|\d+%\|")


def clean(seg):
    out = []
    for l in seg:
        if NOISE.search(l):
            continue
        if l.strip():
            out.append(l.rstrip())
    return out


marks = [(i, l.strip()) for i, l in enumerate(lines)
         if l.strip().startswith(("ROLE ", "--- Q"))]
marks.append((len(lines), "END"))
for k in range(len(marks) - 1):
    i, lab = marks[k]
    j = marks[k + 1][0]
    if not lab.startswith("--- Q"):
        print("\n" + "#" * 78)
        print("#", lab)
        print("#" * 78)
        for l in clean(lines[i:min(j, i + 4)]):
            print("  ", l[:200])
        continue
    print("\n" + "=" * 78)
    print(lab[:200])
    print("=" * 78)
    for l in clean(lines[i + 1:j]):
        print("  ", l[:400])

print("\n\n" + "#" * 78)
print("# F-51 records: framing / kind / turn1_is_correct / label")
print("#" * 78)
for p in sorted((R / "cap" / "run" / "smoke_tf").glob("*.jsonl")):
    print(f"\n-- {p.name}")
    for r in (json.loads(l) for l in p.open(encoding="utf-8")):
        print(f"   qidx={r.get('question_index')} framing={r.get('framing')!r} "
              f"kind={r.get('kind')!r} label={r.get('label')!r} "
              f"turn1={r.get('turn1')!r} turn1_is_correct={r.get('turn1_is_correct')!r} "
              f"correct={r.get('correct_answer')!r} pidx={r.get('prompt_index')}")

print("\n-- cap/questions bank used by the smoke (first 6 rows of each jsonl) --")
qd = R / "cap" / "questions"
for p in sorted(qd.glob("*.jsonl")):
    rows = [json.loads(l) for l in p.open(encoding="utf-8")]
    print(f"\n   {p.name}: {len(rows)} rows, keys {sorted(rows[0])}")
    for i, r in enumerate(rows[:6]):
        print(f"      [{i}] {str(r)[:220]}")
