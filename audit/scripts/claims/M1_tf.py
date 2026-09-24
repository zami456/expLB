#!/usr/bin/env python
"""M1 F-46 / F-51 -- teacher forcing.  Extracts the raw rollout text the two entries rest on.

F-46: logs/cap_smoke_v2.log -- pc_caver and pc_position_holder, variant 0, items Q0/Q13/Q16.
F-51: cap/run/smoke_tf/{pc_caver,pc_position_holder}.jsonl -- 4 records each, prefilled turn 1.

READ-ONLY.
"""
import json
import re
from pathlib import Path

R = Path("C:/Users/user4/expLB")

print("#" * 78)
print("# F-51  cap/run/smoke_tf")
print("#" * 78)
d = R / "cap" / "run" / "smoke_tf"
if not d.exists():
    print("MISSING", d)
else:
    for p in sorted(d.iterdir()):
        print(f"  {p.name}  {p.stat().st_size:,} B  mtime {p.stat().st_mtime}")
    turn1 = {}
    for p in sorted(d.glob("*.jsonl")):
        recs = [json.loads(l) for l in p.open(encoding="utf-8")]
        print(f"\n===== {p.name}: {len(recs)} records; keys {sorted(recs[0])}")
        for i, r in enumerate(recs):
            conv = r.get("conversation") or r.get("messages") or []
            sysp = r.get("system_prompt")
            if sysp is None:
                sysp = next((m["content"] for m in conv if m.get("role") == "system"), None)
            nonsys = [m for m in conv if m.get("role") != "system"]
            print(f"\n  --- record {i}  item={r.get('item_id') or r.get('question_id')} "
                  f"variant={r.get('variant')} role={r.get('role') or r.get('persona')}")
            print(f"      system: {str(sysp)[:220]}")
            for m in nonsys:
                print(f"      [{m['role']:9}] {m['content'][:300]!r}")
            # remember turn 1 (first assistant message) per item, to test byte-identity
            a = [m for m in nonsys if m["role"] == "assistant"]
            key = r.get("item_id") or r.get("question_id") or i
            if a:
                turn1.setdefault(key, {})[p.stem] = a[0]["content"]
    print("\n===== turn-1 byte identity across the two personas =====")
    for k, v in turn1.items():
        vals = list(v.values())
        same = len(set(vals)) == 1
        print(f"  item {k}: {len(vals)} personas, identical={same}  len={[len(x) for x in vals]}")
        if not same:
            for kk, vv in v.items():
                print(f"      {kk}: {vv[:200]!r}")

print()
print("#" * 78)
print("# F-46  logs/cap_smoke_v2.log")
print("#" * 78)
L = R / "logs" / "cap_smoke_v2.log"
raw = L.read_bytes()
enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
txt = raw.decode(enc, "replace")
lines = txt.splitlines()
print(f"{L}  {len(raw):,} B  enc={enc}  lines={len(lines)}  mtime {L.stat().st_mtime}")
for pat in ("ROLE", "QUESTION", "TURN", "SYSTEM", "variant", "temperature"):
    n = sum(1 for l in lines if pat in l)
    print(f"   lines containing {pat!r}: {n}")
print("\n--- first 40 non-empty lines ---")
shown = 0
for l in lines:
    if l.strip():
        print("   ", l[:200])
        shown += 1
        if shown >= 40:
            break

print("\n--- index of ROLE / item markers ---")
for i, l in enumerate(lines):
    if re.search(r"^\s*(ROLE|=== |--- |\[Q\d|Q\d+\b.*\?)", l) and len(l) < 200:
        print(f"   [{i}] {l[:180]}")
