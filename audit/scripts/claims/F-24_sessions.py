"""F-24 / F-14 - recover the console evidence from the owner's session transcript.

Streams the big session JSONL line-by-line (never loads it whole) and prints the
tool_result / text blocks in the windows that the Phase-1 mining pointed at:
  F-24: the 40.22 GB reading (~line 3710) and the 56.64 GB retraction (~line 4293);
  F-14: the 271-file scp comparison (~line 1877) and the rclone pull (~line 1892).

Usage: python F-24_sessions.py
"""
import json, os, re

SESS = os.path.expanduser(
    r"~/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB")
FN = [f for f in os.listdir(SESS) if f.startswith("b035ccb3")][0]
P = os.path.join(SESS, FN)
print("session:", P, os.path.getsize(P), "bytes\n")

WINDOWS = [("F-14 scp 271-file compare", 1840, 1900),
           ("F-14 rclone pull", 1885, 1900),
           ("F-24 first GB reading", 3690, 3730),
           ("F-24 retraction window", 4270, 4330)]

KEEP = {}
with open(P, encoding="utf-8", errors="replace") as f:
    for n, line in enumerate(f, 1):
        for name, lo, hi in WINDOWS:
            if lo <= n <= hi:
                KEEP.setdefault(n, line)
                break

def blocks(rec):
    m = rec.get("message") or {}
    c = m.get("content")
    if isinstance(c, str):
        yield ("text", c)
        return
    for b in (c or []):
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            yield ("text", b.get("text", ""))
        elif b.get("type") == "tool_use":
            yield ("tool_use", json.dumps(b.get("input", ""))[:600])
        elif b.get("type") == "tool_result":
            cc = b.get("content")
            if isinstance(cc, list):
                cc = "".join(x.get("text", "") for x in cc if isinstance(x, dict))
            yield ("tool_result", str(cc))

PAT = re.compile(r"(GB|MiB|204800|204,800|271|truncat|rclone|scp|blobs|cache|huggingface|"
                 r"retract|wrong|actually)", re.I)

for name, lo, hi in WINDOWS:
    print("=" * 78)
    print("###", name, f"(lines {lo}-{hi})")
    print("=" * 78)
    for n in range(lo, hi + 1):
        if n not in KEEP:
            continue
        try:
            rec = json.loads(KEEP[n])
        except Exception:
            continue
        ts = rec.get("timestamp", "")
        for kind, txt in blocks(rec):
            if not txt or not PAT.search(txt):
                continue
            txt = txt.strip()
            if len(txt) > 2200:
                txt = txt[:1100] + "\n   ...[cut]...\n" + txt[-900:]
            print(f"\n-- L{n} {ts} [{kind}]")
            for ln in txt.splitlines():
                print("   " + ln)
    print()
