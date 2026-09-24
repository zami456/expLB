"""Phase 0: enumerate F-number headings in the findings ledger on a git ref.
Usage: python audit/scripts/p0_fnumbers.py [ref] [path]
Reads the file via `git show ref:path` so it works for any branch/commit."""
import re, subprocess, sys, collections
ref = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
path = sys.argv[2] if len(sys.argv) > 2 else "kd/docs/FINDINGS.md"
txt = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, encoding="utf-8").stdout
heads = []
for ln, line in enumerate(txt.splitlines(), 1):
    m = re.match(r"^(#{1,4})\s+\**\s*(F-\d+[a-z]?)\b(.*)$", line)
    if m:
        heads.append((ln, m.group(2), m.group(3).strip()))
ids = [h[1] for h in heads]
nums = sorted({int(re.sub(r"\D", "", i.split("-")[1])) for i in ids})
cnt = collections.Counter(ids)
print(f"ref={ref} path={path} lines={len(txt.splitlines())}")
print(f"headings={len(heads)} distinct_ids={len(cnt)} min=F-{nums[0]:02d} max=F-{nums[-1]:02d}")
print("missing numbers:", [n for n in range(nums[0], nums[-1] + 1) if n not in nums])
print("duplicated ids:", {k: v for k, v in cnt.items() if v > 1})
print("ids with letter suffix:", [i for i in cnt if re.search(r"[a-z]$", i)])
# order check: are headings in increasing numeric order?
seq = [int(re.sub(r"\D", "", i.split("-")[1])) for i in ids]
print("out-of-order positions:", [(heads[k][0], ids[k]) for k in range(1, len(seq)) if seq[k] < seq[k-1]])
# date per heading ("### F-01 · 2026-09-19 · title")
dates = collections.Counter()
for ln, i, rest in heads:
    d = re.search(r"20\d\d-\d\d-\d\d", rest)
    dates[d.group(0) if d else "no-date"] += 1
print("headings per date:", dict(sorted(dates.items())))
kw = collections.Counter()
for ln, i, rest in heads:
    for k in ("amend", "retract", "supersed", "correct", "withdraw", "revis", "replac", "overturn", "wrong"):
        if k in rest.lower(): kw[k] += 1
print("title keywords:", dict(kw))
for ln, i, rest in heads:
    print(f"{ln:5d} {i:6s} {rest[:150]}")
