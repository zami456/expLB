"""Stream one (or a few) lines out of a Claude transcript and print only the text blocks,
truncated. Never loads the file. Usage: python S_line.py <session-prefix> <line> [span] [maxchars]"""
import json, sys, glob, os
base = "C:/Users/SURAIYA/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB"
pref, ln = sys.argv[1], int(sys.argv[2])
span = int(sys.argv[3]) if len(sys.argv) > 3 else 0
maxc = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
f = [p for p in glob.glob(base + "/*.jsonl") if os.path.basename(p).startswith(pref)][0]
lo, hi = ln - span, ln + span
with open(f, encoding="utf-8") as fh:
    for i, line in enumerate(fh, 1):
        if i < lo: continue
        if i > hi: break
        try: rec = json.loads(line)
        except Exception: continue
        m = rec.get("message") or {}
        c = m.get("content")
        out = []
        if isinstance(c, str): out.append(c)
        elif isinstance(c, list):
            for b in c:
                if not isinstance(b, dict): continue
                if b.get("type") == "text": out.append(b["text"])
                elif b.get("type") == "tool_use": out.append("[tool_use " + json.dumps(b.get("input"))[:maxc] + "]")
                elif b.get("type") == "tool_result":
                    r = b.get("content")
                    if isinstance(r, list):
                        r = "".join(x.get("text", "") for x in r if isinstance(x, dict))
                    out.append("[tool_result] " + str(r)[:maxc])
        print(f"--- line {i} | {rec.get('type')} | {rec.get('timestamp')} ---")
        print(("\n".join(out))[:maxc])
