"""W6G: stream a transcript and print a trimmed view of a given line (or range).

Usage: python _W6G_peek.py <session_prefix> <line> [<nlines>] [<maxchars>]

Reads line-by-line (never loads the file). Prints only the text/tool_result
content, truncated. Output is for the agent's screen only; it is NOT written
into audit/ (see SESSION_MINING.md rule 3).
"""
import json
import sys
import glob
import os

BASE = r"C:/Users/SURAIYA/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB"


def content_text(msg):
    if msg is None:
        return ""
    c = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(c, str):
        return c
    out = []
    if isinstance(c, list):
        for b in c:
            if not isinstance(b, dict):
                out.append(str(b))
                continue
            t = b.get("type")
            if t == "text":
                out.append(b.get("text", ""))
            elif t == "tool_use":
                out.append("[tool_use %s] %s" % (b.get("name"), json.dumps(b.get("input", {}))[:4000]))
            elif t == "tool_result":
                cc = b.get("content")
                if isinstance(cc, str):
                    out.append("[tool_result] " + cc)
                elif isinstance(cc, list):
                    for x in cc:
                        if isinstance(x, dict) and x.get("type") == "text":
                            out.append("[tool_result] " + x.get("text", ""))
    return "\n".join(out)


def main():
    pref = sys.argv[1]
    start = int(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    mx = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
    matches = glob.glob(os.path.join(BASE, pref + "*.jsonl"))
    if not matches:
        print("NO SESSION MATCHING", pref)
        return
    p = matches[0]
    print("### file:", os.path.basename(p))
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, 1):
            if i < start:
                continue
            if i >= start + n:
                break
            try:
                rec = json.loads(line)
            except Exception:
                print("L%d PARSE-FAIL" % i)
                continue
            txt = content_text(rec.get("message"))
            print("--- L%d [%s] %s" % (i, rec.get("type"), rec.get("timestamp")))
            print(txt[:mx])


if __name__ == "__main__":
    main()
