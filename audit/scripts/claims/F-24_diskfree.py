import json, os, re
SESS = os.path.expanduser(r"~/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB")
P = os.path.join(SESS, [f for f in os.listdir(SESS) if f.startswith("b035ccb3")][0])
pat = re.compile(r"free GB|Free/1GB|GB free|working set|WorkingSet|WS\b", re.I)
n = 0
with open(P, encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f, 1):
        if 3300 <= i <= 3720 and ("free" in line.lower() or "working set" in line.lower()):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            m = rec.get("message") or {}
            c = m.get("content")
            blocks = [c] if isinstance(c, str) else [
                (b.get("text") if b.get("type") == "text" else
                 (json.dumps(b.get("input"))[:400] if b.get("type") == "tool_use" else
                  ("".join(x.get("text","") for x in b.get("content") or [] if isinstance(x, dict))
                   if isinstance(b.get("content"), list) else str(b.get("content")))))
                for b in (c or []) if isinstance(b, dict)]
            for t in blocks:
                if not t:
                    continue
                for ln in str(t).splitlines():
                    if pat.search(ln) or re.search(r"\b\d+(\.\d+)?\s*GB\b", ln):
                        print(f"L{i} {rec.get('timestamp','')}: {ln.strip()[:190]}")
                        n += 1
print("hits:", n)
