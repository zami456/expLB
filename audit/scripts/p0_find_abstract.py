"""Phase 0 follow-up: recover the collaborator's title+abstract as the owner pasted it into earlier Claude sessions.
Reads ~/.claude/projects/<this project>/*.jsonl (read-only), prints every USER message containing the title."""
import json, os, glob
D = os.path.expanduser("~/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB")
KEY = "One Direction Generalizes"
for f in sorted(glob.glob(D + "/*.jsonl")):
    if "4f0b290f" in f:  # this audit session
        continue
    with open(f, encoding="utf-8", errors="replace") as fh:
        for ln, line in enumerate(fh, 1):
            if KEY not in line:
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if o.get("type") != "user":
                continue
            c = o.get("message", {}).get("content")
            texts = [c] if isinstance(c, str) else [x.get("text", "") for x in (c or []) if isinstance(x, dict) and x.get("type") == "text"]
            for t in texts:
                if KEY in t:
                    print(f"=== {os.path.basename(f)} line {ln} timestamp {o.get('timestamp')}")
                    print(t[:6000])
                    print()
