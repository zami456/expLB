"""F-55: dump turn-2 assistant text from the conversation field of each cap probe jsonl."""
import json, os, sys
d = sys.argv[1]
only = sys.argv[2] if len(sys.argv) > 2 else None
for fn in sorted(os.listdir(d)):
    if only and only not in fn: continue
    p = os.path.join(d, fn)
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    print(f"\n===== {fn} =====")
    print("system_prompt:", repr(recs[0].get("system_prompt"))[:200])
    print("kind/framing/label:", recs[0].get("kind"), "|", recs[0].get("framing"), "|", recs[0].get("label"))
    for i, r in enumerate(recs):
        conv = r.get("conversation") or []
        asst = [m for m in conv if (m.get("role") if isinstance(m, dict) else None) == "assistant"]
        last = asst[-1].get("content", "") if asst else ""
        print(f"\n [{i}] Q={r.get('question')!r}")
        print(f"     correct={r.get('correct_answer')!r} turn1_is_correct={r.get('turn1_is_correct')} n_turns={len(conv)}")
        print(f"     turn1: {str(r.get('turn1'))[:150]!r}")
        print(f"     turn2: {str(last)[:400]!r}")
