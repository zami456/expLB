"""F-55: read the three cap/run/probe jsonl files; print item count, system prompt presence,
turn-2 text per item, and search for the five quoted phrases."""
import json, os, re, sys
d = sys.argv[1]
quotes = ["the answer is indeed Tungsten", "36,000 feet", "goldfish have a relatively good",
          "I'm quite sure, actually", "smallest prime number is indeed 2",
          "Tungsten is indeed the element", "Tungsten indeed has the chemical symbol W",
          "it's not like they can remember things for a long time",
          "lightning never strikes the same place twice", "I think it's actually Canberra"]
for fn in sorted(os.listdir(d)):
    p = os.path.join(d, fn)
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    print(f"\n===== {fn}  ({len(recs)} records) =====")
    print("keys:", sorted(recs[0].keys()))
    for i, r in enumerate(recs):
        sysp = r.get("system") or r.get("system_prompt") or r.get("persona_prompt") or ""
        t2 = r.get("turn2") or r.get("response2") or r.get("assistant2") or r.get("reply") or ""
        if not t2:
            for k, v in r.items():
                if isinstance(v, str) and len(v) > 40 and k not in ("system", "system_prompt", "question", "prompt", "challenge"):
                    t2 = f"[{k}] " + v; break
        print(f" [{i}] sys_len={len(sysp)} | {str(r.get('question') or r.get('item') or r.get('q') or '')[:60]!r}")
        print(f"      turn2: {str(t2)[:260]!r}")
    blob = open(p, encoding="utf-8").read()
    hits = [q for q in quotes if q in blob]
    print("  QUOTE HITS:", hits)
