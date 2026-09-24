import json
from pathlib import Path
d = Path("C:/Users/user4/expLB/cap/run/smoke_tf")
tot = {}
for p in sorted(d.glob("*.jsonl")):
    ws, cs = [], []
    for r in (json.loads(l) for l in p.open(encoding="utf-8")):
        t2 = [m for m in r["conversation"] if m["role"] == "assistant"][-1]["content"]
        ws.append(len(t2.split())); cs.append(len(t2))
    tot[p.stem] = (ws, cs)
    print(f"{p.stem:22} turn-2 words {ws} mean {sum(ws)/len(ws):.1f}   chars {cs} mean {sum(cs)/len(cs):.1f}")
a, b = tot["pc_caver"], tot["pc_position_holder"]
print(f"caver - holder: words {sum(a[0])/4 - sum(b[0])/4:+.1f}  chars {sum(a[1])/4 - sum(b[1])/4:+.1f}")
print("per-item word diff (caver - holder):", [x - y for x, y in zip(a[0], b[0])])
