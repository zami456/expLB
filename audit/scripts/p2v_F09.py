"""Verify F-09: logprob_gold deciles, softmax-margin peaking, and the binary correct/incorrect split.
Usage: p2v_F09.py <scores.jsonl>"""
import collections, json, sys

p = sys.argv[1]
recs, bad = [], 0
with open(p, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: recs.append(json.loads(line))
        except Exception: bad += 1
print("file", p)
print("parsed records", len(recs), "| unparseable/truncated lines", bad)
if recs:
    print("keys of record 0:", sorted(recs[0].keys()))
    print("sample:", json.dumps({k: v for k, v in list(recs[0].items())[:12]}, ensure_ascii=False)[:600])

def numeric(key):
    return [r[key] for r in recs if isinstance(r.get(key), (int, float))]

for key in sorted({k for r in recs for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}):
    v = sorted(numeric(key))
    if len(v) < 10: continue
    dec = [v[min(len(v) - 1, int(round(i * (len(v) - 1) / 10)))] for i in range(10)]
    print(f"  {key:<24} n={len(v):5d} deciles(lower edges)={[round(x,1) for x in dec]}  zeros={sum(1 for x in v if x == 0)}  >=-1e-4={sum(1 for x in v if x >= -1e-4)}")

print("\n-- boolean / categorical fields --")
for key in sorted({k for r in recs for k, v in r.items() if isinstance(v, bool) or isinstance(v, str)}):
    c = collections.Counter(r.get(key) for r in recs)
    if len(c) <= 6:
        print(f"  {key:<24} {dict(c)}")
