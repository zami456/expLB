"""W5E-ownerdoc lab-side part 3 -- O-065 only. READ-ONLY.

thesis 4.1 row: own100 | 100 AA roles | AA's 240 shared questions x 5 prompts |
~1,200 per role (kept: median 1,163) | Llama-3.1-70B | ...

Checks: role count, per-role rollout count, distinct question_index / prompt_index,
per-role kept count under keep_scores={3} and {1,2,3}, and the extraction log's own
"N kept" lines plus the judge model recorded in the run logs.
"""
import glob, json, os, re, statistics

E = "C:/Users/user4/expLB"
resp = f"{E}/run/responses"
sc = f"{E}/run/scores"

fs = sorted(f for f in os.listdir(resp) if f.endswith(".jsonl"))
print(f"roles with responses: {len(fs)}")
roles = [f[:-6] for f in fs]
sfs = sorted(f for f in os.listdir(sc) if f.endswith(".json"))
print(f"roles with scores   : {len(sfs)}")
print(f"response roles without a score file: {sorted(set(roles) - {f[:-5] for f in sfs})}")
print(f"score roles without responses     : {sorted({f[:-5] for f in sfs} - set(roles))}")

# structure of one role
qi, pi, n = set(), set(), 0
with open(os.path.join(resp, fs[0]), encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        n += 1
        qi.add(o["question_index"]); pi.add(o["prompt_index"])
print(f"{fs[0]}: rows={n}  distinct question_index={len(qi)} (min {min(qi)} max {max(qi)})"
      f"  distinct prompt_index={len(pi)} {sorted(pi)}")

rows_per = []
for fn in fs:
    c = sum(1 for _ in open(os.path.join(resp, fn), encoding="utf-8"))
    rows_per.append(c)
print(f"rollouts per role: min={min(rows_per)} median={statistics.median(rows_per)} "
      f"max={max(rows_per)} total={sum(rows_per)}")

kept3, kept123, dist = [], [], {}
for fn in sfs:
    d = json.load(open(os.path.join(sc, fn), encoding="utf-8"))
    vals = list(d.values())
    for v in vals:
        dist[v] = dist.get(v, 0) + 1
    kept3.append(sum(1 for v in vals if v == 3))
    kept123.append(sum(1 for v in vals if v in (1, 2, 3)))
print(f"judged entries per role: min={min(len(json.load(open(os.path.join(sc,f),encoding='utf-8'))) for f in sfs)}")
print(f"score distribution over all roles: {dict(sorted(dist.items(), key=lambda x: str(x[0])))}")
print(f"kept (score==3) per role : min={min(kept3)} median={statistics.median(kept3)} "
      f"max={max(kept3)} mean={statistics.fmean(kept3):.1f}")
print(f"kept (score in 1,2,3)    : min={min(kept123)} median={statistics.median(kept123)} "
      f"max={max(kept123)}")

print()
print("-- extraction logs that print 'N kept' for these roles --")
for fp in sorted(glob.glob(f"{E}/logs/*.log")) + sorted(glob.glob(f"{E}/logs/*.txt")):
    b = open(fp, "rb").read()
    txt = None
    for enc in ("utf-8", "utf-16"):
        try:
            t = b.decode(enc)
            if "\x00" not in t:
                txt = t; break
        except Exception:
            pass
    if txt is None:
        continue
    hits = re.findall(r"Role (\S+): (\d+) kept, (\d+) usable", txt)
    if hits and any(h[0] in roles for h in hits):
        kk = [int(h[1]) for h in hits if h[0] in roles]
        uu = [int(h[2]) for h in hits if h[0] in roles]
        print(f"  {os.path.basename(fp)}: {len(kk)} matching roles; "
              f"kept min={min(kk)} median={statistics.median(kk)} max={max(kk)}; "
              f"usable min={min(uu)} median={statistics.median(uu)} max={max(uu)}")
        print("    first 5:", hits[:5])
    if "judge_model" in txt or "Llama-3.1-70B" in txt:
        for line in txt.splitlines():
            if ("judge_model" in line or "Llama-3.1-70B" in line) and ("run/" in line or "responses" in line
                                                                        or "judge" in line.lower()):
                print(f"  [{os.path.basename(fp)}] {line.strip()[:200]}")
                break
