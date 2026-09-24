"""W5E-ownerdoc lab-side part 2. READ-ONLY.

1. O-022 / O-038 LEVEL 2: recompute the four single-source baselines from the 110 raw pairs
   stored inside run-premise/audit/audit_L40_s42.json.
2. O-065: own100 rollout / judge structure -- questions per role, prompt variants, judge model,
   score-3 kept counts.
3. O-082: print logs/validate.log (any encoding) -- the '+0.032 label lift' line.
"""
import codecs, glob, json, os, statistics

E = "C:/Users/user4/expLB"

print("#" * 78)
print("O-022 / O-038 LEVEL 2 -- baselines recomputed from the 110 stored pairs")
p = f"{E}/run-premise/audit/audit_L40_s42.json"
d = json.load(open(p, encoding="utf-8"))
pairs = d["pairs"]
targets = d["targets"]
print(f"  n_pairs stored = {len(pairs)}  targets = {len(targets)}")
by_t = {}
for pr in pairs:
    by_t.setdefault(pr["target"], []).append(pr["auroc"])
mean_t, med_t, ori_t, orc_t = [], [], [], []
for t in targets:
    v = by_t[t]
    mean_t.append(statistics.fmean(v))
    med_t.append(statistics.median(v))
    ori_t.append(statistics.fmean(max(a, 1 - a) for a in v))
    orc_t.append(max(v))
    print(f"    {t:<34} n={len(v):<3} mean={mean_t[-1]:.4f} median={med_t[-1]:.4f} "
          f"oriented={ori_t[-1]:.4f} oracle={orc_t[-1]:.4f}")
print(f"  RECOMPUTED mean_over_sources   = {statistics.fmean(mean_t)!r}   file={d['baselines']['mean_over_sources']!r}")
print(f"  RECOMPUTED median_over_sources = {statistics.fmean(med_t)!r}   file={d['baselines']['median_over_sources']!r}")
print(f"  RECOMPUTED oriented_mean       = {statistics.fmean(ori_t)!r}   file={d['baselines']['oriented_mean']!r}")
print(f"  RECOMPUTED best_source_oracle  = {statistics.fmean(orc_t)!r}   file={d['baselines']['best_source_oracle']!r}")
allv = [pr["auroc"] for pr in pairs]
print(f"  pairs below 0.5 = {sum(1 for a in allv if a < 0.5)} (file {d['pairs_below_half']}); "
      f"below 0.4 = {sum(1 for a in allv if a < 0.4)}; min = {min(allv)!r}")
mn = min(pairs, key=lambda x: x["auroc"])
print(f"  minimum pair: {mn['source']} -> {mn['target']} = {mn['auroc']!r}")
print(f"  flat (per-pair, not per-target) oriented mean = {statistics.fmean(max(a,1-a) for a in allv)!r}")
print(f"  pooled_lodo_full = {d['pooled_lodo_full']!r}; "
      f"pooled - oriented = {d['pooled_lodo_full'] - d['baselines']['oriented_mean']:+.4f} "
      f"(file bootstrap diff {d['bootstrap']['pooled_lodo_full-oriented_mean']['diff']!r})")

print()
print("#" * 78)
print("O-065 -- own100 bank: rollouts, questions, prompt variants, judge, kept rows")
resp = f"{E}/run/responses"
fs = sorted(os.listdir(resp))
print(f"  responses: {len(fs)} role files")
ex = os.path.join(resp, fs[0])
with open(ex, encoding="utf-8") as f:
    first = json.loads(f.readline())
print(f"  example {fs[0]}: keys = {sorted(first.keys())}")
for k, v in first.items():
    s = json.dumps(v) if not isinstance(v, str) else v
    print(f"    {k} = {s[:160]}")
# question / variant structure of one role
qs, vs = set(), set()
with open(ex, encoding="utf-8") as f:
    n = 0
    for line in f:
        o = json.loads(line)
        n += 1
        for kq in ("question_id", "qid", "question", "q_id"):
            if kq in o:
                qs.add(str(o[kq])[:200]); break
        for kv in ("variant", "prompt_variant", "variant_id", "prompt_id", "pid"):
            if kv in o:
                vs.add(str(o[kv])); break
print(f"  {fs[0]}: rows={n} distinct questions={len(qs)} distinct variants={len(vs)} {sorted(vs)[:8]}")

sc = f"{E}/run/scores"
sfs = sorted(os.listdir(sc))
print(f"  scores: {len(sfs)} files; first {sfs[:3]}")
sp = os.path.join(sc, sfs[0])
raw = open(sp, encoding="utf-8").read()
print(f"  {sfs[0]} size={len(raw)} starts: {raw[:300]}")
try:
    sj = json.loads(raw)
    print(f"  parsed as JSON: type={type(sj).__name__} "
          f"{'keys=' + str(sorted(sj.keys())[:20]) if isinstance(sj, dict) else 'len=' + str(len(sj))}")
    if isinstance(sj, list) and sj:
        print(f"    first element: {json.dumps(sj[0])[:300]}")
except Exception as ex2:
    print("  not a single JSON doc:", ex2)


def kept_count(path):
    r = open(path, encoding="utf-8").read()
    try:
        o = json.loads(r)
    except Exception:
        o = [json.loads(l) for l in r.splitlines() if l.strip()]
    if isinstance(o, dict):
        for k in ("scores", "rows", "items", "results"):
            if k in o and isinstance(o[k], list):
                o = o[k]; break
    if not isinstance(o, list):
        return None, None
    tot = len(o)
    k3 = 0
    for it in o:
        if isinstance(it, dict):
            for key in ("score", "judge_score", "rating", "grade"):
                if key in it:
                    try:
                        if int(it[key]) == 3:
                            k3 += 1
                    except Exception:
                        pass
                    break
        elif isinstance(it, (int, float)) and int(it) == 3:
            k3 += 1
    return tot, k3


tots, k3s = [], []
for fn in sfs:
    t, k = kept_count(os.path.join(sc, fn))
    if t is not None:
        tots.append(t); k3s.append(k)
if tots:
    print(f"  judged rows per role: n={len(tots)} min={min(tots)} median={statistics.median(tots)} max={max(tots)}")
    print(f"  score-3 kept per role: min={min(k3s)} median={statistics.median(k3s)} max={max(k3s)} total={sum(k3s)}")
# vectors: how many rows each role vector was built from
import re
for vd in [f"{E}/run/vectors", f"{E}/run/vectors_fp32"]:
    if os.path.isdir(vd):
        fl = sorted(os.listdir(vd))
        print(f"  {vd}: {len(fl)} files; first {fl[:3]}")

print()
print("#" * 78)
print("O-082 -- logs/validate.log")
lp = f"{E}/logs/validate.log"
b = open(lp, "rb").read()
for enc in ("utf-8", "utf-16", "utf-16-le", "latin-1"):
    try:
        txt = b.decode(enc)
        if txt.count("\x00") == 0:
            print(f"  decoded as {enc}, {len(txt)} chars")
            break
    except Exception:
        continue
print(txt)
