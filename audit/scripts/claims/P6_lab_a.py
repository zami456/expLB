"""P6 batch, lab-side part A: F-07, F-10, F-20, F-59.  READ-ONLY.
Run from C:/Users/user4/audit_scratch with the lab analysis venv.
"""
import glob, json, os, re, sys
from collections import Counter

R = "C:/Users/user4/expLB"


def head(t):
    print("\n" + "=" * 78 + "\n== " + t + "\n" + "=" * 78)


def st(p):
    return f"{p}  {os.path.getsize(p)} B  mtime={__import__('datetime').datetime.fromtimestamp(os.path.getmtime(p)):%Y-%m-%d %H:%M:%S}"


# ------------------------------------------------------------------ F-07
head("F-07  split-half reliability of d_knowledge; the 121-item class")
p = f"{R}/kd/results/phase2_gate.json"
print(st(p))
g = json.load(open(p, encoding="utf-8"))
print("pooling:", g["pooling"], "| n_text_matched:", g["n_text_matched"],
      "| n_knowledge_matched:", g["n_knowledge_matched"])
for row in g["rows"]:
    print(f"  L{row['layer']:<3} split_half_cos={row['split_half_cos']:.6f}  "
          f"random_pair_cos={row['random_pair_cos']:.6f}")
print(f"  stated: 0.988 / 0.995 / 0.993  and a random-pair floor of 0.007")
print(f"  rounded to 3dp: " + " / ".join(f"{r['split_half_cos']:.3f}" for r in g["rows"]))
print(f"  random_pair rounded: " + " / ".join(f"{r['random_pair_cos']:.3f}" for r in g["rows"]))
pk = f"{R}/kd/run/factor_probes/kd_knowledge/rows.jsonl"
print(st(pk))
lab = Counter()
keys = None
for line in open(pk, encoding="utf-8"):
    if not line.strip():
        continue
    d = json.loads(line)
    keys = keys or sorted(d.keys())
    lab[d.get("label")] += 1
print("  kd_knowledge rows.jsonl keys:", keys)
print("  label counts:", dict(lab), "| total", sum(lab.values()))
pu = f"{R}/kd/run/factor_probes/kd_utterance/rows.jsonl"
if os.path.exists(pu):
    lu = Counter(json.loads(l)["label"] for l in open(pu, encoding="utf-8") if l.strip())
    print("  kd_utterance label counts:", dict(lu), "| total", sum(lu.values()))

# ------------------------------------------------------------------ F-10
head("F-10  score-3 filtering imposes an asymmetric sample imbalance")
gp = f"{R}/kd/personas/kd_role_groups.json"
print(st(gp))
groups = json.load(open(gp, encoding="utf-8"))["groups"]
per = {}
for f in sorted(glob.glob(f"{R}/run-kd/scores/*.json")):
    role = os.path.splitext(os.path.basename(f))[0]
    d = json.load(open(f, encoding="utf-8"))
    vals = list(d.values()) if isinstance(d, dict) else d
    flat = []
    for v in vals:
        if isinstance(v, (int, float)):
            flat.append(int(v))
        elif isinstance(v, dict):
            flat += [int(x) for x in v.values() if isinstance(x, (int, float))]
        elif isinstance(v, list):
            flat += [int(x) for x in v if isinstance(x, (int, float))]
    per[role] = Counter(flat)
print(f"  score files: {len(per)}   roles in groups: {sum(len(s['roles']) for s in groups.values())}")
print(f"  groups: {list(groups)}")
out = {}
print(f"\n  {'group':20} {'roles':>5} {'s3 tot':>7} {'s3 avg':>7} {'s23 tot':>8} {'s23 avg':>8} {'rollouts':>9}")
for gname in ("knowledge_denial", "sincere_ignorance", "plain_assertion"):
    names = [r for r in groups[gname]["roles"] if r in per]
    s3 = [per[r].get(3, 0) for r in names]
    s23 = [per[r].get(3, 0) + per[r].get(2, 0) for r in names]
    n = [sum(per[r].values()) for r in names]
    out[gname] = (len(names), sum(s3), sum(s3) / len(s3), sum(s23), sum(s23) / len(s23), sum(n))
    print(f"  {gname:20} {len(names):>5} {sum(s3):>7} {sum(s3)/len(s3):>7.1f} "
          f"{sum(s23):>8} {sum(s23)/len(s23):>8.1f} {sum(n):>9}")
kd, si, pa = out["knowledge_denial"], out["sincere_ignorance"], out["plain_assertion"]
print(f"\n  stated per-role: kd 116.8/146.8  sincere 89.0/150.0  plain 73.3/149.0")
print(f"  imbalance s3  kd/sincere = {kd[2]/si[2]:.2f}x   kd/plain = {kd[2]/pa[2]:.2f}x   (stated 1.31x / 1.59x)")
print(f"  imbalance s23 kd/sincere = {kd[4]/si[4]:.2f}x   kd/plain = {kd[4]/pa[4]:.2f}x   (stated 0.98x)")
new = [r for gname in ("knowledge_denial", "sincere_ignorance", "plain_assertion")
       for r in groups[gname]["roles"] if r in per]
z = sum(per[r].get(0, 0) + per[r].get(1, 0) for r in new)
tot = sum(sum(per[r].values()) for r in new)
print(f"  score 0/1 in the three new groups: {z} of {tot}   (stated 16 of 1,500)")
print(f"  discarded by s3 that s23 keeps: {sum(per[r].get(2,0) for r in new)} of {tot} "
      f"= {sum(per[r].get(2,0) for r in new)/tot:.1%}   (stated ~30%)")
print(f"  'totals 467 vs 267' check: s3 totals kd+sincere+plain = "
      f"{kd[1]}+{si[1]}+{pa[1]} = {kd[1]+si[1]+pa[1]}; s23 = {kd[3]}+{si[3]}+{pa[3]} = {kd[3]+si[3]+pa[3]}")
print(f"  kd s3 total {kd[1]} vs sincere+plain s3 {si[1]+pa[1]}")
print(f"  per-role score histograms (three new groups):")
for gname in ("knowledge_denial", "sincere_ignorance", "plain_assertion"):
    for r in groups[gname]["roles"]:
        if r in per:
            print(f"    {gname:20} {r:26} " + " ".join(f"{k}:{per[r].get(k,0)}" for k in (0, 1, 2, 3))
                  + f"  total {sum(per[r].values())}")

# ------------------------------------------------------------------ F-20
head("F-20  the matched slice is reportable on passB ONLY")
for store in ("passA", "passB", "passC"):
    for ds in ("harm-pressure-knowledge-report", "harm-pressure-kr-matched"):
        p = f"{R}/run-coords/{store}/{ds}/rows.jsonl"
        if not os.path.exists(p):
            print(f"  {store}/{ds}: MISSING")
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        c = Counter(r.get("label") for r in rows)
        print(f"  {store:6} {ds:32} n={len(rows):5}  labels={dict(c)}   "
              f"mtime={__import__('datetime').datetime.fromtimestamp(os.path.getmtime(p)):%Y-%m-%d %H:%M}")
print("\n  --- what the logs said at the time (the 98 / 89:9 state) ---")
for lg in (f"{R}/logs/kd_passAC.log", f"{R}/logs/passAC11_view.log"):
    if not os.path.exists(lg):
        print("  MISSING", lg); continue
    print("  " + st(lg))
    try:
        txt = open(lg, encoding="utf-8", errors="replace").read()
    except Exception:
        txt = open(lg, encoding="utf-16", errors="replace").read()
    if "\x00" in txt:
        txt = open(lg, encoding="utf-16", errors="replace").read()
    for ln in txt.splitlines():
        if re.search(r"matched|kr-matched|HP-KR|harm-pressure-knowledge-report|n=\d+", ln) and len(ln) < 300:
            print("    |", ln.strip())

# ------------------------------------------------------------------ F-59
head("F-59  HP-KR growth gate failed")
for name, f in (("smoke(primary, correct==0, harmful)", "rollouts"),
                ("control(fallback 1, correct==0, control)", "control"),
                ("margin(fallback 2, lowest margin, harmful)", "margin")):
    p = f"{R}/kd/run/denial/{f}.jsonl"
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    den = sum(1 for r in rows if r.get("denies"))
    print(f"  {name:42} n={len(rows):3}  denies={den:3}  rate={den/len(rows):.3f}   "
          f"templates={sorted({r['template'] for r in rows})}  "
          f"correct={dict(Counter(r['correct'] for r in rows))}")
    print(f"      distinct item_ids: {len({r['item_id'] for r in rows})}; "
          f"responses: {dict(Counter(r['response'].strip()[:40] for r in rows))}")
print("\n  --- GATE lines in the logs ---")
for lg in ("hpkr_denial_smoke", "hpkr_denial_control", "hpkr_denial_margin"):
    p = f"{R}/logs/{lg}.log"
    raw = open(p, "rb").read()
    txt = raw.decode("utf-16", "replace") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")
    print("  " + st(p))
    for i, ln in enumerate(txt.splitlines(), 1):
        if re.search(r"denial rate|GATE|OOM|selected|items", ln, re.I) and len(ln) < 300:
            print(f"    {lg}:{i} | {ln.strip()}")

print("\n  --- the 312 'wrong' items and the 75%, recomputed from the FULL lab scores.jsonl ---")
p = f"{R}/kd/run/knowledge/scores.jsonl"
print("  " + st(p))
sc = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
print(f"  items: {len(sc)}   accuracy = {sum(r['correct'] for r in sc)/len(sc):.4f}   "
      f"correct==0: {sum(1 for r in sc if r['correct']==0)}   correct==1: {sum(1 for r in sc if r['correct']==1)}")
bysub = Counter(r["subset"] for r in sc)
print("  by subset:", dict(bysub),
      {s: round(sum(r["correct"] for r in sc if r["subset"] == s) / bysub[s], 4) for s in bysub})
ids = {r["item_id"] for r in sc}
for f in ("rollouts", "control", "margin"):
    rows = [json.loads(l) for l in open(f"{R}/kd/run/denial/{f}.jsonl", encoding="utf-8") if l.strip()]
    miss = [r["item_id"] for r in rows if r["item_id"] not in ids]
    sel = {r["item_id"]: r for r in sc if r["item_id"] in {x["item_id"] for x in rows}}
    corr = Counter(sel[i]["correct"] for i in sel)
    print(f"  {f:9}: {len(rows)} rows, item_ids not in scores.jsonl: {len(miss)}; "
          f"their scores.jsonl 'correct': {dict(corr)}; "
          f"margins min/max {min((sel[i]['margin'] for i in sel), default=None)}/"
          f"{max((sel[i]['margin'] for i in sel), default=None)}")
print("  denial rate among wrong items (primary) = 1 - claim rate; "
      "claimed-to-know share = "
      f"{1 - sum(1 for l in open(f'{R}/kd/run/denial/rollouts.jsonl', encoding='utf-8') if l.strip() and json.loads(l)['denies'])/20:.2f}"
      "   (stated 75%)")

# the truncated git copy, for the RF-5 note
print("\n  --- RF-5: what the committed (truncated) copy would give ---")
print("  lab scores.jsonl bytes:", os.path.getsize(p))
