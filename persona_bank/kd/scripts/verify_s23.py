"""Check the score-3 vs score-2+3 balance across the three contrasted groups, exactly."""
import json, glob
from collections import Counter
from pathlib import Path
groups = json.load(open("kd/personas/kd_role_groups.json", encoding="utf-8"))["groups"]
of = {r: g for g, s in groups.items() for r in s["roles"]}
per = {}
for f in sorted(glob.glob("run-kd/scores/*.json")):
    role = Path(f).stem
    d = json.load(open(f, encoding="utf-8"))
    vals = list(d.values()) if isinstance(d, dict) else d
    flat = []
    for v in vals:
        if isinstance(v, (int, float)): flat.append(int(v))
        elif isinstance(v, dict): flat += [int(x) for x in v.values() if isinstance(x, (int, float))]
        elif isinstance(v, list): flat += [int(x) for x in v if isinstance(x, (int, float))]
    per[role] = Counter(flat)

def report(names, label):
    s3 = [per[r].get(3, 0) for r in names if r in per]
    s23 = [per[r].get(3, 0) + per[r].get(2, 0) for r in names if r in per]
    n = [sum(per[r].values()) for r in names if r in per]
    return label, len(s3), sum(s3), sum(s3)/len(s3), min(s3), sum(s23), sum(s23)/len(s23), min(s23), sum(n)

print(f"{'group':20} {'roles':>5} {'s3 tot':>7} {'s3 avg':>7} {'s3 min':>7} | "
      f"{'s23 tot':>8} {'s23 avg':>8} {'s23 min':>8} | {'rollouts':>8}")
out = {}
for g in ("knowledge_denial", "sincere_ignorance", "plain_assertion"):
    r = report(groups[g]["roles"], g); out[g] = r
    print(f"{r[0]:20} {r[1]:>5} {r[2]:>7} {r[3]:>7.1f} {r[4]:>7} | {r[5]:>8} {r[6]:>8.1f} {r[7]:>8} | {r[8]:>8}")

kd, si, pa = out["knowledge_denial"], out["sincere_ignorance"], out["plain_assertion"]
print()
print("CONTRAST 1  knowledge_denial - sincere_ignorance")
print(f"  score-3 only : {kd[3]:.1f} vs {si[3]:.1f} per role   ratio {kd[3]/si[3]:.2f}x"
      f"   (totals {kd[2]} vs {si[2]})")
print(f"  scores 2+3   : {kd[6]:.1f} vs {si[6]:.1f} per role   ratio {kd[6]/si[6]:.2f}x"
      f"   (totals {kd[5]} vs {si[5]})")
print("CONTRAST 2  knowledge_denial - plain_assertion")
print(f"  score-3 only : {kd[3]:.1f} vs {pa[3]:.1f} per role   ratio {kd[3]/pa[3]:.2f}x")
print(f"  scores 2+3   : {kd[6]:.1f} vs {pa[6]:.1f} per role   ratio {kd[6]/pa[6]:.2f}x")
print()
lo = min(r[7] for r in (kd, si, pa))
print(f"min per-role count under s23 across all three groups: {lo}  (min_count is 20)")
zeros = {r: per[r].get(0,0)+per[r].get(1,0) for g in ("knowledge_denial","sincere_ignorance","plain_assertion") for r in groups[g]["roles"] if r in per}
print(f"score 0/1 (refusals or out-of-role) in the new groups: {sum(zeros.values())} of "
      f"{sum(sum(per[r].values()) for r in zeros)} rollouts")
