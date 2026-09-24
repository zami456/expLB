"""READ-ONLY on the lab PC. Recomputes the scale-rung generation/judging numbers behind
F-27, F-30, F-54 and locates the F-58 crossgen log evidence.

Run via:
  python audit/scripts/labpc_run.py audit/scripts/p1_labpc_runpy.ps1 FILE=P4_lab_scores.py ARGS=
Writes nothing.
"""
import collections, glob, json, os, re, sys

R = r"C:\Users\user4\expLB"


def scores(root, label):
    files = sorted(glob.glob(os.path.join(root, "*.json")))
    if not files:
        print(f"[{label}] NO SCORE FILES at {root}")
        return None
    counts = collections.Counter()
    per_role = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        per_role[os.path.basename(f)[:-5]] = collections.Counter(d.values())
        counts.update(collections.Counter(d.values()))
    tot = sum(counts.values())
    keep = counts[2] + counts[3]
    print(f"[{label}] {root}")
    print(f"  files={len(files)}  scored_rollouts={tot}")
    for s in sorted(counts):
        print(f"  score {s}: {counts[s]:5d}  {100*counts[s]/tot:6.2f}%")
    print(f"  keep 2+3 = {keep}/{tot} = {100*keep/tot:.2f}%   drop 0+1 = {tot-keep}/{tot} = {100*(tot-keep)/tot:.2f}%")
    gp = os.path.join(R, "src", "assistant_axis_ndif", "pc_role_groups.json")
    if os.path.exists(gp):
        groups = json.load(open(gp, encoding="utf-8"))["groups"]
        for gname, g in groups.items():
            rows = [per_role[r] for r in g.get("roles", []) if r in per_role]
            if not rows:
                continue
            n = sum(sum(rc.values()) for rc in rows)
            kept = sum(rc[2] + rc[3] for rc in rows)
            print(f"  group {gname:18s} {len(rows):2d} roles  kept {kept:5d}/{n:5d} = {100*kept/n:6.2f}%  mean kept/role {kept/len(rows):6.2f}")
    worst = sorted(per_role.items(), key=lambda kv: (kv[1][0]+kv[1][1])/max(1, sum(kv[1].values())), reverse=True)[:5]
    print("  worst roles by drop rate: " + ", ".join(
        f"{r} {100*(rc[0]+rc[1])/max(1,sum(rc.values())):.1f}%" for r, rc in worst))
    return per_role


print("=" * 78)
print("F-27 / F-30: judge-score distributions")
for root, lab in ((os.path.join(R, "run-paper16", "scores"), "70B paper16"),
                  (os.path.join(R, "run-scale", "8b", "scores"), "8B"),
                  (os.path.join(R, "run-scale", "3b", "scores"), "3B")):
    scores(root, lab)
    print()

print("=" * 78)
print("F-27: rollout counts per rung (responses)")
for rung in ("3b", "8b"):
    d = os.path.join(R, "run-scale", rung, "responses")
    fs = sorted(glob.glob(os.path.join(d, "*.jsonl")))
    tot = 0
    short = []
    for f in fs:
        n = sum(1 for _ in open(f, encoding="utf-8"))
        tot += n
        if n != 100:
            short.append((os.path.basename(f), n))
    print(f"  {rung}: {len(fs)} role files, {tot} rows; non-100-row files: {short if short else 'none'}")

print()
print("=" * 78)
print("F-54: coords row counts per dataset")
for rung in ("3b", "8b"):
    base = os.path.join(R, "run-scale", rung, "coords")
    if not os.path.isdir(base):
        print(f"  {rung}: NO coords dir")
        continue
    tot = 0
    for ds in sorted(os.listdir(base)):
        p = os.path.join(base, ds, "rows.jsonl")
        if os.path.exists(p):
            n = sum(1 for _ in open(p, encoding="utf-8"))
            tot += n
            print(f"  {rung} {ds:28s} {n:6d}")
    print(f"  {rung} TOTAL {tot}")

print()
print("=" * 78)
print("F-58: crossgen scale logs")
for rung in ("3b", "8b"):
    p = os.path.join(R, "logs", f"crossgen_scale_{rung}.log")
    if not os.path.exists(p):
        print(f"  MISSING {p}")
        continue
    raw = open(p, "rb").read()
    txt = raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8", "replace")
    lines = txt.splitlines()
    print(f"  --- {p}  ({len(lines)} lines, {len(raw)} bytes)")
    for i, l in enumerate(lines):
        if re.search(r"datasets|--bank|mask|harm-pressure|wrote|Namespace|python |27_crossgen|matrices", l):
            print(f"    {i+1:5d}| {l.strip()[:200]}")

print()
print("=" * 78)
print("Scale status logs")
for p in sorted(glob.glob(os.path.join(R, "logs", "scale_*.status"))) + \
         sorted(glob.glob(os.path.join(R, "logs", "scale_coords.status"))):
    raw = open(p, "rb").read()
    txt = raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8", "replace")
    print(f"  --- {os.path.basename(p)}")
    for l in txt.splitlines():
        if l.strip():
            print(f"      {l.strip()[:190]}")
