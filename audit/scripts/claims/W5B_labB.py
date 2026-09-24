"""W5B-ledger batch, lab-PC pass B - READ-ONLY.
F-64 keep rates (scores are flat {rollout_id: score} JSON), F-55's missing pc_position_holder
source, F-104 status files, F-22 log quotes, F-50's 8B timing.  Writes nothing.
"""
import collections, glob, json, os, re

R = "C:/Users/user4/expLB"


def sec(t):
    print("\n" + "=" * 78 + "\n== " + t + "\n" + "=" * 78)


def readtext(p):
    b = open(p, "rb").read()
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return b.decode("utf-16", "replace")
    return b.decode("utf-8", "replace")


# ---------------------------------------------------------------- F-64 -----
sec("F-64  score distribution and per-group keep rate, from run-cap/scores/*.json")
sd = f"{R}/run-cap/scores"
per = {}
dist = collections.Counter()
for x in sorted(os.listdir(sd)):
    if not x.endswith(".json"):
        continue
    d = json.load(open(f"{sd}/{x}", encoding="utf-8"))
    if not isinstance(d, dict):
        print("  !! unexpected structure in", x, type(d)); continue
    vals = list(d.values())
    per[os.path.splitext(x)[0]] = vals
    dist.update(vals)
N = sum(dist.values())
print(f"  {len(per)} role files, {N} scored rollouts")
for k in sorted(dist):
    print(f"    score {k}: {dist[k]:>5}  {dist[k]/N:.4%}   (ledger: 0->78/6.5%, 1->0/0.0%, 2->714/59.5%, 3->408/34.0%)")
if 1 not in dist:
    print("    score 1: 0  0.0000%")
keep = sum(v for k, v in dist.items() if k in (2, 3))
print(f"  keep 2+3 = {keep}/{N} = {keep/N:.4%};  drop 0+1 = {N-keep}/{N} = {(N-keep)/N:.4%}")

print("\n  per-role drop (score 0 or 1)  [ledger: pc_deferrer 19%, pc_calibrated_holder 17%]")
for role in sorted(per, key=lambda r: -sum(1 for v in per[r] if v in (0, 1))):
    sc = per[role]
    dr = sum(1 for v in sc if v in (0, 1))
    print(f"    {role:30} {dr:>3}/{len(sc):<5} = {dr/len(sc):.1%}")

print("\n  looking for cap_role_groups.json")
cands = glob.glob(f"{R}/**/cap_role_groups.json", recursive=True)
print("   ", cands[:10])
for gf in cands[:1]:
    g = json.load(open(gf, encoding="utf-8"))
    print(f"  {gf}:")
    print("   ", json.dumps(g)[:900])
    groups = {k: v["roles"] for k, v in g.get("groups", {}).items()} if "groups" in g else {}
    if groups:
        for grp, roles in groups.items():
            tot = kept = miss = 0
            for role in roles:
                sc = per.get(role)
                if sc is None:
                    miss += 1; continue
                tot += len(sc); kept += sum(1 for v in sc if v in (2, 3))
            print(f"    {grp:22} {len(roles)} roles (missing scores: {miss})  "
                  f"kept {kept:>4}/{tot:<5} = {kept/tot:.4%}" if tot else f"    {grp}: no scores")

# ---------------------------------------------------------------- F-55 -----
sec("F-55  where does the pc_position_holder turn-2 quote live?")
needle = "Tungsten is indeed the element"
found = []
for base in (f"{R}/cap/run", f"{R}/run-cap", f"{R}/cap"):
    for p in glob.glob(base + "/**/*.jsonl", recursive=True):
        try:
            t = open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        if needle in t:
            found.append(p)
print("  files containing", repr(needle), "->", found)
for p in found[:2]:
    for l in open(p, encoding="utf-8", errors="replace"):
        if needle in l:
            r = json.loads(l)
            conv = r.get("conversation") or []
            print(f"   {p}")
            print(f"     system_prompt: {str(r.get('system_prompt'))[:120]}")
            print(f"     turn2: {' '.join(conv[-1]['content'].split())[:260]}")
            break
# also: is there a pc_position_holder rollout in run-cap/responses with the 8 probe questions?
p = f"{R}/run-cap/responses/pc_position_holder.jsonl"
if os.path.exists(p):
    n = sum(1 for _ in open(p, encoding="utf-8"))
    print(f"  run-cap/responses/pc_position_holder.jsonl: {n} rows (the 100-rollout cap bank, not the 8-item probe)")

# --------------------------------------------------------------- F-104 -----
sec("F-104  sarc_pilot / cap_controls status files")
for f in ("logs/sarc_pilot.status.2026-09-21-buggy-verdict", "logs/sarc_pilot.status.pre-e2e-2026-09-22",
          "logs/sarc_pilot.status", "logs/cap_controls.status"):
    p = f"{R}/{f}"
    print(f"\n  -- {p}  {'EXISTS' if os.path.exists(p) else 'MISSING'}")
    if os.path.exists(p):
        print(f"     bytes={os.path.getsize(p)}  mtime={__import__('datetime').datetime.fromtimestamp(os.path.getmtime(p)):%Y-%m-%d %H:%M}")
        for l in readtext(p).splitlines():
            print("     |", l[:200])
print("\n  other sarc_pilot.status* files present:")
for p in sorted(glob.glob(f"{R}/logs/sarc_pilot.status*")):
    print("   ", os.path.basename(p), os.path.getsize(p))

# ---------------------------------------------------------------- F-22 -----
sec("F-22  the truncated exception, and the first post-fix message")
for f, pat, n in ((f"{R}/logs/scale_8b_generate.log", r"session attempt", 2),
                  (f"{R}/logs/scale_3b_generate.log", r"session attempt", 2)):
    if not os.path.exists(f):
        print(f"\n  {f}: NOT FOUND"); continue
    L = readtext(f).splitlines()
    hit = [i for i, l in enumerate(L) if re.search(pat, l)]
    print(f"\n  -- {f}: {len(L)} lines, {len(hit)} 'session attempt' lines")
    for i in hit[:n]:
        for j in range(i, min(len(L), i + 12)):
            print(f"     {j+1:>6}: {L[j][:170]}")
        print("     ---")

# ---------------------------------------------------------------- F-50 -----
sec("F-50  8B instructed-deception timing (for the 2.4x ratio)")
L = readtext(f"{R}/logs/scale_coords_8b.log").splitlines()
for i, l in enumerate(L, 1):
    if re.search(r"18:3[0-9]:|18:4[0-9]:", l) and re.search(r"instructed-deception|harm-pressure-choice", l) \
       and re.search(r"to do|already complete|sessions|100%", l):
        print(f"  {i:>6}: {l[:190]}")
print("\n  3B instructed-deception progress-bar end lines:")
L3 = readtext(f"{R}/logs/scale_coords_3b.log").splitlines()
for i, l in enumerate(L3, 1):
    if "instructed-deception: 100%" in l and "400/400" in l:
        print(f"  {i:>6}: {l[:190]}")
print("\nDONE W5B_labB")
