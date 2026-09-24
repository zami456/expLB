"""W5B-ledger batch, lab-PC pass A - READ-ONLY.

Re-derives the file-level numbers of F-40, F-47, F-50, F-55, F-64, plus the log quotes of
F-22 and F-104.  Opens owner files for reading only; writes nothing anywhere.

Run:  python audit/scripts/labpc_run.py audit/scripts/p1_labpc_runpy.ps1 FILE=W5B_labA.py ARGS=
"""
import collections, glob, json, os, re, sys

R = "C:/Users/user4/expLB"


def sec(t):
    print("\n" + "=" * 78)
    print("== " + t)
    print("=" * 78)


def readtext(p, limit=None):
    """Decode a possibly UTF-16 (PowerShell `*>`) log."""
    b = open(p, "rb").read() if limit is None else open(p, "rb").read(limit)
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return b.decode("utf-16", "replace")
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("utf-8", "replace")


def nlines(p):
    n = 0
    with open(p, "rb") as f:
        for _ in f:
            n += 1
    return n


# ---------------------------------------------------------------- F-40 -----
sec("F-40  passB row counts (ledger total 14,718; instructed-deception 5,432 = 37%)")
pb = f"{R}/run-coords/passB"
if os.path.isdir(pb):
    tot = 0
    rows = {}
    for d in sorted(os.listdir(pb)):
        f = f"{pb}/{d}/rows.jsonl"
        if os.path.isfile(f):
            rows[d] = nlines(f)
    for d, n in sorted(rows.items(), key=lambda kv: -kv[1]):
        print(f"  {d:36} {n:>7}")
    grid = {k: v for k, v in rows.items() if k != "alpaca"}
    tot = sum(grid.values())
    print(f"  --- 11-dataset grid total (alpaca excluded) = {tot}")
    if "instructed-deception" in grid:
        print(f"  --- instructed-deception share = {grid['instructed-deception']/tot:.4f}")
    print(f"  --- alpaca = {rows.get('alpaca')}")
else:
    print("  NOT FOUND:", pb)

sec("F-40  8B coords log, the 16:59-17:18 window the rate was read off")
lg = f"{R}/logs/scale_coords_8b.log"
if os.path.isfile(lg):
    t = readtext(lg)
    L = t.splitlines()
    print(f"  {lg}: {len(L)} lines")
    for i, l in enumerate(L[:25], 1):
        print(f"  {i:>6}: {l[:200]}")
    pat = re.compile(r"1[67]:\d\d:\d\d")
    hits = [(i, l) for i, l in enumerate(L, 1)
            if pat.search(l) and re.search(r"to do|chunk|complete|rows|wrote|Done", l, re.I)]
    print(f"  ... {len(hits)} timestamped progress lines in the 16:xx/17:xx window; first 40:")
    for i, l in hits[:40]:
        print(f"  {i:>6}: {l[:200]}")
else:
    print("  NOT FOUND:", lg)

# ---------------------------------------------------------------- F-47 -----
sec("F-47  the stale items.pt (log lines 14819 / 14821) and the CURRENT 8B store")
if os.path.isfile(lg):
    for i, l in enumerate(L, 1):
        if re.search(r"harm-pressure-choice:.*to do|instructed-deception:.*to do|sessions\b", l):
            print(f"  {i:>6}: {l[:220]}")
lg2 = f"{R}/logs/scale_coords_8b_hpc.log"
if os.path.isfile(lg2):
    t2 = readtext(lg2); L2 = t2.splitlines()
    print(f"\n  {lg2}: {len(L2)} lines; first 3 and last 3")
    for l in L2[:3] + ["   ..."] + L2[-3:]:
        print("   ", l[:200])
for d in ("instructed-deception", "harm-pressure-choice"):
    p = f"{R}/run-scale/8b/coords/{d}"
    if os.path.isdir(p):
        fs = os.listdir(p)
        ip = f"{p}/items.pt"
        rj = f"{p}/rows.jsonl"
        print(f"\n  8B {d}: {len(fs)} files; chunks={sum(1 for x in fs if x.startswith('chunk'))}")
        print(f"      items.pt   = {os.path.getsize(ip) if os.path.exists(ip) else 'ABSENT'}")
        print(f"      rows.jsonl = {nlines(rj) if os.path.exists(rj) else 'ABSENT'} rows")
    else:
        print(f"\n  8B {d}: DIR NOT FOUND")

# ---------------------------------------------------------------- F-50 -----
sec("F-50  3B instructed-deception: 800 rows in 400 chunks (2.0 rows/chunk)")
p = f"{R}/run-scale/3b/coords/instructed-deception"
if os.path.isdir(p):
    fs = os.listdir(p)
    ch = [x for x in fs if x.startswith("chunk")]
    print(f"  files={len(fs)}  chunk files={len(ch)}  other={sorted(set(fs)-set(ch))}")
    rj = f"{p}/rows.jsonl"
    nr = nlines(rj) if os.path.exists(rj) else None
    print(f"  rows.jsonl rows = {nr}")
    if nr and ch:
        print(f"  rows/chunk = {nr/len(ch):.3f}")
else:
    print("  NOT FOUND:", p)
lg3 = f"{R}/logs/scale_coords_3b.log"
if os.path.isfile(lg3):
    L3 = readtext(lg3).splitlines()
    print(f"\n  {lg3}: {len(L3)} lines")
    for l in L3[:6]:
        print("   ", l[:200])
    ip = [l for l in L3 if re.search(r"instructed-deception", l)]
    print(f"  instructed-deception lines: {len(ip)}; first 5 / last 5")
    for l in ip[:5] + ["   ..."] + ip[-5:]:
        print("   ", l[:200])
    # chunk-rate: timestamps of the first and last chunk write for that dataset
    ts = [re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)", l) for l in L3]
    ts = [m.group(1) for m in ts if m]
    if ts:
        print(f"  log first ts {ts[0]}  last ts {ts[-1]}")
else:
    print("  NOT FOUND:", lg3)

# ---------------------------------------------------------------- F-55 -----
sec("F-55  cap/run/probe: no-persona vs holding vs folding personas (8 items each)")
for name in ("pc_position_holder_nopersona", "pc_position_holder", "pc_calibrated_holder",
             "pc_consensus_follower", "pc_caver"):
    f = f"{R}/cap/run/probe/{name}.jsonl"
    if not os.path.isfile(f):
        print(f"\n  -- {name}: NOT FOUND")
        continue
    recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    print(f"\n  -- {name}: {len(recs)} rows   keys={sorted(recs[0].keys())}")
    sp = recs[0].get("system_prompt")
    print(f"     system_prompt: {('null' if sp is None else repr(sp)[:120])}")
    for k, r in enumerate(recs):
        conv = r.get("conversation") or r.get("messages") or []
        q = next((m["content"] for m in conv if m.get("role") == "user"), "")
        t2 = conv[-1]["content"] if conv else r.get("turn2", "")
        print(f"     [{k}] Q: {' '.join(str(q).split())[:90]}")
        print(f"         turn2: {' '.join(str(t2).split())[:260]}")

# ---------------------------------------------------------------- F-64 -----
sec("F-64  cap generation / control / score counts and the grader keep rates")
for sub, exp in (("responses", "12 files / 1,200 rows"), ("control", "12 files / 240 rows"),
                 ("scores", "12 score files")):
    d = f"{R}/run-cap/{sub}"
    if os.path.isdir(d):
        fs = sorted(x for x in os.listdir(d) if x.endswith((".jsonl", ".json")))
        tot = 0
        for x in fs:
            p = f"{d}/{x}"
            tot += nlines(p) if x.endswith(".jsonl") else 0
        print(f"  {sub:10} files={len(fs):>3}  jsonl rows={tot:>5}   (ledger: {exp})")
        print(f"             names: {fs}")
    else:
        print(f"  {sub}: NOT FOUND ({d})")

sd = f"{R}/run-cap/scores"
if os.path.isdir(sd):
    per = {}
    dist = collections.Counter()
    for x in sorted(os.listdir(sd)):
        p = f"{sd}/{x}"
        if x.endswith(".jsonl"):
            recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        elif x.endswith(".json"):
            j = json.load(open(p, encoding="utf-8"))
            recs = j if isinstance(j, list) else j.get("rows") or j.get("scores") or []
        else:
            continue
        role = os.path.splitext(x)[0]
        sc = []
        for r in recs:
            v = r.get("score", r.get("judge_score", r.get("grade")))
            sc.append(v)
            dist[v] += 1
        per[role] = sc
    print(f"\n  score distribution over {sum(dist.values())} rollouts:")
    for k in sorted(dist, key=lambda z: (z is None, z)):
        print(f"    score {k}: {dist[k]:>5}  {dist[k]/sum(dist.values()):.4%}")
    keep = sum(v for k, v in dist.items() if k in (2, 3))
    print(f"    keep 2+3 = {keep}/{sum(dist.values())} = {keep/sum(dist.values()):.4%}")
    print("\n  per-role drop (score 0 or 1):")
    for role, sc in sorted(per.items()):
        dr = sum(1 for v in sc if v in (0, 1))
        print(f"    {role:30} {dr:>3}/{len(sc):<4} = {dr/len(sc):.1%}")
    # group split
    for gf in (f"{R}/cap/personas/cap_role_groups.json", f"{R}/cap/cap_role_groups.json",
               f"{R}/run-cap/cap_role_groups.json"):
        if os.path.isfile(gf):
            g = json.load(open(gf, encoding="utf-8"))
            print(f"\n  groups from {gf}: { {k: len(v) for k, v in g.items()} if isinstance(g, dict) else g}")
            if isinstance(g, dict):
                for grp, roles in g.items():
                    tot = kept = 0
                    for role in roles:
                        sc = per.get(role)
                        if sc is None:
                            print(f"     !! role {role} has no score file")
                            continue
                        tot += len(sc); kept += sum(1 for v in sc if v in (2, 3))
                    if tot:
                        print(f"    {grp:22} {len(roles)} roles  kept {kept:>4}/{tot:<5} = {kept/tot:.4%}")
            break
    else:
        print("\n  cap_role_groups.json NOT FOUND in the tried locations")

# ------------------------------------------------------- F-104 / F-22 ------
sec("F-104  sarc_pilot status files, before and after the deployment")
for f in ("logs/sarc_pilot.status.2026-09-21-buggy-verdict", "logs/sarc_pilot.status.pre-e2e-2026-09-22",
          "logs/sarc_pilot.status", "logs/cap_controls.status"):
    p = f"{R}/{f}"
    print(f"\n  -- {p}  {'EXISTS' if os.path.exists(p) else 'MISSING'}")
    if os.path.exists(p):
        print(f"     bytes={os.path.getsize(p)}")
        for l in readtext(p).splitlines():
            print("     |", l[:200])

sec("F-22  the truncated NNsight exception in the 8B generate log, and the post-fix 3B log")
for f, pat in ((f"{R}/logs/scale_8b_generate.log", r"session attempt|batch attempt"),
               (f"{R}/logs/scale_3b_generate.log", r"session attempt|batch attempt|OOM|CUDA|memory")):
    if not os.path.isfile(f):
        print(f"\n  {f}: NOT FOUND"); continue
    Lx = readtext(f).splitlines()
    print(f"\n  -- {f}: {len(Lx)} lines")
    hit = [(i, l) for i, l in enumerate(Lx, 1) if re.search(pat, l)]
    print(f"     {len(hit)} matching lines; first 3 with 6 lines of context each:")
    for i, l in hit[:3]:
        for j in range(i - 1, min(len(Lx), i + 5)):
            print(f"     {j+1:>6}: {Lx[j][:180]}")
        print("     ---")
print("\nDONE W5B_labA")
