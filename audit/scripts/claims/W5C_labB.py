"""W5C-ledger batch, lab-side pass B (READ ONLY).

Covers F-52, F-61, F-68, plus follow-ups for F-41, F-71, F-84.
Run on the lab PC via audit/scripts/labpc_run.py + p1_labpc_runpy.ps1.
"""
import collections, glob, json, os, re, statistics

R = "C:/Users/user4/expLB"


def mtime(p):
    import datetime
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return "-"


def readtext(p):
    with open(p, "rb") as f:
        b = f.read()
    t = b.decode("utf-16") if b[:2] in (b"\xff\xfe", b"\xfe\xff") else b.decode("utf-8", "replace")
    return t.splitlines()


# ------------------------------------------------------------------ F-52
print("=" * 78)
print("### F-52  contrast holds on true answers, collapses on false ones")
print("=" * 78)
for p in sorted(glob.glob(f"{R}/cap/run/smoke_cells/*.jsonl")):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    print(f"\n--- {os.path.basename(p)}  n={len(rows)}  mtime {mtime(p)}")
    if rows:
        print("    keys:", sorted(rows[0]))
    for i, r in enumerate(rows):
        meta = {k: r.get(k) for k in ("framing", "kind", "item_id", "qid", "turn1_is_correct",
                                      "correct", "variant", "persona", "role", "control")
                if k in r}
        t2 = r.get("turn2") or r.get("response") or r.get("text") or ""
        if isinstance(t2, list):
            t2 = " ".join(map(str, t2))
        t1 = r.get("turn1") or r.get("forced") or ""
        if isinstance(t1, list):
            t1 = " ".join(map(str, t1))
        print(f"  [{i}] {meta}")
        print(f"      turn1: {str(t1)[:160].replace(chr(10),' / ')}")
        print(f"      turn2: {str(t2)[:420].replace(chr(10),' / ')}")
print("\n-- spelunking-word count in pc_caver main-condition file --")
for p in sorted(glob.glob(f"{R}/cap/run/smoke_cells/*caver*.jsonl")):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    hits = 0
    for r in rows:
        t = json.dumps(r).lower()
        if any(w in t for w in ("cave", "spelunk", "pickaxe", "helmet", "underground", "mine")):
            hits += 1
    print(f"   {os.path.basename(p)}: {hits}/{len(rows)} rows mention caving words")

# ------------------------------------------------------------------ F-61
print()
print("=" * 78)
print("### F-61  36-vs-26 bank comparison confounded three ways")
print("=" * 78)
for run in ("run-paper16", "run-kd"):
    d = f"{R}/{run}/responses"
    if not os.path.isdir(d):
        print(f"   {d}: NOT FOUND"); continue
    fs = sorted(glob.glob(d + "/*"))
    jl = [f for f in fs if f.endswith(".jsonl")]
    other = [os.path.basename(f) for f in fs if not f.endswith(".jsonl")]
    tot = 0
    per = {}
    qs = collections.Counter()
    vs = collections.Counter()
    for f in jl:
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        per[os.path.basename(f)] = len(rows)
        tot += len(rows)
        for r in rows:
            q = r.get("question_id", r.get("qid", r.get("question")))
            v = r.get("variant", r.get("variant_id"))
            qs[str(q)[:70]] += 1
            vs[str(v)] += 1
    print(f"   {run}: {len(jl)} role .jsonl files, {tot} rollouts, "
          f"{tot/max(len(jl),1):.1f} per role  (non-jsonl: {other})")
    cnt = collections.Counter(per.values())
    print(f"      rollouts-per-file distribution: {dict(sorted(cnt.items()))}")
    odd = {k: v for k, v in per.items() if v != collections.Counter(per.values()).most_common(1)[0][0]}
    print(f"      files differing from the mode: {odd}")
    print(f"      distinct questions: {len(qs)}   distinct variants: {len(vs)} -> {sorted(vs)[:8]}")
    print(f"      first role file: {os.path.basename(jl[0])} n={per[os.path.basename(jl[0])]}")

print("\n-- question-set overlap (paper16 vs kd) --")
def qset(run):
    s = set()
    for f in sorted(glob.glob(f"{R}/{run}/responses/*.jsonl")):
        for l in open(f, encoding="utf-8"):
            if not l.strip():
                continue
            r = json.loads(l)
            q = r.get("question") or r.get("prompt") or r.get("question_text")
            if q:
                s.add(" ".join(str(q).split())[:200])
    return s
p16, kd = qset("run-paper16"), qset("run-kd")
print(f"   paper16 distinct question texts: {len(p16)}")
print(f"   kd      distinct question texts: {len(kd)}")
print(f"   intersection: {len(p16 & kd)}   paper16-only: {len(p16 - kd)}   kd-only: {len(kd - p16)}")
print("   paper16 questions (first 60 chars each):")
for q in sorted(p16):
    print(f"      [{'BOTH' if q in kd else 'P16 '}] {q[:90]}")

print("\n-- pc:unified minus raw at L40, paper probe, both banks --")
def offdiag(path, feat, probe="paper", drop=None):
    d = json.load(open(path, encoding="utf-8"))
    ds = d["datasets"]
    keep = [i for i, n in enumerate(ds) if n != drop]
    for r in d["results"]:
        if r["feature"] == feat and r["probe"] == probe:
            v = [r["auroc"][i][j] for i in keep for j in keep
                 if i != j and r["auroc"][i][j] is not None]
            return statistics.fmean(v), len(v), ds
    return None, 0, ds

for label, path in (("paper16 (kd11 root)", f"{R}/run-crossgen/kd11/matrices/mean_L40_s42.json"),
                    ("kd bank", f"{R}/run-crossgen/kd11/banks/kd/matrices/mean_L40_s42.json")):
    if not os.path.exists(path):
        print(f"   {label}: MISSING {path}"); continue
    d = json.load(open(path, encoding="utf-8"))
    print(f"   {label}: {path}  mtime {mtime(path)}  bank={d.get('bank')} name={d.get('bank_name')} "
          f"k={d.get('k')} datasets={len(d['datasets'])}")
    for drop in (None, "sycophancy_mt"):
        u, nu, _ = offdiag(path, "pc:unified", drop=drop)
        r0, nr, _ = offdiag(path, "raw", drop=drop)
        dp, ndp, _ = offdiag(path, "datapca", drop=drop)
        tag = "all 11" if drop is None else "minus sycophancy_mt"
        if u is None or r0 is None:
            feats = sorted({x["feature"] for x in d["results"]})
            print(f"      features present: {feats}"); break
        print(f"      [{tag}] raw={r0:.4f}({nr})  pc:unified={u:.4f}({nu})  "
              f"pc:unified-raw={u-r0:+.4f}   datapca={dp:.4f}  pc:unified-datapca={u-dp:+.4f}")

# ------------------------------------------------------------------ F-68
print()
print("=" * 78)
print("### F-68  A vs B: holding personas improve the contrast by ~0.045")
print("=" * 78)
TARGET = "sycophancy_mt"
for bank in ("cap_s23", "cap_all"):
    for L in (20, 40, 44):
        p = f"{R}/run-crossgen/kd11/banks/{bank}/matrices/mean_L{L}_s42.json"
        if not os.path.exists(p):
            print(f"   {bank} L{L}: MISSING"); continue
        d = json.load(open(p, encoding="utf-8"))
        j = d["datasets"].index(TARGET)
        print(f"   {bank} L{L}  mtime {mtime(p)} bank={d.get('bank')} name={d.get('bank_name')} k={d.get('k')}")
        for z in d.get("zeroshot", []):
            a = z["auroc"][j]
            if a is None:
                continue
            lo = (z.get("ci95_lo") or [None] * 20)[j]
            hi = (z.get("ci95_hi") or [None] * 20)[j]
            ci = f" [{lo:.4f}, {hi:.4f}]" if lo is not None else ""
            print(f"      {z['direction']:<34} {a:.4f}{ci}")
print("\n   -- A minus B per layer/filter (contrast vs contrast:cap_vs_default) --")
for bank in ("cap_s23", "cap_all"):
    for L in (20, 40, 44):
        p = f"{R}/run-crossgen/kd11/banks/{bank}/matrices/mean_L{L}_s42.json"
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        j = d["datasets"].index(TARGET)
        z = {x["direction"]: x["auroc"][j] for x in d.get("zeroshot", [])}
        A = z.get("contrast")
        B = z.get("contrast:cap_vs_default")
        if A is not None and B is not None:
            print(f"      {bank} L{L}: A={A:.4f} B={B:.4f} A-B={A-B:+.4f}")

# ------------------------------------------------------------------ follow-ups
print()
print("=" * 78)
print("### follow-ups")
print("=" * 78)
print("-- F-84: numpy percentile of the null, and 30c control type --")
try:
    import numpy as np
    d = json.load(open(f"{R}/run-premise/cluster_perm_kd_matched.json", encoding="utf-8"))
    nm = np.array(d["null_max_mass"], dtype=float)
    print(f"   np.percentile(null,95) = {np.percentile(nm,95):.4f}   "
          f"mean={nm.mean():.4f}  max={nm.max():.4f}  n>=obs={(nm>=d['observed_max_mass']).sum()}")
    print(f"   layers list = {d['layers']}")
except Exception as e:
    print("   numpy step failed:", e)
sc = f"{R}/src/assistant_axis_ndif/30c_cluster_perm.py"
if os.path.exists(sc):
    ls = readtext(sc)
    print(f"   {sc}: {len(ls)} lines, mtime {mtime(sc)}")
    for i, l in enumerate(ls, 1):
        if re.search(r"random|rand_|dataset_pca|datapca|k\s*=|B\s*=|permut|store|pass[ABC]|pooling", l, re.I):
            print(f"     {i}: {l.strip()[:130]}")
print("\n-- F-84: premise_extras.log dated context around the two runs --")
pl = readtext(f"{R}/logs/premise_extras.log")
for lo, hi in ((520, 550), (975, 1010)):
    for i in range(lo, min(hi, len(pl))):
        t = pl[i].strip()
        if t:
            print(f"     L{i+1}: {t[:150]}")
    print("     ---")
print("\n-- F-71: layer counts per store --")
for store in ("passA", "passB", "passC"):
    d = f"{R}/run-coords/{store}/harm-pressure-kr-matched"
    if not os.path.isdir(d):
        print(f"   {store}: no kr-matched dir"); continue
    ch = sorted(f for f in os.listdir(d) if f.startswith("chunk"))
    print(f"   {store}: {len(ch)} chunks, e.g. {ch[:2]}")
    try:
        import torch
        t = torch.load(os.path.join(d, ch[0]), map_location="cpu", weights_only=False)
        def describe(o, depth=0):
            if isinstance(o, dict):
                return {k: describe(v, depth+1) for k, v in list(o.items())[:8]} if depth < 2 else "dict"
            if hasattr(o, "shape"):
                return f"tensor{tuple(o.shape)}"
            if isinstance(o, list):
                return f"list[{len(o)}]"
            return type(o).__name__
        print(f"      chunk structure: {describe(t)}")
    except Exception as e:
        print(f"      torch load failed: {e}")
print("\n-- F-41: chunk-count line and the 3B budget --")
ls = readtext(f"{R}/logs/scale_coords_8b.log")
for i, l in enumerate(ls, 1):
    if re.search(r"\b119\b.*chunk|chunk.*\b119\b", l):
        print(f"     8b L{i}: {l.strip()[:160]}")
p3 = f"{R}/logs/scale_coords_3b.log"
if os.path.exists(p3):
    l3 = readtext(p3)
    print(f"   scale_coords_3b.log: {len(l3)} lines, mtime {mtime(p3)}; first 3 lines:")
    for l in l3[:3]:
        print("     " + l.strip()[:150])
sp = f"{R}/scale/scripts/run_scale_coords.ps1"
if os.path.exists(sp):
    print(f"   {sp} mtime {mtime(sp)}: attn/sess settings")
    for i, l in enumerate(readtext(sp), 1):
        if re.search(r"attn|sess|max_length|0\.75|0\.5|0\.42|0\.55", l):
            print(f"     {i}: {l.strip()[:140]}")
print("\nDONE W5C labB")
