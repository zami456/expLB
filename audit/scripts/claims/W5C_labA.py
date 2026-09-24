"""W5C-ledger batch, lab-side pass A (READ ONLY).

Covers F-14, F-24, F-41, F-48, F-56, F-71, F-84.
Run on the lab PC via audit/scripts/labpc_run.py + p1_labpc_runpy.ps1.
Reads only under C:/Users/user4/.  Writes nothing.
"""
import collections, glob, json, os, re, statistics, sys

R = "C:/Users/user4/expLB"


def size(p):
    try:
        return os.path.getsize(p)
    except OSError:
        return None


def mtime(p):
    import datetime
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return "-"


def readtext(p, limit=None):
    """Decode a log that may be UTF-16 (PowerShell *> redirect)."""
    with open(p, "rb") as f:
        b = f.read()
    if b[:2] in (b"\xff\xfe", b"\xfe\xff"):
        t = b.decode("utf-16")
    else:
        t = b.decode("utf-8", "replace")
    lines = t.splitlines()
    return lines[:limit] if limit else lines


# ------------------------------------------------------------------ F-14
print("=" * 78)
print("### F-14  scp truncation at 204,800 bytes")
print("=" * 78)
sj = f"{R}/kd/run/knowledge/scores.jsonl"
print(f"lab  {sj}: {size(sj)} B  mtime {mtime(sj)}")
n = ok = 0
ids = set()
with open(sj, encoding="utf-8") as f:
    for line in f:
        n += 1
        try:
            r = json.loads(line)
            ok += 1
            ids.add(r.get("item_id") or r.get("id"))
        except Exception:
            pass
print(f"lab  scores.jsonl: {n} lines, {ok} parse OK, {len(ids)} distinct ids")
print(f"lab  first 204800 bytes would end mid-line? -> lines fully inside 204800 B:")
with open(sj, "rb") as f:
    head = f.read(204800)
print(f"     {head.count(chr(10).encode())} newlines in the first 204,800 bytes"
      f" (local truncated copy holds 1,095 parseable + 1 cut)")

print("\n-- the 5 files F-14 says were truncated (real sizes on the lab now) --")
for rel in ("summary.json", "matrices/last_L40_s42.json", "matrices/last_L44_s42.json",
            "matrices/mean_L40_s42.json", "matrices/mean_L44_s42.json"):
    p = f"{R}/run-crossgen/kd11/{rel}"
    s = size(p)
    print(f"   {rel:<34} {s}  mtime {mtime(p)}"
          + (f"   204800/{s} = {204800 / s:.1%}" if s else ""))
big = [(size(p), p) for p in glob.glob(f"{R}/run-crossgen/kd11/**/*", recursive=True)
       if os.path.isfile(p) and (size(p) or 0) > 204800]
print(f"   files >204,800 B under run-crossgen/kd11 today: {len(big)}")
allf = [p for p in glob.glob(f"{R}/run-crossgen/kd11/**/*", recursive=True) if os.path.isfile(p)]
print(f"   total files under run-crossgen/kd11 today: {len(allf)}  (F-14 quotes 271 on 09-19)")
print(f"   sum of the 5 files = {sum(size(f'{R}/run-crossgen/kd11/{r}') or 0 for r in ('summary.json','matrices/last_L40_s42.json','matrices/last_L44_s42.json','matrices/mean_L40_s42.json','matrices/mean_L44_s42.json'))/1048576:.3f} MiB"
      f"   (F-14's rclone pull moved 2.660 MiB)")

# ------------------------------------------------------------------ F-24
print()
print("=" * 78)
print("### F-24  missing remote=True downloaded the 70B")
print("=" * 78)
hub = "C:/Users/user4/.cache/huggingface/hub"
for d in sorted(glob.glob(hub + "/models--*")):
    tot = 0
    nf = 0
    for dp, _, fns in os.walk(d):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                if not os.path.islink(p):
                    tot += os.path.getsize(p)
                nf += 1
            except OSError:
                pass
    print(f"   {os.path.basename(d):<48} {nf:>4} files  {tot/1024**3:8.2f} GB")
b = hub + "/models--meta-llama--Llama-3.3-70B-Instruct/blobs"
if os.path.isdir(b):
    fs = os.listdir(b)
    print(f"   blobs/: {len(fs)} entries")
    for fn in sorted(fs)[:12]:
        p = os.path.join(b, fn)
        print(f"      {fn[:44]:<46} {size(p)} B  {mtime(p)}")
print("   smoke_multiturn.py remote= call sites:")
for p in (f"{R}/cap/scripts/smoke_multiturn.py", f"{R}/cap/smoke_multiturn.py"):
    if os.path.exists(p):
        ls = readtext(p)
        hits = [f"{i}:{l.strip()[:90]}" for i, l in enumerate(ls, 1)
                if "generate(" in l or "remote=" in l or "assert" in l]
        print(f"      {p}  ({len(ls)} lines, mtime {mtime(p)})")
        for h in hits[:8]:
            print("        " + h)
print("   other call sites F-24 names:")
for p, ln in ((f"{R}/src/assistant_axis_ndif/1_generate_ndif.py", 178),
              (f"{R}/src/assistant_axis_ndif/3_judge_ndif.py", 181),
              (f"{R}/src/assistant_axis_ndif/2_activations_ndif.py", 330),
              (f"{R}/src/assistant_axis_ndif/_api_probe_07.py", 72)):
    if os.path.exists(p):
        ls = readtext(p)
        t = ls[ln - 1].strip()[:100] if len(ls) >= ln else "(file shorter)"
        print(f"      {os.path.basename(p)}:{ln}  {t}")
    else:
        print(f"      {os.path.basename(p)}  NOT FOUND")
sess = len(re.findall(r"session\(\s*remote\s*=\s*True",
                      "\n".join(sum([readtext(p) for p in glob.glob(f"{R}/src/assistant_axis_ndif/*.py")
                                     + glob.glob(f"{R}/cap/scripts/*.py") + glob.glob(f"{R}/kd/scripts/*.py")], []))))
direct = len(re.findall(r"remote\s*=\s*True",
                        "\n".join(sum([readtext(p) for p in glob.glob(f"{R}/src/assistant_axis_ndif/*.py")
                                       + glob.glob(f"{R}/cap/scripts/*.py") + glob.glob(f"{R}/kd/scripts/*.py")], []))))
print(f"   grep now: session(remote=True) occurrences = {sess}; all remote=True occurrences = {direct}")

# ------------------------------------------------------------------ F-41
print()
print("=" * 78)
print("### F-41  8B coords OOM on HP-KR")
print("=" * 78)
lg = f"{R}/logs/scale_coords_8b.log"
ls = readtext(lg)
print(f"   {lg}: {len(ls)} lines, mtime {mtime(lg)}")
for i, l in enumerate(ls, 1):
    if "902.00" in l or "OutOfMemory" in l or "attempt 4/4" in l:
        print(f"     L{i}: {l.strip()[:200]}")
print("   -- resume lines --")
for i, l in enumerate(ls, 1):
    if "already complete" in l or "to do" in l and "already on disk" in l:
        print(f"     L{i}: {l.strip()[:160]}")
st = f"{R}/logs/scale_coords.status"
sl = readtext(st)
print(f"   {st}: {len(sl)} lines, mtime {mtime(st)}")
for l in sl:
    if "attn" in l or "START" in l or "exit" in l:
        print("     " + l.strip()[:170])
print("   -- OOM occurrences by dataset/time in the 8B log --")
cur = None
for i, l in enumerate(ls, 1):
    m = re.search(r"(instructed-deception|harm-pressure-choice|harm-pressure-knowledge-report|roleplaying|ai_liar|convincing-game)", l)
    if m:
        cur = m.group(1)
    if "OutOfMemory" in l:
        print(f"     L{i} ds~{cur}: {l.strip()[:150]}")

# ------------------------------------------------------------------ F-48
print()
print("=" * 78)
print("### F-48  8B evaluation arm complete")
print("=" * 78)
for l in sl:
    if "8b" in l.lower():
        print("     status: " + l.strip()[:170])
base = f"{R}/run-scale/8b/coords"
if os.path.isdir(base):
    for d in sorted(os.listdir(base)):
        p = os.path.join(base, d)
        if not os.path.isdir(p):
            continue
        fs = os.listdir(p)
        chunks = [f for f in fs if f.startswith("chunk")]
        rows = None
        rp = os.path.join(p, "rows.jsonl")
        if os.path.exists(rp):
            rows = sum(1 for _ in open(rp, encoding="utf-8"))
        print(f"     {d:<38} chunks={len(chunks):>4} rows.jsonl={rows}  other={sorted(set(fs)-set(chunks))[:4]}  mtime {mtime(p)}")
else:
    print("     NOT FOUND", base)

# ------------------------------------------------------------------ F-56
print()
print("=" * 78)
print("### F-56  --sample is proportional, not balancing")
print("=" * 78)
cl = f"{R}/logs/crossgen_scale_8b.log"
if os.path.exists(cl):
    cls = readtext(cl)
    print(f"   {cl}: {len(cls)} lines, mtime {mtime(cl)}")
    for i, l in enumerate(cls[:60], 1):
        print(f"     L{i}: {l.strip()[:160]}")
print("   -- recomputed from run-scale/8b/coords/*/rows.jsonl (label field) --")
for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
    rp = os.path.join(base, d, "rows.jsonl")
    if not os.path.exists(rp):
        continue
    lab = collections.Counter()
    tot = 0
    for line in open(rp, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        tot += 1
        lab[r.get("label", r.get("is_deceptive", "?"))] += 1
    print(f"     {d:<38} n={tot:<6} labels={dict(lab)}")

# ------------------------------------------------------------------ F-71
print()
print("=" * 78)
print("### F-71  all three stores carry the matched slice identically")
print("=" * 78)
hs = f"{R}/logs/hpkr_topup.status"
if os.path.exists(hs):
    for l in readtext(hs):
        print("     status: " + l.strip()[:170])
ID_SETS = {}
for store in ("passA", "passB", "passC"):
    d = f"{R}/run-coords/{store}/harm-pressure-knowledge-report"
    rp = f"{d}/rows.jsonl"
    if not os.path.exists(rp):
        print(f"     {store}: rows.jsonl NOT FOUND at {rp}")
        continue
    n = 0
    matched = []
    lays = set()
    per_token = set()
    for line in open(rp, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        n += 1
        if r.get("kr_matched") or r.get("matched") or r.get("in_kr_matched"):
            matched.append(r)
    # meta
    meta = {}
    for mf in ("meta.json", "store.json", "config.json"):
        p = os.path.join(d, mf)
        if os.path.exists(p):
            meta = json.load(open(p, encoding="utf-8"))
            break
    chunks = [f for f in os.listdir(d) if f.startswith("chunk")]
    print(f"     {store}: rows={n} chunks={len(chunks)} matched_flagged={len(matched)} "
          f"meta_keys={sorted(meta)[:8]} mtime {mtime(rp)}")
    if matched:
        lab = collections.Counter(str(r.get("label", r.get("is_deceptive")))
                                  for r in matched)
        print(f"        matched label counts: {dict(lab)}")
    # the matched view, if it is a separate dataset dir
    vd = f"{R}/run-coords/{store}/harm-pressure-kr-matched"
    if os.path.isdir(vd):
        vrp = f"{vd}/rows.jsonl"
        vn = 0
        vlab = collections.Counter()
        rid = []
        for line in open(vrp, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            vn += 1
            vlab[str(r.get("label", r.get("is_deceptive")))] += 1
            rid.append(r.get("row_id", r.get("id")))
        vch = [f for f in os.listdir(vd) if f.startswith("chunk")]
        print(f"        kr-matched view: rows={vn} chunks={len(vch)} labels={dict(vlab)} "
              f"mtime {mtime(vrp)}  first_ids={rid[:3]}")
        ID_SETS[store] = set(map(str, rid))

if len(ID_SETS) > 1:
    ks = sorted(ID_SETS)
    print("   -- row_id set comparison across stores (kr-matched view) --")
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b_ = ID_SETS[ks[i]], ID_SETS[ks[j]]
            print(f"      {ks[i]} vs {ks[j]}: |A|={len(a)} |B|={len(b_)} "
                  f"identical={a == b_} A-B={len(a - b_)} B-A={len(b_ - a)}")

# ------------------------------------------------------------------ F-84
print()
print("=" * 78)
print("### F-84  cluster-permutation test, p = 0.38")
print("=" * 78)
cp = f"{R}/run-premise/cluster_perm_kd_matched.json"
if os.path.exists(cp):
    d = json.load(open(cp, encoding="utf-8"))
    print(f"   {cp}  mtime {mtime(cp)}  size {size(cp)}")
    print("   top-level keys:", sorted(d))
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            print(f"     {k} = {v}")
        elif isinstance(v, list):
            print(f"     {k}: list[{len(v)}] head={v[:6]}")
        elif isinstance(v, dict):
            print(f"     {k}: dict keys={sorted(v)[:10]}")
    nm = d.get("null_max_mass") or d.get("null") or d.get("null_masses")
    obs = d.get("observed_max_mass", d.get("observed"))
    if isinstance(nm, list) and nm and obs is not None:
        nm2 = [float(x) for x in nm]
        p = (sum(1 for x in nm2 if x >= obs) + 1) / (len(nm2) + 1)
        p_noplus = sum(1 for x in nm2 if x >= obs) / len(nm2)
        nm2s = sorted(nm2)
        q95 = nm2s[int(0.95 * len(nm2s))]
        print(f"   RECOMPUTED: B={len(nm2)} obs={obs:.6f} null_mean={statistics.fmean(nm2):.6f} "
              f"null_p95={q95:.6f}  p(>=obs, +1)={p:.4f}  p(>=obs, plain)={p_noplus:.4f}")
    om = d.get("observed_margins") or d.get("margins")
    if isinstance(om, list):
        print(f"   observed_margins ({len(om)}): {[round(float(x),4) for x in om]}")
        pos = [i for i, x in enumerate(om) if float(x) > 0]
        print(f"   positive layers: {len(pos)} of {len(om)} at idx {pos}")
else:
    print("   NOT FOUND", cp)
print("   -- premise_extras log lines --")
pl = f"{R}/logs/premise_extras.log"
if os.path.exists(pl):
    pls = readtext(pl)
    for i, l in enumerate(pls, 1):
        if "cluster" in l.lower() or "null mean" in l.lower() or "p = " in l:
            print(f"     L{i}: {l.strip()[:170]}")
ps = f"{R}/logs/premise_extras.status"
if os.path.exists(ps):
    print(f"   status ({mtime(ps)}):")
    for l in readtext(ps):
        print("     " + l.strip()[:170])
print("\nDONE W5C labA")
