"""W5C-ledger batch, lab-side pass C (READ ONLY).

F-52 conversation text; F-61 per-role question/variant structure + hunt for the '+0.006'
matrix (109 pairs, raw 0.6800); F-68 direction definitions; F-41 chunk count.
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
print("### F-52  the 24 smoke_cells rollouts, turn-2 text")
print("=" * 78)
for p in sorted(glob.glob(f"{R}/cap/run/smoke_cells/*.jsonl")):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    print(f"\n--- {os.path.basename(p)}  n={len(rows)}")
    print(f"    system_prompt[0][:220]: {str(rows[0].get('system_prompt'))[:220]}")
    for i, r in enumerate(rows):
        conv = r.get("conversation")
        turns = []
        if isinstance(conv, list):
            for m in conv:
                if isinstance(m, dict):
                    turns.append(f"{m.get('role')}: {str(m.get('content'))[:300]}")
                else:
                    turns.append(str(m)[:300])
        print(f"  [{i}] framing={r.get('framing')} kind={r.get('kind')} "
              f"t1_correct={r.get('turn1_is_correct')} label={r.get('label')} "
              f"q={str(r.get('question'))[:70]!r} correct={r.get('correct_answer')!r} turn1={r.get('turn1')!r}")
        for t in turns:
            print(f"        {t}")

# ------------------------------------------------------------------ F-61
print()
print("=" * 78)
print("### F-61  per-role question / variant structure")
print("=" * 78)
for run in ("run-paper16", "run-kd"):
    print(f"\n-- {run} --")
    fs = sorted(glob.glob(f"{R}/{run}/responses/*.jsonl"))
    summary = collections.Counter()
    for f in fs:
        rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        qs = {" ".join(str(r.get("question", "")).split())[:120] for r in rows}
        pi = {r.get("prompt_index") for r in rows}
        qi = {r.get("question_index") for r in rows}
        key = (len(rows), len(qs), len(pi), len(qi))
        summary[key] += 1
        if summary[key] <= 2:
            print(f"   {os.path.basename(f):<34} n={len(rows)} distinct_q={len(qs)} "
                  f"prompt_index={sorted(x for x in pi if x is not None)} "
                  f"question_index_n={len(qi)}  keys={sorted(rows[0])}")
    print(f"   (n, distinct_q, n_prompt_index, n_question_index) -> #roles : {dict(summary)}")
    # which roles use which question set
    if run == "run-paper16":
        groups = collections.defaultdict(list)
        for f in fs:
            rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
            qs = frozenset(" ".join(str(r.get("question", "")).split())[:120] for r in rows)
            groups[qs].append(os.path.basename(f))
        for k, v in groups.items():
            print(f"   question-set of {len(k)} questions used by {len(v)} roles: {sorted(v)[:14]}")

print("\n-- hunt for the matrix behind '+0.006' (raw 0.6800 over 109 pairs, paper probe) --")
def scan(pattern, label):
    out = []
    for p in glob.glob(pattern, recursive=True):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(d, dict) or "datasets" not in d or "results" not in d:
            continue
        idx = {(r["feature"], r["probe"]): r["auroc"] for r in d["results"]}
        blk = idx.get(("raw", "paper"))
        if blk is None:
            continue
        vals = [v for i, row in enumerate(blk) for j, v in enumerate(row) if i != j and v is not None]
        if not vals:
            continue
        m = statistics.fmean(vals)
        if len(vals) == 109 or abs(m - 0.68) < 0.001:
            u = idx.get(("pc:unified", "paper"))
            uv = [v for i, row in enumerate(u) for j, v in enumerate(row)
                  if i != j and v is not None] if u else []
            out.append((p, d.get("layer"), d.get("pooling"), d.get("seed"), len(d["datasets"]),
                        len(vals), round(m, 4),
                        round(statistics.fmean(uv), 4) if uv else None,
                        round(statistics.fmean(uv) - m, 4) if uv else None))
    print(f"   {label}: {len(out)} hit(s)")
    for o in out:
        print("     ", o)

scan(f"{R}/run-crossgen/**/matrices/*.json", "run-crossgen (all stores)")

# ------------------------------------------------------------------ F-68
print()
print("=" * 78)
print("### F-68  what the zero-shot directions ARE")
print("=" * 78)
for f in ("cap/scripts/cap_zeroshot.py", "src/assistant_axis_ndif/27_crossgen.py",
          "src/assistant_axis_ndif/6_axes.py"):
    p = f"{R}/{f}"
    if not os.path.exists(p):
        print(f"   {f}: NOT FOUND"); continue
    ls = readtext(p)
    print(f"\n   -- {f} ({len(ls)} lines, mtime {mtime(p)}) --")
    for i, l in enumerate(ls, 1):
        if re.search(r"contrast:|PC1:|zeroshot|zero_shot|capitulation|position_holding|"
                     r"assistant_anchor|declared", l):
            print(f"     {i}: {l.strip()[:135]}")
print("\n   -- cap bank personas / groups --")
for p in glob.glob(f"{R}/cap/personas/*.json") + glob.glob(f"{R}/cap/personas/*.jsonl"):
    try:
        d = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    print(f"     {os.path.basename(p)}: keys={sorted(d)[:10] if isinstance(d, dict) else type(d).__name__}")
    if isinstance(d, dict):
        for k in ("groups", "contrasts", "declared_contrasts"):
            if k in d:
                print(f"       {k} = {json.dumps(d[k])[:600]}")
try:
    import torch
    for b in ("run-cap/axes_s23.pt", "run-cap/axes_all.pt"):
        p = f"{R}/{b}"
        if os.path.exists(p):
            t = torch.load(p, map_location="cpu", weights_only=False)
            print(f"     {b}: type={type(t).__name__} keys={sorted(t)[:14] if isinstance(t, dict) else '-'}"
                  f"  mtime {mtime(p)}")
            if isinstance(t, dict):
                for k in ("meta", "counts", "groups", "axes", "contrasts"):
                    if k in t and not hasattr(t[k], "shape"):
                        print(f"        {k} = {json.dumps(t[k], default=str)[:700]}")
except Exception as e:
    print("     torch step failed:", e)

# ------------------------------------------------------------------ F-41
print()
print("=" * 78)
print("### F-41  '119 chunks survived' and the disk-free readings")
print("=" * 78)
ls = readtext(f"{R}/logs/scale_coords_8b.log")
for i, l in enumerate(ls, 1):
    if 4870 <= i <= 4935 or re.search(r"chunks? (survive|on disk|present|found)|resum", l, re.I):
        print(f"   L{i}: {l.strip()[:165]}")
print("\nDONE W5C labC")
