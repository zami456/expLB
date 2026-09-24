"""Phase 0: read-only inventory of Claude-written docs OUTSIDE the repo that bear on this project:
~/.claude/plans/*.md (plan files; ONBOARDING.md calls harmonic-cuddling-kay.md 'the single most important file'),
the older project memory dir for a previous clone, and the session transcripts (counted, not read).
Usage: PYTHONIOENCODING=utf-8 python audit/scripts/p0_docs_external.py"""
import os, re, glob, datetime
H = os.path.expanduser("~/.claude")
def stamp(p): return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M")
print("== plans")
for p in sorted(glob.glob(os.path.join(H, "plans", "*.md"))):
    t = open(p, encoding="utf-8", errors="replace").read()
    head = next((l for l in t.splitlines() if l.startswith("#")), "")[:110]
    f = sorted({int(x) for x in re.findall(r"\bF-(\d{1,3})\b", t)})
    q = sorted({int(x) for x in re.findall(r"\bQ(\d{1,2})\b", t)})
    mentions = [k for k in ("expLB", "persona", "NDIF", "paper16", "HP-KR", "Liars", "70B") if k in t]
    print(f"{os.path.basename(p)}\t{os.path.getsize(p)}B\t{t.count(chr(10))}L\tmtime {stamp(p)}\t"
          f"F:{('F-%02d..F-%02d' % (f[0], f[-1])) if f else '-'}\tQ:{','.join(map(str, q)) or '-'}\tmentions:{','.join(mentions)}\t{head}")
print("== other project memory dirs")
for d in glob.glob(os.path.join(H, "projects", "*", "memory")):
    if "STUDY-thesis-expLB" in d: continue
    for p in sorted(glob.glob(os.path.join(d, "*"))):
        print(f"{p}\t{os.path.getsize(p)}B\tmtime {stamp(p)}")
print("== session transcripts for this repo (not read)")
tr = glob.glob(os.path.join(H, "projects", "C--Users-SURAIYA-Documents-STUDY-thesis-expLB", "*.jsonl"))
if tr:
    ts = sorted(os.path.getmtime(p) for p in tr)
    print(f"{len(tr)} files, {sum(os.path.getsize(p) for p in tr)/1e6:.1f} MB, mtime "
          f"{datetime.datetime.fromtimestamp(ts[0]):%Y-%m-%d} .. {datetime.datetime.fromtimestamp(ts[-1]):%Y-%m-%d}")
