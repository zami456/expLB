"""Phase 0: per-document facts for the docs inventory (read-only; git read commands only).
Usage: PYTHONIOENCODING=utf-8 python audit/scripts/p0_docs_inventory.py > audit/phase0/docs_inventory_raw.tsv
       PYTHONIOENCODING=utf-8 python audit/scripts/p0_docs_inventory.py --authority   (prints authority/stale lines)
Candidate set = tracked files at HEAD outside submodules (mask/ pacchiardi/ soft-trigger/ assistant_axis/),
outside the nnsight library docs (docs/{concepts,usage,models,remote,patterns,errors,gotchas,reference,developing}),
with extension .md/.html/.tex/.xlsx or under docs/thesis, plus the draft PDF, plus untracked CLAUDE.local.md.
Per doc: tracked?, lines, bytes, first/last commit date (git log --follow), #commits, counts of numbering schemes."""
import os, re, subprocess, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
LIB = re.compile(r"^docs/(concepts|usage|models|remote|patterns|errors|gotchas|reference|developing)/")
SUB = re.compile(r"^(mask|pacchiardi|soft-trigger|assistant_axis|behavioral-self-awareness)(/|$)")
files = [f for f in git("ls-files").splitlines() if not SUB.match(f) and not LIB.match(f)]
cand = [f for f in files if re.search(r"\.(md|html|tex|xlsx)$", f, re.I) or f.startswith("docs/thesis/")
        or f.startswith("full paper draft")]
cand = [f for f in cand if not f.startswith("figures/") or f.endswith((".tex", ".html"))]
cand += [f for f in ("CLAUDE.local.md",) if os.path.exists(f) and f not in files]
SCHEMES = {
    "F-id": r"\bF-(\d{1,3})\b",
    "Q-id": r"\bQ(\d{1,2})[a-z]?\b",
    "Claim": r"\bClaim\s+(\d+[a-z]?)\b",
    "Result": r"\bResult\s+(\d+[a-z]?)\b",
    "Tier": r"\bTier\s+(\d+(?:\.\d+)?)\b",
    "Gate": r"\bGate\s+(\d+[a-z]?)\b",
    "Stage": r"\bStage\s+(\d+[a-z]?)\b",
    "Hyp": r"\b(H\d{1,2})\b",
    "D-id": r"\b(D-?\d{1,2})\b",
    "Fig": r"\bFig(?:ure)?\.?\s*(\d+)\b",
}
AUTH = re.compile(r"authoritative|source of truth|append-only|\bledger\b|supersed|canonical|single source|stale|obsolete|deprecated|replaced by|do not trust|out of date|outdated|DEPRECATED|historical", re.I)
def textof(f):
    if f.endswith((".xlsx", ".pdf")): return None
    try:
        return open(f, encoding="utf-8", errors="replace").read()
    except OSError:
        return None
if "--authority" in sys.argv:
    for f in cand:
        t = textof(f)
        if not t: continue
        for ln, line in enumerate(t.splitlines(), 1):
            if AUTH.search(line):
                print(f"{f}:{ln}: {line.strip()[:260]}")
    sys.exit()
print("\t".join(["path", "tracked", "lines", "bytes", "first_commit", "first_date", "last_commit", "last_date", "n_commits",
                 *SCHEMES.keys(), "F_range", "Q_distinct"]))
for f in cand:
    tracked = "y" if f in files else "n"
    size = os.path.getsize(f) if os.path.exists(f) else -1
    t = textof(f)
    nl = t.count("\n") if t is not None else ""
    log = git("log", "--follow", "--format=%h %ad", "--date=format:%Y-%m-%d %H:%M", "--", f).splitlines() if tracked == "y" else []
    first = log[-1].split(" ", 1) if log else ["", ""]
    last = log[0].split(" ", 1) if log else ["", ""]
    counts, frange, qd = [], "", ""
    for k, rx in SCHEMES.items():
        vals = re.findall(rx, t) if t else []
        counts.append(str(len(vals)))
        if k == "F-id" and vals:
            nums = sorted({int(v) for v in vals}); frange = f"F-{nums[0]:02d}..F-{nums[-1]:02d} ({len(nums)} distinct)"
        if k == "Q-id" and vals:
            qd = ",".join("Q" + str(x) for x in sorted({int(v) for v in vals}))
    print("\t".join([f, tracked, str(nl), str(size), first[0], first[1], last[0], last[1], str(len(log)), *counts, frange, qd]))
