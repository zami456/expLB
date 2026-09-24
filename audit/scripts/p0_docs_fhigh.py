"""Phase 0: locate F-ids >= F-110 (or any given regex) in the working tree and the auto-memory dir.
Usage: PYTHONIOENCODING=utf-8 python audit/scripts/p0_docs_fhigh.py [regex]
Walks the repo working tree (tracked + untracked + ignored), skipping .git, .venv*, and audit/.
Prints path:line: context for each hit, so you can see whether the token is a finding id or something else."""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEM = os.path.expanduser(r"~/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB/memory")
pat = re.compile(sys.argv[1] if len(sys.argv) > 1 else r"\bF-(1[1-9][0-9]|[2-9][0-9]{2})\b")
EXTS = (".md", ".py", ".ps1", ".json", ".txt", ".csv", ".html", ".yaml", ".yml", ".ipynb", ".tex", ".sh", ".log", ".jsonl")
SKIP_DIRS = {".git", "audit", "node_modules", "__pycache__"}
MAX = 60_000_000
def walk(base):
    for dp, dns, fns in os.walk(base):
        dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith(".venv")]
        for fn in fns:
            if fn.lower().endswith(EXTS):
                yield os.path.join(dp, fn)
hits = 0
for base in (ROOT, MEM):
    for p in walk(base):
        try:
            if os.path.getsize(p) > MAX: continue
            with open(p, encoding="utf-8", errors="replace") as f:
                for ln, line in enumerate(f, 1):
                    for m in pat.finditer(line):
                        hits += 1
                        s = max(0, m.start() - 80); e = min(len(line), m.end() + 80)
                        print(f"{os.path.relpath(p, ROOT) if p.startswith(ROOT) else p}:{ln}: ...{line[s:e].strip()}...")
        except OSError as ex:
            print("ERR", p, ex)
print("total hits:", hits)
