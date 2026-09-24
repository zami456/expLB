"""Grep the Task-B doc corpus (and optionally the secondary prose/code corpus) for regexes. Read-only.
Usage: PYTHONIOENCODING=utf-8 python audit/scripts/p1_anom_grepdocs.py [--sec] [--max N] <regex> [<regex> ...]
Text is lower-cased and backslashes are turned into '/', so write patterns in lower case with '/'.
Prints, per regex: number of docs hit, then up to N hits as  doc:line: snippet.
"""
import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_anom_common import load_docs, load_secondary, git_show, norm, PCOOD

args = sys.argv[1:]
use_sec = "--sec" in args
if use_sec:
    args.remove("--sec")
mx = 6
if "--max" in args:
    i = args.index("--max"); mx = int(args[i + 1]); del args[i:i + 2]
docs = load_docs()
if use_sec:
    sec = load_secondary()
    for k in list(sec):
        if sec[k] is None:
            ref = "HEAD" if k.startswith("code:asix:") else PCOOD
            sec[k] = norm(git_show(ref, k.split(":", 2)[2]))
    docs.update(sec)
for pat in args:
    c = re.compile(pat)
    hits = []
    for d, t in docs.items():
        for i, line in enumerate(t.splitlines(), 1):
            m = c.search(line)
            if m:
                s = max(0, m.start() - 60)
                hits.append(f"  {d}:{i}: {line[s:m.end() + 60].strip()}")
    ndocs = len({h.split(':')[0] + ':' + h.split(':')[1] for h in hits})
    print(f"== {pat}  -> {len(hits)} lines in {ndocs} docs")
    for h in hits[:mx]:
        print(h[:260])
