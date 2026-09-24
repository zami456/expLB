"""Phase 1 Task B: which lab logs does the Task-B doc corpus name? Joins audit/phase1/labpc/log_index.tsv (242 logs)
with a per-log-file name check against the doc corpus (log file name with or without extension; digits generalised
only when the doc uses a brace/placeholder form is NOT attempted - exact names only). Read-only.
Output: audit/phase1/anomalies_detail/logs_mentions.tsv
Usage: PYTHONIOENCODING=utf-8 python audit/scripts/p1_anom_logs.py
"""
import collections, csv, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p1_anom_common import load_docs, REPO

P1 = os.path.join(REPO, "audit", "phase1")
docs = load_docs()
fmap = open(os.path.join(P1, "fmap_all.csv"), encoding="utf-8").read().lower().replace(chr(92), "/")
rows = []
with open(os.path.join(P1, "labpc", "log_index.tsv"), encoding="utf-8", errors="replace") as f:
    txt = f.read().replace("\x00", "")
rd = csv.DictReader(txt.splitlines(), delimiter="\t")
for r in rd:
    name = r["log"].split(" [")[0].strip()
    stem = os.path.splitext(name)[0]
    pat = re.compile(r"(?<![\w-])" + re.escape(name.lower()) + r"(?![\w-])")
    pat2 = re.compile(r"(?<![\w-])" + re.escape(stem.lower()) + r"\.(log|status|txt|err)")
    hits = [d.split(":", 1)[1] for d, t in docs.items() if pat.search(t)]
    lvl = "EXACT" if hits else ""
    if not hits:
        hits = [d.split(":", 1)[1] for d, t in docs.items() if pat2.search(t)]
        lvl = "SIBLING" if hits else "NONE"
    in_fmap = "y" if pat.search(fmap) else ""
    prefix = re.split(r"[_.\-]", stem)[0]
    rows.append([name, prefix, r["bytes"], r["mtime"], r["first_ts"], r["last_ts"], r["lines"], r["fail_lines"],
                 r["done_lines"], lvl, ";".join(hits[:4]), in_fmap, (r.get("scripts_mentioned") or "")[:120],
                 (r.get("last_line") or "")[:140]])
out = os.path.join(P1, "anomalies_detail", "logs_mentions.tsv")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["log", "prefix", "bytes", "mtime", "first_ts", "last_ts", "lines", "fail_lines", "done_lines",
                "doc_mention", "docs", "in_fmap", "scripts_mentioned", "last_line"])
    w.writerows(rows)
c = collections.Counter(r[9] for r in rows)
print(len(rows), c)
byp = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    byp[r[1]][r[9] if r[9] != "NONE" or r[11] != "y" else "FMAP-ONLY"] += 1
for p, cc in sorted(byp.items()):
    print(f"{p:22s} {dict(cc)}")
